"""Multi-robot traffic simulation: shared road network, mutual visibility, right of way.

The fixed-step engine in :mod:`aegisrover.sim.engine` moves every robot independently,
so two robots sharing a junction never react to each other. This layer puts every robot
into *one* shared network and makes their controllers react to the robots they can see:

* a junction (a node with three or more incident lanes) is a conflict zone that only one
  robot may occupy at a time, handed out under a configurable right-of-way policy;
* a single-lane edge carries one direction at a time — oncoming traffic waits at the
  entry node, with a deterministic tie break when both ends arrive together;
* same-direction traffic keeps a safety gap (car following).

Every yield decision is recorded as a directed edge in a wait-for graph (waiter ->
blocker). A cycle in that graph is a candidate deadlock; only a cycle that persists for
``deadlock_timeout`` seconds is reported as an interlock (互锁), so a transient cycle
during handover is not mistaken for a gridlock.

Separation is measured in lane space (arc distance on a shared edge, or the junction
conflict zone). A robot queued at a node before it commits to its next edge occupies an
abstract "slot" there, so two robots swapping over a degree-two node never read as an
overlap; the zone mutex is what governs junction collisions.

Runs are deterministic like the base engine: fixed steps, robots evaluated in name
order, and a sha256 digest over the trace.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from aegisrover.storage.repository import canonical_json

__all__ = (
    'RoadNetwork', 'RobotSpec', 'TrafficConfig', 'TrafficSimulation',
    'WaitEvent', 'DeadlockRecord', 'RobotWaitSummary', 'TrafficReport',
    'TrafficError', 'FCFS', 'PRIORITY', 'RIGHT_YIELD', 'REASON_LABELS',
)

# Policies
FCFS = 'fcfs'                 # 先到先得，同请求时刻按机器人编号
PRIORITY = 'priority'         # 主路优先，平手按到达时刻、再按编号
RIGHT_YIELD = 'right_yield'   # 让右方来车 —— 四向同时到达时会形成环

_REASON_INTERSECTION = 'intersection'
_REASON_HEAD_ON = 'head_on'
_REASON_CAR_FOLLOW = 'car_follow'

_REASON_LABELS = {
    _REASON_INTERSECTION: '路口让行',
    _REASON_HEAD_ON: '窄道会车',
    _REASON_CAR_FOLLOW: '同向跟车',
}
REASON_LABELS = dict(_REASON_LABELS)


class TrafficError(ValueError):
    pass


# ------------------------------------------------------------------------------- network
@dataclass(frozen=True)
class Edge:
    a: str
    b: str
    length: float
    priority: bool = False  # 主路标志


class RoadNetwork:
    """Undirected single-lane graph. Nodes with degree >= 3 are junctions."""

    def __init__(self):
        self.nodes: dict[str, tuple[float, float]] = {}
        self.edges: dict[tuple[str, str], Edge] = {}
        self.adjacent: dict[str, list[str]] = {}

    def add_node(self, name: str, x: float, y: float) -> None:
        if name in self.nodes:
            raise TrafficError(f'duplicate node {name!r}')
        self.nodes[name] = (float(x), float(y))
        self.adjacent[name] = []

    def add_edge(self, a: str, b: str, *, priority: bool = False) -> Edge:
        if a == b:
            raise TrafficError('an edge must connect two different nodes')
        if a not in self.nodes or b not in self.nodes:
            raise TrafficError(f'unknown endpoint in edge {a!r}-{b!r}')
        key = (a, b) if a < b else (b, a)
        if key in self.edges:
            raise TrafficError(f'duplicate edge {key!r}')
        (x1, y1), (x2, y2) = self.nodes[a], self.nodes[b]
        length = math.hypot(x2 - x1, y2 - y1)
        if length <= 0:
            raise TrafficError(f'edge {key!r} has zero length')
        edge = Edge(key[0], key[1], length, priority=priority)
        self.edges[key] = edge
        self.adjacent[a].append(b)
        self.adjacent[b].append(a)
        return edge

    def key(self, a: str, b: str) -> tuple[str, str]:
        candidate = (a, b) if a < b else (b, a)
        if candidate not in self.edges:
            raise TrafficError(f'no edge between {a!r} and {b!r}')
        return candidate

    def is_junction(self, node: str) -> bool:
        return len(self.adjacent[node]) >= 3

    def position_on(self, key: tuple[str, str], from_node: str, s: float) -> tuple[float, float, float]:
        edge = self.edges[key]
        u, v = (edge.a, edge.b) if from_node == edge.a else (edge.b, edge.a)
        x1, y1 = self.nodes[u]
        x2, y2 = self.nodes[v]
        f = max(0.0, min(1.0, s / edge.length))
        yaw = math.atan2(y2 - y1, x2 - x1)
        return x1 + (x2 - x1) * f, y1 + (y2 - y1) * f, yaw

    def validate_route(self, route: Sequence[str]) -> None:
        if len(route) < 2:
            raise TrafficError('a route needs at least two nodes')
        for node in route:
            if node not in self.nodes:
                raise TrafficError(f'unknown route node {node!r}')
        for a, b in zip(route, route[1:]):
            self.key(a, b)


# ------------------------------------------------------------------------------- config
@dataclass(frozen=True)
class RobotSpec:
    name: str
    route: tuple[str, ...]
    max_speed: float = 1.0
    radius: float = 0.3


@dataclass(frozen=True)
class TrafficConfig:
    step: float = 0.2
    vision_range: float = 30.0          # 互相看见的距离
    approach_dist: float = 2.5          # 多远开始申请路口
    zone_radius: float = 1.1            # 路口冲突区半径
    safety_gap: float = 0.4             # 同向跟车最小净距
    accel: float = 4.0                  # m/s^2，加速/减速对称
    deadlock_timeout: float = 1.0       # 等待环持续多久判定为互锁
    policy: str = FCFS

    def __post_init__(self):
        if self.step <= 0:
            raise TrafficError('step must be positive')
        if self.vision_range <= 0 or self.approach_dist <= 0 or self.zone_radius <= 0:
            raise TrafficError('distances must be positive')
        if self.policy not in (FCFS, PRIORITY, RIGHT_YIELD):
            raise TrafficError(f'unknown policy {self.policy!r}')


# ------------------------------------------------------------------------------- runtime
@dataclass
class _RState:
    spec: RobotSpec
    leg: int = 0                  # current edge is route[leg] -> route[leg+1]
    s: float = 0.0                # progress along current edge from route[leg]
    committed: bool = False       # has physically entered the current edge
    speed: float = 0.0
    request_time: float | None = None
    reservation: str | None = None  # junction node the robot may enter
    arrived: bool = False
    finish_time: float | None = None
    # per-step decision outputs
    blocker: str | None = None
    block_reason: str | None = None
    stop_arc: float = math.inf

    @property
    def from_node(self) -> str:
        return self.spec.route[self.leg]

    @property
    def target_node(self) -> str | None:
        route = self.spec.route
        if self.leg >= len(route) - 1:
            return None
        return route[self.leg + 1]


@dataclass(frozen=True)
class WaitEvent:
    robot: str
    reason: str
    start: float
    end: float
    duration: float
    blocker: str
    location: str
    visible: bool
    blocker_distance: float | None

    def to_dict(self) -> dict:
        return {
            'robot': self.robot, 'reason': self.reason,
            'reason_label': REASON_LABELS[self.reason],
            'start': round(self.start, 6), 'end': round(self.end, 6),
            'duration': round(self.duration, 6), 'blocker': self.blocker,
            'location': self.location, 'visible': self.visible,
            'blocker_distance': None if self.blocker_distance is None
            else round(self.blocker_distance, 6),
        }


@dataclass(frozen=True)
class DeadlockRecord:
    members: tuple[str, ...]
    start: float
    confirmed_at: float
    end: float | None
    duration: float | None
    location: str
    resolved: bool

    def to_dict(self) -> dict:
        return {
            'members': list(self.members), 'start': round(self.start, 6),
            'confirmed_at': round(self.confirmed_at, 6),
            'end': None if self.end is None else round(self.end, 6),
            'duration': None if self.duration is None else round(self.duration, 6),
            'location': self.location, 'resolved': self.resolved,
        }


@dataclass(frozen=True)
class RobotWaitSummary:
    robot: str
    total_wait: float
    wait_count: int
    finish_time: float | None
    arrived: bool
    deadlocked_with: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            'robot': self.robot, 'total_wait': round(self.total_wait, 6),
            'wait_count': self.wait_count,
            'finish_time': None if self.finish_time is None else round(self.finish_time, 6),
            'arrived': self.arrived, 'deadlocked_with': list(self.deadlocked_with),
        }


@dataclass(frozen=True)
class TrafficReport:
    duration: float
    steps: int
    policy: str
    digest: str
    trace: tuple[dict, ...]
    waits: tuple[WaitEvent, ...]
    deadlocks: tuple[DeadlockRecord, ...]
    summaries: tuple[RobotWaitSummary, ...]
    min_separation: float

    def to_dict(self) -> dict:
        return {
            'duration': self.duration, 'steps': self.steps, 'policy': self.policy,
            'digest': self.digest, 'waits': [w.to_dict() for w in self.waits],
            'deadlocks': [d.to_dict() for d in self.deadlocks],
            'summaries': [s.to_dict() for s in self.summaries],
            'min_separation': round(self.min_separation, 6),
        }


class TrafficSimulation:
    """Step all robots through one shared network under one right-of-way policy."""

    def __init__(self, net: RoadNetwork, specs: Iterable[RobotSpec],
                 config: TrafficConfig | None = None):
        self.net = net
        self.config = config or TrafficConfig()
        names = [s.name for s in specs]
        if len(names) != len(set(names)):
            raise TrafficError('robot names must be unique')
        self.states: dict[str, _RState] = {}
        for spec in specs:
            net.validate_route(spec.route)
            if spec.max_speed <= 0 or spec.radius <= 0:
                raise TrafficError(f'bad spec for {spec.name!r}')
            self.states[spec.name] = _RState(spec=tuple_route(spec))
        self.time = 0.0
        self._history: list[dict[str, tuple]] = []

    # -- geometry helpers ------------------------------------------------------
    def _pose(self, r: _RState) -> tuple[float, float, float]:
        if r.arrived:
            x, y = self.net.nodes[r.spec.route[-1]]
            return x, y, 0.0
        key = self.net.key(r.from_node, r.target_node)
        return self.net.position_on(key, r.from_node, r.s if r.committed else 0.0)

    def _distance_to_node(self, r: _RState, node: str) -> float:
        x, y = self.net.nodes[node]
        px, py, _ = self._pose(r)
        return math.hypot(px - x, py - y)

    def _visible(self, a: _RState, b: _RState, poses: dict[str, tuple[float, float]]) -> bool:
        ax, ay = poses[a.spec.name]
        bx, by = poses[b.spec.name]
        return math.hypot(ax - bx, ay - by) <= self.config.vision_range + 1e-12

    # -- main loop -------------------------------------------------------------
    def run(self, duration: float) -> TrafficReport:
        if duration <= 0:
            raise TrafficError('duration must be positive')
        cfg = self.config
        steps = int(round(duration / cfg.step))
        trace: list[dict] = []
        # stopped_step[robot] = (reason, blocker, location, visible, distance) or None
        stopped: list[dict[str, tuple]] = []
        active_deadlocks: dict[frozenset[str], dict] = {}
        finished_deadlocks: list[DeadlockRecord] = []

        for _ in range(steps):
            poses = {name: self._pose(r)[:2] for name, r in self.states.items()}
            self._decide(poses)

            # integrate in deterministic name order
            for name in sorted(self.states):
                self._move(self.states[name])

            stopped_now = self._collect_stopped(poses)
            stopped.append(stopped_now)
            self._history.append({name: self._lane_state(r)
                                  for name, r in self.states.items()})

            # wait-for graph -> persistent cycles
            graph = {name: r.blocker for name, r in self.states.items()
                     if r.blocker is not None and not r.arrived}
            cycles = _find_cycles(graph)
            self._update_deadlocks(cycles, active_deadlocks, finished_deadlocks)

            samples = {}
            for name, r in self.states.items():
                x, y, yaw = self._pose(r)
                if r.arrived:
                    state = 'arrived'
                elif name in stopped_now:
                    state = 'waiting'
                else:
                    state = 'moving'
                samples[name] = {'x': round(x, 9), 'y': round(y, 9),
                                 'yaw': round(yaw, 9), 'v': round(r.speed, 9),
                                 'state': state, 'blocker': r.blocker}
            trace.append({'time': round(self.time + cfg.step, 9), 'robots': samples})
            self.time += cfg.step

        # close any *confirmed* deadlock still active at end of run; cycles that
        # never outlived the timeout are transient and are not reported
        for members, info in active_deadlocks.items():
            if info['confirmed']:
                finished_deadlocks.append(self._close_deadlock(members, info, None))

        waits = self._build_wait_events(stopped)
        summaries = self._summarize(waits, finished_deadlocks)
        digest = hashlib.sha256(canonical_json({
            'policy': cfg.policy, 'step': cfg.step,
            'trace': trace,
        }).encode()).hexdigest()
        return TrafficReport(
            duration=self.time, steps=steps, policy=cfg.policy, digest=digest,
            trace=tuple(trace), waits=tuple(waits),
            deadlocks=tuple(finished_deadlocks), summaries=tuple(summaries),
            min_separation=self._min_separation(),
        )

    # -- decision layer --------------------------------------------------------
    def _decide(self, poses: dict[str, tuple[float, float]]) -> None:
        for r in self.states.values():
            r.blocker = None
            r.block_reason = None
            r.stop_arc = math.inf

        self._resolve_junctions(poses)
        self._resolve_queue_entries(poses)
        self._resolve_lane_entries(poses)
        self._resolve_car_following(poses)

    def _active_robots(self) -> list[_RState]:
        return [r for r in self.states.values() if not r.arrived]

    def _resolve_junctions(self, poses) -> None:
        cfg = self.config
        junctions = {n for n in self.net.nodes if self.net.is_junction(n)}
        for node in junctions:
            requesters = self._junction_requesters(node)
            requester_names = {r.spec.name for r in requesters}
            # A robot physically crossing the zone holds it; one queued exactly at the
            # node is still an abstract slot and arbitrates as a requester instead.
            occupants = [r for r in self._active_robots()
                         if self._distance_to_node(r, node) <= cfg.zone_radius + 1e-9
                         and (r.committed or r.spec.name not in requester_names)]
            if not requesters:
                continue

            if occupants:
                guard = min(occupants, key=lambda o: o.spec.name)
                for r in requesters:
                    self._block(r, guard.spec.name, _REASON_INTERSECTION,
                                self._junction_stop_arc(r), poses)
                continue

            if cfg.policy == RIGHT_YIELD:
                self._resolve_right_yield(node, requesters, poses)
                continue

            # FCFS / main-road priority: walk the priority queue and grant every
            # requester whose traversal arc is compatible with the arcs already
            # granted this step and whose exit lane is free of physical oncoming
            # traffic. A skipped head lets later compatible requesters through
            # (work-conserving); incompatible later requesters wait on the first
            # granted robot that conflicts with them.
            ordered = self._ordered_requesters(node, requesters)
            granted: list[_RState] = []
            blocked: dict[str, tuple[str, str]] = {}
            for r in ordered:
                arc = self._arc(r, node)
                conflict = next((g for g in granted if _arcs_conflict(arc, self._arc(g, node), self.net, node, cfg.zone_radius)), None)
                if conflict is not None:
                    blocked[r.spec.name] = (conflict.spec.name, _REASON_INTERSECTION)
                    continue
                oncoming = self._physical_oncoming(r, node, poses)
                if oncoming is not None:
                    blocked[r.spec.name] = (oncoming, _REASON_HEAD_ON)
                    continue
                r.reservation = node
                granted.append(r)

            if not granted:
                # nobody could move; point each waiter at the highest-priority rival
                # or its physical oncoming blocker so the wait-for graph is real
                for r in ordered:
                    name, reason = blocked[r.spec.name]
                    self._block(r, name, reason, self._junction_stop_arc(r), poses)
            else:
                for r in ordered:
                    if r in granted:
                        continue
                    name, reason = blocked[r.spec.name]
                    if reason == _REASON_HEAD_ON:
                        self._block(r, name, reason, self._junction_stop_arc(r), poses)
                    else:
                        first_conflict = next(
                            (g for g in granted
                             if _arcs_conflict(self._arc(r, node), self._arc(g, node),
                                               self.net, node, cfg.zone_radius)),
                            granted[0])
                        self._block(r, first_conflict.spec.name, _REASON_INTERSECTION,
                                    self._junction_stop_arc(r), poses)

    def _junction_requesters(self, node: str) -> list[_RState]:
        cfg = self.config
        out = []
        for r in self._active_robots():
            if r.target_node != node or r.reservation == node:
                continue
            if r.committed:
                edge = self.net.edges[self.net.key(r.from_node, node)]
                if edge.length - r.s > cfg.approach_dist + 1e-9:
                    continue
            if r.request_time is None:
                r.request_time = self.time
            out.append(r)
        return out

    def _arc(self, r: _RState, node: str) -> tuple[str, str]:
        """The directed (entry arm, exit arm) pair of r's traversal through node."""
        route = r.spec.route
        # r approaches node from route[leg]; it exits toward route[leg+2]
        assert route[r.leg + 1] == node
        far = route[r.leg + 2] if r.leg + 2 < len(route) else None
        return (r.from_node, far if far is not None else node)

    def _physical_oncoming(self, r: _RState, node: str, poses) -> str | None:
        """A committed oncoming car physically present on r's exit arm.

        Only a robot driving *toward* node counts; same-direction straight-through
        traffic drives away from node. Queued requesters are handled by arc
        conflicts instead.
        """
        route = r.spec.route
        if r.leg + 2 >= len(route):
            return None
        far = route[r.leg + 2]
        key = self.net.key(node, far)
        for q in self._active_robots():
            if q is r or q.arrived or not q.committed or q.target_node != node:
                continue
            if self.net.key(q.from_node, q.target_node) != key:
                continue
            if not self._visible(r, q, poses):
                continue
            if q.s > self.config.zone_radius + 1e-9:
                return q.spec.name
        return None

    def _resolve_right_yield(self, node: str, requesters: list[_RState], poses) -> None:
        # A contender yields to every visible rival on its right that wants a
        # conflicting arc; robots whose arcs are compatible may both enter, and an
        # oncoming physical car still blocks the exit arm.
        right: dict[str, list[str]] = {r.spec.name: [] for r in requesters}
        for i, r in enumerate(requesters):
            rx, ry, yaw = self._pose(r)
            arc_r = self._arc(r, node)
            for q in requesters[i + 1:]:
                if not _arcs_conflict(arc_r, self._arc(q, node), self.net, node, self.config.zone_radius):
                    continue
                if not self._visible(r, q, poses):
                    continue
                qx, qy = poses[q.spec.name]
                side_r = math.cos(yaw) * (qy - ry) - math.sin(yaw) * (qx - rx)
                _, _, yaw_q = self._pose(q)
                side_q = math.cos(yaw_q) * (rx - qx) - math.sin(yaw_q) * (ry - qy)
                if side_r < -1e-9:
                    right[r.spec.name].append(q.spec.name)
                if side_q < -1e-9:
                    right[q.spec.name].append(r.spec.name)

        granted: list[_RState] = []
        for r in requesters:
            if right[r.spec.name]:
                self._block(r, min(right[r.spec.name]), _REASON_INTERSECTION,
                            self._junction_stop_arc(r), poses)
                continue
            oncoming = self._physical_oncoming(r, node, poses)
            if oncoming is not None:
                self._block(r, oncoming, _REASON_HEAD_ON,
                            self._junction_stop_arc(r), poses)
                continue
            r.reservation = node
            granted.append(r)

    def _ordered_requesters(self, node: str, requesters: list[_RState]) -> list[_RState]:
        if self.config.policy == PRIORITY:
            def key(r: _RState) -> tuple:
                edge = self.net.edges[self.net.key(r.from_node, node)]
                return (0 if edge.priority else 1, r.request_time, r.spec.name)
        else:
            def key(r: _RState) -> tuple:
                return (r.request_time, r.spec.name)
        return sorted(requesters, key=key)

    def _junction_stop_arc(self, r: _RState) -> float:
        if not r.committed:
            return 0.0
        edge = self.net.edges[self.net.key(r.from_node, r.target_node)]
        return edge.length - (self.config.zone_radius + r.spec.radius)

    def _resolve_lane_entries(self, poses) -> None:
        """Arbitrate the moment a robot commits onto an edge (narrow-lane yielding)."""
        for r in sorted(self._active_robots(), key=lambda x: x.spec.name):
            if r.committed:
                continue
            node = r.from_node
            target = r.target_node
            if target is None:
                continue
            # A junction entry needs the reservation first.
            if self.net.is_junction(node) and r.reservation != node:
                continue
            key = self.net.key(node, target)
            edge = self.net.edges[key]

            committed = []
            for q in self._active_robots():
                if q is r:
                    continue
                qkey = self.net.key(q.from_node, q.target_node)
                if qkey != key or q.from_node == node:
                    continue
                if not self._visible(r, q, poses):
                    continue
                if q.committed:
                    # q is coming from the far end; wait while it still occupies the
                    # lane ahead of our entry conflict zone (arc s measured from q's end)
                    if q.s > self.config.zone_radius + 1e-9:
                        committed.append(q)
                else:
                    far_node = q.from_node
                    if self.net.is_junction(far_node) and q.reservation != far_node:
                        continue  # rival cannot enter yet
                    if q.blocker is not None:
                        continue  # rival is queued behind someone and cannot enter
                    # two free robots diverging head-on: deterministic name tie break
                    if q.spec.name < r.spec.name:
                        committed.append(q)

            if committed:
                blocker = min(committed, key=lambda q: q.spec.name)
                self._block(r, blocker.spec.name, _REASON_HEAD_ON, 0.0, poses)

    def _resolve_queue_entries(self, poses) -> None:
        """Multiple robots queued at one node take the same outgoing edge in order.

        Without this, a convoy would pile onto a single lane at the same time. The
        leader is the committed one closest to the node (or the alphabetically first
        uncommitted robot); everyone else waits in the node slot. A robot may commit
        only once the nearest same-direction committed robot has left a launch gap,
        otherwise a convoy starting from one slot would overlap on the lane.
        """
        groups: dict[tuple[str, str], list[_RState]] = {}
        for r in self._active_robots():
            if r.committed or r.target_node is None:
                continue
            if self.net.is_junction(r.from_node) and r.reservation != r.from_node:
                continue  # junction winner is decided by the reservation step
            groups.setdefault((r.from_node, r.target_node), []).append(r)
        for (from_node, target_node), group in groups.items():
            group.sort(key=lambda r: r.spec.name)
            key = self.net.key(from_node, target_node)
            # nearest committed robot on the shared edge, measured from this node
            ahead_s = None
            ahead_name = None
            for q in self._active_robots():
                if not q.committed or q.from_node != from_node:
                    continue
                if self.net.key(q.from_node, q.target_node) != key:
                    continue
                if ahead_s is None or q.s < ahead_s:
                    ahead_s, ahead_name = q.s, q.spec.name
            launch_gap = self.config.safety_gap + 2 * group[0].spec.radius
            for index, r in enumerate(group):
                if index > 0:
                    self._block(r, group[0].spec.name, _REASON_CAR_FOLLOW, 0.0, poses)
                elif ahead_name is not None and ahead_s < launch_gap - 1e-9:
                    # leader waits for the previous convoy member to clear launch gap
                    self._block(r, ahead_name, _REASON_CAR_FOLLOW, 0.0, poses)

    def _resolve_car_following(self, poses) -> None:
        groups: dict[tuple, list[_RState]] = {}
        for r in self._active_robots():
            if not r.committed:
                continue
            key = (self.net.key(r.from_node, r.target_node), r.from_node)
            groups.setdefault(key, []).append(r)
        for group in groups.values():
            group.sort(key=lambda r: r.s)
            for behind, ahead in zip(group, group[1:]):
                gap = ahead.s - behind.s - ahead.spec.radius - behind.spec.radius
                if gap < self.config.safety_gap - 1e-9 and self._visible(behind, ahead, poses):
                    arc = ahead.s - self.config.safety_gap - ahead.spec.radius - behind.spec.radius
                    self._block(behind, ahead.spec.name, _REASON_CAR_FOLLOW, arc, poses)

    def _block(self, r: _RState, blocker: str, reason: str, stop_arc: float, poses) -> None:
        # intersection reservation dominates; head-on dominates car following
        rank = {_REASON_INTERSECTION: 0, _REASON_HEAD_ON: 1, _REASON_CAR_FOLLOW: 2}
        if r.block_reason is None or rank[reason] < rank[r.block_reason]:
            r.blocker = blocker
            r.block_reason = reason
            r.stop_arc = stop_arc

    # -- movement ---------------------------------------------------------------
    def _move(self, r: _RState) -> None:
        cfg = self.config
        dt = cfg.step
        if r.arrived:
            return

        if not r.committed:
            if r.blocker is not None:
                r.speed = 0.0
                return
            r.committed = True
            # leaving a queued slot at a node releases that node's reservation
            r.reservation = None

        target_speed = r.spec.max_speed if r.blocker is None else 0.0
        if r.speed < target_speed:
            r.speed = min(target_speed, r.speed + cfg.accel * dt)
        elif r.speed > target_speed:
            r.speed = max(target_speed, r.speed - cfg.accel * dt)

        budget = r.speed * dt
        if r.blocker is not None:
            budget = min(budget, max(0.0, r.stop_arc - r.s))

        while budget > 1e-12 and not r.arrived:
            edge = self.net.edges[self.net.key(r.from_node, r.target_node)]
            remaining = edge.length - r.s
            if budget + 1e-12 < remaining:
                r.s += budget
                budget = 0.0
                break
            budget -= remaining
            r.s = edge.length
            route = r.spec.route
            if r.leg == len(route) - 2:
                r.arrived = True
                r.finish_time = self.time + cfg.step - budget
                r.speed = 0.0
                r.reservation = None
                break
            next_node = r.target_node
            r.leg += 1
            r.s = 0.0
            r.committed = False
            r.request_time = None
            if r.reservation == next_node:
                r.reservation = None
            # Re-check entry into the next leg with the remaining budget.
            if r.blocker is not None:
                r.speed = 0.0
                break

    # -- reporting --------------------------------------------------------------
    def _collect_stopped(self, poses) -> dict[str, tuple]:
        out = {}
        for name, r in self.states.items():
            if r.arrived or r.blocker is None or r.speed > 1e-9:
                continue
            if r.committed and r.s > r.stop_arc + 1e-6:
                continue  # still rolling toward the stop line, not yet waiting
            q = self.states[r.blocker]
            bx, by = poses[r.blocker]
            px, py = poses[name]
            distance = math.hypot(px - bx, py - by)
            location = self._location_label(r)
            out[name] = (r.block_reason, r.blocker, location,
                         distance <= self.config.vision_range + 1e-12, distance)
        return out

    def _location_label(self, r: _RState) -> str:
        if r.block_reason == _REASON_INTERSECTION:
            return f'路口 {r.target_node}'
        if r.block_reason == _REASON_HEAD_ON:
            if r.committed:
                return f'车道 {r.from_node}-{r.target_node}'
            return f'车道入口 {r.from_node}'
        if r.committed:
            return f'车道 {r.from_node}-{r.target_node}'
        return f'排队 {r.from_node}（同向放行）'

    def _update_deadlocks(self, cycles: list[set[str]], active: dict, finished: list) -> None:
        cfg = self.config
        seen: set[frozenset[str]] = set()
        for cycle in cycles:
            members = frozenset(cycle)
            seen.add(members)
            if members in active:
                active[members]['last_seen'] = self.time + cfg.step
                continue
            start = self.time + cfg.step
            active[members] = {
                'start': start,
                'confirmed_at': start + cfg.deadlock_timeout,
                'last_seen': start,
                'location': self._cycle_location(members),
                'confirmed': False,
            }
        stale = [m for m in active if m not in seen]
        for members in stale:
            if active[members]['confirmed']:
                finished.append(self._close_deadlock(members, active[members],
                                                     active[members]['last_seen']))
            del active[members]
        for info in active.values():
            if not info['confirmed'] and self.time + cfg.step >= info['confirmed_at'] - 1e-9:
                info['confirmed'] = True

    def _close_deadlock(self, members: frozenset[str], info: dict,
                        end: float | None) -> DeadlockRecord:
        confirmed = info['confirmed']
        record_end = end if confirmed and end is not None and end >= info['confirmed_at'] else None
        duration = None if record_end is None else max(0.0, record_end - info['confirmed_at'])
        return DeadlockRecord(
            members=tuple(sorted(members)), start=info['start'],
            confirmed_at=info['confirmed_at'], end=record_end,
            duration=duration, location=info['location'],
            resolved=record_end is not None,
        )

    def _cycle_location(self, members: frozenset[str]) -> str:
        nodes = []
        for name in members:
            r = self.states[name]
            node = r.target_node if r.block_reason == _REASON_INTERSECTION else r.from_node
            if node and node not in nodes:
                nodes.append(node)
        return '、'.join(f'路口 {n}' for n in nodes) if nodes else '未知位置'

    def _build_wait_events(self, stopped_steps: list[dict[str, tuple]]) -> list[WaitEvent]:
        dt = self.config.step
        events: list[WaitEvent] = []
        for name in self.states:
            open_event: dict | None = None
            for index, stopped in enumerate(stopped_steps):
                info = stopped.get(name)
                t_end = (index + 1) * dt
                if info is None:
                    if open_event is not None:
                        events.append(self._finalize_wait(name, open_event))
                        open_event = None
                    continue
                reason, blocker, location, visible, distance = info
                if open_event is None or open_event['reason'] != reason \
                        or open_event['blocker'] != blocker:
                    if open_event is not None:
                        events.append(self._finalize_wait(name, open_event))
                    open_event = {'start': t_end - dt, 'end': t_end, 'reason': reason,
                                  'blocker': blocker, 'location': location,
                                  'visible': visible, 'distance': distance}
                else:
                    open_event['end'] = t_end
                    open_event['visible'] = open_event['visible'] and visible
                    open_event['distance'] = max(open_event['distance'], distance)
            if open_event is not None:
                events.append(self._finalize_wait(name, open_event))
        events.sort(key=lambda e: (e.start, e.robot))
        return events

    @staticmethod
    def _finalize_wait(robot: str, data: dict) -> WaitEvent:
        return WaitEvent(
            robot=robot, reason=data['reason'], start=data['start'], end=data['end'],
            duration=max(0.0, data['end'] - data['start']), blocker=data['blocker'],
            location=data['location'], visible=data['visible'],
            blocker_distance=data['distance'],
        )

    def _lane_state(self, r: _RState) -> tuple:
        """Where the robot occupies network space: an edge with arc, or a node slot."""
        if r.arrived:
            return ('node', r.spec.route[-1], 0.0)
        key = self.net.key(r.from_node, r.target_node)
        if r.committed:
            return ('edge', key, r.from_node, r.s)
        return ('node', r.from_node, 0.0)

    def _min_separation(self) -> float:
        """Smallest clearance across the whole run, measured in lane space.

        On a shared edge it is arc distance minus the two radii; uncommitted robots
        share a node slot (junction queue), whose length is the zone diameter; robots
        on disjoint edges are considered safely separated.
        """
        worst = math.inf
        for snapshot in self._history:
            lanes: dict[tuple, list[tuple[float, float]]] = {}
            slot_count: dict[str, int] = {}
            slot_radii: dict[str, list[float]] = {}
            for name, item in snapshot.items():
                radius = self.states[name].spec.radius
                if item[0] == 'edge':
                    _, key, from_node, s = item
                    arc = s if item[2] == from_node else self.net.edges[key].length - s
                    lanes.setdefault(key, []).append((arc, radius))
                else:
                    slot_count[item[1]] = slot_count.get(item[1], 0) + 1
                    slot_radii.setdefault(item[1], []).append(radius)
            for items in lanes.values():
                items.sort()
                for (s1, rad1), (s2, rad2) in zip(items, items[1:]):
                    worst = min(worst, s2 - s1 - rad1 - rad2)
            for node, radii in slot_radii.items():
                if slot_count[node] > 1:
                    radii = sorted(radii)
                    for i, rad1 in enumerate(radii):
                        for rad2 in radii[i + 1:]:
                            worst = min(worst, 2 * self.config.zone_radius - rad1 - rad2)
        return worst if math.isfinite(worst) else math.inf

    def _summarize(self, waits: list[WaitEvent],
                   deadlocks: list[DeadlockRecord]) -> list[RobotWaitSummary]:
        locked: dict[str, set[str]] = {name: set() for name in self.states}
        for d in deadlocks:
            members = set(d.members)
            for name in d.members:
                locked[name] |= members - {name}
        out = []
        for name, r in self.states.items():
            mine = [w for w in waits if w.robot == name]
            out.append(RobotWaitSummary(
                robot=name,
                total_wait=round(sum(w.duration for w in mine), 9),
                wait_count=len(mine),
                finish_time=None if r.finish_time is None else round(r.finish_time, 9),
                arrived=r.arrived,
                deadlocked_with=tuple(sorted(locked[name])),
            ))
        return out


def tuple_route(spec: RobotSpec) -> RobotSpec:
    return RobotSpec(name=spec.name, route=tuple(spec.route),
                     max_speed=spec.max_speed, radius=spec.radius)


def _arcs_conflict(arc1: tuple[str, str], arc2: tuple[str, str],
                   net: 'RoadNetwork', node: str, radius: float) -> bool:
    """Do two (entry arm, exit arm) traversals through ``node`` collide in the zone?

    A traversal is the chord between stop-line points one radius out on each arm.
    Traversals conflict when they share an arm (head-on or same stream) or when
    their inside-zone chords come within two radii of each other.
    """
    in1, out1 = arc1
    in2, out2 = arc2
    if arc1 == arc2:
        return True
    if in1 == in2 or out1 == out2:
        return True  # same entry or same exit arm: streams merge/diverge in the zone
    if in1 == out2 or out1 == in2:
        return True  # head-on over a shared arm
    p1 = _stop_point(net, node, in1, radius)
    q1 = _stop_point(net, node, out1, radius)
    p2 = _stop_point(net, node, in2, radius)
    q2 = _stop_point(net, node, out2, radius)
    return _segment_distance(p1, q1, p2, q2) < 2 * radius - 1e-9


def _stop_point(net: 'RoadNetwork', node: str, arm: str, radius: float
                ) -> tuple[float, float]:
    nx, ny = net.nodes[node]
    ax, ay = net.nodes[arm]
    dx, dy = ax - nx, ay - ny
    length = math.hypot(dx, dy)
    return nx + dx / length * radius, ny + dy / length * radius


def _segment_distance(a, b, p, q) -> float:
    """Minimum distance between segments a-b and p-q."""
    return min(_point_segment_distance(a, p, q), _point_segment_distance(b, p, q),
               _point_segment_distance(p, a, b), _point_segment_distance(q, a, b))


def _point_segment_distance(p, a, b) -> float:
    vx, vy = b[0] - a[0], b[1] - a[1]
    wx, wy = p[0] - a[0], p[1] - a[1]
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-18:
        return math.hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, (wx * vx + wy * vy) / length_sq))
    cx, cy = a[0] + t * vx, a[1] + t * vy
    return math.hypot(p[0] - cx, p[1] - cy)


def _find_cycles(graph: dict[str, str]) -> list[set[str]]:
    """Distinct simple cycles of a functional graph (each node has <= one blocker)."""
    cycles: list[set[str]] = []
    seen_cycles: set[frozenset[str]] = set()
    for start in graph:
        path: list[str] = []
        index: dict[str, int] = {}
        node = start
        while node is not None and node not in index:
            if node not in graph:
                break
            index[node] = len(path)
            path.append(node)
            node = graph[node]
        if node in index:
            cycle = frozenset(path[index[node]:])
            if cycle not in seen_cycles:
                seen_cycles.add(cycle)
                cycles.append(set(cycle))
    return cycles
