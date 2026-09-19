"""Multi-robot intersection coordination: shared perception, yield rules, wait and deadlock reporting.

The base engine drives every robot with an open-loop twist, so two robots can
cross the same intersection without ever noticing each other. This module closes
that gap for fleet studies: each robot follows a waypoint route, perceives
neighbours within a sensor radius, and yields at conflict zones according to a
policy. The supervisor records every wait episode — who waited, where, and on
whom — and flags cyclic waits (deadlocks) once they persist beyond a threshold,
so a run can be graded instead of eyeballed.

Two yield policies are provided. ``priority`` applies a total order over the
fleet: a robot only waits for higher-priority robots, so the wait-for graph is
acyclic by construction and every wait eventually resolves. ``right_hand``
yields to whoever approaches from the right; with symmetric arrivals that rule
has no global tie-break and four robots can end up each waiting on the next —
exactly the interlock the report is meant to surface.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from aegisrover.core.types import Pose2, Twist2, wrap_angle
from aegisrover.sim.engine import RunResult, SimulationEngine

__all__ = ('ConflictZone', 'DeadlockIncident', 'FleetError', 'FleetReport',
           'FleetSupervisor', 'WaitEpisode', 'run_fleet')

POLICIES = ('priority', 'right_hand')

# Bearing window (radians, relative to own yaw) that counts as "on my right".
_RIGHT_MIN, _RIGHT_MAX = -2.6, -0.15
# How far past the zone boundary a robot must get before the zone counts as cleared.
_EXIT_HYSTERESIS = 0.25


class FleetError(RuntimeError):
    pass


@dataclass(frozen=True)
class ConflictZone:
    """A circular intersection region that robots should not occupy together."""
    zone_id: str
    x: float
    y: float
    radius: float

    def edge_distance(self, x: float, y: float) -> float:
        """Distance from (x, y) to the zone boundary; negative when inside."""
        return math.hypot(x - self.x, y - self.y) - self.radius


@dataclass(frozen=True)
class WaitEpisode:
    """One contiguous interval a robot spent stopped at a zone, waiting for others."""
    robot: str
    zone: str
    start: float
    end: float
    blocked_by: tuple[str, ...]
    x: float
    y: float
    ongoing: bool = False

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 9)

    def to_dict(self) -> dict:
        return {'robot': self.robot, 'zone': self.zone, 'start': self.start,
                'end': self.end, 'duration': self.duration,
                'blocked_by': list(self.blocked_by), 'position': [self.x, self.y],
                'ongoing': self.ongoing}


@dataclass(frozen=True)
class DeadlockIncident:
    """A cyclic wait that persisted past the deadlock threshold."""
    robots: tuple[str, ...]
    zones: tuple[str, ...]
    started_at: float
    detected_at: float
    ended_at: float
    ongoing: bool = False

    @property
    def duration(self) -> float:
        return round(self.ended_at - self.started_at, 9)

    def to_dict(self) -> dict:
        return {'robots': list(self.robots), 'zones': list(self.zones),
                'started_at': self.started_at, 'detected_at': self.detected_at,
                'ended_at': self.ended_at, 'duration': self.duration,
                'ongoing': self.ongoing}


@dataclass(frozen=True)
class FleetReport:
    """Post-run account of who waited where and which waits never resolved."""
    policy: str
    duration: float
    episodes: tuple[WaitEpisode, ...]
    deadlocks: tuple[DeadlockIncident, ...]
    wait_totals: dict[str, float]
    finished: dict[str, bool]

    def waited(self, robot: str) -> float:
        return self.wait_totals.get(robot, 0.0)

    def to_dict(self) -> dict:
        return {'policy': self.policy, 'duration': self.duration,
                'wait_totals': dict(self.wait_totals), 'finished': dict(self.finished),
                'episodes': [e.to_dict() for e in self.episodes],
                'deadlocks': [d.to_dict() for d in self.deadlocks]}

    def summary(self) -> str:
        lines = [f'fleet report: policy={self.policy} duration={self.duration:.2f}s']
        done = sum(1 for ok in self.finished.values() if ok)
        lines.append(f'robots: {len(self.finished)}, routes finished: {done}')
        for robot in sorted(self.wait_totals):
            total = self.wait_totals[robot]
            count = sum(1 for e in self.episodes if e.robot == robot)
            lines.append(f'  {robot}: waited {total:.2f}s in {count} episode(s)')
        for ep in self.episodes:
            tail = ' (still waiting at end of run)' if ep.ongoing else ''
            lines.append(f'    wait [{ep.start:7.2f}s - {ep.end:7.2f}s] {ep.duration:6.2f}s '
                         f'{ep.robot} at zone {ep.zone} ({ep.x:.2f}, {ep.y:.2f}) '
                         f'blocked by {", ".join(ep.blocked_by)}{tail}')
        if not self.deadlocks:
            lines.append('deadlocks: none')
        for dl in self.deadlocks:
            tail = ' (still active at end of run)' if dl.ongoing else ''
            lines.append(f'  DEADLOCK at zone(s) {", ".join(dl.zones)}: '
                         f'{", ".join(dl.robots)} waited on each other for '
                         f'{dl.duration:.2f}s since {dl.started_at:.2f}s{tail}')
        return '\n'.join(lines)


class FleetSupervisor:
    """Per-step coordinator: route following, local perception, yielding, wait accounting.

    One supervisor drives one run. Wire it into the engine via
    ``engine.run(duration, on_step=supervisor.step)`` and read the outcome with
    ``supervisor.report(end_time)`` (``run_fleet`` does both).
    """

    def __init__(self, routes: dict[str, list[tuple[float, float]]],
                 zones: list[ConflictZone], *, policy: str = 'priority',
                 priorities: dict[str, float] | None = None,
                 sensor_radius: float = 4.0, cruise_speed: float = 0.8,
                 approach_distance: float = 1.5, stop_margin: float = 0.35,
                 deadlock_after: float = 1.0, arrive_tolerance: float = 0.12):
        if policy not in POLICIES:
            raise FleetError(f'unknown policy {policy!r}')
        if not routes:
            raise FleetError('routes must not be empty')
        if sensor_radius < 0 or cruise_speed <= 0 or stop_margin < 0:
            raise FleetError('bad fleet parameters')
        self.policy = policy
        self.routes = {name: [(float(x), float(y)) for x, y in pts]
                       for name, pts in routes.items()}
        for name, pts in self.routes.items():
            if not pts:
                raise FleetError(f'route for {name!r} is empty')
        self.zones = list(zones)
        # Larger numbers win. Default: deterministic name order.
        self.priorities = (dict(priorities) if priorities is not None
                           else {name: rank for rank, name in enumerate(sorted(self.routes))})
        if set(self.priorities) != set(self.routes):
            raise FleetError('priorities must cover exactly the routed robots')
        self.sensor_radius = float(sensor_radius)
        self.cruise_speed = float(cruise_speed)
        self.approach_distance = float(approach_distance)
        self.stop_margin = float(stop_margin)
        self.deadlock_after = float(deadlock_after)
        self.arrive_tolerance = float(arrive_tolerance)
        # -- runtime state ------------------------------------------------------
        self._wp_index = {name: 0 for name in self.routes}
        self._finished = {name: False for name in self.routes}
        self._inside = {name: set() for name in self.routes}
        self._cleared = {name: set() for name in self.routes}
        self._held_zone: dict[str, str] = {}
        self._open_episodes: dict[str, dict] = {}
        self._episodes: list[WaitEpisode] = []
        self._wait_totals = {name: 0.0 for name in self.routes}
        self._cycles: dict[frozenset, float] = {}
        self._open_incidents: dict[frozenset, dict] = {}
        self._incidents: list[DeadlockIncident] = []

    # -- engine hook ------------------------------------------------------------
    def step(self, engine: SimulationEngine) -> None:
        start = engine.time - engine.step  # interval this command covers
        robots = engine.world.robots
        self._update_zone_bookkeeping(robots)
        blocked_by, held_zone = self._decide_holds(robots)
        for name in sorted(robots):
            robot = robots[name]
            if name in blocked_by or self._finished.get(name, False):
                robot.twist = Twist2(0.0, 0.0)
            elif name in self.routes:
                robot.twist = self._drive(name, robot.pose)
        self._account_waits(robots, blocked_by, held_zone, start, engine.step)
        self._track_deadlocks(blocked_by, held_zone, start)

    # -- perception and yield rules ---------------------------------------------
    def _update_zone_bookkeeping(self, robots) -> None:
        for name in self.routes:
            pose = robots[name].pose
            for zone in self.zones:
                edge = zone.edge_distance(pose.x, pose.y)
                if edge <= 0.0:
                    self._inside[name].add(zone.zone_id)
                elif zone.zone_id in self._inside[name] and edge > _EXIT_HYSTERESIS:
                    self._inside[name].discard(zone.zone_id)
                    self._cleared[name].add(zone.zone_id)

    def _relevant_zone(self, name: str, pose: Pose2) -> ConflictZone | None:
        """Nearest zone ahead of the robot. Zones are only relevant while the robot
        is outside them — a robot already inside an intersection drives through."""
        best = None
        for zone in self.zones:
            if zone.zone_id in self._cleared[name] or zone.zone_id in self._inside[name]:
                continue
            edge = zone.edge_distance(pose.x, pose.y)
            if 0.0 < edge <= self.approach_distance:
                if best is None or edge < best[1]:
                    best = (zone, edge)
        return best[0] if best else None

    def _claims(self, other: str, zone: ConflictZone, robots) -> bool:
        """Whether ``other`` currently claims the zone: inside it or closing in on it."""
        pose = robots[other].pose
        edge = zone.edge_distance(pose.x, pose.y)
        if edge <= 0.0:
            return True
        return (zone.zone_id not in self._cleared[other]
                and zone.zone_id not in self._inside[other]
                and edge <= self.approach_distance)

    def _must_yield_to(self, name: str, other: str, robots) -> bool:
        if self.policy == 'priority':
            return self.priorities[other] > self.priorities[name]
        pose, opose = robots[name].pose, robots[other].pose
        bearing = wrap_angle(math.atan2(opose.y - pose.y, opose.x - pose.x) - pose.yaw)
        return _RIGHT_MIN <= bearing <= _RIGHT_MAX

    def _decide_holds(self, robots) -> tuple[dict[str, set], dict[str, str]]:
        blocked_by: dict[str, set] = {}
        held_zone: dict[str, str] = {}
        for name in sorted(self.routes):
            if self._finished[name]:
                continue
            pose = robots[name].pose
            zone = self._relevant_zone(name, pose)
            if zone is None or zone.edge_distance(pose.x, pose.y) > self.stop_margin:
                continue
            blockers = set()
            for other in sorted(self.routes):
                if other == name or self._finished[other]:
                    continue
                opose = robots[other].pose
                if math.hypot(opose.x - pose.x, opose.y - pose.y) > self.sensor_radius:
                    continue  # not perceived: coordination is local, not an oracle
                inside = zone.edge_distance(opose.x, opose.y) <= 0.0
                if inside or (self._claims(other, zone, robots)
                              and self._must_yield_to(name, other, robots)):
                    blockers.add(other)
            if blockers:
                blocked_by[name] = blockers
                held_zone[name] = zone.zone_id
        return blocked_by, held_zone

    # -- route following ----------------------------------------------------------
    def _drive(self, name: str, pose: Pose2) -> Twist2:
        waypoints = self.routes[name]
        idx = self._wp_index[name]
        while True:
            tx, ty = waypoints[idx]
            dx, dy = tx - pose.x, ty - pose.y
            dist = math.hypot(dx, dy)
            if dist > self.arrive_tolerance:
                break
            if idx == len(waypoints) - 1:
                self._finished[name] = True
                return Twist2(0.0, 0.0)
            idx += 1
        self._wp_index[name] = idx
        err = wrap_angle(math.atan2(dy, dx) - pose.yaw)
        angular = max(-2.5, min(2.5, 4.0 * err))
        linear = self.cruise_speed * max(0.0, math.cos(err))
        if dist < 0.6:
            linear *= dist / 0.6  # ease into the final waypoint instead of orbiting it
        return Twist2(linear, angular)

    # -- accounting -----------------------------------------------------------------
    def _account_waits(self, robots, blocked_by, held_zone, start: float, step: float) -> None:
        for name, blockers in blocked_by.items():
            self._wait_totals[name] += step
            episode = self._open_episodes.get(name)
            if episode is None:
                pose = robots[name].pose
                self._open_episodes[name] = {'zone': held_zone[name], 'start': start,
                                             'blockers': set(blockers),
                                             'x': pose.x, 'y': pose.y}
            else:
                episode['blockers'] |= blockers
        for name in list(self._open_episodes):
            if name not in blocked_by:
                episode = self._open_episodes.pop(name)
                self._episodes.append(WaitEpisode(
                    robot=name, zone=episode['zone'], start=round(episode['start'], 9),
                    end=round(start, 9), blocked_by=tuple(sorted(episode['blockers'])),
                    x=round(episode['x'], 9), y=round(episode['y'], 9)))

    def _track_deadlocks(self, blocked_by, held_zone, start: float) -> None:
        current = _cycles_in(blocked_by)
        for key in current:
            self._cycles.setdefault(key, start)
        for key in list(self._cycles):
            if key not in current:
                if key in self._open_incidents:
                    self._close_incident(key, start)
                del self._cycles[key]
        for key in current:
            formed = self._cycles[key]
            if start - formed >= self.deadlock_after and key not in self._open_incidents:
                self._open_incidents[key] = {
                    'started_at': formed, 'detected_at': start,
                    'zones': tuple(sorted({held_zone[r] for r in key}))}

    def _close_incident(self, key: frozenset, end: float) -> None:
        opened = self._open_incidents.pop(key)
        self._incidents.append(DeadlockIncident(
            robots=tuple(sorted(key)), zones=opened['zones'],
            started_at=round(opened['started_at'], 9), detected_at=round(opened['detected_at'], 9),
            ended_at=round(end, 9)))

    # -- reporting --------------------------------------------------------------------
    def report(self, end_time: float) -> FleetReport:
        """Build the report without disturbing state, so a run can be continued."""
        episodes = list(self._episodes)
        for name, ep in sorted(self._open_episodes.items()):
            episodes.append(WaitEpisode(
                robot=name, zone=ep['zone'], start=round(ep['start'], 9),
                end=round(end_time, 9), blocked_by=tuple(sorted(ep['blockers'])),
                x=round(ep['x'], 9), y=round(ep['y'], 9), ongoing=True))
        episodes.sort(key=lambda e: (e.start, e.robot))
        incidents = list(self._incidents)
        for key, opened in sorted(self._open_incidents.items(), key=lambda kv: kv[1]['started_at']):
            incidents.append(DeadlockIncident(
                robots=tuple(sorted(key)), zones=opened['zones'],
                started_at=round(opened['started_at'], 9),
                detected_at=round(opened['detected_at'], 9),
                ended_at=round(end_time, 9), ongoing=True))
        incidents.sort(key=lambda d: d.started_at)
        return FleetReport(policy=self.policy, duration=round(end_time, 9),
                           episodes=tuple(episodes), deadlocks=tuple(incidents),
                           wait_totals={n: round(v, 9) for n, v in sorted(self._wait_totals.items())},
                           finished=dict(sorted(self._finished.items())))


def _cycles_in(graph: dict[str, set]) -> set[frozenset]:
    """Frozensets of robots that form a wait cycle. Fleets are small, so a plain
    DFS from every node with dedup by frozenset is enough."""
    keys: set[frozenset] = set()

    def dfs(node: str, stack: list[str]) -> None:
        for nxt in sorted(graph.get(node, ())):
            if nxt in stack:
                keys.add(frozenset(stack[stack.index(nxt):]))
            else:
                dfs(nxt, stack + [nxt])

    for node in sorted(graph):
        dfs(node, [node])
    return keys


def run_fleet(starts: dict[str, tuple[float, float, float]],
              routes: dict[str, list[tuple[float, float]]],
              zones: list[ConflictZone], *, duration: float, step: float = 0.1,
              seed: int = 0, policy: str = 'priority',
              priorities: dict[str, float] | None = None, sensor_radius: float = 4.0,
              cruise_speed: float = 0.8, deadlock_after: float = 1.0) -> tuple[RunResult, FleetReport]:
    """Run a coordinated fleet scenario and return the engine result plus the report."""
    if set(starts) != set(routes):
        raise FleetError('starts and routes must name the same robots')
    engine = SimulationEngine(step=step, seed=seed)
    for name in sorted(starts):
        x, y, yaw = starts[name]
        engine.add_robot(name, Pose2(float(x), float(y), float(yaw)))
    supervisor = FleetSupervisor(routes, zones, policy=policy, priorities=priorities,
                                 sensor_radius=sensor_radius, cruise_speed=cruise_speed,
                                 deadlock_after=deadlock_after)
    result = engine.run(duration, on_step=supervisor.step)
    return result, supervisor.report(result.duration)
