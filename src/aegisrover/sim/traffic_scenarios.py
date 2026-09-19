"""Ready-made multi-robot traffic scenarios for intersection/yield studies.

Four networks exercise the questions the traffic layer exists to answer:

* :func:`four_way_cross` — four robots converge on one junction, but routes are
  chosen so no two of them share an arm in opposite directions (the physically
  impossible "four-way simultaneous go-straight" case has its own scenario). Under
  FCFS/main-road priority everyone serialises and finishes; under ``right_yield``
  with simultaneous arrival the four contenders form a permanent cycle, reported as
  an interlock.
* :func:`opposing_straights` — west- and east-bound robots both go straight through
  the same main-road arms. The single lane cannot serve both at once: this is a
  capacity/geometry deadlock that no right-of-way rule can resolve, and the report
  pins it at junction C.
* :func:`narrow_lane` — one single-lane corridor between two junctions, two robots
  enter head-on. One yields at the entry node, the other crosses, then the waiter
  goes: a long wait but no deadlock.
* :func:`convoy_merge` — three robots on the main road plus one joining from a side
  arm; exercises car following, main-road priority and queueing at the junction.

Each builder returns ``(network, robot_specs)`` so the caller picks policy and
config. ``build`` wraps the common TrafficSimulation construction.
"""
from __future__ import annotations

from typing import Sequence

from aegisrover.sim.traffic import (
    RoadNetwork, RobotSpec, TrafficConfig, TrafficSimulation,
)

__all__ = ('four_way_cross', 'opposing_straights', 'narrow_lane', 'convoy_merge',
           'build')


def _cross_network(*, arm_length: float, offset: float,
                   main_road_priority: bool) -> RoadNetwork:
    net = RoadNetwork()
    coords = {
        'C': (0.0, 0.0),
        'W': (-arm_length, 0.0), 'E': (arm_length, 0.0),
        'N': (0.0, arm_length), 'S': (0.0, -arm_length),
        'W2': (-arm_length - offset, 0.0), 'E2': (arm_length + offset, 0.0),
        'N2': (0.0, arm_length + offset), 'S2': (0.0, -arm_length - offset),
    }
    for name, (x, y) in coords.items():
        net.add_node(name, x, y)
    main_edges = {('W2', 'W'), ('W', 'C'), ('C', 'E'), ('E', 'E2')}
    for a, b in [('W2', 'W'), ('W', 'C'), ('C', 'E'), ('E', 'E2'),
                 ('N2', 'N'), ('N', 'C'), ('C', 'S'), ('S', 'S2')]:
        net.add_edge(a, b, priority=main_road_priority and (a, b) in main_edges)
    return net


def four_way_cross(*, arm_length: float = 12.0, offset: float = 12.0,
                   main_road_priority: bool = True
                   ) -> tuple[RoadNetwork, list[RobotSpec]]:
    """Three contenders at a four-arm junction with one arm left empty.

    The empty east arm is the "spillway": r_west can always exit there first, which
    frees the west arm for r_north, which frees the north arm for r_south — a clean
    chain of yielding with no cycle. All three request C simultaneously, so the wait
    differences between FCFS, main-road priority and right-yield are visible.
    """
    net = _cross_network(arm_length=arm_length, offset=offset,
                         main_road_priority=main_road_priority)
    specs = [
        RobotSpec('r_west', ('W2', 'W', 'C', 'E', 'E2')),    # exits into empty arm
        RobotSpec('r_north', ('N2', 'N', 'C', 'W', 'W2')),   # waits for r_west
        RobotSpec('r_south', ('S2', 'S', 'C', 'N', 'N2')),   # waits for r_north
    ]
    return net, specs


def opposing_straights(*, arm_length: float = 12.0, offset: float = 12.0
                       ) -> tuple[RoadNetwork, list[RobotSpec]]:
    """West- and east-bound straight traffic sharing one single-lane main road.

    Two robots approach junction C on the *same* lane from opposite ends and both
    want to continue straight. Only one can physically occupy the lane, so this is
    a geometry-level interlock: the report must identify the two members and C.
    """
    net = _cross_network(arm_length=arm_length, offset=offset,
                         main_road_priority=False)
    specs = [
        RobotSpec('r_west', ('W2', 'W', 'C', 'E', 'E2')),
        RobotSpec('r_east', ('E2', 'E', 'C', 'W', 'W2')),
    ]
    return net, specs


def narrow_lane(*, arm_length: float = 8.0, lane_length: float = 14.0
                ) -> tuple[RoadNetwork, list[RobotSpec]]:
    """Junctions A and B joined by one single lane; robots enter from opposite ends."""
    net = RoadNetwork()
    coords = {
        'A0': (-arm_length - lane_length, -arm_length),
        'A': (-lane_length, 0.0),
        'B': (0.0, 0.0),
        'B0': (lane_length + arm_length, arm_length),
    }
    for name, (x, y) in coords.items():
        net.add_node(name, x, y)
    net.add_edge('A0', 'A')
    net.add_edge('A', 'B')
    net.add_edge('B', 'B0')
    specs = [
        RobotSpec('alpha', ('A0', 'A', 'B', 'B0')),
        RobotSpec('bravo', ('B0', 'B', 'A', 'A0')),
    ]
    return net, specs


def convoy_merge(*, arm_length: float = 10.0, lane_length: float = 12.0
                 ) -> tuple[RoadNetwork, list[RobotSpec]]:
    """Main road W-C-E carrying a convoy of three, plus a side arm joining at C.

    All three convoy robots start queued at W; car following spaces them out as they
    commit onto the lane one after another. The joining robot comes from N.
    """
    net = RoadNetwork()
    coords = {
        'W': (-lane_length, 0.0), 'C': (0.0, 0.0), 'E': (lane_length, 0.0),
        'E2': (lane_length + arm_length, 0.0),
        'N': (0.0, arm_length), 'N2': (0.0, arm_length * 2),
    }
    for name, (x, y) in coords.items():
        net.add_node(name, x, y)
    net.add_edge('W', 'C', priority=True)
    net.add_edge('C', 'E', priority=True)
    net.add_edge('E', 'E2', priority=True)
    net.add_edge('N', 'C')
    net.add_edge('N2', 'N')
    specs = [
        RobotSpec('lead', ('W', 'C', 'E', 'E2')),
        RobotSpec('mid', ('W', 'C', 'E', 'E2')),
        RobotSpec('trail', ('W', 'C', 'E', 'E2')),
        RobotSpec('joiner', ('N2', 'N', 'C', 'E', 'E2')),
    ]
    return net, specs


def build(scenario: str, *, policy: str, config: TrafficConfig | None = None,
          speeds: Sequence[float] | None = None) -> TrafficSimulation:
    """Construct a simulation for one of the named library scenarios."""
    builders = {
        'four_way_cross': four_way_cross,
        'opposing_straights': opposing_straights,
        'narrow_lane': narrow_lane,
        'convoy_merge': convoy_merge,
    }
    if scenario not in builders:
        raise KeyError(f'unknown scenario {scenario!r}')
    net, specs = builders[scenario]()
    if speeds is not None:
        specs = [RobotSpec(s.name, s.route, max_speed=v, radius=s.radius)
                 for s, v in zip(specs, speeds)]
    return TrafficSimulation(net, specs, config or TrafficConfig(policy=policy))
