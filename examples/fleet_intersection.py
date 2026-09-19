"""Fleet coordination demo: four robots, one intersection, two yield policies.

Run from a checkout with no install:

    PYTHONPATH=src python3 examples/fleet_intersection.py

The same four robots cross one intersection twice. First the ``priority``
policy orders them through with measurable waits; then the ``right_hand``
policy lets them arrive simultaneously and interlock, and the report names the
deadlocked robots, the zone, and how long each of them has been waiting.
"""
import math
import sys
from pathlib import Path

try:
    from aegisrover.sim.fleet import ConflictZone, run_fleet
except ModuleNotFoundError:  # allow running straight from a source checkout
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from aegisrover.sim.fleet import ConflictZone, run_fleet

ZONE = ConflictZone('cross', 0.0, 0.0, 1.0)
ARM = 4.0
STARTS = {
    'r-north': (0.0, ARM, -math.pi / 2),   # above the crossing, heading south
    'r-east': (ARM, 0.0, math.pi),         # right of the crossing, heading west
    'r-south': (0.0, -ARM, math.pi / 2),   # below the crossing, heading north
    'r-west': (-ARM, 0.0, 0.0),            # left of the crossing, heading east
}
ROUTES = {
    'r-north': [(0.0, -ARM)],
    'r-east': [(-ARM, 0.0)],
    'r-south': [(0.0, ARM)],
    'r-west': [(ARM, 0.0)],
}


def main() -> None:
    print('=== priority policy: a total order, waits but no deadlock ===')
    _, report = run_fleet(STARTS, ROUTES, [ZONE], duration=30.0, policy='priority')
    print(report.summary())

    print()
    print('=== right_hand policy: symmetric arrivals interlock ===')
    _, report = run_fleet(STARTS, ROUTES, [ZONE], duration=20.0, policy='right_hand')
    print(report.summary())


if __name__ == '__main__':
    main()
