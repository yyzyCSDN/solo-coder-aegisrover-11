"""Command-line entry: run a shared-network traffic scenario and report waits.

Examples
--------
python -m aegisrover.sim.traffic_cli four_way_cross --policy fcfs --duration 70
python -m aegisrover.sim.traffic_cli four_way_cross --policy right_yield
python -m aegisrover.sim.traffic_cli opposing_straights --json
"""
from __future__ import annotations

import argparse
import sys

from aegisrover.sim.traffic import FCFS, PRIORITY, RIGHT_YIELD, TrafficConfig
from aegisrover.sim.traffic_report import format_report, report_to_json
from aegisrover.sim.traffic_scenarios import build

_SCENARIOS = ('four_way_cross', 'opposing_straights', 'narrow_lane', 'convoy_merge')
_POLICIES = {FCFS, PRIORITY, RIGHT_YIELD}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='traffic-sim',
        description='多机器人共享路网让行仿真：报告等待位置、等待时长与互锁。')
    parser.add_argument('scenario', choices=_SCENARIOS)
    parser.add_argument('--policy', choices=sorted(_POLICIES), default=FCFS)
    parser.add_argument('--duration', type=float, default=80.0, help='仿真时长（秒）')
    parser.add_argument('--step', type=float, default=0.2, help='步长（秒）')
    parser.add_argument('--vision', type=float, default=30.0, help='互相看见距离（米）')
    parser.add_argument('--deadlock-timeout', type=float, default=1.0,
                        help='等待环持续多久判定为互锁（秒）')
    parser.add_argument('--json', action='store_true', help='输出机器可读 JSON')
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = TrafficConfig(
        step=args.step, vision_range=args.vision,
        deadlock_timeout=args.deadlock_timeout, policy=args.policy)
    simulation = build(args.scenario, policy=args.policy, config=config)
    report = simulation.run(args.duration)
    print(report_to_json(report) if args.json else format_report(report))
    # exit non-zero only when a confirmed interlock is still unresolved at the end
    unresolved = any(not d.resolved for d in report.deadlocks)
    return 2 if unresolved else 0


if __name__ == '__main__':
    sys.exit(main())
