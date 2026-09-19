"""Human-readable rendering of a :class:`TrafficReport`.

The report dataclasses are machine-oriented (and JSON-serialisable via
``to_dict``). This module turns one into the summary an operator wants after a
run: where each robot waited, for how long, waiting on whom, and whether a wait
cycle turned into a confirmed interlock.
"""
from __future__ import annotations

import json

from aegisrover.sim.traffic import REASON_LABELS, TrafficReport

__all__ = ('format_report', 'report_to_json')

_POLICY_LABELS = {
    'fcfs': '先到先行 (FCFS)',
    'priority': '主路优先',
    'right_yield': '让右方来车',
}


def format_report(report: TrafficReport) -> str:
    lines: list[str] = []
    policy = _POLICY_LABELS.get(report.policy, report.policy)
    lines.append(f'多机交通仿真报告（让行规则：{policy}，步长 {report.steps and round(report.duration / report.steps, 3)}s，'
                 f'共 {report.steps} 步 / {round(report.duration, 2)}s）')
    lines.append('')

    lines.append('各机器人等待汇总')
    for s in report.summaries:
        status = f'已到达 @{round(s.finish_time, 2)}s' if s.arrived else '未到达'
        locked = ('，互锁对象：' + '、'.join(s.deadlocked_with)) if s.deadlocked_with else ''
        lines.append(f'  - {s.robot}: 累计等待 {round(s.total_wait, 2)}s，'
                     f'等待 {s.wait_count} 段，{status}{locked}')
    lines.append('')

    lines.append('逐段等待事件（等待发生的位置与让行对象）')
    if not report.waits:
        lines.append('  （无等待）')
    for w in report.waits:
        sight = '在视野内看见' if w.visible else '视野外（按规则预判）'
        distance = '' if w.blocker_distance is None else f'，相距 {round(w.blocker_distance, 2)}m'
        lines.append(
            f'  - {w.start:6.2f}s–{w.end:6.2f}s（{round(w.duration, 2):5.2f}s） '
            f'{w.robot} 在 {w.location} {REASON_LABELS[w.reason]}，让行 {w.blocker}'
            f'（{sight}{distance}）')
    lines.append('')

    lines.append('互锁（等待图中持续存在的环）')
    if not report.deadlocks:
        lines.append('  未检测到互锁。')
    else:
        for d in report.deadlocks:
            members = ' → '.join(d.members + (d.members[0],))
            if d.resolved:
                outcome = (f'已于 {round(d.end, 2)}s 自行解开，确认互锁持续 '
                           f'{round(d.duration, 2)}s')
            else:
                outcome = '仿真结束仍未解开（需要外部介入，如重置或信号灯）'
            lines.append(
                f'  - 位置：{d.location}；成员：{members}；等待环出现于 '
                f'{round(d.start, 2)}s，超过阈值在 {round(d.confirmed_at, 2)}s 判定互锁；{outcome}')
    lines.append('')

    sep = '无（单车道占用互斥得到保证）' if report.min_separation == float("inf") else f'{round(report.min_separation, 3)}m'
    lines.append(f'全程最小车间净距：{sep}')
    lines.append(f'轨迹摘要 digest：{report.digest[:16]}…')
    return '\n'.join(lines)


def report_to_json(report: TrafficReport, *, indent: int = 2) -> str:
    return json.dumps(report.to_dict(), ensure_ascii=False, indent=indent)
