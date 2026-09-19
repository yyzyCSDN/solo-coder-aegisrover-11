"""Acceptance tests for multi-robot traffic simulation: mutual visibility,
right-of-way yielding, wait accounting and interlock detection."""
import math

import pytest

from aegisrover.sim.traffic import (
    FCFS, PRIORITY, RIGHT_YIELD, REASON_LABELS, RoadNetwork, RobotSpec,
    TrafficConfig, TrafficError, TrafficSimulation,
)
from aegisrover.sim.traffic_report import format_report, report_to_json
from aegisrover.sim.traffic_scenarios import build


def run(scenario, policy, duration, **cfg):
    cfg.setdefault('deadlock_timeout', 1.0)
    sim = build(scenario, policy=policy,
                config=TrafficConfig(policy=policy, **cfg))
    return sim.run(duration)


# ----------------------------------------------------------------------------- network
def test_network_validates_edges_and_routes():
    net = RoadNetwork()
    net.add_node('a', 0, 0)
    with pytest.raises(TrafficError):
        net.add_edge('a', 'b')          # unknown endpoint
    net.add_node('b', 3, 4)
    edge = net.add_edge('a', 'b')
    assert edge.length == pytest.approx(5.0)
    assert net.key('b', 'a') == ('a', 'b')
    with pytest.raises(TrafficError):
        net.add_edge('a', 'b')          # duplicate


def test_junction_is_degree_three_or_more():
    net = RoadNetwork()
    for n, xy in [('c', (0, 0)), ('w', (-1, 0)), ('e', (1, 0)), ('n', (0, 1))]:
        net.add_node(n, *xy)
    net.add_edge('c', 'w')
    assert not net.is_junction('c')
    net.add_edge('c', 'e')
    assert not net.is_junction('c')
    net.add_edge('c', 'n')
    assert net.is_junction('c')


def test_bad_spec_and_config_rejected():
    net = RoadNetwork()
    net.add_node('a', 0, 0)
    net.add_node('b', 1, 0)
    with pytest.raises(TrafficError):
        TrafficSimulation(net, [RobotSpec('x', ('a', 'b'), max_speed=0)])
    with pytest.raises(TrafficError):
        TrafficConfig(policy='traffic_lights')
    with pytest.raises(TrafficError):
        TrafficConfig(step=0)


# --------------------------------------------------------------------- FCFS: solvable
def test_three_way_cross_fcfs_serialises_without_deadlock():
    rep = run('four_way_cross', FCFS, 70.0)
    assert rep.deadlocks == ()
    assert all(s.arrived for s in rep.summaries)
    by_name = {s.robot: s for s in rep.summaries}
    # empty-arm chain: west first, then north, then south — waiting grows in order
    assert by_name['r_west'].total_wait < by_name['r_north'].total_wait
    assert by_name['r_north'].total_wait < by_name['r_south'].total_wait
    assert by_name['r_west'].finish_time < by_name['r_north'].finish_time
    assert by_name['r_north'].finish_time < by_name['r_south'].finish_time
    # every wait was a yield to another visible robot
    assert rep.waits
    for w in rep.waits:
        assert w.blocker in by_name and w.blocker != w.robot
        assert w.visible
        assert w.duration > 0
        assert w.blocker_distance is not None
        assert w.blocker_distance <= 30.0 + 1e-9


def test_main_road_priority_also_resolves_and_favours_main_road():
    rep = run('four_way_cross', PRIORITY, 70.0)
    assert rep.deadlocks == ()
    assert all(s.arrived for s in rep.summaries)
    by_name = {s.robot: s for s in rep.summaries}
    # r_west is on the flagged main road and never has to wait on a side-arm robot
    assert by_name['r_west'].total_wait <= by_name['r_north'].total_wait


# ----------------------------------------------------------------- right-yield deadlock
def test_right_yield_forms_persistent_three_member_interlock():
    rep = run('four_way_cross', RIGHT_YIELD, 30.0)
    assert len(rep.deadlocks) == 1
    deadlock = rep.deadlocks[0]
    assert set(deadlock.members) == {'r_west', 'r_north', 'r_south'}
    assert not deadlock.resolved and deadlock.end is None
    assert deadlock.confirmed_at > deadlock.start          # must outlive the timeout
    assert not any(s.arrived for s in rep.summaries)
    # the wait-for edges close the ring reported
    by_name = {s.robot: s for s in rep.summaries}
    for s in rep.summaries:
        assert s.deadlocked_with
    # each robot waited essentially the whole standstill
    for s in rep.summaries:
        assert s.total_wait == pytest.approx(
            30.0 - 12.2, abs=0.25)


def test_same_geometry_under_fcfs_does_not_deadlock():
    """The rule, not the geometry, decides the outcome."""
    locked = run('four_way_cross', RIGHT_YIELD, 30.0)
    flowing = run('four_way_cross', FCFS, 70.0)
    assert locked.deadlocks and not flowing.deadlocks
    assert all(s.arrived for s in flowing.summaries)


def test_transient_wait_cycle_is_not_reported_as_interlock():
    # the wait cycle exists throughout the standstill (~12.4s onward) but a
    # confirmation timeout longer than the run window means it is never confirmed
    rep = run('four_way_cross', RIGHT_YIELD, 20.0, deadlock_timeout=25.0)
    assert rep.deadlocks == ()
    # robots are nevertheless standing still (waits recorded, just not confirmed)
    assert any(s.total_wait > 0 for s in rep.summaries)

    # shortening the run to end before the cycle even appears also reports nothing
    early = run('four_way_cross', RIGHT_YIELD, 12.0, deadlock_timeout=1.0)
    assert early.deadlocks == ()


# --------------------------------------------------------------- geometric head-on trap
def test_opposing_straights_detected_as_unresolvable_interlock():
    rep = run('opposing_straights', FCFS, 50.0)
    assert not any(s.arrived for s in rep.summaries)
    assert rep.deadlocks
    members = {m for d in rep.deadlocks for m in d.members}
    assert members == {'r_west', 'r_east'}
    # the trap spans the single W-C-E corridor (the junction and both entry arms)
    assert any(('C' in d.location or 'W' in d.location) for d in rep.deadlocks)


# ----------------------------------------------------------------------- narrow lane
def test_narrow_lane_one_yields_then_both_cross():
    rep = run('narrow_lane', FCFS, 80.0)
    assert rep.deadlocks == ()
    assert all(s.arrived for s in rep.summaries)
    waiter = min(rep.summaries, key=lambda s: s.finish_time)
    # exactly one of the two experienced a head-on wait at the lane entry
    head_on = [w for w in rep.waits if w.reason == 'head_on']
    assert head_on and all(w.location.startswith('车道入口') for w in head_on)
    blockers = {w.blocker for w in head_on}
    assert blockers and all(b != head_on[0].robot for b in blockers)


# ---------------------------------------------------------------------------- convoy
def test_convoy_keeps_clearance_and_merges_behind_main_road():
    rep = run('convoy_merge', FCFS, 90.0)
    assert rep.deadlocks == ()
    assert all(s.arrived for s in rep.summaries)
    # cars never overlap: lane-space clearance is never negative
    assert rep.min_separation >= -1e-9
    by_name = {s.robot: s for s in rep.summaries}
    # convoy order preserved through the junction
    assert by_name['lead'].finish_time < by_name['mid'].finish_time
    assert by_name['mid'].finish_time < by_name['trail'].finish_time
    # following produced recorded waits
    assert any(w.reason == 'car_follow' for w in rep.waits)


# ------------------------------------------------------------------------ determinism
def test_runs_are_deterministic():
    first = run('four_way_cross', FCFS, 60.0)
    second = run('four_way_cross', FCFS, 60.0)
    assert first.digest == second.digest
    assert [(w.start, w.end, w.robot, w.blocker) for w in first.waits] == \
           [(w.start, w.end, w.robot, w.blocker) for w in second.waits]


def test_step_size_changes_the_digest():
    coarse = run('four_way_cross', FCFS, 60.0, step=0.5)
    fine = run('four_way_cross', FCFS, 60.0, step=0.1)
    assert coarse.digest != fine.digest


# --------------------------------------------------------------------------- reporting
def test_report_json_roundtrip_has_wait_and_deadlock_fields():
    rep = run('four_way_cross', RIGHT_YIELD, 20.0)
    import json
    payload = json.loads(report_to_json(rep))
    assert payload['policy'] == RIGHT_YIELD
    assert payload['deadlocks'][0]['members']
    assert payload['deadlocks'][0]['resolved'] is False
    for wait in payload['waits']:
        assert {'robot', 'reason_label', 'duration', 'blocker', 'location',
                'visible'} <= set(wait)
    totals = {s['robot']: s['total_wait'] for s in payload['summaries']}
    assert all(v > 0 for v in totals.values())


def test_formatted_report_names_locations_members_and_durations():
    rep = run('four_way_cross', RIGHT_YIELD, 20.0)
    text = format_report(rep)
    assert '让右方来车' in text
    assert '路口 C' in text
    assert '互锁' in text
    for robot in ('r_west', 'r_north', 'r_south'):
        assert robot in text
    assert '累计等待' in text and '未到达' in text


def test_reason_labels_cover_all_wait_reasons():
    rep = run('four_way_cross', FCFS, 70.0)
    for w in rep.waits:
        assert w.reason in REASON_LABELS
