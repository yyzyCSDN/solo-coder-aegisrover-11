"""Acceptance tests for the simulation and mapping domains."""
import json
import math

import pytest

from aegisrover.core.types import GridShape, Pose2, Twist2, Vec2
from aegisrover.mapping.occupancy import LogOddsGrid
from aegisrover.mapping.revisions import MapFormatError, MapRepository
from aegisrover.sim.engine import SimulationEngine, SimulationError
from aegisrover.sim.fleet import ConflictZone, run_fleet
from aegisrover.sim.scenarios import ScenarioError, ScenarioSpec, ScenarioStore
from aegisrover.storage.audit import AuditLog
from aegisrover.storage.repository import Repository


class Clock:
    def __init__(self, now=1.0):
        self.now = now

    def __call__(self):
        self.now += 0.25
        return self.now


@pytest.fixture()
def repo():
    repository = Repository(':memory:', clock=Clock())
    yield repository
    repository.close()


# ------------------------------------------------------------------------------- engine
def test_engine_is_deterministic_and_seed_sensitive():
    def build(seed):
        engine = SimulationEngine(step=0.1, seed=seed)
        engine.add_robot('r1', Pose2(0, 0, 0), Twist2(1.0, 0.0))
        engine.schedule(0.5, 'set_twist', {'robot': 'r1', 'linear': 0.0, 'angular': 1.0})
        return engine.run(1.0)

    first, second = build(7), build(7)
    assert first.digest == second.digest
    assert first.steps == 10
    assert build(8).digest != first.digest


def test_engine_applies_same_timestamp_events_in_registration_order():
    engine = SimulationEngine(step=0.1)
    engine.add_robot('r1', Pose2(0, 0, 0), Twist2(0.0, 0.0))
    engine.schedule(0.2, 'set_twist', {'robot': 'r1', 'linear': 1.0})
    engine.schedule(0.2, 'set_twist', {'robot': 'r1', 'linear': 0.0})
    result = engine.run(0.3)
    assert [event.sequence for event in result.events_applied] == [1, 2]
    assert result.final_pose('r1')['x'] == pytest.approx(0.0, abs=1e-9)  # last write wins

    reversed_engine = SimulationEngine(step=0.1)
    reversed_engine.add_robot('r1', Pose2(0, 0, 0), Twist2(0.0, 0.0))
    reversed_engine.schedule(0.2, 'set_twist', {'robot': 'r1', 'linear': 0.0})
    reversed_engine.schedule(0.2, 'set_twist', {'robot': 'r1', 'linear': 1.0})
    assert reversed_engine.run(0.3).digest != result.digest


def test_engine_rejects_past_events_and_bad_arguments():
    engine = SimulationEngine(step=0.1)
    engine.add_robot('r1')
    engine.run(0.5)
    with pytest.raises(SimulationError):
        engine.schedule(0.1, 'stop')
    with pytest.raises(SimulationError):
        SimulationEngine(step=0.0)
    with pytest.raises(SimulationError):
        engine.run(0.0)


def test_engine_noise_hook_uses_seed():
    def noise(rng, dt):
        return Twist2(rng.gauss(0, 0.01), 0.0)

    def run(seed):
        engine = SimulationEngine(step=0.2, seed=seed, noise=noise)
        engine.add_robot('r1')
        return engine.run(1.0)

    assert run(3).digest == run(3).digest
    assert run(3).digest != run(4).digest


# ---------------------------------------------------------------------------- scenarios
def spec(**overrides):
    base = dict(scenario_id='yard', duration=1.0, step=0.25, seed=5,
                robots=({'name': 'r1', 'pose': [0.0, 0.0, 0.0], 'twist': [0.5, 0.0]},),
                events=({'time': 0.5, 'kind': 'set_twist', 'payload': {'robot': 'r1', 'linear': 0.0}},))
    base.update(overrides)
    return ScenarioSpec(revision=0, **base)


def test_scenario_store_versions_and_verifies(repo):
    store = ScenarioStore(repo, audit=AuditLog(repo, Clock()), clock=Clock())
    first = store.save(spec(), actor='planner')
    second = store.save(spec(duration=2.0), actor='planner')
    assert (first.revision, second.revision) == (1, 2)
    assert store.latest('yard').revision == 2
    assert store.get('yard', 1).duration == 1.0
    assert store.verify(first) and store.verify(second)
    assert store.diff(first, second)['changed_fields']['duration'] == {'left': 1.0, 'right': 2.0}


def test_scenario_tampering_is_detected(repo):
    store = ScenarioStore(repo, audit=AuditLog(repo, Clock()), clock=Clock())
    saved = store.save(spec(), actor='planner')
    record = repo.get('scenarios', 'yard@0001')
    payload = dict(record.payload)
    payload['duration'] = 9.0
    repo.put('scenarios', 'yard@0001', payload)
    tampered = ScenarioSpec.from_dict(payload)
    assert store.verify(tampered) is False
    with pytest.raises(ScenarioError):
        store.replay(tampered)


def test_scenario_replay_is_reproducible(repo):
    store = ScenarioStore(repo, audit=AuditLog(repo, Clock()), clock=Clock())
    saved = store.save(spec(), actor='planner')
    result = store.require_reproducible(saved)
    assert result.steps == 4
    assert result.events_applied[0].kind == 'set_twist'
    first, second, same = store.replay_twice(saved)
    assert same and first.digest == second.digest


def test_scenario_rejects_unknown_events_and_bad_step(repo):
    store = ScenarioStore(repo, audit=AuditLog(repo, Clock()), clock=Clock())
    bad = store.save(spec(events=({'time': 0.1, 'kind': 'launch_missile', 'payload': {}},)), actor='planner')
    with pytest.raises(ScenarioError):
        store.replay(bad)
    with pytest.raises(ScenarioError):
        store.save(spec(scenario_id='broken', step=0.0), actor='planner')


# ---------------------------------------------------------------------------- occupancy
def grid(width=5, height=5, resolution=1.0):
    return LogOddsGrid(GridShape(width, height, resolution, Vec2(0.0, 0.0)))


def test_occupancy_updates_clamp_and_track_revision():
    g = grid()
    start = g.probability(1, 1)
    g.update(1, 1, True)
    assert g.probability(1, 1) > start
    for _ in range(200):
        g.update(1, 1, True)
    assert g.probability(1, 1) <= g.upper
    for _ in range(400):
        g.update(1, 1, False)
    assert g.probability(1, 1) >= g.lower
    assert g.is_free(1, 1) and g.revision > 0


def test_scan_marks_free_cells_and_occupied_endpoint():
    g = grid(width=6, height=3)
    report = g.integrate_scan(Pose2(0.5, 1.5, 0.0), [3.0],
                              angle_min=0.0, angle_increment=1.0, max_range=10.0)
    assert report.occupied_marked == 1
    assert report.free_marked >= 2
    assert g.is_occupied(3, 1) or g.probability(3, 1) > 0.5
    assert g.is_free(1, 1)


def test_scan_at_max_range_is_not_an_obstacle():
    g = grid(width=4, height=1)
    report = g.integrate_scan(Pose2(0.5, 0.5, 0.0), [10.0],
                              angle_min=0.0, angle_increment=1.0, max_range=10.0)
    assert report.occupied_marked == 0
    assert all(g.probability(x, 0) <= 0.5 for x in range(4))


def test_scan_skips_invalid_beams():
    g = grid()
    report = g.integrate_scan(Pose2(0.5, 0.5, 0.0), [None, float('nan'), 0.01, 2.0],
                              angle_min=0.0, angle_increment=0.3, max_range=5.0)
    assert report.skipped == 3
    assert report.beams == 4


def test_occupancy_payload_roundtrip():
    g = grid()
    g.update(2, 2, True)
    restored = LogOddsGrid.from_payload(g.to_payload())
    assert restored.probability(2, 2) == pytest.approx(g.probability(2, 2))
    assert restored.revision == g.revision


# ---------------------------------------------------------------------------- revisions
def test_map_revisions_history_and_rollback(repo):
    maps = MapRepository(repo, audit=AuditLog(repo, Clock()), clock=Clock())
    first = maps.save('yard', {(0, 0): 1}, actor='robot-1')
    second = maps.save('yard', {(0, 0): 1, (1, 1): 2}, actor='robot-2', expected_revision=1)
    assert (first.revision, second.revision) == (1, 2)
    assert maps.verify(first) and maps.verify(second)
    third = maps.rollback('yard', 1)
    assert third.cells == {'0,0': 1}
    assert maps.diff(second, third)['removed'] == {'1,1': 2}
    with pytest.raises(Exception):
        maps.save('yard', {(9, 9): 1}, expected_revision=1)


def test_three_way_merge_combines_disjoint_edits(repo):
    maps = MapRepository(repo, audit=AuditLog(repo, Clock()), clock=Clock())
    base = maps.save('yard', {'0,0': 1, '1,1': 1}, actor='planner')
    ours = maps.save('yard', {'0,0': 9, '1,1': 1}, actor='robot-1')
    theirs = maps.save('yard', {'0,0': 1, '1,1': 5}, actor='robot-2')
    merged, result = maps.merge_and_save('yard', ours.revision, theirs.revision, actor='merger')
    assert result.clean
    assert result.applied_from_ours == ('0,0',) and result.applied_from_theirs == ('1,1',)
    assert merged.cells == {'0,0': 9, '1,1': 5}
    assert merged.revision == 4 and merged.parent == theirs.revision


def test_three_way_merge_reports_overlapping_edits(repo):
    maps = MapRepository(repo, audit=AuditLog(repo, Clock()), clock=Clock())
    maps.save('yard', {'0,0': 1}, actor='planner')
    ours = maps.save('yard', {'0,0': 9}, actor='robot-1')
    theirs = maps.save('yard', {'0,0': 7}, actor='robot-2')
    result = MapRepository.three_way_merge(maps.get('yard', 1), ours, theirs)
    assert result.conflicts == ('0,0',)
    unchanged, same_result = maps.merge_and_save('yard', ours.revision, theirs.revision)
    assert unchanged.revision == ours.revision
    assert same_result.conflicts == ('0,0',)


def test_map_format_roundtrip_and_migration():
    payload = {'width': 2, 'height': 2, 'resolution': 0.5, 'origin': [1.0, 2.0],
               'cells': {f'{x},{y}': (x + y) for y in range(2) for x in range(2)}}
    text = MapRepository.encode(payload)
    decoded = MapRepository.decode(text)
    assert decoded['cells'] == payload['cells'] and decoded['resolution'] == 0.5
    tampered = json.loads(text)
    tampered['cells']['0,0'] = 42
    with pytest.raises(MapFormatError):
        MapRepository.decode(json.dumps(tampered))
    broken = json.loads(text)
    broken['format'] = 1
    with pytest.raises(MapFormatError):
        MapRepository.decode(json.dumps(broken))
    legacy = json.dumps({'width': 2, 'height': 2, 'cells': [0, 1, 2, 3]})
    migrated = MapRepository.decode(MapRepository.migrate(legacy))
    assert migrated['cells'] == {'0,0': 0, '1,0': 1, '0,1': 2, '1,1': 3}


# ------------------------------------------------------------------------------- fleet coordination
def crossing_world(distance=4.0):
    """Four robots heading straight through one intersection from the four arms."""
    starts = {
        'r-north': (0.0, distance, -math.pi / 2),
        'r-east': (distance, 0.0, math.pi),
        'r-south': (0.0, -distance, math.pi / 2),
        'r-west': (-distance, 0.0, 0.0),
    }
    routes = {
        'r-north': [(0.0, -distance)],
        'r-east': [(-distance, 0.0)],
        'r-south': [(0.0, distance)],
        'r-west': [(distance, 0.0)],
    }
    return starts, routes


def test_engine_on_step_hook_fires_once_per_step():
    engine = SimulationEngine(step=0.1)
    engine.add_robot('r1')
    calls = []
    engine.run(0.5, on_step=lambda eng: calls.append(round(eng.time, 9)))
    assert calls == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_fleet_priority_policy_orders_waits_and_never_deadlocks():
    starts, routes = crossing_world()
    _, report = run_fleet(starts, routes, [ConflictZone('cross', 0.0, 0.0, 1.0)],
                          duration=30.0, policy='priority')
    assert all(report.finished.values())
    assert report.deadlocks == ()
    # default priorities follow name order: r-west crosses first, r-east last
    assert report.waited('r-west') == 0.0
    assert report.waited('r-east') > report.waited('r-north') > report.waited('r-south') > 0.0
    for episode in report.episodes:
        assert episode.zone == 'cross' and episode.duration > 0.0
    for robot in routes:  # episode durations add up to the per-robot totals
        total = sum(e.duration for e in report.episodes if e.robot == robot)
        assert total == pytest.approx(report.waited(robot))


def test_fleet_right_hand_policy_interlocks_and_is_reported():
    starts, routes = crossing_world()
    _, report = run_fleet(starts, routes, [ConflictZone('cross', 0.0, 0.0, 1.0)],
                          duration=20.0, policy='right_hand')
    assert not any(report.finished.values())
    assert len(report.deadlocks) == 1
    deadlock = report.deadlocks[0]
    assert deadlock.robots == ('r-east', 'r-north', 'r-south', 'r-west')
    assert deadlock.zones == ('cross',)
    assert deadlock.ongoing and deadlock.duration > 10.0
    for robot in routes:
        assert report.waited(robot) > 10.0
    json.dumps(report.to_dict())  # the report must stay serialisable


def test_fleet_runs_are_reproducible():
    starts, routes = crossing_world()
    zones = [ConflictZone('cross', 0.0, 0.0, 1.0)]
    first_result, first_report = run_fleet(starts, routes, zones, duration=20.0, policy='right_hand')
    second_result, second_report = run_fleet(starts, routes, zones, duration=20.0, policy='right_hand')
    assert first_result.digest == second_result.digest
    assert first_report.to_dict() == second_report.to_dict()


def test_fleet_without_perception_never_yields():
    starts, routes = crossing_world()
    _, report = run_fleet(starts, routes, [ConflictZone('cross', 0.0, 0.0, 1.0)],
                          duration=15.0, policy='right_hand', sensor_radius=0.0)
    assert all(report.finished.values())
    assert report.episodes == () and report.deadlocks == ()
