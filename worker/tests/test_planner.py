from timbre_worker.app.models import GenerationRequest, SectionRole
from timbre_worker.services.planner import CompositionPlanner, PLAN_VERSION


def test_planner_builds_deterministic_plan() -> None:
    planner = CompositionPlanner()
    request = GenerationRequest(
        prompt="lofi piano over rain",
        duration_seconds=16,
        model_id="riffusion-v1",
    )
    plan_a = planner.build_plan(request)
    plan_b = planner.build_plan(request)
    assert plan_a == plan_b
    assert plan_a.version == PLAN_VERSION
    assert plan_a.sections
    assert plan_a.sections[0].target_seconds > 0
    assert plan_a.total_duration_seconds + 2.0 >= request.duration_seconds


def test_planner_collapses_short_duration() -> None:
    planner = CompositionPlanner()
    request = GenerationRequest(
        prompt="quick arp",
        duration_seconds=8,
        model_id="riffusion-v1",
    )
    plan = planner.build_plan(request)
    assert len(plan.sections) <= 2
    assert any(section.role == SectionRole.MOTIF for section in plan.sections)
    assert all(section.target_seconds >= 2.0 for section in plan.sections)
