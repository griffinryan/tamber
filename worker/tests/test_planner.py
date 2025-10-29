from timbre_worker.app.models import GenerationMode, GenerationRequest, SectionRole
from timbre_worker.services.planner import PLAN_VERSION, CompositionPlanner


def test_planner_builds_deterministic_plan() -> None:
    planner = CompositionPlanner()
    request = GenerationRequest(
        prompt="lofi piano over rain",
        duration_seconds=16,
        model_id="musicgen-stereo-medium",
    )
    plan_a = planner.build_plan(request)
    plan_b = planner.build_plan(request)
    assert plan_a == plan_b
    assert plan_a.version == PLAN_VERSION
    assert plan_a.sections
    assert plan_a.sections[0].target_seconds > 0
    assert plan_a.total_duration_seconds + 2.0 >= request.duration_seconds
    assert plan_a.theme is not None
    assert plan_a.theme.instrumentation
    assert all(section.motif_directive is not None for section in plan_a.sections)
    assert any(section.motif_directive == "state motif" for section in plan_a.sections)
    assert all(isinstance(section.variation_axes, list) for section in plan_a.sections)


def test_planner_collapses_short_duration() -> None:
    planner = CompositionPlanner()
    request = GenerationRequest(
        prompt="quick arp",
        duration_seconds=8,
        model_id="musicgen-stereo-medium",
    )
    plan = planner.build_plan(request)
    assert len(plan.sections) <= 2
    assert any(section.role == SectionRole.MOTIF for section in plan.sections)
    assert all(section.target_seconds >= 2.0 for section in plan.sections)
    assert plan.theme is not None
    assert all(section.motif_directive is not None for section in plan.sections)


def test_planner_builds_motif_mode_plan() -> None:
    planner = CompositionPlanner()
    request = GenerationRequest(
        prompt="motif spotlight",
        duration_seconds=12,
        model_id="musicgen-stereo-medium",
        mode=GenerationMode.MOTIF,
    )
    plan = planner.build_plan(request)
    assert len(plan.sections) == 1
    section = plan.sections[0]
    assert section.role == SectionRole.MOTIF
    assert section.target_seconds >= 2.0
    assert section.target_seconds <= 25.0
    assert section.motif_directive == "state motif"
    assert plan.theme is not None
