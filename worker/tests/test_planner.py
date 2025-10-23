from timbre_worker.app.models import GenerationRequest
from timbre_worker.services.planner import CompositionPlanner, PLAN_VERSION


def test_planner_builds_deterministic_plan() -> None:
    planner = CompositionPlanner()
    request = GenerationRequest(prompt="lofi piano over rain", duration_seconds=16, model_id="riffusion-v1")
    plan_a = planner.build_plan(request)
    plan_b = planner.build_plan(request)
    assert plan_a == plan_b
    assert plan_a.version == PLAN_VERSION
    assert plan_a.sections
    assert plan_a.sections[0].target_seconds > 0
