from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest

from timbre_worker.app.models import (
    CompositionPlan,
    CompositionSection,
    GenerationRequest,
    SectionEnergy,
    SectionRole,
)
from timbre_worker.app.settings import Settings
from timbre_worker.services.orchestrator import ComposerOrchestrator
from timbre_worker.services.types import SectionRender


class PassthroughPlanner:
    def build_plan(self, request: GenerationRequest) -> CompositionPlan:
        assert request.plan is not None
        return request.plan


class DummyBackend:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: List[str] = []

    async def warmup(self) -> None:  # pragma: no cover - no-op for tests
        return None

    async def render_section(
        self,
        request: GenerationRequest,
        section: CompositionSection,
        *,
        plan: CompositionPlan,
        model_id: str | None = None,
    ) -> SectionRender:
        self.calls.append(section.section_id)
        sample_rate = 44100
        samples = max(1, int(round(section.target_seconds * sample_rate)))
        waveform = np.linspace(0.0, 1.0, samples, dtype=np.float32)
        extras = {
            "backend": self.name,
            "section_id": section.section_id,
            "section_label": section.label,
            "placeholder": False,
        }
        return SectionRender(waveform=waveform, sample_rate=sample_rate, extras=extras)


class PlaceholderBackend(DummyBackend):
    async def render_section(
        self,
        request: GenerationRequest,
        section: CompositionSection,
        *,
        plan: CompositionPlan,
        model_id: str | None = None,
    ) -> SectionRender:
        render = await super().render_section(request, section, plan=plan, model_id=model_id)
        render.extras["placeholder"] = True
        render.extras["placeholder_reason"] = "missing"
        return render


def build_plan(duration: float = 6.0) -> CompositionPlan:
    sections = [
        CompositionSection(
            section_id="s00",
            role=SectionRole.INTRO,
            label="Intro",
            prompt="start",
            bars=2,
            target_seconds=duration / 2,
            energy=SectionEnergy.LOW,
            model_id="riffusion-v1",
            seed_offset=0,
            transition=None,
        ),
        CompositionSection(
            section_id="s01",
            role=SectionRole.MOTIF,
            label="Theme",
            prompt="middle",
            bars=2,
            target_seconds=duration / 2,
            energy=SectionEnergy.MEDIUM,
            model_id="musicgen-small",
            seed_offset=1,
            transition=None,
        ),
    ]
    return CompositionPlan(
        version="test",
        tempo_bpm=90,
        time_signature="4/4",
        key="C major",
        total_bars=4,
        total_duration_seconds=duration,
        sections=sections,
    )


def build_request(plan: CompositionPlan) -> GenerationRequest:
    return GenerationRequest(
        prompt="lofi piano",
        duration_seconds=int(round(plan.total_duration_seconds)),
        model_id="composer",
        plan=plan,
    )


@pytest.mark.asyncio
async def test_orchestrator_combines_sections(tmp_path: Path) -> None:
    planner = PassthroughPlanner()
    riffusion = DummyBackend("riffusion")
    musicgen = DummyBackend("musicgen")
    settings = Settings(artifact_root=tmp_path / "artifacts", config_dir=tmp_path / "config")
    settings.ensure_directories()

    orchestrator = ComposerOrchestrator(settings, planner, riffusion, musicgen)

    plan = build_plan()
    request = build_request(plan)

    progress_calls: list[tuple[int, int, str]] = []
    mix_calls: list[float] = []

    async def progress_cb(position: int, total: int, render: SectionRender) -> None:
        progress_calls.append((position, total, render.extras.get("section_id", "")))

    async def mix_cb(duration_seconds: float) -> None:
        mix_calls.append(duration_seconds)

    artifact = await orchestrator.generate(
        job_id="test-job",
        request=request,
        progress_cb=progress_cb,
        mix_cb=mix_cb,
    )

    assert Path(artifact.artifact_path).exists()
    assert len(progress_calls) == len(plan.sections)
    assert progress_calls[0][2] == "s00"
    assert progress_calls[1][2] == "s01"
    assert pytest.approx(mix_calls[0], rel=1e-3) == plan.total_duration_seconds

    extras = artifact.metadata.extras
    assert extras.get("backend") == "composer"
    assert extras.get("placeholder") is False
    sections = extras.get("sections")
    assert isinstance(sections, list)
    assert len(sections) == 2


@pytest.mark.asyncio
async def test_orchestrator_marks_placeholder(tmp_path: Path) -> None:
    planner = PassthroughPlanner()
    riffusion = PlaceholderBackend("riffusion")
    musicgen = DummyBackend("musicgen")
    settings = Settings(artifact_root=tmp_path / "artifacts", config_dir=tmp_path / "config")
    settings.ensure_directories()

    orchestrator = ComposerOrchestrator(settings, planner, riffusion, musicgen)

    plan = build_plan(duration=4.0)
    request = build_request(plan)

    artifact = await orchestrator.generate(job_id="placeholder-job", request=request)

    extras = artifact.metadata.extras
    assert extras.get("placeholder") is True
    sections = extras.get("sections")
    assert isinstance(sections, list)
    assert any(entry.get("placeholder") for entry in sections)
