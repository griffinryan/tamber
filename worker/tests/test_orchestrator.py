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
    SectionOrchestration,
    SectionRole,
)
from timbre_worker.app.settings import Settings
from timbre_worker.services.exceptions import GenerationFailure
from timbre_worker.services.orchestrator import ComposerOrchestrator
from timbre_worker.services.types import BackendStatus, SectionRender


class PassthroughPlanner:
    def build_plan(self, request: GenerationRequest) -> CompositionPlan:
        assert request.plan is not None
        return request.plan


class DummyMusicGen:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: List[str] = []

    async def warmup(self) -> BackendStatus:  # pragma: no cover - lightweight status
        return BackendStatus(
            name=self.name,
            ready=True,
            device="cpu",
            dtype=None,
            error=None,
            details={},
        )

    async def render_section(
        self,
        request: GenerationRequest,
        section: CompositionSection,
        *,
        plan: CompositionPlan,
        model_id: str | None = None,
        render_seconds: float | None = None,
        theme=None,
        previous_render: SectionRender | None = None,
        **_: object,
    ) -> SectionRender:
        self.calls.append(section.section_id)
        sample_rate = 44100
        duration = render_seconds if render_seconds is not None else section.target_seconds
        samples = max(1, int(round(duration * sample_rate)))
        waveform = np.linspace(0.0, 1.0, samples, dtype=np.float32)
        extras = {
            "backend": self.name,
            "section_id": section.section_id,
            "section_label": section.label,
            "placeholder": False,
            "render_seconds": duration,
        }
        return SectionRender(waveform=waveform, sample_rate=sample_rate, extras=extras)


class PlaceholderBackend(DummyMusicGen):
    async def render_section(
        self,
        request: GenerationRequest,
        section: CompositionSection,
        *,
        plan: CompositionPlan,
        model_id: str | None = None,
        render_seconds: float | None = None,
        **kwargs: object,
    ) -> SectionRender:
        render = await super().render_section(
            request,
            section,
            plan=plan,
            model_id=model_id,
            render_seconds=render_seconds,
            **kwargs,
        )
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
            model_id="musicgen-stereo-medium",
            seed_offset=0,
            transition=None,
            orchestration=SectionOrchestration(),
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
            orchestration=SectionOrchestration(),
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
    musicgen = DummyMusicGen("musicgen")
    settings = Settings(
        artifact_root=tmp_path / "artifacts",
        config_dir=tmp_path / "config",
    )
    settings.ensure_directories()

    orchestrator = ComposerOrchestrator(settings, planner, musicgen)

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

    backend_status = orchestrator.backend_status()
    assert backend_status.get("musicgen") is not None

    assert Path(artifact.artifact_path).exists()
    assert len(progress_calls) == 1
    assert progress_calls[0][2] == "s01"
    assert mix_calls, "mix callback should be invoked"

    extras = artifact.metadata.extras
    mix_info = extras.get("mix", {})
    assert extras.get("sample_rate") == settings.export_sample_rate
    assert extras.get("bit_depth") == settings.export_bit_depth
    assert extras.get("format") == settings.export_format
    assert mix_info.get("sample_rate") == settings.export_sample_rate
    assert mix_calls[0] == pytest.approx(mix_info.get("duration_seconds", 0.0), rel=1e-3)
    assert extras.get("backend") == "composer"
    assert extras.get("placeholder") is False
    assert mix_info.get("target_rms", 0.0) == pytest.approx(0.2, rel=1e-3)
    section_rms = mix_info.get("section_rms")
    assert isinstance(section_rms, list)
    assert len(section_rms) == 1
    sections = extras.get("sections")
    assert isinstance(sections, list)
    assert len(sections) == 1
    assert sections[0].get("section_id") == "s01"
    assert sections[0].get("render_seconds", 0.0) > 0.0
    assert isinstance(sections[0].get("arrangement_text"), str)
    assert mix_info.get("crossfades", []) == []
    motif_seed = extras.get("motif_seed")
    assert isinstance(motif_seed, dict)
    assert motif_seed.get("captured") is True
    assert motif_seed.get("section_id") == "s01"
    assert Path(motif_seed.get("path", "")).exists()
    assert extras.get("full_plan") is not None
    preview_plan = extras.get("motif_preview_plan")
    assert isinstance(preview_plan, dict)
    assert preview_plan.get("sections", [{}])[0].get("section_id") == "s01"
    assert extras.get("motif_preview_seconds", 0.0) == pytest.approx(
        mix_info.get("duration_seconds", 0.0), rel=1e-3
    )
    assert artifact.metadata.plan is not None
    assert len(artifact.metadata.plan.sections) == 2
    assert musicgen.calls == ["s01"]


@pytest.mark.asyncio
async def test_orchestrator_warmup_reports_backend_status(tmp_path: Path) -> None:
    planner = PassthroughPlanner()
    musicgen = DummyMusicGen("musicgen")
    settings = Settings(
        artifact_root=tmp_path / "artifacts",
        config_dir=tmp_path / "config",
    )
    settings.ensure_directories()

    orchestrator = ComposerOrchestrator(settings, planner, musicgen)
    statuses = await orchestrator.warmup()
    assert statuses.get("musicgen") is not None
    assert orchestrator.backend_status().keys() >= {"musicgen"}


@pytest.mark.asyncio
async def test_orchestrator_marks_placeholder(tmp_path: Path) -> None:
    planner = PassthroughPlanner()
    musicgen = PlaceholderBackend("musicgen")
    settings = Settings(
        artifact_root=tmp_path / "artifacts",
        config_dir=tmp_path / "config",
    )
    settings.ensure_directories()

    orchestrator = ComposerOrchestrator(settings, planner, musicgen)

    plan = build_plan(duration=4.0)
    request = build_request(plan)

    artifact = await orchestrator.generate(job_id="placeholder-job", request=request)

    extras = artifact.metadata.extras
    assert extras.get("placeholder") is True
    sections = extras.get("sections")
    assert isinstance(sections, list)
    assert len(sections) == 1
    assert sections[0].get("placeholder") is True
    motif_seed = extras.get("motif_seed")
    assert isinstance(motif_seed, dict)
    assert motif_seed.get("captured") is True
    mix_info = extras.get("mix", {})
    assert mix_info.get("crossfades", []) == []
    assert extras.get("full_plan") is not None


@pytest.mark.asyncio
async def test_orchestrator_rejects_unsupported_models(tmp_path: Path) -> None:
    planner = PassthroughPlanner()
    musicgen = DummyMusicGen("musicgen")
    settings = Settings(
        artifact_root=tmp_path / "artifacts",
        config_dir=tmp_path / "config",
    )
    settings.ensure_directories()

    orchestrator = ComposerOrchestrator(settings, planner, musicgen)

    plan = build_plan()
    for section in plan.sections:
        section.model_id = "riffusion-v1"
    request = build_request(plan)

    with pytest.raises(GenerationFailure):
        await orchestrator.generate(job_id="unsupported", request=request)
