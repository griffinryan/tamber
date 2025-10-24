from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from timbre_worker.app.jobs import JobManager
from timbre_worker.app.models import (
    CompositionPlan,
    CompositionSection,
    GenerationArtifact,
    GenerationMetadata,
    GenerationRequest,
    GenerationStatus,
    JobState,
    SectionEnergy,
    SectionOrchestration,
    SectionRole,
)
from timbre_worker.services.riffusion import GenerationFailure
from timbre_worker.services.types import SectionRender


class StubOrchestrator:
    def __init__(self, artifact_root: Path) -> None:
        self._artifact_root = artifact_root
        self._artifact_root.mkdir(parents=True, exist_ok=True)

    async def warmup(self) -> None:  # pragma: no cover - not used in tests
        return None

    async def generate(
        self,
        job_id: str,
        request: GenerationRequest,
        progress_cb=None,
        mix_cb=None,
    ) -> GenerationArtifact:
        artifact_path = self._artifact_root / f"{job_id}.wav"
        artifact_path.write_bytes(b"RIFF")
        if progress_cb is not None:
            await progress_cb(
                1,
                1,
                SectionRender(
                    waveform=np.zeros(10, dtype=np.float32),
                    sample_rate=44100,
                    extras={
                        "backend": "stub",
                        "section_id": "s00",
                        "section_label": "Test",
                        "placeholder": False,
                    },
                ),
            )
        if mix_cb is not None:
            await mix_cb(float(request.duration_seconds))
        metadata = GenerationMetadata(
            prompt=request.prompt,
            seed=request.seed,
            model_id=request.model_id,
            duration_seconds=request.duration_seconds,
            extras={"backend": "stub"},
            plan=self._plan(request),
        )
        return GenerationArtifact(
            job_id=job_id,
            artifact_path=str(artifact_path),
            metadata=metadata,
        )

    def _plan(self, request: GenerationRequest) -> CompositionPlan:
        section = CompositionSection(
            section_id="s00",
            role=SectionRole.MOTIF,
            label="Test",
            prompt=request.prompt,
            bars=4,
            target_seconds=float(request.duration_seconds),
            energy=SectionEnergy.MEDIUM,
            model_id=request.model_id,
            seed_offset=0,
            transition=None,
            orchestration=SectionOrchestration(),
        )
        return CompositionPlan(
            version="test",
            tempo_bpm=90,
            time_signature="4/4",
            key="C major",
            total_bars=4,
            total_duration_seconds=float(request.duration_seconds),
            sections=[section],
        )


class FailingOrchestrator(StubOrchestrator):
    async def generate(
        self,
        job_id: str,
        request: GenerationRequest,
        progress_cb=None,
        mix_cb=None,
    ) -> GenerationArtifact:
        raise GenerationFailure("boom")


@pytest.mark.asyncio
async def test_job_manager_success_flow(tmp_path: Path) -> None:
    orchestrator = StubOrchestrator(tmp_path)
    manager = JobManager(orchestrator)
    request = GenerationRequest(prompt="hello", duration_seconds=2, model_id="riffusion-v1")

    status = await manager.enqueue(request)
    assert status.state == JobState.QUEUED

    result = await _wait_for_terminal_state(manager, status.job_id)
    assert result.state == JobState.SUCCEEDED
    artifact = await manager.get_artifact(status.job_id)
    assert artifact is not None
    assert Path(artifact.artifact_path).exists()


@pytest.mark.asyncio
async def test_job_manager_failure_flow(tmp_path: Path) -> None:
    orchestrator = FailingOrchestrator(tmp_path)
    manager = JobManager(orchestrator)
    request = GenerationRequest(prompt="oops", duration_seconds=2, model_id="riffusion-v1")

    status = await manager.enqueue(request)
    assert status.state == JobState.QUEUED

    result = await _wait_for_terminal_state(manager, status.job_id)
    assert result.state == JobState.FAILED
    artifact = await manager.get_artifact(status.job_id)
    assert artifact is None


async def _wait_for_terminal_state(manager: JobManager, job_id: str) -> GenerationStatus:
    for _ in range(60):
        status = await manager.get_status(job_id)
        if status is None:
            await asyncio.sleep(0.05)
            continue
        if status.state in {JobState.SUCCEEDED, JobState.FAILED}:
            return status
        await asyncio.sleep(0.05)
    raise AssertionError("job did not complete within timeout")
