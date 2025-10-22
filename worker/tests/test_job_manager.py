from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from timbre_worker.app.jobs import JobManager
from timbre_worker.app.models import GenerationArtifact, GenerationMetadata, GenerationRequest, GenerationStatus, JobState
from timbre_worker.services.riffusion import GenerationFailure


class StubService:
    def __init__(self, artifact_root: Path) -> None:
        self.default_model_id = "riffusion-v1"
        self._artifact_root = artifact_root
        self._artifact_root.mkdir(parents=True, exist_ok=True)

    async def warmup(self) -> None:  # pragma: no cover - not used in tests
        return None

    async def generate(
        self,
        job_id: str,
        request: GenerationRequest,
    ) -> GenerationArtifact:
        artifact_path = self._artifact_root / f"{job_id}.wav"
        artifact_path.write_bytes(b"RIFF")
        metadata = GenerationMetadata(
            prompt=request.prompt,
            seed=request.seed,
            model_id=request.model_id,
            duration_seconds=request.duration_seconds,
            extras={"backend": "stub"},
        )
        return GenerationArtifact(
            job_id=job_id,
            artifact_path=str(artifact_path),
            metadata=metadata,
        )


class FailingService(StubService):
    async def generate(self, job_id: str, request: GenerationRequest) -> GenerationArtifact:
        raise GenerationFailure("boom")


@pytest.mark.asyncio
async def test_job_manager_success_flow(tmp_path: Path) -> None:
    service = StubService(tmp_path)
    manager = JobManager(service)
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
    service = FailingService(tmp_path)
    manager = JobManager(service)
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
