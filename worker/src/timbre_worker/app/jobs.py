from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Dict, Optional
from uuid import uuid4

from loguru import logger

from ..services.orchestrator import ComposerOrchestrator
from ..services.riffusion import GenerationFailure
from ..services.types import SectionRender
from .models import GenerationArtifact, GenerationRequest, GenerationStatus, JobState


class JobManager:
    """Coordinates asynchronous generation jobs and exposes status artifacts."""

    def __init__(self, orchestrator: ComposerOrchestrator):
        self._orchestrator = orchestrator
        self._statuses: Dict[str, GenerationStatus] = {}
        self._artifacts: Dict[str, GenerationArtifact] = {}
        self._requests: Dict[str, GenerationRequest] = {}
        self._lock = asyncio.Lock()
        self._tasks: Dict[str, asyncio.Task[None]] = {}

    async def enqueue(self, request: GenerationRequest) -> GenerationStatus:
        job_id = str(uuid4())
        status = GenerationStatus(job_id=job_id, state=JobState.QUEUED, message="queued")
        async with self._lock:
            self._statuses[job_id] = status
            self._requests[job_id] = request
        task = asyncio.create_task(self._execute_job(job_id, request))
        async with self._lock:
            self._tasks[job_id] = task
        return status

    async def get_status(self, job_id: str) -> Optional[GenerationStatus]:
        async with self._lock:
            status = self._statuses.get(job_id)
            if status is None:
                return None
            return status.model_copy(deep=True)

    async def get_artifact(self, job_id: str) -> Optional[GenerationArtifact]:
        async with self._lock:
            artifact = self._artifacts.get(job_id)
            if artifact is None:
                return None
            return artifact

    async def _execute_job(self, job_id: str, request: GenerationRequest) -> None:
        await self._set_status(
            job_id,
            state=JobState.RUNNING,
            progress=0.05,
            message="planning composition",
        )
        try:
            async def progress_cb(position: int, total: int, render: SectionRender) -> None:
                ratio = position / max(total, 1)
                progress = 0.2 + 0.6 * ratio
                label = (
                    render.extras.get("section_label")
                    or render.extras.get("section_id")
                    or "section"
                )
                backend = render.extras.get("backend", "backend")
                message = f"rendering {position}/{total}: {label} ({backend})"
                await self._set_status(
                    job_id,
                    state=JobState.RUNNING,
                    progress=progress,
                    message=message,
                )

            async def mix_cb(duration_seconds: float) -> None:
                await self._set_status(
                    job_id,
                    state=JobState.RUNNING,
                    progress=0.9,
                    message=f"assembling mixdown ({duration_seconds:.1f}s)",
                )

            artifact = await self._orchestrator.generate(
                job_id=job_id,
                request=request,
                progress_cb=progress_cb,
                mix_cb=mix_cb,
            )
        except GenerationFailure as exc:
            await self._set_status(
                job_id,
                state=JobState.FAILED,
                progress=1.0,
                message=str(exc),
            )
            logger.error("job {job_id} failed: {exc}", job_id=job_id, exc=exc)
            return
        except Exception:  # noqa: BLE001
            await self._set_status(
                job_id,
                state=JobState.FAILED,
                progress=1.0,
                message="unexpected error during generation",
            )
            logger.exception("unexpected error during job {}", job_id)
            return

        async with self._lock:
            self._artifacts[job_id] = artifact
        await self._set_status(
            job_id,
            state=JobState.SUCCEEDED,
            progress=1.0,
            message="generation complete",
        )
        async with self._lock:
            self._tasks.pop(job_id, None)
            self._requests.pop(job_id, None)

    async def _set_status(
        self,
        job_id: str,
        *,
        state: JobState,
        progress: Optional[float] = None,
        message: Optional[str] = None,
    ) -> None:
        async with self._lock:
            status = self._statuses[job_id]
            status.state = state
            if progress is not None:
                status.progress = max(0.0, min(progress, 1.0))
            status.message = message
            status.updated_at = datetime.now(tz=UTC)
