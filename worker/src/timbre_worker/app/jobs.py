from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Dict, Optional
from uuid import uuid4

from loguru import logger

from .models import GenerationArtifact, GenerationRequest, GenerationStatus, JobState
from ..services.orchestrator import ComposerOrchestrator
from ..services.riffusion import GenerationFailure


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
            return status.copy(deep=True)

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
            await self._set_status(
                job_id,
                state=JobState.RUNNING,
                progress=0.2,
                message="rendering sections",
            )
            await self._set_status(
                job_id,
                state=JobState.RUNNING,
                progress=0.6,
                message="assembling mixdown",
            )
            artifact = await self._orchestrator.generate(job_id=job_id, request=request)
        except GenerationFailure as exc:
            await self._set_status(
                job_id,
                state=JobState.FAILED,
                progress=1.0,
                message=str(exc),
            )
            logger.error("job {job_id} failed: {exc}", job_id=job_id, exc=exc)
            return
        except Exception as exc:  # noqa: BLE001
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
