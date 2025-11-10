from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from typing import Dict, Optional
from uuid import uuid4

from loguru import logger

from ..services.exceptions import GenerationFailure
from ..services.orchestrator import ComposerOrchestrator
from ..services.planner import CompositionPlanner
from ..services.types import SectionRender
from .models import (
    ClipLayer,
    GenerationArtifact,
    GenerationMode,
    GenerationRequest,
    GenerationStatus,
    JobState,
)
from .sessions import SessionManager, UnknownSessionError


class JobManager:
    """Coordinates asynchronous generation jobs and exposes status artifacts."""

    def __init__(
        self,
        orchestrator: ComposerOrchestrator,
        planner: CompositionPlanner,
        sessions: SessionManager,
    ):
        self._orchestrator = orchestrator
        self._planner = planner
        self._sessions = sessions
        self._statuses: Dict[str, GenerationStatus] = {}
        self._artifacts: Dict[str, GenerationArtifact] = {}
        self._requests: Dict[str, GenerationRequest] = {}
        self._lock = asyncio.Lock()
        self._tasks: Dict[str, asyncio.Task[None]] = {}

    async def enqueue(self, request: GenerationRequest) -> GenerationStatus:
        job_id = str(uuid4())
        status = GenerationStatus(job_id=job_id, state=JobState.QUEUED, message="queued")

        if request.session_id is not None:
            exists = await self._sessions.session_exists(request.session_id)
            if not exists:
                raise UnknownSessionError(request.session_id)
            if request.mode == GenerationMode.CLIP or request.clip_layer is not None:
                await self._prepare_clip_request(request)


        async with self._lock:
            self._statuses[job_id] = status
            self._requests[job_id] = request
        task = asyncio.create_task(self._execute_job(job_id, request))
        async with self._lock:
            self._tasks[job_id] = task

        if request.session_id is not None:
            await self._sessions.register_job(
                request.session_id,
                job_id,
                request.prompt,
                layer=request.clip_layer,
                scene_index=request.clip_scene_index,
                bars=request.clip_bars,
            )

        return status

    async def _prepare_clip_request(self, request: GenerationRequest) -> None:
        session_id = request.session_id
        if session_id is None:
            raise RuntimeError("clip requests require a session")
        summary = await self._sessions.get_summary(session_id)
        if summary is None:
            raise UnknownSessionError(session_id)
        seed_plan = summary.seed_plan
        if seed_plan is None:
            raise RuntimeError("session seed plan not ready")

        clip_layer = request.clip_layer or ClipLayer.RHYTHM
        bars = request.clip_bars or 4
        prompt = request.prompt or summary.seed_prompt or "Session clip"
        request.prompt = prompt

        plan = request.plan
        if plan is None:
            plan = self._planner.build_clip_plan(
                seed_plan=seed_plan,
                session_theme=summary.theme or seed_plan.theme,
                clip_prompt=prompt,
                layer=clip_layer,
                bars=bars,
            )
            request.plan = plan

        request.mode = GenerationMode.CLIP
        request.clip_layer = clip_layer
        request.clip_bars = request.clip_bars or min(int(plan.total_bars), 64)

        loop_seconds = max(1.0, float(plan.total_duration_seconds))
        request.duration_seconds = max(1, int(round(loop_seconds)))

        if request.seed is None:
            request.seed = self._derive_clip_seed(
                session_id=session_id,
                layer=clip_layer,
                scene_index=request.clip_scene_index,
                prompt=prompt,
            )

    @staticmethod
    def _derive_clip_seed(
        *,
        session_id: str,
        layer: ClipLayer,
        scene_index: Optional[int],
        prompt: str,
    ) -> int:
        scene_fragment = str(scene_index) if scene_index is not None else "none"
        payload = f"{session_id}|{layer.value}|{scene_fragment}|{prompt}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        return int.from_bytes(digest[:8], "little") & ((1 << 62) - 1)

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
        session_id = request.session_id
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
        if session_id is not None:
            await self._sessions.record_artifact(session_id, job_id, artifact)
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
        session_id: Optional[str] = None
        async with self._lock:
            status = self._statuses[job_id]
            status.state = state
            if progress is not None:
                status.progress = max(0.0, min(progress, 1.0))
            status.message = message
            status.updated_at = datetime.now(tz=UTC)
            request = self._requests.get(job_id)
            if request is not None:
                session_id = request.session_id
        if session_id is not None:
            await self._sessions.mark_job_state(session_id, job_id, state)
