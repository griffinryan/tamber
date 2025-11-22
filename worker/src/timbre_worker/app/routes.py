from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException, Request

from ..services.planner import PLAN_VERSION
from .jobs import JobManager
from .models import (
    GenerationArtifact,
    GenerationMode,
    GenerationRequest,
    GenerationStatus,
    JobState,
    SessionClipRequest,
    SessionCreateRequest,
    SessionSummary,
)
from .settings import Settings
from .sessions import SessionManager, UnknownSessionError

router = APIRouter()


def get_job_manager(request: Request) -> JobManager:
    return cast(JobManager, request.app.state.job_manager)


def get_session_manager(request: Request) -> SessionManager:
    return cast(SessionManager, request.app.state.session_manager)


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    settings = cast(Settings, request.app.state.settings)
    composer = getattr(request.app.state, "composer", None)
    status_map = getattr(request.app.state, "backend_status", {})
    backend_status: dict[str, object] = {}
    if isinstance(status_map, dict):
        for name, status in status_map.items():
            if hasattr(status, "as_dict"):
                backend_status[name] = status.as_dict()  # type: ignore[attr-defined]
            else:
                backend_status[name] = status
    if composer is not None:
        composer_status = composer.backend_status()
        for name, status in composer_status.items():
            backend_status[name] = status.as_dict()

    available_backends = sorted(backend_status.keys())
    warmup_complete = bool(backend_status) and all(
        isinstance(value, dict) and value.get("ready") for value in backend_status.values()
    )
    session_manager = get_session_manager(request)
    session_summaries = await session_manager.all_summaries()
    return {
        "status": "ok",
        "default_model_id": settings.default_model_id,
        "musicgen_default_model_id": settings.musicgen_default_model_id,
        "artifact_root": str(settings.artifact_root),
        "planner_version": PLAN_VERSION,
        "available_backends": available_backends,
        "backend_status": backend_status,
        "warmup_complete": warmup_complete,
        "session_count": len(session_summaries),
    }


@router.post("/generate", response_model=GenerationStatus)
async def generate(payload: GenerationRequest, request: Request) -> GenerationStatus:
    manager = get_job_manager(request)
    try:
        status = await manager.enqueue(payload)
    except UnknownSessionError as exc:
        raise HTTPException(
            status_code=404, detail=f"session {exc.session_id} not found"
        ) from exc
    return status


@router.get("/status/{job_id}", response_model=GenerationStatus)
async def status(job_id: str, request: Request) -> GenerationStatus:
    manager = get_job_manager(request)
    status = await manager.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job not found")
    return status


@router.get("/artifact/{job_id}", response_model=GenerationArtifact)
async def artifact(job_id: str, request: Request) -> GenerationArtifact:
    manager = get_job_manager(request)
    status = await manager.get_status(job_id)
    if status is None or status.state != JobState.SUCCEEDED:
        raise HTTPException(status_code=404, detail="artifact not available")
    artifact = await manager.get_artifact(job_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="artifact not available")
    return artifact


@router.post("/session", response_model=SessionSummary)
async def create_session(
    payload: SessionCreateRequest, request: Request
) -> SessionSummary:
    manager = get_session_manager(request)
    return await manager.create_session(payload)


@router.get("/session/{session_id}", response_model=SessionSummary)
async def fetch_session(session_id: str, request: Request) -> SessionSummary:
    manager = get_session_manager(request)
    summary = await manager.get_summary(session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="session not found")
    return summary


@router.post("/session/{session_id}/clip", response_model=GenerationStatus)
async def create_session_clip(
    session_id: str, payload: SessionClipRequest, request: Request
) -> GenerationStatus:
    manager = get_job_manager(request)
    session_manager = get_session_manager(request)
    summary = await session_manager.get_summary(session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="session not found")
    if summary.seed_plan is None:
        raise HTTPException(status_code=409, detail="session seed plan not available")

    settings = cast(Settings, request.app.state.settings)
    theme_motif = summary.theme.motif if summary.theme is not None else None
    clip_prompt = payload.prompt or summary.seed_prompt or theme_motif or "Session clip"
    clip_bars = payload.bars

    if payload.generation is not None:
        generation_request = payload.generation.model_copy(deep=True)
        generation_request.session_id = session_id
        generation_request.mode = GenerationMode.CLIP
        generation_request.clip_layer = payload.layer
        generation_request.clip_scene_index = payload.scene_index
        generation_request.clip_bars = clip_bars or generation_request.clip_bars
        generation_request.prompt = clip_prompt
        if generation_request.model_id is None:
            generation_request.model_id = settings.default_model_id
    else:
        generation_request = GenerationRequest(
            prompt=clip_prompt,
            duration_seconds=int(max(1, round(summary.seed_plan.total_duration_seconds))),
            model_id=settings.default_model_id,
            session_id=session_id,
            mode=GenerationMode.CLIP,
            clip_layer=payload.layer,
            clip_scene_index=payload.scene_index,
            clip_bars=clip_bars,
        )

    try:
        status = await manager.enqueue(generation_request)
    except UnknownSessionError as exc:
        raise HTTPException(
            status_code=404, detail=f"session {exc.session_id} not found"
        ) from exc
    return status
