from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Dict, Optional
from uuid import uuid4

from .models import (
    ClipLayer,
    CompositionPlan,
    GenerationArtifact,
    JobState,
    SessionClipSummary,
    SessionCreateRequest,
    SessionSummary,
    ThemeDescriptor,
)


class UnknownSessionError(Exception):
    """Raised when a session lookup fails."""

    def __init__(self, session_id: str) -> None:
        super().__init__(session_id)
        self.session_id = session_id


@dataclass
class SessionClip:
    job_id: str
    prompt: str
    created_at: datetime
    state: JobState = JobState.QUEUED
    duration_seconds: Optional[float] = None
    artifact_path: Optional[str] = None
    layer: Optional[ClipLayer] = None
    scene_index: Optional[int] = None
    bars: Optional[int] = None

    def to_summary(self) -> SessionClipSummary:
        return SessionClipSummary(
            job_id=self.job_id,
            prompt=self.prompt,
            state=self.state,
            duration_seconds=self.duration_seconds,
            artifact_path=self.artifact_path,
            layer=self.layer,
            scene_index=self.scene_index,
            bars=self.bars,
        )


@dataclass
class SessionRecord:
    session_id: str
    created_at: datetime
    updated_at: datetime
    name: Optional[str] = None
    seed_prompt: Optional[str] = None
    seed_job_id: Optional[str] = None
    seed_plan: Optional[CompositionPlan] = None
    theme: Optional[ThemeDescriptor] = None
    tempo_bpm: Optional[int] = None
    key: Optional[str] = None
    time_signature: Optional[str] = None
    clips: Dict[str, SessionClip] = field(default_factory=dict)

    def clip_count(self) -> int:
        return len(self.clips)


class SessionManager:
    """Tracks musical sessions owned by the worker."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionRecord] = {}
        self._lock = asyncio.Lock()

    async def create_session(self, payload: SessionCreateRequest) -> SessionSummary:
        session_id = f"session-{uuid4()}"
        now = datetime.now(tz=UTC)
        record = SessionRecord(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            name=payload.name,
            seed_prompt=payload.prompt,
            tempo_bpm=payload.tempo_bpm,
            key=payload.key,
            time_signature=payload.time_signature,
        )
        async with self._lock:
            self._sessions[session_id] = record
        return self._to_summary(record)

    async def session_exists(self, session_id: str) -> bool:
        async with self._lock:
            return session_id in self._sessions

    async def register_job(
        self,
        session_id: str,
        job_id: str,
        prompt: str,
        *,
        layer: Optional[ClipLayer] = None,
        scene_index: Optional[int] = None,
        bars: Optional[int] = None,
    ) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise UnknownSessionError(session_id)
            session.updated_at = datetime.now(tz=UTC)
            session.clips[job_id] = SessionClip(
                job_id=job_id,
                prompt=prompt,
                created_at=session.updated_at,
                layer=layer,
                scene_index=scene_index,
                bars=bars,
            )

    async def mark_job_state(self, session_id: str, job_id: str, state: JobState) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise UnknownSessionError(session_id)
            session.updated_at = datetime.now(tz=UTC)
            clip = session.clips.get(job_id)
            if clip is None:
                clip = SessionClip(
                    job_id=job_id,
                    prompt="(unknown)",
                    created_at=session.updated_at,
                )
                session.clips[job_id] = clip
            clip.state = state

    async def record_artifact(
        self,
        session_id: str,
        job_id: str,
        artifact: GenerationArtifact,
    ) -> None:
        async with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise UnknownSessionError(session_id)
            session.updated_at = datetime.now(tz=UTC)
            clip = session.clips.get(job_id)
            if clip is None:
                clip = SessionClip(
                    job_id=job_id,
                    prompt=artifact.metadata.prompt,
                    created_at=session.updated_at,
                )
                session.clips[job_id] = clip
            clip.state = JobState.SUCCEEDED
            clip.duration_seconds = float(artifact.metadata.duration_seconds)
            clip.artifact_path = artifact.artifact_path

            plan = artifact.metadata.plan
            if session.seed_plan is None and plan is not None:
                session.seed_plan = plan
                session.seed_job_id = job_id
                session.seed_prompt = artifact.metadata.prompt
                session.theme = plan.theme
                session.tempo_bpm = plan.tempo_bpm
                session.key = plan.key
                session.time_signature = plan.time_signature
            if plan is not None and plan.theme is not None and session.theme is None:
                session.theme = plan.theme

    async def get_summary(self, session_id: str) -> Optional[SessionSummary]:
        async with self._lock:
            record = self._sessions.get(session_id)
            if record is None:
                return None
            return self._to_summary(record)

    async def all_summaries(self) -> list[SessionSummary]:
        async with self._lock:
            return [self._to_summary(session) for session in self._sessions.values()]

    def _to_summary(self, record: SessionRecord) -> SessionSummary:
        clip_refs = sorted(record.clips.values(), key=lambda clip: clip.created_at)
        clips = [clip.to_summary() for clip in clip_refs]
        theme_copy = record.theme.model_copy(deep=True) if record.theme is not None else None
        seed_plan_copy = (
            record.seed_plan.model_copy(deep=True) if record.seed_plan is not None else None
        )
        return SessionSummary(
            session_id=record.session_id,
            created_at=record.created_at,
            updated_at=record.updated_at,
            name=record.name,
            tempo_bpm=record.tempo_bpm,
            key=record.key,
            time_signature=record.time_signature,
            seed_job_id=record.seed_job_id,
            seed_prompt=record.seed_prompt,
            seed_plan=seed_plan_copy,
            theme=theme_copy,
            clip_count=record.clip_count(),
            clips=clips,
        )
