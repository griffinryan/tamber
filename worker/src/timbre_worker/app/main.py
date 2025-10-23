from __future__ import annotations

import asyncio

from fastapi import FastAPI

from ..services.musicgen import MusicGenService
from ..services.orchestrator import ComposerOrchestrator
from ..services.planner import CompositionPlanner
from ..services.riffusion import RiffusionService
from .jobs import JobManager
from .routes import router
from .settings import get_settings


def create_app() -> FastAPI:
    """Create and configure FastAPI instance."""
    settings = get_settings()
    planner = CompositionPlanner()
    riffusion = RiffusionService(settings)
    musicgen = MusicGenService()
    orchestrator = ComposerOrchestrator(settings, planner, riffusion, musicgen)
    manager = JobManager(orchestrator)
    app = FastAPI(title="Timbre Worker", version="0.1.0")
    app.state.settings = settings
    app.state.riffusion_service = riffusion
    app.state.musicgen_service = musicgen
    app.state.composer = orchestrator
    app.state.planner = planner
    app.state.job_manager = manager

    async def _warmup_background() -> None:
        asyncio.create_task(orchestrator.warmup())

    app.add_event_handler("startup", _warmup_background)
    app.include_router(router)
    return app


app = create_app()
