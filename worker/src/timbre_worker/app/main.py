from __future__ import annotations

from fastapi import FastAPI
from loguru import logger

from ..services.musicgen import MusicGenService
from ..services.orchestrator import ComposerOrchestrator
from ..services.planner import CompositionPlanner
from .jobs import JobManager
from .routes import router
from .settings import get_settings


def create_app() -> FastAPI:
    """Create and configure FastAPI instance."""
    settings = get_settings()
    planner = CompositionPlanner()
    musicgen = MusicGenService(settings=settings)
    orchestrator = ComposerOrchestrator(settings, planner, musicgen)
    manager = JobManager(orchestrator)
    app = FastAPI(title="Timbre Worker", version="0.1.0")
    app.state.settings = settings
    app.state.musicgen_service = musicgen
    app.state.composer = orchestrator
    app.state.planner = planner
    app.state.job_manager = manager
    app.state.backend_status = {}

    async def _warmup_background() -> None:
        try:
            statuses = await orchestrator.warmup()
            app.state.backend_status = statuses
            logger.info(
                "Worker warmup complete: {}",
                {name: status.ready for name, status in statuses.items()},
            )
        except Exception:  # noqa: BLE001
            logger.exception("Worker warmup failed")

    app.add_event_handler("startup", _warmup_background)
    app.include_router(router)
    return app


app = create_app()
