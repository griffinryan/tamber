import asyncio

from fastapi import FastAPI

from .jobs import JobManager
from .routes import router
from .settings import get_settings
from ..services.riffusion import RiffusionService


def create_app() -> FastAPI:
    """Create and configure FastAPI instance."""
    settings = get_settings()
    service = RiffusionService(settings)
    manager = JobManager(service)
    app = FastAPI(title="Timbre Worker", version="0.1.0")
    app.state.settings = settings
    app.state.riffusion_service = service
    app.state.job_manager = manager

    async def _warmup_background() -> None:
        asyncio.create_task(service.warmup())

    app.add_event_handler("startup", _warmup_background)
    app.include_router(router)
    return app


app = create_app()
