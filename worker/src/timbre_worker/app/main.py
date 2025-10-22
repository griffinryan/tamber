from fastapi import FastAPI

from .routes import router


def create_app() -> FastAPI:
    """Create and configure FastAPI instance."""
    app = FastAPI(title="Timbre Worker", version="0.1.0")
    app.include_router(router)
    return app


app = create_app()
