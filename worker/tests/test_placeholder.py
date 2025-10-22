from timbre_worker.app.main import create_app


def test_create_app() -> None:
    app = create_app()
    assert app.title == "Timbre Worker"
