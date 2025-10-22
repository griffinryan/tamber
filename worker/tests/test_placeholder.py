from fastapi.testclient import TestClient

from timbre_worker.app.main import create_app


def test_create_app() -> None:
    app = create_app()
    assert app.title == "Timbre Worker"


def test_health_endpoint() -> None:
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
