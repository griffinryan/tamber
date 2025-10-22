from __future__ import annotations

from pathlib import Path

import pytest

from timbre_worker.generate import _run


@pytest.mark.asyncio
async def test_generate_cli_placeholder(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifact_dir = tmp_path / "artifacts"
    config_dir = tmp_path / "config"

    await _run(
        "calm modular arpeggios",
        duration=2,
        model_id="riffusion-v1",
        artifact_dir=artifact_dir,
        config_dir=config_dir,
    )

    captured = capsys.readouterr()
    assert "artifact_path" in captured.out
    assert artifact_dir.exists()
