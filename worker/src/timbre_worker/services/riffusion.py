from __future__ import annotations

from typing import Optional


class RiffusionService:
    """Placeholder service wrapping the Riffusion pipeline.

    Phase 0 only needs to validate environment setup; concrete loading and inference will
    follow once the TUI scaffolding and contracts are stable.
    """

    def __init__(self) -> None:
        self._model = None

    async def warmup(self) -> None:
        """Load model weights lazily; no-op placeholder for now."""

    async def generate(
        self, prompt: str, duration_seconds: int, seed: Optional[int] = None
    ) -> str:
        """Generate audio from prompt and return artifact path.

        Stub implementation returns a placeholder path so the CLI pipeline can be exercised.
        """
        del prompt, duration_seconds, seed
        return "/tmp/riffusion-placeholder.wav"
