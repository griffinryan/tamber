"""Shared service data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class SectionRender:
    waveform: np.ndarray
    sample_rate: int
    extras: Dict[str, Any]


@dataclass
class SectionPhrase:
    section_id: str
    tempo_bpm: int
    bars: int
    beats: float
    seconds: float
    seconds_per_beat: float
    padding_seconds: float
    conditioning_tail_seconds: float

    @property
    def duration_with_padding(self) -> float:
        return self.seconds + self.padding_seconds

    @property
    def beats_with_padding(self) -> float:
        return self.beats + (self.padding_seconds / max(self.seconds_per_beat, 1e-6))


@dataclass
class SectionTrack:
    section_id: str
    phrase: SectionPhrase
    render: SectionRender
    backend: str
    conditioning_tail: Optional[np.ndarray]
    conditioning_rate: Optional[int]

    def render_seconds(self) -> float:
        waveform = self.render.waveform
        if waveform.size == 0 or self.render.sample_rate <= 0:
            return 0.0
        return float(
            np.asarray(waveform, dtype=np.float32).shape[0] / self.render.sample_rate
        )
