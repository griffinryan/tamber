"""Shared service data structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
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


@dataclass
class BackendStatus:
    name: str
    ready: bool
    device: Optional[str]
    dtype: Optional[str]
    error: Optional[str]
    details: Dict[str, Any] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "ready": self.ready,
            "updated_at": self.updated_at.isoformat(),
        }
        if self.device is not None:
            payload["device"] = self.device
        if self.dtype is not None:
            payload["dtype"] = self.dtype
        if self.error is not None:
            payload["error"] = self.error
        if self.details:
            payload["details"] = self.details
        return payload
