"""Shared service data structures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class SectionRender:
    waveform: np.ndarray
    sample_rate: int
    extras: Dict[str, Any]
