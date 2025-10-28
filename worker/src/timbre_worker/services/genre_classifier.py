"""
Genre Classification and Musical Intelligence

This module analyzes user prompts to detect genre, style, era, and vocal characteristics.
It provides semantic understanding to drive genre-aware template selection and orchestration.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml


class NarrativeStructure(str, Enum):
    """Types of musical narrative structures"""
    MOTIF_DRIVEN = "motif_driven"  # Traditional verse/chorus, theme-based
    TEXTURE_DRIVEN = "texture_driven"  # Ambient, evolving layers
    RHYTHM_DRIVEN = "rhythm_driven"  # Hip-hop, EDM, groove-first
    HARMONIC_DRIVEN = "harmonic_driven"  # Jazz, classical, chord-focused
    CINEMATIC = "cinematic"  # Film scores, dramatic arcs


class EnergyProfile(str, Enum):
    """Energy curve patterns for compositions"""
    DRAMATIC_ARC = "dramatic_arc"  # Classical, film scores
    SWINGING_ARC = "swinging_arc"  # Jazz
    GROOVE_STEADY = "groove_steady"  # Hip-hop
    BUILD_DROP = "build_drop"  # EDM
    ANTHEMIC_BUILD = "anthemic_build"  # Rock
    FLAT_LOW = "flat_low"  # Ambient
    CATCHY_BUILD = "catchy_build"  # Pop
    SMOOTH_GROOVE = "smooth_groove"  # R&B
    ORGANIC_FLOW = "organic_flow"  # Folk
    CULTURAL_FLOW = "cultural_flow"  # World
    UNPREDICTABLE = "unpredictable"  # Experimental


@dataclass
class VocalCharacteristics:
    """Detected vocal characteristics from prompt"""
    has_vocals: bool = False
    gender: Optional[str] = None  # "male", "female", "mixed", None
    style: Optional[str] = None  # "rap", "sung", "choir", "opera", "spoken"
    descriptors: list[str] = None  # ["soulful", "aggressive", "ethereal"]

    def __post_init__(self):
        if self.descriptors is None:
            self.descriptors = []


@dataclass
class GenreProfile:
    """Complete musical profile for a detected genre"""
    genre: str
    display_name: str
    confidence: float  # 0.0-1.0
    tempo_range: tuple[int, int]
    typical_tempos: list[int]
    meters: list[str]
    narrative_structure: NarrativeStructure
    vocal_prevalence: float  # 0.0-1.0, how often this genre has vocals
    energy_profile: EnergyProfile
    typical_keys: list[str]
    layers: dict[str, int]  # rhythm, bass, harmony, lead, textures, vocals counts
    instruments: dict[str, list[str]]  # Per-layer instrument suggestions
    era: Optional[str] = None  # "60s", "80s", "90s", "modern"
    subgenre: Optional[str] = None  # "acid jazz", "trap", "lo-fi"


class GenreClassifier:
    """
    Analyzes user prompts to detect genre and extract musical characteristics.

    Uses keyword matching, artist references, and era markers to classify
    prompts into 12 base genres with detailed musical profiles.
    """

    def __init__(self):
        self.profiles = self._load_profiles()
        self._build_keyword_index()
        self._build_artist_index()
        self._build_era_patterns()
        self._build_vocal_patterns()

    def _load_profiles(self) -> dict:
        """Load genre profiles from YAML database"""
        profiles_path = Path(__file__).parent.parent / "data" / "genre_profiles.yaml"
        with open(profiles_path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _build_keyword_index(self):
        """Build reverse index: keyword -> [genres]"""
        self.keyword_to_genres = {}
        for genre, profile in self.profiles.items():
            for keyword in profile.get("keywords", []):
                keyword_lower = keyword.lower()
                if keyword_lower not in self.keyword_to_genres:
                    self.keyword_to_genres[keyword_lower] = []
                self.keyword_to_genres[keyword_lower].append(genre)

    def _build_artist_index(self):
        """Map artist names to genres (extracted from keywords)"""
        self.artist_to_genre = {
            # Classical
            "mozart": "classical",
            "beethoven": "classical",
            "bach": "classical",
            "tchaikovsky": "classical",

            # Film Score
            "john williams": "film_score",
            "hans zimmer": "film_score",
            "ennio morricone": "film_score",

            # Jazz
            "miles davis": "jazz",
            "john coltrane": "jazz",
            "charlie parker": "jazz",
            "herbie hancock": "jazz",

            # Hip-Hop / R&B
            "the weeknd": "r_and_b",
            "drake": "hip_hop",
            "kendrick lamar": "hip_hop",
            "travis scott": "hip_hop",
            "sza": "r_and_b",

            # Pop
            "wham": "pop",
            "madonna": "pop",
            "michael jackson": "pop",
            "ariana grande": "pop",

            # Rock
            "nirvana": "rock",
            "radiohead": "rock",
            "pink floyd": "rock",

            # Electronic
            "daft punk": "electronic",
            "aphex twin": "electronic",
            "deadmau5": "electronic",

            # Ambient
            "brian eno": "ambient",
            "stars of the lid": "ambient",
            "william basinski": "ambient",
        }

    def _build_era_patterns(self):
        """Patterns for detecting musical eras"""
        self.era_patterns = {
            "60s": [r"\b60s?\b", r"\bsixties\b", r"\b1960s?\b"],
            "70s": [r"\b70s?\b", r"\bseventies\b", r"\b1970s?\b"],
            "80s": [r"\b80s?\b", r"\beighties\b", r"\b1980s?\b"],
            "90s": [r"\b90s?\b", r"\bnineties\b", r"\b1990s?\b"],
            "2000s": [r"\b2000s?\b", r"\bearly 2000s?\b"],
            "2010s": [r"\b2010s?\b"],
            "modern": [r"\bmodern\b", r"\bcontemporary\b", r"\bcurrent\b"],
        }

    def _build_vocal_patterns(self):
        """Patterns for detecting vocal characteristics"""
        self.vocal_patterns = {
            "male": [
                r"\bmale\s+(?:vocal|singer|voice|rap)",
                r"\b(?:tenor|baritone|bass)\b",
                r"\bmale\b",
            ],
            "female": [
                r"\bfemale\s+(?:vocal|singer|voice|rap)",
                r"\b(?:soprano|alto)\b",
                r"\bfemale\b",
            ],
            "choir": [r"\bchoir\b", r"\bchoral\b", r"\bensemble\b"],
            "rap": [r"\brap\b", r"\bmc\b", r"\bhip[- ]?hop vocal"],
            "sung": [r"\bsing", r"\bsung\b", r"\bvocal", r"\bvoice"],
            "opera": [r"\bopera\b", r"\baria\b"],
            "spoken": [r"\bspoken\s+word\b", r"\bnarrat"],
        }

        self.vocal_descriptors = {
            "soulful": [r"\bsoulful\b", r"\bsoul\b"],
            "ethereal": [r"\bethereal\b", r"\bangelic\b", r"\bairy\b"],
            "aggressive": [r"\baggressive\b", r"\bscream", r"\bshout"],
            "smooth": [r"\bsmooth\b", r"\bsilky\b"],
            "raspy": [r"\braspy\b", r"\bgritty\b", r"\brough\b"],
            "falsetto": [r"\bfalsetto\b", r"\bhigh\s+voice\b"],
        }

    def classify(self, prompt: str) -> GenreProfile:
        """
        Analyze prompt and return genre profile with confidence score.

        Args:
            prompt: User's text prompt

        Returns:
            GenreProfile with detected genre and characteristics
        """
        prompt_lower = prompt.lower()

        # Detect era first
        era = self._detect_era(prompt_lower)

        # Detect genre through multiple signals
        genre_scores = {}

        # 1. Keyword matching (highest confidence)
        keyword_matches = self._match_keywords(prompt_lower)
        for genre, score in keyword_matches.items():
            genre_scores[genre] = genre_scores.get(genre, 0.0) + score

        # 2. Artist references (very high confidence)
        artist_match = self._match_artists(prompt_lower)
        if artist_match:
            genre_scores[artist_match] = genre_scores.get(artist_match, 0.0) + 0.9

        # 3. Era-based genre hints
        if era in ("60s", "70s") and "jazz" in prompt_lower:
            genre_scores["jazz"] = genre_scores.get("jazz", 0.0) + 0.3
        elif era == "80s":
            genre_scores["pop"] = genre_scores.get("pop", 0.0) + 0.2
        elif era == "90s":
            # 90s can be many things, slight boost to relevant genres
            genre_scores["hip_hop"] = genre_scores.get("hip_hop", 0.0) + 0.1
            genre_scores["rock"] = genre_scores.get("rock", 0.0) + 0.1

        # Select best match or default to pop
        if genre_scores:
            genre = max(genre_scores.items(), key=lambda x: x[1])[0]
            confidence = min(1.0, max(genre_scores.values()))
        else:
            # Default to pop for generic prompts
            genre = "pop"
            confidence = 0.3

        # Detect subgenre
        subgenre = self._detect_subgenre(prompt_lower, genre)

        # Build profile
        return self._build_profile(genre, confidence, era, subgenre)

    def _match_keywords(self, prompt: str) -> dict[str, float]:
        """Match genre keywords and return scores"""
        scores = {}
        for keyword, genres in self.keyword_to_genres.items():
            if keyword in prompt:
                # Exact word match gets higher score
                if re.search(rf"\b{re.escape(keyword)}\b", prompt):
                    for g in genres:
                        scores[g] = scores.get(g, 0.0) + 0.8
                else:
                    # Substring match gets lower score
                    for g in genres:
                        scores[g] = scores.get(g, 0.0) + 0.5
        return scores

    def _match_artists(self, prompt: str) -> Optional[str]:
        """Match artist names to genres"""
        for artist, genre in self.artist_to_genre.items():
            if artist in prompt:
                return genre
        return None

    def _detect_era(self, prompt: str) -> Optional[str]:
        """Detect era from prompt"""
        for era, patterns in self.era_patterns.items():
            for pattern in patterns:
                if re.search(pattern, prompt, re.IGNORECASE):
                    return era
        return None

    def _detect_subgenre(self, prompt: str, genre: str) -> Optional[str]:
        """Detect subgenre within a genre"""
        subgenre_patterns = {
            "jazz": {
                "acid jazz": r"\bacid\s+jazz\b",
                "bebop": r"\bbebop\b",
                "swing": r"\bswing\b",
                "latin jazz": r"\blatin\s+jazz\b",
                "bossa nova": r"\bbossa\s+nova\b",
            },
            "hip_hop": {
                "trap": r"\btrap\b",
                "boom bap": r"\bboom\s+bap\b",
                "lo-fi": r"\blo-?fi\b",
                "conscious": r"\bconscious\b",
            },
            "electronic": {
                "house": r"\bhouse\b",
                "techno": r"\btechno\b",
                "trance": r"\btrance\b",
                "dubstep": r"\bdubstep\b",
                "drum and bass": r"\b(?:drum\s+and\s+bass|dnb)\b",
            },
            "rock": {
                "indie": r"\bindie\b",
                "punk": r"\bpunk\b",
                "grunge": r"\bgrunge\b",
                "metal": r"\bmetal\b",
            },
        }

        if genre in subgenre_patterns:
            for subgenre, pattern in subgenre_patterns[genre].items():
                if re.search(pattern, prompt, re.IGNORECASE):
                    return subgenre
        return None

    def detect_vocals(self, prompt: str) -> VocalCharacteristics:
        """Detect vocal characteristics from prompt"""
        prompt_lower = prompt.lower()

        vocals = VocalCharacteristics()

        # Check for any vocal keywords
        has_any_vocal = any(
            re.search(pattern, prompt_lower)
            for patterns in self.vocal_patterns.values()
            for pattern in patterns
        )

        if not has_any_vocal:
            return vocals

        vocals.has_vocals = True

        # Detect gender
        male_match = any(re.search(p, prompt_lower) for p in self.vocal_patterns["male"])
        female_match = any(re.search(p, prompt_lower) for p in self.vocal_patterns["female"])

        if male_match and female_match:
            vocals.gender = "mixed"
        elif male_match:
            vocals.gender = "male"
        elif female_match:
            vocals.gender = "female"

        # Detect style
        if any(re.search(p, prompt_lower) for p in self.vocal_patterns["rap"]):
            vocals.style = "rap"
        elif any(re.search(p, prompt_lower) for p in self.vocal_patterns["opera"]):
            vocals.style = "opera"
        elif any(re.search(p, prompt_lower) for p in self.vocal_patterns["choir"]):
            vocals.style = "choir"
        elif any(re.search(p, prompt_lower) for p in self.vocal_patterns["spoken"]):
            vocals.style = "spoken"
        elif any(re.search(p, prompt_lower) for p in self.vocal_patterns["sung"]):
            vocals.style = "sung"

        # Detect descriptors
        for descriptor, patterns in self.vocal_descriptors.items():
            if any(re.search(p, prompt_lower) for p in patterns):
                vocals.descriptors.append(descriptor)

        return vocals

    def _build_profile(
        self,
        genre: str,
        confidence: float,
        era: Optional[str],
        subgenre: Optional[str],
    ) -> GenreProfile:
        """Build complete genre profile from database and detections"""
        data = self.profiles[genre]

        return GenreProfile(
            genre=genre,
            display_name=data["display_name"],
            confidence=confidence,
            tempo_range=tuple(data["tempo_range"]),
            typical_tempos=data["typical_tempos"],
            meters=data["meters"],
            narrative_structure=NarrativeStructure(data["narrative_structure"]),
            vocal_prevalence=data["vocal_prevalence"],
            energy_profile=EnergyProfile(data["energy_profile"]),
            typical_keys=data["typical_keys"],
            layers=data["layers"],
            instruments=data["instruments"],
            era=era,
            subgenre=subgenre,
        )
