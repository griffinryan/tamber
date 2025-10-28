"""
Tests for genre classification and vocal detection
"""

import pytest

from timbre_worker.services.genre_classifier import (
    EnergyProfile,
    GenreClassifier,
    NarrativeStructure,
    VocalCharacteristics,
)


@pytest.fixture
def classifier():
    """Create genre classifier instance"""
    return GenreClassifier()


class TestGenreDetection:
    """Tests for genre classification"""

    def test_classical_detection(self, classifier):
        """Should detect classical genre"""
        profile = classifier.classify("orchestral symphony with violin and piano")
        assert profile.genre == "classical"
        assert profile.confidence > 0.5
        assert profile.narrative_structure == NarrativeStructure.HARMONIC_DRIVEN
        assert profile.energy_profile == EnergyProfile.DRAMATIC_ARC

    def test_jazz_detection(self, classifier):
        """Should detect jazz genre"""
        profile = classifier.classify("smooth jazz with saxophone and upright bass")
        assert profile.genre == "jazz"
        assert profile.confidence > 0.7
        assert profile.narrative_structure == NarrativeStructure.HARMONIC_DRIVEN

    def test_hip_hop_detection(self, classifier):
        """Should detect hip-hop genre"""
        profile = classifier.classify("trap beat with 808 bass and rap vocals")
        assert profile.genre == "hip_hop"
        assert profile.confidence > 0.7
        assert profile.narrative_structure == NarrativeStructure.RHYTHM_DRIVEN

    def test_electronic_detection(self, classifier):
        """Should detect electronic genre"""
        profile = classifier.classify("house music with four-on-the-floor kick")
        assert profile.genre == "electronic"
        assert profile.confidence > 0.5
        assert profile.energy_profile == EnergyProfile.BUILD_DROP

    def test_ambient_detection(self, classifier):
        """Should detect ambient genre"""
        profile = classifier.classify("dreamy ambient soundscape with drones")
        assert profile.genre == "ambient"
        assert profile.confidence > 0.5
        assert profile.narrative_structure == NarrativeStructure.TEXTURE_DRIVEN
        assert profile.energy_profile == EnergyProfile.FLAT_LOW

    def test_film_score_detection(self, classifier):
        """Should detect film score genre"""
        profile = classifier.classify("cinematic film score like Hans Zimmer")
        assert profile.genre == "film_score"
        assert profile.confidence > 0.7
        assert profile.narrative_structure == NarrativeStructure.CINEMATIC

    def test_film_score_john_williams(self, classifier):
        """Should detect film score from John Williams reference"""
        profile = classifier.classify("epic orchestral piece like John Williams Harry Potter")
        assert profile.genre == "film_score"
        assert profile.confidence > 0.8  # Artist match gives high confidence

    def test_pop_detection(self, classifier):
        """Should detect pop genre"""
        profile = classifier.classify("upbeat pop song with synths and vocals")
        assert profile.genre == "pop"
        assert profile.confidence > 0.5
        assert profile.vocal_prevalence > 0.9

    def test_r_and_b_detection(self, classifier):
        """Should detect R&B genre"""
        profile = classifier.classify("smooth r&b with soulful vocals")
        assert profile.genre == "r_and_b"
        assert profile.confidence > 0.5
        assert profile.energy_profile == EnergyProfile.SMOOTH_GROOVE

    def test_rock_detection(self, classifier):
        """Should detect rock genre"""
        profile = classifier.classify("indie rock with electric guitar and drums")
        assert profile.genre == "rock"
        assert profile.confidence > 0.7

    def test_folk_detection(self, classifier):
        """Should detect folk genre"""
        profile = classifier.classify("acoustic folk with fingerstyle guitar")
        assert profile.genre == "folk"
        assert profile.confidence > 0.5

    def test_world_detection(self, classifier):
        """Should detect world music genre"""
        profile = classifier.classify("ethnic music with tabla and sitar")
        assert profile.genre == "world"
        assert profile.confidence > 0.7

    def test_experimental_detection(self, classifier):
        """Should detect experimental genre"""
        profile = classifier.classify("avant-garde noise with glitchy textures")
        assert profile.genre == "experimental"
        assert profile.confidence > 0.5
        assert profile.narrative_structure == NarrativeStructure.TEXTURE_DRIVEN


class TestSubgenreDetection:
    """Tests for subgenre detection"""

    def test_acid_jazz_subgenre(self, classifier):
        """Should detect acid jazz subgenre"""
        profile = classifier.classify("60s acid jazz with funky grooves")
        assert profile.genre == "jazz"
        assert profile.subgenre == "acid jazz"
        assert profile.era == "60s"

    def test_trap_subgenre(self, classifier):
        """Should detect trap subgenre"""
        profile = classifier.classify("modern trap beat with hi-hats")
        assert profile.genre == "hip_hop"
        assert profile.subgenre == "trap"

    def test_lofi_subgenre(self, classifier):
        """Should detect lo-fi subgenre"""
        profile = classifier.classify("chill lo-fi hip hop beats")
        assert profile.genre == "hip_hop"
        assert profile.subgenre == "lo-fi"

    def test_house_subgenre(self, classifier):
        """Should detect house subgenre"""
        profile = classifier.classify("deep house with groovy bassline")
        assert profile.genre == "electronic"
        assert profile.subgenre == "house"

    def test_drum_and_bass_subgenre(self, classifier):
        """Should detect drum and bass subgenre"""
        profile = classifier.classify("fast-paced drum and bass")
        assert profile.genre == "electronic"
        assert profile.subgenre == "drum and bass"


class TestEraDetection:
    """Tests for era detection"""

    def test_60s_era(self, classifier):
        """Should detect 60s era"""
        profile = classifier.classify("60s psychedelic rock")
        assert profile.era == "60s"

    def test_80s_era(self, classifier):
        """Should detect 80s era"""
        profile = classifier.classify("80s synthpop like Wham!")
        assert profile.era == "80s"
        assert profile.genre == "pop"

    def test_90s_era(self, classifier):
        """Should detect 90s era"""
        profile = classifier.classify("1990s grunge rock")
        assert profile.era == "90s"

    def test_modern_era(self, classifier):
        """Should detect modern era"""
        profile = classifier.classify("modern electronic music")
        assert profile.era == "modern"


class TestArtistReferences:
    """Tests for artist-based genre detection"""

    def test_the_weeknd(self, classifier):
        """Should detect The Weeknd as R&B"""
        profile = classifier.classify("dark R&B like The Weeknd")
        assert profile.genre == "r_and_b"
        assert profile.confidence > 0.8

    def test_wham(self, classifier):
        """Should detect Wham! as pop"""
        profile = classifier.classify("upbeat pop like Wham! Make It Big album")
        assert profile.genre == "pop"
        assert profile.confidence > 0.8

    def test_brian_eno(self, classifier):
        """Should detect Brian Eno as ambient"""
        profile = classifier.classify("atmospheric ambient like Brian Eno")
        assert profile.genre == "ambient"
        assert profile.confidence > 0.8

    def test_miles_davis(self, classifier):
        """Should detect Miles Davis as jazz"""
        profile = classifier.classify("cool jazz like Miles Davis")
        assert profile.genre == "jazz"
        assert profile.confidence > 0.8


class TestVocalDetection:
    """Tests for vocal characteristic detection"""

    def test_male_vocals(self, classifier):
        """Should detect male vocals"""
        vocals = classifier.detect_vocals("smooth song with male vocals")
        assert vocals.has_vocals is True
        assert vocals.gender == "male"

    def test_female_vocals(self, classifier):
        """Should detect female vocals"""
        vocals = classifier.detect_vocals("pop song with female singer")
        assert vocals.has_vocals is True
        assert vocals.gender == "female"

    def test_mixed_vocals(self, classifier):
        """Should detect mixed vocals"""
        vocals = classifier.detect_vocals("duet with male and female vocals")
        assert vocals.has_vocals is True
        assert vocals.gender == "mixed"

    def test_rap_style(self, classifier):
        """Should detect rap vocal style"""
        vocals = classifier.detect_vocals("hip hop track with aggressive rap")
        assert vocals.has_vocals is True
        assert vocals.style == "rap"

    def test_choir_style(self, classifier):
        """Should detect choir vocal style"""
        vocals = classifier.detect_vocals("epic piece with choir")
        assert vocals.has_vocals is True
        assert vocals.style == "choir"

    def test_opera_style(self, classifier):
        """Should detect opera vocal style"""
        vocals = classifier.detect_vocals("classical aria with opera vocals")
        assert vocals.has_vocals is True
        assert vocals.style == "opera"

    def test_soulful_descriptor(self, classifier):
        """Should detect soulful vocal descriptor"""
        vocals = classifier.detect_vocals("r&b with soulful male vocals")
        assert vocals.has_vocals is True
        assert "soulful" in vocals.descriptors

    def test_ethereal_descriptor(self, classifier):
        """Should detect ethereal vocal descriptor"""
        vocals = classifier.detect_vocals("ambient with ethereal voices")
        assert vocals.has_vocals is True
        assert "ethereal" in vocals.descriptors

    def test_no_vocals(self, classifier):
        """Should detect no vocals"""
        vocals = classifier.detect_vocals("instrumental piano piece")
        assert vocals.has_vocals is False
        assert vocals.gender is None
        assert vocals.style is None


class TestTempoRanges:
    """Tests for genre-specific tempo ranges"""

    def test_ambient_slow_tempo(self, classifier):
        """Ambient should have slow tempo range"""
        profile = classifier.classify("ambient drone")
        assert profile.tempo_range[0] == 40
        assert profile.tempo_range[1] == 90

    def test_electronic_fast_tempo(self, classifier):
        """Electronic should have fast tempo range"""
        profile = classifier.classify("drum and bass")
        assert profile.tempo_range[0] == 100
        assert profile.tempo_range[1] == 180

    def test_classical_moderate_tempo(self, classifier):
        """Classical should have moderate tempo range"""
        profile = classifier.classify("classical symphony")
        assert profile.tempo_range[0] == 40
        assert profile.tempo_range[1] == 120


class TestLayerProfiles:
    """Tests for genre-specific layer profiles"""

    def test_classical_no_rhythm(self, classifier):
        """Classical should have minimal rhythm layer"""
        profile = classifier.classify("orchestral piece")
        assert profile.layers["rhythm"] == 0
        assert profile.layers["harmony"] == 6

    def test_hip_hop_heavy_rhythm(self, classifier):
        """Hip-hop should have heavy rhythm layer"""
        profile = classifier.classify("hip hop beat")
        assert profile.layers["rhythm"] == 3
        assert profile.layers["bass"] == 2

    def test_ambient_heavy_textures(self, classifier):
        """Ambient should have heavy textures layer"""
        profile = classifier.classify("ambient soundscape")
        assert profile.layers["textures"] == 6
        assert profile.layers["rhythm"] == 0

    def test_jazz_heavy_lead(self, classifier):
        """Jazz should have heavy lead layer for solos"""
        profile = classifier.classify("jazz with saxophone solo")
        assert profile.layers["lead"] == 4


class TestInstrumentSuggestions:
    """Tests for genre-specific instrument suggestions"""

    def test_classical_instruments(self, classifier):
        """Classical should suggest orchestral instruments"""
        profile = classifier.classify("classical music")
        harmony_instruments = profile.instruments["harmony"]
        assert any("violin" in inst.lower() for inst in harmony_instruments)

    def test_jazz_instruments(self, classifier):
        """Jazz should suggest jazz instruments"""
        profile = classifier.classify("jazz music")
        lead_instruments = profile.instruments["lead"]
        assert any("saxophone" in inst.lower() or "trumpet" in inst.lower() for inst in lead_instruments)

    def test_hip_hop_instruments(self, classifier):
        """Hip-hop should suggest 808 and trap elements"""
        profile = classifier.classify("hip hop")
        rhythm_instruments = profile.instruments["rhythm"]
        assert any("808" in inst for inst in rhythm_instruments)

    def test_film_score_instruments(self, classifier):
        """Film scores should suggest cinematic instruments"""
        profile = classifier.classify("film score")
        rhythm_instruments = profile.instruments["rhythm"]
        assert any("timpani" in inst.lower() or "epic" in inst.lower() for inst in rhythm_instruments)


class TestDefaultBehavior:
    """Tests for default/fallback behavior"""

    def test_generic_prompt_defaults_to_pop(self, classifier):
        """Generic prompts should default to pop"""
        profile = classifier.classify("nice song with melody")
        assert profile.genre == "pop"
        assert profile.confidence < 0.5  # Low confidence for generic

    def test_instrumental_only_prompt(self, classifier):
        """Should handle instrumental-only prompts"""
        profile = classifier.classify("beautiful melody")
        assert profile.genre is not None
        assert profile.confidence >= 0.0
