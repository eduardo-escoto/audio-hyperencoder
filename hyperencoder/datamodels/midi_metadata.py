"""
MIDI metadata models for hyperencoder.

This module defines the Pydantic models for MIDI metadata extracted from MIDI files.
"""

from typing import Any
from pydantic import Field

from .base import BaseConfig


class MidiMetadata(BaseConfig):
    """
    Extracted MIDI metadata for a song.
    
    This model contains comprehensive musical metadata extracted from MIDI files
    to enrich the training data with musical information.
    """
    
    # Basic file information
    filename: str = Field(description="Name of the MIDI file")
    duration_seconds: float = Field(
        ge=0.0, description="Total duration of the MIDI file in seconds"
    )
    
    # Tempo information
    tempo_bpm: float = Field(
        ge=1.0, le=1000.0, description="Primary tempo in beats per minute"
    )
    tempo_changes: list[tuple[float, float]] = Field(
        default_factory=list,
        description="List of (time_seconds, tempo_bpm) changes throughout the song"
    )
    
    # Time signature information
    time_signature_numerator: int = Field(
        ge=1, le=32, description="Time signature numerator (beats per measure)"
    )
    time_signature_denominator: int = Field(
        ge=1, le=32, description="Time signature denominator (note value)"
    )
    time_signature_changes: list[tuple[float, int, int]] = Field(
        default_factory=list,
        description="List of (time_seconds, numerator, denominator) changes"
    )
    
    # Key signature information
    key_signature: int = Field(
        ge=-7, le=7, description="Key signature (-7 to 7 flats/sharps)"
    )
    key_signature_changes: list[tuple[float, int]] = Field(
        default_factory=list,
        description="List of (time_seconds, key_signature) changes"
    )
    
    # Note statistics
    total_notes: int = Field(ge=0, description="Total number of notes in the MIDI")
    note_range_min: int = Field(
        ge=0, le=127, description="Lowest MIDI note number"
    )
    note_range_max: int = Field(
        ge=0, le=127, description="Highest MIDI note number"
    )
    note_range_span: int = Field(
        ge=0, le=127, description="Span of note range (max - min)"
    )
    average_velocity: float = Field(
        ge=0.0, le=127.0, description="Average note velocity"
    )
    velocity_std: float = Field(
        ge=0.0, description="Standard deviation of note velocities"
    )
    
    # Track information
    num_tracks: int = Field(ge=0, description="Number of MIDI tracks")
    track_names: list[str] = Field(
        default_factory=list, description="Names of MIDI tracks"
    )
    num_channels: int = Field(
        ge=0, le=16, description="Number of MIDI channels used"
    )
    
    # Instrument information
    unique_programs: list[int] = Field(
        default_factory=list,
        description="List of unique MIDI program numbers (instruments)"
    )
    program_changes: list[tuple[float, int, int]] = Field(
        default_factory=list,
        description="List of (time_seconds, channel, program) changes"
    )
    
    # Rhythmic information
    note_density: float = Field(
        ge=0.0, description="Average notes per second"
    )
    beat_density: float = Field(
        ge=0.0, description="Average notes per beat"
    )
    
    # Advanced musical features (optional, computed if requested)
    chord_progressions: list[str] | None = Field(
        default=None,
        description="Detected chord progressions (if harmonic analysis enabled)"
    )
    key_centers: list[str] | None = Field(
        default=None,
        description="Detected key centers (if harmonic analysis enabled)"
    )
    rhythmic_patterns: dict[str, Any] | None = Field(
        default=None,
        description="Detected rhythmic patterns (if rhythmic analysis enabled)"
    )
    
    # Processing metadata
    extraction_timestamp: float = Field(
        description="Unix timestamp when metadata was extracted"
    )
    extractor_version: str = Field(
        default="1.0.0", description="Version of the metadata extractor"
    ) 