"""
MIDI metadata extraction for hyperencoder.

This module provides functionality to extract metadata from MIDI files to enrich
the training data with musical information.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

# Music analysis imports
import miditoolkit
import numpy as np


class MidiMetadataExtractor:
    """
    Extracts metadata from MIDI files for use in hyperencoder training.
    
    This class handles the extraction of musical metadata from MIDI files,
    including tempo, time signature, key signature, and note statistics.
    """
    
    def __init__(
        self,
        extract_harmony: bool = True,
        extract_rhythm: bool = True,
        extract_structure: bool = False,
    ):
        """
        Initialize the MIDI metadata extractor.
        
        Args:
            extract_harmony: Whether to extract harmonic information
            extract_rhythm: Whether to extract rhythmic information
            extract_structure: Whether to extract structural information
        """
        self.extract_harmony = extract_harmony
        self.extract_rhythm = extract_rhythm
        self.extract_structure = extract_structure
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize music analysis components
        self._setup_extractors()
    
    def _setup_extractors(self) -> None:
        """Set up the music analysis components."""
        self.logger.info("Setting up MIDI metadata extractors")
        
        # Initialize any additional extractors here
        # For example, key detection, chord progression analysis, etc.
        
    def extract_metadata(self, midi_path: Path) -> Dict[str, Any]:
        """
        Extract metadata from a MIDI file.
        
        Args:
            midi_path: Path to the MIDI file
            
        Returns:
            Dictionary containing extracted metadata
            
        Raises:
            FileNotFoundError: If the MIDI file doesn't exist
            ValueError: If the MIDI file is invalid
        """
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        
        try:
            # Load MIDI file
            midi_data = miditoolkit.MidiFile(str(midi_path))
            
            # Extract basic metadata
            metadata = {
                "filename": midi_path.name,
                "duration_seconds": self._get_duration_seconds(midi_data),
            }
            
            # Extract tempo information
            if self.extract_rhythm:
                tempo_info = self._extract_tempo_info(midi_data)
                metadata.update(tempo_info)
            
            # Extract time signature information
            if self.extract_rhythm:
                time_sig_info = self._extract_time_signature_info(midi_data)
                metadata.update(time_sig_info)
            
            # Extract key signature information
            if self.extract_harmony:
                key_sig_info = self._extract_key_signature_info(midi_data)
                metadata.update(key_sig_info)
            
            # Extract note statistics
            note_stats = self._extract_note_statistics(midi_data)
            metadata.update(note_stats)
            
            # Extract chord progressions if enabled
            if self.extract_harmony:
                chord_info = self._extract_chord_info(midi_data)
                metadata.update(chord_info)
            
            # Extract structural information if enabled
            if self.extract_structure:
                structure_info = self._extract_structure_info(midi_data)
                metadata.update(structure_info)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error extracting metadata from {midi_path}: {e}")
            raise ValueError(f"Invalid MIDI file: {midi_path}") from e
    
    def _get_duration_seconds(self, midi_data: miditoolkit.MidiFile) -> float:
        """Calculate duration in seconds from MIDI data."""
        if midi_data.max_tick == 0:
            return 0.0
        
        # Get the last tempo change or use default
        tempo_bpm = 120.0  # Default tempo
        if midi_data.tempo_changes:
            tempo_bpm = midi_data.tempo_changes[-1].tempo
        
        # Convert ticks to seconds
        seconds_per_beat = 60.0 / tempo_bpm
        beats = midi_data.max_tick / midi_data.ticks_per_beat
        return beats * seconds_per_beat
    
    def _extract_tempo_info(self, midi_data: miditoolkit.MidiFile) -> Dict[str, Any]:
        """Extract tempo information from MIDI data."""
        tempo_info = {
            "tempo_bpm": 120.0,  # Default tempo
            "tempo_changes": [],
        }
        
        # Get tempo changes
        if midi_data.tempo_changes:
            # Use the first tempo as the main tempo
            tempo_info["tempo_bpm"] = float(midi_data.tempo_changes[0].tempo)
            
            # Convert tempo changes to time-based format
            tempo_info["tempo_changes"] = [
                (self._tick_to_seconds(tc.time, midi_data), float(tc.tempo))
                for tc in midi_data.tempo_changes
            ]
        
        return tempo_info
    
    def _extract_time_signature_info(self, midi_data: miditoolkit.MidiFile) -> Dict[str, Any]:
        """Extract time signature information from MIDI data."""
        time_sig_info = {
            "time_signature_numerator": 4,
            "time_signature_denominator": 4,
            "time_signature_changes": [],
        }
        
        # Get time signature changes
        if midi_data.time_signature_changes:
            first_ts = midi_data.time_signature_changes[0]
            time_sig_info["time_signature_numerator"] = first_ts.numerator
            time_sig_info["time_signature_denominator"] = first_ts.denominator
            
            time_sig_info["time_signature_changes"] = [
                (
                    self._tick_to_seconds(ts.time, midi_data),
                    int(ts.numerator),
                    int(ts.denominator)
                )
                for ts in midi_data.time_signature_changes
            ]
        
        return time_sig_info
    
    def _extract_key_signature_info(self, midi_data: miditoolkit.MidiFile) -> Dict[str, Any]:
        """Extract key signature information from MIDI data."""
        key_sig_info = {
            "key_signature": 0,
            "key_signature_changes": [],
        }
        
        # Get key signature changes
        if midi_data.key_signature_changes:
            first_ks = midi_data.key_signature_changes[0]
            key_sig_info["key_signature"] = first_ks.key_number
            
            key_sig_info["key_signature_changes"] = [
                (
                    self._tick_to_seconds(ks.time, midi_data),
                    int(ks.key_number)
                )
                for ks in midi_data.key_signature_changes
            ]
        
        return key_sig_info
    
    def _extract_note_statistics(self, midi_data: miditoolkit.MidiFile) -> Dict[str, Any]:
        """Extract note statistics from MIDI data."""
        all_notes = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                all_notes.extend(instrument.notes)
        
        if not all_notes:
            return {
                "total_notes": 0,
                "note_range_min": 60,
                "note_range_max": 72,
                "avg_velocity": 64,
                "avg_note_duration": 0.5,
                "polyphony_max": 0,
                "polyphony_avg": 0.0,
            }
        
        note_stats = {
            "total_notes": len(all_notes),
            "note_range_min": min(note.pitch for note in all_notes),
            "note_range_max": max(note.pitch for note in all_notes),
            "avg_velocity": np.mean([note.velocity for note in all_notes]),
            "avg_note_duration": np.mean([
                self._tick_to_seconds(note.end - note.start, midi_data)
                for note in all_notes
            ]),
        }
        
        # Calculate polyphony
        polyphony = self._calculate_polyphony(all_notes, midi_data)
        note_stats.update(polyphony)
        
        return note_stats
    
    def _calculate_polyphony(
        self, 
        notes: List[miditoolkit.Note], 
        midi_data: miditoolkit.MidiFile
    ) -> Dict[str, Any]:
        """Calculate polyphony statistics from notes."""
        if not notes:
            return {"polyphony_max": 0, "polyphony_avg": 0.0}
        
        # Create time grid
        start_time = self._tick_to_seconds(min(note.start for note in notes), midi_data)
        end_time = self._tick_to_seconds(max(note.end for note in notes), midi_data)
        
        # Sample at 100ms intervals
        sample_rate = 10  # 10 samples per second
        times = np.arange(start_time, end_time, 1.0 / sample_rate)
        
        polyphony_values = []
        for time in times:
            active_notes = sum(
                1 for note in notes
                if (self._tick_to_seconds(note.start, midi_data) <= time <= 
                    self._tick_to_seconds(note.end, midi_data))
            )
            polyphony_values.append(active_notes)
        
        return {
            "polyphony_max": max(polyphony_values) if polyphony_values else 0,
            "polyphony_avg": np.mean(polyphony_values) if polyphony_values else 0.0,
        }
    
    def _extract_chord_info(self, midi_data: miditoolkit.MidiFile) -> Dict[str, Any]:
        """Extract chord progression information from MIDI data."""
        # This is a simplified chord extraction
        # In a full implementation, you might use more sophisticated chord detection
        
        chord_info = {
            "has_chords": False,
            "chord_changes": [],
            "chord_complexity": 0.0,
        }
        
        # Basic chord detection logic would go here
        # For now, return empty chord info
        
        return chord_info
    
    def _extract_structure_info(self, midi_data: miditoolkit.MidiFile) -> Dict[str, Any]:
        """Extract structural information from MIDI data."""
        structure_info = {
            "has_structure": False,
            "sections": [],
            "repetitions": [],
        }
        
        # Structural analysis logic would go here
        # For now, return empty structure info
        
        return structure_info
    
    def _tick_to_seconds(self, ticks: int, midi_data: miditoolkit.MidiFile) -> float:
        """Convert MIDI ticks to seconds."""
        if midi_data.ticks_per_beat == 0:
            return 0.0
        
        # Get the appropriate tempo for this tick position
        tempo_bpm = 120.0  # Default tempo
        
        # Find the tempo that applies to this tick
        for tempo_change in midi_data.tempo_changes:
            if tempo_change.time <= ticks:
                tempo_bpm = tempo_change.tempo
            else:
                break
        
        # Convert ticks to seconds
        seconds_per_beat = 60.0 / tempo_bpm
        beats = ticks / midi_data.ticks_per_beat
        return beats * seconds_per_beat 