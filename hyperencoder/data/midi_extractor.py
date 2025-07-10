"""
MIDI metadata extraction using miditoolkit.

This module provides functionality to extract comprehensive metadata from MIDI files
for use in the hyperencoder dataset.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import miditoolkit
import numpy as np
from miditoolkit.midi.containers import Instrument, Note

from hyperencoder.datamodels import MidiMetadata


class MidiMetadataExtractor:
    """
    Extract comprehensive metadata from MIDI files using miditoolkit.
    
    This class provides methods to extract various types of musical metadata
    from MIDI files, including tempo, time signatures, note statistics, and
    optionally harmonic and rhythmic analysis.
    """
    
    def __init__(
        self,
        extract_harmony: bool = False,
        extract_rhythm: bool = False,
        extractor_version: str = "1.0.0"
    ):
        """
        Initialize the MIDI metadata extractor.
        
        Args:
            extract_harmony: Whether to extract harmonic analysis
            extract_rhythm: Whether to extract rhythmic patterns
            extractor_version: Version string for the extractor
        """
        self.extract_harmony = extract_harmony
        self.extract_rhythm = extract_rhythm
        self.extractor_version = extractor_version
        self.logger = logging.getLogger(__name__)
    
    def extract_metadata(self, midi_path: Path) -> MidiMetadata:
        """
        Extract metadata from a MIDI file.
        
        Args:
            midi_path: Path to the MIDI file
            
        Returns:
            MidiMetadata object with extracted metadata
            
        Raises:
            FileNotFoundError: If MIDI file doesn't exist
            RuntimeError: If MIDI file cannot be parsed
        """
        if not midi_path.exists():
            raise FileNotFoundError(f"MIDI file not found: {midi_path}")
        
        try:
            # Load the MIDI file
            midi_obj = miditoolkit.midi.parser.MidiFile(str(midi_path))
            
            # Extract all metadata components
            basic_info = self._extract_basic_info(midi_obj, midi_path)
            tempo_info = self._extract_tempo_info(midi_obj)
            time_sig_info = self._extract_time_signature_info(midi_obj)
            key_sig_info = self._extract_key_signature_info(midi_obj)
            note_stats = self._extract_note_statistics(midi_obj)
            track_info = self._extract_track_info(midi_obj)
            instrument_info = self._extract_instrument_info(midi_obj)
            rhythmic_info = self._extract_rhythmic_info(midi_obj, tempo_info)
            
            # Optional advanced analysis
            harmonic_info = {}
            rhythmic_patterns = {}
            
            if self.extract_harmony:
                harmonic_info = self._extract_harmonic_analysis(midi_obj)
                
            if self.extract_rhythm:
                rhythmic_patterns = self._extract_rhythmic_patterns(midi_obj)
            
            # Combine all metadata
            metadata = MidiMetadata(
                # Basic info
                filename=midi_path.name,
                duration_seconds=basic_info["duration_seconds"],
                
                # Tempo info
                tempo_bpm=tempo_info["primary_tempo"],
                tempo_changes=tempo_info["tempo_changes"],
                
                # Time signature info
                time_signature_numerator=time_sig_info["primary_numerator"],
                time_signature_denominator=time_sig_info["primary_denominator"],
                time_signature_changes=time_sig_info["time_signature_changes"],
                
                # Key signature info
                key_signature=key_sig_info["primary_key"],
                key_signature_changes=key_sig_info["key_signature_changes"],
                
                # Note statistics
                total_notes=note_stats["total_notes"],
                note_range_min=note_stats["note_range_min"],
                note_range_max=note_stats["note_range_max"],
                note_range_span=note_stats["note_range_span"],
                average_velocity=note_stats["average_velocity"],
                velocity_std=note_stats["velocity_std"],
                
                # Track info
                num_tracks=track_info["num_tracks"],
                track_names=track_info["track_names"],
                num_channels=track_info["num_channels"],
                
                # Instrument info
                unique_programs=instrument_info["unique_programs"],
                program_changes=instrument_info["program_changes"],
                
                # Rhythmic info
                note_density=rhythmic_info["note_density"],
                beat_density=rhythmic_info["beat_density"],
                
                # Optional advanced features
                chord_progressions=harmonic_info.get("chord_progressions"),
                key_centers=harmonic_info.get("key_centers"),
                rhythmic_patterns=rhythmic_patterns if rhythmic_patterns else None,
                
                # Processing metadata
                extraction_timestamp=time.time(),
                extractor_version=self.extractor_version
            )
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {midi_path}: {e}")
            raise RuntimeError(f"Failed to parse MIDI file {midi_path}: {e}")
    
    def _extract_basic_info(self, midi_obj: miditoolkit.MidiFile, midi_path: Path) -> Dict:
        """Extract basic MIDI file information."""
        # Calculate duration based on the last note/event
        max_time = 0
        for instrument in midi_obj.instruments:
            if instrument.notes:
                max_time = max(max_time, max(note.end for note in instrument.notes))
        
        # Convert ticks to seconds
        duration_seconds = midi_obj.tick_to_second(max_time) if max_time > 0 else 0.0
        
        return {
            "duration_seconds": duration_seconds
        }
    
    def _extract_tempo_info(self, midi_obj: miditoolkit.MidiFile) -> Dict:
        """Extract tempo information."""
        tempo_changes = []
        primary_tempo = 120.0  # Default tempo
        
        if midi_obj.tempo_changes:
            # Get primary tempo (first tempo change)
            primary_tempo = midi_obj.tempo_changes[0].tempo
            
            # Get all tempo changes
            for tempo_change in midi_obj.tempo_changes:
                time_seconds = midi_obj.tick_to_second(tempo_change.time)
                tempo_changes.append((time_seconds, tempo_change.tempo))
        
        return {
            "primary_tempo": primary_tempo,
            "tempo_changes": tempo_changes
        }
    
    def _extract_time_signature_info(self, midi_obj: miditoolkit.MidiFile) -> Dict:
        """Extract time signature information."""
        time_signature_changes = []
        primary_numerator = 4
        primary_denominator = 4
        
        if midi_obj.time_signature_changes:
            # Get primary time signature
            primary_ts = midi_obj.time_signature_changes[0]
            primary_numerator = primary_ts.numerator
            primary_denominator = primary_ts.denominator
            
            # Get all time signature changes
            for ts_change in midi_obj.time_signature_changes:
                time_seconds = midi_obj.tick_to_second(ts_change.time)
                time_signature_changes.append((
                    time_seconds,
                    ts_change.numerator,
                    ts_change.denominator
                ))
        
        return {
            "primary_numerator": primary_numerator,
            "primary_denominator": primary_denominator,
            "time_signature_changes": time_signature_changes
        }
    
    def _extract_key_signature_info(self, midi_obj: miditoolkit.MidiFile) -> Dict:
        """Extract key signature information."""
        key_signature_changes = []
        primary_key = 0  # C major
        
        if midi_obj.key_signature_changes:
            # Get primary key signature
            primary_key = midi_obj.key_signature_changes[0].key_number
            
            # Get all key signature changes
            for key_change in midi_obj.key_signature_changes:
                time_seconds = midi_obj.tick_to_second(key_change.time)
                key_signature_changes.append((time_seconds, key_change.key_number))
        
        return {
            "primary_key": primary_key,
            "key_signature_changes": key_signature_changes
        }
    
    def _extract_note_statistics(self, midi_obj: miditoolkit.MidiFile) -> Dict:
        """Extract note-level statistics."""
        all_notes = []
        
        for instrument in midi_obj.instruments:
            if not instrument.is_drum:  # Skip drum tracks for pitch analysis
                all_notes.extend(instrument.notes)
        
        if not all_notes:
            return {
                "total_notes": 0,
                "note_range_min": 0,
                "note_range_max": 0,
                "note_range_span": 0,
                "average_velocity": 0.0,
                "velocity_std": 0.0
            }
        
        # Calculate statistics
        pitches = [note.pitch for note in all_notes]
        velocities = [note.velocity for note in all_notes]
        
        note_range_min = min(pitches)
        note_range_max = max(pitches)
        note_range_span = note_range_max - note_range_min
        
        average_velocity = np.mean(velocities)
        velocity_std = np.std(velocities)
        
        return {
            "total_notes": len(all_notes),
            "note_range_min": note_range_min,
            "note_range_max": note_range_max,
            "note_range_span": note_range_span,
            "average_velocity": float(average_velocity),
            "velocity_std": float(velocity_std)
        }
    
    def _extract_track_info(self, midi_obj: miditoolkit.MidiFile) -> Dict:
        """Extract track information."""
        track_names = []
        channels_used = set()
        
        for instrument in midi_obj.instruments:
            # Track names
            if instrument.name:
                track_names.append(instrument.name)
            else:
                track_names.append(f"Track_{instrument.program}")
            
            # Channels used
            channels_used.add(instrument.channel)
        
        return {
            "num_tracks": len(midi_obj.instruments),
            "track_names": track_names,
            "num_channels": len(channels_used)
        }
    
    def _extract_instrument_info(self, midi_obj: miditoolkit.MidiFile) -> Dict:
        """Extract instrument/program information."""
        unique_programs = []
        program_changes = []
        
        for instrument in midi_obj.instruments:
            if instrument.program not in unique_programs:
                unique_programs.append(instrument.program)
        
        # Extract program changes if available
        # Note: miditoolkit doesn't directly expose program changes timeline
        # This would require more complex parsing of the raw MIDI events
        
        return {
            "unique_programs": unique_programs,
            "program_changes": program_changes  # Empty for now
        }
    
    def _extract_rhythmic_info(self, midi_obj: miditoolkit.MidiFile, tempo_info: Dict) -> Dict:
        """Extract rhythmic information."""
        all_notes = []
        
        for instrument in midi_obj.instruments:
            all_notes.extend(instrument.notes)
        
        if not all_notes:
            return {
                "note_density": 0.0,
                "beat_density": 0.0
            }
        
        # Calculate note density (notes per second)
        max_time = max(note.end for note in all_notes)
        duration_seconds = midi_obj.tick_to_second(max_time)
        note_density = len(all_notes) / duration_seconds if duration_seconds > 0 else 0.0
        
        # Calculate beat density (notes per beat)
        # Assuming 4/4 time signature for simplicity
        primary_tempo = tempo_info["primary_tempo"]
        beats_per_second = primary_tempo / 60.0
        total_beats = duration_seconds * beats_per_second
        beat_density = len(all_notes) / total_beats if total_beats > 0 else 0.0
        
        return {
            "note_density": note_density,
            "beat_density": beat_density
        }
    
    def _extract_harmonic_analysis(self, midi_obj: miditoolkit.MidiFile) -> Dict:
        """Extract harmonic analysis (chord progressions, key centers)."""
        # This is a placeholder for advanced harmonic analysis
        # Implementation would require more sophisticated music theory algorithms
        self.logger.info("Harmonic analysis not yet implemented")
        return {
            "chord_progressions": [],
            "key_centers": []
        }
    
    def _extract_rhythmic_patterns(self, midi_obj: miditoolkit.MidiFile) -> Dict:
        """Extract rhythmic patterns."""
        # This is a placeholder for advanced rhythmic analysis
        # Implementation would require pattern recognition algorithms
        self.logger.info("Rhythmic pattern analysis not yet implemented")
        return {} 