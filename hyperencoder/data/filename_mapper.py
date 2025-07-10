"""
Filename mapping utilities for MIDI metadata injection.

This module provides functionality to map latent filenames to corresponding
MIDI filenames based on directory structure.
"""

import logging
from pathlib import Path
from typing import Optional, List


class FilenameMapper:
    """
    Handle filename transformations between latent and MIDI files.
    
    This class implements the mapping strategy where the parent directory
    of the latent file is used as the MIDI filename.
    
    Example:
        Latent file: /path/to/latents/SongName/track_001.safetensors
        MIDI file:   /path/to/midi/SongName.mid
    """
    
    def __init__(
        self,
        default_extension: str = ".mid",
        alternative_extensions: List[str] = None
    ):
        """
        Initialize the filename mapper.
        
        Args:
            default_extension: Default MIDI file extension to try first
            alternative_extensions: List of alternative extensions to try
        """
        self.default_extension = default_extension
        self.alternative_extensions = alternative_extensions or [
            ".mid", ".midi", ".MID", ".MIDI"
        ]
        self.logger = logging.getLogger(__name__)
    
    def map_latent_to_midi(self, latent_path: Path) -> str:
        """
        Map a latent file path to corresponding MIDI filename.
        
        Uses the parent directory name of the latent file as the MIDI filename
        (without extension).
        
        Args:
            latent_path: Path to the latent file
            
        Returns:
            MIDI filename with default extension
            
        Example:
            >>> mapper = FilenameMapper()
            >>> latent_path = Path("/data/latents/SongName/track_001.safetensors")
            >>> mapper.map_latent_to_midi(latent_path)
            "SongName.mid"
        """
        # Get the parent directory name
        parent_dir = latent_path.parent.name
        
        # Create MIDI filename with default extension
        midi_filename = f"{parent_dir}{self.default_extension}"
        
        self.logger.debug(f"Mapped {latent_path} -> {midi_filename}")
        return midi_filename
    
    def find_midi_file(self, latent_path: Path, midi_dir: Path) -> Optional[Path]:
        """
        Find the corresponding MIDI file for a latent file.
        
        Tries the default extension first, then alternative extensions.
        
        Args:
            latent_path: Path to the latent file
            midi_dir: Directory containing MIDI files
            
        Returns:
            Path to the MIDI file if found, None otherwise
        """
        if not midi_dir.exists():
            self.logger.warning(f"MIDI directory does not exist: {midi_dir}")
            return None
        
        # Get the base MIDI filename (without extension)
        parent_dir = latent_path.parent.name
        
        # Try default extension first
        midi_path = midi_dir / f"{parent_dir}{self.default_extension}"
        if midi_path.exists():
            self.logger.debug(f"Found MIDI file: {midi_path}")
            return midi_path
        
        # Try alternative extensions
        for ext in self.alternative_extensions:
            if ext == self.default_extension:
                continue  # Already tried this one
            
            midi_path = midi_dir / f"{parent_dir}{ext}"
            if midi_path.exists():
                self.logger.debug(f"Found MIDI file with alternative extension: {midi_path}")
                return midi_path
        
        self.logger.debug(f"No MIDI file found for {latent_path} in {midi_dir}")
        return None
    
    def validate_mapping(self, latent_path: Path, midi_dir: Path) -> bool:
        """
        Validate that a MIDI file exists for the given latent file.
        
        Args:
            latent_path: Path to the latent file
            midi_dir: Directory containing MIDI files
            
        Returns:
            True if corresponding MIDI file exists, False otherwise
        """
        midi_path = self.find_midi_file(latent_path, midi_dir)
        return midi_path is not None
    
    def get_mapping_info(self, latent_path: Path, midi_dir: Path) -> dict:
        """
        Get detailed mapping information for debugging.
        
        Args:
            latent_path: Path to the latent file
            midi_dir: Directory containing MIDI files
            
        Returns:
            Dictionary with mapping information
        """
        parent_dir = latent_path.parent.name
        midi_path = self.find_midi_file(latent_path, midi_dir)
        
        return {
            "latent_file": str(latent_path),
            "parent_directory": parent_dir,
            "midi_directory": str(midi_dir),
            "expected_midi_filename": f"{parent_dir}{self.default_extension}",
            "found_midi_file": str(midi_path) if midi_path else None,
            "mapping_successful": midi_path is not None
        } 