import os

# import time  # Import time for delay
import random
import logging
from os import walk
from enum import Enum
from typing import Any
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from numpy import ceil, floor
from torch import Tensor, Generator, stack, squeeze
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader, random_split
from safetensors.torch import load_file

from hyperencoder.datamodels import MidiMetadataConfig, MidiMetadata
from .midi_extractor import MidiMetadataExtractor
from .filename_mapper import FilenameMapper

DEFAULT_FILE_SUFFIXES = [
    ".safetensors",
    ".json",
    "_reconstructed.wav",
    "_original_trimmed.wav",
]
DEFAULT_SUFFIX_KEYS = [
    "latents",
    "pre_encode_config",
    "reconstructed_audio",
    "original_audio",
]

DEFAULT_SUFFIX_MAPPING = dict(
    zip(DEFAULT_FILE_SUFFIXES, DEFAULT_SUFFIX_KEYS, strict=False)
)


class LatentLoadStrategy(Enum):
    LAZY = "lazy"
    EAGER = "eager"
    LAZY_CACHED = "lazy_cached"


def create_random_cropper(crop_length, crop_ratio):
    # log = logging.getLogger()
    def random_cropper(latents, infos):
        # crop length is in seconds
        chunk_pct = 1 / crop_ratio
        start_pct = random.uniform(0, 1.0 - chunk_pct)
        end_pct = start_pct + chunk_pct
        # log.info(f"Start pct is: {start_pct}")

        latent_length = latents.shape[-1]
        latent_crop_start = int(round(latent_length * start_pct))
        latent_crop_length = int(round(latent_length * chunk_pct))

        # log.info(f"Latent Shape: {latents.shape}")
        # log.info(f"Crop Length is: {latent_crop_length}")
        # log.info(f"Crop Interval is: {latent_crop_start} to {(latent_crop_start + latent_crop_length)}")

        cropped_latents = latents[
            :, :, latent_crop_start : (latent_crop_start + latent_crop_length)
        ].clone()

        reals = infos["trimmed_input_reals"]
        decoded_reals = infos["decoded_reals"]

        real_length = reals.shape[-1]
        real_crop_start = int(round(real_length * start_pct))
        real_crop_length = int(round(real_length * chunk_pct))

        cropped_reals = reals[
            :, :, real_crop_start : (real_crop_start + real_crop_length)
        ].clone()
        cropped_decoded_reals = decoded_reals[
            :, :, real_crop_start : (real_crop_start + real_crop_length)
        ].clone()

        # crop_length = floor(latent_length / crop_ratio)
        # rand_start = random.randint(0, latent_length - crop_length - 1)

        # log.info(f"Info Members: {list(infos.keys())}")

        new_infos = {
            "root": infos["root"],
            "prefix": infos["prefix"],
            "crop_start_pct": round(number=start_pct, ndigits=4),
            "crop_end_pct": round(number=end_pct, ndigits=4),
            "crop_latent_start": latent_crop_start,
            "crop_latent_length": latent_crop_length,
            "crop_real_start": real_crop_start,
            "crop_real_length": real_crop_length,
            "cropped_reals": cropped_reals,
            "cropped_decoded_reals": cropped_decoded_reals,
        }

        return cropped_latents, new_infos

    return random_cropper


def get_cropper_from_config(crop_config):
    if crop_config["random_crop"]:
        return create_random_cropper(
            crop_config["original_crop_length"], crop_config["crop_ratio"]
        )
    else:
        raise NotImplementedError()


@dataclass(frozen=True)
class EncodedDirectoryInfo:
    root: str
    prefix: str
    latents: str
    pre_encode_config: str
    reconstructed_audio: str
    original_audio: str


"""
This is a dataset class that is used to load pre-encoded latent tensors
that were generated from the pre-encoding script. This is due to the specific 
folder structure, and file-naming conventions that were used to generate the
the ouptus. This is a bit of a hacky solution but ehh we balling brother.
"""


class PreEncodedLatentDataset(Dataset):
    def __init__(
        self,
        file_tuples: list[tuple[str, EncodedDirectoryInfo]],
        loading_strategy: LatentLoadStrategy = LatentLoadStrategy.LAZY,
        crop_config=None,
    ):
        self.file_tuples = file_tuples
        self.loading_strategy = loading_strategy
        self.latent_dict = {}

        self.cropper = None
        if crop_config is not None:
            self.cropper = get_cropper_from_config(crop_config)

        if loading_strategy == LatentLoadStrategy.EAGER:
            self.latent_dict = self.load_all(file_tuples)

    """
    This function gets the required filepaths from the pre-encoded
    base directory. The structure is determined by the outputs 
    from the pre-encoding script. This isn't the cleanest solution
    but it's the best I can do for now. The structure is as follows:
    - {prefix}_original_trimmed.wav
    - {prefix}_reconstructed.wav
    - {prefix}.json
    - {prefix}.safetensors

    The include stems parameter is used to include stems as part of the dataset,
    instead of just the overall mix.
    """

    @staticmethod
    def from_parent_dirs(
        parent_folders: list[str],
        prefix_filter: str | None = None,
        suffix_mapping: dict[str, str] | None = None,
        loading_strategy: LatentLoadStrategy = LatentLoadStrategy.LAZY,
    ):
        file_tuples = PreEncodedLatentDataset.collect_file_path_tuples(
            parent_folders, prefix_filter, suffix_mapping, loading_strategy
        )
        return PreEncodedLatentDataset(file_tuples)

    @staticmethod
    def collect_file_path_tuples(
        parent_folders: list[str],
        prefix_filter: str | None = None,
        suffix_mapping: dict[str, str] | None = None,
    ):
        import time
        
        logger = logging.getLogger()  # Use the existing logger
        start_time = time.time()
        logger.info(f"🔍 Scanning {len(parent_folders)} parent directories for latent files")
        
        prefix_filter_re = None

        if suffix_mapping is None:
            suffix_mapping = DEFAULT_SUFFIX_MAPPING
        if prefix_filter is not None:
            from re import compile

            prefix_filter_re = compile(prefix_filter)

        files_dicts = defaultdict(lambda: defaultdict(dict))
        tuples: list[tuple[str, EncodedDirectoryInfo]] = []
        total_files_scanned = 0

        def get_pref_from_suff(suff, file):
            return file.replace(suff, "")

        for i, path in enumerate(parent_folders):
            logger.debug(f"📁 Scanning directory {i+1}/{len(parent_folders)}: {path}")
            dir_start = time.time()
            dir_files = 0
            
            for root, _, files in walk(path):
                for file in files:
                    total_files_scanned += 1
                    dir_files += 1
                    for suff in suffix_mapping:
                        if file.endswith(suff):
                            pref = get_pref_from_suff(suff, file)
                            files_dicts[root][pref][suffix_mapping[suff]] = file
            
            logger.debug(f"⏱️ Directory scan took {time.time() - dir_start:.2f}s, found {dir_files} files")

        for root, dir_dict in files_dicts.items():
            for pref, suff_dict in dir_dict.items():
                latents_path = os.path.join(root, suff_dict.get("latents", ""))
                if not os.path.exists(latents_path):
                    logger.warning(f"Latents file not found: {latents_path}. Skipping.")
                    continue

                try:
                    out_tuple = (
                        latents_path,
                        EncodedDirectoryInfo(
                            **{"root": root, "prefix": pref, **suff_dict}
                        ),
                    )
                    if prefix_filter_re is not None:
                        if prefix_filter_re.match(pref):
                            tuples.append(out_tuple)
                    else:
                        tuples.append(out_tuple)
                except KeyError as e:
                    warn_msg = f"""Missing expected key in suffix mapping: {e}. 
Skipping {latents_path}."""
                    logger.warning(warn_msg)
                    continue

        scan_duration = time.time() - start_time
        logger.info(f"✅ File scan complete: found {len(tuples)} valid latent files")
        logger.info(f"📊 Scanned {total_files_scanned:,} total files in {scan_duration:.2f}s")
        logger.debug(f"⏱️ Average scan speed: {total_files_scanned/scan_duration:.0f} files/sec")
        
        return tuples

    @staticmethod
    def load_all(tuples):
        latent_dict = {}
        for idx in range(len(tuples)):
            latent, path, info = PreEncodedLatentDataset.load_item(tuples[idx])
            latent_dict[path] = (latent, info)
        return latent_dict

    @staticmethod
    def load_item(latent_tuple: tuple[str, dict[str, Any]]) -> tuple[Tensor, str, dict[str, Any]]:
        # logger = logging.getLogger()  # Use the existing logger

        latents_path, info = latent_tuple

        latents_sf = load_file(latents_path)
        latents = latents_sf["latents"]
        info = {**info.__dict__, **latents_sf}
        return latents, latents_path, info

        # if not os.path.exists(latents_path):
        #     logger.warning(
        # f"File not found during loading: {latents_path}. Skipping."
        # )
        #     return None, latents_path, info

        # retries = 20  # Number of retry attempts
        # delay_seconds = 5  # Delay between retries

        # for attempt in range(1, retries + 1):
        #     try:
        #         latents_sf = load_file(latents_path)
        #         latents = latents_sf["latents"]
        #         info = {**info.__dict__, **latents_sf}
        #         return latents, latents_path, info
        #     except Exception as e:
        #         logger.error(
        #             f"Error loading file {latents_path} (attempt {attempt}/{retries}): {e}"
        #         )
        #         if attempt < retries:
        #             time.sleep(delay_seconds)  # Delay before retrying
        #         else:
        #             logger.error(
        #                 f"Failed to load file {latents_path} after {retries} attempts. Skipping."
        #             )
        #             raise FileNotFoundError()

    def __len__(self):
        return len(self.file_tuples)

    def __getitem__(self, idx):
        if self.loading_strategy == LatentLoadStrategy.EAGER:
            latent_path, info = self.file_tuples[idx]
            latents, info = self.latent_dict[latent_path]
        elif self.loading_strategy == LatentLoadStrategy.LAZY:
            latents, path, info = self.load_item(self.file_tuples[idx])
        elif self.loading_strategy == LatentLoadStrategy.LAZY_CACHED:
            latent_path, info = self.file_tuples[idx]
            if latent_path in self.latent_dict:
                latents, info = self.latent_dict[latent_path]
            else:
                latents, path, info = self.load_item(self.file_tuples[idx])
                self.latent_dict[path] = (latents, info)
        else:
            raise NotImplementedError()
        # log = logging.getLogger()
        # log.info(f"Cropper: {self.cropper is not None}")

        if self.cropper is not None:
            # get a crop of a certain length, and safe start, end, and length in infos
            latents, info = self.cropper(latents, info)

        return latents, info


class PreEncodedLatentDataModule(LightningDataModule):
    def __init__(
        self,
        train_tuples: list[tuple[str, EncodedDirectoryInfo]] | None,
        val_tuples: list[tuple[str, EncodedDirectoryInfo]] | None,
        test_tuples: list[tuple[str, EncodedDirectoryInfo]] | None,
        predict_tuples: list[tuple[str, EncodedDirectoryInfo]] | None,
        loading_strategy: LatentLoadStrategy = LatentLoadStrategy.LAZY,
        batch_size: int = 32,
        num_workers: int = 4,
        persistent_workers=True,
        crop_config=None,
    ):
        super().__init__()
        self.train_tuples = train_tuples
        self.val_tuples = val_tuples
        self.test_tuples = test_tuples
        self.predict_tuples = predict_tuples
        self.loading_strategy = loading_strategy
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.crop_config = crop_config

    @staticmethod
    def from_dirs_per_dataset(
        train_dirs: list[str],
        val_dirs: list[str],
        test_dirs: list[str],
        predict_dirs: list[str],
        batch_size: int = 32,
        num_workers: int = 4,
        loading_strategy: LatentLoadStrategy = LatentLoadStrategy.LAZY,
        persistent_workers=True,
        crop_config=None,
    ):
        train_tuples = PreEncodedLatentDataset.collect_file_path_tuples(train_dirs)
        val_tuples = PreEncodedLatentDataset.collect_file_path_tuples(val_dirs)
        test_tuples = PreEncodedLatentDataset.collect_file_path_tuples(test_dirs)
        predict_tuples = PreEncodedLatentDataset.collect_file_path_tuples(predict_dirs)

        return PreEncodedLatentDataModule(
            train_tuples=train_tuples,
            val_tuples=val_tuples,
            test_tuples=test_tuples,
            predict_tuples=predict_tuples,
            loading_strategy=loading_strategy,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            crop_config=crop_config,
        )

    @staticmethod
    def from_single_dataset_splits(
        datadir_configs: list[dict],
        train_split_pct: float = 0.7,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.1,
        random_seed: int = 42,
        batch_size: int = 32,
        num_workers: int = 4,
        loading_strategy: LatentLoadStrategy = LatentLoadStrategy.LAZY,
        persistent_workers=True,
        crop_config=None,
    ):
        dir_paths = [config["path"] for config in datadir_configs]
        latents_tuples = PreEncodedLatentDataset.collect_file_path_tuples(dir_paths)
        logs = logging.getLogger()
        logs.info(
            f"{int(ceil(len(latents_tuples) * train_split_pct))}, {int(floor(len(latents_tuples) * val_split_pct))}, {int(floor(len(latents_tuples) * test_split_pct))}, {len(latents_tuples)}"
        )
        train_tuples, val_tuples, test_tuples = random_split(
            latents_tuples,
            [
                int(ceil(len(latents_tuples) * train_split_pct)),
                int(ceil(len(latents_tuples) * val_split_pct)),
                int(floor(len(latents_tuples) * test_split_pct)),
            ],
            generator=Generator().manual_seed(random_seed),
        )

        return PreEncodedLatentDataModule(
            train_tuples=train_tuples,
            val_tuples=val_tuples,
            test_tuples=test_tuples,
            predict_tuples=latents_tuples,
            loading_strategy=loading_strategy,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            crop_config=crop_config,
        )

    def setup(self, stage: str | None = None):
        if stage == "fit":
            self.train_dataset = PreEncodedLatentDataset(
                self.train_tuples,
                loading_strategy=self.loading_strategy,
                crop_config=self.crop_config,
            )
            self.val_dataset = PreEncodedLatentDataset(
                self.val_tuples,
                loading_strategy=self.loading_strategy,
                crop_config=self.crop_config,
            )
        if stage == "validate":
            self.val_dataset = PreEncodedLatentDataset(
                self.val_tuples,
                loading_strategy=self.loading_strategy,
                crop_config=self.crop_config,
            )
        if stage == "test":
            self.test_dataset = PreEncodedLatentDataset(
                self.test_tuples,
                loading_strategy=self.loading_strategy,
                crop_config=self.crop_config,
            )
        if stage == "predict":
            self.predict_dataset = PreEncodedLatentDataset(
                self.predict_tuples,
                loading_strategy=self.loading_strategy,
                crop_config=self.crop_config,
            )

    def get_collate_fn(self):
        def custom_collate_fn(batch):
            latents = [
                squeeze(item[0]) if item[0].dim() == 3 else item[0] for item in batch
            ]
            stacked_latents = stack(latents)

            infos = [item[1] for item in batch]

            return stacked_latents, infos

        return custom_collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.get_collate_fn(),
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.get_collate_fn(),
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.get_collate_fn(),
            persistent_workers=self.persistent_workers,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.get_collate_fn(),
            persistent_workers=self.persistent_workers,
        )


class MidiEnhancedLatentDataset(PreEncodedLatentDataset):
    """
    PreEncodedLatentDataset enhanced with MIDI metadata injection.
    
    This dataset extends PreEncodedLatentDataset to automatically extract
    and inject MIDI metadata into the info dict based on the configured
    MIDI metadata settings.
    """
    
    def __init__(
        self,
        file_tuples: list[tuple[str, EncodedDirectoryInfo]],
        loading_strategy: LatentLoadStrategy = LatentLoadStrategy.LAZY,
        crop_config=None,
        midi_metadata_config: MidiMetadataConfig | None = None,
    ):
        """
        Initialize the MIDI-enhanced latent dataset.
        
        Args:
            file_tuples: List of (latents_path, EncodedDirectoryInfo) tuples
            loading_strategy: How to load the latent data
            crop_config: Configuration for cropping operations
            midi_metadata_config: Configuration for MIDI metadata injection
        """
        super().__init__(file_tuples, loading_strategy, crop_config)
        
        self.midi_config = midi_metadata_config
        self.midi_extractor = None
        self.filename_mapper = None
        self.midi_cache = {}  # Cache for extracted MIDI metadata
        
        # Initialize MIDI processing components if configured
        if midi_metadata_config and midi_metadata_config.enabled:
            self._setup_midi_processing()
    
    def _setup_midi_processing(self):
        """Initialize MIDI processing components."""
        if not self.midi_config:
            return
            
        # Initialize MIDI metadata extractor
        self.midi_extractor = MidiMetadataExtractor(
            extract_harmony=self.midi_config.extract_harmony,
            extract_rhythm=self.midi_config.extract_rhythm
        )
        
        # Initialize filename mapper
        self.filename_mapper = FilenameMapper(
            default_extension=self.midi_config.default_midi_extension,
            alternative_extensions=self.midi_config.alternative_extensions
        )
        
        # Set up caching if enabled
        if self.midi_config.enable_caching:
            self._setup_midi_cache()
    
    def _setup_midi_cache(self):
        """Set up MIDI metadata caching."""
        # For now, use in-memory cache
        # In the future, could implement persistent caching to disk
        self.midi_cache = {}
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset with optional MIDI metadata injection.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (latents, info) where info may contain MIDI metadata
        """
        # Get the original item from parent class
        latents, info = super().__getitem__(idx)
        
        # Inject MIDI metadata if configured
        if self.midi_config and self.midi_config.enabled:
            midi_metadata = self._extract_midi_metadata(info)
            if midi_metadata:
                info["midi_metadata"] = midi_metadata.model_dump()
        
        return latents, info
    
    def _extract_midi_metadata(self, info: dict) -> MidiMetadata | None:
        """
        Extract MIDI metadata for the current item.
        
        Args:
            info: Info dict containing file paths and metadata
            
        Returns:
            MidiMetadata object if extraction successful, None otherwise
        """
        if not self.midi_config or not self.midi_config.midi_dir:
            return None
        
        # Get the latent file path from info
        latents_path = info.get("latents", "")
        if not latents_path:
            return None
        
        latents_path = Path(latents_path)
        
        # Check cache first if caching is enabled
        if self.midi_config.enable_caching and str(latents_path) in self.midi_cache:
            return self.midi_cache[str(latents_path)]
        
        # Set up MIDI directory
        midi_dir = Path(self.midi_config.midi_dir)
        if not midi_dir.exists():
            if self.midi_config.fail_on_missing_midi:
                raise FileNotFoundError(f"MIDI directory not found: {midi_dir}")
            return None
        
        # Find corresponding MIDI file
        midi_path = self.filename_mapper.find_midi_file(latents_path, midi_dir)
        if not midi_path:
            if self.midi_config.fail_on_missing_midi:
                expected_name = self.filename_mapper.map_latent_to_midi(latents_path)
                raise FileNotFoundError(
                    f"MIDI file not found for {latents_path}. "
                    f"Expected: {midi_dir / expected_name}"
                )
            return None
        
        # Extract metadata
        try:
            metadata = self.midi_extractor.extract_metadata(midi_path)
            
            # Cache the result if caching is enabled
            if self.midi_config.enable_caching:
                self.midi_cache[str(latents_path)] = metadata
            
            return metadata
            
        except Exception as e:
            if self.midi_config.fail_on_missing_midi:
                raise RuntimeError(
                    f"Failed to extract MIDI metadata from {midi_path}: {e}"
                ) from e
            return None
    
    def get_midi_mapping_info(self, idx: int) -> dict:
        """
        Get detailed MIDI mapping information for debugging.
        
        Args:
            idx: Index of the item
            
        Returns:
            Dictionary with mapping information
        """
        if not self.midi_config or not self.midi_config.enabled:
            return {"midi_enabled": False}
        
        # Get the latent file path
        latents_path, _ = self.file_tuples[idx]
        latents_path = Path(latents_path)
        
        midi_dir = Path(self.midi_config.midi_dir) if self.midi_config.midi_dir else None
        
        if not midi_dir:
            return {"midi_enabled": True, "midi_dir": None, "error": "No MIDI directory configured"}
        
        # Use filename mapper to get detailed info
        mapping_info = self.filename_mapper.get_mapping_info(latents_path, midi_dir)
        mapping_info["midi_enabled"] = True
        
        return mapping_info
    
    def validate_midi_mappings(self) -> dict:
        """
        Validate MIDI mappings for the entire dataset.
        
        Returns:
            Dictionary with validation results
        """
        if not self.midi_config or not self.midi_config.enabled:
            return {"midi_enabled": False}
        
        midi_dir = Path(self.midi_config.midi_dir) if self.midi_config.midi_dir else None
        
        if not midi_dir or not midi_dir.exists():
            return {
                "midi_enabled": True,
                "midi_dir": str(midi_dir) if midi_dir else None,
                "error": "MIDI directory not found"
            }
        
        # Check all mappings
        total_files = len(self.file_tuples)
        successful_mappings = 0
        failed_mappings = []
        
        for idx, (latents_path, _) in enumerate(self.file_tuples):
            latents_path = Path(latents_path)
            
            if self.filename_mapper.validate_mapping(latents_path, midi_dir):
                successful_mappings += 1
            else:
                failed_mappings.append({
                    "index": idx,
                    "latents_path": str(latents_path),
                    "expected_midi": self.filename_mapper.map_latent_to_midi(latents_path)
                })
        
        return {
            "midi_enabled": True,
            "total_files": total_files,
            "successful_mappings": successful_mappings,
            "failed_mappings": len(failed_mappings),
            "success_rate": successful_mappings / total_files if total_files > 0 else 0.0,
            "failed_mapping_details": failed_mappings[:10]  # Show first 10 failures
        }
