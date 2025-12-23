"""
Running this file will cut fine-tuning patches to be used with prithvi.

It assumes that two .tif files have been placed in data:

    SPECTRAL_FILE = 'data/planet_median_stAndrews.tif'
    MASK_FILE = 'data/SIMM_2024_w_seagrass_sand_water.tif'

-------------------------------------------------------------

## Description of files:
    From the SIMM Seagrass project two images are obtained of the St Andrew's sound region.

    1. A seagrass classification .tif created using the mode of classifications on images taken throughout a year.
    2. A spectral mean .tif created using the same images used to generate classifications.

    The aggregations use a total of 138 images covering 2022-2024, one aggregation is created for each year.

    The 2024 year is selected for creating this dataset.
    The 2024 spectral mean image is built from 72 images.
    This image cannot be shared due to licensing restrictions.

    The seagrass image can be created using [this GEE script](https://code.earthengine.google.com/?scriptPath=users%2Ftylarmurray%2Fprithvi%3Aprithvi_planet_median_creation).
    In addition to application of the seagrass classifier (GEE script available [here](From the SIMM Seagrass project two images are obtained of the St Andrew's sound region.

    1. A seagrass classification .tif created using the mode of classifications on images taken throughout a year.
    2. A spectral median .tif created using all Planet SuperDove images (346 images).

    The spectral mean image cannot be shared due to licensing restrictions.
    The GEE script to generate this image is [here](https://code.earthengine.google.com/?scriptPath=users%2Ftylarmurray%2Fprithvi%3Aprithvi_planet_median_creation).
    This script will not work unless you have access to the (restricted) Planet SuperDove image collection asset.

    The seagrass image for 2024 can be downloaded [here](https://usf.box.com/s/xr4zqg7vj9ynqxfn9zz0oj3r91ki66ui).
    In addition to application of the seagrass classifier (GEE script available [here](https://code.earthengine.google.com/23d1c15a67dfbc71564b67afdf394873)), manal adjustments may have been made to improve the final product.

-------------------------------------------------------------

This script uses the PrithviPatchExtractor class to extract training patches.
"""

from pathlib import Path
import shutil
import logging

from prithvi_patch_extractor import PrithviPatchExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    #level=logging.DEBUG,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # Configuration
    # SPECTRAL_FILE can be either:
        """
        Initialize the patch extractor.
        
        Args:
            spectral_path: Path to spectral bands TIF file, ZIP file containing TIF shards, 
                          or directory containing TIF files
            mask_path: Path to seagrass mask TIF file
            patch_size: Size of patches to extract (default 224x224 for Prithvi)
            stride: Step size between patches (default 224 for non-overlapping)
            output_dir: Directory to save extracted patches
        """
        self.spectral_path = Path(spectral_path)
        self.mask_path = mask_path
        self.patch_size = patch_size
        self.stride = stride
        self.output_dir = Path(output_dir)
        
        # Determine input type and collect spectral files
        self.is_zip = False
        self.is_directory = False
        self.temp_dir = None
        self.spectral_files = []
        
        if self.spectral_path.is_dir():
            print(f"Detected directory: {self.spectral_path}")
            self.is_directory = True
            self._collect_tif_files_from_directory()
        elif self.spectral_path.suffix.lower() == '.zip':
            print(f"Detected ZIP archive: {self.spectral_path}")
            self.is_zip = True
            self._extract_zip()
        else:
            print(f"Detected single TIF file: {self.spectral_path}")
            self.spectral_files = [self.spectral_path]
        
        # Create output directories
        self.spectral_dir = self.output_dir / 'spectral'
        self.mask_dir = self.output_dir / 'masks'
        self.spectral_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
    
    def _collect_tif_files_from_directory(self):
        """Collect all TIF files from a directory."""
        # Find all .tif and .tiff files in the directory (not recursive)
        tif_files = list(self.spectral_path.glob('*.tif')) + list(self.spectral_path.glob('*.tiff'))
        
        if not tif_files:
            raise ValueError(f"No .tif files found in directory: {self.spectral_path}")
        
        # Sort files for consistent ordering
        self.spectral_files = sorted(tif_files)
        print(f"Found {len(self.spectral_files)} TIF files in directory")
        
        # Print first few filenames for verification
        if len(self.spectral_files) <= 5:
            for f in self.spectral_files:
                print(f"  - {f.name}")
        else:
            for f in self.spectral_files[:3]:
                print(f"  - {f.name}")
            print(f"  ... and {len(self.spectral_files) - 3} more files")
    
    def _extract_zip(self):
        """Extract TIF files from ZIP archive to temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix='spectral_shards_'))
        print(f"Extracting ZIP to temporary directory: {self.temp_dir}")
        
        with zipfile.ZipFile(self.spectral_path, 'r') as zip_ref:
            # Get all .tif files in the archive
            tif_files = [f for f in zip_ref.namelist() if f.lower().endswith('.tif') or f.lower().endswith('.tiff')]
            
            if not tif_files:
                raise ValueError(f"No .tif files found in {self.spectral_path}")
            
            print(f"Found {len(tif_files)} TIF files in archive")
            
            # Extract all TIF files
            for tif_file in tif_files:
                zip_ref.extract(tif_file, self.temp_dir)
                extracted_path = self.temp_dir / tif_file
                self.spectral_files.append(extracted_path)
        
        print(f"Extracted {len(self.spectral_files)} spectral shard files")
    
    def __del__(self):
        """Cleanup temporary directory if it was created."""
        if self.temp_dir and self.temp_dir.exists():
            print(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
    
    def cleanup(self):
        """Manually cleanup temporary directory."""
        if self.temp_dir and self.temp_dir.exists():
            print(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
        
    def _calculate_overlap_window(self, spectral_src, mask_src):
        """
        Calculate the overlapping spatial extent between spectral and mask files.
        Handles CRS mismatches by transforming bounds to a common coordinate system.
        
        Args:
            spectral_src: Opened rasterio spectral dataset
            mask_src: Opened rasterio mask dataset
        
        Returns:
            Tuple of (spectral_window, mask_window, overlap_shape)
        """
        from rasterio.windows import from_bounds
        from rasterio.warp import transform_bounds
        
        print(f"Spectral CRS: {spectral_src.crs}")
        print(f"Mask CRS: {mask_src.crs}")
        
        # Get bounds for both datasets
        spec_bounds = spectral_src.bounds
        mask_bounds = mask_src.bounds
        
        print(f"Spectral bounds (native): {spec_bounds}")
        print(f"Mask bounds (native): {mask_bounds}")
        
        # If CRS don't match, transform mask bounds to spectral CRS
        if spectral_src.crs != mask_src.crs:
            print(f"\nCRS mismatch detected! Transforming mask bounds to spectral CRS...")
            mask_bounds_transformed = transform_bounds(
                mask_src.crs,
                spectral_src.crs,
                mask_bounds.left,
                mask_bounds.bottom,
                mask_bounds.right,
                mask_bounds.top
            )
            print(f"Mask bounds (transformed to spectral CRS): {mask_bounds_transformed}")
            mask_bounds = type('Bounds', (), {
                'left': mask_bounds_transformed[0],
                'bottom': mask_bounds_transformed[1],
                'right': mask_bounds_transformed[2],
                'top': mask_bounds_transformed[3]
            })()
        
        # Calculate intersection (overlap) in spectral CRS
        overlap_left = max(spec_bounds.left, mask_bounds.left)
        overlap_bottom = max(spec_bounds.bottom, mask_bounds.bottom)
        overlap_right = min(spec_bounds.right, mask_bounds.right)
        overlap_top = min(spec_bounds.top, mask_bounds.top)
        
        print(f"\nOverlap bounds (in spectral CRS): left={overlap_left:.2f}, bottom={overlap_bottom:.2f}, right={overlap_right:.2f}, top={overlap_top:.2f}")
        
        # Check if there's any overlap
        if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
            raise ValueError(
                f"No spatial overlap between spectral and mask files!\n"
                f"Spectral bounds: {spec_bounds}\n"
                f"Mask bounds (transformed): {mask_bounds}\n"
                f"Check that both files cover the same geographic area."
            )
        
        # Create windows for each dataset based on overlap bounds
        spec_window = from_bounds(
            overlap_left, overlap_bottom, overlap_right, overlap_top,
            transform=spectral_src.transform
        )
        
        # For mask, we need to transform overlap bounds back to mask CRS if they differ
        if spectral_src.crs != mask_src.crs:
            mask_overlap_bounds = transform_bounds(
                spectral_src.crs,
                mask_src.crs,
                overlap_left,
                overlap_bottom,
                overlap_right,
                overlap_top
            )
            mask_window = from_bounds(
                mask_overlap_bounds[0],
                mask_overlap_bounds[1],
                mask_overlap_bounds[2],
                mask_overlap_bounds[3],
                transform=mask_src.transform
            )
        else:
            mask_window = from_bounds(
                overlap_left, overlap_bottom, overlap_right, overlap_top,
                transform=mask_src.transform
            )
        
        # Round windows to integer pixels
        spec_window = Window(
            int(round(spec_window.col_off)),
            int(round(spec_window.row_off)),
            int(round(spec_window.width)),
            int(round(spec_window.height))
        )
        
        mask_window = Window(
            int(round(mask_window.col_off)),
            int(round(mask_window.row_off)),
            int(round(mask_window.width)),
            int(round(mask_window.height))
        )
        
        # Use the minimum dimensions to ensure exact match
        overlap_height = min(spec_window.height, mask_window.height)
        overlap_width = min(spec_window.width, mask_window.width)
        
        spec_window = Window(spec_window.col_off, spec_window.row_off, overlap_width, overlap_height)
        mask_window = Window(mask_window.col_off, mask_window.row_off, overlap_width, overlap_height)
        
        print(f"\nSpectral file shape: {spectral_src.shape}")
        print(f"Mask file shape: {mask_src.shape}")
        print(f"Overlap region: {overlap_height}x{overlap_width} pixels")
        print(f"Spectral window: {spec_window}")
        print(f"Mask window: {mask_window}\n")
        
        return spec_window, mask_window, (overlap_height, overlap_width)
    
    def extract_patches(self, min_valid_pixels=0.00001, save_metadata=True):
        """
        Extract patches from both spectral and mask files.
        Processes all spectral shards if a ZIP file was provided.
        
        Args:
            min_valid_pixels: Minimum fraction of valid (non-nodata) pixels required (0-1)
            save_metadata: Whether to save metadata JSON file
        
        Returns:
            Dictionary with extraction statistics
        """
        from rasterio.warp import reproject, Resampling
        
        stats = {
            'total_patches': 0,
            'valid_patches': 0,
            'skipped_patches': 0,
            'patch_size': self.patch_size,
            'stride': self.stride,
            'spectral_files_processed': 0
        }
        
        patch_id = 0
        
        # Open mask file once (used for all spectral shards)
        with rasterio.open(self.mask_path) as mask_src:
            
            # Process each spectral file (single file or multiple shards from ZIP)
            for shard_idx, spectral_file in enumerate(self.spectral_files):
                print(f"\n{'='*60}")
                print(f"Processing spectral file {shard_idx + 1}/{len(self.spectral_files)}: {spectral_file.name}")
                print(f"{'='*60}")
                
                stats['spectral_files_processed'] += 1
                
                with rasterio.open(spectral_file) as spectral_src:
                    # Debug: Read a small raw sample before any processing
                    raw_sample = spectral_src.read(1, window=Window(0, 0, min(10, spectral_src.width), min(10, spectral_src.height)))
                    logger.debug(f"  RAW DATA CHECK:")
                    logger.debug(f"    Raw 10x10 sample from (0,0): {raw_sample.flatten()[:20]}")
                    logger.debug(f"    Raw data type: {raw_sample.dtype}")
                    logger.debug(f"    Raw contains NaN: {np.isnan(raw_sample).any()}")
                    logger.debug(f"    Raw unique values (first 10): {np.unique(raw_sample)[:10]}")
                    
                    patch_id = self._extract_patches_from_shard(
                        spectral_src, mask_src, patch_id, min_valid_pixels, stats
                    )
        
        # Save metadata
        if save_metadata:
            metadata = {
                'spectral_path': str(self.spectral_path),
                'input_type': 'zip' if self.is_zip else ('directory' if self.is_directory else 'single_file'),
                'spectral_files_processed': stats['spectral_files_processed'],
                'mask_path': str(self.mask_path),
                'patch_size': self.patch_size,
                'stride': self.stride,
                'stats': stats
            }
            
            with open(self.output_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Extraction complete!")
        print(f"{'='*60}")
        print(f"Spectral files processed: {stats['spectral_files_processed']}")
        print(f"Valid patches saved: {stats['valid_patches']}")
        print(f"Patches skipped: {stats['skipped_patches']}")
        print(f"Output directory: {self.output_dir}")
        
        return stats
    
    def _extract_patches_from_shard(self, spectral_src, mask_src, patch_id, min_valid_pixels, stats):
        """
        Extract patches from a single spectral shard.
        
        Args:
            spectral_src: Opened rasterio spectral dataset
            mask_src: Opened rasterio mask dataset
            patch_id: Starting patch ID number
            min_valid_pixels: Minimum fraction of valid pixels
            stats: Statistics dictionary to update
        
        Returns:
            Updated patch_id for next shard
        """
        from rasterio.warp import reproject, Resampling
        
        # Check if CRS match
        if spectral_src.crs != mask_src.crs:
            print(f"CRS mismatch detected: {spectral_src.crs} vs {mask_src.crs}")
            print("Reprojecting mask to match spectral image...")
            
            # Create in-memory reprojected mask
            mask_reprojected = np.zeros((spectral_src.height, spectral_src.width), dtype=mask_src.dtypes[0])
            
            reproject(
                source=rasterio.band(mask_src, 1),
                destination=mask_reprojected,
                src_transform=mask_src.transform,
                src_crs=mask_src.crs,
                dst_transform=spectral_src.transform,
                dst_crs=spectral_src.crs,
                resampling=Resampling.nearest
            )
            
            print(f"Mask reprojected to match spectral image: {mask_reprojected.shape}")
            logger.debug(f"  Mask value range after reprojection: [{mask_reprojected.min():.6f}, {mask_reprojected.max():.6f}]")
            logger.debug(f"  Mask unique values: {np.unique(mask_reprojected)}")
            logger.debug(f"  Mask non-zero count: {np.count_nonzero(mask_reprojected)} / {mask_reprojected.size}")
            height, width = spectral_src.shape
            
        else:
            # If CRS match, calculate overlapping region
            spec_base_window, mask_base_window, (height, width) = self._calculate_overlap_window(
                spectral_src, mask_src
            )
            mask_reprojected = None
        
        n_bands = spectral_src.count
        
        # Debug: Check spectral data before extraction
        logger.debug(f"\n  Checking spectral source data:")
        logger.debug(f"  - Shape: {spectral_src.shape}")
        logger.debug(f"  - Bands: {n_bands}")
        logger.debug(f"  - Dtype: {spectral_src.dtypes[0]}")
        logger.debug(f"  - Nodata value: {spectral_src.nodata}")
        
        # Read a small sample from center to check data validity
        sample_size = min(100, spectral_src.height, spectral_src.width)
        sample_y = spectral_src.height // 2
        sample_x = spectral_src.width // 2
        sample_window = Window(sample_x, sample_y, sample_size, sample_size)
        sample_data = spectral_src.read(1, window=sample_window)  # Read first band
        
        logger.debug(f"  - Sample from center (band 1): shape={sample_data.shape}")
        logger.debug(f"    Value range: [{np.nanmin(sample_data):.6f}, {np.nanmax(sample_data):.6f}]")
        logger.debug(f"    NaN count: {np.isnan(sample_data).sum()} / {sample_data.size}")
        logger.debug(f"    Zero count: {(sample_data == 0).sum()}")
        logger.debug(f"    Non-zero, non-NaN count: {np.logical_and(~np.isnan(sample_data), sample_data != 0).sum()}")
        
        print(f"\nExtracting {self.patch_size}x{self.patch_size} patches with stride {self.stride}")
        print(f"Number of spectral bands: {n_bands}")
        
        # Calculate number of patches
        n_rows = (height - self.patch_size) // self.stride + 1
        n_cols = (width - self.patch_size) // self.stride + 1
        
        print(f"Will extract up to {n_rows * n_cols} patches from this shard\n")
        
        shard_patch_count = 0
        debug_first_few = 3  # Debug first few patches
        
        # Iterate through patch locations
        for row_idx in range(n_rows):
            for col_idx in range(n_cols):
                row_start = row_idx * self.stride
                col_start = col_idx * self.stride
                
                if mask_reprojected is not None:
                    # Use reprojected mask - simple indexing
                    spectral_window = Window(col_start, row_start, self.patch_size, self.patch_size)
                    spectral_patch = spectral_src.read(window=spectral_window)
                    mask_patch = mask_reprojected[
                        row_start:row_start+self.patch_size,
                        col_start:col_start+self.patch_size
                    ]
                else:
                    # Use overlap windows
                    spectral_window = Window(
                        spec_base_window.col_off + col_start,
                        spec_base_window.row_off + row_start,
                        self.patch_size,
                        self.patch_size
                    )
                    
                    mask_window = Window(
                        mask_base_window.col_off + col_start,
                        mask_base_window.row_off + row_start,
                        self.patch_size,
                        self.patch_size
                    )
                    
                    spectral_patch = spectral_src.read(window=spectral_window)
                    mask_patch = mask_src.read(1, window=mask_window)
                
                stats['total_patches'] += 1
                
                # Debug first few patches
                should_debug = stats['total_patches'] <= debug_first_few
                
                # Check validity (skip if too many nodata pixels)
                if self._is_valid_patch(spectral_patch, mask_patch, min_valid_pixels, debug=should_debug):
                    # Save patches as GeoTIFF files
                    spectral_filename = f'patch_{patch_id:05d}_spectral.tif'
                    mask_filename = f'patch_{patch_id:05d}_mask.tif'
                    
                    # Calculate transform for this patch
                    if mask_reprojected is not None:
                        patch_transform = spectral_src.transform * rasterio.Affine.translation(col_start, row_start)
                    else:
                        patch_transform = spectral_src.transform * rasterio.Affine.translation(
                            spec_base_window.col_off + col_start,
                            spec_base_window.row_off + row_start
                        )
                    
                    # Save spectral patch
                    with rasterio.open(
                        self.spectral_dir / spectral_filename,
                        'w',
                        driver='GTiff',
                        height=self.patch_size,
                        width=self.patch_size,
                        count=spectral_patch.shape[0],
                        dtype=spectral_patch.dtype,
                        crs=spectral_src.crs,
                        transform=patch_transform,
                        compress='lzw'
                    ) as dst:
                        dst.write(spectral_patch)
                    
                    # Save mask patch
                    with rasterio.open(
                        self.mask_dir / mask_filename,
                        'w',
                        driver='GTiff',
                        height=self.patch_size,
                        width=self.patch_size,
                        count=1,
                        dtype=mask_patch.dtype,
                        crs=spectral_src.crs,
                        transform=patch_transform,
                        compress='lzw'
                    ) as dst:
                        dst.write(mask_patch, 1)
                    
                    stats['valid_patches'] += 1
                    patch_id += 1
                    shard_patch_count += 1
                    
                    if patch_id % 100 == 0:
                        print(f"Extracted {patch_id} total valid patches...")
                else:
                    stats['skipped_patches'] += 1
        
        print(f"Shard complete: {shard_patch_count} valid patches extracted")
        
        return patch_id
    
    def _is_valid_patch(self, spectral_patch, mask_patch, min_valid_fraction, debug=False):
        """
        Check if a patch has sufficient valid data.
        
        Args:
            spectral_patch: Spectral data patch (bands, height, width)
            mask_patch: Mask data patch (height, width)
            min_valid_fraction: Minimum fraction of valid pixels
            debug: Whether to print debug information
        
        Returns:
            Boolean indicating if patch is valid
        """
        # Check for nodata in spectral bands (assuming 0 or NaN as nodata)
        valid_spectral = ~(np.isnan(spectral_patch) | (spectral_patch == 0)).all(axis=0)
        
        # Check for nodata in mask
        valid_mask = ~np.isnan(mask_patch)
        
        # Combined validity
        valid_pixels = valid_spectral & valid_mask
        valid_fraction = valid_pixels.sum() / valid_pixels.size
        
        if debug:
            logger.debug(f"\n  _is_valid_patch:")
            logger.debug(f"    Spectral patch shape: {spectral_patch.shape}, dtype: {spectral_patch.dtype}")
            logger.debug(f"    Spectral value range: [{np.nanmin(spectral_patch):.6f}, {np.nanmax(spectral_patch):.6f}]")
            logger.debug(f"    Spectral NaN count: {np.isnan(spectral_patch).sum()}")
            logger.debug(f"    Spectral zero count: {(spectral_patch == 0).sum()}")
            logger.debug(f"    Spectral valid pixels: {valid_spectral.sum()} / {valid_spectral.size}")
            logger.debug(f"    Mask patch shape: {mask_patch.shape}, dtype: {mask_patch.dtype}")
            logger.debug(f"    Mask unique values: {np.unique(mask_patch)}")
            logger.debug(f"    Mask NaN count: {np.isnan(mask_patch).sum()}")
            logger.debug(f"    Mask valid pixels: {valid_mask.sum()} / {valid_mask.size}")
            logger.debug(f"    Combined valid fraction: {valid_fraction:.4f} (threshold: {min_valid_fraction:.4f})")
            logger.debug(f"    Result: {'VALID' if valid_fraction >= min_valid_fraction else 'INVALID'}")
        
        return valid_fraction >= min_valid_fraction
    
    def create_train_val_split(self, val_fraction=0.2, random_seed=42):
        """
        Create train/validation split with Prithvi-compatible directory structure.
        Creates:
        - training_chips/ and validation_chips/ subdirectories
        - training_data.txt and validation_data.txt listing chip names
        
        Args:
            val_fraction: Fraction of data for validation (0-1)
            random_seed: Random seed for reproducibility
        """
        import shutil
        
        np.random.seed(random_seed)
        
        # Create training and validation directories
        training_dir = self.output_dir / 'training_chips'
        validation_dir = self.output_dir / 'validation_chips'
        training_dir.mkdir(exist_ok=True)
        validation_dir.mkdir(exist_ok=True)
        
        # Get all patch files
        spectral_files = sorted(self.spectral_dir.glob('patch_*_spectral.tif'))
        n_patches = len(spectral_files)
        
        if n_patches == 0:
            print("Warning: No patches found to split!")
            return
        
        # Create random split
        indices = np.arange(n_patches)
        np.random.shuffle(indices)
        
        n_val = int(n_patches * val_fraction)
        val_indices = set(indices[:n_val])
        
        # Lists for .txt files (just chip names, no extensions)
        train_chip_names = []
        val_chip_names = []
        
        print(f"\nOrganizing patches into training and validation sets...")
        
        for idx, spec_file in enumerate(spectral_files):
            patch_id = spec_file.stem.split('_')[1]  # Extract ID from 'patch_XXXXX_spectral'
            mask_file = self.mask_dir / f'patch_{patch_id}_mask.tif'
            
            # Create chip name (e.g., "chip_00001")
            chip_name = f'chip_{patch_id}'
            
            # Determine target directory
            if idx in val_indices:
                target_dir = validation_dir
                val_chip_names.append(chip_name)
            else:
                target_dir = training_dir
                train_chip_names.append(chip_name)
            
            # Copy/move files to appropriate directory with new naming
            # Copy spectral bands
            target_spectral = target_dir / f'{chip_name}_merged.tif'
            shutil.copy2(spec_file, target_spectral)
            
            # Copy mask
            target_mask = target_dir / f'{chip_name}.mask.tif'
            if mask_file.exists():
                shutil.copy2(mask_file, target_mask)
        
        # Save .txt files with chip names (one per line)
        with open(self.output_dir / 'training_data.txt', 'w') as f:
            for chip_name in train_chip_names:
                f.write(f'{chip_name}\n')
        
        with open(self.output_dir / 'validation_data.txt', 'w') as f:
            for chip_name in val_chip_names:
                f.write(f'{chip_name}\n')
        
        print(f"\nDataset split created:")
        print(f"Training samples: {len(train_chip_names)}")
        print(f"  - Files in: {training_dir}")
        print(f"  - List file: training_data.txt")
        print(f"Validation samples: {len(val_chip_names)}")
        print(f"  - Files in: {validation_dir}")
        print(f"  - List file: validation_data.txt")
        print(f"\nChip naming format:")
        print(f"  - Spectral: chip_XXXXX_merged.tif")
        print(f"  - Mask: chip_XXXXX.mask.tif")
        
        # Clean up temporary directories
        print(f"\nCleaning up temporary directories...")
        if self.spectral_dir.exists():
            shutil.rmtree(self.spectral_dir)
            print(f"  - Removed: {self.spectral_dir}")
        if self.mask_dir.exists():
            shutil.rmtree(self.mask_dir)
            print(f"  - Removed: {self.mask_dir}")
        print(f"✓ Cleanup complete!")


if __name__ == '__main__':
    # Configuration
    # SPECTRAL_FILE can be either:
    #   - A single .tif file: 'data/planet_median_stAndrews.tif'
    #   - A .zip file containing multiple .tif shards: 'data/median_images.zip'
    #   - A directory containing .tif files: 'data/spectral_shards/'
    # SPECTRAL_FILE = 'data/planet_median_stAndrews.tif'
    # SPECTRAL_FILE = 'data/median_images_8band_shards.zip'
    # SPECTRAL_FILE = 'data/median_images'  # v03
    SPECTRAL_FILE = 'data/seasonal_s2_stack.tif'
    MASK_FILE = 'data/SIMM_2024_seagrass_sand_water_land.tif'
    PATCH_SIZE = 224  # Prithvi model input size
    STRIDE = 224  # Non-overlapping patches
    OUTPUT_DIR = 'data/tuning_patches'

    # clear the output directory if it exists
    output_path = Path(OUTPUT_DIR)
    if output_path.exists():
        print(f"Clearing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(output_path)
    
    # Enable debug logging if needed (set to True to see detailed debug information)
    # Debug output includes: raw data checks, mask statistics, spectral data validation,
    # and per-patch validity information for the first few patches
    DEBUG_MODE = False
    if DEBUG_MODE:
        logger.setLevel(logging.DEBUG)
    
    # Initialize extractor
    extractor = PrithviPatchExtractor(
        spectral_path=SPECTRAL_FILE,
        mask_path=MASK_FILE,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        output_dir=OUTPUT_DIR
    )
    
    try:
        # Extract patches
        stats = extractor.extract_patches()
        
        # Create train/validation split
        extractor.create_train_val_split(val_fraction=0.2, random_seed=42)
        
        print("\n✓ Patch extraction complete! Ready for Prithvi fine-tuning.")
    
    finally:
        # Cleanup temporary files if ZIP was used
        extractor.cleanup()
