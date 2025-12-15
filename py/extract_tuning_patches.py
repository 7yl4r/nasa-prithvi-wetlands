"""
Running this file will cut fine-tuning patches to be used with prithvi.

It assumes that two .tif files have been placed in data:

    SPECTRAL_FILE = 'data/planet_median_stAndrews.tif'
    MASK_FILE = 'data/SAB2024_SVM_clean_smoothed.tif'

-------------------------------------------------------------

## Description of files:
    From the SIMM Seagrass project two images are obtained of the St Andrew's sound region.

    1. A seagrass classification .tif created using the mode of classifications on images taken throughout a year.
    2. A spectral mean .tif created using the same images used to generate classifications.

    The aggregations use a total of 138 images covering 2022-2024, one aggregation is created for each year.

    The 2024 year is selected for creating this dataset.
    The 2024 spectral mean image is built from 72 images.
    This image cannot be shared due to licensing restrictions.

    The seagrass image can be downloaded [here](https://usf.box.com/s/xr4zqg7vj9ynqxfn9zz0oj3r91ki66ui).
    In addition to application of the seagrass classifier (GEE script available [here](From the SIMM Seagrass project two images are obtained of the St Andrew's sound region.

    1. A seagrass classification .tif created using the mode of classifications on images taken throughout a year.
    2. A spectral median .tif created using all Planet SuperDove images (346 images).

    The spectral mean image cannot be shared due to licensing restrictions.
    The GEE script to generate this image is [here](https://code.earthengine.google.com/de9e9e1dc344d4cf8c234b30809665b8).
    This script will not work unless you have access to the (restricted) Planet SuperDove image collection asset.

    The seagrass image for 2024 can be downloaded [here](https://usf.box.com/s/xr4zqg7vj9ynqxfn9zz0oj3r91ki66ui).
    In addition to application of the seagrass classifier (GEE script available [here](https://code.earthengine.google.com/23d1c15a67dfbc71564b67afdf394873)), manal adjustments may have been made to improve the final product.

-------------------------------------------------------------

The class defined below is generic, but the only current usage is in the main section 
at the bottom of this file.
"""


import numpy as np
import rasterio
from rasterio.windows import Window
from pathlib import Path
import json

class PrithviPatchExtractor:
    """Extract paired patches from spectral bands and mask for Prithvi model fine-tuning."""
    
    def __init__(self, spectral_path, mask_path, patch_size=224, stride=224, output_dir='patches'):
        """
        Initialize the patch extractor.
        
        Args:
            spectral_path: Path to spectral bands TIF file
            mask_path: Path to seagrass mask TIF file
            patch_size: Size of patches to extract (default 224x224 for Prithvi)
            stride: Step size between patches (default 224 for non-overlapping)
            output_dir: Directory to save extracted patches
        """
        self.spectral_path = spectral_path
        self.mask_path = mask_path
        self.patch_size = patch_size
        self.stride = stride
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.spectral_dir = self.output_dir / 'spectral'
        self.mask_dir = self.output_dir / 'masks'
        self.spectral_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def extract_patches(self, min_valid_pixels=0.7, save_metadata=True):
        """
        Extract patches from both spectral and mask files.
        
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
            'stride': self.stride
        }
        
        with rasterio.open(self.spectral_path) as spectral_src, \
             rasterio.open(self.mask_path) as mask_src:
            
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
                height, width = spectral_src.shape
                
            else:
                # If CRS match, calculate overlapping region
                spec_base_window, mask_base_window, (height, width) = self._calculate_overlap_window(
                    spectral_src, mask_src
                )
                mask_reprojected = None
            
            n_bands = spectral_src.count
            
            print(f"Extracting {self.patch_size}x{self.patch_size} patches with stride {self.stride}")
            print(f"Number of spectral bands: {n_bands}")
            
            # Calculate number of patches
            n_rows = (height - self.patch_size) // self.stride + 1
            n_cols = (width - self.patch_size) // self.stride + 1
            
            print(f"Will extract up to {n_rows * n_cols} patches\n")
            
            patch_id = 0
            
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
                    
                    # Check validity (skip if too many nodata pixels)
                    if self._is_valid_patch(spectral_patch, mask_patch, min_valid_pixels):
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
                        
                        if patch_id % 100 == 0:
                            print(f"Extracted {patch_id} valid patches...")
                    else:
                        stats['skipped_patches'] += 1
            
            # Save metadata
            if save_metadata:
                metadata = {
                    'spectral_path': str(self.spectral_path),
                    'mask_path': str(self.mask_path),
                    'patch_size': self.patch_size,
                    'stride': self.stride,
                    'n_bands': n_bands,
                    'processing_shape': [height, width],
                    'spectral_full_shape': list(spectral_src.shape),
                    'mask_full_shape': list(mask_src.shape),
                    'spectral_crs': str(spectral_src.crs),
                    'mask_crs': str(mask_src.crs),
                    'reprojected': mask_reprojected is not None,
                    'transform': list(spectral_src.transform)[:6],
                    'stats': stats
                }
                
                with open(self.output_dir / 'metadata.json', 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        print(f"\nExtraction complete!")
        print(f"Valid patches saved: {stats['valid_patches']}")
        print(f"Patches skipped: {stats['skipped_patches']}")
        print(f"Output directory: {self.output_dir}")
        
        return stats
    
    def _is_valid_patch(self, spectral_patch, mask_patch, min_valid_fraction):
        """
        Check if a patch has sufficient valid data.
        
        Args:
            spectral_patch: Spectral data patch (bands, height, width)
            mask_patch: Mask data patch (height, width)
            min_valid_fraction: Minimum fraction of valid pixels
        
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
            target_mask = target_dir / f'{chip_name}_label.tif'
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
        print(f"  - Label: chip_XXXXX_label.tif")
        
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
    SPECTRAL_FILE = 'data/planet_median_stAndrews.tif'
    MASK_FILE = 'data/SAB2024_SVM_clean_smoothed.tif'
    PATCH_SIZE = 224  # Prithvi model input size
    STRIDE = 224  # Non-overlapping patches
    OUTPUT_DIR = 'data/tuning_patches'
    MIN_VALID_PIXELS = 0.7  # Require 70% valid pixels
    
    # Initialize extractor
    extractor = PrithviPatchExtractor(
        spectral_path=SPECTRAL_FILE,
        mask_path=MASK_FILE,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        output_dir=OUTPUT_DIR
    )
    
    # Extract patches
    stats = extractor.extract_patches(min_valid_pixels=MIN_VALID_PIXELS)
    
    # Create train/validation split
    extractor.create_train_val_split(val_fraction=0.2, random_seed=42)
    
    print("\n✓ Patch extraction complete! Ready for Prithvi fine-tuning.")