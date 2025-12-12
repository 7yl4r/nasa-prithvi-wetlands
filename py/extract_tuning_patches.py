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
        
    def extract_patches(self, min_valid_pixels=0.7, save_metadata=True):
        """
        Extract patches from both spectral and mask files.
        
        Args:
            min_valid_pixels: Minimum fraction of valid (non-nodata) pixels required (0-1)
            save_metadata: Whether to save metadata JSON file
        
        Returns:
            Dictionary with extraction statistics
        """
        stats = {
            'total_patches': 0,
            'valid_patches': 0,
            'skipped_patches': 0,
            'patch_size': self.patch_size,
            'stride': self.stride
        }
        
        with rasterio.open(self.spectral_path) as spectral_src, \
             rasterio.open(self.mask_path) as mask_src:
            
            # Verify spatial alignment
            if spectral_src.shape != mask_src.shape:
                raise ValueError(f"Shape mismatch: spectral {spectral_src.shape} vs mask {mask_src.shape}")
            
            height, width = spectral_src.shape
            n_bands = spectral_src.count
            
            print(f"Input dimensions: {height}x{width}, Bands: {n_bands}")
            print(f"Extracting {self.patch_size}x{self.patch_size} patches with stride {self.stride}")
            
            # Calculate number of patches
            n_rows = (height - self.patch_size) // self.stride + 1
            n_cols = (width - self.patch_size) // self.stride + 1
            
            print(f"Will extract up to {n_rows * n_cols} patches")
            
            patch_id = 0
            
            # Iterate through patch locations
            for row_idx in range(n_rows):
                for col_idx in range(n_cols):
                    row_start = row_idx * self.stride
                    col_start = col_idx * self.stride
                    
                    # Create window
                    window = Window(col_start, row_start, self.patch_size, self.patch_size)
                    
                    # Read patches
                    spectral_patch = spectral_src.read(window=window)
                    mask_patch = mask_src.read(1, window=window)
                    
                    stats['total_patches'] += 1
                    
                    # Check validity (skip if too many nodata pixels)
                    if self._is_valid_patch(spectral_patch, mask_patch, min_valid_pixels):
                        # Save patches
                        spectral_filename = f'patch_{patch_id:05d}_spectral.npy'
                        mask_filename = f'patch_{patch_id:05d}_mask.npy'
                        
                        np.save(self.spectral_dir / spectral_filename, spectral_patch)
                        np.save(self.mask_dir / mask_filename, mask_patch)
                        
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
                    'input_shape': [height, width],
                    'crs': str(spectral_src.crs),
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
        Create train/validation split and save file lists.
        
        Args:
            val_fraction: Fraction of data for validation (0-1)
            random_seed: Random seed for reproducibility
        """
        np.random.seed(random_seed)
        
        # Get all patch files
        spectral_files = sorted(self.spectral_dir.glob('patch_*_spectral.npy'))
        n_patches = len(spectral_files)
        
        # Create random split
        indices = np.arange(n_patches)
        np.random.shuffle(indices)
        
        n_val = int(n_patches * val_fraction)
        val_indices = set(indices[:n_val])
        
        # Create file lists
        train_list = []
        val_list = []
        
        for idx, spec_file in enumerate(spectral_files):
            patch_id = spec_file.stem.split('_')[1]
            mask_file = self.mask_dir / f'patch_{patch_id}_mask.npy'
            
            pair = {
                'spectral': str(spec_file.relative_to(self.output_dir)),
                'mask': str(mask_file.relative_to(self.output_dir))
            }
            
            if idx in val_indices:
                val_list.append(pair)
            else:
                train_list.append(pair)
        
        # Save splits
        with open(self.output_dir / 'train_split.json', 'w') as f:
            json.dump(train_list, f, indent=2)
        
        with open(self.output_dir / 'val_split.json', 'w') as f:
            json.dump(val_list, f, indent=2)
        
        print(f"\nDataset split created:")
        print(f"Training samples: {len(train_list)}")
        print(f"Validation samples: {len(val_list)}")


if __name__ == '__main__':
    # Configuration
    SPECTRAL_FILE = 'data/annual_median_selectedBands_2024.tif'
    MASK_FILE = 'data/SAB2024_SVM_clean_smoothed.tif'
    PATCH_SIZE = 224  # Prithvi model input size
    STRIDE = 224  # Non-overlapping patches
    OUTPUT_DIR = 'prithvi_patches'
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
