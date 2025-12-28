"""
Calculate mean and standard deviation statistics for all bands in tuning patch chips.

This script processes all *_merged.tif files in the training_chips and validation_chips
directories and computes per-band statistics across all pixels in all images.

Output format matches the requirements for Prithvi model normalization.
"""

import numpy as np
import rasterio
from pathlib import Path
from tqdm import tqdm


def calculate_chip_statistics(
        data_root='data/output/tuning_patches', 
        use_validation=True,
        band_names = [     # Band names for Planet SuperDove imagery
            "COASTAL BLUE",
            "BLUE",
            "GREEN I",
            "GREEN II",
            "YELLOW",
            "RED",
            "RED-EDGE",
            "NEAR-INFRARED",
        ],
    ):
    """
    Calculate mean and std for all bands across all chip images.
    
    Args:
        data_root: Root directory containing training_chips and validation_chips
        use_validation: Whether to include validation chips in statistics
    
    Returns:
        Tuple of (means_dict, stds_dict) with band names as keys
    """
    data_root = Path(data_root)
        
    # Collect all chip files
    chip_files = []
    
    training_dir = data_root / 'training_chips'
    if training_dir.exists():
        chip_files.extend(sorted(training_dir.glob('chip_*_merged.tif')))
        print(f"Found {len(chip_files)} training chips")
    else:
        print(f"Warning: Training directory not found: {training_dir}")
    
    if use_validation:
        validation_dir = data_root / 'validation_chips'
        if validation_dir.exists():
            val_chips = list(sorted(validation_dir.glob('chip_*_merged.tif')))
            chip_files.extend(val_chips)
            print(f"Found {len(val_chips)} validation chips")
        else:
            print(f"Warning: Validation directory not found: {validation_dir}")
    
    if len(chip_files) == 0:
        raise ValueError(f"No chip files found in {data_root}")
    
    print(f"\nProcessing {len(chip_files)} total chips...")
    
    # Read first file to get number of bands
    with rasterio.open(chip_files[0]) as src:
        n_bands = src.count
        print(f"Number of bands: {n_bands}")
        
        if n_bands != len(band_names):
            print(f"Warning: Expected {len(band_names)} bands but found {n_bands}")
            # Adjust band names if needed
            if n_bands < len(band_names):
                band_names = band_names[:n_bands]
            else:
                band_names.extend([f"BAND_{i+1}" for i in range(len(band_names), n_bands)])
    
    # Initialize accumulators for online mean/std calculation
    # Using Welford's online algorithm for numerical stability
    n_pixels = 0
    means = np.zeros(n_bands, dtype=np.float64)
    M2 = np.zeros(n_bands, dtype=np.float64)  # For variance calculation
    
    print("\nCalculating statistics...")
    for chip_file in tqdm(chip_files, desc="Processing chips"):
        with rasterio.open(chip_file) as src:
            # Read all bands
            data = src.read()  # Shape: (bands, height, width)
            
            # Flatten spatial dimensions for each band
            for band_idx in range(n_bands):
                band_data = data[band_idx].flatten()
                
                # Filter out nodata values (assuming 0 or NaN)
                valid_mask = ~(np.isnan(band_data) | (band_data == 0))
                valid_data = band_data[valid_mask]
                
                if len(valid_data) == 0:
                    continue
                
                # Update statistics using Welford's online algorithm
                for value in valid_data:
                    n_pixels += 1
                    delta = value - means[band_idx]
                    means[band_idx] += delta / n_pixels
                    delta2 = value - means[band_idx]
                    M2[band_idx] += delta * delta2
    
    # Calculate final standard deviations
    if n_pixels > 1:
        variances = M2 / (n_pixels - 1)  # Sample variance
        stds = np.sqrt(variances)
    else:
        stds = np.zeros(n_bands)
    
    # Create dictionaries with band names
    means_dict = {band_names[i]: float(means[i]) for i in range(n_bands)}
    stds_dict = {band_names[i]: float(stds[i]) for i in range(n_bands)}
    
    return means_dict, stds_dict


def print_statistics(means_dict, stds_dict):
    """Print statistics in the required format."""
    print("\n" + "="*60)
    print("CHIP STATISTICS")
    print("="*60)
    
    print("\nMEANS = {")
    for i, (band, value) in enumerate(means_dict.items()):
        comma = "," if i < len(means_dict) - 1 else ""
        print(f'    "{band}": {value:.6f}{comma}')
    print("}")
    
    print("\nSTDS = {")
    for i, (band, value) in enumerate(stds_dict.items()):
        comma = "," if i < len(stds_dict) - 1 else ""
        print(f'    "{band}": {value:.6f}{comma}')
    print("}")
    print()


def save_statistics(means_dict, stds_dict, output_file='data/tuning_patches/chip_statistics.py'):
    """Save statistics to a Python file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('"""Statistics calculated from tuning patch chips."""\n\n')
        
        f.write('MEANS = {\n')
        for i, (band, value) in enumerate(means_dict.items()):
            comma = "," if i < len(means_dict) - 1 else ""
            f.write(f'    "{band}": {value:.6f}{comma}\n')
        f.write('}\n\n')
        
        f.write('STDS = {\n')
        for i, (band, value) in enumerate(stds_dict.items()):
            comma = "," if i < len(stds_dict) - 1 else ""
            f.write(f'    "{band}": {value:.6f}{comma}\n')
        f.write('}\n')
    
    print(f"Statistics saved to: {output_path}")
