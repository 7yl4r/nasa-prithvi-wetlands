import sys
from pathlib import Path
import shutil
import logging

# Add the parent directory to the path so we can import from ../src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.PrithviPatchExtractor import PrithviPatchExtractor


# Configuration
# SPECTRAL_FILE can be either:
#   - A single .tif file: 'data/input/spectral/seasonal_s2_stack.tif'
#   - A .zip file containing multiple .tif shards: 'ddata/input/spectral/spectral_median_planet_shards.zip'
#   - A directory containing .tif files: 'data/input/spectral/spectral_median_planet_2023_to_2025/'
# SPECTRAL_FILE = 'data/planet_median_stAndrews.tif'
# SPECTRAL_FILE = 'data/input/spectral/spectral_median_planet_shards.zip'
# SPECTRAL_FILE = 'data/input/spectral/spectral_median_planet_2023_to_2025'  # v03
SPECTRAL_FILE = 'data/input/spectral/seasonal_s2_stack.tif'
MASK_FILE = 'data/input/classmaps/SIMM_2024_seagrass_sand_water_land.tif'
PATCH_SIZE = 224  # Prithvi model input size
STRIDE = 224  # Non-overlapping patches
OUTPUT_DIR = 'data/output/tuning_patches'

# clear the output directory if it exists
output_path = Path(OUTPUT_DIR)
if output_path.exists():
    print(f"Clearing existing output directory: {OUTPUT_DIR}")
    shutil.rmtree(output_path)

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
