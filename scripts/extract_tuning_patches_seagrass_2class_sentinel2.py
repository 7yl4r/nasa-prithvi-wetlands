import sys
from pathlib import Path
import shutil
import logging

# Add the parent directory to the path so we can import from ../src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.PrithviPatchExtractor import extract_patches

# =======================================================================================
# === Configuration
extract_patches(
    spectral_path = 'data/input/spectral/stAndrews_seasonal_s2_stack_2023_to_2025.tif',
    mask_path = 'data/input/classmaps/seagrass_superdove_SIMM_SAB2024_SVM_clean_smoothed.tif',
    patch_size = 224,
    stride = 224,
    output_dir = 'data/output/tuning_patches',
    output_file = 'stAndrews_seagrass_tuning_patches.tar.bz2'
)
# =======================================================================================