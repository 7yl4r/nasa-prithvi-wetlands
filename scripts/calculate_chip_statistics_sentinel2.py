import sys
from pathlib import Path

# Add the parent directory to the path so we can import from ../src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.calculate_chip_statistics import calculate_chip_statistics, print_statistics, save_statistics
    

# Calculate statistics
means_dict, stds_dict = calculate_chip_statistics(
    data_root='data/output/tuning_patches',
    band_names=[
        "BLUE",
        "GREEN",
        "RED",
        "RED_EDGE",
        "RED_EDGE_2",
        "NIR",
        "SWIR",
        "SWIR_2"
    ]
)

# Print statistics
print_statistics(means_dict, stds_dict)

# Save to file
# save_statistics(means_dict, stds_dict, output_file=args.output)

print("\n✓ Statistics calculation complete!")
