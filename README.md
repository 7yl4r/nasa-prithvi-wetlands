# nasa-prithvi-wetlands

An attempt to map coastal wetlands with NASA's Prithvi-EO Geo Foundation Model.

Research Notebooks here can be run in google colab with no setup.
It is recommended to use a GPU-enabled runtime.

For advanced usage (such as tuning dataset creation), use the setup below.

## Setup

This project uses modern Python dependency management with `pyproject.toml` and includes the `dwc_tools` package as a git submodule.

### Installation

```bash
# Clone the repository
git clone https://github.com/7yl4r/nasa-prithvi-wetlands.git
cd nasa-prithvi-wetlands

# Initialize and update git submodules
git submodule update --init --recursive

# Install the project and all dependencies
pip install -e .
```

Dependencies are managed in `pyproject.toml`.


### Workflows

#### Fine-Tuning Dataset Creation 
```bash
# 1) Manually place images 'planet_median_stAndrews.tif' and 'SAB2024_SVM_clean_smoothed.tif'
#    in data/ as described at top of extract_tuning_patches.py.

# 2) Create fine tuning patch chips from the spectral and SVM files.
python py/extract_tuning_patches.py

# 3) compress & zip the patches 
tar -cvjf data/seagrass_tuning_patches.tar.bz2 data/tuning_patches

# 4) Manually upload the file to google drive

# 5) Update any links in .ipynb files to use new file ID.
```

