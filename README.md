# nasa-prithvi-wetlands

An attempt to map coastal wetlands with NASA's Prithvi-EO Geo Foundation Model 

## Setup

This project uses modern Python dependency management with `pyproject.toml` and includes the `dwc_tools` package as a git submodule.

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

#### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/7yl4r/nasa-prithvi-wetlands.git
cd nasa-prithvi-wetlands

# Initialize and update git submodules
git submodule update --init --recursive

# Install the project and all dependencies (including dwc_tools)
pip install -e .

# Install dwc_tools from the submodule
pip install -e ./dwc_tools
```


#### Option 2: Development Installation

For development with additional tools (pytest, black, flake8):

```bash
# Clone the repository
git clone https://github.com/7yl4r/nasa-prithvi-wetlands.git
cd nasa-prithvi-wetlands

# Initialize and update git submodules
git submodule update --init --recursive

# Install with development dependencies
pip install -e ".[dev]"
pip install -e ./dwc_tools
```


### Running Scripts

Once installed, you can run the data download scripts:

```bash
python py/download_seagrass_and_mangrove.py
```

## Dependencies

Main dependencies are managed in `pyproject.toml`:
- **pandas** - Data manipulation and analysis
- **dwc_tools** - Darwin Core taxonomic occurrence data tools (git submodule)
