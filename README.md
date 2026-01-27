# Glacier Ice Thickness Change Analysis
## Sikkim, Bhutan, and Arunachal Pradesh (Eastern Himalaya)

This project analyzes glacier surface elevation changes in the Eastern Himalayan region using TanDEM-X DEM Change Maps (DCM) and RGI 7.0 glacier outlines.

## Key Results

| Region | Glaciers | Area (km²) | Mean Δh (m) | Rate (m/yr) |
|--------|----------|------------|-------------|-------------|
| Sikkim | 812 | 1,025 | -0.37 | -0.046 |
| Bhutan | 1,887 | 1,630 | -1.45 | -0.182 |
| Arunachal Pradesh | 3,774 | 2,746 | -0.96 | -0.120 |
| **Total** | **6,473** | **5,401** | **-1.03** | **-0.129** |

**Volume Loss**: ~14.87 km³ over 8 years (~1.86 km³/year)

## Data Sources

1. **TanDEM-X DEM Change Maps (DCM)** - German Aerospace Center (DLR)
   - 30m resolution pre-calculated elevation difference maps
   - 37 tiles covering the study region

2. **Randolph Glacier Inventory (RGI) v7.0**
   - Glacier outlines for South Asia East region
   - Filtered to Sikkim, Bhutan, and Arunachal Pradesh

## Repository Structure

```
GNR_618/
├── analysis.ipynb          # Main analysis notebook
├── methodology.txt         # Detailed methodology description
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── DCM/                    # DCM ZIP files (not tracked - too large)
├── RGI_Data/               # Glacier shapefiles
│   └── sikkim_bhutan_arunachal_clean.*
└── results/                # Output files
    ├── regional_summary.csv
    ├── per_glacier_results.csv
    ├── glaciers_with_change.shp
    ├── glacier_thinning_map.png
    ├── elevation_change_histogram.png
    ├── regional_comparison.png
    └── elevation_dependence.png
```

## Setup & Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install rasterio geopandas numpy pandas matplotlib shapely jupyter
```

## Data Preparation

The DCM TIF files are not included in this repository due to size constraints.
To reproduce the analysis:

1. Download TanDEM-X DCM tiles from DLR for the region (N26-N29, E087-E097)
2. Place the ZIP files in the `DCM/` folder
3. Run the notebook - it will extract the TIF files automatically

## Running the Analysis

```bash
# Activate virtual environment
source venv/bin/activate

# Run Jupyter notebook
jupyter notebook analysis.ipynb
```

Or execute directly:
```bash
jupyter nbconvert --to notebook --execute analysis.ipynb
```

## Output Files

| File | Description |
|------|-------------|
| `regional_summary.csv` | Summary statistics by region |
| `per_glacier_results.csv` | Individual glacier statistics (6,473 glaciers) |
| `glaciers_with_change.shp` | Shapefile with elevation change attributes |
| `glacier_thinning_map.png` | Map of glaciers colored by thinning rate |
| `elevation_change_histogram.png` | Distribution of pixel-level changes |
| `regional_comparison.png` | Box/bar plots comparing regions |
| `elevation_dependence.png` | Scatter plot of Δh vs altitude |

## Methodology

See `methodology.txt` for a detailed, non-technical description of the analysis process.

### Brief Overview:
1. Extract TIF files from DCM ZIP archives
2. Load glacier boundaries from RGI shapefile
3. Clip elevation change rasters to glacier extents
4. Filter invalid/noisy pixels (NoData, values >+100m or <-150m)
5. Calculate per-tile and per-glacier statistics
6. Aggregate regional results and create visualizations

## Limitations

- Time period is approximate (~8 years based on TanDEM-X naming conventions)
- Radar penetration into snow/firn not corrected
- Volume-to-mass conversion assumes ice density of 900 kg/m³

## License

This project is for academic/research purposes.

## Acknowledgments

- German Aerospace Center (DLR) for TanDEM-X data
- RGI Consortium for glacier inventory data
