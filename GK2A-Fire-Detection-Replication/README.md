# GK2A/AMI Fire Detection Algorithm Replication

Experimental Python replication of the **GEO-KOMPSAT-2A (GK2A) Advanced Meteorological Imager (AMI) fire detection algorithm** for the Korean Peninsula.

This repository implements a research-oriented version of the GK2A fire-detection workflow using calibrated AMI spectral observations together with cloud, land/sea, topographic, land-cover, and satellite-geometry information.

The implementation includes preprocessing, auxiliary input generation, contextual fire detection, quality-flag generation, and geographic fire-point export.

This is an **experimental research replication** and is not the official KMA/NMSC operational GK2A fire-detection software. Results may therefore differ from the official GK2A fire product.

---

## Overview

The general processing workflow is:

1. Prepare calibrated GK2A spectral observations.
2. Generate or correct the GK2A latitude/longitude grid.
3. Prepare cloud, land/sea, and DEM data.
4. Calculate Solar Zenith Angle (SZA) and View Zenith Angle (VZA).
5. Generate land-cover and industrial heat-source masks.
6. Apply topographic temperature correction.
7. Detect absolute and potential fire pixels.
8. Apply contextual fire-detection tests.
9. Apply cloud, land-cover, and industrial-source rejection.
10. Generate the final fire mask and DQF output.
11. Export detected fire pixels to GeoJSON.

---

## Main Scripts

### `gk2a_ff_algorithm.py`

Core implementation of the experimental GK2A fire-detection algorithm.

Main functions include:

- day/night separation
- observation-quality screening
- topographic temperature correction
- SWIR/TIR background estimation
- absolute fire detection
- potential-fire detection
- contextual fire detection
- cloud-related rejection
- industrial heat-source rejection
- optional temporal stability checking
- binary fire-mask generation
- DQF generation

The algorithm expects calibrated brightness temperatures and reflectances rather than raw digital numbers. 

### `gk2a_inputs.py`

Generates auxiliary inputs used by the fire algorithm, including:

- Solar Zenith Angle
- View Zenith Angle
- industrial heat-source mask
- urban/cropland/bare-soil mask
- topographic reference pixels

The `build_all_inputs()` function can generate these inputs together. 
### `run_example.py`

Example workflow showing how to:

- load prepared GK2A observations
- crop the full-disk data to the Korean Peninsula
- build auxiliary inputs
- run the fire-detection algorithm
- create fire-point coordinates
- export GeoJSON and NumPy outputs

The example currently uses case-specific file paths and should be modified for the user's own data. 

---

## Preprocessing Scripts

### `generate_gk2a_geolocation.py`

Generates/corrects the GK2A full-disk latitude and longitude grid using the CGMS geostationary projection convention.

The current implementation uses the GK2A sub-satellite longitude near **128.2°E** and a 5500 × 5500 full-disk grid. 

### `make_cloud_mask.py`

Converts the GK2A Level-2 cloud product into a boolean cloud mask.

The current interpretation is:

```text
0 = cloud
1 = probably cloud
2 = clear
```

Only clear pixels are treated as cloud-free. 

### `make_land_mask.py`

Converts the GK2A land/sea mask into a boolean land mask.

```text
1 = land
0 = sea
-999 = invalid
```

### `make_dem_on_gk2a_grid.py`

Resamples a DEM to the GK2A grid for use in topographic temperature correction.

The current example uses a Korea SRTM DEM. 

---

## Required Input Data

The main spectral inputs are:

| Input | GK2A channel | Purpose |
|---|---|---|
| SWIR brightness temperature | SW038 / Ch07 | Main fire-sensitive thermal channel |
| TIR brightness temperature | IR112 / Ch14 | Background thermal temperature |
| Red reflectance | VI006 / Ch03 | Visible reflectance |
| NIR reflectance | VI008 / Ch04 | Near-infrared reflectance |

Additional inputs include:

```text
GK2A cloud mask
GK2A land/sea mask
DEM
land-cover data
latitude / longitude
scan time
```

The algorithm expects the spectral observations to already be converted to brightness temperature or reflectance.

---

## Study Area

The example script focuses on the Korean Peninsula using approximately:

```text
Longitude: 124°E – 132°E
Latitude : 33°N – 39.5°N
```



---

## Requirements

Python 3.9 or later is recommended.

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

Current requirements:

```text
numpy>=1.24
scipy>=1.10
netCDF4>=1.6
rasterio>=1.3
geopandas>=0.14
pyproj>=3.5
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/GK2A-Fire-Detection-Replication.git
```

Move into the repository:

```bash
cd GK2A-Fire-Detection-Replication
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Example

Before running the example, update the paths in `run_example.py` for:

```text
preprocessed GK2A NetCDF
land mask
cloud mask
DEM
land-cover dataset
output directory
```

Also set the correct UTC scan time:

```python
scan_dt = datetime(YYYY, MM, DD, HH, MM)
```

Then run:

```bash
python run_example.py
```

The example uses `build_all_inputs()` to generate SZA, VZA, industrial, and land-cover rejection masks before passing the data to the main fire-detection algorithm. 
---

## Outputs

The example can generate:

```text
FF_binary_*.npy
DQF_FF_*.npy
T_SWIR_corrected_*.npy
T_TIR_corrected_*.npy
delta_T_*.npy
base_SWIR_*.npy
base_dT_*.npy
LAT_*.npy
LON_*.npy
FF_*.geojson
```

The GeoJSON output contains the geographic locations of detected fire pixels and can be opened directly in GIS software such as QGIS. 

---

## Important Note

This repository provides an **experimental replication** of the GK2A/AMI fire-detection algorithm.

The implementation was developed from available algorithm documentation and research testing. It does not reproduce the official operational processing system exactly.

Differences may result from:

- cloud-screening implementation
- ancillary datasets
- topographic correction
- land-cover information
- industrial-source masking
- contextual-background selection
- operational processing details that are not available publicly

Therefore, results should be used for research and algorithm evaluation rather than operational wildfire warning.

---

## Research Use

This repository is intended for:

- GK2A fire-product evaluation
- satellite wildfire research
- algorithm replication
- fire-detection comparison
- geolocation assessment
- methodological testing

---

## Disclaimer

This repository is an independent research implementation and is not an official KMA/NMSC GK2A processing system.
