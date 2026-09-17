# GK2A/AMI Fire Radiative Power Retrieval

Research Python implementation for calculating **Fire Radiative Power (FRP)** from GEO-KOMPSAT-2A (GK2A) Advanced Meteorological Imager (AMI) observations.

This repository implements an event-based FRP retrieval workflow using calibrated GK2A SW038 radiance, official GK2A fire detections, cloud information, local background estimation, and satellite pixel area.

The FRP calculation is based on a mid-infrared radiance approach and produces pixel-level FRP values in megawatts (MW).

This is a **research implementation** and is not an official KMA/NMSC operational FRP product.

---

## Overview

The general FRP processing workflow is:

1. Identify GK2A fire pixels.
2. Read and calibrate SW038 observations.
3. Use the cloud and fire products to identify suitable non-fire background pixels.
4. Estimate the local SW038 background radiance.
5. Calculate the ground area of each GK2A pixel.
6. Calculate the fire-pixel radiance excess above the local background.
7. Calculate pixel-level FRP.
8. Export the results to CSV.

The current implementation follows a Wooster-style mid-infrared radiance approach. 

---

## Main Script

### `gk2a_frp.py`

Main implementation of the GK2A/AMI FRP retrieval.

The script performs:

- GK2A SW038 radiance calibration
- SW038 brightness-temperature calculation
- IR112 brightness-temperature calculation
- GK2A fire-pixel selection
- cloud screening
- local background-radiance estimation
- adaptive background-window selection
- GK2A pixel-area calculation
- FRP calculation
- event-based processing
- pixel-level CSV output

The core FRP calculation is:

```text
FRP = A_pixel × σ / (a_AMI × τ) × (R_fire - R_background)
```

where:

```text
A_pixel      = satellite pixel ground area
R_fire       = SW038 radiance of the fire pixel
R_background = local background SW038 radiance
a_AMI        = AMI-specific MIR coefficient
τ            = atmospheric transmittance
σ            = Stefan-Boltzmann constant
```

FRP is reported in **MW**. 

---

## Required Input Data

The current workflow uses the following GK2A products:

| Product | Purpose |
|---|---|
| GK2A FF | Identification of fire pixels |
| GK2A DQF_FF | Fire-pixel quality/confidence information |
| GK2A CLD | Selection of clear background pixels |
| GK2A SW038 | Main mid-infrared radiance used for FRP |
| GK2A IR112 | Thermal diagnostic information |

The main FRP calculation is based on **SW038 radiance**.

IR112 is used for additional brightness-temperature information and diagnostics rather than directly in the FRP equation. 

---

## Local Background Estimation

FRP requires an estimate of the non-fire radiance that would be observed at the fire location without the active fire.

The current implementation searches surrounding clear, normal-land, non-fire pixels using progressively larger windows:

```text
5 × 5
7 × 7
9 × 9
11 × 11
15 × 15
```

The smallest window containing enough valid background pixels is selected.

The default minimum number of valid background pixels is:

```text
8
```

The median SW038 radiance of the valid surrounding pixels is used as the local background radiance. 
---

## Atmospheric Transmittance

The current default configuration uses:

```text
τ = 1.0
```

This means that no explicit atmospheric-transmittance correction is applied.

The value can be changed using the command-line option:

```bash
--tau
```

This parameter should be considered when performing sensitivity analysis or comparing FRP values with other satellite products. 

---

## Event-Based Processing

The current implementation processes individual wildfire events.

An event is specified using:

```bash
--event
```

For example:

```bash
python gk2a_frp.py \
  --event 2023-04-03_Hampyeong
```

The script automatically processes the available GK2A observations for the selected event.

Optional start and end times can also be specified:

```bash
python gk2a_frp.py \
  --event 2023-04-03_Hampyeong \
  --start 202304030240 \
  --end 202304030930
```

Times are specified in UTC using:

```text
YYYYMMDDHHMM
```

---

## Main Options

Useful command-line options include:

```text
--event
--start
--end
--step-min
--crop-buffer-km
--match-buffer-km
--event-buffer-km
--windows
--min-valid
--tau
--output-dir
```

For example:

```bash
python gk2a_frp.py \
  --event 2023-04-03_Hampyeong \
  --step-min 10 \
  --windows 5,7,9,11,15 \
  --min-valid 8 \
  --tau 1.0
```

The default background windows and other processing parameters can therefore be modified for sensitivity testing. 

---

## Supplementary Event Preparation

The wildfire cases used in the broader research workflow were prepared separately using:

```text
GK2A data downloading
event metadata
KFS burned-area information
spatial fire matching
solar-geometry calculations
frame-quality checks
```

These steps were used to identify, organize, and evaluate the exact wildfire cases included in the research.

They are **supporting event-preparation and validation procedures** and are not part of the core FRP equation itself.

The FRP retrieval presented in this repository focuses on:

```text
fire radiance
background radiance
pixel area
AMI MIR coefficient
atmospheric transmittance
```

---

## Requirements

Python 3.9 or later is recommended.

Install the required packages with:

```bash
pip install -r requirements.txt
```

The main scientific dependencies are:

```text
numpy>=1.24
netCDF4>=1.6
pyproj>=3.5
geopandas>=0.14
shapely>=2.0
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/GK2A-AMI-FRP-Retrieval.git
```

Move into the repository:

```bash
cd GK2A-AMI-FRP-Retrieval
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Script

Basic execution:

```bash
python gk2a_frp.py \
  --event 2023-04-03_Hampyeong
```

The script can infer the processing period from the available SW038 files if `--start` and `--end` are not supplied.

To specify the output directory:

```bash
python gk2a_frp.py \
  --event 2023-04-03_Hampyeong \
  --output-dir ./results/Hampyeong
```

---

## Output

The main output is a pixel-level CSV file:

```text
EVENT_gk2a_frp_pixels.csv
```

For example:

```text
2023-04-03_Hampyeong_gk2a_frp_pixels.csv
```

The CSV contains fields including:

```text
latitude
longitude
bt38
bt112
acq_date
acq_time
satellite
instrument
confidence
dqf_ff
frp
daynight
kfs_associated
```

FRP values are reported in **MW**, while brightness temperatures are reported in **K**. 

Event-level FRP time series can be generated later by grouping the pixel detections by acquisition time and summing the individual pixel FRP values. 

---

## Important Note

This repository contains a **research FRP retrieval**, not an official NMSC FRP product.

The implementation uses the official GK2A fire product to identify fire pixels but independently calculates FRP from calibrated SW038 observations.

Results may be influenced by:

- local background selection
- number of available clear background pixels
- atmospheric transmittance
- pixel-area calculation
- daytime reflected solar radiation
- cloud conditions
- calibration assumptions

Therefore, the output should be used for scientific analysis and evaluation rather than as an operational FRP product.

---

## Research Use

This repository is intended for:

- GK2A fire-energy analysis
- FRP retrieval research
- wildfire-event analysis
- satellite fire-product comparison
- FRP sensitivity testing
- comparison with VIIRS or other satellite FRP products

---

## Disclaimer

This repository is an independent research implementation and is not an official KMA/NMSC GK2A processing system.
