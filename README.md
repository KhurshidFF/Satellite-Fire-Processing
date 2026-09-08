# Suomi-NPP VIIRS Active Fire Detection Replication

Experimental Python replication of the **Suomi-NPP VIIRS 375 m active fire detection algorithm** for South Korea.

This repository implements a research-oriented version of the VIIRS active fire detection workflow using original Suomi-NPP VIIRS Level-1B and geolocation products. The implementation includes thermal-band preprocessing, quality screening, nighttime fire-pixel detection, contextual background tests, fire confidence classification, Fire Radiative Power (FRP) estimation, persistent-anomaly screening, and FIRMS-style CSV and GeoJSON output generation.

The implementation was developed for research and algorithm evaluation purposes. It is not an official NASA, NOAA, CSPP, or FIRMS implementation and may not reproduce the operational VIIRS active fire product exactly.

---

## Overview

The algorithm processes Suomi-NPP VIIRS observations at approximately 375 m spatial resolution using the I-band thermal channels together with the M13 moderate-resolution thermal channel.

The main processing workflow is:

1. Read VIIRS I04 and I05 brightness temperatures.
2. Read geolocation and observation geometry information.
3. Apply geolocation and quality-control screening.
4. Identify land, water, cloud, and valid background pixels.
5. Detect potential fire pixels using fixed thermal thresholds.
6. Apply contextual fire tests using surrounding background pixels.
7. Assign low, nominal, and high fire-confidence classes.
8. Map detected I-band fire pixels to the M13 grid.
9. Estimate Fire Radiative Power (FRP).
10. Apply South Korea geographic filtering.
11. Identify known persistent thermal anomalies.
12. Export detected fires as CSV and GeoJSON files.

---

## Main Script

### `viirs_snpp_fire_detection.py`

Main implementation of the experimental Suomi-NPP VIIRS active fire detection algorithm.

The script performs:

- VIIRS Level-1B data reading
- brightness-temperature conversion
- geolocation handling
- solar zenith angle screening
- sensor zenith angle handling
- land/water masking
- quality-flag screening
- bow-tie pixel rejection
- cloud screening
- fixed-threshold fire detection
- contextual fire detection
- confidence classification
- I-band to M-band pixel mapping
- M13-based FRP estimation
- persistent thermal-anomaly screening
- South Korea geographic filtering
- CSV output generation
- GeoJSON output generation

---

## Required VIIRS Input Products

The current implementation uses four Suomi-NPP VIIRS products.

| Product | Purpose | Main Variables Used |
|---|---|---|
| `VNP02IMG` | VIIRS imagery radiometric data | I04, I05, quality flags |
| `VNP03IMG` | I-band geolocation | latitude, longitude, SZA, VZA, land/water mask, quality flag |
| `VNP02MOD` | Moderate-resolution radiometric data | M13 radiance |
| `VNP03MOD` | Moderate-resolution geolocation | M13 latitude, longitude, VZA, quality flag |

The I04 and I05 brightness temperatures are derived using the brightness-temperature lookup tables provided in the VIIRS Level-1B files.

The algorithm currently expects matching observation times among the four VIIRS products.

---

## Main Spectral Inputs

### I04

VIIRS I04 is the primary mid-wave infrared thermal channel used for active fire detection.

The algorithm uses I04 brightness temperature for identifying very hot pixels and contextual temperature anomalies.

### I05

VIIRS I05 is a long-wave infrared thermal channel.

The difference between I04 and I05 brightness temperatures is calculated as:

```text
ΔT = BT(I04) - BT(I05)
