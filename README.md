# Satellite Fire Processing

A collection of Python tools for satellite-based wildfire research over the Korean Peninsula.

This repository brings together workflows for downloading satellite fire products, replicating active-fire detection algorithms, and retrieving Fire Radiative Power (FRP) using data from geostationary and polar-orbiting satellite sensors.

The repository currently includes tools and experimental implementations for **GK2A/AMI**, **Suomi-NPP VIIRS**, and **Sentinel-3** satellite fire products.

## Repository Structure

### 1. FireProductDownloader

Tools for downloading satellite-based fire products over the Korean Peninsula.

Supported data sources include:

- GK2A/AMI
- Suomi-NPP VIIRS
- Sentinel-3

Repository directory:

[`FireProductDownloader/`](./FireProductDownloader)

---

### 2. GK2A Fire Detection Replication

Experimental Python implementation for replicating the **GK2A/AMI forest fire detection algorithm**.

The workflow includes:

- GK2A input-data preprocessing
- Auxiliary input generation
- Spectral and contextual fire-detection tests
- Fire-pixel identification
- Comparison with official GK2A fire products

Repository directory:

[`GK2A-Fire-Detection-Replication/`](./GK2A-Fire-Detection-Replication)

---

### 3. VIIRS Fire Detection Replication

Experimental implementation of the **Suomi-NPP VIIRS 375 m active-fire detection algorithm**.

The workflow includes:

- VIIRS Level-1B input handling
- Contextual fire-detection tests
- Active-fire pixel identification
- Fire Radiative Power (FRP) estimation
- Generation of FIRMS-style fire-detection outputs

Repository directory:

[`VIIRS-Fire-Detection-Replication/`](./VIIRS-Fire-Detection-Replication)

---

### 4. GK2A/AMI FRP Retrieval

Python implementation for estimating **Fire Radiative Power (FRP)** from GK2A/AMI observations.

The workflow uses:

- AMI 3.8 µm (`SW038`) observations
- Official GK2A fire detections
- Adaptive background estimation
- Wooster MIR-radiance FRP methodology

Repository directory:

[`GK2A-AMI-FRP-Retrieval/`](./GK2A-AMI-FRP-Retrieval)

---

## Main Applications

The tools in this repository are intended for research involving:

- Satellite-based wildfire detection
- Near-real-time fire monitoring
- Active-fire algorithm evaluation
- GEO–LEO satellite comparison
- Fire Radiative Power retrieval
- Validation of satellite fire products
- Wildfire monitoring over the Korean Peninsula

## Satellite Sensors

| Satellite / Sensor | Orbit | Application |
|---|---|---|
| GK2A / AMI | Geostationary (GEO) | Fire detection and FRP retrieval |
| Suomi-NPP / VIIRS | Low Earth Orbit (LEO) | 375 m active-fire detection and FRP |
| Sentinel-3 | Low Earth Orbit (LEO) | Satellite fire-product acquisition |

## Requirements

The individual projects have different dependencies. Please refer to the README and configuration files inside each project directory for detailed installation and usage instructions.

Most workflows are implemented in Python and commonly use scientific and geospatial packages such as:

- NumPy
- Pandas
- Xarray
- Rasterio
- GeoPandas
- SciPy
- NetCDF4

## Data Access

Some scripts require external satellite-data services and personal API credentials.

API keys, authentication information, passwords, and other private credentials should **not** be committed to this repository.

Users should obtain access credentials directly from the corresponding satellite-data providers.

## Research Status

These implementations are intended primarily for **research, algorithm replication, validation, and experimental analysis**.

The replicated detection and FRP algorithms may not reproduce official operational satellite products exactly because operational processing systems can contain additional calibration, preprocessing, auxiliary datasets, quality-control procedures, and implementation details that are not completely available in public documentation.

Therefore, outputs from these tools should not be interpreted as official operational fire products.

## Study Region

The primary study region is the **Korean Peninsula**, although several components of the code may be adaptable to other geographic regions with appropriate modifications.

## License

Please refer to the license information provided in this repository and the individual project directories.

## Author

**Khurshedjon Farkhodov**

Research interests include satellite remote sensing, wildfire detection, active-fire product evaluation, geostationary and polar-orbiting satellite observations, and Fire Radiative Power retrieval.
