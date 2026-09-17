# FireProductDownloader

Python tools for downloading satellite-based active-fire and wildfire-related products over the Korean Peninsula from Sentinel-3, VIIRS, and GEO-KOMPSAT-2A (GK2A).

This repository provides research-oriented scripts for accessing and organizing fire-related satellite observations from multiple low-Earth-orbit (LEO) and geostationary (GEO) satellite systems.

The scripts are intended to simplify the acquisition of satellite fire products for wildfire monitoring, product evaluation, algorithm development, and multi-sensor comparison studies.

---

## Overview

The repository currently supports three satellite data sources:

1. **GK2A/AMI** — geostationary satellite observations over East Asia.
2. **VIIRS** — polar-orbiting active-fire observations.
3. **Sentinel-3** — satellite fire and Fire Radiative Power (FRP) products.

The general workflow is:

```text
Select satellite/product
        ↓
Provide API credentials
        ↓
Define date/time or study period
        ↓
Define Korean Peninsula study area
        ↓
Request satellite data
        ↓
Download available files/products
        ↓
Store data locally for further processing
```

The downloaded products can subsequently be used for fire detection analysis, FRP analysis, satellite-product validation, geolocation assessment, and comparison between GEO and LEO fire observations.

---

## Scripts

### `GK2A_api_crawling_GEL.py`

Downloads fire-related products from the GEO-KOMPSAT-2A (GK2A) satellite.

GK2A provides high-temporal-resolution geostationary observations over the Korean Peninsula and surrounding East Asian region.

Depending on the configuration, the script can be used to retrieve products such as:

```text
GK2A/AMI fire products
cloud-related products
thermal infrared observations
shortwave infrared observations
```

Typical research applications include:

- wildfire monitoring
- GK2A fire-product evaluation
- fire-detection algorithm development
- temporal analysis of wildfire development
- comparison with VIIRS and other satellite observations

---

### `VIIRS_api_crawling_GEL.py`

Downloads VIIRS active-fire observations for the Korean Peninsula.

VIIRS is carried by polar-orbiting satellites and provides high-spatial-resolution active-fire observations that are commonly used for wildfire detection and satellite fire-product validation.

The downloaded VIIRS observations can be used for:

- active-fire detection analysis
- comparison with GK2A fire detections
- comparison with Sentinel-3 fire observations
- fire-location validation
- wildfire-event analysis
- multi-sensor fire-product evaluation

---

### `Sen3_api_crawling_GEL.py`

Downloads Sentinel-3 fire-related products.

Sentinel-3 observations can provide information related to active fires and Fire Radiative Power (FRP), making them useful for comparison with other satellite fire products.

Typical applications include:

- Sentinel-3 fire-product analysis
- FRP comparison
- comparison with VIIRS active-fire observations
- comparison with GK2A/AMI fire products
- multi-satellite wildfire studies

---

## Study Area

The scripts were developed primarily for satellite fire research over the Korean Peninsula.

A typical study domain is approximately:

```text
Longitude: 124°E – 132°E
Latitude : 33°N – 39.5°N
```

The exact spatial domain can be modified according to the requirements of the study.

---

## Satellite Systems

| Satellite / Sensor | Orbit Type | Main Use in Repository |
|---|---|---|
| GK2A / AMI | GEO | High-temporal-resolution fire monitoring |
| VIIRS | LEO | High-spatial-resolution active-fire observations |
| Sentinel-3 | LEO | Fire and FRP-related observations |

Combining GEO and LEO satellite observations can provide complementary information because geostationary satellites offer frequent temporal observations while polar-orbiting satellites generally provide finer spatial information.

---

## Requirements

Python 3.9 or later is recommended.

Depending on the selected downloader, commonly required packages may include:

```text
requests
numpy
pandas
```

Install the required packages with:

```bash
pip install requests numpy pandas
```

If a `requirements.txt` file is included in the repository, use:

```bash
pip install -r requirements.txt
```

---

## API Credentials

Some satellite services require personal API credentials, access tokens, or API keys.

**Do not store personal API keys directly in the public GitHub repository.**

A recommended approach is to store credentials as environment variables.

For example:

```bash
export FIRMS_MAP_KEY="YOUR_API_KEY"
```

and access the value in Python using:

```python
import os

api_key = os.environ["FIRMS_MAP_KEY"]
```

The same approach can be used for other satellite-data services that require authentication.

API credentials must be obtained directly from the corresponding data provider.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/FireProductDownloader.git
```

Move into the repository:

```bash
cd FireProductDownloader
```

Install the required Python packages:

```bash
pip install requests numpy pandas
```

---

## Running the Downloaders

Before running a script:

1. Configure the required API credentials.
2. Check the requested date or time period.
3. Check the satellite product name.
4. Check the geographic domain.
5. Set the local output directory.

---

### GK2A

Run:

```bash
python GK2A_api_crawling_GEL.py
```

The script should be configured with the required GK2A product, acquisition period, API access information, and output directory.

---

### VIIRS

Run:

```bash
python VIIRS_api_crawling_GEL.py
```

Before running, provide the required VIIRS/FIRMS API credentials and configure the requested acquisition period and study region.

---

### Sentinel-3

Run:

```bash
python Sen3_api_crawling_GEL.py
```

Configure the required Sentinel-3 access information, fire product, acquisition period, and output directory before execution.

---

## Suggested Data Organization

For research projects involving several satellite sensors, the downloaded data can be organized as:

```text
data/
├── gk2a/
│   ├── fire/
│   ├── cloud/
│   ├── sw038/
│   └── ir112/
│
├── viirs/
│   └── fire/
│
└── sentinel3/
    └── fire/
```

For event-based wildfire studies, another useful structure is:

```text
data/
└── EVENT_NAME/
    ├── gk2a/
    ├── viirs/
    └── sentinel3/
```

This makes it easier to compare observations from multiple satellites for the same wildfire event.

---

## Output

Depending on the satellite service and selected product, downloaded data may include:

```text
NetCDF (.nc)
CSV (.csv)
HDF/HDF5
satellite fire-product files
geolocation information
thermal observations
FRP-related information
```

The downloaded files are intended to be processed separately using the appropriate satellite-product analysis or fire-detection workflow.

---

## Research Applications

The repository can support:

- wildfire monitoring
- active-fire product acquisition
- satellite fire-product comparison
- GEO versus LEO fire analysis
- GK2A fire-product validation
- VIIRS fire validation
- Sentinel-3 FRP analysis
- fire-detection algorithm development
- geolocation assessment
- multi-sensor wildfire studies
- event-based wildfire analysis

---

## Important Notes

This repository provides **data-download utilities only**.

It does not itself perform:

```text
fire detection
FRP retrieval
satellite-product validation
geolocation correction
machine-learning classification
```

Those analyses should be performed using separate processing and evaluation workflows after the required satellite observations have been downloaded.

Product availability, API structure, authentication requirements, filenames, and data-access policies may change according to the corresponding satellite-data provider.

Users should consult the official documentation of each data provider when configuring or updating the download scripts.

---

## Data and API Usage

Satellite products remain subject to the terms, licenses, access policies, and citation requirements of their respective data providers.

Users are responsible for:

- obtaining valid API credentials
- following provider data-access policies
- respecting request and download limits
- properly citing the original satellite products
- checking redistribution restrictions before sharing downloaded data

This repository does not redistribute the original satellite datasets.

---

## Security

Never commit:

```text
API keys
access tokens
passwords
personal credentials
.env files containing secrets
```

to a public repository.

If an API credential has previously been committed publicly, it should be revoked or regenerated through the corresponding data provider.

---

## Disclaimer

This repository contains independent research utilities for downloading satellite fire-related products.

The scripts are not official software of KMA/NMSC, NASA, NOAA, ESA, EUMETSAT, or other satellite-data providers.

Users should verify downloaded products, metadata, timestamps, and geographic coverage before using the data for scientific analysis or operational applications.
