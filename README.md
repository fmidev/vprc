# vprc

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.txt)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Status: Beta](https://img.shields.io/badge/status-beta-yellow.svg)]()
[![Docker Repository on Quay](https://quay.io/repository/fmi/vprc/status "Docker Repository on Quay")](https://quay.io/repository/fmi/vprc)

Python implementation of the **Koistinen & Pohjola VPR (Vertical Profile Reflectivity) correction algorithm** for weather radar data. This modernization of a circa-2003 Perl codebase is designed for use in Apache Airflow workflows at FMI.

📖 [Introduction](docs/introduction.md) · [Quick Start](docs/quickstart.md) · [Configuration](docs/configuration.md)

## Features

- **Ground clutter removal** – Gradient-based filtering of low-altitude echoes
- **Spike smoothing** – Boundary correction and isolated echo removal
- **Profile classification** – Automatic layer segmentation (Precipitation, Altostratus, Clear Air Echo, Clutter)
- **Bright band detection** – Melting layer identification using gradient analysis
- **VPR correction** – Range-dependent correction factors for CAPPI products and individual elevations
- **Climatology blending** – Quality weight based climatology fallback for VPR correction
- **Compositing** - Gridded correction fields for radar composite products
- **TOML configuration** – Flexible radar metadata management with environment variable support

## Installation

Requires Python 3.12+.

```bash
# From source (development)
git clone https://github.com/fmidev/vprc.git
cd vprc
pip install -e .

# Or directly from GitHub
pip install git+https://github.com/fmidev/vprc.git
```

## Project Structure

```
src/vprc/          # Package implementation
tests/             # Test suite (see tests/README.md)
docs/              # Documentation
```

## Testing

```bash
pytest tests/
```

See [tests/README.md](tests/README.md) for details on test structure and coverage.

## Contributing

Contributions are welcome through Github.


## References

Koistinen, J., and H. Pohjola, 2014: Estimation of Ground-Level Reflectivity Factor in Operational Weather Radar Networks Using VPR-Based Correction Ensembles. *J. Appl. Meteor. Climatol.*, **53**, 2394–2411, https://doi.org/10.1175/JAMC-D-13-0343.1.
