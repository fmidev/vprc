# vprc

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/fmidev/vprc/blob/main/LICENSE.txt)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![Status: Beta](https://img.shields.io/badge/status-beta-yellow.svg)]()
[![Docker Repository on Quay](https://quay.io/repository/fmi/vprc/status "Docker Repository on Quay")](https://quay.io/repository/fmi/vprc)

Python implementation of the **Koistinen & Pohjola VPR (Vertical Profile Reflectivity) correction algorithm** for weather radar data. This modernization of a circa-2003 Perl codebase is designed for use in Apache Airflow workflows at FMI.

📖 [Introduction](https://github.com/fmidev/vprc/blob/main/docs/introduction.md) · [Quick Start](https://github.com/fmidev/vprc/blob/main/docs/quickstart.md) · [Configuration](https://github.com/fmidev/vprc/blob/main/docs/configuration.md)

## Features

- **Ground clutter removal** – Gradient-based filtering of low-altitude echoes
- **Spike smoothing** – Boundary correction and isolated echo removal
- **Profile classification** – Automatic layer segmentation (Precipitation, Altostratus, Clear Air Echo, Clutter)
- **Bright band detection** – Melting layer identification using gradient analysis
- **VPR correction** – Range-dependent correction factors for CAPPI products and individual elevations
- **Climatology blending** – Quality weight based climatology fallback for VPR correction
- **Compositing** - Gridded correction fields for radar composite products
- **TOML configuration** – Flexible radar metadata management with programmatic API access

## Installation

Requires Python 3.14+.

```bash
pip install vprc
```

For development, clone the repository and create the environment, including
development tools, with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/fmidev/vprc.git
cd vprc
uv sync
```

A container image for operational use is published at
[quay.io/fmi/vprc](https://quay.io/repository/fmi/vprc).

## Project Structure

```
src/vprc/          # Package implementation
tests/             # Test suite (see tests/README.md)
docs/              # Documentation
```

## Testing

```bash
uv run pytest tests/
```

See [tests/README.md](https://github.com/fmidev/vprc/blob/main/tests/README.md) for details on test structure and coverage.

## Contributing

Contributions are welcome through Github.

Coding agents: see [AGENTS.md](https://github.com/fmidev/vprc/blob/main/AGENTS.md) for development guidance.


## References

Koistinen, J., and H. Pohjola, 2014: Estimation of Ground-Level Reflectivity Factor in Operational Weather Radar Networks Using VPR-Based Correction Ensembles. *J. Appl. Meteor. Climatol.*, **53**, 2394–2411, https://doi.org/10.1175/JAMC-D-13-0343.1.
