# Agent guide for vprc

Python 3.14+ implementation of the Koistinen & Pohjola vertical profile
reflectivity (VPR) correction algorithm for weather radar data. FMI runs this
package in containers from external Airflow workflows; DAGs do not belong here.

## Where to work

- [src/vprc/__init__.py](src/vprc/__init__.py): public API and `process_vvp()`,
  returning a `ProcessedProfile`. It orchestrates parsing, clutter removal,
  smoothing, classification, bright band detection and VPR correction.
- [src/vprc/](src/vprc/): separate algorithm modules, including climatology,
  temporal averaging and multi-radar compositing. Temporal averaging and
  compositing are separate API calls, not part of `process_vvp()`.
- [src/vprc/io/](src/vprc/io/): VVP text parsing and GeoTIFF/COG export.
- [src/scripts/](src/scripts/): composite creation and diagnostic scripts.
- [tests/](tests/): pytest modules and checked-in input/reference fixtures.
- [pyproject.toml](pyproject.toml): dependencies, Hatch build and type-check
  configuration. Versioning uses hatch-vcs and Git metadata.

See [quickstart](docs/quickstart.md), [algorithm overview](docs/introduction.md),
[configuration](docs/configuration.md), [test guide](tests/README.md) and
[release guide](docs/releasing.md) for details rather than duplicating them here.

## Setup and checks

The environment is managed with [uv](https://docs.astral.sh/uv/) from
[pyproject.toml](pyproject.toml) and the committed `uv.lock`; the Python
version comes from [.python-version](.python-version) and matches the
production container. Run commands from the repository root with `uv run`,
which creates or syncs `.venv` as needed; do not rely on shell activation.
Without uv: `python -m pip install -e . --group dev` (pip 25.1+).

- Run the relevant tests first, e.g. `uv run pytest tests/test_clutter.py`.
- For algorithm or pipeline changes, also run
  `uv run pytest tests/test_legacy_comparison.py tests/test_temporal_e2e.py`
  as relevant. Full suite: `uv run pytest tests/`.
- Lint and format touched Python files with Ruff (configured in
  [pyproject.toml](pyproject.toml)): `uv run ruff check --fix <files>` and
  `uv run ruff format <files>`. [CI](.github/workflows/ci.yml) runs Ruff,
  the full test suite and a wheel build on the whole tree, and repeats the
  tests with the oldest runtime dependency versions allowed by
  `pyproject.toml`.
- Type checking: `uv run mypy src/vprc tests`. Existing errors remain; do not
  add new ones in touched code.
- Wheel build, for packaging changes: `uv build --wheel`.
  [Containerfile](Containerfile) defines the reproducible production image:
  base images are pinned by digest and dependencies come from `uv.lock`.
- Add or update dependencies with `uv add` (`uv add --dev` for development
  tools) so `uv.lock` stays in sync; commit both files. Give runtime
  dependencies a lower bound that passes the oldest-dependency CI job.
- Releases are Git tags; the version comes from hatch-vcs, never from the
  source. Follow the [release guide](docs/releasing.md); do not tag or publish
  unless asked.
- Add regression tests in the corresponding test module when behavior changes.
  Reuse fixtures in [tests/data/](tests/data/) and the existing legacy parsers.
- Report checks run and any failures or unavailable dependencies. Documentation-only
  changes do not require Python tests or builds.

## Algorithm guardrails

- The original Perl scripts (`allprof_prodx2.pl`, `pystycappi.pl`,
  `pystycappi_ka.pl`) are the authoritative algorithm reference. They may be
  available in the gitignored [legacy/](legacy/) directory, alongside sample
  inputs/outputs. **Do not modify or delete anything in that directory.**
- A fresh clone does not include the Perl source or [local/](local/) development
  data/tools. Use checked-in tests and reference outputs; if a change needs the
  missing Perl source to resolve algorithm intent, request it rather than guess.
- Preserve numerical behavior, not Perl structure. Prefer existing NumPy, xarray,
  SciPy and wradlib operations over hand-written numerical or radar algorithms.
  Reuse project helpers and constants before adding new ones.
- Validate against legacy outputs with the established test tolerances. Do not
  rewrite reference fixtures or relax tolerances merely to make a change pass.
- Pay particular attention to missing echoes, unusable profiles and bright bands
  at or near the surface: incorrect handling can produce large false corrections.

## Data and configuration conventions

- Vertical profiles use the xarray `height` dimension. Parsed heights are integer
  meters **above antenna**, converted from input heights above sea level (ASL),
  with the standard 200 m spacing (`STEP` in [constants.py](src/vprc/constants.py)).
  `read_vvp()` starts the grid at the radar's `lowest_level_offset_m` (Perl
  `$alintaso`), even when the file lacks that level, and fills levels absent
  from the file as no-echo levels so stages can index `height ± STEP`; their
  heights are in `attrs["missing_levels_m"]`. `smooth_spikes()` drops levels
  at or above `PROFILE_TOP_M` (10 km above antenna) after using them as
  neighbours, as the legacy loops stop at `$i < 10000`.
  The correction calculation also uses a finer grid; do not assume every grid
  has 200 m spacing.
- `lin_dbz` is reflectivity in **dBZ**, not linear reflectivity Z, despite its name.
  Use it rather than `log_dbz`; processing updates `corrected_dbz`.
- Profile missing/no-echo values use `MDS = -45` dBZ. Preserve each algorithm's
  threshold comparisons; do not substitute zero or NaN indiscriminately.
- `read_vvp()` accepts `radar_metadata["freezing_level_m"]` in meters ASL and
  converts it to above-antenna metadata. `None` means unknown; zero means no
  melting layer above the antenna. Check the receiving API's height convention
  when passing freezing levels or CAPPI heights between stages.
- Profile labels are `Prec.`, `As`, `CAE` and `Clutter`; use existing types.
- Use [config.py](src/vprc/config.py) for radar lookup rather than hardcoding
  stations, coordinates or network size. VVP header names must match a radar
  `name` or `aliases` entry; unknown names raise rather than use `[defaults]`.
  `antenna_height_m` and `lowest_level_offset_m` come only from the radar
  section or from runtime overrides given together; `[defaults]` has none.
  Configuration selects `VPRC_RADAR_CONFIG` if set, otherwise the packaged
  [radar_defaults.toml](src/vprc/radar_defaults.toml). Within the selected config,
  runtime metadata overrides radar-specific values, which override `[defaults]`.
  Config loading is cached; tests changing configuration must account for caches.

## Change conventions

- Write Pythonic, type-hinted code with English names and documentation.
- Keep comments focused on non-obvious algorithm reasoning. Mention corresponding
  Perl variable names or quote a short expression when useful; avoid fragile
  legacy line-number references.
- Use `logging` for processing diagnostics and surface invalid inputs explicitly;
  do not hide errors behind success-shaped fallback results. Raise
  `NoUsableProfileError` only for valid files that contain no usable profile.
- Keep changes scoped and update related tests/docs when APIs or behavior change.
  Maintain this file when development commands or important conventions change.
