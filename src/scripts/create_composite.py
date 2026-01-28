#!/usr/bin/env python
"""Create composite VPR correction COGs from VVP profile files.

Usage:
    python src/scripts/create_composite.py [VVP_FILE ...]

If no files are given, uses the three test files from tests/data/.

Example:
    python src/scripts/create_composite.py tests/data/202308281000_*.txt
    python src/scripts/create_composite.py --output-dir /tmp/composite
"""

import sys
from pathlib import Path

import click

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vprc import (
    process_vvp,
    CompositeGrid,
    composite_corrections,
    create_empty_composite,
)
from vprc.composite import RadarCorrection
from vprc.io import write_composite_cogs, Compression

# Default test files
DEFAULT_FILES = (
    "tests/data/202308281000_KAN.VVP_40.txt",
    "tests/data/202308281000_KOR.VVP_40.txt",
    "tests/data/202308281000_VIH.VVP_40.txt",
)

# Radar coordinates from radar_defaults.toml
RADAR_COORDS = {
    "KAN": (61.81085, 22.50204),
    "KANKAANPAA": (61.81085, 22.50204),
    "KOR": (60.128469, 21.643379),
    "VIH": (60.5561915, 24.49558603),
    "VIHTI": (60.5561915, 24.49558603),
    "VAN": (60.270620, 24.869024),
    "VANTAA": (60.270620, 24.869024),
    "KUO": (62.862598, 27.381468),
    "KUOPIO": (62.862598, 27.381468),
    "VIM": (63.104835, 23.82086),
    "VIMPELI": (63.104835, 23.82086),
}


def extract_radar_code(filepath: Path) -> str:
    """Extract radar code from filename like 202308281000_KAN.VVP_40.txt."""
    name = filepath.stem  # e.g., "202308281000_KAN.VVP_40"
    parts = name.split("_")
    if len(parts) >= 2:
        return parts[1].split(".")[0]  # "KAN"
    return "UNKNOWN"


def extract_timestamp(filepath: Path) -> str:
    """Extract timestamp from filename like 202308281000_KAN.VVP_40.txt."""
    name = filepath.stem
    parts = name.split("_")
    if parts:
        return parts[0]  # "202308281000"
    return "unknown"


@click.command()
@click.argument(
    "files",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(path_type=Path),
    default=Path("local/output"),
    show_default=True,
    help="Output directory for COG files.",
)
@click.option(
    "--freezing-level", "-f",
    type=float,
    default=3000.0,
    show_default=True,
    help="Freezing level in meters.",
)
@click.option(
    "--resolution", "-r",
    type=float,
    default=1000.0,
    show_default=True,
    help="Grid resolution in meters.",
)
@click.option(
    "--max-range",
    type=float,
    default=250.0,
    show_default=True,
    help="Maximum radar range in km.",
)
@click.option(
    "--compress",
    type=click.Choice(["DEFLATE", "LZW", "ZSTD", "NONE"], case_sensitive=False),
    default="DEFLATE",
    show_default=True,
    help="Compression method.",
)
@click.option(
    "--force-quality",
    type=float,
    default=None,
    help="Override quality weight for all radars (useful for testing).",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Verbose output.",
)
def main(
    files: tuple[Path, ...],
    output_dir: Path,
    freezing_level: float,
    resolution: float,
    max_range: float,
    compress: str,
    force_quality: float | None,
    verbose: bool,
):
    """Create composite VPR correction COGs from VVP profile files.

    If no FILES are given, uses the three test files from tests/data/.
    """
    # Use default files if none provided
    if not files:
        files = tuple(Path(f) for f in DEFAULT_FILES)
        click.echo("Using default test files...")

    # Process each VVP file
    radar_corrections = []
    attempted_radars = []  # Track all attempted radars for grid bounds
    timestamp = None

    for filepath in files:
        if not filepath.exists():
            # Missing input file is an error - it was explicitly requested
            raise click.ClickException(f"Input file not found: {filepath}")

        radar_code = extract_radar_code(filepath)
        if timestamp is None:
            timestamp = extract_timestamp(filepath)

        if verbose:
            click.echo(f"Processing {filepath} (radar: {radar_code})")

        # Get radar coordinates
        if radar_code not in RADAR_COORDS:
            click.echo(f"Warning: Unknown radar code '{radar_code}', skipping", err=True)
            continue

        lat, lon = RADAR_COORDS[radar_code]
        attempted_radars.append((radar_code, lat, lon))

        # Process through pipeline
        try:
            result = process_vvp(filepath, freezing_level_m=freezing_level)
        except Exception as e:
            click.echo(f"Error processing {filepath}: {e}", err=True)
            continue

        if result.vpr_correction is None:
            click.echo(f"Info: No VPR correction for {radar_code} (profile not usable)", err=True)
            continue

        # Create radar correction object
        # Use forced quality weight if specified, otherwise use computed value
        quality_weight = (
            force_quality
            if force_quality is not None
            else result.vpr_correction.quality_weight
        )

        radar_corr = RadarCorrection(
            radar_code=radar_code,
            latitude=lat,
            longitude=lon,
            correction=result.vpr_correction,
            quality_weight=quality_weight,
        )

        if verbose:
            click.echo(f"  Quality weight: {radar_corr.quality_weight:.3f}")
            click.echo(f"  Ground reference: {result.vpr_correction.z_ground_dbz:.1f} dBZ")

        radar_corrections.append(radar_corr)

    # Determine grid bounds from attempted radars (or use Finland fallback)
    if attempted_radars:
        # Use locations of all attempted radars for grid bounds
        lats = [r[1] for r in attempted_radars]
        lons = [r[2] for r in attempted_radars]
        attempted_codes = [r[0] for r in attempted_radars]

        # Transform to ETRS-TM35FIN for grid bounds
        from pyproj import CRS, Transformer
        transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_epsg(3067), always_xy=True)

        xs, ys = [], []
        for lat, lon in zip(lats, lons):
            x, y = transformer.transform(lon, lat)
            xs.append(x)
            ys.append(y)

        # Extend bounds by max range
        margin = max_range * 1000  # km to m
        xmin = min(xs) - margin
        xmax = max(xs) + margin
        ymin = min(ys) - margin
        ymax = max(ys) + margin

        if verbose:
            click.echo(f"\nGrid bounds (ETRS-TM35FIN):")
            click.echo(f"  X: {xmin:.0f} to {xmax:.0f}")
            click.echo(f"  Y: {ymin:.0f} to {ymax:.0f}")

        grid = CompositeGrid.from_bounds(
            xmin, xmax, ymin, ymax,
            resolution_m=resolution,
        )
    else:
        # No radars at all - use Finland-wide grid as fallback
        click.echo("Warning: No radar locations available, using Finland-wide grid", err=True)
        grid = CompositeGrid.for_finland(resolution_m=resolution)
        attempted_codes = []

    click.echo(f"Grid size: {len(grid.x)} x {len(grid.y)} = {len(grid.x) * len(grid.y)} cells")

    if not radar_corrections:
        # No usable corrections - this is a valid scenario (e.g., no precipitation)
        click.echo("\nNo usable VPR corrections (no precipitation detected by any radar)")
        click.echo("Creating empty composite with no corrections...")
        composite = create_empty_composite(grid, radar_codes=attempted_codes)
    else:
        click.echo(f"\nCreating composite from {len(radar_corrections)} radars:")
        for rc in radar_corrections:
            click.echo(f"  {rc.radar_code}: quality={rc.quality_weight:.3f}")

        # Create composite
        composite = composite_corrections(
            radar_corrections,
            grid,
            max_range_km=max_range,
        )

    # Export COGs
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = timestamp or "composite"
    compress_typed: Compression = compress.upper()  # type: ignore[assignment]
    outputs = write_composite_cogs(
        composite,
        output_dir,
        prefix=prefix,
        compress=compress_typed,
    )

    click.echo(f"\nCreated {len(outputs)} COG files:")
    for var, path in outputs.items():
        size_kb = path.stat().st_size / 1024
        click.echo(f"  {path.name} ({size_kb:.1f} KB)")

    if composite.attrs.get("empty_composite", False):
        click.echo("\nNote: These are empty COGs (no precipitation corrections).")

    click.echo(f"\nDone! Output directory: {output_dir}")


if __name__ == "__main__":
    main()
