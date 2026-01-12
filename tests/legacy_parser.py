"""Parser for legacy Perl output files.

These parsers read the output files from allprof_prodx2.pl for comparison
with the Python reimplementation.
"""

from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import xarray as xr

from vprc.constants import MDS


@dataclass
class LegacyProfileHeader:
    """Parsed header from .profile file."""

    radar_name: str
    product: str
    step: int
    mds: float
    timestamp: datetime


@dataclass
class LegacyProfileFooter:
    """Parsed footer metadata from .profile file.

    Based on the Finnish column headers in the legacy output:
    kerros, alaraja, yläraja, maxdbz, maakaiku, KIK, As, sade, Unclf, BB,
    BB korkeus, BB amplitudi, 0-rajan korkeus, bb:n alaraja, bb:n yläraja,
    haihtuminen, laatupaino, pv, kk, vv, h, min
    """

    layer_num: int
    bottom_height: int
    top_height: int
    max_dbz: float
    ground_clutter: int  # boolean flag
    clear_air_echo: int  # KIK = boolean flag
    altostratus: int  # As = boolean flag
    precipitation: int  # sade = precipitation type (0=none, 1=snow, 2=rain, 3=sleet)
    unclassified: int  # Unclf = boolean flag
    bright_band: int  # BB = boolean flag
    bb_height: float
    bb_amplitude: float
    freezing_level: float  # 0-rajan korkeus
    bb_bottom: float
    bb_top: float
    evaporation: float  # haihtuminen
    quality_weight: float  # laatupaino
    day: int
    month: int
    year: int
    hour: int
    minute: int


def parse_legacy_profile(path: str | Path) -> tuple[xr.Dataset, LegacyProfileHeader, LegacyProfileFooter]:
    """Parse a legacy .profile file from allprof_prodx2.pl.

    The .profile format consists of:
    - Line 1: Header with radar name, product, step, MDS, date, time
    - Lines 2-N: Data rows with height, lin_dbz, log_dbz, corrected_dbz,
                 kor_profile, gradient, sample_count
    - Line N+1: Column description (Finnish)
    - Line N+2: Footer with classification/metadata values

    Args:
        path: Path to the .profile file

    Returns:
        Tuple of (dataset, header, footer) where dataset contains the
        profile data with 'height' as coordinate.
    """
    path = Path(path)
    lines = path.read_text().strip().split('\n')

    # Parse header (first line)
    header = _parse_header(lines[0])

    # Parse footer (last line, skip the column description line before it)
    footer = _parse_footer(lines[-1])

    # Parse data lines (everything between header and footer description)
    # Skip header (1 line) and footer (2 lines: description + data)
    data_lines = lines[1:-2]
    ds = _parse_data(data_lines, header)

    return ds, header, footer


def _parse_header(line: str) -> LegacyProfileHeader:
    """Parse the header line of .profile file.

    Format: KANKAANPAA VVP_40 200 -45 24.08. 2025 11:00
    """
    parts = line.split()
    radar_name = parts[0]
    product = parts[1]
    step = int(parts[2])
    mds = float(parts[3])

    # Parse date: "24.08." "2025" "11:00"
    date_str = parts[4].rstrip('.')  # Remove trailing dot
    year = int(parts[5])
    time_str = parts[6]

    day, month = map(int, date_str.split('.'))
    hour, minute = map(int, time_str.split(':'))

    timestamp = datetime(year, month, day, hour, minute)

    return LegacyProfileHeader(
        radar_name=radar_name,
        product=product,
        step=step,
        mds=mds,
        timestamp=timestamp,
    )


def _parse_footer(line: str) -> LegacyProfileFooter:
    """Parse the footer data line of .profile file.

    Format: 1 126 6726 34.8 1 0 0 2 0 0 0 0 0 0 0 0.0 0.0 3.0 0.66 24 8 2025 11 0
    Indices:0 1   2    3    4 5 6 7 8 9 10 11 12 13 14 15  16  17  18   19 20 21  22 23
    """
    parts = line.split()

    return LegacyProfileFooter(
        layer_num=int(parts[0]),
        bottom_height=int(parts[1]),
        top_height=int(parts[2]),
        max_dbz=float(parts[3]),
        ground_clutter=int(parts[4]),
        clear_air_echo=int(parts[5]),
        altostratus=int(parts[6]),
        precipitation=int(parts[7]),
        unclassified=int(parts[8]),
        bright_band=int(parts[9]),
        bb_height=float(parts[10]),
        bb_amplitude=float(parts[11]),
        freezing_level=float(parts[12]),
        bb_bottom=float(parts[13]),
        bb_top=float(parts[14]),
        evaporation=float(parts[15]),
        # parts[16] and parts[17] are additional fields (unclear meaning)
        quality_weight=float(parts[18]),
        day=int(parts[19]),
        month=int(parts[20]),
        year=int(parts[21]),
        hour=int(parts[22]),
        minute=int(parts[23]),
    )


def _parse_data(lines: list[str], header: LegacyProfileHeader) -> xr.Dataset:
    """Parse the data lines into an xarray Dataset.

    Columns: height lin_dbz log_dbz corrected_dbz kor_profile gradient sample_count
    """
    heights = []
    lin_dbz = []
    log_dbz = []
    corrected_dbz = []
    kor_profile = []
    gradient = []
    sample_count = []

    for line in lines:
        parts = line.split()
        if len(parts) < 7:
            continue

        heights.append(int(parts[0]))
        lin_dbz.append(float(parts[1]))
        log_dbz.append(float(parts[2]))
        corrected_dbz.append(float(parts[3]))
        kor_profile.append(float(parts[4]))
        gradient.append(float(parts[5]))
        sample_count.append(int(parts[6]))

    height_arr = np.array(heights)

    ds = xr.Dataset(
        {
            'lin_dbz': (['height'], np.array(lin_dbz)),
            'log_dbz': (['height'], np.array(log_dbz)),
            'corrected_dbz': (['height'], np.array(corrected_dbz)),
            'kor_profile': (['height'], np.array(kor_profile)),
            'gradient': (['height'], np.array(gradient)),
            'sample_count': (['height'], np.array(sample_count)),
        },
        coords={'height': height_arr},
        attrs={
            'radar_name': header.radar_name,
            'product': header.product,
            'step': header.step,
            'mds': header.mds,
            'timestamp': header.timestamp.isoformat(),
        },
    )

    return ds


@dataclass
class LegacyCorHeader:
    """Parsed header from .cor file.

    Format: KANKAANPAA VVP_40 200 -45 24 08 2025 11 00 sade 2 max_dBZ 34.8 max_height 1126 0.66 0 ELEVS 0.7 1.5 3.0 3.3
    """

    radar_name: str
    product: str
    step: int
    mds: float
    timestamp: datetime
    precip_type: str  # 'sade' = rain
    precip_code: int  # 0=none, 1=snow, 2=rain, 3=sleet
    max_dbz: float
    max_height: int
    quality_weight: float
    bb_flag: int
    elevation_angles: list[float]


def parse_legacy_cor(path: str | Path) -> tuple[xr.Dataset, LegacyCorHeader]:
    """Parse a legacy .cor file with VPR correction factors.

    The .cor format consists of:
    - Line 1: Header with metadata and elevation angles
    - Lines 2+: Data rows with range and per-elevation correction/weight/height

    Args:
        path: Path to the .cor file

    Returns:
        Tuple of (dataset, header) where dataset contains correction factors
        with dimensions (range_km, elevation).
    """
    path = Path(path)
    lines = path.read_text().strip().split('\n')

    # Parse header
    header = _parse_cor_header(lines[0])

    # Parse data lines
    ds = _parse_cor_data(lines[1:], header)

    return ds, header


def _parse_cor_header(line: str) -> LegacyCorHeader:
    """Parse the header line of .cor file.

    Format: KANKAANPAA VVP_40 200 -45 24 08 2025 11 00 sade 2 max_dBZ 34.8 max_height 1126 0.66 0 ELEVS 0.7 1.5 3.0 3.3
    """
    parts = line.split()

    radar_name = parts[0]
    product = parts[1]
    step = int(parts[2])
    mds = float(parts[3])

    # Date/time: day month year hour minute
    day = int(parts[4])
    month = int(parts[5])
    year = int(parts[6])
    hour = int(parts[7])
    minute = int(parts[8])
    timestamp = datetime(year, month, day, hour, minute)

    precip_type = parts[9]  # 'sade'
    precip_code = int(parts[10])

    # Find max_dBZ and max_height by label
    max_dbz_idx = parts.index('max_dBZ') + 1
    max_dbz = float(parts[max_dbz_idx])

    max_height_idx = parts.index('max_height') + 1
    max_height = int(parts[max_height_idx])

    # Quality weight and BB flag follow max_height
    quality_weight = float(parts[max_height_idx + 1])
    bb_flag = int(parts[max_height_idx + 2])

    # Elevation angles after 'ELEVS'
    elevs_idx = parts.index('ELEVS') + 1
    elevation_angles = [float(e) for e in parts[elevs_idx:]]

    return LegacyCorHeader(
        radar_name=radar_name,
        product=product,
        step=step,
        mds=mds,
        timestamp=timestamp,
        precip_type=precip_type,
        precip_code=precip_code,
        max_dbz=max_dbz,
        max_height=max_height,
        quality_weight=quality_weight,
        bb_flag=bb_flag,
        elevation_angles=elevation_angles,
    )


def _parse_cor_data(lines: list[str], header: LegacyCorHeader) -> xr.Dataset:
    """Parse the data lines of .cor file into an xarray Dataset.

    The .cor format has 6 triplets per line (for rain cases):
    1. CAPPI 500m: (correction_klim, correction_rain, beam_height)
    2. CAPPI 1000m: (correction_klim, correction_rain, beam_height)
    3-6. Elevation angles 1-4: (correction_klim, correction_rain, beam_height)

    Returns dataset with dimensions (range_km, correction_type) where
    correction_type is ['cappi_500', 'cappi_1000', 'elev_1', 'elev_2', 'elev_3', 'elev_4']
    """
    n_elevs = len(header.elevation_angles)
    n_triplets = 2 + n_elevs  # 2 CAPPI heights + n elevation angles
    has_precip = header.precip_code > 0

    ranges = []
    corrections_klim = []  # [range, type]
    corrections_rain = []
    beam_heights = []

    for line in lines:
        parts = line.split()
        if len(parts) < 1 + 3 * n_triplets:
            continue

        ranges.append(int(parts[0]))

        row_corr_klim = []
        row_corr_rain = []
        row_height = []

        for i in range(n_triplets):
            base_idx = 1 + i * 3
            if has_precip:
                # For rain: (klim, rain, height)
                row_corr_klim.append(float(parts[base_idx]))
                row_corr_rain.append(float(parts[base_idx + 1]))
            else:
                # For non-rain: (klim, ref=100, height)
                row_corr_klim.append(float(parts[base_idx]))
                row_corr_rain.append(np.nan)
            row_height.append(int(parts[base_idx + 2]))

        corrections_klim.append(row_corr_klim)
        corrections_rain.append(row_corr_rain)
        beam_heights.append(row_height)

    range_arr = np.array(ranges)
    type_labels = ['cappi_500', 'cappi_1000'] + [f'elev_{i+1}' for i in range(n_elevs)]

    ds = xr.Dataset(
        {
            'correction_klim_db': (['range_km', 'correction_type'], np.array(corrections_klim)),
            'correction_rain_db': (['range_km', 'correction_type'], np.array(corrections_rain)),
            'beam_height_m': (['range_km', 'correction_type'], np.array(beam_heights)),
        },
        coords={
            'range_km': range_arr,
            'correction_type': type_labels,
        },
        attrs={
            'radar_name': header.radar_name,
            'product': header.product,
            'max_dbz': header.max_dbz,
            'max_height': header.max_height,
            'quality_weight': header.quality_weight,
            'timestamp': header.timestamp.isoformat(),
            'elevation_angles': header.elevation_angles,
        },
    )

    return ds
