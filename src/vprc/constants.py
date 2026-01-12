#!/usr/bin/env python3
"""
Algorithm constants from the Koistinen & Pohjola VPR correction reference.
"""

# Minimum detectable signal threshold (dBZ)
# Values below this are considered missing/no echo
MDS = -45

# Gradient threshold for ground clutter detection (dBZ/m)
# Equivalent to -1 dBZ per 200m vertical step
# Steeper negative gradients indicate possible ground clutter
# Perl variable: $mkkynnys
GROUND_CLUTTER_GRADIENT_THRESHOLD = -0.005

# Vertical resolution / height step (meters)
STEP = 200

# Minimum sample count for valid data
# Data points with fewer samples are considered unreliable
MIN_SAMPLES = 30

# Freezing level threshold (meters)
# Ground clutter correction is skipped if freezing level is between 0-1000m
# (allprof_prodx2.pl line 392)
FREEZING_LEVEL_MIN = 1000

# Spike smoothing thresholds (dBZ)
# Amplitude threshold for detecting spikes
# Perl: hardcoded as 3 in spike detection conditions
SPIKE_AMPLITUDE_THRESHOLD = 3.0

# Threshold for large positive spikes - use 2-point average instead of 3-point
# Perl variable: $dbzkynnys1
LARGE_POSITIVE_SPIKE_THRESHOLD = 4.0

# Threshold for large negative spikes - use 2-point average instead of 3-point
# Perl variable: $dbzkynnys2
LARGE_NEGATIVE_SPIKE_THRESHOLD = -10.0

# -----------------------------------------------------------------------------
# VPR Correction Constants (from pystycappi.pl)
# -----------------------------------------------------------------------------

# Fine grid resolution for VPR correction (meters)
# The coarse 200m VVP profile is interpolated to this resolution
# Perl variable: $kerrospaksuus
FINE_GRID_RESOLUTION_M = 25

# Maximum height for VPR profile (meters)
# Perl: $kerroksia = int(15000 / $kerrospaksuus)
MAX_PROFILE_HEIGHT_M = 15000

# Default one-way half-power beamwidth (degrees)
# Used for Gaussian beam weighting in VPR correction
# Can be overridden per-radar in radar_defaults.toml
# Perl variable: $puolentehonleveys
DEFAULT_BEAMWIDTH_DEG = 0.95

# Maximum allowed VPR correction (dB)
# Corrections larger than this are capped
# Perl variable: $max_dBZ_korjaus
MAX_CORRECTION_DB = 30.0

# Minimum correction threshold (dB)
# Corrections smaller than this are set to zero
MIN_CORRECTION_THRESHOLD_DB = 0.05

# Default CAPPI heights for VPR correction (meters)
DEFAULT_CAPPI_HEIGHTS_M = (500, 1000)

# Default maximum range for VPR correction (km)
DEFAULT_MAX_RANGE_KM = 250

# Default range step for VPR correction (km)
DEFAULT_RANGE_STEP_KM = 1
