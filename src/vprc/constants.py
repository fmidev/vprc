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

# -----------------------------------------------------------------------------
# Bright Band Slope Adjustment Constants (from allprof_prodx2.pl)
# -----------------------------------------------------------------------------

# Slope threshold for BB adjustment (dBZ per 200m step)
# When slope exceeds this, ground reference is adjusted to avoid over-correction
# Perl variable: $bbkallistus threshold (0.2)
BB_SLOPE_THRESHOLD = 0.2

# Maximum height for BB lower edge to trigger slope adjustment (meters)
# Adjustment only applied when BB lower edge (zala) is below this height
# Perl: condition `$zala < 600`
BB_SLOPE_HEIGHT_LIMIT_M = 600

# Maximum allowed BB amplitude above peak when BB is at surface (dBZ)
# When BB peak is at lowest level and amplitude_above exceeds this,
# the peak value is reduced to cap effective amplitude at this threshold.
# Perl: `if ( $ylaamp >= 10 and $bbalku == $alintaso )` (lines 1091-1098)
BB_PEAK_AMPLITUDE_CAP_DB = 10.0

# Minimum linear Z (mm^6/m^3) required for valid near-surface bright band
# When BB bottom is at lowest 1-2 levels, sample count at both BB bottom and top
# must exceed this threshold.
# Perl: `$isotaulu[$bbalku][$j][1] < 500` (lines 1051-1075) where [1] is Zcount
NEAR_SURFACE_BB_MIN_ZCOUNT = 500

# Minimum ratio of sample count at BB top to BB bottom for valid near-surface BB
# When BB bottom is at lowest levels, this prevents false detection from clutter.
# Perl: `$isotaulu[$zyla][$j][1] / $isotaulu[$bbalku][$j][1] < 0.7` (lines 1051-1075)
NEAR_SURFACE_BB_MIN_ZCOUNT_RATIO = 0.7

# -----------------------------------------------------------------------------
# Post-BB Ground Clutter Correction Constants (from allprof_prodx2.pl)
# -----------------------------------------------------------------------------

# Maximum dBZ increase allowed between adjacent levels for clutter correction
# If lower level > upper level + this threshold, lower is capped
# Perl: `$isotaulu[$i - 200][$j][0] > $isotaulu[$i][$j][0] + 1`
POST_BB_CLUTTER_THRESHOLD_DB = 1.0

# BB bottom height threshold for triggering post-BB clutter correction (meters)
# When BB bottom <= this value, clutter correction is applied below BB
# Perl: `if ( $zala <= 800 and $bb == 1 )`
LOW_BB_HEIGHT_M = 800

# Height range above lowest level for post-BB clutter correction (meters)
# Correction is applied from lowest level to lowest + this value
# Perl: `for ( $i = $alintaso + 600 ; $i > $alintaso ; $i = $i - 200 )`
POST_BB_CLUTTER_HEIGHT_M = 600

# Threshold for large dBZ jump smoothing in post-BB correction (dB)
# If jump between levels > this, apply additional smoothing
# Perl: `if ( $isotaulu[ $i + 200 ][$j][5] - $isotaulu[$i][$j][5] > 6 )`
LARGE_JUMP_THRESHOLD_DB = 6.0

# Maximum freezing level for applying clutter correction when no BB detected
# Perl: `$bb == 0 and $nollaraja <= 1000`
NO_BB_CLUTTER_FREEZING_LEVEL_M = 1000

# Height offsets for BB spike restoration (meters)
# After spike smoothing, restore original values at these offsets from BB peak
# Perl: restores at bbalku-400, bbalku-200, bbalku, bbalku+200, bbalku+400
BB_SPIKE_RESTORATION_OFFSETS = (-2*STEP, -STEP, 0, STEP, 2*STEP)

# -----------------------------------------------------------------------------
# Classification constants (from allprof_prodx2.pl)
# -----------------------------------------------------------------------------

# Maximum allowed dBZ drop from max to min below (dB)
# If reflectivity decreases by more than this from peak toward ground,
# precipitation is likely evaporating before reaching surface.
# Perl: `if ( $evapor > 20 )` (lines 1174-1185)
EVAPORATION_THRESHOLD_DB = 20.0

# -----------------------------------------------------------------------------
# Temporal Averaging Constants (from pystycappi_ka.pl)
# -----------------------------------------------------------------------------

# Weight for climatological profile when blending with precipitation profile
# Perl: `corr_klim * 0.2 + corr_sade * $wq`
CLIMATOLOGICAL_WEIGHT = 0.2
