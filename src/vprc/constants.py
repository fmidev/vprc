#!/usr/bin/env python3
"""
Algorithm constants from the Koistinen & Pohjola VPR correction reference.

Based on allprof_prodx2.pl lines 16-28.
"""

# Minimum detectable signal threshold (dBZ)
# Values below this are considered missing/no echo
MDS = -45

# Missing gradient indicator
# Used in Perl to mark invalid/unavailable gradient values
# In Python we may prefer NaN, but this is kept for reference
GMDS = 100

# Gradient threshold for ground clutter detection (dBZ/m)
# Equivalent to -1 dBZ per 200m vertical step
# Steeper negative gradients indicate possible ground clutter
MKKYNNYS = -0.005

# Vertical resolution / height step (meters)
STEP = 200

# Minimum sample count for valid data
# Data points with fewer samples are considered unreliable
MIN_SAMPLES = 30

# Spike detection thresholds (dBZ)
# Used in spike smoothing algorithm (lines 490-686)
DBZ_KYNNYS1 = 4
DBZ_KYNNYS2 = -10
DBZ_KYNNYS3 = -10
DBZ_KYNNYS4 = 10

# Freezing level threshold (meters)
# Ground clutter correction is skipped if freezing level is between 0-1000m
# (allprof_prodx2.pl line 392)
FREEZING_LEVEL_MIN = 1000
