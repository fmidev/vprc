#!/usr/bin/env python3
"""
Algorithm constants from the Koistinen & Pohjola VPR correction reference.

Based on allprof_prodx2.pl lines 16-28.
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
