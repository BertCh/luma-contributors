# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
# Run: /usr/bin/python3 modules/analysis/test/gpu-spatial-analysis/polygon-buffer/buffer-output-oracles.oracle.py
from shapely.geometry import LineString
import struct

# Flat caps (2); join resolution is 8 segments per quadrant, irrelevant for a straight line.
surface = LineString([(0, 0), (4, 0)]).buffer(1, resolution=8, cap_style=2, join_style=2)
print({"area": surface.area, "rings": 1 + len(surface.interiors),
       "vertices": sorted(surface.exterior.coords[:-1])})
translated = LineString([(0, 5000000), (0, 5000004)]).buffer(0.5, resolution=8, cap_style=3)
assert translated.area == 5
print({"translated_square_cap_area": translated.area, "rings": 1 + len(translated.interiors)})
for cap_style in (1, 2, 3):
    collapsed = LineString([(3, 7), (3, 7)]).buffer(1, resolution=8, cap_style=cap_style)
    print({"collapsed_cap_style": cap_style, "area": collapsed.area})

# Independent IEEE-754 bound, shared by buffer and overlay after snap rounding. Every nonzero
# admitted coordinate is an integer in [1, 2**22+1]. The raw 24-bit-significand product exponent
# therefore spans at most 44 bits, well below the orientation predicate's 200-bit refusal span.
def significand_exponent(value):
    bits = struct.unpack("!I", struct.pack("!f", value))[0]
    return ((bits >> 23) & 255) - 150

minimum_exponent = significand_exponent(1)
maximum_exponent = significand_exponent(2**22 + 1)
product_span = 2 * (maximum_exponent - minimum_exponent)
assert (minimum_exponent, maximum_exponent, product_span) == (-23, -1, 44)
assert product_span < 200
print({"lattice_coordinate_exponents": [minimum_exponent, maximum_exponent],
       "maximum_raw_product_exponent_span": product_span,
       "uncertain_retained_pair_count": 0})
