# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

"""Run: /usr/bin/python3 modules/analysis/test/gpu-network/network-isochrones/isochrones-output-oracles.oracle.py

Independent Web Mercator / Quadbin equations, no implementation imports.
The spec stores the results as literals and never invokes Python.
"""
import math

zoom, column, row = 2, 1, 1
morton = sum(((column >> bit) & 1) << (2 * bit) |
             ((row >> bit) & 1) << (2 * bit + 1) for bit in range(zoom))
footer_bits = 52 - 2 * zoom
key = (0x4800000000000000 | (zoom << 52) | (morton << footer_bits) |
       ((1 << footer_bits) - 1))
north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * row / 2**zoom))))
south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (row + 1) / 2**zoom))))
print('key:', [key & 0xffffffff, key >> 32])
print('north:', north, 'south:', south)
print('area:', (360 / 2**zoom) * (north - south))

# Rich fixture: a detached tile precedes an eight-tile annulus in Morton order.
# Enumerate tile neighbors directly; the resulting boundary is a detached rectangle,
# a 3x3 outer rectangle with 12 edges and a one-tile hole with four edges.
zoom = 4
tiles = [(1, 1)] + [(x, y) for y in range(7, 10) for x in range(7, 10) if (x, y) != (8, 8)]
def tile_key(tile):
    x, y = tile
    path = sum((((x >> bit) & 1) << (2 * bit)) |
               (((y >> bit) & 1) << (2 * bit + 1)) for bit in range(zoom))
    return (0x4800000000000000 | (zoom << 52) | (path << (52 - 2 * zoom)) |
            ((1 << (52 - 2 * zoom)) - 1))
def latitude(y):
    return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / 2**zoom))))
tiles.sort(key=tile_key)
print('tiles:', tiles)
print('positions:', [v for x, y in tiles for v in ((x + .5) / 16 * 360 - 180, latitude(y + .5))])
print('keys:', [v for tile in tiles for v in (tile_key(tile) & 0xffffffff, tile_key(tile) >> 32)])
edges = [(index, edge) for index, (x, y) in enumerate(tiles)
         for edge, neighbor in enumerate([(x, y-1), (x+1, y), (x, y+1), (x-1, y)])
         if neighbor not in tiles]
print('rows:', [index for index, edge in edges])
print('edgeIndices:', [edge for index, edge in edges])
print('groups:', [3 if tiles[index] == (1, 1) else 7 for index, edge in edges])
print('latitudes:', [(y, latitude(y)) for y in [1, 2, 7, 8, 9, 10]])
print('areas:', [22.5*(latitude(1)-latitude(2)), 67.5*(latitude(7)-latitude(10)), -22.5*(latitude(8)-latitude(9))])

# R2: the high-valued bottom-right corner (9) is clipped at level 5, its
# bottom neighbor is 3 and top neighbor is 1. Linear interpolation determines
# the three vertices without using the marching-squares implementation.
from fractions import Fraction
triangle = [(Fraction(10, 3), Fraction(1, 2)), (6, Fraction(1, 2)), (6, 1)]
centroid = tuple(sum(point[axis] for point in triangle) / 3 for axis in (0, 1))
samples = [(2, Fraction(1, 2), 2), (2, Fraction(3, 2), 2), (6, Fraction(3, 2), 5)]
print('R2 triangle:', triangle, 'centroid:', centroid)
for scale in [(1, 1), (4, 1)]:
    distances = [sum(((centroid[axis]-sample[axis])/scale[axis])**2 for axis in (0, 1)) for sample in samples]
    print('R2 scale:', scale, 'distances:', [float(value) for value in distances],
          'nearest facility:', samples[min(range(3), key=distances.__getitem__)][2])
print('R5 segment starts:', [
    [(tiles[index][0]/16*360-180, latitude(tiles[index][1])),
     ((tiles[index][0]+1)/16*360-180, latitude(tiles[index][1])),
     ((tiles[index][0]+1)/16*360-180, latitude(tiles[index][1]+1)),
     (tiles[index][0]/16*360-180, latitude(tiles[index][1]+1))][edge] for index, edge in edges])
