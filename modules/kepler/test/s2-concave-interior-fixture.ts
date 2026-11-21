// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Independent Shapely enumeration in the face-0 central-projection chart, not a cover engine:
// ring -> (tan(lon),tan(lat)/cos(lon)); quadratic S2 ST corners at i/64,j/64.
// Polygon.covers(cell) gives these literal face/level/i/j tuples.
export const S2_CONCAVE_INTERIOR_COORDINATES = [
  '0/6/29/29',
  '0/6/29/30',
  '0/6/29/31',
  '0/6/29/32',
  '0/6/29/33',
  '0/6/29/34',
  '0/6/30/29',
  '0/6/30/30',
  '0/6/30/31',
  '0/6/30/32',
  '0/6/30/33',
  '0/6/30/34',
  '0/6/31/29',
  '0/6/31/30',
  '0/6/32/29',
  '0/6/32/30',
  '0/6/33/29',
  '0/6/33/30',
  '0/6/33/31',
  '0/6/33/32',
  '0/6/33/33',
  '0/6/33/34',
  '0/6/34/29',
  '0/6/34/30',
  '0/6/34/31',
  '0/6/34/32',
  '0/6/34/33',
  '0/6/34/34'
];
