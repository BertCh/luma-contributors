// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Exact input f32 integers. Hand-worked at latitude zero: Quadbin/Quadkey clip to
// +/-180, both wrap to column 0; the equatorial row is 2^(zoom-1). Geohash keeps
// positive endpoint bin 2^30-1 and negative endpoint bin 0, latitude bin 2^29.
// Interleave those bits (geohash longitude first), then OR the family/level header.
export const CLIPPED_LONGITUDE_FIXTURES = [
  {family: 'quadbin', resolution: 26, negative: 0x49a8000000000000n, positive: 0x49a8000000000000n},
  {family: 'quadkey', resolution: 29, negative: 0x7600000000000000n, positive: 0x7600000000000000n},
  {family: 'geohash', resolution: 12, negative: 0xc400000000000000n, positive: 0xceaaaaaaaaaaaaaan}
] as const;
export const CLIPPED_LONGITUDES = Float32Array.of(-(2 ** 30), 0, 2 ** 30, 0, -1e30, 0, 1e30, 0);
