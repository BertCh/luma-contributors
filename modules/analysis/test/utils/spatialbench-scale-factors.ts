// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Row counts of the SpatialBench tables (Apache Sedona, December 2025, Apache-2.0) per scale
 * factor, from https://sedona.apache.org/spatialbench/datasets-generators/. Used only to size
 * opt-in benchmarks (PLAT-07); no SpatialBench data or generator is bundled.
 */
export type SpatialBenchRowCounts = {
  scaleFactor: number;
  trip: number;
  building: number;
  customer: number;
  driver: number;
  vehicle: number;
  zone: number;
};

export function getSpatialBenchRowCounts(scaleFactor: number): SpatialBenchRowCounts {
  if (!(scaleFactor > 0)) throw new Error(`Invalid SpatialBench scale factor ${scaleFactor}`);
  // The published zone table is tiered by scale factor.
  const zone =
    scaleFactor < 10 ? 156095 : scaleFactor < 100 ? 455711 : scaleFactor < 1000 ? 1035371 : 1035749;
  return {
    scaleFactor,
    trip: Math.round(6e6 * scaleFactor),
    // 20K x (1 + log2(SF)), clamped to at least one building for SF below 0.5.
    building: Math.max(1, Math.round(20000 * (1 + Math.log2(scaleFactor)))),
    customer: Math.round(30000 * scaleFactor),
    driver: Math.round(500 * scaleFactor),
    vehicle: Math.round(100 * scaleFactor),
    zone
  };
}
