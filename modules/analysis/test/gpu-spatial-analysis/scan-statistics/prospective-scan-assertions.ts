// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {
  GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX,
  GPU_SPATIAL_SCAN_STATISTIC_CLUSTER,
  GPU_SPATIAL_SCAN_STATISTIC_SUMMARY
} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-parameters';

// Independent enumeration of all 25 centers, 1..8 nearest zones and all <=3-bucket cylinders.
// Python binary64 inside/outside Poisson density; full derivation/script is recorded in the stream status.
export const EXPECTED_SCAN_WINDOWS = {
  retrospective: {center: 0, firstBucket: 2, lastBucket: 3, zoneCount: 4, llr: 338.9490178620206},
  prospective: {center: 4, firstBucket: 6, lastBucket: 7, zoneCount: 4, llr: 79.81643822504134}
} as const;

export type ScanWindowResult = {
  indices: ArrayLike<number>;
  clusters: ArrayLike<number>;
  summary: ArrayLike<number>;
};

export function expectScanWindow(
  result: ScanWindowResult,
  mode: keyof typeof EXPECTED_SCAN_WINDOWS
): void {
  const expected = EXPECTED_SCAN_WINDOWS[mode];
  expect(result.summary[GPU_SPATIAL_SCAN_STATISTIC_SUMMARY.clusterCount]).toBeGreaterThan(0);
  for (const field of ['center', 'zoneCount', 'firstBucket', 'lastBucket'] as const) {
    expect(result.indices[GPU_SPATIAL_SCAN_STATISTIC_CLUSTER_INDEX[field]], field).toBe(
      expected[field]
    );
  }
  const score = result.clusters[GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.logLikelihoodRatio];
  expect(Number.isFinite(score)).toBe(true);
  expect(score).toBeGreaterThan(0);
  expect(Math.abs(score - expected.llr)).toBeLessThanOrEqual(0.002 + 2e-5 * expected.llr);
}
