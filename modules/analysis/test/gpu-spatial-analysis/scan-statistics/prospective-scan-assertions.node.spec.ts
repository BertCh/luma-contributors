// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {expectScanWindow, EXPECTED_SCAN_WINDOWS} from './prospective-scan-assertions';
import {
  GPU_SPATIAL_SCAN_STATISTIC_CLUSTER,
  GPU_SPATIAL_SCAN_STATISTIC_SUMMARY
} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-parameters';

it('S3 rejects a zeroed retrospective record with no published cluster', () => {
  expect(() =>
    expectScanWindow(
      {indices: new Uint32Array(4), clusters: new Float32Array(8), summary: new Uint32Array(4)},
      'retrospective'
    )
  ).toThrow();
});

for (const mode of ['retrospective', 'prospective'] as const) {
  it(`S3 accepts the independently enumerated ${mode} window`, () => {
    const expected = EXPECTED_SCAN_WINDOWS[mode];
    const clusters = new Float32Array(8),
      summary = new Uint32Array(4);
    clusters[GPU_SPATIAL_SCAN_STATISTIC_CLUSTER.logLikelihoodRatio] = expected.llr;
    summary[GPU_SPATIAL_SCAN_STATISTIC_SUMMARY.clusterCount] = 1;
    expectScanWindow(
      {
        indices: Uint32Array.of(
          expected.center,
          expected.zoneCount,
          expected.firstBucket,
          expected.lastBucket
        ),
        clusters,
        summary
      },
      mode
    );
  });
}
