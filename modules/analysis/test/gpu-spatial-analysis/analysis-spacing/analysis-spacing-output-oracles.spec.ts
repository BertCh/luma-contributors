// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUAnalysisSpacing,
  GPU_ANALYSIS_SPACING_SUMMARY_LENGTH,
  decodeGPUAnalysisSpacingStatistics
} from '../../../src/gpu-spatial-analysis/analysis-spacing/index';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  RECTANGLE_POSITIONS,
  RECTANGLE_STATISTICS
} from './analysis-spacing-output-oracles-fixtures';

it('writes every rectangle statistic from closed form into a poisoned summary', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOutputOracleFixture(device);
  const summary = fixture.output('summary', 'uint32', GPU_ANALYSIS_SPACING_SUMMARY_LENGTH);
  fixture.graph.add(
    new GPUAnalysisSpacing({
      positions: fixture.input('positions', 'float32x2', RECTANGLE_POSITIONS),
      spatialContext: {coordinateSpace: 'planar', metric: 'native'},
      neighborRank: 3,
      gridSize: [4, 4],
      output: {summary}
    })
  );
  const compiled = fixture.graph.compile();
  try {
    const words = Uint32Array.from(
      await fixture.submitTwice(compiled, () => fixture.readRows([summary]))
    );
    // All expected values are exactly representable in f32; equality also rejects NaN.
    expect(decodeGPUAnalysisSpacingStatistics(words)).toEqual(RECTANGLE_STATISTICS);
    expect(Array.from(words.slice(0, 8))).toEqual([4, 0, 0, 12, 0, 3, 4, 1]);
    expect(Array.from(words.slice(36))).toEqual(Array(508).fill(0));
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
