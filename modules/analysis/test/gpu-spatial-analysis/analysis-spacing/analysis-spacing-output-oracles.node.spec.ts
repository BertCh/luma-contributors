// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeGPUAnalysisSpacingStatistics} from '../../../src/gpu-spatial-analysis/ontology/analysis-experience/index';
import {
  GPUAnalysisSpacing,
  GPU_ANALYSIS_SPACING_SUMMARY_LENGTH,
  decodeGPUAnalysisSpacingStatistics
} from '../../../src/gpu-spatial-analysis/analysis-spacing/index';
import {createOutputOracleFixture} from '../cell-topology/output-oracle-test-utils';
import {
  RECTANGLE_POSITIONS,
  RECTANGLE_STATISTICS
} from './analysis-spacing-output-oracles-fixtures';

it('resolves the public spacing module and poisoned-readback helper imported by the GPU oracle', () => {
  expect(typeof GPUAnalysisSpacing).toBe('function');
  expect(GPU_ANALYSIS_SPACING_SUMMARY_LENGTH).toBe(544);
  expect(typeof decodeGPUAnalysisSpacingStatistics).toBe('function');
  expect(typeof createOutputOracleFixture).toBe('function');
});

it('matches every rectangle statistic to its closed form', () => {
  expect(
    computeGPUAnalysisSpacingStatistics(RECTANGLE_POSITIONS, {
      coordinateModel: 'planar',
      neighborRank: 3
    })
  ).toEqual(RECTANGLE_STATISTICS);
});
