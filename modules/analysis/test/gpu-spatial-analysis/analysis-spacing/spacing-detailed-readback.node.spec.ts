// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  decodeGPUAnalysisSpacingStatistics,
  decodeGPUAnalysisSpacingStatisticsDetailed
} from '../../../src/gpu-spatial-analysis/analysis-spacing/gpu-analysis-spacing';

function singleton(invalidCount = 0) {
  const words = new Uint32Array(544);
  words[0] = 1;
  words[1] = invalidCount;
  words[5] = 1;
  words[7] = 1;
  return words;
}

it('keeps invalid counts in the ordinary statistics', () => {
  expect(decodeGPUAnalysisSpacingStatistics(singleton(1))?.invalidCount).toBe(1);
});

it('distinguishes empty, all-invalid, mixed and incomplete summaries', () => {
  const words = singleton();
  words[0] = 0;
  expect(decodeGPUAnalysisSpacingStatisticsDetailed(words)).toMatchObject({
    pointCount: 0,
    invalidCount: 0,
    unavailableReason: 'empty'
  });
  words[1] = 2;
  expect(decodeGPUAnalysisSpacingStatisticsDetailed(words)).toMatchObject({
    pointCount: 0,
    invalidCount: 2,
    unavailableReason: 'all-invalid'
  });
  words[0] = 1;
  expect(decodeGPUAnalysisSpacingStatisticsDetailed(words).statistics?.invalidCount).toBe(2);
  words[27] = 1;
  words[3] = 300;
  expect(decodeGPUAnalysisSpacingStatisticsDetailed(words)).toMatchObject({
    pointCount: 1,
    invalidCount: 2,
    candidateOverflow: 1,
    requiredCount: 300,
    complete: 0,
    unavailableReason: 'incomplete'
  });
  expect(decodeGPUAnalysisSpacingStatisticsDetailed(words).statistics).toBeUndefined();
  expect(() => decodeGPUAnalysisSpacingStatistics(words)).toThrow(/candidateOverflow/);
});

it('reads bounded byte subviews and rejects partial or short elements', () => {
  const storage = new Uint8Array(2184);
  storage.set(new Uint8Array(singleton(300).buffer), 4);
  const bytes = storage.subarray(4, 2180);
  expect(decodeGPUAnalysisSpacingStatistics(bytes)?.invalidCount).toBe(300);
  expect(decodeGPUAnalysisSpacingStatistics(bytes.slice().buffer)?.pointCount).toBe(1);
  expect(() => decodeGPUAnalysisSpacingStatistics(bytes.subarray(1))).toThrow(/whole/);
  expect(() => decodeGPUAnalysisSpacingStatistics(bytes.subarray(4))).toThrow(/544/);
  const words = singleton();
  new Float32Array(words.buffer)[14] = Infinity;
  expect(decodeGPUAnalysisSpacingStatisticsDetailed(words)).toMatchObject({
    complete: 0,
    unavailableReason: 'nonfinite-statistics'
  });
  expect(() => decodeGPUAnalysisSpacingStatistics(words)).toThrow(/finite/);
});
