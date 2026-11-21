// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeTrajectorySplitOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-split-operation';
import {computeTrajectoryThinningOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-cleaning/cpu-trajectory-thinning-operation';
import {
  OFFSETS,
  POSITIONS,
  SPLIT_TIMES,
  VALUES,
  SPLIT_POSITIONS,
  SPLIT_OUTPUT_TIMES,
  THIN_TIMES,
  THIN_POSITIONS,
  THIN_OUTPUT_TIMES,
  THIN_VALIDITY
} from './cleaning-output-oracles-fixture';

it('split copies the repeated boundary row and both tracks from hand-derived rows', () => {
  const result = computeTrajectorySplitOnCPU(
    {positions: POSITIONS, timestamps: SPLIT_TIMES, trackOffsets: OFFSETS, values: VALUES},
    {mode: 'value-change', segmentCapacity: 3, rowCapacity: 7}
  );
  expect(Array.from(result.outputPositions)).toEqual(SPLIT_POSITIONS);
  expect(Array.from(result.outputTimestamps)).toEqual(SPLIT_OUTPUT_TIMES);
  expect(Array.from(result.rowIds)).toEqual([0, 1, 2, 2, 3, 4, 5]);
});

it('thinning copies the hand-selected rows and writes the original-row selection mask', () => {
  const result = computeTrajectoryThinningOnCPU(
    {
      positions: POSITIONS,
      timestamps: THIN_TIMES,
      trackOffsets: OFFSETS,
      parameters: Float32Array.of(3, 0, 0, 0)
    },
    {mode: 'minimum-time', capacity: 5}
  );
  expect(Array.from(result.sourceSelectionMask)).toEqual(THIN_VALIDITY);
  expect(Array.from(result.outputPositions)).toEqual(THIN_POSITIONS);
  expect(Array.from(result.outputTimestamps)).toEqual(THIN_OUTPUT_TIMES);
});
