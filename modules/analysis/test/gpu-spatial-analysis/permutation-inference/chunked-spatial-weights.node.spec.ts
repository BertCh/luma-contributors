// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {planGPUSpatialWeightsChunks} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {getGPUSpatialWeightsNeighborLoopWGSL} from '../../../src/gpu-spatial-analysis/spatial-weights/spatial-weights';

test('chunk plan keeps each chunk within the slot limit and covers every row', () => {
  const counts = [3, 5, 2, 7, 1, 1, 9, 4];
  const plan = planGPUSpatialWeightsChunks(counts, 10);
  expect(plan.map(chunk => chunk.rowStart)).toEqual([0, 3, 6, 7]);
  expect(plan.reduce((total, chunk) => total + chunk.rowCount, 0)).toBe(counts.length);
  for (const chunk of plan) expect(chunk.slotCount).toBeLessThanOrEqual(10);
  expect(plan.reduce((total, chunk) => total + chunk.slotCount, 0)).toBe(32);
});

test('a row above the limit gets its own chunk', () => {
  const plan = planGPUSpatialWeightsChunks([2, 50, 2], 10);
  expect(plan.map(chunk => chunk.rowCount)).toEqual([1, 1, 1]);
  expect(plan[1].slotCount).toBe(50);
});

test('the neighbor loop addresses chunk offsets by the row minus the chunk start only when asked', () => {
  const whole = getGPUSpatialWeightsNeighborLoopWGSL('true', 'x');
  expect(whole).toContain('offsets[offsetsOffset + activeRow]');
  const chunked = getGPUSpatialWeightsNeighborLoopWGSL('true', 'x', 'index', 'WEIGHTS_ROW_START');
  expect(chunked).toContain('offsets[offsetsOffset + activeRow - (WEIGHTS_ROW_START)]');
  expect(chunked).toContain('activeRow - (WEIGHTS_ROW_START) + 1u');
  expect(chunked).toContain('neighbor == activeRow');
});
