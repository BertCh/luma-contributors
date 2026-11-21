// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPULagrangianFlowMapParameterValues} from '../../../src/raster-analysis/lagrangian-diagnostics/index';

const settings = {startTime: 0, timeStep: 1, sliceStartTime: 0, sliceEndTime: 1};

// RT-129: the flow map documents signed cell sizes (as GPUParticleAdvection and GPUStreamlines
// do), so a north-up field with its top edge as origin and a negative cell height is accepted.
it('RT-129 the flow map packer accepts a north-up field with a negative cell height', () => {
  const values = getGPULagrangianFlowMapParameterValues({
    ...settings,
    fieldPlacement: {origin: [0, 4], cellSize: [1, -1]}
  });
  expect(Array.from(values)).toEqual([0, 4, 1, -1, 0, 1, 0, 1]);
  const mirrored = getGPULagrangianFlowMapParameterValues({
    ...settings,
    fieldPlacement: {origin: [4, 0], cellSize: [-1, 1]}
  });
  expect(Array.from(mirrored.subarray(0, 4))).toEqual([4, 0, -1, 1]);
});

it('RT-129 the flow map packer still rejects zero and non-finite cell sizes', () => {
  for (const [x, y, cellWidth, cellHeight] of [
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, Number.NaN],
    [0, 0, Number.POSITIVE_INFINITY, 1],
    // Rounds to zero in float32.
    [0, 0, 1, 1e-50]
  ] as const) {
    const fieldPlacement = {origin: [x, y], cellSize: [cellWidth, cellHeight]} as const;
    expect(() => getGPULagrangianFlowMapParameterValues({...settings, fieldPlacement})).toThrow(
      /^GPULagrangianFlowMap fieldPlacement cell(Width|Height) must be a finite number other than 0/
    );
  }
});
