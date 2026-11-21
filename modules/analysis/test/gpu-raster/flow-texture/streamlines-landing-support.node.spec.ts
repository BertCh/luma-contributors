// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeStreamlinesOnCPU} from '../../../src/raster-analysis/flow-texture/cpu-streamlines-operation';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../../src/raster-analysis/flow-texture/streamlines-parameters';

// RT-124: one unit field cell [0, 1) x [0, 1) with velocity [1, 0] inside a two-column occupancy
// grid of cell width 2, [0, 4) x [0, 1), with two seed columns. Seed 8 jitters seed 0 to (0.6033, 0.2658); a 0.75
// step has its midpoint at 0.978 (inside the field) and lands at 1.353, outside the field but
// inside the grid. The landing has no velocity support, so the forward step must end the line;
// the backward step lands at -0.147, outside both. Seed 1 starts outside the field. Each seed
// alone is one point, below minimumPoints 2: no line.
const options = {
  fieldWidth: 1,
  fieldHeight: 1,
  gridWidth: 2,
  gridHeight: 1,
  seedColumns: 2,
  seedRows: 1,
  stepsPerDirection: 1,
  lineCapacity: 2,
  pointCapacity: 8
};

it('RT-124 a second-order step that lands outside the velocity field ends the line', () => {
  const result = computeStreamlinesOnCPU(
    {
      velocities: Float32Array.of(1, 0),
      parameters: getGPUStreamlinesParameterValues({
        fieldPlacement: {origin: [0, 0], cellSize: [1, 1]},
        gridPlacement: {origin: [0, 0], cellSize: [2, 1]},
        stepLength: 0.75
      }),
      uint32Parameters: getGPUStreamlinesUint32ParameterValues({seed: 8, minimumPoints: 2})
    },
    options
  );
  for (let point = 0; point < result.positionCount[0]; point++) {
    const x = result.positions[point * 2];
    expect(Number.isFinite(x)).toBe(true);
    expect(x >= 0 && x < 1, `point ${point} x = ${x} lies in the field`).toBe(true);
  }
  expect(result.count[0]).toBe(0);
  expect(result.requiredCount[0]).toBe(0);
});
