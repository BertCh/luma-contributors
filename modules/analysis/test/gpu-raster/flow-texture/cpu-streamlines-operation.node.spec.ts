// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  CPUStreamlinesOperation,
  computeStreamlinesOnCPU
} from '../../../src/raster-analysis/flow-texture/cpu-streamlines-operation';
import {
  GPUStreamlinesOperation,
  type StreamlinesOperationProps
} from '../../../src/raster-analysis/flow-texture/gpu-streamlines-operation';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues,
  STREAMLINES_SEED_PURPOSE
} from '../../../src/raster-analysis/flow-texture/streamlines-parameters';
import {getPhilox4x32, getPhiloxUnitFloat} from '../../../src/analysis-core/kernels/philox-random';

const options = {
  fieldWidth: 1,
  fieldHeight: 1,
  gridWidth: 1,
  gridHeight: 1,
  seedColumns: 4,
  seedRows: 1,
  stepsPerDirection: 2,
  lineCapacity: 4,
  pointCapacity: 20
};
const parameters = getGPUStreamlinesParameterValues({
  fieldPlacement: {origin: [0, 0], cellSize: [10, 10]},
  gridPlacement: {origin: [0, 0], cellSize: [10, 10]},
  stepLength: 0.125
});
const uint32Parameters = getGPUStreamlinesUint32ParameterValues({seed: 7});

it('selects the highest priority of four colliding horizontal traces and integrates the closed form', () => {
  const random = Array.from({length: 4}, (_, seed) =>
    getPhilox4x32([seed, 0, 0, 0], [7, STREAMLINES_SEED_PURPOSE])
  );
  const priority = random.map((words, seed) => (words[2] >>> 16) * 65536 + 65535 - seed);
  const winner = priority.indexOf(Math.max(...priority));
  const result = computeStreamlinesOnCPU(
    {velocities: Float32Array.of(3, 0), parameters, uint32Parameters},
    options
  );
  expect(result.count[0]).toBe(1);
  expect(result.requiredCount[0]).toBe(1);
  expect(result.ids[0]).toBe(winner);
  expect(Array.from(result.lineOffsets)).toEqual([0, 5, 5, 5, 5]);
  expect(result.positionCount[0]).toBe(5);
  const startX = (winner + getPhiloxUnitFloat(random[winner][0])) * 2.5;
  const startY = getPhiloxUnitFloat(random[winner][1]) * 10;
  for (let point = 0; point < 5; point++) {
    expect(result.positions[point * 2]).toBe(Math.fround(startX + (point - 2) * 0.125));
    expect(result.positions[point * 2 + 1]).toBe(Math.fround(startY));
  }
  expect(result.overflow[0]).toBe(0);
  expect(result.iterationLimitReached[0]).toBe(0);
  const overflow = computeStreamlinesOnCPU(
    {velocities: Float32Array.of(3, 0), parameters, uint32Parameters},
    {...options, pointCapacity: 1}
  );
  expect(overflow.count[0]).toBe(0);
  expect(overflow.requiredCount[0]).toBe(1);
  expect(overflow.overflow[0]).toBe(1);
  expect(Array.from(overflow.lineOffsets)).toEqual([0, 0, 0, 0, 0]);
});

it('rejects slow, missing and too-short traces, checks arrays and preserves inputs', () => {
  for (const velocities of [Float32Array.of(0, 0), Float32Array.of(NaN, 1)]) {
    const result = computeStreamlinesOnCPU({velocities, parameters, uint32Parameters}, options);
    expect(result.count[0]).toBe(0);
    expect(result.requiredCount[0]).toBe(0);
    expect(result.overflow[0]).toBe(0);
  }
  const result = computeStreamlinesOnCPU(
    {
      velocities: Float32Array.of(3, 0),
      parameters,
      uint32Parameters: getGPUStreamlinesUint32ParameterValues({minimumPoints: 6})
    },
    options
  );
  expect(result.requiredCount[0]).toBe(0);
  expect(() =>
    computeStreamlinesOnCPU(
      {velocities: new Float32Array(1), parameters, uint32Parameters},
      options
    )
  ).toThrow(/velocities/);
  expect(() =>
    computeStreamlinesOnCPU(
      {velocities: Float32Array.of(1, 0), parameters: new Float32Array(1), uint32Parameters},
      options
    )
  ).toThrow(/parameters/);
});

it('validates CPU dimensions and both graph output contracts at construction', () => {
  const props: StreamlinesOperationProps = {
    ...options,
    id: 'stream-test',
    velocities: 'velocities',
    parameters: 'parameters',
    uint32Parameters: 'uint32Parameters',
    output: {
      lines: {ids: 'ids', status: {count: 'count', overflow: 'overflow'}},
      positions: 'positions',
      lineOffsets: 'lineOffsets',
      positionCount: 'positionCount'
    }
  };
  expect(() => new CPUStreamlinesOperation({...props, seedColumns: 65536})).toThrow(
    /stream-test.*seedColumns/
  );
  for (const Operation of [GPUStreamlinesOperation, CPUStreamlinesOperation])
    expect(() => new Operation({...props, pointCapacity: 0})).toThrow(/stream-test/);
});

it('follows the analytic parabola of a bilinearly sampled field (1, x)', () => {
  const sceneOptions = {
    ...options,
    fieldWidth: 4,
    fieldHeight: 4,
    seedColumns: 1,
    stepsPerDirection: 10,
    pointCapacity: 21
  };
  const result = computeStreamlinesOnCPU(
    {
      velocities: Float32Array.from({length: 32}, (_, component) =>
        component % 2 === 0 ? 1 : (Math.floor(component / 2) % 4) + 0.5
      ),
      parameters: getGPUStreamlinesParameterValues({
        fieldPlacement: {origin: [0, 0], cellSize: [1, 1]},
        gridPlacement: {origin: [0, 0], cellSize: [4, 4]},
        stepLength: 0.05
      }),
      uint32Parameters: getGPUStreamlinesUint32ParameterValues({seed: 3})
    },
    sceneOptions
  );
  expect(result.count[0]).toBe(1);
  expect(result.positionCount[0]).toBe(21);
  // Integral curve y - x²/2 is constant. RK2 has O(stepLength²) global error; the bound
  // below also includes one float32 output rounding, and detects nearest-neighbor sampling.
  const centerX = result.positions[20];
  const centerY = result.positions[21];
  const invariant = centerY - (centerX * centerX) / 2;
  for (let point = 0; point < 21; point++) {
    const horizontal = result.positions[point * 2];
    const vertical = result.positions[point * 2 + 1];
    expect(Math.abs(vertical - (horizontal * horizontal) / 2 - invariant)).toBeLessThan(1e-5);
    if (point > 0) expect(horizontal).toBeGreaterThan(result.positions[point * 2 - 2]);
  }
  // Negative control: first-order Euler on this scene misses the invariant by over 1e-3.
  let maximumEulerError = 0;
  for (const sign of [-1, 1]) {
    let horizontal = centerX;
    let vertical = centerY;
    for (let step = 0; step < 10; step++) {
      const speed = Math.hypot(1, horizontal);
      vertical += (sign * Math.fround(0.05) * horizontal) / speed;
      horizontal += (sign * Math.fround(0.05)) / speed;
      maximumEulerError = Math.max(
        maximumEulerError,
        Math.abs(vertical - (horizontal * horizontal) / 2 - invariant)
      );
    }
  }
  expect(maximumEulerError).toBeGreaterThan(1e-3);
});

it('rejects a seed whose final float32 x lies on the exclusive upper grid boundary', () => {
  const result = computeStreamlinesOnCPU(
    {
      velocities: Float32Array.of(1, 0),
      parameters: getGPUStreamlinesParameterValues({
        fieldPlacement: {origin: [16777216, 0], cellSize: [2, 2]},
        gridPlacement: {origin: [16777216, 0], cellSize: [2, 2]},
        stepLength: 0.125
      }),
      uint32Parameters: getGPUStreamlinesUint32ParameterValues({seed: 0, minimumPoints: 1})
    },
    {...options, seedColumns: 1, stepsPerDirection: 1, lineCapacity: 1, pointCapacity: 3}
  );
  // Philox x jitter for seed zero is above 0.5. Adding it to 2^24 rounds to 2^24+2,
  // the excluded upper endpoint of this two-unit cell: no line or candidate can be emitted.
  expect(Array.from(result.count)).toEqual([0]);
  expect(Array.from(result.requiredCount)).toEqual([0]);
  expect(Array.from(result.positionCount)).toEqual([0]);
  expect(result.candidatePositions.every(Number.isNaN)).toBe(true);
  expect(Array.from(result.candidateSpans)).toEqual([0, 0, 0, 0]);
});

it('publishes three horizontal candidate positions and the four diagnostic span words', () => {
  const random = getPhilox4x32([0, 0, 0, 0], [3, STREAMLINES_SEED_PURPOSE]);
  const startX = getPhiloxUnitFloat(random[0]) * 10;
  const startY = getPhiloxUnitFloat(random[1]) * 10;
  const result = computeStreamlinesOnCPU(
    {
      velocities: Float32Array.of(1, 0),
      parameters,
      uint32Parameters: getGPUStreamlinesUint32ParameterValues({seed: 3})
    },
    {...options, seedColumns: 1, stepsPerDirection: 1, lineCapacity: 1, pointCapacity: 3}
  );
  expect(Array.from(result.candidatePositions)).toEqual(
    Array.from(Float32Array.of(startX - 0.125, startY, startX, startY, startX + 0.125, startY))
  );
  expect(Array.from(result.candidateSpans)).toEqual([1, 1, 1, 1]);
  expect(Array.from(result.positions)).toEqual(Array.from(result.candidatePositions));
});

it('truncates a step that rounds onto the upper boundary despite a downward-rounded reciprocal', () => {
  const result = computeStreamlinesOnCPU(
    {
      velocities: Float32Array.of(1, 0),
      parameters: getGPUStreamlinesParameterValues({
        fieldPlacement: {origin: [0, 0], cellSize: [25, 25]},
        gridPlacement: {origin: [0, 0], cellSize: [25, 25]},
        stepLength: 7.925207614898682
      }),
      uint32Parameters: getGPUStreamlinesUint32ParameterValues({seed: 3, minimumPoints: 1})
    },
    {...options, seedColumns: 1, stepsPerDirection: 1, lineCapacity: 1, pointCapacity: 3}
  );
  // The forward point is 24.999999701976776 in float64 and 25 in float32. Although
  // 25 * float32(1/25) is below one in float64, the shader's rounded product is exactly one.
  expect(Array.from(result.candidateSpans)).toEqual([1, 0, 1, 0]);
  expect(result.positionCount[0]).toBe(2);
  expect(result.candidatePositions[4]).toBeNaN();
  for (let point = 0; point < 2; point++) expect(result.positions[point * 2]).toBeLessThan(25);
});
