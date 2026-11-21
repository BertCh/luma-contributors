// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeStreamlinesOnCPU} from '../../../src/raster-analysis/flow-texture/cpu-streamlines-operation';
import {
  getGPUStreamlinesParameterValues,
  getGPUStreamlinesUint32ParameterValues
} from '../../../src/raster-analysis/flow-texture/streamlines-parameters';

it('B1 rejects an interior seed that rounds onto the excluded field edge', () => {
  const result = computeStreamlinesOnCPU(
    {
      velocities: Float32Array.of(1, 0),
      parameters: getGPUStreamlinesParameterValues({
        fieldPlacement: {origin: [1000000, 0], cellSize: [1, 1]},
        gridPlacement: {origin: [1000000, 0], cellSize: [1, 1]},
        stepLength: 0
      }),
      uint32Parameters: getGPUStreamlinesUint32ParameterValues({seed: 16, minimumPoints: 1})
    },
    {
      fieldWidth: 1,
      fieldHeight: 1,
      gridWidth: 2,
      gridHeight: 1,
      seedColumns: 2,
      seedRows: 1,
      stepsPerDirection: 1,
      lineCapacity: 2,
      pointCapacity: 6
    }
  );
  // Philox(seed16,purpose2) first jitter .9903802275657654 is interior in binary64,
  // but 1000000+jitter rounds to 1000001 in float32. The second seed is outside the field.
  expect(Array.from(result.count)).toEqual([0]);
  expect(Array.from(result.candidatePositions).every(Number.isNaN)).toBe(true);
});

it('B1 published float32 streamline seeds and landings stay inside velocity support', () => {
  const result = computeStreamlinesOnCPU(
    {
      velocities: Float32Array.of(1, 0),
      parameters: getGPUStreamlinesParameterValues({
        fieldPlacement: {origin: [1000000, 0], cellSize: [1, 1]},
        gridPlacement: {origin: [1000000, 0], cellSize: [1, 1]},
        stepLength: 7 / 16
      }),
      uint32Parameters: getGPUStreamlinesUint32ParameterValues({seed: 0, minimumPoints: 3})
    },
    {
      fieldWidth: 1,
      fieldHeight: 1,
      gridWidth: 2,
      gridHeight: 1,
      seedColumns: 2,
      seedRows: 1,
      stepsPerDirection: 1,
      lineCapacity: 2,
      pointCapacity: 6
    }
  );
  // At 1e6, one float32 ULP is 1/16. A mathematically interior endpoint can round
  // to 1000001, the excluded edge. Neither jittered seed can supply 3 supported published points.
  expect(result.count[0]).toBe(0);
  for (let point = 0; point < result.candidatePositions.length / 2; point++) {
    const horizontal = result.candidatePositions[2 * point];
    if (Number.isNaN(horizontal)) continue;
    expect(Number.isFinite(horizontal)).toBe(true);
    expect(horizontal >= 1000000 && horizontal < 1000001, `candidate ${point}: ${horizontal}`).toBe(
      true
    );
  }
});

it('streamlines retain a supported below-threshold terminal landing, then stop', () => {
  const result = computeStreamlinesOnCPU(
    {
      velocities: Float32Array.of(1, 0, 1, 0, 0, 0, 0, 0),
      parameters: getGPUStreamlinesParameterValues({
        fieldPlacement: {origin: [0, 0], cellSize: [1, 1]},
        gridPlacement: {origin: [0, 0], cellSize: [1, 1]},
        stepLength: 1.375,
        minimumSpeed: 0.75
      }),
      uint32Parameters: getGPUStreamlinesUint32ParameterValues({seed: 6, minimumPoints: 2})
    },
    {
      fieldWidth: 4,
      fieldHeight: 1,
      gridWidth: 4,
      gridHeight: 1,
      seedColumns: 1,
      seedRows: 1,
      stepsPerDirection: 2,
      lineCapacity: 1,
      pointCapacity: 5
    }
  );
  // Philox fixes x0=0.4723553657531738. Midpoint 1.159855... has speed 1;
  // landing x0+1.375=1.847355... has speed 2.5-x=0.6526446 < .75.
  // Existing streamline semantics retain that supported endpoint, but cannot start another step there.
  expect(result.count[0]).toBe(1);
  expect(result.positionCount[0]).toBe(2);
  expect(Array.from(result.positions.subarray(0, 4))).toEqual([
    0.47235536575317383, 0.45453524589538574, 1.8473553657531738, 0.45453524589538574
  ]);
  expect(Number.isFinite(result.positions[1])).toBe(true);
});
