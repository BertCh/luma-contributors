// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {runInNewContext} from 'node:vm';
import {
  calibrateRasterOperationBand,
  validateRasterOperationBand,
  type CPURasterOperationBand
} from '../../../src/raster-analysis/raster-grid/raster-operation-band';
import {computeTerrainFlowOnCPU} from '../../../src/gpu-terrain/hydrology/cpu-terrain-flow-operation';
import {getGPUTerrainFlowParameterValues} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';

// @ts-expect-error URT-13 declared uint32 storage requires Uint32Array.
const mismatchedBand: CPURasterOperationBand = {
  id: 'dem',
  format: 'uint32',
  values: Float32Array.of(3.25)
};
void mismatchedBand;

const GRID = {
  width: 3,
  height: 1,
  affine: [1, 0, 0, 0, -1, 0] as const,
  pixelInterpretation: 'area' as const
};

it('URT-13 rejects mismatched CPU elevation storage before conversion', () => {
  for (const [format, values, required] of [
    ['uint32', Float32Array.of(3.25, 2.25, 1.25), 'Uint32Array'],
    ['sint32', Uint32Array.of(3, 2, 1), 'Int32Array'],
    ['float32', Int32Array.of(3, 2, 1), 'Float32Array']
  ] as const) {
    // JavaScript callers and casts still need the construction-time guard.
    const elevation = {id: 'dem', format, values};
    expect(() =>
      Reflect.apply(computeTerrainFlowOnCPU, undefined, [
        {elevation, parameters: getGPUTerrainFlowParameterValues({}, GRID)},
        {grid: GRID}
      ])
    ).toThrow(new RegExp(`^computeTerrainFlowOnCPU.*elevation.values.*${required}.*${format}`));
  }
});

it('URT-13 retains correctly typed subarrays at nonzero offsets', () => {
  for (const elevation of [
    {
      id: 'dem',
      format: 'float32' as const,
      values: Float32Array.of(99, 3.25, 2.25, 1.25, 99).subarray(1, 4)
    },
    {id: 'dem', format: 'sint32' as const, values: Int32Array.of(99, 3, 2, 1, 99).subarray(1, 4)},
    {id: 'dem', format: 'uint32' as const, values: Uint32Array.of(99, 3, 2, 1, 99).subarray(1, 4)}
  ]) {
    expect(elevation.values.byteOffset).toBe(4);
    validateRasterOperationBand('typed-subarray', elevation, GRID);
    expect(Array.from(calibrateRasterOperationBand(elevation, 3))).toEqual(
      Array.from(elevation.values)
    );
  }
});

it('B1 accepts cross-realm native subarrays without copying or changing their byte offset', () => {
  const values = runInNewContext('Int32Array.of(91,9,4,2).subarray(1)') as Int32Array;
  expect(values.byteOffset).toBe(4);
  const elevation: CPURasterOperationBand = {id: 'dem', format: 'sint32', values};
  validateRasterOperationBand('cross-realm', elevation, GRID);
  const result = computeTerrainFlowOnCPU(
    {elevation, parameters: getGPUTerrainFlowParameterValues({}, GRID)},
    {grid: GRID}
  );
  expect(Array.from(result.flowDirections)).toEqual([1, 1, 0]);
  expect(elevation.values).toBe(values);
  expect(elevation.values.byteOffset).toBe(4);
});

it('B1 rejects an array that spoofs its typed-array name', () => {
  const values = new Float32Array(3);
  Object.defineProperty(values, Symbol.toStringTag, {value: 'Int32Array'});
  expect(() =>
    Reflect.apply(validateRasterOperationBand, undefined, [
      'spoof',
      {id: 'dem', format: 'sint32', values},
      GRID
    ])
  ).toThrow(/must be Int32Array/);
});

it('S3 names the public elevation or friction band in shared validation errors', () => {
  for (const name of ['elevation', 'friction']) {
    const malformed = {
      id: 'dem',
      format: 'sint32',
      values: new Float32Array(3)
    };
    expect(() =>
      Reflect.apply(validateRasterOperationBand, undefined, ['named-band', malformed, GRID, name])
    ).toThrow(new RegExp(`^named-band.*${name}.values must be Int32Array`));
  }
});
