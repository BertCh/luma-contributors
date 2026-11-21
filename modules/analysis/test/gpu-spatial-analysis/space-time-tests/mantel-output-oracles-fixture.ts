// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import type {MantelTestOperationProps} from '../../../src/gpu-spatial-analysis/space-time-tests/gpu-mantel-test-operation';
import {getGPUSpaceTimeParameterValues} from '../../../src/gpu-spatial-analysis/space-time-tests/space-time-parameters';
import type {OperationGraphRows} from '../../utils/mixed-execution-test-utils';

export function getMantelProps(stopping: boolean): MantelTestOperationProps {
  return {
    rowCount: 3,
    pairs: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights', distances: 'distances'},
    times: 'times',
    parameters: 'parameters',
    maximumPermutations: 7,
    ...(stopping ? {stopAfterExceedances: 3} : {}),
    output: {
      statistics: 'statistics',
      summary: 'summary',
      ...(stopping ? {permutationCount: 'used'} : {})
    }
  };
}
export function getMantelInputs(stopping: boolean) {
  return {
    offsets: {format: 'uint32' as const, values: new Uint32Array([0, 2, 3, 3])},
    neighbors: {format: 'uint32' as const, values: new Uint32Array([1, 2, 2])},
    weights: {format: 'float32' as const, values: new Float32Array([1, 1, 1])},
    distances: {format: 'float32' as const, values: new Float32Array([1, 2, 4])},
    times: {format: 'float32' as const, values: new Float32Array(stopping ? [2, 2, 2] : [0, 1, 3])},
    parameters: {
      format: 'uint32' as const,
      values: getGPUSpaceTimeParameterValues({seed: 17, permutationCount: 5})
    }
  };
}

export function getMantelTieProps(stopping: boolean): MantelTestOperationProps {
  return {...getMantelProps(stopping), ...(stopping ? {stopAfterExceedances: 1} : {})};
}

export function getMantelTieInputs() {
  return {
    ...getMantelInputs(false),
    offsets: {format: 'uint32' as const, values: new Uint32Array([0, 2, 2, 2])},
    neighbors: {format: 'uint32' as const, values: new Uint32Array([1, 2])},
    weights: {format: 'float32' as const, values: new Float32Array([1, 1])},
    distances: {format: 'float32' as const, values: new Float32Array([1, 2])},
    parameters: {
      format: 'uint32' as const,
      values: getGPUSpaceTimeParameterValues({seed: 0, permutationCount: 5})
    }
  };
}

export function getMantelRescaledTieInputs() {
  return {
    ...getMantelTieInputs(),
    times: {format: 'float32' as const, values: new Float32Array([0, 1 / 8192, 1])}
  };
}

export function getMantelLateAnchorInputs() {
  return {
    ...getMantelInputs(false),
    offsets: {
      format: 'uint32' as const,
      values: Uint32Array.from({length: 301}, (_, row) => (row <= 270 ? 0 : row === 271 ? 2 : 3))
    },
    neighbors: {format: 'uint32' as const, values: new Uint32Array([271, 272, 272])},
    distances: {format: 'float32' as const, values: new Float32Array([1000001, 1000002, 1000001])},
    times: {format: 'float32' as const, values: Float32Array.from({length: 300}, (_, row) => row)},
    parameters: {
      format: 'uint32' as const,
      values: getGPUSpaceTimeParameterValues({seed: 0, permutationCount: 1})
    }
  };
}

export function expectMantelLateAnchorOracle(result: Record<string, OperationGraphRows>): void {
  // First active pair block270 exercises both hierarchical reductions past row256.
  // Independent integer Random123 oracle maps [270,271,272] -> [270,53,13].
  // Pearson([1,2,1],[1,2,1])=1; permuted gaps [217,257,40] give r=.6424277309451744.
  // A wrongly selected zero anchor loses the spatial variance at the million-unit offset.
  const expected = [1, 0.6424277309451744, 0, 0, 0, 0, 0, 0];
  for (const [index, value] of expected.entries()) {
    const actual = result['statistics'][index];
    expect(Number.isFinite(actual)).toBe(true);
    if (value === 0) expect(actual).toBe(0);
    else expect(Math.abs(actual - value)).toBeLessThanOrEqual(2e-6);
  }
  expect(result['summary'][1]).toBe(3);
  expect(result['summary'][6]).toBe(0);
  expect(result['summary'][8]).toBe(0.5);
}

export function expectMantelTieOracle(
  result: Record<string, OperationGraphRows>,
  stopping: boolean
): void {
  // Two nonconstant pairs have Pearson r = sign((x1-x0)*(y1-y0)) exactly.
  // Independent integer shuffle/Pearson oracle.py gives [1,-1,1,-1,1,1].
  expect(Array.from(result['statistics'])).toEqual(
    stopping ? [1, -1, 1, 0, 0, 0, 0, 0] : [1, -1, 1, -1, 1, 1, 0, 0]
  );
  expect(result['summary'][6]).toBe(stopping ? 1 : 3);
  expect(result['summary'][8]).toBe(Math.fround(stopping ? 0.5 : 4 / 6));
  if (stopping) expect(Array.from(result['used'])).toEqual([2]);
}

export function getMantelIdenticalInputs() {
  return {
    ...getMantelInputs(false),
    distances: {format: 'float32' as const, values: new Float32Array([1, 3, 8])},
    parameters: {
      format: 'uint32' as const,
      values: getGPUSpaceTimeParameterValues({seed: 96, permutationCount: 3})
    }
  };
}

export function getMantelThreePairInputs(seed = 0, permutationCount = 5) {
  return {
    ...getMantelInputs(false),
    distances: {format: 'float32' as const, values: new Float32Array([1, 2, 1])},
    times: {format: 'float32' as const, values: new Float32Array([0, 3, 4])},
    parameters: {
      format: 'uint32' as const,
      values: getGPUSpaceTimeParameterValues({seed, permutationCount})
    }
  };
}

export function expectMantelThreePairOracle(
  result: Record<string, OperationGraphRows>,
  stopping: boolean
): void {
  // Integer Pearson moments: n=3, sumX=4,sumY=8, centered denominator sqrt(28).
  // Independent shuffle dot products [12,9,9,11,12,12] give numerators [4,-5,-5,1,4,4].
  const expected = [
    0.7559289460184544,
    -0.944911182523068,
    -0.944911182523068,
    0.1889822365046136,
    0.7559289460184544,
    stopping ? 0 : 0.7559289460184544,
    0,
    0
  ];
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(result['statistics'][index]).toBeNaN();
      continue;
    }
    expect(Number.isFinite(result['statistics'][index])).toBe(true);
    expect(Math.abs(result['statistics'][index] - value)).toBeLessThanOrEqual(2e-6);
  }
  expect(result['summary'][6]).toBe(stopping ? 1 : 2);
  expect(result['summary'][8]).toBe(stopping ? 0.25 : 0.5);
  if (stopping) expect(Array.from(result['used'])).toEqual([4]);
}

export function expectMantelEqualCorrelationOracle(
  result: Record<string, OperationGraphRows>
): void {
  // Seed 1 draws different orders [2,1,0], [0,1,2] but identical exact Pearson moments.
  expect(result['summary'][5]).toBe(0);
  expect(Number.isNaN(result['summary'][11])).toBe(true);
  for (const value of result['statistics'].slice(0, 3)) {
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs(value - 0.7559289460184544)).toBeLessThanOrEqual(2e-6);
  }
}

export function expectMantelIdenticalOracle(result: Record<string, OperationGraphRows>): void {
  // Independent shuffle oracle: all draws order [1,0,2]; Pearson r=21/sqrt(78*6).
  expect(result['summary'][5]).toBe(0);
  expect(Number.isNaN(result['summary'][11])).toBe(true);
  for (const value of result['statistics'].slice(1, 4)) {
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs(value - 0.9707253433941511)).toBeLessThanOrEqual(2e-6);
  }
}
export function expectMantelOracle(
  result: Record<string, OperationGraphRows>,
  stopping: boolean
): void {
  // Pearson: cross product 1, sum squares 14/3 and 2 => r=sqrt(3/28).
  // Other slot literals independently generated by mantel-output-oracles.oracle.py (NumPy).
  const expected = stopping
    ? Array(8).fill(NaN)
    : [
        0.3273268353539886, 0.9819805060619659, -0.3273268353539886, -0.3273268353539886,
        0.3273268353539886, 0.3273268353539886, 0, 0
      ];
  expect(result['statistics'].length).toBe(8);
  for (const [index, value] of expected.entries()) {
    if (Number.isNaN(value)) {
      expect(result['statistics'][index]).toBeNaN();
      continue;
    }
    expect(Number.isFinite(result['statistics'][index])).toBe(true);
    // Centered three-pair f32 correlation: 2e-6 absolute, tail clearing is exact.
    if (value === 0) expect(result['statistics'][index]).toBe(0);
    else expect(Math.abs(result['statistics'][index] - value)).toBeLessThanOrEqual(2e-6);
  }
  expect(result['summary'][1]).toBe(3);
  if (stopping) {
    // A constant time vector has no Pearson coefficient or usable null inference.
    expect(Array.from(result['used'])).toEqual([0]);
    expect(Array.from(result['summary'])).toEqual([NaN, 3, 0, 0, ...Array(8).fill(NaN)]);
  } else {
    expect(Number.isFinite(result['summary'][0])).toBe(true);
    expect(Math.abs(result['summary'][0] - 0.3273268353539886)).toBeLessThanOrEqual(2e-6);
  }
}
