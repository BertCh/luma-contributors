// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import {
  computeColumnQuantilesOnCPU,
  CPUColumnQuantilesOperation
} from '../../../src/dataframe-analysis/column-classification/cpu-column-quantiles-operation';
import {
  GPUColumnQuantilesOperation,
  type ColumnQuantilesOperationProps
} from '../../../src/dataframe-analysis/column-classification/gpu-column-quantiles-operation';
import {
  getGPUColumnQuantilesParameterValues,
  getGPUColumnQuantilesRationalProbabilityValues,
  type GPUColumnQuantileInterpolation
} from '../../../src/dataframe-analysis/column-classification/column-quantiles-parameters';

const PROPS: ColumnQuantilesOperationProps = {
  id: 'quantiles-check',
  values: 'values',
  parameters: 'parameters',
  rowCount: 8,
  quantileCount: 5,
  output: {quantiles: 'quantiles', validCount: 'validCount'}
};

it('matches hand-selected order statistics and inclusive percentile bounds', () => {
  // Surviving sorted sample: [-4, 0, 2, 8]. Quartile ranks: 0, .75, 1.5, 2.25, 3.
  const answers: Record<GPUColumnQuantileInterpolation, number[]> = {
    lower: [-4, -4, 0, 2, 8],
    higher: [-4, 0, 2, 8, 8],
    nearest: [-4, 0, 2, 2, 8],
    linear: [-4, -1, 1, 3.5, 8],
    midpoint: [-4, -2, 1, 5, 8]
  };
  for (const interpolation of Object.keys(answers) as GPUColumnQuantileInterpolation[]) {
    const result = computeColumnQuantilesOnCPU(
      {
        values: new Float32Array([8, -4, -0, 2, 100, NaN, Infinity, -Infinity]),
        mask: new Uint32Array([1, 1, 1, 1, 0, 1, 1, 1]),
        parameters: getGPUColumnQuantilesParameterValues({
          quantiles: [0, 0.25, 0.5, 0.75, 1],
          interpolation,
          filterRange: [0.25, 0.75]
        })
      },
      5
    );
    expect(Array.from(result.quantiles)).toEqual(answers[interpolation]);
    expect(Array.from(result.validCount)).toEqual([4]);
    expect(Array.from(result.filterBounds)).toEqual([0, 2]);
    expect(Array.from(result.outputMask)).toEqual([0, 0, 1, 1, 0, 0, 0, 0]);
  }
});

it('delegates GPU limits to contributor construction and drops that GPU-only cap on CPU', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {
      values: {format: 'float32', length: 8},
      parameters: {format: 'float32', length: 69},
      quantiles: {format: 'float32', length: 65},
      validCount: {format: 'uint32', length: 1}
    });
    const props = {...PROPS, quantileCount: 65};
    expect(() => new GPUColumnQuantilesOperation(props).createContributor(getView)).toThrow(
      /quantileCount/
    );
    expect(() => new CPUColumnQuantilesOperation(props)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('uses exact rational ranks and half-even ties, independent of float probabilities', () => {
  const result = computeColumnQuantilesOnCPU(
    {
      values: new Float32Array([10, 20, 30, 40]),
      parameters: getGPUColumnQuantilesParameterValues({
        quantiles: [NaN, NaN, NaN, NaN],
        interpolation: 'nearest'
      }),
      rationalProbabilities: new Uint32Array([1, 6, 1, 2, 5, 6, 2, 1])
    },
    4
  );
  expect(Array.from(result.quantiles)).toEqual([10, 30, 30, NaN]);
  const exact = computeColumnQuantilesOnCPU(
    {
      values: Float32Array.from({length: 101}, (_, row) => row),
      parameters: getGPUColumnQuantilesParameterValues({quantiles: [0], interpolation: 'higher'}),
      rationalProbabilities: getGPUColumnQuantilesRationalProbabilityValues([[55, 100]])
    },
    1
  );
  expect(exact.quantiles[0]).toBe(55);
});

it('handles empty samples, invalid probabilities, and large finite interpolation', () => {
  const parameters = getGPUColumnQuantilesParameterValues({quantiles: [-1, NaN, 2]});
  const empty = computeColumnQuantilesOnCPU(
    {values: new Float32Array([NaN, Infinity]), parameters},
    3
  );
  expect(Array.from(empty.quantiles)).toEqual([NaN, NaN, NaN]);
  expect(Array.from(empty.filterBounds)).toEqual([NaN, NaN]);
  expect(Array.from(empty.outputMask)).toEqual([0, 0]);
  expect(empty.validCount[0]).toBe(0);
  expect(
    Array.from(
      computeColumnQuantilesOnCPU({values: new Float32Array([1]), parameters}, 3).quantiles
    )
  ).toEqual([NaN, NaN, NaN]);
  expect(
    computeColumnQuantilesOnCPU(
      {
        values: new Float32Array([-3e38, 3e38]),
        parameters: getGPUColumnQuantilesParameterValues({quantiles: [0.5]})
      },
      1
    ).quantiles[0]
  ).toBe(0);
});

it('retains the packed probability in float64 instead of rounding its rank to float32', () => {
  // The packed .07 is slightly above 7/100; multiplying by 100 in float64 is above 7,
  // whereas the GPU rounds that product back to exactly 7 before taking ceil.
  const result = computeColumnQuantilesOnCPU(
    {
      values: Float32Array.from({length: 101}, (_, row) => row),
      parameters: getGPUColumnQuantilesParameterValues({quantiles: [0.07], interpolation: 'higher'})
    },
    1
  );
  expect(result.quantiles[0]).toBe(8);
  const filtered = computeColumnQuantilesOnCPU(
    {
      values: Float32Array.from({length: 100}, (_, row) => row),
      parameters: getGPUColumnQuantilesParameterValues({quantiles: [0.5], filterRange: [0, 0.07]})
    },
    1
  );
  expect(Array.from(filtered.filterBounds)).toEqual([0, 7]);
});

it('distinguishes all interpolation methods on both packed probability paths', () => {
  // Same masked sample as the comparison spec: sorted [-4, 0, 2, 8], fractional quarter ranks.
  const expected: Record<GPUColumnQuantileInterpolation, number[]> = {
    lower: [-4, -4, 0, 2, 8, NaN],
    higher: [-4, 0, 2, 8, 8, NaN],
    nearest: [-4, 0, 2, 2, 8, NaN],
    linear: [-4, -1, 1, 3.5, 8, NaN],
    midpoint: [-4, -2, 1, 5, 8, NaN]
  };
  for (const interpolation of Object.keys(expected) as GPUColumnQuantileInterpolation[]) {
    for (const rational of [false, true]) {
      const result = computeColumnQuantilesOnCPU(
        {
          values: new Float32Array([9, -4, 0, 2, 8, 100, NaN, Infinity]),
          mask: new Uint32Array([0, 1, 1, 1, 1, 0, 1, 1]),
          parameters: getGPUColumnQuantilesParameterValues({
            quantiles: [0, 0.25, 0.5, 0.75, 1, NaN],
            interpolation,
            filterRange: [0.25, 0.75]
          }),
          rationalProbabilities: rational
            ? getGPUColumnQuantilesRationalProbabilityValues(
                [
                  [0, 1],
                  [1, 4],
                  [1, 2],
                  [3, 4],
                  [1, 1]
                ],
                6
              )
            : undefined
        },
        6
      );
      expect(Array.from(result.quantiles)).toEqual(expected[interpolation]);
      expect(result.validCount[0]).toBe(4);
      expect(Array.from(result.filterBounds)).toEqual([0, 2]);
      expect(Array.from(result.outputMask)).toEqual([0, 0, 1, 1, 0, 0, 0, 0]);
    }
  }
});

it('rejects inconsistent arrays and validates operation construction', () => {
  const input = {
    values: new Float32Array(2),
    parameters: getGPUColumnQuantilesParameterValues({quantiles: [0.5]})
  };
  expect(() => computeColumnQuantilesOnCPU({...input, mask: new Uint32Array(1)}, 1)).toThrow(
    'mask'
  );
  expect(() => computeColumnQuantilesOnCPU({...input, parameters: new Float32Array(4)}, 1)).toThrow(
    'parameters'
  );
  expect(() =>
    computeColumnQuantilesOnCPU({...input, rationalProbabilities: new Uint32Array(1)}, 1)
  ).toThrow('rationalProbabilities');
  for (const Operation of [GPUColumnQuantilesOperation, CPUColumnQuantilesOperation]) {
    expect(() => new Operation({...PROPS, quantileCount: 0})).toThrow(/^quantiles-check/);
    expect(new Operation(PROPS).id).toBe('quantiles-check');
  }
  expect(() => new CPUColumnQuantilesOperation({...PROPS, rowCount: 0})).toThrow(
    /^quantiles-check.*rowCount/
  );
});
