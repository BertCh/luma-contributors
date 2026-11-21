// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createQuantilesSnapshotFixture} from './quantiles-snapshot.test-utils';
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
import {
  getColumnQuantilesParameterValues,
  getColumnQuantilesRationalProbabilityValues
} from '../../../src/dataframe-analysis/index';

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

it('rounds packed float ranks to float32 before selecting order statistics', () => {
  // The packed .07 is slightly above 7/100; f32(100 * f32(.07)) is exactly 7.
  // Both forms take ceil only after that float32 product.
  const result = computeColumnQuantilesOnCPU(
    {
      values: Float32Array.from({length: 101}, (_, row) => row),
      parameters: getGPUColumnQuantilesParameterValues({quantiles: [0.07], interpolation: 'higher'})
    },
    1
  );
  expect(result.quantiles[0]).toBe(7);
  const filtered = computeColumnQuantilesOnCPU(
    {
      values: Float32Array.from({length: 100}, (_, row) => row),
      parameters: getGPUColumnQuantilesParameterValues({quantiles: [0.5], filterRange: [0, 0.07]})
    },
    1
  );
  // Upper filter rank is ceil(f32(100 * f32(.07))) - 1 = 6.
  expect(Array.from(filtered.filterBounds)).toEqual([0, 6]);
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

// DPC-117. Without an outputMask the GPU form never used rowCount, so a declared rowCount 3
// for a length-2 values input lowered silently; the CPU form throws for the same declaration.
it('rejects a declared rowCount that differs from the values length on the GPU form', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {
      values: {format: 'float32', length: 2},
      parameters: {format: 'float32', length: 69},
      quantiles: {format: 'float32', length: 1},
      validCount: {format: 'uint32', length: 1}
    });
    const props = {...PROPS, rowCount: 3, quantileCount: 1};
    expect(() => new GPUColumnQuantilesOperation(props).createContributor(getView)).toThrow(
      /^quantiles-check.*values length 2 must equal rowCount 3/
    );
    const matching = {...props, rowCount: 2};
    expect(() =>
      new GPUColumnQuantilesOperation(matching).createContributor(getView)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('TWN-41 pins packed 0.7 quantile and filter ranks independently', () => {
  // f32(10 * f32(.7)) = 7: row 7 is both the lower quantile and lower filter bound.
  const parameters = getGPUColumnQuantilesParameterValues({
    quantiles: [0.7],
    interpolation: 'lower',
    filterRange: [0.7, 1]
  });
  const quantile = computeColumnQuantilesOnCPU(
    {values: Float32Array.from({length: 11}, (_, row) => row), parameters},
    1
  );
  expect(Array.from(quantile.quantiles)).toEqual([7]);
  const filtered = computeColumnQuantilesOnCPU(
    {values: Float32Array.from({length: 10}, (_, row) => row), parameters},
    1
  );
  expect(Array.from(filtered.filterBounds)).toEqual([7, 9]);
  expect(Array.from(filtered.outputMask)).toEqual([0, 0, 0, 0, 0, 0, 0, 1, 1, 1]);
});

it('TWN-14 packs 65 CPU probabilities through named inputs on both routes', () => {
  const floatParameters = getColumnQuantilesParameterValues({quantiles: Array(65).fill(0.5)});
  const rationalProbabilities = getColumnQuantilesRationalProbabilityValues(
    Array.from({length: 65}, () => [1, 2] as const)
  );
  for (const ratios of [undefined, rationalProbabilities]) {
    // Rational wiring must override unusable float slots, rather than pass via float fallback.
    const parameters = ratios
      ? getColumnQuantilesParameterValues({quantiles: Array(65).fill(NaN)})
      : floatParameters;
    const result = computeColumnQuantilesOnCPU(
      {values: Float32Array.of(0, 2), parameters, rationalProbabilities: ratios},
      65
    );
    // Every half-probability is the midpoint of [0, 2], exactly 1.
    expect(Array.from(result.quantiles)).toEqual(Array(65).fill(1));
    const rows = (values: Float32Array | Uint32Array) => ({
      format: values instanceof Float32Array ? ('float32' as const) : ('uint32' as const),
      length: values.length,
      chunks: [{values, length: values.length}],
      pack: () => values
    });
    const operation = new CPUColumnQuantilesOperation({
      ...PROPS,
      rowCount: 2,
      quantileCount: 65,
      rationalProbabilities: ratios ? 'ratios' : undefined
    });
    const outputs = operation.execute({
      inputs: {
        values: rows(Float32Array.of(0, 2)),
        parameters: rows(parameters),
        ...(ratios ? {rationalProbabilities: rows(ratios)} : {})
      }
    }) as Record<string, Float32Array | Uint32Array>;
    expect(Array.from(outputs['quantiles'])).toEqual(Array(65).fill(1));
  }
});

it('UDF-08 keeps out-of-domain binary64 probabilities invalid after packing', () => {
  const parameters = getGPUColumnQuantilesParameterValues({
    quantiles: [-1e-50, 1 + Number.EPSILON]
  });
  expect(Array.from(parameters.slice(4))).toEqual([NaN, NaN]);
  expect(
    Array.from(
      computeColumnQuantilesOnCPU({values: Float32Array.of(0, 10, 20), parameters}, 2).quantiles
    )
  ).toEqual([NaN, NaN]);
});

it('ING-41 validates every neutral rational pair before target writes', () => {
  const target = new Uint32Array(4).fill(9);
  expect(() =>
    getColumnQuantilesRationalProbabilityValues(
      [
        [1, 2],
        [1, 0]
      ],
      2,
      target
    )
  ).toThrow(/denominator/);
  expect(Array.from(target)).toEqual([9, 9, 9, 9]);
  expect(() =>
    getColumnQuantilesRationalProbabilityValues([[1, 2]], 2, new Uint32Array(3))
  ).toThrow(/target/);
  expect(() => getGPUColumnQuantilesParameterValues({quantiles: Array(65).fill(0.5)})).toThrow(
    /quantileCount/
  );
  expect(() =>
    getGPUColumnQuantilesRationalProbabilityValues(Array.from({length: 65}, () => [1, 2] as const))
  ).toThrow(/quantileCount/);
});

it.each([
  2 ** 24 + 1,
  2 ** 24 + 2,
  2 ** 24 + 4
])('B1 preserves CPU extrema above GPU row capacity: %i', rowCount => {
  // A sorted sample of zeros followed by one 37 has maximum 37 at every length.
  // The default [0,1] filter must include every row, including the sole maximum.
  const values = new Float32Array(rowCount);
  values[rowCount - 1] = 37;
  const result = computeColumnQuantilesOnCPU(
    {values, parameters: getGPUColumnQuantilesParameterValues({quantiles: [1]})},
    1
  );
  expect(Array.from(result.quantiles)).toEqual([37]);
  expect(Array.from(result.filterBounds)).toEqual([0, 37]);
  expect(result.outputMask[rowCount - 1]).toBe(1);
}, 30000);

it.each([false, true])('B1 captures operation names and rowCount once, inherited=%s', inherited => {
  const rows = (values: Float32Array) => ({
    format: 'float32' as const,
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  for (const Operation of [CPUColumnQuantilesOperation, GPUColumnQuantilesOperation]) {
    const fixture = createQuantilesSnapshotFixture(inherited);
    const operation = new Operation(fixture.props);
    const expectedReads = {
      values: 1,
      parameters: 1,
      rowCount: 1,
      quantiles: 1,
      validCount: 1,
      filterBounds: 1
    };
    expect(fixture.reads).toEqual(expectedReads);
    fixture.mutate();
    if (operation instanceof CPUColumnQuantilesOperation) {
      expect(operation.outputs).toHaveProperty('initial-bounds');
      const result = operation.execute({
        inputs: {
          values: rows(Float32Array.of(1, 3)),
          parameters: rows(
            getGPUColumnQuantilesParameterValues({quantiles: [0.5], filterRange: [0, 1]})
          )
        }
      }) as Record<string, Float32Array | Uint32Array>;
      expect(Array.from(result['initial-bounds'])).toEqual([1, 3]);
      expect(result).not.toHaveProperty('filterBounds');
      expect(Array.from(result['quantiles'])).toEqual([2]);
      expect(result).not.toHaveProperty('later-filterBounds');
    } else {
      expect(operation.outputs).toHaveProperty('filterBounds.value', 'initial-bounds');
      const device = createNullWebGPUDevice();
      try {
        const getView = createOperationTestViewGetter(new GPUCommandGraph(device), {
          values: {format: 'float32', length: 2},
          parameters: {format: 'float32', length: 5},
          quantiles: {format: 'float32', length: 1},
          validCount: {format: 'uint32', length: 1},
          filterBounds: {format: 'float32', length: 2}
        });
        const contributor = operation.createContributor(getView);
        expect(contributor).toHaveProperty('props.output.filterBounds', getView('filterBounds'));
        expect(operation.inputs).toMatchObject({values: 'values', parameters: 'parameters'});
      } finally {
        device.destroy();
      }
    }
    expect(fixture.reads).toEqual(expectedReads);
  }
});
