// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUColumnQuantiles} from '../../../src/dataframe-analysis/column-classification/gpu-column-quantiles';
import type {GPUColumnQuantilesProps} from '../../../src/dataframe-analysis/column-classification/gpu-column-quantiles';
import {
  getGPUColumnQuantilesParameterLength,
  getGPUColumnQuantilesRationalProbabilityLength,
  getGPUColumnQuantilesRationalProbabilityValues,
  GPU_COLUMN_QUANTILES_MAXIMUM_RATIONAL_TERM
} from '../../../src/dataframe-analysis/column-classification/column-quantiles-parameters';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  rows: number,
  overrides: (
    view: <Format extends 'uint32' | 'float32'>(
      format: Format,
      length: number
    ) => GraphDataView<Format>
  ) => Partial<GPUColumnQuantilesProps>
): GPUColumnQuantilesProps {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `rational-view-${serial++}`, format, length);
  return {
    values: view('float32', rows),
    parameters: view('float32', getGPUColumnQuantilesParameterLength(3)),
    quantileCount: 3,
    output: {quantiles: view('float32', 3), validCount: view('uint32', 1)},
    ...overrides(view)
  };
}

function construct(rows: number, overrides: Parameters<typeof createProps>[2]): GPUColumnQuantiles {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    return new GPUColumnQuantiles(createProps(graph, rows, overrides));
  } finally {
    device.destroy();
  }
}

it('getGPUColumnQuantilesRationalProbabilityValues packs numerator and denominator pairs', () => {
  expect(getGPUColumnQuantilesRationalProbabilityLength(3)).toBe(6);
  expect(
    Array.from(
      getGPUColumnQuantilesRationalProbabilityValues(
        [
          [19, 20],
          [0, 1],
          [GPU_COLUMN_QUANTILES_MAXIMUM_RATIONAL_TERM, GPU_COLUMN_QUANTILES_MAXIMUM_RATIONAL_TERM]
        ],
        4
      )
    )
  ).toEqual([19, 20, 0, 1, 0xffffffff, 0xffffffff, 0, 0]);
  // A ratio above 1 is packed as given; the kernel returns NaN for it.
  expect(Array.from(getGPUColumnQuantilesRationalProbabilityValues([[3, 2]]))).toEqual([3, 2]);
  const target = new Uint32Array(6).fill(9);
  expect(getGPUColumnQuantilesRationalProbabilityValues([[1, 2]], 2, target)).toBe(target);
  expect(Array.from(target)).toEqual([1, 2, 0, 0, 9, 9]);
});

it('getGPUColumnQuantilesRationalProbabilityValues names the term and the limit it rejects', () => {
  expect(() => getGPUColumnQuantilesRationalProbabilityValues([[1, 0]])).toThrow(
    /^GPUColumnQuantiles rational probability 0 denominator must be an integer in \[1, 4294967295\]$/
  );
  expect(() => getGPUColumnQuantilesRationalProbabilityValues([[0.5, 1]])).toThrow(
    /^GPUColumnQuantiles rational probability 0 numerator must be an integer in \[0, 4294967295\]$/
  );
  expect(() =>
    getGPUColumnQuantilesRationalProbabilityValues([
      [1, 2],
      [1, 2 ** 32]
    ])
  ).toThrow(/rational probability 1 denominator/);
  expect(() => getGPUColumnQuantilesRationalProbabilityValues([[-1, 2]])).toThrow(/numerator/);
  expect(() =>
    getGPUColumnQuantilesRationalProbabilityValues(
      [
        [1, 2],
        [1, 3]
      ],
      1
    )
  ).toThrow(/exceed quantileCount/);
  expect(() => getGPUColumnQuantilesRationalProbabilityValues([[1, 2]], 0)).toThrow(
    /quantileCount/
  );
  expect(() =>
    getGPUColumnQuantilesRationalProbabilityValues([[1, 2]], 2, new Uint32Array(3))
  ).toThrow(/target must hold 4/);
});

it('GPUColumnQuantiles validates rationalProbabilities and the row limits of the two paths', () => {
  expect(
    construct(10, view => ({rationalProbabilities: view('uint32', 6)})).props.rationalProbabilities
  ).toBeDefined();
  expect(() => construct(10, view => ({rationalProbabilities: view('uint32', 5)}))).toThrow(
    /^column-quantiles rationalProbabilities must hold 6 uint32 values/
  );
  expect(() =>
    construct(10, view => ({rationalProbabilities: view('float32', 6) as never}))
  ).toThrow(/rationalProbabilities/);
  // The float32 path keeps its 2^24 limit and its message.
  expect(() => construct(2 ** 24 + 1, () => ({}))).toThrow(
    /^column-quantiles values length must be at most 16777216 \(2\^24\): .*float32 and are exact only up to 2\^24$/
  );
  // The rational path lifts it, unless a filter output is requested.
  expect(
    construct(2 ** 24 + 1, view => ({rationalProbabilities: view('uint32', 6)})).props.values.length
  ).toBe(2 ** 24 + 1);
  expect(() =>
    construct(2 ** 24 + 1, view => ({
      rationalProbabilities: view('uint32', 6),
      output: {
        quantiles: view('float32', 3),
        validCount: view('uint32', 1),
        filterBounds: view('float32', 2)
      }
    }))
  ).toThrow(
    /^column-quantiles values length must be at most 16777216 \(2\^24\) when output\.outputMask or output\.filterBounds is requested/
  );
  expect(() => construct(2 ** 31, view => ({rationalProbabilities: view('uint32', 6)}))).toThrow(
    /^column-quantiles values length must be at most 2147483647 \(2\^31 - 1\) with rationalProbabilities/
  );
  // An output may not alias the rational probabilities.
  expect(() =>
    construct(10, view => {
      const shared = view('uint32', 6);
      return {
        rationalProbabilities: shared,
        output: {quantiles: view('float32', 3), validCount: shared}
      };
    })
  ).toThrow(/column-quantiles/);
});

it('ROAD-113 rational total-row boundary admits 2^31-1 without payload allocation', () => {
  expect(
    construct(2 ** 31 - 1, view => ({rationalProbabilities: view('uint32', 6)})).props.values.length
  ).toBe(2 ** 31 - 1);
  expect(() => construct(2 ** 31, view => ({rationalProbabilities: view('uint32', 6)}))).toThrow(
    /values length must be at most 2147483647 \(2\^31 - 1\) with rationalProbabilities/
  );
});

it.each([
  'outputMask',
  'filterBounds'
] as const)('ROAD-113 %s separately pins the rational filter boundary', port => {
  const overrides: Parameters<typeof createProps>[2] = view => ({
    rationalProbabilities: view('uint32', 6),
    output: {
      quantiles: view('float32', 3),
      validCount: view('uint32', 1),
      ...(port === 'outputMask'
        ? {outputMask: view('uint32', 2 ** 24)}
        : {filterBounds: view('float32', 2)})
    }
  });
  expect(construct(2 ** 24, overrides).props.values.length).toBe(2 ** 24);
  expect(() => construct(2 ** 24 + 1, overrides)).toThrow(
    /^column-quantiles values length must be at most 16777216 \(2\^24\) when output\.outputMask or output\.filterBounds is requested/
  );
});

it('ING-41 leaves the rational target untouched when a later pair is invalid', () => {
  const target = new Uint32Array(4).fill(9);
  expect(() =>
    getGPUColumnQuantilesRationalProbabilityValues(
      [
        [1, 2],
        [1, 0]
      ],
      2,
      target
    )
  ).toThrow(/denominator/);
  expect(Array.from(target)).toEqual([9, 9, 9, 9]);
});

it('UDF-08 rejects the first decreasing packed custom edge before publication', async () => {
  const {getGPUClassBreaksParameterValues} = await import(
    '../../../src/dataframe-analysis/column-classification/class-breaks-parameters'
  );
  const target = new Float32Array(12).fill(9);
  expect(() =>
    getGPUClassBreaksParameterValues({method: 'custom', customEdges: [0, 10, 5, 20]}, 3, target)
  ).toThrow(/customEdges\[1\].*customEdges\[2\]/);
  expect(Array.from(target)).toEqual(Array(12).fill(9));
  expect(
    Array.from(
      getGPUClassBreaksParameterValues({method: 'custom', customEdges: [0, 5, 5, 20]}, 3).slice(8)
    )
  ).toEqual([0, 5, 5, 20]);
});

it('S3 packs overlapping probability controls without overwriting unread source lanes', async () => {
  const {getGPUColumnQuantilesParameterValues} = await import(
    '../../../src/dataframe-analysis/column-classification/column-quantiles-parameters'
  );
  const target = Float32Array.of(0.25, 0.75, 9, 9, 9, 9);
  expect(
    Array.from(getGPUColumnQuantilesParameterValues({quantiles: target.subarray(0, 2)}, target))
  ).toEqual([3, 0, 1, 0, 0.25, 0.75]);
});

it('S3 packs custom edges from the reusable target data slice', async () => {
  const {getGPUClassBreaksParameterValues} = await import(
    '../../../src/dataframe-analysis/column-classification/class-breaks-parameters'
  );
  const target = new Float32Array(11).fill(9);
  target.set([-2, 1, 4], 8);
  expect(
    Array.from(
      getGPUClassBreaksParameterValues(
        {method: 'custom', customEdges: target.subarray(8)},
        2,
        target
      ).slice(8)
    )
  ).toEqual([-2, 1, 4]);
});

it('S4 rejects inherited names in both option lookups before publication', async () => {
  const {getGPUClassBreaksParameterValues} = await import(
    '../../../src/dataframe-analysis/column-classification/class-breaks-parameters'
  );
  const {getGPUColumnQuantilesParameterValues} = await import(
    '../../../src/dataframe-analysis/column-classification/column-quantiles-parameters'
  );
  expect(() => getGPUClassBreaksParameterValues({method: 'constructor' as never}, 5)).toThrow(
    /unknown method constructor/
  );
  expect(() =>
    getGPUColumnQuantilesParameterValues({quantiles: [0.5], interpolation: 'toString' as never})
  ).toThrow(/unknown interpolation toString/);
});
