// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import {
  computeClassBreaksOnCPU,
  CPUClassBreaksOperation
} from '../../../src/dataframe-analysis/column-classification/cpu-class-breaks-operation';
import {
  GPUClassBreaksOperation,
  type ClassBreaksOperationProps
} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks-operation';
import {
  getGPUClassBreaksParameterValues,
  type GPUClassBreaksSettings
} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';

const PROPS: ClassBreaksOperationProps = {
  id: 'breaks-check',
  values: 'values',
  parameters: 'parameters',
  rowCount: 8,
  maximumClassCount: 6,
  output: {breaks: 'breaks', classCount: 'classCount', classCounts: 'classCounts'}
};
function computeBreaks(values: number[], settings: GPUClassBreaksSettings) {
  return computeClassBreaksOnCPU(
    Float32Array.from(values),
    getGPUClassBreaksParameterValues(settings, 6),
    {maximumClassCount: 6}
  );
}

// Enumerate every partition into k nonempty classes at distinct-value boundaries. Score it by
// direct deviations about each class's independently recomputed mean, not the twin's recurrence.
function findBestPartitionCost(sorted: number[], classes: number): number {
  let best = Infinity;
  for (let bits = 0; bits < 2 ** (sorted.length - 1); bits++) {
    const edges = [0];
    for (let row = 1; row < sorted.length; row++) if (bits & (1 << (row - 1))) edges.push(row);
    edges.push(sorted.length);
    if (
      edges.length !== classes + 1 ||
      edges.slice(1, -1).some(row => sorted[row - 1] === sorted[row])
    )
      continue;
    let cost = 0;
    for (let group = 1; group < edges.length; group++) {
      const sample = sorted.slice(edges[group - 1], edges[group]);
      const mean = sample.reduce((sum, value) => sum + value, 0) / sample.length;
      cost += sample.reduce((sum, value) => sum + (value - mean) ** 2, 0);
    }
    best = Math.min(best, cost);
  }
  return best;
}

it('finds the global Fisher-Jenks optimum by exhaustive independent partition checks', () => {
  const values = [0, 0, 1, 2, 8, 9, 10, 10];
  for (const classes of [2, 3, 4]) {
    const result = computeBreaks(values, {method: 'natural-breaks', classCount: classes});
    let objective = 0;
    for (let group = 0; group < result.classCount[0]; group++) {
      const sample = values.filter(value => {
        let assigned = 0;
        while (assigned + 1 < classes && value >= result.breaks[assigned + 1]) assigned++;
        return assigned === group;
      });
      const mean = sample.reduce((sum, value) => sum + value, 0) / sample.length;
      objective += sample.reduce((sum, value) => sum + (value - mean) ** 2, 0);
    }
    expect(result.classCount[0]).toBe(classes);
    expect(objective).toBeCloseTo(findBestPartitionCost(values, classes), 10);
    expect(Array.from(result.classCounts).reduce((sum, count) => sum + count, 0)).toBe(
      values.length
    );
  }
  expect(
    Array.from(
      computeBreaks([0, 1, 2], {method: 'natural-breaks', classCount: 2}).breaks.subarray(0, 3)
    )
  ).toEqual([0, 1, 2]);
  const distinct = computeBreaks([2, 2, 8, 8], {method: 'natural-breaks', classCount: 6});
  expect(distinct.classCount[0]).toBe(2);
  expect(Array.from(distinct.breaks)).toEqual([2, 8, 8, NaN, NaN, NaN, NaN]);
  // Treating distinct values as equally weighted chooses edge 5, which costs 21.4666...;
  // the four copies of zero make edge 4 optimal, with within-class cost 17.
  const weighted = computeBreaks([0, 0, 0, 0, 4, 5, 8, 9], {
    method: 'natural-breaks',
    classCount: 2
  });
  expect(Array.from(weighted.breaks.subarray(0, 3))).toEqual([0, 4, 9]);
  expect(Array.from(weighted.classCounts.subarray(0, 2))).toEqual([4, 4]);
});

it('delegates GPU limits to contributor construction and drops that GPU-only cap on CPU', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {
      values: {format: 'float32', length: 8},
      parameters: {format: 'float32', length: 74},
      breaks: {format: 'float32', length: 66},
      classCount: {format: 'uint32', length: 1},
      classCounts: {format: 'uint32', length: 65}
    });
    const props = {...PROPS, maximumClassCount: 65, methods: ['equal-interval'] as const};
    expect(() => new GPUClassBreaksOperation(props).createContributor(getView)).toThrow(
      /maximumClassCount/
    );
    expect(() => new CPUClassBreaksOperation(props)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('matches hand-computed breaks and counts for arithmetic progression data', () => {
  const values = [0, 1, 2, 3, 4, 5, 6, 7];
  for (const method of ['equal-interval', 'quantile'] as const) {
    const result = computeBreaks(values, {method, classCount: 4});
    expect(Array.from(result.breaks)).toEqual([0, 1.75, 3.5, 5.25, 7, NaN, NaN]);
    expect(Array.from(result.classCounts)).toEqual([2, 2, 2, 2, 0, 0]);
  }
  const standard = computeBreaks(values, {method: 'standard-deviation', classCount: 4});
  // Mean 3.5; sample variance 6; symmetric one-sigma edges.
  expect(Array.from(standard.breaks.subarray(0, 5))).toEqual(
    Array.from(Float32Array.from([0, 3.5 - Math.sqrt(6), 3.5, 3.5 + Math.sqrt(6), 7]))
  );
  const box = computeBreaks(values, {method: 'box-plot'});
  expect(Array.from(box.breaks)).toEqual([0, 0, 1.75, 3.5, 5.25, 7, 7]);
  expect(Array.from(box.classCounts)).toEqual([0, 2, 2, 2, 1, 1]);
  const maximum = computeBreaks(values, {method: 'maximum-breaks', classCount: 3});
  expect(Array.from(maximum.breaks.subarray(0, 4))).toEqual([0, 0.5, 1.5, 7]);
  expect(Array.from(maximum.classCounts)).toEqual([1, 1, 6, 0, 0, 0]);
});

it('matches head-tail means, strict heads, and the minority stopping rule', () => {
  for (const [ratio, expected] of [
    [1, [0, 6, 15, 20]],
    [0.2, [0, 6, 20]]
  ] as const) {
    const result = computeBreaks([0, 1, 2, 3, 10, 20], {
      method: 'head-tail',
      classCount: 4,
      headTailRatio: ratio
    });
    expect(Array.from(result.breaks.subarray(0, result.classCount[0] + 1))).toEqual(expected);
  }
});

it('handles custom edges, repeated edges, missing/constant columns and integer conversion', () => {
  const custom = computeBreaks([-Infinity, 0, 1, 3, Infinity, NaN], {
    method: 'custom',
    customEdges: [-Infinity, 0, 3, Infinity]
  });
  expect(Array.from(custom.classCounts)).toEqual([1, 2, 2, 0, 0, 0]);
  const empty = computeBreaks([NaN, Infinity], {method: 'equal-interval', classCount: 3});
  expect(empty.classCount[0]).toBe(0);
  expect(empty.breaks.every(Number.isNaN)).toBe(true);
  const constant = computeBreaks([-0, 0, 0], {method: 'quantile', classCount: 4});
  expect(constant.classCount[0]).toBe(1);
  expect(constant.breaks[0] === 0 && constant.breaks[1] === 0).toBe(true);
  const parameters = getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 1}, 6);
  const converted = computeClassBreaksOnCPU(new Uint32Array([16777217, 16777217]), parameters, {
    maximumClassCount: 6
  });
  expect(Array.from(converted.breaks.subarray(0, 2))).toEqual([16777216, 16777216]);
  const huge = computeBreaks([-3e38, 3e38], {method: 'equal-interval', classCount: 2});
  expect(huge.breaks[1]).toBe(0);
});

it('uses ordered keys for uploaded signed zeros in custom counts and maximum gaps', () => {
  const custom = computeClassBreaksOnCPU(
    new Float32Array([-0, +0]),
    getGPUClassBreaksParameterValues({method: 'custom', customEdges: [-1, +0, 1]}, 2),
    {maximumClassCount: 2, methods: ['custom']}
  );
  // Negative zero's key is below positive zero's key, so each class receives one row.
  expect(custom.classCount[0]).toBe(2);
  expect(Array.from(custom.classCounts)).toEqual([1, 1]);
  const maximum = computeClassBreaksOnCPU(
    new Float32Array([-1, -0, +0, 1]),
    getGPUClassBreaksParameterValues({method: 'maximum-breaks', classCount: 4}, 4),
    {maximumClassCount: 4, methods: ['maximum-breaks']}
  );
  // Three key-distinct gaps have widths 1, 0, 1. All three are selected for four classes.
  expect(maximum.classCount[0]).toBe(4);
  expect(Array.from(maximum.classCounts)).toEqual([1, 1, 1, 1]);
  [-1, -0.5, 0, 0.5, 1].forEach((value, edge) => expect(maximum.breaks[edge] === value).toBe(true));
});

it('checks arrays, uncompiled methods, malformed parameter statuses and operation construction', () => {
  const parameters = getGPUClassBreaksParameterValues({method: 'quantile', classCount: 3}, 6);
  expect(() =>
    computeClassBreaksOnCPU(
      new Float32Array(2),
      parameters,
      {maximumClassCount: 6},
      new Uint32Array(1)
    )
  ).toThrow('mask');
  expect(() =>
    computeClassBreaksOnCPU(new Float32Array(2), parameters.subarray(1), {maximumClassCount: 6})
  ).toThrow('parameters');
  expect(
    computeClassBreaksOnCPU(new Float32Array([1, 2]), parameters, {
      maximumClassCount: 6,
      methods: ['custom']
    }).classCount[0]
  ).toBe(0);
  parameters.fill(NaN);
  expect(
    computeClassBreaksOnCPU(new Float32Array([1, 2]), parameters, {maximumClassCount: 6})
      .classCount[0]
  ).toBe(0);
  for (const Operation of [GPUClassBreaksOperation, CPUClassBreaksOperation]) {
    expect(() => new Operation({...PROPS, output: {...PROPS.output, breaks: 'values'}})).toThrow(
      /^breaks-check/
    );
    expect(new Operation(PROPS).id).toBe('breaks-check');
  }
  for (const replacement of [
    {rowCount: 0},
    {maximumClassCount: 0},
    {maximumClassCount: 5},
    {methods: []},
    {naturalBreaksBinCount: 1}
  ])
    expect(() => new CPUClassBreaksOperation({...PROPS, ...replacement})).toThrow(/^breaks-check/);
});
