// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {computeNaturalBreaksMirror} from './natural-breaks-mirror';
import {
  getComparisonFloat32Key,
  getOrderedFloat32Key
} from '../../../src/dataframe-analysis/column-classification/column-classification-shared';
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

function createInputRows(values: Float32Array | Uint32Array | Int32Array) {
  return {
    format:
      values instanceof Float32Array
        ? ('float32' as const)
        : values instanceof Uint32Array
          ? ('uint32' as const)
          : ('sint32' as const),
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  };
}

for (const inherited of [false, true]) {
  it.each(['cpu', 'gpu'])(`%s class breaks retains getter mask, inherited=${inherited}`, form => {
    const descriptor = {get: () => 'mask', enumerable: false};
    const props: ClassBreaksOperationProps = Object.assign(
      inherited
        ? Object.create(Object.defineProperty({}, 'mask', descriptor))
        : Object.defineProperty({}, 'mask', descriptor),
      {...PROPS, rowCount: 2, maximumClassCount: 2, methods: ['equal-interval'] as const}
    );
    if (form === 'cpu') {
      const result = new CPUClassBreaksOperation(props).execute({
        inputs: {
          values: createInputRows(Float32Array.of(1, 100)),
          mask: createInputRows(Uint32Array.of(1, 0)),
          parameters: createInputRows(
            getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2)
          )
        }
      }) as Record<string, Float32Array | Uint32Array>;
      expect(Array.from(result['breaks'])).toEqual([1, 1, NaN]);
    } else {
      const operation = new GPUClassBreaksOperation(props);
      expect(operation.inputs).toHaveProperty('mask', 'mask');
      const device = createNullWebGPUDevice();
      try {
        const getView = createOperationTestViewGetter(new GPUCommandGraph(device), {
          values: {format: 'float32', length: 2},
          mask: {format: 'uint32', length: 2},
          parameters: {format: 'float32', length: 11},
          breaks: {format: 'float32', length: 3},
          classCount: {format: 'uint32', length: 1},
          classCounts: {format: 'uint32', length: 2}
        });
        expect(operation.createContributor(getView)).toHaveProperty('props.mask', getView('mask'));
      } finally {
        device.destroy();
      }
    }
  });

  it.each([
    {form: 'cpu', Operation: CPUClassBreaksOperation},
    {form: 'gpu', Operation: GPUClassBreaksOperation}
  ])(`$form class-break outputs retain getter fields, inherited=${inherited}`, ({Operation}) => {
    const fields = Object.fromEntries(
      Object.entries(PROPS.output).map(([name, value]) => [
        name,
        {get: () => value, enumerable: false}
      ])
    );
    const output = inherited
      ? Object.create(Object.defineProperties({}, fields))
      : Object.defineProperties({}, fields);
    const operation = new Operation({...PROPS, output});
    if (operation instanceof CPUClassBreaksOperation) {
      expect(Object.keys(operation.outputs).sort()).toEqual([
        'breaks',
        'classCount',
        'classCounts'
      ]);
    } else {
      expect(operation.outputs).toHaveProperty('breaks.value', 'breaks');
      expect(operation.outputs).toHaveProperty('classCount.value', 'classCount');
      expect(operation.outputs).toHaveProperty('classCounts.value', 'classCounts');
    }
  });
}

it.each([
  {form: 'cpu', Operation: CPUClassBreaksOperation},
  {form: 'gpu', Operation: GPUClassBreaksOperation}
])('$form class-break operation rejects exact storage without exact mode', ({Operation}) => {
  expect(
    () =>
      new Operation({...PROPS, methods: ['equal-interval'], naturalBreaksExactStorage: 'recompute'})
  ).toThrow(/naturalBreaksExactStorage requires exact mode/);
  expect(
    () =>
      new Operation({
        ...PROPS,
        methods: ['equal-interval'],
        naturalBreaksMode: 'exact',
        naturalBreaksExactStorage: 'recompute'
      })
  ).not.toThrow();
});

it.each([
  'uint32',
  'sint32'
] as const)('infers omitted valuesFormat for %s on both forms', format => {
  const values = format === 'uint32' ? new Uint32Array([0, 2]) : new Int32Array([-2, 0]);
  const props = {
    ...PROPS,
    rowCount: undefined,
    maximumClassCount: 2,
    methods: ['equal-interval'] as const
  };
  const operation = new CPUClassBreaksOperation(props);
  const result = operation.execute({
    inputs: {
      values: createInputRows(values),
      parameters: createInputRows(
        getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2)
      )
    }
  }) as Record<string, Float32Array | Uint32Array>;
  expect(Array.from(result['breaks'])).toEqual(format === 'uint32' ? [0, 1, 2] : [-2, -1, 0]);
  expect(Array.from(result['classCounts'])).toEqual([1, 1]);
  const device = createNullWebGPUDevice();
  try {
    const getView = createOperationTestViewGetter(new GPUCommandGraph(device), {
      values: {format, length: 2},
      parameters: {format: 'float32', length: 11},
      breaks: {format: 'float32', length: 3},
      classCount: {format: 'uint32', length: 1},
      classCounts: {format: 'uint32', length: 2}
    });
    expect(() => new GPUClassBreaksOperation(props).createContributor(getView)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it.each([
  {rowCount: 3},
  {valuesFormat: 'float32' as const}
])('both forms reject inconsistent assertions %o', assertion => {
  const props = {
    ...PROPS,
    rowCount: 2,
    maximumClassCount: 2,
    methods: ['equal-interval'] as const,
    ...assertion
  };
  const expected = assertion.rowCount !== undefined ? /rowCount/ : /valuesFormat/;
  expect(() =>
    new CPUClassBreaksOperation(props).execute({
      inputs: {
        values: createInputRows(new Uint32Array([0, 2])),
        parameters: createInputRows(
          getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2)
        )
      }
    })
  ).toThrow(expected);
  const device = createNullWebGPUDevice();
  try {
    const getView = createOperationTestViewGetter(new GPUCommandGraph(device), {
      values: {format: 'uint32', length: 2},
      parameters: {format: 'float32', length: 11},
      breaks: {format: 'float32', length: 3},
      classCount: {format: 'uint32', length: 1},
      classCounts: {format: 'uint32', length: 2}
    });
    expect(() => new GPUClassBreaksOperation(props).createContributor(getView)).toThrow(expected);
  } finally {
    device.destroy();
  }
});
function computeBreaks(values: number[], settings: GPUClassBreaksSettings) {
  return computeClassBreaksOnCPU(
    Float32Array.from(values),
    getGPUClassBreaksParameterValues(settings, 6),
    {maximumClassCount: 6}
  );
}

it.each([
  0, 1
])('CPU class breaks accepts empty input while GPU requires positive inferred rows: %s', rowCount => {
  const props: ClassBreaksOperationProps = {
    ...PROPS,
    rowCount: undefined,
    maximumClassCount: 2,
    methods: ['equal-interval']
  };
  const device = createNullWebGPUDevice();
  try {
    const getView = createOperationTestViewGetter(new GPUCommandGraph(device), {
      values: {format: 'uint32', length: rowCount},
      parameters: {format: 'float32', length: 11},
      breaks: {format: 'float32', length: 3},
      classCount: {format: 'uint32', length: 1},
      classCounts: {format: 'uint32', length: 2}
    });
    const runCPU = () =>
      new CPUClassBreaksOperation(props).execute({
        inputs: {
          values: createInputRows(new Uint32Array(rowCount).fill(7)),
          parameters: createInputRows(
            getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2)
          )
        }
      });
    const runGPU = () => new GPUClassBreaksOperation(props).createContributor(getView);
    if (rowCount === 0) {
      const result = runCPU() as Record<string, Float32Array | Uint32Array>;
      expect(Array.from(result['breaks'])).toEqual([NaN, NaN, NaN]);
      expect(Array.from(result['classCount'])).toEqual([0]);
      expect(runGPU).toThrow(/skip the GPU operation/);
    } else {
      const result = runCPU() as Record<string, Float32Array | Uint32Array>;
      expect(Array.from(result['breaks']).slice(0, 2)).toEqual([7, 7]);
      expect(Array.from(result['classCount'])).toEqual([1]);
      expect(runGPU).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it.each([
  'cpu',
  'gpu'
])('existing %s class-break operation retains its contract after caller mutation', form => {
  const methods: NonNullable<ClassBreaksOperationProps['methods']>[number][] = ['equal-interval'];
  const props: ClassBreaksOperationProps = {
    ...PROPS,
    rowCount: 2,
    maximumClassCount: 2,
    methods,
    output: {...PROPS.output}
  };
  const cpu = new CPUClassBreaksOperation(props);
  const gpu = new GPUClassBreaksOperation(props);
  const inputs = {
    values: createInputRows(new Uint32Array([0, 2])),
    parameters: createInputRows(
      getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2)
    )
  };
  const device = createNullWebGPUDevice();
  try {
    const getView = createOperationTestViewGetter(new GPUCommandGraph(device), {
      values: {format: 'uint32', length: 2},
      parameters: {format: 'float32', length: 11},
      breaks: {format: 'float32', length: 3},
      classCount: {format: 'uint32', length: 1},
      classCounts: {format: 'uint32', length: 2}
    });
    for (const run of [0, 1]) {
      if (run === 1) {
        props.rowCount = 3;
        props.valuesFormat = 'float32';
        props.maximumClassCount = 6;
        props.values = 'other-values';
        props.mask = 'other-mask';
        props.output.breaks = 'other-breaks';
        props.output.classCounts = undefined;
        methods.splice(0, methods.length, 'quantile');
      }
      if (form === 'cpu') {
        const result = cpu.execute({inputs}) as Record<string, Float32Array | Uint32Array>;
        expect(Object.keys(result).sort()).toEqual(['breaks', 'classCount', 'classCounts']);
        expect(Array.from(result['breaks'])).toEqual([0, 1, 2]);
        expect(Array.from(result['classCounts'])).toEqual([1, 1]);
      } else {
        const contributor = gpu.createContributor(getView);
        expect(contributor).toHaveProperty('methods', ['equal-interval']);
        expect(contributor).toHaveProperty('props.values.length', 2);
        expect(contributor).toHaveProperty('props.output.classCounts.length', 2);
      }
    }
  } finally {
    device.destroy();
  }
});

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

it('uses numeric equality for signed zeros in custom counts and maximum gaps (DPC-110)', () => {
  const custom = computeClassBreaksOnCPU(
    new Float32Array([-0, +0]),
    getGPUClassBreaksParameterValues({method: 'custom', customEdges: [-1, +0, 1]}, 2),
    {maximumClassCount: 2, methods: ['custom']}
  );
  // Both zeros equal the inner edge and therefore enter the upper class.
  expect(custom.classCount[0]).toBe(2);
  expect(Array.from(custom.classCounts)).toEqual([0, 2]);
  const maximum = computeClassBreaksOnCPU(
    new Float32Array([-1, -0, +0, 1]),
    getGPUClassBreaksParameterValues({method: 'maximum-breaks', classCount: 4}, 4),
    {maximumClassCount: 4, methods: ['maximum-breaks']}
  );
  // Three numeric distinct values have two gaps of width 1, giving three classes.
  expect(maximum.classCount[0]).toBe(3);
  expect(Array.from(maximum.classCounts)).toEqual([1, 2, 1, 0]);
  [-1, -0.5, 0.5, 1].forEach((value, edge) => expect(maximum.breaks[edge] === value).toBe(true));
});

it.each([
  {
    values: [1, 1.0000001192092896],
    edges: [1, 1.0000001192092896, 1.0000001192092896],
    counts: [1, 1]
  },
  {
    values: [7.703719777548943e-34, 7.703720695903905e-34],
    edges: [7.703719777548943e-34, 7.703720695903905e-34, 7.703720695903905e-34],
    counts: [1, 1]
  },
  {
    values: [-1.9999998807907104, -2, -2, -1.9999998807907104, -2],
    edges: [-2, -1.9999998807907104, -1.9999998807907104],
    counts: [3, 2]
  }
])('W5-01 maximum breaks separates adjacent float32 observations: %j', fixture => {
  const result = computeClassBreaksOnCPU(
    Float32Array.from(fixture.values),
    getGPUClassBreaksParameterValues({method: 'maximum-breaks', classCount: 2}, 2),
    {maximumClassCount: 2, methods: ['maximum-breaks']}
  );
  // The exact midpoint 1 + 2^-24 rounds to 1. A lower edge must instead use the upper row.
  expect(Array.from(result.classCount)).toEqual([2]);
  expect(Array.from(result.breaks)).toEqual(fixture.edges);
  expect(Array.from(result.classCounts)).toEqual(fixture.counts);
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
    {rowCount: -1},
    {maximumClassCount: 0},
    {maximumClassCount: 5, methods: ['box-plot'] as const},
    {methods: []},
    {naturalBreaksBinCount: 1}
  ])
    expect(() => new CPUClassBreaksOperation({...PROPS, ...replacement})).toThrow(/^breaks-check/);
});

it('DPC-110 ordering: normalizes only zero keys and retains adjacent floats, invalid rows and masks', () => {
  expect(getOrderedFloat32Key(-0)).toBe(0x7fffffff);
  expect(getOrderedFloat32Key(+0)).toBe(0x80000000);
  expect(getComparisonFloat32Key(-0)).toBe(0x80000000);
  expect(getComparisonFloat32Key(+0)).toBe(0x80000000);
  // IEEE binary32 neighbors of +/-1 have a spacing of 2^-23 on the outer side.
  const adjacent = 1 + 2 ** -23;
  for (const value of [-adjacent, -1, 1, adjacent]) {
    expect(getComparisonFloat32Key(value)).toBe(getOrderedFloat32Key(value));
  }
  expect(getComparisonFloat32Key(-adjacent)).toBeLessThan(getComparisonFloat32Key(-1));
  expect(getComparisonFloat32Key(1)).toBeLessThan(getComparisonFloat32Key(adjacent));
  const values = Float32Array.of(-adjacent, -1, -0, +0, 1, adjacent, NaN, adjacent);
  const mask = Uint32Array.of(1, 1, 1, 1, 1, 1, 1, 0);
  const result = computeClassBreaksOnCPU(
    values,
    getGPUClassBreaksParameterValues({method: 'custom', customEdges: [-2, -1, adjacent, 2]}, 3),
    {maximumClassCount: 3, methods: ['custom']},
    mask
  );
  expect(Array.from(result.classCounts)).toEqual([1, 4, 1]);
});

it('DPC-111: the float64 natural-break twin separates the two extreme finite clusters', () => {
  // Independent exhaustive-partition reference: natural-reference.py in this stream's scratch
  // evaluates all five splits. Split 3 has SSD 3.999998932932241e74; the next best is 2.4392500429471075e77.
  const values = Float32Array.of(-3e38, -2.9e38, -2.8e38, 2.8e38, 2.9e38, 3e38);
  const result = computeClassBreaksOnCPU(
    values,
    getGPUClassBreaksParameterValues({method: 'natural-breaks', classCount: 2}, 2),
    {maximumClassCount: 2, methods: ['natural-breaks']}
  );
  expect(Array.from(result.classCounts)).toEqual([3, 3]);
  expect(result.breaks.every(Number.isFinite)).toBe(true);
  expect(result.breaks[1]).toBe(values[3]);
});

it('CMP-05: classifies only the declared four-row grid prefix, retaining explicit fifth observations', () => {
  const allocation = Float32Array.of(2, 4, 6, 8, 1000);
  const parameters = getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2);
  for (const rows of [4, 5]) {
    const result = computeClassBreaksOnCPU(allocation.subarray(0, rows), parameters, {
      maximumClassCount: 2,
      methods: ['equal-interval']
    });
    expect(Array.from(result.breaks)).toEqual(rows === 4 ? [2, 5, 8] : [2, 501, 1000]);
    expect(Array.from(result.classCounts)).toEqual(rows === 4 ? [2, 2] : [4, 1]);
  }
});

it('DEF-07: omitted methods admit two equal-interval classes in pure and graph forms', () => {
  const parameters = getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2);
  const result = computeClassBreaksOnCPU(Float32Array.of(0, 2, 4), parameters, {
    maximumClassCount: 2
  });
  expect(Array.from(result.breaks)).toEqual([0, 2, 4]);
  const operation = new CPUClassBreaksOperation({...PROPS, rowCount: 3, maximumClassCount: 2});
  const graphResult = operation.execute({
    inputs: {
      values: createInputRows(Float32Array.of(0, 2, 4)),
      parameters: createInputRows(parameters)
    }
  }) as Record<string, Float32Array | Uint32Array>;
  expect(Array.from(graphResult['breaks'])).toEqual([0, 2, 4]);
});

it('FRM-10: both graph forms reject three approximate classes in two bins before resolving ports', () => {
  const props: ClassBreaksOperationProps = {
    ...PROPS,
    maximumClassCount: 3,
    methods: ['natural-breaks'],
    naturalBreaksMode: 'approximate',
    naturalBreaksBinCount: 2
  };
  for (const Operation of [GPUClassBreaksOperation, CPUClassBreaksOperation])
    expect(() => new Operation(props)).toThrow(/maximumClassCount 3.*naturalBreaksBinCount 2/);
  // The pure function remains the exact reference: six points form three pairs, SSD = 1.5.
  const result = computeClassBreaksOnCPU(
    Float32Array.of(0, 1, 2, 3, 4, 5),
    getGPUClassBreaksParameterValues({method: 'natural-breaks', classCount: 3}, 3),
    {maximumClassCount: 3, methods: ['natural-breaks']}
  );
  expect(Array.from(result.breaks)).toEqual([0, 2, 4, 5]);
});

it('DEF-06: reports the resolved mode and exact CPU algorithm for the twelve-value fixture', () => {
  // Exhaustive Fisher-Jenks partitions separate [15,36,56], [79,104,114],
  // [157,166,174,177,179] and [100000]: inner lower edges 79,157,100000.
  const values = Float32Array.of(174, 15, 79, 177, 157, 114, 36, 56, 179, 166, 104, 100000);
  const result = computeClassBreaksOnCPU(
    values,
    getGPUClassBreaksParameterValues({method: 'natural-breaks', classCount: 4}, 4),
    {maximumClassCount: 4, methods: ['natural-breaks']}
  );
  expect(Array.from(result.breaks.subarray(1, 4))).toEqual([79, 157, 100000]);
  expect(result).toHaveProperty('provenance.naturalBreaksMode', 'approximate');
  expect(result).toHaveProperty('provenance.naturalBreaksAlgorithm', 'exact-fisher-jenks-float64');
  expect(
    new CPUClassBreaksOperation({...PROPS, maximumClassCount: 4, methods: ['natural-breaks']})
  ).toHaveProperty('provenance.naturalBreaksAlgorithm', 'exact-fisher-jenks-float64');
});

it('FRM-12: the CPU graph publishes the pure empty-input result', () => {
  const operation = new CPUClassBreaksOperation({
    ...PROPS,
    rowCount: 0,
    maximumClassCount: 2,
    methods: ['equal-interval']
  });
  const result = operation.execute({
    inputs: {
      values: createInputRows(new Float32Array()),
      parameters: createInputRows(
        getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2)
      )
    }
  }) as Record<string, Float32Array | Uint32Array>;
  expect(Array.from(result['breaks'])).toEqual([NaN, NaN, NaN]);
  expect(Array.from(result['classCount'])).toEqual([0]);
  expect(Array.from(result['classCounts'])).toEqual([0, 0]);
});

it('FRM-12: the published mixed executor supports empty CPU class breaks', async () => {
  const device = createNullWebGPUDevice();
  try {
    const result = await runOperationGraph(device, {
      operations: [
        new CPUClassBreaksOperation({
          ...PROPS,
          rowCount: 0,
          maximumClassCount: 2,
          methods: ['equal-interval']
        })
      ],
      outputs: ['breaks', 'classCount', 'classCounts'],
      inputs: {
        values: {format: 'float32', values: new Float32Array()},
        parameters: {
          format: 'float32',
          values: getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2)
        }
      }
    });
    expect(Array.from(result['breaks'])).toEqual([NaN, NaN, NaN]);
    expect(Array.from(result['classCount'])).toEqual([0]);
    expect(Array.from(result['classCounts'])).toEqual([0, 0]);
  } finally {
    device.destroy();
  }
});

it('DEF-06: pins the approximate host-mirror observation separately from device evidence', () => {
  const values = Float32Array.of(174, 15, 79, 177, 157, 114, 36, 56, 179, 166, 104, 100000);
  expect(computeNaturalBreaksMirror(values, 15, 100000, 4, 2048).edges).toEqual([104, 157, 100000]);
});

it.each([
  undefined,
  ['maximum-breaks'] as const
])('R1 B1: CPU maximum breaks packs chunks with methods=%s', methods => {
  const operation = new CPUClassBreaksOperation({...PROPS, rowCount: 4, methods});
  expect(operation.methods).toContain('maximum-breaks');
  const result = operation.execute({
    inputs: {
      values: {
        ...createInputRows(Float32Array.of(1, 2, 20, 21)),
        chunks: [
          {values: Float32Array.of(1, 2), length: 2},
          {values: Float32Array.of(20, 21), length: 2}
        ]
      },
      parameters: createInputRows(
        getGPUClassBreaksParameterValues({method: 'maximum-breaks', classCount: 2}, 6)
      )
    }
  }) as Record<string, Float32Array | Uint32Array>;
  // Gaps 1,18,1; midpoint of the largest gap (2,20) is 11.
  expect(Array.from(result['breaks']).slice(0, 3)).toEqual([1, 11, 21]);
  expect(Array.from(result['classCount'])).toEqual([2]);
  expect(Array.from(result['classCounts'])).toEqual([2, 2, 0, 0, 0, 0]);
  expect(
    Object.values(result).every(
      value => value instanceof Float32Array || value instanceof Uint32Array
    )
  ).toBe(true);
});

it.each([
  undefined,
  2048,
  4096
])('R1 B2: CPU equal intervals preserves unused bins=%s at 2049 classes', naturalBreaksBinCount => {
  const operation = new CPUClassBreaksOperation({
    ...PROPS,
    rowCount: 3,
    maximumClassCount: 2049,
    methods: naturalBreaksBinCount === 4096 ? ['equal-interval'] : undefined,
    naturalBreaksBinCount
  });
  const result = operation.execute({
    inputs: {
      values: createInputRows(Float32Array.of(0, 2, 4)),
      parameters: createInputRows(
        getGPUClassBreaksParameterValues({method: 'equal-interval', classCount: 2}, 2049)
      )
    }
  }) as Record<string, Float32Array | Uint32Array>;
  expect(Array.from(result['breaks']).slice(0, 3)).toEqual([0, 2, 4]);
  expect(Array.from(result['classCount'])).toEqual([2]);
});

it.each([
  undefined,
  'exact',
  'monotone'
] as const)('R1 S1: GPU operation exposes %s provenance before lowering', naturalBreaksMode => {
  const operation = new GPUClassBreaksOperation({...PROPS, naturalBreaksMode});
  expect(operation).toHaveProperty(
    'provenance.naturalBreaksMode',
    naturalBreaksMode ?? 'approximate'
  );
  expect(operation).toHaveProperty(
    'provenance.naturalBreaksAlgorithm',
    naturalBreaksMode === 'exact'
      ? 'exact-fisher-jenks-normalized-float32'
      : naturalBreaksMode === 'monotone'
        ? 'monotone-quantized-dynamic-programming'
        : 'histogram-refinement'
  );
});

it('R1 B2: pure exact reference retains natural breaks beyond the graph bin default', () => {
  const result = computeClassBreaksOnCPU(
    Float32Array.of(0, 1, 2),
    getGPUClassBreaksParameterValues({method: 'natural-breaks', classCount: 2}, 2049),
    {maximumClassCount: 2049}
  );
  expect(Array.from(result.breaks).slice(0, 3)).toEqual([0, 1, 2]);
  expect(result.provenance?.methods).toContain('natural-breaks');
});

it.each([
  undefined,
  2048
])('R1 B2: CPU graph defaults preserve exact natural breaks with bins=%s', naturalBreaksBinCount => {
  const operation = new CPUClassBreaksOperation({
    ...PROPS,
    rowCount: 3,
    maximumClassCount: 2049,
    naturalBreaksBinCount
  });
  const result = operation.execute({
    inputs: {
      values: createInputRows(Float32Array.of(0, 1, 2)),
      parameters: createInputRows(
        getGPUClassBreaksParameterValues({method: 'natural-breaks', classCount: 2}, 2049)
      )
    }
  }) as Record<string, Float32Array | Uint32Array>;
  expect(operation.methods).toContain('natural-breaks');
  expect(Array.from(result['classCount'])).toEqual([2]);
  expect(Array.from(result['breaks']).slice(0, 3)).toEqual([0, 1, 2]);
});
