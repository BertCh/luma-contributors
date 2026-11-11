// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {getGPUClassBreaksParameterValues} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {GPUClassBreaks} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import {
  getGPUSpatialAutocorrelationParameterValues,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {
  GPUClassAssignment,
  GPULISAMarkov,
  GPUSpatialMarkov,
  GPUTransitionDiagnostics,
  GPUTransitionMatrix,
  GPU_LISA_MARKOV_STATE_COUNT,
  GPU_TRANSITION_HOMOGENEITY,
  GPU_TRANSITION_STATE_SPARSITY
} from '../../../src/gpu-spatial-analysis/distribution-dynamics/index';
import {
  createAutocorrelatedScene,
  createDistanceBandWeights
} from '../spatial-autocorrelation/spatial-autocorrelation-oracle';
import {AnalysisRig, createSeededRandom} from '../catchment-accessibility/rig';
import {
  computeLisaStatesOracle,
  computeTransitionOracle,
  getOracleClass,
  getOracleQuantile
} from './distribution-oracle';
import * as giddy from './markov-giddy-fixture';
import digitizeFixtureJson from '../../oracles/fixtures/distribution-class-assignment-numpy.json';
import {decodeOracleFixture} from '../../oracles/oracle-fixture';
import {requireTestDevice} from '../../utils/require-test-device';

type DigitizeClosureCase = {classes: number[]; counts: number[]};
/** NumPy `digitize` references, see `test/oracles/distribution-class-assignment-numpy.py`. */
const digitizeFixture = decodeOracleFixture<{
  versions: Record<string, string>;
  reference: {
    edges: {edges: number[]; values: number[]; right: number[]; left: number[]};
    pooled: {
      rows: number;
      periods: number;
      classCount: number;
      values: number[];
      edges: number[];
      tiedRowCount: number;
      right: DigitizeClosureCase;
      left: DigitizeClosureCase;
    };
  };
}>(digitizeFixtureJson);

const NO_CLASS = 0xffffffff;

/** Like `expectClose` but NaN must match NaN exactly. */
function expectCloseOrNaN(
  actual: number[],
  expected: number[],
  label: string,
  relative = 1e-4,
  absolute = 1e-6
): void {
  expect(actual.length, `${label} length`).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    if (Number.isNaN(expected[index]) || Number.isNaN(actual[index])) {
      if (!(Number.isNaN(expected[index]) && Number.isNaN(actual[index]))) {
        throw new Error(
          `${label}: [${index}] ${actual[index]} != ${expected[index]} (NaN mismatch)`
        );
      }
    } else if (
      Math.abs(actual[index] - expected[index]) >
      absolute + relative * Math.abs(expected[index])
    ) {
      throw new Error(`${label}: [${index}] ${actual[index]} != ${expected[index]}`);
    }
  }
}

function expectClose(actual: number[], expected: number[], label: string, relative = 1e-5): void {
  expect(actual.length, `${label} length`).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    if (Math.abs(actual[index] - expected[index]) > 1e-6 + relative * Math.abs(expected[index])) {
      throw new Error(`${label}: [${index}] ${actual[index]} != ${expected[index]}`);
    }
  }
}

it('GPUClassAssignment follows the shared class rule with ties, NaN, infinities and mask', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const edges = [0, 10, 20, 30, 40];
  const values = [
    -5,
    0,
    9.99,
    10,
    10.01,
    20,
    29,
    30,
    39.9,
    40,
    100,
    NaN,
    Infinity,
    -Infinity,
    15,
    25,
    35,
    5
  ];
  const mask = values.map((_, index) => (index === 14 ? 0 : 1));
  const rig = new AnalysisRig(device);
  const output = rig.output('uint32', values.length);
  rig.run(
    new GPUClassAssignment({
      values: rig.input(Float32Array.from(values), 'float32'),
      breaks: rig.input(Float32Array.from(edges), 'float32'),
      classCount: rig.input(Uint32Array.of(4), 'uint32'),
      mask: rig.input(Uint32Array.from(mask), 'uint32'),
      output
    })
  );
  const actual = await rig.readUint(output);
  const expected = values.map((value, index) =>
    mask[index] === 0 ? NO_CLASS : getOracleClass(value, edges, 4)
  );
  expect(actual).toEqual(expected);
  expect(new Set(actual).size).toBeGreaterThan(3);
  rig.destroy();
});

it("GPUClassAssignment intervalClosure 'right' keeps a value equal to an edge in the lower class", async () => {
  // Math review point-process-clustering-03. The rule of mapclassify and giddy is
  // numpy.digitize(values, bins, right=True): class j holds bins[j - 1] < x <= bins[j], so a value
  // equal to an edge stays below it and the minimum is in class 0. Expected classes are written
  // by hand from that rule for edges [0, 10, 20, 30, 40]; nothing here calls the oracle.
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const edges = [0, 10, 20, 30, 40];
  const cases: [value: number, rightClosed: number, leftClosed: number][] = [
    [-5, 0, 0],
    [0, 0, 0],
    [9.99, 0, 0],
    [10, 0, 1],
    [10.01, 1, 1],
    [20, 1, 2],
    [29, 2, 2],
    [30, 2, 3],
    [39.9, 3, 3],
    [40, 3, 3],
    [100, 3, 3],
    [NaN, NO_CLASS, NO_CLASS],
    [Infinity, 3, 3],
    [-Infinity, 0, 0]
  ];
  const values = Float32Array.from(cases.map(([value]) => value));
  const rig = new AnalysisRig(device);
  const rightClosed = rig.output('uint32', values.length);
  const leftClosed = rig.output('uint32', values.length);
  const breaks = rig.input(Float32Array.from(edges), 'float32');
  const classCount = rig.input(Uint32Array.of(4), 'uint32');
  const input = rig.input(values, 'float32');
  rig.run(
    new GPUClassAssignment({
      id: 'right-closed',
      values: input,
      breaks,
      classCount,
      intervalClosure: 'right',
      output: rightClosed
    }),
    new GPUClassAssignment({
      id: 'left-closed',
      values: input,
      breaks,
      classCount,
      output: leftClosed
    })
  );
  expect(await rig.readUint(rightClosed)).toEqual(cases.map(([, expected]) => expected));
  // The default is unchanged: an edge is the lower edge of the class above it.
  expect(await rig.readUint(leftClosed)).toEqual(cases.map(([, , expected]) => expected));
  rig.destroy();
  expect(
    () =>
      new GPUClassAssignment({
        values: input,
        breaks,
        classCount,
        intervalClosure: 'both' as 'left',
        output: rightClosed
      })
  ).toThrow(/intervalClosure/);
});

it("pooled quantile breaks with rows tied at a break count giddy's transitions with 'right'", async () => {
  // Five rows over three periods, period-major. Sorted, the pooled values are
  //   1 2 3 4 5 5 6 7 8 9 9 10 11 12 13
  // so the R-7 terciles sit at positions 14/3 and 28/3, between two 5s and between two 9s: the
  // inner edges are exactly 5 and 9 and two rows are tied at each. mapclassify.Quantiles(k=3)
  // has bins [5, 9, 13] and yb = (x <= 5 -> 0, x <= 9 -> 1, else 2); giddy.Markov counts the
  // transitions of those classes between consecutive periods. By hand:
  //   classes  period 0: 0 0 1 1 2   period 1: 0 0 1 1 2   period 2: 0 0 1 2 2
  //   counts   0->0: 4, 1->1: 3, 1->2: 1, 2->2: 2
  // With the default 'left' the tied rows sit one class higher:
  //   classes  period 0: 0 1 1 2 2   period 1: 0 1 1 2 2   period 2: 0 0 1 2 2
  //   counts   0->0: 2, 1->0: 1, 1->1: 3, 2->2: 4
  // (mapclassify and giddy are not installed on the machine these were written on, so they are
  // from the documented rule, not from a giddy run. The next case checks the same rule against
  // numpy.digitize, which is installed.)
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const pooled = Float32Array.of(1, 5, 6, 9, 13, 2, 5, 7, 9, 12, 3, 4, 8, 10, 11);
  const expected = {
    right: {
      classes: [0, 0, 1, 1, 2, 0, 0, 1, 1, 2, 0, 0, 1, 2, 2],
      counts: [4, 0, 0, 0, 3, 1, 0, 0, 2]
    },
    left: {
      classes: [0, 1, 1, 2, 2, 0, 1, 1, 2, 2, 0, 0, 1, 2, 2],
      counts: [2, 0, 0, 1, 3, 0, 0, 0, 4]
    }
  };
  for (const intervalClosure of ['right', 'left'] as const) {
    const rig = new AnalysisRig(device);
    const values = rig.input(pooled, 'float32');
    const breaks = rig.output('float32', 4);
    const classCount = rig.output('uint32', 1);
    const classes = rig.output('uint32', pooled.length);
    const counts = rig.output('uint32', 9);
    rig.run(
      new GPUClassBreaks({
        values,
        maximumClassCount: 3,
        methods: ['quantile'],
        parameters: rig.input(
          getGPUClassBreaksParameterValues({method: 'quantile', classCount: 3}, 3),
          'float32'
        ),
        output: {breaks, classCount}
      }),
      new GPUClassAssignment({values, breaks, classCount, intervalClosure, output: classes}),
      new GPUTransitionMatrix({classes, periodCount: 3, classCount: 3, output: {counts}})
    );
    expect(await rig.readFloat(breaks), intervalClosure).toEqual([1, 5, 9, 13]);
    expect(await rig.readUint(classes), intervalClosure).toEqual(expected[intervalClosure].classes);
    expect(await rig.readUint(counts), intervalClosure).toEqual(expected[intervalClosure].counts);
    rig.destroy();
  }
});

it('GPUClassAssignment matches numpy.digitize for both closures, on fixed edges and pooled quantiles', async () => {
  // Fix review fr-models-09: an independent reference for the two closures. The fixture is
  // NumPy 2.0.2 (`test/oracles/distribution-class-assignment-numpy.py`): `digitize(right=True)`
  // is the right-closed rule of mapclassify's bins, `digitize(right=False)` the left-closed one.
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {edges: fixed, pooled} = digitizeFixture.reference;
  expect(pooled.tiedRowCount).toBeGreaterThan(2);
  for (const intervalClosure of ['right', 'left'] as const) {
    const rig = new AnalysisRig(device);
    const fixedClasses = rig.output('uint32', fixed.values.length);
    const pooledValues = rig.input(Float32Array.from(pooled.values), 'float32');
    const breaks = rig.output('float32', pooled.classCount + 1);
    const classCount = rig.output('uint32', 1);
    const pooledClasses = rig.output('uint32', pooled.values.length);
    const counts = rig.output('uint32', pooled.classCount * pooled.classCount);
    rig.run(
      new GPUClassAssignment({
        id: 'fixed-edges',
        values: rig.input(Float32Array.from(fixed.values), 'float32'),
        breaks: rig.input(Float32Array.from(fixed.edges), 'float32'),
        classCount: rig.input(Uint32Array.of(fixed.edges.length - 1), 'uint32'),
        intervalClosure,
        output: fixedClasses
      }),
      new GPUClassBreaks({
        values: pooledValues,
        maximumClassCount: pooled.classCount,
        methods: ['quantile'],
        parameters: rig.input(
          getGPUClassBreaksParameterValues(
            {method: 'quantile', classCount: pooled.classCount},
            pooled.classCount
          ),
          'float32'
        ),
        output: {breaks, classCount}
      }),
      new GPUClassAssignment({
        id: 'pooled-quantiles',
        values: pooledValues,
        breaks,
        classCount,
        intervalClosure,
        output: pooledClasses
      }),
      new GPUTransitionMatrix({
        classes: pooledClasses,
        periodCount: pooled.periods,
        classCount: pooled.classCount,
        output: {counts}
      })
    );
    expect(await rig.readUint(fixedClasses), intervalClosure).toEqual(fixed[intervalClosure]);
    expect(await rig.readFloat(breaks), intervalClosure).toEqual(pooled.edges);
    expect(await rig.readUint(pooledClasses), intervalClosure).toEqual(
      pooled[intervalClosure].classes
    );
    expect(await rig.readUint(counts), intervalClosure).toEqual(pooled[intervalClosure].counts);
    rig.destroy();
  }
  // The two closures count different transitions on this data.
  expect(pooled.right.counts).not.toEqual(pooled.left.counts);
});

it("GPUSpatialMarkov lagIntervalClosure 'right' keeps a lag equal to a lag edge in the lower lag class", async () => {
  // Fix review fr-models-04. Six rows in three pairs (0-1, 2-3, 4-5), each row's only neighbor
  // its partner with weight 1 and `normalize: false`, so the lag of a row is exactly its
  // partner's value. Values, period-major:
  //   period 0: 0 8 9 1 2 10    period 1: 3 8 11 4 12 5    period 2: 13 6 7 14 15 16
  // The 18 pooled lags are the same numbers; sorted, positions 8 and 9 are both 8, so the median
  // (the one inner edge of two quantile lag classes) is exactly 8 and two lags are tied at it:
  // row 0 in periods 0 and 1 (its partner, row 1, holds 8 in both).
  //   lags     period 0: 8 0 1 9 10 2    period 1: 8 3 4 11 5 12
  // Classes of the values are given (0 for a value up to 7, else 1):
  //   period 0: 0 1 1 0 0 1    period 1: 0 1 1 0 1 0    period 2: 1 0 0 1 1 1
  // so the transitions are, by row, 0->0 1->1 1->1 0->0 0->1 1->0 from period 0 and
  // 0->1 1->0 1->0 0->1 1->1 0->1 from period 1.
  // 'right' (lag <= 8 is lag class 0, as giddy): lag classes 0 0 0 1 1 0 and 0 0 0 1 0 1.
  //   lag class 0: 0->0 1, 0->1 1, 1->0 3, 1->1 3    lag class 1: 0->0 1, 0->1 3
  // 'left' (default, lag >= 8 is lag class 1): row 0 moves to lag class 1 in both periods.
  //   lag class 0: 1->0 3, 1->1 3                     lag class 1: 0->0 2, 0->1 4
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const values = Float32Array.of(0, 8, 9, 1, 2, 10, 3, 8, 11, 4, 12, 5, 13, 6, 7, 14, 15, 16);
  const classes = Uint32Array.of(0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1);
  const pairs = {
    offsets: [0, 1, 2, 3, 4, 5, 6],
    neighbors: [1, 0, 3, 2, 5, 4],
    weights: [1, 1, 1, 1, 1, 1]
  };
  const expected = {
    right: {
      lagClasses: [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1],
      counts: [1, 1, 3, 3, 1, 3, 0, 0]
    },
    left: {
      lagClasses: [1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
      counts: [0, 0, 3, 3, 2, 4, 0, 0]
    }
  };
  for (const lagIntervalClosure of ['right', 'left', undefined] as const) {
    const rig = new AnalysisRig(device);
    const counts = rig.output('uint32', 8);
    const lagBreaks = rig.output('float32', 3);
    const lagClassCount = rig.output('uint32', 1);
    const lagValues = rig.output('float32', 18);
    const lagClasses = rig.output('uint32', 18);
    rig.run(
      new GPUSpatialMarkov({
        values: rig.input(values, 'float32'),
        classes: rig.input(classes, 'uint32'),
        classCount: 2,
        weights: rig.weights(pairs),
        periodCount: 3,
        normalize: false,
        maximumLagClassCount: 2,
        lagParameters: rig.input(
          getGPUClassBreaksParameterValues({method: 'quantile', classCount: 2}, 2),
          'float32'
        ),
        ...(lagIntervalClosure ? {lagIntervalClosure} : {}),
        output: {counts, lagBreaks, lagClassCount, lagValues, lagClasses}
      })
    );
    const closure = lagIntervalClosure ?? 'left';
    expect(await rig.readFloat(lagValues), closure).toEqual([
      8, 0, 1, 9, 10, 2, 8, 3, 4, 11, 5, 12, 6, 13, 14, 7, 16, 15
    ]);
    expect(await rig.readFloat(lagBreaks), closure).toEqual([0, 8, 16]);
    expect(await rig.readUint(lagClassCount), closure).toEqual([2]);
    // The lag classes of the last period condition no transition; the first two are checked.
    expect((await rig.readUint(lagClasses)).slice(0, 12), closure).toEqual(
      expected[closure].lagClasses
    );
    expect(await rig.readUint(counts), closure).toEqual(expected[closure].counts);
    rig.destroy();
  }
  // A wrong value is refused at construction, under the contributor's own ID.
  const rig = new AnalysisRig(device);
  expect(
    () =>
      new GPUSpatialMarkov({
        id: 'tied-lag',
        values: rig.input(values, 'float32'),
        classes: rig.input(classes, 'uint32'),
        classCount: 2,
        weights: rig.weights(pairs),
        periodCount: 3,
        maximumLagClassCount: 2,
        lagParameters: rig.input(
          getGPUClassBreaksParameterValues({method: 'quantile', classCount: 2}, 2),
          'float32'
        ),
        lagIntervalClosure: 'both' as 'left',
        output: {
          counts: rig.output('uint32', 8),
          lagBreaks: rig.output('float32', 3),
          lagClassCount: rig.output('uint32', 1)
        }
      })
  ).toThrow(/tied-lag lagIntervalClosure/);
  rig.destroy();
});

it("natural-breaks lower edges with 'left' keep the smallest value of each class in its class", async () => {
  // Three clusters. GPUClassBreaks natural breaks returns lower edges: each inner edge is the
  // smallest value of its class, here 10 and 20. With the default 'left' rule 10 and 20 open
  // their classes; with 'right' each would fall into the class below, which is why the closure is
  // an option and not a new default.
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const clustered = Float32Array.of(1, 2, 3, 10, 11, 12, 20, 21, 22);
  const rig = new AnalysisRig(device);
  const values = rig.input(clustered, 'float32');
  const breaks = rig.output('float32', 4);
  const classCount = rig.output('uint32', 1);
  const leftClosed = rig.output('uint32', clustered.length);
  const rightClosed = rig.output('uint32', clustered.length);
  rig.run(
    new GPUClassBreaks({
      values,
      maximumClassCount: 3,
      methods: ['natural-breaks'],
      parameters: rig.input(
        getGPUClassBreaksParameterValues({method: 'natural-breaks', classCount: 3}, 3),
        'float32'
      ),
      output: {breaks, classCount}
    }),
    new GPUClassAssignment({id: 'left-closed', values, breaks, classCount, output: leftClosed}),
    new GPUClassAssignment({
      id: 'right-closed',
      values,
      breaks,
      classCount,
      intervalClosure: 'right',
      output: rightClosed
    })
  );
  expect(await rig.readFloat(breaks)).toEqual([1, 10, 20, 22]);
  expect(await rig.readUint(leftClosed)).toEqual([0, 0, 0, 1, 1, 1, 2, 2, 2]);
  expect(await rig.readUint(rightClosed)).toEqual([0, 0, 0, 0, 1, 1, 1, 2, 2]);
  rig.destroy();
});

it('GPUTransitionMatrix matches the oracle for conditions, period lag and invalid classes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const {rows, periods, classCount, conditionCount, periodLag, seed} of [
    {rows: 700, periods: 6, classCount: 5, conditionCount: 1, periodLag: 1, seed: 1},
    {rows: 333, periods: 7, classCount: 4, conditionCount: 3, periodLag: 2, seed: 2},
    // Several workgroup tiles with a partial last tile (workgroup-private histogram).
    {rows: 5003, periods: 5, classCount: 5, conditionCount: 1, periodLag: 1, seed: 3},
    {rows: 3001, periods: 4, classCount: 6, conditionCount: 7, periodLag: 1, seed: 5},
    // 64 * 64 cells exceed the private histogram: global atomics.
    {rows: 2100, periods: 4, classCount: 64, conditionCount: 1, periodLag: 1, seed: 4}
  ]) {
    const random = createSeededRandom(seed);
    const classes = Uint32Array.from({length: rows * periods}, () =>
      random() < 0.05 ? NO_CLASS : Math.floor(random() * classCount)
    );
    const conditions = Uint32Array.from({length: rows * periods}, () =>
      random() < 0.03 ? NO_CLASS : Math.floor(random() * conditionCount)
    );
    const mask = Uint32Array.from({length: rows}, () => (random() < 0.9 ? 1 : 0));
    const cells = conditionCount * classCount * classCount;
    const rig = new AnalysisRig(device);
    const counts = rig.output('uint32', cells + 2);
    const probabilities = rig.output('float32', cells);
    const rowTotals = rig.output('uint32', conditionCount * classCount);
    const ignored = rig.output('uint32', 1);
    rig.run(
      new GPUTransitionMatrix({
        classes: rig.input(classes, 'uint32'),
        periodCount: periods,
        classCount,
        periodLag,
        conditionClasses: conditionCount > 1 ? rig.input(conditions, 'uint32') : undefined,
        conditionCount,
        mask: rig.input(mask, 'uint32'),
        output: {counts, probabilities, rowTotals, ignored}
      })
    );
    const expected = computeTransitionOracle({
      classes,
      conditions: conditionCount > 1 ? conditions : undefined,
      mask,
      rows,
      periods,
      classCount,
      conditionCount,
      periodLag
    });
    const actualCounts = (await rig.readUint(counts)).slice(0, cells);
    expect(actualCounts.reduce((sum, value) => sum + value, 0)).toBeGreaterThan(rows);
    expect(actualCounts).toEqual(expected.counts);
    expect(await rig.readUint(rowTotals)).toEqual(expected.rowTotals);
    expect((await rig.readUint(ignored))[0]).toBe(expected.ignored);
    expectClose(await rig.readFloat(probabilities), expected.probabilities, 'probabilities');
    rig.destroy();
  }
});

it('GPUSpatialMarkov conditions transitions on pooled spatial-lag classes', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 90;
  const periods = 5;
  const classCount = 4;
  const lagClassCount = 3;
  const scene = createAutocorrelatedScene(7, rows, [14]);
  const weights = createDistanceBandWeights(scene.positions, 14, {rowStandardize: true});
  const random = createSeededRandom(5);
  const values = new Float32Array(rows * periods);
  for (let period = 0; period < periods; period++) {
    for (let row = 0; row < rows; row++) {
      values[period * rows + row] = Math.fround(
        scene.values[row] * (1 + 0.15 * period) + random() * 3
      );
    }
  }
  const classes = Uint32Array.from({length: rows * periods}, () =>
    Math.floor(random() * classCount)
  );
  const mask = Uint32Array.from({length: rows}, (_, row) => (row % 11 === 0 ? 0 : 1));
  const rig = new AnalysisRig(device);
  const csr = {
    offsets: Array.from(weights.offsets),
    neighbors: Array.from(weights.neighbors),
    weights: Array.from(weights.weights)
  };
  const cells = lagClassCount * classCount * classCount;
  const counts = rig.output('uint32', cells);
  const lagBreaks = rig.output('float32', lagClassCount + 1);
  const lagClassCountView = rig.output('uint32', 1);
  const lagValues = rig.output('float32', rows * periods);
  const lagClasses = rig.output('uint32', rows * periods);
  rig.run(
    new GPUSpatialMarkov({
      values: rig.input(values, 'float32'),
      classes: rig.input(classes, 'uint32'),
      classCount,
      weights: rig.weights(csr),
      periodCount: periods,
      maximumLagClassCount: lagClassCount,
      lagParameters: rig.input(
        getGPUClassBreaksParameterValues(
          {method: 'quantile', classCount: lagClassCount},
          lagClassCount
        ),
        'float32'
      ),
      mask: rig.input(mask, 'uint32'),
      output: {
        counts,
        lagBreaks,
        lagClassCount: lagClassCountView,
        lagValues,
        lagClasses
      }
    })
  );
  // Lag against a CPU row-standardized lag over unmasked neighbors.
  const actualLag = await rig.readFloat(lagValues);
  const expectedLag: number[] = [];
  for (let period = 0; period < periods; period++) {
    for (let row = 0; row < rows; row++) {
      let sum = 0;
      let weightSum = 0;
      if (mask[row]) {
        for (let slot = csr.offsets[row]; slot < csr.offsets[row + 1]; slot++) {
          const neighbor = csr.neighbors[slot];
          if (mask[neighbor]) {
            sum += csr.weights[slot] * values[period * rows + neighbor];
            weightSum += csr.weights[slot];
          }
        }
      }
      expectedLag.push(weightSum > 0 ? sum / weightSum : 0);
    }
  }
  expectClose(actualLag, expectedLag, 'lag values', 1e-4);
  // Pooled quantile edges of the unmasked lag column.
  const pooled: number[] = [];
  for (let index = 0; index < rows * periods; index++) {
    if (mask[index % rows]) pooled.push(actualLag[index]);
  }
  pooled.sort((a, b) => a - b);
  const actualBreaks = await rig.readFloat(lagBreaks);
  const expectedEdges = [0, 1, 2, 3].map(index =>
    index === 0
      ? pooled[0]
      : index === lagClassCount
        ? pooled[pooled.length - 1]
        : getOracleQuantile(pooled, Math.fround(index / lagClassCount))
  );
  expectClose(actualBreaks, expectedEdges, 'pooled lag breaks', 1e-5);
  expect((await rig.readUint(lagClassCountView))[0]).toBe(lagClassCount);
  // Exact counts from the GPU lag values and edges.
  const lagClassOracle = actualLag.map((value, index) =>
    mask[index % rows] ? getOracleClass(value, actualBreaks, lagClassCount) : NO_CLASS
  );
  expect(await rig.readUint(lagClasses)).toEqual(lagClassOracle);
  const expected = computeTransitionOracle({
    classes,
    conditions: lagClassOracle,
    mask,
    rows,
    periods,
    classCount,
    conditionCount: lagClassCount
  });
  const actualCounts = await rig.readUint(counts);
  expect(actualCounts).toEqual(expected.counts);
  for (let condition = 0; condition < lagClassCount; condition++) {
    const slice = actualCounts.slice(
      condition * classCount * classCount,
      (condition + 1) * classCount * classCount
    );
    expect(slice.reduce((sum, value) => sum + value, 0)).toBeGreaterThan(0);
  }
  rig.destroy();
});

it('GPULISAMarkov counts transitions between per-period LISA quadrants', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = 120;
  const periods = 4;
  const scene = createAutocorrelatedScene(3, rows, [16]);
  const weights = createDistanceBandWeights(scene.positions, 16, {rowStandardize: true});
  const random = createSeededRandom(9);
  const values = new Float32Array(rows * periods);
  for (let period = 0; period < periods; period++) {
    for (let row = 0; row < rows; row++) {
      values[period * rows + row] = Math.fround(
        scene.values[row] * (period % 2 === 0 ? 1 : -0.6) + random() * 2 + period
      );
    }
  }
  const parameters = getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.1});
  expect(parameters.length).toBe(GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH);
  const rig = new AnalysisRig(device);
  const cells = GPU_LISA_MARKOV_STATE_COUNT ** 2;
  const counts = rig.output('uint32', cells);
  const rowTotals = rig.output('uint32', GPU_LISA_MARKOV_STATE_COUNT);
  const quadrants = rig.output('uint32', rows * periods);
  rig.run(
    new GPULISAMarkov({
      values: rig.input(values, 'float32'),
      weights: rig.weights({
        offsets: Array.from(weights.offsets),
        neighbors: Array.from(weights.neighbors),
        weights: Array.from(weights.weights)
      }),
      periodCount: periods,
      parameters: rig.input(parameters, 'float32'),
      gating: 'significance',
      output: {counts, rowTotals, quadrants}
    })
  );
  // Gated states: the quadrant of an analytically significant row, else 0; rows outside the
  // analysis (islands) are the out-of-range class instead of state 0.
  const expectedQuadrants = computeLisaStatesOracle({
    weights,
    values,
    rows,
    periods,
    gating: 'significance',
    significanceLevel: 0.1
  });
  const actualQuadrants = await rig.readUint(quadrants);
  expect(actualQuadrants).toEqual(expectedQuadrants);
  expect(actualQuadrants.filter(value => value > 0 && value < 5).length).toBeGreaterThan(10);
  expect(actualQuadrants.filter(value => value === 0).length).toBeGreaterThan(10);
  const expected = computeTransitionOracle({
    classes: expectedQuadrants,
    rows,
    periods,
    classCount: GPU_LISA_MARKOV_STATE_COUNT
  });
  expect(await rig.readUint(counts)).toEqual(expected.counts);
  expect(await rig.readUint(rowTotals)).toEqual(expected.rowTotals);
  rig.destroy();
});

// Reference from giddy 'Markov' (PySAL venv): 40 locations x 5 periods, 4 classes (seeded numpy).
const GIDDY_CLASSES = [
  2, 3, 1, 0, 1, 1, 3, 3, 1, 2, 1, 3, 0, 3, 3, 2, 3, 0, 3, 3, 2, 3, 3, 1, 2, 3, 0, 1, 1, 3, 1, 2, 2,
  1, 2, 2, 3, 1, 3, 2, 3, 3, 1, 2, 3, 0, 1, 2, 2, 0, 1, 3, 2, 2, 3, 0, 2, 2, 0, 0, 2, 1, 1, 0, 1, 0,
  3, 3, 0, 0, 2, 1, 0, 2, 2, 0, 0, 0, 1, 0, 3, 0, 2, 3, 0, 3, 2, 0, 1, 0, 3, 1, 2, 2, 0, 1, 0, 1, 3,
  3, 0, 2, 3, 3, 0, 1, 2, 3, 3, 2, 0, 2, 0, 1, 2, 0, 2, 0, 2, 3, 2, 1, 3, 0, 1, 3, 3, 1, 3, 2, 2, 0,
  2, 0, 1, 1, 3, 1, 0, 0, 0, 0, 1, 0, 3, 1, 0, 0, 0, 1, 0, 3, 2, 3, 2, 1, 1, 3, 2, 2, 3, 2, 2, 2, 3,
  3, 2, 2, 0, 1, 1, 0, 3, 1, 1, 0, 2, 1, 3, 1, 1, 3, 1, 1, 3, 1, 0, 3, 0, 0, 3, 2, 2, 1, 2, 0, 3, 2,
  2, 2
];
const GIDDY_COUNTS = [9, 11, 8, 14, 9, 9, 10, 8, 12, 9, 12, 7, 15, 7, 12, 8];
const GIDDY_P = [
  0.214286, 0.261905, 0.190476, 0.333333, 0.25, 0.25, 0.277778, 0.222222, 0.3, 0.225, 0.3, 0.175,
  0.357143, 0.166667, 0.285714, 0.190476
];

it('GPUTransitionMatrix reproduces giddy Markov counts and probabilities', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const counts = rig.output('uint32', 16);
  const probabilities = rig.output('float32', 16);
  rig.run(
    new GPUTransitionMatrix({
      classes: rig.input(Uint32Array.from(GIDDY_CLASSES), 'uint32'),
      periodCount: 5,
      classCount: 4,
      output: {counts, probabilities}
    })
  );
  expect(await rig.readUint(counts)).toEqual(GIDDY_COUNTS);
  expectClose(await rig.readFloat(probabilities), GIDDY_P, 'giddy p', 1e-5);
  rig.destroy();
});

it('GPUTransitionMatrix homogeneity, steady states and sparsity match giddy Spatial_Markov', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const periods = 5;
  const classCount = 3;
  const conditionCount = 3;
  const rig = new AnalysisRig(device);
  const counts = rig.output('uint32', conditionCount * classCount * classCount);
  const homogeneity = rig.output('float32', 5);
  const steadyStates = rig.output('float32', (conditionCount + 1) * classCount);
  const stateSparsity = rig.output('uint32', 2);
  const converged = rig.output('uint32', 1);
  rig.run(
    new GPUTransitionMatrix({
      classes: rig.input(Uint32Array.from(giddy.A_CLASSES), 'uint32'),
      periodCount: periods,
      classCount,
      conditionClasses: rig.input(Uint32Array.from(giddy.A_CONDITIONS), 'uint32'),
      conditionCount,
      minimumTransitions: giddy.A_SPARSITY_THRESHOLD,
      output: {counts, homogeneity, steadyStates, stateSparsity, status: {converged}}
    })
  );
  expect(await rig.readUint(counts)).toEqual(giddy.A_COUNTS);
  const actual = await rig.readFloat(homogeneity);
  const expected = giddy.A_HOMOGENEITY;
  // Q and LR are f32 sums; the p-values go through the f32 incomplete gamma (about 6e-5 relative).
  expectCloseOrNaN([actual[GPU_TRANSITION_HOMOGENEITY.Q]], [expected.Q], 'Q', 1e-4);
  expectCloseOrNaN([actual[GPU_TRANSITION_HOMOGENEITY.LR]], [expected.LR], 'LR', 1e-4);
  expect(actual[GPU_TRANSITION_HOMOGENEITY.DEGREES_OF_FREEDOM]).toBe(expected.dof);
  expectCloseOrNaN(
    [actual[GPU_TRANSITION_HOMOGENEITY.Q_P_VALUE]],
    [expected.Qp],
    'Q p',
    2e-3,
    1e-9
  );
  expectCloseOrNaN(
    [actual[GPU_TRANSITION_HOMOGENEITY.LR_P_VALUE]],
    [expected.LRp],
    'LR p',
    2e-3,
    1e-9
  );
  expectCloseOrNaN(await rig.readFloat(steadyStates), giddy.A_STEADY_STATES, 'steady states', 1e-4);
  const sparsity = await rig.readUint(stateSparsity);
  expect(sparsity[GPU_TRANSITION_STATE_SPARSITY.SPARSE]).toBe(giddy.A_SPARSITY.sparse);
  expect(sparsity[GPU_TRANSITION_STATE_SPARSITY.EMPTY]).toBe(giddy.A_SPARSITY.empty);
  expect((await rig.readUint(converged))[0]).toBe(1);
  rig.destroy();
});

it('GPUTransitionDiagnostics handles empty rows, periodic, reducible and dead-end chains like giddy', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const classCount = 4;
  for (const {name, counts: countValues, homogeneity: expected, steady, sparsity, converged} of [
    {
      name: 'ergodic and periodic conditions',
      counts: giddy.B_GOOD_COUNTS,
      homogeneity: giddy.B_GOOD_HOMOGENEITY,
      steady: giddy.B_GOOD_STEADY_STATES,
      sparsity: giddy.B_GOOD_SPARSITY,
      converged: 1
    },
    {
      name: 'plus a reducible and a dead-end condition',
      counts: giddy.B_ALL_COUNTS,
      homogeneity: giddy.B_ALL_HOMOGENEITY,
      steady: giddy.B_ALL_STEADY_STATES,
      sparsity: giddy.B_ALL_SPARSITY,
      converged: 0
    }
  ]) {
    const conditionCount = countValues.length / (classCount * classCount);
    const rig = new AnalysisRig(device);
    const homogeneity = rig.output('float32', 5);
    const steadyStates = rig.output('float32', (conditionCount + 1) * classCount);
    const stateSparsity = rig.output('uint32', 2);
    const convergedView = rig.output('uint32', 1);
    rig.run(
      new GPUTransitionDiagnostics({
        counts: rig.input(Uint32Array.from(countValues), 'uint32'),
        classCount,
        conditionCount,
        output: {homogeneity, steadyStates, stateSparsity, status: {converged: convergedView}}
      })
    );
    const actual = await rig.readFloat(homogeneity);
    // dof excludes the one from-state without any transition (giddy adds 1 for it, see fixture).
    expect(actual[GPU_TRANSITION_HOMOGENEITY.DEGREES_OF_FREEDOM], name).toBe(expected.dof);
    expectCloseOrNaN([actual[GPU_TRANSITION_HOMOGENEITY.Q]], [expected.Q], `${name} Q`, 1e-4);
    expectCloseOrNaN([actual[GPU_TRANSITION_HOMOGENEITY.LR]], [expected.LR], `${name} LR`, 1e-4);
    expectCloseOrNaN(
      [actual[GPU_TRANSITION_HOMOGENEITY.Q_P_VALUE]],
      [expected.Qp],
      `${name} Q p`,
      2e-3,
      1e-9
    );
    expectCloseOrNaN(
      [actual[GPU_TRANSITION_HOMOGENEITY.LR_P_VALUE]],
      [expected.LRp],
      `${name} LR p`,
      2e-3,
      1e-9
    );
    expectCloseOrNaN(await rig.readFloat(steadyStates), steady, `${name} steady states`, 1e-4);
    const actualSparsity = await rig.readUint(stateSparsity);
    expect(actualSparsity[GPU_TRANSITION_STATE_SPARSITY.SPARSE]).toBe(sparsity.sparse);
    expect(actualSparsity[GPU_TRANSITION_STATE_SPARSITY.EMPTY]).toBe(sparsity.empty);
    expect((await rig.readUint(convergedView))[0], `${name} converged`).toBe(converged);
    rig.destroy();
  }
});

/** Row-standardized rook weights of a `side x side` grid plus one isolated row at the end. */
function createGridWeightsWithIsland(side: number): {
  offsets: number[];
  neighbors: number[];
  weights: number[];
} {
  const offsets = [0];
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < side * side; row++) {
    const r = Math.floor(row / side);
    const c = row % side;
    const list: number[] = [];
    if (r > 0) list.push(row - side);
    if (c > 0) list.push(row - 1);
    if (c < side - 1) list.push(row + 1);
    if (r < side - 1) list.push(row + side);
    for (const neighbor of list) {
      neighbors.push(neighbor);
      weights.push(Math.fround(1 / list.length));
    }
    offsets.push(neighbors.length);
  }
  // Island: no neighbors.
  offsets.push(neighbors.length);
  return {offsets, neighbors, weights};
}

it('GPULISAMarkov excludes NaN rows and islands and matches giddy LISA_Markov ungated', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const side = 6;
  const rows = side * side + 1;
  const periods = 4;
  const csr = createGridWeightsWithIsland(side);
  const oracleWeights = {
    offsets: Uint32Array.from(csr.offsets),
    neighbors: Uint32Array.from(csr.neighbors),
    weights: Float32Array.from(csr.weights)
  };
  for (const {name, values: fixtureValues, counts: expectedCounts, states, ignored} of [
    {
      // Row 7 is NaN in every period: giddy LISA_Markov over the 34 complete rows (row 7 and the
      // island removed, weights rebuilt and row-standardized) has the same counts.
      name: 'NaN row in every period',
      values: giddy.C1_VALUES,
      counts: giddy.C1_COUNTS,
      states: giddy.C1_STATES,
      ignored: giddy.C1_IGNORED
    },
    {
      // Row 20 is NaN in period 2 only: expected states come from esda Moran_Local per period on
      // that period's valid rows, counted over pairs valid at both ends (giddy has no missing data).
      name: 'NaN row in one period',
      values: giddy.C2_VALUES,
      counts: giddy.C2_COUNTS,
      states: giddy.C2_STATES,
      ignored: giddy.C2_IGNORED
    }
  ]) {
    const values = Float32Array.from(fixtureValues);
    const rig = new AnalysisRig(device);
    const counts = rig.output('uint32', 25);
    const quadrants = rig.output('uint32', rows * periods);
    const ignoredView = rig.output('uint32', 1);
    const steadyStates = rig.output('float32', GPU_LISA_MARKOV_STATE_COUNT);
    const stateSparsity = rig.output('uint32', 2);
    rig.run(
      new GPULISAMarkov({
        values: rig.input(values, 'float32'),
        weights: {
          ...rig.weights(csr),
          information: {rowCount: rows, transform: 'row', symmetric: false}
        },
        periodCount: periods,
        parameters: rig.input(getGPUSpatialAutocorrelationParameterValues({}), 'float32'),
        output: {counts, quadrants, ignored: ignoredView, steadyStates, stateSparsity}
      })
    );
    const expectedStates = states.map(state => (state < 0 ? NO_CLASS : state));
    const actualStates = await rig.readUint(quadrants);
    expect(actualStates, `${name} states`).toEqual(expectedStates);
    // Never state 0 under the default ungated chain, and no missing row is state 0.
    expect(actualStates.filter(state => state === 0).length).toBe(0);
    expect(await rig.readUint(counts), `${name} counts`).toEqual(expectedCounts);
    expect((await rig.readUint(ignoredView))[0], `${name} ignored`).toBe(ignored);
    // The CPU oracle built on the double-precision local Moran oracle agrees with the fixture.
    expect(
      computeLisaStatesOracle({
        weights: oracleWeights,
        values,
        rows,
        periods,
        gating: 'none',
        rowStandardized: true
      }),
      `${name} oracle`
    ).toEqual(expectedStates);
    // Steady state of the 4-state chain: state 0 is unused (probability 0).
    if (name === 'NaN row in every period') {
      expectCloseOrNaN(
        await rig.readFloat(steadyStates),
        giddy.C1_STEADY_STATE,
        'LISA steady',
        1e-4
      );
    }
    rig.destroy();
  }
});
