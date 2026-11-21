// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPULocationChangePoints} from '../../../src/gpu-spatial-analysis/change-points/gpu-location-change-points';
import {GPUPettittTest} from '../../../src/gpu-spatial-analysis/change-points/gpu-pettitt-test';
import {GPUPettittTestOperation} from '../../../src/gpu-spatial-analysis/change-points/gpu-pettitt-test-operation';
import {CPUPettittTestOperation} from '../../../src/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation';
import {CPULocationChangePointsOperation} from '../../../src/gpu-spatial-analysis/change-points/cpu-location-change-points-operation';
import {GPULocationChangePointsOperation} from '../../../src/gpu-spatial-analysis/change-points/gpu-location-change-points-operation';
import {computeLocationChangePointsOnCPU} from '../../../src/gpu-spatial-analysis/change-points/cpu-location-change-points-operation';
import {computePettittTestOnCPU} from '../../../src/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';
import {
  AUDIT_CHANGE_CASES,
  AUDIT_CHANGE_SERIES,
  AUDIT_CHANGE_AFFINE_COSTS,
  AUDIT_PETTITT_EXPECTED,
  AUDIT_PETTITT_SERIES,
  REVIEW_TWO_CHANGE_CASES,
  TRANSLATED_CHANGE_TIES,
  WIDE_MEAN_COST_EXPECTED,
  WIDE_MEAN_COST_VALUES,
  WIDE_MEAN_SQUARES_EXPECTED,
  WIDE_MEAN_SQUARES_VALUES,
  WIDE_STANDARDIZATION_CASES,
  packAuditSeries
} from './audit-change-points-fixture';

it('W5-21 preserves finite mean costs for large positive and negative totals', () => {
  const actual = computeLocationChangePointsOnCPU(WIDE_MEAN_COST_VALUES, Float32Array.of(1, 8), {
    layout: 'cell-major',
    cellCount: 3,
    sliceCount: 8,
    cost: 'mean',
    maximumChangePoints: 1
  });
  expectNumericArrayClose(actual.changePointCount, [0, 0, 0]);
  expectNumericArrayClose(actual.optimalCost, WIDE_MEAN_COST_EXPECTED);
});

it('W5-21 CPU preserves finite mean costs with wide squared offsets', () => {
  const actual = computeLocationChangePointsOnCPU(WIDE_MEAN_SQUARES_VALUES, Float32Array.of(1, 5), {
    layout: 'cell-major',
    cellCount: 4,
    sliceCount: 5,
    cost: 'mean',
    maximumChangePoints: 1
  });
  expectNumericArrayClose(actual.changePointCount, [0, 0, 0, 0]);
  expectNumericArrayClose(actual.optimalCost, Float32Array.from(WIDE_MEAN_SQUARES_EXPECTED));
});

for (const fixture of WIDE_STANDARDIZATION_CASES) {
  it(`B7-CP-02 CPU standardizes wide ${fixture.cost} inputs`, () => {
    const actual = computeLocationChangePointsOnCPU(
      fixture.values,
      Float32Array.of(1, fixture.minimumLength),
      {
        layout: 'cell-major',
        cellCount: fixture.counts.length,
        sliceCount: fixture.sliceCount,
        cost: fixture.cost,
        maximumChangePoints: 1
      }
    );
    expectNumericArrayClose(actual.changePointCount, fixture.counts);
    expectNumericArrayClose(actual.requiredChangePointCounts, fixture.counts);
    expectNumericArrayClose(actual.changePoints, fixture.points);
    expectNumericArrayClose(actual.overflow, [0]);
    expectNumericArrayClose(actual.optimalCost, fixture.expectedCost, {relativeTolerance: 2e-7});
  });
}

for (const fixture of REVIEW_TWO_CHANGE_CASES) {
  it(`review 2 CPU ${fixture.name}`, () => {
    const actual = computeLocationChangePointsOnCPU(fixture.values, fixture.parameters, {
      layout: 'cell-major',
      cellCount: fixture.points.length,
      sliceCount: fixture.sliceCount,
      cost: fixture.cost,
      maximumChangePoints: 1
    });
    expectNumericArrayClose(actual.changePoints, fixture.points);
    expectNumericArrayClose(
      actual.changePointCount,
      fixture.points.map(() => 1)
    );
    expectNumericArrayClose(actual.requiredChangePointCounts, fixture.required);
    expectNumericArrayClose(actual.optimalCost, fixture.expected, {
      relativeTolerance: Math.min(2e-7, fixture.relativeTolerance)
    });
    expectNumericArrayClose(actual.overflow, [fixture.overflow]);
  });
}

it('B1 float32 slope transcription keeps slice 4 across the exact translation', () => {
  const fixture = TRANSLATED_CHANGE_TIES.find(fixture => fixture.cost === 'slope')!;
  for (const translation of [0, fixture.translation]) {
    const values = fixture.source.map(value => value * fixture.amplitude + translation);
    for (const reciprocal of [false, true]) {
      const result = transcribeSlopeTie(values, fixture.penalty, reciprocal);
      expect(result.points).toEqual([fixture.point]);
      expect(result.cost).toBe(fixture.expected[0]);
    }
  }
});

function transcribeSlopeTie(values: number[], penalty: number, reciprocal: boolean) {
  const round = Math.fround;
  const divide = (numerator: number, denominator: number) =>
    round(reciprocal ? numerator * round(1 / denominator) : numerator / denominator);
  const sum = (terms: number[]) => round(terms.reduce((total, term) => total + term, 0));
  const maximum = Math.max(...values.map(Math.abs));
  const shift = maximum >= 2 ** 54 ? Math.floor(Math.log2(maximum)) - 50 : 0;
  const factor = 2 ** -shift;
  const scaled = values.map(value => round(value * factor));
  const relative = scaled.map(value => round(value - scaled[0]));
  const center = divide(sum(relative), values.length);
  const deviations = relative.map(value => round(value - center));
  const variance = divide(sum(deviations.map(value => round(value * value))), values.length);
  const deviationScale = round(Math.sqrt(variance));
  const standardized = deviations.map(value => divide(value, deviationScale));
  const scaledPenalty = round(round(divide(penalty, variance) * factor) * factor);
  const segment = (first: number, last: number) => {
    const length = last - first;
    let total = 0;
    let squares = 0;
    let timeValue = 0;
    let timeFirst = 0;
    let timeSecond = 0;
    for (let time = 0; time < length; time++) {
      const row = first + time;
      const offset = shift
        ? divide(round(scaled[row] - scaled[first]), deviationScale)
        : round(standardized[row] - standardized[first]);
      total = round(total + offset);
      squares = round(squares + round(offset * offset));
      timeValue = round(timeValue + round(time * offset));
      timeFirst += time;
      timeSecond += time * time;
    }
    const centeredSquares = round(squares - divide(round(total * total), length));
    const timeVariance = divide(length * timeSecond - timeFirst * timeFirst, length);
    const covariance = round(timeValue - divide(round(timeFirst * total), length));
    return Math.max(
      0,
      round(centeredSquares - divide(round(covariance * covariance), timeVariance))
    );
  };
  let points: number[] = [];
  let best = segment(0, values.length);
  for (const candidatePoints of [[3], [4], [5], [6], [3, 6]]) {
    let prefix = 0;
    let first = 0;
    for (const last of [...candidatePoints, values.length]) {
      prefix = round(round(prefix + segment(first, last)) + (first ? scaledPenalty : 0));
      first = last;
    }
    if (prefix < best) {
      best = prefix;
      points = candidatePoints;
    }
  }
  return {points, cost: round(best * variance) * 2 ** (2 * shift)};
}

it('retains all required mean changes in a mixed-sign, large-dynamic-range series', () => {
  const actual = computeLocationChangePointsOnCPU(
    Float32Array.of(1e7, -1e7, 1, 1, 1.25, 1.25, 1, 1),
    Float32Array.of(1 / 64, 1),
    {layout: 'cell-major', cellCount: 1, sliceCount: 8, cost: 'mean', maximumChangePoints: 2}
  );
  expectNumericArrayClose(actual.changePointCount, [2]);
  expectNumericArrayClose(actual.changePoints, [1, 2]);
  expectNumericArrayClose(actual.requiredChangePointCounts, [4]);
  expectNumericArrayClose(actual.optimalCost, [0.0625]);
  expectNumericArrayClose(actual.overflow, [1]);
});

it('reads an accessor-backed CPU layout once before validating and indexing', () => {
  let reads = 0;
  const options = {
    get layout() {
      reads++;
      return (reads === 1 ? 'cell-major' : 'time-major') as 'cell-major';
    },
    cellCount: 2,
    sliceCount: 4,
    cost: 'mean' as const,
    maximumChangePoints: 3
  };
  const actual = computeLocationChangePointsOnCPU(
    Float32Array.of(1, 1, 8, 8, 9, 9, 2, 2),
    Float32Array.of(3, 1),
    options
  );
  expect(Array.from(actual.changePointCount)).toEqual([1, 1]);
  expectNumericArrayClose(actual.optimalCost, [3, 3]);
  expect(reads).toBe(1);
});

it('rejects misspelled layouts at construction in every contributor/operation and both CPU helpers', () => {
  const layout = 'time-major' as 'cell-major';
  const location = {
    values: 'values',
    parameters: 'parameters',
    layout,
    cellCount: 2,
    sliceCount: 4,
    cost: 'mean' as const,
    maximumChangePoints: 1,
    output: {changePointCount: 'count', changePoints: 'points'}
  };
  const pettitt = {
    values: 'values',
    layout,
    cellCount: 2,
    sliceCount: 4,
    output: {statistic: 'statistic', changeSlice: 'change', pValues: 'p'}
  };
  for (const Operation of [GPULocationChangePointsOperation, CPULocationChangePointsOperation])
    expect(() => new Operation(location)).toThrow(/layout/);
  for (const Operation of [GPUPettittTestOperation, CPUPettittTestOperation])
    expect(() => new Operation(pettitt)).toThrow(/layout/);
  const values = Float32Array.of(1, 1, 8, 8, 9, 9, 2, 2);
  expect(() => computeLocationChangePointsOnCPU(values, Float32Array.of(3, 1), location)).toThrow(
    /layout/
  );
  expect(() => computePettittTestOnCPU(values, pettitt)).toThrow(/layout/);
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const input = createTransientView(graph, 'values', 'float32', 8);
    expect(
      () =>
        new GPULocationChangePoints({
          ...location,
          values: input,
          parameters: createTransientView(graph, 'parameters', 'float32', 2),
          output: {
            changePointCount: createTransientView(graph, 'count', 'uint32', 2),
            changePoints: createTransientView(graph, 'points', 'uint32', 2)
          }
        })
    ).toThrow(/layout/);
    expect(
      () =>
        new GPUPettittTest({
          ...pettitt,
          values: input,
          output: {
            statistic: createTransientView(graph, 'statistic', 'uint32', 2),
            changeSlice: createTransientView(graph, 'change', 'uint32', 2),
            pValues: createTransientView(graph, 'p', 'float32', 2)
          }
        })
    ).toThrow(/layout/);
  } finally {
    device.destroy();
  }
});

it('retains the true variance partition under a large representable translation', () => {
  for (const offset of [0, 1000000]) {
    const actual = computeLocationChangePointsOnCPU(
      Float32Array.from([0, 0, 0, 0, 0.0625, 0.0625, 0.0625, 0.0625], value => value + offset),
      Float32Array.of(3, 2),
      {layout: 'cell-major', cellCount: 1, sliceCount: 8, cost: 'variance', maximumChangePoints: 2}
    );
    expectNumericArrayClose(actual.changePointCount, [0]);
    expectNumericArrayClose(actual.changePoints, [0xffffffff, 0xffffffff]);
    expectNumericArrayClose(actual.requiredChangePointCounts, [0]);
    expectNumericArrayClose(actual.optimalCost, [-32.74875791352086], {absoluteTolerance: 2e-5});
    expectNumericArrayClose(actual.overflow, [0]);
  }
});

for (const compute of [computeLocationChangePointsOnCPU]) {
  it(`audits all five partition costs and every output by exhaustive enumeration: ${compute.name}`, () => {
    for (const {cost, minimumLength, expected} of AUDIT_CHANGE_CASES) {
      for (const layout of ['cell-major', 'period-major'] as const) {
        for (const affine of [false, true]) {
          const penalty = affine
            ? cost === 'poisson'
              ? 6
              : cost === 'mean' || cost === 'slope'
                ? 12
                : 3
            : 3;
          const target = {
            ...expected,
            optimalCost: affine ? AUDIT_CHANGE_AFFINE_COSTS[cost] : expected.optimalCost
          };
          const actual = compute(
            packAuditSeries(AUDIT_CHANGE_SERIES, layout, value =>
              affine ? 2 * value + (cost === 'poisson' ? 0 : 3) : value
            ),
            Float32Array.of(penalty, minimumLength),
            {cost, layout, cellCount: 5, sliceCount: 8, maximumChangePoints: 1}
          );
          for (const port of Object.keys(expected) as (keyof typeof expected)[]) {
            expectNumericArrayClose(actual[port], target[port], {
              absoluteTolerance: port === 'optimalCost' ? 2e-5 : 0,
              relativeTolerance: port === 'optimalCost' ? 1e-7 : 0,
              label: `${cost}, ${layout}, ${port}`
            });
          }
        }
      }
    }
  });
}

it('audits every Pettitt CPU output with cross-split sign sums, ties, missing slices and affine invariance', () => {
  // DGN-01: the pinned constant is tested (p=1); zero/single-observation rows are untested (NaN).
  for (const layout of ['cell-major', 'period-major'] as const) {
    for (const affine of [false, true]) {
      const actual = computePettittTestOnCPU(
        packAuditSeries(AUDIT_PETTITT_SERIES, layout, value => (affine ? 2 * value + 3 : value)),
        {layout, cellCount: 5, sliceCount: 8}
      );
      for (const port of Object.keys(
        AUDIT_PETTITT_EXPECTED
      ) as (keyof typeof AUDIT_PETTITT_EXPECTED)[]) {
        expectNumericArrayClose(actual[port], AUDIT_PETTITT_EXPECTED[port], {
          absoluteTolerance: port === 'pValues' ? 1e-7 : 0,
          label: port
        });
      }
    }
  }
});

it('retains the unsplit RSS when the penalty dwarfs the residual', () => {
  const actual = computeLocationChangePointsOnCPU(
    Float32Array.of(0, 1),
    Float32Array.of(2 ** 60, 1),
    {cost: 'mean', layout: 'cell-major', cellCount: 1, sliceCount: 2, maximumChangePoints: 1}
  );
  // One-segment mean 1/2, RSS = (1/2)^2 + (1/2)^2 = 1/2; no penalty applies.
  expectNumericArrayClose(actual.optimalCost, [0.5]);
  expectNumericArrayClose(actual.changePointCount, [0]);
  expectNumericArrayClose(actual.changePoints, [0xffffffff]);
  expectNumericArrayClose(actual.requiredChangePointCounts, [0]);
  expectNumericArrayClose(actual.overflow, [0]);
});

it('uses the stated Poisson likelihood for one finite observation', () => {
  const actual = computeLocationChangePointsOnCPU(Float32Array.of(7, NaN), Float32Array.of(3, 1), {
    cost: 'poisson',
    layout: 'cell-major',
    cellCount: 1,
    sliceCount: 2,
    maximumChangePoints: 1
  });
  // C = 2*x*(ln(n)-ln(x)), x=7, n=1; no change penalty.
  expectNumericArrayClose(actual.optimalCost, [-27.242742086774385], {absoluteTolerance: 2e-6});
});

it('rejects inherited object names as unsupported segment costs at operation construction', () => {
  expect(
    () =>
      new GPULocationChangePointsOperation({
        values: 'values',
        parameters: 'parameters',
        layout: 'cell-major',
        cellCount: 1,
        sliceCount: 8,
        cost: 'constructor' as 'mean',
        maximumChangePoints: 1,
        output: {changePointCount: 'counts', changePoints: 'changes'}
      })
  ).toThrow(/location-change-points cost/);
});
