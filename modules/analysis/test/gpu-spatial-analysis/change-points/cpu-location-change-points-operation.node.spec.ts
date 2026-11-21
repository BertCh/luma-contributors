// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {AnalysisValidationError} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import fixture from '../../oracles/fixtures/stx06-change-points.json';
import {
  computeLocationChangePointsOnCPU,
  CPULocationChangePointsOperation
} from '../../../src/gpu-spatial-analysis/change-points/cpu-location-change-points-operation';
import {
  GPULocationChangePointsOperation,
  type LocationChangePointsOperationProps
} from '../../../src/gpu-spatial-analysis/change-points/gpu-location-change-points-operation';
import type {GPULocationChangePointCost} from '../../../src/gpu-spatial-analysis/change-points/gpu-location-change-points';

const PROPS: LocationChangePointsOperationProps = {
  id: 'changes-check',
  values: 'values',
  parameters: 'parameters',
  cellCount: 1,
  sliceCount: 8,
  cost: 'mean',
  layout: 'cell-major',
  maximumChangePoints: 2,
  output: {
    changePointCount: 'counts',
    changePoints: 'changes',
    requiredChangePointCounts: 'required',
    optimalCost: 'optimum',
    status: {overflow: 'overflow'}
  }
};

// Exhaustive partition enumeration, with direct two-pass residuals instead of running moments.
function findExhaustiveOptimum(
  values: number[],
  cost: GPULocationChangePointCost,
  penalty: number,
  minimumLength: number
): number {
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const variance = values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length;
  let best = Infinity;
  for (let boundaries = 0; boundaries < 2 ** (values.length - 1); boundaries++) {
    const edges = [0];
    for (let index = 1; index < values.length; index++)
      if (boundaries & (1 << (index - 1))) edges.push(index);
    edges.push(values.length);
    if (edges.some((edge, index) => index > 0 && edge - edges[index - 1] < minimumLength)) continue;
    let objective = penalty * (edges.length - 2);
    for (let segment = 1; segment < edges.length; segment++) {
      const start = edges[segment - 1];
      const sample = values.slice(start, edges[segment]);
      const count = sample.length;
      const total = sample.reduce((sum, value) => sum + value, 0);
      const localMean = total / count;
      const squares = sample.reduce((sum, value) => sum + (value - localMean) ** 2, 0);
      if (cost === 'mean') objective += squares;
      else if (cost === 'poisson') objective += total ? -2 * total * Math.log(total / count) : 0;
      else if (cost === 'slope') {
        const timeMean = start + (count - 1) / 2;
        const timeSquares = sample.reduce(
          (sum, _, index) => sum + (start + index - timeMean) ** 2,
          0
        );
        const covariance = sample.reduce(
          (sum, value, index) => sum + (start + index - timeMean) * (value - localMean),
          0
        );
        const slope = covariance / timeSquares;
        objective += sample.reduce(
          (sum, value, index) =>
            sum + (value - localMean - slope * (start + index - timeMean)) ** 2,
          0
        );
      } else {
        const residual =
          cost === 'variance'
            ? sample.reduce((sum, value) => sum + (value - mean) ** 2, 0)
            : squares;
        objective +=
          count * (Math.log(2 * Math.PI * Math.max(residual / count, 1e-11 * variance)) + 1);
      }
    }
    best = Math.min(best, objective);
  }
  return best;
}

it('matches exhaustive partitions for every segment cost', () => {
  const values = [1, 2, 1, 3, 9, 14, 10, 12];
  for (const cost of ['mean', 'variance', 'mean-variance', 'poisson', 'slope'] as const) {
    const minimumLength = cost === 'slope' ? 3 : 2;
    const result = computeLocationChangePointsOnCPU(
      Float32Array.from(values),
      new Float32Array([3, minimumLength]),
      {...PROPS, cost}
    );
    expect(result.optimalCost[0]).toBeCloseTo(
      findExhaustiveOptimum(values, cost, 3, minimumLength),
      4
    );
  }
});

it('resolves the small Poisson improvement on a large count baseline', () => {
  // Mean r = 2^30 + 128, observations r +/- 128. The split gains
  // 2*r*(u^2 + u^4/6 + ...), u = 128/r: 3.05175745e-5, above penalty 2^-16.
  const result = computeLocationChangePointsOnCPU(
    new Float32Array([1073741824, 1073742080]),
    new Float32Array([2 ** -16, 1]),
    {...PROPS, cost: 'poisson', sliceCount: 2, maximumChangePoints: 1}
  );
  expect(Array.from(result.changePointCount)).toEqual([1]);
  expect(Array.from(result.requiredChangePointCounts)).toEqual([1]);
  expect(Array.from(result.changePoints)).toEqual([1]);
  expect(Array.from(result.overflow)).toEqual([0]);
  // The likelihood constant is about -8.93e10; the gain and penalty cannot affect its f32 rounding.
  expect(result.optimalCost[0]).toBe(-89311346688);
});

it('keeps the unsplit residual when the change penalty is much larger', () => {
  // The one-segment mean is 1/2 and RSS = (0 - 1/2)^2 + (1 - 1/2)^2 = 1/2.
  const result = computeLocationChangePointsOnCPU(
    new Float32Array([0, 1]),
    new Float32Array([2 ** 60, 1]),
    {...PROPS, sliceCount: 2, maximumChangePoints: 1}
  );
  expect(Array.from(result.changePointCount)).toEqual([0]);
  expect(Array.from(result.requiredChangePointCounts)).toEqual([0]);
  expect(Array.from(result.changePoints)).toEqual([0xffffffff]);
  expect(Array.from(result.overflow)).toEqual([0]);
  expect(result.optimalCost[0]).toBe(0.5);
});

it('enforces GPU operation limits at construction and retains the larger CPU shape', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {
      values: {format: 'float32', length: 513},
      parameters: {format: 'float32', length: 2},
      changePointCount: {format: 'uint32', length: 1},
      changePoints: {format: 'uint32', length: 2},
      requiredChangePointCounts: {format: 'uint32', length: 1},
      optimalCost: {format: 'float32', length: 1},
      overflow: {format: 'uint32', length: 1}
    });
    const props = {...PROPS, sliceCount: 513};
    expect(() => new GPULocationChangePointsOperation(props).createContributor(getView)).toThrow(
      /sliceCount/
    );
    expect(() => new CPULocationChangePointsOperation(props)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('matches the independent ruptures fixtures for mean, mean-variance and slope', () => {
  for (const location of fixture.locations) {
    for (const cost of ['mean', 'mean-variance', 'slope'] as const) {
      const reference = location.changePoints[cost];
      const result = computeLocationChangePointsOnCPU(
        Float32Array.from(location.series[cost]),
        new Float32Array([reference.penalty, reference.minimumSegmentLength]),
        {...PROPS, cost, sliceCount: fixture.length, maximumChangePoints: 12}
      );
      expect(result.requiredChangePointCounts[0]).toBe(reference.breakpoints.length);
      expect(Array.from(result.changePoints.subarray(0, result.changePointCount[0]))).toEqual(
        reference.breakpoints
      );
      expect(Math.abs(result.optimalCost[0] - reference.optimalCost)).toBeLessThan(
        1e-6 * Math.abs(reference.optimalCost)
      );
    }
  }
});

it('keeps earliest overflow changes, true gap positions, and exact tie order', () => {
  const overflow = computeLocationChangePointsOnCPU(
    new Float32Array([0, 8, 0, 8, 0, 8, 0, 8]),
    new Float32Array([0.25, 1]),
    PROPS
  );
  expect(Array.from(overflow.changePoints)).toEqual([1, 2]);
  expect(overflow.changePointCount[0]).toBe(2);
  expect(overflow.requiredChangePointCounts[0]).toBe(7);
  expect(overflow.overflow[0]).toBe(1);
  expect(overflow.optimalCost[0]).toBe(1.75);
  const gappy = computeLocationChangePointsOnCPU(
    new Float32Array([0, NaN, 0, 0, NaN, 8, 8, 8]),
    new Float32Array([1, 2]),
    PROPS
  );
  expect(Array.from(gappy.changePoints)).toEqual([5, 0xffffffff]);
  const tie = computeLocationChangePointsOnCPU(
    new Float32Array([0, 1, 2]),
    new Float32Array([1.5, 1]),
    {...PROPS, sliceCount: 3}
  );
  expect(tie.changePointCount[0]).toBe(0); // no split and split are both cost 2: smallest start is 0.
});

it('supports both layouts, flat/empty series, relative floors and series beyond GPU caps', () => {
  for (const layout of ['cell-major', 'period-major'] as const) {
    const series = [
      [3, 3, 3],
      [NaN, NaN, NaN],
      [1, 4, 9]
    ];
    const values = Float32Array.from(
      layout === 'cell-major'
        ? series.flat()
        : Array.from({length: 3}, (_, slice) => series.map(row => row[slice])).flat()
    );
    const result = computeLocationChangePointsOnCPU(values, new Float32Array([100, 2]), {
      ...PROPS,
      layout,
      cellCount: 3,
      sliceCount: 3
    });
    expect(Array.from(result.optimalCost)).toEqual([0, NaN, Math.fround(98 / 3)]);
  }
  const flat = computeLocationChangePointsOnCPU(
    new Float32Array([4, 4]),
    new Float32Array([1, 2]),
    {...PROPS, cost: 'variance', sliceCount: 2}
  );
  expect(flat.optimalCost[0]).toBeCloseTo(2 * (Math.log(2 * Math.PI) + Math.log(1e-11) + 1), 4);
  const long = computeLocationChangePointsOnCPU(
    Float32Array.from({length: 600}, (_, slice) => (slice < 300 ? 0 : 10)),
    new Float32Array([1, 2]),
    {...PROPS, sliceCount: 600}
  );
  expect(Array.from(long.changePoints)).toEqual([300, 0xffffffff]);
  expect(long.optimalCost[0]).toBe(1);
});

it('checks arrays and both operation constructors', () => {
  expect(() =>
    computeLocationChangePointsOnCPU(new Float32Array(7), new Float32Array([1, 2]), PROPS)
  ).toThrow('values');
  expect(() =>
    computeLocationChangePointsOnCPU(new Float32Array(8), new Float32Array(1), PROPS)
  ).toThrow('parameters');
  for (const Operation of [GPULocationChangePointsOperation, CPULocationChangePointsOperation]) {
    expect(() => new Operation({...PROPS, cellCount: 0})).toThrow(/^changes-check/);
    expect(
      () => new Operation({...PROPS, output: {...PROPS.output, changePoints: 'values'}})
    ).toThrow(/^changes-check/);
    expect(new Operation(PROPS).id).toBe('changes-check');
  }
  expect(() => new CPULocationChangePointsOperation({...PROPS, cost: 'unknown' as 'mean'})).toThrow(
    /^changes-check.*cost/
  );
  expect(() => new CPULocationChangePointsOperation({...PROPS, sliceCount: 0})).toThrow(
    /^changes-check.*sliceCount/
  );
});

it('W2-15 validates both operation forms at construction with the caller ID', () => {
  for (const Operation of [GPULocationChangePointsOperation, CPULocationChangePointsOperation]) {
    expect(() => new Operation({...PROPS, id: 'my-change', cost: 'bogus' as 'mean'})).toThrow(
      AnalysisValidationError
    );
    expect(() => new Operation({...PROPS, id: 'my-change', cost: 'bogus' as 'mean'})).toThrow(
      /^my-change.*cost.*mean.*variance.*poisson.*slope.*bogus/
    );
  }
  expect(() =>
    computeLocationChangePointsOnCPU(new Float32Array(7), Float32Array.of(1, 2), {
      ...PROPS,
      id: 'my-cpu-change'
    })
  ).toThrow(/^my-cpu-change.*values.*8.*7/);
});
