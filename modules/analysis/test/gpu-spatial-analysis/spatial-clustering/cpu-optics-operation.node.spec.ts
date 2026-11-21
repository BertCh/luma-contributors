// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createOPTICSStatusFields, getOPTICSStatusValues} from './optics-status-test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computeOPTICSOnCPU,
  CPUOPTICSOperation,
  type CPUOPTICSOptions
} from '../../../src/gpu-spatial-analysis/spatial-clustering/cpu-optics-operation';
import {
  GPUOPTICSOperation,
  type OPTICSOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics-operation';
import {
  getGPUSphereRadius,
  resolveGPUSpatialContext,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
} from '../../../src/analysis-core/contracts/spatial-context';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/temporal-context';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import fixture from './optics-operation-fixtures.json';

const TEMPORAL_CONTEXT: GPUTemporalContext = {
  encoding: 'float32-relative',
  unit: 'second',
  epoch: 0n,
  axis: {kind: 'irregular'}
};
const PROPS: OPTICSOperationProps = {
  id: 'optics-check',
  positions: 'positions',
  rowCount: 3,
  minimumSamples: 2,
  maximumDistance: 4,
  extractionDistance: 2,
  output: {
    ordering: 'ordering',
    coreDistances: 'coreDistances',
    reachability: 'reachability',
    predecessors: 'predecessors',
    labels: 'labels',
    ports: {status: createOPTICSStatusFields(name => name)}
  }
};

it.each([
  'dbscan',
  'xi'
] as const)('X7 respects the explicit visit budget before %s extraction', extractionMethod => {
  const positions = Float32Array.of(0, 0, 1, 0, 4, 0, 5, 0);
  for (const budget of [1, 4]) {
    const result = computeOPTICSOnCPU(positions, {
      minimumSamples: 2,
      maximumDistance: 2,
      extractionDistance: 1.5,
      extractionMethod,
      maximumIterations: budget
    });
    expect(Array.from(result.ordering)).toEqual(
      budget === 1 ? [0, 0xffffffff, 0xffffffff, 0xffffffff] : [0, 1, 2, 3]
    );
    expect(getOPTICSStatusValues(result)).toEqual([budget, budget === 4 ? 1 : 0, 0]);
    expect(Array.from(result.iterationCount)).toEqual([budget]);
    expect(Array.from(result.converged)).toEqual([budget === 4 ? 1 : 0]);
    expect(Array.from(result.iterationLimitReached)).toEqual([budget === 1 ? 1 : 0]);
    expect(Array.from(result.labels)).toEqual(
      budget === 1 ? [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff] : [0, 0, 1, 1]
    );
    if (budget === 1) {
      expect(Array.from(result.clusterSummary)).toEqual([0, 0]);
      expect(result.clusterIntervals.every(value => value === 0xffffffff)).toBe(true);
    }
  }
});

it('X7 CPU operation retains its validated non-enumerable visit budget', async () => {
  let budget = 1;
  const props = {...PROPS, rowCount: 4};
  Object.defineProperty(props, 'maximumIterations', {get: () => budget});
  const operation = new CPUOPTICSOperation(props);
  budget = 4;
  const values = Float32Array.of(0, 0, 1, 0, 4, 0, 5, 0);
  const result = await operation.execute({
    inputs: {
      positions: {format: 'float32x2', length: 4, chunks: [{values, length: 4}], pack: () => values}
    }
  } as Parameters<typeof operation.execute>[0]);
  expect(Array.from(result['ordering'] as Uint32Array)).toEqual([
    0, 0xffffffff, 0xffffffff, 0xffffffff
  ]);
});

it('X7 GPU operation retains the visit budget until contributor lowering', () => {
  const props = {...PROPS, rowCount: 4, maximumIterations: 1};
  const operation = new GPUOPTICSOperation(props);
  props.maximumIterations = 4;
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {
      positions: {format: 'float32x2', length: 4},
      ordering: {format: 'uint32', length: 4},
      coreDistances: {format: 'float32', length: 4},
      reachability: {format: 'float32', length: 4},
      predecessors: {format: 'uint32', length: 4},
      labels: {format: 'uint32', length: 4},
      ...createOPTICSStatusFields(() => ({format: 'uint32' as const, length: 1}))
    });
    expect(operation.createContributor(getView)).toMatchObject({props: {maximumIterations: 1}});
  } finally {
    device.destroy();
  }
});

function expectDistances(actual: Float32Array, expected: (number | string)[]) {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, row) => {
    const distance = value === 'Infinity' ? Infinity : Number(value);
    if (!Number.isFinite(distance)) expect(actual[row]).toBe(distance);
    else
      expect(Math.abs(actual[row] - distance), `distance row ${row}`).toBeLessThanOrEqual(
        1e-6 * Math.max(1, distance)
      );
  });
}

for (const scene of fixture.scenes) {
  it(`matches scikit-learn ordering and both extraction methods: ${scene.name}`, () => {
    const positions = Float32Array.from(scene.positions);
    const times = scene.times ? Float32Array.from(scene.times) : undefined;
    const options: CPUOPTICSOptions = {
      ...scene.options,
      temporalContext: times ? TEMPORAL_CONTEXT : undefined
    };
    const dbscan = computeOPTICSOnCPU(
      positions,
      {...options, extractionDistance: scene.dbscan.extractionDistance},
      times
    );
    expect(Array.from(dbscan.ordering)).toEqual(scene.ordering);
    expect(Array.from(dbscan.predecessors)).toEqual(scene.predecessors);
    expect(Array.from(dbscan.labels)).toEqual(scene.dbscan.labels);
    expectDistances(dbscan.coreDistances, scene.coreDistances);
    expectDistances(dbscan.reachability, scene.reachability);
    expect(getOPTICSStatusValues(dbscan)).toEqual([scene.ordering.length, 1, 0]);
    expect(Array.from(dbscan.candidateOverflow)).toEqual([0]);
    for (const extraction of scene.xi) {
      const result = computeOPTICSOnCPU(
        positions,
        {...options, ...extraction, extractionMethod: 'xi'},
        times
      );
      expect(Array.from(result.ordering)).toEqual(scene.ordering);
      expect(Array.from(result.predecessors)).toEqual(scene.predecessors);
      expectDistances(result.coreDistances, scene.coreDistances);
      expectDistances(result.reachability, scene.reachability);
      expect(getOPTICSStatusValues(result)).toEqual([scene.ordering.length, 1, 0]);
      expect(Array.from(result.labels)).toEqual(extraction.labels);
      expect(Array.from(result.clusterSummary)).toEqual([extraction.intervals.length, 0]);
      expect(
        Array.from(result.clusterIntervals.subarray(0, extraction.intervals.length * 2))
      ).toEqual(extraction.intervals.flat());
      expect(
        result.clusterIntervals
          .subarray(extraction.intervals.length * 2)
          .every(value => value === 0xffffffff)
      ).toBe(true);
    }
  });
}

it('matches hand-computed duplicate, inclusive-boundary, missing-value and invalid-time cases', () => {
  const positions = Float32Array.of(0, 0, 0, 0, 1, 0, NaN, 0);
  const result = computeOPTICSOnCPU(positions, {
    minimumSamples: 2,
    maximumDistance: 1,
    extractionDistance: 1
  });
  expect(Array.from(result.ordering)).toEqual([0, 1, 2, 0xffffffff]);
  expect(Array.from(result.coreDistances)).toEqual([0, 0, 1, Infinity]);
  expect(Array.from(result.reachability)).toEqual([Infinity, 0, 1, Infinity]);
  expect(Array.from(result.predecessors)).toEqual([0xffffffff, 0, 0, 0xffffffff]);
  expect(Array.from(result.labels)).toEqual([0, 0, 0, 0xffffffff]);
  expect(getOPTICSStatusValues(result)).toEqual([3, 1, 1]);
  const times = computeOPTICSOnCPU(
    Float32Array.of(0, 0, 1, 0),
    {
      minimumSamples: 1,
      maximumDistance: 2,
      extractionDistance: 2,
      maximumTimeDistance: 1,
      temporalContext: TEMPORAL_CONTEXT
    },
    Float32Array.of(0, NaN)
  );
  expect(getOPTICSStatusValues(times)).toEqual([1, 1, 1]);
  expect(Array.from(times.labels)).toEqual([0, 0xffffffff]);
});

it('matches an equatorial closed form across the antimeridian and excludes invalid geographic coordinates', () => {
  const spatialContext = resolveGPUSpatialContext(
    'optics-sphere',
    {coordinateSpace: 'longitude-latitude', metric: 'great-circle', units: 'meters'},
    GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
  );
  const truth = (2 * Math.PI * getGPUSphereRadius(spatialContext)) / 180;
  const result = computeOPTICSOnCPU(Float32Array.of(179, 0, -179, 0, 181, 0), {
    spatialContext,
    minimumSamples: 2,
    maximumDistance: truth * 2,
    extractionDistance: truth * 1.5
  });
  expectDistances(result.coreDistances, [truth, truth, Infinity]);
  expectDistances(result.reachability, [Infinity, truth, Infinity]);
  expect(Array.from(result.ordering)).toEqual([0, 1, 0xffffffff]);
  expect(Array.from(result.predecessors)).toEqual([0xffffffff, 0, 0xffffffff]);
  expect(Array.from(result.labels)).toEqual([0, 0, 0xffffffff]);
  expect(getOPTICSStatusValues(result)).toEqual([2, 1, 1]);
  for (const coincident of [
    [180, 0, -180, 0],
    [0, 90, 123, 90]
  ]) {
    const zero = computeOPTICSOnCPU(Float32Array.from(coincident), {
      spatialContext,
      minimumSamples: 2,
      maximumDistance: 1,
      extractionDistance: 1
    });
    expect(Array.from(zero.coreDistances)).toEqual([0, 0]);
    expect(Array.from(zero.reachability)).toEqual([Infinity, 0]);
    expect(Array.from(zero.labels)).toEqual([0, 0]);
  }
});

it('drops GPU scheduling and neighbor caps while preserving the explicit visit budget', () => {
  const result = computeOPTICSOnCPU(Float32Array.of(0, 0, 1, 0, 2, 0), {
    minimumSamples: 2,
    maximumDistance: 4,
    extractionDistance: 2,
    maximumIterations: 1,
    neighborIndex: 'grid',
    neighborCapacity: 1
  });
  expect(getOPTICSStatusValues(result)).toEqual([1, 0, 0]);
  expect(Array.from(result.candidateOverflow)).toEqual([0]);
  expect(Array.from(result.labels)).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
  const invalid = computeOPTICSOnCPU(new Float32Array(4097 * 2).fill(NaN), {
    minimumSamples: 257,
    maximumDistance: 1,
    extractionMethod: 'xi',
    minimumClusterSize: 2
  });
  expect(getOPTICSStatusValues(invalid)).toEqual([0, 1, 4097]);
  expect(invalid.ordering.every(value => value === 0xffffffff)).toBe(true);
  expect(invalid.labels.every(value => value === 0xffffffff)).toBe(true);
  expect(invalid.coreDistances.every(value => value === Infinity)).toBe(true);
  expect(invalid.clusterIntervals.every(value => value === 0xffffffff)).toBe(true);
  expect(Array.from(invalid.clusterSummary)).toEqual([0, 0]);
});

it('rejects inconsistent array lengths and invalid operation props at construction', () => {
  for (const Operation of [CPUOPTICSOperation, GPUOPTICSOperation]) {
    expect(() => new Operation({...PROPS, output: {...PROPS.output, labels: 'positions'}})).toThrow(
      /^optics-check outputs.labels/
    );
  }
  expect(() => computeOPTICSOnCPU(Float32Array.of(0), PROPS)).toThrow(/positions/);
  expect(() =>
    computeOPTICSOnCPU(
      new Float32Array(6),
      {...PROPS, temporalContext: TEMPORAL_CONTEXT, maximumTimeDistance: 1},
      Float32Array.of(0)
    )
  ).toThrow(/times length/);
  for (const override of [
    {minimumSamples: 0},
    {maximumDistance: NaN},
    {extractionDistance: 5},
    {xi: 1},
    {times: 'times'},
    {temporalContext: TEMPORAL_CONTEXT}
  ]) {
    expect(() => new CPUOPTICSOperation({...PROPS, ...override})).toThrow(/^optics-check /);
  }
  expect(() => new CPUOPTICSOperation({...PROPS, rowCount: 0})).toThrow(/^optics-check rowCount/);
  expect(() => new GPUOPTICSOperation({...PROPS, rowCount: 0})).toThrow(
    /^optics-check outputs.ordering.length/
  );
  expect(
    () =>
      new CPUOPTICSOperation({
        ...PROPS,
        minimumSamples: 257,
        frontierBatchSize: 512,
        maximumIterations: 99
      })
  ).not.toThrow();
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {
      positions: {format: 'float32x2', length: 3},
      ordering: {format: 'uint32', length: 3},
      coreDistances: {format: 'float32', length: 3},
      reachability: {format: 'float32', length: 3},
      predecessors: {format: 'uint32', length: 3},
      labels: {format: 'uint32', length: 3},
      ...createOPTICSStatusFields(() => ({format: 'uint32' as const, length: 1}))
    });
    expect(() =>
      new GPUOPTICSOperation({...PROPS, minimumSamples: 257}).createContributor(getView)
    ).toThrow(/^optics-check minimumSamples/);
    expect(() => new GPUOPTICSOperation({...PROPS, xi: 1}).createContributor(getView)).toThrow(
      /^optics-check xi/
    );
    expect(() =>
      new GPUOPTICSOperation({...PROPS, maximumIterations: 4}).createContributor(getView)
    ).toThrow(/^optics-check maximumIterations/);
    expect(() =>
      new GPUOPTICSOperation({...PROPS, extractionDistance: 5}).createContributor(getView)
    ).toThrow(/^optics-check extractionDistance/);
  } finally {
    device.destroy();
  }
});
