// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import {
  computeSinkhornDivergenceOnCPU,
  CPUSinkhornDivergenceOperation
} from '../../../src/gpu-spatial-analysis/optimal-transport/cpu-sinkhorn-divergence-operation';
import {
  GPUSinkhornDivergenceOperation,
  type SinkhornDivergenceOperationProps
} from '../../../src/gpu-spatial-analysis/optimal-transport/gpu-sinkhorn-divergence-operation';
import {getGPUSinkhornDivergenceParameterValues} from '../../../src/gpu-spatial-analysis/optimal-transport/sinkhorn-divergence-parameters';
import {computeSinkhornDivergenceReference} from './stm09-sinkhorn-reference';

const GRID = {
  width: 3,
  height: 2,
  pixelInterpretation: 'area' as const,
  affine: [1, 0, 0, 0, -1, 0] as [number, number, number, number, number, number]
};
const SOURCE = new Float32Array([1, 3, 0, 2, 0, 1]);
const TARGET = new Float32Array([0, 1, 2, 0, 4, 1]);
const PROPS: SinkhornDivergenceOperationProps = {
  id: 'sinkhorn-check',
  grid: GRID,
  source: 'source',
  target: 'target',
  parameters: 'parameters',
  output: {
    divergence: 'divergence',
    components: 'components',
    marginalError: 'error',
    status: {
      converged: 'converged',
      iterationLimitReached: 'limited',
      iterationCount: 'iterations',
      invalidCount: 'invalid',
      approximation: 'approximation'
    }
  }
};

it('matches a dense coupling/primal oracle on a sparse, anisotropic grid', () => {
  const settings = {cellWidth: 1.5, cellHeight: 0.75, epsilon: 2, tolerance: 1e-10};
  const parameters = getGPUSinkhornDivergenceParameterValues(settings);
  const result = computeSinkhornDivergenceOnCPU(SOURCE, TARGET, parameters, {
    grid: GRID,
    maximumIterations: 5000
  });
  const reference = computeSinkhornDivergenceReference(
    {...GRID, ...settings, source: Array.from(SOURCE), target: Array.from(TARGET)},
    1e-12
  );
  expect(result.converged[0]).toBe(1);
  expect(result.iterationLimitReached[0]).toBe(0);
  expect(result.approximation[0]).toBe(0);
  expect(result.invalidCount[0]).toBe(0);
  expect(result.marginalError[0]).toBeLessThan(1.1e-10);
  expect(result.divergence[0]).toBeCloseTo(reference.divergence, 6);
  [reference.crossCost, reference.sourceCost, reference.targetCost].forEach((expected, index) =>
    expect(result.components[index]).toBeCloseTo(expected, 6)
  );
  const scheduled = computeSinkhornDivergenceOnCPU(
    SOURCE,
    TARGET,
    getGPUSinkhornDivergenceParameterValues({...settings, epsilonStart: 16, scalingFactor: 0.5}),
    {grid: GRID, maximumIterations: 5000, blurRadius: 0}
  );
  expect(scheduled.divergence[0]).toBeCloseTo(reference.divergence, 6);
});

it('delegates GPU limits to contributor construction and drops that GPU-only cap on CPU', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {
      source: {format: 'float32', length: 6},
      target: {format: 'float32', length: 6},
      parameters: {format: 'float32', length: 6},
      divergence: {format: 'float32', length: 1},
      components: {format: 'float32', length: 3},
      marginalError: {format: 'float32', length: 1},
      converged: {format: 'uint32', length: 1},
      iterationLimitReached: {format: 'uint32', length: 1},
      iterationCount: {format: 'uint32', length: 1},
      invalidCount: {format: 'uint32', length: 1},
      approximation: {format: 'uint32', length: 1}
    });
    const props = {...PROPS, maximumIterations: 1001};
    expect(() => new GPUSinkhornDivergenceOperation(props).createContributor(getView)).toThrow(
      /maximumIterations/
    );
    expect(() => new CPUSinkhornDivergenceOperation(props)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('matches the point-mass closed form and keeps full blur support despite a short radius', () => {
  const parameters = getGPUSinkhornDivergenceParameterValues({
    cellWidth: 2,
    cellHeight: 3,
    epsilon: 0.125,
    tolerance: 1e-8
  });
  const result = computeSinkhornDivergenceOnCPU(
    new Float32Array([5, 0, 0, 0, 0, 0]),
    new Float32Array([0, 0, 0, 0, 0, 7]),
    parameters,
    {grid: GRID, blurRadius: 0}
  );
  expect(result.divergence[0]).toBe(25); // (2 cells * 2)^2 + (1 cell * 3)^2, KL = 0.
  expect(Array.from(result.components)).toEqual([25, 0, 0]);
  expect(result.converged[0]).toBe(1);
  expect(result.approximation[0]).toBe(0);
  const singleton = computeSinkhornDivergenceOnCPU(
    Float32Array.of(3),
    Float32Array.of(9),
    parameters,
    {grid: {...GRID, width: 1, height: 1}}
  );
  expect(singleton.divergence[0] === 0).toBe(true);
});

it('publishes budget-limited results, counts invalid masses and refuses unusable frames', () => {
  const parameters = getGPUSinkhornDivergenceParameterValues({
    cellWidth: 1,
    cellHeight: 1,
    epsilon: 0.5,
    tolerance: 1e-8
  });
  const limited = computeSinkhornDivergenceOnCPU(SOURCE, TARGET, parameters, {
    grid: GRID,
    maximumIterations: 1
  });
  expect(limited.iterationCount[0]).toBe(1);
  expect(limited.iterationLimitReached[0]).toBe(1);
  expect(limited.converged[0]).toBe(0);
  expect(Number.isFinite(limited.divergence[0])).toBe(true);
  const dirty = computeSinkhornDivergenceOnCPU(
    new Float32Array([NaN, 3, -2, 2, 0, 1]),
    TARGET,
    parameters,
    {grid: GRID}
  );
  const clean = computeSinkhornDivergenceOnCPU(
    new Float32Array([0, 3, 0, 2, 0, 1]),
    TARGET,
    parameters,
    {grid: GRID}
  );
  expect(dirty.invalidCount[0]).toBe(2);
  expect(dirty.divergence[0]).toBe(clean.divergence[0]);
  for (const invalid of [
    computeSinkhornDivergenceOnCPU(SOURCE, new Float32Array(6), parameters, {grid: GRID}),
    computeSinkhornDivergenceOnCPU(SOURCE, TARGET, new Float32Array(6), {grid: GRID})
  ]) {
    expect(invalid.divergence[0]).toBeNaN();
    expect(Array.from(invalid.components)).toEqual([NaN, NaN, NaN]);
    expect(invalid.marginalError[0]).toBeNaN();
    expect(invalid.iterationCount[0]).toBe(0);
    expect(invalid.converged[0]).toBe(0);
    expect(invalid.iterationLimitReached[0]).toBe(0);
  }
});

it('validates arrays, props and graph ports at construction', () => {
  const parameters = new Float32Array(6);
  expect(() =>
    computeSinkhornDivergenceOnCPU(SOURCE.subarray(1), TARGET, parameters, {grid: GRID})
  ).toThrow('source');
  expect(() =>
    computeSinkhornDivergenceOnCPU(SOURCE, TARGET.subarray(1), parameters, {grid: GRID})
  ).toThrow('target');
  expect(() =>
    computeSinkhornDivergenceOnCPU(SOURCE, TARGET, parameters.subarray(1), {grid: GRID})
  ).toThrow('parameters');
  for (const Operation of [GPUSinkhornDivergenceOperation, CPUSinkhornDivergenceOperation]) {
    expect(
      () => new Operation({...PROPS, output: {...PROPS.output, divergence: 'source'}})
    ).toThrow(/^sinkhorn-check/);
    expect(new Operation(PROPS).id).toBe('sinkhorn-check');
  }
  expect(() => new CPUSinkhornDivergenceOperation({...PROPS, maximumIterations: 0})).toThrow(
    /^sinkhorn-check.*maximumIterations/
  );
  expect(() => new CPUSinkhornDivergenceOperation({...PROPS, grid: {...GRID, width: 0}})).toThrow(
    /^sinkhorn-check.*grid/
  );
  expect(
    () => new CPUSinkhornDivergenceOperation({...PROPS, maximumIterations: 1001})
  ).not.toThrow();
});
