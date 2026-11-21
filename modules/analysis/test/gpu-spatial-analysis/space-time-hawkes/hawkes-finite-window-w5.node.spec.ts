// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {
  computeSpaceTimeHawkesOnCPU,
  CPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS
} from '../../../src/gpu-spatial-analysis/space-time-hawkes/cpu-space-time-hawkes-operation';
import {getSquareInput, evaluateSquare, expectSquareFit} from './hawkes-finite-window-w5-fixture';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUSpaceTimeHawkes} from '../../../src/gpu-spatial-analysis/space-time-hawkes/gpu-space-time-hawkes';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('UPT-01 optimizes finite-window decay without decreasing the first-step likelihood', () => {
  const input = getSquareInput();
  const first = computeSpaceTimeHawkesOnCPU(input, {maximumIterations: 1});
  const entering = evaluateSquare(input.parameters.subarray(5));
  expect(Math.abs(entering - -10.5944629)).toBeLessThan(2e-6);
  expect(Number.isFinite(first.fittedLogLikelihood[0])).toBe(true);
  expect(first.fittedLogLikelihood[0]).toBeGreaterThanOrEqual(entering);
  expect(Math.abs(first.fittedLogLikelihood[0] - evaluateSquare(first.estimates))).toBeLessThan(
    2e-6
  );
  const final = computeSpaceTimeHawkesOnCPU(input, {maximumIterations: 1000});
  expectSquareFit(final.estimates, final.fittedLogLikelihood[0], final.converged[0]);
});

it('UPT-01 generated one-step kernels increase the independently evaluated square likelihood', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const input = getSquareInput();
  const memory = createCPUGraphMemory();
  const parameters = createTransientView(graph, 'parameters', 'float32', 9);
  memory.write(parameters, input.parameters);
  const estimates = createTransientView(graph, 'estimates', 'float32', 4);
  const fittedLogLikelihood = createTransientView(graph, 'fitted', 'float32', 1);
  try {
    const nodes = new GPUSpaceTimeHawkes({
      maximumIterations: 1,
      maximumParents: 4,
      positions: createTransientView(graph, 'positions', 'float32x2', 4),
      times: createTransientView(graph, 'times', 'float32', 4),
      parameters,
      output: {estimates, fittedLogLikelihood}
    }).getCommandNodes(graph);
    const selected = nodes.filter(
      node =>
        node.id.includes('-initialize') ||
        node.id.includes('-iteration-') ||
        node.id.includes('-fitted-sums') ||
        node.id.endsWith('-publish-estimates')
    );
    const events = getWGSLKernelNodeRecord(
      graph,
      selected.find(node => node.id.includes('-sums-tiles'))!.id
    )!.bindings.find(binding => binding.name === 'events')!.view;
    memory.write(
      events,
      Float32Array.from(
        Array.from(input.times).flatMap((time, row) => [
          input.positions[2 * row],
          input.positions[2 * row + 1],
          time,
          0
        ])
      )
    );
    executeCommandNodesOnCPU(selected, memory);
    const actual = memory.readFloat32(estimates);
    const fitted = memory.readFloat32(fittedLogLikelihood)[0];
    expect(Number.isFinite(fitted)).toBe(true);
    expect(fitted).toBeGreaterThanOrEqual(evaluateSquare(input.parameters.subarray(5)));
    expect(Math.abs(fitted - evaluateSquare(actual))).toBeLessThan(2e-5);
  } finally {
    device.destroy();
  }
});

it('DEF-13 shares the ordinary budget of 100 and reports the resolved budget', () => {
  const input = {
    positions: Float32Array.of(2, 4, 2, 5, 3, 5, 4, 3, 1, 3, 0, 1, 4, 3, 7, 4),
    times: Float32Array.of(0, 1, 2, 3, 4, 5, 6, 7),
    parameters: Float32Array.of(30, 8, 8, 15, 1e-5, 0.05, 0.5, 1, 2)
  };
  expect(CPU_SPACE_TIME_HAWKES_DEFAULT_MAXIMUM_ITERATIONS).toBe(100);
  const omitted = computeSpaceTimeHawkesOnCPU(input);
  expect(omitted).toEqual(computeSpaceTimeHawkesOnCPU(input, {maximumIterations: 100}));
  expect(omitted.maximumIterations[0]).toBe(100);
  expect(computeSpaceTimeHawkesOnCPU(input, {maximumIterations: 1000}).maximumIterations[0]).toBe(
    1000
  );
});
