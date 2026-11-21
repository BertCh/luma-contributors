// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUSinkhornDivergenceOperation,
  type SinkhornDivergenceOperationProps
} from '../../../src/gpu-spatial-analysis/optimal-transport/gpu-sinkhorn-divergence-operation';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';
import {GPUSinkhornDivergence} from '../../../src/gpu-spatial-analysis/optimal-transport/gpu-sinkhorn-divergence';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('exposes default stopping destinations and rejects float32 schedules that cannot advance', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const view = (name: string, length: number) =>
      createTransientView(graph, name, 'float32', length);
    const solver = new GPUSinkhornDivergence({
      id: 'transport',
      grid: {width: 2, height: 1, pixelInterpretation: 'area', affine: [1, 0, 0, 0, 1, 0]},
      source: view('source', 2),
      target: view('target', 2),
      parameters: view('parameters', 6),
      maximumIterations: 1,
      output: {divergence: view('divergence', 1)}
    });
    const settings = {cellWidth: 1, cellHeight: 1, epsilon: 1, epsilonStart: 2};
    for (const scalingFactor of [undefined, 1 - 2 ** -26]) {
      expect(() => solver.getParameterValues({...settings, scalingFactor})).toThrow(
        /transport.*scalingFactor/
      );
    }
    expect(solver.getParameterValues({...settings, scalingFactor: 0.5})[5]).toBe(0.5);
    graph.add(solver);
    expect(solver.output.status?.converged?.length).toBe(1);
    expect(solver.output.status?.iterationLimitReached?.length).toBe(1);
    expect(solver.output.marginalError?.length).toBe(1);
  } finally {
    device.destroy();
  }
});

it('validates radius spellings at named operation construction and captures them', () => {
  const props: SinkhornDivergenceOperationProps = {
    id: 'transport',
    grid: {width: 2, height: 1, pixelInterpretation: 'area', affine: [1, 0, 0, 0, 1, 0]},
    source: 'source',
    target: 'target',
    parameters: 'parameters',
    output: {divergence: 'divergence'},
    transportRadius: 1
  };
  expect(() => new GPUSinkhornDivergenceOperation({...props, ...{blurRadius: 1}})).toThrow(
    /blurRadius was removed.*transportRadius/
  );
  const operation = new GPUSinkhornDivergenceOperation(props);
  props.transportRadius = 0;
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {
      source: {format: 'float32', length: 2},
      target: {format: 'float32', length: 2},
      parameters: {format: 'float32', length: 6},
      divergence: {format: 'float32', length: 1}
    });
    expect((operation.createContributor(getView) as GPUSinkhornDivergence).transportRadius).toBe(1);
  } finally {
    device.destroy();
  }
});
