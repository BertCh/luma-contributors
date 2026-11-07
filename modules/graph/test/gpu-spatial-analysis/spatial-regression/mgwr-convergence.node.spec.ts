// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma-contributors/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {GPUGeneralizedGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-generalized-geographically-weighted-regression';
import {GPUMultiscaleGeographicallyWeightedRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-multiscale-geographically-weighted-regression';

function createProps(graph: GPUCommandGraph, rowCount: number) {
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `convergence-${serial++}`, format, length);
  return {
    view,
    regression: {
      positions: view('float32x2', rowCount),
      predictors: view('float32', rowCount * 2),
      response: view('float32', rowCount),
      predictorCount: 2,
      mask: view('uint32', rowCount),
      maximumIterations: 3,
      indexGridSize: [4, 4] as const,
      output: {
        coefficients: view('float32', rowCount * 3),
        fitted: view('float32', rowCount),
        status: {codes: view('uint32', rowCount), iterationCount: view('uint32', 1)}
      }
    }
  };
}

function generalizedProps(
  view: ReturnType<typeof createProps>['view'],
  regression: ReturnType<typeof createProps>['regression']
) {
  return {
    ...regression,
    parameters: view('float32', 2),
    family: 'poisson' as const,
    output: {
      ...regression.output,
      status: {
        ...regression.output.status,
        iterationCount: view('uint32', regression.response.length)
      }
    }
  };
}

it('multiscale convergence reduces in parallel within eight storage bindings', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {view, regression} = createProps(graph, 5000);
  const nodes = new GPUMultiscaleGeographicallyWeightedRegression({
    ...regression,
    parameters: view('float32', 4)
  }).getCommandNodes(graph);
  const sources = getGPUCommandNodeKernelSources(nodes);
  const reduce = sources.filter(
    source => source.id.includes('-convergence-') && !source.id.includes('decide')
  );
  const decide = sources.filter(source => source.id.includes('-convergence-decide-'));
  expect(reduce.length).toBe(3);
  expect(decide.length).toBe(3);
  for (const source of [...reduce, ...decide]) {
    expect(new WgslReflect(source.source).storage.length, source.id).toBeLessThanOrEqual(8);
  }
  // Workgroup tree reduction instead of one serial thread over rows x columns.
  expect(reduce[0].source).toContain('var<workgroup> sharedChange');
  expect(reduce[0].source).toContain('workgroupBarrier()');
  expect(reduce[0].source).not.toMatch(/for \(var row = 0u; row < 5000u; row\+\+\)/);
  device.destroy();
});

it('gaussianCutoff is validated and puts the Gaussian kernel on the grid path', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {view, regression} = createProps(graph, 64);
  const props = generalizedProps(view, regression);
  for (const gaussianCutoff of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
    expect(
      () => new GPUGeneralizedGeographicallyWeightedRegression({...props, gaussianCutoff})
    ).toThrow(/gaussianCutoff/);
  }
  const sources = getGPUCommandNodeKernelSources(
    new GPUGeneralizedGeographicallyWeightedRegression({
      ...props,
      kernel: 'gaussian',
      gaussianCutoff: 5
    }).getCommandNodes(graph)
  );
  const fit = sources.find(source => source.id.endsWith('-fit'))!;
  expect(fit.source).toContain('const GAUSSIAN_CUTOFF');
  expect(fit.source).toContain('getSupportRadius');
  device.destroy();
});

it('adaptive neighbor selection walks grid rings instead of scanning every row', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const {view, regression} = createProps(graph, 64);
  const props = {
    ...generalizedProps(view, regression),
    bandwidthMode: 'adaptive' as const,
    maximumNeighborCount: 16
  };
  const find = (producerProps: typeof props) =>
    getGPUCommandNodeKernelSources(
      new GPUGeneralizedGeographicallyWeightedRegression(producerProps).getCommandNodes(graph)
    ).find(source => source.id.endsWith('-fit'))!;
  const ring = find(props);
  expect(ring.source).toContain('fn selectNeighborDistances');
  expect(ring.source).not.toContain('var distances');
  expect(new WgslReflect(ring.source).storage.length).toBeLessThanOrEqual(8);
  const scan = find({...props, id: 'scan', indexGridSize: undefined as never});
  expect(scan.source).toContain('var distances');
  device.destroy();
});
