// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  GPUPointCloudNormalScale,
  getGPUPointCloudNormalScaleParameterValues,
  type GPUPointCloudNormalScaleProps
} from '@luma-contributors/analysis/pointcloud-analysis';
import {createNullWebGPUDevice, getParsedKernelSources} from '../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';

function makeDeclaration(rows = 4, candidates = 25) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let sequence = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x3'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `normal-scale-${sequence++}`, format, length);
  const props: GPUPointCloudNormalScaleProps = {
    parameters: view('float32', 5),
    positions: view('float32x3', candidates),
    queryRows: view('uint32', rows),
    radii: [0.1, 0.4, 0.8],
    capacity: 64,
    normals: view('float32x3', rows),
    normalScales: view('float32', rows),
    normalVariances: view('float32', rows),
    status: view('uint32', rows)
  };
  return {graph, props, view};
}

it('validates scale, capacity, support, view and coordinate limits at construction', () => {
  const {props, view} = makeDeclaration();
  for (const radii of [
    [],
    [0],
    [-1],
    [NaN],
    [Infinity],
    [1e-30],
    [1e30],
    [0.8, 0.4],
    [0.4, 0.4],
    [0.4, 0.4 + 1e-10],
    Array.from({length: 33}, (_, index) => index + 1)
  ]) {
    expect(() => new GPUPointCloudNormalScale({...props, radii})).toThrow(
      /^gpu-pointcloud-normal-scale radii/
    );
  }
  for (const capacity of [0, 2, 65, 3.5]) {
    expect(() => new GPUPointCloudNormalScale({...props, capacity})).toThrow(/capacity/);
  }
  for (const minimumSupport of [2, 65, 3.5]) {
    expect(() => getGPUPointCloudNormalScaleParameterValues({minimumSupport})).toThrow(
      /minimumSupport/
    );
  }
  expect(() => new GPUPointCloudNormalScale({...props, normals: view('float32x3', 3)})).toThrow(
    /normals/
  );
  expect(() => getGPUPointCloudNormalScaleParameterValues({viewpoint: [Infinity, 0, 0]})).toThrow(
    /viewpoint/
  );
  expect(
    () =>
      new GPUPointCloudNormalScale({
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      })
  ).toThrow(/longitude-latitude/);
  const large = makeDeclaration(65536, 65536);
  expect(() => new GPUPointCloudNormalScale(large.props)).toThrow(/pass grid/);
});

it('audits shaders and bounds command counts at 32 scales, with and without an index', () => {
  for (const indexed of [false, true]) {
    const {graph, props} = makeDeclaration();
    const contributor = new GPUPointCloudNormalScale({
      ...props,
      radii: Array.from({length: 32}, (_, index) => 0.1 + index * 0.1),
      ...(indexed
        ? {grid: {origin: [-4, -4, -4] as const, cellSize: 1, dimensions: [8, 8, 8] as const}}
        : {})
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    for (const [identifier, source] of getParsedKernelSources(nodes)) {
      expect(auditWGSLModule(source), identifier).toEqual([]);
    }
  }
});

it('checks row-proportional scratch against the device before building kernels', () => {
  const {graph, props} = makeDeclaration(1000000, 1);
  graph.device.limits.maxStorageBufferBindingSize = 128 * 1024 * 1024;
  expect(() => new GPUPointCloudNormalScale(props).getCommandNodes(graph)).toThrow(
    /rows x capacity scratch/
  );
});

it('accepts an empty query selection and emits no kernels', () => {
  const {graph, props} = makeDeclaration(0);
  expect(new GPUPointCloudNormalScale(props).getCommandNodes(graph)).toEqual([]);
});

it('packs mutable eligibility/orientation settings and rejects parameter shape and aliases', () => {
  expect(Array.from(getGPUPointCloudNormalScaleParameterValues())).toEqual([3, 0, 0, 0, 0]);
  expect(
    Array.from(
      getGPUPointCloudNormalScaleParameterValues({minimumSupport: 6, viewpoint: [1, 2, 3]})
    )
  ).toEqual([6, 1, 2, 3, 1]);
  const {props, view} = makeDeclaration();
  expect(() => new GPUPointCloudNormalScale({...props, parameters: view('float32', 4)})).toThrow(
    /parameters/
  );
  expect(
    () => new GPUPointCloudNormalScale({...props, normalVariances: props.normalScales})
  ).toThrow(/distinct buffers/);
});

it('bounds the published index allowance at the storage-limited position count', () => {
  const {graph, props} = makeDeclaration(1, Math.floor((128 * 1024 * 1024) / 12));
  const contributor = new GPUPointCloudNormalScale({
    ...props,
    grid: {origin: [0, 0, 0], cellSize: 1, dimensions: [1, 1, 1]}
  });
  expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
    contributor.estimatedCommandNodeCount
  );
});
