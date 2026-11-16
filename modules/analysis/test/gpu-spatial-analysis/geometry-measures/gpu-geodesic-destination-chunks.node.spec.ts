// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {
  WGSL_RESERVED_WORDS,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {GPUGeodesicDestination} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geodesic-destination';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
afterEach(() => vi.restoreAllMocks());
let nextPropsId = 0;
function createProps(graph: GPUCommandGraph, lengths: number[], padded = false) {
  const propsId = nextPropsId++;
  const rows = <Format extends 'float32' | 'float32x2' | 'uint32'>(name: string, format: Format) =>
    createVectorView(
      name,
      format,
      lengths.map((length, chunkIndex) => {
        const byteOffset = padded ? (chunkIndex % 63) * 4 : 0;
        const components = format === 'float32x2' ? 2 : 1;
        const backing = createTransientView(
          graph,
          `${name}-${propsId}-${chunkIndex}`,
          'uint32',
          Math.max(length, 1) * components + byteOffset / 4
        );
        return graph.createDataView(backing.buffer, {format, length, byteOffset});
      })
    );
  return {
    id: 'chunk-destination',
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal'} as const,
    origins: rows('origins', 'float32x2'),
    bearings: rows('bearings', 'float32'),
    distances: rows('distances', 'float32'),
    output: {
      finalBearings: rows('finalBearings', 'float32'),
      destinations: rows('destinations', 'float32x2'),
      status: {
        converged: rows('converged', 'uint32'),
        approximation: rows('approximation', 'uint32'),
        iterationLimitReached: rows('iterationLimitReached', 'uint32')
      }
    }
  };
}

it('validates every input/output topology and all writable aliases at construction', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 4, 2]);
  expect(new GPUGeodesicDestination(props).estimatedCommandNodeCount).toBe(7);
  expect(
    () => new GPUGeodesicDestination({...props, bearings: createProps(graph, [2, 3, 2]).bearings})
  ).toThrow(/^chunk-destination bearings chunk 0.*topology/);
  expect(
    () =>
      new GPUGeodesicDestination({...props, output: {...props.output, destinations: props.origins}})
  ).toThrow(/share buffers/);
  expect(
    () =>
      new GPUGeodesicDestination({
        ...props,
        output: {
          ...props.output,
          status: {...props.output.status, approximation: props.output.status.converged}
        }
      })
  ).toThrow(/byte ranges must not overlap/);
  const backing = createTransientView(graph, 'repeated', 'float32', 2);
  const repeated = createProps(graph, [2, 2]);
  expect(
    () =>
      new GPUGeodesicDestination({
        ...repeated,
        output: {finalBearings: createVectorView('repeat', 'float32', [backing, backing])}
      })
  ).toThrow(/byte ranges must not overlap/);
});

it('rejects a later aligned binding before allocating chunk parameters or declaring nodes', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUGeodesicDestination(createProps(graph, [1, 16_777_216], true));
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const addNode = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /^chunk-destination origins chunk 1 needs 134217732 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(addNode).not.toHaveBeenCalled();
});

it('supports 30 million rows, empty chunks and construction-time command budgets', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUGeodesicDestination(
    createProps(graph, [0, 10_000_000, 10_000_000, 10_000_000, 0])
  );
  expect(contributor.getCommandNodes(graph)).toHaveLength(7);
  const input = createTransientView(graph, 'budget-input', 'float32x2', 1);
  const output = createTransientView(graph, 'budget-output', 'float32', 32768 * 64);
  const outputChunks = Array.from({length: 32768}, (_, row) =>
    graph.createDataView(output.buffer, {format: 'float32', length: 1, byteOffset: row * 256})
  );
  expect(
    () =>
      new GPUGeodesicDestination({
        ...contributor.props,
        origins: createVectorView('origins', 'float32x2', new Array(32768).fill(input)),
        bearings: createVectorView('bearings', 'float32', outputChunks),
        distances: createVectorView('distances', 'float32', outputChunks),
        output: {finalBearings: createVectorView('distances', 'float32', outputChunks)}
      })
  ).toThrow(/estimated command node count.*32768/);
});

it('shares a bounded pipeline count for all five outputs as chunk count and unaligned prefixes grow', () => {
  for (const metric of ['great-circle', 'rhumb', 'ellipsoidal'] as const) {
    const keys = new Set<string>();
    for (const chunkCount of [1, 3, 100]) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(
        graph,
        Array.from({length: chunkCount}, (_, index) => [1, 257, 513][index % 3]),
        true
      );
      if (metric !== 'ellipsoidal') delete (props.output.status as {converged?: unknown}).converged;
      const contributor = new GPUGeodesicDestination({
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric}
      });
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      expect(getParsedKernelSources(nodes).size).toBe(metric === 'ellipsoidal' ? 2 : 1);
      for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
        keys.add(key);
      for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
        expect(Object.keys(constants)).toHaveLength(0);
        // Strip comments: this checks generated identifiers rather than English documentation.
        const identifiers = stripWGSLComments(source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
      }
    }
    expect(keys.size).toBe(metric === 'ellipsoidal' ? 2 : 1);
  }
});

it('rejects custom contexts whose generated float32 literals or Vincenty constants are invalid', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 4, 2]);
  for (const ellipsoid of [
    {semiMajorAxis: 1e-200, flattening: 0.003},
    {semiMajorAxis: 1e200, flattening: 0.003},
    {semiMajorAxis: 6378137, flattening: 1 - 1e-9}
  ]) {
    expect(
      () =>
        new GPUGeodesicDestination({
          ...props,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'ellipsoidal', ellipsoid}
        })
    ).toThrow(/^chunk-destination spatialContext\.ellipsoid\./);
  }
  for (const sphereRadius of [1e-200, 3.5e38]) {
    expect(
      () =>
        new GPUGeodesicDestination({
          ...props,
          spatialContext: {
            coordinateSpace: 'longitude-latitude',
            metric: 'great-circle',
            sphereRadius
          }
        })
    ).toThrow(/^chunk-destination radius/);
  }
});

it('prefixes chunk scratch resource IDs so two contributors can share one graph', () => {
  const graph = new GPUCommandGraph(device);
  const first = new GPUGeodesicDestination({...createProps(graph, [1, 4, 2]), id: 'first'});
  const second = new GPUGeodesicDestination({...createProps(graph, [1, 4, 2]), id: 'second'});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const firstNodes = first.getCommandNodes(graph);
  const secondNodes = second.getCommandNodes(graph);
  expect(allocate.mock.calls.map(([props]) => props.id)).toEqual(['first-chunks', 'second-chunks']);
  const nodeIds = [...firstNodes, ...secondNodes].map(node => node.id);
  expect(new Set(nodeIds).size).toBe(nodeIds.length);
  expect(firstNodes.every(node => node.id.startsWith('first-'))).toBe(true);
  expect(secondNodes.every(node => node.id.startsWith('second-'))).toBe(true);
});
