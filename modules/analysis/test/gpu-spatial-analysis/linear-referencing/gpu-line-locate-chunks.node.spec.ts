// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphVectorView} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPULineLocate,
  type GPULineLocateProps
} from '../../../src/gpu-spatial-analysis/linear-referencing/gpu-line-locate';
import {
  createGraphChunkGroupKernelNode,
  planGraphChunkGroups
} from '../../../src/analysis-core/kernels/graph-chunk-groups';
import {expectValidQueryShaderLiterals} from './gpu-linear-referencing-chunks.test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

type ChunkedProps = GPULineLocateProps<
  GraphVectorView<'float32x2'>,
  GraphVectorView<'uint32'>,
  GraphVectorView<'float32x2'>,
  GraphVectorView<'float32'>,
  GraphVectorView<'uint32'>,
  GraphVectorView<'uint32'>,
  GraphVectorView<'float32'>,
  GraphVectorView<'float32'>
>;

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());

function createProps(graph: GPUCommandGraph, lengths: number[]): ChunkedProps {
  const rows = <Format extends 'float32x2' | 'float32' | 'uint32'>(name: string, format: Format) =>
    createVectorView(
      name,
      format,
      lengths.map((length, chunkIndex) => {
        const backing = createTransientView(graph, `${name}-${chunkIndex}`, format, length + 64);
        return graph.createDataView(backing.buffer, {
          format,
          length,
          byteOffset: ((chunkIndex % 3) + 1) * 4
        });
      })
    );
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', 2),
    lineOffsets: createTransientView(graph, 'line-offsets', 'uint32', 2),
    eventLineIndices: rows('events', 'uint32'),
    eventMeasures: rows('measures', 'float32'),
    eventOffsets: rows('offsets', 'float32'),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {positions: rows('output', 'float32x2'), statuses: rows('status', 'uint32')}
  };
}

it('validates chunk topology and aligned output overlap at construction', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 3, 2]);
  expect(new GPULineLocate(props).estimatedCommandNodeCount).toBe(7);
  expect(
    () =>
      new GPULineLocate({
        ...props,
        eventMeasures: createTransientView(graph, 'wrong', 'float32', 6)
      })
  ).toThrow(/topology/);
  const backing = createTransientView(graph, 'alias', 'float32x2', 128);
  const first = graph.createDataView(backing.buffer, {
    format: 'float32x2',
    length: 1,
    byteOffset: 4
  });
  const adjacent = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 12
  });
  const single = createProps(new GPUCommandGraph(device), [1]);
  expect(
    () => new GPULineLocate({...single, output: {positions: first, statuses: adjacent}})
  ).toThrow(/aligned binding byte ranges must not overlap/);
});

it('prechecks derived result scratch before graph allocation', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPULineLocate(createProps(graph, [1, 5_592_406]));
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/eventLineIndices chunk rows.*5592405/);
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
});

it('accepts 30 million events with one shared prefix and chunk-sized dispatches', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPULineLocate(createProps(graph, Array(6).fill(5_000_000)));
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes).toHaveLength(13);
  expect(nodes.filter(node => node.id === 'line-locate-vertex-measures')).toHaveLength(1);
});

it('does not pad dispatches or specialize shaders/pipelines as uneven chunk counts grow', () => {
  const pipelines = new Set<string>();
  for (const count of [1, 3, 100]) {
    const graph = new GPUCommandGraph(device);
    const lengths = Array.from({length: count}, (_, index) => [1, 257, 513][index % 3]);
    const nodes = new GPULineLocate(createProps(graph, lengths)).getCommandNodes(graph);
    expect(() => graph.add(nodes)).not.toThrow();
    const queryNodes = nodes.filter(node => node.id.endsWith('-locate'));
    expect(queryNodes.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
      lengths.map(length => Math.ceil(length / 256))
    );
    for (const key of getGPUCommandNodePipelineKeys(nodes)) pipelines.add(key);
    for (const kernel of getGPUCommandNodeKernelSources(nodes)) {
      expectValidQueryShaderLiterals(kernel.source);
      const reflection = new WgslReflect(kernel.source);
      expect(reflection.storage.length).toBeLessThanOrEqual(8);
      const identifiers =
        kernel.source
          .replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '')
          .match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(identifiers.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
      if (kernel.source.includes('GraphChunkGroupParameters')) expect(kernel.constants).toEqual({});
    }
  }
  expect(pipelines.size).toBe(3);
});

it('plans grouped support, drift and weights without allocating graph resources', () => {
  const graph = new GPUCommandGraph(device);
  const supports = createVectorView(
    'supports',
    'float32x2',
    [2, 6, 4].map((length, index) =>
      createTransientView(graph, `supports-${index}`, 'float32x2', length)
    )
  );
  const weights = createVectorView(
    'weights',
    'float32',
    [3, 9, 6].map((length, index) =>
      createTransientView(graph, `weights-${index}`, 'float32', length)
    )
  );
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const plan = planGraphChunkGroups(
    'groups',
    {view: supports, rowsPerQuery: 2},
    {supports: {view: supports, rowsPerQuery: 2}, weights: {view: weights, rowsPerQuery: 3}}
  );
  expect(plan.chunks.map(chunk => [chunk.rowBase, chunk.length])).toEqual([
    [0, 1],
    [1, 3],
    [4, 2]
  ]);
  expect(allocate).not.toHaveBeenCalled();
  expect(() => planGraphChunkGroups('groups', {view: supports, rowsPerQuery: 3}, {})).toThrow(
    /query chunk 0 rows/
  );
  expect(() =>
    planGraphChunkGroups(
      'groups',
      {view: supports, rowsPerQuery: 2},
      {weights: {view: weights, rowsPerQuery: 2}}
    )
  ).toThrow(/weights total rows must match the query groups/);
  const packed = createTransientView(graph, 'packed-output', 'float32', 30);
  expect(
    planGraphChunkGroups(
      'capacity',
      {view: packed},
      {output: {view: packed, allowPackedCapacity: true}}
    ).totalLength
  ).toBe(30);
});

it('validates uint32-limit literals and the published workgroup-size minimum', () => {
  const graph = new GPUCommandGraph(device);
  expect(() =>
    createGraphChunkGroupKernelNode(graph, {
      id: 'chunk-groups-minimum',
      operation: 'ChunkGroupRows',
      bindings: [],
      body: '',
      workgroupSize: 1,
      invocationCount: 1
    })
  ).toThrow(/workgroupSize.*2/);
  for (const workgroupSize of [2, 32, 256]) {
    const sources = new Set<string>();
    for (const invocationCount of [1, 33, 0xffffffff]) {
      const graph = new GPUCommandGraph(device);
      const node = createGraphChunkGroupKernelNode(graph, {
        id: 'chunk-groups-row-limit',
        operation: 'ChunkGroupRows',
        bindings: [],
        body: '',
        workgroupSize,
        invocationCount
      });
      expect(() => graph.add(node)).not.toThrow();
      for (const {source, constants} of getGPUCommandNodeKernelSources([node])) {
        expectValidQueryShaderLiterals(source);
        expect(constants).toEqual({});
        const identifiers = source.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(identifiers.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
        sources.add(source);
      }
    }
    expect(sources.size).toBe(1);
  }
});

it('rejects sphere radii that overflow or vanish in shader float32 at construction', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 3, 2]);
  for (const sphereRadius of [1e100, Number.MIN_VALUE])
    expect(
      () =>
        new GPULineLocate({
          ...props,
          spatialContext: {
            coordinateSpace: 'longitude-latitude',
            metric: 'great-circle',
            sphereRadius
          }
        })
    ).toThrow(/sphereRadius.*finite/);
});
