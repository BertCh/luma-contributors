// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphVectorView,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {
  GPUSphericalConvexContains,
  type GPUSphericalConvexContainsProps
} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-contains';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {expectValidQueryShaderLiterals} from '../linear-referencing/gpu-linear-referencing-chunks.test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

type ChunkedProps = GPUSphericalConvexContainsProps<
  GraphVectorView<'float32x3'>,
  GraphVectorView<'uint32'>,
  GraphVectorView<'uint32'>,
  GraphVectorView<'uint32'>
>;

const device = createNullWebGPUDevice();
afterEach(() => vi.restoreAllMocks());
afterAll(() => device.destroy());

function createProps(graph: GPUCommandGraph, lengths: number[]): ChunkedProps {
  const createRows = <Format extends 'float32x3' | 'uint32'>(name: string, format: Format) =>
    createVectorView(
      name,
      format,
      lengths.map((length, chunkIndex) => {
        const backing = createTransientView(graph, `${name}-${chunkIndex}`, format, length + 16);
        return graph.createDataView(backing.buffer, {
          format,
          length,
          byteOffset: ((chunkIndex % 3) + 1) * 4
        });
      })
    );
  return {
    positions: createTransientView(graph, 'positions', 'float32x3', 3),
    offsets: createTransientView(graph, 'offsets', 'uint32', 2),
    queries: createRows('queries', 'float32x3'),
    polygonIndices: createRows('selectors', 'uint32'),
    output: {contains: createRows('contains', 'uint32'), status: createRows('statuses', 'uint32')}
  };
}

it('validates uneven row topology and writable aliases before any declaration', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 3, 2]);
  const contributor = new GPUSphericalConvexContains(props);
  expect(contributor.estimatedCommandNodeCount).toBe(4);
  expect(
    () =>
      new GPUSphericalConvexContains({
        ...props,
        polygonIndices: createTransientView(graph, 'wrong-topology', 'uint32', 6)
      })
  ).toThrow(/polygonIndices.*topology/);
  expect(
    () =>
      new GPUSphericalConvexContains({
        ...props,
        output: {contains: props.output.contains, status: props.output.contains}
      })
  ).toThrow(/must not overlap/);
  expect(
    () =>
      new GPUSphericalConvexContains({
        ...props,
        positions: createVectorView('packed-only', 'float32x3', [
          props.positions
        ]) as unknown as GraphDataView<'float32x3'>
      })
  ).toThrow(/positions must be one packed view/);
});

it('prechecks later aligned chunks and graph ownership without graph mutations', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUSphericalConvexContains(createProps(graph, [1, 11_184_811]));
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const addNode = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/queries chunk 1 needs 134217740 bytes/);
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(addNode).not.toHaveBeenCalled();
  expect(() => contributor.getCommandNodes(new GPUCommandGraph(device))).toThrow(
    /views must belong/
  );
});

it('accepts 30 million rows with resident packed geometry and skips empty chunks', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUSphericalConvexContains(
    createProps(graph, [0, 10_000_000, 10_000_000, 10_000_000, 0])
  );
  expect(contributor.getCommandNodes(graph)).toHaveLength(4);
  expect(contributor.estimatedCommandNodeCount).toBe(4);
  const emptyGraph = new GPUCommandGraph(device);
  expect(
    new GPUSphericalConvexContains(createProps(emptyGraph, [])).getCommandNodes(emptyGraph)
  ).toEqual([]);
});

it('keeps one source and pipeline as query counts and uneven unaligned chunk counts grow', () => {
  const pipelines = new Set<string>();
  for (const chunkCount of [1, 3, 100]) {
    const graph = new GPUCommandGraph(device);
    const lengths = Array.from({length: chunkCount}, (_, index) => [1, 257, 513][index % 3]);
    const nodes = new GPUSphericalConvexContains(createProps(graph, lengths)).getCommandNodes(
      graph
    );
    expect(() => graph.add(nodes)).not.toThrow();
    expect(getParsedKernelSources(nodes).size).toBe(1);
    expect(
      nodes.filter(node => node.workload).map(node => node.workload?.maximumWorkgroupCount)
    ).toEqual(lengths.map(length => Math.ceil(length / 256)));
    for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
      pipelines.add(key);
    for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
      expect(constants).toEqual({});
      expectValidQueryShaderLiterals(source);
      const identifiers =
        source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(identifiers.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
    }
  }
  expect(pipelines.size).toBe(1);
});
