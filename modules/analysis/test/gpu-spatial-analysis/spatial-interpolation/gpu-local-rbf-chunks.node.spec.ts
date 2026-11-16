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
  GPULocalRBF,
  type GPULocalRBFProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-rbf';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {expectValidQueryShaderLiterals} from '../linear-referencing/gpu-linear-referencing-chunks.test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

type ChunkedProps = GPULocalRBFProps<
  GraphVectorView<'float32x2'>,
  GraphVectorView<'float32x4'>,
  GraphVectorView<'uint32'>
>;

const device = createNullWebGPUDevice();
afterEach(() => vi.restoreAllMocks());
afterAll(() => device.destroy());

function createProps(graph: GPUCommandGraph, lengths: number[], unaligned = false): ChunkedProps {
  const createRows = <Format extends 'float32x2' | 'float32x4' | 'uint32'>(
    name: string,
    format: Format
  ) =>
    createVectorView(
      name,
      format,
      lengths.map((length, chunkIndex) => {
        const backing = createTransientView(graph, `${name}-${chunkIndex}`, format, length + 16);
        return graph.createDataView(backing.buffer, {
          format,
          length,
          byteOffset: unaligned ? ((chunkIndex % 3) + 1) * 4 : 0
        });
      })
    );
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', 3),
    values: createTransientView(graph, 'values', 'float32', 3),
    queries: createRows('queries', 'float32x2'),
    neighborCount: 3,
    output: {values: createRows('results', 'float32x4'), status: createRows('statuses', 'uint32')}
  };
}

it('validates row topology, packed shared views, and writable aliases at construction', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 3, 2]);
  expect(new GPULocalRBF(props).estimatedCommandNodeCount).toBe(3);
  expect(
    () =>
      new GPULocalRBF({
        ...props,
        output: {...props.output, status: createTransientView(graph, 'wrong-topology', 'uint32', 6)}
      })
  ).toThrow(/status.*topology/);
  expect(
    () =>
      new GPULocalRBF({
        ...props,
        positions: createVectorView('shared', 'float32x2', [
          props.positions
        ]) as unknown as GraphDataView<'float32x2'>
      })
  ).toThrow(/positions must be one packed view/);
  const overlappingGraph = new GPUCommandGraph(device);
  const overlapping = createProps(overlappingGraph, [2, 2]);
  const chunks = 'data' in overlapping.output.values ? overlapping.output.values.data : [];
  overlapping.output.status = createVectorView(
    'alias',
    'uint32',
    chunks.map(chunk =>
      overlappingGraph.createDataView(chunk.buffer, {format: 'uint32', length: chunk.length})
    )
  );
  expect(() => new GPULocalRBF(overlapping)).toThrow(/must not overlap/);
  const repeated = createProps(new GPUCommandGraph(device), [2, 2]);
  const repeatedValues = 'data' in repeated.output.values ? repeated.output.values.data : [];
  repeated.output.values = createVectorView('repeated', 'float32x4', [
    repeatedValues[0],
    repeatedValues[0]
  ]);
  expect(() => new GPULocalRBF(repeated)).toThrow(/must not overlap/);
});

it('prechecks the actual aligned range of later chunks before graph mutations', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 8_388_608], true);
  const contributor = new GPULocalRBF(props);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const addNode = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/results chunk 1 needs 134217736 bytes/);
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(addNode).not.toHaveBeenCalled();
});

it('keeps CSR candidates and sample smoothing within eight storage bindings', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 3, 2]);
  props.neighborhoods = createTransientView(graph, 'neighbors', 'uint32', 6);
  props.neighborhoodOffsets = createTransientView(graph, 'offsets', 'uint32', 7);
  props.sampleSmoothing = createTransientView(graph, 'smoothing', 'float32', 3);
  const contributor = new GPULocalRBF(props);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(contributor.getCommandNodes(graph)).toHaveLength(3);
  expect(allocate).not.toHaveBeenCalled();
});

it('accepts 30 million query rows when each chunk fits the device binding limit', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPULocalRBF(
    createProps(graph, [6_000_000, 6_000_000, 6_000_000, 6_000_000, 6_000_000])
  );
  expect(contributor.getCommandNodes(graph)).toHaveLength(5);
});

it('keeps one shader and pipeline across growing query counts and uneven unaligned chunks', () => {
  const pipelines = new Set<string>();
  for (const chunkCount of [1, 3, 100]) {
    const graph = new GPUCommandGraph(device);
    const props = createProps(
      graph,
      Array.from({length: chunkCount}, (_, index) => [1, 33, 65][index % 3]),
      true
    );
    const nodes = new GPULocalRBF(props).getCommandNodes(graph);
    expect(() => graph.add(nodes)).not.toThrow();
    expect(getParsedKernelSources(nodes).size).toBe(1);
    expect(nodes.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
      Array.from({length: chunkCount}, (_, index) => Math.ceil([1, 33, 65][index % 3] / 32))
    );
    for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
      pipelines.add(key);
    for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
      expect(constants).toEqual({});
      expectValidQueryShaderLiterals(source);
      const words =
        source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
      expect(words.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
    }
  }
  expect(pipelines.size).toBe(1);
});

it('keeps CSR candidate lengths out of pipeline specialization', () => {
  const pipelines = new Set<string>();
  for (const lengths of [[1], [1, 3, 2], [65, 1, 33, 2]]) {
    const graph = new GPUCommandGraph(device);
    const props = createProps(graph, lengths, true);
    const rowCount = lengths.reduce((total, length) => total + length, 0);
    props.neighborhoods = createTransientView(graph, 'neighborhoods', 'uint32', rowCount * 3);
    props.neighborhoodOffsets = createTransientView(graph, 'offsets', 'uint32', rowCount + 1);
    const nodes = new GPULocalRBF(props).getCommandNodes(graph);
    expect(getParsedKernelSources(nodes).size).toBe(1);
    for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
      pipelines.add(key);
  }
  expect(pipelines.size).toBe(1);
});

it('preserves packed oversized output capacities and validates graph ownership before mutation', () => {
  const graph = new GPUCommandGraph(device);
  const props: GPULocalRBFProps = {
    positions: createTransientView(graph, 'positions', 'float32x2', 3),
    values: createTransientView(graph, 'values', 'float32', 3),
    queries: createTransientView(graph, 'queries', 'float32x2', 2),
    output: {
      values: createTransientView(graph, 'output-values', 'float32x4', 5),
      status: createTransientView(graph, 'output-status', 'uint32', 5)
    }
  };
  const contributor = new GPULocalRBF(props);
  expect(contributor.getCommandNodes(graph)).toHaveLength(1);
  const foreignGraph = new GPUCommandGraph(device);
  const allocate = vi.spyOn(foreignGraph, 'createTransientBuffer');
  expect(() => contributor.getCommandNodes(foreignGraph)).toThrow(
    /views must belong to the target graph/
  );
  expect(allocate).not.toHaveBeenCalled();
});

it('validates float32 model literals at construction and preserves unbounded searches', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 3, 2]);
  for (const prop of ['epsilon', 'smoothing', 'searchRadius'] as const)
    for (const value of [1e100, Number.NaN])
      expect(() => new GPULocalRBF({...props, [prop]: value})).toThrow(
        new RegExp(`${prop}.*finite`)
      );
  expect(() => new GPULocalRBF({...props, epsilon: Number.MIN_VALUE})).toThrow(/epsilon.*finite/);
  const nodes = new GPULocalRBF({...props, searchRadius: Infinity}).getCommandNodes(graph);
  expect(() => graph.add(nodes)).not.toThrow();
  for (const {source} of getGPUCommandNodeKernelSources(nodes))
    expectValidQueryShaderLiterals(source);
});
