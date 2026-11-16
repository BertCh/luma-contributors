// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphVectorView} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {
  GPULocalKriging,
  type GPULocalKrigingProps
} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {expectValidQueryShaderLiterals} from '../linear-referencing/gpu-linear-referencing-chunks.test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';

type ChunkedProps = GPULocalKrigingProps<
  GraphVectorView<'float32x2' | 'float32x4'>,
  GraphVectorView<'float32x4'>,
  GraphVectorView<'uint32'>,
  GraphVectorView<'float32'>,
  GraphVectorView<'float32' | 'float32x4'>
>;

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());

for (const candidates of ['fixed', 'csr'] as const) {
  it(`keeps source-row tie order and one supplied-${candidates} solve pipeline across growing chunks`, () => {
    const pipelines = new Set<string>();
    for (const chunkCount of [1, 3, 9]) {
      const graph = new GPUCommandGraph(device);
      const lengths = Array.from({length: chunkCount}, (_, index) => [31, 32, 33][index % 3]);
      const props = createProps(graph, lengths, 'brute-force');
      props.mode = 'ordinary';
      props.sampleDrift = undefined;
      props.queryDrift = undefined;
      props.output.weights = undefined;
      const total = lengths.reduce((sum, length) => sum + length, 0);
      props.neighborhoods = createTransientView(graph, 'neighbors', 'uint32', total * 2);
      props.neighborhoodCapacity = 2;
      if (candidates === 'csr')
        props.neighborhoodOffsets = createTransientView(graph, 'offsets', 'uint32', total + 1);
      const nodes = new GPULocalKriging(props)
        .getCommandNodes(graph)
        .filter(node => node.workload?.variant === 'ordinary');
      expect(nodes.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
        lengths.map(length => Math.ceil(length / 32))
      );
      for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
        pipelines.add(key);
      for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
        expect(source).toContain(
          'separation == distances[neighbor] && sampleIndex < neighbors[neighbor]'
        );
        expectValidQueryShaderLiterals(source);
        const words =
          source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(words.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
      }
    }
    expect(pipelines.size).toBe(1);
  });
}
function createProps(
  graph: GPUCommandGraph,
  lengths: number[],
  sampleIndex: 'grid' | 'brute-force' = 'grid'
): ChunkedProps {
  const rows = <Format extends 'float32x2' | 'float32x4' | 'float32' | 'uint32'>(
    name: string,
    format: Format,
    width = 1
  ) =>
    createVectorView(
      name,
      format,
      lengths.map((length, index) => {
        const backing = createTransientView(graph, `${name}-${index}`, format, length * width + 64);
        return graph.createDataView(backing.buffer, {
          format,
          length: length * width,
          byteOffset: ((index % 3) + 1) * 4
        });
      })
    );
  return {
    positions: createTransientView(graph, 'positions', 'float32x2', 3),
    observations: createTransientView(graph, 'observations', 'float32x4', 3),
    supports: rows('supports', 'float32x4', 2),
    supportCount: 2,
    neighborCount: 3,
    mode: 'external-drift',
    driftCount: 3,
    sampleDrift: createTransientView(graph, 'sample-drift', 'float32', 9),
    queryDrift: rows('query-drift', 'float32', 3),
    sampleIndex,
    sampleIndexChunkQueryCount: 33,
    covariance: {model: 'exponential', sill: 1, range: 2},
    output: {
      values: rows('results', 'float32x4'),
      status: rows('status', 'uint32'),
      weights: rows('weights', 'float32', 3)
    }
  };
}

it('validates complete support, drift and weight groups and aligned writable aliases', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 3, 2]);
  expect(
    () =>
      new GPULocalKriging({
        ...props,
        queryDrift: createTransientView(graph, 'wrong', 'float32', 18)
      })
  ).toThrow(/queryDrift.*topology/);
  const backing = createTransientView(graph, 'broken-supports', 'float32x4', 13);
  const supports = createVectorView('split-support', 'float32x4', [
    graph.createDataView(backing.buffer, {format: 'float32x4', length: 3}),
    graph.createDataView(backing.buffer, {format: 'float32x4', length: 5}),
    graph.createDataView(backing.buffer, {format: 'float32x4', length: 4})
  ]);
  expect(() => new GPULocalKriging({...props, supports})).toThrow(/query chunk 0 rows/);
  expect(
    () =>
      new GPULocalKriging({
        ...props,
        output: {
          ...props.output,
          weights: createTransientView(graph, 'wrong-weights', 'float32', 18)
        }
      })
  ).toThrow(/weights.*topology/);
  const aliasProps = createProps(new GPUCommandGraph(device), [1, 3, 2]);
  aliasProps.output.weights = aliasProps.queryDrift as NonNullable<
    ChunkedProps['output']['weights']
  >;
  expect(() => new GPULocalKriging(aliasProps)).toThrow(
    /outputs must not share buffers with inputs/
  );
});

it('prechecks derived observation/covariance packing before graph mutation', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 3, 2]);
  props.positions = createTransientView(graph, 'many-positions', 'float32x2', 8_388_608);
  props.observations = createTransientView(graph, 'many-observations', 'float32x4', 8_388_608);
  props.sampleDrift = createTransientView(graph, 'many-drift', 'float32', 8_388_608 * 3);
  const contributor = new GPULocalKriging(props);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  expect(() => contributor.getCommandNodes(graph)).toThrow(/observations plus covariance needs/);
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
});

it('accepts 30 million query rows without a combined samples-plus-queries target binding', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, Array(10).fill(3_000_000));
  props.supportCount = 2;
  props.sampleIndexChunkQueryCount = undefined;
  const contributor = new GPULocalKriging(props);
  const nodes = contributor.getCommandNodes(graph);
  expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
  expect(nodes.filter(node => node.id.endsWith('-index-prepare'))).toHaveLength(1);
  expect(nodes.filter(node => node.workload?.variant === 'sample-index-candidates')).toHaveLength(
    10
  );
});

for (const sampleIndex of ['grid', 'brute-force'] as const)
  it(`keeps shader/pipeline counts constant across external/internal chunks (${sampleIndex})`, () => {
    const pipelines = new Set<string>();
    let pipelineCount = 0;
    for (const count of [1, 3, 100]) {
      const graph = new GPUCommandGraph(device);
      const lengths = Array.from({length: count}, (_, index) => [1, 65, 129][index % 3]);
      const contributor = new GPULocalKriging(createProps(graph, lengths, sampleIndex));
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      const keys = getGPUCommandNodePipelineKeys(nodes);
      pipelineCount = keys.size;
      for (const key of keys) pipelines.add(key);
      const kernels = getGPUCommandNodeKernelSources(nodes);
      for (const {source} of kernels) {
        expect(new WgslReflect(source).storage.length).toBeLessThanOrEqual(8);
        expectValidQueryShaderLiterals(source);
        const identifiers =
          source.replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(identifiers.filter(word => WGSL_RESERVED_WORDS.has(word))).toEqual([]);
        expect(source).not.toMatch(/QUERY_START|POSITIONS_BASE/);
      }
      expect(() => graph.add(nodes)).not.toThrow();
      const queryNodes = nodes.filter(node => node.workload?.variant === 'external-drift');
      const expectedLengths = lengths.flatMap(length =>
        sampleIndex === 'grid'
          ? Array.from({length: Math.ceil(length / 33)}, (_, index) =>
              Math.min(33, length - index * 33)
            )
          : [length]
      );
      expect(queryNodes.map(node => node.workload?.maximumInvocationCount)).toEqual(
        expectedLengths
      );
      expect(queryNodes.map(node => node.workload?.maximumWorkgroupCount)).toEqual(
        expectedLengths.map(length => Math.ceil(length / 32))
      );
    }
    expect(pipelines.size).toBe(pipelineCount);
  });

it('chooses the shared grid using the actual device binding limit above default-device sample sizes', () => {
  const largerDevice = createNullWebGPUDevice();
  Object.defineProperty(largerDevice.limits, 'maxStorageBufferBindingSize', {value: 536870912});
  Object.defineProperty(largerDevice.limits, 'maxBufferSize', {value: 536870912});
  try {
    const graph = new GPUCommandGraph(largerDevice);
    const props = createProps(graph, [1000]);
    props.positions = createTransientView(graph, 'large-positions', 'float32x2', 10_000_000);
    props.observations = createTransientView(graph, 'large-observations', 'float32x4', 10_000_000);
    props.sampleDrift = createTransientView(graph, 'large-drift', 'float32', 30_000_000);
    props.sampleIndex = 'auto';
    const contributor = new GPULocalKriging(props);
    expect(contributor.sampleIndex).toBe('grid');
    expect(contributor.getCommandNodes(graph).length).toBeLessThanOrEqual(
      contributor.estimatedCommandNodeCount
    );
  } finally {
    largerDevice.destroy();
  }
});

it('validates float32 model literals before allocating covariance or index scratch', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 3, 2]);
  for (const value of [1e100, Number.NaN]) {
    expect(() => new GPULocalKriging({...props, means: [value]})).toThrow(/means\[0\].*finite/);
    expect(() => new GPULocalKriging({...props, searchRadius: value})).toThrow(
      /searchRadius.*finite/
    );
  }
  const nodes = new GPULocalKriging({...props, searchRadius: Infinity}).getCommandNodes(graph);
  expect(() => graph.add(nodes)).not.toThrow();
  for (const {source} of getGPUCommandNodeKernelSources(nodes))
    expectValidQueryShaderLiterals(source);
});
