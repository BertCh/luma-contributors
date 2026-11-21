// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, afterEach, expect, expectTypeOf, it, vi} from 'vitest';
import {
  WGSL_RESERVED_WORDS,
  stripWGSLComments
} from '../analysis-core/kernel-audit/kernel-audit-wgsl';
import {readPointBufferRing} from './gpu-point-buffer-chunks.test-utils';
import {
  GPUPointBuffer,
  type GPUPointBufferProps,
  getGPUPointBufferParameterValues
} from '../../src/geospatial-analysis/point-buffer/index';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../utils/gpu-contributor-test-utils';

const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
afterEach(() => vi.restoreAllMocks());
let nextPropsId = 0;
function createProps(graph: GPUCommandGraph, lengths: number[], padded = false) {
  const propsId = nextPropsId++;
  const rows = (name: string, multiplier: number) =>
    createVectorView(
      name,
      'float32x2',
      lengths.map((pointCount, chunkIndex) => {
        const length = pointCount * multiplier;
        const byteOffset = padded ? (chunkIndex % 63) * 4 : 0;
        const backing = createTransientView(
          graph,
          `${name}-${propsId}-${chunkIndex}`,
          'uint32',
          Math.max(length, 1) * 2 + byteOffset / 4
        );
        return graph.createDataView(backing.buffer, {format: 'float32x2', length, byteOffset});
      })
    );
  return {
    id: 'chunk-point-buffer',
    spatialContext: {
      coordinateSpace: 'longitude-latitude' as const,
      metric: 'great-circle' as const
    },
    points: rows('points', 1),
    offsets: rows('offsets', 4),
    verticesPerRing: 4,
    parameters: createTransientView(graph, `parameters-${propsId}`, 'float32', 2)
  };
}

it('validates whole-ring topology, output overlap and uint32 expanded row counts', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 4, 2]);
  expect(new GPUPointBuffer(props).estimatedCommandNodeCount).toBe(4);
  expect(
    () => new GPUPointBuffer({...props, offsets: createProps(graph, [2, 3, 2]).offsets})
  ).toThrow(/^chunk-point-buffer offsets chunk 0.*verticesPerRing/);
  const repeated = createProps(graph, [1, 1]);
  expect(
    () =>
      new GPUPointBuffer({
        ...repeated,
        offsets: createVectorView('repeated', 'float32x2', [
          repeated.offsets.data[0],
          repeated.offsets.data[0]
        ])
      })
  ).toThrow(/byte ranges must not overlap/);
  expect(() => new GPUPointBuffer({...props, verticesPerRing: 0xffffffff})).toThrow(
    /points\.length \* verticesPerRing \(30064771065\).*GRAPH_CHUNK_MAXIMUM_ROW_COUNT \(4294967295\)/
  );
  const shared = createTransientView(graph, 'shared', 'float32x2', 5);
  expect(
    () =>
      new GPUPointBuffer({
        ...props,
        points: graph.createDataView(shared.buffer, {format: 'float32x2', length: 1}),
        offsets: graph.createDataView(shared.buffer, {
          format: 'float32x2',
          length: 4,
          byteOffset: 8
        })
      })
  ).toThrow(/share buffers/);
});

it('checks expanded later output bindings before parameter allocation or node declaration', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUPointBuffer(createProps(graph, [1, 4_194_304], true));
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const addNode = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /^chunk-point-buffer offsets chunk 1 needs 134217732 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(addNode).not.toHaveBeenCalled();
});

it('supports 30 million points with whole rings in bounded chunks', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUPointBuffer(
    createProps(graph, [0, ...new Array(10).fill(3_000_000), 0])
  );
  expect(contributor.getCommandNodes(graph)).toHaveLength(11);
});

it('keeps one source and pipeline per coordinate system as chunk count grows', () => {
  for (const coordinateSystem of ['planar', 'geographic'] as const) {
    const keys = new Set<string>();
    for (const chunkCount of [1, 3, 100]) {
      const graph = new GPUCommandGraph(device);
      const contributor = new GPUPointBuffer({
        ...createProps(
          graph,
          Array.from({length: chunkCount}, (_, index) => [1, 257, 513][index % 3]),
          true
        ),
        spatialContext:
          coordinateSystem === 'geographic'
            ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
            : {coordinateSpace: 'planar', metric: 'native'}
      });
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      expect(getParsedKernelSources(nodes).size).toBe(1);
      for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
        keys.add(key);
      for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
        expect(Object.keys(constants)).toHaveLength(0);
        const identifiers = stripWGSLComments(source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? [];
        expect(identifiers.filter(identifier => WGSL_RESERVED_WORDS.has(identifier))).toEqual([]);
      }
    }
    expect(keys.size).toBe(1);
  }
});

it('reinterprets partial ring readbacks as float32 values rather than numeric bytes', async () => {
  const rings = new Float32Array([0, 2, 2, 0, 0, -2, -2, 0, 0, 3, 3, 0, 0, -3, -3, 0]);
  const buffer = device.createBuffer({data: rings});
  try {
    const bytes = await buffer.readAsync(0, 32);
    // This is the previous headless decoder: it reproduces the reported zero instead of 2.
    expect(new Float32Array(bytes)[1]).toBe(0);
    expect(await readPointBufferRing(buffer, 0, 4)).toEqual([0, 2, 2, 0, 0, -2, -2, 0]);
    expect(await readPointBufferRing(buffer, 1, 4)).toEqual([0, 3, 3, 0, 0, -3, -3, 0]);
    // A backend may return a view into a larger allocation; its byteOffset must be honored too.
    const padded = new Uint8Array(40);
    padded.set(bytes, 4);
    const readAsync = vi.fn(async () => padded.subarray(4, 36));
    expect(await readPointBufferRing({readAsync}, 1, 4)).toEqual([0, 2, 2, 0, 0, -2, -2, 0]);
    expect(readAsync).toHaveBeenCalledWith(32, 32);
  } finally {
    buffer.destroy();
  }
});

it('prefixes chunk scratch resource IDs so two contributors can share one graph', () => {
  const graph = new GPUCommandGraph(device);
  const first = new GPUPointBuffer({...createProps(graph, [1, 4, 2]), id: 'first'});
  const second = new GPUPointBuffer({...createProps(graph, [1, 4, 2]), id: 'second'});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const firstNodes = first.getCommandNodes(graph);
  const secondNodes = second.getCommandNodes(graph);
  expect(allocate.mock.calls.map(([props]) => props.id)).toEqual(['first-chunks', 'second-chunks']);
  const nodeIds = [...firstNodes, ...secondNodes].map(node => node.id);
  expect(new Set(nodeIds).size).toBe(nodeIds.length);
  expect(firstNodes.every(node => node.id.startsWith('first-'))).toBe(true);
  expect(secondNodes.every(node => node.id.startsWith('second-'))).toBe(true);
});

it('ERG-102 distinguishes product overflow from a legal product with a short offsets view', () => {
  const graph = new GPUCommandGraph(device);
  // Transient views only declare sizes: no huge host or device allocation is needed.
  const overflow = createProps(graph, [1, 4, 2]);
  expect(() => new GPUPointBuffer({...overflow, verticesPerRing: 0xffffffff})).toThrow(
    'chunk-point-buffer points.length * verticesPerRing (30064771065) must not exceed GRAPH_CHUNK_MAXIMUM_ROW_COUNT (4294967295)'
  );
  const legalProduct = createProps(graph, [1]);
  const shortOffsets = createTransientView(graph, 'short-offsets', 'float32x2', 1);
  expect(
    () =>
      new GPUPointBuffer({
        ...legalProduct,
        points: legalProduct.points.data[0],
        offsets: shortOffsets,
        verticesPerRing: 0xffffffff
      })
  ).toThrow(/^chunk-point-buffer offsets must have verticesPerRing rows per point$/);
});

it('S1 reports the exact expanded product beyond Number.MAX_SAFE_INTEGER', () => {
  const graph = new GPUCommandGraph(device);
  // Exact integer arithmetic: 3,000,001 * 4,294,967,295 = 12,884,906,179,967,295.
  // Only view extents are declared; no large allocation is submitted.
  const props = createProps(graph, [3_000_001]);
  expect(() => new GPUPointBuffer({...props, verticesPerRing: 0xffffffff})).toThrow(
    'points.length * verticesPerRing (12884906179967295) must not exceed GRAPH_CHUNK_MAXIMUM_ROW_COUNT (4294967295)'
  );
});

it('NAM-07 rejects unknown modes and resolves one captured spatial context mode', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1]);
  for (const coordinateSystem of ['longitude-latitude', 'unrelated-token']) {
    const legacyProps = {...props, coordinateSystem};
    expect(() => new GPUPointBuffer(legacyProps)).toThrow(/coordinateSystem.*spatialContext/);
  }
  for (const mode of ['planar', 'geographic'] as const) {
    const spatialContext =
      mode === 'planar'
        ? {coordinateSpace: 'planar' as const, metric: 'native' as const}
        : {
            coordinateSpace: 'longitude-latitude' as const,
            metric: 'great-circle' as const,
            units: 'meters' as const
          };
    const input = {
      ...props,
      id: `mode-${mode}`,
      spatialContext
    };
    const contributor = new GPUPointBuffer(input);
    expectTypeOf(contributor.props.spatialContext).toEqualTypeOf<
      GPUPointBufferProps['spatialContext']
    >();
    input.spatialContext = {coordinateSpace: 'planar', metric: 'native'};
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.at(-1)?.workload?.variant).toBe(mode);
    const source = getGPUCommandNodeKernelSources(nodes)
      .map(kernel => kernel.source)
      .join('\n');
    expect(source.includes('let clampedLatitude')).toBe(mode === 'geographic');
  }
  expect(
    () =>
      new GPUPointBuffer({
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'rhumb', units: 'meters'}
      })
  ).toThrow();
});

it('VAL-02 validates encoded radius and full parameter destinations before writing', () => {
  for (const radius of [Number.MIN_VALUE, 1e-50, Number.MAX_VALUE, 0, -1, NaN, Infinity]) {
    const target = new Float32Array([77, 88, 99]);
    expect(() => getGPUPointBufferParameterValues(radius, target)).toThrow(/radius/);
    expect([...target]).toEqual([77, 88, 99]);
  }
  for (const length of [0, 1]) {
    const target = new Float32Array(length).fill(77);
    expect(() => getGPUPointBufferParameterValues(10, target)).toThrow(/target.length/);
    expect([...target]).toEqual(new Array(length).fill(77));
  }
  for (const length of [2, 4]) {
    const target = new Float32Array(length).fill(77);
    expect(getGPUPointBufferParameterValues(10, target)).toBe(target);
    expect([...target]).toEqual([10, 0, ...new Array(length - 2).fill(77)]);
  }
});

it('B1 accepts effective metre units', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1]);
  expect(
    () =>
      new GPUPointBuffer({
        ...props,
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
      })
  ).not.toThrow();
});

it('B1 rejects explicit null context instead of resolving a default', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1]);
  expect(() => new GPUPointBuffer({...props, spatialContext: null as never})).toThrow(
    /spatialContext.*null/
  );
});

it('S4 snapshots all nested spatial context metadata exposed in props', () => {
  const graph = new GPUCommandGraph(device);
  const spatialContext = {
    coordinateSpace: 'planar' as const,
    metric: 'native' as const,
    frame: {
      id: 'local',
      kind: 'crs' as const,
      origin: [1, 2] as [number, number],
      areaOfUse: [0, 0, 3, 4] as [number, number, number, number]
    }
  };
  const contributor = new GPUPointBuffer({
    ...createProps(graph, [1]),
    spatialContext
  });
  spatialContext.frame.origin[0] = 99;
  spatialContext.frame.areaOfUse[0] = 99;
  spatialContext.frame.id = 'changed';
  spatialContext.coordinateSpace = 'longitude-latitude' as never;
  expect(contributor.props.spatialContext).toEqual({
    coordinateSpace: 'planar',
    metric: 'native',
    frame: {id: 'local', kind: 'crs', origin: [1, 2], areaOfUse: [0, 0, 3, 4]}
  });
});

it('B5 requires a mode declaration in the public input type', () => {
  expectTypeOf<Omit<GPUPointBufferProps, 'spatialContext'>>().not.toExtend<GPUPointBufferProps>();
});
