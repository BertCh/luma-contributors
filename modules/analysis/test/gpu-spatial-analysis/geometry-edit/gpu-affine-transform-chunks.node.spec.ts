// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {
  WGSL_RESERVED_WORDS,
  stripWGSLComments
} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {GPUAffineTransform} from '../../../src/gpu-spatial-analysis/geometry-edit/gpu-affine-transform';
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
  const rows = (name: string) =>
    createVectorView(
      name,
      'float32x2',
      lengths.map((length, chunkIndex) => {
        const byteOffset = padded ? 4 * (chunkIndex + (name === 'positions' ? 1 : 2)) : 0;
        const backing = createTransientView(
          graph,
          `${name}-${propsId}-${chunkIndex}`,
          'float32',
          Math.max(length, 1) * 2 + byteOffset / 4
        );
        return graph.createDataView(backing.buffer, {format: 'float32x2', length, byteOffset});
      })
    );
  return {
    id: 'chunk-affine',
    positions: rows('positions'),
    parameters: createTransientView(graph, `parameters-${propsId}`, 'float32', 12),
    output: {positions: rows('output')}
  };
}

it('validates topology, formats, computed origins, input aliases and overlapping writable chunks', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 4, 2]);
  expect(new GPUAffineTransform(props).estimatedCommandNodeCount).toBe(4);
  const wrongTopology = createProps(graph, [2, 3, 2]).output;
  expect(() => new GPUAffineTransform({...props, output: wrongTopology})).toThrow(
    /^chunk-affine output.positions chunk 0.*topology/
  );
  expect(
    () =>
      new GPUAffineTransform({
        ...props,
        origins: ['center'],
        lineOffsets: createTransientView(graph, 'lines', 'uint32', 3)
      })
  ).toThrow(/computed origins must be one packed view/);
  expect(() => new GPUAffineTransform({...props, output: {positions: props.positions}})).toThrow(
    /share buffers/
  );
  const repeated = createTransientView(graph, 'repeated', 'float32x2', 2);
  const aliasProps = createProps(graph, [2, 2]);
  expect(
    () =>
      new GPUAffineTransform({
        ...aliasProps,
        output: {positions: createVectorView('alias', 'float32x2', [repeated, repeated])}
      })
  ).toThrow(/byte ranges must not overlap/);
  const invalid = graph.createDataView(repeated.buffer, {format: 'uint32x2', length: 2});
  expect(
    () =>
      new GPUAffineTransform({
        ...createProps(graph, [2]),
        positions: createVectorView('bad', 'uint32x2', [invalid])
      } as never)
  ).toThrow(/must be packed.*float32x2/);
});

it('checks later aligned bindings and graph ownership before declaring any resources or nodes', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 16_777_216], true);
  const contributor = new GPUAffineTransform(props);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const addNode = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /^chunk-affine positions chunk 1 needs 134217736 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(addNode).not.toHaveBeenCalled();
  const foreign = new GPUCommandGraph(device);
  expect(() => new GPUAffineTransform(createProps(graph, [1])).getCommandNodes(foreign)).toThrow(
    /target graph/
  );
});

it('declares 30 million rows in bounded chunks and skips empty chunks', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUAffineTransform(
    createProps(graph, [0, 10_000_000, 10_000_000, 10_000_000, 0])
  );
  expect(contributor.getCommandNodes(graph)).toHaveLength(4);
});

it('shares one source and pipeline for 1, 3 and 100 uneven, unaligned chunks', () => {
  for (const perFeature of [false, true]) {
    const keys = new Set<string>();
    for (const chunkCount of [1, 3, 100]) {
      const graph = new GPUCommandGraph(device);
      const props = createProps(
        graph,
        Array.from({length: chunkCount}, (_, index) => [1, 257, 513][index % 3]),
        true
      );
      const contributor = new GPUAffineTransform(
        perFeature
          ? {
              ...props,
              parameters: undefined,
              geometryType: 'polygons',
              ringOffsets: createTransientView(graph, 'rings', 'uint32', 3),
              polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 3),
              featureOffsets: createTransientView(graph, 'features', 'uint32', 3),
              featureTransforms: createTransientView(graph, 'transforms', 'float32', 12)
            }
          : props
      );
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount!);
      expect(getParsedKernelSources(nodes).size).toBe(1);
      for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
        keys.add(key);
      for (const {source, constants} of getGPUCommandNodeKernelSources(nodes)) {
        expect(Object.keys(constants)).toHaveLength(0);
        expect(
          (stripWGSLComments(source).match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(identifier =>
            WGSL_RESERVED_WORDS.has(identifier)
          )
        ).toEqual([]);
      }
    }
    expect(keys.size).toBe(1);
  }
});

it('pre-checks computed-origin feature-statistics scratch before allocating bounds or nodes', () => {
  const graph = new GPUCommandGraph(device);
  // Bounds fit at 32 MiB, but the measurement stage's 64-byte statistics records exceed 128 MiB.
  const featureCount = 2_097_153;
  const contributor = new GPUAffineTransform({
    id: 'origin-scratch',
    geometryType: 'lines',
    origins: ['center'],
    positions: createTransientView(graph, 'origin-positions', 'float32x2', 1),
    lineOffsets: createTransientView(graph, 'origin-lines', 'uint32', featureCount + 1),
    parameters: createTransientView(graph, 'origin-parameters', 'float32', 12),
    output: {positions: createTransientView(graph, 'origin-output', 'float32x2', 1)}
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /^origin-scratch origin feature stats needs 134217792 bytes in one binding/
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
});

it('rejects offset counts outside the uint32 domain before generating binary-search literals', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 4, 2]);
  const lineOffsets = createTransientView(graph, 'unaddressable-lines', 'uint32', 0x100000001);
  expect(() => new GPUAffineTransform({...props, lineOffsets})).toThrow(
    /^chunk-affine ringOffsets.length.*4294967295/
  );
});

it('pre-checks computed-origin scratch against maxBufferSize before declaring any resources', () => {
  const constrainedDevice = createNullWebGPUDevice();
  Object.defineProperty(constrainedDevice.limits, 'maxStorageBufferBindingSize', {value: 1024});
  Object.defineProperty(constrainedDevice.limits, 'maxBufferSize', {value: 512});
  const graph = new GPUCommandGraph(constrainedDevice);
  const contributor = new GPUAffineTransform({
    id: 'origin-allocation',
    geometryType: 'lines',
    origins: ['center'],
    positions: createTransientView(graph, 'positions', 'float32x2', 1),
    lineOffsets: createTransientView(graph, 'lines', 'uint32', 10),
    parameters: createTransientView(graph, 'parameters', 'float32', 12),
    output: {positions: createTransientView(graph, 'output', 'float32x2', 1)}
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  try {
    // Nine 64-byte statistics records fit the binding limit but exceed the allocation limit.
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /^origin-allocation origin feature stats byteLength must be an integer in \[0, 512\]$/
    );
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  } finally {
    constrainedDevice.destroy();
  }
});

it('prefixes chunk scratch resource IDs so two contributors can share one graph', () => {
  const graph = new GPUCommandGraph(device);
  const first = new GPUAffineTransform({...createProps(graph, [1, 4, 2]), id: 'first'});
  const second = new GPUAffineTransform({...createProps(graph, [1, 4, 2]), id: 'second'});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const firstNodes = first.getCommandNodes(graph);
  const secondNodes = second.getCommandNodes(graph);
  expect(allocate.mock.calls.map(([props]) => props.id)).toEqual(['first-chunks', 'second-chunks']);
  const nodeIds = [...firstNodes, ...secondNodes].map(node => node.id);
  expect(new Set(nodeIds).size).toBe(nodeIds.length);
  expect(firstNodes.every(node => node.id.startsWith('first-'))).toBe(true);
  expect(secondNodes.every(node => node.id.startsWith('second-'))).toBe(true);
});
