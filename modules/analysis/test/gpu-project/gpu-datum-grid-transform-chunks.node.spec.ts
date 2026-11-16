// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {
  WGSL_RESERVED_WORDS,
  stripWGSLComments
} from '../analysis-core/kernel-audit/kernel-audit-wgsl';
import {GPUDatumGridTransform} from '../../src/geospatial-analysis/gpu-datum-grid-transform';
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
  const rows = <Format extends 'float32x3' | 'uint32'>(name: string, format: Format) =>
    createVectorView(
      name,
      format,
      lengths.map((length, chunkIndex) => {
        const byteOffset = padded ? (chunkIndex % 63) * 4 : 0;
        const components = format === 'float32x3' ? 3 : 1;
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
    id: 'chunk-datum',
    positions: rows('positions', 'float32x3'),
    positionsLow: rows('positionsLow', 'float32x3'),
    samples: createTransientView(graph, `samples-${propsId}`, 'float32x3', 4),
    mask: createTransientView(graph, `mask-${propsId}`, 'uint32', 4),
    grids: [
      {width: 2, height: 2, sampleOffset: 0, origin: [0, 0] as const, spacing: [1, 1] as const}
    ],
    operation: 'combined' as const,
    sourceVerticalReference: 'source',
    targetVerticalReference: 'target',
    output: {
      positions: rows('output', 'float32x3'),
      positionsLow: rows('outputLow', 'float32x3'),
      status: rows('status', 'uint32')
    }
  };
}

it('validates all row topologies, low-word pairing and writable ranges at construction', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 4, 2]);
  expect(new GPUDatumGridTransform(props).estimatedCommandNodeCount).toBe(4);
  expect(
    () =>
      new GPUDatumGridTransform({
        ...props,
        positionsLow: createProps(graph, [2, 3, 2]).positionsLow
      })
  ).toThrow(/^chunk-datum positionsLow chunk 0.*topology/);
  expect(
    () =>
      new GPUDatumGridTransform({
        ...props,
        output: {...props.output, positionsLow: props.output.positions}
      })
  ).toThrow(/byte ranges must not overlap/);
  expect(
    () => new GPUDatumGridTransform({...props, output: {...props.output, positionsLow: undefined}})
  ).toThrow(/must be given together/);
  expect(
    () =>
      new GPUDatumGridTransform({...props, output: {...props.output, positions: props.positions}})
  ).toThrow(/share buffers/);
});

it('checks later aligned row bindings and the resident grid before any graph mutation', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 11_184_811], true);
  const contributor = new GPUDatumGridTransform(props);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const addNode = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /^chunk-datum positions chunk 1 needs 134217736 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(addNode).not.toHaveBeenCalled();
  const small = createProps(graph, [1]);
  const oversizedSamples = createTransientView(graph, 'large-grid', 'float32x3', 11_184_811);
  const oversizedMask = createTransientView(graph, 'large-mask', 'uint32', 11_184_811);
  const resident = new GPUDatumGridTransform({
    ...small,
    samples: oversizedSamples,
    mask: oversizedMask
  });
  allocate.mockClear();
  createView.mockClear();
  expect(() => resident.getCommandNodes(graph)).toThrow(/samples chunk 0 needs 134217732 bytes/);
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
});

it('supports 30 million rows in bounded chunks, skipping empty chunks', () => {
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUDatumGridTransform(
    createProps(graph, [0, 10_000_000, 10_000_000, 10_000_000, 0])
  );
  expect(contributor.getCommandNodes(graph)).toHaveLength(4);
});

it('keeps one source and pipeline per direction as chunk count grows', () => {
  for (const direction of ['forward', 'inverse'] as const) {
    const keys = new Set<string>();
    for (const chunkCount of [1, 3, 100]) {
      const graph = new GPUCommandGraph(device);
      const contributor = new GPUDatumGridTransform({
        ...createProps(
          graph,
          Array.from({length: chunkCount}, (_, index) => [1, 257, 513][index % 3]),
          true
        ),
        direction
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

it('bounds grid indices and rejects numbers that cannot become valid WGSL literals at construction', () => {
  const graph = new GPUCommandGraph(device);
  const props = createProps(graph, [1, 4, 2]);
  // Metadata-only graph resources: no GPU allocation is created for this impossible binding.
  const samples = createTransientView(graph, 'unaddressable-samples', 'float32x3', 0x100000000);
  const mask = createTransientView(graph, 'unaddressable-mask', 'uint32', 0x100000000);
  expect(() => new GPUDatumGridTransform({...props, samples, mask})).toThrow(
    /^chunk-datum samples.length.*1431655701/
  );
  for (const value of [NaN, Infinity, -Infinity, 3.5e38]) {
    expect(
      () =>
        new GPUDatumGridTransform({
          ...props,
          grids: [{...props.grids[0], origin: [value, 0]}]
        })
    ).toThrow(/^chunk-datum grids\[0\].origin\[0\]/);
    expect(
      () =>
        new GPUDatumGridTransform({
          ...props,
          grids: [{...props.grids[0], spacing: [value, 1]}]
        })
    ).toThrow(/^chunk-datum grids\[0\].spacing\[0\]/);
    expect(() => new GPUDatumGridTransform({...props, tolerance: value})).toThrow(
      /^chunk-datum tolerance/
    );
    expect(() => new GPUDatumGridTransform({...props, heightUnitMeters: value})).toThrow(
      /^chunk-datum heightUnitMeters/
    );
    expect(() => new GPUDatumGridTransform({...props, maximumIterations: value})).toThrow(
      /^chunk-datum maximumIterations/
    );
  }
  expect(
    () =>
      new GPUDatumGridTransform({
        ...props,
        grids: [{...props.grids[0], width: 0xffffffff}]
      })
  ).toThrow(/sampleOffset \+ width \* height/);
});

it('prefixes chunk scratch resource IDs so two contributors can share one graph', () => {
  const graph = new GPUCommandGraph(device);
  const first = new GPUDatumGridTransform({...createProps(graph, [1, 4, 2]), id: 'first'});
  const second = new GPUDatumGridTransform({...createProps(graph, [1, 4, 2]), id: 'second'});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const firstNodes = first.getCommandNodes(graph);
  const secondNodes = second.getCommandNodes(graph);
  expect(allocate.mock.calls.map(([props]) => props.id)).toEqual(['first-chunks', 'second-chunks']);
  const nodeIds = [...firstNodes, ...secondNodes].map(node => node.id);
  expect(new Set(nodeIds).size).toBe(nodeIds.length);
  expect(firstNodes.every(node => node.id.startsWith('first-'))).toBe(true);
  expect(secondNodes.every(node => node.id.startsWith('second-'))).toBe(true);
});
