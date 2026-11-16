// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPURasterRegionGraphOperation} from '../../../src/raster-analysis/remote-sensing/gpu-raster-region-graph-operation';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPURasterRegionGraph} from '../../../src/raster-analysis/remote-sensing/gpu-raster-region-graph';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

let serial = 0;
function createCase(graph: GPUCommandGraph, regionCount: number, maximumIterations: number) {
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `boruvka-${serial++}`, format, length);
  return {
    grid: makeGPURasterMetadata({width: regionCount, height: 1}),
    regionCount,
    maximumIterations,
    mergeThreshold: 1,
    searchStructure: 'boruvka' as const,
    labels: view('uint32', regionCount),
    values: view('float32', regionCount),
    output: {
      adjacency: view('uint32', regionCount ** 2),
      weights: view('float32', regionCount ** 2),
      mapping: view('uint32', regionCount),
      status: {iterationLimitReached: view('uint32', 1)}
    }
  };
}

it('GPU-13 accepts more than 256 regions and reuses pipelines across Boruvka rounds', () => {
  const device = createNullWebGPUDevice();
  try {
    const signatures: Set<string>[] = [];
    for (const rounds of [1, 10]) {
      const graph = new GPUCommandGraph(device);
      const contributor = new GPURasterRegionGraph(createCase(graph, 513, rounds));
      const nodes = contributor.getCommandNodes(graph);
      const kernels = getGPUCommandNodeKernelSources(nodes.filter(node => node.type === 'compute'));
      for (const kernel of kernels) {
        expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
        expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      }
      signatures.push(
        new Set(kernels.map(kernel => `${kernel.source}\n${JSON.stringify(kernel.constants)}`))
      );
      expect(nodes).toHaveLength(contributor.estimatedCommandNodeCount);
    }
    expect([...signatures[0]].sort()).toEqual([...signatures[1]].sort());
  } finally {
    device.destroy();
  }
});

it('GPU-13 preflights dense outputs before declaring scratch or nodes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPURasterRegionGraph(createCase(graph, 16, 4));
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 512});
    const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/limit 512/);
    expect(createBuffer).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('GPU-13 retains explicit dense/adaptive limits and auto bounded-propagation semantics', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createCase(graph, 8, 1);
    expect(new GPURasterRegionGraph({...props, searchStructure: 'auto'}).searchStructure).toBe(
      'dense'
    );
    expect(
      new GPURasterRegionGraph({...props, searchStructure: 'auto', maximumIterations: 7})
        .searchStructure
    ).toBe('boruvka');
    for (const mergeMode of ['fixed', 'adaptive'] as const) {
      const contributor = new GPURasterRegionGraph({
        ...props,
        id: `dense-${mergeMode}`,
        searchStructure: 'dense',
        mergeMode
      });
      expect(contributor.getCommandNodes(graph)).toHaveLength(
        contributor.estimatedCommandNodeCount
      );
    }
    const large = createCase(graph, 257, 9);
    expect(() => new GPURasterRegionGraph({...large, searchStructure: 'dense'})).toThrow(
      /\[1, 256\]/
    );
    expect(
      () => new GPURasterRegionGraph({...large, searchStructure: 'auto', mergeMode: 'adaptive'})
    ).toThrow(/\[1, 256\]/);
    expect(() => new GPURasterRegionGraph({...props, mergeMode: 'adaptive'})).toThrow(
      /boruvka.*fixed/
    );
    expect(() => new GPURasterRegionGraph({...large, regionCount: 65536})).toThrow(/65535/);
  } finally {
    device.destroy();
  }
});

it('GPU-13 isolates every aligned input/output binding failure and boundary-edge scratch', () => {
  const device = createNullWebGPUDevice();
  try {
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 192});
    for (const field of [
      'labels',
      'values',
      'adjacency',
      'weights',
      'mapping',
      'iterationLimitReached'
    ]) {
      const graph = new GPUCommandGraph(device);
      const props = createCase(graph, 4, 2);
      const original =
        field === 'labels' || field === 'values'
          ? props[field]
          : field === 'iterationLimitReached'
            ? props.output.status[field]
            : props.output[field as 'mapping'];
      const buffer = createTransientView(
        graph,
        `bad-${field}`,
        original.format,
        original.length + 49
      );
      const bad = graph.createDataView(buffer.buffer, {
        format: original.format,
        length: original.length,
        byteOffset: 196
      });
      const modified =
        field === 'labels' || field === 'values'
          ? {...props, [field]: bad}
          : {
              ...props,
              output:
                field === 'iterationLimitReached'
                  ? {...props.output, status: {iterationLimitReached: bad}}
                  : {...props.output, [field]: bad}
            };
      const contributor = new GPURasterRegionGraph(modified as typeof props);
      const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
      const createView = vi.spyOn(graph, 'createDataView');
      expect(() => contributor.getCommandNodes(graph), field).toThrow(/limit 192/);
      expect(createBuffer, field).not.toHaveBeenCalled();
      expect(createView, field).not.toHaveBeenCalled();
    }
    const graph = new GPUCommandGraph(device);
    const props = createCase(graph, 4, 2);
    const contributor = new GPURasterRegionGraph({
      ...props,
      grid: makeGPURasterMetadata({width: 25, height: 1}),
      labels: createTransientView(graph, 'many-labels', 'uint32', 25),
      values: createTransientView(graph, 'many-values', 'float32', 25)
    });
    const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/edge endpoints.*200.*limit 192/);
    expect(createBuffer).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('GPU-13 rejects adjacency read-only and weights writable in one physical buffer at construction', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createCase(graph, 4, 2);
    const backing = createTransientView(graph, 'aliased', 'uint32', 32);
    const adjacency = graph.createDataView(backing.buffer, {format: 'uint32', length: 16});
    const weights = graph.createDataView(backing.buffer, {
      format: 'float32',
      length: 16,
      byteOffset: 64
    });
    expect(
      () => new GPURasterRegionGraph({...props, output: {...props.output, adjacency, weights}})
    ).toThrow(/same physical buffer/);
  } finally {
    device.destroy();
  }
});

it('GPU-13 auto preserves small dense workloads when boundary scratch cannot fit the device', () => {
  const device = createNullWebGPUDevice();
  try {
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 192});
    const graph = new GPUCommandGraph(device);
    const props = createCase(graph, 4, 3);
    const contributor = new GPURasterRegionGraph({
      ...props,
      searchStructure: 'auto',
      grid: makeGPURasterMetadata({width: 25, height: 1}),
      labels: createTransientView(graph, 'fallback-labels', 'uint32', 25),
      values: createTransientView(graph, 'fallback-values', 'float32', 25)
    });
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes).toHaveLength(13);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    expect(nodes.some(node => node.id.includes('boruvka'))).toBe(false);
  } finally {
    device.destroy();
  }
});

it('GPU-13 forwards searchStructure through the operation wrapper', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createCase(graph, 4, 3);
    const operation = new GPURasterRegionGraphOperation({
      ...props,
      searchStructure: 'dense',
      labels: 'labels',
      values: 'values',
      output: {
        adjacency: 'adjacency',
        weights: 'weights',
        mapping: 'mapping',
        status: {iterationLimitReached: 'pending'}
      }
    });
    const views = {
      ...props.output,
      labels: props.labels,
      values: props.values,
      iterationLimitReached: props.output.status.iterationLimitReached
    };
    const contributor = operation.createContributor(
      port => views[port as 'labels'] as never
    ) as GPURasterRegionGraph;
    expect(contributor.props.searchStructure).toBe('dense');
    expect(contributor.searchStructure).toBe('dense');
  } finally {
    device.destroy();
  }
});

it('GPU-13 validates and formats finite float32 thresholds on both routes', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const searchStructure of ['dense', 'boruvka'] as const) {
      for (const mergeThreshold of [0, 1e-50, 3.4028234663852886e38]) {
        const graph = new GPUCommandGraph(device);
        const contributor = new GPURasterRegionGraph({
          ...createCase(graph, 4, 3),
          searchStructure,
          mergeThreshold
        });
        const kernels = getGPUCommandNodeKernelSources(
          contributor.getCommandNodes(graph).filter(node => node.type === 'compute')
        );
        for (const kernel of kernels) {
          expect(() => new WgslReflect(kernel.source)).not.toThrow();
          expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
        }
      }
      const graph = new GPUCommandGraph(device);
      const props = {...createCase(graph, 4, 3), searchStructure};
      for (const mergeThreshold of [NaN, Infinity, Number.MAX_VALUE, -1])
        expect(() => new GPURasterRegionGraph({...props, mergeThreshold})).toThrow(
          /mergeThreshold/
        );
    }
  } finally {
    device.destroy();
  }
});

it('GPU-13 accepts disjoint packed fields and Boruvka mapping/status sharing an alignment window', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createCase(graph, 4, 2);
    const backing = createTransientView(graph, 'packed-region', 'uint32', 512);
    const output = {
      adjacency: props.output.adjacency,
      weights: props.output.weights,
      mapping: graph.createDataView(backing.buffer, {format: 'uint32', length: 4}),
      status: {
        iterationLimitReached: graph.createDataView(backing.buffer, {
          format: 'uint32',
          byteOffset: 16,
          length: 1
        })
      }
    };
    const contributor = new GPURasterRegionGraph({...props, output});
    expect(contributor.getCommandNodes(graph)).toHaveLength(contributor.estimatedCommandNodeCount);
    // Dense finish binds mapping and status together, so its aligned ranges must be separate.
    expect(() => new GPURasterRegionGraph({...props, output, searchStructure: 'dense'})).toThrow(
      /aligned binding.*overlap/
    );
  } finally {
    device.destroy();
  }
});

it('GPU-13 rejects missing STORAGE usage on each caller binding before declaring resources', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const field of [
      'labels',
      'values',
      'adjacency',
      'weights',
      'mapping',
      'iterationLimitReached'
    ]) {
      const graph = new GPUCommandGraph(device);
      const props = createCase(graph, 4, 2);
      const view =
        field === 'labels' || field === 'values'
          ? props[field]
          : field === 'iterationLimitReached'
            ? props.output.status[field]
            : props.output[field as 'mapping'];
      const contributor = new GPURasterRegionGraph(props);
      Object.defineProperty(view.buffer, 'usage', {value: 0});
      const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
      const createView = vi.spyOn(graph, 'createDataView');
      expect(() => contributor.getCommandNodes(graph), field).toThrow(/Buffer.STORAGE/);
      expect(createBuffer).not.toHaveBeenCalled();
      expect(createView).not.toHaveBeenCalled();
    }
  } finally {
    device.destroy();
  }
});

it('GPU-13 preflights boundary scratch against maxBufferSize before any declaration', () => {
  const device = createNullWebGPUDevice();
  try {
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 1024});
    Object.defineProperty(device.limits, 'maxBufferSize', {value: 400});
    const graph = new GPUCommandGraph(device);
    const props = createCase(graph, 8, 3);
    const contributor = new GPURasterRegionGraph({
      ...props,
      grid: makeGPURasterMetadata({width: 80, height: 1}),
      labels: createTransientView(graph, 'buffer-labels', 'uint32', 80),
      values: createTransientView(graph, 'buffer-values', 'float32', 80)
    });
    const createBuffer = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/edge endpoints.*maxBufferSize/);
    expect(createBuffer).not.toHaveBeenCalled();
    const automatic = new GPURasterRegionGraph({
      ...contributor.props,
      id: 'automatic-buffer',
      searchStructure: 'auto',
      maximumIterations: 7
    });
    expect(automatic.getCommandNodes(graph)).toHaveLength(17);
  } finally {
    device.destroy();
  }
});
