// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPURoseStatistic} from '../../../src/gpu-spatial-analysis/geographic-distribution/gpu-rose-statistic';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {expectSpatialAnalysisWGSL} from './spatial-analysis-wgsl-test-utils';

function createContext() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const props = {
    starts: createTransientView(graph, 'starts', 'float32x2', 4),
    ends: createTransientView(graph, 'ends', 'float32x2', 4),
    circular: {
      mode: 'directional' as const,
      angleUnit: 'degrees' as const,
      origin: 'positive-y-clockwise' as const,
      binCount: 4
    },
    output: {
      summary: createTransientView(graph, 'summary', 'float32', 5),
      bins: createTransientView(graph, 'bins', 'float32', 4)
    }
  };
  return {device, graph, props};
}

it('accepts degree bearings and converts the mean after circular moments', () => {
  const {device, graph, props} = createContext();
  try {
    const kernels = getGPUCommandNodeKernelSources(
      new GPURoseStatistic(props).getCommandNodes(graph)
    );
    expect(kernels).toHaveLength(1);
    expect(kernels[0].source).toContain('var direction = delta.yx;');
    expect(kernels[0].source).toContain('meanAngle * (180.0 / PI)');
    expect(kernels[0].source).toContain('guardedAtan2(meanSine, meanCosine)');
    expect(kernels[0].source).toContain('getScaledQuotient(sharedCosine[0],totalWeight,1.0,0)');
    expect(kernels[0].source).toContain('length(vec2<f32>(meanCosine,meanSine))');
    expect(kernels[0].source).toContain('for(var group=workgroupIndex;');
    expect(kernels[0].source).not.toContain('if (index >= INVOCATION_COUNT)');
    for (const kernel of kernels) expectSpatialAnalysisWGSL(kernel.source, kernel.id);
  } finally {
    device.destroy();
  }
});

it.each([
  'directional',
  'axial'
] as const)('%s keeps the curated circular-angle binding attached to histogram binning', mode => {
  const {device, graph, props} = createContext();
  try {
    const kernels = getGPUCommandNodeKernelSources(
      new GPURoseStatistic({...props, circular: {...props.circular, mode}}).getCommandNodes(graph)
    );
    const source = kernels[0].source;
    expect(source).toContain(`let circularAngle = ${mode === 'axial' ? '2.0 * angle' : 'angle'};`);
    expect(source).toContain('var binPosition = (circularAngle / (2.0 * PI)) * f32(BIN_COUNT);');
    // The binding represents phi, but moments still use the stable vector identities.
    expect(source).not.toMatch(/\b(?:sin|cos)\(circularAngle\)/);
    expectSpatialAnalysisWGSL(source, kernels[0].id);
  } finally {
    device.destroy();
  }
});

it('rejects physically aliased output bindings at construction across imported handles', () => {
  const {device, graph, props} = createContext();
  const buffer = device.createBuffer({byteLength: 512, usage: Buffer.STORAGE});
  try {
    const first = graph.importBuffer({id: 'first', byteLength: 512, usage: Buffer.STORAGE}, buffer);
    const second = graph.importBuffer(
      {id: 'second', byteLength: 512, usage: Buffer.STORAGE},
      buffer
    );
    expect(
      () =>
        new GPURoseStatistic({
          ...props,
          output: {
            summary: graph.createDataView(first, {format: 'float32', length: 5}),
            bins: graph.createDataView(second, {format: 'float32', byteOffset: 32, length: 4})
          }
        })
    ).toThrow(/aligned binding byte ranges must not overlap/);
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('checks the actual aligned binding size before allocating sorting scratch', () => {
  const {device, graph, props} = createContext();
  try {
    const storage = graph.createTransientBuffer({
      id: 'long-starts',
      byteLength: 134217736,
      usage: Buffer.STORAGE
    });
    const starts = graph.createDataView(storage, {
      format: 'float32x2',
      byteOffset: 8,
      length: 16777216
    });
    const ends = createTransientView(graph, 'long-ends', 'float32x2', starts.length);
    const groupIds = createTransientView(graph, 'groups', 'uint32', starts.length);
    const summary = createTransientView(graph, 'group-summary', 'float32', 10);
    const bins = createTransientView(graph, 'group-bins', 'float32', 8);
    const contributor = new GPURoseStatistic({
      ...props,
      starts,
      ends,
      groupIds,
      groupCount: 2,
      output: {summary, bins}
    });
    const state = graph as unknown as {buffers: Map<string, unknown>; nodes: unknown[]};
    const resourceCount = state.buffers.size;
    expect(() => contributor.getCommandNodes(graph)).toThrow(/needs 134217736 bytes/);
    expect(state.buffers.size).toBe(resourceCount);
    expect(state.nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});

it('preserves group counts above 2^24 with a bounded cooperative group loop', () => {
  const {device, graph, props} = createContext();
  try {
    Object.assign(device.limits, {
      maxStorageBufferBindingSize: 1073741824,
      maxBufferSize: 1073741824
    });
    const groupCount = 0x1000001;
    const contributor = new GPURoseStatistic({
      ...props,
      groupCount,
      groupIds: createTransientView(graph, 'groups', 'uint32', 4),
      circular: {...props.circular, binCount: 2},
      output: {
        summary: createTransientView(graph, 'large-summary', 'float32', 5 * groupCount),
        bins: createTransientView(graph, 'large-bins', 'float32', 2 * groupCount)
      }
    });
    const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
    const reduce = sources.find(kernel => kernel.id === 'rose-statistic-reduce')!;
    expect(reduce.source).toContain('group+=INVOCATION_COUNT / LANES');
    expect(reduce.constants?.['INVOCATION_COUNT']).toBe(65535 * 256);
    for (const kernel of sources) expectSpatialAnalysisWGSL(kernel.source, kernel.id);
  } finally {
    device.destroy();
  }
});

it('rejects physical input/output aliases while allowing aligned disjoint regions in one buffer', () => {
  const {device, graph, props} = createContext();
  const buffer = device.createBuffer({byteLength: 1024, usage: Buffer.STORAGE});
  try {
    const first = graph.importBuffer(
      {id: 'read-handle', byteLength: 1024, usage: Buffer.STORAGE},
      buffer
    );
    const second = graph.importBuffer(
      {id: 'write-handle', byteLength: 1024, usage: Buffer.STORAGE},
      buffer
    );
    const starts = graph.createDataView(first, {format: 'float32x2', length: 4});
    const bins = graph.createDataView(second, {format: 'float32', length: 4});
    expect(() => new GPURoseStatistic({...props, starts, output: {...props.output, bins}})).toThrow(
      /logical byte ranges must not overlap/
    );
    const output = {
      // Valid execution uses one imported handle; the duplicate above tests early rejection only.
      summary: graph.createDataView(first, {format: 'float32', byteOffset: 256, length: 5}),
      bins: graph.createDataView(first, {format: 'float32', byteOffset: 512, length: 4})
    };
    expect(() =>
      new GPURoseStatistic({...props, starts, output}).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    buffer.destroy();
    device.destroy();
  }
});

it('segmented Rose permits logically disjoint group IDs and outputs in one alignment window', () => {
  const {device, graph, props} = createContext();
  try {
    const buffer = createTransientView(graph, 'packed-group-ids-and-bins', 'uint32', 8).buffer;
    const groupIds = graph.createDataView(buffer, {format: 'uint32', length: 4});
    const bins = graph.createDataView(buffer, {format: 'float32', byteOffset: 16, length: 4});
    expect(() =>
      new GPURoseStatistic({
        ...props,
        groupCount: 2,
        groupIds,
        circular: {...props.circular, binCount: 2},
        output: {summary: createTransientView(graph, 'two-summaries', 'float32', 10), bins}
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});
