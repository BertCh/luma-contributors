// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterEach, expect, it, vi} from 'vitest';
import {GPUCalendarBuckets} from '../../../src/dataframe-analysis/calendar-buckets/gpu-calendar-buckets';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

import {expectValidShaderIdentifiers} from './chunk-shader-validation.test-utils';

afterEach(() => vi.restoreAllMocks());

function createFixture(lengths: number[], byteOffsets = lengths.map((_, index) => index * 4)) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const column = <Format extends 'uint32' | 'uint32x2' | 'sint32'>(name: string, format: Format) =>
    createVectorView(
      name,
      format,
      lengths.map((length, index) => {
        const backing = createTransientView(graph, `${name}-${index}`, format, length + 256);
        return graph.createDataView(backing.buffer, {
          format,
          length,
          byteOffset: byteOffsets[index]
        });
      })
    );
  const props = {
    id: 'calendar',
    timestamps: column('timestamps', 'uint32x2'),
    mask: column('mask', 'uint32'),
    utcOffsets: column('utcOffsets', 'sint32'),
    parameters: createTransientView(graph, 'parameters', 'sint32', 2),
    output: {
      year: column('year', 'sint32'),
      hour: column('hour', 'uint32'),
      hourWeekdayCounts: createTransientView(graph, 'counts', 'uint32', 168)
    }
  };
  return {device, graph, props};
}

it('accepts 30 million rows and empty chunks with a bounded command count', () => {
  const {device, graph, props} = createFixture([0, 10_000_000, 9_000_000, 11_000_000]);
  try {
    const contributor = new GPUCalendarBuckets(props);
    expect(contributor.rowCount).toBe(30_000_000);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes).toHaveLength(contributor.estimatedCommandNodeCount);
    expect(nodes.filter(node => node.id.endsWith('fill-matrix'))).toHaveLength(1);
    expect(nodes.some(node => node.id.endsWith('chunk-0'))).toBe(false);
  } finally {
    device.destroy();
  }
});

it('validates chunk boundaries, formats and overlapping writable ranges at construction', () => {
  const {device, graph, props} = createFixture([1, 3, 2]);
  try {
    expect(
      () =>
        new GPUCalendarBuckets({
          ...props,
          mask: createVectorView('bad-mask', 'uint32', [...props.mask.data].reverse())
        })
    ).toThrow(/^calendar mask chunk 0.*topology/);
    expect(
      () =>
        new GPUCalendarBuckets({
          ...props,
          output: {
            ...props.output,
            hour: createVectorView('repeat', 'uint32', [
              props.output.hour.data[0],
              props.output.hour.data[1],
              graph.createDataView(props.output.hour.data[1].buffer, {
                format: 'uint32',
                byteOffset: 4,
                length: 2
              })
            ])
          }
        })
    ).toThrow(/overlap/);
    const alias = graph.createDataView(props.output.hour.data[1].buffer, {
      format: 'sint32',
      byteOffset: 4,
      length: 3
    });
    expect(
      () =>
        new GPUCalendarBuckets({
          ...props,
          output: {
            ...props.output,
            year: createVectorView('alias', 'sint32', [
              props.output.year.data[0],
              alias,
              props.output.year.data[2]
            ])
          }
        })
    ).toThrow(/overlap/);
    expect(
      () =>
        new GPUCalendarBuckets({
          ...props,
          output: {
            ...props.output,
            hourWeekdayCounts: graph.createDataView(props.output.hour.data[1].buffer, {
              format: 'uint32',
              length: 168
            })
          }
        })
    ).toThrow(/overlap/);
    const strided = graph.createDataView(props.mask.data[1].buffer, {
      format: 'uint32',
      length: 3,
      byteStride: 8
    });
    expect(
      () =>
        new GPUCalendarBuckets({
          ...props,
          mask: createVectorView('strided', 'uint32', [
            props.mask.data[0],
            strided,
            props.mask.data[2]
          ])
        })
    ).toThrow(/mask must be packed/);
  } finally {
    device.destroy();
  }
});

it('prechecks later chunks, side inputs and aligned prefixes before any graph mutation', () => {
  const {device, graph, props} = createFixture([1, 33_554_432], [0, 4]);
  const contributor = new GPUCalendarBuckets(props);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  expect(() => contributor.getCommandNodes(graph)).toThrow(
    /needs .*bytes|aligned binding must fit/
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
  device.destroy();
});

it('uses a constant shader and pipeline count for unequal chunk lengths and offsets', () => {
  const keys = new Set<string>();
  const sources = new Set<string>();
  for (const chunkCount of [1, 3, 30]) {
    const lengths = Array.from({length: chunkCount}, (_, index) => [513, 1, 257][index % 3]);
    const {device, graph, props} = createFixture(
      lengths,
      lengths.map((_, index) => (index % 64) * 4)
    );
    try {
      const nodes = new GPUCalendarBuckets(props).getCommandNodes(graph);
      expect(getParsedKernelSources(nodes).size).toBe(chunkCount + 1);
      for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
        keys.add(key);
      for (const {source} of getGPUCommandNodeKernelSources(nodes)) sources.add(source);
    } finally {
      device.destroy();
    }
  }
  expect(keys.size).toBe(2); // One matrix clear and one decode pipeline.
  expect(sources.size).toBe(2);
  expectValidShaderIdentifiers(sources);
});

it('prechecks output prefixes, parameters and zone side tables before graph mutation', () => {
  for (const field of ['hour', 'parameters', 'transitions', 'zoneOffsets'] as const) {
    const {device, graph, props} = createFixture([1, 3, 2]);
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 128});
    const unsigned = createTransientView(graph, 'unsigned', 'uint32', 128);
    const signed = createTransientView(graph, 'signed', 'sint32', 128);
    const transitions = graph.createDataView(unsigned.buffer, {
      format: 'uint32x2',
      length: 1,
      byteOffset: field === 'transitions' ? 252 : 0
    });
    const offsets = graph.createDataView(signed.buffer, {
      format: 'sint32',
      length: 2,
      byteOffset: field === 'zoneOffsets' ? 252 : 0
    });
    const hour =
      field === 'hour'
        ? createVectorView('bad-hour', 'uint32', [
            props.output.hour.data[0],
            graph.createDataView(unsigned.buffer, {format: 'uint32', length: 3, byteOffset: 252}),
            props.output.hour.data[2]
          ])
        : props.output.hour;
    const contributor = new GPUCalendarBuckets({
      ...props,
      utcOffsets: undefined,
      parameters:
        field === 'parameters'
          ? graph.createDataView(signed.buffer, {format: 'sint32', length: 2, byteOffset: 252})
          : props.parameters,
      timeZone:
        field === 'transitions' || field === 'zoneOffsets' ? {transitions, offsets} : undefined,
      output: {hour}
    });
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/needs .*bytes/);
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    device.destroy();
  }
});
