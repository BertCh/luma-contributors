// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUCellPeriodKeys} from '../../../src/dataframe-analysis/value-cube/gpu-cell-period-keys';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {expectValidShaderIdentifiers} from '../calendar-buckets/chunk-shader-validation.test-utils';

function createFixture(lengths: number[]) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const column = <Format extends 'uint32' | 'uint32x2'>(name: string, format: Format) =>
    createVectorView(
      name,
      format,
      lengths.map((length, index) => {
        const backing = createTransientView(graph, `${name}-${index}`, format, length + 128);
        return graph.createDataView(backing.buffer, {format, byteOffset: (index % 64) * 4, length});
      })
    );
  const props = {
    id: 'keys',
    cells: column('cells', 'uint32x2'),
    periods: column('periods', 'uint32'),
    mask: column('mask', 'uint32'),
    periodCount: 4,
    fill: 'ones' as const,
    layout: 'legacy-one-bit' as const,
    output: {
      keys: column('keys', 'uint32x2'),
      status: {invalidCount: createTransientView(graph, 'invalid', 'uint32', 1)}
    }
  };
  return {device, graph, props, column};
}

it('validates chunk formats, boundaries and aliases of every writable field and chunk', () => {
  const {device, graph, props} = createFixture([1, 3, 2]);
  try {
    expect(
      () =>
        new GPUCellPeriodKeys({
          ...props,
          periods: createVectorView('reversed', 'uint32', [...props.periods.data].reverse())
        })
    ).toThrow(/periods chunk 0.*topology/);
    expect(
      () =>
        new GPUCellPeriodKeys({
          ...props,
          output: {
            ...props.output,
            keys: createVectorView('repeat', 'uint32x2', [
              props.output.keys.data[1],
              props.output.keys.data[1]
            ])
          }
        })
    ).toThrow(/must not overlap/);
    const invalidCount = graph.createDataView(props.output.keys.data[1].buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: 1
    });
    expect(
      () => new GPUCellPeriodKeys({...props, output: {...props.output, status: {invalidCount}}})
    ).toThrow(/must not overlap/);
    const strided = graph.createDataView(props.cells.data[1].buffer, {
      format: 'uint32x2',
      length: 3,
      byteStride: 16
    });
    expect(
      () =>
        new GPUCellPeriodKeys({
          ...props,
          cells: createVectorView('strided', 'uint32x2', [
            props.cells.data[0],
            strided,
            props.cells.data[2]
          ])
        })
    ).toThrow(/cells must be packed/);
  } finally {
    device.destroy();
  }
});

it('prechecks every binding including later output chunks and status before declaring resources', () => {
  for (const failure of ['input', 'output', 'status']) {
    const {device, graph, props} = createFixture([1, 3, 2]);
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {
      value: failure === 'input' ? 24 : failure === 'status' ? 128 : 256
    });
    if (failure === 'output') {
      const backing = createTransientView(graph, 'late-output', 'uint32x2', 64);
      const view = graph.createDataView(backing.buffer, {
        format: 'uint32x2',
        byteOffset: 252,
        length: 3
      });
      props.output.keys = createVectorView('late', 'uint32x2', [
        props.output.keys.data[0],
        view,
        props.output.keys.data[2]
      ]);
    }
    if (failure === 'status') {
      const backing = createTransientView(graph, 'oversized-status', 'uint32', 128);
      // One uint32 row is a valid status scalar. Its binding starts at byte 0 after
      // 256-byte alignment, so the 252-byte prefix plus 4-byte row exceeds the 128-byte limit.
      props.output.status.invalidCount = graph.createDataView(backing.buffer, {
        format: 'uint32',
        byteOffset: 252,
        length: 1
      });
    }
    const contributor = new GPUCellPeriodKeys(props);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      failure === 'status'
        ? /invalidCount chunk 0 needs 256 bytes in one binding \(limit 128\)/
        : /needs .*bytes/
    );
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    expect(() => contributor.getCommandNodes(new GPUCommandGraph(device))).toThrow(
      /views must belong/
    );
    device.destroy();
  }
});

it('accepts 30 million rows, skips empty chunks and keeps shader/pipeline counts constant', () => {
  const keys = new Set<string>();
  const sources = new Set<string>();
  for (const lengths of [
    [513],
    [513, 1, 257],
    Array.from({length: 30}, (_, index) => [513, 1, 257][index % 3]),
    [0, 10_000_000, 9_000_000, 11_000_000]
  ]) {
    const {device, graph, props} = createFixture(lengths);
    const contributor = new GPUCellPeriodKeys(props);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes).toHaveLength(contributor.estimatedCommandNodeCount);
    expect(getParsedKernelSources(nodes).size).toBe(
      lengths.filter(length => length > 0).length + 1
    );
    expect(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size).toBe(2);
    if (lengths[0] === 513)
      for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
        keys.add(key);
    for (const {source} of getGPUCommandNodeKernelSources(nodes)) sources.add(source);
    device.destroy();
  }
  expect(keys.size).toBe(2);
  expect(sources.size).toBe(2);
  expectValidShaderIdentifiers(sources);
});
