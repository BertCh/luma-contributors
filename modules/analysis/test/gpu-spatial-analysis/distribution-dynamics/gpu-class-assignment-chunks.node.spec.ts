// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUClassAssignment} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-class-assignment';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {expectValidShaderIdentifiers} from '../../gpu-dataframe/calendar-buckets/chunk-shader-validation.test-utils';

function createFixture(lengths: number[]) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const column = <Format extends 'uint32' | 'float32'>(name: string, format: Format) =>
    createVectorView(
      name,
      format,
      lengths.map((length, index) => {
        const backing = createTransientView(graph, `${name}-${index}`, format, length + 128);
        return graph.createDataView(backing.buffer, {format, byteOffset: (index % 64) * 4, length});
      })
    );
  const props = {
    id: 'classes',
    values: column('values', 'float32'),
    mask: column('mask', 'uint32'),
    output: column('output', 'uint32'),
    breaks: createTransientView(graph, 'breaks', 'float32', 4),
    classCount: createTransientView(graph, 'class-count', 'uint32', 1)
  };
  return {device, graph, props};
}

it('validates chunk topology, packed format and writable aliases at construction', () => {
  const {device, graph, props} = createFixture([1, 3, 2]);
  try {
    expect(
      () =>
        new GPUClassAssignment({
          ...props,
          mask: createVectorView('bad-mask', 'uint32', [...props.mask.data].reverse())
        })
    ).toThrow(/mask chunk 0.*topology/);
    expect(
      () =>
        new GPUClassAssignment({
          ...props,
          output: createVectorView('repeat', 'uint32', [props.output.data[1], props.output.data[1]])
        })
    ).toThrow(/must not overlap/);
    const strided = graph.createDataView(props.values.data[1].buffer, {
      format: 'float32',
      byteStride: 8,
      length: 3
    });
    expect(
      () =>
        new GPUClassAssignment({
          ...props,
          values: createVectorView('strided', 'float32', [
            props.values.data[0],
            strided,
            props.values.data[2]
          ])
        })
    ).toThrow(/values must be packed/);
  } finally {
    device.destroy();
  }
});

it('prechecks every aligned row binding and side input before declaring graph resources', () => {
  for (const field of ['values', 'output', 'breaks', 'classCount'] as const) {
    const {device, graph, props} = createFixture([1, 3, 2]);
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 256});
    const format = field === 'values' || field === 'breaks' ? 'float32' : 'uint32';
    const backing = createTransientView(graph, 'oversized', format, 128);
    const oversized = graph.createDataView(backing.buffer, {
      format,
      byteOffset: 252,
      length: field === 'classCount' ? 2 : 3
    });
    const changed =
      field === 'values' || field === 'output'
        ? {
            ...props,
            [field]: createVectorView('late', format, [
              props[field].data[0],
              oversized,
              props[field].data[2]
            ])
          }
        : {...props, [field]: oversized};
    const contributor = new GPUClassAssignment(changed as typeof props);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/needs .*bytes/);
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    expect(() => contributor.getCommandNodes(new GPUCommandGraph(device))).toThrow(
      /views must belong/
    );
    device.destroy();
  }
});

for (const intervalClosure of ['left', 'right'] as const) {
  it(`keeps ${intervalClosure} shader/pipeline counts constant and handles 45 million rows`, () => {
    const keys = new Set<string>();
    const sources = new Set<string>();
    for (const lengths of [
      [513],
      [513, 1, 257],
      Array.from({length: 30}, (_, index) => [513, 1, 257][index % 3]),
      [0, 15_000_000, 14_000_000, 16_000_000]
    ]) {
      const {device, graph, props} = createFixture(lengths);
      const contributor = new GPUClassAssignment({...props, intervalClosure});
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes).toHaveLength(contributor.estimatedCommandNodeCount);
      expect(getParsedKernelSources(nodes).size).toBe(lengths.filter(length => length > 0).length);
      expect(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size).toBe(1);
      if (lengths[0] === 513)
        for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
          keys.add(key);
      for (const {source} of getGPUCommandNodeKernelSources(nodes)) sources.add(source);
      expect(contributor.classifiedValues.classes).toBe(props.output);
      device.destroy();
    }
    expect(keys.size).toBe(1);
    expect(sources.size).toBe(1);
    expectValidShaderIdentifiers(sources);
  });
}

it('handles a zero-chunk vector without declaring nodes or scratch', () => {
  const {device, graph, props} = createFixture([]);
  try {
    const contributor = new GPUClassAssignment(props);
    expect(contributor.estimatedCommandNodeCount).toBe(0);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(contributor.getCommandNodes(graph)).toEqual([]);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});
