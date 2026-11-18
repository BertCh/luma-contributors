// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {Buffer} from '@luma.gl/core';
import {planBoundsFilterCompaction} from '../../../src/gpu-spatial-analysis/bounds-filter/bounds-filter-compaction';
import {getReservedWGSLWords} from '../spatial-join/wgsl-reserved-words';
import {getChunkDispatches} from '../spatial-join/chunk-dispatch-test-utils';
import {expect, expectTypeOf, it, vi} from 'vitest';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUBoundsFilter,
  type GPUBoundsFilterMode,
  type GPUBoundsFilterOutput,
  type GPUBoundsFilterProps
} from '../../../src/gpu-spatial-analysis/bounds-filter/gpu-bounds-filter';
import {
  createNullWebGPUDevice,
  createVectorView,
  getParsedKernelSources,
  getGPUCommandNodePipelineKeys,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function createFixture(chunkLengths: number[], chunked = true) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device, {id: 'bounds-chunks'});
  const chunks = chunkLengths.map((length, index) =>
    createTransientView(graph, `bounds-${index}`, 'float32x4', length)
  );
  const props: GPUBoundsFilterProps = {
    id: 'bounds-test',
    bounds: chunked ? createVectorView('bounds', 'float32x4', chunks) : chunks[0],
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    output: {
      ids: createTransientView(graph, 'ids', 'uint32', 5),
      status: {
        count: createTransientView(graph, 'count', 'uint32', 1),
        overflow: createTransientView(graph, 'overflow', 'uint32', 1),
        requiredCount: createTransientView(graph, 'required', 'uint32', 1)
      }
    }
  };
  return {device, graph, props};
}

for (const mode of ['intersects', 'within', 'contains'] as GPUBoundsFilterMode[]) {
  it(`GPUBoundsFilter ${mode} declares nine million chunked bounds above the former packed binding ceiling`, () => {
    // 9M float32x4 bounds need 144MB as a packed input, above the core 128MiB binding limit.
    // Graph declaration and WGSL parsing allocate no GPU buffers.
    const {device, graph, props} = createFixture([3_000_000, 2_000_000, 4_000_000]);
    try {
      const contributor = new GPUBoundsFilter({...props, mode});
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      expect(new Set(nodes.map(node => node.id)).size).toBe(nodes.length);
      const sources = getParsedKernelSources(nodes);
      expect([...sources.keys()].filter(id => /^bounds-test-mask-\d+$/.test(id))).toHaveLength(3);
      for (const {source} of sources.configurations) {
        expect(getStorageBindingCount(source)).toBeLessThanOrEqual(8);
        expect(getReservedWGSLWords(source)).toEqual([]);
      }
      expect(getGPUCommandNodePipelineKeys(nodes, {idPattern: /^bounds-test-mask-\d+$/}).size).toBe(
        1
      );
    } finally {
      device.destroy();
    }
  });
}

it('GPUBoundsFilter checks each input binding before declaring any scratch', () => {
  const {device, graph, props} = createFixture([9_000_000], false);
  try {
    const contributor = new GPUBoundsFilter(props);
    const createScratch = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/bounds-test bounds chunk.*limit/);
    expect(createScratch).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('GPUBoundsFilter compacts 35 million rows without a row-sized scratch binding', () => {
  const {device, graph, props} = createFixture([
    7_000_000, 7_000_000, 7_000_000, 7_000_000, 7_000_000
  ]);
  try {
    const contributor = new GPUBoundsFilter(props);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
    const plan = planBoundsFilterCompaction(
      device,
      'bounds-test',
      [7_000_000, 7_000_000, 7_000_000, 7_000_000, 7_000_000]
    );
    expect(plan.blockCount).toBe(136720);
    expect(plan.chunks.map(chunk => chunk.firstRow)).toEqual([
      0, 7_000_000, 14_000_000, 21_000_000, 28_000_000
    ]);
    expect(plan.chunks.every(chunk => chunk.wordCount === 218750)).toBe(true);
  } finally {
    device.destroy();
  }
});

it('GPUBoundsFilter requires source-aligned output masks to preserve chunk boundaries', () => {
  const {device, graph, props} = createFixture([2, 3, 5]);
  try {
    for (const lengths of [
      [3, 2, 5],
      [2, 3, 4]
    ]) {
      const outputMask = createVectorView(
        `mask-${lengths.join('-')}`,
        'uint32',
        lengths.map((length, index) =>
          createTransientView(graph, `mask-${lengths.join('-')}-${index}`, 'uint32', length)
        )
      );
      expect(() => new GPUBoundsFilter({...props, output: {...props.output, outputMask}})).toThrow(
        /bounds-test output.outputMask must match the source topology/
      );
    }
    const packedMask = createTransientView(graph, 'packed-mask', 'uint32', 10);
    expect(
      () => new GPUBoundsFilter({...props, output: {...props.output, outputMask: packedMask}})
    ).toThrow(/bounds-test output.outputMask must match the source topology/);
  } finally {
    device.destroy();
  }
});

it('GPUBoundsFilter skips empty chunks and stays within its node estimate with many uneven chunks', () => {
  for (const lengths of [[0, 2, 0, 3, 5, 0], Array.from({length: 64}, (_, index) => index % 7)]) {
    const {device, graph, props} = createFixture(lengths);
    try {
      const contributor = new GPUBoundsFilter(props);
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      const sources = getParsedKernelSources(nodes);
      expect([...sources.keys()].filter(id => /^bounds-test-mask-\d+$/.test(id))).toHaveLength(
        lengths.filter(length => length > 0).length
      );
    } finally {
      device.destroy();
    }
  }
});

it('GPUBoundsFilter validates the command budget at construction', () => {
  const {device, graph, props} = createFixture([1]);
  try {
    const repeatedBound = createTransientView(graph, 'repeated-bound', 'float32x4', 1);
    const bounds = createVectorView('too-many-chunks', 'float32x4', [
      ...Array.from({length: 1024}, () => repeatedBound),
      createTransientView(graph, 'last-bound', 'float32x4', 1)
    ]);
    expect(() => new GPUBoundsFilter({...props, bounds})).toThrow(
      /bounds-test estimated command node count/
    );
  } finally {
    device.destroy();
  }
});

it('GPUBoundsFilter rejects overlapping writable mask chunks and other outputs at construction', () => {
  const {device, graph, props} = createFixture([1, 1, 1]);
  try {
    const sameMaskRow = createTransientView(graph, 'same-mask-row', 'uint32', 1);
    const outputMask = createVectorView('aliased-mask', 'uint32', [
      sameMaskRow,
      sameMaskRow,
      sameMaskRow
    ]);
    expect(() => new GPUBoundsFilter({...props, output: {...props.output, outputMask}})).toThrow(
      /bounds-test.*must not overlap/
    );
    const disjointMask = createVectorView('mask-status-alias', 'uint32', [
      props.output.status.count,
      createTransientView(graph, 'mask-1', 'uint32', 1),
      createTransientView(graph, 'mask-2', 'uint32', 1)
    ]);
    expect(
      () => new GPUBoundsFilter({...props, output: {...props.output, outputMask: disjointMask}})
    ).toThrow(/bounds-test.*must not overlap/);
    expect(
      () =>
        new GPUBoundsFilter({
          ...props,
          output: {
            ...props.output,
            status: {...props.output.status, overflow: props.output.status.count}
          }
        })
    ).toThrow(/bounds-test.*must not overlap/);
  } finally {
    device.destroy();
  }
});

it('GPUBoundsFilter reuses mask pipelines with nonzero-offset caller masks and uneven chunks', () => {
  const {device, graph, props} = createFixture([1, 255, 256]);
  try {
    const bounds = createVectorView(
      'offset-bounds',
      'float32x4',
      [1, 255, 256].map((length, index) => {
        const backing = createTransientView(
          graph,
          `offset-bounds-${index}`,
          'float32x4',
          length + 64
        );
        return graph.createDataView(backing.buffer, {
          format: 'float32x4',
          length,
          byteOffset: 4 + index * 12
        });
      })
    );
    const outputMask = createVectorView(
      'offset-masks',
      'uint32',
      [1, 255, 256].map((length, index) => {
        const backing = createTransientView(graph, `offset-mask-${index}`, 'uint32', length + 64);
        return graph.createDataView(backing.buffer, {
          format: 'uint32',
          length,
          byteOffset: 12 + index * 4
        });
      })
    );
    const nodes = new GPUBoundsFilter({
      ...props,
      bounds,
      output: {...props.output, outputMask}
    }).getCommandNodes(graph);
    expect(getGPUCommandNodePipelineKeys(nodes, {idPattern: /^bounds-test-mask-\d+$/}).size).toBe(
      1
    );
    for (const source of getParsedKernelSources(nodes).values())
      expect(getReservedWGSLWords(source)).toEqual([]);
  } finally {
    device.destroy();
  }
});

it('GPUBoundsFilter preserves packed mask types and infers vector masks', () => {
  const {device, graph, props} = createFixture([1]);
  try {
    expectTypeOf<GPUBoundsFilterOutput['outputMask']>().toEqualTypeOf<
      GraphDataView<'uint32'> | undefined
    >();
    expectTypeOf<GPUBoundsFilterProps['output']['outputMask']>().toEqualTypeOf<
      GraphDataView<'uint32'> | undefined
    >();
    const mask = createTransientView(graph, 'type-packed-mask', 'uint32', 1);
    const packed = new GPUBoundsFilter({
      ...props,
      bounds: createTransientView(graph, 'type-packed-bounds', 'float32x4', 1),
      output: {...props.output, outputMask: mask}
    });
    expectTypeOf(packed.props.output.outputMask).toEqualTypeOf<
      GraphDataView<'uint32'> | undefined
    >();
    const vector = createVectorView('type-vector-mask', 'uint32', [mask]);
    const chunked = new GPUBoundsFilter({...props, output: {...props.output, outputMask: vector}});
    expectTypeOf(chunked.props.output.outputMask).toEqualTypeOf<typeof vector | undefined>();
  } finally {
    device.destroy();
  }
});

it('GPUBoundsFilter checks aligned co-bound status ranges and accepts consecutive mask chunks', () => {
  const {device, graph, props} = createFixture([1, 1, 1]);
  try {
    const storage = createTransientView(graph, 'shared-writes', 'uint32', 3);
    const chunks = [0, 4, 8].map(byteOffset =>
      graph.createDataView(storage.buffer, {format: 'uint32', length: 1, byteOffset})
    );
    const outputMask = createVectorView('consecutive-masks', 'uint32', chunks);
    expect(
      () => new GPUBoundsFilter({...props, output: {...props.output, outputMask}})
    ).not.toThrow();
    expect(
      () =>
        new GPUBoundsFilter({
          ...props,
          output: {
            ...props.output,
            status: {
              ...props.output.status,
              count: chunks[0],
              overflow: chunks[1]
            }
          }
        })
    ).toThrow(/bounds-test.*aligned binding byte ranges.*must not overlap/);
  } finally {
    device.destroy();
  }
});

it('GPUBoundsFilter keeps shader and pipeline counts constant as chunk count grows', () => {
  const keyCounts: number[] = [];
  const shaderCounts: number[] = [];
  for (const lengths of [[512], [1, 255, 256], [1, 1, 1, 1, 1, 251, 256]]) {
    const {device, graph, props} = createFixture(lengths);
    try {
      const nodes = new GPUBoundsFilter(props).getCommandNodes(graph);
      const masks = nodes.filter(node => /^bounds-test-mask-\d+$/.test(node.id));
      expect(getChunkDispatches(device, masks)).toEqual(
        lengths.map(length => [Math.ceil(length / (32 * 256)), 1, 1])
      );
      // One global published scan changes size with metadata, while the domain kernels keep
      // one source and pipeline each across every chunk. No per-chunk primitive scan is built.
      keyCounts.push(getGPUCommandNodePipelineKeys(nodes).size);
      shaderCounts.push(new Set(getParsedKernelSources(nodes).values()).size);
      for (const expression of [/bounds-test-mask-/, /compact-count-/, /compact-scatter-/]) {
        expect(getGPUCommandNodePipelineKeys(nodes, {idPattern: expression}).size).toBe(1);
      }
    } finally {
      device.destroy();
    }
  }
  for (let index = 1; index < keyCounts.length; index++) {
    expect(keyCounts[index]).toEqual(keyCounts[0]);
    expect(shaderCounts[index]).toEqual(shaderCounts[0]);
  }
});

it('GPUBoundsFilter preflights aligned input and parameter ranges before declaring resources', () => {
  for (const field of ['bounds', 'parameters'] as const) {
    const {device, graph, props} = createFixture([1], false);
    try {
      Object.defineProperty(device, 'limits', {
        value: {...device.limits, maxStorageBufferBindingSize: 256}
      });
      const backing = createTransientView(graph, 'offset-input', 'float32', 128);
      const view = graph.createDataView(backing.buffer, {
        format: field === 'bounds' ? 'float32x4' : 'float32',
        length: field === 'bounds' ? 1 : 4,
        byteOffset: 244
      });
      const contributor =
        field === 'bounds'
          ? new GPUBoundsFilter({...props, bounds: view as GraphDataView<'float32x4'>})
          : new GPUBoundsFilter({...props, parameters: view as GraphDataView<'float32'>});
      const createScratch = vi.spyOn(graph, 'createTransientBuffer');
      expect(() => contributor.getCommandNodes(graph)).toThrow(/bounds-test.*260.*limit.*256/);
      expect(createScratch).not.toHaveBeenCalled();
    } finally {
      device.destroy();
    }
  }
});

it('GPUBoundsFilter preflights the smaller block metadata before declaration', () => {
  const {device, graph, props} = createFixture(Array.from({length: 65}, () => 4));
  try {
    Object.defineProperty(device, 'limits', {value: {...device.limits, maxBufferSize: 256}});
    const contributor = new GPUBoundsFilter(props);
    const createScratch = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /bounds-test compaction block counts\/offsets byteLength.*maxBufferSize/
    );
    expect(createScratch).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});

it('bounds accepts aligned packed writable fields and preflights STORAGE usage before allocation', () => {
  const {device, graph, props} = createFixture([3], false);
  try {
    const storage = createTransientView(graph, 'packed-output', 'uint32', 256);
    const field = (byteOffset: number, length = 1) =>
      graph.createDataView(storage.buffer, {format: 'uint32', byteOffset, length});
    const output = {
      ids: field(0, 3),
      outputMask: field(256, 3),
      status: {count: field(512), requiredCount: field(768), overflow: props.output.status.overflow}
    };
    expect(() => new GPUBoundsFilter({...props, output}).getCommandNodes(graph)).not.toThrow();
    // IDs and masks are written by different passes; their aligned binding prefixes may overlap.
    expect(() =>
      new GPUBoundsFilter({
        ...props,
        id: 'separate-passes',
        output: {...output, outputMask: field(12, 3)}
      }).getCommandNodes(graph)
    ).not.toThrow();
    expect(
      () =>
        new GPUBoundsFilter({
          ...props,
          output: {...output, outputMask: field(8, 3)}
        })
    ).toThrow(/logical output byte ranges must not overlap/);
    const invalid = new GPUBoundsFilter({
      ...props,
      output: {
        ...output,
        ids: graph.createDataView(
          graph.createTransientBuffer({id: 'no-storage', byteLength: 12, usage: Buffer.COPY_DST}),
          {format: 'uint32', length: 3}
        )
      }
    });
    const allocation = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => invalid.getCommandNodes(graph)).toThrow(/STORAGE/);
    expect(allocation).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});
