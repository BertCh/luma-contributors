// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {prepareNullableArrowTimeFixture} from './nullable-arrow-time-fixture';
import {makeGPUTimeVectorFromPreparedArrowColumn} from '../../../src/analysis-core/graph-support/arrow-analytics-time-upload';

import {createTransientView, GPUCommandGraph, type GraphVectorView} from '@luma.gl/graph/gpu-core';
import {describe, expect, it, vi} from 'vitest';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/index';
import {
  GPUTimeWordsRelative,
  type GPUTimeWordsRelativeProps
} from '../../../src/dataframe-analysis/time-words-relative/index';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {expectValidShaderIdentifiers} from '../calendar-buckets/chunk-shader-validation.test-utils';

const CONTEXT: GPUTemporalContext = {
  unit: 'millisecond',
  epoch: 'unix',
  encoding: 'int64-words',
  timeZone: 'UTC',
  axis: {kind: 'regular', step: 1000}
};
let serial = 0;

function createProps(
  graph: GPUCommandGraph,
  overrides: Partial<GPUTimeWordsRelativeProps> = {}
): GPUTimeWordsRelativeProps {
  return {
    timestamps: createTransientView(graph, `times-${serial++}`, 'uint32x2', 5),
    temporalContext: CONTEXT,
    origin: 1_700_000_000_000n,
    output: createTransientView(graph, `relative-${serial++}`, 'float32', 5),
    ...overrides
  };
}

function expectThrows(
  overrides: (graph: GPUCommandGraph) => Partial<GPUTimeWordsRelativeProps>,
  message: RegExp
): void {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  expect(() => new GPUTimeWordsRelative(createProps(graph, overrides(graph)))).toThrow(message);
  device.destroy();
}

describe('GPUTimeWordsRelative', () => {
  it('F6: masked-row NaN uses a runtime bitcast operand for WGSL constant evaluation', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    try {
      const contributor = new GPUTimeWordsRelative(
        createProps(graph, {
          mask: createTransientView(graph, 'mask', 'uint32', 5)
        })
      );
      const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      expect(sources).toHaveLength(1);
      expect(sources[0].source).toContain('bitcast<f32>((index & 0u) | 0x7fc00000u)');
    } finally {
      device.destroy();
    }
  });
  it('exposes the input context as float32-relative counted from the origin', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const node = new GPUTimeWordsRelative(createProps(graph));
    expect(node.outputTemporalContext).toEqual({
      ...CONTEXT,
      encoding: 'float32-relative',
      epoch: 1_700_000_000_000n
    });
    // The input is not modified.
    expect(CONTEXT.encoding).toBe('int64-words');
    expect(CONTEXT.epoch).toBe('unix');
    expect(node.getCommandNodes(graph).map(command => command.id)).toEqual([
      'time-words-relative-convert-chunk-0'
    ]);
    device.destroy();
  });

  it('takes the origin as an instant, whatever epoch the words count from', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const node = new GPUTimeWordsRelative(
      createProps(graph, {temporalContext: {...CONTEXT, epoch: 1_000n}, origin: 5_000n})
    );
    expect(node.outputTemporalContext.epoch).toBe(5_000n);
    device.destroy();
  });

  it('resolves the input as Int64 words only', () => {
    expectThrows(
      () => ({temporalContext: {...CONTEXT, encoding: 'float32-relative', epoch: 5n}}),
      /int64-words/
    );
    expectThrows(() => ({temporalContext: {...CONTEXT, encoding: 'double-single'}}), /int64-words/);
    expectThrows(() => ({temporalContext: {...CONTEXT, unit: 'minute' as never}}), /temporal unit/);
  });

  it('validates the origin', () => {
    expectThrows(() => ({origin: 2n ** 63n}), /origin must be a signed Int64/);
    expectThrows(() => ({origin: 12 as never}), /origin must be a signed Int64/);
    expectThrows(
      () => ({temporalContext: {...CONTEXT, epoch: -(2n ** 62n)}, origin: 2n ** 62n + 1n}),
      /more than 2\^63 ticks from the input epoch/
    );
  });

  it('validates the views', () => {
    expectThrows(
      graph => ({timestamps: createTransientView(graph, 'empty', 'uint32x2', 0)}),
      /at least one row/
    );
    expectThrows(
      graph => ({output: createTransientView(graph, 'short', 'float32', 4)}),
      /output must have 5 rows/
    );
    expectThrows(
      graph => ({output: createTransientView(graph, 'wrong', 'uint32', 5) as never}),
      /output/
    );
  });

  it('refuses timestamps larger than one storage binding at declaration', () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const limit = device.limits.maxStorageBufferBindingSize / 8;
    const add = (rows: number) =>
      graph.add(
        new GPUTimeWordsRelative({
          ...createProps(graph),
          id: `time-words-relative-${rows}`,
          timestamps: createTransientView(graph, `big-times-${rows}`, 'uint32x2', rows),
          output: createTransientView(graph, `big-relative-${rows}`, 'float32', rows)
        })
      );
    expect(() => add(limit)).not.toThrow();
    expect(() => add(limit + 1)).toThrow(/timestamps chunk 0 needs 134217736 bytes in one binding/);
    device.destroy();
  });
});

function createChunkedProps(
  graph: GPUCommandGraph,
  lengths: number[],
  outputOffset = 0
): GPUTimeWordsRelativeProps<GraphVectorView> {
  const column = <Format extends 'uint32x2' | 'float32'>(
    name: string,
    format: Format,
    extraOffset = 0
  ) =>
    createVectorView(
      name,
      format,
      lengths.map((length, index) => {
        const backing = createTransientView(graph, `${name}-${index}`, format, length + 128);
        return graph.createDataView(backing.buffer, {
          format,
          byteOffset: (index % 64) * 4 + extraOffset,
          length
        });
      })
    );
  return {
    id: 'chunked',
    timestamps: column('timestamps', 'uint32x2'),
    output: column('output', 'float32', outputOffset),
    temporalContext: CONTEXT,
    origin: 0n
  };
}

it('validates row chunk topology, packed layouts and writable overlap', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createChunkedProps(graph, [1, 3, 2]);
    const timestamps = props.timestamps as ReturnType<typeof createVectorView<'uint32x2'>>;
    const output = props.output as ReturnType<typeof createVectorView<'float32'>>;
    expect(
      () =>
        new GPUTimeWordsRelative({
          ...props,
          output: createVectorView('reversed', 'float32', [...output.data].reverse())
        })
    ).toThrow(/^chunked output chunk 0.*topology/);
    expect(
      () =>
        new GPUTimeWordsRelative({
          ...props,
          output: createVectorView('overlap', 'float32', [output.data[1], output.data[1]])
        })
    ).toThrow(/must not overlap/);
    const strided = graph.createDataView(timestamps.data[1].buffer, {
      format: 'uint32x2',
      length: 3,
      byteStride: 16
    });
    expect(
      () =>
        new GPUTimeWordsRelative({
          ...props,
          timestamps: createVectorView('stride', 'uint32x2', [
            timestamps.data[0],
            strided,
            timestamps.data[2]
          ])
        })
    ).toThrow(/timestamps must be packed/);
    const empty = new GPUTimeWordsRelative(
      createChunkedProps(new GPUCommandGraph(device), [0, 2, 1])
    );
    expect(empty.rowCount).toBe(3);
    expect(empty.estimatedCommandNodeCount).toBe(2);
  } finally {
    device.destroy();
  }
});

it('prechecks every aligned chunk range and graph ownership before allocations or nodes', () => {
  for (const outputOffset of [0, 248]) {
    const device = createNullWebGPUDevice();
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {
      value: outputOffset ? 128 : 64
    });
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUTimeWordsRelative(
      createChunkedProps(graph, [1, 8, 1], outputOffset)
    );
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      outputOffset ? /output chunk 0 needs/ : /timestamps chunk 1 needs 68 bytes/
    );
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    expect(() => contributor.getCommandNodes(new GPUCommandGraph(device))).toThrow(
      /^chunked views must belong/
    );
    device.destroy();
  }
});

it('accepts 30 million chunked rows and keeps shader and pipeline counts constant', () => {
  const keys = new Set<string>();
  const sources = new Set<string>();
  for (const lengths of [
    [513],
    [513, 1, 257],
    Array.from({length: 30}, (_, index) => [513, 1, 257][index % 3]),
    [10_000_000, 9_000_000, 11_000_000]
  ]) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const contributor = new GPUTimeWordsRelative(createChunkedProps(graph, lengths));
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes).toHaveLength(contributor.estimatedCommandNodeCount);
    expect(getParsedKernelSources(nodes).size).toBe(lengths.length);
    expect(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size).toBe(1);
    // Compare inputs with a common maximum chunk length; total rows and chunk offsets still vary.
    if (lengths[0] === 513) {
      for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
        keys.add(key);
    }
    for (const {source} of getGPUCommandNodeKernelSources(nodes)) sources.add(source);
    device.destroy();
  }
  expect(keys.size).toBe(1);
  expect(sources.size).toBe(1);
  expectValidShaderIdentifiers(sources);
});

it('validates the optional missing-time mask and aligns it with timestamp chunks (UNI-08)', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const props = createChunkedProps(graph, [1, 2]);
    const mask = createVectorView('mask', 'uint32', [
      createTransientView(graph, 'mask-0', 'uint32', 2),
      createTransientView(graph, 'mask-1', 'uint32', 1)
    ]);
    expect(() => new GPUTimeWordsRelative({...props, mask})).toThrow(/mask chunk 0.*topology/);
    const alignedMask = createVectorView('aligned-mask', 'uint32', [
      createTransientView(graph, 'aligned-mask-0', 'uint32', 1),
      createTransientView(graph, 'aligned-mask-1', 'uint32', 2)
    ]);
    const nodes = new GPUTimeWordsRelative({...props, mask: alignedMask}).getCommandNodes(graph);
    expect(getParsedKernelSources(nodes).size).toBe(2);
    for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
      expect(source).toContain('timeValidity');
      expect(source).toContain('bitcast<f32>((index & 0u) | 0x7fc00000u)');
    }
  } finally {
    device.destroy();
  }
});

it('preserves actual Arrow null validity and batch boundaries through upload (S2)', () => {
  const device = createNullWebGPUDevice();
  const {table, column} = prepareNullableArrowTimeFixture();
  expect(table.batches.map(batch => batch.numRows)).toEqual([2, 1]);
  expect(Array.from(table.getChild('time')!)).toEqual([0n, null, 1n]);
  expect(column.validity.map(mask => Array.from(mask))).toEqual([[1, 0], [1]]);
  const upload = makeGPUTimeVectorFromPreparedArrowColumn(device, column);
  try {
    const graph = new GPUCommandGraph(device);
    const timestamps = graph.importGPUVector('arrow-times', upload.timestamps);
    const mask = graph.importGPUVector('arrow-validity', upload.validity);
    expect(timestamps.data.map(chunk => chunk.length)).toEqual([2, 1]);
    expect(mask.data.map(chunk => chunk.length)).toEqual([2, 1]);
    const output = createVectorView('arrow-relative', 'float32', [
      createTransientView(graph, 'arrow-relative-0', 'float32', 2),
      createTransientView(graph, 'arrow-relative-1', 'float32', 1)
    ]);
    expect(
      new GPUTimeWordsRelative({
        timestamps,
        mask,
        temporalContext: upload.temporalContext,
        origin: 0n,
        output
      }).getCommandNodes(graph)
    ).toHaveLength(2);
  } finally {
    upload.destroy();
    device.destroy();
  }
});
