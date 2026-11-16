// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, ComputePipeline, type ComputePass} from '@luma.gl/core';
import {Kernel} from '@luma.gl/graph';
import {
  createTransientView,
  createTransientVectorView,
  GraphDataView,
  GPUCommandGraph,
  getViewBindingRange,
  getViewElementOffset
} from '@luma.gl/graph/gpu-core';
import {afterAll, afterEach, expect, it, vi} from 'vitest';
import {
  createGraphChunkDispatchParameters,
  GRAPH_CHUNK_MAXIMUM_ROW_COUNT,
  GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT,
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings
} from '../../src/analysis-core/kernels/graph-chunk-dispatch';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../utils/gpu-contributor-test-utils';
import {createExampleChunkNodes} from './graph-chunk-dispatch.test-utils';

const device = createNullWebGPUDevice();
afterEach(() => vi.restoreAllMocks());
afterAll(() => device.destroy());

function createFields(graph: GPUCommandGraph, lengths: number[]) {
  return Object.fromEntries(
    ['input', 'output'].map(name => [
      name,
      createVectorView(
        name,
        'uint32',
        lengths.map((length, chunkIndex) =>
          createTransientView(graph, `${name}-${chunkIndex}`, 'uint32', length)
        )
      )
    ])
  );
}

it('plans uneven row-aligned fields and empty chunks without declaring resources', () => {
  const graph = new GPUCommandGraph(device);
  const fields = createFields(graph, [0, 3, 257, 1, 0]);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const plan = planGraphChunkDispatch('plan', {...fields, optionalMask: undefined});
  expect(plan.totalLength).toBe(261);
  expect(plan.nonemptyChunkCount).toBe(3);
  expect(plan.chunks.map(({chunkIndex, rowBase, length}) => [chunkIndex, rowBase, length])).toEqual(
    [
      [0, 0, 0],
      [1, 0, 3],
      [2, 3, 257],
      [3, 260, 1],
      [4, 261, 0]
    ]
  );
  expect(plan.chunks[2].views['input']).toBe(fields['input'].data[2]);
  expect(plan.chunks[2].views).not.toHaveProperty('optionalMask');
  expect(allocate).not.toHaveBeenCalled();
});

it('accepts packed and singleton vector topology, but rejects unequal totals, boundaries and strides', () => {
  const graph = new GPUCommandGraph(device);
  const input = createTransientView(graph, 'input', 'uint32', 4);
  expect(
    planGraphChunkDispatch('single', {input, vector: createVectorView('one', 'uint32', [input])})
      .totalLength
  ).toBe(4);
  expect(() => planGraphChunkDispatch('none', {optional: undefined})).toThrow(/^none views/);
  const split = createVectorView('split', 'uint32', [input, input]);
  expect(() => planGraphChunkDispatch('total', {input, split})).toThrow(/^total split.*topology/);
  const fields = createFields(graph, [1, 3]);
  const reversed = createVectorView('reversed', 'uint32', [...fields['input'].data].reverse());
  expect(() => planGraphChunkDispatch('boundary', {input: fields['input'], reversed})).toThrow(
    /^boundary reversed chunk 0.*topology/
  );
  const strided = graph.createDataView(input.buffer, {format: 'uint32', length: 2, byteStride: 8});
  expect(() => planGraphChunkDispatch('stride', {strided})).toThrow(
    /^stride strided chunk 0 must be packed/
  );
});

it('bounds uint32 totals before packing parameters and accepts 30 million chunked rows', () => {
  const graph = new GPUCommandGraph(device);
  const large = createFields(graph, [10_000_000, 10_000_000, 10_000_000]);
  const plan = planGraphChunkDispatch('large', large);
  expect(plan.totalLength).toBe(30_000_000);
  expect(plan.chunks.map(chunk => chunk.rowBase)).toEqual([0, 10_000_000, 20_000_000]);
  expect(() => validateGraphChunkDispatchBindings(graph, 'large', plan)).not.toThrow();
  const maximum = createTransientView(graph, 'maximum', 'uint32', GRAPH_CHUNK_MAXIMUM_ROW_COUNT);
  const extra = createTransientView(graph, 'extra', 'uint32', 1);
  expect(planGraphChunkDispatch('maximum', {maximum}).totalLength).toBe(0xffffffff);
  const tooLarge = createVectorView('too-large', 'uint32', [maximum, extra]);
  expect(() => planGraphChunkDispatch('overflow', {input: tooLarge})).toThrow(
    /^overflow input total rows.*4294967295/
  );
});

it('allocates a single parameter resource and binds three words at each aligned slot', () => {
  const graph = new GPUCommandGraph(device);
  const plan = planGraphChunkDispatch('parameters', createFields(graph, [0, 3, 257, 0]));
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const {slots, nodes} = createGraphChunkDispatchParameters(graph, {id: 'parameters', plan});
  expect(allocate).toHaveBeenCalledTimes(1);
  expect(nodes.map(node => node.id)).toEqual(['parameters-upload']);
  expect(new Set(slots.map(slot => slot.buffer)).size).toBe(1);
  expect(slots.map(slot => slot.byteOffset)).toEqual([0, 256, 512, 768]);
  expect(slots.every(slot => slot.length === 3 && getViewElementOffset(slot) === 0)).toBe(true);
  expect(slots.every(slot => getViewBindingRange(slot).size === 12)).toBe(true);
  expect(slots[0].buffer.byteLength).toBe(4 * GRAPH_CHUNK_PARAMETER_SLOT_WORD_COUNT * 4);
  // Inspect the upload's immutable initial words, independently of the GPU shader.
  const uploaded: Uint32Array[] = [];
  const originalCreateBuffer = device.createBuffer.bind(device);
  const createBuffer = vi.spyOn(device, 'createBuffer');
  createBuffer.mockImplementation(props => {
    const created = originalCreateBuffer(props);
    vi.spyOn(created, 'write').mockImplementation(values => {
      uploaded.push(new Uint32Array(values as Uint32Array));
    });
    return created;
  });
  const compiled = nodes[0].compile({device});
  expect(uploaded).toHaveLength(1);
  expect(
    [0, 1, 2, 3].map(index => Array.from(uploaded[0].slice(index * 64, index * 64 + 3)))
  ).toEqual([
    [0, 0, 260],
    [0, 3, 260],
    [3, 257, 260],
    [260, 0, 260]
  ]);
  compiled.destroy?.();
});

it('handles zero chunks and all-empty chunks without dispatches', () => {
  const graph = new GPUCommandGraph(device);
  const plan = planGraphChunkDispatch('empty', {input: createVectorView('empty', 'uint32', [])});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(createGraphChunkDispatchParameters(graph, {id: 'empty', plan})).toEqual({
    slots: [],
    nodes: []
  });
  expect(allocate).not.toHaveBeenCalled();
  const allEmpty = planGraphChunkDispatch('all-empty', createFields(graph, [0, 0, 0]));
  expect(allEmpty.nonemptyChunkCount).toBe(0);
  expect(createExampleChunkNodes(graph, allEmpty)).toHaveLength(1);
});

it('rejects writable aliases across fields and chunks, allowing adjacent aligned bindings', () => {
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'backing', 'uint32', 1024);
  const first = graph.createDataView(backing.buffer, {format: 'uint32', length: 100});
  const partial = graph.createDataView(backing.buffer, {
    format: 'uint32x2',
    length: 10,
    byteOffset: 396
  });
  const adjacent = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 100,
    byteOffset: 512
  });
  const empty = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 0,
    byteOffset: 1024
  });
  expect(() => validateDisjointWritableViews('alias', {first, second: partial})).toThrow(
    /^alias first chunk 0 and second chunk 0.*must not overlap/
  );
  expect(() =>
    validateDisjointWritableViews('repeated', {
      output: createVectorView('repeat', 'uint32', [first, first])
    })
  ).toThrow(/output chunk 0 and output chunk 1/);
  expect(() => validateDisjointWritableViews('nested', {outer: backing, first, adjacent})).toThrow(
    /must not overlap/
  );
  expect(() =>
    validateDisjointWritableViews('disjoint', {first, adjacent, empty, optional: undefined})
  ).not.toThrow();
});

it('checks every aligned range, empty backing row and graph membership before graph mutations', () => {
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'backing', 'uint32', 33_554_433);
  const offset = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 33_554_432,
    byteOffset: 4
  });
  const plan = planGraphChunkDispatch('offset', {input: offset});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const addNode = vi.spyOn(graph, 'add');
  expect(() => createGraphChunkDispatchParameters(graph, {id: 'offset', plan})).toThrow(
    /^offset input chunk 0 needs 134217732 bytes.*limit 134217728/
  );
  const otherGraph = new GPUCommandGraph(device);
  expect(() => validateGraphChunkBindingRanges(otherGraph, 'foreign', {input: offset})).toThrow(
    /^foreign views must belong/
  );
  // GraphDataView permits a zero-length view at buffer end, but a bound empty row does not fit.
  const emptyEnd = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 0,
    byteOffset: backing.buffer.byteLength
  });
  createView.mockClear();
  expect(() => validateGraphChunkBindingRanges(graph, 'empty', {input: emptyEnd})).toThrow(
    /empty input chunk 0 aligned binding must fit/
  );
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(addNode).not.toHaveBeenCalled();
});

it('checks later output chunks and preserves their indices before allocating parameter scratch', () => {
  const graph = new GPUCommandGraph(device);
  const input = createTransientView(graph, 'shared-input', 'uint32', 33_554_432);
  const firstOutput = createTransientView(graph, 'first-output', 'uint32', 33_554_432);
  const secondBacking = createTransientView(graph, 'second-output', 'uint32', 33_554_433);
  const secondOutput = graph.createDataView(secondBacking.buffer, {
    format: 'uint32',
    length: 33_554_432,
    byteOffset: 4
  });
  const plan = planGraphChunkDispatch('late', {
    input: createVectorView('input', 'uint32', [input, input]),
    output: createVectorView('output', 'uint32', [firstOutput, secondOutput])
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => createGraphChunkDispatchParameters(graph, {id: 'late', plan})).toThrow(
    /^late output chunk 1 needs 134217732 bytes/
  );
  expect(allocate).not.toHaveBeenCalled();
});

it('checks a parameter slot against the storage binding limit before allocating anything', () => {
  const tinyDevice = createNullWebGPUDevice();
  Object.defineProperty(tinyDevice.limits, 'maxStorageBufferBindingSize', {value: 8});
  const graph = new GPUCommandGraph(tinyDevice);
  const plan = planGraphChunkDispatch('tiny', createFields(graph, [1]));
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() => createGraphChunkDispatchParameters(graph, {id: 'tiny', plan})).toThrow(
    /^tiny chunk parameter slot needs 12 bytes in one binding \(limit 8\)/
  );
  expect(allocate).not.toHaveBeenCalled();
  tinyDevice.destroy();
});

it('checks actual slot binding size and whole-buffer allocation limits separately before allocation', () => {
  const smallDevice = createNullWebGPUDevice();
  Object.defineProperty(smallDevice.limits, 'maxStorageBufferBindingSize', {value: 16});
  Object.defineProperty(smallDevice.limits, 'maxBufferSize', {value: 1024});
  const graph = new GPUCommandGraph(smallDevice);
  const plan = planGraphChunkDispatch('slots', createFields(graph, [1, 1, 1]));
  expect(() => validateGraphChunkDispatchBindings(graph, 'slots', plan)).not.toThrow();
  const largerPlan = planGraphChunkDispatch('too-many', {
    input: createVectorView('five', 'uint32', new Array(5).fill(plan.chunks[0].views['input']))
  });
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  expect(() =>
    createGraphChunkDispatchParameters(graph, {id: 'too-many', plan: largerPlan})
  ).toThrow(/^too-many chunk parameter buffer byteLength.*1024/);
  expect(allocate).not.toHaveBeenCalled();
  smallDevice.destroy();
});

it('uses one shader and pipeline for 1, 3 and 100 unequal chunks and bounded 3D dispatch shapes', () => {
  const keys = new Set<string>();
  for (const chunkCount of [1, 3, 100]) {
    const graph = new GPUCommandGraph(device);
    const fields = createFields(
      graph,
      Array.from({length: chunkCount}, (_, index) => [1, 257, 513][index % 3])
    );
    const nodes = createExampleChunkNodes(graph, planGraphChunkDispatch('example', fields));
    expect(getParsedKernelSources(nodes).size).toBe(1); // Fixed pipeline ID, fixed source.
    const sources = getGPUCommandNodeKernelSources(nodes);
    expect(sources).toHaveLength(chunkCount);
    expect(sources.every(source => Object.keys(source.constants).length === 0)).toBe(true);
    for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
      keys.add(key);
  }
  const smallDevice = createNullWebGPUDevice();
  Object.defineProperty(smallDevice.limits, 'maxComputeWorkgroupsPerDimension', {value: 2});
  const graph = new GPUCommandGraph(smallDevice);
  const nodes = createExampleChunkNodes(
    graph,
    planGraphChunkDispatch('three-dimensional', createFields(graph, [257, 1025]))
  );
  for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
    keys.add(key);
  expect(keys.size).toBe(1);
  smallDevice.destroy();
});

// Full W3C WGSL §16.2 list, checked 2026-11-15, plus backend-sensitive `sample`.
// https://www.w3.org/TR/WGSL/#reserved-words
const RESERVED_WORDS = new Set(
  `
NULL Self abstract active alignas alignof as asm asm_fragment async attribute auto await become
cast catch class co_await co_return co_yield coherent column_major common compile compile_fragment
concept const_cast consteval constexpr constinit crate debugger decltype delete demote demote_to_helper
do dynamic_cast enum explicit export extends extern external fallthrough filter final finally friend
from fxgroup get goto groupshared highp impl implements import inline instanceof interface layout lowp
macro macro_rules match mediump meta mod module move mut mutable namespace new nil noexcept noinline
nointerpolation non_coherent noncoherent noperspective null nullptr of operator package packoffset
partition pass patch pixelfragment precise precision premerge priv protected pub public readonly ref
regardless register reinterpret_cast require resource restrict sample self set shared sizeof smooth snorm
static static_assert static_cast std subroutine super target template this thread_local throw trait try
type typedef typeid typename typeof union unless unorm unsafe unsized use using varying virtual volatile
wgsl where with writeonly yield
`
    .trim()
    .split(/\s+/)
);

it('checks every generated shader identifier against the full reserved word list', () => {
  const graph = new GPUCommandGraph(device);
  const nodes = createExampleChunkNodes(
    graph,
    planGraphChunkDispatch('identifiers', createFields(graph, [1, 257, 513]))
  );
  for (const {source} of getGPUCommandNodeKernelSources(nodes)) {
    expect(
      (source.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(identifier =>
        RESERVED_WORDS.has(identifier)
      )
    ).toEqual([]);
  }
});

it('rejects subword storage binding sizes through both validators before declaring scratch or nodes', () => {
  const graph = new GPUCommandGraph(device);
  const input = createTransientView(graph, 'subword', 'uint8x2', 1);
  const plan = planGraphChunkDispatch('subword', {input});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  const message = /^subword input chunk 0 aligned binding size \(2\).*positive.*multiple.*\(4\)/;
  expect(() => validateGraphChunkBindingRanges(graph, 'subword', {input})).toThrow(message);
  expect(() => validateGraphChunkDispatchBindings(graph, 'subword', plan)).toThrow(message);
  expect(() => createGraphChunkDispatchParameters(graph, {id: 'subword', plan})).toThrow(message);
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
});

it.each([
  'input',
  'output'
])('rejects imported %s without STORAGE through both validators before allocation', name => {
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({byteLength: 4, usage: Buffer.COPY_DST});
  try {
    const view = importGraphBuffer(graph, name, buffer, 'uint32', 1);
    const plan = planGraphChunkDispatch('usage', {[name]: view});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    const message = `usage ${name} chunk 0 buffer usage must include Buffer.STORAGE`;
    expect(() => validateGraphChunkBindingRanges(graph, 'usage', {[name]: view})).toThrow(message);
    expect(() => validateGraphChunkDispatchBindings(graph, 'usage', plan)).toThrow(message);
    expect(() => createGraphChunkDispatchParameters(graph, {id: 'usage', plan})).toThrow(message);
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  } finally {
    buffer.destroy();
  }
});

it('requests exactly one backend pipeline per graph as chunk count increases', () => {
  const counts: number[] = [];
  for (const chunkCount of [1, 3, 100]) {
    const graph = new GPUCommandGraph(device);
    const fields = createFields(
      graph,
      Array.from({length: chunkCount}, (_, index) => [255, 256, 257, 513][index % 4])
    );
    // NullDevice has no compute backend; retain the real published factory and count its calls.
    const pipelines = vi.spyOn(device, 'createComputePipeline').mockImplementation(
      props =>
        Object.assign(Object.create(ComputePipeline.prototype), {
          id: props.id,
          hash: '',
          destroy: () => {}
        }) as ComputePipeline
    );
    for (const node of createExampleChunkNodes(graph, planGraphChunkDispatch('creation', fields)))
      graph.add(node);
    pipelines.mockClear();
    const compiled = graph.compile();
    try {
      counts.push(pipelines.mock.calls.length);
      expect(pipelines).toHaveBeenCalledTimes(1);
    } finally {
      compiled.destroy();
      pipelines.mockRestore();
    }
  }
  expect(counts).toEqual([1, 1, 1]);
});

it('proves the repeated-encoding fixture reuses one physical parameter buffer for two contributors', () => {
  const graph = new GPUCommandGraph(device);
  const inputBuffer = device.createBuffer({byteLength: 192 * 4, usage: Buffer.STORAGE});
  const firstOutputBuffer = device.createBuffer({byteLength: 320 * 4, usage: Buffer.STORAGE});
  const secondOutputBuffer = device.createBuffer({byteLength: 320 * 4, usage: Buffer.STORAGE});
  const input = importGraphBuffer(graph, 'input', inputBuffer, 'uint32', 192);
  const firstOutput = importGraphBuffer(graph, 'first-output', firstOutputBuffer, 'uint32', 320);
  const secondOutput = importGraphBuffer(graph, 'second-output', secondOutputBuffer, 'uint32', 320);
  const splitViews = (id: string, backing: typeof input, prefix: number, lengths: number[]) => {
    let rowBase = prefix;
    return createVectorView(
      id,
      'uint32',
      lengths.map(length => {
        const view = graph.createDataView(backing.buffer, {
          format: 'uint32',
          length,
          byteOffset: rowBase * 4
        });
        rowBase += length;
        return view;
      })
    );
  };
  const firstPlan = planGraphChunkDispatch('first', {
    input: splitViews('first-input', input, 0, [64, 128]),
    output: splitViews('first-output', firstOutput, 64, [64, 128])
  });
  const secondPlan = planGraphChunkDispatch('second', {
    input: splitViews('second-input', firstOutput, 64, [128, 64]),
    output: splitViews('second-output', secondOutput, 64, [128, 64])
  });
  for (const plan of [firstPlan, secondPlan]) {
    validateDisjointWritableViews('outputs', {
      output: createVectorView(
        'output-chunks',
        'uint32',
        plan.chunks.map(chunk => chunk.views['output'])
      )
    });
  }
  for (const node of createExampleChunkNodes(graph, firstPlan, {id: 'first'})) graph.add(node);
  for (const node of createExampleChunkNodes(graph, secondPlan, {id: 'second'})) graph.add(node);
  const pipelines = vi.spyOn(device, 'createComputePipeline').mockImplementation(
    props =>
      Object.assign(Object.create(ComputePipeline.prototype), {
        id: props.id,
        hash: '',
        destroy: () => {}
      }) as ComputePipeline
  );
  const compiled = graph.compile();
  try {
    expect(compiled.stats.logicalTransientBufferCount).toBe(2);
    expect(compiled.stats.physicalTransientBufferCount).toBe(1);
    expect(compiled.stats.nodeOrder).toEqual([
      'first-parameters-upload',
      'first-chunk-0',
      'first-chunk-1',
      'second-parameters-upload',
      'second-chunk-0',
      'second-chunk-1'
    ]);
  } finally {
    compiled.destroy();
    pipelines.mockRestore();
    inputBuffer.destroy();
    firstOutputBuffer.destroy();
    secondOutputBuffer.destroy();
  }
});

it.each([
  0, -4, 2, 6
])('rejects invalid aligned binding byte length %i in a supplied view', byteLength => {
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'backing', 'uint32', 2);
  // Deliberately malformed metadata exercises the standalone validator's positive-size guard.
  const input = new GraphDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 0,
    byteStride: 4,
    rowByteLength: byteLength
  });
  expect(() => validateGraphChunkBindingRanges(graph, 'size', {input})).toThrow(
    /^size input chunk 0 aligned binding size.*must be positive and a multiple/
  );
});

it('accepts subword rows whose complete aligned storage binding is a valid four-byte range', () => {
  const graph = new GPUCommandGraph(device);
  const input = createTransientView(graph, 'two-subword-rows', 'uint8x2', 2);
  const plan = planGraphChunkDispatch('subword', {input});
  expect(() => validateGraphChunkBindingRanges(graph, 'subword', {input})).not.toThrow();
  expect(createGraphChunkDispatchParameters(graph, {id: 'subword', plan}).slots).toHaveLength(1);
});

it('allows separate chunk prefixes but rejects co-bound field prefixes', () => {
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'shared-outputs', 'uint32', 128);
  const first = graph.createDataView(backing.buffer, {format: 'uint32', length: 1});
  const second = graph.createDataView(backing.buffer, {format: 'uint32', length: 1, byteOffset: 4});
  const separated = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 256
  });
  const empty = graph.createDataView(backing.buffer, {format: 'uint32', length: 0, byteOffset: 4});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  expect(() => validateDisjointWritableViews('aligned', {first, second})).toThrow(
    /^aligned first chunk 0 and second chunk 0 aligned binding byte ranges must not overlap/
  );
  expect(() =>
    validateDisjointWritableViews('aligned-chunks', {
      output: createVectorView('chunks', 'uint32', [first, second])
    })
  ).not.toThrow();
  expect(() =>
    validateDisjointWritableViews(
      'strict-chunks',
      {
        output: createVectorView('strict-chunks', 'uint32', [first, second])
      },
      {strictAlignedBindings: true}
    )
  ).toThrow(/output chunk 0 and output chunk 1 aligned binding byte ranges must not overlap/);
  expect(() => validateDisjointWritableViews('empty', {first, empty})).toThrow(
    /first chunk 0 and empty chunk 0 aligned binding byte ranges must not overlap/
  );
  expect(() => validateDisjointWritableViews('separated', {first, separated})).not.toThrow();
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
});

it('groups separate imported handles by physical buffer across fields and chunks', () => {
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({byteLength: 512, usage: Buffer.STORAGE});
  const otherBuffer = device.createBuffer({byteLength: 512, usage: Buffer.STORAGE});
  try {
    const firstImport = importGraphBuffer(graph, 'first-import', buffer, 'uint32');
    const secondImport = importGraphBuffer(graph, 'second-import', buffer, 'uint32');
    const independentImport = importGraphBuffer(graph, 'independent-import', otherBuffer, 'uint32');
    expect(firstImport.buffer).not.toBe(secondImport.buffer);
    const first = graph.createDataView(firstImport.buffer, {format: 'uint32', length: 1});
    const repeated = graph.createDataView(secondImport.buffer, {format: 'uint32', length: 1});
    const prefixOverlap = graph.createDataView(secondImport.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    const separated = graph.createDataView(secondImport.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 256
    });
    const independent = graph.createDataView(independentImport.buffer, {
      format: 'uint32',
      length: 1
    });
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    expect(() => validateDisjointWritableViews('physical', {first, repeated})).toThrow(
      /^physical first chunk 0 and repeated chunk 0 aligned binding byte ranges must not overlap/
    );
    expect(() =>
      validateDisjointWritableViews('physical-prefix', {first, second: prefixOverlap})
    ).toThrow(/first chunk 0 and second chunk 0 aligned binding byte ranges must not overlap/);
    expect(() =>
      validateDisjointWritableViews('physical-chunks', {
        output: createVectorView('chunks', 'uint32', [first, repeated])
      })
    ).toThrow(/output chunk 0 and output chunk 1 logical byte ranges must not overlap/);
    expect(() => validateDisjointWritableViews('separated', {first, separated})).not.toThrow();
    expect(() => validateDisjointWritableViews('independent', {first, independent})).not.toThrow();
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  } finally {
    buffer.destroy();
    otherBuffer.destroy();
  }
});

it('checks aligned ranges across same-index vector fields and packed outputs against later chunks', () => {
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'shared', 'uint32', 256);
  const separate = createTransientView(graph, 'separate', 'uint32', 1);
  const first = graph.createDataView(backing.buffer, {format: 'uint32', length: 1});
  const next = graph.createDataView(backing.buffer, {format: 'uint32', length: 1, byteOffset: 4});
  const far = graph.createDataView(backing.buffer, {format: 'uint32', length: 1, byteOffset: 512});
  expect(() =>
    validateDisjointWritableViews('same-index', {
      first: createVectorView('first', 'uint32', [separate, first]),
      second: createVectorView('second', 'uint32', [far, next])
    })
  ).toThrow(/first chunk 1 and second chunk 1 aligned binding byte ranges must not overlap/);
  expect(() =>
    validateDisjointWritableViews('packed', {
      packed: first,
      chunked: createVectorView('chunked', 'uint32', [far, next])
    })
  ).toThrow(/packed chunk 0 and chunked chunk 1 aligned binding byte ranges must not overlap/);
  // A singleton vector is bound only in dispatch zero, unlike a packed GraphDataView.
  expect(() =>
    validateDisjointWritableViews('singleton', {
      singleton: createVectorView('singleton', 'uint32', [first]),
      chunked: createVectorView('chunked', 'uint32', [far, next])
    })
  ).not.toThrow();
});

it('rejects logical overlap across different-index fields and ignores empty cross-dispatch chunks', () => {
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'shared', 'uint32', 128);
  const firstOther = createTransientView(graph, 'first-other', 'uint32', 2);
  const secondOther = createTransientView(graph, 'second-other', 'uint32', 2);
  const first = graph.createDataView(backing.buffer, {format: 'uint32', length: 2});
  const overlap = graph.createDataView(backing.buffer, {
    format: 'uint32',
    length: 2,
    byteOffset: 4
  });
  const empty = graph.createDataView(backing.buffer, {format: 'uint32', length: 0});
  expect(() =>
    validateDisjointWritableViews('cross-index', {
      first: createVectorView('first', 'uint32', [first, firstOther]),
      second: createVectorView('second', 'uint32', [secondOther, overlap])
    })
  ).toThrow(/first chunk 0 and second chunk 1 logical byte ranges must not overlap/);
  expect(() =>
    validateDisjointWritableViews('empty-chunks', {
      output: createVectorView('empty-chunks', 'uint32', [first, empty, empty])
    })
  ).not.toThrow();
  expect(() =>
    validateDisjointWritableViews(
      'strict-empty',
      {
        output: createVectorView('strict-empty', 'uint32', [first, empty])
      },
      {strictAlignedBindings: true}
    )
  ).toThrow(/aligned binding byte ranges must not overlap/);
});

it('allows adjacent chunks through separately imported handles while retaining physical overlap checks', () => {
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({byteLength: 16, usage: Buffer.STORAGE});
  try {
    const firstImport = importGraphBuffer(graph, 'first', buffer, 'uint32');
    const secondImport = importGraphBuffer(graph, 'second', buffer, 'uint32');
    const first = graph.createDataView(firstImport.buffer, {format: 'uint32', length: 1});
    const next = graph.createDataView(secondImport.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    const repeated = graph.createDataView(secondImport.buffer, {format: 'uint32', length: 1});
    expect(first.buffer).not.toBe(next.buffer);
    expect(() =>
      validateDisjointWritableViews('adjacent', {
        output: createVectorView('adjacent', 'uint32', [first, next])
      })
    ).not.toThrow();
    expect(() =>
      validateDisjointWritableViews('co-bound', {
        first: createVectorView('first', 'uint32', [first]),
        second: createVectorView('second', 'uint32', [next])
      })
    ).toThrow(/aligned binding byte ranges must not overlap/);
    expect(() =>
      validateDisjointWritableViews('repeated', {
        output: createVectorView('repeated', 'uint32', [first, repeated])
      })
    ).toThrow(/logical byte ranges must not overlap/);
  } finally {
    buffer.destroy();
  }
});

it('allows published transient outputs with shared empty chunks within and across fields', () => {
  const graph = new GPUCommandGraph(device);
  const template = createVectorView(
    'template',
    'uint32',
    [0, 3, 0].map((length, index) =>
      createTransientView(graph, `template-${index}`, 'uint32', length)
    )
  );
  const first = createTransientVectorView(graph, 'first', template);
  const second = createTransientVectorView(graph, 'second', template);
  expect(first.data[0]).toBe(first.data[2]);
  expect(second.data[0]).toBe(second.data[2]);
  const sharedSecond = createVectorView('shared-second', 'uint32', [
    first.data[0],
    second.data[1],
    first.data[2]
  ]);
  expect(sharedSecond.data[0]).toBe(first.data[0]);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  expect(() => validateDisjointWritableViews('published', {first, second})).not.toThrow();
  expect(() =>
    validateDisjointWritableViews('shared-empty', {first, second: sharedSecond})
  ).not.toThrow();
  expect(() =>
    validateDisjointWritableViews(
      'strict-empty',
      {first, second: sharedSecond},
      {strictAlignedBindings: true}
    )
  ).toThrow(/aligned binding byte ranges must not overlap/);
  const overlappingNonempty = createVectorView('overlapping-nonempty', 'uint32', [
    first.data[0],
    first.data[1],
    first.data[2]
  ]);
  expect(() =>
    validateDisjointWritableViews('nonempty', {first, second: overlappingNonempty})
  ).toThrow(/first chunk 1 and second chunk 1 aligned binding byte ranges must not overlap/);
  expect(allocate).not.toHaveBeenCalled();
  expect(createView).not.toHaveBeenCalled();
  expect(add).not.toHaveBeenCalled();
});

it('dispatches each nonempty chunk at its own workgroup count instead of padding to the largest', () => {
  const graph = new GPUCommandGraph(device);
  const plan = planGraphChunkDispatch(
    'dispatch-sizes',
    createFields(graph, [0, 1, 255, 256, 257, 1025, 0])
  );
  const nodes = createExampleChunkNodes(graph, plan);
  const pipelines = vi.spyOn(device, 'createComputePipeline').mockImplementation(
    props =>
      Object.assign(Object.create(ComputePipeline.prototype), {
        id: props.id,
        hash: '',
        destroy: () => {}
      }) as ComputePipeline
  );
  const dispatch = vi.spyOn(Kernel.prototype, 'dispatch').mockImplementation(() => {});
  const buffer = device.createBuffer({byteLength: 8192, usage: Buffer.STORAGE});
  const commandEncoder = device.createCommandEncoder();
  try {
    for (const node of nodes) {
      if (node.type !== 'compute') continue;
      const compiled = node.compile({device});
      try {
        compiled.encode({
          commandEncoder,
          computePass: {} as ComputePass,
          parameters: undefined,
          getBuffer: () => buffer,
          getTexture: () => {
            throw new Error('unexpected texture');
          },
          getTextureView: () => {
            throw new Error('unexpected texture view');
          },
          getExternalTexture: () => {
            throw new Error('unexpected external texture');
          }
        });
      } finally {
        compiled.destroy?.();
      }
    }
    expect(dispatch.mock.calls.map(([, options]) => [options.x, options.y, options.z])).toEqual([
      [1, 1, 1],
      [1, 1, 1],
      [1, 1, 1],
      [2, 1, 1],
      [5, 1, 1]
    ]);
  } finally {
    dispatch.mockRestore();
    pipelines.mockRestore();
    commandEncoder.destroy();
    buffer.destroy();
  }
});
