// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUCommandGraph,
  createTransientView,
  createTransientVectorView,
  type GraphDataView,
  type GPUCommandNodeProducer,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  planGraphChunkDispatch,
  createGraphChunkDispatchParameters,
  validateGraphChunkBindingRanges,
  validateDisjointWritableViews,
  type GraphChunkDispatchPlan
} from '../../src/analysis-core/kernels/graph-chunk-dispatch';
import {
  importGraphBuffer,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs
} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createOutputBuffer,
  createVectorView,
  readUint32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {createExampleChunkNodes} from './graph-chunk-dispatch.test-utils';

/** Tiny test contributor: output[row] = input[row] * 3 + globalRow + totalRows. */
class ExampleChunkContributor implements GPUCommandNodeProducer {
  readonly plan: GraphChunkDispatchPlan;
  readonly estimatedCommandNodeCount: number;

  constructor(
    input: GraphVectorView<'uint32'>,
    output: GraphVectorView<'uint32'>,
    readonly options: {id?: string; maximumWorkgroupsPerDimension?: number} = {}
  ) {
    this.plan = planGraphChunkDispatch('example', {input, output});
    validateDisjointWritableViews('example', {output});
    validateGraphOutputsDisjointFromInputs('example', [output], [input]);
    this.estimatedCommandNodeCount =
      this.plan.nonemptyChunkCount + (this.plan.chunks.length ? 1 : 0);
    validateCommandNodeBudget('example', this.estimatedCommandNodeCount);
  }

  getCommandNodes<Parameters>(graph: GPUCommandGraph<Parameters>) {
    return createExampleChunkNodes(graph, this.plan, this.options);
  }
}

it('dispatches 1, 3 and 100 unequal chunks with one pipeline and CPU-expected global rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const pipelines = vi.spyOn(device, 'createComputePipeline');
  const pipelineCounts: number[] = [];
  try {
    for (const chunkCount of [1, 3, 100]) {
      const graph = new GPUCommandGraph(device);
      const lengths = Array.from({length: chunkCount}, (_, index) => [1, 257, 513][index % 3]);
      const totalLength = lengths.reduce((total, length) => total + length, 0);
      let rowBase = 0;
      const arrays = lengths.map(length => {
        const data = Uint32Array.from({length}, (_, index) => (rowBase + index) % 101);
        rowBase += length;
        return data;
      });
      // Nonzero 256-byte offsets check that binding slot offsets remain zero on the GPU.
      const inputBuffers = arrays.map(data => {
        const padded = new Uint32Array(data.length + 64);
        padded.set(data, 64);
        return device.createBuffer({data: padded, usage: Buffer.STORAGE | Buffer.COPY_DST});
      });
      const outputBuffers = lengths.map(length => createOutputBuffer(device, length + 64));
      const input = createVectorView(
        'input',
        'uint32',
        inputBuffers.map((buffer, chunkIndex) => {
          const backing = importGraphBuffer(graph, `input-${chunkIndex}`, buffer, 'uint32');
          return graph.createDataView(backing.buffer, {
            format: 'uint32',
            length: lengths[chunkIndex],
            byteOffset: 256
          });
        })
      );
      const output = createVectorView(
        'output',
        'uint32',
        outputBuffers.map((buffer, chunkIndex) => {
          const backing = importGraphBuffer(graph, `output-${chunkIndex}`, buffer, 'uint32');
          return graph.createDataView(backing.buffer, {
            format: 'uint32',
            length: lengths[chunkIndex],
            byteOffset: 256
          });
        })
      );
      const contributor = new ExampleChunkContributor(input, output);
      graph.add(contributor);
      pipelines.mockClear();
      const compiled = graph.compile();
      try {
        pipelineCounts.push(pipelines.mock.calls.length);
        expect(pipelines).toHaveBeenCalledTimes(1);
        for (const increment of [0, 7]) {
          if (increment) {
            for (const [chunkIndex, buffer] of inputBuffers.entries()) {
              buffer.write(
                arrays[chunkIndex].map(value => value + increment),
                256
              );
            }
          }
          submitGraph(device, compiled, undefined);
          let globalRow = 0;
          for (const [chunkIndex, buffer] of outputBuffers.entries()) {
            const actual = (await readUint32(buffer)).slice(64);
            const expected = Array.from(
              arrays[chunkIndex],
              value => (value + increment) * 3 + globalRow++ + totalLength
            );
            expect(actual).toEqual(expected);
          }
        }
      } finally {
        compiled.destroy();
        for (const buffer of [...inputBuffers, ...outputBuffers]) buffer.destroy();
      }
    }
    expect(pipelineCounts).toEqual([1, 1, 1]);
    const signatures = pipelines.mock.calls.map(([props]) =>
      JSON.stringify({
        source: props.shader.props.source,
        constants: props.constants,
        shaderLayout: props.shaderLayout
      })
    );
    expect(new Set(signatures).size).toBe(1);
  } finally {
    pipelines.mockRestore();
  }
});

it('handles empty chunks and two million rows against an independent CPU formula', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lengths = [0, 1_000_003, 0, 999_997, 0];
  const totalLength = 2_000_000;
  const graph = new GPUCommandGraph(device);
  let rowBase = 0;
  const inputBuffers = lengths.map(length => {
    const values = Uint32Array.from(
      {length: Math.max(length, 1)},
      (_, index) => (rowBase + index) % 17
    );
    rowBase += length;
    return device.createBuffer({data: values, usage: Buffer.STORAGE});
  });
  const outputBuffers = lengths.map(length => createOutputBuffer(device, length));
  const input = createVectorView(
    'input',
    'uint32',
    inputBuffers.map((buffer, chunkIndex) =>
      importGraphBuffer(graph, `input-${chunkIndex}`, buffer, 'uint32', lengths[chunkIndex])
    )
  );
  const output = createVectorView(
    'output',
    'uint32',
    outputBuffers.map((buffer, chunkIndex) =>
      importGraphBuffer(graph, `output-${chunkIndex}`, buffer, 'uint32', lengths[chunkIndex])
    )
  );
  const contributor = new ExampleChunkContributor(input, output);
  expect(contributor.estimatedCommandNodeCount).toBe(3);
  graph.add(contributor);
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    let globalRow = 0;
    for (const [chunkIndex, buffer] of outputBuffers.entries()) {
      if (lengths[chunkIndex] === 0) continue;
      const bytes = await buffer.readAsync();
      const actual = new Uint32Array(bytes.buffer, bytes.byteOffset, lengths[chunkIndex]);
      const firstRow = globalRow;
      expect(
        actual.findIndex((value, index) => {
          const row = firstRow + index;
          return value !== (row % 17) * 3 + row + totalLength;
        })
      ).toBe(-1);
      globalRow += lengths[chunkIndex];
    }
    expect(globalRow).toBe(totalLength);
  } finally {
    compiled.destroy();
    for (const buffer of [...inputBuffers, ...outputBuffers]) buffer.destroy();
  }
});

it('rejects one-row uint8x2 storage bindings before declaring parameter scratch', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const input = createTransientView(graph, 'subword', 'uint8x2', 1);
  const plan = planGraphChunkDispatch('subword', {input});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  try {
    expect(() => createGraphChunkDispatchParameters(graph, {id: 'subword', plan})).toThrow(
      /^subword input chunk 0 aligned binding size \(2\).*positive.*multiple.*\(4\)/
    );
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  } finally {
    allocate.mockRestore();
    createView.mockRestore();
    add.mockRestore();
  }
});

it.each([
  'input',
  'output'
])('rejects imported %s lacking STORAGE before declaring scratch', async name => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffer = device.createBuffer({byteLength: 4, usage: Buffer.COPY_DST});
  const view = importGraphBuffer(graph, name, buffer, 'uint32', 1);
  const plan = planGraphChunkDispatch('usage', {[name]: view});
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  try {
    const message = `usage ${name} chunk 0 buffer usage must include Buffer.STORAGE`;
    expect(() => validateGraphChunkBindingRanges(graph, 'usage', {[name]: view})).toThrow(message);
    expect(() => createGraphChunkDispatchParameters(graph, {id: 'usage', plan})).toThrow(message);
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  } finally {
    allocate.mockRestore();
    createView.mockRestore();
    add.mockRestore();
    buffer.destroy();
  }
});

it('flattens forced 2-by-2 dispatch dimensions correctly at workgroup tails', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const lengths = [255, 256, 257, 1025];
  const totalLength = lengths.reduce((total, length) => total + length, 0);
  const inputBuffers = lengths.map(length =>
    device.createBuffer({
      data: Uint32Array.from({length}, (_, index) => index % 13),
      usage: Buffer.STORAGE
    })
  );
  const outputBuffers = lengths.map(length => createOutputBuffer(device, length));
  const input = createVectorView(
    'input',
    'uint32',
    inputBuffers.map((buffer, index) =>
      importGraphBuffer(graph, `input-${index}`, buffer, 'uint32', lengths[index])
    )
  );
  const output = createVectorView(
    'output',
    'uint32',
    outputBuffers.map((buffer, index) =>
      importGraphBuffer(graph, `output-${index}`, buffer, 'uint32', lengths[index])
    )
  );
  graph.add(new ExampleChunkContributor(input, output, {maximumWorkgroupsPerDimension: 2}));
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    let rowBase = 0;
    for (const [chunkIndex, buffer] of outputBuffers.entries()) {
      const expected = Array.from(
        {length: lengths[chunkIndex]},
        (_, index) => (index % 13) * 3 + rowBase + index + totalLength
      );
      expect(await readUint32(buffer)).toEqual(expected);
      rowBase += lengths[chunkIndex];
    }
  } finally {
    compiled.destroy();
    for (const buffer of [...inputBuffers, ...outputBuffers]) buffer.destroy();
  }
});

it('refreshes parameter slots after another contributor reuses scratch, preserving adjacent outputs and canaries', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const totalLength = 192;
  const canary = 0x1234abcd;
  // A 64-word prefix and suffix surround adjacent aligned chunks in each imported output.
  const inputBuffer = device.createBuffer({
    data: Uint32Array.from({length: totalLength}, (_, row) => row % 23),
    usage: Buffer.STORAGE | Buffer.COPY_DST
  });
  const firstOutputBuffer = device.createBuffer({
    data: new Uint32Array(totalLength + 128).fill(canary),
    usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
  });
  const secondOutputBuffer = device.createBuffer({
    data: new Uint32Array(totalLength + 128).fill(canary),
    usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
  });
  const inputBacking = importGraphBuffer(
    graph,
    'input-backing',
    inputBuffer,
    'uint32',
    totalLength
  );
  const firstBacking = importGraphBuffer(graph, 'first-backing', firstOutputBuffer, 'uint32');
  const secondBacking = importGraphBuffer(graph, 'second-backing', secondOutputBuffer, 'uint32');
  const splitViews = (
    id: string,
    backing: GraphDataView<'uint32'>,
    prefixRows: number,
    lengths: number[]
  ) => {
    let rowBase = prefixRows;
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
  const firstInput = splitViews('first-input', inputBacking, 0, [64, 128]);
  const firstOutput = splitViews('first-output', firstBacking, 64, [64, 128]);
  // Reversed chunk lengths overwrite the shared slots with different lengths and row bases.
  const secondInput = splitViews('second-input', firstBacking, 64, [128, 64]);
  const secondOutput = splitViews('second-output', secondBacking, 64, [128, 64]);
  graph.add(new ExampleChunkContributor(firstInput, firstOutput, {id: 'first'}));
  graph.add(new ExampleChunkContributor(secondInput, secondOutput, {id: 'second'}));
  const compiled = graph.compile();
  try {
    expect(compiled.stats.logicalTransientBufferCount).toBe(2);
    expect(compiled.stats.physicalTransientBufferCount).toBe(1);
    for (const increment of [0, 5, 11]) {
      inputBuffer.write(
        Uint32Array.from({length: totalLength}, (_, row) => (row % 23) + increment)
      );
      submitGraph(device, compiled, undefined);
      const firstExpected = Array.from(
        {length: totalLength},
        (_, row) => ((row % 23) + increment) * 3 + row + totalLength
      );
      const secondExpected = firstExpected.map((value, row) => value * 3 + row + totalLength);
      for (const [buffer, expected] of [
        [firstOutputBuffer, firstExpected],
        [secondOutputBuffer, secondExpected]
      ] as const) {
        const actual = await readUint32(buffer);
        expect(actual.slice(0, 64)).toEqual(new Array(64).fill(canary));
        expect(actual.slice(64, 64 + totalLength)).toEqual(expected);
        expect(actual.slice(64 + totalLength)).toEqual(new Array(64).fill(canary));
      }
    }
  } finally {
    compiled.destroy();
    inputBuffer.destroy();
    firstOutputBuffer.destroy();
    secondOutputBuffer.destroy();
  }
});

it.each([
  'aligned-prefix',
  'physical-import'
])('validates %s output chunk bindings at construction before declaring nodes', async mode => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const outputBuffer = device.createBuffer({byteLength: 512, usage: Buffer.STORAGE});
  const firstImport = importGraphBuffer(graph, 'first-output', outputBuffer, 'uint32');
  const secondHandle =
    mode === 'physical-import'
      ? importGraphBuffer(graph, 'second-output', outputBuffer, 'uint32').buffer
      : firstImport.buffer;
  const input = createVectorView(
    'input',
    'uint32',
    [0, 1].map(index => createTransientView(graph, `input-${index}`, 'uint32', 1))
  );
  const output = createVectorView('output', 'uint32', [
    graph.createDataView(firstImport.buffer, {format: 'uint32', length: 1}),
    graph.createDataView(secondHandle, {
      format: 'uint32',
      length: 1,
      byteOffset: mode === 'aligned-prefix' ? 4 : 0
    })
  ]);
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  const createView = vi.spyOn(graph, 'createDataView');
  const add = vi.spyOn(graph, 'add');
  try {
    if (mode === 'physical-import') {
      expect(() => new ExampleChunkContributor(input, output)).toThrow(
        /^example output chunk 0 and output chunk 1 logical byte ranges must not overlap/
      );
    } else {
      const contributor = new ExampleChunkContributor(input, output);
      expect(contributor.plan.chunks.map(chunk => [chunk.rowBase, chunk.length])).toEqual([
        [0, 1],
        [1, 1]
      ]);
      expect(() =>
        validateDisjointWritableViews('strict', {output}, {strictAlignedBindings: true})
      ).toThrow(/aligned binding byte ranges must not overlap/);
    }
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
  } finally {
    allocate.mockRestore();
    createView.mockRestore();
    add.mockRestore();
    outputBuffer.destroy();
  }
});

it('runs published transient topology [0,3,0] with shared empty views in two output fields', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const inputBuffer = device.createBuffer({
    data: new Uint32Array([2, 5, 9]),
    usage: Buffer.STORAGE
  });
  const firstResultBuffer = createOutputBuffer(device, 3);
  const secondResultBuffer = createOutputBuffer(device, 3);
  const inputEmpty = createTransientView(graph, 'input-empty', 'uint32', 0);
  const input = createVectorView('input', 'uint32', [
    inputEmpty,
    importGraphBuffer(graph, 'input-values', inputBuffer, 'uint32', 3),
    inputEmpty
  ]);
  const first = createTransientVectorView(graph, 'first-transient', input);
  const secondTransient = createTransientVectorView(graph, 'second-transient', input);
  const second = createVectorView('second-shared-empty', 'uint32', [
    first.data[0],
    secondTransient.data[1],
    first.data[2]
  ]);
  expect(first.data[0]).toBe(first.data[2]);
  expect(secondTransient.data[0]).toBe(secondTransient.data[2]);
  expect(second.data[0]).toBe(first.data[0]);
  expect(() => validateDisjointWritableViews('two-output-fields', {first, second})).not.toThrow();
  // Consume the transient values into imported results using the same fixed shader a second time.
  const resultEmpty = createTransientView(graph, 'result-empty', 'uint32', 0);
  const firstResult = createVectorView('first-result', 'uint32', [
    resultEmpty,
    importGraphBuffer(graph, 'first-result-values', firstResultBuffer, 'uint32', 3),
    resultEmpty
  ]);
  const secondResult = createVectorView('second-result', 'uint32', [
    resultEmpty,
    importGraphBuffer(graph, 'second-result-values', secondResultBuffer, 'uint32', 3),
    resultEmpty
  ]);
  graph.add(new ExampleChunkContributor(input, first, {id: 'first'}));
  graph.add(new ExampleChunkContributor(input, second, {id: 'second'}));
  graph.add(new ExampleChunkContributor(first, firstResult, {id: 'first-consumer'}));
  graph.add(new ExampleChunkContributor(second, secondResult, {id: 'second-consumer'}));
  const compiled = graph.compile();
  try {
    for (let encodingIndex = 0; encodingIndex < 2; encodingIndex++) {
      submitGraph(device, compiled, undefined);
      // f(f(value, row), row) = 9*value + 4*row + 12, calculated independently.
      expect(await readUint32(firstResultBuffer)).toEqual([30, 61, 101]);
      expect(await readUint32(secondResultBuffer)).toEqual([30, 61, 101]);
    }
  } finally {
    compiled.destroy();
    inputBuffer.destroy();
    firstResultBuffer.destroy();
    secondResultBuffer.destroy();
  }
});
