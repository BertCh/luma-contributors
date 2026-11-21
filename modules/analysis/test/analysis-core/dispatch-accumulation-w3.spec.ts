// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createWGSLActiveCountDispatch,
  createWGSLKernelNode
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createGraphChunkGroupKernelNode} from '../../src/analysis-core/kernels/graph-chunk-groups';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../utils/gpu-contributor-test-utils';
import {
  createFixedPointScaleNodes,
  createFixedPointGroupedScaleNodes,
  createFixedPointDecodeNode,
  FIXED_POINT_WGSL
} from '../../src/analysis-core/kernels/fixed-point-accumulator';

it('WGS-11 reads maximum-count dispatch without launching its workload', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const countBuffer = device.createBuffer({
    data: new Uint32Array([0xffffffff]),
    usage: Buffer.STORAGE
  });
  const outputBuffer = createOutputBuffer(device, 3);
  const count = importGraphBuffer(graph, 'count', countBuffer, 'uint32');
  const output = importGraphBuffer(graph, 'output', outputBuffer, 'uint32');
  const dispatch = createWGSLActiveCountDispatch(graph, {
    id: 'maximum',
    operation: 'Maximum',
    count,
    maximumItemCount: 0xffffffff,
    workgroupSize: 256
  });
  graph.add(dispatch.updateNode);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'read-dispatch',
      operation: 'ReadDispatch',
      invocationCount: 3,
      bindings: [
        {
          name: 'dispatchWords',
          view: graph.createDataView(dispatch.condition.buffer, {format: 'uint32', length: 3}),
          type: 'u32',
          access: 'read'
        },
        {name: 'output', view: output, type: 'u32', access: 'read_write'}
      ],
      body: 'output[outputOffset + index] = dispatchWords[dispatchWordsOffset + index];'
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(outputBuffer)).toEqual([65535, 257, 1]);
  } finally {
    compiled.destroy();
    countBuffer.destroy();
    outputBuffer.destroy();
  }
});

it('WGS-15 submits both builders with an unused storage binding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const builder of [createWGSLKernelNode, createGraphChunkGroupKernelNode]) {
    const graph = new GPUCommandGraph(device);
    const outputBuffer = createOutputBuffer(device, 1);
    const weightBuffer = createOutputBuffer(device, 1);
    const output = importGraphBuffer(graph, 'values', outputBuffer, 'uint32');
    const weights = importGraphBuffer(graph, 'weights', weightBuffer, 'uint32');
    graph.add(
      builder(graph, {
        id: 'unused',
        operation: 'Unused',
        invocationCount: 1,
        bindings: [
          {name: 'values', view: output, type: 'u32', access: 'read_write'},
          {name: 'weights', view: weights, type: 'u32', access: 'read'}
        ],
        body: 'values[valuesOffset + index] = 1u;'
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(outputBuffer)).toEqual([1]);
    } finally {
      compiled.destroy();
      outputBuffer.destroy();
      weightBuffer.destroy();
    }
  }
});

it('UDF-06 isolates two occupied cells while preserving the shared-scale route', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const values of [[1], [1, 2 ** 100]]) {
    for (const perCell of [false, true]) {
      const graph = new GPUCommandGraph(device);
      const buffers: Buffer[] = [];
      const inputBuffer = device.createBuffer({
        data: new Float32Array(values),
        usage: Buffer.STORAGE
      });
      buffers.push(inputBuffer);
      const cellBuffer = device.createBuffer({
        data: Uint32Array.from(values, (_, index) => index),
        usage: Buffer.STORAGE
      });
      buffers.push(cellBuffer);
      const scaleBuffer = createOutputBuffer(device, perCell ? values.length : 1);
      buffers.push(scaleBuffer);
      const wordsBuffer = createOutputBuffer(device, values.length * 3);
      buffers.push(wordsBuffer);
      const outputBuffer = createOutputBuffer(device, values.length);
      buffers.push(outputBuffer);
      const input = importGraphBuffer(graph, 'input', inputBuffer, 'float32');
      const cells = importGraphBuffer(graph, 'cells', cellBuffer, 'uint32');
      const scale = importGraphBuffer(graph, 'scale', scaleBuffer, 'uint32');
      const words = importGraphBuffer(graph, 'words', wordsBuffer, 'uint32');
      const output = importGraphBuffer(graph, 'output', outputBuffer, 'float32');
      const scaleProps = {id: 'scales', operation: 'Scales', values: input, scale};
      for (const node of perCell
        ? createFixedPointGroupedScaleNodes(graph, {...scaleProps, cellRows: cells})
        : createFixedPointScaleNodes(graph, scaleProps))
        graph.add(node);
      // One term per occupied cell is an independent exact fixture: sum(1)=1, sum(2^100)=2^100.
      // For the old two-term shared route h=2, unit=256, so 1 quantizes to 0.
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'accumulate',
          operation: 'Accumulate',
          invocationCount: values.length,
          bindings: [
            {name: 'terms', view: input, type: 'f32', access: 'read'},
            {name: 'scales', view: scale, type: 'u32', access: 'read'},
            {name: 'words', view: words, type: 'u32', access: 'read_write'}
          ],
          declarations: FIXED_POINT_WGSL,
          body: `let unit = getFixedPointUnitExponent(scales[scalesOffset + ${perCell ? 'index' : '0u'}], ${values.length === 1 ? 1 : 2}u);
  let value = quantizeFixedPoint(terms[termsOffset + index], unit);
  words[wordsOffset + index * 3u] = value.x;
  words[wordsOffset + index * 3u + 1u] = value.y;
  words[wordsOffset + index * 3u + 2u] = value.z;`
        })
      );
      graph.add(
        createFixedPointDecodeNode(graph, {
          id: 'decode',
          operation: 'Decode',
          words,
          scale,
          maximumTermCount: values.length,
          output,
          scaleMode: perCell ? 'per-cell' : 'shared'
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        const actual = await readFloat32(outputBuffer);
        expect(actual.every(Number.isFinite)).toBe(true);
        expect(actual).toEqual(perCell || values.length === 1 ? values : [0, 2 ** 100]);
      } finally {
        compiled.destroy();
        for (const buffer of buffers) buffer.destroy();
      }
    }
  }
});

it('COMMON-W3 keeps foreign vector topology for gather and scratch on device', async () => {
  const {createGraphChunkGather} = await import(
    '../../src/analysis-core/kernels/graph-chunk-gather'
  );
  const {createTransientUint32Rows} = await import(
    '../../src/analysis-core/kernels/wgsl-kernel-nodes'
  );
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const lengths of [[], [0], [1], [0, 1, 0]]) {
    const graph = new GPUCommandGraph(device);
    const length = lengths.reduce((total, count) => total + count, 0);
    const buffers = lengths.map(() =>
      device.createBuffer({data: new Uint32Array([7]), usage: Buffer.STORAGE | Buffer.COPY_SRC})
    );
    const source = new GraphVectorView({
      id: 'foreign',
      name: 'foreign',
      format: 'uint32',
      length,
      valueLength: length,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data: lengths.map((count, index) => {
        const imported = importGraphBuffer(graph, `source-${index}`, buffers[index], 'uint32');
        return graph.createDataView(imported.buffer, {format: 'uint32', length: count});
      })
    });
    Object.setPrototypeOf(source, Object.prototype);
    const scratch = createTransientUint32Rows(graph, 'scratch', source);
    expect(scratch).toBeInstanceOf(GraphVectorView);
    expect((scratch as GraphVectorView).data.map(chunk => chunk.length)).toEqual(lengths);
    const gathered = createGraphChunkGather(graph, {id: 'gather', source});
    expect(gathered.view).not.toBe(source);
    for (const node of gathered.nodes) graph.add(node);
    if (length === 0) {
      for (const buffer of buffers) buffer.destroy();
      continue;
    }
    const outputBuffer = createOutputBuffer(device, 1);
    const output = importGraphBuffer(graph, 'output', outputBuffer, 'uint32');
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'read',
        operation: 'Read',
        invocationCount: 1,
        bindings: [
          {name: 'input', view: gathered.view, type: 'u32', access: 'read'},
          {name: 'output', view: output, type: 'u32', access: 'read_write'}
        ],
        body: 'output[outputOffset] = input[inputOffset];'
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(outputBuffer)).toEqual([7]);
    } finally {
      compiled.destroy();
      outputBuffer.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});

it('S2 groups permuted cell rows with repeated terms into independent sums', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  // Hand-worked: cell 0 receives -3 and -1 => -4; cell 1 receives 2^101.
  // A mutation that selects scale[index] instead of scale[cellRows[index]] gives
  // cell 0 the huge term's exponent and quantizes both negative terms to zero.
  const inputBuffer = device.createBuffer({
    data: new Float32Array([2 ** 101, -3, -1]),
    usage: Buffer.STORAGE
  });
  const cellBuffer = device.createBuffer({data: new Uint32Array([1, 0, 0]), usage: Buffer.STORAGE});
  const scaleBuffer = createOutputBuffer(device, 2);
  const wordsBuffer = createOutputBuffer(device, 6);
  const outputBuffer = createOutputBuffer(device, 2);
  const values = importGraphBuffer(graph, 'values', inputBuffer, 'float32');
  const cellRows = importGraphBuffer(graph, 'cells', cellBuffer, 'uint32');
  const scale = importGraphBuffer(graph, 'scale', scaleBuffer, 'uint32');
  const words = importGraphBuffer(graph, 'words', wordsBuffer, 'uint32');
  const output = importGraphBuffer(graph, 'output', outputBuffer, 'float32');
  for (const node of createFixedPointGroupedScaleNodes(graph, {
    id: 'scales',
    operation: 'GroupedScales',
    values,
    cellRows,
    scale
  }))
    graph.add(node);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'sum-cells',
      operation: 'SumCells',
      invocationCount: 2,
      bindings: [
        {name: 'terms', view: values, type: 'f32', access: 'read'},
        {name: 'cells', view: cellRows, type: 'u32', access: 'read'},
        {name: 'scales', view: scale, type: 'u32', access: 'read'},
        {name: 'words', view: words, type: 'u32', access: 'read_write'}
      ],
      declarations: FIXED_POINT_WGSL,
      body: `let unit = getFixedPointUnitExponent(scales[scalesOffset + index], 2u);
  var total = vec3<u32>(0u);
  for (var row = 0u; row < 3u; row++) {
    if (cells[cellsOffset + row] == index) {
      total = addFixedPoint(total, quantizeFixedPoint(terms[termsOffset + row], unit));
    }
  }
  words[wordsOffset + index * 3u] = total.x;
  words[wordsOffset + index * 3u + 1u] = total.y;
  words[wordsOffset + index * 3u + 2u] = total.z;`
    })
  );
  graph.add(
    createFixedPointDecodeNode(graph, {
      id: 'decode',
      operation: 'Decode',
      words,
      scale,
      maximumTermCount: 3,
      output,
      scaleMode: 'per-cell'
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(scaleBuffer)).toEqual([128, 228]);
    const actual = await readFloat32(outputBuffer);
    expect(actual.every(Number.isFinite)).toBe(true);
    expect(actual).toEqual([-4, 2 ** 101]);
  } finally {
    compiled.destroy();
    for (const buffer of [inputBuffer, cellBuffer, scaleBuffer, wordsBuffer, outputBuffer])
      buffer.destroy();
  }
});
