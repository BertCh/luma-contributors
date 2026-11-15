// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, GPUScan} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it, vi} from 'vitest';
import {createPreparationStatusNode} from '../../src/gpu-spatial-analysis/lon-lat-polygons/lon-lat-polygon-preparation-kernels';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createSaturatedCountScanNodes,
  createSaturatedScanTotalNodes,
  SCAN_TOTAL_WGSL,
  SATURATED_SCAN_TOTAL
} from '../../src/analysis-core/kernels/scan-total';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from './gpu-contributor-test-utils';
import {requireTestDevice} from './require-test-device';

/** True sum of the counts, saturated like the GPU total. */
function getExpectedTotal(counts: readonly number[]): number {
  const sum = counts.reduce((total, count) => total + count, 0);
  return sum >= 2 ** 32 ? SATURATED_SCAN_TOTAL : sum;
}

// S-8: GPUScan wraps modulo 2^32, so a count total of 2^32 or more used to read back as a small
// number and pass every capacity check. The saturated total must equal the true sum below 2^32 and
// 0xffffffff from 2^32 on, also when the wrap happens deep inside a multi-block scan.
it('saturates exclusive count-scan totals that reach 2^32', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const many = (length: number, count: number) => Array.from({length}, () => count);
  const cases: readonly (readonly number[])[] = [
    [1, 2, 3],
    [0xfffffffe, 1],
    [0xffffffff, 1],
    [0x80000000, 0x80000000, 7],
    [5, 0xffffffff, 0, 3],
    many(5000, 850_000),
    many(5000, 900_000),
    [...many(4095, 0), 0xffffffff, ...many(4096, 1)]
  ];
  for (const counts of cases) {
    const graph = new GPUCommandGraph(device, {id: 'scan-total'});
    const countsBuffer = createInputBuffer(device, Uint32Array.from(counts));
    const offsetsBuffer = createOutputBuffer(device, counts.length);
    const totalBuffer = createOutputBuffer(device, 2);
    const countsView = importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', counts.length);
    const offsetsView = importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', counts.length);
    const totalView = importGraphBuffer(graph, 'total', totalBuffer, 'uint32', 2);
    for (const node of new GPUScan({
      id: 'scan',
      input: countsView,
      output: offsetsView,
      mode: 'exclusive'
    }).getCommandNodes(graph)) {
      graph.add(node);
    }
    for (const node of createSaturatedScanTotalNodes(graph, {
      id: 'scan-total',
      operation: 'ScanTotalTest',
      counts: countsView,
      offsets: offsetsView,
      rowCount: counts.length,
      total: totalView,
      totalRow: 1
    })) {
      graph.add(node);
    }
    const compiled = graph.compile();
    submitGraph(device, compiled, undefined);
    const [, total] = await readUint32(totalBuffer, 2);
    compiled.destroy();
    for (const buffer of [countsBuffer, offsetsBuffer, totalBuffer]) buffer.destroy();
    expect(total, `counts of length ${counts.length}`).toBe(getExpectedTotal(counts));
  }
});

it('saturates every exclusive offset after carries, preserving padding and total rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases = [
    [],
    [0],
    [0, 2, 0, 3, 0],
    [0xfffffffe, 1, 0, 1, 7],
    [2, 0xffffffff, 0, 2],
    [0xffffffff, 0xffffffff, 0xffffffff, 7, 0],
    [...Array.from({length: 4095}, () => 0), 0xffffffff, 1, 0, 3],
    Array.from({length: 5000}, () => 900_000)
  ];
  for (const counts of cases) {
    const graph = new GPUCommandGraph(device, {id: 'saturated-count-scan'});
    // Both inputs and outputs are nonzero-offset subviews with untouched padding.
    const inputBuffer = createInputBuffer(device, Uint32Array.from([99, ...counts, 88]));
    const offsetsBuffer = createOutputBuffer(device, counts.length + 2);
    offsetsBuffer.write(new Uint32Array(counts.length + 2).fill(77));
    const totalBuffer = createOutputBuffer(device, 3);
    totalBuffer.write(Uint32Array.from([66, 55, 44]));
    const endsBuffer = createOutputBuffer(device, Math.max(1, counts.length));
    const input = importGraphBuffer(graph, 'input', inputBuffer, 'uint32', counts.length + 2);
    const offsets = importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', counts.length + 2);
    const total = importGraphBuffer(graph, 'total', totalBuffer, 'uint32', 3);
    const ends = importGraphBuffer(graph, 'ends', endsBuffer, 'uint32', Math.max(1, counts.length));
    const countView = graph.createDataView(input.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: counts.length
    });
    const offsetView = graph.createDataView(offsets.buffer, {
      format: 'uint32',
      byteOffset: 4,
      length: counts.length
    });
    for (const node of createSaturatedCountScanNodes(graph, {
      id: 'counts',
      operation: 'ScanTotalTest',
      counts: countView,
      offsets: offsetView,
      rowCount: counts.length,
      total,
      totalRow: 1
    })) {
      graph.add(node);
    }
    graph.add(
      createWGSLKernelNode(graph, {
        id: 'ends',
        operation: 'ScanTotalTest',
        variant: 'saturated-ends',
        bindings: [
          {name: 'counts', view: countView, type: 'u32', access: 'read'},
          {name: 'offsets', view: offsetView, type: 'u32', access: 'read'},
          {name: 'ends', view: ends, type: 'u32', access: 'read_write'}
        ],
        invocationCount: counts.length,
        declarations: SCAN_TOTAL_WGSL,
        body: 'ends[endsOffset + index] = getSaturatedScanSum(offsets[offsetsOffset + index], counts[countsOffset + index]);'
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      let sum = 0n;
      const expectedOffsets: number[] = [];
      const expectedEnds: number[] = [];
      const saturate = (value: bigint) => Number(value > 0xffffffffn ? 0xffffffffn : value);
      for (const count of counts) {
        expectedOffsets.push(saturate(sum));
        sum += BigInt(count);
        expectedEnds.push(saturate(sum));
      }
      expect(Array.from(await readUint32(offsetsBuffer, counts.length + 2))).toEqual([
        77,
        ...expectedOffsets,
        77
      ]);
      expect(Array.from(await readUint32(totalBuffer, 3))).toEqual([66, saturate(sum), 44]);
      if (counts.length > 0) {
        expect(Array.from(await readUint32(endsBuffer, counts.length))).toEqual(expectedEnds);
      }
    } finally {
      compiled.destroy();
      for (const buffer of [inputBuffer, offsetsBuffer, totalBuffer, endsBuffer]) buffer.destroy();
    }
  }
});

it('saturates scan arithmetic and resize products with integer boundary oracles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases = [
    [0, 0, 8],
    [0xffffffff, 0, 8],
    [0xfffffffc, 3, 1],
    [0xfffffffc, 4, 8],
    [0x20000000, 0, 8],
    [2, 3, 0],
    [123, 456, 7]
  ];
  const graph = new GPUCommandGraph(device, {id: 'scan-arithmetic'});
  const inputBuffer = createInputBuffer(device, Uint32Array.from(cases.flat()));
  const outputBuffer = createOutputBuffer(device, cases.length * 2);
  const input = importGraphBuffer(graph, 'input', inputBuffer, 'uint32', cases.length * 3);
  const output = importGraphBuffer(graph, 'output', outputBuffer, 'uint32', cases.length * 2);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'arithmetic',
      operation: 'ScanTotalTest',
      variant: 'arithmetic',
      bindings: [
        {name: 'input', view: input, type: 'u32', access: 'read'},
        {name: 'output', view: output, type: 'u32', access: 'read_write'}
      ],
      invocationCount: cases.length,
      declarations: SCAN_TOTAL_WGSL,
      body: `let sum = getSaturatedScanSum(input[inputOffset + index * 3u], input[inputOffset + index * 3u + 1u]);
  output[outputOffset + index * 2u] = sum;
  output[outputOffset + index * 2u + 1u] = getSaturatedScanProduct(sum, input[inputOffset + index * 3u + 2u]);`
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const saturate = (value: bigint) => Number(value > 0xffffffffn ? 0xffffffffn : value);
    const expected = cases.flatMap(([offset, count, multiplier]) => {
      const sum = BigInt(offset) + BigInt(count);
      return [saturate(sum), saturate(sum * BigInt(multiplier))];
    });
    expect(Array.from(await readUint32(outputBuffer, cases.length * 2))).toEqual(expected);
  } finally {
    compiled.destroy();
    inputBuffer.destroy();
    outputBuffer.destroy();
  }
});

it.each([
  765, 768, 771, 1536
])('saturates arithmetic with %i compile-time resize rows', async addedRows => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const offsets = [
    0,
    1,
    0x1fffffff - addedRows,
    0x20000000 - addedRows,
    0xffffffff - addedRows,
    0x100000000 - addedRows,
    0xffffffff
  ];
  const graph = new GPUCommandGraph(device, {id: `scan-resize-${addedRows}`});
  const inputBuffer = createInputBuffer(device, Uint32Array.from(offsets));
  const outputBuffer = createOutputBuffer(device, 2 * offsets.length);
  const input = importGraphBuffer(graph, 'input', inputBuffer, 'uint32', offsets.length);
  const output = importGraphBuffer(graph, 'output', outputBuffer, 'uint32', 2 * offsets.length);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'resize-arithmetic',
      operation: 'ScanTotalTest',
      variant: 'resize-arithmetic',
      bindings: [
        {name: 'input', view: input, type: 'u32', access: 'read'},
        {name: 'output', view: output, type: 'u32', access: 'read_write'}
      ],
      invocationCount: offsets.length,
      declarations: SCAN_TOTAL_WGSL,
      body: `let sum = getSaturatedScanSum(input[inputOffset + index], ${addedRows}u);
  output[outputOffset + 2u * index] = sum;
  output[outputOffset + 2u * index + 1u] = getSaturatedScanProduct(sum, 8u);`
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const saturate = (value: bigint) => Number(value > 0xffffffffn ? 0xffffffffn : value);
    expect(await readUint32(outputBuffer, 2 * offsets.length)).toEqual(
      offsets.flatMap(offset => {
        const sum = BigInt(offset) + BigInt(addedRows);
        return [saturate(sum), saturate(8n * sum)];
      })
    );
  } finally {
    compiled.destroy();
    inputBuffer.destroy();
    outputBuffer.destroy();
  }
});

it('polygon preparation publishes a saturated resize requirement after dense scan overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Each value overflowed the old 8 * (denseTotal + 3 * ringCount) estimate, sometimes to 24.
  for (const denseTotal of [0x20000000, 0xffffffff]) {
    const graph = new GPUCommandGraph(device, {id: 'preparation-resize-total'});
    const buffers = [
      createInputBuffer(device, Uint32Array.from([0, 3])),
      createInputBuffer(device, Uint32Array.from([0, 0])),
      createOutputBuffer(device, 6),
      createInputBuffer(device, new Uint32Array(4)),
      createInputBuffer(device, new Uint32Array(2))
    ];
    buffers[2].write(Uint32Array.from([denseTotal, 0, 0, 0, 0, 0]));
    const [counts, starts, stageFlags, metadata, diagnostics] = buffers.map((buffer, index) =>
      importGraphBuffer(graph, `input-${index}`, buffer, 'uint32', [2, 2, 6, 4, 2][index])
    );
    graph.add(
      createPreparationStatusNode(graph, {
        id: 'status',
        counts,
        starts,
        stageFlags,
        capacity: 4,
        metadata,
        diagnostics
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      const result = await readUint32(buffers[2], 6);
      expect(result[3]).toBe(3);
      expect(result[4]).toBe(0xffffffff);
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});

it.each([
  [255, 0],
  [256, 0],
  [256, 2],
  [257, 0],
  [512, 0]
])('publishes ring flags and scalar preparation status for %i rings with %i padding rows', async (ringCount, paddingRows) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: `preparation-status-${ringCount}`});
  recordWGSLKernelNodes(graph);
  const total = 3 * ringCount;
  const storageRowCount = 2 * ringCount + paddingRows + Number(paddingRows > 0);
  const countsBuffer = createInputBuffer(
    device,
    Uint32Array.from({length: storageRowCount}, (_, row) =>
      row < paddingRows || row >= paddingRows + 2 * ringCount
        ? 91
        : row < paddingRows + ringCount
          ? 3
          : 0
    )
  );
  const startsBuffer = createInputBuffer(
    device,
    Uint32Array.from({length: storageRowCount}, (_, row) =>
      row < paddingRows || row >= paddingRows + 2 * ringCount
        ? 92
        : 3 * Math.min(row - paddingRows, ringCount)
    )
  );
  const metadataBuffer = createInputBuffer(
    device,
    Uint32Array.from({length: 4 * ringCount}, (_, word) => (word % 4 === 0 ? word / 4 : 0))
  );
  const diagnosticsBuffer = createInputBuffer(device, Uint32Array.from([2, 1]));
  const stageFlagsBuffer = createOutputBuffer(device, 6);
  stageFlagsBuffer.write(Uint32Array.from([total, total, total, total, total, total]));
  const flagsBuffer = createOutputBuffer(device, ringCount);
  const invalidCountBuffer = createOutputBuffer(device, 1);
  const approximationBuffer = createOutputBuffer(device, 1);
  const countsStorage = importGraphBuffer(graph, 'counts', countsBuffer, 'uint32', storageRowCount);
  const startsStorage = importGraphBuffer(graph, 'starts', startsBuffer, 'uint32', storageRowCount);
  const counts = graph.createDataView(countsStorage.buffer, {
    format: 'uint32',
    byteOffset: 4 * paddingRows,
    length: 2 * ringCount
  });
  const starts = graph.createDataView(startsStorage.buffer, {
    format: 'uint32',
    byteOffset: 4 * paddingRows,
    length: 2 * ringCount
  });
  const metadata = importGraphBuffer(graph, 'metadata', metadataBuffer, 'uint32', 4 * ringCount);
  const diagnostics = importGraphBuffer(graph, 'diagnostics', diagnosticsBuffer, 'uint32', 2);
  const stageFlags = importGraphBuffer(graph, 'stage-flags', stageFlagsBuffer, 'uint32', 6);
  const ringFlags = importGraphBuffer(graph, 'flags', flagsBuffer, 'uint32', ringCount);
  const invalidCount = importGraphBuffer(graph, 'invalid', invalidCountBuffer, 'uint32', 1);
  const approximation = importGraphBuffer(graph, 'approximation', approximationBuffer, 'uint32', 1);
  graph.add(
    createPreparationStatusNode(graph, {
      id: 'status',
      counts,
      starts,
      metadata,
      ringFlags,
      diagnostics,
      stageFlags,
      capacity: total + 1,
      invalidCount,
      approximation
    })
  );
  // Preserve luma.gl's nested error scopes, but also include their native messages in the
  // thrown assertion. A bind-group failure invalidates the submission and otherwise looks
  // like a zero-filled output, hiding the reason in the browser's console.
  const nativeDevice: GPUDevice = device.handle;
  const validationErrors: string[] = [];
  const pendingScopes: Promise<void>[] = [];
  const popErrorScope = nativeDevice.popErrorScope.bind(nativeDevice);
  const errorScopeSpy = vi.spyOn(nativeDevice, 'popErrorScope').mockImplementation(() => {
    const result = popErrorScope();
    pendingScopes.push(
      result.then(error => {
        if (error) validationErrors.push(error.message);
      })
    );
    return result;
  });
  nativeDevice.pushErrorScope('validation');
  let outerScopePopped = false;
  let compiled: ReturnType<typeof graph.compile> | undefined;
  try {
    compiled = graph.compile();
    // An invalid synchronous pipeline can enter the upstream cache and obscure its original
    // compiler error on later runs. Check the exact generated module and overrides with the
    // async API too: it rejects with the original compilation reason. Construct the graph's
    // synchronous pipeline first so this diagnostic cannot warm its native pipeline cache.
    const statusKernel = getWGSLKernelNodeRecord(graph, 'status')!;
    const shader = nativeDevice.createShaderModule({code: statusKernel.source});
    const compilation = await shader.getCompilationInfo();
    const compilerErrors = compilation.messages.filter(message => message.type === 'error');
    expect(compilerErrors.map(message => message.message).join(' | ')).toBe('');
    try {
      await nativeDevice.createComputePipelineAsync({
        label: `preparation-status-${ringCount}-compilation-check`,
        layout: 'auto',
        compute: {
          module: shader,
          entryPoint: statusKernel.entryPoint,
          constants: statusKernel.constants
        }
      });
    } catch (error) {
      throw new Error(
        `Preparation status compiler rejection for ${ringCount} rings: ${String(error).replaceAll('\n', ' ')}`,
        {cause: error}
      );
    }
    for (let repeat = 0; repeat < 2; repeat++) {
      submitGraph(device, compiled, undefined);
      await nativeDevice.queue.onSubmittedWorkDone();
      if (!outerScopePopped) {
        await nativeDevice.popErrorScope();
        outerScopePopped = true;
      }
      await Promise.all(pendingScopes);
      expect(
        validationErrors,
        `Native WebGPU validation for ${ringCount} rings: ${validationErrors.join(' | ').replaceAll('\n', ' ')}`
      ).toEqual([]);
      expect(await readUint32(flagsBuffer, ringCount), `ring flags for ${ringCount} rings`).toEqual(
        Array.from({length: ringCount}, (_, row) => row)
      );
      expect(await readUint32(stageFlagsBuffer, 6), `totals for ${ringCount} rings`).toEqual([
        total,
        0,
        0,
        total,
        total,
        total
      ]);
      expect(await readUint32(invalidCountBuffer, 1)).toEqual([2]);
      expect(await readUint32(approximationBuffer, 1)).toEqual([1]);
    }
  } finally {
    if (!outerScopePopped) await nativeDevice.popErrorScope();
    await Promise.all(pendingScopes);
    errorScopeSpy.mockRestore();
    compiled?.destroy();
    for (const buffer of [
      countsBuffer,
      startsBuffer,
      metadataBuffer,
      diagnosticsBuffer,
      stageFlagsBuffer,
      flagsBuffer,
      invalidCountBuffer,
      approximationBuffer
    ])
      buffer.destroy();
  }
});
