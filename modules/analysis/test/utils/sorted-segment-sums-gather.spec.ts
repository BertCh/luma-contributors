// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUGather, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {getSortedSegmentSumNodes} from '../../src/analysis-core/kernels/sorted-segment-sums';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from './gpu-contributor-test-utils';
import {requireTestDevice} from './require-test-device';

it('the sorted-segment float gather preserves NaN payloads and negative zero at a source offset', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const sourceBuffer = createInputBuffer(
    device,
    new Uint32Array([0xdeadbeef, 0x7fc12345, 0x80000000, 0x3f800000, 0xcafebabe])
  );
  const indexBuffer = createInputBuffer(device, new Uint32Array([1, 0, 2]));
  const outputBuffer = createOutputBuffer(device, 3);
  try {
    const sourceStorage = importGraphBuffer(graph, 'source', sourceBuffer, 'float32');
    const source = graph.createDataView(sourceStorage.buffer, {
      format: 'float32',
      length: 3,
      byteOffset: 4
    });
    // Sums perform float arithmetic, which cannot expose a copied NaN payload or -0 unchanged;
    // exercise the same published gather used by getSortedSegmentSumNodes directly.
    graph.add(
      new GPUGather({
        id: 'gather',
        source,
        indices: importGraphBuffer(graph, 'indices', indexBuffer, 'uint32', 3),
        output: importGraphBuffer(graph, 'output', outputBuffer, 'float32', 3)
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(outputBuffer)).toEqual([0x80000000, 0x7fc12345, 0x3f800000]);
    } finally {
      compiled.destroy();
    }
  } finally {
    sourceBuffer.destroy();
    indexBuffer.destroy();
    outputBuffer.destroy();
  }
});

it.each([
  {
    name: 'unsorted rows with excluded keys and a source offset',
    keys: [1, 0, 2, 1, 4, 0xffffffff],
    contributions: [2, 4, 8, 16, 32, 64],
    expected: [4, 18, 8, 0]
  },
  {
    name: 'zero rows',
    keys: [],
    contributions: [],
    expected: [0, 0, 0, 0]
  }
])('getSortedSegmentSumNodes produces correct sums for $name', async testCase => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  // Zero-length logical views still have backing storage for the guarded kernels to bind.
  const keyBuffer = createInputBuffer(
    device,
    new Uint32Array(testCase.keys.length ? testCase.keys : [0xffffffff])
  );
  const contributionBuffer = createInputBuffer(
    device,
    new Float32Array([99, ...testCase.contributions, 88])
  );
  const outputBuffer = createOutputBuffer(device, testCase.expected.length);
  outputBuffer.write(new Float32Array(testCase.expected.length).fill(123));
  try {
    const contributions = importGraphBuffer(graph, 'contributions', contributionBuffer, 'float32');
    const nodes = getSortedSegmentSumNodes(graph, {
      id: 'sums',
      operation: 'SortedSegmentSumGatherTest',
      segmentCount: testCase.expected.length,
      segmentKeys: importGraphBuffer(graph, 'keys', keyBuffer, 'uint32', testCase.keys.length),
      sumContributions: graph.createDataView(contributions.buffer, {
        format: 'float32',
        length: testCase.contributions.length,
        byteOffset: 4
      }),
      sums: importGraphBuffer(graph, 'sums', outputBuffer, 'float32', testCase.expected.length)
    });
    for (const node of nodes) graph.add(node);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readFloat32(outputBuffer)).toEqual(testCase.expected);
      if (testCase.keys.length === 0) {
        expect(await readUint32(outputBuffer)).toEqual([0, 0, 0, 0]);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    keyBuffer.destroy();
    contributionBuffer.destroy();
    outputBuffer.destroy();
  }
});
