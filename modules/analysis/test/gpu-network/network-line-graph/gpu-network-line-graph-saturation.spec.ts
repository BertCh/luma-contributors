// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkLineGraph,
  getGPUNetworkLineGraphParameterValues
} from '../../../src/gpu-network/network-line-graph/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  recordWGSLKernelNodes,
  getWGSLKernelNodeRecord,
  createWGSLKernelNode
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

it('keeps CSR offsets monotone when turn counts exceed uint32, including graph reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    values: Uint32Array | Float32Array,
    format: Format
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        `saturation-input-${serial++}`,
        buffer,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      )
    };
  };
  const output = <Format extends 'uint32' | 'float32'>(length: number, format: Format) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `saturation-output-${serial++}`, buffer, format, length)
    };
  };
  const offsets = output(4, 'uint32');
  const count = output(1, 'uint32');
  const overflow = output(1, 'uint32');
  const injected = input(Uint32Array.of(0xfffffffe, 3, 4, 0), 'uint32');
  recordWGSLKernelNodes(graph);
  const contributor = new GPUNetworkLineGraph({
    id: 'line',
    topology: {
      offsets: input(Uint32Array.of(0, 1, 2, 3), 'uint32').view,
      neighbors: input(Uint32Array.of(1, 2, 0), 'uint32').view,
      weights: input(Float32Array.of(1, 1, 1), 'float32').view,
      nodePositions: input(Float32Array.of(0, 0, 1, 0, 0, 1), 'float32x2').view
    },
    parameters: input(getGPUNetworkLineGraphParameterValues({}), 'float32').view,
    output: {
      topology: {
        offsets: offsets.view,
        neighbors: output(4, 'uint32').view,
        weights: output(4, 'float32').view
      },
      arcCount: count.view,
      status: {overflow: overflow.view}
    }
  });
  for (const node of contributor.getCommandNodes(graph)) {
    graph.add(node);
    if (node.id === 'line-count') {
      // Inject the output of a large count stage, avoiding billions of turn tests while running
      // the real wide scan, publication, fill and clamp stages on the GPU.
      const counts = getWGSLKernelNodeRecord(graph, node.id)!.bindings.find(
        binding => binding.name === 'counts'
      )!.view;
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'inject-turn-counts',
          operation: 'LineGraphSaturationProbe',
          invocationCount: 4,
          bindings: [
            {name: 'inputCounts', view: injected.view, type: 'u32', access: 'read'},
            {name: 'counts', view: counts, type: 'u32', access: 'read_write'}
          ],
          body: 'counts[countsOffset + index] = inputCounts[inputCountsOffset + index];'
        })
      );
    }
  }
  const compiled = graph.compile();
  try {
    await submitGraph(device, compiled, undefined);
    expect(Array.from(await readUint32(offsets.buffer, 4))).toEqual([0, 4, 4, 4]);
    expect(Array.from(await readUint32(count.buffer, 1))).toEqual([4]);
    expect(Array.from(await readUint32(overflow.buffer, 1))).toEqual([1]);
    injected.buffer.write(Uint32Array.of(1, 1, 1, 0));
    await submitGraph(device, compiled, undefined);
    expect(Array.from(await readUint32(offsets.buffer, 4))).toEqual([0, 1, 2, 3]);
    expect(Array.from(await readUint32(count.buffer, 1))).toEqual([3]);
    expect(Array.from(await readUint32(overflow.buffer, 1))).toEqual([0]);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
