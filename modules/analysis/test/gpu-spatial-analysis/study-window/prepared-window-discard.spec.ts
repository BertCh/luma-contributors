// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const discard of ['full', 'submitted-prefix'] as const) {
  it(`ERG-02 study window retains pending area and bounds after ${discard} refresh`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = <Format extends 'float32x2' | 'uint32'>(
      name: string,
      format: Format,
      values: Float32Array | Uint32Array
    ) => {
      const buffer = device.createBuffer({
        data: values,
        usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
      });
      buffers.push(buffer);
      return {
        buffer,
        view: importGraphBuffer(
          graph,
          name,
          buffer,
          format,
          values.length / (format === 'float32x2' ? 2 : 1)
        )
      };
    };
    const positions = input('positions', 'float32x2', Float32Array.of(0, 0, 2, 0, 2, 3, 0, 3));
    const summary = device.createBuffer({
      byteLength: 32,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(summary);
    const preparation = new GPUStudyWindowPreparation({
      positions: positions.view,
      featureOffsets: input('features', 'uint32', Uint32Array.of(0, 1)).view,
      polygonOffsets: input('polygons', 'uint32', Uint32Array.of(0, 1)).view,
      ringOffsets: input('rings', 'uint32', Uint32Array.of(0, 4)).view,
      samplingGrid: [2, 2],
      output: {summary: importGraphBuffer(graph, 'summary', summary, 'float32', 8)}
    });
    graph.add(preparation);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // Closed form: 2x3 rectangle area 6, perimeter 10; refreshed 4x2 area 8, perimeter 12.
      // Both have exactly four nondegenerate edges and no orientation/empty-window flags.
      expect(await readFloat32(summary, 8)).toEqual([0, 0, 2, 3, 6, 10, 4, 0]);
      positions.buffer.write(Float32Array.of(10, 20, 14, 20, 14, 22, 10, 22));
      preparation.invalidate();
      const encoder = device.createCommandEncoder();
      try {
        if (discard === 'full') compiled.encode(encoder, {parameters: undefined});
        else {
          const execution = compiled.createExecution({
            maximumInvocationCount: 1_000_000,
            maximumNodeCount: 2
          });
          expect(execution.encodeNext(encoder, {parameters: undefined}).nodeCount).toBe(2);
        }
        const commandBuffer = encoder.finish();
        if (discard === 'submitted-prefix') device.submit(commandBuffer);
        commandBuffer.destroy();
      } finally {
        encoder.destroy();
      }
      // The abandoned full encoding or submitted request-and-prepare prefix cannot commit a refresh.
      expect(await readFloat32(summary, 8)).toEqual([0, 0, 2, 3, 6, 10, 4, 0]);
      submitGraph(device, compiled, undefined);
      expect(await readFloat32(summary, 8)).toEqual([10, 20, 14, 22, 8, 12, 4, 0]);
    } finally {
      compiled.destroy();
      preparation.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
