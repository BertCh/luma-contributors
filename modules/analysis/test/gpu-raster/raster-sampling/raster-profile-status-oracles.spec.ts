// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPURasterProfile,
  getGPURasterProfileParameterValues
} from '../../../src/raster-analysis/raster-sampling/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('reports per-line truncation when all clamped profile samples fit, then clears it', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const grid = {
    width: 1,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const values = createInputBuffer(device, Float32Array.of(5));
  const positions = createInputBuffer(device, Float32Array.of(0, 0.5, 1, 0.5));
  // Line zero has length one; all 65,534 other lines have no vertices.
  const offsets = new Uint32Array(65536).fill(2);
  offsets[0] = 0;
  const lineOffsets = createInputBuffer(device, offsets);
  const count = createOutputBuffer(device, 1);
  const requiredCount = createOutputBuffer(device, 1);
  const overflow = createOutputBuffer(device, 1);
  for (const buffer of [count, requiredCount, overflow]) buffer.write(Uint32Array.of(0xdeadbeef));
  const parameters = new GPUParameterBuffer(device, {
    id: 'profile-status-parameters',
    format: 'float32',
    length: 12
  });
  graph.add(
    new GPURasterProfile({
      grid,
      values: importGraphBuffer(graph, 'values', values, 'float32', 1),
      positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 2),
      lineOffsets: importGraphBuffer(graph, 'offsets', lineOffsets, 'uint32', 65536),
      sampleCapacity: 65535,
      parameters: parameters.importToGraph(graph),
      output: {
        status: {
          count: importGraphBuffer(graph, 'count', count, 'uint32', 1),
          requiredCount: importGraphBuffer(graph, 'required', requiredCount, 'uint32', 1),
          overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
        }
      }
    })
  );
  const compiled = graph.compile();
  try {
    // floor(0xffffffff/(65535+1))=65535 is the per-line cap.
    // Spacing 1/65536 requests 65,537 points including both endpoints, so truncation
    // is forced although all 65,535 retained rows fit the output capacity exactly.
    parameters.write(getGPURasterProfileParameterValues({spacing: 1 / 65536}, grid));
    submitGraph(device, compiled);
    expect(await readUint32(count, 1)).toEqual([65535]);
    // requiredCount is before total-capacity clamping, after the per-line truncation above.
    expect(await readUint32(requiredCount, 1)).toEqual([65535]);
    expect(await readUint32(overflow, 1)).toEqual([1]);
    // One segment at spacing one requests only its two endpoints; the same graph must
    // clear the earlier truncation flag even though no capacity or topology changed.
    parameters.write(getGPURasterProfileParameterValues({spacing: 1}, grid));
    submitGraph(device, compiled);
    expect(await readUint32(count, 1)).toEqual([2]);
    expect(await readUint32(requiredCount, 1)).toEqual([2]);
    expect(await readUint32(overflow, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    parameters.destroy();
    for (const buffer of [values, positions, lineOffsets, count, requiredCount, overflow])
      buffer.destroy();
  }
}, 120000);
