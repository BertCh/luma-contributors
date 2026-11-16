// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createTrackTableOffsetsNode} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/movement-common';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {WGSL_RESERVED_WORDS} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('searches each sorted track boundary independently, including empty and invalid trailing tracks', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const tracks = createTransientView(graph, 'tracks', 'uint32', 8);
    const count = createTransientView(graph, 'count', 'uint32', 1);
    const offsets = createTransientView(graph, 'offsets', 'uint32', 6);
    const node = createTrackTableOffsetsNode(graph, {
      id: 'offsets',
      operation: 'TestTrackOffsets',
      tracks,
      count,
      offsets,
      capacity: 8,
      trackCount: 5
    });
    const kernel = getGPUCommandNodeKernelSources([node])[0];
    expect(kernel.constants['INVOCATION_COUNT']).toBe(6);
    const source = kernel.source;
    expect(source).not.toContain('for (var track');
    expect(
      (source.match(/[A-Za-z_][A-Za-z_0-9]*/g) ?? []).filter(identifier =>
        WGSL_RESERVED_WORDS.has(identifier)
      )
    ).toEqual([]);
    const memory = createCPUGraphMemory();
    memory.write(tracks, new Uint32Array([0, 0, 0, 2, 3, 3, 6, 0xffffffff]));
    for (const [total, expected] of [
      [7, [0, 3, 3, 4, 6, 6]],
      [2, [0, 2, 2, 2, 2, 2]],
      [0, [0, 0, 0, 0, 0, 0]]
    ] as const) {
      memory.write(count, new Uint32Array([total]));
      executeCommandNodesOnCPU([node], memory);
      expect(Array.from(memory.readUint32(offsets))).toEqual(expected);
    }
  } finally {
    device.destroy();
  }
});
