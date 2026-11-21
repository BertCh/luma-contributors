// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUDorlingCartogram} from '../../../src/gpu-spatial-analysis/dorling-cartogram/gpu-dorling-cartogram';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

it('the Dorling guide covers the generated neighbor-grid storage binding count', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `guide-${serial++}`, format, length);
    const nodes = new GPUDorlingCartogram({
      positions: view('float32x2', 2),
      values: view('float32', 2),
      neighborGrid: true,
      maximumIterations: 1,
      output: {
        positions: view('float32x2', 2),
        radii: view('float32', 2),
        sourceIds: view('uint32', 2),
        metrics: view('float32x2', 1),
        status: view('uint32', 4)
      }
    }).getCommandNodes(graph);
    const guide = readFileSync(
      new URL(
        '../../../../../docs/api/gpu-spatial-analysis/gpu-dorling-cartogram.md',
        import.meta.url
      ),
      'utf8'
    );
    const boundWord = guide.match(/Kernels use at most (\w+) storage buffers/)?.[1];
    const words: Record<string, number> = {seven: 7, eight: 8};
    expect(boundWord).toBeDefined();
    const limit = words[boundWord!];
    expect(Number.isFinite(limit)).toBe(true);
    const kernels = getGPUCommandNodeKernelSources(nodes);
    expect(kernels.some(({id}) => id.includes('-relax-1'))).toBe(true);
    for (const {id, source} of kernels)
      expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(limit);
  } finally {
    device.destroy();
  }
});
