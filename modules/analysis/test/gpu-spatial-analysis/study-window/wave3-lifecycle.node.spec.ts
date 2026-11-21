// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUStudyWindowPreparation} from '../../../src/gpu-spatial-analysis/study-window/gpu-study-window-preparation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {Buffer} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';

it('LIF-04 captures samplingGrid before validation and reports the same grid after declaration', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const samplingGrid: [number, number] = [1, 1];
  const prepared = new GPUStudyWindowPreparation({
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    featureOffsets: createTransientView(graph, 'features', 'uint32', 2),
    polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 2),
    ringOffsets: createTransientView(graph, 'rings', 'uint32', 2),
    samplingGrid
  });
  try {
    samplingGrid[0] = 2;
    expect(prepared.sampleGrid).toEqual([1, 1]);
    graph.add(prepared);
    expect(prepared.getViews(graph).sampleGrid).toEqual([1, 1]);
  } finally {
    prepared.destroy();
    device.destroy();
  }
});

it('LIF-11 releases study-window acquisitions after edges allocation fails', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const inputs = Array.from({length: 4}, () =>
    device.createBuffer({byteLength: 32, usage: Buffer.STORAGE | Buffer.COPY_DST})
  );
  const prepared = new GPUStudyWindowPreparation({
    id: 'failure-window',
    positions: importGraphBuffer(graph, 'positions', inputs[0], 'float32x2', 4),
    featureOffsets: importGraphBuffer(graph, 'features', inputs[1], 'uint32', 2),
    polygonOffsets: importGraphBuffer(graph, 'polygons', inputs[2], 'uint32', 2),
    ringOffsets: importGraphBuffer(graph, 'rings', inputs[3], 'uint32', 2),
    samplingGrid: [1, 1]
  });
  const created: {destroyed: boolean}[] = [];
  const allocate = device.createBuffer.bind(device);
  const spy = vi.spyOn(device, 'createBuffer').mockImplementation(settings => {
    if (
      typeof settings === 'object' &&
      settings !== null &&
      'id' in settings &&
      settings.id === 'failure-window-edges'
    )
      throw new Error('injected edges');
    const buffer = allocate(settings);
    created.push(buffer);
    return buffer;
  });
  try {
    expect(() => graph.add(prepared)).toThrow('injected edges');
    expect(prepared.isDeclaredIn(graph)).toBe(false);
    expect(created.filter(buffer => !buffer.destroyed)).toHaveLength(0);
    spy.mockRestore();
    const fresh = new GPUCommandGraph(device);
    expect(() => fresh.add(prepared)).not.toThrow();
    expect(prepared.isDeclaredIn(fresh)).toBe(true);
    expect(prepared.getViews(fresh).sampleGrid).toEqual([1, 1]);
  } finally {
    spy.mockRestore();
    prepared.destroy();
    for (const input of inputs) input.destroy();
    device.destroy();
  }
});
