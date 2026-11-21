// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi, type MockInstance} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUTerrainSpikeRepair} from '../../../src/gpu-terrain/terrain-decode/gpu-terrain-spike-repair';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('W5-04 releases partial work-buffer acquisition and supports retry without owning caller buffers', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const elevation = device.createBuffer({
    byteLength: 16,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const validity = device.createBuffer({
    byteLength: 16,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const output = device.createBuffer({
    byteLength: 16,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const callerDestructions = [
    vi.spyOn(elevation, 'destroy'),
    vi.spyOn(output, 'destroy'),
    vi.spyOn(validity, 'destroy')
  ];
  const contributor = new GPUTerrainSpikeRepair({
    id: 'fault-repair',
    grid: makeGPURasterMetadata({width: 2, height: 2}),
    elevation: {
      id: 'elevation',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: importGraphBuffer(graph, 'elevation', elevation, 'float32', 4)
      }
    },
    output: {
      values: importGraphBuffer(graph, 'output', output, 'float32', 4),
      outputValidity: importGraphBuffer(graph, 'validity', validity, 'uint32', 4)
    }
  });
  const allocate = device.createBuffer.bind(device);
  const destructions: MockInstance<() => void>[] = [];
  let allocationCount = 0;
  const allocation = vi.spyOn(device, 'createBuffer').mockImplementation(properties => {
    allocationCount++;
    if (allocationCount === 3) throw new Error('injected third allocation failure');
    const buffer = allocate(properties);
    destructions.push(vi.spyOn(buffer, 'destroy'));
    return buffer;
  });
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow('injected third allocation failure');
    expect(destructions).toHaveLength(2);
    for (const destruction of destructions) expect(destruction).toHaveBeenCalledTimes(1);
    // Both earlier acquisitions must be released exactly once; retry acquires a fresh set of four.
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    expect(destructions).toHaveLength(6);
    for (const destruction of destructions.slice(2)) expect(destruction).not.toHaveBeenCalled();
    contributor.destroy();
    contributor.destroy();
    for (const destruction of destructions) expect(destruction).toHaveBeenCalledTimes(1);
    for (const destruction of callerDestructions) expect(destruction).not.toHaveBeenCalled();
  } finally {
    allocation.mockRestore();
    contributor.destroy();
    elevation.destroy();
    output.destroy();
    validity.destroy();
    vi.restoreAllMocks();
    device.destroy();
  }
});
