// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPURasterSpectralUnmixing} from '../../../src/raster-analysis/remote-sensing/gpu-raster-spectral-unmixing';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../../gpu-spatial-analysis/segment-intersection/wgsl-graph-executor';

it('executes a two-iteration NNLS count rather than a boolean iteration flag', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const values = createTransientView(graph, 'values', 'float32', 4);
  const endmembers = createTransientView(graph, 'endmembers', 'float32', 4);
  const abundances = createTransientView(graph, 'abundances', 'float32', 4);
  const residual = createTransientView(graph, 'residual', 'float32', 2);
  const status = createTransientView(graph, 'status', 'uint32', 2);
  const iterationCounts = createTransientView(graph, 'iterations', 'uint32', 2);
  const contributor = new GPURasterSpectralUnmixing({
    cellCount: 2,
    bandCount: 2,
    endmemberCount: 2,
    values,
    endmembers,
    constraint: 'nonnegative',
    maximumIterations: 2,
    tolerance: 1e-6,
    output: {abundances, residual, status, iterationCounts}
  });
  const memory = createCPUGraphMemory();
  memory.write(values, Float32Array.of(2, 4, 0, 0));
  memory.write(endmembers, Float32Array.of(1, 0, 0, 1));
  memory.write(iterationCounts, new Uint32Array(2).fill(0xdeadbeef));
  memory.write(status, new Uint32Array(2).fill(0xdeadbeef));
  try {
    executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
    // Identity design and step 1/trace(E'E)=1/2: [0,0] -> [1,2] -> [1.5,3].
    // The zero spectrum finishes on its first zero update; the nonzero spectrum
    // still has gradient [-0.5,-1]. A boolean performed>0 would incorrectly give [1,1].
    expect(Array.from(memory.readUint32(iterationCounts))).toEqual([2, 1]);
    expect(Array.from(memory.readUint32(status))).toEqual([4, 0]);
    expect(Array.from(memory.readFloat32(abundances))).toEqual([1.5, 3, 0, 0]);
    expect(Array.from(memory.readFloat32(residual))).toEqual([1.25, 0]);
  } finally {
    device.destroy();
  }
});
