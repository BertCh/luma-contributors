// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPUSpatialKMeans,
  getGPUSpatialKMeansParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-clustering';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('distinguishes last-round Lloyd convergence from exhaustion and retains the mean', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const scale of [1, 2 ** -80]) {
    for (const maximumIterations of [1, 2]) {
      const graph = new GPUCommandGraph(device);
      const positions = createInputBuffer(device, Float32Array.of(0, 0, 2 * scale, 0));
      const parameters = createInputBuffer(
        device,
        getGPUSpatialKMeansParameterValues({seed: 1, tolerance: 0})
      );
      const labels = createOutputBuffer(device, 2);
      const centers = createOutputBuffer(device, 2);
      const flags = createOutputBuffer(device, 2);
      const flagsView = importGraphBuffer(graph, 'flags', flags, 'uint32', 2);
      const solver = new GPUSpatialKMeans({
        positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 2),
        parameters: importGraphBuffer(graph, 'parameters', parameters, 'uint32', 2),
        initialization: 'first-valid',
        clusterCount: 1,
        maximumIterations,
        output: {
          labels: importGraphBuffer(graph, 'labels', labels, 'uint32', 2),
          centers: importGraphBuffer(graph, 'centers', centers, 'float32x2', 1)
        }
      });
      graph.add(solver);
      graph.add(
        createWGSLKernelNode(graph, {
          id: 'copy-flags',
          operation: 'test',
          variant: 'copy',
          invocationCount: 1,
          bindings: [
            {
              name: 'converged',
              view: solver.output.status!.converged!,
              type: 'u32',
              access: 'read'
            },
            {
              name: 'limited',
              view: solver.output.status!.iterationLimitReached!,
              type: 'u32',
              access: 'read'
            },
            {name: 'flags', view: flagsView, type: 'u32', access: 'read_write'}
          ],
          body: 'flags[flagsOffset] = converged[convergedOffset]; flags[flagsOffset + 1u] = limited[limitedOffset];'
        })
      );
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      // First round moves (0,0) to arithmetic mean (1,0); second round has zero displacement.
      expect(await readFloat32(centers, 2)).toEqual([scale, 0]);
      expect(await readUint32(flags, 2)).toEqual(maximumIterations === 1 ? [0, 1] : [1, 0]);
      compiled.destroy();
      for (const buffer of [positions, parameters, labels, centers, flags]) buffer.destroy();
    }
  }
});
