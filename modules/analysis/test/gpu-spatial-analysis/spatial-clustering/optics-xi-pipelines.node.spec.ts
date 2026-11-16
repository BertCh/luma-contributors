// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import type {GPUOPTICSProps} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {createOPTICSXiNodes} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics-xi';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('xi uses the GPU cursor without specializing shader source or pipeline constants per batch', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const rows of [128, 65536]) {
      const graph = new GPUCommandGraph(device);
      const props: GPUOPTICSProps = {
        positions: createTransientView(graph, 'positions', 'float32x2', rows),
        minimumSamples: 2,
        maximumDistance: 10,
        extractionMethod: 'xi',
        output: {
          ordering: createTransientView(graph, 'ordering', 'uint32', rows),
          coreDistances: createTransientView(graph, 'core', 'float32', rows),
          reachability: createTransientView(graph, 'reachability', 'float32', rows),
          predecessors: createTransientView(graph, 'predecessors', 'uint32', rows),
          labels: createTransientView(graph, 'labels', 'uint32', rows),
          status: createTransientView(graph, 'status', 'uint32', 3)
        }
      };
      const nodes = createOPTICSXiNodes(graph, props, 'xi-pipeline-check');
      for (const variant of ['xi-extraction', 'xi-gate']) {
        const batches = nodes.filter(node => node.workload?.variant === variant);
        expect(batches.length).toBeGreaterThan(1);
        // Sample both ends and an interior batch; avoid compiling 2,048 identical
        // templates on the memory-constrained node test worker.
        const sources = getGPUCommandNodeKernelSources([
          batches[0],
          batches[Math.floor(batches.length / 2)],
          batches[batches.length - 1]
        ]);
        expect(new Set(sources.map(kernel => kernel.source)).size).toBe(1);
        expect(new Set(sources.map(kernel => JSON.stringify(kernel.constants))).size).toBe(1);
        expect(new Set(sources.map(kernel => JSON.stringify(kernel.shaderLayout))).size).toBe(1);
        if (variant === 'xi-extraction') {
          expect(sources[0].source).toContain('areas[areasOffset + BATCH_START_OFFSET]');
        }
      }
    }
  } finally {
    device.destroy();
  }
});
