// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUCrossVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-cross-variogram';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('the legacy default forwards to the shared cell index without row-sized serial loops', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const rows = 4093;
  const contributor = new GPUCrossVariogram({
    positions: createTransientView(graph, 'positions', 'float32x2', rows),
    values: createTransientView(graph, 'values', 'float32', rows),
    parameters: createTransientView(graph, 'parameters', 'float32', 1),
    lagCount: 4,
    output: {
      semivariances: createTransientView(graph, 'semivariances', 'float32', 4),
      pairCounts: createTransientView(graph, 'counts', 'uint32', 4),
      status: createTransientView(graph, 'status', 'uint32', 4)
    }
  });
  const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
  expect([...sources.keys()].some(name => name.includes('cell-keys'))).toBe(true);
  expect([...sources.keys()].some(name => name.includes('legacy-pairs'))).toBe(true);
  for (const source of sources.values()) {
    expect(source).not.toMatch(/for\s*\([^;]*;[^;]*<\s*4093u\s*;/);
  }
});

it('preflights the legacy compatibility scratch before allocating indexed intermediates', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const contributor = new GPUCrossVariogram({
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    values: createTransientView(graph, 'values', 'float32', 4),
    parameters: createTransientView(graph, 'parameters', 'float32', 1),
    lagCount: 10,
    gridSize: [1, 1],
    output: {
      semivariances: createTransientView(graph, 'semivariances', 'float32', 10),
      pairCounts: createTransientView(graph, 'counts', 'uint32', 10),
      status: createTransientView(graph, 'status', 'uint32', 10)
    }
  });
  Object.assign(device.limits, {maxStorageBufferBindingSize: 100});
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /legacy variogram partials.*160 bytes.*limit 100/
    );
    expect(allocations).not.toHaveBeenCalled();
  } finally {
    allocations.mockRestore();
    device.destroy();
  }
});
