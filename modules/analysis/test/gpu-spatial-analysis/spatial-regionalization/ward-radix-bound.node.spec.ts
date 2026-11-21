// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPU_WARD_MUTUAL_SELECTION_RADIX_BIN_COUNT} from '../../../src/gpu-spatial-analysis/spatial-regionalization/ward-mutual-selection';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUWardRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-ward-regions';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('keeps Ward byte-radix reset and increasing digit selection at 256 bins', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const view = (name: string, length: number) =>
      createTransientView(graph, name, 'uint32', length);
    // 4097 rows select the production distributed radix route (the cooperative cutoff is 4096).
    new GPUWardRegions({
      id: 'ward',
      weights: {
        offsets: view('offsets', 4098),
        neighbors: view('neighbors', 4097),
        weights: createTransientView(graph, 'weights', 'float32', 4097)
      },
      values: createTransientView(graph, 'values', 'float32', 4097),
      targetRegionCount: 2,
      maximumIterations: 1,
      mergeSchedule: 'mutual-nearest',
      parameters: view('parameters', 1),
      output: {
        labels: view('labels', 4097),
        objective: createTransientView(graph, 'objective', 'float32', 1),
        status: view('status', 6)
      }
    }).getCommandNodes(graph);
    expect(GPU_WARD_MUTUAL_SELECTION_RADIX_BIN_COUNT).toBe(256);
    const reset = getWGSLKernelNodeRecord(graph, 'ward-round-0-selection-reset')!;
    expect(reset.invocationCount).toBe(256);
    expect(reset.bindings.find(binding => binding.name === 'histogram')!.view.length).toBe(
      GPU_WARD_MUTUAL_SELECTION_RADIX_BIN_COUNT
    );
    const choose = getWGSLKernelNodeRecord(graph, 'ward-round-0-selection-choose-0')!;
    expect(choose.body).toContain('for (var digit = 0u; digit < 256u; digit++)');
    expect(choose.body).toContain('if (remaining > count) { remaining -= count; }');
  } finally {
    device.destroy();
  }
});
