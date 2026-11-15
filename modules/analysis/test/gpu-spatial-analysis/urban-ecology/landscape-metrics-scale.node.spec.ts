// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {addLandscapeMetricsRecipe} from '../../../src/gpu-spatial-analysis/landscape-ecology/landscape-metrics';
import {reconcileLandscapePatchLabels} from '../../../src/gpu-spatial-analysis/landscape-ecology/reconcile-patches';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

it.each([
  true,
  false
])('landscape summaries use raster patch measurements and bounded raster walks (countBackground=%s)', countBackground => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    addLandscapeMetricsRecipe(graph, {
      patches: createTransientView(graph, 'patches', 'uint32', 257 * 129),
      patchClasses: createTransientView(graph, 'classes', 'uint32', 8193),
      width: 257,
      height: 129,
      classCount: 7,
      cellSize: [2, 3],
      connectivity: 8,
      countBackground,
      reconcilePatches: true
    });
    const kernels = getGPUCommandNodeKernelSources((graph as unknown as {nodes: unknown[]}).nodes);
    expect(kernels.some(kernel => kernel.operation === 'GPURasterPatchMetrics')).toBe(true);
    for (const kernel of kernels) {
      expect(getStorageBindingCount(kernel.source), kernel.id).toBeLessThanOrEqual(8);
      expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
      // No invocation may rescan all cells or all patches. Raster loops use the shared bounded
      // row segments, and class sums use the shared parallel segmented reduction instead.
      expect(kernel.source, kernel.id).not.toMatch(/for\s*\(var\s+(cell|patchIndex)\s*=\s*0u/);
    }
    expect(kernels.length).toBeLessThan(512);
  } finally {
    device.destroy();
  }
});

it('seam reconciliation dispatches one union invocation per cell and one root invocation per patch', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    reconcileLandscapePatchLabels(graph, {
      id: 'seams',
      width: 257,
      height: 129,
      connectivity: 8,
      patches: createTransientView(graph, 'patches', 'uint32', 257 * 129),
      patchClasses: createTransientView(graph, 'classes', 'uint32', 8193)
    });
    const kernels = getGPUCommandNodeKernelSources((graph as unknown as {nodes: unknown[]}).nodes);
    const union = kernels.find(kernel => kernel.id.includes('seams-reconcile'))!;
    expect(union.source).toContain('atomicCompareExchangeWeak');
    expect(union.source).not.toMatch(/for\s*\(var\s+cell\s*=/);
    expect(kernels.some(kernel => kernel.id.includes('patch-roots'))).toBe(true);
    for (const kernel of kernels) expect(() => new WgslReflect(kernel.source)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it.each([
  'raster',
  'patches',
  'counter'
] as const)('landscape rejects the %s limit before mutating the graph', limit => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const width = limit === 'counter' ? 0x40000000 : limit === 'raster' ? 33554433 : 2;
    const patchCount = limit === 'patches' ? 8388609 : 1;
    const expected =
      limit === 'counter'
        ? /width times height must be at most 1073741823/
        : /needs .* bytes in one binding \(limit 134217728\)/;
    expect(() =>
      addLandscapeMetricsRecipe(graph, {
        patches: createTransientView(graph, 'patches', 'uint32', width),
        patchClasses: createTransientView(graph, 'classes', 'uint32', patchCount),
        width,
        height: 1,
        classCount: 1,
        cellSize: [1, 1],
        connectivity: 4,
        reconcilePatches: true
      })
    ).toThrow(expected);
    expect((graph as unknown as {nodes: unknown[]}).nodes).toHaveLength(0);
  } finally {
    device.destroy();
  }
});
