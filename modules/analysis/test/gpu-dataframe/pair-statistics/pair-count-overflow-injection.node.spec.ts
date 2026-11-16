// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUCrossVariogram} from '../../../src/dataframe-analysis/pair-statistics/gpu-cross-variogram';
import {GPUMarkedPointStatistics} from '../../../src/dataframe-analysis/pair-statistics/gpu-marked-point-statistics';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys
} from '../../utils/gpu-contributor-test-utils';
import {getPairCountOverflowPartials} from './pair-count-overflow-injection';

it.each([
  'cross',
  'marked'
] as const)('%s overflow injection reaches the real count tree and finish kernel', estimator => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const rows = estimator === 'cross' ? 92683 : 65537;
  const bins = 3;
  let serial = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `view-${serial++}`, format, length);
  try {
    const positions = view('float32x2', rows);
    const pairCounts = view('uint32', bins);
    const status = view('uint32', bins);
    const contributor =
      estimator === 'cross'
        ? new GPUCrossVariogram({
            id: 'cross',
            positions,
            values: view('float32', rows),
            parameters: view('float32', 1),
            lagCount: bins,
            output: {semivariances: view('float32', bins), pairCounts, status}
          })
        : new GPUMarkedPointStatistics({
            id: 'marked',
            positions,
            parameters: view('float32', 6),
            radiusCount: bins,
            edgeCorrection: 'none',
            output: {statistics: view('float32', 4 * bins), pairCounts, status}
          });
    const nodes = contributor.getCommandNodes(graph);
    const counts = nodes.filter(node => node.workload?.variant === 'segment-total-u32');
    expect(counts).toHaveLength(3);
    // The headless injection must work even though this builder has no WGSL record.
    expect(getWGSLKernelNodeRecord(graph, counts[0].id)).toBeUndefined();
    const partials = getPairCountOverflowPartials(counts[0]);
    expect(partials.format).toBe('uint32');
    expect(partials.length).toBe(rows * bins);
    for (let level = 1; level < counts.length; level++) {
      const previousOutput = counts[level - 1].resources?.find(
        resource => 'buffer' in resource && resource.usage === 'storage-read-write'
      );
      expect(previousOutput && 'buffer' in previousOutput && previousOutput.buffer).toBe(
        getPairCountOverflowPartials(counts[level])
      );
    }
    const finish = nodes.find(
      node => node.workload?.variant === (estimator === 'cross' ? 'legacy-finish' : 'grid-finish')
    )!;
    const record = getWGSLKernelNodeRecord(graph, finish.id)!;
    const totals = record.bindings.find(binding => binding.name === 'totalCounts')!.view;
    expect(totals.length).toBe(2 * bins);
    expect(counts.at(-1)!.resources).toContainEqual({
      buffer: totals,
      usage: 'storage-read-write'
    });
    expect(record.bindings.find(binding => binding.name === 'totalSums')).toBeDefined();
    for (const kernel of getGPUCommandNodeKernelSources([...counts, finish]))
      expect(auditWGSLModule(kernel.source, kernel.constants), kernel.id).toEqual([]);
    expect(getGPUCommandNodePipelineKeys(counts, {includeShaderLayout: true}).size).toBe(1);
  } finally {
    device.destroy();
  }
});
