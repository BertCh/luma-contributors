// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUMaxPRegions} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-max-p-regions';
import {GPURegionPartitionEvaluation} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-region-partition-evaluation';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {HASH_UINT32_WGSL as coreHash} from '../../../src/analysis-core/kernels/hash-uint32';
import {HASH_UINT32_WGSL as shimHash} from '../../../src/gpu-spatial-analysis/shared/hash-uint32';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('bounds each mean reduction stage and preserves the shared hash text', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const rows = 65537; // 257 row tiles require two stages: 257 -> 2 -> 1.
    new GPURegionPartitionEvaluation({
      id: 'staged',
      values: createTransientView(graph, 'values', 'float32', rows),
      labels: createTransientView(graph, 'labels', 'uint32', rows),
      labelCapacity: rows,
      output: {summary: createTransientView(graph, 'summary', 'float32', 9)}
    }).getCommandNodes(graph);
    expect(getWGSLKernelNodeRecord(graph, 'staged-mean-stage-0')!.invocationCount).toBe(2);
    expect(getWGSLKernelNodeRecord(graph, 'staged-mean-stage-1')!.invocationCount).toBe(1);
    expect(getWGSLKernelNodeRecord(graph, 'staged-mean-stage-0')!.body).toContain(
      '(group + 1u) * 256u'
    );
    expect(getWGSLKernelNodeRecord(graph, 'staged-means')!.body).not.toContain('for (');
    expect(coreHash).toBe(shimHash);
  } finally {
    device.destroy();
  }
});

it('writes optional Max-p evidence on both refinement routes', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const maximumIterations of [0, 1]) {
      const graph = new GPUCommandGraph(device);
      recordWGSLKernelNodes(graph);
      const view = (name: string, length: number) =>
        createTransientView(graph, name, 'uint32', length);
      const contributor = new GPUMaxPRegions({
        id: 'evidence',
        weights: {
          offsets: view('offsets', 3),
          neighbors: view('neighbors', 2),
          weights: createTransientView(graph, 'weights', 'float32', 2)
        },
        values: createTransientView(graph, 'values', 'float32', 2),
        extensiveValues: createTransientView(graph, 'extensive', 'float32', 2),
        threshold: createTransientView(graph, 'threshold', 'float32', 1),
        parameters: view('parameters', 1),
        startCount: 1,
        maximumGrowthSteps: 1,
        maximumIterations,
        output: {
          labels: view('labels', 2),
          objective: createTransientView(graph, 'objective', 'float32', 1),
          status: view('status', 6),
          evidence: view('evidence', 5)
        }
      });
      contributor.getCommandNodes(graph);
      expect(getWGSLKernelNodeRecord(graph, 'evidence-initialize')!.invocationCount).toBe(6);
      expect(getWGSLKernelNodeRecord(graph, 'evidence-initialize')!.body).not.toContain('for (');
      const record = getWGSLKernelNodeRecord(graph, 'evidence-evidence')!;
      expect(record.invocationCount).toBe(5);
      expect(record.bindings.some(binding => binding.name === 'improvement')).toBe(
        maximumIterations > 0
      );
    }
  } finally {
    device.destroy();
  }
});
