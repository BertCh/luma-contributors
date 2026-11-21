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
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

it('B3 interprets the emitted enclave probe, reduction and evidence writer', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const uint = (name: string, length: number) =>
      createTransientView(graph, name, 'uint32', length);
    const float = (name: string, length: number) =>
      createTransientView(graph, name, 'float32', length);
    const nodes = new GPUMaxPRegions({
      id: 'enclave',
      weights: {
        offsets: uint('offsets', 6),
        neighbors: uint('neighbors', 8),
        weights: float('weights', 8)
      },
      values: float('values', 5),
      extensiveValues: float('extensive', 5),
      threshold: float('threshold', 1),
      parameters: uint('parameters', 1),
      startCount: 1,
      maximumGrowthSteps: 3,
      output: {
        labels: uint('labels', 5),
        objective: float('objective', 1),
        status: uint('status', 6),
        evidence: uint('evidence', 5)
      }
    }).getCommandNodes(graph);
    const binding = (suffix: string, name: string) =>
      getWGSLKernelNodeRecord(graph, `enclave-${suffix}`)!.bindings.find(
        value => value.name === name
      )!.view;
    for (const unfinished of [true, false]) {
      const memory = createCPUGraphMemory();
      memory.write(binding('enclave-pending', 'offsets'), Uint32Array.of(0, 1, 3, 5, 7, 8));
      memory.write(binding('enclave-pending', 'neighbors'), Uint32Array.of(1, 0, 2, 1, 3, 2, 4, 3));
      // Frozen end-of-enclave state at three rounds versus four rounds, before splitting leftovers.
      memory.write(
        binding('enclave-pending', 'rowState'),
        Uint32Array.of(0, 0, 0, 0, 0, 0, 0, 0, unfinished ? 4 : 0, 0)
      );
      memory.write(binding('enclave-pending', 'regionPhase'), Uint32Array.of(2, 0, 0, 0, 3));
      executeCommandNodesOnCPU(
        nodes.filter(node =>
          [
            'enclave-enclave-pending',
            'enclave-enclave-pending-stage-0',
            'enclave-evidence'
          ].includes(node.id)
        ),
        memory
      );
      expect(Array.from(memory.readUint32(binding('evidence', 'evidence'), 5))).toEqual(
        unfinished ? [1, 0, 0, 0, 0] : [0, 1, 0, 0, 0]
      );
    }
  } finally {
    device.destroy();
  }
});

it('B8 interprets two staged mean reductions with nonzero contributions from tile 257', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const nodes = new GPURegionPartitionEvaluation({
      id: 'mean',
      values: createTransientView(graph, 'values', 'float32', 2 * 65537),
      columnCount: 2,
      labels: createTransientView(graph, 'labels', 'uint32', 65537),
      output: {summary: createTransientView(graph, 'summary', 'float32', 9)}
    }).getCommandNodes(graph);
    const binding = (suffix: string, name: string) =>
      getWGSLKernelNodeRecord(graph, `mean-${suffix}`)!.bindings.find(value => value.name === name)!
        .view;
    const memory = createCPUGraphMemory();
    const counts = new Uint32Array(257 * 5),
      sums = new Float32Array(257 * 2 * 2);
    // x alternates 0,2 in the first 65536 rows; y=2*x+3 has origin 3.
    for (let tile = 0; tile < 256; tile++) {
      counts[5 * tile] = 256;
      // Current row sums interleave (high, residual) for each column. Residuals are zero here.
      sums[4 * tile] = 256;
      sums[4 * tile + 2] = 512;
    }
    counts[5 * 256] = 1;
    sums[4 * 256] = 4096;
    sums[4 * 256 + 2] = 8192;
    memory.write(binding('mean-stage-0', 'rows'), counts);
    memory.write(binding('mean-stage-0', 'sums'), sums);
    executeCommandNodesOnCPU(
      nodes.filter(node =>
        ['mean-mean-stage-0', 'mean-mean-stage-1', 'mean-means'].includes(node.id)
      ),
      memory
    );
    const actual = memory.readFloat32(binding('means', 'means'), 2);
    // Exact sums/count: 69632/65537 and 139264/65537 in origin-relative coordinates.
    expect(Array.from(actual)).toEqual([Math.fround(69632 / 65537), Math.fround(139264 / 65537)]);
    expect(actual.every(Number.isFinite)).toBe(true);
  } finally {
    device.destroy();
  }
});
