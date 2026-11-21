// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGeographicDistribution} from '../../../src/gpu-spatial-analysis/geographic-distribution/gpu-geographic-distribution';
import {getGPUGeographicDistributionParameterValues} from '../../../src/gpu-spatial-analysis/geographic-distribution/geographic-distribution-parameters';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {expectNumericArrayClose} from '../../../src/testing/expect-numeric-array-close';

for (const [label, positions, shortAxis] of [
  [
    'thin',
    [
      100 + 1 / 64,
      99 + 1 / 64,
      100 - 1 / 64,
      99 - 1 / 64,
      -100 + 1 / 64,
      -99 + 1 / 64,
      -100 - 1 / 64,
      -99 - 1 / 64
    ],
    0.0001110392344137977
  ],
  ['collinear', [1, Math.fround(0.1), -1, -Math.fround(0.1)], 0]
] as const) {
  it(`projects the ${label} cloud before rounding its transverse variance`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    recordWGSLKernelNodes(graph);
    const rows = positions.length / 2;
    const parameters = getGPUGeographicDistributionParameterValues({ellipseConvention: 'standard'});
    const input = createTransientView(graph, 'positions', 'float32x2', rows);
    const parameterView = createTransientView(graph, 'parameters', 'float32', parameters.length);
    const ellipses = createTransientView(graph, 'ellipses', 'float32', 3);
    try {
      const nodes = new GPUGeographicDistribution({
        positions: input,
        parameters: parameterView,
        reduction: 'sorted',
        output: {ellipses}
      }).getCommandNodes(graph);
      const memory = createCPUGraphMemory();
      memory.write(parameterView, parameters);
      const x = Float32Array.from({length: rows}, (_, index) => positions[2 * index]);
      const y = Float32Array.from({length: rows}, (_, index) => positions[2 * index + 1]);
      for (const node of nodes) {
        const record = getWGSLKernelNodeRecord(graph, node.id);
        if (!record) continue;
        for (const binding of record.bindings) {
          if (binding.name === 'weightSums') memory.write(binding.view, Float32Array.of(rows));
          if (binding.name === 'sumXX')
            memory.write(
              binding.view,
              Float32Array.of(x.reduce((sum, value) => sum + value * value, 0))
            );
          if (binding.name === 'sumYY')
            memory.write(
              binding.view,
              Float32Array.of(y.reduce((sum, value) => sum + value * value, 0))
            );
          if (binding.name === 'sumXY')
            memory.write(
              binding.view,
              Float32Array.of(x.reduce((sum, value, index) => sum + value * y[index], 0))
            );
          if (binding.name === 'mean') memory.write(binding.view, Float32Array.of(0, 0));
          if (binding.name === 'sortedKeys') memory.write(binding.view, new Uint32Array(rows));
          if (binding.name === 'sortedX') memory.write(binding.view, x);
          if (binding.name === 'sortedY') memory.write(binding.view, y);
          if (binding.name === 'sortedWeights')
            memory.write(binding.view, new Float32Array(rows).fill(1));
        }
      }
      for (const node of nodes.filter(
        node => node.id.endsWith('-spread') || node.id.includes('-projection-')
      )) {
        const record = getWGSLKernelNodeRecord(graph, node.id)!;
        if (node.id.includes('-sum-projection-')) {
          // Only the published cooperative sum is emulated; all owned projection arithmetic runs.
          const source = record.bindings.find(binding => binding.name === 'segmentInput')!.view;
          const result = record.bindings.find(binding => binding.name === 'segmentOutput')!.view;
          memory.write(
            result,
            Float32Array.of(memory.readFloat32(source).reduce((sum, value) => sum + value, 0))
          );
        } else executeCommandNodesOnCPU([node], memory);
      }
      const actual = memory.readFloat32(ellipses);
      expect(actual.every(Number.isFinite)).toBe(true);
      expectNumericArrayClose([actual[1]], [shortAxis], {
        absoluteTolerance: shortAxis === 0 ? 0 : 2e-8
      });
      expect(actual[2]).toBeGreaterThan(0);
    } finally {
      device.destroy();
    }
  });
}

for (const reduction of ['sorted', 'chunked'] as const) {
  it(`keeps ${reduction} projection shaders within eight storage bindings`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, name, format, length);
    try {
      const contributor = new GPUGeographicDistribution({
        positions: view('p', 'float32x2', 4),
        weights: view('w', 'float32', 4),
        mask: view('m', 'uint32', 4),
        groupIds: view('g', 'uint32', 4),
        groupCount: 2,
        parameters: view('parameters', 'float32', 8),
        reduction,
        output: {ellipses: view('e', 'float32', 6), standardDistances: view('s', 'float32', 2)}
      });
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes.length).toBeLessThanOrEqual(contributor.estimatedCommandNodeCount);
      for (const {id, source} of getGPUCommandNodeKernelSources(nodes))
        expect(getStorageBindingCount(source), id).toBeLessThanOrEqual(8);
    } finally {
      device.destroy();
    }
  });
}
