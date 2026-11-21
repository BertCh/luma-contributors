// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUCommandGraph,
  GPUElementwise,
  createTransientView,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUNetworkCostMatrix,
  getGPUNetworkAccessibilityParameterValues,
  getGPUNetworkCostWeightsParameterValues
} from '@luma-contributors/analysis/gpu-network';
import {importGraphBuffer} from '../../../../analysis/src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../../analysis/src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../../../../analysis/src/testing/gpu-contributor-test-utils';
import {runExample as runAccessibility} from '../../../src/computation-lenses/scenarios/gpu-network/network-accessibility.example';
import {runExample as runWeights} from '../../../src/computation-lenses/scenarios/gpu-network/network-cost-weights.example';

it('executable network examples certify the five-node horizon and recover after incomplete routing', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Directed unit chain: horizon 1 reaches nodes [0,1]; horizon 4 reaches [0,1,2,3,4].
  // A 512-node chain with one local relaxation and one round cannot finish routing.
  for (const [nodeCount, horizon, iterations, expectedCount, expectedComplete] of [
    [512, 512, 1, undefined, 0],
    [5, 1, 24, 2, 0],
    [5, 4, 24, 5, 1]
  ] as const) {
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = <Format extends 'float32' | 'uint32'>(
      name: string,
      values: Float32Array | Uint32Array,
      format: Format
    ) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, format, values.length);
    };
    const observe = (name: string, view: GraphDataView<'uint32'> | GraphDataView<'float32'>) => {
      const buffer = createOutputBuffer(device, view.length);
      buffers.push(buffer);
      graph.add(
        new GPUElementwise({
          id: `observe-${name}`,
          operation: 'copy',
          input: view,
          output: importGraphBuffer(graph, `observed-${name}`, buffer, view.format, view.length)
        })
      );
      return buffer;
    };
    const matrixCosts = createTransientView(graph, 'matrix', 'float32', nodeCount);
    const matrixComplete = createTransientView(graph, 'matrix-complete', 'uint32', 1);
    const matrixCostLimit = input('horizon', Float32Array.of(horizon), 'float32');
    graph.add(
      new GPUNetworkCostMatrix({
        topology: {
          offsets: input(
            'offsets',
            Uint32Array.from({length: nodeCount + 1}, (_, index) => Math.min(index, nodeCount - 1)),
            'uint32'
          ),
          neighbors: input(
            'neighbors',
            Uint32Array.from({length: nodeCount - 1}, (_, index) => index + 1),
            'uint32'
          ),
          weights: input('costs', new Float32Array(nodeCount - 1).fill(1), 'float32')
        },
        startNodes: input('start', Uint32Array.of(0), 'uint32'),
        costLimit: matrixCostLimit,
        maximumIterations: iterations,
        localIterations: iterations === 1 ? 1 : 32,
        output: {costs: matrixCosts, status: {converged: matrixComplete}}
      })
    );
    const accessibility = await runAccessibility(device, {
      graph,
      matrixCosts,
      matrixComplete,
      matrixCostLimit,
      opportunityWeights: input('opportunities', Float32Array.of(1), 'float32'),
      parameters: input(
        'scoring',
        getGPUNetworkAccessibilityParameterValues({threshold: 4}),
        'float32'
      ),
      nodeCount
    });
    const weights = await runWeights(device, {
      graph,
      matrixCosts,
      matrixComplete,
      matrixCostLimit,
      parameters: input(
        'decay',
        getGPUNetworkCostWeightsParameterValues({costLimit: 4}),
        'float32'
      ),
      facilityCount: 1,
      nodeCount
    });
    const scoreComplete = observe('score-complete', accessibility.complete!);
    const weightsComplete = observe('weights-complete', weights.status.complete!);
    const offsets = observe('weight-offsets', weights.offsets);
    const scores = observe('scores', accessibility.gravity);
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      expect(await readUint32(scoreComplete, 1)).toEqual([expectedComplete]);
      expect(await readUint32(weightsComplete, 1)).toEqual([expectedComplete]);
      if (expectedCount !== undefined) {
        expect(await readUint32(offsets, 2)).toEqual([0, expectedCount]);
        const values = await readFloat32(scores, nodeCount);
        expect(values.every(Number.isFinite)).toBe(true);
        expect(values).toEqual(expectedCount === 2 ? [1, 1, 0, 0, 0] : [1, 1, 1, 1, 1]);
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  }
});
