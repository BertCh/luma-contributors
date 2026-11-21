// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUOPTICS} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics';
import {GPUOPTICSOperation} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics-operation';
import {computeOPTICSOnCPU} from '../../../src/gpu-spatial-analysis/spatial-clustering/cpu-optics-operation';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

it('carries the grid loss word into named finalization without a candidateOverflow port', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const orderingMethod of ['sequential', 'parallel'] as const) {
      const graph = new GPUCommandGraph(device);
      const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(
        name: string,
        format: Format,
        length: number
      ) => createTransientView(graph, name, format, length);
      const optics = new GPUOPTICS({
        positions: view('positions', 'float32x2', 3),
        minimumSamples: 2,
        maximumDistance: 2,
        extractionDistance: 1,
        neighborIndex: 'grid',
        neighborCapacity: 1,
        orderingMethod,
        output: {
          ordering: view('ordering', 'uint32', 3),
          coreDistances: view('core', 'float32', 3),
          reachability: view('reachability', 'float32', 3),
          predecessors: view('predecessors', 'uint32', 3),
          labels: view('labels', 'uint32', 3),
          ports: {
            status: {
              iterationLimitReached: view('limit', 'uint32', 1),
              reason: view('reason', 'uint32', 1)
            }
          }
        }
      });
      const nodes = optics.getCommandNodes(graph);
      const finalize = nodes.find(node => node.id.endsWith('-iteration-status'))!;
      const source = getGPUCommandNodeKernelSources([finalize])[0].source;
      expect(source).toContain('overflow[overflowOffset]');
      expect(source).toContain('reason[reasonOffset]');
      expect(nodes.length).toBeLessThanOrEqual(optics.estimatedCommandNodeCount);
    }
  } finally {
    device.destroy();
  }
});

it('publishes the optional reason and invalid-row count as separate operation outputs', () => {
  const operation = new GPUOPTICSOperation({
    positions: 'positions',
    rowCount: 3,
    minimumSamples: 2,
    maximumDistance: 2,
    extractionDistance: 1,
    output: {
      ordering: 'ordering',
      coreDistances: 'core',
      reachability: 'reachability',
      predecessors: 'predecessors',
      labels: 'labels',
      ports: {status: {reason: 'reason', invalidCount: 'invalidCount'}}
    }
  });
  expect(operation.outputs['reason']).toMatchObject({value: 'reason', format: 'uint32', length: 1});
  expect(operation.outputs['invalidCount']).toMatchObject({
    value: 'invalidCount',
    format: 'uint32',
    length: 1
  });
  expect(operation.outputs['status']).toBeUndefined();
});

it('reports completion and visit exhaustion in the exhaustive CPU form', () => {
  for (const maximumIterations of [2, 3]) {
    const result = computeOPTICSOnCPU(Float32Array.of(0, 0, 0.5, 0, 10, 0), {
      minimumSamples: 2,
      maximumDistance: 1,
      extractionDistance: 1,
      maximumIterations
    });
    expect(Array.from(result.reason)).toEqual([maximumIterations === 3 ? 0 : 2]);
    expect(Array.from(result.candidateOverflow)).toEqual([0]);
    expect(Array.from(result.labels)).toEqual(
      maximumIterations === 3 ? [0, 0, 0xffffffff] : [0xffffffff, 0xffffffff, 0xffffffff]
    );
  }
});
