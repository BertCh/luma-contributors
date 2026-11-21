// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {computeOPTICSOnCPU} from '../../../src/gpu-spatial-analysis/spatial-clustering/cpu-optics-operation';
import {GPUOPTICSOperation} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics-operation';
import {GPU_CONTRIBUTOR_OPERATION_LOWERING} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('isolated unequal pairs have literal OPTICS core distances, reachability and predecessors', () => {
  // Self-inclusive minimumSamples=2 gives the distance to the only other point.
  // Each component starts with infinite reachability and no predecessor; its partner
  // is reached from that first row. The 7-unit gap exceeds maximumDistance=3.
  const result = computeOPTICSOnCPU(Float32Array.of(0, 0, 1, 0, 8, 0, 10, 0), {
    minimumSamples: 2,
    maximumDistance: 3,
    extractionDistance: 3
  });
  expect(Array.from(result.coreDistances)).toEqual([1, 1, 2, 2]);
  expect(Array.from(result.reachability)).toEqual([Infinity, 1, Infinity, 2]);
  expect(Array.from(result.predecessors)).toEqual([0xffffffff, 0, 0xffffffff, 2]);
  expect(Array.from(result.candidateOverflow)).toEqual([0]);
});

it('real OPTICS lowering resolves gridLoss and rejects a missing declaration that a direct getter bypasses', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const operation = new GPUOPTICSOperation({
      positions: 'positions',
      rowCount: 4,
      minimumSamples: 2,
      maximumDistance: 3,
      extractionDistance: 3,
      neighborIndex: 'grid',
      neighborCapacity: 1,
      output: {
        ordering: 'ordering',
        coreDistances: 'coreDistances',
        reachability: 'reachability',
        predecessors: 'predecessors',
        labels: 'labels',
        ports: {status: {candidateOverflow: 'gridLoss'}}
      }
    });
    const views: Record<string, GraphDataView> = {
      positions: createTransientView(graph, 'positions', 'float32x2', 4)
    };
    for (const output of Object.values(operation.outputs))
      views[output.value] = createTransientView(graph, output.value, output.format, output.length);
    const resolved: string[] = [];
    let lowered = false;
    const context = {
      resolveVector(value: string) {
        resolved.push(value);
        return {data: [views[value]]};
      },
      lower() {
        lowered = true;
      }
    };
    type LoweringContext = Parameters<typeof GPU_CONTRIBUTOR_OPERATION_LOWERING.lower>[1];
    GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(operation, context as unknown as LoweringContext);
    expect(lowered).toBe(true);
    expect(resolved).toContain('gridLoss');
    const outputs = {...operation.outputs};
    delete outputs['candidateOverflow'];
    Reflect.set(operation, 'outputs', outputs);
    // Reproduce the regression-test gap: an unconstrained getter still wires the
    // flag without its operation declaration; the production resolver must refuse.
    expect(() =>
      operation.createContributor(
        port => (port === 'candidateOverflow' ? views['gridLoss'] : views[port]) as never
      )
    ).not.toThrow();
    expect(() =>
      GPU_CONTRIBUTOR_OPERATION_LOWERING.lower(operation, context as unknown as LoweringContext)
    ).toThrow('has no port "candidateOverflow"');
  } finally {
    device.destroy();
  }
});
