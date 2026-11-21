// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUOPTICSOperation} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-optics-operation';
import {requireTestDevice} from '../../utils/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {createPoisonedOracleGraph, runPoisonedOracleGraph} from './output-oracle-test-utils';

it('OPTICS operation writes literal isolated-pair distances and candidate status', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const neighborIndex of ['brute-force', 'grid'] as const) {
    for (const candidatePresent of neighborIndex === 'grid' ? [true, false] : [false]) {
      const harness = createPoisonedOracleGraph(
        device,
        {
          positions: {
            format: 'float32x2',
            length: 4,
            values: Float32Array.of(0, 0, 1, 0, 8, 0, 10, 0)
          }
        },
        {
          ordering: {format: 'uint32', length: 4},
          coreDistances: {format: 'float32', length: 4},
          reachability: {format: 'float32', length: 4},
          predecessors: {format: 'uint32', length: 4},
          labels: {format: 'uint32', length: 4},
          iterationCount: {format: 'uint32', length: 1},
          converged: {format: 'uint32', length: 1},
          invalidCount: {format: 'uint32', length: 1},
          ...(candidatePresent ? {candidateOverflow: {format: 'uint32' as const, length: 1}} : {})
        }
      );
      const operation = new GPUOPTICSOperation({
        positions: 'positions',
        rowCount: 4,
        minimumSamples: 2,
        maximumDistance: 3,
        extractionDistance: 3,
        neighborIndex,
        ...(neighborIndex === 'grid' ? {neighborCapacity: 16} : {}),
        output: {
          ordering: 'ordering',
          coreDistances: 'coreDistances',
          reachability: 'reachability',
          predecessors: 'predecessors',
          labels: 'labels',
          ports: {
            status: {
              iterationCount: 'iterationCount',
              converged: 'converged',
              invalidCount: 'invalidCount',
              ...(candidatePresent ? {candidateOverflow: 'candidateOverflow'} : {})
            }
          }
        }
      });
      harness.graph.add(operation.createContributor(harness.getView));
      // Exact float32 integers; see the hand derivation in the adjacent node spec.
      for (const result of await runPoisonedOracleGraph(harness, device)) {
        expect(result['ordering']).toEqual([0, 1, 2, 3]);
        expect(result['coreDistances']).toEqual([1, 1, 2, 2]);
        expect(result['reachability']).toEqual([Infinity, 1, Infinity, 2]);
        expect(result['predecessors']).toEqual([0xffffffff, 0, 0xffffffff, 2]);
        expect(result['labels']).toEqual([0, 0, 1, 1]);
        expect(result['iterationCount']).toEqual([4]);
        expect(result['converged']).toEqual([1]);
        expect(result['invalidCount']).toEqual([0]);
        if (candidatePresent) expect(result['candidateOverflow']).toEqual([0]);
      }
    }
  }
});

it('OPTICS graph lowering resolves renamed gridLoss and reports positive candidate loss', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const candidatePresent of [true, false]) {
    // Four directed radius neighbors exceed global capacity one; no valid ordering
    // can start. Renaming the value requires production operation.outputs resolution.
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
        ports: {
          status: {
            iterationCount: 'iterationCount',
            converged: 'converged',
            invalidCount: 'invalidCount',
            ...(candidatePresent ? {candidateOverflow: 'gridLoss'} : {})
          }
        }
      }
    });
    const result = await runOperationGraph(device, {
      inputs: {positions: {format: 'float32x2', values: Float32Array.of(0, 0, 1, 0, 8, 0, 10, 0)}},
      operations: [operation],
      outputs: [
        'iterationCount',
        'converged',
        'invalidCount',
        'ordering',
        'labels',
        ...(candidatePresent ? ['gridLoss'] : [])
      ]
    });
    expect(Array.from(result['iterationCount'])).toEqual([0]);
    expect(Array.from(result['converged'])).toEqual([0]);
    expect(Array.from(result['invalidCount'])).toEqual([0]);
    expect(Array.from(result['ordering'])).toEqual([
      0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff
    ]);
    expect(Array.from(result['labels'])).toEqual([0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff]);
    if (candidatePresent) expect(Array.from(result['gridLoss'])).toEqual([1]);
  }
});
