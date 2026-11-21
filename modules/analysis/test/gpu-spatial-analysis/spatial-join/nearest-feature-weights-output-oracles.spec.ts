// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNearestFeatureWeights} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-weights';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createPoisonedOracleGraph,
  runPoisonedOracleGraph
} from '../spatial-clustering/output-oracle-test-utils';

it('nearest adapter publishes only its defined flags and preserves literal CSR with candidate flag omitted', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Hand oracle: rows retain [3,1] and [9], then sort to [1,3],[9]. Binary
  // weights are one. Capacity two clips the last row; capacity three retains all.
  for (const capacity of [2, 3]) {
    for (const candidatePresent of [true, false]) {
      const harness = createPoisonedOracleGraph(
        device,
        {
          neighborRows: {format: 'uint32', length: 4, values: Uint32Array.of(3, 1, 9, 0xffffffff)},
          neighborCounts: {format: 'uint32', length: 2, values: Uint32Array.of(2, 1)},
          ...(candidatePresent
            ? {upstream: {format: 'uint32' as const, length: 1, values: Uint32Array.of(1)}}
            : {})
        },
        {
          offsets: {format: 'uint32', length: 3},
          neighbors: {format: 'uint32', length: capacity},
          weights: {format: 'float32', length: capacity},
          overflow: {format: 'uint32', length: 1},
          ...(candidatePresent ? {candidateOverflow: {format: 'uint32' as const, length: 1}} : {})
        }
      );
      const view = harness.getView;
      harness.graph.add(
        new GPUNearestFeatureWeights({
          slotCapacity: 2,
          neighborRows: view('neighborRows'),
          neighborCounts: view('neighborCounts'),
          ...(candidatePresent ? {neighborStatus: {overflow: view<'uint32'>('upstream')}} : {}),
          output: {
            offsets: view('offsets'),
            neighbors: view('neighbors'),
            weights: view('weights'),
            status: {
              overflow: view('overflow'),
              ...(candidatePresent ? {candidateOverflow: view<'uint32'>('candidateOverflow')} : {})
            }
          }
        })
      );
      const results = await runPoisonedOracleGraph(harness, device, submission => {
        if (candidatePresent)
          harness.buffers['upstream'].write(Uint32Array.of(submission === 0 ? 1 : 0));
      });
      for (const [submission, result] of results.entries()) {
        expect(result['offsets']).toEqual(capacity === 2 ? [0, 2, 2] : [0, 2, 3]);
        expect(result['neighbors']).toEqual(capacity === 2 ? [1, 3] : [1, 3, 9]);
        expect(result['weights']).toEqual(capacity === 2 ? [1, 1] : [1, 1, 1]);
        expect(result['overflow']).toEqual([capacity === 2 ? 1 : 0]);
        if (candidatePresent)
          expect(result['candidateOverflow']).toEqual([submission === 0 ? 1 : 0]);
      }
    }
  }
});
