// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUContiguityWeightsOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  createWeightsTopologyAuditFixture,
  createWeightsCornerContactAuditFixture,
  CORNER_CONTACT_EXPECTED,
  TOPOLOGY_AUDIT_EXPECTED as EXPECTED
} from './weights-topology-audit-fixture';

it('GPU exact-source adjacency matches the hand-derived eight-row graph on every output', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const [shift, scale] of [
    [0, 1],
    [100, 2]
  ]) {
    for (const corner of [false, true]) {
      const topology = corner
        ? createWeightsCornerContactAuditFixture(shift, scale)
        : createWeightsTopologyAuditFixture(shift, scale);
      for (const criterion of ['queen', 'rook'] as const) {
        const expected = corner
          ? CORNER_CONTACT_EXPECTED[criterion]
          : {offsets: EXPECTED.offsets, neighbors: [1, 0, 7, 6], weights: [1, 1, 1, 1]};
        // Every row is either empty or valid, as documented by the preparation.
        const operation = new GPUContiguityWeightsOperation({
          criterion,
          capacity: 6,
          featureCount: topology.rowCount,
          pairCapacity: 128,
          snapTolerance: 0,
          polygons: {
            positions: 'positions',
            ringOffsets: 'ringOffsets',
            polygonOffsets: 'polygonOffsets',
            featureOffsets: 'featureOffsets'
          },
          output: {
            offsets: 'offsets',
            neighbors: 'neighbors',
            weights: 'weights',
            status: {
              overflow: 'overflow',
              requiredCount: 'requiredCount',
              candidateOverflow: 'candidateOverflow'
            }
          }
        });
        try {
          const result = await runOperationGraph(device, {
            inputs: {
              positions: {format: 'float32x2', values: topology.positions},
              ringOffsets: {format: 'uint32', values: topology.ringOffsets},
              polygonOffsets: {format: 'uint32', values: topology.polygonOffsets},
              featureOffsets: {format: 'uint32', values: topology.featureOffsets}
            },
            operations: [operation],
            outputs: [
              'offsets',
              'neighbors',
              'weights',
              'overflow',
              'requiredCount',
              'candidateOverflow'
            ]
          });
          expect(Array.from(result['offsets'])).toEqual(expected.offsets);
          // Only used slots of contiguity output are defined.
          expect(Array.from(result['neighbors']).slice(0, expected.neighbors.length)).toEqual(
            expected.neighbors
          );
          expect(Array.from(result['weights']).slice(0, expected.weights.length)).toEqual(
            expected.weights
          );
          expect(Array.from(result['overflow'])).toEqual([0]);
          expect(Array.from(result['requiredCount'])).toEqual([expected.neighbors.length]);
          expect(Array.from(result['candidateOverflow'])).toEqual([0]);
        } finally {
          operation.destroy();
        }
      }
    }
  }
});
