// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {
  GPUContiguityWeightsOperation,
  computePolygonContiguityOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-weights/index';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {makeExactPolygonTopology} from './polygon-contiguity-preparation-fixture';

test('GPU adjacency uses exact source equality tokens, including final-capacity overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const topology = makeExactPolygonTopology();
  for (const criterion of ['queen', 'rook'] as const) {
    for (const capacity of [1, 2]) {
      const {status, ...weights} = computePolygonContiguityOnCPU(topology, {criterion, capacity});
      const expected = {...weights, ...status};
      const operation = new GPUContiguityWeightsOperation({
        criterion,
        capacity,
        featureCount: topology.rowCount,
        pairCapacity: 64,
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
      const actual = await runOperationGraph(device, {
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
      for (const name of [
        'offsets',
        'neighbors',
        'weights',
        'overflow',
        'requiredCount',
        'candidateOverflow'
      ] as const) {
        // All slots are used for these capacities, so every output word is defined.
        expect(Array.from(actual[name]), `${criterion}/${capacity}/${name}`).toEqual(
          Array.from(expected[name])
        );
      }
      operation.destroy();
    }
  }
});
