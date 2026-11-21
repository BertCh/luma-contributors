// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createPoisonedOracleGraph,
  runPoisonedOracleGraph
} from '../spatial-clustering/output-oracle-test-utils';

it('nearest compact status distinguishes final capacity from candidate loss and clears on radius recovery', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Distances to the single feature are exactly 0 and 2. Radius 3 retains both;
  // radius 0 retains only row 0. Stable compaction emits query source ID 71 first.
  for (const candidateCapacity of [1, 2]) {
    for (const optionalPort of ['overflow', 'requiredCount', 'candidateOverflow', 'absent']) {
      const compactPresent = optionalPort !== 'absent';
      const harness = createPoisonedOracleGraph(
        device,
        {
          queries: {format: 'float32x2', length: 2, values: Float32Array.of(0, 0, 2, 0)},
          queryIds: {format: 'uint32', length: 2, values: Uint32Array.of(71, 93)},
          features: {format: 'float32x2', length: 1, values: Float32Array.of(0, 0)},
          radius: {format: 'float32', length: 1, values: Float32Array.of(3)}
        },
        {
          nearestIds: {format: 'uint32', length: 2},
          overallOverflow: {format: 'uint32', length: 1},
          ...(compactPresent
            ? {
                ids: {format: 'uint32' as const, length: 1},
                count: {format: 'uint32' as const, length: 1},
                overflow: {format: 'uint32' as const, length: 1}
              }
            : {}),
          ...(optionalPort === 'requiredCount'
            ? {requiredCount: {format: 'uint32' as const, length: 1}}
            : {}),
          ...(optionalPort === 'candidateOverflow'
            ? {candidateOverflow: {format: 'uint32' as const, length: 1}}
            : {})
        }
      );
      harness.graph.add(
        new GPUNearestFeatureJoin({
          queries: {
            kind: 'points',
            positions: harness.getView('queries'),
            sourceIds: harness.getView('queryIds')
          },
          features: {kind: 'points', positions: harness.getView('features')},
          radius: harness.getView('radius'),
          candidateCapacity,
          output: {
            nearestFeatureIds: harness.getView<'uint32'>('nearestIds'),
            status: {overflow: harness.getView('overallOverflow')},
            ...(compactPresent
              ? {
                  matches: {
                    ids: harness.getView<'uint32'>('ids'),
                    status: {
                      count: harness.getView<'uint32'>('count'),
                      overflow: harness.getView<'uint32'>('overflow'),
                      ...(optionalPort === 'requiredCount'
                        ? {requiredCount: harness.getView<'uint32'>('requiredCount')}
                        : {}),
                      ...(optionalPort === 'candidateOverflow'
                        ? {candidateOverflow: harness.getView<'uint32'>('candidateOverflow')}
                        : {})
                    }
                  }
                }
              : {})
          }
        })
      );
      const results = await runPoisonedOracleGraph(harness, device, submission =>
        harness.buffers['radius'].write(Float32Array.of(submission === 0 ? 3 : 0))
      );
      for (const [submission, result] of results.entries()) {
        const matched = submission === 0 && candidateCapacity === 2 ? 2 : 1;
        expect(result['nearestIds']).toEqual([0, matched === 2 ? 0 : 0xffffffff]);
        expect(result['overallOverflow']).toEqual([
          submission === 0 && (candidateCapacity === 1 || compactPresent) ? 1 : 0
        ]);
        if (compactPresent) {
          expect(result['ids']).toEqual([71]);
          expect(result['count']).toEqual([1]);
          expect(result['overflow']).toEqual([matched === 2 ? 1 : 0]);
        }
        if (optionalPort === 'requiredCount') expect(result['requiredCount']).toEqual([matched]);
        if (optionalPort === 'candidateOverflow')
          expect(result['candidateOverflow']).toEqual([
            submission === 0 && candidateCapacity === 1 ? 1 : 0
          ]);
      }
    }
  }
});
