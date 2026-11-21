// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUContiguityWeights} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights';
import {GPUContiguityWeightsOperation} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-contiguity-weights-operation';
import {GPUAnalysisContiguityStatistics} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-analysis-contiguity-statistics';
import {requireTestDevice} from '../../utils/require-test-device';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {WeightsRig} from './spatial-weights-harness';
import {
  AUDIT_CONTIGUITY_EXPECTED,
  createAuditContiguityInput
} from './audit-weights-a-contiguity-fixture';

it('audits direct and named queen/rook outputs against literal pairwise-set oracles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Binary weights and integer coordinates/snaps are exact in f32, including the affine copy.
  for (const [scale, shift] of [
    [1, 0],
    [8, 1048576]
  ]) {
    const input = createAuditContiguityInput(scale, shift);
    for (const criterion of ['queen', 'rook'] as const) {
      const expected = AUDIT_CONTIGUITY_EXPECTED[criterion];
      for (const capacity of [5, expected.neighbors.length]) {
        const expectedOffsets = expected.offsets.map(value => Math.min(capacity, value));
        const options = {criterion, capacity, pairCapacity: 128, snapTolerance: scale / 4};
        const rig = new WeightsRig(device);
        const output = rig.weightsOutput(8, capacity);
        const overflow = rig.output('uint32', 1);
        const required = rig.output('uint32', 1);
        const candidate = rig.output('uint32', 1);
        try {
          rig.run(
            new GPUContiguityWeights({
              ...options,
              polygons: rig.polygons(input),
              output: {
                ...output.spatialWeights,
                status: {
                  overflow: overflow.view,
                  requiredCount: required.view,
                  candidateOverflow: candidate.view
                }
              }
            })
          );
          const actual = await output.read();
          expect(actual.offsets).toEqual(expectedOffsets);
          expect(actual.neighbors).toEqual(expected.neighbors.slice(0, capacity));
          expect(actual.weights).toEqual(Array(capacity).fill(1));
          expect(await readUint32(required.buffer, 1)).toEqual([expected.neighbors.length]);
          expect(await readUint32(overflow.buffer, 1)).toEqual([
            Number(capacity < expected.neighbors.length)
          ]);
          expect(await readUint32(candidate.buffer, 1)).toEqual([0]);
        } finally {
          rig.destroy();
        }
        const actual = await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: input.positions},
            ringOffsets: {format: 'uint32', values: input.ringOffsets},
            polygonOffsets: {format: 'uint32', values: input.polygonOffsets},
            featureOffsets: {format: 'uint32', values: input.featureOffsets}
          },
          operations: [
            new GPUContiguityWeightsOperation({
              ...options,
              featureCount: 8,
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
                  requiredCount: 'required',
                  candidateOverflow: 'candidate'
                }
              }
            })
          ],
          outputs: ['offsets', 'neighbors', 'weights', 'overflow', 'required', 'candidate']
        });
        expect(Array.from(actual['offsets'])).toEqual(expectedOffsets);
        expect(Array.from(actual['neighbors'])).toEqual(expected.neighbors.slice(0, capacity));
        expect(Array.from(actual['weights'])).toEqual(Array(capacity).fill(1));
        expect(Array.from(actual['required'])).toEqual([expected.neighbors.length]);
        expect(Array.from(actual['overflow'])).toEqual([
          Number(capacity < expected.neighbors.length)
        ]);
        expect(Array.from(actual['candidate'])).toEqual([0]);
      }
    }
  }
});

it('audits every contiguity-profile port against literal degree/median expectations including capacity refusal', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const criterion of ['queen', 'rook'] as const) {
    const expected = AUDIT_CONTIGUITY_EXPECTED[criterion];
    for (const capacity of [5, expected.neighbors.length]) {
      const rig = new WeightsRig(device);
      const weights = rig.weightsOutput(8, capacity);
      const overflow = rig.output('uint32', 1);
      const required = rig.output('uint32', 1);
      const candidate = rig.output('uint32', 1);
      const cardinality = rig.output('uint32', 8);
      const counts = rig.output('uint32', 6);
      const median = rig.output('float32', 1);
      const validity = rig.output('uint32', 1);
      const truncated = capacity < expected.neighbors.length;
      try {
        rig.run(
          new GPUAnalysisContiguityStatistics({
            criterion,
            pairCapacity: 128,
            polygons: rig.polygons(createAuditContiguityInput()),
            output: {
              weights: {
                ...weights.spatialWeights,
                status: {
                  overflow: overflow.view,
                  requiredCount: required.view,
                  candidateOverflow: candidate.view
                }
              },
              cardinality: cardinality.view,
              counts: counts.view,
              median: median.view,
              validity: validity.view
            }
          })
        );
        const actual = await weights.read();
        expect(actual.offsets).toEqual(expected.offsets.map(value => Math.min(capacity, value)));
        expect(actual.neighbors).toEqual(expected.neighbors.slice(0, capacity));
        expect(actual.weights).toEqual(Array(capacity).fill(1));
        expect(await readUint32(required.buffer, 1)).toEqual([expected.neighbors.length]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([Number(truncated)]);
        expect(await readUint32(candidate.buffer, 1)).toEqual([0]);
        expect(await readUint32(validity.buffer, 1)).toEqual([Number(!truncated)]);
        expect(await readUint32(cardinality.buffer, 8)).toEqual(
          truncated
            ? Array(8).fill(0xffffffff)
            : criterion === 'queen'
              ? [3, 3, 3, 3, 1, 1, 0, 0]
              : [2, 2, 2, 2, 1, 1, 0, 0]
        );
        expect(await readUint32(counts.buffer, 6)).toEqual(
          truncated
            ? Array(6).fill(0xffffffff)
            : criterion === 'queen'
              ? [14, 0, 2, 0, 3, 2]
              : [10, 0, 2, 0, 2, 2]
        );
        const [actualMedian] = await readFloat32(median.buffer, 1);
        if (truncated) expect(actualMedian).toBeNaN();
        else expect(actualMedian).toBe(criterion === 'queen' ? 2 : 1.5);
      } finally {
        rig.destroy();
      }
    }
  }
});
