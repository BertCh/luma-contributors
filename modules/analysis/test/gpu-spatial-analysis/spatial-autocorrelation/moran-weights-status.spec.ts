// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  GPULocalGeary,
  getGPUSpatialAutocorrelationParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {
  createSpatialAutocorrelationHarness,
  type SpatialAutocorrelationReadback
} from './spatial-autocorrelation-harness';

const VALUES = new Float32Array([1, 2, 4, 8]);
const POSITIONS = new Float32Array([0, 0, 1, 0, 2, 0, 3, 0]);
const BOUNDS = [-1, -1, 4, 1] as const;

function expectSuppressed(
  result: SpatialAutocorrelationReadback,
  contributor: 'local-moran' | 'hot-spot'
) {
  expect(result.zScores.every(Number.isNaN)).toBe(true);
  expect(result.pValues.every(Number.isNaN)).toBe(true);
  expect(result.adjustedPValues.every(Number.isNaN)).toBe(true);
  expect(result.neighborCounts).toEqual([0, 0, 0, 0]);
  expect(contributor === 'local-moran' ? result.quadrants : result.bins).toEqual([0, 0, 0, 0]);
}

function expectAvailable(result: SpatialAutocorrelationReadback, undefinedRows: number[] = []) {
  expect(result.status).toEqual({overflow: 0, candidateOverflow: 0});
  for (const values of [result.zScores, result.pValues, result.adjustedPValues]) {
    for (const [row, value] of values.entries()) {
      expect(
        undefinedRows.includes(row) ? Number.isNaN(value) : Number.isFinite(value),
        `row ${row}`
      ).toBe(true);
    }
  }
}

it('neighbor search capacity loss reaches Moran and clears after the radius shrinks', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const capacity of [1, 6, 12]) {
    const harness = createSpatialAutocorrelationHarness(device, {
      contributor: 'local-moran',
      scene: {values: VALUES},
      weights: {
        kind: 'neighbor-search',
        positions: POSITIONS,
        parameters: {bounds: BOUNDS, searchRadius: 1.1},
        capacity
      },
      multipleTesting: 'benjamini-hochberg'
    });
    try {
      for (const radius of [1.1, 2.1, 1.1]) {
        const result = await harness.run({}, {bounds: BOUNDS, searchRadius: radius});
        // The line has 6 directed edges at radius 1.1 and 10 at radius 2.1.
        const requiredCount = radius < 2 ? 6 : 10;
        expect(result.status).toEqual({
          overflow: Number(requiredCount > capacity),
          candidateOverflow: 0
        });
        if (requiredCount > capacity) {
          expectSuppressed(result, 'local-moran');
        } else {
          // At radius 2.1, interior rows see all three other observations: conditional variance
          // is zero, so those two tests are undefined even with complete weights.
          expectAvailable(result, radius > 2 ? [1, 2] : []);
        }
      }
      expect(harness.buildCount).toBe(1);
    } finally {
      harness.destroy();
    }
  }
});

for (const contributor of ['local-moran', 'hot-spot'] as const) {
  for (const multipleTesting of ['none', 'benjamini-hochberg'] as const) {
    it(`${contributor} suppresses candidate-only inference with ${multipleTesting} and clears status on reuse`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createSpatialAutocorrelationHarness(device, {
        contributor,
        scene: {values: VALUES},
        weights: {
          kind: 'csr',
          offsets: new Uint32Array([0, 1, 3, 5, 6]),
          neighbors: new Uint32Array([1, 0, 2, 1, 3, 2]),
          weights: new Float32Array(6).fill(1),
          status: {overflow: 0, candidateOverflow: 0}
        },
        multipleTesting,
        // Also check the ungated quadrant path: incomplete neighborhoods cannot get a class.
        quadrantGating: multipleTesting === 'none' ? 'none' : 'analytic'
      });
      try {
        const complete = await harness.run();
        expectAvailable(complete);
        expect(complete.neighborCounts).toEqual([1, 2, 2, 1]);
        if (contributor === 'local-moran' && multipleTesting === 'none') {
          expect(complete.quadrants.some(quadrant => quadrant !== 0)).toBe(true);
        }
        for (const status of [
          {overflow: 0, candidateOverflow: 1},
          {overflow: 1, candidateOverflow: 0},
          {overflow: 1, candidateOverflow: 1}
        ]) {
          harness.writeWeightsStatus(status);
          const result = await harness.run();
          expect(result.status).toEqual(status);
          expectSuppressed(result, contributor);
          harness.writeWeightsStatus({overflow: 0, candidateOverflow: 0});
          const recovered = await harness.run();
          expectAvailable(recovered);
          expect(recovered.neighborCounts).toEqual([1, 2, 2, 1]);
          expect(recovered.zScoreBits).toEqual(complete.zScoreBits);
        }
        expect(harness.buildCount).toBe(1);
      } finally {
        harness.destroy();
      }
    });
  }
}

it('local Geary suppresses candidate loss for packed and chunked weights without sticky status', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const chunked of [false, true]) {
    const rig = new WeightsRig(device);
    try {
      const weights = rig.uploadWeights({
        offsets: [0, 1, 3, 5, 6],
        neighbors: [1, 0, 2, 1, 3, 2],
        weights: [1, 1, 1, 1, 1, 1],
        distances: []
      });
      const candidateOverflow = rig.input(new Uint32Array([0]), 'uint32', 1);
      const inputWeights = chunked
        ? {rowCount: 4, chunks: [{rowStart: 0, rowCount: 4, weights}], status: {candidateOverflow}}
        : {...weights, status: {candidateOverflow}};
      const statistic = rig.output('float32', 4);
      const status = rig.output('uint32', 1);
      rig.graph.add(
        new GPULocalGeary({
          weights: inputWeights,
          values: rig.input(VALUES, 'float32', 4),
          parameters: rig.input(getGPUSpatialAutocorrelationParameterValues(), 'float32', 8),
          output: {statistic: statistic.view, status: {candidateOverflow: status.view}}
        })
      );
      const compiled = rig.graph.compile();
      try {
        for (const flag of [0, 1, 0]) {
          rig.bufferOf(candidateOverflow).write(new Uint32Array([flag]));
          submitGraph(device, compiled, undefined);
          expect(await readUint32(status.buffer, 1)).toEqual([flag]);
          const result = await readFloat32(statistic.buffer, 4);
          if (flag) {
            expect(result.every(Number.isNaN)).toBe(true);
          } else {
            // Population variance is 115 / 16. Edge squared differences sum to [1, 5, 20, 16].
            for (const [row, squaredDifference] of [1, 5, 20, 16].entries()) {
              expect(Number.isFinite(result[row])).toBe(true);
              expect(result[row]).toBeCloseTo((squaredDifference * 16) / 115, 5);
            }
          }
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  }
});
