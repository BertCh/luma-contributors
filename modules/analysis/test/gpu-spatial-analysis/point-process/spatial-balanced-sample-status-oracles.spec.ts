// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUSpatialBalancedSample} from '../../../src/gpu-spatial-analysis/point-process/gpu-spatial-balanced-sample';
import {requireTestDevice} from '../../utils/require-test-device';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';

it('balanced sample rewrites every stratum status through invalid-frame recovery', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      positions: {format: 'float32x2', values: Float32Array.of(NaN, 0, 1, 0, 2, 0, 3, 0)},
      frameIds: {format: 'uint32', values: Uint32Array.of(10, 11, 12, 13)},
      strata: {format: 'uint32', values: Uint32Array.of(0, 0, 1, 1)}
    },
    outputs: {
      frameIds: {format: 'uint32', length: 3},
      probabilities: {format: 'float32', length: 3},
      weights: {format: 'float32', length: 3},
      status: {format: 'uint32', length: 4}
    },
    create: ({inputs, outputs}) =>
      new GPUSpatialBalancedSample({
        positions: inputs['positions'] as never,
        frameIds: inputs['frameIds'] as never,
        strata: inputs['strata'] as never,
        allocations: [
          {stratumId: 0, sampleCount: 2},
          {stratumId: 1, sampleCount: 1}
        ],
        bounds: [0, 0, 4, 1],
        seed: 7,
        output: {
          frameIds: outputs['frameIds'] as never,
          inclusionProbabilities: outputs['probabilities'] as never,
          designWeights: outputs['weights'] as never,
          status: outputs['status'] as never
        }
      })
  });
  try {
    // Any invalid frame row invalidates every allocation: no selections, code 1 per stratum.
    expect((await fixture.run())['status']).toEqual([0, 1, 0, 1]);
    fixture.writeInput('positions', Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0));
    // Both allocations now fit their two-row strata. Every formerly nonzero code must clear.
    expect((await fixture.run())['status']).toEqual([2, 0, 1, 0]);
    fixture.writeInput('strata', Uint32Array.of(0, 1, 1, 1));
    // All positions remain finite. The first allocation now has only one eligible row for
    // two slots, while the second has three eligible rows for one slot. Their codes differ.
    expect((await fixture.run())['status']).toEqual([0, 1, 1, 0]);
    fixture.writeInput('strata', Uint32Array.of(0, 0, 1, 1));
    expect((await fixture.run())['status']).toEqual([2, 0, 1, 0]);
    fixture.writeInput('positions', Float32Array.of(NaN, 0, 1, 0, 2, 0, 3, 0));
    expect((await fixture.run())['status']).toEqual([0, 1, 0, 1]);
  } finally {
    fixture.destroy();
  }
});
