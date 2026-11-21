// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPULocalMoran,
  getGPUSpatialAutocorrelationParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {GPUZonalStatistics} from '../../../src/gpu-spatial-analysis/zonal-statistics/index';
import {
  GPULocalPermutationTest,
  getGPUPermutationParameterValues
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {GPUMultipleTestingAdjustment} from '../../../src/gpu-spatial-analysis/multiple-testing/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

it('CMP-07 means to Moran and permutation adjustment retains membership loss and an empty tested family', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  try {
    const assignments = rig.input(Uint32Array.of(0, 1, 2, 3, 0xffffffff), 'uint32', 5);
    const candidateOverflow = rig.input(Uint32Array.of(1), 'uint32', 1);
    const valuesComplete = rig.input(Uint32Array.of(1), 'uint32', 1);
    const membershipComplete = rig.input(Uint32Array.of(1), 'uint32', 1);
    const membership = rig.input(Uint32Array.of(1, 1, 1, 1), 'uint32', 4);
    const means = rig.output('float32', 4);
    const membershipLoss = rig.output('uint32', 1);
    rig.graph.add(
      new GPUZonalStatistics({
        features: {
          kind: 'feature-rows',
          featureCount: 4,
          pointFeatureRows: assignments,
          status: {candidateOverflow}
        },
        values: rig.input(Float32Array.of(1, 3, 2, 4, 100), 'float32', 5),
        output: {means: means.view, status: {candidateOverflow: membershipLoss.view}}
      })
    );
    const common = GPULocalMoran.createCommonMaskNodes(rig.graph, {
      rowCount: 4,
      valuesComplete,
      mask: membership,
      membershipStatus: {complete: membershipComplete, candidateOverflow: membershipLoss.view}
    });
    for (const node of common.nodes) rig.graph.add(node);
    const weights = rig.uploadWeights({
      offsets: [0, 1, 3, 5, 6],
      neighbors: [1, 0, 2, 1, 3, 2],
      weights: [1, 1, 1, 1, 1, 1],
      distances: []
    });
    const localI = rig.output('float32', 4);
    const analyticP = rig.output('float32', 4);
    rig.graph.add(
      new GPULocalMoran({
        weights,
        values: means.view,
        mask: common.mask,
        parameters: rig.input(
          getGPUSpatialAutocorrelationParameterValues({significanceLevel: 0.05}),
          'float32',
          8
        ),
        output: {
          localI: localI.view,
          zScores: rig.output('float32', 4).view,
          pValues: analyticP.view
        }
      })
    );
    const rawP = rig.output('float32', 4);
    const observed = rig.output('float32', 4);
    rig.graph.add(
      new GPULocalPermutationTest({
        weights,
        values: means.view,
        mask: common.mask,
        statistic: 'localMoran',
        maximumPermutations: 19,
        parameters: rig.input(
          getGPUPermutationParameterValues({seed: 7, permutationCount: 19}),
          'uint32',
          4
        ),
        output: {
          observed: observed.view,
          pseudoPValues: rawP.view,
          exceedances: rig.output('uint32', 4).view,
          status: {overflow: rig.output('uint32', 1).view}
        }
      })
    );
    const adjusted = rig.output('float32', 4);
    const tested = rig.output('uint32', 1);
    rig.graph.add(
      new GPUMultipleTestingAdjustment({
        pValues: rawP.view,
        method: 'benjamini-yekutieli',
        output: {adjustedPValues: adjusted.view, effectiveFamilySize: tested.view}
      })
    );
    const compiled = rig.graph.compile();
    try {
      // First lost membership, then clean recovery, scalar-only loss, and genuinely empty membership.
      for (const phase of [
        'lost',
        'clean',
        'membership-incomplete',
        'scalar-loss',
        'empty'
      ] as const) {
        if (phase === 'clean') {
          rig.bufferOf(assignments).write(Uint32Array.of(0, 1, 2, 3, 3));
          rig.bufferOf(candidateOverflow).write(Uint32Array.of(0));
          rig.bufferOf(valuesComplete).write(Uint32Array.of(1));
        }
        if (phase === 'membership-incomplete')
          rig.bufferOf(membershipComplete).write(Uint32Array.of(0));
        if (phase === 'scalar-loss') {
          rig.bufferOf(membershipComplete).write(Uint32Array.of(1));
          rig.bufferOf(valuesComplete).write(Uint32Array.of(0));
        }
        if (phase === 'empty') {
          rig.bufferOf(valuesComplete).write(Uint32Array.of(1));
          rig.bufferOf(membership).write(new Uint32Array(4));
        }
        submitGraph(device, compiled, undefined);
        expect(await readUint32(rig.bufferOf(common.membershipStatus.complete!), 1)).toEqual([
          phase === 'membership-incomplete' ? 0 : 1
        ]);
        expect(await readUint32(membershipLoss.buffer, 1)).toEqual([phase === 'lost' ? 1 : 0]);
        if (phase === 'clean') {
          expect(await readFloat32(means.buffer, 4)).toEqual([1, 3, 2, 52]);
          // n=4, SS=1877, (n-1)*(-13.5)*(-11.5)/1877 = 0.2481353223.
          for (const result of [localI, observed]) {
            const values = await readFloat32(result.buffer, 4);
            expect(values).toHaveLength(4);
            const value = values[0];
            expect(Number.isFinite(value)).toBe(true);
            expect(value).toBeCloseTo(0.2481353223, 5);
          }
          expect((await readUint32(tested.buffer, 1))[0]).toBe(4);
          const adjustedValues = await readFloat32(adjusted.buffer, 4);
          expect(adjustedValues).toHaveLength(4);
          for (const value of adjustedValues) expect(Number.isFinite(value)).toBe(true);
        } else {
          for (const result of [localI, analyticP, observed, rawP, adjusted]) {
            const values = await readFloat32(result.buffer, 4);
            expect(values).toHaveLength(4);
            expect(values.every(Number.isNaN)).toBe(true);
          }
          expect(await readUint32(tested.buffer, 1)).toEqual([0]);
        }
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    rig.destroy();
  }
});
