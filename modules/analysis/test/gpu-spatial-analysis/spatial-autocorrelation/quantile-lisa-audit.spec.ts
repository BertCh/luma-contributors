// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUQuantileLISA} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

it('quantile LISA writes every port against exact tied-class conditional expectations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new WeightsRig(device);
  // Independent derivation and exhaustive combinations: quantile-lisa-audit.oracle.py.
  const values = rig.input(Float32Array.from([...new Array(10).fill(5), NaN]), 'float32', 11);
  const settings = rig.input(Uint32Array.of(4, 4), 'uint32', 2);
  const membership = rig.output('uint32', 11);
  const boundaries = rig.output('float32', 2);
  const validCount = rig.output('uint32', 1);
  const analysisCount = rig.output('uint32', 1);
  const classes = rig.output('uint32', 11);
  const invalidQuantile = rig.output('uint32', 1);
  const exceedances = rig.output('uint32', 11);
  const probabilities = rig.output('float32', 11);
  const observed = rig.output('float32', 11);
  const significant = rig.output('uint32', 11);
  const islandCount = rig.output('uint32', 1);
  const overflow = rig.output('uint32', 1);
  const invalidCount = rig.output('uint32', 1);
  rig.graph.add(
    new GPUQuantileLISA({
      alternative: 'geoda-legacy',
      values,
      mask: rig.input(Uint32Array.from([...new Array(9).fill(1), 0, 1]), 'uint32', 11),
      quantileParameters: settings,
      parameters: rig.input(
        getGPUPermutationParameterValues({seed: 43, permutationCount: 9, significanceLevel: 0.2}),
        'uint32',
        4
      ),
      maximumPermutations: 9,
      maximumNeighbors: 2,
      weights: rig.uploadWeights({
        offsets: [0, 2, 4, 6, 8, 10, 12, 14, 16, 16, 16, 16],
        neighbors: [1, 7, 0, 2, 1, 3, 2, 4, 3, 5, 4, 6, 5, 7, 0, 6],
        weights: new Array(16).fill(1),
        distances: []
      }),
      output: {
        membership: membership.view,
        boundaries: boundaries.view,
        validCount: validCount.view,
        analysisCount: analysisCount.view,
        classes: classes.view,
        invalidQuantile: invalidQuantile.view,
        exceedances: exceedances.view,
        pseudoPValues: probabilities.view,
        observed: observed.view,
        significant: significant.view,
        islandCount: islandCount.view,
        status: {overflow: overflow.view, invalidCount: invalidCount.view}
      }
    })
  );
  const compiled = rig.graph.compile();
  try {
    for (const value of [5, 42]) {
      rig.bufferOf(values).write(Float32Array.from([...new Array(10).fill(value), NaN]));
      for (const selected of [4, 1]) {
        rig.bufferOf(settings).write(Uint32Array.of(4, selected));
        submitGraph(device, compiled, undefined);
        const joined = selected === 4;
        expect(await readUint32(membership.buffer, 11)).toEqual([
          ...new Array(9).fill(Number(joined)),
          0,
          0
        ]);
        expect(await readFloat32(boundaries.buffer, 2)).toEqual(
          joined ? [value, Infinity] : [-Infinity, value]
        );
        expect(await readUint32(validCount.buffer, 1)).toEqual([9]);
        expect(await readUint32(analysisCount.buffer, 1)).toEqual([8]);
        expect(await readUint32(islandCount.buffer, 1)).toEqual([1]);
        expect(await readUint32(invalidQuantile.buffer, 1)).toEqual([0]);
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(invalidCount.buffer, 1)).toEqual([0]);
        expect(await readUint32(exceedances.buffer, 11)).toEqual([
          ...new Array(8).fill(joined ? 0 : 0xffffffff),
          0xffffffff,
          0xffffffff,
          0xffffffff
        ]);
        const probabilityValues = await readFloat32(probabilities.buffer, 11);
        const observedValues = await readFloat32(observed.buffer, 11);
        for (let row = 0; row < 11; row++) {
          if (row < 8 && joined) {
            // One division to f32; no random-stream tolerance is needed because every draw ties.
            expect(Number.isFinite(probabilityValues[row])).toBe(true);
            expect(probabilityValues[row]).toBeCloseTo(0.1, 7);
          } else expect(probabilityValues[row]).toBeNaN();
          if (row < 8) expect(observedValues[row]).toBe(joined ? 2 : 0);
          else expect(observedValues[row]).toBeNaN();
        }
        expect(await readUint32(significant.buffer, 11)).toEqual([
          ...new Array(8).fill(Number(joined)),
          0,
          0,
          0
        ]);
        expect(await readUint32(classes.buffer, 11)).toEqual([
          ...new Array(8).fill(Number(joined)),
          3,
          2,
          2
        ]);
      }
    }
    rig.bufferOf(settings).write(Uint32Array.of(1, 1));
    submitGraph(device, compiled, undefined);
    expect(await readUint32(invalidQuantile.buffer, 1)).toEqual([1]);
    expect(await readUint32(analysisCount.buffer, 1)).toEqual([0]);
    expect(await readUint32(validCount.buffer, 1)).toEqual([9]);
    expect(await readUint32(membership.buffer, 11)).toEqual(new Array(11).fill(0));
    expect(await readUint32(classes.buffer, 11)).toEqual(new Array(11).fill(2));
    expect((await readFloat32(probabilities.buffer, 11)).every(Number.isNaN)).toBe(true);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});
