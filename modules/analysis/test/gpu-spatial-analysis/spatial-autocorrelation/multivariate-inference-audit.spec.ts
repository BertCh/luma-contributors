// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUMultivariateLocalGearyPermutationTest} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

for (const strategy of ['exact', 'tiled'] as const) {
  it(`${strategy} multivariate inference matches a fully enumerated conditional population on every output`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // Two complete vectors; the island is the sole donor for row 0. Standardized vectors are
    // [-1,-1],[1,1], so mean squared distance=4 for the observed and every one of nine draws.
    // Derivation plus independent CPU-twin stream oracle: multivariate-inference-audit.oracle.py.
    for (const mode of ['equal', 'min', 'legacy', 'family'] as const) {
      const rig = new WeightsRig(device);
      const makeValues = (affine: boolean) =>
        Float32Array.from(
          [0, 2, 2, 6, 0, 2, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN],
          value => (affine ? 2 * value + 32 : value)
        );
      const values = rig.input(makeValues(false), 'float32', 16);
      const loss = rig.input(Uint32Array.of(0), 'uint32', 1);
      const candidateLoss = rig.input(Uint32Array.of(0), 'uint32', 1);
      const base = rig.uploadWeights({
        offsets: [0, 1, 1, 1, 1, 1, 1, 1, 1],
        neighbors: [1],
        weights: [1],
        distances: []
      });
      const weights =
        strategy === 'exact'
          ? {...base, status: {overflow: loss, candidateOverflow: candidateLoss}}
          : {
              rowCount: 8,
              chunks: [
                {
                  rowStart: 0,
                  rowCount: 8,
                  weights: {...base, status: {candidateOverflow: candidateLoss}}
                }
              ],
              status: {overflow: loss}
            };
      const statistic = rig.output('float32', 8),
        pValues = rig.output('float32', 8),
        classes = rig.output('uint32', 8);
      const simulations = strategy === 'exact' ? rig.output('float32', 72) : undefined;
      rig.graph.add(
        new GPUMultivariateLocalGearyPermutationTest({
          strategy,
          values,
          variableCount: 2,
          weights,
          mask: rig.input(Uint32Array.of(1, 1, 0, 1, 1, 1, 1, 1), 'uint32', 8),
          permutationCount: 9,
          significanceLevel: 0.15,
          tail: mode === 'equal' ? 'equal-tail' : 'esda-min-tail',
          correction: mode === 'legacy' ? 'bonferroni' : 'none',
          multipleTesting: mode === 'family' ? 'bonferroni' : 'none',
          rowTileSize: 3,
          permutationTileSize: 4,
          maximumNeighbors: 1,
          output: {
            statistic: statistic.view,
            adjustedPValues: pValues.view,
            classes: classes.view,
            simulations: simulations?.view
          }
        })
      );
      const compiled = rig.graph.compile();
      try {
        for (const affine of [false, true]) {
          rig.bufferOf(values).write(makeValues(affine));
          for (const flags of [
            [0, 0],
            [7, 0],
            [0, 9],
            [0, 0]
          ]) {
            rig.bufferOf(loss).write(Uint32Array.of(flags[0]));
            rig.bufferOf(candidateLoss).write(Uint32Array.of(flags[1]));
            submitGraph(device, compiled, undefined);
            const incomplete = flags.some(Boolean);
            const expectedProbability = mode === 'equal' ? 1 : mode === 'legacy' ? 0.2 : 0.1;
            const actual = await readFloat32(statistic.buffer, 8),
              probabilities = await readFloat32(pValues.buffer, 8);
            for (let row = 0; row < 8; row++) {
              if (row === 0 && !incomplete) {
                expect(Number.isFinite(actual[row])).toBe(true);
                expect(actual[row]).toBeCloseTo(4, 5);
                expect(Number.isFinite(probabilities[row])).toBe(true);
                expect(probabilities[row]).toBeCloseTo(expectedProbability, 6);
              } else {
                expect(actual[row]).toBeNaN();
                expect(probabilities[row]).toBeNaN();
              }
            }
            expect(await readUint32(classes.buffer, 8)).toEqual([
              incomplete ? 0xffffffff : mode === 'equal' || mode === 'legacy' ? 0 : 2,
              ...new Array(7).fill(0xffffffff)
            ]);
            if (simulations) {
              const draws = await readFloat32(simulations.buffer, 72);
              for (let index = 0; index < 72; index++) {
                if (index < 9 && !incomplete) {
                  expect(Number.isFinite(draws[index])).toBe(true);
                  expect(draws[index]).toBeCloseTo(4, 5);
                } else expect(draws[index]).toBeNaN();
              }
            }
          }
        }
        rig
          .bufferOf(values)
          .write(
            Float32Array.from([0, 2, 2, 2, 0, 2, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN])
          );
        submitGraph(device, compiled, undefined);
        expect((await readFloat32(statistic.buffer, 8)).every(Number.isNaN)).toBe(true);
        expect((await readFloat32(pValues.buffer, 8)).every(Number.isNaN)).toBe(true);
        expect(await readUint32(classes.buffer, 8)).toEqual(new Array(8).fill(0xffffffff));
        if (simulations)
          expect((await readFloat32(simulations.buffer, 72)).every(Number.isNaN)).toBe(true);
      } finally {
        compiled.destroy();
        rig.destroy();
      }
    }
  });
}
