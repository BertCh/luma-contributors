// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUMultivariateLocalGearyPermutationTest,
  getGPUMultivariateLocalGearyPermutationTestParameterValues,
  type GPUMultivariateLocalGearyPermutationTestSettings
} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {
  computeMultivariateGearyRuntimeOracle,
  MULTIVARIATE_GEARY_RUNTIME_PERMUTATIONS,
  MULTIVARIATE_GEARY_RUNTIME_SEED,
  MULTIVARIATE_GEARY_RUNTIME_VALUES
} from './multivariate-geary-runtime-oracle';

function expectClose(actual: number[], expected: number[]) {
  expect(actual.length).toBe(expected.length);
  for (let index = 0; index < expected.length; index++) {
    if (Number.isNaN(expected[index])) expect(actual[index], `row ${index}`).toBeNaN();
    else
      expect(Math.abs(actual[index] - expected[index]), `row ${index}`).toBeLessThanOrEqual(
        1e-5 * Math.max(1, Math.abs(expected[index]))
      );
  }
}

for (const strategy of ['exact', 'tiled'] as const) {
  for (const mode of ['none', 'bonferroni', 'benjamini-hochberg'] as const) {
    it(`${strategy} ${mode} rereads full-width seeds and levels without recompilation, matching a degree-one oracle`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      for (const chunked of [false, true]) {
        const rig = new WeightsRig(device);
        const parameters = new GPUParameterBuffer(device, {
          id: 'geary-runtime-parameters',
          format: 'uint32',
          length: 5,
          values: new Uint32Array([0xdeadbeef, 0, 0, 0, 0xcafebabe])
        });
        const imported = parameters.importToGraph(rig.graph);
        const parameterView = rig.graph.createDataView(imported.buffer, {
          format: 'uint32',
          length: 3,
          byteOffset: 4
        });
        const values = rig.input(
          new Float32Array([
            ...MULTIVARIATE_GEARY_RUNTIME_VALUES.flatMap(value => [value, 2 * value + 5]),
            NaN,
            NaN,
            NaN,
            9
          ]),
          'float32',
          16
        );
        const mask = rig.input(new Uint32Array([1, 1, 1, 1, 1, 1, 0, 1]), 'uint32', 8);
        const weightOverflow = rig.input(new Uint32Array([0]), 'uint32', 1);
        const whole = rig.uploadWeights(
          {
            offsets: [0, 1, 2, 3, 4, 5, 5, 5, 5],
            neighbors: [1, 2, 3, 4, 0],
            weights: [1, 1, 1, 1, 1],
            distances: []
          },
          4
        );
        const first = rig.uploadWeights(
          {offsets: [0, 1, 2, 3], neighbors: [1, 2, 3], weights: [1, 1, 1], distances: []},
          2
        );
        const second = rig.uploadWeights(
          {offsets: [0, 1, 2, 2, 2, 2], neighbors: [4, 0], weights: [1, 1], distances: []},
          2
        );
        const weights = chunked
          ? {
              rowCount: 8,
              status: {overflow: weightOverflow},
              chunks: [
                {rowStart: 0, rowCount: 3, weights: first},
                {rowStart: 3, rowCount: 5, weights: second}
              ]
            }
          : {...whole, status: {overflow: weightOverflow}};
        const statistic = rig.output('float32', 10);
        const pValues = rig.output('float32', 10);
        const classes = rig.output('uint32', 10);
        const simulations =
          strategy === 'exact'
            ? rig.output('float32', 8 * MULTIVARIATE_GEARY_RUNTIME_PERMUTATIONS)
            : undefined;
        const contributor = new GPUMultivariateLocalGearyPermutationTest({
          weights,
          values,
          mask,
          variableCount: 2,
          permutationCount: MULTIVARIATE_GEARY_RUNTIME_PERMUTATIONS,
          strategy,
          rowTileSize: 3,
          permutationTileSize: 7,
          parameters: parameterView,
          // Deliberately different defaults: the resident parameter view must override both.
          seed: [123, 456],
          significanceLevel: 0.5,
          correction: mode === 'bonferroni' ? 'bonferroni' : 'none',
          multipleTesting: mode === 'benjamini-hochberg' ? 'benjamini-hochberg' : 'none',
          tail: mode === 'bonferroni' ? 'esda-min-tail' : 'equal-tail',
          output: {
            statistic: rig.graph.createDataView(statistic.view.buffer, {
              format: 'float32',
              length: 8,
              byteOffset: 4
            }),
            adjustedPValues: rig.graph.createDataView(pValues.view.buffer, {
              format: 'float32',
              length: 8,
              byteOffset: 4
            }),
            classes: rig.graph.createDataView(classes.view.buffer, {
              format: 'uint32',
              length: 8,
              byteOffset: 4
            }),
            simulations: simulations?.view
          }
        });
        rig.graph.add(contributor);
        const compiled = rig.graph.compile();
        const check = async (
          settings: GPUMultivariateLocalGearyPermutationTestSettings,
          empty = false
        ) => {
          const packed = getGPUMultivariateLocalGearyPermutationTestParameterValues(settings);
          parameters.write(packed, 1);
          submitGraph(device, compiled, undefined);
          const expected = empty
            ? {
                statistic: new Array<number>(8).fill(NaN),
                pValues: new Array<number>(8).fill(NaN),
                classes: new Array<number>(8).fill(0xffffffff),
                simulations: new Array<number>(8 * MULTIVARIATE_GEARY_RUNTIME_PERMUTATIONS).fill(
                  NaN
                )
              }
            : computeMultivariateGearyRuntimeOracle(
                settings.seed as readonly [number, number],
                settings.significanceLevel!,
                mode
              );
          expectClose((await readFloat32(statistic.buffer, 10)).slice(1, 9), expected.statistic);
          expectClose((await readFloat32(pValues.buffer, 10)).slice(1, 9), expected.pValues);
          expect((await readUint32(classes.buffer, 10)).slice(1, 9)).toEqual(expected.classes);
          if (simulations)
            expectClose(
              await readFloat32(simulations.buffer, 8 * MULTIVARIATE_GEARY_RUNTIME_PERMUTATIONS),
              expected.simulations
            );
          for (const buffer of [statistic.buffer, pValues.buffer, classes.buffer]) {
            const words = await readUint32(buffer, 10);
            expect([words[0], words[9]]).toEqual([0x7f7f7f7f, 0x7f7f7f7f]);
          }
          expect(await readUint32(parameters.buffer, 5)).toEqual([
            0xdeadbeef,
            ...packed,
            0xcafebabe
          ]);
        };
        try {
          const firstSettings = {seed: MULTIVARIATE_GEARY_RUNTIME_SEED, significanceLevel: 0.01};
          for (const settings of [
            firstSettings,
            {...firstSettings, significanceLevel: 0.99},
            {seed: [0xffffffff, 0xffffffff] as const, significanceLevel: 0.99},
            {seed: [0x80000000, 1] as const, significanceLevel: 0.99},
            firstSettings
          ])
            await check(settings);
          rig.bufferOf(mask).write(new Uint32Array(8));
          await check({...firstSettings, significanceLevel: 0.99}, true);
          rig.bufferOf(mask).write(new Uint32Array([1, 1, 1, 1, 1, 1, 0, 1]));
          rig.bufferOf(weightOverflow).write(new Uint32Array([1]));
          await check(firstSettings, true);
          rig.bufferOf(weightOverflow).write(new Uint32Array([0]));
          await check({...firstSettings, significanceLevel: 0.99});
        } finally {
          compiled.destroy();
          parameters.destroy();
          rig.destroy();
        }
      }
    });
  }
}
