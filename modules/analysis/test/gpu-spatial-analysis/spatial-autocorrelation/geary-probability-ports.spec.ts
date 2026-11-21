// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUMultivariateLocalGearyPermutationTest} from '../../../src/gpu-spatial-analysis/spatial-autocorrelation/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';

for (const strategy of ['exact', 'tiled'] as const) {
  it(`UST-12 ${strategy} keeps raw probabilities while selecting four tested or five complete rows`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    try {
      const values = rig.input(Float32Array.of(0, 1, 3, 7, 31), 'float32', 5);
      // Row zero's ordered neighbors minimize the weighted distance among 4P3=24
      // draws: nearest receives weight 4, next weight 2, third weight 1. Its inclusive
      // equal-tail probability is near 2/24; the fifth included row is an island.
      const weights = rig.uploadWeights({
        offsets: [0, 3, 6, 9, 12, 12],
        neighbors: [1, 2, 3, 0, 2, 3, 0, 1, 3, 0, 1, 2],
        weights: [4, 2, 1, 4, 2, 1, 4, 2, 1, 4, 2, 1],
        distances: []
      });
      const results = [];
      for (const [index, options] of [
        {},
        {correction: 'bonferroni' as const},
        {multipleTesting: 'bonferroni' as const},
        {correction: 'bonferroni' as const, family: 'tested' as const},
        {multipleTesting: 'bonferroni' as const, family: 'complete' as const}
      ].entries()) {
        const raw = rig.output('float32', 5);
        const adjusted = rig.output('float32', 5);
        rig.graph.add(
          new GPUMultivariateLocalGearyPermutationTest({
            id: `geary-ports-${index}`,
            strategy,
            values,
            weights,
            variableCount: 1,
            permutationCount: 999,
            seed: 7,
            significanceLevel: 0.01,
            ...options,
            output: {
              statistic: rig.output('float32', 5).view,
              pseudoPValues: raw.view,
              adjustedPValues: adjusted.view,
              classes: rig.output('uint32', 5).view
            }
          })
        );
        results.push({
          raw,
          adjusted,
          multiplier: index === 0 ? 1 : index === 1 || index === 4 ? 5 : 4
        });
      }
      const compiled = rig.graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        let referenceRaw: number[] | undefined;
        for (const result of results) {
          const raw = await readFloat32(result.raw.buffer, 5);
          const adjusted = await readFloat32(result.adjusted.buffer, 5);
          for (const values of [raw, adjusted]) expect(values).toHaveLength(5);
          if (referenceRaw) {
            for (let row = 0; row < 4; row++) expect(raw[row]).toBe(referenceRaw[row]);
          } else referenceRaw = raw;
          expect(Number.isFinite(raw[0])).toBe(true);
          expect(raw[0]).toBeGreaterThan(0);
          expect(raw[0]).toBeLessThan(0.2);
          // Independent integer Philox4x32-10 replay: 47 inclusive lower-tail draws of
          // 999; raw p=2*(47+1)/1000=.096, then four/five-row Bonferroni=.384/.480.
          expect(raw[0]).toBeCloseTo(0.096, 6);
          expect(adjusted[0]).toBeCloseTo(
            result.multiplier === 1 ? 0.096 : result.multiplier === 4 ? 0.384 : 0.48,
            6
          );
          for (let row = 0; row < 4; row++) {
            expect(Number.isFinite(adjusted[row])).toBe(true);
            expect(adjusted[row]).toBeCloseTo(Math.min(1, raw[row] * result.multiplier), 6);
          }
          expect(raw[4]).toBeNaN();
          expect(adjusted[4]).toBeNaN();
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  });
}

for (const strategy of ['exact', 'tiled'] as const) {
  it(`N1 ${strategy} suppresses false weights certificates and recovers with spare capacity`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const rig = new WeightsRig(device);
    try {
      const complete = rig.input(Uint32Array.of(0), 'uint32', 1);
      const weights = {
        ...rig.uploadWeights(
          {offsets: [0, 1, 2, 3, 4], neighbors: [1, 2, 3, 0], weights: [1, 1, 1, 1], distances: []},
          8
        ),
        status: {complete}
      };
      const statistic = rig.output('float32', 4);
      const raw = rig.output('float32', 4);
      const adjusted = rig.output('float32', 4);
      rig.graph.add(
        new GPUMultivariateLocalGearyPermutationTest({
          strategy,
          weights,
          values: rig.input(Float32Array.of(0, 1, 3, 7), 'float32', 4),
          variableCount: 1,
          permutationCount: 19,
          multipleTesting: 'bonferroni',
          output: {
            statistic: statistic.view,
            pseudoPValues: raw.view,
            adjustedPValues: adjusted.view,
            classes: rig.output('uint32', 4).view
          }
        })
      );
      const compiled = rig.graph.compile();
      try {
        for (const certificate of [0, 1, 0]) {
          rig.bufferOf(complete).write(Uint32Array.of(certificate));
          submitGraph(device, compiled, undefined);
          for (const output of [statistic, raw, adjusted]) {
            const values = await readFloat32(output.buffer, 4);
            expect(values).toHaveLength(4);
            expect(values.every(certificate ? Number.isFinite : Number.isNaN)).toBe(true);
          }
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      rig.destroy();
    }
  });
}
