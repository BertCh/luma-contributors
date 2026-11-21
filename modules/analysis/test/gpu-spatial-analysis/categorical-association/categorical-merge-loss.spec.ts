// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {GPUColocationQuotient} from '../../../src/gpu-spatial-analysis/categorical-association/gpu-colocation-quotient';
import {GPULocalCategoryDependence} from '../../../src/gpu-spatial-analysis/categorical-association/gpu-local-category-dependence';
import {createInputBuffer, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {
  MERGE_GLOBAL_QUOTIENTS,
  MERGE_LOCAL_QUOTIENTS
} from './categorical-merge-fixture.test-utils';

it.each([
  'colocation',
  'dependence'
] as const)('AMG merge %s replays complete lost complete with preserved tails', async kind => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  const overflowBuffer = createInputBuffer(device, Uint32Array.of(0));
  const candidateBuffer = createInputBuffer(device, Uint32Array.of(0));
  const completeBuffer = createInputBuffer(device, Uint32Array.of(1));
  const categoryBuffer = createInputBuffer(device, Uint32Array.of(0, 0, 1));
  let compiled: ReturnType<typeof rig.graph.compile> | undefined;
  try {
    const weights = {
      ...rig.weights({offsets: [0, 1, 2, 3], neighbors: [1, 2, 0], weights: [1, 1, 1]}),
      status: {
        overflow: importGraphBuffer(rig.graph, 'overflow', overflowBuffer, 'uint32', 1),
        candidateOverflow: importGraphBuffer(rig.graph, 'candidate', candidateBuffer, 'uint32', 1),
        complete: importGraphBuffer(rig.graph, 'complete', completeBuffer, 'uint32', 1),
        requiredCount: rig.input(Uint32Array.of(3), 'uint32')
      }
    };
    const categories = importGraphBuffer(rig.graph, 'categories', categoryBuffer, 'uint32', 3);
    const categoryRowCounts = rig.output('uint32', 4);
    // Hand fixture: one neighbor per row. Chance shares are (N_b - [a=b])/2,
    // N=[2,1]. LICD uses n=1 and p=2/3,2/3,1/3; its upper tails are p,1,1.
    const expectations =
      kind === 'colocation'
        ? [MERGE_GLOBAL_QUOTIENTS, MERGE_LOCAL_QUOTIENTS]
        : [
            [
              1,
              1,
              1,
              2 / 3,
              2 / 3,
              0.7071067811865476,
              0,
              1,
              0,
              2 / 3,
              1,
              -1.4142135623730951,
              0,
              1,
              0,
              1 / 3,
              1,
              -0.7071067811865476
            ]
          ];
    const results = expectations.map(expected => rig.output('float32', expected.length + 2));
    rig.graph.add(
      kind === 'colocation'
        ? new GPUColocationQuotient({
            weights,
            categories,
            categoryCount: 2,
            output: {
              globalQuotients: results[0],
              localQuotients: results[1],
              categoryRowCounts
            }
          })
        : new GPULocalCategoryDependence({
            weights,
            categories,
            categoryCount: 2,
            output: {
              statistics: results[0],
              categoryRowCounts
            }
          })
    );
    compiled = rig.graph.compile();
    const sentinel = new Float32Array(Uint32Array.of(0x7f7f7f7f).buffer)[0];
    for (const flags of [
      [0, 0, 1],
      [8, 0, 1],
      [0, 0, 1],
      [0, 128, 1],
      [0, 0, 1],
      [8, 128, 1],
      [0, 0, 0],
      [0, 0, 1]
    ]) {
      overflowBuffer.write(Uint32Array.of(flags[0]));
      candidateBuffer.write(Uint32Array.of(flags[1]));
      completeBuffer.write(Uint32Array.of(flags[2]));
      const lost = Boolean(flags[0] || flags[1] || !flags[2]);
      categoryBuffer.write(lost ? Uint32Array.of(1, 0, 1) : Uint32Array.of(0, 0, 1));
      submitGraph(device, compiled, undefined);
      for (let resultIndex = 0; resultIndex < results.length; resultIndex++) {
        const actual = await rig.readFloat(results[resultIndex]);
        const expected = expectations[resultIndex];
        expect(actual.slice(expected.length)).toEqual([sentinel, sentinel]);
        expected.forEach((value, index) => {
          if (lost || Number.isNaN(value)) expect(actual[index]).toBeNaN();
          else {
            expect(Number.isFinite(actual[index])).toBe(true);
            expect(Math.abs(actual[index] - value)).toBeLessThanOrEqual(2e-6);
          }
        });
      }
      expect(await rig.readUint(categoryRowCounts)).toEqual(
        lost ? [1, 2, 0x7f7f7f7f, 0x7f7f7f7f] : [2, 1, 0x7f7f7f7f, 0x7f7f7f7f]
      );
    }
  } finally {
    compiled?.destroy();
    overflowBuffer.destroy();
    candidateBuffer.destroy();
    completeBuffer.destroy();
    categoryBuffer.destroy();
    rig.destroy();
  }
});

it('review tiny colocation weights preserve unit quotients on GPU', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const weight of [2 ** -149, 2 ** -130, 2 ** -126, 2 ** -100]) {
    const rig = new AnalysisRig(device);
    try {
      const globalQuotients = rig.output('float32', 4);
      const localQuotients = rig.output('float32', 4);
      rig.run(
        new GPUColocationQuotient({
          weights: rig.weights({offsets: [0, 1, 2], neighbors: [1, 0], weights: [weight, weight]}),
          categories: rig.input(Uint32Array.of(0, 1), 'uint32'),
          categoryCount: 2,
          output: {globalQuotients, localQuotients}
        })
      );
      // A single row in each category, linked only to the other: chance and share both 1.
      for (const output of [globalQuotients, localQuotients]) {
        const actual = await rig.readFloat(output);
        expect(actual[0]).toBeNaN();
        expect(actual[3]).toBeNaN();
        for (const index of [1, 2]) {
          expect(Number.isFinite(actual[index])).toBe(true);
          expect(actual[index]).toBe(1);
        }
      }
    } finally {
      rig.destroy();
    }
  }
});
