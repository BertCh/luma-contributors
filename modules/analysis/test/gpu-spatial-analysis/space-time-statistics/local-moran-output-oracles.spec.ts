// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {
  GPUSpaceTimeLocalMoran,
  type GPUSpaceTimeLocalMoranProps
} from '../../../src/gpu-spatial-analysis/space-time-statistics/index';
import {getGPUPermutationParameterValues} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {WeightsRig} from '../spatial-weights/spatial-weights-harness';
import {
  CYCLE_CSR,
  TIED_CSR,
  MORAN_CASES,
  expectMoranOracle
} from './local-moran-output-oracles-fixture';

it('space-time Moran publishes every poisoned optional output alone and omitted against an exhaustive tied null', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const layout of ['period-major', 'cell-major'] as const) {
    for (const optional of [...MORAN_CASES, ['cycle'], ['cycle-half']]) {
      const cycle = optional[0].startsWith('cycle');
      const halfLevel = optional[0] === 'cycle-half';
      const names = halfLevel
        ? ['significant']
        : cycle
          ? [
              'analysisCount',
              'islandCount',
              'invalidCount',
              'adjustedPValues',
              'significant',
              'quadrants',
              'observed',
              'permutationCounts'
            ]
          : optional;
      const rig = new WeightsRig(device);
      try {
        const allocated = Object.fromEntries(
          ['exceedances', 'pseudoPValues', 'overflow', ...names].map(name => {
            const scalar = ['analysisCount', 'islandCount', 'invalidCount', 'overflow'].includes(
              name
            );
            return [
              name,
              rig.output(
                ['pseudoPValues', 'adjustedPValues', 'observed'].includes(name)
                  ? 'float32'
                  : 'uint32',
                scalar ? 1 : 4
              )
            ];
          })
        );
        const views: Record<string, GraphDataView> = Object.fromEntries(
          Object.entries(allocated)
            .filter(([name]) => !['overflow', 'invalidCount'].includes(name))
            .map(([name, output]) => [name, output.view])
        );
        const values = cycle ? [1, 2, 4, NaN] : [1, 3, 9, NaN];
        const cube =
          layout === 'period-major'
            ? [...values, ...values]
            : values.flatMap(value => [value, value]);
        rig.run(
          new GPUSpaceTimeLocalMoran({
            weights: rig.uploadWeights(cycle ? CYCLE_CSR : TIED_CSR),
            cube: rig.input(new Float32Array(cube), 'float32', 8),
            layout,
            periodCount: 2,
            period: 1,
            variant: 'lagged',
            parameters: rig.input(
              getGPUPermutationParameterValues({
                seed: 11,
                permutationCount: 7,
                significanceLevel: halfLevel ? 0.5 : cycle ? 0.75 : 0.05
              }),
              'uint32',
              4
            ),
            maximumPermutations: 7,
            maximumNeighbors: 1,
            alternative: 'greater',
            islands: 'exclude',
            multipleTesting:
              names.includes('significant') || names.includes('adjustedPValues')
                ? 'benjamini-yekutieli'
                : 'none',
            ...(cycle ? {} : {stopAfterExceedances: 3}),
            output: {
              ...views,
              status: {
                overflow: allocated['overflow'].view,
                invalidCount: allocated['invalidCount']?.view
              }
            } as GPUSpaceTimeLocalMoranProps['output']
          })
        );
        const actual: Record<string, number[]> = {};
        for (const [name, output] of Object.entries(allocated))
          actual[name] =
            output.view.format === 'float32'
              ? await readFloat32(output.buffer, output.view.length)
              : await readUint32(output.buffer, output.view.length);
        expectMoranOracle(actual, cycle, halfLevel ? 0.5 : 0.75);
      } finally {
        rig.destroy();
      }
    }
  }
});
