// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {it} from 'vitest';
import {GPUSpaceTimeDiggleTest} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-diggle-test';
import {GPUSpaceTimeDiggleTestOperation} from '../../../src/dataframe-analysis/pair-statistics/gpu-space-time-diggle-test-operation';
import {RecipeTestFixture} from '../../gpu-spatial-analysis/recipes/recipe-harness';
import {
  expectLiteralRows,
  poisonOperation
} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  DIGGLE_INPUT,
  DIGGLE_OPTIONS,
  DIGGLE_EXPECTED,
  createDiggleOutputPoison
} from './pair-output-oracles-fixture';
import {GPUCoregionalizationFitOperation} from '../../../src/dataframe-analysis/pair-statistics/gpu-coregionalization-fit-operation';
import {getGPUCoregionalizationFitParameterValues} from '../../../src/dataframe-analysis/pair-statistics/coregionalization-parameters';

it('direct Diggle optional moments and status match independent permutations, individually and absent', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const selected of [
    [],
    ['permutationMean'],
    ['permutationStandardDeviation'],
    ['status']
  ] as const) {
    for (const empty of [false, true]) {
      const fixture = new RecipeTestFixture(device, 'diggle-direct-oracle');
      const output = {
        d: fixture.output('d', 'float32', 4),
        statistics: fixture.output('statistics', 'float32', 6),
        summary: fixture.output('summary', 'float32', 11),
        permutationMean: fixture.output('mean', 'float32', 4),
        permutationStandardDeviation: fixture.output('deviation', 'float32', 4),
        status: fixture.output('status', 'uint32', 1)
      };
      for (const [name, value] of Object.entries(output))
        value.buffer.write(createDiggleOutputPoison(name, value.view.length));
      fixture.graph.add(
        new GPUSpaceTimeDiggleTest({
          ...DIGGLE_OPTIONS,
          gridSize: [2, 2],
          positions: fixture.input('positions', DIGGLE_INPUT.positions, 'float32x2', 4),
          times: fixture.input('times', DIGGLE_INPUT.times, 'float32', 4),
          parameters: fixture.input('parameters', DIGGLE_INPUT.parameters, 'float32', 11),
          randomParameters: fixture.input('random', DIGGLE_INPUT.randomParameters, 'uint32', 2),
          mask: fixture.input('mask', new Uint32Array(4).fill(empty ? 0 : 1), 'uint32', 4),
          output: {
            d: output.d.view,
            statistics: output.statistics.view,
            summary: output.summary.view,
            permutationMean: selected.includes('permutationMean' as never)
              ? output.permutationMean.view
              : undefined,
            permutationStandardDeviation: selected.includes('permutationStandardDeviation' as never)
              ? output.permutationStandardDeviation.view
              : undefined,
            status: selected.includes('status' as never) ? output.status.view : undefined
          }
        })
      );
      try {
        fixture.run();
        for (const name of ['d', 'statistics', 'summary', ...selected] as const) {
          const expected = empty
            ? name === 'status'
              ? [1]
              : name === 'summary'
                ? [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0, 0]
                : DIGGLE_EXPECTED[name].map(() => NaN)
            : DIGGLE_EXPECTED[name];
          const rows =
            name === 'status'
              ? await fixture.readUint32(output[name], 1)
              : await fixture.readFloat32(output[name], expected.length);
          expectLiteralRows(rows, expected, 2e-4);
        }
      } finally {
        fixture.destroy();
      }
    }
  }
});

it('coregionalization contributor through its operation writes literal diagnostics and invalidCount', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const imperfect of [false, true]) {
    const rows = await runOperationGraph(device, {
      inputs: {
        distances: {format: 'float32', values: new Float32Array([1, 2])},
        semivariances: {
          format: 'float32',
          values: new Float32Array(imperfect ? [2, 0, 3, 6, 0, 3] : [2, 0, 3, 2, NaN, 3])
        },
        pairCounts: {format: 'uint32', values: new Uint32Array(imperfect ? [1, 3] : [10, 5])},
        parameters: {
          format: 'float32',
          values: getGPUCoregionalizationFitParameterValues([
            {model: 'spherical', range: 1, sill: 1}
          ])
        }
      },
      operations: [
        poisonOperation(
          new GPUCoregionalizationFitOperation({
            variableCount: 2,
            structureCount: 1,
            maximumIterations: imperfect ? 1 : 4,
            distances: 'distances',
            semivariances: 'semivariances',
            pairCounts: 'pairCounts',
            parameters: 'parameters',
            output: {
              matrices: 'matrices',
              loadings: 'loadings',
              diagnostics: 'diagnostics',
              status: {
                validity: 'validity',
                iterationCount: 'iterationCount',
                converged: 'converged',
                iterationLimitReached: 'iterationLimitReached',
                invalidCount: 'invalidCount'
              }
            }
          })
        )
      ],
      outputs: ['matrices', 'diagnostics', 'invalidCount']
    });
    // One unit shape fits diagonal gamma=[2,3] exactly; NaN bin excluded, leaving weight 10.
    // Two unit shapes with diagonals [2,3]/[6,3] and weights [1,3] fit [5,3] with MSE=3.
    expectLiteralRows(rows['matrices'], imperfect ? [5, 0, 0, 3] : [2, 0, 0, 3]);
    expectLiteralRows(rows['diagnostics'], imperfect ? [3, 1, 2, 4] : [0, 0, 1, 10]);
    expectLiteralRows(rows['invalidCount'], [imperfect ? 0 : 1], 0);
  }
});

it('Diggle operation ports match independent literals with each optional output alone and omitted', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const required = ['d', 'statistics', 'summary'] as const;
  const optional = [
    'permutationMean',
    'permutationStandardDeviation',
    'k',
    'spatialK',
    'temporalK',
    'status'
  ] as const;
  for (const selected of [
    [...required, ...optional],
    ...optional.map(name => [...required, name]),
    ...optional.map(name => [...required, ...optional.filter(other => other !== name)])
  ]) {
    for (const empty of [false, true]) {
      const output = {
        d: 'd',
        statistics: 'statistics',
        summary: 'summary',
        ...Object.fromEntries(selected.map(name => [name, name]))
      };
      const rows = await runOperationGraph(device, {
        inputs: {
          positions: {format: 'float32x2', values: DIGGLE_INPUT.positions},
          times: {format: 'float32', values: DIGGLE_INPUT.times},
          parameters: {format: 'float32', values: DIGGLE_INPUT.parameters},
          random: {format: 'uint32', values: DIGGLE_INPUT.randomParameters},
          mask: {format: 'uint32', values: new Uint32Array(4).fill(empty ? 0 : 1)}
        },
        operations: [
          poisonOperation(
            new GPUSpaceTimeDiggleTestOperation({
              ...DIGGLE_OPTIONS,
              gridSize: [2, 2],
              rowCount: 4,
              positions: 'positions',
              times: 'times',
              mask: 'mask',
              parameters: 'parameters',
              randomParameters: 'random',
              output
            })
          )
        ],
        outputs: selected
      });
      // 2e-4 allows float32 fixed-point K reduction and signed residual cancellation.
      for (const name of selected)
        expectLiteralRows(
          rows[name],
          empty
            ? name === 'status'
              ? [1]
              : name === 'summary'
                ? [NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, 0, 0]
                : DIGGLE_EXPECTED[name].map(() => NaN)
            : DIGGLE_EXPECTED[name],
          2e-4
        );
    }
  }
});
