// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUProbitRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-probit-regression-operation';
import {GPUProbitRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import {PROBIT_SPATIAL_LAG_PROPS} from './probit-spatial-lag-test-utils';
import largeFixture from './probit-spatial-lag-fixture.json';
import auditFixture from './probit-audit-fixture.json';

for (const scene of [
  'oracle',
  'audit',
  'budget',
  'zeroWeights',
  'outOfRange',
  'negativeWeights',
  'invalidOrigin',
  'candidateOverflow',
  'notConvergedWeights',
  'excluded',
  'secondStageSingular'
] as const) {
  it(`linearized spatial probit matches independent expectations on every optional port: ${scene}`, async () => {
    const fixture = scene === 'audit' ? auditFixture : largeFixture;
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const Operation of [GPUProbitRegressionOperation, CPUProbitRegressionOperation]) {
      const secondStageSingular = scene === 'secondStageSingular';
      const outputs = [
        'coefficients',
        'standardErrors',
        'fitted',
        'marginalEffects',
        'summary',
        'status',
        'tests',
        'diagnosticStatus',
        'lagCoefficients',
        'lagStandardErrors',
        'lagStatus'
      ];
      const result = await runOperationGraph(device, {
        inputs: {
          predictors: {
            format: 'float32',
            values: secondStageSingular
              ? Float32Array.of(-2, -2, -1, -1, 1, 1, 2, 2)
              : Float32Array.from(fixture.predictors)
          },
          response: {
            format: 'float32',
            values: secondStageSingular
              ? Float32Array.of(0, 1, 0, 1, 0, 1, 0, 1)
              : Float32Array.from(fixture.response)
          },
          mask: {
            format: 'uint32',
            values: secondStageSingular
              ? new Uint32Array(8).fill(1)
              : scene === 'excluded'
                ? new Uint32Array(60)
                : Uint32Array.from(fixture.mask)
          },
          offsets: {
            format: 'uint32',
            values: secondStageSingular
              ? Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8)
              : Uint32Array.from(fixture.offsets, (value, index) =>
                  scene === 'invalidOrigin' && index === 0 ? 1 : value
                )
          },
          neighbors: {
            format: 'uint32',
            values: secondStageSingular
              ? Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 0)
              : Uint32Array.from(fixture.neighbors)
          },
          weights: {
            format: 'float32',
            values: secondStageSingular
              ? new Float32Array(8).fill(1)
              : Float32Array.from(fixture.weights, value =>
                  scene === 'zeroWeights'
                    ? 0
                    : scene === 'outOfRange'
                      ? value / 4
                      : scene === 'negativeWeights'
                        ? -value
                        : value
                )
          },
          weightFlag: {
            format: 'uint32',
            values: Uint32Array.of(scene === 'notConvergedWeights' ? 0 : 1)
          }
        },
        operations: [
          new Operation({
            ...PROBIT_SPATIAL_LAG_PROPS,
            rowCount: secondStageSingular ? 8 : fixture.response.length,
            predictorCount: secondStageSingular ? 1 : 2,
            maximumIterations: scene === 'budget' ? 1 : undefined,
            spatialDiagnostics: {
              ...PROBIT_SPATIAL_LAG_PROPS.spatialDiagnostics!,
              weights: {
                ...PROBIT_SPATIAL_LAG_PROPS.spatialDiagnostics!.weights,
                status:
                  scene === 'candidateOverflow'
                    ? {candidateOverflow: 'weightFlag'}
                    : scene === 'notConvergedWeights'
                      ? {converged: 'weightFlag'}
                      : undefined
              }
            },
            spatialLag: {
              ...PROBIT_SPATIAL_LAG_PROPS.spatialLag!,
              weights: {
                ...PROBIT_SPATIAL_LAG_PROPS.spatialLag!.weights,
                status:
                  scene === 'candidateOverflow'
                    ? {candidateOverflow: 'weightFlag'}
                    : scene === 'notConvergedWeights'
                      ? {converged: 'weightFlag'}
                      : undefined
              }
            }
          })
        ],
        outputs
      });
      const expectedStatus =
        scene === 'budget' || scene === 'excluded'
          ? 1
          : scene === 'zeroWeights'
            ? 2
            : scene === 'outOfRange'
              ? 4
              : scene === 'secondStageSingular'
                ? 3
                : [
                      'negativeWeights',
                      'invalidOrigin',
                      'candidateOverflow',
                      'notConvergedWeights'
                    ].includes(scene)
                  ? 5
                  : 0;
      expect(result['lagStatus'][0]).toBe(expectedStatus);
      if (expectedStatus === 0 || expectedStatus === 4) {
        const factor = scene === 'outOfRange' ? 4 : 1;
        // Float32 sparse sums, erfc, Gram factors and two projections: 0.2% relative,
        // deliberately well clear of zero, missing rho or a projected-residual sandwich.
        expectRegressionRowsClose(
          result['lagCoefficients'],
          fixture.expected.coefficients.map((value, index) => value * (index === 3 ? factor : 1)),
          2e-3
        );
        expectRegressionRowsClose(
          result['lagStandardErrors'],
          fixture.expected.standardErrors.map((value, index) => value * (index === 3 ? factor : 1)),
          2e-3
        );
      } else {
        expect(result['lagCoefficients'].every(Number.isNaN)).toBe(true);
        expect(result['lagStandardErrors'].every(Number.isNaN)).toBe(true);
      }
      if (scene === 'oracle' || scene === 'audit') {
        for (const [name, expected] of Object.entries({
          coefficients: fixture.expected.ordinaryCoefficients,
          standardErrors: fixture.expected.ordinaryStandardErrors,
          fitted: fixture.expected.fitted.map(value => (value === null ? NaN : value)),
          marginalEffects: fixture.expected.marginalEffects,
          tests: fixture.expected.tests
        }))
          expectRegressionRowsClose(result[name], expected, 5e-4);
        expectRegressionRowsClose(result['summary'].subarray(0, 3), fixture.expected.summary, 5e-4);
        expect(result['status'][0]).toBe(0);
        expect(result['diagnosticStatus'][0]).toBe(0);
      }
      if (scene === 'budget') {
        expect(result['status'][0]).toBe(3);
        expect(result['summary'][3]).toBe(1);
      }
      if (scene === 'candidateOverflow' || scene === 'notConvergedWeights') {
        expect(result['diagnosticStatus'][0]).toBe(2);
        expect(result['tests'].every(Number.isNaN)).toBe(true);
      }
      if (scene === 'secondStageSingular') {
        // Closed form: paired opposite responses at every x -> eta=0, p=1/2,
        // information=(2/pi) diag(8,20); directed ring has T=8 and e'We=-2.
        expect(result['status'][0]).toBe(0);
        expectRegressionRowsClose(result['coefficients'], [0, 0], 0);
        expectRegressionRowsClose(
          result['standardErrors'],
          [Math.sqrt(Math.PI / 16), Math.sqrt(Math.PI / 40)],
          5e-5
        );
        expectRegressionRowsClose(result['fitted'], new Array(8).fill(0.5), 0);
        expectRegressionRowsClose(result['marginalEffects'], [0], 0);
        expectRegressionRowsClose(
          result['summary'],
          [8, -8 * Math.log(2), 16 * Math.log(2), 1],
          5e-5
        );
        expect(result['diagnosticStatus'][0]).toBe(0);
        expectRegressionRowsClose(
          result['tests'],
          [
            8,
            0.004677734981047266,
            -2 * Math.sqrt(2),
            0.004677734981047266,
            8,
            0.004677734981047266
          ],
          5e-4
        );
      }
      if (scene === 'excluded') {
        expect(result['status'][0]).toBe(1);
        expect(result['summary'][0]).toBe(0);
      }
    }
  });
}
