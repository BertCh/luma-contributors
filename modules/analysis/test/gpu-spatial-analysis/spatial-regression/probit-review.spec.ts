// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUProbitRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-probit-regression-operation';
import {GPUProbitRegressionOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-probit-regression-operation';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {expectRegressionRowsClose} from './regression-operation-test-utils';
import {PROBIT_SPATIAL_LAG_PROPS} from './probit-spatial-lag-test-utils';
import audit from './probit-audit-fixture.json';
import large from './probit-spatial-lag-fixture.json';
import ring from './probit-review-ring-fixture.json';
import boundary from './probit-review-boundary-fixture.json';

for (const scene of [
  'R1',
  'R2',
  'R2shift',
  'R3',
  'R3shift3000',
  'R3shift4000',
  'R5complete',
  'R5loss',
  'R5capacity',
  'complete-certificate',
  'incomplete-certificate',
  'complete-aggregated-certificate',
  'incomplete-aggregated-certificate',
  'required-csr-span',
  'R6',
  'R8empty',
  'R8scalar'
] as const) {
  it(`reviewed spatial probit counterexample ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const isRing = scene.startsWith('R3') || scene === 'R8empty';
    const source = isRing ? ring : scene.startsWith('R2') || scene === 'R6' ? audit : large;
    const count = scene === 'R1' ? 8 : source.response.length;
    const columns = scene === 'R1' || isRing ? 1 : 2;
    const origin =
      scene === 'R2shift' ? 5 : scene === 'R3shift3000' ? 3000 : scene === 'R3shift4000' ? 4000 : 0;
    const status = scene.endsWith('-certificate')
      ? {
          complete: 'complete',
          ...(scene.includes('aggregated') ? {requiredCount: 'accessible'} : {})
        }
      : scene.startsWith('R5')
        ? {count: 'written', requiredCount: 'required'}
        : scene === 'R8scalar'
          ? {candidateOverflow: 'flag'}
          : scene === 'required-csr-span'
            ? {requiredCount: 'required'}
            : undefined;
    for (const Operation of [GPUProbitRegressionOperation, CPUProbitRegressionOperation]) {
      const execute = () =>
        runOperationGraph(device, {
          inputs: {
            predictors: {
              format: 'float32',
              values:
                scene === 'R1'
                  ? Float32Array.of(0, 1, 3, 8, 0, 1, 3, 8)
                  : Float32Array.from(
                      source.predictors,
                      (value, index) => value + (index % columns === 0 ? origin : 0)
                    )
            },
            response: {
              format: 'float32',
              values:
                scene === 'R1'
                  ? Float32Array.of(0, 0, 0, 0, 1, 1, 1, 1)
                  : Float32Array.from(source.response)
            },
            mask: {
              format: 'uint32',
              values: scene === 'R1' ? new Uint32Array(8).fill(1) : Uint32Array.from(source.mask)
            },
            offsets: {
              format: 'uint32',
              values:
                scene === 'R1'
                  ? Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8)
                  : scene === 'R8empty'
                    ? new Uint32Array(count + 1)
                    : Uint32Array.from(source.offsets)
            },
            neighbors: {
              format: 'uint32',
              values:
                scene === 'R1'
                  ? Uint32Array.of(1, 2, 3, 4, 5, 6, 7, 0)
                  : scene === 'R8empty'
                    ? new Uint32Array()
                    : Uint32Array.from(
                        scene === 'required-csr-span' ? [...source.neighbors, 0] : source.neighbors
                      )
            },
            weights: {
              format: 'float32',
              values:
                scene === 'R1'
                  ? new Float32Array(8).fill(1)
                  : scene === 'R8empty'
                    ? new Float32Array()
                    : Float32Array.from(
                        scene === 'required-csr-span' ? [...source.weights, 0] : source.weights,
                        (value, slot) => (scene === 'R6' && slot === source.offsets[4] ? -1 : value)
                      )
            },
            written: {format: 'uint32', values: Uint32Array.of(118)},
            required: {
              format: 'uint32',
              values: Uint32Array.of(scene === 'R5complete' ? 118 : 119)
            },
            complete: {
              format: 'uint32',
              values: Uint32Array.of(scene.startsWith('incomplete-') ? 0 : 1)
            },
            accessible: {format: 'uint32', values: Uint32Array.of(118)},
            flag: {format: 'uint32', values: Uint32Array.of(0, 1)}
          },
          operations: [
            new Operation({
              ...PROBIT_SPATIAL_LAG_PROPS,
              rowCount: count,
              predictorCount: columns,
              spatialDiagnostics: {
                ...PROBIT_SPATIAL_LAG_PROPS.spatialDiagnostics!,
                weights: {
                  ...PROBIT_SPATIAL_LAG_PROPS.spatialDiagnostics!.weights,
                  status: scene === 'R5capacity' ? {requiredCount: 'required'} : status
                }
              },
              spatialLag: {
                ...PROBIT_SPATIAL_LAG_PROPS.spatialLag!,
                weights: {
                  ...PROBIT_SPATIAL_LAG_PROPS.spatialLag!.weights,
                  status: scene === 'R5capacity' ? {requiredCount: 'required'} : status
                }
              }
            })
          ],
          outputs: [
            'status',
            'diagnosticStatus',
            'tests',
            'lagStatus',
            'lagCoefficients',
            'lagStandardErrors'
          ]
        });
      if (scene === 'R8scalar') {
        await expect(execute()).rejects.toThrow(/candidateOverflow/);
        continue;
      }
      const output = await execute();
      expect(output['status'][0]).toBe(0);
      const invalid =
        scene === 'R5loss' ||
        scene === 'R5capacity' ||
        scene.startsWith('incomplete-') ||
        scene === 'required-csr-span' ||
        scene === 'R6';
      const code = scene === 'R1' ? 3 : scene === 'R8empty' ? 2 : invalid ? 5 : 0;
      expect(output['lagStatus'][0]).toBe(code);
      expect(output['diagnosticStatus'][0]).toBe(scene === 'R8empty' ? 3 : invalid ? 2 : 0);
      if (code) {
        expect(output['lagCoefficients'].every(Number.isNaN)).toBe(true);
        expect(output['lagStandardErrors'].every(Number.isNaN)).toBe(true);
        if (invalid || scene === 'R8empty') expect(output['tests'].every(Number.isNaN)).toBe(true);
      } else {
        // Independent SciPy/NumPy full-instrument projection. Float32 ordinary centering at
        // an offset of 4000, sparse products and dense solves allow 0.3% relative error.
        expectRegressionRowsClose(
          output['lagCoefficients'],
          source.expected.coefficients.map((value, index) =>
            index === 0 ? value - origin * source.expected.coefficients[1] : value
          ),
          3e-3
        );
        expectRegressionRowsClose(
          output['lagStandardErrors'].subarray(origin ? 1 : 0),
          source.expected.standardErrors.slice(origin ? 1 : 0),
          3e-3
        );
      }
    }
  });
}

it('R7 named CPU publication classifies float32 boundary and overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const extreme of [false, true]) {
    const data = extreme ? large : boundary;
    const result = await runOperationGraph(device, {
      inputs: {
        predictors: {format: 'float32', values: Float32Array.from(data.predictors)},
        response: {format: 'float32', values: Float32Array.from(data.response)},
        mask: {format: 'uint32', values: Uint32Array.from(data.mask)},
        offsets: {format: 'uint32', values: Uint32Array.from(data.offsets)},
        neighbors: {format: 'uint32', values: Uint32Array.from(data.neighbors)},
        weights: {
          format: 'float32',
          values: Float32Array.from(data.weights, value => value * (extreme ? 2 ** -135 : 1))
        }
      },
      operations: [
        new CPUProbitRegressionOperation({...PROBIT_SPATIAL_LAG_PROPS, tolerance: 1e-9})
      ],
      outputs: ['lagCoefficients', 'lagStandardErrors', 'lagStatus']
    });
    expect(result['lagStatus'][0]).toBe(extreme ? 3 : 4);
    if (extreme) {
      expect(result['lagCoefficients'].every(Number.isNaN)).toBe(true);
      expect(result['lagStandardErrors'].every(Number.isNaN)).toBe(true);
    } else {
      expect(result['lagCoefficients'][3]).toBe(1);
      expectRegressionRowsClose(
        result['lagStandardErrors'],
        boundary.expected.standardErrors,
        3e-6
      );
    }
  }
});
