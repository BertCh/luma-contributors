// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSpatialRegressionDiagnosticsOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-spatial-regression-diagnostics-operation';
import {GPUSpatialRegressionDiagnosticsOperation} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regression-diagnostics-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import fixtures from './spatial-regression-diagnostics-operation-fixtures.json';
import {expectRegressionRowsClose} from './regression-operation-test-utils';

it('G3 operation snapshots keep candidate loss active after caller maps mutate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = fixtures[0];
  for (const Operation of [
    GPUSpatialRegressionDiagnosticsOperation,
    CPUSpatialRegressionDiagnosticsOperation
  ]) {
    const props = {
      rowCount: 30,
      predictorCount: 2,
      weights: {
        offsets: 'offsets',
        neighbors: 'neighbors',
        weights: 'weights',
        status: {candidateOverflow: 'flag' as string | undefined}
      },
      predictors: 'predictors',
      response: 'response',
      residuals: 'residuals',
      output: {tests: 'tests', summary: 'summary', status: {code: 'code'}}
    };
    const operation = new Operation(props);
    delete props.weights.status.candidateOverflow;
    props.output.status.code = 'replacement';
    for (const loss of [1, 0]) {
      const result = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.from(fixture.weights.offsets)},
          neighbors: {format: 'uint32', values: Uint32Array.from(fixture.weights.neighbors)},
          weights: {format: 'float32', values: Float32Array.from(fixture.weights.weights)},
          predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
          response: {format: 'float32', values: Float32Array.from(fixture.response)},
          residuals: {format: 'float32', values: Float32Array.from(fixture.residuals)},
          flag: {format: 'uint32', values: Uint32Array.of(loss)}
        },
        operations: [operation],
        outputs: ['code', 'tests']
      });
      expect(Array.from(result['code'])).toEqual([loss ? 3 : 0]);
      if (loss) {
        for (let test = 0; test < 6; test++) {
          expect(result['tests'][test * 3]).toBeNaN();
          expect(result['tests'][test * 3 + 2]).toBeNaN();
        }
      } else expect(Number.isFinite(result['tests'][0])).toBe(true);
    }
  }
});

for (const scene of [
  'ordinary',
  'masked',
  'restandardized',
  'excluded',
  'zero-residuals',
  'zero-links'
] as const) {
  it(`diagnostic operations agree on ${scene}: relative tolerance 5e-3`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = fixtures[scene === 'masked' ? 2 : scene === 'restandardized' ? 3 : 0];
    const outputs = [];
    for (const Operation of [
      GPUSpatialRegressionDiagnosticsOperation,
      CPUSpatialRegressionDiagnosticsOperation
    ]) {
      outputs.push(
        await runOperationGraph(device, {
          inputs: {
            offsets: {
              format: 'uint32',
              values:
                scene === 'zero-links'
                  ? new Uint32Array(31)
                  : Uint32Array.from(fixture.weights.offsets)
            },
            neighbors: {format: 'uint32', values: Uint32Array.from(fixture.weights.neighbors)},
            weights: {format: 'float32', values: Float32Array.from(fixture.weights.weights)},
            predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
            response: {format: 'float32', values: Float32Array.from(fixture.response)},
            residuals: {
              format: 'float32',
              values:
                scene === 'zero-residuals'
                  ? new Float32Array(30)
                  : Float32Array.from(fixture.residuals)
            },
            mask: {
              format: 'uint32',
              values: scene === 'excluded' ? new Uint32Array(30) : Uint32Array.from(fixture.mask)
            }
          },
          operations: [
            new Operation({
              rowCount: 30,
              predictorCount: 2,
              restandardizeRows: fixture.restandardizeRows,
              weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
              predictors: 'predictors',
              response: 'response',
              residuals: 'residuals',
              mask: 'mask',
              output: {tests: 'tests', summary: 'summary', status: {code: 'code'}}
            })
          ],
          outputs: ['tests', 'summary', 'code']
        })
      );
    }
    expect(Array.from(outputs[0]['code'])).toEqual(Array.from(outputs[1]['code']));
    expectRegressionRowsClose(outputs[0]['tests'], outputs[1]['tests'], 5e-3);
    expectRegressionRowsClose(outputs[0]['summary'], outputs[1]['summary'], 5e-3);
    for (const slot of [0, 12])
      expect(outputs[0]['summary'][slot]).toBe(outputs[1]['summary'][slot]);
    for (let test = 0; test < 6; test++)
      expect(outputs[0]['tests'][test * 3 + 1]).toBe(outputs[1]['tests'][test * 3 + 1]);
  });
}

it('BC2 operation snapshots preserve complete suppression and recovery after caller maps mutate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = fixtures[0];
  for (const Operation of [
    GPUSpatialRegressionDiagnosticsOperation,
    CPUSpatialRegressionDiagnosticsOperation
  ]) {
    const props = {
      rowCount: 30,
      predictorCount: 2,
      weights: {
        offsets: 'offsets',
        neighbors: 'neighbors',
        weights: 'weights',
        status: {complete: 'flag' as string | undefined}
      },
      predictors: 'predictors',
      response: 'response',
      residuals: 'residuals',
      output: {tests: 'tests', summary: 'summary', status: {code: 'code'}}
    };
    const operation = new Operation(props);
    delete props.weights.status.complete;
    props.output.status.code = 'replacement';
    for (const loss of [0, 1, 0]) {
      const result = await runOperationGraph(device, {
        inputs: {
          offsets: {format: 'uint32', values: Uint32Array.from(fixture.weights.offsets)},
          neighbors: {format: 'uint32', values: Uint32Array.from(fixture.weights.neighbors)},
          weights: {format: 'float32', values: Float32Array.from(fixture.weights.weights)},
          predictors: {format: 'float32', values: Float32Array.from(fixture.predictors)},
          response: {format: 'float32', values: Float32Array.from(fixture.response)},
          residuals: {format: 'float32', values: Float32Array.from(fixture.residuals)},
          flag: {format: 'uint32', values: Uint32Array.of(loss)}
        },
        operations: [operation],
        outputs: ['code', 'tests']
      });
      expect(Array.from(result['code'])).toEqual([loss ? 0 : 3]);
      if (!loss) {
        for (let test = 0; test < 6; test++) {
          expect(result['tests'][test * 3]).toBeNaN();
          expect(result['tests'][test * 3 + 2]).toBeNaN();
        }
      } else expect(Number.isFinite(result['tests'][0])).toBe(true);
    }
  }
});
