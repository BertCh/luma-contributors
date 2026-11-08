// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import type {CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUGeneralizedGeographicallyWeightedRegression,
  GPUSeeminglyUnrelatedRegression
} from '../../../src/gpu-spatial-analysis/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {RecipeTestFixture} from '../recipes/recipe-harness';
import {getBenchmarkSettings, getBenchmarkDistribution} from '../../utils/benchmark-settings';
import {calculateSURBenchmarkReference} from './statistics-performance-reference';
import {calculateSURReferenceWithQR} from './sur-qr-reference';

const environment = (import.meta as unknown as {env?: Record<string, string>}).env;
const enabled = environment?.['VITE_STATISTICS_BENCHMARK'] === 'true';
it.skipIf(!enabled)(
  'statistical fitting and nested bandwidth selection scaling',
  {timeout: 300000},
  async () => {
    const settings = getBenchmarkSettings(environment);
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const queue = (device as unknown as {handle: {queue: {onSubmittedWorkDone(): Promise<void>}}})
      .handle.queue;
    console.log(
      'STATISTICS_DEVICE',
      JSON.stringify({
        info: device.info,
        sourceSnapshot: settings.sourceSnapshot,
        sampling: settings,
        userAgent: navigator.userAgent,
        recordedAt: new Date().toISOString(),
        limits: {
          maxBufferSize: device.limits.maxBufferSize,
          maxStorageBufferBindingSize: device.limits.maxStorageBufferBindingSize,
          maxStorageBuffersPerShaderStage: device.limits.maxStorageBuffersPerShaderStage,
          maxComputeWorkgroupStorageSize: device.limits.maxComputeWorkgroupStorageSize,
          maxComputeInvocationsPerWorkgroup: device.limits.maxComputeInvocationsPerWorkgroup
        },
        maximumStorageBindingBytes: device.limits.maxStorageBufferBindingSize,
        clock:
          'CPU encode + submit + queue completion; readback separate; no claimed GPU timestamp or driver cold start'
      })
    );
    let measuredWorkloads = 0;
    for (const kind of ['sur', 'poisson-bandwidth-selection'] as const)
      for (const rows of kind === 'sur' ? [64, 1024, 16384] : [32, 128, 512]) {
        if (!`${kind}-${rows}`.includes(settings.workloadFilter)) continue;
        measuredWorkloads++;
        const fixture = new RecipeTestFixture(device, `${kind}-${rows}`);
        let compiled: CompiledGPUCommandGraph | undefined;
        try {
          const start = performance.now();
          let validateOutputs: (() => Promise<void>) | undefined;
          let inputBytes = 0;
          let outputBytes = 0;
          const summary = fixture.output('summary', 'float32', kind === 'sur' ? 4 : 3);
          if (kind === 'sur') {
            const predictors = new Float32Array(rows * 2),
              responses = new Float32Array(rows * 2);
            for (let row = 0; row < rows; row++) {
              const predictor = Math.sin(row * 0.71);
              const error = Math.cos(row * 1.31) * 0.1;
              predictors[row] = predictor;
              predictors[rows + row] = Math.cos(row * 0.27);
              responses[row] = 2 + predictor * 0.7 + error;
              responses[rows + row] =
                1 - predictors[rows + row] * 0.3 + error * 0.4 + Math.sin(row * 1.99) * 0.07;
            }
            inputBytes = predictors.byteLength + responses.byteLength;
            const coefficients = summary,
              covariance = fixture.output('covariance', 'float32', 16),
              residualCovariance = fixture.output('residual-covariance', 'float32', 4),
              residuals = fixture.output('residuals', 'float32', rows * 2),
              status = fixture.output('status', 'uint32', 1);
            outputBytes = (4 + 16 + 4 + rows * 2 + 1) * 4;
            validateOutputs = async () => {
              const expected = calculateSURBenchmarkReference(predictors, responses);
              const independent = calculateSURReferenceWithQR(predictors, responses);
              for (const key of [
                'coefficients',
                'covariance',
                'residualCovariance',
                'residuals'
              ] as const) {
                expected[key].forEach((value, index) =>
                  expect(Math.abs(value - independent[key][index])).toBeLessThanOrEqual(
                    1e-8 * Math.max(1, Math.abs(independent[key][index]))
                  )
                );
              }
              expect(await fixture.readUint32(status, 1)).toEqual([0]);
              for (const [name, output, values] of [
                ['coefficients', coefficients, expected.coefficients],
                ['covariance', covariance, expected.covariance],
                ['residualCovariance', residualCovariance, expected.residualCovariance],
                ['residuals', residuals, expected.residuals]
              ] as const) {
                const actual = await fixture.readFloat32(output, values.length);
                for (let index = 0; index < actual.length; index++) {
                  expect(Number.isFinite(actual[index])).toBe(true);
                  expect(
                    Math.abs(actual[index] - values[index]),
                    `${name}[${index}]: actual ${actual[index]}, expected ${values[index]}`
                  ).toBeLessThanOrEqual(1e-4 * Math.max(1, Math.abs(values[index])));
                }
              }
            };
            fixture.graph.add(
              new GPUSeeminglyUnrelatedRegression({
                rowCount: rows,
                equationCount: 2,
                predictorCount: 1,
                predictors: fixture.input('predictors', predictors, 'float32', predictors.length),
                responses: fixture.input('responses', responses, 'float32', responses.length),
                output: {
                  coefficients: coefficients.view,
                  covariance: covariance.view,
                  residualCovariance: residualCovariance.view,
                  residuals: residuals.view,
                  status: status.view
                }
              })
            );
          } else {
            const positions = new Float32Array(rows * 2),
              predictors = new Float32Array(rows),
              response = new Float32Array(rows),
              mask = new Uint32Array(rows);
            for (let row = 0; row < rows; row++) {
              positions[row * 2] = (row % 16) / 16;
              positions[row * 2 + 1] = Math.floor(row / 16) / Math.ceil(rows / 16);
              predictors[row] = Math.sin(row * 0.71);
              response[row] = 1 + ((row * 7) % 9);
              mask[row] = row % 4 === 0 ? 0 : 1;
            }
            inputBytes =
              positions.byteLength +
              predictors.byteLength +
              response.byteLength +
              mask.byteLength +
              8;
            const coefficients = fixture.output('coefficients', 'float32', rows * 2),
              fitted = fixture.output('fitted', 'float32', rows),
              codes = fixture.output('codes', 'uint32', rows),
              iterationCount = fixture.output('iterations', 'uint32', rows),
              bandwidth = fixture.output('bandwidth', 'float32', 1),
              status = fixture.output('status', 'uint32', 1);
            outputBytes = (rows * 5 + 5) * 4;
            fixture.graph.add(
              new GPUGeneralizedGeographicallyWeightedRegression({
                family: 'poisson',
                positions: fixture.input('positions', positions, 'float32x2', rows),
                predictors: fixture.input('predictors', predictors, 'float32', rows),
                predictorCount: 1,
                response: fixture.input('response', response, 'float32', rows),
                mask: fixture.input('mask', mask, 'uint32', rows),
                parameters: fixture.input('parameters', new Float32Array([1, 1e-5]), 'float32', 2),
                predictExcluded: true,
                maximumIterations: 32,
                bandwidthSelection: {
                  bandwidths: [0.25, 0.5, 1],
                  output: {scores: summary.view, bandwidth: bandwidth.view, status: status.view}
                },
                output: {
                  coefficients: coefficients.view,
                  fitted: fitted.view,
                  status: {codes: codes.view, iterationCount: iterationCount.view}
                }
              })
            );
          }
          await queue.onSubmittedWorkDone();
          const preparationMilliseconds = performance.now() - start;
          const compileStart = performance.now();
          compiled = fixture.graph.compile();
          const compileMilliseconds = performance.now() - compileStart;
          const submit = async () => {
            const start = performance.now();
            const encoder = device.createCommandEncoder();
            compiled!.encode(encoder, {parameters: undefined});
            device.submit(encoder.finish());
            await queue.onSubmittedWorkDone();
            return performance.now() - start;
          };
          const firstExecutionMilliseconds = await submit();
          for (let warmup = 0; warmup < settings.warmupCount; warmup++) await submit();
          const residentMilliseconds = [];
          for (let repeat = 0; repeat < settings.sampleCount; repeat++)
            residentMilliseconds.push(await submit());
          const readStart = performance.now();
          const certificate = await fixture.readFloat32(summary, summary.view.length);
          const readbackMilliseconds = performance.now() - readStart;
          const validationStart = performance.now();
          let validationError: unknown;
          try {
            expect(certificate.every(Number.isFinite)).toBe(true);
            if (kind === 'sur') {
              expect(certificate[0]).toBeCloseTo(2, 1);
              expect(certificate[1]).toBeCloseTo(0.7, 1);
            } else {
              expect(certificate.every(value => value >= 0)).toBe(true);
            }
            await validateOutputs?.();
          } catch (error) {
            validationError = error;
          }
          const validationMilliseconds = performance.now() - validationStart;
          console.log(
            'STATISTICS_MEASUREMENT',
            JSON.stringify({
              kind,
              rows,
              preparationMilliseconds,
              compileMilliseconds,
              firstExecutionMilliseconds,
              residentMilliseconds,
              residentDistribution: getBenchmarkDistribution(residentMilliseconds),
              validationMilliseconds,
              correctnessStatus: validationError ? 'failed' : 'passed',
              validationError: validationError instanceof Error ? validationError.message : null,
              validationScope:
                kind === 'sur'
                  ? 'all outputs against f64 block GLS cross-checked with independent whitened QR'
                  : 'finite nonnegative bandwidth scores',
              readbackScope:
                'coefficient summary or bandwidth scores only; full validation readback excluded',
              readbackMilliseconds,
              inputBytes,
              outputBytes,
              transientBytes: compiled.stats.physicalTransientResourceBytes,
              logicalTransientBytes: compiled.stats.logicalTransientBytes,
              nodeCount: compiled.stats.nodeOrder.length,
              certificate
            })
          );
          if (validationError) throw validationError;
        } finally {
          compiled?.destroy();
          fixture.destroy();
        }
      }
    if (!measuredWorkloads)
      throw new Error(`No statistics workload matched ${settings.workloadFilter}`);
  }
);
