// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, type GraphBufferHandle} from '@luma.gl/graph/gpu-core';
import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {TEMPORAL_AUDIT_MODELS, createTemporalAuditCase} from './temporal-gwr-audit-fixtures';
import {
  createFinalTemporalCase,
  type FinalTemporalScenario
} from './temporal-regression-w5-fixtures';
import {getGPUGeographicallyTemporallyWeightedRegressionModelResult as readModel} from '../../../src/gpu-spatial-analysis/spatial-regression/geographically-weighted-regression-model-result';

for (const multiscale of [false, true]) {
  it.each(
    TEMPORAL_AUDIT_MODELS
  )(`${multiscale ? 'MGTWR' : 'GTWR'} $kernel/adaptive=$adaptive matches every independent output with ties, masks and changed units`, async model => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const options of [
      {multiscale},
      {multiscale, straightLine: true, indexed: !multiscale && model.kernel === 'bisquare'},
      {multiscale, rescaled: true, indexed: !multiscale && model.kernel === 'bisquare'},
      {multiscale, search: true}
    ]) {
      const graph: GPUCommandGraph = new GPUCommandGraph(device);
      const buffers = new Map<GraphBufferHandle, Buffer>();
      let serial = 0;
      const port = <Format extends 'float32' | 'float32x2' | 'uint32'>(
        format: Format,
        values: number[]
      ) => {
        const buffer = createOutputBuffer(device, values.length);
        buffer.write(format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values));
        const view = importGraphBuffer(
          graph,
          `temporal-oracle-${serial++}`,
          buffer,
          format,
          values.length / (format === 'float32x2' ? 2 : 1)
        );
        buffers.set(view.buffer, buffer);
        return view;
      };
      let compiled: ReturnType<typeof graph.compile> | undefined;
      try {
        const {nodes, outputs, expected} = createTemporalAuditCase(graph, port, model, options);
        nodes.forEach(node => graph.add(node));
        compiled = graph.compile();
        submitGraph(device, compiled, undefined);
        for (const [name, values] of Object.entries(expected)) {
          // R2 targets exact slope/predictions/residuals; a zero RSS has an unbounded likelihood.
          if ('straightLine' in options && ['summary', 'scores'].includes(name)) continue;
          const view = outputs[name];
          const buffer = buffers.get(view.buffer)!;
          const actual =
            view.format === 'uint32'
              ? await readUint32(buffer, view.length)
              : await readFloat32(buffer, view.length);
          expect(actual.length).toBe(values.length);
          values.forEach((value, index) => {
            if (Number.isNaN(value)) expect(actual[index], `${name}[${index}]`).toBeNaN();
            else {
              expect(Number.isFinite(actual[index]), `${name}[${index}]`).toBe(true);
              // Short f32 local reductions and summary statistics; SOC absolute error 1e-4.
              const tolerance = [
                'localStatus',
                'iterationCount',
                'converged',
                'iterationLimitReached',
                'selectedParameters'
              ].includes(name)
                ? 0
                : 1e-4;
              expect(Math.abs(actual[index] - value), `${name}[${index}]`).toBeLessThanOrEqual(
                tolerance
              );
            }
          });
        }
      } finally {
        compiled?.destroy();
        buffers.forEach(buffer => buffer.destroy());
      }
    }
  });
}

for (const route of ['dense', 'indexed', 'search'] as const) {
  // Profile selection requires a finite AICc; genuine exact-fit infinities are checked on fits.
  for (const scenario of (route === 'search'
    ? ['mixed-sign', 'rounded']
    : [
        'mixed-sign',
        'rounded',
        'exact',
        'negligible',
        ...(route === 'dense' ? ['component-outliers', 'orthogonal-rounded'] : [])
      ]) as FinalTemporalScenario[]) {
    it(`W5-48 W5-53 ${route} ${scenario} fit and model readback preserve independent evidence`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const graph = new GPUCommandGraph(device);
      const buffers = new Map<GraphBufferHandle, Buffer>();
      let serial = 0;
      const port = <Format extends 'float32' | 'float32x2' | 'uint32'>(
        format: Format,
        values: number[]
      ) => {
        const buffer = createOutputBuffer(device, values.length);
        buffer.write(format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values));
        const view = importGraphBuffer(
          graph,
          `w5-final-${serial++}`,
          buffer,
          format,
          values.length / (format === 'float32x2' ? 2 : 1)
        );
        buffers.set(view.buffer, buffer);
        return view;
      };
      let compiled: ReturnType<typeof graph.compile> | undefined;
      try {
        const {nodes, outputs} = createFinalTemporalCase(
          graph,
          port,
          scenario,
          route,
          scenario === 'component-outliers' || scenario === 'orthogonal-rounded'
        );
        nodes.forEach(node => graph.add(node));
        compiled = graph.compile();
        submitGraph(device, compiled, undefined);
        const coefficients = await readFloat32(
          buffers.get(outputs.coefficients.buffer)!,
          outputs.coefficients.length
        );
        const fitted = await readFloat32(
          buffers.get(outputs.fitted.buffer)!,
          outputs.fitted.length
        );
        const residuals = await readFloat32(
          buffers.get(outputs.residuals.buffer)!,
          outputs.residuals.length
        );
        const status = await readUint32(
          buffers.get(outputs.localStatus.buffer)!,
          outputs.localStatus.length
        );
        const summary = await readFloat32(buffers.get(outputs.summary.buffer)!, 12);
        expect(Array.from(status)).toEqual(Array(status.length).fill(0));
        if (scenario === 'negligible') {
          for (let row = 0; row < 4; row++) {
            expect(Number.isFinite(coefficients[2 * row])).toBe(true);
            expect(coefficients[2 * row]).toBeCloseTo(2.0000003166, 5);
            expect(coefficients[2 * row + 1]).toBeCloseTo(1, 5);
            expect(Number.isFinite(residuals[row])).toBe(true);
            expect(Math.abs(residuals[row] + 0.0000003166041)).toBeLessThan(2.5e-7);
          }
        } else if (scenario === 'component-outliers') {
          for (let row = 0; row < 8; row++) {
            expect(Number.isFinite(coefficients[2 * row])).toBe(true);
            expect(coefficients[2 * row]).toBeCloseTo(0.75, 6);
            expect(coefficients[2 * row + 1]).toBeCloseTo(0, 6);
          }
        } else if (scenario === 'orthogonal-rounded') {
          const expected = [1, 0, -1, -2, 2, 1, 0, -1].map(value => value / 96);
          for (let row = 0; row < 8; row++) {
            expect(Number.isFinite(coefficients[2 * row + 1])).toBe(true);
            expect(coefficients[2 * row + 1]).toBeCloseTo(1 / 96, 7);
            expect(Number.isFinite(residuals[row])).toBe(true);
            expect(Math.abs(residuals[row] - expected[row])).toBeLessThan(1e-7);
          }
        } else if (scenario === 'mixed-sign') {
          // Category means 1/6 and 1 imply intercept 7/12 and slope 5/12.
          for (const row of [0, 1]) {
            for (const value of [coefficients[2 * row], coefficients[2 * row + 1], fitted[row]])
              expect(Number.isFinite(value)).toBe(true);
            expect(coefficients[2 * row]).toBeCloseTo(7 / 12, 5);
            expect(coefficients[2 * row + 1]).toBeCloseTo(5 / 12, 5);
            expect(fitted[row]).toBeCloseTo(1 / 6, 5);
          }
        } else {
          const model = readModel({summary});
          expect(model.rowCount).toBe(8);
          expect(model.status.code).toBe(0);
          if (scenario === 'exact') {
            expect(Array.from(residuals)).toEqual(Array(8).fill(0));
            expect(model.residualSumOfSquares).toBe(0);
            expect(model.logLikelihood).toBe(Infinity);
            expect(model.correctedAkaikeInformationCriterion).toBe(-Infinity);
          } else {
            // Rational OLS: b1=9/1376, residuals below /1376, RSS=5/11008.
            const expectedResiduals = [11, 2, -7, -16, 16, 7, -2, -11].map(value => value / 1376);
            expectedResiduals.forEach((expected, row) => {
              expect(Number.isFinite(residuals[row])).toBe(true);
              expect(Math.abs(residuals[row] - expected)).toBeLessThan(2e-5);
              expect(fitted[row]).toBe(1e6 + (row >= 4 ? 1 / 16 : 0));
            });
            expect(Number.isFinite(model.residualSumOfSquares)).toBe(true);
            expect(Math.abs(model.residualSumOfSquares - 5 / 11008)).toBeLessThan(2e-7);
            expect(Number.isFinite(model.logLikelihood)).toBe(true);
            expect(Math.abs(model.logLikelihood! - 27.75401649203806)).toBeLessThan(2e-3);
            expect(Number.isFinite(model.correctedAkaikeInformationCriterion)).toBe(true);
            expect(
              Math.abs(model.correctedAkaikeInformationCriterion! + 43.50803298407612)
            ).toBeLessThan(4e-3);
          }
        }
      } finally {
        compiled?.destroy();
        buffers.forEach(buffer => buffer.destroy());
      }
    });
  }
}
