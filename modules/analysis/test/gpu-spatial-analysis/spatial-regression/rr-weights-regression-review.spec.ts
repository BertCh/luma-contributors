// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUSpatialWeightsTraces} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-weights-traces';
import {GPUSpatialErrorGM} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-error-gm';
import {GPUSpatialTwoStageLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-two-stage-least-squares';
import {addSpatialRegressionRecipe} from '../../../src/gpu-spatial-analysis/recipes/spatial-regression-recipe';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {REVIEW_REGRESSION_FIXTURE} from './rr-weights-regression-review-fixture';
import {getGPUSpatialTwoStageLeastSquaresDiagnostics} from '../../../src/gpu-spatial-analysis/spatial-regression/spatial-two-stage-least-squares-model-result';

for (const scope of ['whole', 'root', 'child'] as const) {
  for (const hasCount of [false, true]) {
    it(`R1 count-only ${scope} loss suppresses traces and recovers with count=${hasCount}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const graph = new GPUCommandGraph(device);
      const buffers: Buffer[] = [];
      let serial = 0;
      const input = <Format extends 'float32' | 'uint32'>(
        format: Format,
        values: Float32Array | Uint32Array
      ) => {
        const buffer = createInputBuffer(device, values);
        buffers.push(buffer);
        return {
          buffer,
          view: importGraphBuffer(graph, `count-input-${serial++}`, buffer, format, values.length)
        };
      };
      const output = () => {
        const buffer = createOutputBuffer(device, 3);
        buffers.push(buffer);
        buffer.write(Float32Array.of(77, 98, 99));
        return {
          buffer,
          view: importGraphBuffer(graph, `count-output-${serial++}`, buffer, 'float32', 3)
        };
      };
      const capacity = scope === 'child' ? 2 : 3;
      const required = input('uint32', Uint32Array.of(capacity));
      const count = input('uint32', Uint32Array.of(capacity));
      const status = {requiredCount: required.view, ...(hasCount ? {count: count.view} : {})};
      const createWeights = (offsets: number[], neighbors: number[]) => ({
        offsets: input('uint32', Uint32Array.from(offsets)).view,
        neighbors: input('uint32', Uint32Array.from(neighbors)).view,
        weights: input(
          'float32',
          Float32Array.from(neighbors, () => 1)
        ).view
      });
      const weights =
        scope === 'whole'
          ? {...createWeights([0, 1, 2, 3], [1, 0, 1, 99, 99]), status}
          : {
              rowCount: 3,
              ...(scope === 'root' ? {status} : {}),
              chunks: [
                {rowStart: 0, rowCount: 1, weights: createWeights([0, 1], [1])},
                {
                  rowStart: 1,
                  rowCount: 2,
                  weights: {
                    ...createWeights([0, 1, 2], [0, 1, 99, 99]),
                    ...(scope === 'child' ? {status} : {})
                  }
                }
              ]
            };
      const norm = output();
      const traces = output();
      const sums = output();
      try {
        graph.add(
          new GPUSpatialWeightsTraces({
            weights,
            maximumPower: 1,
            mode: 'exact',
            output: {infinityNorm: norm.view, powerTraces: traces.view, powerSums: sums.view}
          })
        );
        const compiled = graph.compile();
        try {
          for (const requirement of [capacity, capacity + 1, capacity + 3, capacity]) {
            const missing = requirement > capacity;
            required.buffer.write(Uint32Array.of(requirement));
            submitGraph(device, compiled, undefined);
            // Three unit-weight rows give norm 1, zero diagonal and total row sum 3.
            for (const {result, expected} of [
              {result: norm, expected: 1},
              {result: traces, expected: 0},
              {result: sums, expected: 3}
            ]) {
              const actual = await readFloat32(result.buffer, 3);
              expect(Array.from(actual)).toEqual([missing ? NaN : expected, 98, 99]);
            }
          }
          if (hasCount) {
            count.buffer.write(Uint32Array.of(capacity - 1));
            submitGraph(device, compiled, undefined);
            expect(Array.from(await readFloat32(norm.buffer, 3))).toEqual([NaN, 98, 99]);
          }
        } finally {
          compiled.destroy();
        }
      } finally {
        for (const buffer of buffers) buffer.destroy();
      }
    });
  }
}

it('R2/R6 complete fitted models and recipe suppress both flags and recover without output reseeding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  const input = <Format extends 'float32' | 'uint32'>(
    format: Format,
    values: Float32Array | Uint32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `fit-input-${serial++}`, buffer, format, values.length)
    };
  };
  const output = <Format extends 'float32' | 'uint32'>(format: Format, length: number) => {
    const buffer = createOutputBuffer(device, length + 2);
    buffers.push(buffer);
    buffer.write(
      format === 'float32'
        ? Float32Array.from([...new Array(length).fill(77), 98, 99])
        : Uint32Array.from([...new Array(length).fill(77), 98, 99])
    );
    return {
      buffer,
      view: importGraphBuffer(graph, `fit-output-${serial++}`, buffer, format, length + 2)
    };
  };
  const fixture = REVIEW_REGRESSION_FIXTURE;
  const overflow = input('uint32', Uint32Array.of(0));
  const candidateOverflow = input('uint32', Uint32Array.of(0));
  const props = {
    weights: {
      offsets: input('uint32', fixture.weights.offsets).view,
      neighbors: input('uint32', fixture.weights.neighbors).view,
      weights: input('float32', fixture.weights.weights).view,
      status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
    },
    predictors: input('float32', fixture.predictors).view,
    response: input('float32', fixture.response).view,
    predictorCount: 1
  };
  const models = [
    {
      kind: 'error',
      table: output('float32', 12),
      summary: output('float32', 13),
      residuals: output('float32', 8),
      code: output('uint32', 1),
      coefficients: fixture.errorCoefficients,
      expectedResiduals: fixture.errorResiduals
    },
    {
      kind: 'twoStage',
      table: output('float32', 12),
      summary: output('float32', 16),
      residuals: output('float32', 8),
      code: output('uint32', 1),
      coefficients: fixture.twoStageCoefficients,
      expectedResiduals: fixture.twoStageResiduals
    }
  ];
  const parameters = new GPUParameterBuffer(device, {
    id: 'recipe-parameters',
    format: 'float32',
    length: 8
  });
  const decision = output('uint32', 1);
  const diagnosticsCode = output('uint32', 1);
  const scalar = (view: typeof decision.view) =>
    graph.createDataView(view.buffer, {format: 'uint32', length: 1, byteOffset: view.byteOffset});
  try {
    for (const model of models) {
      const output = {
        table: model.table.view,
        summary: model.summary.view,
        residuals: model.residuals.view,
        status: {code: model.code.view}
      };
      graph.add(
        model.kind === 'error'
          ? new GPUSpatialErrorGM({...props, id: 'fit-error', estimator: 'kelejian-prucha', output})
          : new GPUSpatialTwoStageLeastSquares({...props, id: 'fit-two-stage', output})
      );
    }
    addSpatialRegressionRecipe(graph, {
      ...props,
      id: 'fit-recipe',
      parameters,
      output: {
        diagnostics: {status: {code: scalar(diagnosticsCode.view)}},
        specification: {decision: scalar(decision.view)}
      }
    });
    const compiled = graph.compile();
    try {
      for (const selected of [candidateOverflow, overflow]) {
        for (const loss of [0, 1, 0]) {
          selected.buffer.write(Uint32Array.of(loss));
          // No output is reseeded: normal writers, suppression and recovery must run in order.
          submitGraph(device, compiled, undefined);
          for (const model of models) {
            expect(Array.from(await readUint32(model.code.buffer, 3)), model.kind).toEqual([
              loss ? 3 : 0,
              98,
              99
            ]);
            const table = await readFloat32(model.table.buffer, 14);
            const residuals = await readFloat32(model.residuals.buffer, 10);
            const summary = await readFloat32(model.summary.buffer, model.summary.view.length);
            if (model.kind === 'twoStage')
              expect(
                getGPUSpatialTwoStageLeastSquaresDiagnostics(Float32Array.from(summary))
                  .rhoOutsideAdmissibleRange
              ).toBe(loss ? null : false);
            for (const values of [table, residuals, summary])
              expect(Array.from(values.slice(-2))).toEqual([98, 99]);
            if (loss) {
              for (const values of [table, residuals, summary])
                expect(Array.from(values.slice(0, -2)).every(Number.isNaN)).toBe(true);
            } else {
              expect(summary[0]).toBe(8);
              for (const [index, expected] of model.coefficients.entries()) {
                expect(Number.isFinite(table[index * 4])).toBe(true);
                // 2e-3 covers float32 moment solves and bounded GM parameter minimization.
                expect(Math.abs(table[index * 4] - expected)).toBeLessThanOrEqual(2e-3);
              }
              for (const [index, expected] of model.expectedResiduals.entries()) {
                expect(Number.isFinite(residuals[index])).toBe(true);
                expect(Math.abs(residuals[index] - expected)).toBeLessThanOrEqual(2e-3);
              }
            }
          }
          expect(Array.from(await readUint32(diagnosticsCode.buffer, 3))).toEqual([
            loss ? 3 : 0,
            98,
            99
          ]);
          // Independent LM p-values 0.4229 and 0.1041 imply OLS on clean weights; loss is undetermined.
          expect(Array.from(await readUint32(decision.buffer, 3))).toEqual([
            loss ? 4 : fixture.specificationDecision,
            98,
            99
          ]);
        }
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    parameters.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
