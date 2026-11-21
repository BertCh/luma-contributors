// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUVectorFormat} from '@luma.gl/graph/gpu-data';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPUSpatialRegimeRegression} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-spatial-regime-regression';
import {getRegimeMomentNodes} from '../../../src/gpu-spatial-analysis/spatial-regression/regime-moments';
import {
  GPUSpatialCrossValidatedOLS,
  type GPUSpatialCrossValidatedOLSProps
} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validated-ols';
import {
  GPUOrdinaryLeastSquares,
  type GPUOrdinaryLeastSquaresProps
} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {AUDIT_REGRESSION_MODELS} from './audit-regression-models-fixtures';
import {AUDIT_OLS_FIXTURES} from './audit-ols-fixtures';

async function createHarness() {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  let serial = 0;
  let cleanup = () => {};
  return {
    graph,
    input<Format extends GPUVectorFormat>(values: number[], format: Format, components = 1) {
      const buffer = createInputBuffer(
        device,
        format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
      );
      buffers.push(buffer);
      return importGraphBuffer(
        graph,
        `input-${serial++}`,
        buffer,
        format,
        values.length / components
      );
    },
    output<Format extends 'float32' | 'uint32'>(length: number, format: Format) {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {
        view: importGraphBuffer(graph, `output-${serial++}`, buffer, format, length),
        read: () => (format === 'uint32' ? readUint32(buffer, length) : readFloat32(buffer, length))
      };
    },
    run() {
      const compiled = graph.compile();
      cleanup = () => compiled.destroy();
      submitGraph(device, compiled, undefined);
    },
    destroy() {
      cleanup();
      for (const buffer of buffers) buffer.destroy();
    }
  };
}

function expectRows(actual: ArrayLike<number>, expected: number[], tolerance = 5e-4) {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, index) => {
    if (Number.isNaN(value)) expect(actual[index], `row ${index}`).toBeNaN();
    else {
      expect(Number.isFinite(actual[index]), `row ${index}`).toBe(true);
      // Small fixed fixtures: relative f32 reduction/solve bound with an absolute floor.
      expect(Math.abs(actual[index] - value), `row ${index}`).toBeLessThanOrEqual(
        tolerance * Math.max(1, Math.abs(value))
      );
    }
  });
}

it('direct GPU OLS retains every oracle output after caller mutates validated topology', async () => {
  const fixture = AUDIT_OLS_FIXTURES[0];
  const harness = await createHarness();
  try {
    const output = {
      coefficients: harness.output(3, 'float32'),
      standardErrors: harness.output(3, 'float32'),
      tStatistics: harness.output(3, 'float32'),
      summary: harness.output(22, 'float32'),
      status: harness.output(1, 'uint32'),
      residuals: harness.output(12, 'float32'),
      fitted: harness.output(12, 'float32'),
      predictions: harness.output(12, 'float32')
    };
    const props: GPUOrdinaryLeastSquaresProps = {
      predictors: harness.input(fixture.predictors, 'float32'),
      response: harness.input(fixture.response, 'float32'),
      mask: harness.input(fixture.mask, 'uint32'),
      predictorCount: 2,
      output: {
        coefficients: output.coefficients.view,
        standardErrors: output.standardErrors.view,
        tStatistics: output.tStatistics.view,
        summary: output.summary.view,
        status: {code: output.status.view},
        fitted: output.fitted.view,
        residuals: output.residuals.view,
        predictions: output.predictions.view
      }
    };
    const contributor = new GPUOrdinaryLeastSquares(props);
    props.predictorCount = 1;
    props.output.predictions = harness.output(1, 'float32').view;
    props.output.status.code = harness.output(1, 'uint32').view;
    harness.graph.add(contributor);
    harness.run();
    for (const name of Object.keys(output) as (keyof typeof output)[])
      expectRows(await output[name].read(), fixture.expected[name], name === 'status' ? 0 : 5e-4);
  } finally {
    harness.destroy();
  }
});

it('R6 regime regression honors an inherited mask instead of fitting the masked outlier', async () => {
  const harness = await createHarness();
  try {
    const mask = harness.input([1, 1, 1, 0], 'uint32');
    const coefficients = harness.output(2, 'float32');
    const fitted = harness.output(4, 'float32');
    const counts = harness.output(1, 'uint32');
    const status = harness.output(1, 'uint32');
    const props = Object.assign(
      Object.create({
        get mask() {
          return mask;
        }
      }),
      {
        predictorCount: 1,
        regimeCount: 1,
        predictors: harness.input([0, 1, 2, 3], 'float32'),
        response: harness.input([0, 1, 2, 99], 'float32'),
        regimes: harness.input([0, 0, 0, 0], 'uint32'),
        output: {
          coefficients: coefficients.view,
          fitted: fitted.view,
          counts: counts.view,
          status: status.view
        }
      }
    );
    harness.graph.add(new GPUSpatialRegimeRegression(props));
    harness.run();
    // The three unmasked pairs lie on y=x; including the outlier instead fits [-19.2,29.8].
    expectRows(await coefficients.read(), [0, 1], 1e-6);
    expectRows(await fitted.read(), [0, 1, 2, NaN], 1e-6);
    expectRows(await counts.read(), [3], 0);
    expectRows(await status.read(), [0], 0);
  } finally {
    harness.destroy();
  }
});

for (const origin of [0, 1000000]) {
  it(`R2 held-out metrics are invariant to the exactly representable origin ${origin}`, async () => {
    const harness = await createHarness();
    try {
      const metrics = harness.output(12, 'float32');
      const metricStatus = harness.output(2, 'uint32');
      const fitStatus = harness.output(2, 'uint32');
      harness.graph.add(
        new GPUSpatialCrossValidatedOLS({
          predictorCount: 1,
          predictors: harness.input([-1, -1, 1, 1, -1, -1, 1, 1], 'float32'),
          response: harness.input(
            [0, 0.0625, 0, 0.0625, 0, 0, 0, 0].map(value => origin + value),
            'float32'
          ),
          folds: {
            positions: harness.input(
              Array.from({length: 8}, (_, row) => [row, 0]).flat(),
              'float32x2',
              2
            ),
            blocks: harness.input([0, 0, 0, 0, 1, 1, 1, 1], 'uint32'),
            design: {mode: 'groups'},
            foldCount: 2,
            output: {
              folds: harness.output(8, 'uint32').view,
              training: harness.output(16, 'uint32').view,
              testing: harness.output(16, 'uint32').view,
              counts: harness.output(6, 'uint32').view,
              status: harness.output(2, 'uint32').view
            }
          },
          output: {
            coefficients: harness.output(4, 'float32').view,
            predictions: harness.output(16, 'float32').view,
            fitStatus: fitStatus.view,
            metrics: metrics.view,
            metricStatus: metricStatus.view
          }
        })
      );
      harness.run();
      expectRows(await fitStatus.read(), [0, 0], 0);
      // Fold 0 predicts origin. SSE=1/128, centered test TSS=1/256, so R²=-1.
      expectRows(
        (await metrics.read()).slice(0, 6),
        [4, 0, 0.001953125, 0.03125, -0.03125, -1],
        1e-6
      );
      expect((await metricStatus.read())[0]).toBe(0);
    } finally {
      harness.destroy();
    }
  });
}

for (const fixture of AUDIT_REGRESSION_MODELS.regimes) {
  it(`regime OLS every output matches SciPy with empty/singular groups: ${fixture.name}`, async () => {
    const harness = await createHarness();
    try {
      const output = {
        coefficients: harness.output(15, 'float32'),
        fitted: harness.output(20, 'float32'),
        status: harness.output(5, 'uint32'),
        counts: harness.output(5, 'uint32')
      };
      const inputs = {
        predictors: harness.input(fixture.predictors, 'float32'),
        response: harness.input(fixture.response, 'float32'),
        regimes: harness.input(fixture.regimes, 'uint32'),
        mask: harness.input(fixture.mask, 'uint32'),
        predictorCount: 2,
        regimeCount: 5
      };
      harness.graph.add(
        new GPUSpatialRegimeRegression({
          ...inputs,
          output: {
            coefficients: output.coefficients.view,
            fitted: output.fitted.view,
            status: output.status.view,
            counts: output.counts.view
          }
        })
      );
      harness.run();
      for (const name of Object.keys(output) as (keyof typeof output)[]) {
        expectRows(
          await output[name].read(),
          fixture.expected[name],
          name === 'status' || name === 'counts' ? 0 : 5e-4
        );
      }
    } finally {
      harness.destroy();
    }
  });

  it(`regime moments and sorted offsets match the independent definition: ${fixture.name}`, async () => {
    const harness = await createHarness();
    try {
      const {nodes, views} = getRegimeMomentNodes(harness.graph, {
        id: 'moments-audit',
        operation: 'regime-moments-audit',
        predictorCount: 2,
        regimeCount: 5,
        predictors: harness.input(fixture.predictors, 'float32'),
        response: harness.input(fixture.response, 'float32'),
        regimes: harness.input(fixture.regimes, 'uint32'),
        mask: harness.input(fixture.mask, 'uint32')
      });
      // Publish helper transients through copies so readback uses caller-owned buffers.
      const output = {offsets: harness.output(6, 'uint32'), moments: harness.output(60, 'float32')};
      for (const node of nodes) harness.graph.add(node);
      harness.graph.add(
        createWGSLKernelNode(harness.graph, {
          id: 'copy-offsets',
          operation: 'audit',
          variant: 'copy-offsets',
          invocationCount: 6,
          bindings: [
            {name: 'source', view: views.segmentOffsets, type: 'u32', access: 'read'},
            {name: 'destination', view: output.offsets.view, type: 'u32', access: 'read_write'}
          ],
          body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
        })
      );
      harness.graph.add(
        createWGSLKernelNode(harness.graph, {
          id: 'copy-moments',
          operation: 'audit',
          variant: 'copy-moments',
          invocationCount: 60,
          bindings: [
            {name: 'source', view: views.moments, type: 'f32', access: 'read'},
            {name: 'destination', view: output.moments.view, type: 'f32', access: 'read_write'}
          ],
          body: 'destination[destinationOffset + index] = source[sourceOffset + index];'
        })
      );
      harness.run();
      expectRows(await output.offsets.read(), fixture.offsets, 0);
      expectRows(await output.moments.read(), fixture.moments, 1e-5);
    } finally {
      harness.destroy();
    }
  });
}

for (const fixture of AUDIT_REGRESSION_MODELS.crossValidation) {
  it(`cross-validated OLS every output matches train-only SciPy fits: ${fixture.name}`, async () => {
    const harness = await createHarness();
    try {
      const output = {
        coefficients: harness.output(4, 'float32'),
        predictions: harness.output(24, 'float32'),
        fitStatus: harness.output(2, 'uint32'),
        metrics: harness.output(12, 'float32'),
        metricStatus: harness.output(2, 'uint32')
      };
      const folds = {
        folds: harness.output(12, 'uint32'),
        training: harness.output(24, 'uint32'),
        testing: harness.output(24, 'uint32'),
        counts: harness.output(6, 'uint32'),
        status: harness.output(2, 'uint32')
      };
      const props: GPUSpatialCrossValidatedOLSProps = {
        predictors: harness.input(fixture.predictors, 'float32'),
        response: harness.input(fixture.response, 'float32'),
        predictorCount: 1,
        folds: {
          positions: harness.input(
            Array.from({length: 12}, (_, row) => [row, 0]).flat(),
            'float32x2',
            2
          ),
          blocks: harness.input(fixture.blocks, 'uint32'),
          mask: harness.input(fixture.mask, 'uint32'),
          design: {mode: 'groups'},
          foldCount: 2,
          output: {
            folds: folds.folds.view,
            training: folds.training.view,
            testing: folds.testing.view,
            counts: folds.counts.view,
            status: folds.status.view
          }
        },
        output: {
          coefficients: output.coefficients.view,
          predictions: output.predictions.view,
          fitStatus: output.fitStatus.view,
          metrics: output.metrics.view,
          metricStatus: output.metricStatus.view
        }
      };
      const contributor = new GPUSpatialCrossValidatedOLS(props);
      props.predictorCount = 2;
      props.folds.foldCount = 3;
      props.folds.output.training = harness.output(1, 'uint32').view;
      props.output.predictions = harness.output(1, 'float32').view;
      harness.graph.add(contributor);
      harness.run();
      for (const name of Object.keys(output) as (keyof typeof output)[])
        expectRows(
          await output[name].read(),
          fixture.expected[name],
          name.includes('Status') ? 0 : 5e-4
        );
      for (const name of Object.keys(folds) as (keyof typeof folds)[])
        expectRows(await folds[name].read(), fixture.foldExpected[name], 0);
    } finally {
      harness.destroy();
    }
  });
}
