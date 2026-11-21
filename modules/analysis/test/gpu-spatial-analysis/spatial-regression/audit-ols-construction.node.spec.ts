// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUOrdinaryLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import {GPUSpatialCrossValidatedOLS} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-spatial-cross-validated-ols';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('OLS preserves the validated design width, optional predictions and status view after caller mutation', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
      createTransientView(graph, `snapshot-${serial++}`, format, length);
    const status = view('uint32', 1);
    const predictions = view('float32', 8);
    const props = {
      predictorCount: 1,
      predictors: view('float32', 8),
      response: view('float32', 8),
      output: {
        coefficients: view('float32', 2),
        standardErrors: view('float32', 2),
        tStatistics: view('float32', 2),
        summary: view('float32', 22),
        status: {code: status},
        predictions
      }
    };
    const contributor = new GPUOrdinaryLeastSquares(props);
    props.predictorCount = 2;
    props.output.status.code = view('uint32', 1);
    props.output.predictions = view('float32', 1);
    expect(contributor.props.predictorCount).toBe(1);
    expect(contributor.props.output.status.code).toBe(status);
    expect(contributor.props.output.predictions).toBe(predictions);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('cross-validated OLS preserves validated fold design and output maps after caller mutation', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `cv-snapshot-${serial++}`, format, length);
    const props = {
      predictorCount: 1,
      predictors: view('float32', 8),
      response: view('float32', 8),
      folds: {
        positions: view('float32x2', 8),
        blocks: view('uint32', 8),
        foldCount: 2,
        design: {mode: 'groups' as const},
        output: {
          folds: view('uint32', 8),
          training: view('uint32', 16),
          testing: view('uint32', 16),
          counts: view('uint32', 6),
          status: view('uint32', 2)
        }
      },
      output: {
        coefficients: view('float32', 4),
        predictions: view('float32', 16),
        fitStatus: view('uint32', 2),
        metrics: view('float32', 12),
        metricStatus: view('uint32', 2)
      }
    };
    const predictions = props.output.predictions;
    const training = props.folds.output.training;
    const contributor = new GPUSpatialCrossValidatedOLS(props);
    props.predictorCount = 2;
    props.folds.foldCount = 3;
    props.folds.output.training = view('uint32', 1);
    props.output.predictions = view('float32', 1);
    expect(contributor.props.predictorCount).toBe(1);
    expect(contributor.props.folds.foldCount).toBe(2);
    expect(contributor.props.folds.output.training).toBe(training);
    expect(contributor.props.output.predictions).toBe(predictions);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});
