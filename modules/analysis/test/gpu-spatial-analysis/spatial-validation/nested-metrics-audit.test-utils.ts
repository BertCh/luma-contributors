// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUNestedInterpolationValidation} from '../../../src/gpu-spatial-analysis/spatial-validation/gpu-nested-interpolation-validation';
import {getRegressionTileLayout} from '../../../src/gpu-spatial-analysis/spatial-regression/regression-row-tiles';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

export function getNestedMetricAuditKernels(rows: number) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let nextView = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `nested-audit-${nextView++}`, format, length);
  try {
    const contributor = new GPUNestedInterpolationValidation({
      positions: view('float32x2', rows),
      response: view('float32', rows),
      training: view('uint32', rows * 2),
      testing: view('uint32', rows * 2),
      innerBlocks: view('uint32', rows),
      foldCount: 2,
      innerFoldCount: 2,
      neighborCount: 2,
      candidates: [{mode: 'rbf', kernel: 'gaussian', epsilon: 1}],
      output: {
        selected: view('uint32', 2),
        scores: view('float32', 6),
        predictions: view('float32', rows * 2),
        metrics: view('float32', 12),
        status: view('uint32', 2)
      }
    });
    return getGPUCommandNodeKernelSources(
      contributor
        .getCommandNodes(graph)
        .filter(
          node => node.id.endsWith('fold-0-metric-tiles') || node.id.endsWith('fold-0-metric-merge')
        )
    );
  } finally {
    device.destroy();
  }
}

export function createNestedMetricAuditValues(
  rows: number,
  shift: number
): Record<string, Float32Array | Uint32Array> {
  const response = new Float32Array(rows);
  const predictions = new Float32Array(rows);
  const testing = new Uint32Array(rows);
  response[0] = shift;
  response[rows - 1] = shift + 1;
  predictions[0] = shift + 1;
  predictions[rows - 1] = shift + 2;
  testing[0] = testing[rows - 1] = 1;
  const tileCount = rows <= 256 ? 1 : getRegressionTileLayout(rows).tileCount;
  return {
    response,
    predictions,
    testing,
    partial: new Float32Array(tileCount * 9),
    metrics: new Float32Array(6),
    status: new Uint32Array(1)
  };
}
