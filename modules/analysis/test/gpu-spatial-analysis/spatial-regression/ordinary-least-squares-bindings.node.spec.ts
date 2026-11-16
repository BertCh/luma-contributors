// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {GPUOrdinaryLeastSquares} from '../../../src/gpu-spatial-analysis/spatial-regression/gpu-ordinary-least-squares';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('OLS validates actual aligned caller ranges before declaring either fit workspace', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `ols-bindings-${serial++}`, format, length);
  try {
    const predictors = graph.createDataView(view('float32', 129).buffer, {
      format: 'float32',
      byteOffset: 4,
      length: 128
    });
    const contributor = new GPUOrdinaryLeastSquares({
      id: 'ols-bindings',
      predictorCount: 1,
      predictors,
      response: view('float32', 128),
      output: {
        coefficients: view('float32', 2),
        standardErrors: view('float32', 2),
        tStatistics: view('float32', 2),
        summary: view('float32', 22),
        status: {code: view('uint32', 1)}
      }
    });
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 512});
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/ols-bindings predictors.*516 bytes/);
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});
