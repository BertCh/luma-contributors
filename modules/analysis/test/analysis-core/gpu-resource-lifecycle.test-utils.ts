// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUFittingCoordinateEncoder} from '../../src/analysis-core/kernels/gpu-fitting-coordinate-encoder';
import {GPUFittingGradientSums} from '../../src/analysis-core/kernels/gpu-fitting-gradient-sums';
import {GPUFittingMinibatchOrder} from '../../src/analysis-core/kernels/gpu-fitting-minibatch-order';
import {createFittingTestGraph} from './fitting/fitting-test-graph';

/** Shared with a node construction smoke test, so invalid fixture props fail without a GPU. */
export function createResourceLifecycleFixture(device: Device) {
  const test = createFittingTestGraph(device);
  try {
    const rowCount = 513;
    const coordinates = test.floats('coordinates', new Float32Array(rowCount * 2));
    const features = test.floats('features', new Float32Array(rowCount * 2));
    const validity = test.uints('validity', new Uint32Array(rowCount));
    const sums = test.floats('sums', [0, 0]);
    const order = test.uints('order', new Uint32Array(rowCount));
    test.graph.add(
      new GPUFittingCoordinateEncoder({
        rowCount,
        coordinates,
        dimensions: 2,
        encoding: 'fourier',
        frequencyCount: 1,
        frequencies: test.floats('frequencies', [1, 0]),
        output: {features, validity}
      })
    );
    test.graph.add(
      new GPUFittingGradientSums({
        rowCount,
        parameterCount: 2,
        gradients: features,
        output: {sums}
      })
    );
    test.graph.add(
      new GPUFittingMinibatchOrder({
        rowCount,
        parameters: test.uints('order-parameters', [1, 2, 3]),
        output: {order}
      })
    );
    return {test, rowCount, sums, order};
  } catch (error) {
    test.destroy();
    throw error;
  }
}
