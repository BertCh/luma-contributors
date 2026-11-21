// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/src/testing/require-test-device';
import {getCatalogBinding, runCatalogOperationAsync} from '../src/analysis/catalog-bindings';

// Include the example's GPU-written extent-mask tests in the standard modules inventory.
import '../../../examples/deck/spatial-analysis-showcase/src/scenes/joins/raster-join-mismatch.spec';

test('MGTWR catalog retains centered small residuals around a large response', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const predictors = [-4, -3, -2, -1, 1, 2, 3, 4];
  const result = await runCatalogOperationAsync(
    getCatalogBinding('regression-mgtwr')!,
    {
      coordinateSystem: 'planar',
      predictorProperties: 'predictor',
      responseProperty: 'observed',
      bandwidth: 30,
      timeScale: 0,
      componentBandwidths: [30, 30],
      componentTimeScales: [0, 0],
      iterationCount: 2
    },
    {
      points: {
        type: 'FeatureCollection',
        features: predictors.map(predictor => ({
          geometry: {type: 'Point', coordinates: [0, 0]},
          properties: {predictor, observed: 1000000 + (predictor > 0 ? 1 / 16 : 0), time: 0}
        }))
      }
    },
    {device}
  );
  expect(result.engine).toBe('gpu');
  const table = result.tables[0];
  // Sum x²=60, sum x*(y-mean)=10/16; slope=(10/16)/60=1/96.
  const expectedResiduals = [1, 0, -1, -2, 2, 1, 0, -1].map(value => value / 96);
  table.columns['coef_predictor'].forEach(value => {
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs((value as number) - 1 / 96)).toBeLessThan(1e-7);
  });
  table.columns['residual'].forEach((value, row) => {
    expect(Number.isFinite(value)).toBe(true);
    expect(Math.abs((value as number) - expectedResiduals[row])).toBeLessThan(1e-7);
  });
});

test('distance-band kernel follows resident bandwidth updates without adaptive kNN provenance', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {GPUCommandGraph} = await import('@luma.gl/graph/gpu-core');
  const {GPUSpatialWeightsTransform, getGPUSpatialWeightsTransformParameterValues} = await import(
    '@luma-contributors/analysis/gpu-spatial-analysis'
  );
  const {importGraphBuffer} = await import(
    '../../analysis/src/analysis-core/kernels/gpu-contributor-utils'
  );
  const {createInputBuffer, readFloat32, submitGraph} = await import(
    '../../analysis/src/testing/gpu-contributor-test-utils'
  );
  const graph = new GPUCommandGraph(device, {id: 'showcase-fixed-distance-band-kernel'});
  const offsets = createInputBuffer(device, Uint32Array.of(0, 1, 2));
  const neighbors = createInputBuffer(device, Uint32Array.of(1, 0));
  const distances = createInputBuffer(device, Float32Array.of(1, 1));
  // Written in place and read back, so it also needs COPY_SRC (input buffers are not readable).
  const weights = device.createBuffer({
    data: Float32Array.of(1, 1),
    usage: Buffer.STORAGE | Buffer.COPY_DST | Buffer.COPY_SRC
  });
  const parameters = createInputBuffer(
    device,
    getGPUSpatialWeightsTransformParameterValues({bandwidth: 1})
  );
  graph.add(
    new GPUSpatialWeightsTransform({
      operation: 'kernel',
      kernel: 'gaussian',
      weights: {
        offsets: importGraphBuffer(graph, 'offsets', offsets, 'uint32', 3),
        neighbors: importGraphBuffer(graph, 'neighbors', neighbors, 'uint32', 2),
        weights: importGraphBuffer(graph, 'weights', weights, 'float32', 2),
        distances: importGraphBuffer(graph, 'distances', distances, 'float32', 2)
      },
      bandwidth: importGraphBuffer(graph, 'parameters', parameters, 'float32', 1)
    })
  );
  const compiled = graph.compile();
  try {
    // Independent Gaussian K(1)=exp(-1/2)/sqrt(2*pi), K(1/2)=exp(-1/8)/sqrt(2*pi).
    for (const [bandwidth, expected] of [
      [1, 0.24197072451914337],
      [2, 0.3520653267642995]
    ]) {
      parameters.write(getGPUSpatialWeightsTransformParameterValues({bandwidth}));
      submitGraph(device, compiled, undefined);
      for (const value of await readFloat32(weights, 2)) {
        expect(Number.isFinite(value)).toBe(true);
        expect(Math.abs(value - expected)).toBeLessThan(1e-6);
      }
    }
  } finally {
    compiled.destroy();
    for (const buffer of [offsets, neighbors, distances, weights, parameters]) buffer.destroy();
  }
});
