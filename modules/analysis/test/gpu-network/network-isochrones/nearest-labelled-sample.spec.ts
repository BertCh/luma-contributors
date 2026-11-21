// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUNetworkIsochrones,
  getGPUNetworkIsochronesParameterValues
} from '../../../src/gpu-network/network-isochrones/index';
import {createIsochronesTriangleFacilityNodes} from '../../../src/gpu-network/network-isochrones/network-isochrones-passes';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createIsochroneOracleHarness,
  expectOracleValues,
  NONE
} from './isochrones-output-oracle-utils';

it('R8 labels both upper-cell triangles from a sample outside their four corners', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createIsochroneOracleHarness(device);
  const result = await harness.run(
    new GPUNetworkIsochrones({
      topology: {
        offsets: harness.input('uint32', [0, 1, 2]),
        neighbors: harness.input('uint32', [0, 1]),
        weights: harness.input('float32', [0, 0]),
        nodePositions: harness.input('float32x2', [2, 1.5, 8, 0.5])
      },
      costs: harness.input('float32', [3, 3]),
      assignments: harness.input('uint32', [2, 5]),
      costBreaks: harness.input('float32', [6]),
      parameters: harness.input(
        'float32',
        Array.from(
          getGPUNetworkIsochronesParameterValues({
            breakCount: 1,
            extent: [0, 0, 8, 3],
            firstBand: 1,
            lastBand: 1
          })
        )
      ),
      raster: {
        width: 2,
        height: 3,
        unreachedCost: 9,
        output: {
          values: harness.output('values', 'float32', 6),
          pixelFacilities: harness.output('pixels', 'uint32', 6),
          triangles: harness.output('triangles', 'float32x2', 12),
          triangleBands: harness.output('bands', 'uint32', 4),
          triangleFacilities: harness.output('facilities', 'uint32', 4),
          status: {
            count: harness.output('count', 'uint32', 1),
            overflow: harness.output('overflow', 'uint32', 1)
          }
        }
      }
    })
  );
  expectOracleValues(result['values'], [3, 3, 3, 9, 3, 9]);
  expectOracleValues(result['pixels'], [2, 5, 2, NONE, 2, NONE]);
  expectOracleValues(result['count'], [3]);
  expectOracleValues(result['overflow'], [0]);
  // The two triangles after the lower-cell triangle lie in y >= 1.5. The upper-cell
  // quad can be triangulated with centroids (16/3,11/6),(14/3,13/6), squared
  // distances 20/9,41/9 to label 5. Either diagonal chooses 5 for both triangles.
  for (const coordinate of [7, 9, 11, 13, 15, 17]) {
    expect(Number.isFinite(result['triangles'][coordinate])).toBe(true);
    expect(result['triangles'][coordinate]).toBeGreaterThanOrEqual(1.5);
  }
  expectOracleValues(result['facilities'].slice(1), [5, 5, NONE]);
});

it.each([
  1,
  2 ** 50
])('R8 preserves ties, unlabelled recovery and inactive slots (scale=%s)', async scale => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createIsochroneOracleHarness(device);
  const triangles = harness.input('float32x2', new Array(3).fill([4 * scale, 0.5 * scale]).flat());
  const triangleCount = harness.input('uint32', [1]);
  const parameters = harness.input(
    'float32',
    Array.from(
      getGPUNetworkIsochronesParameterValues({breakCount: 1, extent: [0, 0, 8 * scale, 3 * scale]})
    )
  );
  const pixelFacilities = harness.input('uint32', [5, 2, NONE, NONE, NONE, NONE]);
  const triangleFacilities = harness.output('facilities', 'uint32', 2);
  const nodes = createIsochronesTriangleFacilityNodes(harness.graph, {
    id: 'nearest',
    width: 2,
    height: 3,
    triangles,
    triangleCount,
    parameters,
    pixelFacilities,
    triangleFacilities
  });
  // The centroid is (4,.5): both labelled sample centers are two world units away.
  const result = await harness.run({getCommandNodes: () => nodes}, first => {
    expectOracleValues(first['facilities'], [2, NONE]);
    harness.write(pixelFacilities, new Array(6).fill(NONE));
  });
  expectOracleValues(result['facilities'], [NONE, NONE]);
});

it.each([0, 5000001])('B1 preserves fan ties and translated non-ties (minX=%s)', async minimumX => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createIsochroneOracleHarness(device);
  const props = {
    id: 'fan-nearest',
    width: 2,
    height: 2,
    triangles: harness.input('float32x2', [
      minimumX + 0.5,
      0.5,
      minimumX + 1.5,
      0.5,
      minimumX + 1.5,
      1.5,
      minimumX + 0.5,
      0.5,
      minimumX + 1.5,
      1.5,
      minimumX + 0.5,
      1.5
    ]),
    triangleCount: harness.input('uint32', [2]),
    parameters: harness.input(
      'float32',
      Array.from(
        getGPUNetworkIsochronesParameterValues({
          breakCount: 1,
          extent: [minimumX, 0, minimumX + 2, 2]
        })
      )
    ),
    pixelFacilities: harness.input('uint32', minimumX === 0 ? [9, NONE, NONE, 2] : [1, 9, 1, 9]),
    triangleFacilities: harness.output('facilities', 'uint32', 2)
  };
  const nodes = createIsochronesTriangleFacilityNodes(harness.graph, props);
  const result = await harness.run({getCommandNodes: () => nodes});
  // Exact squared centroid distances tie at 5/9; the translated striped case has
  // right/left centroids strictly nearer the corresponding column of sample centers.
  expectOracleValues(result['facilities'], minimumX === 0 ? [2, 2] : [9, 1]);
});

it('S1 prunes dense and empty raster regions on device', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createIsochroneOracleHarness(device);
  const pixelFacilities = harness.input('uint32', new Array(1024).fill(7));
  const nodes = createIsochronesTriangleFacilityNodes(harness.graph, {
    id: 'pruned',
    width: 32,
    height: 32,
    triangles: harness.input('float32x2', [0.5, 0.5, 1.5, 0.5, 1.5, 1.5]),
    triangleCount: harness.input('uint32', [1]),
    parameters: harness.input(
      'float32',
      Array.from(getGPUNetworkIsochronesParameterValues({breakCount: 1, extent: [0, 0, 32, 32]}))
    ),
    pixelFacilities,
    triangleFacilities: harness.output('facilities', 'uint32', 1),
    visitedSamples: harness.output('visits', 'uint32', 1)
  });
  const result = await harness.run({getCommandNodes: () => nodes}, first => {
    expectOracleValues(first['facilities'], [7]);
    expect(Number.isFinite(first['visits'][0])).toBe(true);
    expect(first['visits'][0]).toBeGreaterThan(0);
    expect(first['visits'][0]).toBeLessThan(16);
    harness.write(pixelFacilities, new Array(1024).fill(NONE));
  });
  expectOracleValues(result['facilities'], [NONE]);
  expectOracleValues(result['visits'], [0]);
});
