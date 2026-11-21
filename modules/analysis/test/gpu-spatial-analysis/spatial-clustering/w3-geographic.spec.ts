// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPUDBSCAN,
  getGPUDBSCANParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-clustering';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it('uses reversible longitude rotation rather than unsupported unwrapping for seam centroids', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const spatialContext = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters'
  } as const;
  for (const fixture of [
    {longitudes: [179.9, -179.9], latitude: 0, mean: 0, inverse: -180},
    {longitudes: [179.98, -179.86], latitude: 1, mean: 0.06, inverse: -179.94}
  ]) {
    for (const route of ['original', 'unwrapped', 'rotated'] as const) {
      const longitudes = fixture.longitudes.map(longitude =>
        route === 'original'
          ? longitude
          : route === 'unwrapped'
            ? longitude < 0
              ? longitude + 360
              : longitude
            : ((longitude - 180 + 540) % 360) - 180
      );
      const bounds =
        route === 'original'
          ? [-180, -1, 180, 1]
          : route === 'unwrapped'
            ? [179, -1, 181, 1]
            : [-1, -1, 1, 1];
      const graph = new GPUCommandGraph(device);
      const positions = createInputBuffer(
        device,
        Float32Array.of(longitudes[0], fixture.latitude, longitudes[1], fixture.latitude)
      );
      const parameters = createInputBuffer(
        device,
        getGPUDBSCANParameterValues({
          spatialContext,
          bounds: bounds as [number, number, number, number],
          epsilon: 30000,
          minimumSamples: 2
        })
      );
      const labels = createOutputBuffer(device, 2);
      const ids = createOutputBuffer(device, 1);
      const count = createOutputBuffer(device, 1);
      const overflow = createOutputBuffer(device, 1);
      const centroids = createOutputBuffer(device, 2);
      labels.write(Uint32Array.of(77, 77));
      ids.write(Uint32Array.of(77));
      count.write(Uint32Array.of(77));
      overflow.write(Uint32Array.of(77));
      centroids.write(Float32Array.of(1234, 5678));
      graph.add(
        new GPUDBSCAN({
          positions: importGraphBuffer(graph, 'positions', positions, 'float32x2', 2),
          parameters: importGraphBuffer(graph, 'parameters', parameters, 'float32', 8),
          spatialContext,
          gridSize: [8, 8],
          output: {
            labels: importGraphBuffer(graph, 'labels', labels, 'uint32', 2),
            clusters: {
              ids: importGraphBuffer(graph, 'ids', ids, 'uint32', 1),
              status: {
                count: importGraphBuffer(graph, 'count', count, 'uint32', 1),
                overflow: importGraphBuffer(graph, 'overflow', overflow, 'uint32', 1)
              }
            },
            clusterCentroids: importGraphBuffer(graph, 'centroids', centroids, 'float32x2', 1)
          }
        })
      );
      const compiled = graph.compile();
      submitGraph(device, compiled, undefined);
      expect(await readUint32(labels, 2)).toEqual(
        route === 'unwrapped' ? [0xffffffff, 0xffffffff] : [0, 0]
      );
      expect(await readUint32(count, 1)).toEqual([route === 'unwrapped' ? 0 : 1]);
      if (route !== 'unwrapped') {
        const result = await readFloat32(centroids, 2);
        expect(Number.isFinite(result[0])).toBe(true);
        expect(Number.isFinite(result[1])).toBe(true);
        expect(result[0]).toBeCloseTo(fixture.mean, 4);
        expect(result[1]).toBeCloseTo(fixture.latitude, 4);
        // Inverting the 180-degree rotation gives a centroid at the seam, rather than Greenwich.
        if (route === 'rotated') {
          const inverseLongitude = ((result[0] + 180 + 540) % 360) - 180;
          expect(inverseLongitude).toBeCloseTo(fixture.inverse, 4);
        }
      }
      compiled.destroy();
      for (const buffer of [positions, parameters, labels, ids, count, overflow, centroids])
        buffer.destroy();
    }
  }
});
