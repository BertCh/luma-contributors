// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-geometry-measures';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

it('preserves the second word of a geometry invalidCount view on clean reuse', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createParityGraph(device);
  const {graph, input, output} = fixture;
  try {
    const invalidCount = output('invalid-count', 2, 'uint32');
    const vertexCounts = output('vertex-counts', 1, 'uint32');
    graph.add(
      new GPUGeometryMeasures({
        geometryType: 'points',
        positions: input('positions', [0, 0], 'float32x2').view,
        output: {vertexCounts: vertexCounts.view, status: {invalidCount: invalidCount.view}}
      })
    );
    const compiled = graph.compile();
    try {
      for (let submission = 0; submission < 2; submission++) {
        invalidCount.buffer.write(Uint32Array.of(99, 0x12345678));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(invalidCount.buffer, 2)).toEqual([0, 0x12345678]);
        expect(await readUint32(vertexCounts.buffer, 1)).toEqual([1]);
      }
    } finally {
      compiled.destroy();
    }
  } finally {
    fixture.destroy();
  }
});

for (const cooperativeRingRows of [0, 1]) {
  it(`preserves the extreme-longitude multipoint bounds remainder (cooperative=${cooperativeRingRows})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createParityGraph(device);
    const {graph, input, output} = fixture;
    try {
      const bounds = output('bounds', 8, 'float32');
      const centroids = output('centroids', 2, 'float32x2');
      graph.add(
        new GPUGeometryMeasures({
          geometryType: 'points',
          positions: input('positions', [0, 0, 2 ** 62, 0, 0, 0, -176, 0], 'float32x2').view,
          featureOffsets: input('features', [0, 2, 4], 'uint32').view,
          cooperativeRingRows,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
          output: {
            centroids: centroids.view,
            bounds: graph.createDataView(bounds.view.buffer, {format: 'float32x4', length: 2})
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readFloat32(bounds.buffer, 8)).toEqual([-176, 0, 0, 0, -176, 0, 0, 0]);
        expect(await readFloat32(centroids.buffer, 4)).toEqual([-88, 0, -88, 0]);
      } finally {
        compiled.destroy();
      }
    } finally {
      fixture.destroy();
    }
  });
}
