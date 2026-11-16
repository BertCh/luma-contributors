// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPURectangleClip,
  getGPURectangleClipParameterValues
} from '../../../src/gpu-spatial-analysis/rectangle-clip/gpu-rectangle-clip';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {requireTestDevice} from '../../utils/require-test-device';

it('clips a two-polygon feature and publishes feature provenance with an empty second ring', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      positions: {
        format: 'float32x2',
        values: new Float32Array([-1, -1, 3, -1, 3, 3, -1, 3, 10, 10, 11, 10, 10, 11])
      },
      rings: {format: 'uint32', values: new Uint32Array([0, 4, 7])},
      polygons: {format: 'uint32', values: new Uint32Array([0, 1, 2])},
      features: {format: 'uint32', values: new Uint32Array([0, 2])}
    },
    outputs: {
      positions: {format: 'float32x2', length: 16},
      offsets: {format: 'uint32', length: 3},
      ringCount: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1},
      sourceRows: {format: 'uint32', length: 1},
      featureCount: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPURectangleClip({
        geometryType: 'polygons',
        positions: inputs['positions'] as never,
        ringOffsets: inputs['rings'] as never,
        polygonOffsets: inputs['polygons'] as never,
        featureOffsets: inputs['features'] as never,
        parameters,
        output: {
          positions: outputs['positions'] as never,
          lineOffsets: outputs['offsets'] as never,
          status: {count: outputs['ringCount'] as never, overflow: outputs['overflow'] as never},
          polygonFeatures: {
            sourceRows: outputs['sourceRows'] as never,
            status: {count: outputs['featureCount'] as never}
          }
        }
      })
  });
  try {
    const result = await fixture.run(
      getGPURectangleClipParameterValues({minX: 0, minY: 0, maxX: 2, maxY: 2})
    );
    const points = Array.from({length: 4}, (_, index) =>
      result['positions'].slice(index * 2, index * 2 + 2)
    );
    const start = points.findIndex(point => point[0] === 0 && point[1] === 0);
    expect([...points.slice(start), ...points.slice(0, start)]).toEqual([
      [0, 0],
      [2, 0],
      [2, 2],
      [0, 2]
    ]);
    expect(result['offsets']).toEqual([0, 4, 4]);
    expect(result['ringCount']).toEqual([2]);
    expect(result['featureCount']).toEqual([1]);
    expect(result['sourceRows']).toEqual([0]);
    expect(result['overflow']).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});
