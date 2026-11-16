// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUOutlineGeometry,
  getGPUOutlineGeometryParameterValues
} from '../../../src/gpu-spatial-analysis/outline-geometry/gpu-outline-geometry';
import {createGeometryFixture} from './geometry-fixture';
import {requireTestDevice} from '../../utils/require-test-device';

it('publishes compact triangles, end offsets and fresh zero-distance status', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      positions: {values: new Float32Array([0, 0, 2, 0]), format: 'float32x2'},
      lines: {values: new Uint32Array([0, 2]), format: 'uint32'}
    },
    outputs: {
      positions: {format: 'float32x2', length: 21},
      offsets: {format: 'uint32', length: 3},
      sources: {format: 'uint32', length: 21},
      count: {format: 'uint32', length: 1},
      required: {format: 'uint32', length: 1},
      overflow: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPUOutlineGeometry({
        geometryType: 'lines',
        joinSegments: 4,
        positions: inputs['positions'] as never,
        lineOffsets: inputs['lines'] as never,
        parameters,
        output: {
          layout: 'compact',
          positions: outputs['positions'] as never,
          vertexOffsets: outputs['offsets'] as never,
          sourceRows: outputs['sources'] as never,
          status: {
            count: outputs['count'] as never,
            requiredCount: outputs['required'] as never,
            overflow: outputs['overflow'] as never
          }
        }
      })
  });
  try {
    const result = await fixture.run(getGPUOutlineGeometryParameterValues({distance: 1}));
    const expected = [
      0, 0, 1, 0, 0, 1, 0, 0, 0, 1, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 1, 0, 0, 1, 0, -1, 2,
      -1, 0, 1, 2, -1, 2, 1, 2, 0, 3, 0, 2, 1
    ];
    expected.forEach((value, index) => expect(result['positions'][index]).toBeCloseTo(value, 5));
    expect(result['offsets']).toEqual([0, 18, 21]);
    expect(result['sources']).toEqual([...Array(18).fill(0), 1, 1, 1]);
    expect(result['required']).toEqual([30]);
    expect(result['count']).toEqual([21]);
    expect(result['overflow']).toEqual([1]);
    const zero = await fixture.run(getGPUOutlineGeometryParameterValues({distance: 0}));
    expect(zero['count']).toEqual([0]);
    expect(zero['required']).toEqual([0]);
    expect(zero['overflow']).toEqual([0]);
    expect(zero['offsets']).toEqual([0, 0, 0]);
  } finally {
    fixture.destroy();
  }
});
