// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUOffsetCurve,
  getGPUOffsetCurveParameterValues
} from '../../../src/gpu-spatial-analysis/outline-geometry/gpu-offset-curve';
import {requireTestDevice} from '../../utils/require-test-device';
import {createGeometryFixture} from './geometry-fixture';

it('offset curve pins each bevel count instead of only their sum', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      positions: {format: 'float32x2', values: Float32Array.of(0, 0, 1, 0, 1, -1)},
      offsets: {format: 'uint32', values: Uint32Array.of(0, 3)}
    },
    outputs: {
      positions: {format: 'float32x2', length: 6},
      counts: {format: 'uint32', length: 3}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPUOffsetCurve({
        positions: inputs['positions'] as never,
        lineOffsets: inputs['offsets'] as never,
        parameters,
        joinStyle: 'bevel',
        output: {positions: outputs['positions'] as never, counts: outputs['counts'] as never}
      })
  });
  try {
    // One shifted point at each open endpoint; the outside right turn emits both bevel ends.
    const result = await fixture.run(getGPUOffsetCurveParameterValues({distance: 1}));
    expect(result['counts']).toEqual([1, 2, 1]);
    // Reversing the offset makes that corner an inside intersection (one point).
    const reversed = await fixture.run(getGPUOffsetCurveParameterValues({distance: -1}));
    expect(reversed['counts']).toEqual([1, 1, 1]);
  } finally {
    fixture.destroy();
  }
});

for (const capacity of [2, 4]) {
  it(`compact offset publishes line requirements independently of vertex capacity ${capacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createGeometryFixture(device, {
      inputs: {
        positions: {format: 'float32x2', values: Float32Array.of(0, 0, 1, 0, 1, -1)},
        offsets: {format: 'uint32', values: Uint32Array.of(0, 3)}
      },
      outputs: {
        positions: {format: 'float32x2', length: capacity},
        offsets: {format: 'uint32', length: 2},
        count: {format: 'uint32', length: 1},
        requiredCount: {format: 'uint32', length: 1},
        overflow: {format: 'uint32', length: 1},
        requiredVertexCount: {format: 'uint32', length: 1}
      },
      create: ({inputs, outputs, parameters}) =>
        new GPUOffsetCurve({
          positions: inputs['positions'] as never,
          lineOffsets: inputs['offsets'] as never,
          parameters,
          joinStyle: 'bevel',
          output: {
            layout: 'compact',
            positions: outputs['positions'] as never,
            lineOffsets: outputs['offsets'] as never,
            status: {
              count: outputs['count'] as never,
              requiredCount: outputs['requiredCount'] as never,
              overflow: outputs['overflow'] as never,
              requiredVertexCount: outputs['requiredVertexCount'] as never
            }
          }
        })
    });
    try {
      // One input line remains one line, even when its four generated vertices are truncated.
      const result = await fixture.run(getGPUOffsetCurveParameterValues({distance: 1}));
      expect(result['count']).toEqual([1]);
      expect(result['requiredCount']).toEqual([1]);
      expect(result['requiredVertexCount']).toEqual([4]);
      expect(result['overflow']).toEqual([capacity === 2 ? 1 : 0]);
    } finally {
      fixture.destroy();
    }
  });
}
