// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUVertexSnap,
  GPU_VERTEX_SNAP_NO_REFERENCE
} from '../../../src/gpu-spatial-analysis/vertex-snap/gpu-vertex-snap';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';
import {requireTestDevice} from '../../utils/require-test-device';

it('GEO-130 projected vertex snap refreshes references and tolerance on the same graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    // Metre coordinates are exact integers in float32, including these projected offsets.
    inputs: {
      positions: {format: 'float32x2', values: Float32Array.of(500000, 4000000, 500010, 4000010)},
      references: {format: 'float32x2', values: Float32Array.of(500003, 4000004, 500010, 4000012)}
    },
    outputs: {
      positions: {format: 'float32x2', length: 2},
      references: {format: 'uint32', length: 2}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPUVertexSnap({
        spatialContext: {coordinateSpace: 'planar', metric: 'native', units: 'meters'},
        positions: inputs['positions'] as never,
        referencePositions: inputs['references'] as never,
        parameters,
        output: {
          positions: outputs['positions'] as never,
          referenceRows: outputs['references'] as never
        }
      })
  });
  try {
    // Distances: row 0 -> reference 0 is sqrt(3^2+4^2)=5; row 1 -> reference 1 is 2.
    const first = await fixture.run(Float32Array.of(6, 0, 0, 0));
    expect(first['positions']).toEqual([500003, 4000004, 500010, 4000012]);
    expect(first['references']).toEqual([0, 1]);
    fixture.writeInput('references', Float32Array.of(500001, 4000000, 500010, 4000014));
    // Now distances 1 and 4; tolerance 2 snaps only row 0. The second row must clear its old ID.
    const second = await fixture.run(Float32Array.of(2, 0, 0, 0));
    expect(second['positions']).toEqual([500001, 4000000, 500010, 4000010]);
    expect(second['references']).toEqual([0, GPU_VERTEX_SNAP_NO_REFERENCE]);
    expect(fixture.getCompileCount()).toBe(0);
  } finally {
    fixture.destroy();
  }
});
