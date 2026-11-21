// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPULineDensity,
  getGPULineDensityParameterValues
} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-density';
import {createGeometryFixture} from '../outline-geometry/geometry-fixture';

it('PTS-118 counts geographic seam uncertainty and clears it on resubmission', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGeometryFixture(device, {
    inputs: {
      positions: {
        format: 'float32x2',
        values: new Float32Array([179, 0, -179, 0, -179, 0, 179, 0, 10, 0, 12, 0])
      },
      offsets: {format: 'uint32', values: new Uint32Array([0, 2, 4, 6])}
    },
    outputs: {
      lengths: {format: 'float32', length: 360},
      overflow: {format: 'uint32', length: 1},
      seams: {format: 'uint32', length: 1}
    },
    create: ({inputs, outputs, parameters}) =>
      new GPULineDensity({
        positions: inputs['positions'] as never,
        lineOffsets: inputs['offsets'] as never,
        columns: 360,
        rows: 1,
        parameters,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'meters'
        },
        output: {
          lengths: outputs['lengths'] as never,
          status: {
            overflow: outputs['overflow'] as never,
            antimeridianUncertainCount: outputs['seams'] as never
          }
        }
      })
  });
  try {
    const result = await fixture.run(
      getGPULineDensityParameterValues({minX: -180, minY: -0.5, cellWidth: 1, cellHeight: 1})
    );
    expect(result['seams']).toEqual([2]);
    expect(result['lengths']).toHaveLength(360);
    expect(result['lengths'].flatMap((value, cell) => (value > 0 ? [cell] : []))).toEqual(
      Array.from({length: 358}, (_, index) => index + 1)
    );
    expect(result['overflow']).toEqual([0]);
    // Independent interval/grid intersection: [-179,179] overlaps cells 1..358 with positive
    // width twice, and [10,12] adds one degree to cells 190,191. Endpoints carry zero length.
    for (let cell = 0; cell < 360; cell++) {
      const degrees = cell >= 1 && cell <= 358 ? (cell === 190 || cell === 191 ? 3 : 2) : 0;
      const actual = result['lengths'][cell];
      expect(Number.isFinite(actual)).toBe(true);
      expect(Math.abs(actual - ((degrees * Math.PI) / 180) * 6371008.8)).toBeLessThan(2000);
    }
    // Worst-case 357 f32 additions of tDelta cost <=357*2^-24 in t, or 850 metres per
    // wide segment; allow 2000 metres for their sum and endpoint rounding. The exact extent and
    // telescoping integrated equatorial length below are substantially stricter independent checks.
    const integrated = result['lengths'].reduce((sum, value) => sum + value, 0);
    expect(Math.abs(integrated - ((718 * Math.PI) / 180) * 6371008.8)).toBeLessThan(100);
    fixture.writeInput('positions', new Float32Array([10, 0, 12, 0, 10, 0, 12, 0, 10, 0, 12, 0]));
    const recovered = await fixture.run();
    expect(recovered['seams']).toEqual([0]);
    expect(recovered['lengths']).toHaveLength(360);
    expect(recovered['lengths'].filter(value => value > 0)).toHaveLength(2);
    expect(fixture.getCompileCount()).toBe(0);
  } finally {
    fixture.destroy();
  }
});
