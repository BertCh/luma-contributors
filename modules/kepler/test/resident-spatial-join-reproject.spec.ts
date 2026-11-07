// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {
  ResidentSpatialJoinPointReprojector,
  transformResidentSpatialJoinPoint
} from '../../../examples/deck/kepler-compute/src/resident-spatial-join-reproject';

it('GPU point reprojection matches the CPU reference within Float32 error', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const points = Float32Array.from([0, 0, -122.4, 37.8, 139.7, 35.7, 179.9, -84, -0.1, 51.5]);
  const rowCount = points.length / 2;
  for (const transform of [
    'longitude-latitude-to-mercator',
    'mercator-to-longitude-latitude'
  ] as const) {
    const input =
      transform === 'longitude-latitude-to-mercator'
        ? points
        : Float32Array.from(
            Array.from({length: rowCount}, (_, row) =>
              transformResidentSpatialJoinPoint(
                'longitude-latitude-to-mercator',
                points[row * 2],
                points[row * 2 + 1]
              )
            ).flat()
          );
    const buffer = device.createBuffer({
      data: input,
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    const reprojector = new ResidentSpatialJoinPointReprojector(device, transform, rowCount);
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass({});
    reprojector.encode(pass, buffer);
    pass.end();
    device.submit(encoder.finish());
    const bytes = await buffer.readAsync(0, input.byteLength);
    const output = new Float32Array(bytes.slice().buffer);
    for (let row = 0; row < rowCount; row++) {
      const [x, y] = transformResidentSpatialJoinPoint(
        transform,
        input[row * 2],
        input[row * 2 + 1]
      );
      const tolerance = transform === 'longitude-latitude-to-mercator' ? 8 : 1e-3;
      expect(Math.abs(output[row * 2] - x)).toBeLessThan(tolerance);
      expect(Math.abs(output[row * 2 + 1] - y)).toBeLessThan(tolerance);
    }
    reprojector.destroy();
    buffer.destroy();
  }
});
