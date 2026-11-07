// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import proj4 from 'proj4';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../analysis/test/utils/require-test-device';
import {normalizeProjectionLabel} from '../../../examples/deck/kepler-compute/src/resident-spatial-join-crs';
import {
  ResidentSpatialJoinPointReprojector,
  getResidentSpatialJoinPointTransform
} from '../../../examples/deck/kepler-compute/src/resident-spatial-join-reproject';

const UTM_33N = '+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs';
const UTM_34N = 'EPSG:32634';
const UTM_55S = '+proj=utm +zone=55 +south +datum=WGS84 +units=m +no_defs';
const TMERC =
  '+proj=tmerc +lat_0=49 +lon_0=-2 +k=0.9996012717 +x_0=400000 +y_0=-100000 +ellps=GRS80 +units=m +no_defs';

async function runReprojection(
  device: Awaited<ReturnType<typeof getWebGPUTestDevice>>,
  from: string,
  to: string,
  input: Float32Array
): Promise<Float32Array> {
  const points = normalizeProjectionLabel(from)!;
  const polygons = normalizeProjectionLabel(to)!;
  const transform = getResidentSpatialJoinPointTransform(points, polygons);
  expect(transform).toBe('projected');
  const buffer = device!.createBuffer({
    data: input,
    usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
  });
  const reprojector = new ResidentSpatialJoinPointReprojector(
    device!,
    transform as 'projected',
    input.length / 2,
    {points, polygons}
  );
  try {
    const encoder = device!.createCommandEncoder();
    const pass = encoder.beginComputePass({});
    reprojector.encode(pass, buffer);
    pass.end();
    device!.submit(encoder.finish());
    const bytes = await buffer.readAsync(0, input.byteLength);
    return new Float32Array(bytes.slice().buffer);
  } finally {
    reprojector.destroy();
    buffer.destroy();
  }
}

it('GPU general transverse Mercator and UTM reprojection matches proj4 within Float32 error', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases: {from: string; to: string; lonLat: [number, number][]; tolerance: number}[] = [
    {
      from: 'EPSG:4326',
      to: UTM_33N,
      lonLat: [
        [15, 52],
        [10.7, 59.9],
        [12, 0.5],
        [17.9, 70.1]
      ],
      tolerance: 3
    },
    {
      from: UTM_55S,
      to: 'EPSG:4326',
      lonLat: [
        [147, -37.8],
        [150.5, -33.9]
      ],
      tolerance: 1e-4
    },
    {
      from: 'EPSG:4326',
      to: TMERC,
      lonLat: [
        [-2, 49],
        [-1.5, 51.5],
        [-4.5, 53.9]
      ],
      tolerance: 3
    },
    // Between zones through longitude/latitude.
    {
      from: UTM_33N,
      to: UTM_34N,
      lonLat: [
        [15, 52],
        [17, 49]
      ],
      tolerance: 6
    },
    {
      from: 'EPSG:3857',
      to: UTM_33N,
      lonLat: [
        [14, 50],
        [11, 60]
      ],
      tolerance: 6
    }
  ];
  for (const testCase of cases) {
    const converter = proj4(
      testCase.from.startsWith('+') || testCase.from.startsWith('EPSG')
        ? testCase.from
        : testCase.from,
      testCase.to
    );
    const source = proj4('EPSG:4326', testCase.from);
    const input = Float32Array.from(
      testCase.lonLat.flatMap(([longitude, latitude]) => source.forward([longitude, latitude]))
    );
    const output = await runReprojection(device, testCase.from, testCase.to, input);
    testCase.lonLat.forEach((_, row) => {
      // Reference: proj4 on the exact Float32 inputs the kernel saw.
      const [expectedX, expectedY] = converter.forward([input[row * 2], input[row * 2 + 1]]);
      expect(Math.abs(output[row * 2] - expectedX), `${testCase.to} x row ${row}`).toBeLessThan(
        testCase.tolerance
      );
      expect(Math.abs(output[row * 2 + 1] - expectedY), `${testCase.to} y row ${row}`).toBeLessThan(
        testCase.tolerance
      );
    });
  }
});
