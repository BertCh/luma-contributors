// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUSphericalConvexIntersection} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-intersection';
import {GPUSphericalConvexContains} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-convex-contains';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {createGlobalCoordinateFixture} from './global-coordinate-fixture';
import {expectConvexReadbackLengths} from './global-coordinate-output-assertions';

function square(
  minimumFirst: number,
  maximumFirst: number,
  minimumSecond: number,
  maximumSecond: number
) {
  return [
    [1, minimumFirst, minimumSecond],
    [1, maximumFirst, minimumSecond],
    [1, maximumFirst, maximumSecond],
    [1, minimumFirst, maximumSecond]
  ].flatMap(point => point.map(value => value / Math.hypot(...point)));
}

it('convex spherical partial intersection and disjoint pair match independent gnomonic clipping', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGlobalCoordinateFixture(device);
  const positions = fixture.output('float32x3', 16);
  const counts = fixture.output('uint32x2', 2);
  const status = fixture.output('uint32', 2);
  const measures = fixture.output('float32x4', 2);
  fixture.graph.add(
    new GPUSphericalConvexIntersection({
      firstPositions: fixture.input(
        'float32x3',
        Array(2)
          .fill(square(-0.6, 0.4, -0.5, 0.5))
          .flat()
      ),
      secondPositions: fixture.input('float32x3', [
        ...square(-0.2, 0.8, -0.3, 0.7),
        ...square(1, 1.4, 0, 0.4)
      ]),
      firstOffsets: fixture.input('uint32', [0, 4, 8]),
      secondOffsets: fixture.input('uint32', [0, 4, 8]),
      capacity: 8,
      output: {
        positions: positions.view,
        counts: counts.view,
        status: status.view,
        measures: measures.view
      }
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(status.buffer, 2)).toEqual([0, 0]);
    expect(await readUint32(counts.buffer, 4)).toEqual([4, 4, 0, 0]);
    const actual = await readFloat32(positions.buffer, 48);
    const areas = await readFloat32(measures.buffer, 8);
    expectConvexReadbackLengths(actual, areas);
    expect(actual.every(Number.isFinite)).toBe(true);
    // In x=1, intersect the two rectangles by max of lower bounds/min of upper bounds:
    // [-.2,.4] x [-.3,.5]. Inverse projection and the signed spherical-triangle sum
    // about (1,0,0) give these literals (/usr/bin/python3 convex-reference.py in scratch).
    const expected = [
      [0.9407208683835973, -0.18814417367671946, -0.28221626051507914],
      [0.8944271909999159, 0.35777087639996635, -0.2683281572999747],
      [0.8421519210665189, 0.3368607684266076, 0.42107596053325946],
      [0.8804509063256238, -0.1760901812651248, 0.4402254531628119]
    ];
    for (const point of expected)
      expect(
        Array.from({length: 4}, (_, row) =>
          Math.hypot(...point.map((value, axis) => value - actual[row * 3 + axis]))
        ).filter(distance => distance < 2e-6)
      ).toHaveLength(1);
    expect(actual.slice(12).every(value => value === 0)).toBe(true);
    expect(areas.every(Number.isFinite)).toBe(true);
    expect(areas[0]).toBeCloseTo(0.41798823397060403, 5);
    expect(areas[4]).toBe(0);
    expect(areas[5]).toBe(0);
    expect(areas[6]).toBe(0);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('spherical buffer queries retain the looser squared-length check (DGGS-112)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createGlobalCoordinateFixture(device);
  const contains = fixture.output('uint32', 4);
  const status = fixture.output('uint32', 4);
  fixture.graph.add(
    new GPUSphericalConvexContains({
      positions: fixture.input('float32x3', square(-0.5, 0.5, -0.5, 0.5)),
      offsets: fixture.input('uint32', [0, 4]),
      // 1.000075 has length error <1e-4 but squared-length error >1e-4: it must fail.
      queries: fixture.input('float32x3', [1, 0, 0, 1.00002, 0, 0, 0.99998, 0, 0, 1.000075, 0, 0]),
      polygonIndices: fixture.input('uint32', [0, 0, 0, 0]),
      output: {contains: contains.view, status: status.view}
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(status.buffer, 4)).toEqual([0, 0, 0, 1]);
    expect(await readUint32(contains.buffer, 4)).toEqual([1, 1, 1, 0]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
