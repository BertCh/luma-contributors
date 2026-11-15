// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {runMakeValid, type Point} from './topology-property-harness';
import {requireTestDevice} from '../../utils/require-test-device';

function getRectangle(x: number, y: number, width: number, height: number): Point[] {
  return [
    [x, y],
    [x + width, y],
    [x + width, y + height],
    [x, y + height]
  ];
}

function getArea(rings: readonly number[][][]): number {
  return rings.reduce((sum, ring, ringIndex) => {
    let doubledArea = 0;
    for (let vertex = 0; vertex < ring.length; vertex++) {
      const next = ring[(vertex + 1) % ring.length];
      doubledArea += ring[vertex][0] * next[1] - next[0] * ring[vertex][1];
    }
    return sum + ((ringIndex === 0 ? 1 : -1) * Math.abs(doubledArea)) / 2;
  }, 0);
}

it('GPUMakeValid indexes many features without mixing uint32 source IDs that share an f32 value', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rings: Point[][] = [];
  const sourceIds: number[] = [];
  for (let feature = 0; feature < 17; feature++) {
    // Every annulus is independent, including several at the same ray height. IDs close to
    // uint32's upper bound intentionally have the same float32 representation.
    const x = (feature % 5) * 16;
    const y = Math.floor(feature / 5) * 16;
    rings.push(getRectangle(x, y, 10, 10), getRectangle(x + 2, y + 2, 6, 6).reverse());
    sourceIds.push(0xfffffffe - feature);
  }
  // 8 edges per annulus, with ample unwritten noded capacity and an incomplete final tree level.
  const result = await runMakeValid(
    device,
    rings,
    {
      segments: 257,
      intersections: 256,
      rings: 73,
      vertices: 600
    },
    {ringOffsets: Array.from({length: 18}, (_, index) => 2 * index), sourceIds}
  );
  expect(result.overflow).toBe(0);
  expect(result.candidateOverflow).toBe(0);
  expect(result.count).toBe(17);
  expect([...result.sourceIds].sort((first, second) => first - second)).toEqual(
    [...sourceIds].sort((first, second) => first - second)
  );
  for (const polygon of result.polygons) {
    expect(polygon).toHaveLength(2);
    // Hand-derived annulus area 10*10 - 6*6.
    expect(getArea(polygon)).toBe(64);
  }
  device.destroy?.();
});

it('GPUMakeValid preserves coincident ray heights, one-step probes and empty source rings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const step = 2 ** -23;
  const ring: Point[] = [
    [0, 1 + step],
    [0, 1],
    [8, 1],
    [8, 9],
    [0, 9]
  ];
  const result = await runMakeValid(
    device,
    [[], ring, []],
    {
      segments: 33,
      intersections: 16,
      rings: 9,
      vertices: 64,
      vertexTolerance: 2 ** -26
    },
    {ringOffsets: [0, 1, 2, 3], sourceIds: [71, 72, 73]}
  );
  expect(result.overflow).toBe(0);
  expect(result.count).toBe(1);
  expect(result.sourceIds).toEqual([72]);
  expect(getArea(result.polygons[0])).toBeCloseTo(64, 6);
  const empty = await runMakeValid(device, [[[0, 0]]], {
    segments: 3,
    intersections: 1,
    rings: 3,
    vertices: 8
  });
  expect(empty.count).toBe(0);
  expect(empty.overflow).toBe(0);
  device.destroy?.();
});
