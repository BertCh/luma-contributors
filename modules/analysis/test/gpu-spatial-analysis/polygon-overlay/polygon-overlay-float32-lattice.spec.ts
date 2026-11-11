// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Math correctness review 2026-11-10, polygon-topology-06: the overlay lattice must not be finer
// than float32 can store at the coordinates of the data. Expected values are hand derived.

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import {FIXTURE_CAPACITY, runOverlay, type OverlayOperand} from './overlay-gpu-harness';

type Ring = [number, number][];

function getDoubledArea(ring: Ring, originX: number): number {
  let doubled = 0;
  for (let index = 0; index < ring.length; index++) {
    const [x0, y0] = ring[index];
    const [x1, y1] = ring[(index + 1) % ring.length];
    doubled += (x0 - originX) * y1 - (x1 - originX) * y0;
  }
  return doubled;
}

/** Ring vertices without a repeated closing vertex. */
function getOpenRing(ring: Ring): Ring {
  const last = ring[ring.length - 1];
  return ring.length > 1 && ring[0][0] === last[0] && ring[0][1] === last[1]
    ? ring.slice(0, -1)
    : ring;
}

it('GPUPolygonOverlay keeps distinct lattice vertices distinct far from zero (review 06)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // x near 2^23, where one float32 step is 1. The strip's slanted sides (x = 1 + y / 2 and
  // x = 2 + y / 2 past the offset) cross the rectangle's bottom edge y = 1 at x = offset + 1.5
  // and offset + 2.5; the intersection is the part of the strip above that edge.
  const offset = 2 ** 23;
  const strip: OverlayOperand = [
    [
      [
        [
          [offset + 1, 0],
          [offset + 2, 0],
          [offset + 4, 4],
          [offset + 3, 4]
        ]
      ]
    ]
  ];
  const rectangle: OverlayOperand = [
    [
      [
        [
          [offset, 1],
          [offset + 6, 1],
          [offset + 6, 5],
          [offset, 5]
        ]
      ]
    ]
  ];
  const readout = await runOverlay(
    device,
    {
      implementation: 'current',
      operation: 'intersection',
      left: strip,
      right: rectangle,
      // A grid of 2^-10 is finer than float32 can store here. Sized from the extent alone (6
      // units), the lattice held both crossings as distinct points, and both were then written
      // as the float32 value offset + 2 (round to even): a ring with two equal vertices.
      vertexTolerance: 2 ** -10,
      capacity: FIXTURE_CAPACITY
    },
    'float32-lattice'
  );
  expect(readout.overflow).toBe(0);
  expect(readout.featureCount).toBe(1);
  expect(readout.rings.length).toBe(1);
  const ring = getOpenRing(readout.rings[0].vertices);
  expect(new Set(ring.map(([x, y]) => `${x},${y}`)).size).toBe(ring.length);
  expect(ring.length).toBe(4);
  // On a lattice of one float32 step the crossings round half up to offset + 2 and offset + 3,
  // and the right side is rerouted through both of those hot pixels: the part of the strip above
  // y = 1 is the parallelogram (2,1), (3,1), (4,4), (3,4) of area 3, as it is before rounding.
  expect(ring.map(([x, y]) => `${x - offset},${y}`).sort()).toEqual(['2,1', '3,1', '3,4', '4,4']);
  expect(Math.abs(getDoubledArea(ring, offset))).toBe(6);
  device.destroy?.();
});

it('GPUPolygonOverlay lattice is unchanged where the extent decides it', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The same shapes at the origin: every coordinate is within four extents of zero, so the grid
  // of 2^-10 stands and the crossings keep their half-unit positions exactly.
  const strip: OverlayOperand = [
    [
      [
        [
          [1, 0],
          [2, 0],
          [4, 4],
          [3, 4]
        ]
      ]
    ]
  ];
  const rectangle: OverlayOperand = [
    [
      [
        [
          [0, 1],
          [6, 1],
          [6, 5],
          [0, 5]
        ]
      ]
    ]
  ];
  const readout = await runOverlay(
    device,
    {
      implementation: 'current',
      operation: 'intersection',
      left: strip,
      right: rectangle,
      vertexTolerance: 2 ** -10,
      capacity: FIXTURE_CAPACITY
    },
    'float32-lattice-origin'
  );
  expect(readout.overflow).toBe(0);
  expect(readout.rings.length).toBe(1);
  const ring = getOpenRing(readout.rings[0].vertices);
  expect(ring.map(([x, y]) => `${x},${y}`).sort()).toEqual(['1.5,1', '2.5,1', '3,4', '4,4']);
  expect(Math.abs(getDoubledArea(ring, 0))).toBe(6);
  device.destroy?.();
});
