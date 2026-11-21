// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {describe, expect, it} from 'vitest';
import {addStreetProfilesRecipe} from '../../../src/gpu-spatial-analysis/urban-morphology/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {expectUrbanMissingValues} from './urban-ecology-assertions';
import {RecipeTestFixture} from '../recipes/recipe-harness';

// Exact facade lengths of `addStreetProfilesRecipe` at coordinates that are not small integers.
// Every expected value is a hand-computed length of the untransformed geometry (a rigid motion
// keeps lengths), or a closed form evaluated in float64 on the float32 inputs.

type FacadeCase = {
  /** Street start and end: x0, y0, x1, y1. */
  street: number[];
  /** Obstacle segments, four numbers per row. */
  segments: number[];
  /** Building row of each segment. */
  owners: number[];
  maximumDistance: number;
  buildingCount?: number;
};

type Motion = {angle?: number; cosine?: number; sine?: number; scale?: number; offset?: number[]};

/** Scales, rotates and then translates a flat list of x, y pairs, in float64. */
function moveCoordinates(coordinates: number[], motion: Motion): number[] {
  const cosine = motion.cosine ?? Math.cos(motion.angle ?? 0);
  const sine = motion.sine ?? Math.sin(motion.angle ?? 0);
  const scale = motion.scale ?? 1;
  const [offsetX, offsetY] = motion.offset ?? [0, 0];
  const moved: number[] = [];
  for (let index = 0; index < coordinates.length; index += 2) {
    const x = scale * coordinates[index];
    const y = scale * coordinates[index + 1];
    moved.push(cosine * x - sine * y + offsetX, sine * x + cosine * y + offsetY);
  }
  return moved;
}

function moveCase(facadeCase: FacadeCase, motion: Motion): FacadeCase {
  return {
    street: moveCoordinates(facadeCase.street, motion),
    segments: moveCoordinates(facadeCase.segments, motion),
    owners: facadeCase.owners,
    maximumDistance: facadeCase.maximumDistance * (motion.scale ?? 1)
  };
}

/** Runs the recipe and returns left and right facade length per segment and per building. */
async function getFacades(
  facadeCase: FacadeCase,
  obstacleIndex: 'brute-force' | 'bvh'
): Promise<{lengths: number[]; frontage: number[]}> {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = new RecipeTestFixture(device, `street-facades-${obstacleIndex}`);
  try {
    const segmentCount = facadeCase.segments.length / 4;
    const buildingCount = facadeCase.buildingCount ?? Math.max(...facadeCase.owners) + 1;
    const [startX, startY, endX, endY] = facadeCase.street;
    const lengths = fixture.output('facade-lengths', 'float32x2', segmentCount);
    const frontage = fixture.output('facade-frontage', 'float32x2', buildingCount);
    const result = addStreetProfilesRecipe(fixture.graph, {
      positions: fixture.input(
        'positions',
        new Float32Array([0.5 * (startX + endX), 0.5 * (startY + endY)]),
        'float32x2',
        1
      ),
      normals: fixture.input(
        'normals',
        new Float32Array([startY - endY, endX - startX]),
        'float32x2',
        1
      ),
      segments: fixture.input(
        'segments',
        Float32Array.from(facadeCase.segments),
        'float32x4',
        segmentCount
      ),
      segmentBuildingIds: fixture.input(
        'owners',
        Uint32Array.from(facadeCase.owners),
        'uint32',
        segmentCount
      ),
      buildingHeights: fixture.input(
        'heights',
        new Float32Array(buildingCount).fill(6),
        'float32',
        buildingCount
      ),
      maximumDistance: facadeCase.maximumDistance,
      sampleSpacing: 1,
      streetSegment: fixture.input('street', Float32Array.from(facadeCase.street), 'float32x4', 1),
      obstacleIndex,
      output: {facadeLengths: lengths.view, facadeFrontage: frontage.view}
    });
    expect(result.facades!.lengths).toBe(lengths.view);
    expect(result.facades!.frontage).toBe(frontage.view);
    expect((result.facades!.lengths.length * result.facades!.lengths.rowByteLength) / 4).toBe(
      2 * segmentCount
    );
    expect((result.facades!.frontage.length * result.facades!.frontage.rowByteLength) / 4).toBe(
      2 * buildingCount
    );
    fixture.run();
    return {
      lengths: await fixture.readFloat32(lengths, 2 * segmentCount),
      frontage: await fixture.readFloat32(frontage, 2 * buildingCount)
    };
  } finally {
    fixture.destroy();
  }
}

function expectRelativelyClose(actual: number[], expected: number[], relative = 1e-4) {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, index) => {
    expect(Number.isFinite(actual[index])).toBe(true);
    expect(
      Math.abs(actual[index] - value),
      `value ${index}: ${actual[index]} against ${value}`
    ).toBeLessThanOrEqual(relative * Math.max(Math.abs(value), 1));
  });
}

/** The fixture of the integer-coordinate facade spec: a street with two walls left and one right. */
const THREE_WALLS: FacadeCase = {
  street: [-4, 0, 4, 0],
  segments: [-10, 2, 10, 2, -10, 3, 10, 3, -10, -3, 10, -3],
  owners: [0, 1, 2],
  maximumDistance: 10
};
/** Per building, left then right: the near left wall and the right wall face 8 units of street. */
const THREE_WALLS_FRONTAGE = [8, 0, 0, 0, 0, 8];

const GENERIC_MOTION: Motion = {angle: 0.3, offset: [317.3, -152.9]};

describe.each(['brute-force', 'bvh'] as const)('exact facade lengths, %s', obstacleIndex => {
  it('counts an unobstructed facade whose coordinates are not integers', async () => {
    // Street direction (100, 25.2); the facade lies 14.1 to its left, between 24 and 55 along it.
    // Its length is hypot(60.4 - 30.1, 47.9 - 40.2) = sqrt(977.38) = 31.2631.
    const facades = await getFacades(
      {
        street: [10.3, 20.7, 110.3, 45.9],
        segments: [30.1, 40.2, 60.4, 47.9],
        owners: [0],
        maximumDistance: 30
      },
      obstacleIndex
    );
    expectRelativelyClose(facades.lengths, [31.2631, 0]);
    expectRelativelyClose(facades.frontage, [31.2631, 0]);
  });

  it('gives the frontage of the integer fixture after a rotation by 0.3 radians', async () => {
    const facades = await getFacades(moveCase(THREE_WALLS, {angle: 0.3}), obstacleIndex);
    expectRelativelyClose(facades.frontage, THREE_WALLS_FRONTAGE);
  });

  it('gives the same frontage at projected-metre coordinates (500000, 4500000)', async () => {
    // float32 resolves 0.5 near 4,500,000. A scale of 12.5 and the 3-4-5 rotation put every
    // coordinate on that grid, so the geometry is the scaled fixture exactly and each length is
    // 12.5 times the fixture's.
    const moved = moveCase(THREE_WALLS, {
      cosine: 0.6,
      sine: 0.8,
      scale: 12.5,
      offset: [500000, 4500000]
    });
    for (const coordinate of [...moved.street, ...moved.segments]) {
      expect(Math.fround(coordinate)).toBe(coordinate);
    }
    const facades = await getFacades(moved, obstacleIndex);
    expectRelativelyClose(
      facades.frontage,
      THREE_WALLS_FRONTAGE.map(length => 12.5 * length)
    );
  });

  it('measures the float32 geometry it is given at (500000, 4500000) after a generic rotation', async () => {
    // Rounding to float32 moves each coordinate by up to 0.25 here, so the expected value comes
    // from the rounded input. A wall that spans the corridor with nothing in front of it shows
    // streetLength / cos(angle between wall and street) = streetLength * |wall| / |wall . street axis|.
    const moved = moveCase(THREE_WALLS, {angle: 0.3, offset: [500000, 4500000]});
    const street = moved.street.map(Math.fround);
    const segments = moved.segments.map(Math.fround);
    const streetLength = Math.hypot(street[2] - street[0], street[3] - street[1]);
    const axis = [(street[2] - street[0]) / streetLength, (street[3] - street[1]) / streetLength];
    const getSpanningWallLength = (row: number) => {
      const deltaX = segments[4 * row + 2] - segments[4 * row];
      const deltaY = segments[4 * row + 3] - segments[4 * row + 1];
      return (
        (streetLength * Math.hypot(deltaX, deltaY)) / Math.abs(deltaX * axis[0] + deltaY * axis[1])
      );
    };
    const facades = await getFacades(moved, obstacleIndex);
    expectRelativelyClose(facades.frontage, [
      getSpanningWallLength(0),
      0,
      0,
      0,
      0,
      getSpanningWallLength(2)
    ]);
  });

  it('counts two walls that meet at a corner once each', async () => {
    // A chevron (-3, 4) - (0, 2) - (3, 4) of one building, all of it in view of the street
    // (-4, 0) - (4, 0). Each wall is sqrt(9 + 4) = 3.605551 long.
    const facades = await getFacades(
      moveCase(
        {
          street: [-4, 0, 4, 0],
          segments: [-3, 4, 0, 2, 0, 2, 3, 4],
          owners: [0, 0],
          maximumDistance: 10
        },
        GENERIC_MOTION
      ),
      obstacleIndex
    );
    expectRelativelyClose(facades.lengths, [3.605551, 0, 3.605551, 0]);
    expectRelativelyClose(facades.frontage, [7.211103, 0]);
  });

  it('removes exactly the stretch hidden behind a nearer wall', async () => {
    // Wall 0 at y = 5 spans the 8 units of street; wall 1 at y = 2 covers x from -1 to 2 and
    // hides 3 units of wall 0.
    const facades = await getFacades(
      moveCase(
        {
          street: [-4, 0, 4, 0],
          segments: [-10, 5, 10, 5, -1, 2, 2, 2],
          owners: [0, 1],
          maximumDistance: 10
        },
        GENERIC_MOTION
      ),
      obstacleIndex
    );
    expectRelativelyClose(facades.lengths, [5, 0, 3, 0]);
  });

  it('counts a wall listed twice, in opposite directions, for the lower row only', async () => {
    const facades = await getFacades(
      moveCase(
        {
          street: [-4, 0, 4, 0],
          segments: [-10, 2, 10, 2, 10, 2, -10, 2],
          owners: [0, 1],
          maximumDistance: 10
        },
        GENERIC_MOTION
      ),
      obstacleIndex
    );
    expectRelativelyClose(facades.lengths, [8, 0, 0, 0]);
  });

  it('counts the front of a box and nothing for its edge-on sides and hidden back', async () => {
    // Box (-1, 2) - (1, 4): front wall 2 long, two walls perpendicular to the street, back wall
    // hidden by the front one.
    const facades = await getFacades(
      moveCase(
        {
          street: [-4, 0, 4, 0],
          segments: [-1, 2, 1, 2, 1, 2, 1, 4, 1, 4, -1, 4, -1, 4, -1, 2],
          owners: [0, 0, 0, 0],
          maximumDistance: 10
        },
        GENERIC_MOTION
      ),
      obstacleIndex
    );
    expectRelativelyClose(facades.lengths, [2, 0, 0, 0, 0, 0, 0, 0]);
    expectRelativelyClose(facades.frontage, [2, 0]);
  });

  it.each([
    // Reverse row order: a pairwise "lower row within the slack hides me" rule hides all three in
    // a cycle (2 by 1, 1 by 0, 0 by 2, which is nearer by more than the slack).
    {name: 'reverse row order', offsets: [0.0016, 0.0008, 0], lengths: [0, 0, 60, 0, 0, 0]},
    {name: 'row order', offsets: [0, 0.0008, 0.0016], lengths: [60, 0, 0, 0, 0, 0]}
  ])('gives three nearly coincident walls exactly one owner ($name)', async ({
    offsets,
    lengths
  }) => {
    // Street (0, 0) - (100, 0), so the slack is 1e-6 + 1e-5 * 100 = 0.001001. Three walls span x
    // from 20 to 80, 0.0008 apart. The nearest is at y = 10; the one 0.0008 behind it is the same
    // stretch, the one 0.0016 behind it is hidden. The lower row of the two coincident walls owns
    // the 60 units; the visible length is counted once.
    const facades = await getFacades(
      {
        street: [0, 0, 100, 0],
        segments: offsets.flatMap(offset => [20, 10 + offset, 80, 10 + offset]),
        owners: [0, 1, 2],
        maximumDistance: 50
      },
      obstacleIndex
    );
    expectRelativelyClose(facades.lengths, lengths);
  });

  it('treats a wall within the tolerance behind the street line as touching the sightline', async () => {
    // A segment within `tolerance` (1e-6) of the point where a sightline leaves the street axis
    // blocks it at distance 0, whichever side it is on. Wall 1 lies 5e-7 behind the axis along
    // the whole street, so wall 0 shows nothing, and wall 1 itself is on the axis.
    const touching = await getFacades(
      {
        street: [-4, 0, 4, 0],
        segments: [-10, 2, 10, 2, -4, -5e-7, 4, -5e-7],
        owners: [0, 1],
        maximumDistance: 10
      },
      obstacleIndex
    );
    expect(touching.lengths).toEqual([0, 0, 0, 0]);
    // 5e-5 behind the axis it is outside the tolerance: wall 0 shows its 8 units on the left and
    // wall 1 its 8 units on the right.
    const clear = await getFacades(
      {
        street: [-4, 0, 4, 0],
        segments: [-10, 2, 10, 2, -4, -5e-5, 4, -5e-5],
        owners: [0, 1],
        maximumDistance: 10
      },
      obstacleIndex
    );
    expectRelativelyClose(clear.lengths, [8, 0, 0, 8]);
  });
});

// URB-104: two unobstructed four-unit walls, one on each side of an eight-unit street.
// Each wall is wholly in the corridor, so its arc length is exactly 4.
it.each([
  'brute-force',
  'bvh'
] as const)('URB-104 invalid segment ownership suppresses every facade and building summary (%s)', async obstacleIndex => {
  const geometry = {
    street: [-4, 0, 4, 0],
    segments: [-2, 2, 2, 2, -2, -3, 2, -3],
    maximumDistance: 5,
    buildingCount: 2
  };
  const valid = await getFacades({...geometry, owners: [0, 1]}, obstacleIndex);
  expectRelativelyClose(valid.lengths, [4, 0, 0, 4]);
  expectRelativelyClose(valid.frontage, [4, 0, 0, 4]);
  const invalid = await getFacades({...geometry, owners: [0, 2]}, obstacleIndex);
  expectUrbanMissingValues(invalid.lengths, 4);
  expectUrbanMissingValues(invalid.frontage, 4);
});
