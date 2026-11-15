// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {_GlobeViewport as GlobeViewport, WebMercatorViewport} from '@deck.gl/core';
import {SolidPolygonLayer} from '@deck.gl/layers';
import {Matrix4} from '@math.gl/core';
import {cutPolygonByGrid} from '@math.gl/geometry';
import {expect, test, vi} from 'vitest';
import {GlobeBasemapPolygonLayer} from '../../../examples/deck/kepler-compute/src/globe-basemap-polygon-layer';

// Same linear units as deck's OSMTile512 descriptor. These are the fields its subdivision method
// reads; initialize just that CPU state, without constructing attributes or requiring a device.
const EARTH_RADIUS_METERS = 6378137;
const METERS_PER_TILE_UNIT = (2 * Math.PI * EARTH_RADIUS_METERS) / 512;
const TILE_CRS = {
  signature: 'OSMTile512',
  coordinateFrame: 'projected',
  ellipsoid: {semiMajorAxis: EARTH_RADIUS_METERS},
  xUnit: {metersPerUnit: METERS_PER_TILE_UNIT},
  yUnit: {metersPerUnit: METERS_PER_TILE_UNIT}
};

function createPolygonLayer(zoom: number) {
  const layer = new GlobeBasemapPolygonLayer({
    data: [],
    coordinateSystem: 'OSMTile512',
    modelMatrix: new Matrix4().scale([512 / 2 ** zoom, 512 / 2 ** zoom, 1])
  });
  Object.assign(layer, {
    context: {viewport: new GlobeViewport({width: 800, height: 600, zoom: 1.2, resolution: 72})},
    internalState: {sourceCrs: TILE_CRS}
  });
  return layer;
}

function getRingArea(positions: readonly number[], start: number, end: number): number {
  let twiceArea = 0;
  for (let index = start, previous = end - 3; index < end; previous = index, index += 3) {
    twiceArea +=
      positions[previous] * positions[index + 1] - positions[index] * positions[previous + 1];
  }
  return Math.abs(twiceArea) / 2;
}

function getCutArea(pieces: ReturnType<typeof cutPolygonByGrid>): number {
  return pieces.reduce((area, piece) => {
    const positions = Array.from(piece.positions);
    const boundaries = [0, ...Array.from(piece.holeIndices ?? []), positions.length];
    let polygonArea = getRingArea(positions, boundaries[0], boundaries[1]);
    for (let hole = 1; hole < boundaries.length - 1; hole++) {
      polygonArea -= getRingArea(positions, boundaries[hole], boundaries[hole + 1]);
    }
    return area + polygonArea;
  }, 0);
}

test('reproduces native null.pos and cuts the full z0 water polygon without losing fill area', () => {
  const layer = createPolygonLayer(0);
  const nativeStep = SolidPolygonLayer.prototype.getSubdivisionStep.call(layer);
  const positions = [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0];
  expect(nativeStep).toBeCloseTo(1 / 72, 15);
  // This is the driver's SolidPolygonLayer initialization exception, isolated to the published
  // CPU cutter. Keep the reproduction until the pinned math.gl release fixes it.
  expect(() =>
    cutPolygonByGrid(positions, null, {size: 3, gridResolution: nativeStep, edgeTypes: true})
  ).toThrow("Cannot read properties of null (reading 'pos')");
  const step = layer.getSubdivisionStep();
  expect(step).toBe(1 / 128);
  const pieces = cutPolygonByGrid(positions, null, {
    size: 3,
    gridResolution: step,
    edgeTypes: true
  });
  expect(pieces).toHaveLength(128 * 128);
  expect(getCutArea(pieces)).toBe(1);
  for (const piece of pieces) {
    expect(piece.edgeTypes).toHaveLength(piece.positions.length / 3);
    expect(Array.from(piece.positions).every(Number.isFinite)).toBe(true);
  }
});

test('preserves tile buffers and holes on the finer globe grid', () => {
  const outer = [
    -0.0625, -0.0625, 0, 1.0625, -0.0625, 0, 1.0625, 1.0625, 0, -0.0625, 1.0625, 0, -0.0625,
    -0.0625, 0
  ];
  const hole = [
    0.375, 0.375, 0, 0.375, 0.625, 0, 0.625, 0.625, 0, 0.625, 0.375, 0, 0.375, 0.375, 0
  ];
  const pieces = cutPolygonByGrid([...outer, ...hole], [outer.length], {
    size: 3,
    gridResolution: createPolygonLayer(0).getSubdivisionStep(),
    edgeTypes: true
  });
  expect(getCutArea(pieces)).toBe(1.125 ** 2 - 0.25 ** 2);
});

test.each([0, 1, 4, 9, 14])('tile zoom %s keeps at least the native globe resolution', zoom => {
  const layer = createPolygonLayer(zoom);
  const nativeStep = SolidPolygonLayer.prototype.getSubdivisionStep.call(layer);
  const step = layer.getSubdivisionStep();
  expect(Number.isInteger(Math.log2(step))).toBe(true);
  expect(step).toBeLessThanOrEqual(nativeStep);
  expect(step).toBeGreaterThanOrEqual(nativeStep / 2);
});

test('a planar viewport keeps subdivision disabled', () => {
  const layer = createPolygonLayer(0);
  Object.assign(layer, {
    context: {viewport: new WebMercatorViewport({width: 800, height: 600, zoom: 1.2})}
  });
  expect(layer.getSubdivisionStep()).toBe(0);
});

test('a native step immediately below a power of two is never rounded to a coarser grid', () => {
  const getNativeStep = vi
    .spyOn(SolidPolygonLayer.prototype, 'getSubdivisionStep')
    .mockReturnValue(1 - Number.EPSILON / 2);
  try {
    expect(createPolygonLayer(0).getSubdivisionStep()).toBe(0.5);
  } finally {
    getNativeStep.mockRestore();
  }
});
