// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {cellToBoundary, cellToLatLng, getPentagons, gridDisk, latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {
  CELL_COVER_H3_LATTICE_SPACING_DEGREES,
  CELL_COVER_H3_MAXIMUM_LATITUDE,
  CELL_COVER_H3_MINIMUM_INRADIUS_DEGREES
} from '../../../src/gpu-spatial-analysis/cell-cover/cell-cover-h3-constants';

const RESOLUTIONS = [11, 12, 13, 14, 15];

/** Distance from the cell center to its nearest boundary edge in the local equirectangular metric. */
function getCellInradiusDegrees(cell: string): number {
  const [centerLatitude, centerLongitude] = cellToLatLng(cell);
  const scale = Math.cos((centerLatitude * Math.PI) / 180);
  const boundary = cellToBoundary(cell).map(([latitude, longitude]) => {
    let longitudeDifference = longitude - centerLongitude;
    if (longitudeDifference > 180) longitudeDifference -= 360;
    if (longitudeDifference < -180) longitudeDifference += 360;
    return [longitudeDifference * scale, latitude - centerLatitude] as const;
  });
  let minimum = Infinity;
  for (let index = 0; index < boundary.length; index++) {
    const [startX, startY] = boundary[index];
    const [endX, endY] = boundary[(index + 1) % boundary.length];
    const edgeX = endX - startX;
    const edgeY = endY - startY;
    const lengthSquared = edgeX * edgeX + edgeY * edgeY;
    const t =
      lengthSquared === 0
        ? 0
        : Math.max(0, Math.min(1, -(startX * edgeX + startY * edgeY) / lengthSquared));
    minimum = Math.min(minimum, Math.hypot(startX + t * edgeX, startY + t * edgeY));
  }
  return minimum;
}

it('cell cover H3 inradius table is monotone and spacing is half the inradius', () => {
  expect(CELL_COVER_H3_MINIMUM_INRADIUS_DEGREES.length).toBe(16);
  for (let resolution = 1; resolution < 16; resolution++) {
    expect(CELL_COVER_H3_MINIMUM_INRADIUS_DEGREES[resolution]).toBeLessThan(
      CELL_COVER_H3_MINIMUM_INRADIUS_DEGREES[resolution - 1]
    );
  }
  CELL_COVER_H3_MINIMUM_INRADIUS_DEGREES.forEach((inradius, resolution) => {
    expect(CELL_COVER_H3_LATTICE_SPACING_DEGREES[resolution]).toBeCloseTo(0.5 * inradius, 6);
  });
});

it.each(
  RESOLUTIONS
)('cell cover H3 inradius table bounds measured inradii near pentagons at resolution %i', resolution => {
  const cells = new Set<string>();
  for (const pentagon of getPentagons(resolution)) {
    for (const cell of gridDisk(pentagon, 4)) cells.add(cell);
  }
  // Deterministic low-discrepancy sample away from the poles.
  for (let sample = 0; sample < 400; sample++) {
    const latitude = -80 + (160 * sample) / 399;
    const longitude = ((sample * 137.50776405003785 + 180) % 360) - 180;
    cells.add(latLngToCell(latitude, longitude, resolution));
  }
  const tabulated = CELL_COVER_H3_MINIMUM_INRADIUS_DEGREES[resolution];
  let measuredMinimum = Infinity;
  for (const cell of cells) {
    if (Math.abs(cellToLatLng(cell)[0]) > CELL_COVER_H3_MAXIMUM_LATITUDE) continue;
    measuredMinimum = Math.min(measuredMinimum, getCellInradiusDegrees(cell));
  }
  // The table already carries a 0.85 safety factor over the measured minimum.
  expect(tabulated, `resolution ${resolution} measured ${measuredMinimum}`).toBeLessThanOrEqual(
    measuredMinimum
  );
});
