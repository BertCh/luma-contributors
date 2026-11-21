// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Independent GEOS/Hilbert oracle from the S2 review: the gnomonic rectangle spans
// (i-.4,j-.4) to (i+1.4,j+1.4), so exactly the central level-15 square is interior.
export const FINE_S2_RECTANGLES = [
  {first: 23000, second: 21000, cellId: '1123fdc040000000'},
  {first: 24503, second: 19621, cellId: '10d2c5c240000000'},
  {first: 19170, second: 31311, cellId: '12c46378c0000000'}
] as const;

export function getFineS2RectangleRing(first: number, second: number): number[][] {
  const surface = (coordinate: number) => (4 * (coordinate / 32768) ** 2 - 1) / 3;
  return [
    [first - 0.4, second - 0.4],
    [first + 1.4, second - 0.4],
    [first + 1.4, second + 1.4],
    [first - 0.4, second + 1.4]
  ].map(([horizontal, vertical]) => {
    const y = surface(horizontal);
    const z = surface(vertical);
    return [(Math.atan2(y, 1) * 180) / Math.PI, (Math.atan2(z, Math.hypot(1, y)) * 180) / Math.PI];
  });
}

export function getThinSeamTriangleRing(): number[][] {
  const halfAngleDegrees = (0.75e-6 * 180) / Math.PI;
  return [
    [45 - halfAngleDegrees, 0],
    [45 + halfAngleDegrees, 0],
    [45, 10]
  ];
}

export const FINE_S2_COVER_PARAMETERS = {
  mode: 'interior',
  minimumLevel: 15,
  maximumLevel: 15,
  maximumCells: 4,
  maximumVisitedCells: 512
} as const;

export const SEAM_S2_COVER_PARAMETERS = {
  mode: 'exterior',
  minimumLevel: 0,
  maximumLevel: 0,
  maximumCells: 6,
  maximumVisitedCells: 6
} as const;
