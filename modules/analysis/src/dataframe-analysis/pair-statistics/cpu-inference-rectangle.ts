// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Float64 counterpart of the rectangle side-arc kernel, retaining the estimator's weight cap. @internal */
export function getRectangleIsotropicWeight(
  sides: readonly number[],
  distance: number,
  weightCap: number
): number {
  const angles = sides.map(side =>
    distance > side
      ? side === 0
        ? Math.PI / 2
        : Math.atan2(Math.sqrt((distance - side) * (distance + side)), side)
      : 0
  );
  let overlap = 0;
  for (let side = 0; side < 4; side++)
    overlap += Math.max(0, angles[side] + angles[(side + 1) % 4] - Math.PI / 2);
  const inside =
    1 - angles.reduce((sum, angle) => sum + angle, 0) / Math.PI + overlap / (2 * Math.PI);
  return Math.min(1 / Math.max(inside, 1 / weightCap), weightCap);
}

/** Rectangle distances in left, bottom, right, top order. @internal */
export function getRectangleSides(x: number, y: number, bounds: readonly number[]): number[] {
  const [minimumX, minimumY, maximumX, maximumY] = bounds;
  return [x - minimumX, y - minimumY, maximumX - x, maximumY - y];
}

/** The pair-statistics lattice inclusion rule, evaluated in float64. @internal */
export function isRectangleLatticeValid(
  bounds: readonly number[],
  maximumDistance: number
): boolean {
  const [minimumX, minimumY, maximumX, maximumY] = bounds;
  return (
    bounds.every(Number.isFinite) &&
    Number.isFinite(maximumDistance) &&
    maximumDistance > 0 &&
    maximumX >= minimumX &&
    maximumY >= minimumY
  );
}
