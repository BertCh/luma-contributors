// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export type MedialAxisReference = {
  medialAxis: number[];
  radius: number[];
  squaredRadius: number[];
  nearestBackground: [number, number][];
};

/** Independent dense oracle for small fixtures: exact integer disk containment, not thinning. */
export function computeMedialAxisReference(
  width: number,
  height: number,
  values: ArrayLike<number>
): MedialAxisReference {
  const count = width * height;
  const background: [number, number][] = [];
  for (let row = -1; row <= height; row++) {
    for (let column = -1; column <= width; column++) {
      if (
        row < 0 ||
        row === height ||
        column < 0 ||
        column === width ||
        values[row * width + column] === 0
      )
        background.push([column, row]);
    }
  }
  const nearestBackground: [number, number][] = [];
  const squaredRadius: number[] = [];
  for (let cell = 0; cell < count; cell++) {
    const column = cell % width;
    const row = Math.floor(cell / width);
    let nearest: [number, number] = background[0];
    let minimum = Infinity;
    for (const candidate of background) {
      const squaredDistance = (column - candidate[0]) ** 2 + (row - candidate[1]) ** 2;
      if (squaredDistance < minimum) {
        minimum = squaredDistance;
        nearest = candidate;
      }
    }
    nearestBackground.push(nearest);
    squaredRadius.push(minimum);
  }
  const medialAxis = squaredRadius.map((ownRadius, cell) => {
    if (ownRadius === 0) return 0;
    for (let candidate = 0; candidate < count; candidate++) {
      if (candidate === cell) continue;
      const squaredSeparation =
        ((candidate % width) - (cell % width)) ** 2 +
        (Math.floor(candidate / width) - Math.floor(cell / width)) ** 2;
      // sqrt(R) >= sqrt(r) + sqrt(d) iff R-r-d >= 0 and (R-r-d)^2 >= 4rd.
      const difference = squaredRadius[candidate] - ownRadius - squaredSeparation;
      if (difference < 0) continue;
      const left = difference * difference;
      const right = 4 * ownRadius * squaredSeparation;
      const contained =
        Number.isSafeInteger(left) && Number.isSafeInteger(right)
          ? left >= right
          : BigInt(difference) ** 2n >= 4n * BigInt(ownRadius) * BigInt(squaredSeparation);
      if (contained) return 0;
    }
    return 1;
  });
  return {
    medialAxis,
    squaredRadius,
    nearestBackground,
    radius: squaredRadius.map(value => Math.fround(Math.sqrt(value)))
  };
}

/** Analytic filled-rectangle reference, independent of the nearest-seed/gcd implementation. */
export function computeFilledRectangleReference(width: number, height: number) {
  const radius: number[] = [];
  const medialAxis: number[] = [];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const boundaries = [column + 1, width - column, row + 1, height - row];
      const minimum = Math.min(...boundaries);
      radius.push(minimum);
      const tiedBoundaries = boundaries.filter(value => value === minimum).length > 1;
      const horizontalPlateau =
        Math.min(boundaries[0], boundaries[1]) === minimum &&
        Math.abs(boundaries[0] - boundaries[1]) <= 1;
      const verticalPlateau =
        Math.min(boundaries[2], boundaries[3]) === minimum &&
        Math.abs(boundaries[2] - boundaries[3]) <= 1;
      medialAxis.push(Number(tiedBoundaries || horizontalPlateau || verticalPlateau));
    }
  }
  return {radius, medialAxis};
}
