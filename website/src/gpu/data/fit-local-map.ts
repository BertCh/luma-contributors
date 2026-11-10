// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Fit a local east/north meter grid into scene coordinates with one scale on both axes. */
export function fitLocalMap(
  positions: readonly (readonly number[])[],
  world: {width: number; height: number}
): (east: number, north: number) => [number, number] {
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;
  for (const [x, y] of positions) {
    west = Math.min(west, x);
    east = Math.max(east, x);
    south = Math.min(south, y);
    north = Math.max(north, y);
  }
  const scale =
    0.94 * Math.min(world.width / (east - west || 1), world.height / (north - south || 1));
  return (x, y) => [
    world.width / 2 + (x - (west + east) / 2) * scale,
    world.height / 2 + (y - (south + north) / 2) * scale
  ];
}
