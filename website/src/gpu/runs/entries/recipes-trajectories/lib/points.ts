// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** The point with the most other points within `radius`: a busy anchor for a pointer ring. */
export function findBusiestPoint(positions: ArrayLike<number>, radius: number): [number, number] {
  const count = positions.length >> 1;
  let best: [number, number] = [positions[0], positions[1]];
  let bestCount = -1;
  for (let point = 0; point < count; point++) {
    let neighbors = 0;
    for (let other = 0; other < count; other++) {
      if (
        Math.hypot(
          positions[other * 2] - positions[point * 2],
          positions[other * 2 + 1] - positions[point * 2 + 1]
        ) < radius
      ) {
        neighbors++;
      }
    }
    if (neighbors > bestCount) {
      bestCount = neighbors;
      best = [positions[point * 2], positions[point * 2 + 1]];
    }
  }
  return best;
}
