// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const PLANE_TRANSFER_BLOCK_ROWS = 256;
export const PLANE_TRANSFER_ANGLE = 0.002;
export const PLANE_TRANSFER_TRANSLATION = [0.02, -0.01, 0.015];

// Each block has one position and normal. Rotation is observable only between blocks.
export const PLANE_TRANSFER_PAIRS = [
  {position: [2, 0, 0], normal: [1, 0, 0]},
  {position: [0, 2, 0], normal: [0, 1, 0]},
  {position: [0, 0, 2], normal: [0, 0, 1]},
  {position: [0, 0, 4], normal: [0, 1, 0]},
  {position: [5, 0, 0], normal: [0, 0, 1]},
  {position: [0, 6, 0], normal: [1, 0, 0]}
];

export function createPlaneTransferFixture(center: readonly number[]) {
  const cosine = Math.cos(PLANE_TRANSFER_ANGLE);
  const sine = Math.sin(PLANE_TRANSFER_ANGLE);
  const source: number[] = [];
  const target: number[] = [];
  const normals: number[] = [];
  for (let row = 0; row < PLANE_TRANSFER_BLOCK_ROWS * PLANE_TRANSFER_PAIRS.length; row++) {
    const pair = PLANE_TRANSFER_PAIRS[Math.floor(row / PLANE_TRANSFER_BLOCK_ROWS)];
    const horizontal = pair.position[0] - PLANE_TRANSFER_TRANSLATION[0];
    const vertical = pair.position[1] - PLANE_TRANSFER_TRANSLATION[1];
    source.push(
      cosine * horizontal + sine * vertical + center[0],
      -sine * horizontal + cosine * vertical + center[1],
      pair.position[2] - PLANE_TRANSFER_TRANSLATION[2] + center[2]
    );
    target.push(...pair.position.map((value, axis) => value + center[axis]));
    normals.push(...pair.normal);
  }
  return {source, target, normals};
}
