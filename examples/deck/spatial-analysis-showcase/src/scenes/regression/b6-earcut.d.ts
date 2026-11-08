// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

declare module 'earcut' {
  /** Triangulates a polygon with optional holes (flat coordinates, hole start vertex indices). */
  export default function earcut(
    data: ArrayLike<number>,
    holeIndices?: ArrayLike<number> | null,
    dimensions?: number
  ): number[];
}
