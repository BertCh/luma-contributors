// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

declare module 'earcut' {
  /** Triangulates a flat polygon (exterior ring first, then holes at `holeIndices`). */
  export default function earcut(
    data: ArrayLike<number>,
    holeIndices?: ArrayLike<number> | null,
    dimensions?: number
  ): number[];
}
