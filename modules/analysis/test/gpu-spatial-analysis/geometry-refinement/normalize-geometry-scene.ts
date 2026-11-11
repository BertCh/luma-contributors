// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Hand-computed postorder tree spanning polygon holes, multis and nested collections. */
export function getNormalizeGeometryScene() {
  const positions = new Float32Array([
    5, 5, 3, 0, 2, 1, 1, 0, 7, 0, 6, 0, 4, 0, 4, 4, 0, 4, 0, 0, 4, 0, 1, 1, 1, 2, 2, 2, 2, 1, 1, 1,
    3, 1, 3, 2, 3.5, 2, 3.5, 1, 3, 1
  ]);
  const nodes = new Uint32Array([
    0, 0, 1, 50, 2, 1, 3, 40, 2, 4, 2, 30, 3, 6, 5, 20, 3, 11, 5, 10, 3, 16, 5, 0, 5, 0, 3, 6, 4, 3,
    2, 7, 1, 5, 1, 8, 7, 6, 3, 9, 7, 9, 2, 10, 7, 11, 1, 11, 7, 12, 1, 12, 7, 13, 2, 13
  ]);
  const children = new Uint32Array([3, 4, 5, 2, 1, 0, 8, 6, 7, 0, 9, 8, 7, 11, 12]);
  const expectedPositions = new Float32Array([
    5, 5, 1, 0, 2, 1, 3, 0, 6, 0, 7, 0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 1, 1, 2, 1, 2, 2, 1, 2, 1, 1,
    3, 1, 3.5, 1, 3.5, 2, 3, 2, 3, 1
  ]);
  const expectedChildren = new Uint32Array([3, 5, 4, 1, 2, 0, 6, 7, 8, 9, 0, 8, 7, 12, 11]);
  return {
    input: {positions, nodes, children},
    expected: {
      positions: expectedPositions,
      children: expectedChildren,
      status: new Uint32Array(nodes.length / 4)
    }
  };
}
