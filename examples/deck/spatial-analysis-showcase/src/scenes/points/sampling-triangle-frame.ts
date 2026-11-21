// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Removes zero-area counted rows from an area frame, retaining original triangle row identity. */
export function prepareSamplingTriangleFrame(
  positions: Float32Array,
  indices: Uint32Array,
  count: number
) {
  const coordinates: number[] = [];
  const sourceTriangleRows: number[] = [];
  if (!Number.isInteger(count) || count < 0 || count * 3 > indices.length)
    throw new Error('sampling-triangle-frame: count exceeds the readable index prefix');
  for (let row = 0; row < count; row++) {
    const corners = [0, 1, 2].map(corner => {
      const vertex = indices[row * 3 + corner];
      if (vertex * 2 + 1 >= positions.length)
        throw new Error('sampling-triangle-frame: invalid vertex index');
      return [positions[vertex * 2], positions[vertex * 2 + 1]];
    });
    const [first, second, third] = corners;
    const determinant =
      (second[0] - first[0]) * (third[1] - first[1]) -
      (second[1] - first[1]) * (third[0] - first[0]);
    if (!Number.isFinite(determinant))
      throw new Error('sampling-triangle-frame: nonfinite triangle area');
    if (determinant === 0) continue;
    let ordered: number[][] | undefined;
    for (let start = 0; start < 3; start++) {
      const first = corners[start],
        second = corners[(start + 1) % 3],
        third = corners[(start + 2) % 3];
      const firstX = Math.fround(second[0] - first[0]),
        firstY = Math.fround(second[1] - first[1]);
      const secondX = Math.fround(third[0] - first[0]),
        secondY = Math.fround(third[1] - first[1]);
      const packedDeterminant = Math.fround(
        Math.fround(firstX * secondY) - Math.fround(firstY * secondX)
      );
      if (Number.isFinite(packedDeterminant) && packedDeterminant !== 0) {
        ordered = [first, second, third];
        break;
      }
    }
    if (!ordered)
      throw new Error(
        'sampling-triangle-frame: nonzero triangle has no usable float32 corner ordering; translate or rescale the frame'
      );
    for (const corner of ordered) coordinates.push(corner[0], corner[1]);
    sourceTriangleRows.push(row);
  }
  return {
    positions: Float32Array.from(coordinates),
    sourceTriangleRows: Uint32Array.from(sourceTriangleRows)
  };
}

/** Word zero is a legacy-site ID for role 2, and a compact triangle row for random samples. */
export function getSamplingSourceRow(
  packedSource: number,
  role: number,
  sourceTriangleRows: Uint32Array
): number {
  return role === 2 ? packedSource : (sourceTriangleRows[packedSource] ?? packedSource);
}
