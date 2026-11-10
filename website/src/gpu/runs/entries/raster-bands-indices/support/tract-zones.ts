// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {TractsInput} from '../../../inputs';

/** Chicago Census tract shells and their pre-triangulated display mesh. */
export function createTractFeatures(tracts: TractsInput) {
  const {count, vertices, centers, ringOffsets} = tracts;
  const singleParts = Uint32Array.from({length: count + 1}, (_, index) => index);
  const extents = new Float32Array(count * 2);
  for (let row = 0; row < count; row++) {
    const begin = ringOffsets[row];
    const end = ringOffsets[row + 1];
    for (let axis = 0; axis < 2; axis++) {
      let low = Infinity;
      let high = -Infinity;
      for (let vertex = begin; vertex < end; vertex++) {
        const value = vertices[vertex * 2 + axis];
        low = Math.min(low, value);
        high = Math.max(high, value);
      }
      extents[row * 2 + axis] = high - low;
    }
  }
  function getPositions(scale: number) {
    const positions = vertices.slice();
    for (let row = 0; row < count; row++) {
      for (let vertex = ringOffsets[row]; vertex < ringOffsets[row + 1]; vertex++) {
        for (let axis = 0; axis < 2; axis++) {
          const center = centers[row * 2 + axis];
          positions[vertex * 2 + axis] = center + (vertices[vertex * 2 + axis] - center) * scale;
        }
      }
    }
    return positions;
  }
  function getOutline(positions: Float32Array) {
    const outline = new Float32Array(positions.length * 2);
    for (let row = 0; row < count; row++) {
      for (let vertex = ringOffsets[row]; vertex < ringOffsets[row + 1]; vertex++) {
        const next = vertex + 1 < ringOffsets[row + 1] ? vertex + 1 : ringOffsets[row];
        outline.set(
          [
            positions[vertex * 2],
            positions[vertex * 2 + 1],
            positions[next * 2],
            positions[next * 2 + 1]
          ],
          vertex * 4
        );
      }
    }
    return outline;
  }
  return {
    featureCount: count,
    positions: vertices,
    featureOffsets: singleParts,
    polygonOffsets: singleParts,
    ringOffsets,
    meshVertices: vertices,
    meshIndices: tracts.triangleIndices,
    meshOwners: tracts.triangleOwners,
    meshTriangleCount: tracts.triangleCount,
    centers,
    extents,
    outline: tracts.outline,
    outlineCount: tracts.outlineCount,
    getPositions,
    getOutline,
    values: tracts.values,
    geoids: tracts.geoids
  };
}

export type TractFeatures = ReturnType<typeof createTractFeatures>;
