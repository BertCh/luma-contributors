// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Packed four-level polygon rows used by the geometry CPU operations. */
export type CPUPolygonRows = {
  positions: Float32Array;
  ringOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  featureOffsets: Uint32Array;
};

/** A ring's packed vertex interval and positional role. @internal */
export type CPUPolygonRing = {start: number; end: number; shell: boolean};

/** Contiguous feature interval used by label bounds and descriptor hull membership. @internal */
export type CPUPolygonFeatureRows = {start: number; end: number};

/**
 * Resolves the GPU feature-level row interval independently of the individual ring walks.
 * A descending final ring offset can make this interval empty while an earlier ring is nonempty.
 * @internal
 */
export function getCPUPolygonFeatureRows(
  input: CPUPolygonRows,
  feature: number
): CPUPolygonFeatureRows {
  const polygonCount = input.polygonOffsets.length - 1;
  const ringCount = input.ringOffsets.length - 1;
  const rowCount = input.positions.length / 2;
  const firstPolygon = Math.min(input.featureOffsets[feature], polygonCount);
  const lastPolygon = Math.min(input.featureOffsets[feature + 1], polygonCount);
  const firstRing = Math.min(input.polygonOffsets[firstPolygon], ringCount);
  const lastRing = Math.max(firstRing, Math.min(input.polygonOffsets[lastPolygon], ringCount));
  const start = Math.min(input.ringOffsets[firstRing], rowCount);
  return {start, end: Math.max(start, Math.min(input.ringOffsets[lastRing], rowCount))};
}

/** Checks array consistency, without rejecting offsets the GPU deliberately clamps. @internal */
export function validateCPUPolygonRows(input: CPUPolygonRows): void {
  if (input.positions.length % 2 !== 0) {
    throw new Error('positions must contain complete float32x2 rows');
  }
  for (const name of ['ringOffsets', 'polygonOffsets', 'featureOffsets'] as const) {
    if (input[name].length < 2) {
      throw new Error(`${name} must contain at least two rows`);
    }
  }
}

/** Resolves the same clamped feature/ring intervals as the geometry kernels. @internal */
export function getCPUPolygonRings(input: CPUPolygonRows, feature: number): CPUPolygonRing[] {
  const polygonCount = input.polygonOffsets.length - 1;
  const ringCount = input.ringOffsets.length - 1;
  const rowCount = input.positions.length / 2;
  const firstPolygon = Math.min(input.featureOffsets[feature], polygonCount);
  const lastPolygon = Math.min(input.featureOffsets[feature + 1], polygonCount);
  const firstRing = Math.min(input.polygonOffsets[firstPolygon], ringCount);
  const lastRing = Math.min(input.polygonOffsets[lastPolygon], ringCount);
  const rings: CPUPolygonRing[] = [];
  for (let ring = firstRing; ring < lastRing; ring++) {
    let lower = 0;
    let upper = polygonCount;
    while (lower < upper) {
      const middle = Math.floor((lower + upper) / 2);
      if (input.polygonOffsets[middle] < ring) lower = middle + 1;
      else upper = middle;
    }
    const shell =
      ring < input.polygonOffsets[0] || ring >= input.polygonOffsets[polygonCount]
        ? ring === firstRing
        : input.polygonOffsets[lower] === ring;
    rings.push({
      start: Math.min(input.ringOffsets[ring], rowCount),
      end: Math.min(input.ringOffsets[ring + 1], rowCount),
      shell
    });
  }
  return rings;
}

/** Signed even-odd boundary distance and first closest segment witness. @internal */
export function getCPUPolygonDistance(
  positions: Float32Array,
  rings: readonly CPUPolygonRing[],
  queryX: number,
  queryY: number
): {distance: number; witnessX: number; witnessY: number} {
  let inside = false;
  let minimumSquared = Number.POSITIVE_INFINITY;
  let witnessX = queryX;
  let witnessY = queryY;
  for (const ring of rings) {
    if (ring.end <= ring.start) continue;
    let previousX = positions[(ring.end - 1) * 2];
    let previousY = positions[(ring.end - 1) * 2 + 1];
    for (let row = ring.start; row < ring.end; row++) {
      const currentX = positions[row * 2];
      const currentY = positions[row * 2 + 1];
      const edgeX = currentX - previousX;
      const edgeY = currentY - previousY;
      const denominator = edgeX * edgeX + edgeY * edgeY;
      const fraction =
        denominator > 0
          ? Math.max(
              0,
              Math.min(
                1,
                ((queryX - previousX) * edgeX + (queryY - previousY) * edgeY) / denominator
              )
            )
          : 0;
      const candidateX = previousX + fraction * edgeX;
      const candidateY = previousY + fraction * edgeY;
      const squared = (queryX - candidateX) ** 2 + (queryY - candidateY) ** 2;
      if (squared < minimumSquared) {
        minimumSquared = squared;
        witnessX = candidateX;
        witnessY = candidateY;
      }
      if (
        currentY > queryY !== previousY > queryY &&
        queryX < previousX + ((queryY - previousY) * edgeX) / edgeY
      )
        inside = !inside;
      previousX = currentX;
      previousY = currentY;
    }
  }
  return {distance: (inside ? 1 : -1) * Math.sqrt(minimumSquared), witnessX, witnessY};
}
