// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** GeoArrow polygon columns, as `GPUPolygonRasterization` and `GPUPointInPolygonJoin` take them. */
export type DotDensityCPUPolygons = {
  /** Flattened `(x, y)` vertices. */
  positions: Float32Array;
  /** Feature-to-polygon offsets, `featureCount + 1` entries. */
  featureOffsets: Uint32Array;
  /** Polygon-to-ring offsets with a terminal entry. */
  polygonOffsets: Uint32Array;
  /** Ring-to-vertex offsets with a terminal entry. */
  ringOffsets: Uint32Array;
};

/** Input of {@link generateDotCountsOnCPU}. */
export type DotDensityCPUInput = DotDensityCPUPolygons & {
  /** Float values, `featureCount * categoryCount` rows (dot density), or `undefined` with `counts`. */
  values?: Float32Array;
  /** Integer point counts per feature (random points in polygon). */
  counts?: Uint32Array;
  /** Categories per feature. Defaults to 1. */
  categoryCount?: number;
  /** Parameter words from `getGPUDotDensityParameterValues`. */
  parameters: Uint32Array;
  /** Output capacity in dots. */
  capacity: number;
};

/** Result of {@link generateDotCountsOnCPU}. */
export type DotDensityCPUResult = {
  /** Dots per slot after clamping. */
  slotCounts: Uint32Array;
  /** Exclusive prefix of `slotCounts`. */
  /** Pre-clamp slot starts without a terminal. */
  slotStarts: Uint32Array;
  /** Unclamped total dot count. */
  requiredCount: number;
  /** `min(requiredCount, capacity)`. */
  count: number;
  /** 1 when the retained total exceeds final capacity. */
  overflow: number;
  /** 1 when a per-slot limit discarded requests before final publication. */
  candidateOverflow: number;
  /** Clamped requests discarded because their feature has zero area. */
  failedCount: number;
  /** Feature row per dot. */
  featureIds: Uint32Array;
  /** Category per dot. */
  categories: Uint32Array;
};

const fround = Math.fround;

/** Independent Random123 Philox4x32-10 arithmetic using exact BigInt products, not split words. */
export function computeDotPhiloxReference(
  counter: readonly number[],
  key: readonly number[]
): number[] {
  const mask = 0xffffffffn;
  let words = counter.map(BigInt);
  let firstKey = BigInt(key[0]);
  let secondKey = BigInt(key[1]);
  for (let round = 0; round < 10; round++) {
    const firstProduct = words[0] * 0xd2511f53n;
    const secondProduct = words[2] * 0xcd9e8d57n;
    words = [
      ((secondProduct >> 32n) ^ words[1] ^ firstKey) & mask,
      secondProduct & mask,
      ((firstProduct >> 32n) ^ words[3] ^ secondKey) & mask,
      firstProduct & mask
    ];
    firstKey = (firstKey + 0x9e3779b9n) & mask;
    secondKey = (secondKey + 0xbb67ae85n) & mask;
  }
  return words.map(Number);
}

/** Shoelace reference for valid finite polygons; does not model triangulation budget failures. */
function computeFeatureArea(input: DotDensityCPUPolygons, feature: number): number {
  let featureArea = 0;
  for (
    let polygon = input.featureOffsets[feature];
    polygon < input.featureOffsets[feature + 1];
    polygon++
  ) {
    let polygonArea = 0;
    const firstRing = input.polygonOffsets[polygon];
    for (let ring = firstRing; ring < input.polygonOffsets[polygon + 1]; ring++) {
      const first = input.ringOffsets[ring];
      const end = input.ringOffsets[ring + 1];
      let twiceArea = 0;
      for (let vertex = first; vertex < end; vertex++) {
        const next = vertex + 1 === end ? first : vertex + 1;
        twiceArea +=
          input.positions[2 * vertex] * input.positions[2 * next + 1] -
          input.positions[2 * next] * input.positions[2 * vertex + 1];
      }
      polygonArea += ((ring === firstRing ? 1 : -1) * Math.abs(twiceArea)) / 2;
      // A zero-area shell makes the whole polygon empty, regardless of its holes.
      if (ring === firstRing && twiceArea === 0) break;
    }
    featureArea += Math.max(0, polygonArea);
  }
  return featureArea;
}

/**
 * CPU oracle of the dot counts of `GPUDotDensity` (with `values`) and `GPURandomPointsInPolygon`
 * (with `counts`): slot counts, offsets, totals, overflow, feature IDs and categories, all exact.
 *
 * Positions are not generated here: the GPU samples inside a GPU triangulation, so positions are
 * checked by property (all dots placed, inside the feature per shapely) in the GPU spec.
 */
export function generateDotCountsOnCPU(input: DotDensityCPUInput): DotDensityCPUResult {
  const categoryCount = input.categoryCount ?? 1;
  const featureCount = input.featureOffsets.length - 1;
  const slotCount = featureCount * categoryCount;
  const words = input.parameters;
  const floats = new Float32Array(words.buffer, words.byteOffset, words.length);
  const seed = words[0];
  const dotsPerUnit = floats[1];
  // Independent contract arithmetic: reserve a capacity+1 witness, bound the u32 prefix,
  // and cap each slot at the exact integer conversion envelope 2^31.
  const limit = Math.min(
    input.capacity + 1,
    Math.floor((2 ** 32 - 1) / Math.max(slotCount, 1)),
    2 ** 31
  );
  const featureAreas = Array.from({length: featureCount}, (_, feature) =>
    computeFeatureArea(input, feature)
  );

  const slotCounts = new Uint32Array(slotCount);
  let clamped = false;
  let failedCount = 0;
  for (let slot = 0; slot < slotCount; slot++) {
    let dots: number;
    if (input.counts) {
      dots = input.counts[slot];
    } else {
      const value = input.values![slot];
      const scaled = fround(value * dotsPerUnit);
      // Pinned public draw convention: remainder key = [seed, 6], high 24 bits / 2^24.
      const remainder = (computeDotPhiloxReference([slot, 0, 0, 0], [seed, 6])[0] >>> 8) / 2 ** 24;
      // Compare the fractional part directly: subtracting the draw from a large integer
      // first can round down one whole dot in float32 before ceil is applied.
      const integral = Math.floor(scaled);
      const ceiling = integral + Number(remainder < scaled - integral);
      // NaN, negative values and a non-positive scale draw nothing.
      dots = Number.isFinite(scaled) && scaled > 0 && ceiling > 0 ? ceiling : 0;
      if (scaled === Infinity) {
        dots = Infinity;
      }
    }
    if (dots > limit) {
      clamped = true;
      dots = limit;
    }
    if (featureAreas[Math.floor(slot / categoryCount)] === 0) {
      failedCount += dots;
      dots = 0;
    }
    slotCounts[slot] = dots;
  }
  const slotStarts = new Uint32Array(slotCount);
  let requiredCount = 0;
  for (let slot = 0; slot < slotCount; slot++) {
    slotStarts[slot] = requiredCount;
    requiredCount += slotCounts[slot];
  }
  const count = Math.min(requiredCount, input.capacity);
  const featureIds = new Uint32Array(count);
  const categories = new Uint32Array(count);
  let dot = 0;
  for (let slot = 0; slot < slotCount && dot < count; slot++) {
    for (let rank = 0; rank < slotCounts[slot] && dot < count; rank++, dot++) {
      featureIds[dot] = Math.floor(slot / categoryCount);
      categories[dot] = slot % categoryCount;
    }
  }
  return {
    slotCounts,
    slotStarts,
    requiredCount,
    count,
    overflow: requiredCount > input.capacity ? 1 : 0,
    candidateOverflow: clamped ? 1 : 0,
    failedCount,
    featureIds,
    categories
  };
}
