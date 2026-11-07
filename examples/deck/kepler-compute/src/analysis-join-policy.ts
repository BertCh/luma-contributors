// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Pure policy planners for attribute and spatial joins. They make duplicate, right-column, null,
 * precision, predicate, CRS, boundary and cardinality behavior explicit before any GPU work, so the
 * host can show the policy, route or refuse, and tests can assert it without a device.
 */

export type JoinType = 'inner' | 'left' | 'right' | 'full';
export type JoinCardinality = 'one-to-one' | 'one-to-many' | 'many-to-one' | 'many-to-many';
export type JoinKeyFormat =
  | 'string'
  | 'int32'
  | 'uint32'
  | 'float32'
  | 'float64'
  | 'int64'
  | 'uint64';

/** Largest integer magnitude every Float32 represents exactly. */
export const FLOAT32_EXACT_INTEGER_LIMIT = 16777216;

export type JoinKeyPrecisionPolicy = {
  format: JoinKeyFormat;
  /** exact: GPU canonical key is lossless; refuse: values cannot be canonicalized exactly. */
  mode: 'exact' | 'exact-within-limit' | 'refuse';
  reason?: string;
};

/**
 * Decide whether a numeric key column joins exactly on the GPU. Float32 and 32-bit integers are
 * canonicalized losslessly; 64-bit and Float64 keys are admitted only when every value is an integer
 * within the 32-bit range (explicit narrowing), otherwise refused instead of silently rounded.
 */
export function planJoinKeyPrecision(
  format: JoinKeyFormat,
  values?: ArrayLike<number | bigint>
): JoinKeyPrecisionPolicy {
  if (format === 'string' || format === 'int32' || format === 'uint32' || format === 'float32')
    return {format, mode: 'exact'};
  if (!values)
    return {format, mode: 'refuse', reason: `${format} keys need a value scan before narrowing`};
  for (let index = 0; index < values.length; index++) {
    const value = Number(values[index]);
    const fits = Number.isInteger(value) && value >= -2147483648 && value <= 4294967295;
    if (!fits || (typeof values[index] === 'bigint' && BigInt(value) !== values[index]))
      return {
        format,
        mode: 'refuse',
        reason: `${format} key at row ${index} does not narrow exactly to 32 bits`
      };
  }
  return {format, mode: 'exact-within-limit'};
}

export type JoinRightColumnPolicy = {
  /** Suffix appended to a right column whose name collides with a left column. */
  suffix: string;
  /** Drop the right key column when it has the same name as the left key (native keeps both). */
  dropRightKey: boolean;
};

export type JoinOutputColumn = {
  side: 'left' | 'right';
  sourceName: string;
  outputName: string;
};

/**
 * Deterministic output naming. Collisions (against left names and earlier outputs, case sensitive)
 * receive the suffix and then a numeric counter, so output names are unique and stable.
 */
export function planJoinOutputColumns(
  leftNames: readonly string[],
  rightNames: readonly string[],
  rightKeyName: string,
  policy: JoinRightColumnPolicy
): JoinOutputColumn[] {
  const used = new Set<string>();
  const columns: JoinOutputColumn[] = [];
  for (const name of leftNames) {
    used.add(name);
    columns.push({side: 'left', sourceName: name, outputName: name});
  }
  for (const name of rightNames) {
    if (policy.dropRightKey && name === rightKeyName && leftNames.includes(name)) continue;
    let outputName = name;
    if (used.has(outputName)) outputName = `${name}${policy.suffix}`;
    for (let counter = 2; used.has(outputName); counter++)
      outputName = `${name}${policy.suffix}${counter}`;
    used.add(outputName);
    columns.push({side: 'right', sourceName: name, outputName});
  }
  return columns;
}

export type JoinCardinalityPlan = {
  cardinality: JoinCardinality;
  outputRowCount: number;
  matchedPairCount: number;
  unmatchedLeftRowCount: number;
  unmatchedRightRowCount: number;
  /** Null keys never match (SQL and native semantics); they are counted as unmatched. */
  nullKeyLeftRowCount: number;
  nullKeyRightRowCount: number;
};

/**
 * Exact result cardinality from key arrays (null/undefined keys never match). Right-only rows are
 * appended after left order for right and full joins, as the native FULL join does.
 */
export function planJoinCardinality(
  leftKeys: ArrayLike<string | number | null | undefined>,
  rightKeys: ArrayLike<string | number | null | undefined>,
  joinType: JoinType
): JoinCardinalityPlan {
  const rightCounts = new Map<string | number, number>();
  let nullKeyRightRowCount = 0;
  for (let index = 0; index < rightKeys.length; index++) {
    const key = rightKeys[index];
    if (key === null || key === undefined) nullKeyRightRowCount++;
    else rightCounts.set(key, (rightCounts.get(key) ?? 0) + 1);
  }
  const leftCounts = new Map<string | number, number>();
  let nullKeyLeftRowCount = 0;
  let matchedPairCount = 0;
  let unmatchedLeftRowCount = 0;
  for (let index = 0; index < leftKeys.length; index++) {
    const key = leftKeys[index];
    if (key === null || key === undefined) {
      nullKeyLeftRowCount++;
      unmatchedLeftRowCount++;
      continue;
    }
    leftCounts.set(key, (leftCounts.get(key) ?? 0) + 1);
    const matches = rightCounts.get(key) ?? 0;
    if (matches === 0) unmatchedLeftRowCount++;
    matchedPairCount += matches;
  }
  let unmatchedRightRowCount = nullKeyRightRowCount;
  let leftDuplicated = false;
  let rightDuplicated = false;
  for (const [key, count] of rightCounts) {
    if (!leftCounts.has(key)) unmatchedRightRowCount += count;
    if (count > 1 && leftCounts.has(key)) rightDuplicated = true;
  }
  for (const [key, count] of leftCounts)
    if (count > 1 && rightCounts.has(key)) leftDuplicated = true;
  const keepLeft = joinType === 'left' || joinType === 'full';
  const keepRight = joinType === 'right' || joinType === 'full';
  const outputRowCount =
    matchedPairCount +
    (keepLeft ? unmatchedLeftRowCount : 0) +
    (keepRight ? unmatchedRightRowCount : 0);
  const cardinality: JoinCardinality =
    leftDuplicated && rightDuplicated
      ? 'many-to-many'
      : rightDuplicated
        ? 'one-to-many'
        : leftDuplicated
          ? 'many-to-one'
          : 'one-to-one';
  return {
    cardinality,
    outputRowCount,
    matchedPairCount,
    unmatchedLeftRowCount,
    unmatchedRightRowCount,
    nullKeyLeftRowCount,
    nullKeyRightRowCount
  };
}

// ---- Spatial join semantics ------------------------------------------------------------------

export type SpatialJoinPredicate = 'intersects' | 'within';
export type PointPolygonRelation = 'inside' | 'boundary' | 'outside';
export type SpatialJoinPoint = readonly [number, number];
/** Rings: first is the exterior, the rest are holes. Closing vertex optional. */
export type SpatialJoinPolygon = readonly (readonly SpatialJoinPoint[])[];

/**
 * Classify a point against a polygon with holes. Boundary membership (on an edge or vertex of any
 * ring) is exact for axis-aligned and collinear cases through a cross-product test with an
 * absolute tolerance of 0, so a point on a hole edge is 'boundary', not 'inside'.
 */
export function classifyPointInPolygon(
  point: SpatialJoinPoint,
  polygon: SpatialJoinPolygon
): PointPolygonRelation {
  const [x, y] = point;
  let insideCount = 0;
  for (const [ringIndex, ring] of polygon.entries()) {
    let inside = false;
    for (let a = 0, b = ring.length - 1; a < ring.length; b = a++) {
      const [ax, ay] = ring[a];
      const [bx, by] = ring[b];
      const cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax);
      if (
        cross === 0 &&
        x >= Math.min(ax, bx) &&
        x <= Math.max(ax, bx) &&
        y >= Math.min(ay, by) &&
        y <= Math.max(ay, by)
      )
        return 'boundary';
      if (ay > y !== by > y && x < ((bx - ax) * (y - ay)) / (by - ay) + ax) inside = !inside;
    }
    if (ringIndex === 0 && !inside) return 'outside';
    if (ringIndex > 0 && inside) return 'outside';
    insideCount++;
  }
  return insideCount > 0 ? 'inside' : 'outside';
}

/** intersects admits boundary points; within requires the interior (boundary points do not match). */
export function matchesSpatialJoinPredicate(
  relation: PointPolygonRelation,
  predicate: SpatialJoinPredicate
): boolean {
  return predicate === 'intersects' ? relation !== 'outside' : relation === 'inside';
}

export type SpatialJoinSemantics = {
  predicate: SpatialJoinPredicate;
  boundary: 'inclusive' | 'exclusive';
  crs: string;
  nullGeometry: 'unmatched';
  cardinality: 'many-to-many';
};

/**
 * Declare spatial join semantics. Both sides must declare the same CRS: there is no reprojection,
 * so a mismatch is refused (returns a reason) rather than joined in the wrong plane.
 */
export function planSpatialJoinSemantics(
  predicate: SpatialJoinPredicate,
  leftCrs: string,
  rightCrs: string
): {semantics?: SpatialJoinSemantics; refusal?: string} {
  if (leftCrs !== rightCrs)
    return {
      refusal: `Spatial join sides declare different CRSs (${leftCrs}, ${rightCrs}); reproject first`
    };
  return {
    semantics: {
      predicate,
      boundary: predicate === 'intersects' ? 'inclusive' : 'exclusive',
      crs: leftCrs,
      nullGeometry: 'unmatched',
      cardinality: 'many-to-many'
    }
  };
}

/** Reference pair expansion for point-in-polygon, used as the test oracle for GPU output. */
export function listSpatialJoinPairs(
  points: readonly (SpatialJoinPoint | null)[],
  polygons: readonly (SpatialJoinPolygon | null)[],
  predicate: SpatialJoinPredicate
): [pointIndex: number, polygonIndex: number][] {
  const pairs: [number, number][] = [];
  for (const [pointIndex, point] of points.entries()) {
    if (!point) continue;
    for (const [polygonIndex, polygon] of polygons.entries())
      if (polygon && matchesSpatialJoinPredicate(classifyPointInPolygon(point, polygon), predicate))
        pairs.push([pointIndex, polygonIndex]);
  }
  return pairs;
}
