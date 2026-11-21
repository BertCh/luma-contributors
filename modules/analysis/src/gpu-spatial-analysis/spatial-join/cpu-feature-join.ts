// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {computeTriangulationOrientation} from '../../analysis-core/polygon-triangulation/cpu-triangulation-predicates';
import {validateCPUHostOffsets} from '../../analysis-core/operations/compact-host-offsets';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {computeFeatureRelate} from './cpu-feature-relate';
import {createCPUFeatureDistanceRank} from './cpu-nearest-distance-rank';
import {
  getPredicateRelatePatterns,
  parseRelatePattern,
  getRelatePatterns,
  doesRelatePatternAdmitDisjoint,
  type GPUSpatialRelatePattern
} from './spatial-relate-types';
import type {SpatialPredicateName} from './spatial-predicate-wgsl';

/** Packed planar CPU geometry; offsets and sourceIds have the shared geometry-port meanings. */
export type CPUSpatialJoinGeometry = {
  kind: 'points' | 'lines' | 'polygons';
  positions: Float32Array;
  featureOffsets?: Uint32Array;
  lineOffsets?: Uint32Array;
  polygonOffsets?: Uint32Array;
  ringOffsets?: Uint32Array;
  sourceIds?: Uint32Array;
};
/** Predicates with feature-level CPU and GPU reductions. */
export type MultipartJoinPredicate = SpatialPredicateName;
type Position = readonly [number, number];
/** Parsed CPU geometry used by exact feature-level topology. @internal */
export type CPUSpatialFeature = {
  kind: CPUSpatialJoinGeometry['kind'];
  parts: Position[][];
  bounds: number[];
};
type Feature = CPUSpatialFeature;

/** Compile-time predicate options shared by the pure CPU form and operation adapters. */
export type CPUFeatureJoinOptions = {
  predicate: MultipartJoinPredicate;
  pattern?: GPUSpatialRelatePattern;
  distance?: number;
  engine?: 'auto' | 'fast' | 'relate';
  how?: 'inner' | 'anti';
  /** Materialize packed matrices for retained inner pairs. Defaults to false (an empty array). */
  relate?: boolean;
  candidateCapacity: number;
  pairCapacity: number;
  /** Completeness of the supplied left prefix; loss suppresses unknown unmatched rows. */
  leftStatus?: {overflow?: number; candidateOverflow?: number; complete?: number};
};

/** Feature-pair rows and completeness diagnostics from the CPU predicate join. */
export type CPUSpatialPredicateJoinResult = {
  leftRows: Uint32Array;
  rightRows: Uint32Array;
  relate: Uint32Array;
  candidateTruncated: Uint32Array;
  candidateBoundsTestCount: number;
  /** Separate ascending absence result, derived before matched-pair truncation. */
  unmatchedRows: Uint32Array;
  unmatchedStatus: {
    count: number;
    requiredCount: number;
    overflow: number;
    /** Always returned, including for anti-only joins. One when candidates or left input were lost. */
    candidateOverflow: number;
  };
  status: {count: number; requiredCount: number; overflow: number; candidateOverflow: number};
};
/** One nearest feature per query; rows precede application-ID remapping. */
export type CPUNearestFeatureJoinResult = {
  neighborRows: Uint32Array;
  neighborIds: Uint32Array;
  neighborCounts: Uint32Array;
  neighborDistances: Float32Array;
  status: {overflow: number; candidateOverflow: number};
};

/** Shared signed-compatible cutoff for two-word predicate candidate addressing. @internal */
export const FEATURE_JOIN_MAXIMUM_PAIR_CAPACITY = 0x7ffffffe;
/** Shared CPU/GPU threshold limit: the GPU literal must remain finite float32. @internal */
export const FEATURE_JOIN_MAXIMUM_DISTANCE = 3.4028234663852886e38;

/** Shared construction validation for the planar operation forms. @internal */
export function validateFeatureJoinOptions(
  id: string,
  options: CPUFeatureJoinOptions,
  multipart: boolean
): void {
  validateCapacity(
    id,
    'candidateCapacity',
    options.candidateCapacity,
    1,
    FEATURE_JOIN_MAXIMUM_PAIR_CAPACITY
  );
  validateCapacity(id, 'pairCapacity', options.pairCapacity, 1, FEATURE_JOIN_MAXIMUM_PAIR_CAPACITY);
  for (const field of ['overflow', 'candidateOverflow', 'complete'] as const) {
    const value = options.leftStatus?.[field];
    if (value !== undefined) validateCapacity(id, `leftStatus.${field}`, value, 0, 0xffffffff);
  }
  validateRequirement(
    id,
    [
      'intersects',
      'within',
      'coveredBy',
      'contains',
      'covers',
      'containsProperly',
      'touches',
      'crosses',
      'overlaps',
      'equals',
      'relate',
      'dwithin'
    ].includes(options.predicate),
    'predicate must be a named spatial predicate or relate'
  );
  validateRequirement(
    id,
    options.how === undefined || ['inner', 'anti'].includes(options.how),
    'how must be inner or anti'
  );
  validateRequirement(
    id,
    options.engine === undefined || ['auto', 'fast', 'relate'].includes(options.engine),
    'engine must be auto, fast or relate'
  );
  validateRequirement(
    id,
    options.engine !== 'fast' ||
      ['intersects', 'within', 'contains', 'dwithin'].includes(options.predicate) ||
      (multipart && options.predicate === 'coveredBy'),
    'engine fast requires intersects, within, contains or dwithin (or coveredBy for multipart input); use engine relate for other predicates'
  );
  validateRequirement(
    id,
    options.predicate !== 'dwithin' ||
      (Number.isFinite(options.distance) && options.distance! >= 0),
    'distance must be finite and non-negative for dwithin'
  );
  if (options.predicate === 'dwithin')
    validateFiniteNumber(id, 'distance', options.distance!, 0, FEATURE_JOIN_MAXIMUM_DISTANCE);
  validateRequirement(
    id,
    !options.relate || (options.predicate !== 'dwithin' && options.how !== 'anti'),
    'relate requires an inner topological predicate; distance and anti results have no matrix output'
  );
  validateRequirement(
    id,
    options.predicate === 'relate' ? options.pattern !== undefined : options.pattern === undefined,
    'pattern is required exactly when predicate is relate'
  );
  if (options.predicate === 'relate') {
    const patterns = getRelatePatterns(options.pattern!);
    validateRequirement(
      id,
      patterns.length > 0 && !doesRelatePatternAdmitDisjoint(patterns),
      'pattern must be nonempty and exclude disjoint geometries; use how anti with intersects for disjoint'
    );
  }
}

function readFeatures(geometry: CPUSpatialJoinGeometry, skipNonFinite = false): Feature[] {
  const id = 'CPUFeatureJoin';
  const {positions, kind, featureOffsets, lineOffsets, polygonOffsets, ringOffsets} = geometry;
  validateRequirement(id, positions.length % 2 === 0, 'positions must contain complete XY rows');
  const vertexCount = positions.length / 2;
  const checkOffsets = (name: string, offsets: Uint32Array | undefined, childCount: number) => {
    validateRequirement(id, Boolean(offsets?.length), `${name} requires a terminal entry`);
    validateCPUHostOffsets(id, name, offsets!, childCount);
  };
  if (kind === 'lines') checkOffsets('lineOffsets', lineOffsets, vertexCount);
  if (kind === 'polygons') {
    checkOffsets('ringOffsets', ringOffsets, vertexCount);
    checkOffsets('polygonOffsets', polygonOffsets, ringOffsets!.length - 1);
    checkOffsets('featureOffsets', featureOffsets, polygonOffsets!.length - 1);
  } else if (featureOffsets)
    checkOffsets(
      'featureOffsets',
      featureOffsets,
      kind === 'points' ? vertexCount : lineOffsets!.length - 1
    );
  const featureCount = featureOffsets
    ? featureOffsets.length - 1
    : kind === 'points'
      ? vertexCount
      : lineOffsets!.length - 1;
  validateRequirement(
    id,
    !geometry.sourceIds || geometry.sourceIds.length === featureCount,
    'sourceIds length must equal feature count'
  );
  const features: Feature[] = [];
  for (let row = 0; row < featureCount; row++) {
    let first = featureOffsets ? featureOffsets[row] : row;
    let end = featureOffsets ? featureOffsets[row + 1] : row + 1;
    if (kind === 'polygons') {
      first = polygonOffsets![first];
      end = polygonOffsets![end];
    }
    const parts: Position[][] = [];
    const bounds = [Infinity, Infinity, -Infinity, -Infinity];
    for (let part = first; part < end; part++) {
      const start =
        kind === 'points' ? part : (kind === 'lines' ? lineOffsets! : ringOffsets!)[part];
      const stop =
        kind === 'points' ? part + 1 : (kind === 'lines' ? lineOffsets! : ringOffsets!)[part + 1];
      const vertices: Position[] = [];
      for (let vertex = start; vertex < stop; vertex++) {
        const position: Position = [positions[vertex * 2], positions[vertex * 2 + 1]];
        const finite = position.every(Number.isFinite);
        if (!skipNonFinite) validateRequirement(id, finite, 'positions must be finite');
        vertices.push(position);
        if (!finite) continue;
        bounds[0] = Math.min(bounds[0], position[0]);
        bounds[1] = Math.min(bounds[1], position[1]);
        bounds[2] = Math.max(bounds[2], position[0]);
        bounds[3] = Math.max(bounds[3], position[1]);
      }
      // Bounds retain short-part vertices like the GPU builder; exact predicates treat those
      // parts as empty. This preserves candidate ordering and budget loss on both forms.
      if (stop - start >= (kind === 'points' ? 1 : kind === 'lines' ? 2 : 3)) {
        const usable =
          !skipNonFinite ||
          vertices.some(
            (position, vertex) =>
              position.every(Number.isFinite) &&
              (kind === 'points' ||
                ((kind === 'polygons' || vertex + 1 < vertices.length) &&
                  vertices[(vertex + 1) % vertices.length].every(Number.isFinite)))
          );
        if (usable) parts.push(vertices);
      }
    }
    features.push({kind, parts, bounds});
  }
  return features;
}

function* getFeatureEdges(feature: Feature): Generator<readonly [Position, Position]> {
  for (const part of feature.parts)
    for (
      let vertex = 0;
      vertex <
      (feature.kind === 'points' ? 1 : feature.kind === 'polygons' ? part.length : part.length - 1);
      vertex++
    ) {
      const start = part[vertex];
      const end = part[feature.kind === 'points' ? vertex : (vertex + 1) % part.length];
      // Nearest geometry permits unusable vertices; never bridge a gap across one.
      if (start.every(Number.isFinite) && end.every(Number.isFinite)) yield [start, end];
    }
}
function arePositionsEqual(first: Position, second: Position): boolean {
  return first[0] === second[0] && first[1] === second[1];
}
function getOrientationSign(first: Position, second: Position, query: Position): number {
  const orientation = computeTriangulationOrientation(first, second, query);
  // The shared predicate uses 2 for nonfinite inputs; do not treat that as a positive sign.
  return orientation === 2 ? Number.NaN : orientation;
}
function isPositionOnSegment(first: Position, second: Position, query: Position): boolean {
  return (
    getOrientationSign(first, second, query) === 0 &&
    query[0] >= Math.min(first[0], second[0]) &&
    query[0] <= Math.max(first[0], second[0]) &&
    query[1] >= Math.min(first[1], second[1]) &&
    query[1] <= Math.max(first[1], second[1])
  );
}
function locatePosition(feature: Feature, query: Position): number {
  if (feature.kind === 'points')
    return feature.parts.some(part => arePositionsEqual(part[0], query)) ? 1 : 0;
  let inside = false;
  let found = false;
  for (const [first, second] of getFeatureEdges(feature)) {
    if (isPositionOnSegment(first, second, query)) found = true;
    if (
      feature.kind === 'polygons' &&
      first[1] > query[1] !== second[1] > query[1] &&
      second[1] > first[1] === getOrientationSign(first, second, query) > 0
    )
      inside = !inside;
  }
  if (feature.kind === 'polygons') return found ? 2 : inside ? 1 : 0;
  let endpoints = 0;
  for (const part of feature.parts)
    if (!arePositionsEqual(part[0], part[part.length - 1]))
      endpoints +=
        Number(arePositionsEqual(part[0], query)) +
        Number(arePositionsEqual(part[part.length - 1], query));
  return found ? (endpoints % 2 ? 2 : 1) : 0;
}
function intersectSegments(
  first: Position,
  second: Position,
  third: Position,
  fourth: Position
): boolean {
  return (
    isPositionOnSegment(first, second, third) ||
    isPositionOnSegment(first, second, fourth) ||
    isPositionOnSegment(third, fourth, first) ||
    isPositionOnSegment(third, fourth, second) ||
    (getOrientationSign(first, second, third) * getOrientationSign(first, second, fourth) < 0 &&
      getOrientationSign(third, fourth, first) * getOrientationSign(third, fourth, second) < 0)
  );
}
function doFeaturesIntersect(first: Feature, second: Feature): boolean {
  if (!first.parts.length || !second.parts.length) return false;
  for (const [start, end] of getFeatureEdges(first)) {
    if (second.kind === 'polygons' && locatePosition(second, start)) return true;
    for (const [otherStart, otherEnd] of getFeatureEdges(second))
      if (intersectSegments(start, end, otherStart, otherEnd)) return true;
  }
  return (
    first.kind === 'polygons' &&
    second.parts.some(part =>
      part.some(
        position => position.every(Number.isFinite) && locatePosition(first, position) !== 0
      )
    )
  );
}

function doBoundsOverlap(first: number[], second: number[]): boolean {
  return (
    first[0] <= second[2] && second[0] <= first[2] && first[1] <= second[3] && second[1] <= first[3]
  );
}

/** Exact planar matrix of one feature pair, including empty operands. @internal */
export function computeSpatialRelateOnCPU(
  left: CPUSpatialJoinGeometry,
  right: CPUSpatialJoinGeometry,
  leftRow = 0,
  rightRow = 0
): number {
  const leftFeatures = readFeatures(left),
    rightFeatures = readFeatures(right);
  validateRequirement(
    'CPUFeatureJoin',
    Boolean(leftFeatures[leftRow] && rightFeatures[rightRow]),
    'leftRow and rightRow must identify existing features'
  );
  return computeFeatureRelate(leftFeatures[leftRow], rightFeatures[rightRow]);
}

/**
 * Planar feature-level predicate join. Candidate slots count logical feature pairs in row order.
 * What differs from the GPU contributor: rational CPU topology has no orientation uncertainty;
 * distances compute in float64. Empty features have no candidates. In anti mode leftRows are
 * unmatched left rows and rightRows retain the missing-row sentinel.
 */
export function computeSpatialPredicateJoinOnCPU(
  left: CPUSpatialJoinGeometry,
  right: CPUSpatialJoinGeometry,
  options: CPUFeatureJoinOptions
): CPUSpatialPredicateJoinResult {
  const capturedOptions = snapshotConstructionFields(options, [
    'predicate',
    'pattern',
    'distance',
    'engine',
    'how',
    'relate',
    'candidateCapacity',
    'pairCapacity',
    'leftStatus'
  ]);
  options = {
    ...capturedOptions,
    leftStatus:
      capturedOptions.leftStatus &&
      snapshotConstructionFields(capturedOptions.leftStatus, [
        'overflow',
        'candidateOverflow',
        'complete'
      ])
  };
  validateFeatureJoinOptions(
    'CPUFeatureJoin',
    options,
    Boolean(
      (left.kind !== 'polygons' && left.featureOffsets) ||
        (right.kind !== 'polygons' && right.featureOffsets)
    )
  );
  const patterns =
    options.predicate === 'relate'
      ? getRelatePatterns(options.pattern!)
      : options.predicate === 'dwithin'
        ? []
        : getPredicateRelatePatterns(
            options.predicate,
            left.kind === 'points' ? 0 : left.kind === 'lines' ? 1 : 2,
            right.kind === 'points' ? 0 : right.kind === 'lines' ? 1 : 2
          );
  const masks = patterns.map(parseRelatePattern);
  const leftFeatures = readFeatures(left),
    rightFeatures = readFeatures(right);
  const ordered = rightFeatures
    .map((feature, row) => ({feature, row}))
    .sort((first, second) => first.feature.bounds[0] - second.feature.bounds[0]);
  let leafCapacity = 1;
  while (leafCapacity < ordered.length) leafCapacity *= 2;
  const boundsIndex: number[][] = Array.from({length: leafCapacity * 2}, () => [
    Infinity,
    Infinity,
    -Infinity,
    -Infinity
  ]);
  for (let row = 0; row < ordered.length; row++)
    boundsIndex[leafCapacity + row] = ordered[row].feature.bounds;
  for (let node = leafCapacity - 1; node > 0; node--) {
    const first = boundsIndex[node * 2],
      second = boundsIndex[node * 2 + 1];
    boundsIndex[node] = [
      Math.min(first[0], second[0]),
      Math.min(first[1], second[1]),
      Math.max(first[2], second[2]),
      Math.max(first[3], second[3])
    ];
  }
  const leftRows = new Uint32Array(options.pairCapacity).fill(0xffffffff),
    rightRows = new Uint32Array(options.pairCapacity).fill(0xffffffff);
  const relate = new Uint32Array(options.relate ? options.pairCapacity : 0);
  const matched = new Uint32Array(leftFeatures.length);
  const candidateTruncated = new Uint32Array(leftFeatures.length);
  const inputLoss = Boolean(
    options.leftStatus?.overflow ||
      options.leftStatus?.candidateOverflow ||
      options.leftStatus?.complete === 0
  );
  let candidateCount = 0,
    requiredCount = 0;
  let candidateBoundsTestCount = 0;
  for (const [leftRow, feature] of leftFeatures.entries()) {
    if (options.leftStatus?.complete === 0) continue;
    const candidates: typeof ordered = [];
    const pending = [1];
    while (pending.length) {
      const node = pending.pop()!;
      candidateBoundsTestCount++;
      const margin = options.predicate === 'dwithin' ? options.distance! : 0;
      if (
        !doBoundsOverlap(
          [
            feature.bounds[0] - margin,
            feature.bounds[1] - margin,
            feature.bounds[2] + margin,
            feature.bounds[3] + margin
          ],
          boundsIndex[node]
        )
      )
        continue;
      if (node >= leafCapacity) {
        const row = node - leafCapacity;
        if (row < ordered.length) candidates.push(ordered[row]);
      } else {
        pending.push(node * 2 + 1, node * 2);
      }
    }
    candidates.sort((first, second) => first.row - second.row);
    for (const candidate of candidates) {
      if (candidateCount++ >= options.candidateCapacity) {
        candidateTruncated[leftRow] = 1;
        continue;
      }
      const matrix =
        options.relate || (options.predicate !== 'intersects' && options.predicate !== 'dwithin')
          ? computeFeatureRelate(feature, candidate.feature)
          : 0;
      const match =
        options.predicate === 'intersects' && !options.relate
          ? doFeaturesIntersect(feature, candidate.feature)
          : options.predicate === 'dwithin'
            ? areFeaturesWithinDistance(feature, candidate.feature, options.distance!)
            : masks.some(pattern =>
                pattern.every((mask, cell) => (mask & (1 << ((matrix >>> (cell * 2)) & 3))) !== 0)
              );
      if (!match) continue;
      matched[leftRow] = 1;
      if (options.how === 'anti') continue;
      if (requiredCount < options.pairCapacity) {
        leftRows[requiredCount] = leftRow;
        rightRows[requiredCount] = candidate.row;
        if (options.relate) relate[requiredCount] = matrix;
      }
      requiredCount++;
    }
  }
  const unmatchedRows = new Uint32Array(options.pairCapacity).fill(0xffffffff);
  let unmatchedRequiredCount = 0;
  for (let row = 0; row < leftFeatures.length; row++) {
    if (inputLoss) candidateTruncated[row] = 1;
    if (matched[row] || candidateTruncated[row]) continue;
    if (unmatchedRequiredCount < options.pairCapacity) unmatchedRows[unmatchedRequiredCount] = row;
    unmatchedRequiredCount++;
  }
  const candidateOverflow = Number(inputLoss || candidateCount > options.candidateCapacity);
  if (options.how === 'anti') {
    leftRows.set(unmatchedRows);
    requiredCount = unmatchedRequiredCount;
  }
  return {
    leftRows,
    rightRows,
    relate,
    candidateTruncated,
    candidateBoundsTestCount,
    unmatchedRows,
    unmatchedStatus: {
      count: Math.min(unmatchedRequiredCount, options.pairCapacity),
      requiredCount: unmatchedRequiredCount,
      overflow: Number(unmatchedRequiredCount > options.pairCapacity),
      candidateOverflow
    },
    status: {
      count: Math.min(requiredCount, options.pairCapacity),
      requiredCount,
      overflow: Number(requiredCount > options.pairCapacity),
      candidateOverflow
    }
  };
}

function areFeaturesWithinDistance(first: Feature, second: Feature, distance: number): boolean {
  if (!first.parts.length || !second.parts.length) return false;
  if (doFeaturesIntersect(first, second)) return true;
  if (distance === 0) return false;
  for (const [start, end] of getFeatureEdges(first))
    for (const [otherStart, otherEnd] of getFeatureEdges(second))
      if (
        isPointWithinSegmentDistance(start, otherStart, otherEnd, distance) ||
        isPointWithinSegmentDistance(end, otherStart, otherEnd, distance) ||
        isPointWithinSegmentDistance(otherStart, start, end, distance) ||
        isPointWithinSegmentDistance(otherEnd, start, end, distance)
      )
        return true;
  return false;
}

export function isPointWithinSegmentDistance(
  query: Position,
  start: Position,
  end: Position,
  distance: number
): boolean {
  const direction = [end[0] - start[0], end[1] - start[1]];
  const offset = [query[0] - start[0], query[1] - start[1]];
  const lengthSquared = direction[0] ** 2 + direction[1] ** 2;
  const projection = offset[0] * direction[0] + offset[1] * direction[1];
  const fromEnd = [query[0] - end[0], query[1] - end[1]];
  const endProjection = fromEnd[0] * direction[0] + fromEnd[1] * direction[1];
  const firstProduct = offset[0] * direction[1];
  const secondProduct = offset[1] * direction[0];
  const cross = firstProduct - secondProduct;
  const perpendicular = lengthSquared > 0 && projection > 0 && endProjection < 0;
  const actual = perpendicular
    ? cross ** 2
    : projection <= 0 || lengthSquared === 0
      ? offset[0] ** 2 + offset[1] ** 2
      : fromEnd[0] ** 2 + fromEnd[1] ** 2;
  const limit = distance ** 2 * (perpendicular ? lengthSquared : 1);
  const crossError = (Math.abs(firstProduct) + Math.abs(secondProduct)) * Number.EPSILON * 16;
  const comparisonError =
    (Math.abs(actual) + Math.abs(limit)) * Number.EPSILON * 32 +
    (perpendicular ? crossError * (2 * Math.abs(cross) + crossError) : 0);
  // Dot products at opposite endpoints avoid subtracting lengthSquared from a rounded
  // projection. Include coordinate subtraction error before trusting either sign.
  const coordinateScale = Math.max(
    ...query.map(Math.abs),
    ...start.map(Math.abs),
    ...end.map(Math.abs)
  );
  const projectionError =
    Number.EPSILON * 32 * coordinateScale * (Math.abs(direction[0]) + Math.abs(direction[1]));
  if (
    ((start[0] === end[0] && start[1] === end[1]) ||
      (Math.abs(projection) > projectionError && Math.abs(endProjection) > projectionError)) &&
    Math.abs(actual - limit) > comparisonError
  )
    return actual < limit;
  // Resolve threshold ties on the exact represented coordinates and caller threshold. All
  // integers use the common 2^-1074 unit, so no rounded projection or square root is needed.
  const bits = new DataView(new ArrayBuffer(8));
  const integer = (value: number): bigint => {
    bits.setFloat64(0, value);
    const word = bits.getBigUint64(0),
      exponent = Number((word >> 52n) & 2047n);
    const significand = (word & ((1n << 52n) - 1n)) | (exponent ? 1n << 52n : 0n);
    return (word >> 63n ? -significand : significand) << BigInt(Math.max(0, exponent - 1));
  };
  const directionX = integer(end[0]) - integer(start[0]),
    directionY = integer(end[1]) - integer(start[1]);
  let offsetX = integer(query[0]) - integer(start[0]),
    offsetY = integer(query[1]) - integer(start[1]);
  const exactLength = directionX ** 2n + directionY ** 2n;
  const exactProjection = offsetX * directionX + offsetY * directionY;
  const exactLimit = integer(distance) ** 2n;
  if (exactLength > 0n && exactProjection > 0n && exactProjection < exactLength)
    return (offsetX * directionY - offsetY * directionX) ** 2n <= exactLimit * exactLength;
  if (exactProjection > 0n) {
    offsetX = integer(query[0]) - integer(end[0]);
    offsetY = integer(query[1]) - integer(end[1]);
  }
  return offsetX ** 2n + offsetY ** 2n <= exactLimit;
}

/** Distance on original float32 coordinates, retaining direction-invariant determinants. @internal */
export function getPointSegmentDistance(query: Position, start: Position, end: Position): number {
  const difference = [end[0] - start[0], end[1] - start[1]];
  const length = Math.hypot(...difference);
  if (!length) return Math.hypot(query[0] - start[0], query[1] - start[1]);
  const fromStart = [query[0] - start[0], query[1] - start[1]];
  const fromEnd = [query[0] - end[0], query[1] - end[1]];
  if (fromStart[0] * difference[0] + fromStart[1] * difference[1] <= 0)
    return Math.hypot(...fromStart);
  if (fromEnd[0] * difference[0] + fromEnd[1] * difference[1] >= 0) return Math.hypot(...fromEnd);
  // Original float32 coordinates are dyadic integers in units of 2^-149. Computing the
  // six-product determinant in that integer domain avoids subtraction/projection cancellation
  // and gives identical magnitudes when a segment's vertices are reversed.
  const bits = new DataView(new ArrayBuffer(4));
  const integer = (value: number): bigint => {
    bits.setFloat32(0, value);
    const word = bits.getUint32(0),
      exponent = (word >>> 23) & 255;
    const significand = BigInt((word & 0x7fffff) | (exponent ? 0x800000 : 0));
    return (word >>> 31 ? -significand : significand) << BigInt(Math.max(0, exponent - 1));
  };
  const startX = integer(start[0]),
    startY = integer(start[1]);
  const endX = integer(end[0]),
    endY = integer(end[1]);
  const queryX = integer(query[0]),
    queryY = integer(query[1]);
  const determinant = (queryX - startX) * (endY - startY) - (queryY - startY) * (endX - startX);
  return Math.abs(Number(determinant) * 2 ** -298) / length;
}
/**
 * Exact one-neighbor planar feature search, with minimum distance over all usable parts.
 * Nonfinite point parts and edges are skipped; no usable edge means a missing neighbor.
 */
export function computeNearestFeatureJoinOnCPU(
  queries: CPUSpatialJoinGeometry,
  targets: CPUSpatialJoinGeometry
): CPUNearestFeatureJoinResult {
  const queryFeatures = readFeatures(queries, true),
    targetFeatures = readFeatures(targets, true);
  const getDistanceRank = createCPUFeatureDistanceRank();
  // A sorted x index with prefix/suffix union bounds supports outward branch-and-bound search.
  // Whole remaining ranges are pruned before any part-distance evaluation, including wide features
  // whose minimum x is far from the query but whose later parts reach it.
  const ordered = targetFeatures
    .map((feature, row) => ({feature, row}))
    .filter(item => item.feature.parts.length)
    .sort((first, second) => first.feature.bounds[0] - second.feature.bounds[0]);
  const merge = (first: number[], second: number[]) => [
    Math.min(first[0], second[0]),
    Math.min(first[1], second[1]),
    Math.max(first[2], second[2]),
    Math.max(first[3], second[3])
  ];
  const prefix: number[][] = [],
    suffix: number[][] = [];
  for (let row = 0; row < ordered.length; row++)
    prefix[row] = row
      ? merge(prefix[row - 1], ordered[row].feature.bounds)
      : ordered[row].feature.bounds;
  for (let row = ordered.length - 1; row >= 0; row--)
    suffix[row] =
      row + 1 < ordered.length
        ? merge(suffix[row + 1], ordered[row].feature.bounds)
        : ordered[row].feature.bounds;
  const boundDistance = (first: number[], second: number[]) =>
    getDistanceRank(
      Math.hypot(
        Math.max(0, first[0] - second[2], second[0] - first[2]),
        Math.max(0, first[1] - second[3], second[1] - first[3])
      )
    );
  const neighborRows = new Uint32Array(queryFeatures.length).fill(0xffffffff),
    neighborIds = neighborRows.slice();
  const neighborCounts = new Uint32Array(queryFeatures.length),
    neighborDistances = new Float32Array(queryFeatures.length).fill(-1);
  for (const [queryRow, query] of queryFeatures.entries()) {
    if (!query.parts.length) continue;
    let best = 0xff7fffff;
    let low = 0,
      high = ordered.length;
    const center = query.bounds[0] / 2 + query.bounds[2] / 2;
    while (low < high) {
      const middle = low + Math.floor((high - low) / 2);
      if (ordered[middle].feature.bounds[0] < center) low = middle + 1;
      else high = middle;
    }
    let before = low - 1,
      after = low;
    while (before >= 0 || after < ordered.length) {
      const beforeDistance = before >= 0 ? boundDistance(query.bounds, prefix[before]) : Infinity;
      const afterDistance =
        after < ordered.length ? boundDistance(query.bounds, suffix[after]) : Infinity;
      if (Math.min(beforeDistance, afterDistance) > best) break;
      const entry =
        before >= 0 && (after >= ordered.length || beforeDistance <= afterDistance)
          ? ordered[before--]
          : ordered[after++];
      const {feature: target, row: targetRow} = entry;
      if (boundDistance(query.bounds, target.bounds) > best) continue;
      let distance = doFeaturesIntersect(query, target) ? 0 : Infinity;
      if (distance !== 0)
        for (const [start, end] of getFeatureEdges(query))
          for (const [otherStart, otherEnd] of getFeatureEdges(target))
            distance = Math.min(
              distance,
              getPointSegmentDistance(start, otherStart, otherEnd),
              getPointSegmentDistance(end, otherStart, otherEnd),
              getPointSegmentDistance(otherStart, start, end),
              getPointSegmentDistance(otherEnd, start, end)
            );
      const distanceRank = getDistanceRank(distance);
      if (
        distanceRank < best ||
        (Number.isFinite(distance) && distanceRank === best && targetRow < neighborRows[queryRow])
      ) {
        best = distanceRank;
        neighborRows[queryRow] = targetRow;
        neighborIds[queryRow] = targets.sourceIds?.[targetRow] ?? targetRow;
        neighborCounts[queryRow] = 1;
        neighborDistances[queryRow] =
          distanceRank === 0xff7ffffc
            ? Math.fround(3.402823466e38)
            : distanceRank === 0xff7ffffd
              ? Infinity
              : Math.fround(distance);
      }
    }
  }
  return {
    neighborRows,
    neighborIds,
    neighborCounts,
    neighborDistances,
    status: {overflow: 0, candidateOverflow: 0}
  };
}
