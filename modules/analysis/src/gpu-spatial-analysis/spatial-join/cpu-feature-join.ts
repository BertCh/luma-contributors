// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

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
export type MultipartJoinPredicate = 'intersects' | 'within' | 'coveredBy';
type Position = readonly [number, number];
type Feature = {kind: CPUSpatialJoinGeometry['kind']; parts: Position[][]; bounds: number[]};

/** Shared signed-compatible cutoff for two-word predicate candidate addressing. @internal */
export const FEATURE_JOIN_MAXIMUM_PAIR_CAPACITY = 0x7ffffffe;

function readFeatures(geometry: CPUSpatialJoinGeometry): Feature[] {
  const id = 'CPUFeatureJoin';
  const {positions, kind, featureOffsets, lineOffsets, polygonOffsets, ringOffsets} = geometry;
  validateRequirement(id, positions.length % 2 === 0, 'positions must contain complete XY rows');
  const vertexCount = positions.length / 2;
  const checkOffsets = (name: string, offsets: Uint32Array | undefined, childCount: number) => {
    validateRequirement(id, Boolean(offsets?.length), `${name} requires a terminal entry`);
    validateRequirement(
      id,
      offsets![0] === 0 && offsets![offsets!.length - 1] === childCount,
      `${name} must span all child rows`
    );
    for (let row = 1; row < offsets!.length; row++)
      validateRequirement(id, offsets![row - 1] <= offsets![row], `${name} must be nondecreasing`);
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
        validateRequirement(id, position.every(Number.isFinite), 'positions must be finite');
        vertices.push(position);
        bounds[0] = Math.min(bounds[0], position[0]);
        bounds[1] = Math.min(bounds[1], position[1]);
        bounds[2] = Math.max(bounds[2], position[0]);
        bounds[3] = Math.max(bounds[3], position[1]);
      }
      // Bounds retain short-part vertices like the GPU builder; exact predicates treat those
      // parts as empty. This preserves candidate ordering and budget loss on both forms.
      if (stop - start >= (kind === 'points' ? 1 : kind === 'lines' ? 2 : 3)) parts.push(vertices);
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
      yield [part[vertex], part[feature.kind === 'points' ? vertex : (vertex + 1) % part.length]];
    }
}
function arePositionsEqual(first: Position, second: Position): boolean {
  return first[0] === second[0] && first[1] === second[1];
}
function getOrientationSign(first: Position, second: Position, query: Position): number {
  const firstProduct = (second[0] - first[0]) * (query[1] - first[1]);
  const secondProduct = (second[1] - first[1]) * (query[0] - first[0]);
  const determinant = firstProduct - secondProduct;
  if (
    Math.abs(determinant) >
    (Math.abs(firstProduct) + Math.abs(secondProduct)) * Number.EPSILON * 8
  )
    return Math.sign(determinant);
  // Exact dyadic fallback for the represented doubles, including derived segment probes.
  const bits = new DataView(new ArrayBuffer(8));
  const integer = (value: number) => {
    bits.setFloat64(0, value);
    const word = bits.getBigUint64(0);
    const exponent = Number((word >> 52n) & 2047n);
    const significand = (word & ((1n << 52n) - 1n)) | (exponent ? 1n << 52n : 0n);
    return (word >> 63n ? -significand : significand) << BigInt(Math.max(0, exponent - 1));
  };
  const exact =
    (integer(second[0]) - integer(first[0])) * (integer(query[1]) - integer(first[1])) -
    (integer(second[1]) - integer(first[1])) * (integer(query[0]) - integer(first[0]));
  return exact < 0n ? -1 : exact > 0n ? 1 : 0;
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
    second.parts.some(part => part.some(position => locatePosition(first, position) !== 0))
  );
}

// Keep boundary crossings as rational parameters over the supplied dyadic coordinates. A
// reconstructed floating-point intersection or midpoint can move off an oblique boundary.
function getPolygonSegmentCoverage(
  polygon: Feature,
  start: Position,
  end: Position
): [boolean, boolean] {
  if (arePositionsEqual(start, end)) {
    const location = locatePosition(polygon, start);
    return [location !== 0, location === 1];
  }
  const bits = new DataView(new ArrayBuffer(8));
  const integer = (value: number): bigint => {
    bits.setFloat64(0, value);
    const word = bits.getBigUint64(0),
      exponent = Number((word >> 52n) & 2047n);
    const significand = (word & ((1n << 52n) - 1n)) | (exponent ? 1n << 52n : 0n);
    return (word >> 63n ? -significand : significand) << BigInt(Math.max(0, exponent - 1));
  };
  type ExactPosition = readonly [bigint, bigint];
  type Fraction = readonly [bigint, bigint];
  const exact = (point: Position): ExactPosition => [integer(point[0]), integer(point[1])];
  const subtract = (first: ExactPosition, second: ExactPosition): ExactPosition => [
    first[0] - second[0],
    first[1] - second[1]
  ];
  const cross = (first: ExactPosition, second: ExactPosition) =>
    first[0] * second[1] - first[1] * second[0];
  const first = exact(start),
    direction = subtract(exact(end), first);
  const edges = Array.from(
    getFeatureEdges(polygon),
    ([edgeStart, edgeEnd]) => [exact(edgeStart), exact(edgeEnd)] as const
  );
  const breaks: Fraction[] = [
    [0n, 1n],
    [1n, 1n]
  ];
  const append = (numerator: bigint, denominator: bigint) => {
    if (denominator < 0n) {
      numerator = -numerator;
      denominator = -denominator;
    }
    if (numerator >= 0n && numerator <= denominator) breaks.push([numerator, denominator]);
  };
  const axis = direction[0] * direction[0] >= direction[1] * direction[1] ? 0 : 1;
  for (const [edgeStart, edgeEnd] of edges) {
    const edge = subtract(edgeEnd, edgeStart),
      offset = subtract(edgeStart, first),
      denominator = cross(direction, edge);
    if (denominator !== 0n) {
      const alongEdge = cross(offset, direction);
      if (
        denominator > 0n
          ? alongEdge >= 0n && alongEdge <= denominator
          : alongEdge <= 0n && alongEdge >= denominator
      )
        append(cross(offset, edge), denominator);
    } else if (cross(offset, direction) === 0n) {
      append(edgeStart[axis] - first[axis], direction[axis]);
      append(edgeEnd[axis] - first[axis], direction[axis]);
    }
  }
  breaks.sort((firstFraction, secondFraction) => {
    const difference = firstFraction[0] * secondFraction[1] - secondFraction[0] * firstFraction[1];
    return difference < 0n ? -1 : difference > 0n ? 1 : 0;
  });
  const locate = ([numerator, denominator]: Fraction): number => {
    const query: ExactPosition = [
      first[0] * denominator + direction[0] * numerator,
      first[1] * denominator + direction[1] * numerator
    ];
    let inside = false;
    for (const [edgeStart, edgeEnd] of edges) {
      const side = cross(subtract(edgeEnd, edgeStart), [
        query[0] - edgeStart[0] * denominator,
        query[1] - edgeStart[1] * denominator
      ]);
      if (
        side === 0n &&
        query[0] >= (edgeStart[0] < edgeEnd[0] ? edgeStart[0] : edgeEnd[0]) * denominator &&
        query[0] <= (edgeStart[0] > edgeEnd[0] ? edgeStart[0] : edgeEnd[0]) * denominator &&
        query[1] >= (edgeStart[1] < edgeEnd[1] ? edgeStart[1] : edgeEnd[1]) * denominator &&
        query[1] <= (edgeStart[1] > edgeEnd[1] ? edgeStart[1] : edgeEnd[1]) * denominator
      )
        return 2;
      if (
        edgeStart[1] * denominator > query[1] !== edgeEnd[1] * denominator > query[1] &&
        edgeEnd[1] > edgeStart[1] === side > 0n
      )
        inside = !inside;
    }
    return inside ? 1 : 0;
  };
  let interior = false;
  for (let index = 0; index < breaks.length; index++) {
    const current = breaks[index],
      next = breaks[index + 1];
    const samples: Fraction[] = [current];
    if (next && current[0] * next[1] < next[0] * current[1])
      samples.push([current[0] * next[1] + next[0] * current[1], 2n * current[1] * next[1]]);
    for (const fraction of samples) {
      const location = locate(fraction);
      if (!location) return [false, false];
      interior ||= location === 1;
    }
  }
  return [true, interior];
}

function containsFeature(outer: Feature, inner: Feature, includeBoundary: boolean): boolean {
  if (!outer.parts.length || !inner.parts.length) return false;
  if (inner.kind === 'polygons' && outer.kind !== 'polygons') return false;
  if (outer.kind === 'points' && inner.kind !== 'points') return false;
  let interior = false;
  for (const [start, end] of getFeatureEdges(inner)) {
    if (outer.kind === 'lines' && !arePositionsEqual(start, end)) {
      // Merge supplied endpoint intervals, never reconstructed parameter midpoints. A rounded
      // midpoint of an oblique segment need not remain exactly collinear with its endpoints.
      const axis = Math.abs(end[0] - start[0]) >= Math.abs(end[1] - start[1]) ? 0 : 1;
      let reach = Math.min(start[axis], end[axis]);
      const limit = Math.max(start[axis], end[axis]);
      const intervals: number[][] = [];
      for (const [first, second] of getFeatureEdges(outer)) {
        if (
          getOrientationSign(start, end, first) === 0 &&
          getOrientationSign(start, end, second) === 0
        )
          intervals.push([
            Math.min(first[axis], second[axis]),
            Math.max(first[axis], second[axis])
          ]);
      }
      intervals.sort((first, second) => first[0] - second[0]);
      for (const [first, last] of intervals) {
        if (first > reach) break;
        reach = Math.max(reach, last);
        if (reach >= limit) break;
      }
      if (reach < limit) return false;
      interior = true;
      continue;
    }
    if (outer.kind === 'polygons') {
      const [covered, hasInterior] = getPolygonSegmentCoverage(outer, start, end);
      if (!covered) return false;
      interior ||= hasInterior;
    } else {
      const location = locatePosition(outer, start);
      if (!location) return false;
      interior ||= location === 1;
    }
  }
  return includeBoundary || interior;
}
function doBoundsOverlap(first: number[], second: number[]): boolean {
  return (
    first[0] <= second[2] && second[0] <= first[2] && first[1] <= second[3] && second[1] <= first[3]
  );
}

/** Planar feature-level predicate join. Candidate slots count logical feature pairs in row order. */
export function computeSpatialPredicateJoinOnCPU(
  left: CPUSpatialJoinGeometry,
  right: CPUSpatialJoinGeometry,
  options: {predicate: MultipartJoinPredicate; candidateCapacity: number; pairCapacity: number}
) {
  validateCapacity(
    'CPUFeatureJoin',
    'candidateCapacity',
    options.candidateCapacity,
    1,
    FEATURE_JOIN_MAXIMUM_PAIR_CAPACITY
  );
  validateCapacity(
    'CPUFeatureJoin',
    'pairCapacity',
    options.pairCapacity,
    1,
    FEATURE_JOIN_MAXIMUM_PAIR_CAPACITY
  );
  validateRequirement(
    'CPUFeatureJoin',
    ['intersects', 'within', 'coveredBy'].includes(options.predicate),
    'predicate must be intersects, within or coveredBy'
  );
  validateRequirement(
    'CPUFeatureJoin',
    left.kind !== 'polygons' || right.kind !== 'polygons' || options.predicate === 'intersects',
    'polygon/polygon containment is unsupported by the feature-join operation forms'
  );
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
  const candidateTruncated = new Uint32Array(leftFeatures.length);
  let candidateCount = 0,
    requiredCount = 0;
  let candidateBoundsTestCount = 0;
  for (const [leftRow, feature] of leftFeatures.entries()) {
    const candidates: typeof ordered = [];
    const pending = [1];
    while (pending.length) {
      const node = pending.pop()!;
      candidateBoundsTestCount++;
      if (!doBoundsOverlap(feature.bounds, boundsIndex[node])) continue;
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
      const match =
        options.predicate === 'intersects'
          ? doFeaturesIntersect(feature, candidate.feature)
          : containsFeature(candidate.feature, feature, options.predicate === 'coveredBy');
      if (!match) continue;
      if (requiredCount < options.pairCapacity) {
        leftRows[requiredCount] = leftRow;
        rightRows[requiredCount] = candidate.row;
      }
      requiredCount++;
    }
  }
  return {
    leftRows,
    rightRows,
    candidateTruncated,
    candidateBoundsTestCount,
    status: {
      count: Math.min(requiredCount, options.pairCapacity),
      requiredCount,
      overflow: Number(requiredCount > options.pairCapacity),
      candidateOverflow: Number(candidateCount > options.candidateCapacity)
    }
  };
}

function getPointSegmentDistance(query: Position, start: Position, end: Position): number {
  const difference = [end[0] - start[0], end[1] - start[1]];
  const length = Math.hypot(...difference);
  if (!length) return Math.hypot(query[0] - start[0], query[1] - start[1]);
  const fromStart = [query[0] - start[0], query[1] - start[1]];
  const fromEnd = [query[0] - end[0], query[1] - end[1]];
  if (fromStart[0] * difference[0] + fromStart[1] * difference[1] <= 0)
    return Math.hypot(...fromStart);
  if (fromEnd[0] * difference[0] + fromEnd[1] * difference[1] >= 0) return Math.hypot(...fromEnd);
  const projection = Math.max(
    0,
    Math.min(
      length,
      (query[0] - start[0]) * (difference[0] / length) +
        (query[1] - start[1]) * (difference[1] / length)
    )
  );
  return Math.hypot(
    query[0] - start[0] - (difference[0] / length) * projection,
    query[1] - start[1] - (difference[1] / length) * projection
  );
}
/** Exact one-neighbor planar feature search, with minimum distance over all usable parts. */
export function computeNearestFeatureJoinOnCPU(
  queries: CPUSpatialJoinGeometry,
  targets: CPUSpatialJoinGeometry
) {
  const queryFeatures = readFeatures(queries),
    targetFeatures = readFeatures(targets);
  const rankFloat = new Float32Array(1),
    rankBits = new Uint32Array(rankFloat.buffer);
  const getDistanceRank = (distance: number) => {
    if (distance === 0) return 0;
    if (!Number.isFinite(distance)) return 0xff7ffffd;
    const squared = distance * distance;
    let exponent = Math.floor(Math.log2(squared)) + 1;
    rankFloat[0] = squared / 2 ** exponent;
    if (rankFloat[0] === 1) {
      rankFloat[0] = 0.5;
      exponent++;
    }
    if (exponent > 256 || (exponent === 256 && rankFloat[0] > 0.9999998807907104))
      return 0xff7ffffd;
    if (exponent < -251) return 0x00800000;
    let word = (exponent + 252) * 0x800000 + (rankBits[0] & 0x7fffff);
    if (word >= 0x7f800000) word += 0x01000000;
    return Math.min(word, 0xff7ffffc);
  };
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
