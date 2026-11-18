// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Exact-keyed planar point retained in float64 until publication. @internal */
export type CPURefinementPoint = {x: number; y: number; key: string};

/** One noded atomic edge, in input traversal order. @internal */
export type CPURefinementSegment = {
  start: CPURefinementPoint;
  end: CPURefinementPoint;
  line: number;
  source: number;
};

/** Noded traversal and its atomic segments. @internal */
export type CPURefinementNoding = {
  lines: {point: CPURefinementPoint; source: number}[][];
  segments: CPURefinementSegment[];
  incomplete: number;
};

type IntegerPoint = {x: bigint; y: bigint};
type Split = {numerator: bigint; denominator: bigint; point: CPURefinementPoint};
type InputSegment = {
  start: IntegerPoint;
  end: IntegerPoint;
  line: number;
  source: number;
  splits: Split[];
};

// Every finite float32 coordinate is an integer multiple of 2^-149. This scaling is exact in
// float64, including subnormals, and all resulting BigInts are at most 277 bits wide.
const FLOAT32_LATTICE_SCALE = 2 ** 149;

function getGreatestCommonDivisor(first: bigint, second: bigint): bigint {
  let left = first < 0n ? -first : first;
  let right = second < 0n ? -second : second;
  while (right !== 0n) [left, right] = [right, left % right];
  return left;
}

function getRationalKey(numerator: bigint, denominator: bigint): string {
  const divisor = getGreatestCommonDivisor(numerator, denominator);
  return `${numerator / divisor}/${denominator / divisor}`;
}

function getIntegerPoint(x: number, y: number): IntegerPoint {
  return {x: BigInt(x * FLOAT32_LATTICE_SCALE), y: BigInt(y * FLOAT32_LATTICE_SCALE)};
}

function getInputPoint(x: number, y: number): CPURefinementPoint {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return {x, y, key: `nonfinite:${x},${y}`};
  const point = getIntegerPoint(x, y);
  return {x, y, key: `${point.x}/1,${point.y}/1`};
}

function getCrossProduct(first: IntegerPoint, second: IntegerPoint): bigint {
  return first.x * second.y - first.y * second.x;
}

function subtractPoints(first: IntegerPoint, second: IntegerPoint): IntegerPoint {
  return {x: first.x - second.x, y: first.y - second.y};
}

function insertEndpoint(segment: InputSegment, endpoint: IntegerPoint): void {
  const direction = subtractPoints(segment.end, segment.start);
  let numerator = direction.x !== 0n ? endpoint.x - segment.start.x : endpoint.y - segment.start.y;
  let denominator = direction.x !== 0n ? direction.x : direction.y;
  if (denominator < 0n) {
    numerator = -numerator;
    denominator = -denominator;
  }
  if (numerator < 0n || numerator > denominator) return;
  segment.splits.push({
    numerator,
    denominator,
    point: {
      x: Number(endpoint.x) / FLOAT32_LATTICE_SCALE,
      y: Number(endpoint.y) / FLOAT32_LATTICE_SCALE,
      key: `${endpoint.x}/1,${endpoint.y}/1`
    }
  });
}

function insertIntersections(
  first: InputSegment,
  second: InputSegment
): 'none' | 'touch' | 'overlap' | 'crossing' {
  const firstDirection = subtractPoints(first.end, first.start);
  const secondDirection = subtractPoints(second.end, second.start);
  const displacement = subtractPoints(second.start, first.start);
  let denominator = getCrossProduct(firstDirection, secondDirection);
  if (denominator === 0n) {
    if (getCrossProduct(displacement, firstDirection) !== 0n) return 'none';
    const firstStart = firstDirection.x !== 0n ? first.start.x : first.start.y;
    const firstEnd = firstDirection.x !== 0n ? first.end.x : first.end.y;
    const secondStart = firstDirection.x !== 0n ? second.start.x : second.start.y;
    const secondEnd = firstDirection.x !== 0n ? second.end.x : second.end.y;
    const firstMinimum = firstStart < firstEnd ? firstStart : firstEnd;
    const firstMaximum = firstStart > firstEnd ? firstStart : firstEnd;
    const secondMinimum = secondStart < secondEnd ? secondStart : secondEnd;
    const secondMaximum = secondStart > secondEnd ? secondStart : secondEnd;
    const minimum = firstMinimum > secondMinimum ? firstMinimum : secondMinimum;
    const maximum = firstMaximum < secondMaximum ? firstMaximum : secondMaximum;
    if (minimum > maximum) return 'none';
    // Collinear overlaps split at both endpoints, including doubled-back adjacent segments.
    insertEndpoint(first, second.start);
    insertEndpoint(first, second.end);
    insertEndpoint(second, first.start);
    insertEndpoint(second, first.end);
    return minimum === maximum ? 'touch' : 'overlap';
  }
  let firstNumerator = getCrossProduct(displacement, secondDirection);
  let secondNumerator = getCrossProduct(displacement, firstDirection);
  if (denominator < 0n) {
    denominator = -denominator;
    firstNumerator = -firstNumerator;
    secondNumerator = -secondNumerator;
  }
  if (
    firstNumerator < 0n ||
    firstNumerator > denominator ||
    secondNumerator < 0n ||
    secondNumerator > denominator
  )
    return 'none';
  const numeratorX = first.start.x * denominator + firstDirection.x * firstNumerator;
  const numeratorY = first.start.y * denominator + firstDirection.y * firstNumerator;
  // The largest numerator is below 2^837, so conversion cannot overflow float64.
  const point: CPURefinementPoint = {
    x: Number(numeratorX) / Number(denominator) / FLOAT32_LATTICE_SCALE,
    y: Number(numeratorY) / Number(denominator) / FLOAT32_LATTICE_SCALE,
    key: `${getRationalKey(numeratorX, denominator)},${getRationalKey(numeratorY, denominator)}`
  };
  first.splits.push({numerator: firstNumerator, denominator, point});
  second.splits.push({numerator: secondNumerator, denominator, point});
  return firstNumerator === 0n ||
    firstNumerator === denominator ||
    secondNumerator === 0n ||
    secondNumerator === denominator
    ? 'touch'
    : 'crossing';
}

/** Validates packed line arrays, allowing unowned padding before/after the lines. @internal */
export function validateCPURefinementLines(
  positions: Float32Array,
  lineOffsets: Uint32Array,
  argument = 'lines'
): void {
  if (positions.length % 2 !== 0)
    throw new Error(`${argument}.positions must contain float32x2 rows`);
  if (lineOffsets.length < 1)
    throw new Error(`${argument}.lineOffsets must contain a terminal offset`);
  for (let index = 0; index < lineOffsets.length; index++) {
    if (
      lineOffsets[index] > positions.length / 2 ||
      (index > 0 && lineOffsets[index] < lineOffsets[index - 1])
    ) {
      throw new Error(`${argument}.lineOffsets must be nondecreasing and within positions length`);
    }
  }
}

/**
 * Exact rational intersections of float32 input edges, with float64 coordinates and canonical
 * rational visit keys. An optional pair capacity suppresses all geometry and marks incomplete
 * when exhausted; adjacent shared-endpoint touches do not consume it, but overlaps do. There
 * is no successor scan cap, uncertain predicate or intermediate f32 rounding. Intended for the
 * serial refinement twins, not large data-parallel linework.
 * @internal
 */
export function nodeCPURefinementLines(
  positions: Float32Array,
  lineOffsets: Uint32Array,
  options: {intersectionCapacity?: number} = {}
): CPURefinementNoding {
  validateCPURefinementLines(positions, lineOffsets);
  const segments: InputSegment[] = [];
  const segmentsBySource = new Map<number, InputSegment>();
  let incomplete = 0;
  for (let line = 0; line + 1 < lineOffsets.length; line++) {
    for (let vertex = lineOffsets[line]; vertex < lineOffsets[line + 1]; vertex++) {
      const startX = positions[vertex * 2];
      const startY = positions[vertex * 2 + 1];
      if (!Number.isFinite(startX) || !Number.isFinite(startY)) incomplete = 1;
      if (vertex + 1 >= lineOffsets[line + 1]) continue;
      const endX = positions[(vertex + 1) * 2];
      const endY = positions[(vertex + 1) * 2 + 1];
      if (
        ![startX, startY, endX, endY].every(Number.isFinite) ||
        (startX === endX && startY === endY)
      )
        continue;
      const start = getIntegerPoint(startX, startY);
      const end = getIntegerPoint(endX, endY);
      const segment: InputSegment = {start, end, line, source: vertex, splits: []};
      insertEndpoint(segment, start);
      insertEndpoint(segment, end);
      segments.push(segment);
      segmentsBySource.set(vertex, segment);
    }
  }
  let intersectionCount = 0;
  for (let first = 0; first < segments.length; first++) {
    for (let second = first + 1; second < segments.length; second++) {
      const firstSegment = segments[first];
      const secondSegment = segments[second];
      const kind = insertIntersections(firstSegment, secondSegment);
      if (kind === 'none') continue;
      if (kind === 'touch' && firstSegment.line === secondSegment.line) {
        const line = firstSegment.line;
        const firstVertex = lineOffsets[line];
        const lastVertex = lineOffsets[line + 1] - 1;
        const closed =
          positions[firstVertex * 2] === positions[lastVertex * 2] &&
          positions[firstVertex * 2 + 1] === positions[lastVertex * 2 + 1];
        const adjacent =
          second === first + 1 ||
          (closed &&
            (first === 0 || segments[first - 1].line !== line) &&
            (second + 1 === segments.length || segments[second + 1].line !== line));
        if (adjacent) continue;
      }
      intersectionCount++;
      if (
        options.intersectionCapacity !== undefined &&
        intersectionCount > options.intersectionCapacity
      ) {
        return {lines: [], segments: [], incomplete: 1};
      }
    }
  }
  const result: CPURefinementNoding = {lines: [], segments: [], incomplete};
  for (let line = 0; line + 1 < lineOffsets.length; line++) {
    const traversal: CPURefinementNoding['lines'][number] = [];
    for (let vertex = lineOffsets[line]; vertex < lineOffsets[line + 1]; vertex++) {
      const inputPoint = getInputPoint(positions[vertex * 2], positions[vertex * 2 + 1]);
      traversal.push({point: inputPoint, source: vertex});
      const segment = segmentsBySource.get(vertex);
      if (!segment) continue;
      segment.splits.sort((left, right) => {
        const difference = left.numerator * right.denominator - right.numerator * left.denominator;
        return difference < 0n ? -1 : difference > 0n ? 1 : 0;
      });
      let previous: CPURefinementPoint | undefined;
      for (const split of segment.splits) {
        if (previous?.key === split.point.key) continue;
        if (previous)
          result.segments.push({start: previous, end: split.point, line, source: vertex});
        if (split.numerator > 0n && split.numerator < split.denominator)
          traversal.push({point: split.point, source: vertex});
        previous = split.point;
      }
    }
    result.lines.push(traversal);
  }
  return result;
}
