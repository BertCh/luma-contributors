// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CPUSpatialFeature} from './cpu-feature-join';

type Position = readonly [bigint, bigint];
type Query = readonly [bigint, bigint, bigint];
type Fraction = readonly [bigint, bigint];
type ExactFeature = {kind: CPUSpatialFeature['kind']; parts: Position[][]};

function subtract(first: Position, second: Position): Position {
  return [first[0] - second[0], first[1] - second[1]];
}
function cross(first: Position, second: Position): bigint {
  return first[0] * second[1] - first[1] * second[0];
}
function areEqual(first: Position, second: Position): boolean {
  return first[0] === second[0] && first[1] === second[1];
}
function* getEdges(feature: ExactFeature): Generator<readonly [Position, Position]> {
  for (const part of feature.parts)
    for (
      let index = 0;
      index <
      (feature.kind === 'points' ? 1 : feature.kind === 'lines' ? part.length - 1 : part.length);
      index++
    )
      yield [part[index], part[feature.kind === 'points' ? index : (index + 1) % part.length]];
}
function compareFractions(first: Fraction, second: Fraction): number {
  const difference = first[0] * second[1] - second[0] * first[1];
  return difference < 0n ? -1 : difference > 0n ? 1 : 0;
}
function getQuery(start: Position, direction: Position, fraction: Fraction): Query {
  return [
    start[0] * fraction[1] + direction[0] * fraction[0],
    start[1] * fraction[1] + direction[1] * fraction[0],
    fraction[1]
  ];
}
function isOnSegment(start: Position, end: Position, query: Query): boolean {
  return (
    cross(subtract(end, start), [
      query[0] - start[0] * query[2],
      query[1] - start[1] * query[2]
    ]) === 0n &&
    query[0] >= (start[0] < end[0] ? start[0] : end[0]) * query[2] &&
    query[0] <= (start[0] > end[0] ? start[0] : end[0]) * query[2] &&
    query[1] >= (start[1] < end[1] ? start[1] : end[1]) * query[2] &&
    query[1] <= (start[1] > end[1] ? start[1] : end[1]) * query[2]
  );
}
// Matrix classes: 0 interior, 1 boundary, 2 exterior. With a normal, locate the
// symbolic point query + epsilon * normal; no floating-point probe is formed.
function locate(feature: ExactFeature, query: Query, normal?: Position): number {
  const equals = (position: Position) =>
    query[0] === position[0] * query[2] && query[1] === position[1] * query[2];
  if (feature.kind === 'points') return feature.parts.some(part => equals(part[0])) ? 0 : 2;
  let found = false;
  let inside = false;
  const sign = (constant: bigint, coefficient: bigint) => {
    const value = constant || coefficient;
    return value < 0n ? -1 : value > 0n ? 1 : 0;
  };
  for (const [start, end] of getEdges(feature)) {
    if (!normal && isOnSegment(start, end, query)) found = true;
    if (feature.kind !== 'polygons') continue;
    const startAbove = sign(start[1] * query[2] - query[1], -(normal?.[1] ?? 0n)) > 0;
    const endAbove = sign(end[1] * query[2] - query[1], -(normal?.[1] ?? 0n)) > 0;
    const direction = subtract(end, start);
    const side = sign(
      cross(direction, [query[0] - start[0] * query[2], query[1] - start[1] * query[2]]),
      normal ? cross(direction, normal) : 0n
    );
    if (startAbove !== endAbove && end[1] > start[1] === side > 0) inside = !inside;
  }
  if (feature.kind === 'polygons') return found ? 1 : inside ? 0 : 2;
  if (!found) return 2;
  let endpoints = 0;
  for (const part of feature.parts) {
    const first = part[0],
      last = part[part.length - 1];
    if (!areEqual(first, last)) endpoints += Number(equals(first)) + Number(equals(last));
  }
  return endpoints % 2 ? 1 : 0;
}

/** Exact rational arrangement of two features, including whole-feature mod-2 line boundaries. @internal */
export function computeFeatureRelate(first: CPUSpatialFeature, second: CPUSpatialFeature): number {
  const bits = new DataView(new ArrayBuffer(4));
  const integer = (value: number): bigint => {
    bits.setFloat32(0, value);
    const word = bits.getUint32(0),
      exponent = (word >>> 23) & 255;
    const significand = BigInt((word & 0x7fffff) | (exponent ? 0x800000 : 0));
    return (word >>> 31 ? -significand : significand) << BigInt(Math.max(0, exponent - 1));
  };
  const convert = (feature: CPUSpatialFeature): ExactFeature => ({
    kind: feature.kind,
    parts: feature.parts.map(part =>
      part.map(position => [integer(position[0]), integer(position[1])])
    )
  });
  const left = convert(first),
    right = convert(second);
  const cells = new Array<number>(9).fill(0);
  const raise = (leftClass: number, rightClass: number, dimension: number) => {
    const cell = leftClass * 3 + rightClass;
    cells[cell] = Math.max(cells[cell], dimension + 1);
  };
  raise(2, 2, 2);
  const visit = (source: ExactFeature, target: ExactFeature, reverse: boolean) => {
    const record = (sourceClass: number, targetClass: number, dimension: number) =>
      reverse
        ? raise(targetClass, sourceClass, dimension)
        : raise(sourceClass, targetClass, dimension);
    for (const part of source.parts)
      for (const position of part) {
        const query: Query = [position[0], position[1], 1n];
        record(locate(source, query), locate(target, query), 0);
      }
    if (source.kind === 'points') return;
    for (const [start, end] of getEdges(source)) {
      const direction = subtract(end, start);
      if (areEqual(start, end)) continue;
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
      const axis = direction[0] !== 0n ? 0 : 1;
      for (const [otherStart, otherEnd] of getEdges(target)) {
        const edge = subtract(otherEnd, otherStart),
          offset = subtract(otherStart, start);
        const denominator = cross(direction, edge);
        if (denominator !== 0n) {
          const alongEdge = cross(offset, direction);
          if (
            denominator > 0n
              ? alongEdge >= 0n && alongEdge <= denominator
              : alongEdge <= 0n && alongEdge >= denominator
          )
            append(cross(offset, edge), denominator);
        } else if (cross(offset, direction) === 0n) {
          append(otherStart[axis] - start[axis], direction[axis]);
          append(otherEnd[axis] - start[axis], direction[axis]);
        }
      }
      breaks.sort(compareFractions);
      for (let index = 0; index < breaks.length; index++) {
        const current = breaks[index],
          next = breaks[index + 1];
        const query = getQuery(start, direction, current);
        record(locate(source, query), locate(target, query), 0);
        if (!next || compareFractions(current, next) === 0) continue;
        const middle = getQuery(start, direction, [
          current[0] * next[1] + next[0] * current[1],
          2n * current[1] * next[1]
        ]);
        record(source.kind === 'polygons' ? 1 : 0, locate(target, middle), 1);
        if (source.kind === 'polygons') {
          for (const side of [-1n, 1n]) {
            const normal: Position = [-direction[1] * side, direction[0] * side];
            record(
              locate(source, middle, normal),
              target.kind === 'polygons' ? locate(target, middle, normal) : 2,
              2
            );
          }
        }
      }
    }
  };
  visit(left, right, false);
  visit(right, left, true);
  return cells.reduce((packed, value, cell) => packed | (value << (cell * 2)), 0);
}
