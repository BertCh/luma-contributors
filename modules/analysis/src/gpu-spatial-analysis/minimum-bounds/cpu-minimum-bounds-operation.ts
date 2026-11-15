// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getPhilox4x32, PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUMinimumBoundsOperation,
  type MinimumBoundsOperationProps
} from './gpu-minimum-bounds-operation';

/** Packed points with either labels or the polygon CSR hierarchy. */
export type CPUMinimumBoundsInput = {
  positions: Float32Array;
} & (
  | {labels: Uint32Array; groupCount: number}
  | {ringOffsets: Uint32Array; polygonOffsets: Uint32Array; featureOffsets: Uint32Array}
);

/** All fixed-size shape columns, including the completeness scalars. */
export type CPUMinimumBoundsResult = {
  rectangleCorners: Float32Array;
  rectangleSizes: Float32Array;
  circles: Float32Array;
  longestLines: Float32Array;
  diameters: Float32Array;
  bounds: Float32Array;
  hullSizes: Uint32Array;
  overflow: Uint32Array;
  /** Always zero: the CPU twin has no per-group hull limit. */
  groupOverflow: Uint32Array;
};

type Point = {x: number; y: number; latticeX: number; latticeY: number; row: number};
type Circle = {x: number; y: number; squaredRadius: number};

// The contributor's default bounds GPU per-group work; the CPU accepts it but computes all hulls.
const DEFAULT_MAXIMUM_HULL_VERTICES = 2048;
// Largest float32 below pi: rounding a direction to float32 must keep its documented (-pi, pi] range.
const MAXIMUM_RECTANGLE_ANGLE = Math.fround(Math.PI - 2 ** -23);

function roundRectangleAngle(angle: number): number {
  const rounded = Math.fround(angle);
  return rounded > Math.PI || rounded <= -Math.PI ? MAXIMUM_RECTANGLE_ANGLE : rounded || 0;
}

function validateOffsets(name: string, offsets: Uint32Array, limit: number): void {
  if (offsets.length < 2) throw new Error(`${name} must contain at least two entries`);
  for (let index = 0; index < offsets.length; index++) {
    if (offsets[index] > limit || (index > 0 && offsets[index] < offsets[index - 1])) {
      throw new Error(`${name} must be nondecreasing and in [0, ${limit}]`);
    }
  }
}

function findOwner(offsets: Uint32Array, value: number): number {
  if (value < offsets[0] || value >= offsets[offsets.length - 1]) return -1;
  let lower = 0;
  let upper = offsets.length - 2;
  while (lower < upper) {
    const middle = Math.floor((lower + upper + 1) / 2);
    if (offsets[middle] <= value) lower = middle;
    else upper = middle - 1;
  }
  return lower;
}

function computeOrientation(first: Point, second: Point, third: Point): bigint {
  return (
    BigInt(second.latticeX - first.latticeX) * BigInt(third.latticeY - first.latticeY) -
    BigInt(second.latticeY - first.latticeY) * BigInt(third.latticeX - first.latticeX)
  );
}

function computeHull(points: Point[]): Point[] {
  points.sort(
    (first, second) =>
      first.latticeX - second.latticeX || first.latticeY - second.latticeY || first.row - second.row
  );
  const distinct = points.filter(
    (point, index) =>
      index === 0 ||
      point.latticeX !== points[index - 1].latticeX ||
      point.latticeY !== points[index - 1].latticeY
  );
  if (distinct.length < 2) return distinct;
  const lower: Point[] = [];
  const upper: Point[] = [];
  for (const point of distinct) {
    while (
      lower.length >= 2 &&
      computeOrientation(lower[lower.length - 2], lower[lower.length - 1], point) <= 0n
    )
      lower.pop();
    lower.push(point);
  }
  for (const point of [...distinct].reverse()) {
    while (
      upper.length >= 2 &&
      computeOrientation(upper[upper.length - 2], upper[upper.length - 1], point) <= 0n
    )
      upper.pop();
    upper.push(point);
  }
  return [...lower.slice(0, -1), ...upper.slice(0, -1)];
}

function computeSquaredDistance(
  first: {x: number; y: number},
  second: {x: number; y: number}
): number {
  return (first.x - second.x) ** 2 + (first.y - second.y) ** 2;
}

function computeCircle(hull: Point[], group: number): Circle {
  const points = hull.map(point => ({x: point.x - hull[0].x, y: point.y - hull[0].y}));
  for (let slot = points.length - 1; slot > 0; slot--) {
    const pick =
      getPhilox4x32([group, slot, 0, 0], [0, PHILOX_PURPOSES.minimumBoundsCircleShuffle.id])[0] %
      (slot + 1);
    [points[slot], points[pick]] = [points[pick], points[slot]];
  }
  let circle: Circle = {...points[0], squaredRadius: 0};
  const isOutside = (point: {x: number; y: number}) =>
    computeSquaredDistance(point, circle) > circle.squaredRadius * (1 + 16 * Number.EPSILON);
  for (let first = 1; first < points.length; first++) {
    const firstPoint = points[first];
    if (!isOutside(firstPoint)) continue;
    circle = {...firstPoint, squaredRadius: 0};
    for (let second = 0; second < first; second++) {
      const secondPoint = points[second];
      if (!isOutside(secondPoint)) continue;
      circle = {
        x: (firstPoint.x + secondPoint.x) / 2,
        y: (firstPoint.y + secondPoint.y) / 2,
        squaredRadius: 0
      };
      circle.squaredRadius = computeSquaredDistance(firstPoint, circle);
      for (let third = 0; third < second; third++) {
        const thirdPoint = points[third];
        if (!isOutside(thirdPoint)) continue;
        const secondDelta = {x: secondPoint.x - firstPoint.x, y: secondPoint.y - firstPoint.y};
        const thirdDelta = {x: thirdPoint.x - firstPoint.x, y: thirdPoint.y - firstPoint.y};
        const determinant = 2 * (secondDelta.x * thirdDelta.y - secondDelta.y * thirdDelta.x);
        if (determinant === 0) {
          const pairs = [
            [firstPoint, secondPoint],
            [firstPoint, thirdPoint],
            [secondPoint, thirdPoint]
          ];
          pairs.sort(
            (firstPair, secondPair) =>
              computeSquaredDistance(secondPair[0], secondPair[1]) -
              computeSquaredDistance(firstPair[0], firstPair[1])
          );
          circle = {
            x: (pairs[0][0].x + pairs[0][1].x) / 2,
            y: (pairs[0][0].y + pairs[0][1].y) / 2,
            squaredRadius: 0
          };
        } else {
          const secondSquared = secondDelta.x ** 2 + secondDelta.y ** 2;
          const thirdSquared = thirdDelta.x ** 2 + thirdDelta.y ** 2;
          circle = {
            x:
              firstPoint.x +
              (thirdDelta.y * secondSquared - secondDelta.y * thirdSquared) / determinant,
            y:
              firstPoint.y +
              (secondDelta.x * thirdSquared - thirdDelta.x * secondSquared) / determinant,
            squaredRadius: 0
          };
        }
        circle.squaredRadius = Math.max(
          computeSquaredDistance(firstPoint, circle),
          computeSquaredDistance(secondPoint, circle),
          computeSquaredDistance(thirdPoint, circle)
        );
      }
    }
  }
  circle.squaredRadius = 0;
  for (const point of points)
    circle.squaredRadius = Math.max(circle.squaredRadius, computeSquaredDistance(point, circle));
  circle.x += hull[0].x;
  circle.y += hull[0].y;
  return circle;
}

/**
 * Computes group hulls and their bounding shapes in float64, rounding each output once.
 * Hulls use the documented global power-of-two lattice, exact integer orientation and lowest-row
 * representatives. Rectangle ties retain the first edge within the documented 1e-6 relative band.
 *
 * What differs from the GPU contributor:
 * - No hull budget: every hull is computed, overflow and groupOverflow are always zero, and maximumHullVertices on
 *   the operation is accepted for interchangeability but does not truncate the CPU result.
 * - Lattice extent/subtraction and shape arithmetic use float64. Near lattice boundaries hulls
 *   can differ from float32 quantization; diameter ties use float64 squared distances.
 * - Rectangle and diameter search exhaust all edges/pairs instead of using calipers.
 * - One-/two-vertex hull rectangles copy their corners directly and have exactly zero height
 *   and area. GPU projection/reconstruction can leave float32 cancellation residuals in these
 *   quantities and in other mathematically zero computed coordinates.
 * - Rectangle angles are range-normalized after float32 rounding to stay in (-pi, pi].
 * - Circles use the same Philox shuffle, but float64 containment slack (16 epsilon) replaces
 *   4e-6, and only exactly collinear triples use a diametral circle (no near-collinear cutoff).
 * - Nonfinite positions and out-of-range labels are excluded, as on the GPU. Empty groups emit NaN.
 * - The GPU's row-count cap is dropped; uint32 group/output indexing limits remain.
 *
 * @throws If positions, labels or any offset array has an inconsistent length or range.
 */
export function computeMinimumBoundsOnCPU(input: CPUMinimumBoundsInput): CPUMinimumBoundsResult {
  if (input.positions.length % 2 !== 0) throw new Error('positions must contain float32x2 rows');
  const rowCount = input.positions.length / 2;
  let groupCount: number;
  let labels: Uint32Array;
  if ('labels' in input) {
    groupCount = input.groupCount;
    validateCapacity('computeMinimumBoundsOnCPU', 'groupCount', groupCount);
    if (input.labels.length !== rowCount)
      throw new Error('labels length must equal positions row count');
    labels = input.labels;
  } else {
    validateOffsets('ringOffsets', input.ringOffsets, rowCount);
    validateOffsets('polygonOffsets', input.polygonOffsets, input.ringOffsets.length - 1);
    validateOffsets('featureOffsets', input.featureOffsets, input.polygonOffsets.length - 1);
    groupCount = input.featureOffsets.length - 1;
    labels = new Uint32Array(rowCount).fill(0xffffffff);
    for (let row = 0; row < rowCount; row++) {
      const ring = findOwner(input.ringOffsets, row);
      const polygon = ring < 0 ? -1 : findOwner(input.polygonOffsets, ring);
      const feature = polygon < 0 ? -1 : findOwner(input.featureOffsets, polygon);
      if (feature >= 0) labels[row] = feature;
    }
  }
  const groups: Point[][] = Array.from({length: groupCount}, () => []);
  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let row = 0; row < rowCount; row++) {
    const x = input.positions[row * 2];
    const y = input.positions[row * 2 + 1];
    if (labels[row] >= groupCount || !Number.isFinite(x) || !Number.isFinite(y)) continue;
    minimumX = Math.min(minimumX, x);
    minimumY = Math.min(minimumY, y);
    maximumX = Math.max(maximumX, x);
    maximumY = Math.max(maximumY, y);
    groups[labels[row]].push({x, y, latticeX: 0, latticeY: 0, row});
  }
  const extent = Math.max(maximumX - minimumX, maximumY - minimumY);
  const scale = extent > 0 ? 2 ** (28 - Math.max(-99, Math.floor(Math.log2(extent)))) : 1;
  const result: CPUMinimumBoundsResult = {
    rectangleCorners: new Float32Array(groupCount * 8).fill(NaN),
    rectangleSizes: new Float32Array(groupCount * 4).fill(NaN),
    circles: new Float32Array(groupCount * 3).fill(NaN),
    longestLines: new Float32Array(groupCount * 4).fill(NaN),
    diameters: new Float32Array(groupCount).fill(NaN),
    bounds: new Float32Array(groupCount * 4).fill(NaN),
    hullSizes: new Uint32Array(groupCount),
    overflow: new Uint32Array(1),
    groupOverflow: new Uint32Array(1)
  };
  for (let group = 0; group < groupCount; group++) {
    for (const point of groups[group]) {
      point.latticeX = Math.floor((point.x - minimumX) * scale + 0.5);
      point.latticeY = Math.floor((point.y - minimumY) * scale + 0.5);
    }
    const hull = computeHull(groups[group]);
    result.hullSizes[group] = hull.length;
    if (hull.length === 0) continue;
    const anchor = hull[0];
    let bestArea = Infinity;
    let bestDirection = [1, 0];
    let bestRange = [0, 0, 0, 0];
    let longestSquared = -1;
    let longestPair = [anchor, anchor];
    let lowX = Infinity;
    let lowY = Infinity;
    let highX = -Infinity;
    let highY = -Infinity;
    for (let edge = 0; edge < hull.length; edge++) {
      const point = hull[edge];
      lowX = Math.min(lowX, point.x);
      lowY = Math.min(lowY, point.y);
      highX = Math.max(highX, point.x);
      highY = Math.max(highY, point.y);
      for (let other = edge; other < hull.length; other++) {
        const squared = computeSquaredDistance(point, hull[other]);
        if (squared > longestSquared) {
          longestSquared = squared;
          longestPair = [point, hull[other]];
        }
      }
      const next = hull[(edge + 1) % hull.length];
      const length = Math.hypot(next.x - point.x, next.y - point.y);
      const direction =
        length > 0 ? [(next.x - point.x) / length, (next.y - point.y) / length] : [1, 0];
      const range = [Infinity, Infinity, -Infinity, -Infinity];
      for (const vertex of hull) {
        const along = (vertex.x - anchor.x) * direction[0] + (vertex.y - anchor.y) * direction[1];
        const across = -(vertex.x - anchor.x) * direction[1] + (vertex.y - anchor.y) * direction[0];
        range[0] = Math.min(range[0], along);
        range[1] = Math.min(range[1], across);
        range[2] = Math.max(range[2], along);
        range[3] = Math.max(range[3], across);
      }
      const area = (range[2] - range[0]) * (range[3] - range[1]);
      if (edge === 0 || area < bestArea * (1 - 1e-6)) {
        bestArea = area;
        bestDirection = direction;
        bestRange = range;
      }
    }
    for (let corner = 0; corner < 4; corner++) {
      const along = bestRange[corner === 1 || corner === 2 ? 2 : 0];
      const across = bestRange[corner >= 2 ? 3 : 1];
      result.rectangleCorners.set(
        [
          anchor.x + bestDirection[0] * along - bestDirection[1] * across,
          anchor.y + bestDirection[1] * along + bestDirection[0] * across
        ],
        group * 8 + corner * 2
      );
    }
    result.rectangleSizes.set(
      [
        bestRange[2] - bestRange[0],
        bestRange[3] - bestRange[1],
        roundRectangleAngle(Math.atan2(bestDirection[1], bestDirection[0])),
        bestArea
      ],
      group * 4
    );
    if (hull.length <= 2) {
      // The documented point/segment rectangle has zero height and area. Construct it directly:
      // normalized dot products can otherwise leave even a float64 cancellation residual.
      const end = hull[hull.length - 1];
      result.rectangleCorners.set(
        [anchor.x, anchor.y, end.x, end.y, end.x, end.y, anchor.x, anchor.y],
        group * 8
      );
      result.rectangleSizes.set(
        [
          Math.hypot(end.x - anchor.x, end.y - anchor.y),
          0,
          roundRectangleAngle(Math.atan2(end.y - anchor.y, end.x - anchor.x)),
          0
        ],
        group * 4
      );
    }
    const circle = computeCircle(hull, group);
    result.circles.set([circle.x, circle.y, Math.sqrt(circle.squaredRadius)], group * 3);
    result.longestLines.set(
      [longestPair[0].x, longestPair[0].y, longestPair[1].x, longestPair[1].y],
      group * 4
    );
    result.diameters[group] = Math.sqrt(longestSquared);
    result.bounds.set([lowX, lowY, highX, highY], group * 4);
  }
  return result;
}

/** CPU minimum bounds twin with no GPU row or hull budget. */
export class CPUMinimumBoundsOperation extends GPUCPUOperation {
  constructor(props: MinimumBoundsOperationProps) {
    const id = props.id ?? 'minimum-bounds';
    resolveGPUSpatialContext(id, props.spatialContext);
    validateCapacity(id, 'rowCount', props.rowCount);
    validateCapacity(
      id,
      props.labels !== undefined ? 'groupCount' : 'featureCount',
      props.labels !== undefined ? props.groupCount : props.featureCount
    );
    validateCapacity(
      id,
      'maximumHullVertices',
      props.maximumHullVertices ?? DEFAULT_MAXIMUM_HULL_VERTICES,
      3
    );
    validateRequirement(
      id,
      ['rectangleCorners', 'rectangleSizes', 'circles', 'longestLines', 'diameters', 'bounds'].some(
        port => props.output[port as keyof typeof props.output] !== undefined
      ),
      'output requires at least one shape column'
    );
    const operation = new GPUMinimumBoundsOperation(props);
    super({
      id,
      inputs: operation.inputs,
      outputs: Object.fromEntries(
        Object.values(operation.outputs).map(output => [
          output.value,
          gpuGraph.vector(output.format, {length: output.length})
        ])
      ),
      execute({inputs}) {
        const positions = getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2');
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions length must equal rowCount`);
        const input: CPUMinimumBoundsInput =
          props.labels !== undefined
            ? {
                positions,
                labels: getCPUOperationUint32Rows(id, inputs, 'labels'),
                groupCount: props.groupCount
              }
            : {
                positions,
                ringOffsets: getCPUOperationUint32Rows(id, inputs, 'ringOffsets'),
                polygonOffsets: getCPUOperationUint32Rows(id, inputs, 'polygonOffsets'),
                featureOffsets: getCPUOperationUint32Rows(id, inputs, 'featureOffsets')
              };
        if ('featureOffsets' in input && input.featureOffsets.length !== props.featureCount! + 1)
          throw new Error(`${id} featureOffsets length must equal featureCount + 1`);
        const result = computeMinimumBoundsOnCPU(input);
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUMinimumBoundsResult]
          ])
        );
      }
    });
  }
}
