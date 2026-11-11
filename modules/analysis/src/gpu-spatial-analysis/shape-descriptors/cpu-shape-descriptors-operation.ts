// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {readCPUPolygonRows} from '../label-point/cpu-label-point-operation';
import {
  getCPUPolygonFeatureRows,
  getCPUPolygonRings,
  validateCPUPolygonRows,
  type CPUPolygonRows,
  type CPUPolygonRing
} from '../shared/cpu-polygon-rows';
import {GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH} from './gpu-shape-descriptors';
import {
  GPUShapeDescriptorsOperation,
  type ShapeDescriptorsOperationProps
} from './gpu-shape-descriptors-operation';

/** All GPUShapeDescriptors output columns. */
export type CPUShapeDescriptorsResult = {
  areas: Float32Array;
  perimeters: Float32Array;
  polsbyPopper: Float32Array;
  schwartzberg: Float32Array;
  elongation: Float32Array;
  orientation: Float32Array;
  convexity: Float32Array;
  clockwise: Uint32Array;
  sliver: Uint32Array;
};

function integrateRing(
  positions: Float32Array,
  ring: CPUPolygonRing,
  originX: number,
  originY: number
) {
  let twiceArea = 0;
  let firstMomentX = 0;
  let firstMomentY = 0;
  let secondMomentX = 0;
  let secondMomentY = 0;
  let productMoment = 0;
  let perimeter = 0;
  for (let row = ring.start; row < ring.end; row++) {
    const previousRow = row === ring.start ? ring.end - 1 : row - 1;
    const previousX = positions[previousRow * 2] - originX;
    const previousY = positions[previousRow * 2 + 1] - originY;
    const currentX = positions[row * 2] - originX;
    const currentY = positions[row * 2 + 1] - originY;
    const cross = previousX * currentY - currentX * previousY;
    twiceArea += cross;
    firstMomentX += cross * (previousX + currentX);
    firstMomentY += cross * (previousY + currentY);
    secondMomentX += (cross * (previousX ** 2 + previousX * currentX + currentX ** 2)) / 12;
    secondMomentY += (cross * (previousY ** 2 + previousY * currentY + currentY ** 2)) / 12;
    productMoment +=
      (cross *
        (previousX * currentY +
          2 * previousX * previousY +
          2 * currentX * currentY +
          currentX * previousY)) /
      24;
    perimeter += Math.hypot(currentX - previousX, currentY - previousY);
  }
  return {
    twiceArea,
    firstMomentX,
    firstMomentY,
    secondMomentX,
    secondMomentY,
    productMoment,
    perimeter
  };
}

type Point = {positionX: number; positionY: number};

// Float32 inputs are dyadic rationals. This rare fallback preserves the GPU exact hull predicate
// even when a binary64 determinant cancels. Scale all coordinates to integer multiples of 2^-149.
function getExactTurn(first: Point, second: Point, third: Point): number {
  const determinant =
    (second.positionX - first.positionX) * (third.positionY - first.positionY) -
    (second.positionY - first.positionY) * (third.positionX - first.positionX);
  const magnitude =
    Math.abs((second.positionX - first.positionX) * (third.positionY - first.positionY)) +
    Math.abs((second.positionY - first.positionY) * (third.positionX - first.positionX));
  if (Math.abs(determinant) > magnitude * Number.EPSILON * 8) return Math.sign(determinant);
  const scale = 2 ** 149;
  const firstX = BigInt(first.positionX * scale);
  const firstY = BigInt(first.positionY * scale);
  const secondX = BigInt(second.positionX * scale);
  const secondY = BigInt(second.positionY * scale);
  const thirdX = BigInt(third.positionX * scale);
  const thirdY = BigInt(third.positionY * scale);
  const exact = (secondX - firstX) * (thirdY - firstY) - (secondY - firstY) * (thirdX - firstX);
  return exact > 0n ? 1 : exact < 0n ? -1 : 0;
}

function computeHullArea(points: Point[]): number {
  if (
    points.length < 3 ||
    points.some(point => !Number.isFinite(point.positionX) || !Number.isFinite(point.positionY))
  )
    return 0;
  points.sort(
    (first, second) => first.positionX - second.positionX || first.positionY - second.positionY
  );
  const lower: Point[] = [];
  const upper: Point[] = [];
  for (const point of points) {
    while (
      lower.length >= 2 &&
      getExactTurn(lower[lower.length - 2], lower[lower.length - 1], point) <= 0
    )
      lower.pop();
    lower.push(point);
  }
  for (let row = points.length - 1; row >= 0; row--) {
    const point = points[row];
    while (
      upper.length >= 2 &&
      getExactTurn(upper[upper.length - 2], upper[upper.length - 1], point) <= 0
    )
      upper.pop();
    upper.push(point);
  }
  const hull = [...lower.slice(0, -1), ...upper.slice(0, -1)];
  const origin = points[0];
  let twiceArea = 0;
  for (let row = 0; row < hull.length; row++) {
    const first = hull[row];
    const second = hull[(row + 1) % hull.length];
    twiceArea +=
      (first.positionX - origin.positionX) * (second.positionY - origin.positionY) -
      (second.positionX - origin.positionX) * (first.positionY - origin.positionY);
  }
  return Math.abs(twiceArea) * 0.5;
}

/**
 * Integrates shell/hole moments and computes every shape descriptor in binary64.
 * `parameters` is the getGPUShapeDescriptorsParameterValues packed layout.
 * Measurements treat zero signed area as nonnegative when weighting centroid numerators;
 * central moments retain a zero sign for those rings. Hull membership follows the GPU's
 * contiguous feature interval, even when descending offsets leave nonempty individual rings.
 *
 * What differs from the GPU contributor:
 * - All measures, centroids, central moments and ratios stay binary64 until each output is
 *   rounded once. Close-to-threshold sliver decisions can therefore differ from float32.
 * - Convexity always uses a monotone-chain hull with exact float32-coordinate turn predicates;
 *   convexityMethod and maximumVerticesPerFeature only steer GPU execution and are unused.
 * - No GPU sorting, storage-binding or per-invocation hull limits apply.
 * - Empty position storage is accepted for empty features; the GPU requires at least one row.
 *
 * @throws If positions, offsets or parameters arrays are inconsistent.
 */
export function computeShapeDescriptorsOnCPU(
  input: CPUPolygonRows,
  parameters: Float32Array
): CPUShapeDescriptorsResult {
  validateCPUPolygonRows(input);
  if (parameters.length < GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH) {
    throw new Error(
      `parameters must hold ${GPU_SHAPE_DESCRIPTORS_PARAMETER_LENGTH} float32 values`
    );
  }
  // The packer documents [sliverThreshold, 0, 0, 0]; no other parameter fields exist.
  const [sliverThreshold] = parameters;
  const featureCount = input.featureOffsets.length - 1;
  const createColumn = () => new Float32Array(featureCount);
  const output: CPUShapeDescriptorsResult = {
    areas: createColumn(),
    perimeters: createColumn(),
    polsbyPopper: createColumn(),
    schwartzberg: createColumn(),
    elongation: createColumn(),
    orientation: createColumn(),
    convexity: createColumn(),
    clockwise: new Uint32Array(featureCount),
    sliver: new Uint32Array(featureCount)
  };
  for (let feature = 0; feature < featureCount; feature++) {
    const rings = getCPUPolygonRings(input, feature);
    const points: Point[] = [];
    for (const ring of rings)
      for (let row = ring.start; row < ring.end; row++) {
        points.push({positionX: input.positions[row * 2], positionY: input.positions[row * 2 + 1]});
      }
    const originX = points[0]?.positionX ?? 0;
    const originY = points[0]?.positionY ?? 0;
    const sums = rings.map(ring => integrateRing(input.positions, ring, originX, originY));
    let twiceArea = 0;
    let firstMomentX = 0;
    let firstMomentY = 0;
    let perimeter = 0;
    for (let ring = 0; ring < rings.length; ring++) {
      const factor = (rings[ring].shell ? 1 : -1) * (sums[ring].twiceArea >= 0 ? 1 : -1);
      twiceArea += factor * sums[ring].twiceArea;
      firstMomentX += factor * sums[ring].firstMomentX;
      firstMomentY += factor * sums[ring].firstMomentY;
      perimeter += sums[ring].perimeter;
    }
    const area = Math.abs(twiceArea) * 0.5;
    const centroidX =
      twiceArea !== 0
        ? originX + firstMomentX / (3 * twiceArea)
        : points.reduce((total, point) => total + point.positionX, 0) / points.length;
    const centroidY =
      twiceArea !== 0
        ? originY + firstMomentY / (3 * twiceArea)
        : points.reduce((total, point) => total + point.positionY, 0) / points.length;
    let secondMomentX = 0;
    let secondMomentY = 0;
    let productMoment = 0;
    if (Number.isFinite(centroidX) && Number.isFinite(centroidY)) {
      for (let ring = 0; ring < rings.length; ring++) {
        if (rings[ring].end - rings[ring].start < 3) continue;
        const central = integrateRing(input.positions, rings[ring], centroidX, centroidY);
        const factor =
          (rings[ring].shell ? 1 : -1) * Math.sign(central.twiceArea) * (twiceArea < 0 ? -1 : 1);
        secondMomentX += factor * central.secondMomentX;
        secondMomentY += factor * central.secondMomentY;
        productMoment += factor * central.productMoment;
      }
    }
    const mean = (secondMomentX + secondMomentY) * 0.5;
    const radius = Math.hypot((secondMomentX - secondMomentY) * 0.5, productMoment);
    const major = mean + radius;
    const minor = Math.max(0, mean - radius);
    const ratio = (4 * Math.PI * area) / perimeter / perimeter;
    const featureRows = getCPUPolygonFeatureRows(input, feature);
    const hullPoints: Point[] = [];
    for (let row = featureRows.start; row < featureRows.end; row++) {
      hullPoints.push({
        positionX: input.positions[row * 2],
        positionY: input.positions[row * 2 + 1]
      });
    }
    const hullArea = computeHullArea(hullPoints);
    output.areas[feature] = area;
    output.perimeters[feature] = perimeter;
    output.polsbyPopper[feature] = area > 0 && perimeter > 0 ? ratio : NaN;
    output.schwartzberg[feature] = area > 0 && perimeter > 0 ? Math.sqrt(ratio) : NaN;
    output.elongation[feature] = major > 0 ? 1 - Math.sqrt(minor / major) : NaN;
    output.orientation[feature] =
      major > 0
        ? productMoment === 0 && secondMomentX === secondMomentY
          ? 0
          : 0.5 * Math.atan2(2 * productMoment || 0, secondMomentX - secondMomentY || 0)
        : NaN;
    output.convexity[feature] = hullArea > 0 && area > 0 ? Math.min(area / hullArea, 1) : NaN;
    output.clockwise[feature] =
      sums[0]?.twiceArea < 0 && Number.isFinite(centroidX) && Number.isFinite(centroidY) ? 1 : 0;
    output.sliver[feature] = perimeter > 0 && (area <= 0 || ratio < sliverThreshold) ? 1 : 0;
  }
  return output;
}

/** CPU twin; hull selection hints and GPU hull resource limits are unused. */
export class CPUShapeDescriptorsOperation extends GPUCPUOperation {
  constructor(props: ShapeDescriptorsOperationProps) {
    const operation = new GPUShapeDescriptorsOperation(props);
    const {id} = operation;
    validateCapacity(id, 'featureCount', props.featureCount, 1);
    validateRequirement(
      id,
      Object.keys(operation.outputs).length > 0,
      'output must name at least one column'
    );
    resolveGPUSpatialContext(id, props.spatialContext);
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
        const input = readCPUPolygonRows(id, inputs);
        validateRequirement(
          id,
          input.featureOffsets.length === props.featureCount + 1,
          'featureOffsets length must equal featureCount + 1'
        );
        const result = computeShapeDescriptorsOnCPU(
          input,
          getCPUOperationFloat32Rows(id, inputs, 'parameters')
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUShapeDescriptorsResult]
          ])
        );
      }
    });
  }
}
