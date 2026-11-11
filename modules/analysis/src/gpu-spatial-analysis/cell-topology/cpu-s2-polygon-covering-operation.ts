// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateMaximumCount,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {assertUnitVector} from '../global-coordinates/assert-unit-vector';
import {
  GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
  GPU_SPHERICAL_MINIMUM_TOLERANCE,
  GPU_SPHERICAL_MAXIMUM_ARRANGEMENT_TOLERANCE,
  GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT,
  validateAngularTolerance
} from '../global-coordinates/spherical-limits';
import {
  GPUS2PolygonCoveringOperation,
  type S2PolygonCoveringOperationProps
} from './gpu-s2-polygon-covering-operation';
import {
  CPU_S2_COVERING_MAXIMUM_LEVEL,
  type CPUCoveringVector,
  getCoveringDot,
  getCoveringCross,
  normalizeCoveringVector,
  negateCoveringVector,
  getCoveringCellGeometry,
  isPointOnCoveringArc
} from './cpu-s2-covering-geometry';

/** Float32 xyz vertices and uint32 offsets; rings need not consume the entire positions array. */
export type CPUS2PolygonCoveringInput = {positions: Float32Array; ringOffsets: Uint32Array};
/** GPU scheduling budgets are accepted for props compatibility and ignored. */
export type CPUS2PolygonCoveringOptions = Omit<
  S2PolygonCoveringOperationProps,
  'id' | 'positions' | 'ringOffsets' | 'positionCount' | 'ringOffsetCount' | 'output'
>;
/** Counts are [written, required]; status bits 1 invalid, 2 output capacity. */
export type CPUS2PolygonCoveringResult = {
  cells: Uint32Array;
  counts: Uint32Array;
  status: Uint32Array;
  visitedCount: Uint32Array;
};

/**
 * Adaptive S2 cover, in ascending cell-ID order, with the contributor's signed winding,
 * CCW-hole subtraction, inclusive tolerance bands and invalid-ring status.
 *
 * What differs from the GPU contributor:
 * - Predicates and geometry use float64; cell-edge decisions within GPU rounding error can differ.
 * - maximumVisitedCells and refinement are ignored. The traversal completes, status bit 4 is
 *   never set, and required count is complete. visitedCount counts the full depth-first walk.
 * - Levels up to 30 are accepted (the S2 key layout limit), dropping the GPU level-15 precision
 *   cap and fixed stack. The uint32 counts/visitedCount layout limits still apply.
 * - Malformed ring ranges, non-unit/non-finite vertices and degenerate rings return status 1,
 *   as on the GPU. Inconsistent packed array lengths throw. Unreferenced vertices are ignored.
 *
 * @throws Error naming positions or ringOffsets for inconsistent packed arrays, or naming the
 * uint32 output whose complete count cannot be represented.
 */
export function computeS2PolygonCoveringOnCPU(
  input: CPUS2PolygonCoveringInput,
  options: CPUS2PolygonCoveringOptions
): CPUS2PolygonCoveringResult {
  validateOptions('s2-polygon-covering', options);
  const {positions, ringOffsets} = input;
  if (positions.length % 3 !== 0) throw new Error('positions must hold float32x3 rows');
  if (ringOffsets.length < 2) throw new Error('ringOffsets must hold at least two entries');
  const result: CPUS2PolygonCoveringResult = {
    cells: new Uint32Array(options.capacity * 2),
    counts: new Uint32Array(2),
    status: new Uint32Array(1),
    visitedCount: new Uint32Array(1)
  };
  const tolerance = options.tolerance ?? 1e-6;
  const rings: CPUCoveringVector[][] = [];
  const getInvalid = () => {
    result.status[0] = 1;
    return result;
  };
  for (let ring = 0; ring + 1 < ringOffsets.length; ring++) {
    const begin = ringOffsets[ring];
    const end = ringOffsets[ring + 1];
    if (begin > end || end > positions.length / 3 || end - begin < 3) return getInvalid();
    const vertices: CPUCoveringVector[] = [];
    const sum = [0, 0, 0];
    for (let row = begin; row < end; row++) {
      const point = Array.from(positions.subarray(row * 3, row * 3 + 3)) as [
        number,
        number,
        number
      ];
      const squared = getCoveringDot(point, point);
      if (!Number.isFinite(squared) || Math.abs(squared - 1) > 1e-4) return getInvalid();
      vertices.push(point);
      for (let axis = 0; axis < 3; axis++) sum[axis] += point[axis];
    }
    const center = options.hemisphereCenter ?? (sum as [number, number, number]);
    if (Math.hypot(...center) <= tolerance) return getInvalid();
    const unitCenter = normalizeCoveringVector(center);
    for (let vertex = 0; vertex < vertices.length; vertex++) {
      const next = (vertex + 1) % vertices.length;
      const first = vertices[vertex];
      const second = vertices[next];
      if (
        getCoveringDot(unitCenter, first) <= tolerance ||
        Math.hypot(...getCoveringCross(first, second)) <= tolerance
      )
        return getInvalid();
      for (let other = vertex + 1; other < vertices.length; other++) {
        const otherNext = (other + 1) % vertices.length;
        if (other === next || otherNext === vertex) continue;
        if (doArcsIntersect(first, second, vertices[other], vertices[otherNext], tolerance))
          return getInvalid();
      }
    }
    rings.push(vertices);
  }
  const contains = (point: CPUCoveringVector): [boolean, boolean] => {
    let inside = false;
    let boundary = false;
    rings.forEach((ring, ringIndex) => {
      let winding = 0;
      ring.forEach((first, vertex) => {
        const second = ring[(vertex + 1) % ring.length];
        boundary ||= isPointOnCoveringArc(first, second, point, tolerance);
        const numerator = getCoveringDot(point, getCoveringCross(first, second));
        const denominator =
          getCoveringDot(first, second) -
          getCoveringDot(point, first) * getCoveringDot(point, second);
        if (Math.abs(numerator) + Math.abs(denominator) > 1e-12)
          winding +=
            numerator === 0 && denominator === 0
              ? 0
              : Math.atan2(numerator, denominator === 0 ? 0 : denominator);
      });
      inside = ringIndex === 0 ? winding > 3.14159265 : inside && winding <= 3.14159265;
    });
    return [options.complement ? !inside : inside, boundary];
  };
  const classify = (cell: bigint, level: number): [boolean, boolean] => {
    const {vertices, normals} = getCoveringCellGeometry(cell, level);
    let intersects = false;
    let contained = true;
    let touched = false;
    for (const ring of rings)
      for (const point of ring) {
        if (normals.every(normal => getCoveringDot(normal, point) >= -tolerance)) {
          intersects = true;
          touched = true;
        }
      }
    for (let edge = 0; edge < 4; edge++) {
      const first = vertices[edge];
      const second = vertices[(edge + 1) % 4];
      const [inside, boundary] = contains(first);
      intersects ||= inside || boundary;
      contained &&= inside && !boundary;
      for (const ring of rings)
        for (let vertex = 0; vertex < ring.length; vertex++) {
          if (
            doArcsIntersect(
              first,
              second,
              ring[vertex],
              ring[(vertex + 1) % ring.length],
              tolerance
            )
          ) {
            intersects = true;
            touched = true;
          }
        }
    }
    return [intersects, contained && !touched];
  };
  const stack: {cell: bigint; level: number}[] = [];
  for (let face = 5; face >= 0; face--) stack.push({cell: BigInt(2 * face + 1) << 60n, level: 0});
  let required = 0;
  let visited = 0;
  while (stack.length) {
    const {cell, level} = stack.pop()!;
    visited++;
    validateMaximumCount('s2-polygon-covering', 'visitedCount', visited, 0xffffffff);
    const [intersects, contained] = classify(cell, level);
    if (!intersects) continue;
    if (
      level >= (options.minimumLevel ?? 0) &&
      (contained || (options.mode !== 'interior' && level === options.maximumLevel))
    ) {
      if (required < options.capacity) {
        result.cells[required * 2] = Number(cell & 0xffffffffn);
        result.cells[required * 2 + 1] = Number(cell >> 32n);
      }
      required++;
    } else if (level < options.maximumLevel) {
      const marker = 1n << BigInt(60 - 2 * level);
      const childMarker = marker >> 2n;
      for (let child = 3; child >= 0; child--)
        stack.push({
          cell: cell - marker + childMarker + BigInt(child) * 2n * childMarker,
          level: level + 1
        });
    }
    validateMaximumCount('s2-polygon-covering', 'requiredCount', required, 0xffffffff);
  }
  result.counts.set([Math.min(required, options.capacity), required]);
  result.status[0] = Number(required > options.capacity) * 2;
  result.visitedCount[0] = visited;
  return result;
}

/** CPU traversal without GPU scheduling, stack-size or precision caps. */
export class CPUS2PolygonCoveringOperation extends GPUCPUOperation {
  constructor(props: S2PolygonCoveringOperationProps) {
    const operation = new GPUS2PolygonCoveringOperation(props);
    const {id} = operation;
    validateOptions(id, props);
    validateCapacity(id, 'positionCount', props.positionCount, 0);
    validateCapacity(id, 'ringOffsetCount', props.ringOffsetCount, 2);
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
        const positions = getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x3');
        const ringOffsets = getCPUOperationUint32Rows(id, inputs, 'ringOffsets');
        if (positions.length !== props.positionCount * 3)
          throw new Error(`${id} positions length must equal positionCount * 3`);
        if (ringOffsets.length !== props.ringOffsetCount)
          throw new Error(`${id} ringOffsets length must equal ringOffsetCount`);
        const result = computeS2PolygonCoveringOnCPU({positions, ringOffsets}, props);
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUS2PolygonCoveringResult]
          ])
        );
      }
    });
  }
}

function validateOptions(id: string, options: CPUS2PolygonCoveringOptions): void {
  validateCapacity(id, 'maximumLevel', options.maximumLevel, 0, CPU_S2_COVERING_MAXIMUM_LEVEL);
  validateCapacity(id, 'minimumLevel', options.minimumLevel ?? 0, 0, options.maximumLevel);
  validateCapacity(id, 'capacity', options.capacity);
  validateAngularTolerance(
    id,
    options.tolerance ?? 1e-6,
    GPU_SPHERICAL_MINIMUM_TOLERANCE,
    GPU_SPHERICAL_MAXIMUM_ARRANGEMENT_TOLERANCE
  );
  resolveGPUSpatialContext(
    id,
    options.spatialContext ?? GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
    GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT
  );
  if (options.hemisphereCenter) assertUnitVector(options.hemisphereCenter, id, 'hemisphereCenter');
  validateRequirement(
    id,
    options.refinement === undefined ||
      ['serial', 'level-synchronous'].includes(options.refinement),
    'refinement must be serial or level-synchronous'
  );
}

function doArcsIntersect(
  first: CPUCoveringVector,
  second: CPUCoveringVector,
  third: CPUCoveringVector,
  fourth: CPUCoveringVector,
  tolerance: number
): boolean {
  const crossing = getCoveringCross(
    getCoveringCross(first, second),
    getCoveringCross(third, fourth)
  );
  if (Math.hypot(...crossing) <= tolerance * tolerance)
    return (
      isPointOnCoveringArc(first, second, third, tolerance) ||
      isPointOnCoveringArc(first, second, fourth, tolerance)
    );
  const point = normalizeCoveringVector(crossing);
  const opposite = negateCoveringVector(point);
  return (
    (isPointOnCoveringArc(first, second, point, tolerance) &&
      isPointOnCoveringArc(third, fourth, point, tolerance)) ||
    (isPointOnCoveringArc(first, second, opposite, tolerance) &&
      isPointOnCoveringArc(third, fourth, opposite, tolerance))
  );
}
