// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';
import {validateSphericalOverlayOperation} from './spherical-overlay-options';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateMaximumCount
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getCoveringCross,
  getCoveringDot,
  normalizeCoveringVector,
  negateCoveringVector,
  type CPUCoveringVector
} from '../cell-topology/cpu-s2-covering-geometry';
import {validateUnitVector} from './assert-unit-vector';
import {
  GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
  GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT,
  GPU_SPHERICAL_MINIMUM_TOLERANCE,
  GPU_SPHERICAL_MAXIMUM_ARRANGEMENT_TOLERANCE,
  validateAngularTolerance
} from './spherical-limits';
import {
  GPU_SPHERICAL_ARC_OVERLAY_STATUS_INVALID,
  GPU_SPHERICAL_ARC_OVERLAY_STATUS_CAPACITY,
  GPU_SPHERICAL_ARC_OVERLAY_STATUS_UNRESOLVED
} from './gpu-spherical-arc-overlay';
import {
  GPUSphericalArcOverlayOperation,
  captureSphericalArcOverlayOperationProps,
  type SphericalArcOverlayOperationProps
} from './gpu-spherical-arc-overlay-operation';

/** Open unit-xyz rings, with reference-anchored parity membership. */
export type CPUSphericalArcOverlayInput = {
  left: {positions: Float32Array; ringOffsets: Uint32Array; referenceInside: boolean};
  right: {positions: Float32Array; ringOffsets: Uint32Array; referenceInside: boolean};
};
/** Shape and geometric options shared with the GPU operation. */
export type CPUSphericalArcOverlayOptions = Pick<
  SphericalArcOverlayOperationProps,
  'spatialContext' | 'referencePoint' | 'operation' | 'tolerance' | 'capacity' | 'ringCapacity'
>;
/** counts: [written vertices, written rings, required segments, reference membership]. */
export type CPUSphericalArcOverlayResult = {
  positions: Float32Array;
  ringOffsets: Uint32Array;
  counts: Uint32Array;
  status: Uint32Array;
};
type Vector = CPUCoveringVector;
type Edge = {start: Vector; end: Vector; next: number};
type Operand = {edges: Edge[]; referenceInside: boolean};
type Segment = {start: Vector; end: Vector; used: boolean};

/**
 * Nodes minor great-circle arcs, classifies every piece on both sides, then chases oriented rings
 * with the result interior on the left. Input and ring iteration order and maximum-left-turn
 * junction selection match the contributor. Failure empties all geometry, retaining the required
 * segment count accumulated before failure and the Boolean reference membership.
 *
 * What differs from the GPU contributor:
 * - Geometry and predicates use float64, rounded only when output positions are stored. Results
 *   near tolerance thresholds or tied junctions can differ from float32; compare positions with
 *   absolute tolerance 3e-6 on well-separated inputs. Integer fields agree on those inputs.
 * - The GPU-only 128-input, 130-split-point and 1024-output limits are dropped. Both forms retain
 *   every unique segment before output capacity is checked, so counts[2] is independent of the
 *   caller's output capacity, including on coincident boundaries.
 * - Tolerance-based invalid/unresolved rules and the seven reference-anchored containment paths
 *   are retained as the contributor's definition of numerical separation. Status bits 1, 2 and 4
 *   retain their meanings; a capacity failure publishes zero vertices and rings, not a prefix.
 * - Malformed ring ranges and non-unit/non-finite coordinates return status 1 as on GPU;
 *   inconsistent packed positions or missing offset arrays throw an Error naming the argument.
 * - Unknown runtime operation values throw in every form.
 *
 * @throws For inconsistent packed arrays, invalid options, or an unrepresentable uint32 count.
 */
export function computeSphericalArcOverlayOnCPU(
  input: CPUSphericalArcOverlayInput,
  options: CPUSphericalArcOverlayOptions
): CPUSphericalArcOverlayResult {
  const captured = snapshotConstructionFields(options, [
    'spatialContext',
    'referencePoint',
    'operation',
    'tolerance',
    'capacity',
    'ringCapacity'
  ]);
  options = Object.freeze({
    ...captured,
    referencePoint: Object.freeze([...captured.referencePoint]) as readonly [
      number,
      number,
      number
    ],
    ...(captured.spatialContext
      ? {spatialContext: snapshotConstructionSpatialContext(captured.spatialContext)}
      : {})
  });
  validateOptions('spherical-arc-overlay', options);
  for (const name of ['left', 'right'] as const) {
    if (input[name].positions.length % 3)
      throw new Error(`${name}.positions must contain float32x3 rows`);
    if (input[name].ringOffsets.length < 1)
      throw new Error(`${name}.ringOffsets must contain at least one entry`);
  }
  const tolerance = options.tolerance ?? 1e-6;
  const result: CPUSphericalArcOverlayResult = {
    positions: new Float32Array(options.capacity * 3),
    ringOffsets: new Uint32Array(options.ringCapacity + 1),
    counts: new Uint32Array(4),
    status: new Uint32Array(1)
  };
  const combineMembership = (left: boolean, right: boolean) =>
    options.operation === 'intersection'
      ? left && right
      : options.operation === 'union'
        ? left || right
        : options.operation === 'difference'
          ? left && !right
          : left !== right;
  result.counts[3] = Number(
    combineMembership(input.left.referenceInside, input.right.referenceInside)
  );
  const failResult = (status: number) => {
    result.positions.fill(0);
    result.ringOffsets.fill(0);
    result.counts[0] = result.counts[1] = 0;
    result.status[0] =
      status |
      (result.counts[2] > options.capacity ? GPU_SPHERICAL_ARC_OVERLAY_STATUS_CAPACITY : 0);
    return result;
  };
  const onArc = (first: Vector, second: Vector, point: Vector, margin: number) => {
    const normal = normalizeCoveringVector(getCoveringCross(first, second));
    return (
      Math.abs(getCoveringDot(normal, point)) <= Math.max(margin, tolerance) &&
      getCoveringDot(getCoveringCross(first, point), normal) >= -margin &&
      getCoveringDot(getCoveringCross(point, second), normal) >= -margin
    );
  };
  const operands: Operand[] = [];
  for (const source of [input.left, input.right]) {
    const {positions, ringOffsets} = source;
    const vertexCount = positions.length / 3;
    if (ringOffsets.length - 1 > Math.floor(vertexCount / 3))
      return failResult(GPU_SPHERICAL_ARC_OVERLAY_STATUS_INVALID);
    if (ringOffsets[0] !== 0 || ringOffsets[ringOffsets.length - 1] !== vertexCount)
      return failResult(GPU_SPHERICAL_ARC_OVERLAY_STATUS_INVALID);
    const vertices: Vector[] = Array.from({length: vertexCount}, (_, row) => [
      positions[row * 3],
      positions[row * 3 + 1],
      positions[row * 3 + 2]
    ]);
    const edges: Edge[] = [];
    for (let ring = 0; ring + 1 < ringOffsets.length; ring++) {
      const begin = ringOffsets[ring];
      const end = ringOffsets[ring + 1];
      if (begin > end || end > vertexCount || end - begin < 3)
        return failResult(GPU_SPHERICAL_ARC_OVERLAY_STATUS_INVALID);
      for (let row = begin; row < end; row++) {
        const next = row + 1 === end ? begin : row + 1;
        edges.push({start: vertices[row], end: vertices[next], next});
      }
    }
    for (let edge = 0; edge < edges.length; edge++) {
      const {start, end, next} = edges[edge];
      const squared = getCoveringDot(start, start);
      if (
        !Number.isFinite(squared) ||
        Math.abs(squared - 1) > 1e-4 ||
        Math.hypot(...getCoveringCross(start, end)) <= tolerance ||
        onArc(start, end, options.referencePoint, tolerance)
      )
        return failResult(GPU_SPHERICAL_ARC_OVERLAY_STATUS_INVALID);
      for (let other = edge + 1; other < edges.length; other++) {
        if (other === next || edges[other].next === edge) continue;
        const third = edges[other].start;
        const fourth = edges[other].end;
        const crossing = getCoveringCross(
          getCoveringCross(start, end),
          getCoveringCross(third, fourth)
        );
        if (Math.hypot(...crossing) > tolerance * tolerance) {
          const point = normalizeCoveringVector(crossing);
          const opposite = negateCoveringVector(point);
          if (
            (onArc(start, end, point, tolerance) && onArc(third, fourth, point, tolerance)) ||
            (onArc(start, end, opposite, tolerance) && onArc(third, fourth, opposite, tolerance))
          )
            return failResult(GPU_SPHERICAL_ARC_OVERLAY_STATUS_INVALID);
        } else if (onArc(start, end, third, tolerance) || onArc(start, end, fourth, tolerance))
          return failResult(GPU_SPHERICAL_ARC_OVERLAY_STATUS_INVALID);
      }
    }
    operands.push({edges, referenceInside: source.referenceInside});
  }
  const choices: Vector[] = [
    options.referencePoint,
    ...(
      [
        [1, 0.371, 0.127],
        [-0.253, 1, 0.419],
        [0.173, -0.391, 1],
        [-1, 0.137, -0.293],
        [0.317, -1, 0.193],
        [-0.227, 0.433, -1]
      ] as Vector[]
    ).map(normalizeCoveringVector)
  ];
  const containsPoint = (operand: Operand, point: Vector): boolean | undefined => {
    for (const edge of operand.edges)
      if (onArc(edge.start, edge.end, point, tolerance)) return undefined;
    for (const choice of choices) {
      let parity = operand.referenceInside;
      let ambiguous = false;
      for (const [first, second] of [
        [options.referencePoint, choice],
        [choice, point]
      ]) {
        const crossing = getCoveringCross(first, second);
        if (Math.hypot(...crossing) < tolerance) {
          if (getCoveringDot(first, second) < 0) ambiguous = true;
          continue;
        }
        const normal = normalizeCoveringVector(crossing);
        for (const {start, end} of operand.edges) {
          const firstSide = getCoveringDot(normal, start);
          const secondSide = getCoveringDot(normal, end);
          if (
            (Math.abs(firstSide) <= tolerance && onArc(first, second, start, tolerance)) ||
            (Math.abs(secondSide) <= tolerance && onArc(first, second, end, tolerance))
          )
            ambiguous = true;
          if (firstSide * secondSide < 0) {
            const intersection = normalizeCoveringVector(
              getCoveringCross(crossing, getCoveringCross(start, end))
            );
            const opposite = negateCoveringVector(intersection);
            if (
              (onArc(first, second, intersection, -tolerance) &&
                onArc(start, end, intersection, -tolerance)) ||
              (onArc(first, second, opposite, -tolerance) &&
                onArc(start, end, opposite, -tolerance))
            )
              parity = !parity;
          }
        }
      }
      if (!ambiguous) return parity;
    }
    return undefined;
  };
  const distanceToArc = (first: Vector, second: Vector, point: Vector) => {
    const normal = normalizeCoveringVector(getCoveringCross(first, second));
    const projected = getCoveringCross(normal, getCoveringCross(point, normal));
    let distance = Math.min(getArcLength(first, point), getArcLength(second, point));
    if (Math.hypot(...projected) > 1e-10) {
      const nearest = normalizeCoveringVector(projected);
      if (onArc(first, second, nearest, 0))
        distance = Math.min(distance, getArcLength(nearest, point));
    }
    return distance;
  };
  const segments: Segment[] = [];
  for (let operand = 0; operand < 2; operand++) {
    for (const {start: first, end: second} of operands[operand].edges) {
      const points = [
        {point: first, angle: 0},
        {point: second, angle: getArcLength(first, second)}
      ];
      for (const {start: third, end: fourth} of operands[1 - operand].edges) {
        const crossing = getCoveringCross(
          getCoveringCross(first, second),
          getCoveringCross(third, fourth)
        );
        const intersection =
          Math.hypot(...crossing) > tolerance * tolerance
            ? normalizeCoveringVector(crossing)
            : undefined;
        const candidates = intersection
          ? [intersection, negateCoveringVector(intersection)]
          : [third, fourth];
        for (const point of candidates) {
          if (onArc(first, second, point, tolerance) && onArc(third, fourth, point, tolerance)) {
            const angle = getArcLength(first, point);
            if (!points.some(previous => Math.abs(previous.angle - angle) <= 2 * tolerance)) {
              let position = points.length;
              while (position > 0 && points[position - 1].angle > angle) position--;
              points.splice(position, 0, {point, angle});
            }
          }
        }
      }
      for (let piece = 0; piece + 1 < points.length; piece++) {
        const start = points[piece].point;
        const end = points[piece + 1].point;
        if (getArcLength(start, end) <= 4 * tolerance)
          return failResult(GPU_SPHERICAL_ARC_OVERLAY_STATUS_UNRESOLVED);
        const middle = normalizeCoveringVector(addVectors(start, end));
        const normal = normalizeCoveringVector(getCoveringCross(start, end));
        let separation = 16 * tolerance;
        for (const source of operands)
          for (const edge of source.edges)
            if (!onArc(edge.start, edge.end, middle, tolerance))
              separation = Math.min(separation, distanceToArc(edge.start, edge.end, middle) * 0.25);
        if (separation <= 2 * tolerance)
          return failResult(GPU_SPHERICAL_ARC_OVERLAY_STATUS_UNRESOLVED);
        const leftPoint = normalizeCoveringVector(
          addVectors(middle, scaleVector(normal, separation))
        );
        const rightPoint = normalizeCoveringVector(
          addVectors(middle, scaleVector(normal, -separation))
        );
        const firstLeft = containsPoint(operands[0], leftPoint);
        const secondLeft = containsPoint(operands[1], leftPoint);
        const firstRight = containsPoint(operands[0], rightPoint);
        const secondRight = containsPoint(operands[1], rightPoint);
        if (
          firstLeft === undefined ||
          secondLeft === undefined ||
          firstRight === undefined ||
          secondRight === undefined
        )
          return failResult(GPU_SPHERICAL_ARC_OVERLAY_STATUS_UNRESOLVED);
        const leftInside = combineMembership(firstLeft, secondLeft);
        const rightInside = combineMembership(firstRight, secondRight);
        if (leftInside === rightInside) continue;
        const startPoint = leftInside ? start : end;
        const endPoint = leftInside ? end : start;
        if (
          !segments.some(
            previous =>
              getVectorDistance(previous.start, startPoint) <= 4 * tolerance &&
              getVectorDistance(previous.end, endPoint) <= 4 * tolerance
          )
        ) {
          segments.push({start: startPoint, end: endPoint, used: false});
          validateMaximumCount('spherical-arc-overlay', 'counts[2]', segments.length, 0xffffffff);
          result.counts[2] = segments.length;
        }
      }
    }
  }
  if (segments.length > options.capacity)
    return failResult(GPU_SPHERICAL_ARC_OVERLAY_STATUS_CAPACITY);
  let vertexCount = 0;
  let ringCount = 0;
  for (let seed = 0; seed < segments.length; seed++) {
    if (segments[seed].used) continue;
    if (ringCount >= options.ringCapacity)
      return failResult(GPU_SPHERICAL_ARC_OVERLAY_STATUS_CAPACITY);
    result.ringOffsets[ringCount] = vertexCount;
    let current = seed;
    let closed = false;
    for (let step = 0; step < segments.length; step++) {
      const segment = segments[current];
      if (segment.used) return failResult(GPU_SPHERICAL_ARC_OVERLAY_STATUS_UNRESOLVED);
      segment.used = true;
      result.positions.set(segment.start, vertexCount++ * 3);
      const junction = segment.end;
      const incoming = normalizeCoveringVector(
        addVectors(
          scaleVector(junction, getCoveringDot(segment.start, junction)),
          negateCoveringVector(segment.start)
        )
      );
      let next = -1;
      let bestTurn = -4;
      let failure = false;
      for (let candidate = 0; candidate < segments.length; candidate++) {
        const target = segments[candidate];
        if (
          (!target.used || candidate === seed) &&
          getVectorDistance(target.start, junction) <= 4 * tolerance
        ) {
          const outgoing = normalizeCoveringVector(
            addVectors(target.end, scaleVector(junction, -getCoveringDot(junction, target.end)))
          );
          const turn = getGuardedAngle(
            getCoveringDot(junction, getCoveringCross(incoming, outgoing)),
            getCoveringDot(incoming, outgoing)
          );
          if (Math.abs(turn - bestTurn) <= tolerance) failure = true;
          if (turn > bestTurn) {
            bestTurn = turn;
            next = candidate;
          }
        }
      }
      if (next === seed) {
        closed = true;
        if (failure) return failResult(GPU_SPHERICAL_ARC_OVERLAY_STATUS_UNRESOLVED);
        break;
      }
      if (next < 0 || failure) return failResult(GPU_SPHERICAL_ARC_OVERLAY_STATUS_UNRESOLVED);
      current = next;
    }
    if (!closed) return failResult(GPU_SPHERICAL_ARC_OVERLAY_STATUS_UNRESOLVED);
    result.ringOffsets[++ringCount] = vertexCount;
  }
  result.ringOffsets.fill(vertexCount, ringCount + 1);
  result.counts[0] = vertexCount;
  result.counts[1] = ringCount;
  return result;
}

/**
 * CPU arrangement without GPU input, private-array or output scratch-size caps.
 * Operand lengths come from the supplied vectors, just as they do on the GPU path.
 */
export class CPUSphericalArcOverlayOperation extends GPUCPUOperation {
  constructor(props: SphericalArcOverlayOperationProps) {
    props = captureSphericalArcOverlayOperationProps(props);
    const operation = new GPUSphericalArcOverlayOperation(props);
    const {id} = operation;
    validateOptions(id, props);
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
        const getOperand = (name: 'left' | 'right') => {
          const positions = getCPUOperationFloat32Rows(id, inputs, `${name}Positions`, 'float32x3');
          const ringOffsets = getCPUOperationUint32Rows(id, inputs, `${name}RingOffsets`);
          return {positions, ringOffsets, referenceInside: props[name].referenceInside};
        };
        const result = computeSphericalArcOverlayOnCPU(
          {left: getOperand('left'), right: getOperand('right')},
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUSphericalArcOverlayResult]
          ])
        );
      }
    });
  }
}

function validateOptions(id: string, options: CPUSphericalArcOverlayOptions): void {
  validateCapacity(id, 'capacity', options.capacity, 3);
  validateCapacity(id, 'ringCapacity', options.ringCapacity, 1, 0xfffffffe);
  validateUnitVector(options.referencePoint, id, 'referencePoint');
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
  validateSphericalOverlayOperation(id, options.operation);
}
function addVectors(first: Vector, second: Vector): Vector {
  return [first[0] + second[0], first[1] + second[1], first[2] + second[2]];
}
function scaleVector(vector: Vector, scale: number): Vector {
  return [vector[0] * scale, vector[1] * scale, vector[2] * scale];
}
function getVectorDistance(first: Vector, second: Vector): number {
  return Math.hypot(first[0] - second[0], first[1] - second[1], first[2] - second[2]);
}
function getGuardedAngle(numerator: number, denominator: number): number {
  return numerator === 0 && denominator === 0
    ? 0
    : Math.atan2(numerator, denominator === 0 ? 0 : denominator);
}
function getArcLength(first: Vector, second: Vector): number {
  return getGuardedAngle(
    Math.hypot(...getCoveringCross(first, second)),
    getCoveringDot(first, second)
  );
}
