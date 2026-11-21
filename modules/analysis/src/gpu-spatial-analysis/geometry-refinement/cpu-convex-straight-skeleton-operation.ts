// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotRefinementOperationProps} from './refinement-construction-snapshot';

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {resolveGPUSpatialContext} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  GPUConvexStraightSkeletonOperation,
  type ConvexStraightSkeletonOperationProps
} from './gpu-convex-straight-skeleton-operation';

type Point = readonly [number, number];
type WavefrontEdge = {normal: Point; constant: number; source: number; birth: Point};

/** Packed input, with implicit or repeated closure for each independent convex ring. */
export type CPUConvexStraightSkeletonInput = {
  positions: Float32Array;
  ringOffsets: Uint32Array;
};

/** Output block dimensions and planar wavefront parameters. */
export type CPUConvexStraightSkeletonOptions = Pick<
  ConvexStraightSkeletonOperationProps,
  | 'maximumVertices'
  | 'maximumIterations'
  | 'segmentCapacity'
  | 'offsetDistance'
  | 'eventTolerance'
  | 'spatialContext'
>;

/** Fixed polygon blocks, including zero-filled unused capacity. */
export type CPUConvexStraightSkeletonResult = {
  segments: Float32Array;
  sourceEdges: Uint32Array;
  offsetPositions: Float32Array;
  /** [writtenArcs, requiredArcs, offsetVertexCount, flags]; flags: input=1, capacity=2, work=4, numerical=8. */
  status: Uint32Array;
};

function getCrossProduct(left: Point, right: Point): number {
  return left[0] * right[1] - left[1] * right[0];
}

function subtractPoints(left: Point, right: Point): Point {
  return [left[0] - right[0], left[1] - right[1]];
}

function getIntersection(left: WavefrontEdge, right: WavefrontEdge, time: number): Point {
  const determinant = getCrossProduct(left.normal, right.normal);
  return [
    ((left.constant + time) * right.normal[1] - left.normal[1] * (right.constant + time)) /
      determinant,
    (left.normal[0] * (right.constant + time) - (left.constant + time) * right.normal[0]) /
      determinant
  ];
}

function validateOptions(id: string, options: CPUConvexStraightSkeletonOptions): void {
  const maximumVertices = options.maximumVertices ?? 32;
  validateCapacity(id, 'maximumVertices', maximumVertices, 3);
  if (options.maximumIterations !== undefined)
    validateCapacity(id, 'maximumIterations', options.maximumIterations, 0);
  validateCapacity(id, 'segmentCapacity', options.segmentCapacity ?? maximumVertices * 2);
  validateFiniteNumber(id, 'offsetDistance', options.offsetDistance, 0);
  validateFiniteNumber(id, 'eventTolerance', options.eventTolerance ?? 1e-5, 0);
  validateRequirement(id, (options.eventTolerance ?? 1e-5) > 0, 'eventTolerance must be > 0');
  resolveGPUSpatialContext(id, options.spatialContext);
}

/**
 * Advances strictly convex planar rings through simultaneous edge collapses, retaining the GPU's
 * vertex order, source edge IDs, tolerance grouping, strict offset-before-event test and ridge ties.
 * Ring offsets are clamped to positions length; decreasing clamped endpoints report input flag 1
 * for that ring, without preventing subsequent rings from being processed.
 *
 * What differs from the GPU contributor:
 * - Arithmetic is float64, rounded once at output; near event/tolerance boundaries topology can differ.
 * - Collapse events run to completion when maximumIterations is omitted; an explicit budget
 *   binds at the same point as the contributor, after offset capture and before arc emission.
 * - The GPU's 64-vertex ceiling and 3e38 event-time sentinel are removed. maximumVertices still
 *   determines output block size; larger rings receive input flag 1, as on the GPU.
 * - Unused output rows are zeroed. The contributor leaves those rows untouched on reused buffers.
 * - Nonfinite coordinates set numerical flag 8 (possibly with input flag 1); undefined nonfinite
 *   intermediate intersections also set flag 8 instead of emitting nonfinite arcs.
 *
 * @throws Error naming positions, ringOffsets or an option for inconsistent input.
 */
export function computeConvexStraightSkeletonOnCPU(
  input: CPUConvexStraightSkeletonInput,
  options: CPUConvexStraightSkeletonOptions
): CPUConvexStraightSkeletonResult {
  validateOptions('convex-straight-skeleton', options);
  const {positions, ringOffsets} = input;
  if (positions.length % 2 !== 0) throw new Error('positions must contain float32x2 rows');
  if (ringOffsets.length < 1) throw new Error('ringOffsets must contain at least one offset');
  const positionCount = positions.length / 2;
  const offsets = Array.from(ringOffsets, offset => Math.min(offset, positionCount));
  const maximumVertices = options.maximumVertices ?? 32;
  const segmentCapacity = options.segmentCapacity ?? maximumVertices * 2;
  const eventTolerance = options.eventTolerance ?? 1e-5;
  const polygonCount = offsets.length - 1;
  const result: CPUConvexStraightSkeletonResult = {
    segments: new Float32Array(polygonCount * segmentCapacity * 4),
    sourceEdges: new Uint32Array(polygonCount * segmentCapacity * 2),
    offsetPositions: new Float32Array(polygonCount * maximumVertices * 2),
    status: new Uint32Array(polygonCount * 4)
  };
  const getPoint = (row: number): Point => [positions[row * 2], positions[row * 2 + 1]];
  for (let polygon = 0; polygon < polygonCount; polygon++) {
    const first = offsets[polygon];
    if (offsets[polygon + 1] < first) {
      // The shader's unsigned subtraction exceeds maximumVertices and reports unsupported.
      result.status[polygon * 4 + 3] = 1;
      continue;
    }
    let count = offsets[polygon + 1] - first;
    if (
      count > 1 &&
      positions[first * 2] === positions[(first + count - 1) * 2] &&
      positions[first * 2 + 1] === positions[(first + count - 1) * 2 + 1]
    )
      count--;
    let flags = count < 3 || count > maximumVertices ? 1 : 0;
    let required = 0;
    let offsetCount = 0;
    let offsetWritten = false;
    let edges: WavefrontEdge[] = [];
    const orientation =
      flags === 0
        ? Math.sign(
            getCrossProduct(
              subtractPoints(getPoint(first + 1), getPoint(first)),
              subtractPoints(getPoint(first + 2), getPoint(first + 1))
            )
          )
        : 0;
    if (flags === 0) {
      for (let vertex = 0; vertex < count; vertex++) {
        const nextVertex = (vertex + 1) % count;
        const current = getPoint(first + vertex);
        const next = getPoint(first + nextVertex);
        const tangent = subtractPoints(next, current);
        if (!current.every(Number.isFinite)) flags |= 8;
        for (let other = 0; other < count; other++) {
          if (other === vertex || other === nextVertex) continue;
          if (
            getCrossProduct(tangent, subtractPoints(getPoint(first + other), current)) *
              orientation <
            0
          )
            flags |= 1;
        }
        const following = subtractPoints(getPoint(first + ((vertex + 2) % count)), next);
        if (getCrossProduct(tangent, following) * orientation <= 0) flags |= 1;
        const length = Math.hypot(...tangent);
        const normal: Point = [
          (-orientation * tangent[1]) / length,
          (orientation * tangent[0]) / length
        ];
        edges.push({
          normal,
          constant: normal[0] * current[0] + normal[1] * current[1],
          source: first + vertex,
          birth: current
        });
      }
    }
    const emitArc = (start: Point, end: Point, left: number, right: number) => {
      if (required < segmentCapacity) {
        const destination = polygon * segmentCapacity + required;
        result.segments.set([...start, ...end], destination * 4);
        result.sourceEdges.set([left, right], destination * 2);
      }
      required++;
    };
    let time = 0;
    let iterations = 0;
    while (flags === 0 && edges.length >= 3) {
      count = edges.length;
      const eventTimes = edges.map((edge, vertex) => {
        const previous = edges[(vertex + count - 1) % count];
        const next = edges[(vertex + 1) % count];
        const currentPoint = getIntersection(previous, edge, time);
        const nextPoint = getIntersection(edge, next, time);
        const currentVelocity = getIntersection(
          {...previous, constant: 0},
          {...edge, constant: 0},
          1
        );
        const nextVelocity = getIntersection({...edge, constant: 0}, {...next, constant: 0}, 1);
        const tangent: Point = [orientation * edge.normal[1], -orientation * edge.normal[0]];
        const shrinkRate =
          (nextVelocity[0] - currentVelocity[0]) * tangent[0] +
          (nextVelocity[1] - currentVelocity[1]) * tangent[1];
        const separation =
          (nextPoint[0] - currentPoint[0]) * tangent[0] +
          (nextPoint[1] - currentPoint[1]) * tangent[1];
        return shrinkRate < 0 ? Math.max(time, time - separation / shrinkRate) : Infinity;
      });
      const nextTime = eventTimes.reduce(
        (earliest, eventTime) => Math.min(earliest, eventTime),
        Infinity
      );
      if (!Number.isFinite(nextTime)) {
        flags |= 8;
        break;
      }
      if (!offsetWritten && options.offsetDistance < nextTime) {
        offsetCount = count;
        offsetWritten = true;
        for (let vertex = 0; vertex < count; vertex++) {
          result.offsetPositions.set(
            getIntersection(
              edges[(vertex + count - 1) % count],
              edges[vertex],
              options.offsetDistance
            ),
            (polygon * maximumVertices + vertex) * 2
          );
        }
      }
      if (iterations >= (options.maximumIterations ?? Infinity)) {
        flags |= 4;
        break;
      }
      const epsilon = eventTolerance * Math.max(1, Math.abs(nextTime));
      const removed = eventTimes.map(eventTime => eventTime <= nextTime + epsilon);
      const retained = edges.flatMap((_, vertex) => (removed[vertex] ? [] : [vertex]));
      const eventPositions = edges.map((edge, vertex) =>
        getIntersection(edges[(vertex + count - 1) % count], edge, nextTime)
      );
      if (eventPositions.some(point => !point.every(Number.isFinite))) {
        flags |= 8;
        break;
      }
      for (let vertex = 0; vertex < count; vertex++) {
        const previous = (vertex + count - 1) % count;
        if (removed[previous] || removed[vertex] || retained.length < 3) {
          if (
            Math.hypot(...subtractPoints(edges[vertex].birth, eventPositions[vertex])) > epsilon
          ) {
            emitArc(
              edges[vertex].birth,
              eventPositions[vertex],
              edges[previous].source,
              edges[vertex].source
            );
          }
        }
      }
      if (retained.length < 3) {
        let left = eventPositions[0];
        let right = left;
        let longest = 0;
        for (let vertex = 0; vertex < count; vertex++) {
          for (let other = vertex + 1; other < count; other++) {
            const separation = Math.hypot(
              ...subtractPoints(eventPositions[vertex], eventPositions[other])
            );
            if (separation > longest) {
              longest = separation;
              left = eventPositions[vertex];
              right = eventPositions[other];
            }
          }
        }
        if (longest > epsilon) emitArc(left, right, 0xffffffff, 0xffffffff);
        break;
      }
      edges = retained.map((current, vertex) => {
        const previous = retained[(vertex + retained.length - 1) % retained.length];
        return {
          ...edges[current],
          birth:
            (previous + 1) % count === current
              ? edges[current].birth
              : getIntersection(edges[previous], edges[current], nextTime)
        };
      });
      time = nextTime;
      iterations++;
    }
    if (required > segmentCapacity) flags |= 2;
    result.status.set(
      [Math.min(required, segmentCapacity), required, offsetCount, flags],
      polygon * 4
    );
  }
  return result;
}

/** CPU twin: explicit iteration budgets bind; maximumVertices retains its layout role. */
export class CPUConvexStraightSkeletonOperation extends GPUCPUOperation {
  constructor(props: ConvexStraightSkeletonOperationProps) {
    props = snapshotRefinementOperationProps(props, 'convex');
    const id = props.id ?? 'convex-straight-skeleton';
    validateOptions(id, props);
    validateCapacity(id, 'polygonCount', props.polygonCount);
    validateCapacity(
      id,
      'maximumIterations',
      props.maximumIterations ?? props.maximumVertices ?? 32,
      0
    );
    const operation = new GPUConvexStraightSkeletonOperation(props);
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
        const ringOffsets = getCPUOperationUint32Rows(id, inputs, 'ringOffsets');
        if (ringOffsets.length !== props.polygonCount + 1)
          throw new Error(`${id} ringOffsets must have polygonCount + 1 rows`);
        const result = computeConvexStraightSkeletonOnCPU(
          {
            positions: getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2'),
            ringOffsets
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUConvexStraightSkeletonResult]
          ])
        );
      }
    });
  }
}
