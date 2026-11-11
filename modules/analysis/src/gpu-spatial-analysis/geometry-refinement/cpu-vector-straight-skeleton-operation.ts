// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {validateCanonicalOffsets, type CanonicalPoint} from './cpu-canonical-geometry';
import {
  GPUVectorStraightSkeletonOperation,
  type VectorStraightSkeletonOperationProps
} from './gpu-vector-straight-skeleton-operation';

type Point = CanonicalPoint;
type Edge = {normal: Point; constant: number; source: number};
type Vertex = {
  start: Point;
  velocity: Point;
  birth: Point;
  left: number;
  right: number;
  previous: number;
  next: number;
  active: boolean;
};

/** Packed polygon hierarchy; the first ring of each polygon is its shell. */
export type CPUVectorStraightSkeletonInput = {
  positions: Float32Array;
  ringOffsets: Uint32Array;
  polygonOffsets: Uint32Array;
  featureOffsets: Uint32Array;
};

/** Fixed output block dimensions and planar wavefront parameters. */
export type CPUVectorStraightSkeletonOptions = Pick<
  VectorStraightSkeletonOperationProps,
  'maximumVertices' | 'maximumEvents' | 'segmentCapacity' | 'offsetDistance' | 'eventTolerance'
>;

/** Same fixed polygon blocks and status words as GPUVectorStraightSkeleton. */
export type CPUVectorStraightSkeletonResult = {
  segments: Float32Array;
  sourceEdges: Uint32Array;
  offsetPositions: Float32Array;
  offsetLinks: Uint32Array;
  /** [writtenArcs, requiredArcs, activeOffsetVertices, flags]: input=1, capacity=2, work=4, numerical=8. */
  status: Uint32Array;
};

function subtractPoints(first: Point, second: Point): Point {
  return [first[0] - second[0], first[1] - second[1]];
}

function computeCrossProduct(first: Point, second: Point): number {
  return first[0] * second[1] - first[1] * second[0];
}

function computeDotProduct(first: Point, second: Point): number {
  return first[0] * second[0] + first[1] * second[1];
}

function computeDistance(first: Point, second: Point): number {
  return Math.hypot(first[0] - second[0], first[1] - second[1]);
}

function computePosition(vertex: Vertex, time: number): Point {
  return [vertex.start[0] + time * vertex.velocity[0], vertex.start[1] + time * vertex.velocity[1]];
}

function createVertex(
  edges: readonly Edge[],
  left: number,
  right: number,
  previous: number,
  next: number,
  birth: Point,
  time: number
): Vertex {
  const first = edges[left];
  const second = edges[right];
  const determinant = computeCrossProduct(first.normal, second.normal);
  let start = birth;
  let velocity: Point = [0, 0];
  if (Math.abs(determinant) > 1e-7) {
    start = [
      (first.constant * second.normal[1] - first.normal[1] * second.constant) / determinant,
      (first.normal[0] * second.constant - first.constant * second.normal[0]) / determinant
    ];
    velocity = [
      (second.normal[1] - first.normal[1]) / determinant,
      (first.normal[0] - second.normal[0]) / determinant
    ];
  } else if (computeDotProduct(first.normal, second.normal) > 0.999999) {
    velocity = first.normal;
    start = [birth[0] - time * velocity[0], birth[1] - time * velocity[1]];
  }
  return {start, velocity, birth, left, right, previous, next, active: true};
}

function validateOptions(id: string, options: CPUVectorStraightSkeletonOptions): void {
  validateCapacity(id, 'maximumVertices', options.maximumVertices ?? 128, 3);
  validateCapacity(id, 'maximumEvents', options.maximumEvents ?? 128, 0);
  validateCapacity(id, 'segmentCapacity', options.segmentCapacity);
  validateCapacity(
    id,
    'maximumVertices + 2 * maximumEvents',
    (options.maximumVertices ?? 128) + 2 * (options.maximumEvents ?? 128),
    3
  );
  validateFiniteNumber(id, 'offsetDistance', options.offsetDistance, 0);
  validateRequirement(id, options.offsetDistance >= 0, 'offsetDistance must be >= 0');
  validateFiniteNumber(id, 'eventTolerance', options.eventTolerance ?? 1e-5, 0);
  validateRequirement(id, (options.eventTolerance ?? 1e-5) > 0, 'eventTolerance must be > 0');
}

function validateOffsets(name: string, offsets: Uint32Array, limit: number): void {
  if (offsets.length === 1) {
    if (offsets[0] > limit) throw new Error(`${name} must be in [0, ${limit}]`);
    return;
  }
  validateCanonicalOffsets(name, offsets, limit);
}

function getWavefrontState(vertices: readonly Vertex[], active: readonly number[]): string {
  // Historical slots grow at every event. Renumber only active slots in ascending order so
  // equivalent wavefronts compare equal while preserving the event search's tie-breaking order.
  const ordinals = new Map(active.map((index, ordinal) => [index, ordinal]));
  return JSON.stringify(
    active.map(index => {
      const vertex = vertices[index];
      return [
        vertex.left,
        vertex.right,
        ordinals.get(vertex.previous),
        ordinals.get(vertex.next),
        ...vertex.start,
        ...vertex.velocity,
        ...vertex.birth
      ];
    })
  );
}

/**
 * Advances linked planar wavefronts through edge collapses, reflex splits and hole merges.
 * Original slot order, shell/hole winding, source position IDs, inclusive offset-before-event
 * capture, tolerance tests, terminal ridges and per-polygon capacity status follow the contributor.
 *
 * What differs from the GPU contributor:
 * - Arithmetic is float64 with one float32 rounding at output. Near tolerance/event ties, topology
 *   and therefore slot IDs and arc order can differ; the GPU's tolerance definitions are retained.
 * - The event-count budget is removed. maximumEvents only sizes the fixed offset-slot layout;
 *   flag 4 is set only when that layout is exhausted before capture. After capture, event processing
 *   has no iteration budget. Increase maximumEvents if a later snapshot needs more historical slots.
 * - A repeated active wavefront at the same event time terminates with numerical flag 8,
 *   retaining emitted arcs and the captured offset. The GPU has no repeat-state check and can
 *   instead stop at its work budget with flag 4.
 * - The 1024-vertex and 4096-event ceilings and the 3e38 event-time sentinel are removed.
 *   maximumVertices still bounds initial vertices because it participates in the output layout.
 * - Unwritten float/source rows are zeroed; the GPU leaves them untouched on reused buffers.
 * - Nonfinite input coordinates set input flag 1. Nonfinite motion sets numerical flag 8;
 *   the GPU's result for nonfinite intermediate arithmetic is undefined.
 * - Malformed offset hierarchies throw an Error naming the array; the GPU does not validate them.
 *
 * @throws Error naming an argument when arrays or options are inconsistent.
 */
export function computeVectorStraightSkeletonOnCPU(
  input: CPUVectorStraightSkeletonInput,
  options: CPUVectorStraightSkeletonOptions
): CPUVectorStraightSkeletonResult {
  validateOptions('vector-straight-skeleton', options);
  const {positions, ringOffsets, polygonOffsets, featureOffsets} = input;
  if (positions.length % 2 !== 0) throw new Error('positions must contain float32x2 rows');
  validateOffsets('ringOffsets', ringOffsets, positions.length / 2);
  validateOffsets('polygonOffsets', polygonOffsets, ringOffsets.length - 1);
  validateOffsets('featureOffsets', featureOffsets, polygonOffsets.length - 1);
  const maximumVertices = options.maximumVertices ?? 128;
  const slots = maximumVertices + 2 * (options.maximumEvents ?? 128);
  const {segmentCapacity, offsetDistance} = options;
  const tolerance = options.eventTolerance ?? 1e-5;
  const polygonCount = polygonOffsets.length - 1;
  const result: CPUVectorStraightSkeletonResult = {
    segments: new Float32Array(polygonCount * segmentCapacity * 4),
    sourceEdges: new Uint32Array(polygonCount * segmentCapacity * 2),
    offsetPositions: new Float32Array(polygonCount * slots * 2),
    offsetLinks: new Uint32Array(polygonCount * slots * 2).fill(0xffffffff),
    status: new Uint32Array(polygonCount * 4)
  };
  const getPoint = (row: number): Point => [positions[row * 2], positions[row * 2 + 1]];
  const equalPoints = (first: Point, second: Point) =>
    first[0] === second[0] && first[1] === second[1];
  for (let polygon = 0; polygon < polygonCount; polygon++) {
    let flags = 0;
    let required = 0;
    let offsetCount = 0;
    let captured = false;
    let time = 0;
    let wavefrontTime = -Infinity;
    const wavefrontStates = new Set<string>();
    const edges: Edge[] = [];
    const vertices: Vertex[] = [];
    for (let ring = polygonOffsets[polygon]; ring < polygonOffsets[polygon + 1]; ring++) {
      const first = ringOffsets[ring];
      let last = ringOffsets[ring + 1];
      if (last > first && equalPoints(getPoint(first), getPoint(last - 1))) last--;
      const length = last - first;
      if (length < 3) {
        flags |= 1;
        continue;
      }
      let area = 0;
      for (let vertex = first; vertex < last; vertex++)
        area += computeCrossProduct(
          getPoint(vertex),
          getPoint(vertex + 1 === last ? first : vertex + 1)
        );
      const reverse = ring > polygonOffsets[polygon] ? area > 0 : area < 0;
      const sources: number[] = [];
      for (let ordinal = 0; ordinal < length; ordinal++) {
        const source = first + (reverse ? length - 1 - ordinal : ordinal);
        const location = getPoint(source);
        if (
          equalPoints(location, getPoint(first + ((source - first + length - 1) % length))) ||
          equalPoints(location, getPoint(first + ((source - first + 1) % length)))
        ) {
          flags |= 1;
          continue;
        }
        if (edges.length + sources.length >= maximumVertices) {
          flags |= 1;
          break;
        }
        if (!location.every(Number.isFinite)) flags |= 1;
        sources.push(source);
      }
      if (sources.length < 3 || flags !== 0) {
        flags |= 1;
        break;
      }
      const ringStart = edges.length;
      for (let ordinal = 0; ordinal < sources.length; ordinal++) {
        const start = getPoint(sources[ordinal]);
        const finish = getPoint(sources[(ordinal + 1) % sources.length]);
        const direction = subtractPoints(finish, start);
        const length = Math.hypot(...direction);
        const normal: Point = [-direction[1] / length, direction[0] / length];
        edges.push({normal, constant: computeDotProduct(normal, start), source: sources[ordinal]});
      }
      for (let ordinal = 0; ordinal < sources.length; ordinal++) {
        const previous = ringStart + ((ordinal + sources.length - 1) % sources.length);
        const next = ringStart + ((ordinal + 1) % sources.length);
        vertices.push(
          createVertex(
            edges,
            previous,
            ringStart + ordinal,
            previous,
            next,
            getPoint(sources[ordinal]),
            0
          )
        );
      }
    }
    const emitArc = (vertex: Vertex, end: Point) => {
      if (computeDistance(vertex.birth, end) <= tolerance) return;
      if (required < segmentCapacity) {
        const destination = polygon * segmentCapacity + required;
        result.segments.set([...vertex.birth, ...end], destination * 4);
        result.sourceEdges.set(
          [edges[vertex.left].source, edges[vertex.right].source],
          destination * 2
        );
      }
      required++;
    };
    while (flags === 0) {
      if (vertices.some(vertex => ![...vertex.start, ...vertex.velocity].every(Number.isFinite))) {
        flags |= 8;
        break;
      }
      const active = vertices.flatMap((vertex, index) => (vertex.active ? [index] : []));
      // Two-vertex components emit the ridge between birth points, exactly as the kernel does.
      for (const index of active) {
        const vertex = vertices[index];
        if (!vertex.active) continue;
        if (vertex.next === index) vertex.active = false;
        else if (vertices[vertex.next].next === index) {
          emitArc(vertex, vertices[vertex.next].birth);
          vertex.active = false;
          vertices[vertex.next].active = false;
        }
      }
      const visited = new Set<number>();
      for (const root of active) {
        if (!vertices[root].active || visited.has(root)) continue;
        const component: number[] = [];
        let current = root;
        let area = 0;
        let perimeter = 0;
        do {
          component.push(current);
          visited.add(current);
          const next = vertices[current].next;
          const start = computePosition(vertices[current], time);
          const finish = computePosition(vertices[next], time);
          area += computeCrossProduct(start, finish);
          perimeter += computeDistance(start, finish);
          current = next;
        } while (current !== root && component.length <= vertices.length);
        if (component.length > vertices.length) {
          flags |= 8;
          break;
        }
        if (Math.abs(area) > tolerance * (1 + perimeter)) continue;
        for (const index of component)
          emitArc(vertices[index], computePosition(vertices[index], time));
        for (const index of component) {
          const vertex = vertices[index];
          const start = computePosition(vertex, time);
          const finish = computePosition(vertices[vertex.next], time);
          const direction = subtractPoints(finish, start);
          const lengthSquared = computeDotProduct(direction, direction);
          let keep = lengthSquared > tolerance * tolerance;
          for (const other of component) {
            const otherStart = computePosition(vertices[other], time);
            const otherFinish = computePosition(vertices[vertices[other].next], time);
            const separation = subtractPoints(otherStart, start);
            const fraction =
              computeDotProduct(separation, direction) / Math.max(lengthSquared, 1e-30);
            if (
              fraction > tolerance &&
              fraction < 1 - tolerance &&
              Math.abs(computeCrossProduct(separation, direction)) <=
                tolerance * Math.sqrt(lengthSquared)
            )
              keep = false;
            if (
              other < index &&
              ((computeDistance(start, otherStart) <= tolerance &&
                computeDistance(finish, otherFinish) <= tolerance) ||
                (computeDistance(start, otherFinish) <= tolerance &&
                  computeDistance(finish, otherStart) <= tolerance))
            )
              keep = false;
          }
          if (keep) {
            vertex.birth = start;
            emitArc(vertex, finish);
          }
        }
        for (const index of component) vertices[index].active = false;
      }
      if (flags !== 0 || !vertices.some(vertex => vertex.active)) break;
      if (time !== wavefrontTime) {
        wavefrontStates.clear();
        wavefrontTime = time;
      }
      const wavefrontState = getWavefrontState(
        vertices,
        active.filter(index => vertices[index].active)
      );
      if (wavefrontStates.has(wavefrontState)) {
        flags |= 8;
        break;
      }
      wavefrontStates.add(wavefrontState);
      let eventTime = Infinity;
      let eventVertex = -1;
      let eventEdge = -1;
      let split = false;
      for (const index of active) {
        const vertex = vertices[index];
        if (!vertex.active) continue;
        const next = vertices[vertex.next];
        const delta = subtractPoints(next.start, vertex.start);
        const velocity = subtractPoints(next.velocity, vertex.velocity);
        const denominator = computeDotProduct(velocity, velocity);
        if (denominator <= 1e-12) continue;
        const candidate = -computeDotProduct(delta, velocity) / denominator;
        if (
          candidate >= time - tolerance &&
          candidate < eventTime &&
          Math.hypot(delta[0] + candidate * velocity[0], delta[1] + candidate * velocity[1]) <=
            tolerance * (1 + Math.hypot(...computePosition(vertex, candidate)))
        ) {
          eventTime = Math.max(time, candidate);
          eventVertex = index;
          eventEdge = vertex.next;
          split = false;
        }
      }
      for (const index of active) {
        const vertex = vertices[index];
        if (
          !vertex.active ||
          computeCrossProduct(edges[vertex.left].normal, edges[vertex.right].normal) >= -1e-7
        )
          continue;
        for (const edgeIndex of active) {
          const edgeVertex = vertices[edgeIndex];
          if (!edgeVertex.active || edgeIndex === index || edgeVertex.next === index) continue;
          const edge = edges[edgeVertex.right];
          const rate = computeDotProduct(edge.normal, vertex.velocity) - 1;
          if (rate >= -1e-7) continue;
          const candidate = (edge.constant - computeDotProduct(edge.normal, vertex.start)) / rate;
          if (candidate < time - tolerance || candidate >= eventTime - tolerance) continue;
          const location = computePosition(vertex, candidate);
          const start = computePosition(edgeVertex, candidate);
          const direction = subtractPoints(
            computePosition(vertices[edgeVertex.next], candidate),
            start
          );
          const fraction =
            computeDotProduct(subtractPoints(location, start), direction) /
            computeDotProduct(direction, direction);
          if (fraction >= -tolerance && fraction <= 1 + tolerance) {
            eventTime = Math.max(time, candidate);
            eventVertex = index;
            eventEdge = edgeIndex;
            split = true;
          }
        }
      }
      if (!captured && offsetDistance <= eventTime) {
        for (let index = 0; index < vertices.length; index++) {
          const vertex = vertices[index];
          if (!vertex.active) continue;
          const destination = polygon * slots + index;
          result.offsetPositions.set(computePosition(vertex, offsetDistance), destination * 2);
          result.offsetLinks.set([vertex.previous, vertex.next], destination * 2);
          offsetCount++;
        }
        captured = true;
      }
      if (eventVertex < 0) {
        flags |= 8;
        break;
      }
      if (!captured && vertices.length + (split ? 2 : 1) > slots) {
        flags |= 4;
        break;
      }
      const vertex = vertices[eventVertex];
      const edgeVertex = vertices[eventEdge];
      const location = computePosition(vertex, eventTime);
      const previous = vertex.previous;
      const next = vertex.next;
      const count = vertices.length;
      emitArc(vertex, location);
      vertex.active = false;
      if (split) {
        const edgeNext = edgeVertex.next;
        const edgeId = edgeVertex.right;
        vertices.push(
          createVertex(edges, vertex.left, edgeId, previous, edgeNext, location, eventTime)
        );
        vertices.push(
          createVertex(edges, edgeId, vertex.right, eventEdge, next, location, eventTime)
        );
        vertices[previous].next = count;
        vertices[edgeNext].previous = count;
        edgeVertex.next = count + 1;
        vertices[next].previous = count + 1;
      } else {
        const after = edgeVertex.next;
        emitArc(edgeVertex, location);
        edgeVertex.active = false;
        vertices.push(
          createVertex(edges, vertex.left, edgeVertex.right, previous, after, location, eventTime)
        );
        vertices[previous].next = count;
        vertices[after].previous = count;
      }
      time = eventTime;
    }
    if (required > segmentCapacity) flags |= 2;
    result.status.set(
      [Math.min(required, segmentCapacity), required, offsetCount, flags],
      polygon * 4
    );
  }
  return result;
}

/** CPU twin; maximumEvents sizes historical offset slots but does not limit event iterations. */
export class CPUVectorStraightSkeletonOperation extends GPUCPUOperation {
  constructor(props: VectorStraightSkeletonOperationProps) {
    const id = props.id ?? 'vector-straight-skeleton';
    validateOptions(id, props);
    validateCapacity(id, 'polygonCount', props.polygonCount, 0);
    const operation = new GPUVectorStraightSkeletonOperation(props);
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
        const polygonOffsets = getCPUOperationUint32Rows(id, inputs, 'polygonOffsets');
        if (polygonOffsets.length !== props.polygonCount + 1)
          throw new Error(`${id} polygonOffsets must have polygonCount + 1 rows`);
        const result = computeVectorStraightSkeletonOnCPU(
          {
            positions: getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2'),
            ringOffsets: getCPUOperationUint32Rows(id, inputs, 'ringOffsets'),
            polygonOffsets,
            featureOffsets: getCPUOperationUint32Rows(id, inputs, 'featureOffsets')
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUVectorStraightSkeletonResult]
          ])
        );
      }
    });
  }
}
