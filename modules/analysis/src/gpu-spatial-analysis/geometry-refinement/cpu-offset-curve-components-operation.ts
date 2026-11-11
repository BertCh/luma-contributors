// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  nodeCPURefinementLines,
  validateCPURefinementLines,
  type CPURefinementPoint
} from './cpu-refinement-line-noding';
import {
  GPUOffsetCurveComponentsOperation,
  type OffsetCurveComponentsOperationProps
} from './gpu-offset-curve-components-operation';

/** Packed original lines and their raw offset curves, in matching line order. */
export type CPUOffsetCurveComponentsInput = {
  original: {positions: Float32Array; lineOffsets: Uint32Array};
  raw: {positions: Float32Array; lineOffsets: Uint32Array};
};

/** Output dimensions and the contributor's geometric slack/provenance parameters. */
export type CPUOffsetCurveComponentsOptions = Pick<
  OffsetCurveComponentsOperationProps,
  'distance' | 'tolerance' | 'rowsPerVertex' | 'vertexCapacity' | 'componentCapacity'
>;

/** All graph outputs, including partial unreferenced vertex prefixes and zeroed unused rows. */
export type CPUOffsetCurveComponentsResult = {
  positions: Float32Array;
  components: Uint32Array;
  sourceEdges: Uint32Array;
  /** [writtenComponents,writtenVertices,requiredComponents,requiredVertices,overflow,incompleteNoding,0,0]. */
  status: Uint32Array;
};

type Point = {x: number; y: number};
type Fragment = {start: Point; end: Point; line: number; source: number; state: number};

// Match the contributor's finite separation initializer, including lines with no usable edges.
const MAXIMUM_SEPARATION = Math.fround(3e38);

function validateOptions(id: string, options: CPUOffsetCurveComponentsOptions): void {
  validateCapacity(id, 'vertexCapacity', options.vertexCapacity, 0);
  validateCapacity(id, 'componentCapacity', options.componentCapacity, 0);
  validateCapacity(id, 'rowsPerVertex', options.rowsPerVertex ?? 1);
  validateFiniteNumber(id, 'distance', options.distance);
  validateFiniteNumber(id, 'tolerance', options.tolerance, 0);
}

/**
 * Nodes raw planar offsets, trims their source-distance bands, cancels duplicate/opposite fragments,
 * and chains components in the contributor's seed/successor order. Preserves its geometric slack,
 * 1e-7 cut-interval rule, first-match ties, whole-component metadata publication and partial writes.
 *
 * What differs from the GPU contributor:
 * - Arithmetic is float64, rounded once at output; distance-band, coincidence and cut decisions
 *   near tolerance boundaries can differ.
 * - Noding is uncapped and uses exact rational predicates/visit keys on float32 input edges;
 *   intersectionCapacity, segmentCapacity, the 1024-successor scan and GPU binding limits are
 *   dropped. No uncertain-predicate status is produced; finite raw inputs set incompleteNoding=0.
 * - Intersections remain float64 through clipping instead of being rounded in intermediate buffers.
 * - Unused positions/sourceEdges are zeroed; the GPU leaves them untouched on reused buffers.
 * - Empty coordinate arrays with valid zero offsets are accepted. Empty raw arrays return
 *   zero-filled outputs with all-zero status; the GPU noding delegate requires a raw coordinate row.
 * - Inconsistent offsets throw instead of relying on out-of-bounds shader reads. Nonfinite raw
 *   vertices set incompleteNoding=1 and their edges are skipped. Nonfinite original edges are
 *   ignored by distance tests; the GPU does not define their geometric result.
 *
 * @throws Error naming original, raw or an option for inconsistent arrays/properties.
 */
export function computeOffsetCurveComponentsOnCPU(
  input: CPUOffsetCurveComponentsInput,
  options: CPUOffsetCurveComponentsOptions
): CPUOffsetCurveComponentsResult {
  validateOptions('offset-curve-components', options);
  validateCPURefinementLines(input.original.positions, input.original.lineOffsets, 'original');
  validateCPURefinementLines(input.raw.positions, input.raw.lineOffsets, 'raw');
  if (input.original.lineOffsets.length !== input.raw.lineOffsets.length)
    throw new Error('raw.lineOffsets must match original.lineOffsets length');
  const noded = nodeCPURefinementLines(input.raw.positions, input.raw.lineOffsets);
  const distance = Math.abs(options.distance);
  const tolerance = options.tolerance;
  const rowsPerVertex = options.rowsPerVertex ?? 1;
  const pointAt = (row: number): Point => ({
    x: input.original.positions[row * 2],
    y: input.original.positions[row * 2 + 1]
  });
  const offsets = input.original.lineOffsets;
  const getSeparation = (query: Point, line: number, owner: number, outerJoin: boolean): number => {
    let minimum = MAXIMUM_SEPARATION;
    for (let vertex = offsets[line]; vertex + 1 < offsets[line + 1]; vertex++) {
      if (outerJoin && (vertex === owner || vertex + 1 === owner)) continue;
      const start = pointAt(vertex);
      const finish = pointAt(vertex + 1);
      if (![start.x, start.y, finish.x, finish.y].every(Number.isFinite)) continue;
      const directionX = finish.x - start.x;
      const directionY = finish.y - start.y;
      const denominator = directionX ** 2 + directionY ** 2;
      if (denominator === 0) continue;
      const fraction = Math.max(
        0,
        Math.min(
          1,
          ((query.x - start.x) * directionX + (query.y - start.y) * directionY) / denominator
        )
      );
      minimum = Math.min(
        minimum,
        Math.hypot(
          query.x - start.x - fraction * directionX,
          query.y - start.y - fraction * directionY
        )
      );
    }
    return minimum;
  };
  const fragments: Fragment[] = [];
  for (const segment of noded.segments) {
    const {start, end, line} = segment;
    const directionX = end.x - start.x;
    const directionY = end.y - start.y;
    const squaredLength = directionX ** 2 + directionY ** 2;
    if (squaredLength === 0) continue;
    // GPULineNoding publishes the nearest input edge to the atomic segment's midpoint,
    // with ties to the smallest edge ID, including coincident traversals of the same line.
    const midpointX = (start.x + end.x) * 0.5;
    const midpointY = (start.y + end.y) * 0.5;
    let rawEdge = 0xffffffff;
    let bestError = Infinity;
    for (
      let candidate = input.raw.lineOffsets[line];
      candidate + 1 < input.raw.lineOffsets[line + 1];
      candidate++
    ) {
      const startX = input.raw.positions[candidate * 2];
      const startY = input.raw.positions[candidate * 2 + 1];
      const deltaX = input.raw.positions[(candidate + 1) * 2] - startX;
      const deltaY = input.raw.positions[(candidate + 1) * 2 + 1] - startY;
      const lengthSquared = deltaX ** 2 + deltaY ** 2;
      if (lengthSquared === 0 || !Number.isFinite(lengthSquared)) continue;
      const parameter = Math.max(
        0,
        Math.min(1, ((midpointX - startX) * deltaX + (midpointY - startY) * deltaY) / lengthSquared)
      );
      const error =
        (midpointX - (startX + parameter * deltaX)) ** 2 +
        (midpointY - (startY + parameter * deltaY)) ** 2;
      if (error < bestError) {
        bestError = error;
        rawEdge = candidate;
      }
    }
    const owner = Math.floor(rawEdge / rowsPerVertex);
    let outerJoin = false;
    if (
      (rawEdge % rowsPerVertex) + 1 < rowsPerVertex &&
      owner > offsets[line] &&
      owner + 1 < offsets[line + 1]
    ) {
      const before = pointAt(owner - 1);
      const current = pointAt(owner);
      const after = pointAt(owner + 1);
      outerJoin =
        ((current.x - before.x) * (after.y - current.y) -
          (current.y - before.y) * (after.x - current.x)) *
          options.distance <
        0;
    }
    const cuts = [0, 1];
    const insertCut = (fraction: number) => {
      if (fraction > 0 && fraction < 1) cuts.push(fraction);
    };
    for (let vertex = offsets[line]; vertex + 1 < offsets[line + 1]; vertex++) {
      const first = pointAt(vertex);
      const last = pointAt(vertex + 1);
      const edgeX = last.x - first.x;
      const edgeY = last.y - first.y;
      const lengthSquared = edgeX ** 2 + edgeY ** 2;
      if (lengthSquared === 0 || !Number.isFinite(lengthSquared)) continue;
      const normalX = -edgeY / Math.sqrt(lengthSquared);
      const normalY = edgeX / Math.sqrt(lengthSquared);
      const denominator = directionX * normalX + directionY * normalY;
      if (Math.abs(denominator) > 1e-20) {
        for (const side of [-distance, distance]) {
          const fraction =
            (side - ((start.x - first.x) * normalX + (start.y - first.y) * normalY)) / denominator;
          const projection =
            ((start.x + fraction * directionX - first.x) * edgeX +
              (start.y + fraction * directionY - first.y) * edgeY) /
            lengthSquared;
          if (projection >= 0 && projection <= 1) insertCut(fraction);
        }
      }
      for (const endpoint of [first, last]) {
        const deltaX = start.x - endpoint.x;
        const deltaY = start.y - endpoint.y;
        const linear = deltaX * directionX + deltaY * directionY;
        const constant = deltaX ** 2 + deltaY ** 2 - distance ** 2;
        const discriminant = linear ** 2 - squaredLength * constant;
        if (discriminant >= 0) {
          const root = Math.sqrt(discriminant);
          insertCut((-linear - root) / squaredLength);
          insertCut((-linear + root) / squaredLength);
        }
      }
    }
    cuts.sort((left, right) => left - right);
    const outwardX = -directionY / Math.sqrt(squaredLength);
    const outwardY = directionX / Math.sqrt(squaredLength);
    const epsilon = Math.max(1e-5, tolerance * 2, distance * 1e-5);
    for (let slot = 0; slot + 1 < cuts.length; slot++) {
      const first = cuts[slot];
      const last = cuts[slot + 1];
      if (last - first <= 1e-7) continue;
      const middle = {
        x: start.x + (first + last) * 0.5 * directionX,
        y: start.y + (first + last) * 0.5 * directionY
      };
      if (getSeparation(middle, line, owner, outerJoin) + tolerance < distance) continue;
      if (
        Math.max(
          getSeparation(
            {x: middle.x + epsilon * outwardX, y: middle.y + epsilon * outwardY},
            line,
            owner,
            outerJoin
          ),
          getSeparation(
            {x: middle.x - epsilon * outwardX, y: middle.y - epsilon * outwardY},
            line,
            owner,
            outerJoin
          )
        ) <=
        distance + epsilon * 0.05
      )
        continue;
      fragments.push({
        start: {x: start.x + first * directionX, y: start.y + first * directionY},
        end: {x: start.x + last * directionX, y: start.y + last * directionY},
        line,
        source: owner,
        state: 1
      });
    }
  }
  const matchTolerance = Math.max(1e-5, distance * 1e-5);
  const matches = (first: Point, second: Point) =>
    Math.hypot(first.x - second.x, first.y - second.y) <= matchTolerance;
  for (let candidate = 0; candidate < fragments.length; candidate++) {
    const first = fragments[candidate];
    if (first.state !== 1) continue;
    for (let other = candidate + 1; other < fragments.length; other++) {
      const second = fragments[other];
      if (second.state !== 1 || second.line !== first.line) continue;
      if (matches(first.start, second.end) && matches(first.end, second.start)) {
        first.state = 0;
        second.state = 0;
        break;
      }
      if (matches(first.start, second.start) && matches(first.end, second.end)) second.state = 0;
    }
  }
  const result: CPUOffsetCurveComponentsResult = {
    positions: new Float32Array(options.vertexCapacity * 2),
    components: new Uint32Array(options.componentCapacity * 4),
    sourceEdges: new Uint32Array(options.vertexCapacity),
    status: new Uint32Array(8)
  };
  let componentCount = 0;
  let vertexCount = 0;
  let writtenComponents = 0;
  let writtenVertices = 0;
  const writeVertex = (point: Point | CPURefinementPoint, source: number) => {
    if (vertexCount < options.vertexCapacity) {
      result.positions.set([point.x, point.y], vertexCount * 2);
      result.sourceEdges[vertexCount] = source;
    }
    vertexCount++;
  };
  while (true) {
    let seed: Fragment | undefined;
    for (const candidate of fragments) {
      if (candidate.state !== 1) continue;
      seed ??= candidate;
      if (
        !fragments.some(
          other =>
            other !== candidate &&
            other.state === 1 &&
            other.line === candidate.line &&
            matches(other.end, candidate.start)
        )
      ) {
        seed = candidate;
        break;
      }
    }
    if (!seed) break;
    const componentStart = vertexCount;
    writeVertex(seed.start, seed.source);
    let fragment: Fragment | undefined = seed;
    while (fragment) {
      fragment.state = 2;
      const finish: Point = fragment.end;
      writeVertex(finish, fragment.source);
      fragment = fragments.find(
        candidate =>
          candidate.state === 1 && candidate.line === seed.line && matches(finish, candidate.start)
      );
    }
    if (componentCount < options.componentCapacity && vertexCount <= options.vertexCapacity) {
      result.components.set(
        [componentStart, vertexCount - componentStart, seed.line, 0],
        componentCount * 4
      );
      writtenComponents++;
      writtenVertices = vertexCount;
    }
    componentCount++;
  }
  result.status.set([
    writtenComponents,
    writtenVertices,
    componentCount,
    vertexCount,
    Number(componentCount > options.componentCapacity || vertexCount > options.vertexCapacity),
    noded.incomplete,
    0,
    0
  ]);
  return result;
}

/** CPU twin: complete noding without GPU intermediate capacity or binding limits. */
export class CPUOffsetCurveComponentsOperation extends GPUCPUOperation {
  constructor(props: OffsetCurveComponentsOperationProps) {
    const id = props.id ?? 'offset-curve-components';
    validateOptions(id, props);
    validateCapacity(id, 'lineCount', props.lineCount);
    validateCapacity(id, 'intersectionCapacity', props.intersectionCapacity);
    validateCapacity(id, 'segmentCapacity', props.segmentCapacity);
    const operation = new GPUOffsetCurveComponentsOperation(props);
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
        const originalOffsets = getCPUOperationUint32Rows(id, inputs, 'originalOffsets');
        const rawOffsets = getCPUOperationUint32Rows(id, inputs, 'rawOffsets');
        if (
          originalOffsets.length !== props.lineCount + 1 ||
          rawOffsets.length !== props.lineCount + 1
        )
          throw new Error(`${id} original/raw lineOffsets must have lineCount + 1 rows`);
        const result = computeOffsetCurveComponentsOnCPU(
          {
            original: {
              positions: getCPUOperationFloat32Rows(id, inputs, 'originalPositions', 'float32x2'),
              lineOffsets: originalOffsets
            },
            raw: {
              positions: getCPUOperationFloat32Rows(id, inputs, 'rawPositions', 'float32x2'),
              lineOffsets: rawOffsets
            }
          },
          props
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUOffsetCurveComponentsResult]
          ])
        );
      }
    });
  }
}
