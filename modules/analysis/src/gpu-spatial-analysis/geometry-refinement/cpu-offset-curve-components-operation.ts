// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
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
  original: {positions: Float32Array; lineOffsets: Uint32Array; featureOffsets?: Uint32Array};
  raw: {positions: Float32Array; lineOffsets: Uint32Array; featureOffsets?: Uint32Array};
};

/** Output dimensions, geometric parameters and optional noding budgets (uncapped when omitted). */
export type CPUOffsetCurveComponentsOptions = Pick<
  OffsetCurveComponentsOperationProps,
  'distance' | 'tolerance' | 'rowsPerVertex' | 'vertexCapacity' | 'componentCapacity'
> &
  Partial<Pick<OffsetCurveComponentsOperationProps, 'intersectionCapacity' | 'segmentCapacity'>>;

/** All graph outputs, including partial unreferenced vertex prefixes and zeroed unused rows. */
export type CPUOffsetCurveComponentsResult = {
  positions: Float32Array;
  components: Uint32Array;
  sourceEdges: Uint32Array;
  /** [writtenComponents,writtenVertices,requiredComponents,requiredVertices,overflow,incompleteNoding,iterationLimit,0]. */
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
  if (options.intersectionCapacity !== undefined)
    validateCapacity(id, 'intersectionCapacity', options.intersectionCapacity);
  if (options.segmentCapacity !== undefined)
    validateCapacity(id, 'segmentCapacity', options.segmentCapacity);
}

/**
 * Nodes raw planar offsets, trims their source-distance bands, cancels duplicate/opposite fragments,
 * and chains components in the contributor's seed/successor order. Preserves its geometric slack,
 * first-match ties, whole-component metadata publication and partial writes.
 *
 * What differs from the GPU contributor:
 * - Arithmetic is float64, with fragment endpoints captured as float32; distance-band and cut decisions
 *   near tolerance boundaries can differ.
 * - Noding uses exact rational predicates/visit keys on float32 input edges. Optional segmentCapacity
 *   retains a traversal-order prefix; optional intersectionCapacity suppresses geometry when its
 *   pair list is exhausted, since the GPU's spatial-index-dependent partial prefix is not replayed.
 *   Either loss sets incompleteNoding=1 without setting final-output overflow. Omitted budgets are
 *   uncapped. No uncertain-predicate status, 1024-successor scan or GPU binding limit is imposed.
 * - Intersections remain float64 through cut construction; fragment endpoints are rounded before
 *   topology assembly to match their published representation.
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
  const featureOffsets = input.original.featureOffsets;
  const rawFeatureOffsets = input.raw.featureOffsets;
  if (featureOffsets || rawFeatureOffsets) {
    if (
      !featureOffsets ||
      !rawFeatureOffsets ||
      featureOffsets.length !== rawFeatureOffsets.length ||
      featureOffsets.length < 1
    )
      throw new Error(
        'offset-curve-components original.featureOffsets and raw.featureOffsets must have identical grouping; supply both arrays'
      );
    if (
      featureOffsets[0] !== 0 ||
      featureOffsets[featureOffsets.length - 1] !== input.original.lineOffsets.length - 1
    )
      throw new Error(
        'offset-curve-components original.featureOffsets must span every source line part'
      );
    for (let row = 0; row < featureOffsets.length; row++) {
      if (
        featureOffsets[row] !== rawFeatureOffsets[row] ||
        (row > 0 && featureOffsets[row] < featureOffsets[row - 1])
      )
        throw new Error(
          'offset-curve-components original.featureOffsets and raw.featureOffsets must be identical nondecreasing feature-to-part offsets'
        );
    }
  }
  const getFeatureRow = (line: number): number => {
    if (!featureOffsets) return 0;
    let low = 0;
    let high = featureOffsets.length - 1;
    while (high - low > 1) {
      const middle = Math.floor((low + high) / 2);
      if (featureOffsets[middle] <= line) low = middle;
      else high = middle;
    }
    return low;
  };
  const noded = nodeCPURefinementLines(input.raw.positions, input.raw.lineOffsets, {
    intersectionCapacity: options.intersectionCapacity
  });
  const segments =
    options.segmentCapacity === undefined
      ? noded.segments
      : noded.segments.slice(0, options.segmentCapacity);
  const incompleteNoding = Number(
    Boolean(noded.incomplete) || segments.length < noded.segments.length
  );
  const distance = Math.abs(options.distance);
  const tolerance = options.tolerance;
  const rowsPerVertex = options.rowsPerVertex ?? 1;
  const pointAt = (row: number): Point => ({
    x: input.original.positions[row * 2],
    y: input.original.positions[row * 2 + 1]
  });
  const offsets = input.original.lineOffsets;
  const getSeparation = (
    query: Point,
    line: number,
    owner: number,
    outerJoin: boolean,
    roundingSign: number
  ): number => {
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
      const residualX = query.x - start.x - fraction * directionX;
      const residualY = query.y - start.y - fraction * directionY;
      const separation = Math.hypot(residualX, residualY);
      const length = Math.sqrt(denominator);
      const normalX =
        fraction > 0 && fraction < 1
          ? -directionY / length
          : separation > 0
            ? residualX / separation
            : 0;
      const normalY =
        fraction > 0 && fraction < 1
          ? directionX / length
          : separation > 0
            ? residualY / separation
            : 0;
      // Coordinate rounding matters only along this source-distance measurement direction.
      const roundingFloor =
        1.1920928955078125e-7 *
        (distance + Math.abs(query.x * normalX) + Math.abs(query.y * normalY));
      minimum = Math.min(minimum, separation + roundingSign * roundingFloor);
    }
    return minimum;
  };
  const fragments: Fragment[] = [];
  for (const segment of segments) {
    const {start, end, line} = segment;
    const directionX = end.x - start.x;
    const directionY = end.y - start.y;
    const squaredLength = directionX ** 2 + directionY ** 2;
    if (squaredLength === 0) continue;
    // Traversal identity survives cancellation: a later coincident edge must not
    // inherit the earliest edge's row merely because its geometry is identical.
    const rawEdge = segment.source;
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
    const cuts: {fraction: number; point: Point}[] = [
      {fraction: 0, point: start},
      {fraction: 1, point: end}
    ];
    const cutFractions = new Set([0, 1]);
    const insertCut = (fraction: number, point: Point) => {
      if (fraction > 0 && fraction < 1 && !cutFractions.has(fraction)) {
        cutFractions.add(fraction);
        cuts.push({fraction, point});
      }
    };
    for (let vertex = offsets[line]; vertex + 1 < offsets[line + 1]; vertex++) {
      if (outerJoin && (vertex === owner || vertex + 1 === owner)) continue;
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
          const point = {x: start.x + fraction * directionX, y: start.y + fraction * directionY};
          if (normalX === 0) point.y = first.y + side * normalY;
          if (normalY === 0) point.x = first.x + side * normalX;
          const projection =
            ((point.x - first.x) * edgeX + (point.y - first.y) * edgeY) / lengthSquared;
          if (projection >= 0 && projection <= 1) insertCut(fraction, point);
        }
      }
      for (const endpoint of [first, last]) {
        const deltaX = start.x - endpoint.x;
        const deltaY = start.y - endpoint.y;
        const endDeltaX = end.x - endpoint.x;
        const endDeltaY = end.y - endpoint.y;
        // A circle tangent at an existing endpoint adds no interior cut. Construct this tie
        // before normalizing the direction: reciprocal division can otherwise split the tie.
        if (
          (deltaX * directionX + deltaY * directionY === 0 &&
            deltaX ** 2 + deltaY ** 2 === distance ** 2) ||
          (endDeltaX * directionX + endDeltaY * directionY === 0 &&
            endDeltaX ** 2 + endDeltaY ** 2 === distance ** 2)
        )
          continue;
        const segmentLength = Math.sqrt(squaredLength);
        const unitX = directionX / segmentLength;
        const unitY = directionY / segmentLength;
        const along = deltaX * unitX + deltaY * unitY;
        const across = deltaX * unitY - deltaY * unitX;
        const discriminant = (distance - Math.abs(across)) * (distance + Math.abs(across));
        if (discriminant >= 0) {
          const root = Math.sqrt(discriminant);
          const foot = {x: endpoint.x + across * unitY, y: endpoint.y - across * unitX};
          insertCut((-along - root) / segmentLength, {
            x: foot.x - root * unitX,
            y: foot.y - root * unitY
          });
          insertCut((-along + root) / segmentLength, {
            x: foot.x + root * unitX,
            y: foot.y + root * unitY
          });
        }
      }
    }
    cuts.sort((left, right) => left.fraction - right.fraction);
    const outwardX = -directionY / Math.sqrt(squaredLength);
    const outwardY = directionX / Math.sqrt(squaredLength);
    for (let slot = 0; slot + 1 < cuts.length; slot++) {
      const firstCut = cuts[slot];
      const lastCut = cuts[slot + 1];
      const first = firstCut.fraction;
      const last = lastCut.fraction;
      if (last <= first) continue;
      // Topology is published as float32: distinct cut parameters can describe one vertex.
      const publishedStart = {x: Math.fround(firstCut.point.x), y: Math.fround(firstCut.point.y)};
      const publishedEnd = {x: Math.fround(lastCut.point.x), y: Math.fround(lastCut.point.y)};
      if (publishedStart.x === publishedEnd.x && publishedStart.y === publishedEnd.y) continue;
      const middle = {
        x: start.x + (first + last) * 0.5 * directionX,
        y: start.y + (first + last) * 0.5 * directionY
      };
      const probeFloor =
        1.1920928955078125e-7 *
        Math.max(distance, Math.abs(middle.x * outwardX), Math.abs(middle.y * outwardY));
      const epsilon = Math.max(probeFloor * 4, 1e-5, tolerance * 2, distance * 1e-5);
      if (getSeparation(middle, line, owner, outerJoin, 1) + tolerance < distance) continue;
      const positiveProbe = getSeparation(
        {x: middle.x + epsilon * outwardX, y: middle.y + epsilon * outwardY},
        line,
        owner,
        outerJoin,
        -1
      );
      const negativeProbe = getSeparation(
        {x: middle.x - epsilon * outwardX, y: middle.y - epsilon * outwardY},
        line,
        owner,
        outerJoin,
        -1
      );
      if (
        Math.hypot(publishedEnd.x - publishedStart.x, publishedEnd.y - publishedStart.y) >
          epsilon &&
        Math.max(positiveProbe, negativeProbe) <= distance + epsilon * 0.05
      )
        continue;
      fragments.push({
        start: publishedStart,
        end: publishedEnd,
        line,
        source: owner,
        state: 1
      });
    }
  }
  // Shared noded/clipped coordinates identify endpoints. Proximity cannot establish topology:
  // long nearby legs and gaps removed by clipping must remain distinct at every scale.
  const matches = (first: Point, second: Point) => first.x === second.x && first.y === second.y;
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
  let consumed = 0;
  let iterationLimit = 0;
  for (
    let componentRound = 0;
    componentRound < fragments.length && consumed < fragments.length;
    componentRound++
  ) {
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
    for (; fragment && consumed < fragments.length; consumed++) {
      fragment.state = 2;
      const finish: Point = fragment.end;
      writeVertex(finish, fragment.source);
      fragment = fragments.find(
        candidate =>
          candidate.state === 1 && candidate.line === seed.line && matches(finish, candidate.start)
      );
    }
    if (fragment) {
      iterationLimit = 1;
      break;
    }
    if (componentCount < options.componentCapacity && vertexCount <= options.vertexCapacity) {
      result.components.set(
        [componentStart, vertexCount - componentStart, seed.line, getFeatureRow(seed.line)],
        componentCount * 4
      );
      writtenComponents++;
      writtenVertices = vertexCount;
    }
    componentCount++;
  }
  if (fragments.some(fragment => fragment.state === 1)) iterationLimit = 1;
  result.status.set([
    iterationLimit ? 0 : writtenComponents,
    iterationLimit ? 0 : writtenVertices,
    componentCount,
    vertexCount,
    Number(componentCount > options.componentCapacity || vertexCount > options.vertexCapacity),
    incompleteNoding,
    iterationLimit,
    0
  ]);
  return result;
}

/** CPU twin: honours noding budgets; pair exhaustion suppresses geometry with incompleteNoding=1. */
export class CPUOffsetCurveComponentsOperation extends GPUCPUOperation {
  constructor(props: OffsetCurveComponentsOperationProps) {
    const snapshot = snapshotConstructionFields(props, [
      'id',
      'original',
      'raw',
      'lineCount',
      'distance',
      'tolerance',
      'rowsPerVertex',
      'intersectionCapacity',
      'segmentCapacity',
      'vertexCapacity',
      'componentCapacity',
      'output'
    ]);
    props = Object.freeze({
      ...snapshot,
      original: snapshotConstructionFields(snapshot.original, [
        'positions',
        'lineOffsets',
        'featureOffsets'
      ]),
      raw: snapshotConstructionFields(snapshot.raw, ['positions', 'lineOffsets', 'featureOffsets']),
      output: snapshotConstructionFields(snapshot.output, [
        'positions',
        'components',
        'sourceEdges',
        'status'
      ])
    });
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
              lineOffsets: originalOffsets,
              featureOffsets: props.original.featureOffsets
                ? getCPUOperationUint32Rows(id, inputs, 'originalFeatures')
                : undefined
            },
            raw: {
              positions: getCPUOperationFloat32Rows(id, inputs, 'rawPositions', 'float32x2'),
              lineOffsets: rawOffsets,
              featureOffsets: props.raw.featureOffsets
                ? getCPUOperationUint32Rows(id, inputs, 'rawFeatures')
                : undefined
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
