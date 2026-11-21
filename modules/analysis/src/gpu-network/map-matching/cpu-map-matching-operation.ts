// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {snapshotMapMatchingConstructionProps} from './map-matching-construction';
import {
  getGPUGreatCircleRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  computeCosLatitude,
  computeLongitudeDelta,
  wrapLongitude
} from '../../analysis-core/kernels/cpu-geodesic';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {getCPUTrajectoryDistance} from '../../analysis-core/operations/cpu-trajectory-distance';
import {GPU_MAP_MATCHING_NONE, GPU_MAP_MATCHING_PARAMETER_LENGTH} from './gpu-map-matching';
import {
  GPUMapMatchingOperation,
  type MapMatchingOperationProps
} from './gpu-map-matching-operation';

const NONE = GPU_MAP_MATCHING_NONE;
const DEGREES_TO_RADIANS = Math.PI / 180;
// Preserve the GPU's finite sentinel and strict validity boundary, including their f32 values.
const INVALID_SCORE = Math.fround(-1e30);
const MINIMUM_VALID_SCORE = Math.fround(-1e29);

/** Packed input rows, including the optional GPU acceleration table and unused network weights. */
export type CPUMapMatchingInput = {
  positions: Float32Array;
  trackOffsets: Uint32Array;
  parameters: Float32Array;
  topology: {
    offsets: Uint32Array;
    neighbors: Uint32Array;
    nodePositions: Float32Array;
    weights?: Float32Array;
  };
  routeTable?: {
    offsets: Uint32Array;
    targets: Uint32Array;
    costs: Float32Array;
    /** Current legacy horizon; captured builtCostLimit takes precedence when supplied. */
    costLimit: Float32Array;
    builtCostLimit?: Float32Array;
    complete?: Uint32Array;
  };
};

/** Compile-time model choices. Explicit route budgets bind the CPU fallback too. */
export type CPUMapMatchingOptions = {
  /** Explicit private route capacity, as on GPU; omitted keeps CPU fallback searches uncapped. */
  routeNodeBudget?: number;
  spatialContext?: GPUSpatialContext;
  cellSize: number;
  candidateCount?: number;
  minimumFixSpacingSigmas?: number;
  /** Omitted or zero disables traversed output in the pure function. */
  traversedEdgeCapacity?: number;
};

/** Contributor output layouts; traversedCounts is the UNCLAMPED requirement for each track. */
export type CPUMapMatchingResult = {
  matchedEdges: Uint32Array;
  matchedFractions: Float32Array;
  matchedOffsets: Float32Array;
  snapDistances: Float32Array;
  snappedPositions: Float32Array;
  breaks: Uint32Array;
  trackLogLikelihoods: Float32Array;
  matchedCount: Uint32Array;
  breakCount: Uint32Array;
  overflow: Uint32Array;
  candidateOverflow: Uint32Array;
  routeOverflow: Uint32Array;
  traversedEdges: Uint32Array;
  traversedCounts: Uint32Array;
  traversedOverflow: Uint32Array;
  traversedCandidateOverflow: Uint32Array;
};

type Candidate = {edge: number; fraction: number; distance: number};
type SearchEntry = {
  node: number;
  distance: number;
  key: number;
  order: number;
  parent: number;
  done: boolean;
};

function resolveContext(id: string, context?: GPUSpatialContext): GPUSpatialContext {
  return resolveGPUSpatialContext(id, context, {
    ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
    measure: 'distance'
  });
}

function validateOptions(id: string, options: CPUMapMatchingOptions): void {
  if (options.routeNodeBudget !== undefined)
    validateCapacity(id, 'routeNodeBudget', options.routeNodeBudget, 1);
  validateCapacity(id, 'candidateCount', options.candidateCount ?? 8);
  validateFiniteNumber(id, 'cellSize', options.cellSize);
  validateRequirement(id, options.cellSize > 0, 'cellSize must be positive');
  validateFiniteNumber(id, 'minimumFixSpacingSigmas', options.minimumFixSpacingSigmas ?? 0, 0);
  validateRequirement(
    id,
    (options.minimumFixSpacingSigmas ?? 0) >= 0,
    'minimumFixSpacingSigmas must be non-negative'
  );
  resolveContext(id, options.spatialContext);
}

function validateOffsets(name: string, offsets: Uint32Array, maximum?: number): void {
  if (
    offsets.length < 2 ||
    offsets.some(
      (value, row) =>
        (row > 0 && value < offsets[row - 1]) || (maximum !== undefined && value > maximum)
    )
  )
    throw new Error(
      `${name} must be nondecreasing offsets${maximum === undefined ? '' : ` within [0, ${maximum}]`} with at least two rows`
    );
}

/**
 * A*, preserving the contributor's discovery-order queue ties and strict parent
 * improvements. A single target is used again when recovering traversed edges, just as on GPU.
 * The generic CPU routing helper takes float32 weights and has different parent ties, so cannot
 * represent these float64 geometric edge lengths and this route's deliberate tie definition.
 */
function searchRoute(
  start: number,
  targets: readonly number[],
  limit: number,
  offsets: Uint32Array,
  neighbors: Uint32Array,
  lengths: Float64Array,
  nodePositions: Float32Array,
  radius: number | undefined,
  routeNodeBudget = Infinity
): Map<number, SearchEntry> & {droppedCount: number} {
  const entries = Object.assign(new Map<number, SearchEntry>(), {droppedCount: 0});
  const pending = new Set(targets);
  const heap: SearchEntry[] = [];
  const precedes = (first: SearchEntry, second: SearchEntry) =>
    first.key < second.key || (first.key === second.key && first.order < second.order);
  const pushEntry = (entry: SearchEntry) => {
    let position = heap.length;
    heap.push(entry);
    while (position > 0) {
      const parent = Math.floor((position - 1) / 2);
      if (!precedes(entry, heap[parent])) break;
      heap[position] = heap[parent];
      position = parent;
    }
    heap[position] = entry;
  };
  const heuristic = (node: number) => {
    let nearest = Infinity;
    for (const target of targets)
      nearest = Math.min(nearest, getCPUTrajectoryDistance(nodePositions, node, target, radius));
    return targets.length ? nearest * 0.999999 : 0;
  };
  const initial = {
    node: start,
    distance: 0,
    key: heuristic(start),
    order: 0,
    parent: NONE,
    done: false
  };
  entries.set(start, initial);
  pushEntry(initial);
  while (heap.length) {
    const entry = heap[0];
    const last = heap.pop()!;
    if (heap.length) {
      let position = 0;
      while (position * 2 + 1 < heap.length) {
        let child = position * 2 + 1;
        if (child + 1 < heap.length && precedes(heap[child + 1], heap[child])) child++;
        if (!precedes(heap[child], last)) break;
        heap[position] = heap[child];
        position = child;
      }
      heap[position] = last;
    }
    if (entries.get(entry.node) !== entry || entry.done) continue;
    if (entry.key > limit) break;
    entry.done = true;
    pending.delete(entry.node);
    if (!pending.size) break;
    for (
      let edge = offsets[entry.node];
      edge < Math.min(offsets[entry.node + 1], neighbors.length);
      edge++
    ) {
      if (!(lengths[edge] >= 0)) continue;
      const distance = entry.distance + lengths[edge];
      if (distance > limit) continue;
      const node = neighbors[edge];
      const previous = entries.get(node);
      if (previous?.done || distance >= (previous?.distance ?? Infinity)) continue;
      const next = {
        node,
        distance,
        key: distance + heuristic(node),
        order: previous?.order ?? entries.size,
        parent: edge,
        done: false
      };
      if (!previous && entries.size >= routeNodeBudget) {
        entries.droppedCount++;
        let worst: SearchEntry | undefined;
        for (const candidate of entries.values()) {
          if (
            !candidate.done &&
            (!worst ||
              candidate.key > worst.key ||
              (candidate.key === worst.key && candidate.order > worst.order))
          )
            worst = candidate;
        }
        if (!worst || next.key >= worst.key) continue;
        entries.delete(worst.node);
        next.order = worst.order;
      }
      entries.set(node, next);
      pushEntry(next);
    }
  }
  return entries;
}

/**
 * Newson-Krumm matching with exact geometric candidates, optionally bounded A* and per-track
 * Viterbi decoding. Distances and likelihoods remain float64 until output. Equal candidate
 * distances prefer the lowest edge row; equal Viterbi scores prefer the earliest candidate.
 * The finite score sentinel and cutoff of the contributor are retained: only scores strictly
 * above its float32 -1e29 boundary are normalized and eligible for backward selection.
 *
 * What differs from the GPU contributor:
 * - Float64 intermediates and one float32 output rounding replace float32 arithmetic. Near ties,
 *   radius/bound thresholds and extreme magnitudes can change selected paths. Compare ordinary
 *   metric outputs with relative tolerance 5e-5 (absolute 2e-5 near zero); integers compare exactly.
 *   Geographic snapped positions are compared in spatial units using endpoint/output float32
 *   ULPs and a segment-length projection-error allowance, not absolute longitude magnitudes.
 * - All valid edges are searched, without the grid's 16-cell span or entryCapacity truncation.
 *   The searchRadius clamp to twice the cell size in metric units is retained. CPU candidateCount
 *   may exceed 8, but the requested candidateCount and default 8 still define the model.
 * - An explicit routeNodeBudget limits fallback and recovery searches as on GPU, including
 *   candidateOverflow, routeOverflow and budget breaks (2). Omitted, CPU searches remain uncapped.
 *   A complete routeTable within its cost limit certifies scoring transitions without a search.
 *   Grid candidate loss is absent on CPU. Fixed matched outputs always have overflow=0.
 * - Traversed counts remain unclamped, as on GPU; traversedOverflow reports output truncation
 *   only; traversedCandidateOverflow reports incomplete recovery and inserts a NONE separator
 *   before the destination edge. Unwritten slots are zero on CPU (GPU leaves them untouched).
 *   This inherited layout is a section 8 contract departure, not compact output: counts is the
 *   per-track requirement and no separate requiredCount exists. Correcting it needs a coordinated
 *   contributor/twin API change; clamping only the twin would violate the same-graph-values rule.
 * - Malformed packed lengths, decreasing CSR offsets and out-of-range track offsets throw.
 *   Monotonic network offsets are clamped to edge capacity as on GPU; invalid targets and unused
 *   edge slots are ignored. Points outside the track ranges are deterministically unmatched.
 * - Nonfinite coordinates or parameters throw; their GPU results are undefined. Subnormal and
 *   very large finite inputs are evaluated in float64; final float32 outputs may overflow.
 */
export function computeMapMatchingOnCPU(
  input: CPUMapMatchingInput,
  options: CPUMapMatchingOptions
): CPUMapMatchingResult {
  options = snapshotMapMatchingConstructionProps(options);
  validateOptions('computeMapMatchingOnCPU', options);
  if (options.traversedEdgeCapacity !== undefined)
    validateCapacity(
      'computeMapMatchingOnCPU',
      'traversedEdgeCapacity',
      options.traversedEdgeCapacity,
      0
    );
  const {positions, trackOffsets, parameters, topology} = input;
  const {offsets, neighbors, nodePositions, weights} = topology;
  if (!positions.length || positions.length % 2)
    throw new Error('positions must contain nonempty float32x2 rows');
  validateOffsets('topology.offsets', offsets);
  const nodeCount = offsets.length - 1;
  const edgeCount = neighbors.length;
  const pointCount = positions.length / 2;
  if (!edgeCount) throw new Error('topology.neighbors must contain at least one edge slot');
  if (nodePositions.length !== nodeCount * 2)
    throw new Error('topology.nodePositions must contain one float32x2 row per node');
  if (weights && weights.length !== edgeCount)
    throw new Error('topology.weights length must equal topology.neighbors length');
  validateOffsets('trackOffsets', trackOffsets, pointCount);
  if (parameters.length !== GPU_MAP_MATCHING_PARAMETER_LENGTH)
    throw new Error(`parameters must contain ${GPU_MAP_MATCHING_PARAMETER_LENGTH} words`);
  // Packed layout of getGPUMapMatchingParameterValues; the contributor exposes a length constant
  // but no individual offset constants. Destructure its five settings, leaving padding unused.
  const [sigmaValue, betaValue, searchRadius, routeFactor, routeSlack] = parameters;
  if (![sigmaValue, betaValue, searchRadius, routeFactor, routeSlack].every(Number.isFinite))
    throw new Error('parameters settings must be finite');
  if (!positions.every(Number.isFinite)) throw new Error('positions coordinates must be finite');
  if (!nodePositions.every(Number.isFinite))
    throw new Error('topology.nodePositions coordinates must be finite');
  const tableCostLimit = input.routeTable?.builtCostLimit ?? input.routeTable?.costLimit;
  if (input.routeTable) {
    const table = input.routeTable;
    if (table.offsets.length !== nodeCount + 1)
      throw new Error('routeTable.offsets length must equal nodeCount + 1');
    if (table.costs.length !== table.targets.length)
      throw new Error('routeTable.costs length must equal routeTable.targets length');
    validateOffsets('routeTable.offsets', table.offsets, table.targets.length);
    if (table.costLimit.length !== 1) throw new Error('routeTable.costLimit must contain one row');
    if (table.builtCostLimit)
      validateRequirement(
        'computeMapMatchingOnCPU',
        table.builtCostLimit.length === 1,
        'routeTable.builtCostLimit must contain one row'
      );
    if (table.complete && table.complete.length !== 1)
      throw new Error('routeTable.complete must contain one row');
  }
  const radius = getGPUGreatCircleRadius(
    resolveContext('computeMapMatchingOnCPU', options.spatialContext)
  );
  const maximumRadius =
    2 * options.cellSize * (radius === undefined ? 1 : DEGREES_TO_RADIANS * radius);
  const candidateRadius = Math.min(searchRadius, maximumRadius);
  const sigma = Math.max(sigmaValue, 1e-6);
  const beta = Math.max(betaValue, 1e-6);
  const candidateCount = options.candidateCount ?? 8;
  const trackCount = trackOffsets.length - 1;
  const capacity = options.traversedEdgeCapacity ?? 0;
  const sources = new Uint32Array(edgeCount).fill(NONE);
  const lengths = new Float64Array(edgeCount).fill(-1);
  for (let source = 0; source < nodeCount; source++) {
    for (let edge = offsets[source]; edge < Math.min(offsets[source + 1], edgeCount); edge++) {
      sources[edge] = source;
      if (neighbors[edge] < nodeCount)
        lengths[edge] = getCPUTrajectoryDistance(nodePositions, source, neighbors[edge], radius);
    }
  }
  const candidates: Candidate[][] = [];
  for (let point = 0; point < pointCount; point++) {
    const pointLongitude = positions[point * 2];
    const pointLatitude = positions[point * 2 + 1];
    const found: Candidate[] = [];
    for (let edge = 0; edge < edgeCount; edge++) {
      if (!(lengths[edge] >= 0)) continue;
      const startLongitude = nodePositions[sources[edge] * 2];
      const startLatitude = nodePositions[sources[edge] * 2 + 1];
      const endLongitude = nodePositions[neighbors[edge] * 2];
      const endLatitude = nodePositions[neighbors[edge] * 2 + 1];
      const cosine = radius === undefined ? 1 : computeCosLatitude(pointLatitude);
      const startEast =
        (radius === undefined
          ? startLongitude - pointLongitude
          : computeLongitudeDelta(pointLongitude, startLongitude)) * cosine;
      const startNorth = startLatitude - pointLatitude;
      const segmentEast =
        radius === undefined
          ? endLongitude - startLongitude
          : computeLongitudeDelta(startLongitude, endLongitude) * cosine;
      const segmentNorth = endLatitude - startLatitude;
      const denominator = segmentEast ** 2 + segmentNorth ** 2;
      let fraction =
        denominator > 0
          ? Math.max(
              0,
              Math.min(1, -(startEast * segmentEast + startNorth * segmentNorth) / denominator)
            )
          : 0;
      let distance: number;
      if (radius === undefined) {
        distance = Math.hypot(
          pointLongitude - (startLongitude + fraction * segmentEast),
          pointLatitude - (startLatitude + fraction * segmentNorth)
        );
      } else {
        const longitudeOffset =
          computeLongitudeDelta(pointLongitude, startLongitude) +
          fraction * computeLongitudeDelta(startLongitude, endLongitude);
        const latitudeOffset = startNorth + fraction * segmentNorth;
        const term =
          Math.sin(0.5 * latitudeOffset * DEGREES_TO_RADIANS) ** 2 +
          computeCosLatitude(pointLatitude) *
            computeCosLatitude(pointLatitude + latitudeOffset) *
            Math.sin(0.5 * longitudeOffset * DEGREES_TO_RADIANS) ** 2;
        distance = 2 * radius * Math.asin(Math.sqrt(Math.max(0, Math.min(1, term))));
        const startDistance = getCPUTrajectoryDistance(
          [pointLongitude, pointLatitude, startLongitude, startLatitude],
          0,
          1,
          radius
        );
        const endDistance = getCPUTrajectoryDistance(
          [pointLongitude, pointLatitude, endLongitude, endLatitude],
          0,
          1,
          radius
        );
        const endpointDistance = Math.min(startDistance, endDistance);
        if (fraction === 0 || fraction === 1 || endpointDistance < distance) {
          fraction = startDistance <= endDistance ? 0 : 1;
          distance = endpointDistance;
        }
      }
      if (distance <= candidateRadius) found.push({edge, fraction, distance});
    }
    found.sort((first, second) => first.distance - second.distance || first.edge - second.edge);
    candidates.push(found.slice(0, candidateCount));
  }
  const result: CPUMapMatchingResult = {
    matchedEdges: new Uint32Array(pointCount).fill(NONE),
    matchedFractions: new Float32Array(pointCount).fill(-1),
    matchedOffsets: new Float32Array(pointCount).fill(-1),
    snapDistances: new Float32Array(pointCount).fill(-1),
    snappedPositions: positions.slice(),
    breaks: new Uint32Array(pointCount),
    trackLogLikelihoods: new Float32Array(trackCount),
    matchedCount: new Uint32Array(1),
    breakCount: new Uint32Array(1),
    overflow: new Uint32Array(1),
    candidateOverflow: new Uint32Array(1),
    routeOverflow: new Uint32Array(1),
    traversedEdges: new Uint32Array(trackCount * capacity),
    traversedCounts: new Uint32Array(trackCount),
    traversedOverflow: new Uint32Array(1),
    traversedCandidateOverflow: new Uint32Array(1)
  };
  const chosen: (Candidate | undefined)[] = new Array(pointCount);
  const budgetLimited = new Uint8Array(pointCount);
  const predecessors = new Float64Array(pointCount).fill(-1);
  const emissionConstant = -Math.log(sigma) - 0.5 * Math.log(2 * Math.PI);
  for (let track = 0; track < trackCount; track++) {
    const retained: number[] = [];
    const scores: number[][] = [];
    const backPointers: number[][] = [];
    const scoreOffsets: number[] = [];
    for (let point = trackOffsets[track]; point < trackOffsets[track + 1]; point++) {
      const previous = retained.length ? retained[retained.length - 1] : -1;
      if (
        previous >= 0 &&
        getCPUTrajectoryDistance(positions, previous, point, radius) <
          sigma * (options.minimumFixSpacingSigmas ?? 0)
      )
        continue;
      predecessors[point] = previous;
      const here = candidates[point];
      const score = here.map(() => INVALID_SCORE);
      const back = here.map(() => -1);
      let reachable = false;
      if (previous >= 0) {
        const straight = getCPUTrajectoryDistance(positions, point, previous, radius);
        const bound = straight * routeFactor + routeSlack;
        candidates[previous].forEach((from, fromSlot) => {
          const fromScore = scores[scores.length - 1][fromSlot];
          if (fromScore < MINIMUM_VALID_SCORE) return;
          const tail = (1 - from.fraction) * lengths[from.edge];
          const remaining = bound - tail;
          const targets = here
            .filter(to => !(to.edge === from.edge && to.fraction >= from.fraction))
            .map(to => sources[to.edge]);
          const search =
            remaining >= 0 && !(input.routeTable?.complete?.[0] && remaining <= tableCostLimit![0])
              ? searchRoute(
                  neighbors[from.edge],
                  targets,
                  remaining,
                  offsets,
                  neighbors,
                  lengths,
                  nodePositions,
                  radius,
                  options.routeNodeBudget
                )
              : undefined;
          if (search?.droppedCount) {
            budgetLimited[point] = 1;
            result.candidateOverflow[0] = 1;
            result.routeOverflow[0] = Math.min(
              0xffffffff,
              result.routeOverflow[0] + search.droppedCount
            );
          }
          const certified =
            remaining >= 0 &&
            Boolean(input.routeTable?.complete?.[0]) &&
            remaining <= tableCostLimit![0];
          here.forEach((to, toSlot) => {
            const network = search?.get(sources[to.edge]);
            const table = input.routeTable;
            let entry = -1;
            if (certified) {
              const target = sources[to.edge];
              const end = table!.offsets[neighbors[from.edge] + 1];
              let low = table!.offsets[neighbors[from.edge]];
              let high = end;
              while (low < high) {
                const middle = low + Math.floor((high - low) / 2);
                if (table!.targets[middle] < target) low = middle + 1;
                else high = middle;
              }
              if (low < end && table!.targets[low] === target) entry = low;
            }
            const tableCost =
              entry >= 0 && entry < table!.offsets[neighbors[from.edge] + 1]
                ? table!.costs[entry]
                : -1;
            const route =
              to.edge === from.edge && to.fraction >= from.fraction
                ? (to.fraction - from.fraction) * lengths[from.edge]
                : certified
                  ? tableCost >= 0 && tableCost <= remaining
                    ? tail + tableCost + to.fraction * lengths[to.edge]
                    : -1
                  : network?.done
                    ? tail + network.distance + to.fraction * lengths[to.edge]
                    : -1;
            if (route < 0 || route > bound) return;
            const candidateScore = fromScore - Math.log(beta) - Math.abs(straight - route) / beta;
            if (candidateScore > score[toSlot]) {
              score[toSlot] = candidateScore;
              back[toSlot] = fromSlot;
              reachable = true;
            }
          });
        });
      }
      here.forEach((candidate, slot) => {
        const emission = emissionConstant - 0.5 * (candidate.distance / sigma) ** 2;
        if (!reachable) score[slot] = emission;
        else if (score[slot] > MINIMUM_VALID_SCORE) score[slot] += emission;
      });
      let maximum = INVALID_SCORE;
      for (const value of score)
        if (value > MINIMUM_VALID_SCORE) maximum = Math.max(maximum, value);
      const offset = maximum > MINIMUM_VALID_SCORE ? maximum : 0;
      scores.push(score.map(value => (value > MINIMUM_VALID_SCORE ? value - offset : value)));
      scoreOffsets.push(offset);
      backPointers.push(back);
      retained.push(point);
    }
    let current = -1;
    let picking = true;
    let logLikelihood = 0;
    for (let step = retained.length - 1; step >= 0; step--) {
      const point = retained[step];
      const score = scores[step];
      logLikelihood += scoreOffsets[step];
      if (picking) {
        let best = MINIMUM_VALID_SCORE;
        current = -1;
        score.forEach((value, slot) => {
          if (value > best) {
            best = value;
            current = slot;
          }
        });
        if (current >= 0) logLikelihood += best;
      }
      let isBreak = current < 0;
      picking = true;
      if (current >= 0) {
        const candidate = candidates[point][current];
        chosen[point] = candidate;
        result.matchedEdges[point] = candidate.edge;
        result.matchedFractions[point] = candidate.fraction;
        result.matchedOffsets[point] = candidate.fraction * lengths[candidate.edge];
        result.snapDistances[point] = candidate.distance;
        const source = sources[candidate.edge];
        const target = neighbors[candidate.edge];
        const longitude =
          nodePositions[source * 2] +
          candidate.fraction *
            (radius === undefined
              ? nodePositions[target * 2] - nodePositions[source * 2]
              : computeLongitudeDelta(nodePositions[source * 2], nodePositions[target * 2]));
        result.snappedPositions[point * 2] =
          radius === undefined ? longitude : wrapLongitude(longitude);
        result.snappedPositions[point * 2 + 1] =
          nodePositions[source * 2 + 1] +
          candidate.fraction * (nodePositions[target * 2 + 1] - nodePositions[source * 2 + 1]);
        current = backPointers[step][current];
        picking = current < 0;
        isBreak = picking;
        result.matchedCount[0]++;
      }
      if (isBreak && point > trackOffsets[track]) {
        result.breaks[point] = chosen[point] && budgetLimited[point] ? 2 : 1;
        result.breakCount[0]++;
      }
    }
    result.trackLogLikelihoods[track] = logLikelihood;
    if (!capacity) continue;
    let emitted = 0;
    const emitEdge = (edge: number) => {
      if (emitted < capacity) result.traversedEdges[track * capacity + emitted] = edge;
      else result.traversedOverflow[0] = 1;
      emitted++;
    };
    for (const point of retained) {
      const to = chosen[point];
      if (!to) continue;
      const previous = predecessors[point];
      const from = previous < 0 ? undefined : chosen[previous];
      if (!from || result.breaks[point]) {
        if (emitted) emitEdge(NONE);
        emitEdge(to.edge);
      } else if (!(to.edge === from.edge && to.fraction >= from.fraction)) {
        const bound =
          getCPUTrajectoryDistance(positions, point, previous, radius) * routeFactor + routeSlack;
        const start = neighbors[from.edge];
        const target = sources[to.edge];
        const search = searchRoute(
          start,
          [target],
          bound - (1 - from.fraction) * lengths[from.edge],
          offsets,
          neighbors,
          lengths,
          nodePositions,
          radius,
          options.routeNodeBudget
        );
        const path: number[] = [];
        let recovered = true;
        if (search.droppedCount) {
          result.traversedCandidateOverflow[0] = 1;
          result.candidateOverflow[0] = 1;
        }
        for (let node = target; node !== start; ) {
          const entry = search.get(node);
          if (
            !entry?.done ||
            entry.parent === NONE ||
            path.length >= (options.routeNodeBudget ?? Infinity)
          ) {
            recovered = false;
            break;
          }
          path.push(entry.parent);
          node = sources[entry.parent];
        }
        if (!recovered) {
          result.traversedCandidateOverflow[0] = 1;
          result.candidateOverflow[0] = 1;
          emitEdge(NONE);
        } else for (let step = path.length - 1; step >= 0; step--) emitEdge(path[step]);
        emitEdge(to.edge);
      }
    }
    result.traversedCounts[track] = emitted;
  }
  if (capacity && result.candidateOverflow[0]) result.traversedCandidateOverflow[0] = 1;
  return result;
}

/** CPU twin; explicit route budgets and output status meanings match the GPU form. */
export class CPUMapMatchingOperation extends GPUCPUOperation {
  constructor(props: MapMatchingOperationProps) {
    props = snapshotMapMatchingConstructionProps(props);
    const id = props.id ?? 'map-matching';
    validateOptions(id, props);
    for (const name of ['pointCount', 'trackCount', 'nodeCount', 'edgeCount'] as const)
      validateCapacity(id, name, props[name]);
    for (const axis of [0, 1]) {
      validateFiniteNumber(id, `bounds.minimum[${axis}]`, props.bounds.minimum[axis]);
      validateFiniteNumber(id, `bounds.maximum[${axis}]`, props.bounds.maximum[axis]);
      validateRequirement(
        id,
        props.bounds.maximum[axis] > props.bounds.minimum[axis],
        'bounds must have a positive extent'
      );
    }
    if (props.output.traversedEdges)
      validateCapacity(id, 'traversedEdgeCapacity', props.traversedEdgeCapacity ?? 0);
    if (props.routeTable)
      validateCapacity(id, 'routeTable.entryCount', props.routeTable.entryCount, 0);
    const operation = new GPUMapMatchingOperation(props);
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
        const trackOffsets = getCPUOperationUint32Rows(id, inputs, 'trackOffsets');
        const offsets = getCPUOperationUint32Rows(id, inputs, 'offsets');
        const neighbors = getCPUOperationUint32Rows(id, inputs, 'neighbors');
        for (const [name, actual, expected] of [
          ['positions', positions.length, props.pointCount * 2],
          ['trackOffsets', trackOffsets.length, props.trackCount + 1],
          ['topology.offsets', offsets.length, props.nodeCount + 1],
          ['topology.neighbors', neighbors.length, props.edgeCount]
        ] as const)
          if (actual !== expected) throw new Error(`${id} ${name} length must equal ${expected}`);
        const tableTargets = props.routeTable
          ? getCPUOperationUint32Rows(id, inputs, 'tableTargets')
          : undefined;
        if (tableTargets && tableTargets.length !== props.routeTable!.entryCount)
          throw new Error(`${id} routeTable.targets length must equal routeTable.entryCount`);
        const result = computeMapMatchingOnCPU(
          {
            positions,
            trackOffsets,
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters'),
            topology: {
              offsets,
              neighbors,
              nodePositions: getCPUOperationFloat32Rows(id, inputs, 'nodePositions', 'float32x2'),
              weights:
                props.topology.weights === undefined
                  ? undefined
                  : getCPUOperationFloat32Rows(id, inputs, 'weights')
            },
            routeTable: props.routeTable
              ? {
                  offsets: getCPUOperationUint32Rows(id, inputs, 'tableOffsets'),
                  targets: tableTargets!,
                  costs: getCPUOperationFloat32Rows(id, inputs, 'tableCosts'),
                  costLimit: getCPUOperationFloat32Rows(id, inputs, 'tableCostLimit'),
                  complete:
                    props.routeTable.status?.complete === undefined
                      ? undefined
                      : getCPUOperationUint32Rows(id, inputs, 'tableComplete')
                }
              : undefined
          },
          {
            ...props,
            traversedEdgeCapacity: props.output.traversedEdges
              ? props.traversedEdgeCapacity
              : undefined
          }
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUMapMatchingResult]
          ])
        );
      }
    });
  }
}
