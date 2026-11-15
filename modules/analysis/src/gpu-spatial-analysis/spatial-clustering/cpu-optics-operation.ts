// luma.gl
// SPDX-License-Identifier: BSD-3-Clause
// SPDX-FileCopyrightText: Copyright (c) 2007-2024 The scikit-learn developers
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// Xi extraction adapts gpu-optics-xi.ts, derived from scikit-learn 1.7.2 _xi_cluster.
// Original BSD notice and disclaimer are reproduced in optics-xi-license.txt beside this file.

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateFiniteNumber,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getCPUOperationFloat32Rows} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getGPUSphereRadius,
  resolveGPUSpatialContext,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT
} from '../../analysis-core/contracts/spatial-context';
import {resolveGPUTemporalContext} from '../../analysis-core/contracts/temporal-context';
import {GPUOPTICSOperation, type OPTICSOperationProps} from './gpu-optics-operation';
import {GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS, OPTICS_ITERATION_STATUS_FIELDS} from './gpu-optics';

const MISSING_ROW = 0xffffffff;

export type CPUOPTICSOptions = Omit<
  OPTICSOperationProps,
  'id' | 'positions' | 'times' | 'rowCount' | 'output'
>;
export type CPUOPTICSResult = {
  ordering: Uint32Array;
  coreDistances: Float32Array;
  reachability: Float32Array;
  predecessors: Uint32Array;
  labels: Uint32Array;
  status: Uint32Array;
  clusterIntervals: Uint32Array;
  clusterSummary: Uint32Array;
  candidateOverflow: Uint32Array;
  iterationCount: Uint32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
};

function validateOPTICSOptions(
  id: string,
  rowCount: number,
  hasTimes: boolean,
  options: CPUOPTICSOptions
) {
  validateCapacity(id, 'rowCount', rowCount);
  validateCapacity(id, 'minimumSamples', options.minimumSamples);
  validateFiniteNumber(id, 'maximumDistance', options.maximumDistance, 0);
  validateRequirement(
    id,
    Math.fround(options.maximumDistance) > 0,
    'maximumDistance must be positive'
  );
  validateRequirement(
    id,
    ['auto', 'grid', 'brute-force'].includes(options.neighborIndex ?? 'auto'),
    "neighborIndex must be 'auto', 'brute-force' or 'grid'"
  );
  validateRequirement(
    id,
    ['dbscan', 'xi'].includes(options.extractionMethod ?? 'dbscan'),
    "extractionMethod must be 'dbscan' or 'xi'"
  );
  if ((options.extractionMethod ?? 'dbscan') === 'dbscan') {
    validateFiniteNumber(id, 'extractionDistance', options.extractionDistance ?? NaN, 0);
    validateRequirement(
      id,
      Math.fround(options.extractionDistance ?? 0) > 0 &&
        options.extractionDistance! <= options.maximumDistance,
      'extractionDistance must be positive and at most maximumDistance'
    );
  } else
    validateCapacity(
      id,
      'minimumClusterSize',
      options.minimumClusterSize ?? options.minimumSamples,
      2,
      rowCount
    );
  const xi = options.xi ?? 0.05;
  validateFiniteNumber(id, 'xi', xi, 0, 1);
  validateRequirement(id, xi > 0 && xi < 1, 'xi must be in (0, 1)');
  if (options.maximumIterations !== undefined)
    validateCapacity(id, 'maximumIterations', options.maximumIterations);
  if (options.frontierBatchSize !== undefined)
    validateCapacity(id, 'frontierBatchSize', options.frontierBatchSize);
  if (options.neighborCapacity !== undefined)
    validateCapacity(id, 'neighborCapacity', options.neighborCapacity);
  validateRequirement(
    id,
    ['sequential', 'parallel'].includes(options.orderingMethod ?? 'sequential'),
    "orderingMethod must be 'sequential' or 'parallel'"
  );
  const grid =
    options.neighborIndex === 'grid' ||
    ((options.neighborIndex ?? 'auto') === 'auto' &&
      rowCount > GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS);
  validateRequirement(
    id,
    options.orderingMethod !== 'parallel' || grid,
    "orderingMethod 'parallel' needs neighborIndex 'grid'"
  );
  const spatialContext = resolveGPUSpatialContext(id, options.spatialContext, {
    ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
    measure: 'distance'
  });
  if (hasTimes) {
    validateRequirement(id, options.temporalContext !== undefined, 'times need a temporalContext');
    resolveGPUTemporalContext(id, options.temporalContext, {encodings: ['float32-relative']});
    validateFiniteNumber(id, 'maximumTimeDistance', options.maximumTimeDistance ?? NaN, 0);
    validateRequirement(
      id,
      Math.fround(options.maximumTimeDistance ?? 0) > 0,
      'maximumTimeDistance must be positive'
    );
  } else
    validateRequirement(
      id,
      options.temporalContext === undefined && options.maximumTimeDistance === undefined,
      'temporalContext and maximumTimeDistance need times'
    );
  return spatialContext;
}

/**
 * Computes self-inclusive OPTICS core distances, deterministic (reachability, row) ordering and
 * DBSCAN or xi extraction. All numeric work, including xi comparisons, stays float64 until output.
 * Invalid positions/times are excluded; labels and unused ordering/predecessor/interval slots
 * are 0xffffffff. Distances of roots, invalid and non-core rows are infinity as appropriate.
 *
 * What differs from the GPU contributor:
 * - Uses float64 Euclidean/haversine distances and xi comparisons, rounding only output distances.
 *   Ordinary planar distances allow 2e-5 relative error, spherical distances 2e-4 relative.
 *   Near distance ties, radius boundaries or xi thresholds, float32 arithmetic can change discrete
 *   ordering, predecessors, intervals and labels; the strict tie and inclusive bound rules match.
 * - Computes the complete ordering regardless of maximumIterations, frontierBatchSize,
 *   neighborIndex, orderingMethod or neighborCapacity. candidateOverflow and iterationLimitReached are always 0, converged and status.complete are 1.
 * - Drops the GPU row limits (4096/65536), minimumSamples limit (256), batch and command budgets.
 *   The implementation is exhaustive and intended for small collections, with O(n² log n) work.
 * - Xi still has rowCount interval capacity, required by the output layout. Summary is the
 *   clamped interval count and overflow flag; overflow clears all labels, as on the GPU.
 */
export function computeOPTICSOnCPU(
  positions: Float32Array,
  options: CPUOPTICSOptions,
  times?: Float32Array
): CPUOPTICSResult {
  if (positions.length % 2 !== 0)
    throw new Error('computeOPTICSOnCPU positions must contain float32x2 rows');
  const rowCount = positions.length / 2;
  if (times && times.length !== rowCount)
    throw new Error('computeOPTICSOnCPU times length must equal positions row count');
  const spatialContext = validateOPTICSOptions(
    'computeOPTICSOnCPU',
    rowCount,
    times !== undefined,
    options
  );
  const geographic = spatialContext.coordinateSpace === 'longitude-latitude';
  const radius = geographic ? getGPUSphereRadius(spatialContext) : 1;
  const processed = new Uint8Array(rowCount);
  let invalidCount = 0;
  for (let row = 0; row < rowCount; row++) {
    const longitude = positions[2 * row];
    const latitude = positions[2 * row + 1];
    if (
      !Number.isFinite(longitude) ||
      !Number.isFinite(latitude) ||
      (geographic && (Math.abs(longitude) > 180 || Math.abs(latitude) > 90)) ||
      (times && !Number.isFinite(times[row]))
    ) {
      processed[row] = 1;
      invalidCount++;
    }
  }
  const valid = Uint8Array.from(processed, value => 1 - value);
  const getPairDistance = (first: number, second: number): number => {
    if (
      !valid[first] ||
      !valid[second] ||
      (times && Math.abs(times[first] - times[second]) > options.maximumTimeDistance!)
    )
      return Infinity;
    const longitudeDifference = positions[2 * second] - positions[2 * first];
    const latitudeDifference = positions[2 * second + 1] - positions[2 * first + 1];
    if (!geographic) return Math.hypot(longitudeDifference, latitudeDifference);
    const radiansPerDegree = Math.PI / 180;
    const wrappedLongitudeDifference =
      longitudeDifference - 360 * Math.round(longitudeDifference / 360);
    const haversine =
      Math.sin((latitudeDifference * radiansPerDegree) / 2) ** 2 +
      Math.sin((90 - Math.abs(positions[2 * first + 1])) * radiansPerDegree) *
        Math.sin((90 - Math.abs(positions[2 * second + 1])) * radiansPerDegree) *
        Math.sin((wrappedLongitudeDifference * radiansPerDegree) / 2) ** 2;
    return 2 * radius * Math.asin(Math.sqrt(Math.min(1, Math.max(0, haversine))));
  };
  const coreDistances = new Float64Array(rowCount).fill(Infinity);
  for (let row = 0; row < rowCount; row++) {
    if (!valid[row]) continue;
    const distances: number[] = [];
    for (let other = 0; other < rowCount; other++) {
      const distance = getPairDistance(row, other);
      if (distance <= options.maximumDistance) distances.push(distance);
    }
    distances.sort((first, second) => first - second);
    coreDistances[row] = distances[options.minimumSamples - 1] ?? Infinity;
  }
  const ordering = new Uint32Array(rowCount).fill(MISSING_ROW);
  const predecessors = new Uint32Array(rowCount).fill(MISSING_ROW);
  const reachability = new Float64Array(rowCount).fill(Infinity);
  const labels = new Uint32Array(rowCount).fill(MISSING_ROW);
  let visitedCount = 0;
  for (;;) {
    let focus = MISSING_ROW;
    for (let row = 0; row < rowCount; row++) {
      if (!processed[row] && (focus === MISSING_ROW || reachability[row] < reachability[focus]))
        focus = row;
    }
    if (focus === MISSING_ROW) break;
    ordering[visitedCount++] = focus;
    processed[focus] = 1;
    if (!Number.isFinite(coreDistances[focus])) continue;
    for (let row = 0; row < rowCount; row++) {
      if (processed[row]) continue;
      const distance = getPairDistance(focus, row);
      if (distance > options.maximumDistance) continue;
      const candidate = Math.max(coreDistances[focus], distance);
      if (candidate < reachability[row]) {
        reachability[row] = candidate;
        predecessors[row] = focus;
      }
    }
  }
  const clusterIntervals = new Uint32Array(rowCount * 2).fill(MISSING_ROW);
  const clusterSummary = new Uint32Array(2);
  if (options.extractionMethod === 'xi') {
    computeOPTICSXiExtraction(
      ordering,
      visitedCount,
      reachability,
      predecessors,
      labels,
      clusterIntervals,
      clusterSummary,
      options
    );
  } else {
    let cluster = -1;
    for (let rank = 0; rank < visitedCount; rank++) {
      const row = ordering[rank];
      if (reachability[row] > options.extractionDistance!) {
        if (coreDistances[row] <= options.extractionDistance!) labels[row] = ++cluster;
      } else labels[row] = cluster >= 0 ? cluster : MISSING_ROW;
    }
  }
  return {
    ordering,
    predecessors,
    labels,
    coreDistances: Float32Array.from(coreDistances),
    reachability: Float32Array.from(reachability),
    status: Uint32Array.of(visitedCount, 1, invalidCount),
    clusterIntervals,
    clusterSummary,
    iterationCount: Uint32Array.of(visitedCount),
    converged: Uint32Array.of(1),
    iterationLimitReached: Uint32Array.of(0),
    candidateOverflow: Uint32Array.of(0)
  };
}

/** Xi extraction in binary64, retaining GPU interval order and predecessor correction. */
function computeOPTICSXiExtraction(
  ordering: Uint32Array,
  visitedCount: number,
  reachability: Float64Array,
  predecessors: Uint32Array,
  labels: Uint32Array,
  intervals: Uint32Array,
  summary: Uint32Array,
  options: CPUOPTICSOptions
): void {
  const complement = 1 - (options.xi ?? 0.05);
  const getReachability = (rank: number) =>
    rank >= visitedCount ? Infinity : reachability[ordering[rank]];
  const getDirection = (rank: number) => {
    const first = getReachability(rank);
    const second = getReachability(rank + 1);
    return {
      steepUp: first !== second && first <= complement * second,
      steepDown: first !== second && first * complement >= second,
      upward: first < second,
      downward: first > second
    };
  };
  const extendArea = (start: number, downward: boolean) => {
    let end = start;
    let nonSteep = 0;
    for (let rank = start; rank < visitedCount; rank++) {
      const direction = getDirection(rank);
      if (downward ? direction.steepDown : direction.steepUp) {
        nonSteep = 0;
        end = rank;
      } else if (!(downward ? direction.upward : direction.downward)) {
        nonSteep++;
        if (nonSteep > options.minimumSamples) break;
      } else break;
    }
    return end;
  };
  const inverse = new Uint32Array(ordering.length).fill(MISSING_ROW);
  for (let rank = 0; rank < visitedCount; rank++) inverse[ordering[rank]] = rank;
  let areas: {start: number; end: number; maximum: number}[] = [];
  let next = 0;
  let maximumBetween = 0;
  let label = 0;
  for (let steep = 0; steep < visitedCount; steep++) {
    if (steep < next) continue;
    const direction = getDirection(steep);
    if (!direction.steepUp && !direction.steepDown) continue;
    for (let rank = next; rank <= steep; rank++)
      maximumBetween = Math.max(maximumBetween, getReachability(rank));
    areas = areas.filter(
      area =>
        Number.isFinite(maximumBetween) &&
        maximumBetween <= getReachability(area.start) * complement
    );
    for (const area of areas) area.maximum = Math.max(area.maximum, maximumBetween);
    const end = extendArea(steep, direction.steepDown);
    next = end + 1;
    maximumBetween = getReachability(next);
    if (direction.steepDown) {
      areas.push({start: steep, end, maximum: 0});
      continue;
    }
    for (let reverse = areas.length - 1; reverse >= 0; reverse--) {
      const area = areas[reverse];
      let start = area.start;
      let finish = end;
      const upwardMaximum = getReachability(finish + 1);
      if (upwardMaximum * complement < area.maximum) continue;
      const downwardMaximum = getReachability(start);
      if (downwardMaximum * complement >= upwardMaximum) {
        while (start < area.end && getReachability(start + 1) > upwardMaximum) start++;
      } else if (upwardMaximum * complement >= downwardMaximum) {
        while (finish > steep && getReachability(finish - 1) > downwardMaximum) finish--;
      }
      if (options.predecessorCorrection !== false) {
        while (start < finish) {
          if (getReachability(start) > getReachability(finish)) break;
          const predecessor = predecessors[ordering[finish]];
          const predecessorRank = predecessor === MISSING_ROW ? MISSING_ROW : inverse[predecessor];
          if (predecessor !== MISSING_ROW && start <= predecessorRank && predecessorRank < finish)
            break;
          finish--;
        }
        if (start >= finish) continue;
      }
      if (
        finish - start + 1 < (options.minimumClusterSize ?? options.minimumSamples) ||
        start > area.end ||
        finish < steep
      )
        continue;
      const clusterCount = summary[0];
      if (clusterCount >= ordering.length) {
        summary[1] = 1;
        continue;
      }
      intervals[2 * clusterCount] = start;
      intervals[2 * clusterCount + 1] = finish;
      summary[0]++;
      let occupied = false;
      for (let rank = start; rank <= finish; rank++)
        occupied ||= labels[ordering[rank]] !== MISSING_ROW;
      if (!occupied) {
        for (let rank = start; rank <= finish; rank++) labels[ordering[rank]] = label;
        label++;
      }
    }
  }
  if (summary[1] !== 0) labels.fill(MISSING_ROW);
}

/** CPU twin; GPU scheduling/candidate budgets are ignored and the complete result is returned. */
export class CPUOPTICSOperation extends GPUCPUOperation {
  constructor(props: OPTICSOperationProps) {
    const id = props.id ?? 'optics';
    validateOPTICSOptions(id, props.rowCount, props.times !== undefined, props);
    const {output} = props;
    validateRequirement(
      id,
      !(output.clusterIntervals || output.clusterSummary) || props.extractionMethod === 'xi',
      "clusterIntervals and clusterSummary need extractionMethod 'xi'"
    );
    // Match the contributor's structural requirement, independent of which CPU algorithm runs.
    const grid =
      props.neighborIndex === 'grid' ||
      ((props.neighborIndex ?? 'auto') === 'auto' &&
        props.rowCount > GPU_OPTICS_BRUTE_FORCE_MAXIMUM_ROWS);
    validateRequirement(
      id,
      grid ||
        (props.neighborCapacity === undefined &&
          output.ports?.status.candidateOverflow === undefined),
      "neighborCapacity and output.ports.status.candidateOverflow need neighborIndex 'grid'"
    );
    const shape = {length: props.rowCount};
    // Reuse the graph-port shape and disjointness checks; this allocates no GPU resources.
    new GPUOPTICSOperation(props);
    super({
      id,
      inputs: {positions: props.positions, ...(props.times ? {times: props.times} : {})},
      outputs: {
        [output.ordering]: gpuGraph.vector('uint32', shape),
        [output.coreDistances]: gpuGraph.vector('float32', shape),
        [output.reachability]: gpuGraph.vector('float32', shape),
        [output.predecessors]: gpuGraph.vector('uint32', shape),
        [output.labels]: gpuGraph.vector('uint32', shape),
        [output.status]: gpuGraph.vector('uint32', {length: 3}),
        ...(output.clusterIntervals
          ? {[output.clusterIntervals]: gpuGraph.vector('uint32x2', shape)}
          : {}),
        ...(output.clusterSummary
          ? {[output.clusterSummary]: gpuGraph.vector('uint32', {length: 2})}
          : {}),
        ...Object.fromEntries(
          OPTICS_ITERATION_STATUS_FIELDS.filter(name => output.ports?.status[name]).map(name => [
            output.ports!.status[name]!,
            gpuGraph.vector('uint32', {length: 1})
          ])
        ),
        ...(output.ports?.status.candidateOverflow
          ? {[output.ports.status.candidateOverflow]: gpuGraph.vector('uint32', {length: 1})}
          : {})
      },
      execute({inputs}) {
        const positions = getCPUOperationFloat32Rows(id, inputs, 'positions', 'float32x2');
        if (positions.length !== props.rowCount * 2)
          throw new Error(`${id} positions length must equal rowCount * 2`);
        const result = computeOPTICSOnCPU(
          positions,
          props,
          props.times ? getCPUOperationFloat32Rows(id, inputs, 'times') : undefined
        );
        return {
          [output.ordering]: result.ordering,
          [output.coreDistances]: result.coreDistances,
          [output.reachability]: result.reachability,
          [output.predecessors]: result.predecessors,
          [output.labels]: result.labels,
          [output.status]: result.status,
          ...(output.clusterIntervals ? {[output.clusterIntervals]: result.clusterIntervals} : {}),
          ...(output.clusterSummary ? {[output.clusterSummary]: result.clusterSummary} : {}),
          ...Object.fromEntries(
            OPTICS_ITERATION_STATUS_FIELDS.filter(name => output.ports?.status[name]).map(name => [
              output.ports!.status[name]!,
              result[name]
            ])
          ),
          ...(output.ports?.status.candidateOverflow
            ? {[output.ports.status.candidateOverflow]: result.candidateOverflow}
            : {})
        };
      }
    });
  }
}
