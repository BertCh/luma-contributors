// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {isTrajectoryLocationWithin} from './trajectory-location-membership';
import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getCPUTrajectoryDistance,
  roundTrajectoryEven
} from '../../analysis-core/operations/cpu-trajectory-distance';
import {GPU_MOVEMENT_SENTINEL} from './movement-common';
import {getTrajectoryVotePosition} from './geographic-vote';
import {
  GPU_TRAJECTORY_LOCATIONS_PARAMETER_LENGTH,
  GPU_TRAJECTORY_LOCATIONS_DEFAULT_MAXIMUM_ITERATIONS,
  GPU_TRAJECTORY_LOCATIONS_MAXIMUM_ITERATIONS
} from './gpu-trajectory-locations';
import {
  GPUTrajectoryLocationsOperation,
  snapshotTrajectoryLocationsOperationProps,
  type TrajectoryLocationsOperationProps
} from './gpu-trajectory-locations-operation';

/** Sorted staypoint table and packed getGPUTrajectoryLocationsParameterValues settings. */
export type CPUTrajectoryLocationsInput = {
  centroids: Float32Array;
  tracks: Uint32Array;
  count: Uint32Array;
  staypointStatus?: {overflow?: Uint32Array; candidateOverflow?: Uint32Array};
  parameters: Float32Array;
};
/** All output ports; centers have the input table's capacity. */
export type CPUTrajectoryLocationsResult = {
  locationIds: Uint32Array;
  centers: Float32Array;
  locationCount: Uint32Array;
  /** Executed hook/compress rounds; zero for the exact union-find path. */
  iterationCount: Uint32Array;
  converged: Uint32Array;
  complete: Uint32Array;
  iterationLimitReached: Uint32Array;
};

/**
 * Finds same-track radius-connected components with exact union-find or explicitly bounded
 * hook/compress rounds, labels them in first-visit order, and averages distinct coordinates
 * with the GPU's anchor-relative longitude convention.
 * Active count is clamped to input capacity; epsilon is inclusive.
 *
 * What differs from the GPU contributor:
 * - Without an explicit maximumIterations, components are exact. With an explicit budget,
 *   bounded hook/compress rounds and the final convergence check match the GPU algorithm.
 *   Unsettled labels publish only sentinel IDs and a zero location count in both forms.
 * - Distances and centroid sums use float64, rounded once for centers; near-boundary centers
 *   can differ. Planar membership uses inclusive Euclidean distance on stored coordinates.
 *   Wrapped longitude is rewrapped after float32 rounding.
 * - Unordered active tracks throw an Error. Unwritten center capacity rows are zero; the GPU
 *   leaves them untouched.
 * - NaN coordinates have no neighbors and each NaN coordinate is distinct for centroid voting;
 *   GPU behavior for nonfinite coordinates is unspecified.
 *
 * @throws Error naming inconsistent centroids, tracks, count or parameters.
 */
export function computeTrajectoryLocationsOnCPU(
  input: CPUTrajectoryLocationsInput,
  options: {spatialContext?: GPUSpatialContext; maximumIterations?: number} = {}
): CPUTrajectoryLocationsResult {
  const {maximumIterations} = options;
  const {centroids, tracks, count, parameters} = input;
  const capacity = centroids.length / 2;
  if (!Number.isInteger(capacity)) throw new Error('centroids must contain float32x2 rows');
  validateCapacity('trajectory-locations', 'centroids capacity', capacity);
  if (maximumIterations !== undefined)
    validateCapacity(
      'trajectory-locations',
      'maximumIterations',
      maximumIterations,
      0,
      GPU_TRAJECTORY_LOCATIONS_MAXIMUM_ITERATIONS
    );
  if (tracks.length !== capacity) throw new Error('tracks length must match centroids');
  if (count.length < 1) throw new Error('count must contain at least one row');
  if (parameters.length < GPU_TRAJECTORY_LOCATIONS_PARAMETER_LENGTH)
    throw new Error('parameters must hold the trajectory locations parameter layout');
  let inputLoss = false;
  for (const name of ['overflow', 'candidateOverflow'] as const) {
    const flag = input.staypointStatus?.[name];
    validateRequirement(
      'trajectory-locations',
      flag === undefined || flag.length === 1,
      `staypointStatus.${name} must contain exactly one uint32 row; received ${flag?.length}`
    );
    inputLoss ||= Boolean(flag?.[0]);
  }
  const total = inputLoss ? 0 : Math.min(count[0], capacity);
  for (let row = 1; row < total; row++)
    if (tracks[row] < tracks[row - 1]) throw new Error('tracks must be sorted in active rows');
  const context = resolveGPUSpatialContext('trajectory-locations', options.spatialContext, {
    ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
    measure: 'distance'
  });
  const radius = getGPUGreatCircleRadius(context);
  const [epsilon] = parameters;
  const withinRadius = (first: number, second: number) =>
    radius === undefined
      ? isTrajectoryLocationWithin(
          centroids[first * 2],
          centroids[first * 2 + 1],
          centroids[second * 2],
          centroids[second * 2 + 1],
          epsilon
        )
      : getCPUTrajectoryDistance(centroids, first, second, radius) <= epsilon;
  const parents = Uint32Array.from({length: total}, (_, row) => row);
  const findRoot = (row: number): number => {
    let root = row;
    while (parents[root] !== root) root = parents[root];
    while (parents[row] !== row) {
      const next = parents[row];
      parents[row] = root;
      row = next;
    }
    return root;
  };
  const bounded = maximumIterations !== undefined;
  let converged = true;
  if (bounded) {
    const hooked = new Uint32Array(total);
    const trackRanges: {start: number; end: number}[] = [];
    for (let start = 0; start < total; ) {
      let end = start + 1;
      while (end < total && tracks[end] === tracks[start]) end++;
      trackRanges.push({start, end});
      start = end;
    }
    for (let round = 0; round < maximumIterations!; round++) {
      for (const {start, end} of trackRanges)
        for (let row = start; row < end; row++) {
          let label = parents[row];
          for (let other = start; other < end; other++) {
            if (withinRadius(row, other)) label = Math.min(label, parents[other]);
          }
          hooked[row] = label;
        }
      for (let row = 0; row < total; row++) parents[row] = hooked[hooked[row]];
    }
    for (const {start, end} of trackRanges)
      for (let row = start; row < end; row++) {
        for (let other = start; other < end; other++) {
          if (parents[other] < parents[row] && withinRadius(row, other)) converged = false;
        }
      }
  } else {
    for (let first = 0; first < total; first++) {
      for (let second = first + 1; second < total && tracks[first] === tracks[second]; second++) {
        if (withinRadius(first, second)) {
          const firstRoot = findRoot(first);
          const secondRoot = findRoot(second);
          parents[Math.max(firstRoot, secondRoot)] = Math.min(firstRoot, secondRoot);
        }
      }
    }
  }
  const locationIds = new Uint32Array(capacity).fill(GPU_MOVEMENT_SENTINEL);
  const centers = new Float32Array(capacity * 2);
  const groups = new Map<number, number[]>();
  for (let row = 0; converged && row < total; row++) {
    const root = bounded ? parents[row] : findRoot(row);
    const group = groups.get(root);
    if (group) group.push(row);
    else groups.set(root, [row]);
  }
  let location = 0;
  for (const members of groups.values()) {
    const [anchorLongitude, anchorLatitude] = getTrajectoryVotePosition(
      centroids[members[0] * 2],
      centroids[members[0] * 2 + 1],
      radius !== undefined
    );
    const distinct = new Set<string>();
    let longitudeOffset = 0;
    let latitudeOffset = 0;
    for (const row of members) {
      locationIds[row] = location;
      const [longitude, latitude] = getTrajectoryVotePosition(
        centroids[row * 2],
        centroids[row * 2 + 1],
        radius !== undefined
      );
      const key =
        Number.isNaN(longitude) || Number.isNaN(latitude)
          ? `nan-row-${row}`
          : `${longitude},${latitude}`;
      if (distinct.has(key)) continue;
      distinct.add(key);
      const delta = longitude - anchorLongitude;
      longitudeOffset +=
        radius === undefined ? delta : delta - 360 * roundTrajectoryEven(delta / 360);
      latitudeOffset += latitude - anchorLatitude;
    }
    let longitude = anchorLongitude + longitudeOffset / Math.max(distinct.size, 1);
    if (radius !== undefined)
      longitude = longitude + 180 - 360 * Math.floor((longitude + 180) / 360) - 180;
    longitude = Math.fround(longitude);
    if (radius !== undefined && longitude >= 180) longitude -= 360;
    centers[location * 2] = longitude;
    centers[location * 2 + 1] = anchorLatitude + latitudeOffset / Math.max(distinct.size, 1);
    location++;
  }
  return {
    locationIds,
    centers,
    locationCount: Uint32Array.of(location),
    iterationCount: Uint32Array.of(maximumIterations ?? 0),
    converged: Uint32Array.of(Number(converged)),
    complete: Uint32Array.of(Number(converged && !inputLoss)),
    iterationLimitReached: Uint32Array.of(Number(!converged))
  };
}

/**
 * CPU locations operation. When bounded propagation exhausts its rounds, it sets the required
 * `output.status.iterationLimitReached` graph value.
 */
export class CPUTrajectoryLocationsOperation extends GPUCPUOperation {
  constructor(props: TrajectoryLocationsOperationProps) {
    props = snapshotTrajectoryLocationsOperationProps(props);
    const id = props.id ?? 'trajectory-locations';
    const {capacity} = props;
    const maximumIterations =
      props.maximumIterations === undefined
        ? props.output.status.iterationCount !== undefined
          ? GPU_TRAJECTORY_LOCATIONS_DEFAULT_MAXIMUM_ITERATIONS
          : undefined
        : props.maximumIterations;
    validateCapacity(id, 'capacity', capacity);
    if (maximumIterations !== undefined)
      validateCapacity(
        id,
        'maximumIterations',
        maximumIterations,
        0,
        GPU_TRAJECTORY_LOCATIONS_MAXIMUM_ITERATIONS
      );
    const spatialContext = resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
      measure: 'distance'
    });
    const operation = new GPUTrajectoryLocationsOperation(props);
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
        const centroids = getCPUOperationFloat32Rows(id, inputs, 'centroids', 'float32x2');
        if (centroids.length !== capacity * 2)
          throw new Error(`${id} centroids length must match capacity`);
        const result = computeTrajectoryLocationsOnCPU(
          {
            centroids,
            tracks: getCPUOperationUint32Rows(id, inputs, 'tracks'),
            count: getCPUOperationUint32Rows(id, inputs, 'count'),
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters'),
            staypointStatus: {
              overflow:
                props.staypointStatus?.overflow !== undefined
                  ? getCPUOperationUint32Rows(id, inputs, 'stayOverflow')
                  : undefined,
              candidateOverflow:
                props.staypointStatus?.candidateOverflow !== undefined
                  ? getCPUOperationUint32Rows(id, inputs, 'stayCandidateOverflow')
                  : undefined
            }
          },
          {spatialContext, maximumIterations}
        );
        return Object.fromEntries(
          Object.entries(operation.outputs).map(([port, output]) => [
            output.value,
            result[port as keyof CPUTrajectoryLocationsResult]
          ])
        );
      }
    });
  }
}
