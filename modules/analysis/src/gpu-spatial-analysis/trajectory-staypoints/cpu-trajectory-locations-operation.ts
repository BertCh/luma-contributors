// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCPUOperation, gpuGraph} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {validateCapacity} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  getCPUOperationFloat32Rows,
  getCPUOperationUint32Rows
} from '../../analysis-core/operations/cpu-operation-rows';
import {
  getCPUTrajectoryDistance,
  roundTrajectoryEven
} from '../../analysis-core/operations/cpu-trajectory-distance';
import {GPU_MOVEMENT_SENTINEL} from './movement-common';
import {GPU_TRAJECTORY_LOCATIONS_PARAMETER_LENGTH} from './gpu-trajectory-locations';
import {
  GPUTrajectoryLocationsOperation,
  type TrajectoryLocationsOperationProps
} from './gpu-trajectory-locations-operation';

/** Sorted staypoint table and packed getGPUTrajectoryLocationsParameterValues settings. */
export type CPUTrajectoryLocationsInput = {
  centroids: Float32Array;
  tracks: Uint32Array;
  count: Uint32Array;
  parameters: Float32Array;
};
/** All output ports; centers have the input table's capacity. */
export type CPUTrajectoryLocationsResult = {
  locationIds: Uint32Array;
  centers: Float32Array;
  locationCount: Uint32Array;
  converged: Uint32Array;
  iterationLimitReached: Uint32Array;
};

/**
 * Finds exact same-track radius-connected components with union-find, labels them in first-visit
 * order, and averages distinct coordinates with the GPU's anchor-relative longitude convention.
 * Active count is clamped to input capacity; epsilon is inclusive.
 *
 * What differs from the GPU contributor:
 * - Components are exact: maximumIterations is ignored, converged is one, and
 *   iterationLimitReached is zero. The GPU can publish incompletely propagated labels.
 * - Distances and centroid sums use float64, rounded once for centers; threshold decisions and
 *   near-boundary centers can differ. Wrapped longitude is rewrapped after float32 rounding.
 * - Unordered active tracks throw an Error. Unwritten center capacity rows are zero; the GPU
 *   leaves them untouched.
 * - NaN coordinates have no neighbors and each NaN coordinate is distinct for centroid voting;
 *   GPU behavior for nonfinite coordinates is unspecified.
 *
 * @throws Error naming inconsistent centroids, tracks, count or parameters.
 */
export function computeTrajectoryLocationsOnCPU(
  input: CPUTrajectoryLocationsInput,
  options: {spatialContext?: GPUSpatialContext} = {}
): CPUTrajectoryLocationsResult {
  const {centroids, tracks, count, parameters} = input;
  const capacity = centroids.length / 2;
  if (!Number.isInteger(capacity)) throw new Error('centroids must contain float32x2 rows');
  validateCapacity('trajectory-locations', 'centroids capacity', capacity);
  if (tracks.length !== capacity) throw new Error('tracks length must match centroids');
  if (count.length < 1) throw new Error('count must contain at least one row');
  if (parameters.length < GPU_TRAJECTORY_LOCATIONS_PARAMETER_LENGTH)
    throw new Error('parameters must hold the trajectory locations parameter layout');
  const total = Math.min(count[0], capacity);
  for (let row = 1; row < total; row++)
    if (tracks[row] < tracks[row - 1]) throw new Error('tracks must be sorted in active rows');
  const context = resolveGPUSpatialContext('trajectory-locations', options.spatialContext, {
    ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
    measure: 'distance'
  });
  const radius = getGPUGreatCircleRadius(context);
  const [epsilon] = parameters;
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
  for (let first = 0; first < total; first++) {
    for (let second = first + 1; second < total && tracks[first] === tracks[second]; second++) {
      if (getCPUTrajectoryDistance(centroids, first, second, radius) <= epsilon) {
        const firstRoot = findRoot(first);
        const secondRoot = findRoot(second);
        parents[Math.max(firstRoot, secondRoot)] = Math.min(firstRoot, secondRoot);
      }
    }
  }
  const locationIds = new Uint32Array(capacity).fill(GPU_MOVEMENT_SENTINEL);
  const centers = new Float32Array(capacity * 2);
  const groups = new Map<number, number[]>();
  for (let row = 0; row < total; row++) {
    const root = findRoot(row);
    const group = groups.get(root);
    if (group) group.push(row);
    else groups.set(root, [row]);
  }
  let location = 0;
  for (const [root, members] of groups) {
    const anchorLongitude = centroids[root * 2];
    const anchorLatitude = centroids[root * 2 + 1];
    const distinct = new Set<string>();
    let longitudeOffset = 0;
    let latitudeOffset = 0;
    for (const row of members) {
      locationIds[row] = location;
      const longitude = centroids[row * 2];
      const latitude = centroids[row * 2 + 1];
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
    converged: Uint32Array.of(1),
    iterationLimitReached: Uint32Array.of(0)
  };
}

/** CPU exact component twin; the GPU-only iteration budget is dropped. */
export class CPUTrajectoryLocationsOperation extends GPUCPUOperation {
  constructor(props: TrajectoryLocationsOperationProps) {
    const id = props.id ?? 'trajectory-locations';
    validateCapacity(id, 'capacity', props.capacity);
    resolveGPUSpatialContext(id, props.spatialContext, {
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
        if (centroids.length !== props.capacity * 2)
          throw new Error(`${id} centroids length must match capacity`);
        const result = computeTrajectoryLocationsOnCPU(
          {
            centroids,
            tracks: getCPUOperationUint32Rows(id, inputs, 'tracks'),
            count: getCPUOperationUint32Rows(id, inputs, 'count'),
            parameters: getCPUOperationFloat32Rows(id, inputs, 'parameters')
          },
          props
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
