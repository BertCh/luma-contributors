// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  getGPUGreatCircleRadius,
  resolveGPUSpatialContext,
  warnGPUShortLongitudeLatitudeDistance,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import type {GPUIterationStatusPort, GPUStatusScalar} from '../../analysis-core/contracts/index';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import {MOVEMENT_SENTINEL_WGSL} from './movement-common';

const OPERATION = 'GPUTrajectoryLocations';
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

/** Number of float32 elements in a location parameter buffer. */
export const GPU_TRAJECTORY_LOCATIONS_PARAMETER_LENGTH = 4;

/** Default compile-time label propagation rounds of {@link GPUTrajectoryLocations}. */
export const GPU_TRAJECTORY_LOCATIONS_DEFAULT_MAXIMUM_ITERATIONS = 32;

/** Largest accepted label propagation rounds. */
export const GPU_TRAJECTORY_LOCATIONS_MAXIMUM_ITERATIONS = 256;

/** Per-frame settings of {@link GPUTrajectoryLocations}. */
export type GPUTrajectoryLocationsSettings = {
  /** Spatial context of the positions, for the short-distance warning only. */
  spatialContext?: GPUSpatialContext;
  /** Clustering radius (trackintel `epsilon`, inclusive), in position or context length units. */
  epsilon: number;
};

/**
 * Packs location settings into the float32 layout read by {@link GPUTrajectoryLocations}.
 *
 * Layout: `[epsilon, 0, 0, 0]`.
 *
 * @param settings Settings to encode.
 * @param target Optional destination of at least 4 elements.
 * @throws If `epsilon` is not finite and positive.
 */
export function getGPUTrajectoryLocationsParameterValues(
  settings: GPUTrajectoryLocationsSettings,
  target: Float32Array = new Float32Array(GPU_TRAJECTORY_LOCATIONS_PARAMETER_LENGTH)
): Float32Array {
  if (!Number.isFinite(settings.epsilon) || settings.epsilon <= 0) {
    throw new Error('Location epsilon must be finite and positive');
  }
  warnGPUShortLongitudeLatitudeDistance(
    'GPUTrajectoryLocations',
    settings.spatialContext,
    settings.epsilon
  );
  target.set([settings.epsilon, 0, 0, 0]);
  return target;
}

/** Caller-owned outputs of {@link GPUTrajectoryLocations}, one row per staypoint capacity row. */
export type GPUTrajectoryLocationsOutput = {
  /** Compact location per staypoint, in first-visit order within each track, tracks in order; `0xffffffff` past `count`. */
  locationIds: GraphDataView<'uint32'>;
  status: {
    /** Number of locations written. */
    count: GPUStatusScalar;
  } & GPUIterationStatusPort;
  /** Optional centroid of the distinct staypoint coordinates of each location. */
  centers?: GraphDataView<'float32x2'>;
};

/** Properties for {@link GPUTrajectoryLocations}. */
export type GPUTrajectoryLocationsProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'trajectory-locations'`. */
  id?: string;
  /** Planar or longitude/latitude great-circle. */
  spatialContext?: GPUSpatialContext;
  /** Staypoint centroids, `GPUTrajectoryStaypoints` output `centroids`. */
  centroids: GraphDataView<'float32x2'>;
  /** Track of each staypoint, sorted (`GPUTrajectoryStaypoints` output `ids`). */
  tracks: GraphDataView<'uint32'>;
  /** One row: active staypoint count (`GPUTrajectoryStaypoints` `status.count`). */
  count: GraphDataView<'uint32'>;
  /** Per-frame float32 view of {@link GPU_TRAJECTORY_LOCATIONS_PARAMETER_LENGTH} elements. */
  parameters: GraphDataView<'float32'>;
  /** Hook-and-compress rounds, compile-time (default 32, at most 256). */
  maximumIterations?: number;
  output: GPUTrajectoryLocationsOutput;
};

/**
 * Staypoint locations per track with trackintel's `generate_locations` (DBSCAN, `num_samples=1`,
 * `agg_level='user'`) semantics: connected components of the graph joining staypoints of one track
 * within `epsilon` (inclusive, haversine in meters for a great-circle context). Components are
 * labelled by bounded min-label hooking with path compression. Noise does not exist because every
 * staypoint is its own core point.
 *
 * Cost is `O(sum per track of m^2)` distance tests per round, with `m` the staypoints of a track.
 * Dataset-level locations and `num_samples > 1` are out of scope: use `GPUSpatialClustering`, which
 * takes `bounds` and a `minimumSamples`-style core rule. The convex-hull `extent` of trackintel is
 * not produced; `concave-hull` and `buffer` contributors compose on the labels.
 */
export class GPUTrajectoryLocations implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTrajectoryLocationsProps;
  readonly spatialContext: GPUSpatialContext;
  readonly maximumIterations: number;
  readonly capacity: number;

  constructor(props: GPUTrajectoryLocationsProps) {
    this.id = props.id ?? 'trajectory-locations';
    this.props = props;
    const id = this.id;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    const {centroids, tracks, count, parameters, output} = props;
    validatePackedView(centroids, ['float32x2'], `${id} centroids`);
    validatePackedView(tracks, ['uint32'], `${id} tracks`);
    validatePackedView(count, ['uint32'], `${id} count`);
    validatePackedView(parameters, ['float32'], `${id} parameters`);
    validatePackedView(output.locationIds, ['uint32'], `${id} output.locationIds`);
    validateRequirement(id, count.length >= 1, 'count must contain one uint32 row');
    this.capacity = centroids.length;
    validateViewLength(id, 'tracks', tracks, this.capacity);
    validateViewLength(id, 'output.locationIds', output.locationIds, this.capacity);
    validateCapacity(id, 'staypoint capacity', this.capacity);
    validateRequirement(
      id,
      parameters.length >= GPU_TRAJECTORY_LOCATIONS_PARAMETER_LENGTH,
      `parameters must hold ${GPU_TRAJECTORY_LOCATIONS_PARAMETER_LENGTH} float32 values`
    );
    if (output.centers) {
      validatePackedView(output.centers, ['float32x2'], `${id} output.centers`);
      validateViewLength(id, 'output.centers', output.centers, this.capacity);
    }
    this.maximumIterations =
      props.maximumIterations ?? GPU_TRAJECTORY_LOCATIONS_DEFAULT_MAXIMUM_ITERATIONS;
    validateMaximumCount(
      id,
      'maximumIterations',
      this.maximumIterations,
      GPU_TRAJECTORY_LOCATIONS_MAXIMUM_ITERATIONS
    );
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.locationIds,
        output.status.count,
        output.status.converged,
        output.status.iterationLimitReached,
        output.centers
      ],
      [centroids, tracks, count, parameters]
    );
  }

  /** Returns the location nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, capacity, maximumIterations} = this;
    const {centroids, tracks, count, parameters, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      centroids,
      tracks,
      count,
      parameters,
      output.locationIds,
      output.status.count,
      output.status.converged,
      output.status.iterationLimitReached,
      output.centers
    ]);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'centroids',
      capacity * 8,
      'use fewer staypoints per graph'
    );
    const radius = getGPUGreatCircleRadius(this.spatialContext);
    const labelsA = createTransientView(graph, `${id}-labels-a`, 'uint32', capacity);
    const labelsB = createTransientView(graph, `${id}-labels-b`, 'uint32', capacity);
    const groupStarts = createTransientView(graph, `${id}-group-starts`, 'uint32', capacity);
    const groupEnds = createTransientView(graph, `${id}-group-ends`, 'uint32', capacity);
    const changed = createTransientView(graph, `${id}-changed`, 'uint32', 1);
    const distanceSource =
      radius === undefined
        ? 'length(a - b)'
        : `geodesicHaversineDistance(a, b, ${radius.toExponential(10)})`;
    const distanceWGSL = `${radius === undefined ? '' : GEODESIC_WGSL}
fn point(row: u32) -> vec2<f32> {
  return vec2<f32>(centroids[centroidsOffset + 2u * row], centroids[centroidsOffset + 2u * row + 1u]);
}
fn neighborDistance(a: vec2<f32>, b: vec2<f32>) -> f32 { return ${distanceSource}; }
`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createFillNode<Parameters>(graph, {
        id: `${id}-init-a`,
        operation: OPERATION,
        view: labelsA,
        type: 'u32',
        value: 'index'
      }),
      // One serial thread over the sorted staypoint table; O(stays) and trivial.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-groups`,
        operation: OPERATION,
        invocationCount: 1,
        bindings: [
          {name: 'tracks', view: tracks, type: 'u32', access: 'read'},
          {name: 'stayCount', view: count, type: 'u32', access: 'read'},
          {name: 'groupStarts', view: groupStarts, type: 'u32', access: 'read_write'},
          {name: 'groupEnds', view: groupEnds, type: 'u32', access: 'read_write'}
        ],
        body: `let total = min(stayCount[stayCountOffset], ${capacity}u);
  var groupStart = 0u;
  for (var row = 0u; row <= total; row++) {
    if (row == total || (row > 0u && tracks[tracksOffset + row] != tracks[tracksOffset + row - 1u])) {
      for (var member = groupStart; member < row; member++) {
        groupStarts[groupStartsOffset + member] = groupStart;
        groupEnds[groupEndsOffset + member] = row;
      }
      groupStart = row;
    }
  }`
      })
    ];
    const hook = (
      phase: string,
      reachLabels: GraphDataView<'uint32'>,
      writeLabels: GraphDataView<'uint32'>,
      checkChange: boolean
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-hook-${phase}`,
        operation: OPERATION,
        variant: 'hook',
        invocationCount: capacity,
        bindings: [
          {name: 'centroids', view: centroids, type: 'f32', access: 'read'},
          {name: 'parameters', view: parameters, type: 'f32', access: 'read'},
          {name: 'stayCount', view: count, type: 'u32', access: 'read'},
          {name: 'groupStarts', view: groupStarts, type: 'u32', access: 'read'},
          {name: 'groupEnds', view: groupEnds, type: 'u32', access: 'read'},
          {name: 'labelsIn', view: reachLabels, type: 'u32', access: 'read'},
          ...(checkChange
            ? [{name: 'changed', view: changed, type: 'atomic<u32>', access: 'read_write'} as const]
            : [{name: 'labelsOut', view: writeLabels, type: 'u32', access: 'read_write'} as const])
        ],
        declarations: distanceWGSL,
        body: `if (index < min(stayCount[stayCountOffset], ${capacity}u)) {
    let center = point(index);
    var label = labelsIn[labelsInOffset + index];
    let radius = parameters[parametersOffset];
    for (var other = groupStarts[groupStartsOffset + index]; other < groupEnds[groupEndsOffset + index]; other++) {
      if (other != index && neighborDistance(center, point(other)) <= radius) {
        label = min(label, labelsIn[labelsInOffset + other]);
      }
    }
    ${checkChange ? `if (label != labelsIn[labelsInOffset + index]) { atomicStore(&changed[changedOffset], 1u); }` : `labelsOut[labelsOutOffset + index] = label;`}
  }${checkChange ? '' : ` else { labelsOut[labelsOutOffset + index] = ${MOVEMENT_SENTINEL_WGSL}; }`}`
      });
    // Hook A into B, then compress into A: one round of min-label hooking and path halving.
    for (let round = 0; round < maximumIterations; round++) {
      nodes.push(
        hook(`${round}`, labelsA, labelsB, false),
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-compress-${round}`,
          operation: OPERATION,
          variant: 'compress',
          invocationCount: capacity,
          bindings: [
            {name: 'stayCount', view: count, type: 'u32', access: 'read'},
            {name: 'labelsIn', view: labelsB, type: 'u32', access: 'read'},
            {name: 'labelsOut', view: labelsA, type: 'u32', access: 'read_write'}
          ],
          body: `if (index < min(stayCount[stayCountOffset], ${capacity}u)) {
    let label = labelsIn[labelsInOffset + index];
    labelsOut[labelsOutOffset + index] = labelsIn[labelsInOffset + label];
  } else {
    labelsOut[labelsOutOffset + index] = ${MOVEMENT_SENTINEL_WGSL};
  }`
        })
      );
    }
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-changed-reset`,
        operation: OPERATION,
        view: changed,
        type: 'u32',
        value: '0u',
        componentCount: 1
      }),
      hook('check', labelsA, labelsB, true)
    );
    const idMap = createTransientView(graph, `${id}-id-map`, 'uint32', capacity);
    nodes.push(
      // First-visit order within tracks, tracks in order: one serial thread over the table.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-relabel`,
        operation: OPERATION,
        invocationCount: 1,
        bindings: [
          {name: 'stayCount', view: count, type: 'u32', access: 'read'},
          {name: 'labels', view: labelsA, type: 'u32', access: 'read'},
          {name: 'idMap', view: idMap, type: 'u32', access: 'read_write'},
          {name: 'locationCount', view: output.status.count, type: 'u32', access: 'read_write'}
        ],
        body: `var nextId = 0u;
  for (var row = 0u; row < ${capacity}u; row++) {
    if (row < min(stayCount[stayCountOffset], ${capacity}u) && labels[labelsOffset + row] == row) {
      idMap[idMapOffset + row] = nextId;
      nextId++;
    }
  }
  locationCount[locationCountOffset] = nextId;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-ids`,
        operation: OPERATION,
        invocationCount: capacity,
        bindings: [
          {name: 'stayCount', view: count, type: 'u32', access: 'read'},
          {name: 'labels', view: labelsA, type: 'u32', access: 'read'},
          {name: 'idMap', view: idMap, type: 'u32', access: 'read'},
          {name: 'locationIds', view: output.locationIds, type: 'u32', access: 'read_write'}
        ],
        body: `locationIds[locationIdsOffset + index] = select(${MOVEMENT_SENTINEL_WGSL}, idMap[idMapOffset + labels[labelsOffset + index]], index < min(stayCount[stayCountOffset], ${capacity}u));`
      })
    );
    if (output.centers) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-centers`,
          operation: OPERATION,
          invocationCount: capacity,
          bindings: [
            {name: 'centroids', view: centroids, type: 'f32', access: 'read'},
            {name: 'stayCount', view: count, type: 'u32', access: 'read'},
            {name: 'labels', view: labelsA, type: 'u32', access: 'read'},
            {name: 'groupEnds', view: groupEnds, type: 'u32', access: 'read'},
            {name: 'idMap', view: idMap, type: 'u32', access: 'read'},
            {name: 'centers', view: output.centers, type: 'f32', access: 'read_write'}
          ],
          declarations: `fn point(row: u32) -> vec2<f32> {
  return vec2<f32>(centroids[centroidsOffset + 2u * row], centroids[centroidsOffset + 2u * row + 1u]);
}
fn longitudeOffset(delta: f32) -> f32 { return ${radius === undefined ? 'delta' : 'delta - 360.0 * round(delta / 360.0)'}; }`,
          body: `let total = min(stayCount[stayCountOffset], ${capacity}u);
  if (index < total && labels[labelsOffset + index] == index) {
    let anchor = point(index);
    var offset = vec2<f32>(0.0, 0.0);
    var distinctCount = 0u;
    for (var row = index; row < groupEnds[groupEndsOffset + index]; row++) {
      if (labels[labelsOffset + row] != index) { continue; }
      let position = point(row);
      var isDuplicate = false;
      for (var earlier = index; earlier < row; earlier++) {
        if (labels[labelsOffset + earlier] != index) { continue; }
        let other = point(earlier);
        if (other.x == position.x && other.y == position.y) { isDuplicate = true; break; }
      }
      if (!isDuplicate) {
        offset += vec2<f32>(longitudeOffset(position.x - anchor.x), position.y - anchor.y);
        distinctCount++;
      }
    }
    let center = anchor + offset / f32(max(distinctCount, 1u));
    let locationIndex = idMap[idMapOffset + index];
    centers[centersOffset + 2u * locationIndex] = ${radius === undefined ? 'center.x' : '((center.x + 180.0) - 360.0 * floor((center.x + 180.0) / 360.0)) - 180.0'};
    centers[centersOffset + 2u * locationIndex + 1u] = center.y;
  }`
        })
      );
    }
    const {converged, iterationLimitReached} = output.status;
    if (converged || iterationLimitReached) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-status`,
          operation: OPERATION,
          invocationCount: 1,
          bindings: [
            {name: 'changed', view: changed, type: 'u32', access: 'read'},
            ...(converged
              ? [{name: 'converged', view: converged, type: 'u32', access: 'read_write'} as const]
              : []),
            ...(iterationLimitReached
              ? [
                  {
                    name: 'iterationLimitReached',
                    view: iterationLimitReached,
                    type: 'u32',
                    access: 'read_write'
                  } as const
                ]
              : [])
          ],
          body: `let didChange = changed[changedOffset] != 0u;
  ${converged ? `converged[convergedOffset] = select(1u, 0u, didChange);` : ''}
  ${iterationLimitReached ? `iterationLimitReached[iterationLimitReachedOffset] = select(0u, 1u, didChange);` : ''}`
        })
      );
    }
    return nodes;
  }
}
