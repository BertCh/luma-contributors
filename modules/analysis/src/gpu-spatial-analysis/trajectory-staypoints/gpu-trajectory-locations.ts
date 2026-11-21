// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getTrajectoryVotePositionWGSL} from './geographic-vote';
import {
  validateTrajectoryBindings,
  validateTrajectoryWritableDispatches,
  validateTrajectoryOutputRanges
} from '../trajectory-analysis/trajectory-binding-validation';
import {
  createTransientView,
  GPUCompaction,
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
import {
  validateGPUStatusPort,
  type GPUIterationStatusPort,
  type GPUStatusScalar
} from '../../analysis-core/contracts/index';
import {
  validateCapacity,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {TRAJECTORY_LOCATION_MEMBERSHIP_WGSL} from './trajectory-location-membership';
import {GEODESIC_WGSL} from '../geometry-measures/geodesic-wgsl';
import {MOVEMENT_SENTINEL_WGSL} from './movement-common';
import {getBinarySearchWGSL} from '../../analysis-core/kernels/binary-search-wgsl';
import {
  snapshotConstructionFields,
  snapshotConstructionSpatialContext
} from '../../analysis-core/kernels/construction-snapshot';

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
  if (!Number.isFinite(Math.fround(settings.epsilon)) || Math.fround(settings.epsilon) <= 0) {
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
  /**
   * Compact location per staypoint, in first-visit order within each track, tracks in order;
   * `0xffffffff` past `count`, and in every row when the labels did not settle within
   * `maximumIterations`.
   */
  locationIds: GraphDataView<'uint32'>;
  status: {
    /** Number of locations written; 0 when the labels did not settle within `maximumIterations`. */
    count: GPUStatusScalar;
    /** Required one-row flag for exhaustion of the location solver budget. */
    iterationLimitReached: GPUStatusScalar;
    /** Whole input and a settled solver; independent of converged. */
    complete?: GraphDataView<'uint32'>;
  } & GPUIterationStatusPort;
  /** Optional centroid of distinct coordinates; great-circle voting identifies wrapped longitudes and ignores longitude at exact poles. Not written when labels did not settle. */
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
  /** Forward both source flags when using GPUTrajectoryStaypoints. Optional for independently supplied tables. */
  staypointStatus?: {
    overflow?: GraphDataView<'uint32'>;
    candidateOverflow?: GraphDataView<'uint32'>;
  };
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
 * A final hook checks the labels: when no label changes, every label is its component's root.
 * Otherwise (a chain needing more than `maximumIterations` rounds) the partial labels are not
 * published: every active `locationIds` row is `0xffffffff`, `status.count` is 0, `centers` are
 * not written, and the required one-row uint32 `output.status.iterationLimitReached` is 1.
 * Supplied staypoint overflow or candidateOverflow suppresses the table in the same way;
 * `status.complete` is then 0 even if `status.converged` is 1.
 *
 * Cost is `O(sum per track of m^2)` distance tests per round, with `m` the staypoints of a track.
 * Sorted track ranges are found by parallel binary search. Per-label atomic minima identify
 * first members; published stable `GPUCompaction` assigns first-visit location IDs without a
 * single invocation scanning the staypoint table.
 * Dataset-level locations and `num_samples > 1` are out of scope: use `GPUDBSCAN`, which
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
    props = snapshotConstructionFields(props, [
      'id',
      'spatialContext',
      'centroids',
      'tracks',
      'count',
      'staypointStatus',
      'parameters',
      'maximumIterations',
      'output'
    ]);
    const capturedOutput = snapshotConstructionFields(props.output, [
      'locationIds',
      'centers',
      'status'
    ]);
    props = Object.freeze({
      ...props,
      staypointStatus: props.staypointStatus
        ? snapshotConstructionFields(props.staypointStatus, ['overflow', 'candidateOverflow'])
        : undefined,
      spatialContext: props.spatialContext
        ? snapshotConstructionSpatialContext(props.spatialContext)
        : undefined,
      output: Object.freeze({
        ...capturedOutput,
        status: snapshotConstructionFields(
          capturedOutput.status ?? ({} as typeof capturedOutput.status),
          ['count', 'iterationCount', 'converged', 'complete', 'iterationLimitReached']
        )
      })
    });
    this.id = props.id ?? 'trajectory-locations';
    this.props = props;
    const id = this.id;
    validateRequirement(
      id,
      Boolean(props.output.status?.iterationLimitReached),
      'output.status.iterationLimitReached is required; pass a one-row uint32 view'
    );
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    const {centroids, tracks, count, parameters, output} = props;
    validateSinglePackedView(centroids, ['float32x2'], `${id} centroids`);
    validateSinglePackedView(tracks, ['uint32'], `${id} tracks`);
    for (const [name, view] of Object.entries(props.staypointStatus ?? {})) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} staypointStatus.${name}`);
        validateViewLength(id, `staypointStatus.${name}`, view, 1);
      }
    }
    validateSinglePackedView(count, ['uint32'], `${id} count`);
    validateSinglePackedView(parameters, ['float32'], `${id} parameters`);
    validateSinglePackedView(output.locationIds, ['uint32'], `${id} output.locationIds`);
    validateGPUStatusPort(`${id} output.status`, output.status);
    for (const [name, scalar] of Object.entries(output.status)) {
      if (scalar) validateSinglePackedView(scalar, ['uint32'], `${id} output.status.${name}`);
    }
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
      validateSinglePackedView(output.centers, ['float32x2'], `${id} output.centers`);
      validateViewLength(id, 'output.centers', output.centers, this.capacity);
    }
    this.maximumIterations =
      props.maximumIterations === undefined
        ? GPU_TRAJECTORY_LOCATIONS_DEFAULT_MAXIMUM_ITERATIONS
        : props.maximumIterations;
    validateCapacity(
      id,
      'maximumIterations',
      this.maximumIterations,
      0,
      GPU_TRAJECTORY_LOCATIONS_MAXIMUM_ITERATIONS
    );
    validateTrajectoryOutputRanges(this.id, this.props.output);
    validateTrajectoryWritableDispatches(this.id, [
      {
        iterationCount: output.status.iterationCount,
        converged: output.status.converged,
        iterationLimitReached: output.status.iterationLimitReached
      }
    ]);
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        output.locationIds,
        output.status.count,
        output.status.iterationCount,
        output.status.converged,
        output.status.iterationLimitReached,
        output.status.complete,
        output.centers
      ],
      [centroids, tracks, count, parameters, ...Object.values(props.staypointStatus ?? {})]
    );
  }

  /** Returns the location nodes in dependency order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, capacity, maximumIterations} = this;
    const {centroids, tracks, parameters, output} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      centroids,
      tracks,
      props.count,
      parameters,
      ...Object.values(props.staypointStatus ?? {}),
      output.locationIds,
      output.status.count,
      output.status.iterationCount,
      output.status.converged,
      output.status.iterationLimitReached,
      output.status.complete,
      output.centers
    ]);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'centroids',
      capacity * 8,
      'use fewer staypoints per graph'
    );
    validateTrajectoryBindings(graph, this.id, this.props);
    const count = createTransientView(graph, `${id}-usable-count`, 'uint32', 1);
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
${
  radius === undefined
    ? TRAJECTORY_LOCATION_MEMBERSHIP_WGSL
    : `fn withinRadius(a: vec2<f32>, b: vec2<f32>, radius: f32) -> bool { return ${distanceSource} <= radius; }`
}
`;
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-input-loss`,
        operation: OPERATION,
        invocationCount: 1,
        bindings: [
          {name: 'inputCount', view: props.count, type: 'u32', access: 'read'},
          ...Object.entries(props.staypointStatus ?? {}).flatMap(([name, view]) =>
            view ? [{name, view, type: 'u32' as const, access: 'read' as const}] : []
          ),
          {name: 'count', view: count, type: 'u32', access: 'read_write'}
        ],
        body: `count[countOffset] = select(inputCount[inputCountOffset], 0u, ${
          Object.entries(props.staypointStatus ?? {})
            .filter(([, view]) => view)
            .map(([name]) => `${name}[${name}Offset] != 0u`)
            .join(' || ') || 'false'
        });`
      }),
      createFillNode<Parameters>(graph, {
        id: `${id}-init-a`,
        operation: OPERATION,
        view: labelsA,
        type: 'u32',
        value: 'index'
      }),
      // Every row independently finds its contiguous track range in the sorted table.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-groups`,
        operation: OPERATION,
        invocationCount: capacity,
        bindings: [
          {name: 'tracks', view: tracks, type: 'u32', access: 'read'},
          {name: 'stayCount', view: count, type: 'u32', access: 'read'},
          {name: 'groupStarts', view: groupStarts, type: 'u32', access: 'read_write'},
          {name: 'groupEnds', view: groupEnds, type: 'u32', access: 'read_write'}
        ],
        declarations: getBinarySearchWGSL('tracks'),
        body: `let total = min(stayCount[stayCountOffset], ${capacity}u);
  if (index < total) {
    let track = tracks[tracksOffset + index];
    groupStarts[groupStartsOffset + index] = lowerBoundTracks(0u, total, track);
    groupEnds[groupEndsOffset + index] = upperBoundTracks(0u, total, track);
  } else {
    groupStarts[groupStartsOffset + index] = 0u;
    groupEnds[groupEndsOffset + index] = 0u;
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
      if (other != index && withinRadius(center, point(other), radius)) {
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
    // Hook scratch is no longer read after the convergence check. Reuse it for the first
    // member of every provisional label, including labels that are not their own roots.
    const firstRows = labelsB;
    // Track starts are dead after the final hook/check. Reuse their storage for first flags.
    const firstFlags = groupStarts;
    const rowIds = createTransientView(graph, `${id}-row-ids`, 'uint32', capacity);
    const compactFirstRows = createTransientView(
      graph,
      `${id}-compact-first-rows`,
      'uint32',
      capacity
    );
    nodes.push(
      createFillNode<Parameters>(graph, {
        id: `${id}-first-rows-reset`,
        operation: OPERATION,
        view: firstRows,
        type: 'u32',
        value: MOVEMENT_SENTINEL_WGSL
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-first-rows`,
        operation: OPERATION,
        invocationCount: capacity,
        bindings: [
          {name: 'stayCount', view: count, type: 'u32', access: 'read'},
          {name: 'labels', view: labelsA, type: 'u32', access: 'read'},
          {name: 'firstRows', view: firstRows, type: 'atomic<u32>', access: 'read_write'}
        ],
        body: `if (index < min(stayCount[stayCountOffset], ${capacity}u)) {
    let label = labels[labelsOffset + index];
    atomicMin(&firstRows[firstRowsOffset + label], index);
  }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-first-flags`,
        operation: OPERATION,
        invocationCount: capacity,
        bindings: [
          {name: 'stayCount', view: count, type: 'u32', access: 'read'},
          {name: 'labels', view: labelsA, type: 'u32', access: 'read'},
          {name: 'firstRows', view: firstRows, type: 'u32', access: 'read'},
          {name: 'firstFlags', view: firstFlags, type: 'u32', access: 'read_write'},
          {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'}
        ],
        body: `rowIds[rowIdsOffset + index] = index;
  var first = 0u;
  if (index < min(stayCount[stayCountOffset], ${capacity}u)) {
    let label = labels[labelsOffset + index];
    first = select(0u, 1u, firstRows[firstRowsOffset + label] == index);
  }
  firstFlags[firstFlagsOffset + index] = first;`
      }),
      ...new GPUCompaction({
        id: `${id}-compact-locations`,
        input: rowIds,
        flags: firstFlags,
        output: compactFirstRows,
        count: output.status.count
      }).getCommandNodes(graph),
      // Stable compaction supplies location ordinals in first-visit order.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-relabel`,
        operation: OPERATION,
        invocationCount: capacity,
        bindings: [
          {name: 'labels', view: labelsA, type: 'u32', access: 'read'},
          {name: 'idMap', view: idMap, type: 'u32', access: 'read_write'},
          {name: 'compactFirstRows', view: compactFirstRows, type: 'u32', access: 'read'},
          {name: 'locationCount', view: output.status.count, type: 'u32', access: 'read'}
        ],
        body: `if (index < locationCount[locationCountOffset]) {
    let row = compactFirstRows[compactFirstRowsOffset + index];
    let label = labels[labelsOffset + row];
    idMap[idMapOffset + label] = index;
  }`
      }),
      // Unsettled labels are not all roots: publish no locations rather than stale IDs.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-unsettled-count`,
        operation: OPERATION,
        invocationCount: 1,
        bindings: [
          {name: 'changed', view: changed, type: 'u32', access: 'read'},
          {name: 'locationCount', view: output.status.count, type: 'u32', access: 'read_write'}
        ],
        body: `if (changed[changedOffset] != 0u) { locationCount[locationCountOffset] = 0u; }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-write-ids`,
        operation: OPERATION,
        invocationCount: capacity,
        bindings: [
          {name: 'stayCount', view: count, type: 'u32', access: 'read'},
          {name: 'labels', view: labelsA, type: 'u32', access: 'read'},
          {name: 'changed', view: changed, type: 'u32', access: 'read'},
          {name: 'idMap', view: idMap, type: 'u32', access: 'read'},
          {name: 'locationIds', view: output.locationIds, type: 'u32', access: 'read_write'}
        ],
        body: `var locationId = ${MOVEMENT_SENTINEL_WGSL};
  if (changed[changedOffset] == 0u && index < min(stayCount[stayCountOffset], ${capacity}u)) {
    locationId = idMap[idMapOffset + labels[labelsOffset + index]];
  }
  locationIds[locationIdsOffset + index] = locationId;`
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
            {name: 'changed', view: changed, type: 'u32', access: 'read'},
            {name: 'groupEnds', view: groupEnds, type: 'u32', access: 'read'},
            {name: 'idMap', view: idMap, type: 'u32', access: 'read'},
            {name: 'firstRows', view: firstRows, type: 'u32', access: 'read'},
            {name: 'centers', view: output.centers, type: 'f32', access: 'read_write'}
          ],
          declarations: `${getTrajectoryVotePositionWGSL(radius !== undefined)}
fn point(row: u32) -> vec2<f32> {
  return votePosition(vec2<f32>(centroids[centroidsOffset + 2u * row], centroids[centroidsOffset + 2u * row + 1u]));
}
fn longitudeOffset(delta: f32) -> f32 { return ${radius === undefined ? 'delta' : 'delta - 360.0 * round(delta / 360.0)'}; }`,
          body: `let total = min(stayCount[stayCountOffset], ${capacity}u);
  if (changed[changedOffset] == 0u && index < total) {
    let label = labels[labelsOffset + index];
    if (firstRows[firstRowsOffset + label] != index) { return; }
    let anchor = point(index);
    var offset = vec2<f32>(0.0, 0.0);
    var distinctCount = 0u;
    for (var row = index; row < groupEnds[groupEndsOffset + index]; row++) {
      if (labels[labelsOffset + row] != label) { continue; }
      let position = point(row);
      var isDuplicate = false;
      for (var earlier = index; earlier < row; earlier++) {
        if (labels[labelsOffset + earlier] != label) { continue; }
        let other = point(earlier);
        if (other.x == position.x && other.y == position.y) { isDuplicate = true; break; }
      }
      if (!isDuplicate) {
        offset += vec2<f32>(longitudeOffset(position.x - anchor.x), position.y - anchor.y);
        distinctCount++;
      }
    }
    let center = anchor + offset / f32(max(distinctCount, 1u));
    let locationIndex = idMap[idMapOffset + label];
    centers[centersOffset + 2u * locationIndex] = ${radius === undefined ? 'center.x' : 'canonicalLongitude(center.x)'};
    centers[centersOffset + 2u * locationIndex + 1u] = center.y;
  }`
        })
      );
    }
    const {iterationCount, converged, iterationLimitReached} = output.status;
    if (iterationCount || converged || iterationLimitReached) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-status`,
          operation: OPERATION,
          invocationCount: 1,
          bindings: [
            {name: 'changed', view: changed, type: 'u32', access: 'read'},
            ...(iterationCount
              ? [
                  {
                    name: 'iterationCount',
                    view: iterationCount,
                    type: 'u32',
                    access: 'read_write'
                  } as const
                ]
              : []),
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
  ${iterationCount ? `iterationCount[iterationCountOffset] = ${maximumIterations}u;\n  ` : ''}${converged ? `converged[convergedOffset] = select(1u, 0u, didChange);` : ''}
  ${iterationLimitReached ? `iterationLimitReached[iterationLimitReachedOffset] = select(0u, 1u, didChange);` : ''}`
        })
      );
    }
    if (output.status.complete) {
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-complete`,
          operation: OPERATION,
          invocationCount: 1,
          bindings: [
            {name: 'changed', view: changed, type: 'u32', access: 'read'},
            ...Object.entries(props.staypointStatus ?? {}).flatMap(([name, view]) =>
              view ? [{name, view, type: 'u32' as const, access: 'read' as const}] : []
            ),
            {name: 'complete', view: output.status.complete, type: 'u32', access: 'read_write'}
          ],
          body: `complete[completeOffset] = select(1u, 0u, changed[changedOffset] != 0u || ${
            Object.entries(props.staypointStatus ?? {})
              .filter(([, view]) => view)
              .map(([name]) => `${name}[${name}Offset] != 0u`)
              .join(' || ') || 'false'
          });`
        })
      );
    }
    return nodes;
  }
}
