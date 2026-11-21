// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUVisibilityWorkflow,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  createPublishNode,
  createWGSLKernelNode,
  createTransientUint32Rows
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {
  GPUCompactOutput,
  GPUUint32Rows
} from '../../analysis-core/kernels/gpu-contributor-types';
import type {
  GPUPointGeometryPort,
  GPUPackedGeometryPort,
  GPUMultipartPointGeometryPort
} from '../../analysis-core/contracts/index';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  getGraphViewChunks,
  isGraphVectorView,
  validateGraphViewTopology,
  validateCompactOutput,
  validateCapacity,
  validateSinglePackedView,
  validatePowerOfTwo,
  validateRequirement,
  validateNonZeroFiniteNumber
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUNearestFeatureJoin,
  getGPUNearestFeatureJoinMaximumLeafCapacity
} from './gpu-nearest-feature-join';
import {validateStatusScalars} from './spatial-join-geometry';
import {validateGPUMultipartGeometryPort} from '../../analysis-core/contracts/index';
import {
  getNextPowerOfTwo,
  getDefaultSpatialSort,
  getDefaultQuerySort,
  validateDisjointOutputs,
  type SpatialSortCurve
} from './spatial-join-passes';
import {
  validateSpatialJoinBindings,
  validateSpatialJoinReadWriteBindings
} from './spatial-join-binding-preflight';
import {validateNearestNeighborBindings} from './nearest-nodes';
import {NEAREST_NODE_WORDS} from './nearest-wgsl';
import {
  createGraphChunkGroupKernelNode,
  validateGraphChunkGroupKernelSupport
} from '../../analysis-core/kernels/graph-chunk-groups';
import {GPU_SPATIAL_JOIN_NO_FEATURE, type GPUNearestFeatureSource} from './spatial-join-types';

import {
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  resolveGPUSecondarySpatialContext,
  getGPUGreatCircleRadius,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';

/** Planar or great-circle contexts suitable for distance measurements in their length unit. */
const BUFFER_SELECTION_SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

const OPERATION = 'GPUBufferSelection';

/** Public upstream visibility uses its local vector constructor; retain ordered resource views. */
function getVisibilityRows(rows: GPUUint32Rows): GPUUint32Rows {
  return isGraphVectorView(rows) ? new GraphVectorView({...rows, data: rows.data}) : rows;
}

/** Query row bases and compaction totals are uint32. */
const GPU_BUFFER_SELECTION_MAXIMUM_ROWS = 0xffffffff;
/** Published radix sort uses four-bit buckets in 256-key workgroups. */
const GPU_BUFFER_SELECTION_RADIX_BUCKET_COUNT = 16;
/** Matches the published radix sort and spatial-bound reduction tile width. */
const GPU_BUFFER_SELECTION_SORT_WORKGROUP_SIZE = 256;
/** A positive placeholder bypasses unused planar-candidate validation in the nested geodesic route. */
const GPU_BUFFER_SELECTION_UNUSED_CANDIDATE_CAPACITY = 1;

/** Caller-owned outputs of {@link GPUBufferSelection}. */
export type GPUBufferSelectionOutput = {
  /** Optional 0/1 selection mask, chunked like single points or packed per multipart feature. Rewritten on every encoding. */
  outputMask?: GPUUint32Rows;
  /**
   * Optional bounded stable-ID result in ascending row order. `matches.ids.length` must be at
   * least one. `matches.status.overflow` reports final selection capacity loss. Optional
   * `matches.status.candidateOverflow` reports incomplete join candidates separately.
   */
  matches?: GPUCompactOutput;
  /**
   * Optional one-row draw count: the clamped selected count (`matches.count`), written by the same
   * publish node. Bind it as the instance count of an indirect draw, or as the source of an
   * indirect-args copy, so the selected points are drawn without a readback or a hand-copied count.
   * Requires `matches`.
   */
  drawInstanceCount?: GraphDataView<'uint32'>;
  /** Optional distance to the nearest reference feature (in the context length unit), or -1 outside. Multipart rows use the minimum over parts. Chunked like single points or packed per multipart feature. */
  distances?: GraphDataView<'float32'> | GraphVectorView<'float32'>;
  /** Optional nearest reference feature row before application IDs, or GPU_SPATIAL_JOIN_NO_FEATURE. Chunked like single points or packed per multipart feature. */
  nearestFeatureRows?: GPUUint32Rows;
  /**
   * Optional per-point flag, one packed `uint32` per point in row order across chunks, 1 when that
   * point's nearest-feature search may be incomplete because the join dropped candidates
   * (`candidateCapacity`, planar only) or BVH leaves (`leafCapacity`, every point). A flagged point
   * may be left out of the selection although a feature lies within the distance. It is
   * `GPUNearestFeatureJoinOutput.candidateTruncated` passed through. Multipart inputs use one
   * packed row per feature; their feature traversal uses only leafCapacity, not candidateCapacity.
   */
  candidateTruncated?: GraphDataView<'uint32'>;
  /** Completeness scalars. */
  status?: {
    /** Optional one-row flag: 1 when the join's BVH leaf or candidate capacity overflowed. */
    overflow?: GraphDataView<'uint32'>;
    /**
     * Optional one-row flag under the shared status name: 1 when the join's BVH leaf or candidate
     * capacity was too small (the join's `candidateOverflow`). `candidateTruncated` names the
     * points affected.
     */
    candidateOverflow?: GraphDataView<'uint32'>;
  };
};

/** Stable compact selection with candidate completeness available even without caller destinations. */
export type GPUBufferSelectionMatches = GPUCompactOutput & {
  status: GPUCompactOutput['status'] & {candidateOverflow: GraphDataView<'uint32'>};
};

/**
 * Properties for {@link GPUBufferSelection}.
 *
 * Per-frame: `distance` and the contents of every input buffer (points, features, source IDs).
 * Compile-time: view lengths and chunk topology, the feature kind, `candidateCapacity`,
 * `leafCapacity`, `spatialSort`, `output.matches.ids.length`, and which optional views exist.
 */
export type GPUBufferSelectionProps<
  PointGeometry extends GPUPointGeometryPort | GPUMultipartPointGeometryPort =
    | GPUPointGeometryPort
    | GPUMultipartPointGeometryPort
> = {
  /** Prefix for generated node and transient IDs. Defaults to `'buffer-selection'`. */
  id?: string;
  /**
   * Coordinate space and metric of the points and features, forwarded to
   * {@link GPUNearestFeatureJoin}. Omitted means native planar coordinates, with `distance` and
   * `distances` in coordinate units. `{coordinateSpace: 'longitude-latitude', metric:
   * 'great-circle'}` selects points within a great-circle distance in the context length unit
   * (meters by default, or kilometers; `sphereRadius` overrides the mean Earth radius), with segment features straight in
   * longitude/latitude; `candidateCapacity` is then ignored. Other longitude/latitude metrics
   * throw at construction.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Context of `features` when declared separately from `spatialContext`. Must be compatible
   * with `spatialContext` (datum, coordinate space, metric, units, planar frame).
   * Omitted means `spatialContext` describes both inputs.
   */
  featuresSpatialContext?: GPUSpatialContext;
  /**
   * Points to test, possibly chunked; contents are per-frame and at least one row is required.
   * Optional `sourceIds` (same chunk topology) are written to `output.matches.ids`; row indices
   * are used when omitted. Multipart features select when any part is within the distance;
   * outputs and source IDs have one packed row per feature, including empty features.
   */
  points: PointGeometry;
  /**
   * Point features, or segment features. Express a polyline as consecutive segment rows.
   * Contents are per-frame; the feature count is compile-time. Feature `sourceIds` do not affect
   * selection or `output.nearestFeatureRows`; rows identify the reference geometry.
   */
  features: GPUNearestFeatureSource;
  /**
   * Per-frame radius in the context length unit, one float32 row. A point is selected when its distance to
   * the nearest feature is less than or equal to this value: the comparison is inclusive and is
   * made in float32, exactly as `GPUNearestFeatureJoin` does (`distance >= 0 && distance <= radius`).
   * NaN, negative, or infinite values select nothing.
   */
  radius: GraphDataView<'float32'>;
  /** Maximum (point, feature) bounding-box candidates per encoding for single-point planar queries. Validated but unused for multipart feature traversal. Compile-time. */
  candidateCapacity: number;
  /** Power-of-two BVH leaf slots forwarded to the join. Compile-time. */
  leafCapacity?: number;
  /**
   * Spatially sorts features before the BVH build, forwarded to `GPUNearestFeatureJoin` (Hilbert order by default, see `spatialSortCurve`). Compile-time.
   * Results are identical either way. Defaults to on from 256 features, as in
   * `GPUNearestFeatureJoin`; pass `false` to skip it.
   */
  spatialSort?: boolean;
  /**
   * Experimental, compile-time. Curve used by `spatialSort`: `'hilbert'` (default; 10 to 15% faster joins
   * than Morton in paired A/B runs) or `'morton'` (Z-order). Results are identical; only BVH
   * traversal cost changes.
   */
  spatialSortCurve?: SpatialSortCurve;
  /**
   * Processes the points in Hilbert order over the feature BVH bounds, forwarded to
   * `GPUNearestFeatureJoin`. Compile-time. Results are identical either way. Defaults to on from
   * 4096 points and 256 features.
   */
  querySort?: boolean;
  /** Caller-owned outputs; at least one of `outputMask` and `matches` is required. */
  output: GPUBufferSelectionOutput;
};

/**
 * Selects the points within a per-frame planar distance of point or polyline features.
 *
 * The contributor composes {@link GPUNearestFeatureJoin} (a feature BVH, candidate pairs, and exact
 * point-to-segment distances) with a mask kernel (`nearestFeatureRows != no feature`) and, when
 * `output.matches` is requested, `GPUVisibilityWorkflow` over that mask.
 *
 * This differs from the join's own `matches` output, which is unordered (atomic append). The
 * buffer selection publishes a source-aligned 0/1 mask and stable IDs in ascending row order, so
 * results are deterministic and can drive highlighting or draw-call instancing directly.
 *
 * Non-goals: polygon buffers and per-feature distances. Distances are planar in the units of the
 * input coordinates, or great-circle distances in the length unit of the longitude/latitude `spatialContext`.
 */
export class GPUBufferSelection<
  PointGeometry extends GPUPointGeometryPort | GPUMultipartPointGeometryPort =
    | GPUPointGeometryPort
    | GPUMultipartPointGeometryPort
> implements GPUCommandNodeProducer
{
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUBufferSelectionProps;
  /** Number of points. */
  readonly pointCount: number;
  /** Resolved spatial context. */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved nested join topology, validated without declaring its scratch. */
  private readonly featureCount: number;
  private readonly leafCapacity: number;
  private readonly spatialSort: boolean;
  private readonly querySort: boolean;
  private readonly matchesByGraph = new WeakMap<object, GPUBufferSelectionMatches>();

  /** All route storage/binding checks precede compact-port publication, including direct callers. */
  private preflight<Parameters>(graph: GPUCommandGraph<Parameters>): void {
    const {id, props, pointCount} = this;
    const points = props.points.positions;
    const chunks = getGraphViewChunks(points);
    const output = props.output;
    // Pure metadata stand-in for adapter-owned uint32 destinations; never bound or declared.
    const placeholder: GraphDataView<'uint32'> = {...props.radius, format: 'uint32'};
    if (props.points.featureOffsets) {
      validateSpatialJoinBindings(
        graph,
        id,
        Object.fromEntries(
          [...getInputs(props), ...getOutputs(props)].map((view, index) => [
            `binding[${index}]`,
            view
          ])
        ),
        {'feature selection scratch': pointCount * 4}
      );
      // Use existing caller-owned views only as size-checked placeholders. The nested traversal
      // must reject every capacity/binding error before this adapter declares scratch IDs.
      validatePowerOfTwo(
        id,
        'leafCapacity',
        this.leafCapacity,
        1,
        getGPUNearestFeatureJoinMaximumLeafCapacity(graph.device)
      );
      validateNearestNeighborBindings(graph, {
        id,
        operation: OPERATION,
        queries: {
          kind: 'geometry',
          geometry: {
            ...props.points,
            sourceIds: undefined
          } as GPUPackedGeometryPort<GPUMultipartPointGeometryPort>
        },
        queryCount: pointCount,
        features: props.features,
        featureCount: this.featureCount,
        neighborCount: 1,
        capacity: 1,
        leafCapacity: this.leafCapacity,
        spatialSort: this.spatialSort,
        spatialSortCurve: props.spatialSortCurve,
        querySort: this.querySort,
        maxDistance: props.radius,
        sphereRadius: getGPUGreatCircleRadius(this.spatialContext),
        neighborIds: placeholder,
        neighborCounts: placeholder,
        neighborDistances: output.distances as GraphDataView<'float32'> | undefined,
        overflow: output.status?.overflow ?? placeholder
      });

      return;
    }
    const sort = this.spatialSort && this.featureCount > 1;
    const candidateCapacity =
      this.spatialContext.metric === 'great-circle'
        ? GPU_BUFFER_SELECTION_UNUSED_CANDIDATE_CAPACITY
        : props.candidateCapacity;
    const scratchBytes: Record<string, number> = {
      'join state and indirect dispatch': 16,
      'BVH bounds and children': (2 * this.leafCapacity - 1) * 8,
      'feature bounds and sorted bounds': this.featureCount * 8,
      'feature radix histogram': sort
        ? GPU_BUFFER_SELECTION_RADIX_BUCKET_COUNT *
          Math.ceil(this.featureCount / GPU_BUFFER_SELECTION_SORT_WORKGROUP_SIZE) *
          4
        : 0,
      'feature sort tile/scene bounds': sort
        ? Math.max(1, Math.ceil(this.featureCount / GPU_BUFFER_SELECTION_SORT_WORKGROUP_SIZE)) * 16
        : 0,
      'query assignment and scan scratch': pointCount * 4,
      'candidate pair and geometry scratch': candidateCapacity * 8,
      'query radix histogram': this.querySort
        ? GPU_BUFFER_SELECTION_RADIX_BUCKET_COUNT *
          Math.ceil(pointCount / GPU_BUFFER_SELECTION_SORT_WORKGROUP_SIZE) *
          4
        : 0,
      'visibility chunk totals/offsets': chunks.filter(chunk => chunk.length > 0).length * 4,
      'selected IDs': pointCount * 4,
      'selected total': 4
    };
    // Per-chunk mask, identity, normalized flags and scan rows never exceed the packed query
    // assignment envelope. Published scan hierarchies shrink by at least 256 at each level.
    validateSpatialJoinBindings(
      graph,
      id,
      Object.fromEntries(
        [...getInputs(props), ...getOutputs(props)].map((view, index) => [
          `binding[${index}]`,
          view
        ])
      ),
      scratchBytes
    );
    validatePowerOfTwo(
      id,
      'leafCapacity',
      this.leafCapacity,
      1,
      getGPUNearestFeatureJoinMaximumLeafCapacity(graph.device)
    );
    if (this.spatialContext.metric === 'great-circle') {
      // Reuse the nested contributor's pure planner; these existing uint32 views stand in only
      // for adapter-owned outputs whose bindings are checked above before they are declared.
      const scalar = placeholder;
      const plan = validateNearestNeighborBindings(graph, {
        id,
        operation: OPERATION,
        queries: {kind: 'points', points},
        queryCount: pointCount,
        features: props.features,
        featureCount: this.featureCount,
        neighborCount: 1,
        capacity: 1,
        leafCapacity: this.leafCapacity,
        spatialSort: this.spatialSort,
        querySort: this.querySort,
        neighborIds: scalar,
        neighborCounts: scalar,
        overflow: scalar
      });
      validateSpatialJoinBindings(
        graph,
        id,
        {},
        {
          'great-circle node table':
            ((2 * this.leafCapacity - 1) * NEAREST_NODE_WORDS + 2 * pointCount) * 4,
          'great-circle topk': (plan.scratchQueryCount * plan.stride + 3) * 4,
          'great-circle query bounds': this.querySort ? pointCount * 8 : 0
        }
      );
    }
    validateGraphChunkGroupKernelSupport(graph, id, 8);
  }

  /** Returns graph-owned destinations for omitted views, preserving every supplied compact view. */
  getMatches<Parameters>(graph: GPUCommandGraph<Parameters>): GPUBufferSelectionMatches {
    this.preflight(graph);
    const cached = this.matchesByGraph.get(graph);
    if (cached) return cached;
    const supplied = this.props.output.matches;
    const scalar = (name: string) =>
      createTransientView(graph, `${this.id}-matches-${name}`, 'uint32', 1);
    const matches: GPUBufferSelectionMatches = Object.freeze({
      ids:
        supplied?.ids ??
        createTransientView(
          graph,
          `${this.id}-matches-ids`,
          'uint32',
          Math.max(1, this.pointCount)
        ),
      status: Object.freeze({
        count: supplied?.status.count ?? scalar('count'),
        requiredCount: supplied?.status.requiredCount ?? scalar('required-count'),
        overflow: supplied?.status.overflow ?? scalar('overflow'),
        candidateOverflow: supplied?.status.candidateOverflow ?? scalar('candidate-overflow')
      })
    });
    this.matchesByGraph.set(graph, matches);
    return matches;
  }

  constructor(inputProps: GPUBufferSelectionProps<PointGeometry>) {
    validateRequirement(
      inputProps.id ?? 'buffer-selection',
      !('distance' in inputProps),
      'distance was removed; use radius'
    );
    const radius = inputProps.radius;
    validateRequirement(inputProps.id ?? 'buffer-selection', Boolean(radius), 'radius is required');
    const props = Object.freeze({
      ...inputProps,
      radius,
      points: Object.freeze({
        kind: inputProps.points.kind,
        positions: inputProps.points.positions,
        featureOffsets: inputProps.points.featureOffsets,
        sourceIds: inputProps.points.sourceIds
      }),
      features: Object.freeze({...inputProps.features}),
      output: Object.freeze({
        ...inputProps.output,
        ...(inputProps.output.matches
          ? {
              matches: Object.freeze({
                ...inputProps.output.matches,
                status: Object.freeze({...inputProps.output.matches.status})
              })
            }
          : {}),
        ...(inputProps.output.status ? {status: Object.freeze({...inputProps.output.status})} : {})
      })
    });
    validateRequirement(
      props.id ?? 'buffer-selection',
      !('nearestFeatureIds' in props.output),
      'output.nearestFeatureIds is unsupported; use output.nearestFeatureRows'
    );
    this.id = props.id ?? 'buffer-selection';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      BUFFER_SELECTION_SPATIAL_CONTEXT_SUPPORT
    );
    resolveGPUSecondarySpatialContext(
      id,
      'features',
      props.featuresSpatialContext,
      this.spatialContext
    );
    const sphereRadius = getGPUGreatCircleRadius(this.spatialContext);
    if (sphereRadius !== undefined)
      validateNonZeroFiniteNumber(id, 'spatialContext.sphereRadius', sphereRadius);
    const points = props.points.positions;
    const {sourceIds} = props.points;
    const {outputMask, matches, distances, nearestFeatureRows, drawInstanceCount} = props.output;
    const multipart = props.points.featureOffsets !== undefined;
    if (multipart) {
      validateGPUMultipartGeometryPort(id, props.points as GPUMultipartPointGeometryPort);
      validateRequirement(
        id,
        !('data' in points),
        'points.positions multipart queries require packed positions; pack the point parts before selection'
      );
    }
    this.pointCount = multipart ? props.points.featureOffsets!.length - 1 : points.length;
    if (this.pointCount === 0) {
      throw new Error(`${id} needs at least one point`);
    }
    validateCapacity(
      id,
      'points.positions.length',
      this.pointCount,
      1,
      GPU_BUFFER_SELECTION_MAXIMUM_ROWS
    );
    for (const chunk of getGraphViewChunks(points)) {
      validateSinglePackedView(chunk, ['float32x2'], `${id} points.positions`);
    }
    validateSinglePackedView(props.radius, ['float32'], `${id} radius`);
    if (props.radius.length !== 1) {
      throw new Error(`${id} radius needs one float32 row`);
    }
    for (const [name, view] of [
      ['points.sourceIds', sourceIds],
      ['output.outputMask', outputMask],
      ['output.nearestFeatureRows', nearestFeatureRows]
    ] as const) {
      for (const chunk of view ? getGraphViewChunks(view) : []) {
        validateSinglePackedView(chunk, ['uint32'], `${id} ${name}`);
      }
      if (multipart) {
        if (view)
          validateRequirement(
            id,
            !('data' in view) && view.length === this.pointCount,
            `${name} needs one packed row per multipart feature (${this.pointCount})`
          );
      } else validateGraphViewTopology(id, name, points, view);
    }
    for (const chunk of distances ? getGraphViewChunks(distances) : []) {
      validateSinglePackedView(chunk, ['float32'], `${id} output.distances`);
    }
    if (multipart) {
      if (distances)
        validateRequirement(
          id,
          !('data' in distances) && distances.length === this.pointCount,
          `output.distances needs one packed row per multipart feature (${this.pointCount})`
        );
    } else validateGraphViewTopology(id, 'output.distances', points, distances);
    validateStatusScalars(id, {
      overflow: props.output.status?.overflow,
      candidateOverflow: props.output.status?.candidateOverflow
    });
    if (props.output.candidateTruncated) {
      validateSinglePackedView(
        props.output.candidateTruncated,
        ['uint32'],
        `${id} output.candidateTruncated`
      );
      validateRequirement(
        id,
        props.output.candidateTruncated.length === this.pointCount,
        `output.candidateTruncated needs one row per point (${this.pointCount})`
      );
    }
    if (drawInstanceCount) {
      if (!matches) {
        throw new Error(`${id} drawInstanceCount needs matches`);
      }
      validateSinglePackedView(drawInstanceCount, ['uint32'], `${id} output.drawInstanceCount`);
      if (drawInstanceCount.length < 1) {
        throw new Error(`${id} output.drawInstanceCount needs one row`);
      }
    }
    if (matches) {
      validateCompactOutput(id, matches);
      if (matches.ids.length < 1) {
        throw new Error(`${id} output.matches.ids needs at least one row`);
      }
    }
    // Validate nested topology once, before getCommandNodes can allocate adapter storage.
    const {features} = props;
    validateRequirement(
      id,
      features.kind === 'points' || features.kind === 'segments',
      'features must contain points or segments'
    );
    const featureViews =
      features.kind === 'points' ? [features.positions] : [features.starts, features.ends];
    for (const view of featureViews)
      validateSinglePackedView(view, ['float32x2'], `${id} features`);
    this.featureCount = featureViews[0].length;
    validateCapacity(
      id,
      'features.length',
      this.featureCount,
      0,
      GPU_BUFFER_SELECTION_MAXIMUM_ROWS
    );
    validateRequirement(
      id,
      featureViews.every(view => view.length === this.featureCount),
      'features starts and ends must have matching lengths'
    );
    if (features.sourceIds) {
      validateSinglePackedView(features.sourceIds, ['uint32'], `${id} features.sourceIds`);
      validateRequirement(
        id,
        features.sourceIds.length === this.featureCount,
        'features.sourceIds needs one row per feature'
      );
    }
    validateCapacity(id, 'candidateCapacity', props.candidateCapacity);
    if (props.leafCapacity !== undefined) {
      validatePowerOfTwo(id, 'leafCapacity', props.leafCapacity);
    }
    this.leafCapacity = props.leafCapacity ?? getNextPowerOfTwo(Math.max(1, this.featureCount));
    this.spatialSort = props.spatialSort ?? getDefaultSpatialSort(this.featureCount);
    this.querySort = props.querySort ?? getDefaultQuerySort(this.pointCount, this.featureCount);
    validateRequirement(
      id,
      props.spatialSortCurve === undefined ||
        ['hilbert', 'morton'].includes(props.spatialSortCurve),
      'spatialSortCurve must be hilbert or morton'
    );
    validateDisjointOutputs(id, getInputs(props), getOutputs(props), [
      // Preserve the nested join's current writable-output contract before declaring adapters.
      {
        nearestFeatureRows,
        distances,
        overflow: props.output.status?.overflow,
        candidateOverflow: props.output.status?.candidateOverflow,
        candidateTruncated: props.output.candidateTruncated
      },
      {
        ids: matches?.ids,
        count: matches?.status.count,
        requiredCount: matches?.status.requiredCount,
        overflow: matches?.status.overflow,
        drawInstanceCount
      }
    ]);
    validateSpatialJoinReadWriteBindings(id, [nearestFeatureRows], [outputMask]);
    if (matches) {
      validateSpatialJoinReadWriteBindings(
        id,
        [props.output.status?.overflow],
        [matches.status.candidateOverflow]
      );
    }
  }

  /** Returns join, mask, and (with `output`) stable compaction and publish nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const props = {...this.props, output: {...this.props.output, matches: this.getMatches(graph)}};
    if (props.points.featureOffsets) return this.getMultipartCommandNodes(graph, props);
    const {id, pointCount} = this;
    const points = props.points.positions;
    const {sourceIds} = props.points;
    const {outputMask, matches} = props.output;
    const candidateCapacity =
      this.spatialContext.metric === 'great-circle'
        ? GPU_BUFFER_SELECTION_UNUSED_CANDIDATE_CAPACITY
        : props.candidateCapacity;
    const nodes: GPUCommandNode<Parameters>[] = [];

    const nearestFeatureRows =
      props.output.nearestFeatureRows ??
      createTransientUint32Rows(graph, `${id}-nearest-feature-rows`, points);
    const joinOverflow =
      props.output.status?.overflow ??
      createTransientView(graph, `${id}-join-overflow`, 'uint32', 1);
    nodes.push(
      ...new GPUNearestFeatureJoin({
        id: `${id}-join`,
        spatialContext: this.spatialContext,
        featuresSpatialContext: props.featuresSpatialContext,
        queries: {kind: 'points', positions: points},
        features: props.features,
        radius: props.radius,
        candidateCapacity,
        leafCapacity: props.leafCapacity,
        spatialSort: props.spatialSort,
        spatialSortCurve: props.spatialSortCurve,
        querySort: props.querySort,
        output: {
          nearestFeatureRows,
          nearestDistances: props.output.distances,
          candidateTruncated: props.output.candidateTruncated,
          status: {
            overflow: joinOverflow,
            candidateOverflow: props.output.status?.candidateOverflow
          }
        }
      }).getCommandNodes(graph)
    );

    const mask = outputMask ?? createTransientUint32Rows(graph, `${id}-mask`, points);
    const idChunks = getGraphViewChunks(nearestFeatureRows);
    const maskChunks = getGraphViewChunks(mask);
    for (const [chunkIndex, idChunk] of idChunks.entries()) {
      if (idChunk.length === 0) {
        continue;
      }
      nodes.push(
        createGraphChunkGroupKernelNode<Parameters>(graph, {
          id: idChunks.length > 1 ? `${id}-mask-chunk-${chunkIndex}` : `${id}-mask`,
          operation: OPERATION,
          variant: 'mask',
          bindings: [
            {name: 'featureIds', view: idChunk, type: 'u32', access: 'read'},
            {name: 'maskOut', view: maskChunks[chunkIndex], type: 'u32', access: 'read_write'}
          ],
          invocationCount: idChunk.length,
          declarations: `const NO_FEATURE: u32 = ${GPU_SPATIAL_JOIN_NO_FEATURE}u;`,
          body: `maskOut[maskOutOffset + index] =
    select(0u, 1u, featureIds[featureIdsOffset + index] != NO_FEATURE);`
        })
      );
    }

    if (matches) {
      const selectedIds = createTransientView(graph, `${id}-selected-ids`, 'uint32', pointCount);
      const selectedTotal = createTransientView(graph, `${id}-selected-total`, 'uint32', 1);
      nodes.push(
        ...new GPUVisibilityWorkflow({
          id: `${id}-visibility`,
          predicates: [{kind: 'selection', mask: getVisibilityRows(mask)}],
          output: selectedIds,
          count: selectedTotal,
          sourceIds: sourceIds && getVisibilityRows(sourceIds)
        }).getCommandNodes(graph)
      );
      nodes.push(
        createPublishNode<Parameters>(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          requiredCount: selectedTotal,
          compactIds: selectedIds,
          output: {
            ids: matches.ids,
            status: {
              count: matches.status.count,
              overflow: matches.status.overflow,
              requiredCount: matches.status.requiredCount
            }
          },
          extraCounts: props.output.drawInstanceCount ? [props.output.drawInstanceCount] : []
        })
      );
      if (matches.status.candidateOverflow)
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-publish-candidate`,
            operation: OPERATION,
            variant: 'candidate',
            bindings: [
              {name: 'candidateIn', view: joinOverflow, type: 'u32', access: 'read'},
              {
                name: 'candidateOut',
                view: matches.status.candidateOverflow,
                type: 'u32',
                access: 'read_write'
              }
            ],
            invocationCount: 1,
            body: 'candidateOut[candidateOutOffset] = select(0u, 1u, candidateIn[candidateInOffset] != 0u);'
          })
        );
    }
    return nodes;
  }

  /** Feature-level nearest traversal retains the same inclusive radius and stable compaction. */
  private getMultipartCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    props: GPUBufferSelectionProps
  ): GPUCommandNode<Parameters>[] {
    const {id, pointCount} = this;
    const output = props.output;
    const nearestRows =
      (output.nearestFeatureRows as GraphDataView<'uint32'> | undefined) ??
      createTransientView(graph, `${id}-nearest-feature-rows`, 'uint32', pointCount);
    const mask =
      (output.outputMask as GraphDataView<'uint32'> | undefined) ??
      createTransientView(graph, `${id}-mask`, 'uint32', pointCount);
    const overflow =
      output.status?.overflow ?? createTransientView(graph, `${id}-join-overflow`, 'uint32', 1);
    const nodes = [
      ...new GPUNearestFeatureJoin({
        id: `${id}-join`,
        queries: {
          ...props.points,
          sourceIds: undefined
        } as GPUPackedGeometryPort<GPUMultipartPointGeometryPort>,
        features: props.features,
        spatialContext: this.spatialContext,
        featuresSpatialContext: props.featuresSpatialContext,
        radius: props.radius,
        candidateCapacity: props.candidateCapacity,
        leafCapacity: props.leafCapacity,
        spatialSort: props.spatialSort,
        spatialSortCurve: props.spatialSortCurve,
        querySort: props.querySort,
        output: {
          nearestFeatureRows: nearestRows,
          nearestDistances: output.distances as GraphDataView<'float32'> | undefined,
          candidateTruncated: output.candidateTruncated,
          status: {overflow, candidateOverflow: output.status?.candidateOverflow}
        }
      }).getCommandNodes(graph)
    ];
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-mask`,
        operation: OPERATION,
        bindings: [
          {name: 'rows', view: nearestRows, type: 'u32', access: 'read'},
          {name: 'mask', view: mask, type: 'u32', access: 'read_write'}
        ],
        invocationCount: pointCount,
        body: 'mask[maskOffset + index] = select(0u, 1u, rows[rowsOffset + index] != 0xffffffffu);'
      })
    );
    if (output.matches) {
      const selectedIds = createTransientView(graph, `${id}-selected-ids`, 'uint32', pointCount);
      const selectedTotal = createTransientView(graph, `${id}-selected-total`, 'uint32', 1);
      nodes.push(
        ...new GPUVisibilityWorkflow({
          id: `${id}-visibility`,
          predicates: [{kind: 'selection', mask: getVisibilityRows(mask)}],
          output: selectedIds,
          count: selectedTotal,
          sourceIds: props.points.sourceIds && getVisibilityRows(props.points.sourceIds)
        }).getCommandNodes(graph)
      );
      nodes.push(
        createPublishNode(graph, {
          id: `${id}-publish`,
          operation: OPERATION,
          requiredCount: selectedTotal,
          compactIds: selectedIds,
          output: {
            ids: output.matches.ids,
            status: {
              count: output.matches.status.count,
              requiredCount: output.matches.status.requiredCount,
              overflow: output.matches.status.overflow
            }
          },
          extraCounts: output.drawInstanceCount ? [output.drawInstanceCount] : []
        })
      );
      if (output.matches.status.candidateOverflow)
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-publish-candidate`,
            operation: OPERATION,
            bindings: [
              {name: 'inputOverflow', view: overflow, type: 'u32', access: 'read'},
              {
                name: 'outputOverflow',
                view: output.matches.status.candidateOverflow,
                type: 'u32',
                access: 'read_write'
              }
            ],
            invocationCount: 1,
            body: 'outputOverflow[outputOverflowOffset] = inputOverflow[inputOverflowOffset];'
          })
        );
    }
    return nodes;
  }
}

/** Returns every read-only view of a buffer selection. */
function getInputs(
  props: GPUBufferSelectionProps
): (GraphDataView | GraphVectorView | undefined)[] {
  const {features} = props;
  return [
    props.points.positions,
    props.points.featureOffsets,
    props.output.matches ? props.points.sourceIds : undefined,
    props.radius,
    ...(features.kind === 'points' ? [features.positions] : [features.starts, features.ends]),
    features.sourceIds
  ];
}

/** Returns every caller-owned writable view of a buffer selection. */
function getOutputs(
  props: GPUBufferSelectionProps
): (GraphDataView | GraphVectorView | undefined)[] {
  const {output} = props;
  const {matches} = output;
  return [
    output.outputMask,
    output.distances,
    output.nearestFeatureRows,
    output.status?.overflow,
    output.status?.candidateOverflow,
    output.candidateTruncated,
    output.drawInstanceCount,
    matches?.ids,
    matches?.status.count,
    matches?.status.overflow,
    matches?.status.candidateOverflow,
    matches?.status.requiredCount
  ];
}
