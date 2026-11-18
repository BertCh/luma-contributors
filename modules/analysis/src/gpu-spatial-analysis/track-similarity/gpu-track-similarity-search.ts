// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {snapshotSinglePartGeometryPort} from '../../analysis-core/contracts/ports';

import {
  validateTrajectoryBindings,
  validateTrajectoryScratch,
  validateTrajectoryOutputRanges,
  validateTrajectoryWritableDispatches
} from '../trajectory-analysis/trajectory-binding-validation';
import {
  createTransientView,
  doGraphDataViewsOverlap,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPULineGeometryPort, GPUPackedGeometryPort} from '../../analysis-core/contracts/index';
import {
  getGPUGreatCircleRadius,
  GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {createStableLexicographicIndexSortNodes} from '../../analysis-core/kernels/stable-lexicographic-index-sort';
import {getSortKeyBits} from '../../analysis-core/kernels/sorted-segment-sums';
import {GPUTrackSimilarity} from './gpu-track-similarity';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  getTrackSimilarityGreatCircleBoundDeclarations,
  getTrackSimilarityGreatCircleBoundStatements
} from './track-similarity-search-great-circle-bounds';
import {
  getTrackSimilaritySearchBoundStatements,
  TRACK_SIMILARITY_BOUND_DECLARATIONS,
  type TrackSimilaritySearchBoundMetric
} from './track-similarity-search-bounds';
import {
  validateSinglePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs,
  validateCapacity,
  validateFiniteNumber,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Spatial contexts of the search: the same as {@link GPUTrackSimilarity}. */
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_OR_GREAT_CIRCLE_SPATIAL_CONTEXT_SUPPORT,
  measure: 'distance'
} as const);

export type GPUTrackSimilaritySearchProps = {
  id?: string;
  /**
   * Coordinate space of both line sets, as in {@link GPUTrackSimilarityProps.spatialContext}:
   * native planar (the default), or `{coordinateSpace: 'longitude-latitude', metric:
   * 'great-circle'}` with `[longitude, latitude]` degrees, where `threshold` and `distance` are
   * great-circle distances in the context's length unit. The endpoint bound (DTW, Frechet) and the
   * possible-match bound (LCSS) use the haversine distance; the EDR bound is the vertex-count
   * difference, the ERP bound the difference of the gap-distance sums, and Hausdorff and SSPD
   * have no bound in this context (0), so every pair is evaluated.
   */
  spatialContext?: GPUSpatialContext;
  linesA: GPUPackedGeometryPort<GPULineGeometryPort>;
  linesB?: GPUPackedGeometryPort<GPULineGeometryPort>;
  pairA: GraphDataView<'uint32'>;
  pairB: GraphDataView<'uint32'>;
  /**
   * Measure searched. `hausdorff`, `edit-distance-on-real-sequence`,
   * `edit-distance-with-real-penalty` and `symmetric-segment-path` are the measures of the same
   * names in {@link GPUTrackSimilarity}; their lower bounds are in `output.lowerBound`.
   */
  metric:
    | 'dynamic-time-warping'
    | 'frechet'
    | 'longest-common-subsequence'
    | 'hausdorff'
    | 'edit-distance-on-real-sequence'
    | 'edit-distance-with-real-penalty'
    | 'symmetric-segment-path';
  /** Match distance of `edit-distance-on-real-sequence`, as in {@link GPUTrackSimilarityProps}. */
  editDistanceTolerance?: number;
  /** Gap point of `edit-distance-with-real-penalty`, as in {@link GPUTrackSimilarityProps}. */
  editDistanceGapPoint?: readonly [number, number];
  /** Inclusive maximum distance, or minimum LCSS similarity in [0,1]. */
  threshold: GraphDataView<'float32'>;
  maximumFrechetVertexCount?: number;
  dynamicTimeWarpingBandRadius?: number;
  lcssTolerance?: number;
  lcssBandRadius?: number;
  /** Stable GPU compaction bounds dynamic-program scratch. Defaults to pair count. */
  survivorCapacity?: number;
  /** Exact top-k among the supplied threshold-matching pairs, grouped by pairA ID. */
  topK?: number;
  output: {
    distance?: GraphDataView<'float32'>;
    lowerBound?: GraphDataView<'float32'>;
    /** Required for LCSS, normalized matched length / shorter length. */
    similarity?: GraphDataView<'float32'>;
    /** Required for LCSS; possible-match cardinality upper bound. */
    upperBound?: GraphDataView<'float32'>;
    matched: GraphDataView<'uint32'>;
    /** 0 exact evaluated, 1 bound-pruned, 2 invalid pair/data/threshold, 4 dynamic-program capacity exceeded, 8 survivor capacity exceeded. */
    status: GraphDataView<'uint32'>;
    /** Optional [survivor count, evaluated capacity, overflow count]. */
    survivorSummary?: GraphDataView<'uint32'>;
    /** Original pair row indices, ordered by metric then original row; absent slots are 0xffffffff. */
    topKIndices?: GraphDataView<'uint32'>;
    /** Per A trajectory [written count, complete]; invalid or overflow pairs make complete zero. */
    topKSummary?: GraphDataView<'uint32x2'>;
  };
};

/**
 * Batched planar threshold search composing the existing tiled dynamic programs. Endpoint
 * distance is an admissible lower bound for both discrete Frechet and sum-of-distances DTW,
 * including any band. LCSS uses a possible-match cardinality upper bound; evaluating it is O(n*m).
 * The other measures use `O(n + m)` bounds: Hausdorff and SSPD the distance of the vertices to the
 * bounding box of the other track, EDR a count of the vertices that can match inside the box
 * widened by the tolerance, ERP the difference of the two sums of gap distances (ERP is a metric).
 * Pruned pairs skip the measure and have NaN distance, never zero.
 * Input pair IDs are preserved; explicit query batches bound scratch. No approximate pruning.
 */
export class GPUTrackSimilaritySearch implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTrackSimilaritySearchProps;
  readonly value: GraphDataView<'float32'>;
  readonly bound: GraphDataView<'float32'>;
  /** Resolved coordinate context. */
  readonly spatialContext: GPUSpatialContext;
  /** Sphere radius of a great-circle context, `undefined` for planar. */
  readonly sphereRadius: number | undefined;
  constructor(props: GPUTrackSimilaritySearchProps) {
    this.id = props.id ?? 'track-similarity-search';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.sphereRadius = getGPUGreatCircleRadius(this.spatialContext);
    props = {
      ...props,
      linesA: snapshotSinglePartGeometryPort(this.id, 'linesA', props.linesA),
      linesB: props.linesB && snapshotSinglePartGeometryPort(this.id, 'linesB', props.linesB)
    };
    this.props = props;
    const value =
      props.metric === 'longest-common-subsequence'
        ? props.output.similarity
        : props.output.distance;
    const bound =
      props.metric === 'longest-common-subsequence'
        ? props.output.upperBound
        : props.output.lowerBound;
    validateRequirement(
      this.id,
      value !== undefined && bound !== undefined,
      props.metric === 'longest-common-subsequence'
        ? 'output.similarity and output.upperBound are required for longest-common-subsequence'
        : 'output.distance and output.lowerBound are required'
    );
    this.value = value;
    this.bound = bound;
    validateFiniteNumber(this.id, 'lcssTolerance', props.lcssTolerance ?? 1, 0);
    for (const lines of [props.linesA, props.linesB ?? props.linesA]) {
      validateSinglePackedView(lines.positions, ['float32x2'], this.id);
      validateSinglePackedView(lines.lineOffsets, ['uint32'], this.id);
      validateRequirement(
        this.id,
        lines.lineOffsets.length >= 1,
        'lineOffsets must have at least 1 row'
      );
    }
    for (const [name, view] of [
      ['pairA', props.pairA],
      ['pairB', props.pairB],
      ['output.matched', props.output.matched],
      ['output.status', props.output.status]
    ] as const) {
      validateSinglePackedView(view, ['uint32'], this.id);
      validateViewLength(this.id, name, view, props.pairA.length);
    }
    const isSimilarity = props.metric === 'longest-common-subsequence';
    for (const [name, view] of [
      [isSimilarity ? 'output.similarity' : 'output.distance', this.value],
      [isSimilarity ? 'output.upperBound' : 'output.lowerBound', this.bound]
    ] as const) {
      validateSinglePackedView(view, ['float32'], this.id);
      validateViewLength(this.id, name, view, props.pairA.length);
    }
    validateCapacity(
      this.id,
      'survivorCapacity',
      props.survivorCapacity ?? props.pairA.length,
      0,
      props.pairA.length
    );
    if (props.output.survivorSummary) {
      validateSinglePackedView(props.output.survivorSummary, ['uint32'], this.id);
      validateViewLength(this.id, 'output.survivorSummary', props.output.survivorSummary, 3);
    }
    if (props.topK !== undefined) {
      validateCapacity(this.id, 'topK', props.topK, 1, 1024);
      validateRequirement(
        this.id,
        props.output.topKIndices !== undefined && props.output.topKSummary !== undefined,
        'topK needs output.topKIndices and output.topKSummary'
      );
      validateSinglePackedView(props.output.topKIndices, ['uint32'], this.id);
      validateSinglePackedView(props.output.topKSummary, ['uint32x2'], this.id);
      validateViewLength(
        this.id,
        'output.topKIndices',
        props.output.topKIndices,
        (props.linesA.lineOffsets.length - 1) * props.topK
      );
      validateViewLength(
        this.id,
        'output.topKSummary',
        props.output.topKSummary,
        props.linesA.lineOffsets.length - 1
      );
    } else {
      validateRequirement(
        this.id,
        !props.output.topKIndices && !props.output.topKSummary,
        'output.topKIndices and output.topKSummary need topK'
      );
    }
    validateSinglePackedView(props.threshold, ['float32'], this.id);
    validateViewLength(this.id, 'threshold', props.threshold, 1);
    // Separate passes cannot preserve independently named outputs in overlapping storage.
    const outputs = Object.values(props.output).filter(
      (view): view is NonNullable<typeof view> => view !== undefined
    );
    for (const [index, output] of outputs.entries()) {
      validateRequirement(
        this.id,
        !outputs.slice(0, index).some(previous => doGraphDataViewsOverlap(previous, output)),
        'outputs must not overlap each other'
      );
    }
    validateTrajectoryOutputRanges(this.id, this.props.output);
    validateTrajectoryWritableDispatches(this.id, [
      {value, status: props.output.status, matched: props.output.matched},
      {topKIndices: props.output.topKIndices, topKSummary: props.output.topKSummary}
    ]);
    validateGraphOutputsDisjointFromInputs(this.id, [props.output.status], [bound]);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.linesA.positions,
      props.linesA.lineOffsets,
      props.linesB?.positions,
      props.linesB?.lineOffsets,
      props.pairA,
      props.pairB,
      props.threshold
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateTrajectoryBindings(graph, this.id, this.props);
    const {id, props, value, bound, sphereRadius} = this;
    const subsequence = props.metric === 'longest-common-subsequence';
    const isBoxBounded =
      props.metric === 'hausdorff' ||
      props.metric === 'edit-distance-on-real-sequence' ||
      props.metric === 'edit-distance-with-real-penalty' ||
      props.metric === 'symmetric-segment-path';
    const linesB = props.linesB ?? props.linesA;
    const distanceExpression =
      sphereRadius === undefined
        ? 'sqrt(dot(loadA(rowA) - loadB(rowB), loadA(rowA) - loadB(rowB)))'
        : 'pointDistance(loadA(rowA), loadB(rowB))';
    // Hausdorff and SSPD have no dynamic program, so the shorter-track cap does not apply.
    const isUncapped = props.metric === 'hausdorff' || props.metric === 'symmetric-segment-path';
    validateGraphViewsBelongToGraph(id, graph, [
      props.linesA.positions,
      props.linesA.lineOffsets,
      linesB.positions,
      linesB.lineOffsets,
      props.pairA,
      props.pairB,
      props.threshold,
      ...Object.values(props.output)
    ]);
    const capacity = props.survivorCapacity ?? props.pairA.length;
    validateTrajectoryScratch(graph, id, {
      'survivor rows': capacity * 4,
      'survivor summary': 12,
      'pair scratch': props.pairA.length * 4,
      ...(props.topK ? {'sort histograms': 16 * Math.ceil(props.pairA.length / 256) * 4} : {})
    });
    const filteredA = createTransientView(graph, `${id}-filtered-a`, 'uint32', capacity);
    const filteredB = createTransientView(graph, `${id}-filtered-b`, 'uint32', capacity);
    const flags = createTransientView(graph, `${id}-flags`, 'uint32', props.pairA.length);
    const ranks = createTransientView(graph, `${id}-ranks`, 'uint32', props.pairA.length);
    const compactValue = createTransientView(graph, `${id}-compact-value`, 'float32', capacity);
    const survivorSummary =
      props.output.survivorSummary ??
      createTransientView(graph, `${id}-survivor-summary`, 'uint32', 3);
    const programStatus = createTransientView(graph, `${id}-program-status`, 'uint32', capacity);
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-bound`,
        operation: 'GPUTrackSimilaritySearch',
        variant: subsequence
          ? 'possible-match-bound'
          : isBoxBounded
            ? `${props.metric}-bound`
            : 'endpoint-bound',
        invocationCount: props.pairA.length,
        bindings: [
          {name: 'positionsA', view: props.linesA.positions, type: 'f32', access: 'read'},
          {name: 'offsetsA', view: props.linesA.lineOffsets, type: 'u32', access: 'read'},
          {name: 'positionsB', view: linesB.positions, type: 'f32', access: 'read'},
          {name: 'offsetsB', view: linesB.lineOffsets, type: 'u32', access: 'read'},
          {name: 'pairA', view: props.pairA, type: 'u32', access: 'read'},
          {name: 'pairB', view: props.pairB, type: 'u32', access: 'read'},
          {name: 'bounds', view: bound, type: 'f32', access: 'read_write'}
        ],
        declarations: `fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn loadA(row: u32) -> vec2<f32> { return vec2<f32>(positionsA[positionsAOffset + 2u * row], positionsA[positionsAOffset + 2u * row + 1u]); }
fn loadB(row: u32) -> vec2<f32> { return vec2<f32>(positionsB[positionsBOffset + 2u * row], positionsB[positionsBOffset + 2u * row + 1u]); }${
          sphereRadius === undefined
            ? TRACK_SIMILARITY_BOUND_DECLARATIONS
            : getTrackSimilarityGreatCircleBoundDeclarations(sphereRadius)
        }`,
        body: `var nanBits = 0x7fc00000u; bounds[boundsOffset + index] = bitcast<f32>(nanBits);
let first = pairA[pairAOffset + index]; let second = pairB[pairBOffset + index];
if (first >= ${props.linesA.lineOffsets.length - 1}u || second >= ${linesB.lineOffsets.length - 1}u) { return; }
let startA = offsetsA[offsetsAOffset + first]; let endA = offsetsA[offsetsAOffset + first + 1u]; let startB = offsetsB[offsetsBOffset + second]; let endB = offsetsB[offsetsBOffset + second + 1u];
if (startA >= endA || endA > ${props.linesA.positions.length}u || startB >= endB || endB > ${linesB.positions.length}u) { return; }
for (var row = startA; row < endA; row++) { let position = loadA(row); if (!finite(position.x) || !finite(position.y)) { return; } }
for (var row = startB; row < endB; row++) { let position = loadB(row); if (!finite(position.x) || !finite(position.y)) { return; } }
${sphereRadius === undefined ? 'let startDifference = loadA(startA) - loadB(startB); let endDifference = loadA(endA - 1u) - loadB(endB - 1u);' : ''}
${
  isBoxBounded
    ? sphereRadius === undefined
      ? getTrackSimilaritySearchBoundStatements(
          props.metric as TrackSimilaritySearchBoundMetric,
          props.editDistanceTolerance ?? 1,
          props.editDistanceGapPoint ?? [0, 0]
        )
      : getTrackSimilarityGreatCircleBoundStatements(
          props.metric as TrackSimilaritySearchBoundMetric,
          props.editDistanceGapPoint ?? [0, 0]
        )
    : subsequence
      ? `var possibleA = 0u; var possibleB = 0u;
for (var rowA = startA; rowA < endA; rowA++) {
  var possible = false;
  for (var rowB = startB; rowB < endB; rowB++) { if (${distanceExpression} <= f32(${props.lcssTolerance ?? 1})) { possible = true; break; } }
  if (possible) { possibleA++; }
}
for (var rowB = startB; rowB < endB; rowB++) {
  var possible = false;
  for (var rowA = startA; rowA < endA; rowA++) { if (${distanceExpression} <= f32(${props.lcssTolerance ?? 1})) { possible = true; break; } }
  if (possible) { possibleB++; }
}
bounds[boundsOffset + index] = f32(min(possibleA, possibleB)) / f32(min(endA - startA, endB - startB));`
      : sphereRadius === undefined
        ? 'bounds[boundsOffset + index] = max(sqrt(dot(startDifference, startDifference)), sqrt(dot(endDifference, endDifference)));'
        : 'bounds[boundsOffset + index] = max(pointDistance(loadA(startA), loadB(startB)), pointDistance(loadA(endA - 1u), loadB(endB - 1u)));'
}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-prune`,
        operation: 'GPUTrackSimilaritySearch',
        variant: 'prune-pairs',
        invocationCount: props.pairA.length,
        bindings: [
          {name: 'bounds', view: bound, type: 'f32', access: 'read'},
          {name: 'threshold', view: props.threshold, type: 'f32', access: 'read'},
          {name: 'pairA', view: props.pairA, type: 'u32', access: 'read'},
          {name: 'pairB', view: props.pairB, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        body: `let bound = bounds[boundsOffset + index]; let limit = threshold[thresholdOffset];
let valid = (bitcast<u32>(bound) & 0x7f800000u) != 0x7f800000u && (bitcast<u32>(limit) & 0x7f800000u) != 0x7f800000u && limit >= 0.0${subsequence ? ' && limit <= 1.0' : ''};
// The small conservative margin retains threshold-neighbor pairs under f32 roundoff.
let pruned = valid && ${subsequence ? 'bound < limit - 0.000001' : 'bound > limit + max(bound, 1.0) * 0.000001'};
status[statusOffset + index] = select(select(0u, 1u, pruned), 2u, !valid);
flags[flagsOffset + index] = select(0u, 1u, valid && !pruned);`
      })
    ];
    nodes.push(
      ...new GPUScan({
        id: `${id}-survivor-scan`,
        input: flags,
        output: ranks,
        mode: 'exclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clear-survivors`,
        operation: 'GPUTrackSimilaritySearch',
        variant: 'clear-survivors',
        invocationCount: Math.max(capacity, 1),
        bindings: [
          {name: 'filteredA', view: filteredA, type: 'u32', access: 'read_write'},
          {name: 'filteredB', view: filteredB, type: 'u32', access: 'read_write'},
          {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
          {name: 'flags', view: flags, type: 'u32', access: 'read'},
          {name: 'summary', view: survivorSummary, type: 'u32', access: 'read_write'}
        ],
        body: `if (index < ${capacity}u) { filteredA[filteredAOffset + index] = 0xffffffffu; filteredB[filteredBOffset + index] = 0xffffffffu; }
if (index == 0u) {
  let count = ${props.pairA.length ? `ranks[ranksOffset + ${props.pairA.length - 1}u] + flags[flagsOffset + ${props.pairA.length - 1}u]` : '0u'};
  summary[summaryOffset] = count; summary[summaryOffset + 1u] = min(count, ${capacity}u); summary[summaryOffset + 2u] = count - min(count, ${capacity}u);
}`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-compact`,
        operation: 'GPUTrackSimilaritySearch',
        variant: 'stable-survivors',
        invocationCount: props.pairA.length,
        bindings: [
          {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          {name: 'pairA', view: props.pairA, type: 'u32', access: 'read'},
          {name: 'pairB', view: props.pairB, type: 'u32', access: 'read'},
          {name: 'filteredA', view: filteredA, type: 'u32', access: 'read_write'},
          {name: 'filteredB', view: filteredB, type: 'u32', access: 'read_write'}
        ],
        body: `if (status[statusOffset + index] != 0u) { return; } let rank = ranks[ranksOffset + index];
if (rank >= ${capacity}u) { status[statusOffset + index] = 8u; return; }
filteredA[filteredAOffset + rank] = pairA[pairAOffset + index]; filteredB[filteredBOffset + rank] = pairB[pairBOffset + index];`
      })
    );
    if (capacity > 0)
      nodes.push(
        ...new GPUTrackSimilarity({
          id: `${id}-metric`,
          spatialContext: props.spatialContext,
          linesA: props.linesA,
          linesB,
          pairA: filteredA,
          pairB: filteredB,
          maximumFrechetVertexCount: props.maximumFrechetVertexCount,
          dynamicTimeWarpingBandRadius: props.dynamicTimeWarpingBandRadius,
          lcssTolerance: props.lcssTolerance,
          lcssBandRadius: props.lcssBandRadius,
          editDistanceTolerance: props.editDistanceTolerance,
          editDistanceGapPoint: props.editDistanceGapPoint,
          output: {
            hausdorff: props.metric === 'hausdorff' ? compactValue : undefined,
            editDistanceOnRealSequence:
              props.metric === 'edit-distance-on-real-sequence' ? compactValue : undefined,
            editDistanceWithRealPenalty:
              props.metric === 'edit-distance-with-real-penalty' ? compactValue : undefined,
            symmetricSegmentPathDistance:
              props.metric === 'symmetric-segment-path' ? compactValue : undefined,
            frechet: props.metric === 'frechet' ? compactValue : undefined,
            dynamicTimeWarping: props.metric === 'dynamic-time-warping' ? compactValue : undefined,
            longestCommonSubsequence: subsequence ? compactValue : undefined,
            pairStatus: programStatus
          }
        }).getCommandNodes(graph)
      );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish`,
        operation: 'GPUTrackSimilaritySearch',
        variant: 'threshold-match',
        invocationCount: props.pairA.length,
        bindings: [
          {name: 'distance', view: value, type: 'f32', access: 'read_write'},
          {name: 'compactValue', view: compactValue, type: 'f32', access: 'read'},
          {name: 'ranks', view: ranks, type: 'u32', access: 'read'},
          {name: 'threshold', view: props.threshold, type: 'f32', access: 'read'},
          {name: 'programStatus', view: programStatus, type: 'u32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          {name: 'matched', view: props.output.matched, type: 'u32', access: 'read_write'}
        ],
        body: `var nanBits = 0x7fc00000u; distance[distanceOffset + index] = bitcast<f32>(nanBits);
if (status[statusOffset + index] == 0u) { let rank = ranks[ranksOffset + index];
  distance[distanceOffset + index] = compactValue[compactValueOffset + rank];
  if ((programStatus[programStatusOffset + rank] & ${isUncapped ? '5u' : '0xffffffffu'}) != 0u) { status[statusOffset + index] = select(2u, 4u, (programStatus[programStatusOffset + rank] & 2u) != 0u); }
}
matched[matchedOffset + index] = select(0u, 1u, status[statusOffset + index] == 0u && distance[distanceOffset + index] ${subsequence ? '>=' : '<='} threshold[thresholdOffset]);`
      })
    );
    if (props.topK !== undefined && props.output.topKIndices && props.output.topKSummary) {
      nodes.push(...this.createTopKNodes(graph, value, subsequence));
    }
    return nodes;
  }

  /**
   * Segmented select: one stable sort of the pair rows by `(pairA, value)`, then one thread per A
   * track binary-searches its segment and copies the first `topK` rows. Replaces the scan of every
   * pair for each of the K slots with the check of the previous picks.
   */
  private createTopKNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    value: GraphDataView<'float32'>,
    subsequence: boolean
  ): GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const topK = props.topK!;
    const trackCount = props.linesA.lineOffsets.length - 1;
    const pairCount = props.pairA.length;
    const identity = createTransientView(graph, `${id}-top-k-identity`, 'uint32', pairCount);
    const trackKeys = createTransientView(graph, `${id}-top-k-track-keys`, 'uint32', pairCount);
    const valueKeys = createTransientView(graph, `${id}-top-k-value-keys`, 'uint32', pairCount);
    const sort = createStableLexicographicIndexSortNodes(graph, {
      id: `${id}-top-k-sort`,
      operation: 'GPUTrackSimilaritySearch',
      indices: identity,
      keys: [{view: trackKeys, keyBits: getSortKeyBits(trackCount)}, {view: valueKeys}]
    });
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-top-k-keys`,
        operation: 'GPUTrackSimilaritySearch',
        variant: 'top-k-sort-keys',
        invocationCount: pairCount,
        bindings: [
          {name: 'pairA', view: props.pairA, type: 'u32', access: 'read'},
          {name: 'values', view: value, type: 'f32', access: 'read'},
          {name: 'matched', view: props.output.matched, type: 'u32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read'},
          {name: 'identity', view: identity, type: 'u32', access: 'read_write'},
          {name: 'trackKeys', view: trackKeys, type: 'u32', access: 'read_write'},
          {name: 'valueKeys', view: valueKeys, type: 'u32', access: 'read_write'}
        ],
        // Float order as an unsigned key (negative values flip every bit, others the sign bit);
        // `+ 0.0` turns -0 into +0 so equal values tie. Matched rows sort by value (descending
        // similarity for LCSS), then rows with an error status (0xfffffffe), then the rest. Ties
        // keep the pair row order because the sort is stable.
        body: `identity[identityOffset + index] = index;
let track = pairA[pairAOffset + index];
trackKeys[trackKeysOffset + index] = select(${trackCount}u, track, track < ${trackCount}u);
let bits = bitcast<u32>(values[valuesOffset + index] + 0.0);
var ordered = bits ^ select(0x80000000u, 0xffffffffu, (bits & 0x80000000u) != 0u);
${subsequence ? 'ordered = ~ordered;' : ''}
var key = 0xffffffffu;
if (matched[matchedOffset + index] != 0u) { key = ordered; }
else if (status[statusOffset + index] > 1u) { key = 0xfffffffeu; }
valueKeys[valueKeysOffset + index] = key;`
      }),
      ...sort.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-top-k-select`,
        operation: 'GPUTrackSimilaritySearch',
        variant: 'segmented-top-k',
        invocationCount: trackCount,
        bindings: [
          {name: 'sortedIndices', view: sort.sortedIndices, type: 'u32', access: 'read'},
          {name: 'sortedTracks', view: sort.sortedPrimaryKeys, type: 'u32', access: 'read'},
          {name: 'valueKeys', view: valueKeys, type: 'u32', access: 'read'},
          {name: 'selected', view: props.output.topKIndices!, type: 'u32', access: 'read_write'},
          {name: 'summary', view: props.output.topKSummary!, type: 'u32', access: 'read_write'}
        ],
        // First sorted position whose (track, value key) is not below the given one.
        declarations: `fn lowerBound(track: u32, key: u32) -> u32 {
  var low = 0u; var high = ${pairCount}u;
  while (low < high) {
    let middle = low + (high - low) / 2u;
    let middleTrack = sortedTracks[sortedTracksOffset + middle];
    let middleKey = valueKeys[valueKeysOffset + sortedIndices[sortedIndicesOffset + middle]];
    if (middleTrack < track || (middleTrack == track && middleKey < key)) { low = middle + 1u; } else { high = middle; }
  }
  return low;
}`,
        body: `let first = lowerBound(index, 0u);
let matchedEnd = lowerBound(index, 0xfffffffeu);
let errorEnd = lowerBound(index, 0xffffffffu);
let count = min(matchedEnd - first, ${topK}u);
for (var slot = 0u; slot < ${topK}u; slot++) {
  selected[selectedOffset + index * ${topK}u + slot] = select(0xffffffffu, sortedIndices[sortedIndicesOffset + first + slot], slot < count);
}
summary[summaryOffset + index * 2u] = count; summary[summaryOffset + index * 2u + 1u] = select(1u, 0u, errorEnd > matchedEnd);`
      })
    ];
  }
}
