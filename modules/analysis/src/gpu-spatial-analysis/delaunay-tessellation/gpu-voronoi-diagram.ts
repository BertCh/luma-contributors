// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {clamp} from '@math.gl/core';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  createTransientView,
  GPUCompaction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateGPUStorageBindingByteLength,
  validateCapacity,
  validateRequirement,
  validateGraphOutputsDisjointFromInputs,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createStableLexicographicIndexSortNodes} from '../../analysis-core/kernels/stable-lexicographic-index-sort';
import {createFillNode, createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGPUStatusPort,
  type GPUBoundedResultStatusPort
} from '../../analysis-core/contracts/status';
import {EXACT_ORIENTATION_WGSL} from '../../analysis-core/kernels/exact-orientation-wgsl';
import {
  TRIANGLE_CIRCUMCIRCLE_WGSL,
  TRIANGLE_SCALE_WGSL
} from '../shared/triangle-circumcircle-wgsl';

import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateDelaunayLogicalOutputs} from './gpu-delaunay-tessellation';

/** Four endpoint words per candidate, with three candidate edges per input triangle. */
const VORONOI_MAXIMUM_TRIANGLE_COUNT = Math.floor(0xffffffff / 12);
/** Four float32 endpoint words per output segment use uint32 word addresses. */
const VORONOI_MAXIMUM_SEGMENT_COUNT = Math.floor(0xffffffff / 4);

function getVoronoiOutputViews(output: GPUVoronoiDiagramOutput) {
  return {
    segments: output.segments,
    siteIds: output.siteIds,
    ...output.status
  };
}

/** Compact Voronoi edge segments, clipped to `clipBounds`, derived from Delaunay triangle adjacency. */
export type GPUVoronoiDiagramOutput = {
  /** Segment endpoints `(startX, startY, endX, endY)`. */
  segments: GraphDataView<'float32x4'>;
  /**
   * The two Delaunay sites whose bisector owns each segment. To build positive-length
   * adjacency, inspect only rows below status.count and retain a pair only when
   * startX !== endX or startY !== endY in the corresponding segments row.
   */
  siteIds: GraphDataView<'uint32x2'>;
  /** `complete` (optional) is 1 only with no upstream, invalid-triangle or final capacity loss.
   * `converged` reports upstream solver/triangle validity independently of final segment capacity.
   * `unresolvedCount` is unsupported and rejected at construction. */
  status: GPUBoundedResultStatusPort & {requiredCount: GraphDataView<'uint32'>};
};

/** Properties for {@link GPUVoronoiDiagram}. */
export type GPUVoronoiDiagramProps = {
  id?: string;
  /**
   * Planar only. Spatial context of `positions`; omitted means native planar coordinates.
   * Longitude/latitude (or any other non-planar) contexts throw at construction: project
   * positions to a planar frame first, for example with `createGPULocalFrame` and
   * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
   */
  spatialContext?: GPUSpatialContext;
  positions: GraphDataView<'float32x2'>;
  /** Delaunay triples of either winding; only the readable prefix selected by `triangleCount` is read. */
  triangles: GraphDataView<'uint32x3'>;
  /** Requested rows; exceeding triangles.length sets candidateOverflow and clears converged. */
  triangleCount: GraphDataView<'uint32'>;
  /** Upstream completeness. Capacity loss or converged=0 suppresses segments. Only capacity loss sets
   * output overflow/candidateOverflow; solver failure clears output converged and complete when supplied. */
  triangleStatus?: {overflow?: GraphDataView<'uint32'>; converged?: GraphDataView<'uint32'>};
  /** One dynamic `[minimumX, minimumY, maximumX, maximumY]` row for hull-ray clipping. */
  clipBounds: GraphDataView<'float32x4'>;
  output: GPUVoronoiDiagramOutput;
};

/** Maximum number of Voronoi segments emitted from a triangle capacity. */
export function getVoronoiMaximumSegmentCount(triangleCapacity: number): number {
  if (!Number.isSafeInteger(triangleCapacity) || triangleCapacity < 0) {
    throw new Error('triangleCapacity must be a non-negative safe integer');
  }
  return 3 * triangleCapacity;
}

/**
 * Derives vector Voronoi edges from accepted Delaunay topology entirely on the GPU.
 *
 * Internal edges connect adjacent triangle circumcenters. Hull edges extend the outward bisector
 * to the dynamic clip bounds. Every segment is Liang-Barsky clipped to `clipBounds`; segments that
 * miss the bounds entirely are dropped. Triangle and local-edge order provide stable compact output order;
 * each internal edge is emitted by the lower triangle row. Triangle adjacency comes from one stable
 * sort of undirected edge keys, so the build costs a sort rather than a scan of every triangle per edge. Exactly collinear triangles, and
 * triangles whose final circumcenter is unrepresentable in f32, are rejected and counted in `invalidCount` when that
 * canonical status field is supplied.
 * Circumcircle edge equations are independently scaled by powers of two, preserving unequal edge
 * magnitudes; centers are restored after adding the origin, independently of radius overflow.
 * Both clockwise and counter-clockwise input triples emit outward hull rays, clipped without a
 * fixed distance cap. Edge differences that overflow are formed from scaled endpoints.
 *
 * Edges come only from triangles. Two sites, or any set of collinear sites, have no Delaunay
 * triangle, so the output is empty (`count` 0, `invalidCount` 0) although their Voronoi diagram is
 * not: two sites are separated by one bisector and collinear sites by parallel bisectors, which
 * `shapely.voronoi_polygons` returns. Callers that need those cases must build the bisectors of
 * consecutive sites themselves.
 *
 * Upstream triangle capacity loss suppresses segments and sets overflow/candidateOverflow.
 * Upstream solver nonconvergence also suppresses segments, but leaves both capacity words clear
 * when storage suffices. Inspect optional converged or complete to distinguish this from a
 * successful empty diagram. complete also accounts for local segment capacity and invalid triangles.
 *
 * Planar only: `spatialContext` defaults to native planar coordinates and a longitude/latitude
 * context throws at construction. Project first, for example with `createGPULocalFrame` and
 * `GPULocalFrameProjection`, and pass the frame's context so its `frame` identity is kept.
 */
export class GPUVoronoiDiagram implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUVoronoiDiagramProps;
  /** Resolved spatial context (planar). */
  readonly spatialContext: GPUSpatialContext;

  constructor(props: GPUVoronoiDiagramProps) {
    const {
      id,
      spatialContext,
      positions,
      triangles,
      triangleCount,
      triangleStatus,
      clipBounds,
      output
    } = props;
    const {segments, siteIds, status} = output;
    props = {
      id,
      spatialContext,
      positions,
      triangles,
      triangleCount,
      clipBounds,
      triangleStatus: triangleStatus && {
        overflow: triangleStatus.overflow,
        converged: triangleStatus.converged
      },
      output: {
        segments,
        siteIds,
        status: {
          complete: status.complete,
          unresolvedCount: status.unresolvedCount,
          count: status.count,
          requiredCount: status.requiredCount,
          overflow: status.overflow,
          invalidCount: status.invalidCount,
          candidateOverflow: status.candidateOverflow,
          uncertainCount: status.uncertainCount,
          iterationCount: status.iterationCount,
          converged: status.converged,
          iterationLimitReached: status.iterationLimitReached,
          approximation: status.approximation
        }
      }
    };
    this.id = props.id ?? 'voronoi-diagram';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(this.id, props.spatialContext);
    const contributorId = this.id;
    this.validateProps(contributorId, props);
  }

  private validateProps(id: string, props: GPUVoronoiDiagramProps): void {
    validateRequirement(
      id,
      props.output.status.unresolvedCount === undefined,
      'output.status.unresolvedCount is unsupported; use output.status.converged or complete'
    );
    if (props.triangleStatus) validateGPUStatusPort(`${id} triangleStatus`, props.triangleStatus);
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    validateSinglePackedView(props.triangles, ['uint32x3'], `${id} triangles`);
    validateSinglePackedView(props.triangleCount, ['uint32'], `${id} triangleCount`);
    if (props.triangleCount.length < 1) {
      throw new Error(`${id} triangleCount must contain one row`);
    }
    validateSinglePackedView(props.clipBounds, ['float32x4'], `${id} clipBounds`);
    if (props.clipBounds.length !== 1) {
      throw new Error(`${id} clipBounds must contain one row`);
    }
    validateSinglePackedView(props.output.segments, ['float32x4'], `${id} output.segments`);
    validateSinglePackedView(props.output.siteIds, ['uint32x2'], `${id} output.siteIds`);
    if (props.output.segments.length !== props.output.siteIds.length) {
      throw new Error(`${id} output segment and site capacities must match`);
    }
    validateGPUStatusPort(`${id} output.status`, props.output.status);
    validateRequirement(
      id,
      Boolean(props.output.status.requiredCount),
      'output.status.requiredCount is required'
    );
    validateCapacity(
      id,
      'triangles rows',
      props.triangles.length,
      0,
      VORONOI_MAXIMUM_TRIANGLE_COUNT
    );
    validateCapacity(
      id,
      'output.segments rows',
      props.output.segments.length,
      0,
      VORONOI_MAXIMUM_SEGMENT_COUNT
    );
    const views = getVoronoiOutputViews(props.output);
    validateDelaunayLogicalOutputs(id, views);
    validateDisjointWritableViews(id, {
      requiredCount: views.requiredCount,
      count: views.count,
      overflow: views.overflow
    });
    validateDisjointWritableViews(id, {
      requiredCount: views.requiredCount,
      segments: views.segments
    });
    validateDisjointWritableViews(id, {requiredCount: views.requiredCount, siteIds: views.siteIds});
    validateDisjointWritableViews(id, {
      invalidCount: props.output.status.invalidCount,
      converged: props.output.status.converged
    });
    // Disjoint aligned ranges still cannot share read/read_write physical-buffer usages.
    validateGraphOutputsDisjointFromInputs(
      id,
      {
        'output.status.count': props.output.status.count,
        'output.status.overflow': props.output.status.overflow
      },
      {'output.status.requiredCount': props.output.status.requiredCount}
    );
    validateGraphOutputsDisjointFromInputs(
      id,
      {'output.status.converged': props.output.status.converged},
      {'output.status.invalidCount': props.output.status.invalidCount}
    );
    validateGraphOutputsDisjointFromInputs(
      id,
      {'output.status.complete': props.output.status.complete},
      {
        'output.status.invalidCount': props.output.status.invalidCount,
        'output.status.requiredCount': props.output.status.requiredCount
      }
    );
    validateGraphOutputsDisjointFromInputs(id, Object.values(views), [
      props.positions,
      props.triangles,
      props.triangleCount,
      props.clipBounds,
      props.triangleStatus?.overflow,
      props.triangleStatus?.converged
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output} = props;
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      triangles: props.triangles,
      triangleCount: props.triangleCount,
      clipBounds: props.clipBounds,
      triangleOverflow: props.triangleStatus?.overflow,
      triangleConverged: props.triangleStatus?.converged,
      ...getVoronoiOutputViews(output)
    });
    // Endpoint candidates are the largest derived binding. Centers, three edge-key columns,
    // adjacency, compaction, sort padding and scan summaries are all smaller.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'candidate segment scratch',
      Math.max(props.triangles.length * 48, 4),
      'reduce triangles rows'
    );
    const triangleCapacity = props.triangles.length;
    const inputControl = props.triangleStatus
      ? createTransientView(graph, `${id}-input-control`, 'uint32', 2)
      : undefined;
    const readableTriangleCount = inputControl
      ? graph.createDataView(inputControl.buffer, {
          format: 'uint32',
          length: 1,
          byteOffset: inputControl.byteOffset
        })
      : props.triangleCount;
    const upstreamIncomplete =
      inputControl &&
      graph.createDataView(inputControl.buffer, {
        format: 'uint32',
        length: 1,
        byteOffset: inputControl.byteOffset + 4
      });
    const completenessBindings = upstreamIncomplete
      ? [
          {
            name: 'upstreamIncomplete',
            view: upstreamIncomplete,
            type: 'u32' as const,
            access: 'read' as const
          }
        ]
      : [];
    const incompleteExpression = upstreamIncomplete
      ? 'upstreamIncomplete[upstreamIncompleteOffset] != 0u'
      : 'false';
    const capacityLossExpression = upstreamIncomplete
      ? '(upstreamIncomplete[upstreamIncompleteOffset] & 1u) != 0u'
      : 'false';
    const candidateCapacity = 3 * triangleCapacity;
    const centers = createTransientView(graph, `${id}-centers`, 'float32', 3 * triangleCapacity);
    const candidateIds = createTransientView(
      graph,
      `${id}-candidate-ids`,
      'uint32',
      candidateCapacity
    );
    const candidateFlags = createTransientView(
      graph,
      `${id}-candidate-flags`,
      'uint32',
      candidateCapacity
    );
    // Undirected edge key per candidate half-edge; rows with an out-of-range triangle or vertex get
    // the sentinel `(pointCount, pointCount)` and sort last.
    const edgeMinimumKeys = createTransientView(
      graph,
      `${id}-edge-minimum-keys`,
      'uint32',
      candidateCapacity
    );
    const edgeMaximumKeys = createTransientView(
      graph,
      `${id}-edge-maximum-keys`,
      'uint32',
      candidateCapacity
    );
    const keyBits = clamp(32 - Math.clz32(props.positions.length), 1, 32);
    const edgeSort = createStableLexicographicIndexSortNodes<Parameters>(graph, {
      id: `${id}-edge-sort`,
      operation: 'GPUVoronoiDiagram',
      indices: candidateIds,
      keys: [
        {view: edgeMinimumKeys, keyBits},
        {view: edgeMaximumKeys, keyBits}
      ]
    });
    // Neighbor triangle per candidate edge, found once in classification and reused by materialization.
    const neighbors = createTransientView(graph, `${id}-neighbors`, 'uint32', candidateCapacity);
    // Clipped endpoints per candidate half-edge, written before compaction so that segments that
    // lie entirely outside the clip bounds can be dropped.
    const candidateSegments = createTransientView(
      graph,
      `${id}-candidate-segments`,
      'float32',
      4 * candidateCapacity
    );
    const compactCandidateIds = createTransientView(
      graph,
      `${id}-compact-candidate-ids`,
      'uint32',
      candidateCapacity
    );
    const invalidCount =
      output.status.invalidCount ?? createTransientView(graph, `${id}-invalid-count`, 'uint32', 1);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...(inputControl
        ? [
            createWGSLKernelNode<Parameters>(graph, {
              id: `${id}-prepare-input`,
              operation: 'GPUVoronoiDiagram',
              invocationCount: 1,
              bindings: [
                {name: 'requested', view: props.triangleCount, type: 'u32', access: 'read'},
                ...(props.triangleStatus?.overflow
                  ? [
                      {
                        name: 'upstreamOverflow',
                        view: props.triangleStatus.overflow,
                        type: 'u32' as const,
                        access: 'read' as const
                      }
                    ]
                  : []),
                ...(props.triangleStatus?.converged
                  ? [
                      {
                        name: 'upstreamConverged',
                        view: props.triangleStatus.converged,
                        type: 'u32' as const,
                        access: 'read' as const
                      }
                    ]
                  : []),
                {name: 'control', view: inputControl, type: 'u32', access: 'read_write'}
              ],
              body: `let requestedCount = requested[requestedOffset];
let capacityLoss = requestedCount > ${triangleCapacity}u ${props.triangleStatus?.overflow ? '|| upstreamOverflow[upstreamOverflowOffset] != 0u' : ''};
let solverLoss = ${props.triangleStatus?.converged ? 'upstreamConverged[upstreamConvergedOffset] == 0u' : 'false'};
let incomplete = capacityLoss || solverLoss;
control[controlOffset] = select(requestedCount, 0u, incomplete);
control[controlOffset + 1u] = select(0u, 1u, capacityLoss) | select(0u, 2u, solverLoss);`
            })
          ]
        : []),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-initialize`,
        operation: 'GPUVoronoiDiagram',
        variant: 'initialize',
        bindings: [{name: 'invalidCount', view: invalidCount, type: 'u32', access: 'read_write'}],
        invocationCount: 1,
        body: 'invalidCount[invalidCountOffset] = 0u;'
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-circumcenters`,
        operation: 'GPUVoronoiDiagram',
        variant: 'parallel-circumcenters',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'triangles', view: props.triangles, type: 'u32', access: 'read'},
          {name: 'triangleCountIn', view: readableTriangleCount, type: 'u32', access: 'read'},
          {name: 'centers', view: centers, type: 'f32', access: 'read_write'},
          {name: 'invalidCount', view: invalidCount, type: 'atomic<u32>', access: 'read_write'}
        ],
        invocationCount: triangleCapacity,
        declarations: getVoronoiWGSL({
          pointCount: props.positions.length,
          triangleCapacity
        }),
        body: `let triangleCount = min(triangleCountIn[triangleCountInOffset], TRIANGLE_CAPACITY);
  let center = circumcenter(index);
  let base = centersOffset + 3u * index;
  centers[base] = center.x;
  centers[base + 1u] = center.y;
  centers[base + 2u] = select(0.0, center.z, index < triangleCount);
  if (index < triangleCount && center.z == 0.0) { atomicAdd(&invalidCount[invalidCountOffset], 1u); }`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-edge-keys`,
        operation: 'GPUVoronoiDiagram',
        variant: 'parallel-edge-keys',
        bindings: [
          {name: 'triangles', view: props.triangles, type: 'u32', access: 'read'},
          {name: 'triangleCountIn', view: readableTriangleCount, type: 'u32', access: 'read'},
          {name: 'candidateIds', view: candidateIds, type: 'u32', access: 'read_write'},
          {name: 'edgeMinimumKeys', view: edgeMinimumKeys, type: 'u32', access: 'read_write'},
          {name: 'edgeMaximumKeys', view: edgeMaximumKeys, type: 'u32', access: 'read_write'}
        ],
        invocationCount: candidateCapacity,
        declarations: `const POINT_COUNT: u32 = ${props.positions.length}u;
const TRIANGLE_CAPACITY: u32 = ${triangleCapacity}u;`,
        body: `let triangleCount = min(triangleCountIn[triangleCountInOffset], TRIANGLE_CAPACITY);
  let triangle = index / 3u;
  let edge = index % 3u;
  candidateIds[candidateIdsOffset + index] = index;
  var minimum = POINT_COUNT;
  var maximum = POINT_COUNT;
  if (triangle < triangleCount) {
    let a = triangles[trianglesOffset + 3u * triangle + edge];
    let b = triangles[trianglesOffset + 3u * triangle + (edge + 1u) % 3u];
    if (a < POINT_COUNT && b < POINT_COUNT) {
      minimum = min(a, b);
      maximum = max(a, b);
    }
  }
  edgeMinimumKeys[edgeMinimumKeysOffset + index] = minimum;
  edgeMaximumKeys[edgeMaximumKeysOffset + index] = maximum;`
      }),
      ...edgeSort.nodes,
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-classify-edges`,
        operation: 'GPUVoronoiDiagram',
        variant: 'sorted-edge-classification',
        bindings: [
          {name: 'sortedCandidates', view: edgeSort.sortedIndices, type: 'u32', access: 'read'},
          {
            name: 'sortedMinimumKeys',
            view: edgeSort.sortedPrimaryKeys,
            type: 'u32',
            access: 'read'
          },
          {name: 'edgeMaximumKeys', view: edgeMaximumKeys, type: 'u32', access: 'read'},
          {name: 'centers', view: centers, type: 'f32', access: 'read'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'},
          {name: 'candidateFlags', view: candidateFlags, type: 'u32', access: 'read_write'}
        ],
        invocationCount: candidateCapacity,
        declarations: `const POINT_COUNT: u32 = ${props.positions.length}u;
const TRIANGLE_CAPACITY: u32 = ${triangleCapacity}u;
const CANDIDATE_COUNT: u32 = ${candidateCapacity}u;
fn sortedTriangle(row: u32) -> u32 {
  return sortedCandidates[sortedCandidatesOffset + row] / 3u;
}
fn sortedMaximumKey(row: u32) -> u32 {
  return edgeMaximumKeys[edgeMaximumKeysOffset + sortedCandidates[sortedCandidatesOffset + row]];
}
fn sortedKeyEquals(row: u32, minimum: u32, maximum: u32) -> bool {
  return sortedMinimumKeys[sortedMinimumKeysOffset + row] == minimum && sortedMaximumKey(row) == maximum;
}
fn isValidCenter(triangle: u32) -> bool {
  return centers[centersOffset + 3u * triangle + 2u] != 0.0;
}`,
        // One thread per sorted half-edge. Equal undirected keys are contiguous and, because the
        // sort is stable over identity ids, ordered by candidate id and hence by triangle. The
        // lowest other triangle in the run is the neighbor the former all-triangle scan returned.
        body: `let candidate = sortedCandidates[sortedCandidatesOffset + index];
  let triangle = candidate / 3u;
  let minimum = sortedMinimumKeys[sortedMinimumKeysOffset + index];
  var neighbor = TRIANGLE_CAPACITY;
  var accepted = false;
  if (minimum < POINT_COUNT) {
    let maximum = sortedMaximumKey(index);
    // Lower bound of this key's run in [0, index].
    var low = 0u;
    var high = index;
    while (low < high) {
      let middle = (low + high) >> 1u;
      let middleMinimum = sortedMinimumKeys[sortedMinimumKeysOffset + middle];
      if (middleMinimum < minimum || (middleMinimum == minimum && sortedMaximumKey(middle) < maximum)) {
        low = middle + 1u;
      } else {
        high = middle;
      }
    }
    // Skip this triangle's own rows (at most three, contiguous) at the head of the run.
    var row = low;
    while (row < CANDIDATE_COUNT && sortedTriangle(row) == triangle && sortedKeyEquals(row, minimum, maximum)) {
      row++;
    }
    if (row < CANDIDATE_COUNT && sortedKeyEquals(row, minimum, maximum)) {
      neighbor = sortedTriangle(row);
    }
    accepted = isValidCenter(triangle) &&
      (neighbor == TRIANGLE_CAPACITY || (neighbor > triangle && isValidCenter(neighbor)));
  }
  neighbors[neighborsOffset + candidate] = neighbor;
  candidateFlags[candidateFlagsOffset + candidate] = select(0u, 1u, accepted);`
      })
    ];
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-clip-segments`,
        operation: 'GPUVoronoiDiagram',
        variant: 'clip-segments',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'triangles', view: props.triangles, type: 'u32', access: 'read'},
          {name: 'clipBounds', view: props.clipBounds, type: 'f32', access: 'read'},
          {name: 'centers', view: centers, type: 'f32', access: 'read'},
          {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
          {name: 'candidateFlags', view: candidateFlags, type: 'u32', access: 'read_write'},
          {name: 'candidateSegments', view: candidateSegments, type: 'f32', access: 'read_write'}
        ],
        invocationCount: candidateCapacity,
        declarations: `${getVoronoiEdgeWGSL({pointCount: props.positions.length, triangleCapacity})}
fn centerAt(triangle: u32) -> vec2<f32> {
  return vec2<f32>(centers[centersOffset + 3u * triangle], centers[centersOffset + 3u * triangle + 1u]);
}
// Liang-Barsky clip of a segment (0 <= t <= 1) or ray (t >= 0); returns (t0, t1, 1) or zeros.
fn clipParametric(origin: vec2<f32>, direction: vec2<f32>, isRay: bool, bounds: vec4<f32>) -> vec3<f32> {
  var enter = 0.0;
  var leave = 1.0;
  var hasExit = !isRay;
  for (var axis = 0u; axis < 2u; axis++) {
    let low = bounds[axis];
    let high = bounds[axis + 2u];
    if (direction[axis] == 0.0) {
      if (origin[axis] < low || origin[axis] > high) { return vec3<f32>(0.0); }
    } else {
      let first = (low - origin[axis]) / direction[axis];
      let second = (high - origin[axis]) / direction[axis];
      enter = max(enter, min(first, second));
      let exit = max(first, second);
      if (hasExit) { leave = min(leave, exit); } else { leave = exit; hasExit = true; }
    }
  }
  if (!hasExit || !(enter <= leave)) { return vec3<f32>(0.0); }
  return vec3<f32>(enter, leave, 1.0);
}`,
        // Internal edges join two circumcenters; hull edges are rays along the outward bisector.
        // Both are clipped to the bounds, and a segment that misses the bounds is dropped.
        body: `let triangle = index / 3u;
let edge = index % 3u;
let slot = candidateSegmentsOffset + 4u * index;
candidateSegments[slot] = 0.0; candidateSegments[slot + 1u] = 0.0; candidateSegments[slot + 2u] = 0.0; candidateSegments[slot + 3u] = 0.0;
if (candidateFlags[candidateFlagsOffset + index] == 0u) { return; }
let bounds = vec4<f32>(clipBounds[clipBoundsOffset], clipBounds[clipBoundsOffset + 1u], clipBounds[clipBoundsOffset + 2u], clipBounds[clipBoundsOffset + 3u]);
let neighbor = neighbors[neighborsOffset + index];
let center = centerAt(triangle);
var origin = center;
var direction = vec2<f32>(0.0);
let isRay = neighbor >= TRIANGLE_CAPACITY;
if (neighbor < TRIANGLE_CAPACITY) {
  direction = centerAt(neighbor) - center;
} else {
  let a = site(vertex(triangle, edge));
  let b = site(vertex(triangle, (edge + 1u) % 3u));
  let edgeDirection = getScaledTriangleEdge(a, b).direction;
  let edgeExtent = max(abs(edgeDirection.x), abs(edgeDirection.y));
  // The third center word carries the certified triangle orientation, including clockwise input.
  direction = centers[centersOffset + 3u * triangle + 2u] * vec2f(edgeDirection.y, -edgeDirection.x) / edgeExtent;
}
let clipped = clipParametric(origin, direction, isRay, bounds);
if (clipped.z == 0.0) { candidateFlags[candidateFlagsOffset + index] = 0u; return; }
let start = origin + clipped.x * direction;
let end = origin + clipped.y * direction;
candidateSegments[slot] = start.x; candidateSegments[slot + 1u] = start.y; candidateSegments[slot + 2u] = end.x; candidateSegments[slot + 3u] = end.y;`
      }),
      ...new GPUCompaction({
        id: `${id}-compact-edges`,
        input: candidateIds,
        flags: candidateFlags,
        output: compactCandidateIds,
        count: output.status.requiredCount
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-materialize`,
        operation: 'GPUVoronoiDiagram',
        variant: 'parallel-materialization',
        bindings: [
          {name: 'compactCandidateIds', view: compactCandidateIds, type: 'u32', access: 'read'},
          {name: 'candidateSegments', view: candidateSegments, type: 'f32', access: 'read'},
          {name: 'requiredCount', view: output.status.requiredCount, type: 'u32', access: 'read'},
          {name: 'segmentsOut', view: output.segments, type: 'f32', access: 'read_write'}
        ],
        invocationCount: output.segments.length,
        body: `if (index >= requiredCount[requiredCountOffset]) { return; }
  let candidate = compactCandidateIds[compactCandidateIdsOffset + index];
  let source = candidateSegmentsOffset + 4u * candidate;
  let segmentBase = segmentsOutOffset + 4u * index;
  segmentsOut[segmentBase] = candidateSegments[source];
  segmentsOut[segmentBase + 1u] = candidateSegments[source + 1u];
  segmentsOut[segmentBase + 2u] = candidateSegments[source + 2u];
  segmentsOut[segmentBase + 3u] = candidateSegments[source + 3u];`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-materialize-site-ids`,
        operation: 'GPUVoronoiDiagram',
        variant: 'parallel-site-id-materialization',
        bindings: [
          {name: 'triangles', view: props.triangles, type: 'u32', access: 'read'},
          {name: 'compactCandidateIds', view: compactCandidateIds, type: 'u32', access: 'read'},
          {name: 'requiredCount', view: output.status.requiredCount, type: 'u32', access: 'read'},
          {name: 'siteIdsOut', view: output.siteIds, type: 'u32', access: 'read_write'}
        ],
        invocationCount: output.siteIds.length,
        declarations: `fn vertex(triangle: u32, corner: u32) -> u32 {
  return triangles[trianglesOffset + 3u * triangle + corner];
}`,
        body: `if (index >= requiredCount[requiredCountOffset]) { return; }
  let candidate = compactCandidateIds[compactCandidateIdsOffset + index];
  let triangle = candidate / 3u;
  let edge = candidate % 3u;
  let a = vertex(triangle, edge);
  let b = vertex(triangle, (edge + 1u) % 3u);
  let siteBase = siteIdsOutOffset + 2u * index;
  siteIdsOut[siteBase] = min(a, b);
  siteIdsOut[siteBase + 1u] = max(a, b);`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-publish-status`,
        operation: 'GPUVoronoiDiagram',
        variant: 'publish-status',
        bindings: [
          {
            name: 'requiredCountIn',
            view: output.status.requiredCount,
            type: 'u32',
            access: 'read'
          },
          {name: 'countOut', view: output.status.count, type: 'u32', access: 'read_write'},
          {name: 'overflowOut', view: output.status.overflow, type: 'u32', access: 'read_write'},
          ...completenessBindings
        ],
        invocationCount: 1,
        declarations: `const OUTPUT_CAPACITY: u32 = ${output.segments.length}u;`,
        body: `let requiredCount = requiredCountIn[requiredCountInOffset];
  countOut[countOutOffset] = min(requiredCount, OUTPUT_CAPACITY);
  overflowOut[overflowOutOffset] = select(0u, 1u, requiredCount > OUTPUT_CAPACITY || ${capacityLossExpression});`
      })
    );
    for (const name of [
      'uncertainCount',
      'iterationCount',
      'iterationLimitReached',
      'approximation'
    ] as const) {
      const view = output.status[name];
      if (view)
        nodes.push(
          createFillNode<Parameters>(graph, {
            id: `${id}-publish-${name}`,
            operation: 'GPUVoronoiDiagram',
            view,
            type: 'u32',
            value: '0u',
            componentCount: 1
          })
        );
    }
    if (output.status.candidateOverflow)
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish-candidateOverflow`,
          operation: 'GPUVoronoiDiagram',
          bindings: [
            {name: 'requested', view: props.triangleCount, type: 'u32', access: 'read'},
            ...completenessBindings,
            {
              name: 'candidateOverflow',
              view: output.status.candidateOverflow,
              type: 'u32',
              access: 'read_write'
            }
          ],
          invocationCount: 1,
          body: `candidateOverflow[candidateOverflowOffset] = select(0u, 1u, requested[requestedOffset] > ${triangleCapacity}u || ${capacityLossExpression});`
        })
      );
    for (const field of ['converged', 'complete'] as const) {
      const destination = output.status[field];
      if (!destination) continue;
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-publish-${field}`,
          operation: 'GPUVoronoiDiagram',
          bindings: [
            {name: 'requested', view: props.triangleCount, type: 'u32', access: 'read'},
            ...completenessBindings,
            {name: 'invalidCount', view: invalidCount, type: 'u32', access: 'read'},
            ...(field === 'complete'
              ? [
                  {
                    name: 'required',
                    view: output.status.requiredCount,
                    type: 'u32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            {name: 'converged', view: destination, type: 'u32', access: 'read_write'}
          ],
          invocationCount: 1,
          body: `converged[convergedOffset] = select(0u, 1u, invalidCount[invalidCountOffset] == 0u && requested[requestedOffset] <= ${triangleCapacity}u && !(${incompleteExpression}) ${field === 'complete' ? `&& required[requiredOffset] <= ${output.segments.length}u` : ''});`
        })
      );
    }
    return nodes;
  }
}

function getVoronoiWGSL(options: {pointCount: number; triangleCapacity: number}): string {
  return /* wgsl */ `
const POINT_COUNT: u32 = ${options.pointCount}u;
const TRIANGLE_CAPACITY: u32 = ${options.triangleCapacity}u;
${EXACT_ORIENTATION_WGSL}
${TRIANGLE_CIRCUMCIRCLE_WGSL}
fn site(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn vertex(triangle: u32, corner: u32) -> u32 {
  return triangles[trianglesOffset + 3u * triangle + corner];
}
fn circumcenter(triangle: u32) -> vec3<f32> {
  let aRow = vertex(triangle, 0u);
  let bRow = vertex(triangle, 1u);
  let cRow = vertex(triangle, 2u);
  if (aRow >= POINT_COUNT || bRow >= POINT_COUNT || cRow >= POINT_COUNT) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  let a = site(aRow);
  let b = site(bRow);
  let c = site(cRow);
  let circle = getTriangleCircumcircle(a, b, c);
  if (circle.orientation == 0) { return vec3f(0.0); }
  let center = circle.center;
  // Nearly collinear triangles can overflow; NaN fails both comparisons.
  if (!(abs(center.x) <= ORIENT_FLOAT_MAXIMUM && abs(center.y) <= ORIENT_FLOAT_MAXIMUM)) {
    return vec3<f32>(0.0, 0.0, 0.0);
  }
  return vec3<f32>(center, f32(circle.orientation));
}
`;
}

function getVoronoiEdgeWGSL(options: {pointCount: number; triangleCapacity: number}): string {
  return /* wgsl */ `
${TRIANGLE_SCALE_WGSL}
const POINT_COUNT: u32 = ${options.pointCount}u;
const TRIANGLE_CAPACITY: u32 = ${options.triangleCapacity}u;
fn site(row: u32) -> vec2<f32> {
  return vec2<f32>(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
}
fn vertex(triangle: u32, corner: u32) -> u32 {
  return triangles[trianglesOffset + 3u * triangle + corner];
}`;
}
