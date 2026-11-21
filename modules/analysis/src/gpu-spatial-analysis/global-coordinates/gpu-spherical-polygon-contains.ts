// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Kernel} from '@luma.gl/graph';
import {validateUnitVector} from './assert-unit-vector';
import {getSphericalRingOrientationWGSL} from './spherical-ring-orientation-wgsl';
import {
  createTransientView,
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  getViewElementOffset,
  type GraphVectorView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getReadPoint3WGSL} from '../shared/spatial-wgsl-point-helpers';
import {
  isGraphVectorView,
  validateSinglePackedView,
  validateRequirement,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  createGraphChunkDispatchParameters,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/spatial-context';
import {
  GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
  GPU_SPHERICAL_MAXIMUM_TOLERANCE,
  GPU_SPHERICAL_MINIMUM_TOLERANCE,
  GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT,
  validateAngularTolerance
} from './spherical-limits';

/** Status of one query: inside or outside the polygon (or its complement). */
export const GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_VALID = 0;
/** Status: invalid query, polygon index or ring (including an edge shorter than `tolerance`). */
export const GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_INVALID = 1;
/** Status: the query lies on the boundary within `tolerance`; `contains` follows `boundary`. */
export const GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_BOUNDARY = 2;

export type GPUSphericalPolygonContainsProps<
  QueryRows extends
    | GraphDataView<'float32x3'>
    | GraphVectorView<'float32x3'> = GraphDataView<'float32x3'>,
  SelectorRows extends
    | GraphDataView<'uint32'>
    | GraphVectorView<'uint32'> = GraphDataView<'uint32'>,
  ContainsRows extends
    | GraphDataView<'uint32'>
    | GraphVectorView<'uint32'> = GraphDataView<'uint32'>,
  StatusRows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>
> = {
  id?: string;
  /**
   * Coordinate space of the unit vectors: longitude-latitude (their embedding on the sphere) with
   * great-circle edges. Planar contexts are rejected. Defaults to longitude-latitude, no metric.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Unit xyz, simple minor-arc rings, with no repeated closing vertex. Either orientation: each
   * ring stands for the region it bounds inside its hemisphere, so counterclockwise shells with
   * clockwise holes (GeoJSON order) and all-counterclockwise rings give the same result.
   */
  positions: GraphDataView<'float32x3'>;
  ringOffsets: GraphDataView<'uint32'>;
  /** Ring offsets per polygon: first ring is the shell; subsequent rings are holes. */
  polygonOffsets: GraphDataView<'uint32'>;
  /** Packed or chunked queries; selectors and outputs mirror the same ordered chunks. */
  queries: QueryRows;
  polygonIndices: SelectorRows;
  boundary?: 'closed' | 'open';
  complement?: boolean;
  tolerance?: number;
  /** Optional common hemisphere pole. Omitted: each ring uses its normalized vertex mean. */
  hemisphereCenter?: readonly [number, number, number];
  output: {
    /** 0 outside, 1 inside (after `complement`); read it together with `status`. */
    contains: ContainsRows;
    /**
     * Per query: {@link GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_VALID} (0),
     * {@link GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_INVALID} (1) or
     * {@link GPU_SPHERICAL_POLYGON_CONTAINS_STATUS_BOUNDARY} (2).
     */
    status: StatusRows;
  };
};

type PackedSphericalPolygonContainsProps = Omit<
  GPUSphericalPolygonContainsProps,
  'queries' | 'polygonIndices' | 'output'
> & {
  queries: GraphDataView<'float32x3'>;
  polygonIndices: GraphDataView<'uint32'>;
  output: {contains: GraphDataView<'uint32'>; status: GraphDataView<'uint32'>};
};

/** Query output packing plus runtime row parameters fits the default eight storage bindings. */
const SPHERICAL_POLYGON_QUERY_STORAGE_BINDING_COUNT = 8;
/** Independent query rows use the standard 256-thread bounded dispatch. */
const SPHERICAL_POLYGON_CHUNK_WORKGROUP_SIZE = 256;
/** Prevent multiplication by the workgroup size from wrapping the uint32 index. */
const SPHERICAL_POLYGON_MAXIMUM_LINEAR_WORKGROUP_COUNT = 0x1000000;

/**
 * Nonconvex spherical winding containment with holes and explicit complementary-region semantics.
 *
 * Ring validity (unit vertices, hemisphere condition, no crossing or touching nonadjacent arcs) is
 * checked once per ring, not once per query; a query then costs one pass over its rings' edges.
 * Packed resident geometry is shared by every query chunk. Query selectors and output chunks
 * mirror query lengths, including empty chunks. Two-word temporary results are reused between
 * chunks so query kernels stay within eight storage bindings; scatter keeps the public layout.
 *
 * Rings may have either orientation. A ring lies in an open hemisphere (its own vertex mean or
 * `hemisphereCenter`) and stands for the region it bounds there; the other region is selected
 * explicitly with `complement`. The per-ring pass reads the orientation from the sign of the ring's
 * gnomonic (central) projection area about that pole, where great arcs are straight segments, and
 * the query pass orients the winding sum by it, so clockwise or counterclockwise shells and holes
 * (GeoJSON order included) give the same result.
 *
 * Status per query: `0` inside or outside, `1` invalid query, polygon index or ring (this includes
 * a ring with an edge whose cross product is within `tolerance`, i.e. an edge shorter than about
 * `tolerance` radians or nearly antipodal endpoints), `2` on the boundary within `tolerance`. The
 * f32 winding sum is only trustworthy for queries farther than `tolerance` from every vertex, so
 * keep `tolerance` (at least 1e-7, default 1e-6 or about 6 m) below the shortest edge of interest.
 */
export class GPUSphericalPolygonContains<
  QueryRows extends
    | GraphDataView<'float32x3'>
    | GraphVectorView<'float32x3'> = GraphDataView<'float32x3'>,
  SelectorRows extends
    | GraphDataView<'uint32'>
    | GraphVectorView<'uint32'> = GraphDataView<'uint32'>,
  ContainsRows extends
    | GraphDataView<'uint32'>
    | GraphVectorView<'uint32'> = GraphDataView<'uint32'>,
  StatusRows extends GraphDataView<'uint32'> | GraphVectorView<'uint32'> = GraphDataView<'uint32'>
> implements GPUCommandNodeProducer
{
  readonly id: string;
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  private readonly chunked: boolean;
  /** Validated coordinate space of the unit vectors. */
  readonly spatialContext: GPUSpatialContext;
  constructor(
    readonly props: GPUSphericalPolygonContainsProps<
      QueryRows,
      SelectorRows,
      ContainsRows,
      StatusRows
    >
  ) {
    this.id = props.id ?? 'spherical-polygon-contains';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext ?? GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
      GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT
    );
    if (props.hemisphereCenter)
      validateUnitVector(props.hemisphereCenter, this.id, 'hemisphereCenter');
    validateSinglePackedView(props.positions, ['float32x3'], `${this.id} positions`);
    for (const [name, view] of Object.entries({
      ringOffsets: props.ringOffsets,
      polygonOffsets: props.polygonOffsets
    }))
      validateSinglePackedView(view, ['uint32'], `${this.id} ${name}`);
    this.chunkPlan = planGraphChunkDispatch(this.id, {
      queries: props.queries,
      polygonIndices: props.polygonIndices,
      contains: props.output.contains,
      status: props.output.status
    });
    for (const chunk of this.chunkPlan.chunks) {
      validateSinglePackedView(chunk.views['queries'], ['float32x3'], `${this.id} queries`);
      for (const name of ['polygonIndices', 'contains', 'status'])
        validateSinglePackedView(chunk.views[name], ['uint32'], `${this.id} ${name}`);
    }
    validateDisjointWritableViews(this.id, props.output);
    this.chunked = [props.queries, props.polygonIndices, ...Object.values(props.output)].some(
      view => isGraphVectorView(view)
    );
    this.estimatedCommandNodeCount = this.chunked ? 3 + this.chunkPlan.nonemptyChunkCount * 2 : 3;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    validateRequirement(
      this.id,
      props.ringOffsets.length >= 1 && props.polygonOffsets.length >= 1,
      'ringOffsets and polygonOffsets must hold at least one entry'
    );
    validateRequirement(
      this.id,
      props.queries.length === props.polygonIndices.length &&
        props.output.contains.length === props.queries.length &&
        props.output.status.length === props.queries.length,
      'polygonIndices, output.contains and output.status must hold one entry per query'
    );
    validateAngularTolerance(
      this.id,
      props.tolerance ?? 1e-6,
      GPU_SPHERICAL_MINIMUM_TOLERANCE,
      GPU_SPHERICAL_MAXIMUM_TOLERANCE
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.ringOffsets,
      props.polygonOffsets,
      props.queries,
      props.polygonIndices
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphChunkBindingRanges(graph, id, {
      positions: props.positions,
      ringOffsets: props.ringOffsets,
      polygonOffsets: props.polygonOffsets,
      queries: props.queries,
      polygonIndices: props.polygonIndices,
      ...props.output
    });
    if (this.chunked) validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
    const maximumChunkLength = this.chunkPlan.chunks.reduce(
      (maximum, chunk) => Math.max(maximum, chunk.length),
      0
    );
    const scratchRingCount = Math.max(props.ringOffsets.length - 1, 1);
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'ringCenters',
      scratchRingCount * 12,
      'reduce the packed ring count'
    );
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'ringFailures',
      scratchRingCount * 4,
      'reduce the packed ring count'
    );
    if (this.chunked)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'chunk results',
        Math.max(maximumChunkLength, 1) * 8,
        'reduce the query chunk size'
      );
    validateMaximumCount(
      id,
      'storage bindings',
      SPHERICAL_POLYGON_QUERY_STORAGE_BINDING_COUNT,
      graph.device.limits.maxStorageBuffersPerShaderStage
    );
    const layouts = this.chunkPlan.chunks.map(chunk =>
      getBoundedDispatchLayout(
        id,
        chunk.length,
        SPHERICAL_POLYGON_CHUNK_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );
    const tolerance = getWGSLFloatLiteral(props.tolerance ?? 1e-6);
    const ringCount = props.ringOffsets.length - 1;
    // Query-independent ring validation is computed once per ring instead of once per query.
    const ringCenters = createTransientView(
      graph,
      `${id}-ring-centers`,
      'float32x3',
      Math.max(ringCount, 1)
    );
    const ringFailures = createTransientView(
      graph,
      `${id}-ring-failures`,
      'uint32',
      Math.max(ringCount, 1)
    );
    const declarations = `const POLYGON_TOLERANCE: f32 = ${tolerance};
${getReadPoint3WGSL('polygonPosition', 'positions')}
${getSphericalRingOrientationWGSL('polygonPosition')}
fn polygonUnit(point: vec3f) -> bool { let squared = dot(point, point); return (bitcast<u32>(squared) & 0x7f800000u) != 0x7f800000u && abs(squared - 1.0) <= 1e-4; }
fn polygonEdgeNormal(first: vec3f, second: vec3f) -> vec3f {
  let firstDirection = normalize(first);
  let secondDirection = normalize(second);
  // Form a small difference before the cross product to retain tiny off-axis arcs.
  let difference = secondDirection - select(-firstDirection, firstDirection, dot(firstDirection, secondDirection) >= 0.0);
  return cross(firstDirection, difference);
}
fn polygonOnArc(first: vec3f, second: vec3f, point: vec3f, margin: f32) -> bool {
  let firstDirection = normalize(first);
  let secondDirection = normalize(second);
  let pointDirection = normalize(point);
  // Endpoint chord distance can be exactly zero while the incident plane residual
  // rounds above the angular band. Accept the finite endpoint caps first.
  if (min(distance(firstDirection, pointDirection), distance(secondDirection, pointDirection)) <= 2.0 * sin(margin * 0.5)) { return true; }
  let normal = normalize(polygonEdgeNormal(first, second));
  // The projection belongs to the finite minor arc only when both sides are
  // nonnegative. Expanding both side tests by margin aliases a short arc's antipode.
  let firstSide = dot(cross(firstDirection, pointDirection), normal);
  let secondSide = dot(cross(pointDirection, secondDirection), normal);
  if (firstSide >= 0.0 && secondSide >= 0.0) {
    return abs(dot(normal, pointDirection)) <= sin(margin);
  }
  return false;
}`;
    const sharedNodes = [
      createWGSLKernelNode(graph, {
        id: `${id}-ring-center`,
        operation: 'GPUSphericalPolygonContains',
        variant: 'ring-center',
        invocationCount: Math.max(ringCount, 1),
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
          {name: 'ringCenters', view: ringCenters, type: 'f32', access: 'read_write'},
          {name: 'ringFailures', view: ringFailures, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations,
        body: `if (index >= ${ringCount}u) { return; }
let begin = ringOffsets[ringOffsetsOffset + index]; let end = ringOffsets[ringOffsetsOffset + index + 1u];
var failure = 0u; var center = vec3f(0.0);
if (begin > end || end > ${props.positions.length}u || end - begin < 3u) { failure = 1u; }
else {
  for (var vertex = begin; vertex < end; vertex++) { let value = polygonPosition(vertex); if (!polygonUnit(value)) { failure = 1u; } center += value; }
  // The supplied ring represents its smaller region; complement is explicit at polygon level.
  ${props.hemisphereCenter ? `center = vec3f(${props.hemisphereCenter.join(',')});` : ''}
  if (length(center) <= POLYGON_TOLERANCE) { failure = 1u; } else { center = normalize(center); }
}
// Local projected differences retain small off-axis ring area without O(1) cancellation.
var signedArea = 0.0;
if (failure == 0u) {
  signedArea = sphericalRingSignedArea(begin, end, center);
}
let clockwise = select(0u, 2u, signedArea < 0.0);
ringCenters[ringCentersOffset + index * 3u] = center.x; ringCenters[ringCentersOffset + index * 3u + 1u] = center.y; ringCenters[ringCentersOffset + index * 3u + 2u] = center.z;
// Bit 0: failure. Bit 1: clockwise about the pole.
atomicStore(&ringFailures[ringFailuresOffset + index], failure | clockwise);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-ring-edges`,
        operation: 'GPUSphericalPolygonContains',
        variant: 'ring-edges',
        invocationCount: Math.max(props.positions.length, 1),
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
          {name: 'ringCenters', view: ringCenters, type: 'f32', access: 'read'},
          {name: 'ringFailures', view: ringFailures, type: 'atomic<u32>', access: 'read_write'}
        ],
        declarations,
        body: `if (index >= ${props.positions.length}u || ${ringCount}u == 0u) { return; }
// Locate the ring that owns this vertex by binary search over the ring offsets.
var low = 0u; var high = ${ringCount}u;
for (var step = 0u; step < 32u && low + 1u < high; step++) {
  let middle = (low + high) / 2u;
  if (ringOffsets[ringOffsetsOffset + middle] <= index) { low = middle; } else { high = middle; }
}
let ring = low;
let begin = ringOffsets[ringOffsetsOffset + ring]; let end = ringOffsets[ringOffsetsOffset + ring + 1u];
if (index < begin || index >= end || (atomicLoad(&ringFailures[ringFailuresOffset + ring]) & 1u) != 0u) { return; }
let center = vec3f(ringCenters[ringCentersOffset + ring * 3u], ringCenters[ringCentersOffset + ring * 3u + 1u], ringCenters[ringCentersOffset + ring * 3u + 2u]);
let edge = index;
let next = select(edge + 1u, begin, edge + 1u == end);
let first = polygonPosition(edge); let second = polygonPosition(next);
let normal = polygonEdgeNormal(first, second);
var failure = 0u;
if (length(normal) <= POLYGON_TOLERANCE || dot(center, first) <= POLYGON_TOLERANCE) { failure = 1u; }
// Reject crossing/touching nonadjacent arcs instead of assigning an invalid ring a fill.
for (var other = begin; other < end && failure == 0u; other++) {
  let otherNext = select(other + 1u, begin, other + 1u == end);
  if (other == edge || other == next || otherNext == edge) { continue; }
  let third = polygonPosition(other); let fourth = polygonPosition(otherNext);
  let crossing = cross(normal, polygonEdgeNormal(third, fourth));
  if (length(crossing) > POLYGON_TOLERANCE * POLYGON_TOLERANCE) {
    let intersection = normalize(crossing);
    if ((polygonOnArc(first, second, intersection, POLYGON_TOLERANCE) && polygonOnArc(third, fourth, intersection, POLYGON_TOLERANCE)) || (polygonOnArc(first, second, -intersection, POLYGON_TOLERANCE) && polygonOnArc(third, fourth, -intersection, POLYGON_TOLERANCE))) { failure = 1u; }
  } else if (polygonOnArc(first, second, third, POLYGON_TOLERANCE) || polygonOnArc(first, second, fourth, POLYGON_TOLERANCE)) { failure = 1u; }
}
// Bit 1 holds the ring orientation; only bit 0 marks a failure.
if (failure != 0u) { atomicOr(&ringFailures[ringFailuresOffset + ring], 1u); }`
      })
    ];
    if (!this.chunked)
      return [
        ...sharedNodes,
        createWGSLKernelNode(
          graph,
          this.getQueryKernelProps(
            props as PackedSphericalPolygonContainsProps,
            ringFailures,
            declarations
          )
        )
      ];
    const {slots, nodes} = createGraphChunkDispatchParameters(graph, {
      id: `${id}-chunks`,
      plan: this.chunkPlan
    });
    const results = createTransientView(
      graph,
      `${id}-chunk-results`,
      'uint32x2',
      maximumChunkLength
    );
    const queryNodes: GPUCommandNode<Parameters>[] = [];
    for (const chunk of this.chunkPlan.chunks) {
      if (chunk.length === 0) continue;
      const chunkProps: PackedSphericalPolygonContainsProps = {
        ...props,
        queries: chunk.views['queries'] as GraphDataView<'float32x3'>,
        polygonIndices: chunk.views['polygonIndices'] as GraphDataView<'uint32'>,
        output: {
          contains: chunk.views['contains'] as GraphDataView<'uint32'>,
          status: chunk.views['status'] as GraphDataView<'uint32'>
        }
      };
      const kernelProps = this.getQueryKernelProps<Parameters>(
        chunkProps,
        ringFailures,
        declarations,
        results
      );
      queryNodes.push(
        this.createChunkQueryNode(
          `${id}-query-${chunk.chunkIndex}`,
          slots[chunk.chunkIndex],
          layouts[chunk.chunkIndex],
          kernelProps,
          {queries: 3, polygonIndices: 1, results: 2}
        )
      );
      queryNodes.push(
        this.createChunkQueryNode(
          `${id}-scatter-${chunk.chunkIndex}`,
          slots[chunk.chunkIndex],
          layouts[chunk.chunkIndex],
          {
            id: `${id}-scatter`,
            operation: 'GPUSphericalPolygonContains',
            variant: 'scatter',
            invocationCount: chunk.length,
            bindings: [
              {name: 'results', view: results, type: 'u32', access: 'read'},
              {
                name: 'contains',
                view: chunkProps.output.contains,
                type: 'u32',
                access: 'read_write'
              },
              {name: 'status', view: chunkProps.output.status, type: 'u32', access: 'read_write'}
            ],
            body: `contains[containsOffset + index] = results[resultsOffset + index * 2u];
status[statusOffset + index] = results[resultsOffset + index * 2u + 1u];`
          },
          {results: 2, contains: 1, status: 1}
        )
      );
    }
    return [...sharedNodes, ...nodes, ...queryNodes];
  }

  private createChunkQueryNode<Parameters>(
    id: string,
    parameters: GraphDataView<'uint32'>,
    layout: ReturnType<typeof getBoundedDispatchLayout>,
    kernelProps: WGSLKernelProps<Parameters>,
    rowComponents: Record<string, number>
  ): GPUCommandNode<Parameters> {
    // Bound array lengths include only the aligned prefix and the logical view bytes. Scratch
    // results have maximum chunk capacity: each shorter chunk writes its trailing rows, then scatters.
    const source = `${kernelProps.bindings
      .map(
        (binding, location) =>
          `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;\n` +
          (binding.name in rowComponents
            ? `var<private> ${binding.name}Offset: u32;`
            : `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;`)
      )
      .join('\n')}
@group(0) @binding(${kernelProps.bindings.length}) var<storage, read> chunkParameters: array<u32>;
${kernelProps.declarations ?? ''}
@compute @workgroup_size(${SPHERICAL_POLYGON_CHUNK_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(num_workgroups) workgroupCount: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${SPHERICAL_POLYGON_MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${SPHERICAL_POLYGON_CHUNK_WORKGROUP_SIZE}u + localInvocationIndex;
  if (index >= chunkParameters[1]) { return; }
  ${Object.entries(rowComponents)
    .map(
      ([name, components]) =>
        `${name}Offset = arrayLength(&${name}) - chunkParameters[1] * ${components}u;`
    )
    .join('\n  ')}
  ${kernelProps.body}
}`;
    return createGPUComputeCommandNode<Parameters>({
      id,
      workload: {
        operation: kernelProps.operation,
        ...(kernelProps.variant ? {variant: kernelProps.variant} : {}),
        commandCount: 1,
        maximumInvocationCount: kernelProps.invocationCount,
        maximumWorkgroupCount: layout.x * layout.y * layout.z
      },
      resources: [
        ...kernelProps.bindings.map(binding => ({
          buffer: binding.view,
          usage:
            binding.access === 'read' ? ('storage-read' as const) : ('storage-read-write' as const)
        })),
        {buffer: parameters, usage: 'storage-read'}
      ],
      compile: ({device}) => {
        const kernel = new Kernel(device, {
          id: kernelProps.id,
          source,
          shaderLayout: {
            bindings: [
              ...kernelProps.bindings.map((binding, location) => ({
                name: binding.name,
                group: 0,
                location,
                type:
                  binding.access === 'read' ? ('read-only-storage' as const) : ('storage' as const)
              })),
              {
                name: 'chunkParameters',
                group: 0,
                location: kernelProps.bindings.length,
                type: 'read-only-storage'
              }
            ]
          }
        });
        return {
          encode: ({computePass, getBuffer}) =>
            kernel.dispatch(computePass, {
              bindings: Object.fromEntries([
                ...kernelProps.bindings.map(binding => [
                  binding.name,
                  getViewBinding(binding.view, getBuffer)
                ]),
                ['chunkParameters', getViewBinding(parameters, getBuffer)]
              ]),
              ...layout
            }),
          destroy: () => kernel.destroy()
        };
      }
    });
  }

  private getQueryKernelProps<Parameters>(
    props: PackedSphericalPolygonContainsProps,
    ringFailures: GraphDataView<'uint32'>,
    declarations: string,
    results?: GraphDataView<'uint32x2'>
  ): WGSLKernelProps<Parameters> {
    const {id} = this;
    return {
      id,
      operation: 'GPUSphericalPolygonContains',
      variant: `${props.boundary ?? 'closed'}-${Boolean(props.complement)}`,
      invocationCount: props.queries.length,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'ringOffsets', view: props.ringOffsets, type: 'u32', access: 'read'},
        {name: 'polygonOffsets', view: props.polygonOffsets, type: 'u32', access: 'read'},
        {name: 'queries', view: props.queries, type: 'f32', access: 'read'},
        {name: 'polygonIndices', view: props.polygonIndices, type: 'u32', access: 'read'},
        ...(results
          ? [{name: 'results', view: results, type: 'u32' as const, access: 'read_write' as const}]
          : [
              {
                name: 'contains',
                view: props.output.contains,
                type: 'u32' as const,
                access: 'read_write' as const
              },
              {
                name: 'status',
                view: props.output.status,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]),
        {name: 'ringFailures', view: ringFailures, type: 'u32', access: 'read'}
      ],
      declarations,
      body: `let queryOffset = queriesOffset + index * 3u;
let queryPoint = vec3f(queries[queryOffset], queries[queryOffset + 1u], queries[queryOffset + 2u]);
var point = queryPoint;
let polygon = polygonIndices[polygonIndicesOffset + index];
var failure = 0u; var inside = false; var boundary = false;
if (!polygonUnit(point) || polygon >= ${props.polygonOffsets.length - 1}u) { failure = 1u; }
if (failure == 0u) {
  point = normalize(point);
  let firstRing = polygonOffsets[polygonOffsetsOffset + polygon]; let lastRing = polygonOffsets[polygonOffsetsOffset + polygon + 1u];
  if (firstRing > lastRing || lastRing >= ${props.ringOffsets.length}u) { failure = 1u; }
  for (var ring = firstRing; ring < lastRing && failure == 0u; ring++) {
    // Ring validity (range, unit vectors, hemisphere, arc crossings) is precomputed per ring.
    // Bit 0: the ring failed validation. Bit 1: the ring is clockwise about its pole.
    let ringFlags = ringFailures[ringFailuresOffset + ring];
    if ((ringFlags & 1u) != 0u) { failure = 1u; break; }
    let begin = ringOffsets[ringOffsetsOffset + ring]; let end = ringOffsets[ringOffsetsOffset + ring + 1u];
    var winding = 0.0; var ringBoundary = false;
    for (var edge = begin; edge < end; edge++) {
      let next = select(edge + 1u, begin, edge + 1u == end);
      let first = polygonPosition(edge); let second = polygonPosition(next);
      // Boundary directions must be normalized once, just like the packed endpoints.
      ringBoundary = ringBoundary || polygonOnArc(first, second, queryPoint, POLYGON_TOLERANCE);
      // Use small differences near either the query or its antipode before tangent projection.
      // Direct dot(first,second)-dot(point,first)*dot(point,second) loses tiny off-axis angles.
      let firstDifference = first - select(-point, point, dot(point, first) >= 0.0);
      let secondDifference = second - select(-point, point, dot(point, second) >= 0.0);
      let firstTangent = cross(point, firstDifference); let secondTangent = cross(point, secondDifference);
      let numerator = dot(point, cross(firstTangent, secondTangent)); let denominator = dot(firstTangent, secondTangent);
      // Only the zero tangent pair has no angle; a fixed cutoff drops valid tiny rings.
      if (abs(numerator) + abs(denominator) > 0.0) { winding += atan2(numerator, denominator); }
    }
    // A counterclockwise ring winds 2 pi around its region (and -2 pi around the antipodal mirror
    // image of it); a clockwise ring the opposite, so its winding is negated first.
    let orientedWinding = select(winding, -winding, (ringFlags & 2u) != 0u);
    let ringInside = orientedWinding > 3.141592653589793;
    if (ring == firstRing) { inside = ringInside; } else { inside = inside && !ringInside; }
    boundary = boundary || ringBoundary;
  }
}
var accepted = ${props.complement ? '!inside' : 'inside'};
if (boundary && failure == 0u) { failure = 2u; accepted = ${props.boundary !== 'open'}; }
${results ? 'results[resultsOffset + index * 2u]' : 'contains[containsOffset + index]'} = select(0u, 1u, accepted && failure != 1u);
${results ? 'results[resultsOffset + index * 2u + 1u]' : 'status[statusOffset + index]'} = failure;`
    };
  }
}
