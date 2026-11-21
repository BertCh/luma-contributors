// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {Kernel} from '@luma.gl/graph';
import {
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
  type WGSLKernelProps
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  isGraphVectorView,
  validateSinglePackedView,
  validateRequirement,
  validateCommandNodeBudget,
  validateMaximumCount,
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
  GPU_SPHERICAL_CONVEX_MAXIMUM_VERTICES,
  GPU_SPHERICAL_CONVEX_MINIMUM_VERTICES,
  GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
  GPU_SPHERICAL_MAXIMUM_TOLERANCE,
  GPU_SPHERICAL_MINIMUM_TOLERANCE,
  GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT,
  validateAngularTolerance
} from './spherical-limits';

/** Status of one containment row: the value of `output.status`. */
export const GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_VALID = 0;
/** Status: invalid or unsupported polygon, polygon index or query (nonconvex, non-unit, too few or too many vertices). */
export const GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_INVALID = 1;
/** Status: the query lies within `tolerance` of the polygon boundary; `contains` follows `boundary`. */
export const GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_BOUNDARY = 2;
export type GPUSphericalConvexContainsProps<
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
  /** Unit xyz vertices, CCW minor arcs; each polygon is convex and lies in an open hemisphere. */
  positions: GraphDataView<'float32x3'>;
  offsets: GraphDataView<'uint32'>;
  /** Packed or chunked queries; selectors and outputs mirror the same chunk topology. */
  queries: QueryRows;
  polygonIndices: SelectorRows;
  boundary?: 'closed' | 'open';
  /** Query the complementary region; boundary policy still applies explicitly. */
  complement?: boolean;
  tolerance?: number;
  output: {
    /** 0 false, 1 true; invalid rows must be interpreted using status. */
    contains: ContainsRows;
    /**
     * Per query: {@link GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_VALID} (0),
     * {@link GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_INVALID} (1, invalid or unsupported polygon or
     * query) or {@link GPU_SPHERICAL_CONVEX_CONTAINS_STATUS_BOUNDARY} (2, boundary tolerance band).
     */
    status: StatusRows;
  };
};
type PackedSphericalConvexContainsProps = Omit<
  GPUSphericalConvexContainsProps,
  'queries' | 'polygonIndices' | 'output'
> & {
  queries: GraphDataView<'float32x3'>;
  polygonIndices: GraphDataView<'uint32'>;
  output: {contains: GraphDataView<'uint32'>; status: GraphDataView<'uint32'>};
};

/** Six data bindings plus one runtime chunk-parameter binding fit the default device limit. */
const SPHERICAL_CONVEX_CHUNK_STORAGE_BINDING_COUNT = 7;
/** Independent query rows use the standard 256-thread bounded dispatch. */
const SPHERICAL_CONVEX_CHUNK_WORKGROUP_SIZE = 256;
/** Prevent the linear workgroup index from wrapping uint32. */
const SPHERICAL_CONVEX_MAXIMUM_LINEAR_WORKGROUP_COUNT = 0x1000000;

/**
 * Convex spherical containment with explicit boundary and complement semantics.
 * Chunked queries and selectors borrow the same packed resident polygon geometry. Row-aligned
 * output chunks mirror query lengths, including empty chunks. Chunk dispatch reads lengths and
 * aligned binding prefixes at runtime and uses seven storage bindings, independent of row count.
 */
export class GPUSphericalConvexContains<
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
  /** Validated coordinate space of the unit vectors. */
  readonly spatialContext: GPUSpatialContext;
  constructor(
    readonly props: GPUSphericalConvexContainsProps<
      QueryRows,
      SelectorRows,
      ContainsRows,
      StatusRows
    >
  ) {
    this.id = props.id ?? 'spherical-convex-contains';
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext ?? GPU_SPHERICAL_DEFAULT_SPATIAL_CONTEXT,
      GPU_SPHERICAL_SPATIAL_CONTEXT_SUPPORT
    );
    validateSinglePackedView(props.positions, ['float32x3'], `${this.id} positions`);
    validateSinglePackedView(props.offsets, ['uint32'], `${this.id} offsets`);
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
    const chunked = [props.queries, props.polygonIndices, ...Object.values(props.output)].some(
      view => isGraphVectorView(view)
    );
    this.estimatedCommandNodeCount = chunked ? 1 + this.chunkPlan.nonemptyChunkCount : 1;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    // One polygon selector and output per query; polygons contain at most 32 vertices.
    validateRequirement(this.id, props.offsets.length >= 1, 'offsets must hold at least one entry');
    validateRequirement(
      this.id,
      props.polygonIndices.length === props.queries.length,
      'polygonIndices must hold one entry per query'
    );
    validateRequirement(
      this.id,
      props.output.contains.length === props.queries.length &&
        props.output.status.length === props.queries.length,
      'output.contains and output.status must hold one entry per query'
    );
    validateAngularTolerance(
      this.id,
      props.tolerance ?? 1e-6,
      GPU_SPHERICAL_MINIMUM_TOLERANCE,
      GPU_SPHERICAL_MAXIMUM_TOLERANCE
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.positions,
      props.offsets,
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
      offsets: props.offsets,
      queries: props.queries,
      polygonIndices: props.polygonIndices,
      ...props.output
    });
    const chunked = [props.queries, props.polygonIndices, ...Object.values(props.output)].some(
      view => isGraphVectorView(view)
    );
    if (!chunked)
      return [
        createWGSLKernelNode(
          graph,
          this.getKernelProps(props as PackedSphericalConvexContainsProps)
        )
      ];
    validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
    validateMaximumCount(
      id,
      'chunked storage bindings',
      SPHERICAL_CONVEX_CHUNK_STORAGE_BINDING_COUNT,
      graph.device.limits.maxStorageBuffersPerShaderStage
    );
    const layouts = this.chunkPlan.chunks.map(chunk =>
      getBoundedDispatchLayout(
        id,
        chunk.length,
        SPHERICAL_CONVEX_CHUNK_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );
    const {slots, nodes} = createGraphChunkDispatchParameters(graph, {
      id: `${id}-chunks`,
      plan: this.chunkPlan
    });
    return [
      ...nodes,
      ...this.chunkPlan.chunks
        .filter(chunk => chunk.length > 0)
        .map(chunk => {
          const kernelProps = this.getKernelProps({
            ...props,
            queries: chunk.views['queries'] as GraphDataView<'float32x3'>,
            polygonIndices: chunk.views['polygonIndices'] as GraphDataView<'uint32'>,
            output: {
              contains: chunk.views['contains'] as GraphDataView<'uint32'>,
              status: chunk.views['status'] as GraphDataView<'uint32'>
            }
          });
          const rowComponents: Record<string, number> = {
            queries: 3,
            polygonIndices: 1,
            contains: 1,
            status: 1
          };
          // The runtime binding length includes exactly the aligned prefix and logical chunk rows.
          const source = `${kernelProps.bindings
            .map(
              (binding, location) =>
                `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;\n` +
                (binding.name in rowComponents
                  ? `var<private> ${binding.name}Offset: u32;`
                  : `const ${binding.name}Offset: u32 = ${getViewElementOffset(binding.view)}u;`)
            )
            .join('\n')}
@group(0) @binding(6) var<storage, read> chunkParameters: array<u32>;
${kernelProps.declarations}
@compute @workgroup_size(${SPHERICAL_CONVEX_CHUNK_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3<u32>,
  @builtin(num_workgroups) workgroupCount: vec3<u32>,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${SPHERICAL_CONVEX_MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${SPHERICAL_CONVEX_CHUNK_WORKGROUP_SIZE}u + localInvocationIndex;
  if (index >= chunkParameters[1]) { return; }
  ${Object.entries(rowComponents)
    .map(
      ([name, components]) =>
        `${name}Offset = arrayLength(&${name}) - chunkParameters[1] * ${components}u;`
    )
    .join('\n  ')}
  ${kernelProps.body}
}`;
          const parameters = slots[chunk.chunkIndex];
          const layout = layouts[chunk.chunkIndex];
          return createGPUComputeCommandNode<Parameters>({
            id: `${id}-chunk-${chunk.chunkIndex}`,
            workload: {
              operation: 'GPUSphericalConvexContains',
              commandCount: 1,
              maximumInvocationCount: chunk.length,
              maximumWorkgroupCount: layout.x * layout.y * layout.z
            },
            resources: [
              ...kernelProps.bindings.map(binding => ({
                buffer: binding.view,
                usage:
                  binding.access === 'read'
                    ? ('storage-read' as const)
                    : ('storage-read-write' as const)
              })),
              {buffer: parameters, usage: 'storage-read'}
            ],
            compile: ({device}) => {
              const kernel = new Kernel(device, {
                id: `${id}-chunk`,
                source,
                shaderLayout: {
                  bindings: [
                    ...kernelProps.bindings.map((binding, location) => ({
                      name: binding.name,
                      group: 0,
                      location,
                      type:
                        binding.access === 'read'
                          ? ('read-only-storage' as const)
                          : ('storage' as const)
                    })),
                    {name: 'chunkParameters', group: 0, location: 6, type: 'read-only-storage'}
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
        })
    ];
  }

  private getKernelProps<Parameters>(
    props: PackedSphericalConvexContainsProps
  ): WGSLKernelProps<Parameters> {
    const {id} = this;
    return {
      id,
      operation: 'GPUSphericalConvexContains',
      variant: `${props.boundary ?? 'closed'}-${props.complement ? 'complement' : 'interior'}`,
      invocationCount: props.queries.length,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'offsets', view: props.offsets, type: 'u32', access: 'read'},
        {name: 'queries', view: props.queries, type: 'f32', access: 'read'},
        {name: 'polygonIndices', view: props.polygonIndices, type: 'u32', access: 'read'},
        {name: 'contains', view: props.output.contains, type: 'u32', access: 'read_write'},
        {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
      ],
      declarations: `const CONTAINS_TOLERANCE: f32 = ${props.tolerance ?? 1e-6};
fn containsPosition(row: u32) -> vec3f { let offset = positionsOffset + row * 3u; return vec3f(positions[offset], positions[offset + 1u], positions[offset + 2u]); }
fn containsUnit(point: vec3f) -> bool { let squared = dot(point, point); return (bitcast<u32>(squared) & 0x7f800000u) != 0x7f800000u && abs(squared - 1.0) <= 1e-4; }`,
      body: `let queryOffset = queriesOffset + index * 3u; let point = vec3f(queries[queryOffset], queries[queryOffset + 1u], queries[queryOffset + 2u]);
let polygon = polygonIndices[polygonIndicesOffset + index]; var failure = 0u; var inside = true; var boundary = false;
if (!containsUnit(point) || polygon >= ${props.offsets.length - 1}u) { failure = 1u; }
if (failure == 0u) {
  let begin = offsets[offsetsOffset + polygon]; let end = offsets[offsetsOffset + polygon + 1u];
  if (begin > end || end > ${props.positions.length}u || end - begin < ${GPU_SPHERICAL_CONVEX_MINIMUM_VERTICES}u || end - begin > ${GPU_SPHERICAL_CONVEX_MAXIMUM_VERTICES}u) { failure = 1u; }
  if (failure == 0u) {
    var center = vec3f(0.0);
    for (var vertex = begin; vertex < end; vertex++) { let position = containsPosition(vertex); if (!containsUnit(position)) { failure = 1u; } center += position; }
    if (length(center) < CONTAINS_TOLERANCE) { failure = 1u; } else { center = normalize(center); }
    for (var edge = begin; edge < end && failure == 0u; edge++) {
      let first = containsPosition(edge); let second = containsPosition(select(edge + 1u, begin, edge + 1u == end));
      let crossing = cross(first, second);
      if (length(crossing) <= CONTAINS_TOLERANCE || dot(center, first) <= CONTAINS_TOLERANCE) { failure = 1u; break; }
      let normal = normalize(crossing);
      if (dot(normal, center) <= CONTAINS_TOLERANCE) { failure = 1u; break; }
      for (var vertex = begin; vertex < end; vertex++) { if (dot(normal, containsPosition(vertex)) < -CONTAINS_TOLERANCE) { failure = 1u; } }
      let side = dot(normal, point); inside = inside && side >= -CONTAINS_TOLERANCE; boundary = boundary || abs(side) <= CONTAINS_TOLERANCE;
    }
  }
}
boundary = boundary && inside;
var accepted = ${props.complement ? '!inside' : 'inside'};
if (boundary) { accepted = ${props.boundary !== 'open'}; if (failure == 0u) { failure = 2u; } }
contains[containsOffset + index] = select(0u, 1u, accepted && failure != 1u); status[statusOffset + index] = failure;`
    };
  }
}
