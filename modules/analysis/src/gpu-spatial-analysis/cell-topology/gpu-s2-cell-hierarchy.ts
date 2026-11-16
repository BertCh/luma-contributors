// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  createGPUComputeCommandNode,
  getBoundedDispatchLayout,
  getViewBinding,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {Kernel} from '@luma.gl/graph';
import {
  createGraphChunkDispatchParameters,
  planGraphChunkDispatch,
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges,
  validateGraphChunkDispatchBindings,
  type GraphChunkDispatchPlan
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  getGraphViewChunks,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateSinglePackedView,
  validateCapacity,
  validateRequirement,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {WGSLKernelProps} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {dggs} from '@luma.gl/shaders';
import {S2_NEIGHBORS_WGSL} from './s2-neighbor-wgsl';
import {S2_TOPOLOGY_WGSL} from './s2-topology-wgsl';

/** Fixed workgroup size makes chunk lengths runtime dispatch metadata. */
const CHUNK_WORKGROUP_SIZE = 256;
/** Storage component indices of whole fixed-stride records cannot wrap uint32. */
const MAXIMUM_CHUNK_COMPONENT_COUNT = 0xffffffff;
/** Guard padded workgroups before multiplying their index into a uint32 invocation index. */
const MAXIMUM_LINEAR_WORKGROUP_COUNT = 16777216;

/** Status of one cell: the output is valid. */
export const GPU_S2_CELL_HIERARCHY_STATUS_VALID = 0;
/** Status of one cell: the input is not a valid S2 cell ID. */
export const GPU_S2_CELL_HIERARCHY_STATUS_INVALID = 1;
/** Status of one cell: the level is unavailable (children of a leaf, or a parent below the cell). */
export const GPU_S2_CELL_HIERARCHY_STATUS_UNAVAILABLE_LEVEL = 2;

/** Deepest S2 level: leaf cells, whose IDs end in the lowest set bit. */
const S2_MAXIMUM_LEVEL = 30;

export type GPUS2CellHierarchyProps = {
  id?: string;
  /** Exact little-endian low/high S2 cell IDs; no f32 key conversion. */
  cells: GraphDataView<'uint32x2'> | GraphVectorView<'uint32x2'>;
  operation: 'parent' | 'children' | 'range' | 'neighbors';
  /** Parent target level (0–30), ignored for immediate children and leaf ranges. */
  level?: number;
  output: {
    /** One parent, four Hilbert-ordered immediate children, four edge neighbors (left/right/down/up in face coordinates), or two inclusive leaf bounds per input. */
    cells: GraphDataView<'uint32x2'> | GraphVectorView<'uint32x2'>;
    /**
     * Per cell: {@link GPU_S2_CELL_HIERARCHY_STATUS_VALID} (0),
     * {@link GPU_S2_CELL_HIERARCHY_STATUS_INVALID} (1, invalid key) or
     * {@link GPU_S2_CELL_HIERARCHY_STATUS_UNAVAILABLE_LEVEL} (2, leaf children or descendant parent).
     */
    status: GraphDataView<'uint32'> | GraphVectorView<'uint32'>;
  };
};

/**
 * Exact S2 integer hierarchy at all 31 levels, including mixed-level input batches.
 *
 * Accepts packed or chunked row inputs; output chunks mirror inputs and keep whole fixed-stride records.
 */
export class GPUS2CellHierarchy implements GPUCommandNodeProducer {
  readonly id: string;
  readonly stride: number;
  /** Upload plus one dispatch per nonempty chunk. */
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  private readonly recordChunks: Readonly<Record<string, readonly GraphDataView[]>>;

  constructor(readonly props: GPUS2CellHierarchyProps) {
    this.id = props.id ?? 's2-cell-hierarchy';
    this.stride =
      props.operation === 'children' || props.operation === 'neighbors'
        ? 4
        : props.operation === 'range'
          ? 2
          : 1;
    const id = this.id;
    validateRequirement(
      id,
      ['parent', 'children', 'range', 'neighbors'].includes(props.operation),
      "operation must be 'parent', 'children', 'range' or 'neighbors'"
    );
    if (props.operation === 'parent')
      validateCapacity(id, 'level', props.level as number, 0, S2_MAXIMUM_LEVEL);
    this.chunkPlan = planGraphChunkDispatch(id, {cells: props.cells, status: props.output.status});
    validateCapacity(id, 'cells row count', this.chunkPlan.totalLength, 0);
    for (const chunk of this.chunkPlan.chunks) {
      if (chunk.views['cells'])
        validateSinglePackedView(
          chunk.views['cells'],
          ['uint32x2'],
          `${id} cells chunk ${chunk.chunkIndex}`
        );
      validateMaximumCount(
        id,
        'cells chunk components',
        chunk.length * 2,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
      if (chunk.views['status'])
        validateSinglePackedView(
          chunk.views['status'],
          ['uint32'],
          `${id} status chunk ${chunk.chunkIndex}`
        );
      validateMaximumCount(
        id,
        'status chunk components',
        chunk.length,
        MAXIMUM_CHUNK_COMPONENT_COUNT
      );
    }
    const recordChunks: Record<string, readonly GraphDataView[]> = {};
    recordChunks['outputCells'] = props.output.cells ? getGraphViewChunks(props.output.cells) : [];
    if (props.output.cells) {
      validateRequirement(
        id,
        props.output.cells.length === this.chunkPlan.totalLength * this.stride,
        'output.cells length must equal input rows * record stride'
      );
      validateRequirement(
        id,
        recordChunks['outputCells'].length === this.chunkPlan.chunks.length,
        'output.cells must match input chunk topology'
      );
      for (const chunk of this.chunkPlan.chunks) {
        const view = recordChunks['outputCells'][chunk.chunkIndex];
        validateSinglePackedView(view, ['uint32x2'], `${id} outputCells chunk ${chunk.chunkIndex}`);
        validateRequirement(
          id,
          view.length === chunk.length * this.stride,
          `output.cells chunk ${chunk.chunkIndex} length must equal input chunk rows * record stride`
        );
        validateMaximumCount(
          id,
          'outputCells chunk components',
          view.length * 2,
          MAXIMUM_CHUNK_COMPONENT_COUNT
        );
      }
    }
    this.recordChunks = recordChunks;
    validateDisjointWritableViews(id, props.output);
    this.estimatedCommandNodeCount =
      (this.chunkPlan.chunks.length > 0 ? 1 : 0) + this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [props.cells]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, stride} = this;
    validateMaximumCount(
      id,
      'storage bindings',
      4,
      graph.device.limits.maxStorageBuffersPerShaderStage
    );
    validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
    validateGraphChunkBindingRanges(graph, id, {'output.cells': props.output.cells});
    const invocationsPerRow = 1;
    const layouts = this.chunkPlan.chunks.map(chunk =>
      getBoundedDispatchLayout(
        id,
        Math.max(chunk.length * invocationsPerRow, 1),
        CHUNK_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );
    const {slots, nodes} = createGraphChunkDispatchParameters(graph, {
      id: `${id}-chunks-${props.operation}`,
      plan: this.chunkPlan
    });
    // Every chunk closes over the same source string, including assembled fp64 modules.
    let sharedShaderSource: string | undefined;
    const chunkNodes = this.chunkPlan.chunks
      .filter(chunk => chunk.length > 0)
      .map(chunk => {
        const kernelProps: WGSLKernelProps<Parameters> = {
          id,
          operation: 'GPUS2CellHierarchy',
          variant: props.operation,
          invocationCount: chunk.length * invocationsPerRow,
          bindings: [
            {name: 'chunkParameters', view: slots[chunk.chunkIndex], type: 'u32', access: 'read'},
            {name: 'cells', view: chunk.views['cells'], type: 'u32', access: 'read'},
            {
              name: 'output',
              view: this.recordChunks['outputCells'][chunk.chunkIndex],
              type: 'u32',
              access: 'read_write'
            },
            {name: 'status', view: chunk.views['status'], type: 'u32', access: 'read_write'}
          ],
          declarations: `${dggs.source}\n${S2_NEIGHBORS_WGSL}\n${S2_TOPOLOGY_WGSL}`,
          body: `let cell = vec2u(cells[cellsOffset + index * 2u + 1u], cells[cellsOffset + index * 2u]);
var failure = select(1u, 0u, dggs_s2_is_valid_cell_id(cell));
let level = dggs_s2_get_level(cell);
${props.operation === 'parent' ? `if (failure == 0u && level < ${props.level}u) { failure = 2u; }` : props.operation === 'children' ? 'if (failure == 0u && level == 30u) { failure = 2u; }' : ''}
let marker = dggs_u64_shift_left(vec2u(0u, 1u), 2u * (30u - level));
for (var slot = 0u; slot < ${stride}u; slot++) {
  var result = vec2u(0u);
  if (failure == 0u) {
    ${
      props.operation === 'neighbors'
        ? 'result = s2Neighbor(cell, slot);'
        : props.operation === 'parent'
          ? `result = cellTopologyS2GetParent(cell, ${props.level}u);`
          : props.operation === 'children'
            ? 'result = cellTopologyS2GetChild(cell, level + 1u, 1u, slot);'
            : `let delta = dggs_u64_subtract(marker, vec2u(0u, 1u));
    result = select(dggs_u64_subtract(cell, delta), dggs_u64_add(cell, delta), slot == 1u);`
    }
  }
  let offset = outputOffset + (index * ${stride}u + slot) * 2u;
  output[offset] = result.y; output[offset + 1u] = result.x;
}
status[statusOffset + index] = failure;`
        };
        const storageDeclarations = kernelProps.bindings
          .map(
            (binding, location) =>
              `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
          )
          .join('\n');
        const bindingComponentCounts: Record<string, number> = {
          cells: 2,
          output: stride * 2,
          status: 1
        };
        const bindingOffsets = kernelProps.bindings
          .filter(binding => binding.name !== 'chunkParameters')
          .map(
            binding =>
              `let ${binding.name}Offset = arrayLength(&${binding.name}) - rowCount * ${bindingComponentCounts[binding.name]}u;`
          )
          .join('\n  ');
        const source =
          sharedShaderSource ??
          `${storageDeclarations}
${kernelProps.declarations ?? ''}
@compute @workgroup_size(${CHUNK_WORKGROUP_SIZE})
fn main(@builtin(workgroup_id) workgroupId: vec3u,
  @builtin(num_workgroups) workgroupCount: vec3u,
  @builtin(local_invocation_index) localInvocationIndex: u32) {
  let workgroupIndex = (workgroupId.z * workgroupCount.y + workgroupId.y) * workgroupCount.x + workgroupId.x;
  if (workgroupIndex >= ${MAXIMUM_LINEAR_WORKGROUP_COUNT}u) { return; }
  let index = workgroupIndex * ${CHUNK_WORKGROUP_SIZE}u + localInvocationIndex;
  let rowCount = chunkParameters[1];
  if (index >= rowCount * ${invocationsPerRow}u) { return; }
  ${bindingOffsets}
  ${kernelProps.body}
}`;
        sharedShaderSource = source;
        const kernelBindings = kernelProps.bindings;
        const operationName = kernelProps.operation;
        const layout = layouts[chunk.chunkIndex];
        return createGPUComputeCommandNode<Parameters>({
          id: `${kernelProps.id}-${chunk.chunkIndex}`,
          resources: kernelProps.bindings.map(binding => ({
            buffer: binding.view,
            usage: binding.access === 'read' ? 'storage-read' : 'storage-read-write'
          })),
          workload: {
            operation: kernelProps.operation,
            ...(kernelProps.variant ? {variant: kernelProps.variant} : {}),
            commandCount: 1,
            maximumWorkgroupCount: layout.x * layout.y * layout.z,
            maximumInvocationCount: chunk.length * invocationsPerRow
          },
          compile: ({device}) => {
            const kernel = new Kernel(device, {
              id: operationName,
              source,
              shaderLayout: {
                bindings: kernelBindings.map((binding, location) => ({
                  name: binding.name,
                  group: 0,
                  location,
                  type: binding.access === 'read' ? 'read-only-storage' : 'storage'
                }))
              }
            });
            return {
              encode: ({computePass, getBuffer}) =>
                kernel.dispatch(computePass, {
                  bindings: Object.fromEntries(
                    kernelBindings.map(binding => [
                      binding.name,
                      getViewBinding(binding.view, getBuffer)
                    ])
                  ),
                  ...layout
                }),
              destroy: () => kernel.destroy()
            };
          }
        });
      });
    return [...nodes, ...chunkNodes];
  }
}
