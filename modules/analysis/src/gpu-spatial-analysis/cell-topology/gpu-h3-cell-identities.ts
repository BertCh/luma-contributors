// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2020-2021 Uber Technologies, Inc.
// Vertex ownership and orientation follow H3 vertex.c (Apache-2.0).
import {snapshotCellConstruction} from '../cell-indexing/cell-construction-snapshot';
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
  validateRequirement,
  validateCapacity,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import type {WGSLKernelProps} from '../../analysis-core/kernels/wgsl-kernel-nodes';

import {dggs} from '@luma.gl/shaders';

import {H3_NEIGHBOR_WGSL} from './h3-neighbor-wgsl';
import {CELL_INDEX_H3_BASE_CELL_PACKED_TABLE} from '../cell-indexing/h3-index-tables';

/** Fixed workgroup size makes chunk lengths runtime dispatch metadata. */
const CHUNK_WORKGROUP_SIZE = 256;
/** Storage component indices of whole fixed-stride records cannot wrap uint32. */
const MAXIMUM_CHUNK_COMPONENT_COUNT = 0xffffffff;
/** Guard padded workgroups before multiplying their index into a uint32 invocation index. */
const MAXIMUM_LINEAR_WORKGROUP_COUNT = 16777216;

export type GPUH3CellIdentitiesProps = {
  id?: string;
  /** Little-endian (low, high) H3 cells. Invalid cells produce zeros. */
  cells: GraphDataView<'uint32x2'> | GraphVectorView<'uint32x2'>;
  operation: 'directed-edges' | 'vertices';
  /** Six little-endian keys per cell. Edge slot d-1 is direction d; deleted pentagon edge is zero.
   * Vertex slots follow cellToVertexes; pentagon slot 5 is zero. These are H3 edge/vertex IDs,
   * not cell IDs, and must not be passed to cell-only APIs.
   */
  output: GraphDataView<'uint32x2'> | GraphVectorView<'uint32x2'>;
};

/**
 * Exact integer H3 directed edge and canonical vertex identities, including pentagons and faces.
 *
 * Accepts packed or chunked row inputs; output chunks mirror inputs and keep whole fixed-stride records.
 */
export class GPUH3CellIdentities implements GPUCommandNodeProducer {
  readonly id: string;
  /** Upload plus one dispatch per nonempty chunk. */
  readonly estimatedCommandNodeCount: number;
  private readonly chunkPlan: GraphChunkDispatchPlan;
  private readonly recordChunks: Readonly<Record<string, readonly GraphDataView[]>>;

  constructor(readonly props: GPUH3CellIdentitiesProps) {
    props = snapshotCellConstruction(props, ['id', 'cells', 'operation', 'output']);
    this.props = props;
    this.id = props.id ?? 'h3-cell-identities';
    const id = this.id;
    validateRequirement(
      id,
      ['directed-edges', 'vertices'].includes(props.operation),
      "operation must be 'directed-edges' or 'vertices'"
    );
    this.chunkPlan = planGraphChunkDispatch(id, {cells: props.cells});
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
    }
    const recordChunks: Record<string, readonly GraphDataView[]> = {};
    recordChunks['identities'] = props.output ? getGraphViewChunks(props.output) : [];
    if (props.output) {
      validateRequirement(
        id,
        props.output.length === this.chunkPlan.totalLength * 6,
        'output length must equal input rows * record stride'
      );
      validateRequirement(
        id,
        recordChunks['identities'].length === this.chunkPlan.chunks.length,
        'output must match input chunk topology'
      );
      for (const chunk of this.chunkPlan.chunks) {
        const view = recordChunks['identities'][chunk.chunkIndex];
        validateSinglePackedView(view, ['uint32x2'], `${id} identities chunk ${chunk.chunkIndex}`);
        validateRequirement(
          id,
          view.length === chunk.length * 6,
          `output chunk ${chunk.chunkIndex} length must equal input chunk rows * record stride`
        );
        validateMaximumCount(
          id,
          'identities chunk components',
          view.length * 2,
          MAXIMUM_CHUNK_COMPONENT_COUNT
        );
      }
    }
    this.recordChunks = recordChunks;
    validateDisjointWritableViews(id, {output: props.output});
    this.estimatedCommandNodeCount =
      (this.chunkPlan.chunks.length > 0 ? 1 : 0) + this.chunkPlan.nonemptyChunkCount;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    validateGraphOutputsDisjointFromInputs(this.id, [props.output], [props.cells]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateMaximumCount(
      id,
      'storage bindings',
      3,
      graph.device.limits.maxStorageBuffersPerShaderStage
    );
    validateGraphChunkDispatchBindings(graph, id, this.chunkPlan);
    validateGraphChunkBindingRanges(graph, id, {output: props.output});
    const invocationsPerRow = 6;
    const layouts = this.chunkPlan.chunks.map(chunk =>
      getBoundedDispatchLayout(
        id,
        Math.max(chunk.length * invocationsPerRow, 1),
        CHUNK_WORKGROUP_SIZE,
        graph.device.limits.maxComputeWorkgroupsPerDimension
      )
    );
    const {slots, nodes} = createGraphChunkDispatchParameters(graph, {
      id: `${id}-chunks`,
      plan: this.chunkPlan
    });
    // Every chunk closes over the same source string, including assembled fp64 modules.
    let sharedShaderSource: string | undefined;
    const chunkNodes = this.chunkPlan.chunks
      .filter(chunk => chunk.length > 0)
      .map(chunk => {
        const kernelProps: WGSLKernelProps<Parameters> = {
          id,
          operation: 'GPUH3CellIdentities',
          variant: props.operation,
          invocationCount: chunk.length * invocationsPerRow,
          bindings: [
            {name: 'chunkParameters', view: slots[chunk.chunkIndex], type: 'u32', access: 'read'},
            {name: 'cells', view: chunk.views['cells'], type: 'u32', access: 'read'},
            {
              name: 'output',
              view: this.recordChunks['identities'][chunk.chunkIndex],
              type: 'u32',
              access: 'read_write'
            }
          ],
          declarations: `${dggs.source}\n${H3_NEIGHBOR_WGSL}\n${H3_VERTEX_IDENTITY_WGSL}`,
          body: `let row = index / 6u; let slot = index % 6u;
  let cell = vec2u(cells[cellsOffset + row * 2u + 1u], cells[cellsOffset + row * 2u]);
  var key = vec2u(0u);
  if (dggs_h3_is_valid_cell_id(cell)) {
    ${
      props.operation === 'vertices'
        ? 'key = identityVertex(cell, slot);'
        : `let neighbor = cellTopologyH3Neighbor(cell, slot + 1u);
    if (any(neighbor != vec2u(0u))) { key = vec2u((cell.x & 0x80ffffffu) | 0x10000000u | ((slot + 1u) << 24u), cell.y); }`
    }
  }
  output[outputOffset + index * 2u] = key.y;
  output[outputOffset + index * 2u + 1u] = key.x;`
        };
        const storageDeclarations = kernelProps.bindings
          .map(
            (binding, location) =>
              `@group(0) @binding(${location}) var<storage, ${binding.access}> ${binding.name}: array<${binding.type}>;`
          )
          .join('\n');
        const bindingComponentCounts: Record<string, number> = {cells: 2, output: 6 * 2};
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

/** H3 vertex orientation and ownership helpers. @internal */
export const H3_VERTEX_IDENTITY_WGSL = /* wgsl */ `
const IDENTITY_FACE_BASE: array<i32, 540> = array<i32, 540>(${CELL_INDEX_H3_BASE_CELL_PACKED_TABLE.join(',')});
const IDENTITY_PENTAGONS: array<u32, 12> = array<u32,12>(4u,14u,24u,38u,49u,58u,63u,72u,83u,97u,107u,117u);
const IDENTITY_PENTAGON_FACES: array<u32, 60> = array<u32,60>(
4u,0u,2u,1u,3u, 6u,11u,2u,7u,1u, 5u,10u,1u,6u,0u, 7u,12u,3u,8u,2u,
9u,14u,0u,5u,4u, 8u,13u,4u,9u,3u, 11u,6u,15u,10u,16u, 12u,7u,16u,11u,17u,
10u,5u,19u,14u,15u, 13u,8u,17u,12u,18u, 14u,9u,18u,13u,19u, 15u,19u,17u,18u,16u);
fn identityLess(first: vec2u, second: vec2u) -> bool { return first.x < second.x || (first.x == second.x && first.y < second.y); }
fn identityRotations(cell: vec2u) -> u32 {
  let face = dggs_h3_get_center_face_ijk(cell).face;
  let base = dggs_h3_get_base_cell(cell);
  var rotations = 0u;
  for (var slot = 0u; slot < 27u; slot++) {
    let packed = IDENTITY_FACE_BASE[face * 27u + slot];
    if (packed >= 0 && (u32(packed) & 255u) == base) { rotations = u32(packed) >> 8u; break; }
  }
  if (dggs_h3_is_base_cell_pentagon(base)) {
    var pentagon = 0u;
    for (var slot = 0u; slot < 12u; slot++) { if (IDENTITY_PENTAGONS[slot] == base) { pentagon = slot; break; } }
    let faceIK = IDENTITY_PENTAGON_FACES[pentagon * 5u + 3u];
    let faceJK = IDENTITY_PENTAGON_FACES[pentagon * 5u + 1u];
    if (face != dggs_h3_get_base_cell_home(base).face && (base == 4u || base == 117u || face == faceIK)) { rotations = (rotations + 1u) % 6u; }
    let leading = dggs_h3_get_leading_non_zero_digit(cell);
    if (leading == 3u && face == faceIK) { rotations = (rotations + 5u) % 6u; }
    else if (leading == 5u && face == faceJK) { rotations = (rotations + 1u) % 6u; }
  }
  return rotations;
}
fn identityDirection(cell: vec2u, vertex: u32) -> u32 {
  let hexagon = array<u32,6>(6u,2u,3u,1u,5u,4u);
  let pentagon = array<u32,5>(6u,2u,3u,5u,4u);
  let rotation = identityRotations(cell);
  if (cellTopologyH3IsPentagon(cell)) { return pentagon[(vertex + rotation) % 5u]; }
  return hexagon[(vertex + rotation) % 6u];
}
fn identityVertexNumber(cell: vec2u, direction: u32) -> u32 {
  let hexagon = array<u32,7>(0u,3u,1u,2u,5u,4u,0u);
  let pentagon = array<u32,7>(0u,0u,1u,2u,4u,3u,0u);
  let rotation = identityRotations(cell);
  if (cellTopologyH3IsPentagon(cell)) { return (pentagon[direction] + 10u - rotation) % 5u; }
  return (hexagon[direction] + 6u - rotation) % 6u;
}
fn identityReverseDirection(cell: vec2u, destination: vec2u) -> u32 {
  for (var direction = 1u; direction <= 6u; direction++) {
    if (all(cellTopologyH3Neighbor(cell, direction) == destination)) { return direction; }
  }
  return 0u;
}
fn identityVertex(cell: vec2u, vertex: u32) -> vec2u {
  let count = select(6u, 5u, cellTopologyH3IsPentagon(cell));
  if (vertex >= count) { return vec2u(0u); }
  let left = cellTopologyH3Neighbor(cell, identityDirection(cell, vertex));
  let right = cellTopologyH3Neighbor(cell, identityDirection(cell, (vertex + count - 1u) % count));
  var owner = cell; var ownerVertex = vertex;
  if (identityLess(left, owner)) {
    owner = left;
    let ownerCount = select(6u, 5u, cellTopologyH3IsPentagon(owner));
    ownerVertex = (identityVertexNumber(owner, identityReverseDirection(owner, cell)) + 1u) % ownerCount;
  }
  if (identityLess(right, owner)) {
    owner = right; ownerVertex = identityVertexNumber(owner, identityReverseDirection(owner, cell));
  }
  return vec2u((owner.x & 0x80ffffffu) | 0x20000000u | (ownerVertex << 24u), owner.y);
}`;
