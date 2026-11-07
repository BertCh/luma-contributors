// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2020-2021 Uber Technologies, Inc.
// Vertex ownership and orientation follow H3 vertex.c (Apache-2.0).
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {dggs} from '@luma.gl/shaders';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../utils/gpu-contributor-utils';
import {H3_NEIGHBOR_WGSL} from './h3-neighbor-wgsl';
import {CELL_INDEX_H3_BASE_CELL_PACKED_TABLE} from '../cell-indexing/h3-index-tables';

export type GPUH3CellIdentitiesProps = {
  id?: string;
  /** Little-endian (low, high) H3 cells. Invalid cells produce zeros. */
  cells: GraphDataView<'uint32x2'>;
  operation: 'directed-edges' | 'vertices';
  /** Six little-endian keys per cell. Edge slot d-1 is direction d; deleted pentagon edge is zero.
   * Vertex slots follow cellToVertexes; pentagon slot 5 is zero. These are H3 edge/vertex IDs,
   * not cell IDs, and must not be passed to cell-only APIs.
   */
  output: GraphDataView<'uint32x2'>;
};

/** Exact integer H3 directed edge and canonical vertex identities, including pentagons and faces. */
export class GPUH3CellIdentities implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUH3CellIdentitiesProps) {
    this.id = props.id ?? 'h3-cell-identities';
    validatePackedView(props.cells, ['uint32x2'], this.id);
    validatePackedView(props.output, ['uint32x2'], this.id);
    if (props.output.length !== props.cells.length * 6)
      throw new Error('H3 identities require six output rows per cell');
    validateGraphOutputsDisjointFromInputs(this.id, [props.output], [props.cells]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [props.cells, props.output]);
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUH3CellIdentities',
        variant: props.operation,
        invocationCount: props.cells.length * 6,
        bindings: [
          {name: 'cells', view: props.cells, type: 'u32', access: 'read'},
          {name: 'output', view: props.output, type: 'u32', access: 'read_write'}
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
      })
    ];
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
