// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {dggs} from '@luma.gl/shaders';
import {S2_POSITION_TO_IJ, S2_POSITION_TO_ORIENTATION} from '../cell-indexing/cell-index-wgsl';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateSinglePackedView,
  validateCapacity,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

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
  cells: GraphDataView<'uint32x2'>;
  operation: 'parent' | 'children' | 'range' | 'neighbors';
  /** Parent target level (0–30), ignored for immediate children and leaf ranges. */
  level?: number;
  output: {
    /** One parent, four Hilbert-ordered immediate children, four edge neighbors (left/right/down/up in face coordinates), or two inclusive leaf bounds per input. */
    cells: GraphDataView<'uint32x2'>;
    /**
     * Per cell: {@link GPU_S2_CELL_HIERARCHY_STATUS_VALID} (0),
     * {@link GPU_S2_CELL_HIERARCHY_STATUS_INVALID} (1, invalid key) or
     * {@link GPU_S2_CELL_HIERARCHY_STATUS_UNAVAILABLE_LEVEL} (2, leaf children or descendant parent).
     */
    status: GraphDataView<'uint32'>;
  };
};

/** Exact S2 integer hierarchy at all 31 levels, including mixed-level input batches. */
export class GPUS2CellHierarchy implements GPUCommandNodeProducer {
  readonly id: string;
  readonly stride: number;
  constructor(readonly props: GPUS2CellHierarchyProps) {
    this.id = props.id ?? 's2-cell-hierarchy';
    this.stride =
      props.operation === 'children' || props.operation === 'neighbors'
        ? 4
        : props.operation === 'range'
          ? 2
          : 1;
    validateSinglePackedView(props.cells, ['uint32x2'], this.id);
    validateSinglePackedView(props.output.cells, ['uint32x2'], this.id);
    validateSinglePackedView(props.output.status, ['uint32'], this.id);
    // Parent levels and aligned output strides are structural constraints.
    if (props.operation === 'parent') {
      validateCapacity(this.id, 'level', props.level as number, 0, S2_MAXIMUM_LEVEL);
    }
    validateRequirement(
      this.id,
      props.output.cells.length === props.cells.length * this.stride &&
        props.output.status.length === props.cells.length,
      `output.cells must hold ${this.stride} entries per cell and output.status one entry per cell`
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [props.cells]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id, stride} = this;
    validateGraphViewsBelongToGraph(id, graph, [props.cells, ...Object.values(props.output)]);
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUS2CellHierarchy',
        variant: props.operation,
        invocationCount: props.cells.length,
        bindings: [
          {name: 'cells', view: props.cells, type: 'u32', access: 'read'},
          {name: 'output', view: props.output.cells, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${dggs.source}\n${S2_NEIGHBORS_WGSL}`,
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
          ? `let shift = ${2 * (30 - (props.level ?? 0))}u;
    result = dggs_u64_shift_left(dggs_u64_shift_right(cell, shift), shift) | dggs_u64_shift_left(vec2u(0u, 1u), shift);`
          : props.operation === 'children'
            ? `let childMarker = dggs_u64_shift_right(marker, 2u);
    let first = dggs_u64_add(dggs_u64_subtract(cell, marker), childMarker);
    result = first;
    for (var step = 0u; step < slot; step++) { result = dggs_u64_add(result, dggs_u64_shift_left(childMarker, 1u)); }`
            : `let delta = dggs_u64_subtract(marker, vec2u(0u, 1u));
    result = select(dggs_u64_subtract(cell, delta), dggs_u64_add(cell, delta), slot == 1u);`
    }
  }
  let offset = outputOffset + (index * ${stride}u + slot) * 2u;
  output[offset] = result.y; output[offset + 1u] = result.x;
}
status[statusOffset + index] = failure;`
      })
    ];
  }
}

/** Integer cube-face transitions preserve neighbor identities even beyond f32 resolution. */
const S2_NEIGHBORS_WGSL = /* wgsl */ `
const S2_HIERARCHY_IJ_TO_POSITION = array<u32, 16>(${S2_POSITION_TO_IJ.flatMap(orientation => [0, 1, 2, 3].map(value => `${orientation.indexOf(value)}u`)).join(',')});
const S2_HIERARCHY_ORIENTATION = array<u32, 4>(${S2_POSITION_TO_ORIENTATION.map(value => `${value}u`).join(',')});
const S2_FACE_NORMAL = array<vec3i, 6>(vec3i(1,0,0),vec3i(0,1,0),vec3i(0,0,1),vec3i(-1,0,0),vec3i(0,-1,0),vec3i(0,0,-1));
const S2_FACE_COLUMN = array<vec3i, 6>(vec3i(0,1,0),vec3i(-1,0,0),vec3i(-1,0,0),vec3i(0,0,-1),vec3i(0,0,-1),vec3i(0,1,0));
const S2_FACE_ROW = array<vec3i, 6>(vec3i(0,0,1),vec3i(0,0,1),vec3i(0,-1,0),vec3i(0,-1,0),vec3i(1,0,0),vec3i(1,0,0));
fn s2Encode(face: u32, coordinate: vec2u, level: u32) -> vec2u {
  var orientation = face & 1u;
  var key = dggs_u64_shift_left(vec2u(0u, face), 61u) | dggs_u64_shift_left(vec2u(0u, 1u), 60u - 2u * level);
  for (var step = 0u; step < level; step++) {
    let bit = level - step - 1u;
    let quadrant = (((coordinate.x >> bit) & 1u) << 1u) | ((coordinate.y >> bit) & 1u);
    let position = S2_HIERARCHY_IJ_TO_POSITION[orientation * 4u + quadrant];
    orientation ^= S2_HIERARCHY_ORIENTATION[position];
    key |= dggs_u64_shift_left(vec2u(0u, position), 59u - 2u * step);
  }
  return key;
}
fn s2IntegerDot(first: vec3i, second: vec3i) -> i32 { let product = first * second; return product.x + product.y + product.z; }
fn s2Neighbor(cell: vec2u, slot: u32) -> vec2u {
  let level = dggs_s2_get_level(cell); let size = 1u << level;
  let face = dggs_s2_get_face(cell); let coordinate = dggs_s2_get_ij(cell);
  var next = vec2i(coordinate);
  if (slot < 2u) { next.x += select(-1, 1, slot == 1u); } else { next.y += select(-1, 1, slot == 3u); }
  if (all(next >= vec2i(0)) && all(next < vec2i(i32(size)))) { return s2Encode(face, vec2u(next), level); }
  let direction = select(-1, 1, slot == 1u || slot == 3u);
  let neighborNormal = select(S2_FACE_COLUMN[face], S2_FACE_ROW[face], slot >= 2u) * direction;
  let tangent = select(S2_FACE_ROW[face], S2_FACE_COLUMN[face], slot >= 2u);
  let varyingCoordinate = select(coordinate.y, coordinate.x, slot >= 2u);
  var neighborFace = 0u;
  for (var candidate = 0u; candidate < 6u; candidate++) { if (all(S2_FACE_NORMAL[candidate] == neighborNormal)) { neighborFace = candidate; break; } }
  let fixedColumn = s2IntegerDot(S2_FACE_NORMAL[face], S2_FACE_COLUMN[neighborFace]);
  let fixedRow = s2IntegerDot(S2_FACE_NORMAL[face], S2_FACE_ROW[neighborFace]);
  let column = select(select(varyingCoordinate, size - 1u - varyingCoordinate, s2IntegerDot(tangent, S2_FACE_COLUMN[neighborFace]) < 0), select(0u, size - 1u, fixedColumn > 0), fixedColumn != 0);
  let row = select(select(varyingCoordinate, size - 1u - varyingCoordinate, s2IntegerDot(tangent, S2_FACE_ROW[neighborFace]) < 0), select(0u, size - 1u, fixedRow > 0), fixedRow != 0);
  return s2Encode(neighborFace, vec2u(column, row), level);
}
`;
