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
import {S2_NEIGHBORS_WGSL} from './s2-neighbor-wgsl';
import {S2_TOPOLOGY_WGSL} from './s2-topology-wgsl';
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
      })
    ];
  }
}
