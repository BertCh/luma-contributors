// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {dggs, fp64} from '@luma.gl/shaders';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateSinglePackedView,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {S2_ANALYSIS_WGSL} from './s2-analysis-wgsl';
import {CELL_MEASURES_FP64_WGSL} from '../cell-indexing/cell-measures-fp64-wgsl';
import {S2_CELL_MEASURES_FP64_WGSL} from '../cell-indexing/s2-cell-measures-fp64-wgsl';

/** Status of one cell: the measures are valid. */
export const GPU_S2_CELL_MEASURES_STATUS_VALID = 0;
/** Status of one cell: the input is not a valid S2 cell ID, so its measures are NaN. */
export const GPU_S2_CELL_MEASURES_STATUS_INVALID = 1;
export type GPUS2CellMeasuresProps = {
  id?: string;
  cells: GraphDataView<'uint32x2'>;
  /** Compile-time arithmetic choice; defaults to 'float32'. Outputs stay float32. */
  precision?: 'float32' | 'fp64';
  output: {
    /** Spherical area (steradians), perimeter (radians). Multiply by radius squared/radius externally. */
    measures: GraphDataView<'float32x2'>;
    /**
     * Per cell: {@link GPU_S2_CELL_MEASURES_STATUS_VALID} (0) or
     * {@link GPU_S2_CELL_MEASURES_STATUS_INVALID} (1, invalid cell ID).
     */
    status: GraphDataView<'uint32'>;
  };
};
/**
 * S2 spherical cell measurements at levels 0–30 using cancellation-free local increments.
 * Per-frame: cells contents. Compile-time: precision, row count and view layouts.
 * With precision 'fp64', uses the published double-single module for projection and local increments.
 */
export class GPUS2CellMeasures implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUS2CellMeasuresProps) {
    this.id = props.id ?? 's2-cell-measures';
    validateRequirement(
      this.id,
      props.precision === undefined || ['float32', 'fp64'].includes(props.precision),
      "precision must be 'float32' or 'fp64'"
    );
    validateSinglePackedView(props.cells, ['uint32x2'], this.id);
    validateSinglePackedView(props.output.measures, ['float32x2'], this.id);
    validateSinglePackedView(props.output.status, ['uint32'], this.id);
    // One measurement/status row per cell.
    validateRequirement(
      this.id,
      props.output.measures.length === props.cells.length &&
        props.output.status.length === props.cells.length,
      'output.measures and output.status must hold one entry per cell'
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [props.cells]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [props.cells, ...Object.values(props.output)]);
    const usesFP64 = props.precision === 'fp64';
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUS2CellMeasures',
        variant: usesFP64 ? 'unit-sphere-fp64' : 'unit-sphere',
        modules: usesFP64 ? [fp64] : undefined,
        invocationCount: props.cells.length,
        bindings: [
          {name: 'cells', view: props.cells, type: 'u32', access: 'read'},
          {name: 'measures', view: props.output.measures, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${dggs.source}\n${usesFP64 ? `${CELL_MEASURES_FP64_WGSL}\n${S2_CELL_MEASURES_FP64_WGSL}` : S2_ANALYSIS_WGSL}`,
        body: `let cell = vec2u(cells[cellsOffset + index * 2u + 1u], cells[cellsOffset + index * 2u]);
var failure = select(1u, 0u, dggs_s2_is_valid_cell_id(cell));
var area = bitcast<f32>(0x7fc00000u | (index & 0u)); var perimeter = area;
if (failure == 0u) {
  let measured = ${usesFP64 ? 'cellMeasuresGetS2' : 's2AnalysisStableMeasures'}(cell);
  area = measured.${usesFP64 ? 'area' : 'x'}; perimeter = measured.${usesFP64 ? 'perimeter' : 'y'};
}
measures[measuresOffset + index * 2u] = area; measures[measuresOffset + index * 2u + 1u] = perimeter; status[statusOffset + index] = failure;`
      })
    ];
  }
}
