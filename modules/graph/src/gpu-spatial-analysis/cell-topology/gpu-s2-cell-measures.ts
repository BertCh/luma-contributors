// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {assert} from '@luma.gl/core';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {dggs} from '@luma.gl/shaders';
import {createWGSLKernelNode} from '../../utils/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../utils/gpu-contributor-utils';
import {S2_ANALYSIS_WGSL} from './s2-analysis-wgsl';
export type GPUS2CellMeasuresProps = {
  id?: string;
  cells: GraphDataView<'uint32x2'>;
  output: {
    /** Spherical area (steradians), perimeter (radians). Multiply by radius squared/radius externally. */
    measures: GraphDataView<'float32x2'>;
    /** 0 success, 1 invalid ID. */
    status: GraphDataView<'uint32'>;
  };
};
/** S2 spherical cell measurements at levels 0–30 using cancellation-free local increments. */
export class GPUS2CellMeasures implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUS2CellMeasuresProps) {
    this.id = props.id ?? 's2-cell-measures';
    validatePackedView(props.cells, ['uint32x2'], this.id);
    validatePackedView(props.output.measures, ['float32x2'], this.id);
    validatePackedView(props.output.status, ['uint32'], this.id);
    // One measurement/status row per cell.
    assert(
      props.output.measures.length === props.cells.length &&
        props.output.status.length === props.cells.length
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [props.cells]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    validateGraphViewsBelongToGraph(id, graph, [props.cells, ...Object.values(props.output)]);
    return [
      createWGSLKernelNode(graph, {
        id,
        operation: 'GPUS2CellMeasures',
        variant: 'unit-sphere',
        invocationCount: props.cells.length,
        bindings: [
          {name: 'cells', view: props.cells, type: 'u32', access: 'read'},
          {name: 'measures', view: props.output.measures, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `${dggs.source}\n${S2_ANALYSIS_WGSL}`,
        body: `let cell = vec2u(cells[cellsOffset + index * 2u + 1u], cells[cellsOffset + index * 2u]);
var failure = select(1u, 0u, dggs_s2_is_valid_cell_id(cell));
var area = bitcast<f32>(0x7fc00000u | (index & 0u)); var perimeter = area;
if (failure == 0u) {
  let measured = s2AnalysisStableMeasures(cell);
  area = measured.x; perimeter = measured.y;
}
measures[measuresOffset + index * 2u] = area; measures[measuresOffset + index * 2u + 1u] = perimeter; status[statusOffset + index] = failure;`
      })
    ];
  }
}
