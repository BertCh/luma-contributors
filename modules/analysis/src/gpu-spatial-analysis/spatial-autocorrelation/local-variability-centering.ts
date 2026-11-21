// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** @internal Persist reference-centered values before any running moment or standardization. */
export function getLocalVariabilityCenteringNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rows: number;
    columns: number;
    values: GraphDataView<'float32'>;
    bindings: WGSLKernelBinding[];
    declarations: string;
  }
): {nodes: GPUCommandNode<Parameters>[]; values: GraphDataView<'float32'>} {
  const candidates = createTransientView(
    graph,
    `${props.id}-origin-candidates`,
    'uint32',
    props.rows
  );
  const origin = createTransientView(graph, `${props.id}-origin`, 'uint32', 1);
  const values = createTransientView(
    graph,
    `${props.id}-centered-values`,
    'float32',
    props.rows * props.columns
  );
  return {
    values,
    nodes: [
      createWGSLKernelNode(graph, {
        id: `${props.id}-origin-candidates`,
        operation: 'GPUMultivariateLocalGeary',
        variant: 'origin-candidates',
        invocationCount: props.rows,
        bindings: [
          ...props.bindings,
          {name: 'candidates', view: candidates, type: 'u32', access: 'read_write'}
        ],
        declarations: props.declarations,
        body: 'candidates[candidatesOffset + index] = select(0xffffffffu, index, included(index));'
      }),
      ...new GPUReduction({
        id: `${props.id}-origin`,
        input: candidates,
        output: origin,
        operation: 'min'
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${props.id}-center-values`,
        operation: 'GPUMultivariateLocalGeary',
        variant: 'center-values',
        invocationCount: props.rows * props.columns,
        bindings: [
          {name: 'values', view: props.values, type: 'f32', access: 'read'},
          {name: 'origin', view: origin, type: 'u32', access: 'read'},
          {name: 'centered', view: values, type: 'f32', access: 'read_write'}
        ],
        declarations: GET_NAN_WGSL,
        body: `var value = getNaN(); let row = origin[originOffset];
if (row < ${props.rows}u) { value = values[valuesOffset + index] - values[valuesOffset + row * ${props.columns}u + index % ${props.columns}u]; }
centered[centeredOffset + index] = value;`
      })
    ]
  };
}
