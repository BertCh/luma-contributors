// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGraphOutputViews} from '../utils/gpu-contributor-utils';
import {assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import type {GPURasterMetadata} from './types';
import {createWGSLKernelNode} from '../utils/wgsl-kernel-nodes';
import {IS_FINITE_VALUE_WGSL} from '../utils/wgsl-common';
import {
  requireRasterLength,
  validateRasterExpansionGraph,
  validateRasterExpansionViews
} from './raster-expansion-utils';

export type GPURasterReconstructionProps = {
  id?: string;
  grid: GPURasterMetadata;
  /** Finite grayscale marker <= mask for dilation, >= mask for erosion; otherwise invalid. */
  marker: GraphDataView<'float32'>;
  mask: GraphDataView<'float32'>;
  method?: 'dilation' | 'erosion';
  connectivity?: 4 | 8;
  maximumIterations?: number;
  output: {
    values: GraphDataView<'float32'>;
    validity: GraphDataView<'uint32'>;
    status: {converged: GraphDataView<'uint32'>};
  };
};

/** Geodesic reconstruction under an explicit mask, using synchronous bounded GPU rounds. */
export class GPURasterReconstruction implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterReconstructionProps;
  constructor(props: GPURasterReconstructionProps) {
    this.id = props.id ?? 'raster-reconstruction';
    this.props = props;
    const count = validateRasterExpansionViews(
      this.id,
      props.grid,
      [props.marker, props.mask],
      getGraphOutputViews(props.output)
    );
    for (const view of [props.marker, props.mask, props.output.values, props.output.validity])
      requireRasterLength(view, count);
    requireRasterLength(props.output.status.converged, 1);
    assert(Number.isInteger(props.maximumIterations ?? 64) && (props.maximumIterations ?? 64) > 0);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {grid, output} = props;
    const count = grid.width * grid.height;
    const dilation = props.method !== 'erosion';
    validateRasterExpansionGraph(id, graph, [
      props.marker,
      props.mask,
      ...getGraphOutputViews(output)
    ]);
    const first = createTransientView(graph, `${id}-first`, 'float32', count);
    const second = createTransientView(graph, `${id}-second`, 'float32', count);
    const changed = createTransientView(graph, `${id}-changed`, 'uint32', count);
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-init`,
        operation: 'GPURasterReconstruction',
        invocationCount: count,
        bindings: [
          {name: 'marker', view: props.marker, type: 'f32', access: 'read'},
          {name: 'mask', view: props.mask, type: 'f32', access: 'read'},
          {name: 'values', view: first, type: 'f32', access: 'read_write'}
        ],
        declarations: IS_FINITE_VALUE_WGSL,
        body: `let markerValue=marker[markerOffset+index];let maskValue=mask[maskOffset+index];let valid=isFiniteValue(markerValue)&&isFiniteValue(maskValue)&&markerValue ${dilation ? '<=' : '>='} maskValue;values[valuesOffset+index]=select(bitcast<f32>(0x7fc00000u | (index & 0u)),markerValue,valid);`
      })
    ];
    let previous = first;
    let next = second;
    // One extra round tests whether the final iterate is a true fixed point.
    for (let iteration = 0; iteration <= (props.maximumIterations ?? 64); iteration++) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-round-${iteration}`,
          operation: 'GPURasterReconstruction',
          invocationCount: count,
          bindings: [
            {name: 'previous', view: previous, type: 'f32', access: 'read'},
            {name: 'mask', view: props.mask, type: 'f32', access: 'read'},
            {name: 'next', view: next, type: 'f32', access: 'read_write'},
            {name: 'changed', view: changed, type: 'u32', access: 'read_write'}
          ],
          declarations: IS_FINITE_VALUE_WGSL,
          body: `let original=previous[previousOffset+index];var value=original;
if(isFiniteValue(original)){
 for(var rowOffset=-1;rowOffset<=1;rowOffset++){for(var columnOffset=-1;columnOffset<=1;columnOffset++){
  ${props.connectivity === 8 ? '' : 'if(abs(rowOffset)+abs(columnOffset)>1){continue;}'}
  let column=i32(index%${grid.width}u)+columnOffset;let row=i32(index/${grid.width}u)+rowOffset;
  if(column<0||row<0||column>=${grid.width}||row>=${grid.height}){continue;}
  let neighbor=previous[previousOffset+u32(row)*${grid.width}u+u32(column)];if(isFiniteValue(neighbor)){value=${dilation ? 'max' : 'min'}(value,neighbor);}
 }}value=${dilation ? 'min' : 'max'}(value,mask[maskOffset+index]);
}
next[nextOffset+index]=value;changed[changedOffset+index]=u32(isFiniteValue(original)&&value!=original);`
        })
      );
      [previous, next] = [next, previous];
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-status`,
        operation: 'GPURasterReconstruction',
        invocationCount: 1,
        bindings: [
          {name: 'changed', view: changed, type: 'u32', access: 'read'},
          {name: 'converged', view: output.status.converged, type: 'u32', access: 'read_write'}
        ],
        body: `var complete=1u;for(var cell=0u;cell<${count}u;cell++){complete &= 1u-changed[changedOffset+cell];}converged[convergedOffset]=complete;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPURasterReconstruction',
        invocationCount: count,
        bindings: [
          {name: 'previous', view: previous, type: 'f32', access: 'read'},
          {name: 'converged', view: output.status.converged, type: 'u32', access: 'read'},
          {name: 'values', view: output.values, type: 'f32', access: 'read_write'},
          {name: 'validity', view: output.validity, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_VALUE_WGSL,
        body: `let value=previous[previousOffset+index];let valid=converged[convergedOffset]!=0u&&isFiniteValue(value);values[valuesOffset+index]=select(bitcast<f32>(0x7fc00000u | (index & 0u)),value,valid);validity[validityOffset+index]=u32(valid);`
      })
    );
    return nodes;
  }
}

export type GPURasterFillHolesProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: GraphDataView<'uint32'>;
  connectivity?: 4 | 8;
  maximumIterations?: number;
  output: {values: GraphDataView<'uint32'>; status: {converged: GraphDataView<'uint32'>}};
};
/** Fills background components not connected to the boundary, using the selected background connectivity. */
export class GPURasterFillHoles implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterFillHolesProps;
  constructor(props: GPURasterFillHolesProps) {
    this.id = props.id ?? 'raster-fill-holes';
    this.props = props;
    const count = validateRasterExpansionViews(
      this.id,
      props.grid,
      [props.input],
      getGraphOutputViews(props.output)
    );
    requireRasterLength(props.input, count);
    requireRasterLength(props.output.values, count);
    requireRasterLength(props.output.status.converged, 1);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const count = props.grid.width * props.grid.height;
    const marker = createTransientView(graph, `${id}-marker`, 'float32', count);
    const mask = createTransientView(graph, `${id}-mask`, 'float32', count);
    const reconstructed = createTransientView(graph, `${id}-reconstructed`, 'float32', count);
    const validity = createTransientView(graph, `${id}-validity`, 'uint32', count);
    validateRasterExpansionGraph(id, graph, [props.input, ...getGraphOutputViews(props.output)]);
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-prepare`,
        operation: 'GPURasterFillHoles',
        invocationCount: count,
        bindings: [
          {name: 'input', view: props.input, type: 'u32', access: 'read'},
          {name: 'marker', view: marker, type: 'f32', access: 'read_write'},
          {name: 'mask', view: mask, type: 'f32', access: 'read_write'}
        ],
        body: `let background=input[inputOffset+index]==0u;let column=index%${props.grid.width}u;let row=index/${props.grid.width}u;let boundary=column==0u||row==0u||column==${props.grid.width - 1}u||row==${props.grid.height - 1}u;mask[maskOffset+index]=f32(background);marker[markerOffset+index]=f32(background&&boundary);`
      }),
      ...new GPURasterReconstruction({
        id: `${id}-reconstruct`,
        grid: props.grid,
        marker,
        mask,
        connectivity: props.connectivity,
        maximumIterations: props.maximumIterations,
        output: {
          values: reconstructed,
          validity,
          status: {converged: props.output.status.converged}
        }
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPURasterFillHoles',
        invocationCount: count,
        bindings: [
          {name: 'reconstructed', view: reconstructed, type: 'f32', access: 'read'},
          {name: 'validity', view: validity, type: 'u32', access: 'read'},
          {name: 'values', view: props.output.values, type: 'u32', access: 'read_write'}
        ],
        body: 'values[valuesOffset+index]=select(0xfffffffeu,u32(reconstructed[reconstructedOffset+index]==0.0),validity[validityOffset+index]!=0u);'
      })
    ];
  }
}
