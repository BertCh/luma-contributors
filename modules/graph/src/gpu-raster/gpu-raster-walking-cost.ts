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
import type {GPURasterBufferBand, GPURasterMetadata} from './types';
import {getRasterBufferBandNodes, getRasterBandViews} from './raster-band-utils';
import {createWGSLKernelNode} from '../utils/wgsl-kernel-nodes';
import {IS_FINITE_VALUE_WGSL} from '../utils/wgsl-common';
import {
  requireRasterLength,
  validateRasterExpansionViews,
  validateRasterExpansionGraph
} from './raster-expansion-utils';

export type GPURasterWalkingCostProps = {
  id?: string;
  grid: GPURasterMetadata;
  elevation: GPURasterBufferBand;
  friction: GPURasterBufferBand;
  sources: GraphDataView<'uint32'>;
  /** Coefficient multiplying additive average-friction * ground-distance cost. Default 1. */
  frictionWeight?: number;
  model?:
    | {kind: 'tobler'}
    | {
        kind: 'grass' /** a*distance + b*uphill + c*moderate-downhill + d*steep-downhill. */;
        coefficients: readonly [number, number, number, number];
        downhillSlopeThreshold: number;
      };
  maximumIterations?: number;
  output: {
    costs: GraphDataView<'float32'>;
    validity: GraphDataView<'uint32'>;
    status: {converged: GraphDataView<'uint32'>};
  };
};

/**
 * Directed 8-connected least-cost movement. Elevations and affine distances must use the same
 * linear unit (meters for Tobler seconds). Uphill and downhill transitions are evaluated separately.
 * Tobler uses 6*exp(-3.5*abs(slope+0.05)) km/hour. GRASS-style energy coefficients are explicit;
 * negative/nonfinite transition costs are impassable. Jacobi relaxation is bounded and publishes
 * no plausible finite costs until the final iterate is verified. Whole-grid support is required.
 */
export class GPURasterWalkingCost implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterWalkingCostProps;
  constructor(props: GPURasterWalkingCostProps) {
    this.id = props.id ?? 'raster-walking-cost';
    this.props = props;
    const count = validateRasterExpansionViews(
      this.id,
      props.grid,
      [
        ...getRasterBandViews(props.elevation),
        ...getRasterBandViews(props.friction),
        props.sources
      ],
      getGraphOutputViews(props.output)
    );
    for (const view of [props.sources, props.output.costs, props.output.validity])
      requireRasterLength(view, count);
    requireRasterLength(props.output.status.converged, 1);
    assert(
      Number.isInteger(props.maximumIterations ?? 128) && (props.maximumIterations ?? 128) > 0
    );
    assert(Number.isFinite(props.frictionWeight ?? 1) && (props.frictionWeight ?? 1) >= 0);
    if (props.model?.kind === 'grass') {
      assert(
        props.model.coefficients.every(Number.isFinite) &&
          Number.isFinite(props.model.downhillSlopeThreshold) &&
          props.model.downhillSlopeThreshold < 0
      );
    }
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {grid, output} = props;
    const count = grid.width * grid.height;
    validateRasterExpansionGraph(id, graph, [
      ...getRasterBandViews(props.elevation),
      ...getRasterBandViews(props.friction),
      props.sources,
      ...getGraphOutputViews(output)
    ]);
    const elevation = getRasterBufferBandNodes(
      graph,
      `${id}-elevation`,
      props.elevation,
      grid,
      true
    );
    const friction = getRasterBufferBandNodes(graph, `${id}-friction`, props.friction, grid, true);
    const surface = createTransientView(graph, `${id}-surface`, 'float32x2', count);
    const first = createTransientView(graph, `${id}-first`, 'float32', count),
      second = createTransientView(graph, `${id}-second`, 'float32', count),
      changed = createTransientView(graph, `${id}-changed`, 'uint32', count);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...elevation.nodes,
      ...friction.nodes,
      createWGSLKernelNode(graph, {
        id: `${id}-prepare`,
        operation: 'GPURasterWalkingCost',
        invocationCount: count,
        bindings: [
          {name: 'elevation', view: elevation.band.storage.values, type: 'f32', access: 'read'},
          {name: 'elevationValidity', view: elevation.band.validity!, type: 'u32', access: 'read'},
          {name: 'friction', view: friction.band.storage.values, type: 'f32', access: 'read'},
          {name: 'frictionValidity', view: friction.band.validity!, type: 'u32', access: 'read'},
          {name: 'sources', view: props.sources, type: 'u32', access: 'read'},
          {name: 'surface', view: surface, type: 'f32', access: 'read_write'},
          {name: 'first', view: first, type: 'f32', access: 'read_write'}
        ],
        body: `let valid=elevationValidity[elevationValidityOffset+index]!=0u&&frictionValidity[frictionValidityOffset+index]!=0u&&friction[frictionOffset+index]>=0.0;surface[surfaceOffset+index*2u]=select(bitcast<f32>(0x7fc00000u|(index&0u)),elevation[elevationOffset+index],valid);surface[surfaceOffset+index*2u+1u]=friction[frictionOffset+index];first[firstOffset+index]=select(bitcast<f32>(0x7f800000u|(index&0u)),0.0,valid&&sources[sourcesOffset+index]!=0u);`
      })
    ];
    let previous = first,
      next = second;
    const model = props.model;
    for (let iteration = 0; iteration <= (props.maximumIterations ?? 128); iteration++) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-round-${iteration}`,
          operation: 'GPURasterWalkingCost',
          invocationCount: count,
          bindings: [
            {name: 'surface', view: surface, type: 'f32', access: 'read'},
            {name: 'previous', view: previous, type: 'f32', access: 'read'},
            {name: 'next', view: next, type: 'f32', access: 'read_write'},
            {name: 'changed', view: changed, type: 'u32', access: 'read_write'}
          ],
          declarations: IS_FINITE_VALUE_WGSL,
          body: `let elevation=surface[surfaceOffset+index*2u];let friction=surface[surfaceOffset+index*2u+1u];let original=previous[previousOffset+index];var best=original;if(isFiniteValue(elevation)){
for(var rowOffset=-1;rowOffset<=1;rowOffset++){for(var columnOffset=-1;columnOffset<=1;columnOffset++){
 if(columnOffset==0&&rowOffset==0){continue;}let column=i32(index%${grid.width}u)+columnOffset;let row=i32(index/${grid.width}u)+rowOffset;if(column<0||row<0||column>=${grid.width}||row>=${grid.height}){continue;}
 let neighbor=u32(row)*${grid.width}u+u32(column);let neighborElevation=surface[surfaceOffset+neighbor*2u];if(!isFiniteValue(neighborElevation)){continue;}
 let distance=length(vec2<f32>(${grid.affine[0]}*f32(columnOffset)+${grid.affine[1]}*f32(rowOffset),${grid.affine[3]}*f32(columnOffset)+${grid.affine[4]}*f32(rowOffset)));if(distance<=0.0){continue;}let rise=elevation-neighborElevation;let slope=rise/distance;
 ${model?.kind === 'grass' ? `let energy=${model.coefficients[0]}*distance+select(select(${model.coefficients[2]}*rise,${model.coefficients[3]}*rise,slope<${model.downhillSlopeThreshold}),${model.coefficients[1]}*rise,rise>0.0);` : 'let energy=distance*0.6*exp(3.5*abs(slope+0.05));'}
 let cost=energy+${props.frictionWeight ?? 1}*0.5*(friction+surface[surfaceOffset+neighbor*2u+1u])*distance;if(cost>=0.0&&isFiniteValue(cost)){best=min(best,previous[previousOffset+neighbor]+cost);}
}}
}next[nextOffset+index]=best;changed[changedOffset+index]=u32(best<original);`
        })
      );
      [previous, next] = [next, previous];
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-status`,
        operation: 'GPURasterWalkingCost',
        invocationCount: 1,
        bindings: [
          {name: 'changed', view: changed, type: 'u32', access: 'read'},
          {name: 'converged', view: output.status.converged, type: 'u32', access: 'read_write'}
        ],
        body: `var complete=1u;for(var cell=0u;cell<${count}u;cell++){complete&=1u-changed[changedOffset+cell];}converged[convergedOffset]=complete;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPURasterWalkingCost',
        invocationCount: count,
        bindings: [
          {name: 'previous', view: previous, type: 'f32', access: 'read'},
          {name: 'surface', view: surface, type: 'f32', access: 'read'},
          {name: 'converged', view: output.status.converged, type: 'u32', access: 'read'},
          {name: 'costs', view: output.costs, type: 'f32', access: 'read_write'},
          {name: 'validity', view: output.validity, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_VALUE_WGSL,
        body: 'let complete=converged[convergedOffset]!=0u;let valid=complete&&isFiniteValue(surface[surfaceOffset+index*2u]);costs[costsOffset+index]=select(bitcast<f32>(0x7fc00000u|(index&0u)),previous[previousOffset+index],valid);validity[validityOffset+index]=u32(valid);'
      })
    );
    return nodes;
  }
}
