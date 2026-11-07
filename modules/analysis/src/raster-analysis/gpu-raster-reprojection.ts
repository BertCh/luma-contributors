// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {CompiledProjection} from '@luma.gl/graph/gpu-project';
import {GPUProjectionProgram} from '@luma.gl/graph/gpu-project';
import type {GPURasterBufferBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  getRasterBufferBandNodes,
  getRasterBandViews,
  validateRasterGrid
} from './raster-band-utils';
import {createWGSLKernelNode} from '../analysis-core/kernels/wgsl-kernel-nodes';
import {captureGraphCommandNodes} from '../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_VALUE_WGSL} from '../analysis-core/kernels/wgsl-common';
import {
  requireRasterLength,
  validateRasterExpansionViews,
  validateRasterExpansionGraph
} from './raster-expansion-utils';

export type GPURasterReprojectionProps = {
  id?: string;
  sourceGrid: GPURasterMetadata;
  destinationGrid: GPURasterMetadata;
  input: GPURasterBufferBand;
  /** Explicit destination-world to source-world program. Omit only for grids in the same CRS. */
  inverseProjection?: CompiledProjection;
  method?: 'nearest' | 'bilinear';
  noDataPolicy?: 'strict' | 'renormalize';
  output: {values: GraphDataView<'float32'>; validity: GraphDataView<'uint32'>};
};

/**
 * Destination-driven grid alignment/reprojection. Generates destination centers, projects them,
 * inverts the source affine, and samples one explicitly supplied complete source tile. Invalid
 * projection domains and absent source support remain nodata. Rotated/sheared affines are supported.
 * The optional projection uses float32 coordinates; use local coordinates for large projected
 * origins. The source tile must contain the entire requested interpolation support. This adapter
 * does not select/fetch source tiles. Call destroy after the compiled graph to release projection
 * parameters when an inverseProjection is provided.
 */
export class GPURasterReprojection implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterReprojectionProps;
  private projection?: GPUProjectionProgram;
  constructor(props: GPURasterReprojectionProps) {
    this.id = props.id ?? 'raster-reprojection';
    this.props = props;
    const count = validateRasterExpansionViews(
      this.id,
      props.destinationGrid,
      getRasterBandViews(props.input),
      Object.values(props.output)
    );
    validateRasterGrid(this.id, props.sourceGrid);
    requireRasterLength(props.output.values, count);
    requireRasterLength(props.output.validity, count);
    const affine = props.sourceGrid.affine;
    assert(affine[0] * affine[4] - affine[1] * affine[3] !== 0);
    assert(
      !props.inverseProjection ||
        (props.inverseProjection.inputFormat === 'float32x2' &&
          props.inverseProjection.precision !== 'double-single')
    );
    if (!props.inverseProjection) {
      assert(
        JSON.stringify(props.sourceGrid.coordinateReferenceSystem ?? null) ===
          JSON.stringify(props.destinationGrid.coordinateReferenceSystem ?? null)
      );
    }
  }
  destroy(): void {
    this.projection?.destroy();
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    // Projection parameters belong to one graph; rebuilding requires a new contributor.
    assert(!this.projection);
    const {id, props} = this;
    const source = props.sourceGrid,
      destination = props.destinationGrid;
    const count = destination.width * destination.height;
    validateRasterExpansionGraph(id, graph, [
      ...getRasterBandViews(props.input),
      ...Object.values(props.output)
    ]);
    const band = getRasterBufferBandNodes(graph, id, props.input, source, true);
    const destinationPositions = createTransientView(
      graph,
      `${id}-destination`,
      'float32x2',
      count
    );
    const projectedPositions = props.inverseProjection
      ? createTransientView(graph, `${id}-projected`, 'float32x2', count)
      : destinationPositions;
    const projectedValidity = props.inverseProjection
      ? createTransientView(graph, `${id}-projection-validity`, 'uint32', count)
      : undefined;
    const nodes: GPUCommandNode<Parameters>[] = [
      ...band.nodes,
      createWGSLKernelNode(graph, {
        id: `${id}-grid`,
        operation: 'GPURasterReprojection',
        invocationCount: count,
        bindings: [
          {name: 'positions', view: destinationPositions, type: 'f32', access: 'read_write'}
        ],
        body: `let column=f32(index%${destination.width}u)+${destination.pixelInterpretation === 'area' ? '0.5' : '0.0'};let row=f32(index/${destination.width}u)+${destination.pixelInterpretation === 'area' ? '0.5' : '0.0'};positions[positionsOffset+index*2u]=${destination.affine[0]}*column+${destination.affine[1]}*row+${destination.affine[2]};positions[positionsOffset+index*2u+1u]=${destination.affine[3]}*column+${destination.affine[4]}*row+${destination.affine[5]};`
      })
    ];
    if (props.inverseProjection) {
      this.projection = new GPUProjectionProgram({
        id: `${id}-project`,
        projection: props.inverseProjection,
        positions: destinationPositions,
        output: projectedPositions,
        validity: projectedValidity
      });
      nodes.push(...captureGraphCommandNodes(graph, () => this.projection!.addToGraph(graph)));
    }
    const determinant = source.affine[0] * source.affine[4] - source.affine[1] * source.affine[3];
    const origin = props.inverseProjection?.destinationOrigin ?? [0, 0];
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-sample`,
        operation: 'GPURasterReprojection',
        invocationCount: count,
        bindings: [
          {name: 'positions', view: projectedPositions, type: 'f32', access: 'read'},
          ...(projectedValidity
            ? [
                {
                  name: 'projectionValidity',
                  view: projectedValidity,
                  type: 'u32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'source', view: band.band.storage.values, type: 'f32', access: 'read'},
          {name: 'sourceValidity', view: band.band.validity!, type: 'u32', access: 'read'},
          {name: 'values', view: props.output.values, type: 'f32', access: 'read_write'},
          {name: 'validity', view: props.output.validity, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_VALUE_WGSL,
        body: `let x=positions[positionsOffset+index*2u]+${origin[0]}-${source.affine[2]};let y=positions[positionsOffset+index*2u+1u]+${origin[1]}-${source.affine[5]};let column=(${source.affine[4]}*x-${source.affine[1]}*y)/${determinant}-${source.pixelInterpretation === 'area' ? '0.5' : '0.0'};let row=(-${source.affine[3]}*x+${source.affine[0]}*y)/${determinant}-${source.pixelInterpretation === 'area' ? '0.5' : '0.0'};
var valid=isFiniteValue(column)&&isFiniteValue(row)${projectedValidity ? '&&projectionValidity[projectionValidityOffset+index]!=0u' : ''};var total=0.0;var weightSum=0.0;
${props.method === 'nearest' ? `let sampleColumn=i32(floor(column+0.5));let sampleRow=i32(floor(row+0.5));if(sampleColumn<0||sampleRow<0||sampleColumn>=${source.width}||sampleRow>=${source.height}){valid=false;}else{let sampleIndex=u32(sampleRow)*${source.width}u+u32(sampleColumn);valid=valid&&sourceValidity[sourceValidityOffset+sampleIndex]!=0u;total=source[sourceOffset+sampleIndex];weightSum=1.0;}` : `let baseColumn=i32(floor(column));let baseRow=i32(floor(row));let fraction=vec2<f32>(fract(column),fract(row));for(var rowOffset=0;rowOffset<2;rowOffset++){for(var columnOffset=0;columnOffset<2;columnOffset++){let weight=select(1.0-fraction.x,fraction.x,columnOffset==1)*select(1.0-fraction.y,fraction.y,rowOffset==1);if(weight==0.0){continue;}let sampleColumn=baseColumn+columnOffset;let sampleRow=baseRow+rowOffset;var sampleValid=sampleColumn>=0&&sampleRow>=0&&sampleColumn<${source.width}&&sampleRow<${source.height};if(sampleValid){let sampleIndex=u32(sampleRow)*${source.width}u+u32(sampleColumn);sampleValid=sourceValidity[sourceValidityOffset+sampleIndex]!=0u;if(sampleValid){total+=weight*source[sourceOffset+sampleIndex];weightSum+=weight;}}${props.noDataPolicy === 'renormalize' ? '' : 'valid=valid&&sampleValid;'}}}`}
valid=valid&&weightSum>0.0;let result=total/max(weightSum,1.0e-30);valid=valid&&isFiniteValue(result);values[valuesOffset+index]=select(bitcast<f32>(0x7fc00000u | (index & 0u)),result,valid);validity[validityOffset+index]=u32(valid);`
      })
    );
    return nodes;
  }
}
