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
import {
  requireRasterLength,
  validateRasterExpansionViews,
  validateRasterExpansionGraph
} from './raster-expansion-utils';
import {GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION} from './distance-field/distance-field-parameters';
import {getRasterMedialAxisNodes} from './raster-medial-axis';

export type GPURasterSkeletonProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: GraphDataView<'uint32'>;
  maximumIterations?: number;
  output: {skeleton: GraphDataView<'uint32'>; status: {converged: GraphDataView<'uint32'>}};
};
/** Zhang–Suen two-subiteration thinning. Foreground is nonzero; the exterior is background. */
export class GPURasterSkeleton implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterSkeletonProps;
  constructor(props: GPURasterSkeletonProps) {
    this.id = props.id ?? 'raster-skeleton';
    this.props = props;
    const count = validateRasterExpansionViews(
      this.id,
      props.grid,
      [props.input],
      getGraphOutputViews(props.output)
    );
    requireRasterLength(props.input, count);
    requireRasterLength(props.output.skeleton, count);
    requireRasterLength(props.output.status.converged, 1);
    assert(Number.isInteger(props.maximumIterations ?? 64) && (props.maximumIterations ?? 64) > 0);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {grid, output} = props;
    const count = grid.width * grid.height;
    validateRasterExpansionGraph(id, graph, [props.input, ...getGraphOutputViews(output)]);
    const first = createTransientView(graph, `${id}-first`, 'uint32', count),
      second = createTransientView(graph, `${id}-second`, 'uint32', count),
      changed = createTransientView(graph, `${id}-changed`, 'uint32', count);
    const nodes: GPUCommandNode<Parameters>[] = [
      createWGSLKernelNode(graph, {
        id: `${id}-initialize`,
        operation: 'GPURasterSkeleton',
        invocationCount: count,
        bindings: [
          {name: 'input', view: props.input, type: 'u32', access: 'read'},
          {name: 'first', view: first, type: 'u32', access: 'read_write'}
        ],
        body: 'first[firstOffset+index]=u32(input[inputOffset+index]!=0u);'
      })
    ];
    for (let iteration = 0; iteration <= (props.maximumIterations ?? 64); iteration++) {
      for (let phase = 0; phase < 2; phase++) {
        nodes.push(
          createWGSLKernelNode(graph, {
            id: `${id}-${iteration}-${phase}`,
            operation: 'GPURasterSkeleton',
            invocationCount: count,
            bindings: [
              {name: 'previous', view: phase === 0 ? first : second, type: 'u32', access: 'read'},
              {name: 'next', view: phase === 0 ? second : first, type: 'u32', access: 'read_write'},
              {name: 'changed', view: changed, type: 'u32', access: 'read_write'}
            ],
            body: `let original=previous[previousOffset+index];var remove=false;if(original!=0u){var neighborhood:array<u32,8>;let columns=array<i32,8>(0,1,1,1,0,-1,-1,-1);let rows=array<i32,8>(-1,-1,0,1,1,1,0,-1);var neighbors=0u;var transitions=0u;
for(var side=0u;side<8u;side++){let column=i32(index%${grid.width}u)+columns[side];let row=i32(index/${grid.width}u)+rows[side];var value=0u;if(column>=0&&row>=0&&column<${grid.width}&&row<${grid.height}){value=previous[previousOffset+u32(row)*${grid.width}u+u32(column)];}neighborhood[side]=value;neighbors+=value;}
for(var side=0u;side<8u;side++){transitions+=u32(neighborhood[side]==0u&&neighborhood[(side+1u)%8u]!=0u);}
remove=neighbors>=2u&&neighbors<=6u&&transitions==1u&&${phase === 0 ? 'neighborhood[0]*neighborhood[2]*neighborhood[4]==0u&&neighborhood[2]*neighborhood[4]*neighborhood[6]==0u' : 'neighborhood[0]*neighborhood[2]*neighborhood[6]==0u&&neighborhood[0]*neighborhood[4]*neighborhood[6]==0u'};}
next[nextOffset+index]=select(original,0u,remove);changed[changedOffset+index]=${phase === 0 ? 'u32(remove)' : 'changed[changedOffset+index]|u32(remove)'};`
          })
        );
      }
    }
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-status`,
        operation: 'GPURasterSkeleton',
        invocationCount: 1,
        bindings: [
          {name: 'changed', view: changed, type: 'u32', access: 'read'},
          {name: 'converged', view: output.status.converged, type: 'u32', access: 'read_write'}
        ],
        body: `var complete=1u;for(var cell=0u;cell<${count}u;cell++){complete&=1u-changed[changedOffset+cell];}converged[convergedOffset]=complete;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation: 'GPURasterSkeleton',
        invocationCount: count,
        bindings: [
          {name: 'first', view: first, type: 'u32', access: 'read'},
          {name: 'converged', view: output.status.converged, type: 'u32', access: 'read'},
          {name: 'skeleton', view: output.skeleton, type: 'u32', access: 'read_write'}
        ],
        body: 'skeleton[skeletonOffset+index]=select(0xfffffffeu,first[firstOffset+index],converged[convergedOffset]!=0u);'
      })
    );
    return nodes;
  }
}

export type GPURasterMedialAxisProps = {
  id?: string;
  grid: GPURasterMetadata;
  input: GraphDataView<'uint32'>;
  output: {medialAxis: GraphDataView<'uint32'>; radius: GraphDataView<'float32'>};
};
/**
 * Discrete maximal-inscribed-disk centers and their Euclidean radii in pixel units. All exterior
 * pixels are background. A disk center is removed only when its disk is contained in another
 * foreground disk; even-width plateaus can retain several centers. This is a maximal-disk medial
 * axis, distinct from topology-preserving one-pixel thinning. An exact banded distance transform
 * finds the nearest background center, then an integer primitive-lattice-step test classifies
 * each disk independently. Storage is linear in the number of cells; each dimension is at most
 * 32768 and the distance-transform scratch must fit the device's storage binding limits.
 *
 * Containment uses exact integer squared distances, correcting the previous all-pairs float32
 * comparison at tangent disks. Radii are correctly rounded float32 values in pixel units;
 * the grid affine does not rescale them. No thinning or approximate distance transform is used.
 */
export class GPURasterMedialAxis implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterMedialAxisProps;
  constructor(props: GPURasterMedialAxisProps) {
    this.id = props.id ?? 'raster-medial-axis';
    this.props = props;
    const count = validateRasterExpansionViews(
      this.id,
      props.grid,
      [props.input],
      getGraphOutputViews(props.output)
    );
    // Squared pixel offsets, including the virtual exterior, must fit uint32 exactly.
    assert(
      props.grid.width <= GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION &&
        props.grid.height <= GPU_DISTANCE_FIELD_MAXIMUM_DIMENSION
    );
    for (const view of [props.input, ...getGraphOutputViews(props.output)])
      requireRasterLength(view, count);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateRasterExpansionGraph(id, graph, [props.input, ...getGraphOutputViews(props.output)]);
    return getRasterMedialAxisNodes(graph, id, props);
  }
}
