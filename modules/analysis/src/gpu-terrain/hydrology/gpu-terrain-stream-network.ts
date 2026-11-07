// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGraphOutputViews} from '../../analysis-core/kernels/gpu-contributor-utils';
import {assert} from '@luma.gl/core';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getTerrainFlowTracingWGSL} from './terrain-flow-tracing';
import {
  requireRasterLength,
  validateRasterExpansionGraph,
  validateRasterExpansionViews
} from '../../raster-analysis/raster-expansion-utils';

export type GPUTerrainStreamNetworkProps = {
  id?: string;
  grid: GPURasterMetadata;
  flowDirections: GraphDataView<'uint32'>;
  streams: GraphDataView<'uint32'>;
  maximumSteps?: number;
  output: {
    /** Compact reach endpoints; IDs retain original raster cell identity. */
    sourceCells: GraphDataView<'uint32'>;
    targetCells: GraphDataView<'uint32'>;
    /** Reach polyline offsets, capacity sourceCells.length + 1. */
    vertexOffsets: GraphDataView<'uint32'>;
    positions: GraphDataView<'float32x2'>;
    lengths: GraphDataView<'float32'>;
    reachCount: GraphDataView<'uint32'>;
    vertexCount: GraphDataView<'uint32'>;
    status: {overflow: GraphDataView<'uint32'>};
    /** Number of stream cells whose bounded downstream traversal is unresolved, including cycles. */
    unresolvedCount: GraphDataView<'uint32'>;
  };
};

/**
 * Extracts compact directed reaches between headwaters, confluences and terminals. Each reach
 * retains all cell centers as its polyline; junctions occur in each incident reach. A stream path
 * ends when it leaves the supplied mask. Cycles and unresolved directions never produce reaches.
 * Lengths use the planar affine units. One complete D8 grid is required.
 */
export class GPUTerrainStreamNetwork implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTerrainStreamNetworkProps;
  constructor(props: GPUTerrainStreamNetworkProps) {
    this.id = props.id ?? 'terrain-stream-network';
    this.props = props;
    const count = validateRasterExpansionViews(
      this.id,
      props.grid,
      [props.flowDirections, props.streams],
      getGraphOutputViews(props.output)
    );
    requireRasterLength(props.flowDirections, count);
    requireRasterLength(props.streams, count);
    requireRasterLength(props.output.targetCells, props.output.sourceCells.length);
    requireRasterLength(props.output.lengths, props.output.sourceCells.length);
    requireRasterLength(props.output.vertexOffsets, props.output.sourceCells.length + 1);
    for (const view of [
      props.output.reachCount,
      props.output.vertexCount,
      props.output.status.overflow,
      props.output.unresolvedCount
    ])
      requireRasterLength(view, 1);
    assert(Number.isInteger(props.maximumSteps ?? count) && (props.maximumSteps ?? count) > 0);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {grid, output} = props;
    const count = grid.width * grid.height;
    validateRasterExpansionGraph(id, graph, [
      props.flowDirections,
      props.streams,
      ...getGraphOutputViews(output)
    ]);
    const metadata = createTransientView(graph, `${id}-metadata`, 'uint32x4', count);
    const declarations = `${getTerrainFlowTracingWGSL(grid.width, grid.height)}
fn isStream(cell:u32)->bool {return streams[streamsOffset+cell]!=0u;}
fn incomingCount(cell:u32)->u32 {
 var incoming=0u; let column=i32(cell%GRID_WIDTH); let row=i32(cell/GRID_WIDTH);
 for(var rowOffset=-1;rowOffset<=1;rowOffset++){for(var columnOffset=-1;columnOffset<=1;columnOffset++){
 let otherColumn=column+columnOffset;let otherRow=row+rowOffset;
 if(otherColumn<0||otherRow<0||otherColumn>=i32(GRID_WIDTH)||otherRow>=i32(GRID_HEIGHT)||(columnOffset==0&&rowOffset==0)){continue;}
 let other=u32(otherRow)*GRID_WIDTH+u32(otherColumn);
 if(isStream(other)&&getFlowReceiver(other,directions[directionsOffset+other])==cell){incoming++;}
 }}return incoming;
}`;
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-trace`,
        operation: 'GPUTerrainStreamNetwork',
        invocationCount: count,
        bindings: [
          {name: 'directions', view: props.flowDirections, type: 'u32', access: 'read'},
          {name: 'streams', view: props.streams, type: 'u32', access: 'read'},
          {name: 'metadata', view: metadata, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `var vertices=0u;var targetCell=index;var unresolved=0u;
if(isStream(index)) {
 var current=index; var complete=false;
 for(var step=0u;step<${props.maximumSteps ?? count}u;step++){
  let code=directions[directionsOffset+current];
  if(streams[streamsOffset+current]==NO_CELL||streams[streamsOffset+current]==UNRESOLVED_DIRECTION){break;}
  let receiver=getFlowReceiver(current,code);
  if(code!=0u&&receiver==NO_CELL){break;}
  if(receiver==NO_CELL||!isStream(receiver)){complete=true;break;}
  current=receiver;
 }
 unresolved=u32(!complete);
 if(complete&&incomingCount(index)!=1u){
  current=index; vertices=1u;
  for(var step=0u;step<${props.maximumSteps ?? count}u;step++){
   let receiver=getFlowReceiver(current,directions[directionsOffset+current]);
   if(receiver==NO_CELL||!isStream(receiver)){break;}
   current=receiver;vertices++;
   if(incomingCount(current)!=1u){break;}
  }
  targetCell=current;
  if(vertices<2u){vertices=0u;}
 }
}
metadata[metadataOffset+index*4u]=vertices;metadata[metadataOffset+index*4u+1u]=targetCell;metadata[metadataOffset+index*4u+2u]=unresolved;metadata[metadataOffset+index*4u+3u]=0u;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-compact`,
        operation: 'GPUTerrainStreamNetwork',
        invocationCount: 1,
        bindings: [
          {name: 'metadata', view: metadata, type: 'u32', access: 'read'},
          {name: 'sources', view: output.sourceCells, type: 'u32', access: 'read_write'},
          {name: 'offsets', view: output.vertexOffsets, type: 'u32', access: 'read_write'},
          {name: 'reachCount', view: output.reachCount, type: 'u32', access: 'read_write'},
          {name: 'vertexCount', view: output.vertexCount, type: 'u32', access: 'read_write'},
          {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'},
          {name: 'unresolved', view: output.unresolvedCount, type: 'u32', access: 'read_write'}
        ],
        body: `var reaches=0u;var vertices=0u;var truncated=0u;var unresolvedCells=0u;offsets[offsetsOffset]=0u;
for(var cell=0u;cell<${count}u;cell++){
 let entry=vec4<u32>(metadata[metadataOffset+cell*4u],metadata[metadataOffset+cell*4u+1u],metadata[metadataOffset+cell*4u+2u],0u);unresolvedCells+=entry.z;
 if(entry.x==0u){continue;}
 if(reaches>=${output.sourceCells.length}u||vertices+entry.x>${output.positions.length}u){truncated=1u;continue;}
 sources[sourcesOffset+reaches]=cell;vertices+=entry.x;reaches++;offsets[offsetsOffset+reaches]=vertices;
}
reachCount[reachCountOffset]=reaches;vertexCount[vertexCountOffset]=vertices;overflow[overflowOffset]=truncated;unresolved[unresolvedOffset]=unresolvedCells;`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-geometry`,
        operation: 'GPUTerrainStreamNetwork',
        invocationCount: output.sourceCells.length,
        bindings: [
          {name: 'directions', view: props.flowDirections, type: 'u32', access: 'read'},
          {name: 'sources', view: output.sourceCells, type: 'u32', access: 'read'},
          {name: 'offsets', view: output.vertexOffsets, type: 'u32', access: 'read'},
          {name: 'reachCount', view: output.reachCount, type: 'u32', access: 'read'},
          {name: 'positions', view: output.positions, type: 'f32', access: 'read_write'},
          {name: 'targets', view: output.targetCells, type: 'u32', access: 'read_write'},
          {name: 'lengths', view: output.lengths, type: 'f32', access: 'read_write'}
        ],
        declarations: `${getTerrainFlowTracingWGSL(grid.width, grid.height)}
fn position(cell:u32)->vec2<f32>{let column=f32(cell%GRID_WIDTH)+${grid.pixelInterpretation === 'area' ? '0.5' : '0.0'};let row=f32(cell/GRID_WIDTH)+${grid.pixelInterpretation === 'area' ? '0.5' : '0.0'};return vec2<f32>(${grid.affine[0]}*column+${grid.affine[1]}*row+${grid.affine[2]},${grid.affine[3]}*column+${grid.affine[4]}*row+${grid.affine[5]});}`,
        body: `if(index>=reachCount[reachCountOffset]){return;}var current=sources[sourcesOffset+index];var total=0.0;var previousCell=current;
for(var vertex=offsets[offsetsOffset+index];vertex<offsets[offsetsOffset+index+1u];vertex++){
 let point=position(current);positions[positionsOffset+vertex*2u]=point.x;positions[positionsOffset+vertex*2u+1u]=point.y;// Measure the affine displacement before adding the origin, which can round away small edges.
 let delta=vec2<f32>(f32(current%GRID_WIDTH)-f32(previousCell%GRID_WIDTH), f32(current/GRID_WIDTH)-f32(previousCell/GRID_WIDTH));
 total+=length(vec2<f32>(${grid.affine[0]}*delta.x+${grid.affine[1]}*delta.y,${grid.affine[3]}*delta.x+${grid.affine[4]}*delta.y));previousCell=current;
 if(vertex+1u<offsets[offsetsOffset+index+1u]){current=getFlowReceiver(current,directions[directionsOffset+current]);}
}targets[targetsOffset+index]=current;lengths[lengthsOffset+index]=total;`
      })
    ];
  }
}
