// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  validateTerrainBindings,
  getTerrainGateScratchByteLength
} from '../terrain-processing/terrain-binding-preflight';
import {
  getGraphOutputViews,
  validateCapacity,
  validateCommandNodeBudget,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createTransientView,
  GPUCompaction,
  GPUReduction,
  GPUScan,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  validateRasterCellSizeMode,
  type GPURasterCellSizeMode
} from '../../analysis-core/contracts/raster-grid';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateRasterGridAxisAligned} from '../../raster-analysis/raster-band-utils';
import {
  getTerrainFlowTracingWGSL,
  validateTerrainFlowDirectionEncoding,
  type GPUTerrainFlowDirectionEncoding
} from './terrain-flow-tracing';
import {
  createRasterJacobiLoopNodes,
  getRasterJacobiLoopNodeCount,
  requireRasterLength,
  validateRasterExpansionGraph,
  validateRasterExpansionViews,
  validateRasterExpansionPorts
} from '../../raster-analysis/raster-expansion-utils';
import {
  getTerrainGroundMeasureWGSL,
  getTerrainJumpRoundCount
} from '../terrain-processing/terrain-processing-utils';

export type GPUTerrainStreamNetworkProps = {
  id?: string;
  grid: GPURasterMetadata;
  /**
   * D8 codes, one per cell, ESRI by default (1 east, clockwise powers of two; 0 terminal;
   * `0xffffffff` invalid; `0xfffffffe` unresolved). A nonzero code without a receiver is
   * unresolved.
   */
  flowDirections: GraphDataView<'uint32'>;
  /**
   * Numbering of `flowDirections`: `'esri'` (default), `'whitebox'` (the WhiteboxTools default
   * pointer) or `'taudem'`, see {@link GPUTerrainFlowDirectionEncoding}. Compile-time.
   */
  directionEncoding?: GPUTerrainFlowDirectionEncoding;
  /**
   * Stream mask: a nonzero word is a stream cell. The words `0xffffffff` and `0xfffffffe` mark a
   * stream cell of unknown status, which is counted in `output.unresolvedCount`.
   */
  streams: GraphDataView<'uint32'>;
  /**
   * Float64 point that `output.positions` are written relative to; add it back on the CPU or as
   * the layer's coordinate origin. Defaults to `[0, 0]` (absolute world coordinates). Pass the
   * grid origin `[grid.affine[2], grid.affine[5]]` (or a tile corner) for projected grids:
   * absolute float32 positions are quantized to 0.5 m at a UTM northing of 4.5e6 m, so a vertex
   * is up to 0.25 cell off for 1 m cells, while origin-relative ones keep about `1.2e-7` of their
   * distance from the origin (1.2 cm at 100 km). The translation `affine - localOrigin` is formed
   * in float64 before it reaches the shader. Compile-time. The resolved value is the
   * `localOrigin` field of the contributor.
   */
  localOrigin?: readonly [number, number];
  /** Ground model for `output.lengths`. Default `'uniform'` (affine units). `'web-mercator'` and `'geographic'` need an axis-aligned grid. Compile-time. */
  cellSizeMode?: GPURasterCellSizeMode;
  /**
   * Bounds each stream path, counted in cells. Default cell count; longer paths and cycles are
   * unresolved. Compile-time: the pointer-jumping round count is
   * `ceil(log2(min(maximumSteps, cell count)))`.
   */
  maximumSteps?: number;
  output: {
    /** Compact reach endpoints; IDs retain original raster cell identity. */
    sourceCells: GraphDataView<'uint32'>;
    targetCells: GraphDataView<'uint32'>;
    /** Reach polyline offsets, capacity sourceCells.length + 1. */
    vertexOffsets: GraphDataView<'uint32'>;
    /**
     * Cell centers in the grid's world coordinates (the affine's own units) minus `localOrigin`.
     * With the default origin `[0, 0]` they are absolute float32 coordinates, exact only to the
     * float32 spacing at their magnitude (0.5 m at 4.5e6, 7.6e-6 degree at longitude 122).
     */
    positions: GraphDataView<'float32x2'>;
    /** Reach lengths in affine units by default or ground metres in geographic and Web Mercator modes. */
    lengths: GraphDataView<'float32'>;
    reachCount: GraphDataView<'uint32'>;
    vertexCount: GraphDataView<'uint32'>;
    status: {overflow: GraphDataView<'uint32'>};
    /** Number of stream cells whose bounded downstream traversal or junction membership is unresolved, including cycles and upstream paths to uncertain junctions. */
    unresolvedCount: GraphDataView<'uint32'>;
  };
};

/**
 * Extracts compact directed reaches between headwaters, confluences and terminals. Each reach
 * retains all cell centers as its polyline; junctions occur in each incident reach. A stream path
 * ends when it leaves the supplied mask. Cycles and unresolved directions never produce reaches.
 * Unknown donor membership makes a junction uncertain: that cell and paths reaching it are
 * unresolved and produce no reaches. Separate definite reaches remain available.
 * Lengths use affine units by default or ground metres in geographic and Web Mercator modes.
 * One complete D8 grid is required.
 *
 * Reach ends and path completeness are found by pointer jumping (`ceil(log2(path bound))` gated
 * rounds), and reaches are compacted with `GPUCompaction` and `GPUScan`. When the outputs are too
 * small the reaches that fit in order are kept (a prefix) and `status.overflow` is set; the old
 * serial compaction could also keep a later short reach after a skipped long one.
 */
export class GPUTerrainStreamNetwork implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUTerrainStreamNetworkProps;
  /** Number of command nodes `getCommandNodes` returns; bounded by the command node budget. */
  readonly estimatedCommandNodeCount: number;
  /** Float64 point to add to `output.positions` for world coordinates; `[0, 0]` by default. */
  readonly localOrigin: readonly [number, number];
  constructor(props: GPUTerrainStreamNetworkProps) {
    this.id = props.id ?? 'terrain-stream-network';
    this.props = props;
    this.localOrigin = props.localOrigin ?? [0, 0];
    validateRequirement(
      this.id,
      this.localOrigin.length === 2 && this.localOrigin.every(Number.isFinite),
      'localOrigin must be two finite numbers'
    );
    validateTerrainFlowDirectionEncoding(this.id, props.directionEncoding);
    validateRasterExpansionPorts(this.id, props, {
      flowDirections: ['uint32'],
      streams: ['uint32'],
      'output.sourceCells': ['uint32'],
      'output.targetCells': ['uint32'],
      'output.vertexOffsets': ['uint32'],
      'output.positions': ['float32x2'],
      'output.lengths': ['float32'],
      'output.reachCount': ['uint32'],
      'output.vertexCount': ['uint32'],
      'output.status.overflow': ['uint32'],
      'output.unresolvedCount': ['uint32']
    });
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
    validateRasterCellSizeMode(this.id, props.cellSizeMode);
    validateRasterGridAxisAligned(this.id, props.grid, props.cellSizeMode);
    validateCapacity(this.id, 'maximumSteps', props.maximumSteps ?? count);
    // Seed, flag, gather, publish and geometry nodes, the compaction and scan nodes (at most 12), two
    // reductions' nodes (at most 8) and the gated pointer-jumping loop. The sum is an upper bound.
    this.estimatedCommandNodeCount =
      40 +
      getRasterJacobiLoopNodeCount(
        getTerrainJumpRoundCount(Math.min(props.maximumSteps ?? count, count))
      );
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateTerrainBindings(graph, this.id, this.props, {
      'pointer state': 20 * this.props.grid.width * this.props.grid.height,
      'gate table': getTerrainGateScratchByteLength(
        getTerrainJumpRoundCount(
          Math.min(
            this.props.maximumSteps ?? this.props.grid.width * this.props.grid.height,
            this.props.grid.width * this.props.grid.height
          )
        )
      )
    });
    const {id, props} = this;
    const {grid, output} = props;
    const count = grid.width * grid.height;
    validateRasterExpansionGraph(id, graph, [
      props.flowDirections,
      props.streams,
      ...getGraphOutputViews(output)
    ]);
    const operation = 'GPUTerrainStreamNetwork';
    const maximumSteps = props.maximumSteps ?? count;
    const rounds = getTerrainJumpRoundCount(Math.min(maximumSteps, count));
    // Per cell: next stream cell on the path (NO_CELL when done), cells on the path so far, path
    // outcome (0 open, 1 leaves the mask, 2 unresolved), next cell inside the reach and its hops.
    const states = [0, 1].map(side =>
      createTransientView(graph, `${id}-state-${side}`, 'uint32', count * 5)
    );
    const converged = createTransientView(graph, `${id}-converged`, 'uint32', 1);
    const vertexCounts = createTransientView(graph, `${id}-vertex-counts`, 'uint32', count);
    const unresolvedFlags = createTransientView(graph, `${id}-unresolved-flags`, 'uint32', count);
    const reachFlags = createTransientView(graph, `${id}-reach-flags`, 'uint32', count);
    const cellIds = createTransientView(graph, `${id}-cell-ids`, 'uint32', count);
    const compactCells = createTransientView(graph, `${id}-compact-cells`, 'uint32', count);
    const reachTotal = createTransientView(graph, `${id}-reach-total`, 'uint32', 1);
    const rowVertices = createTransientView(graph, `${id}-row-vertices`, 'uint32', count);
    const rowEnds = createTransientView(graph, `${id}-row-ends`, 'uint32', count);
    const tracing = getTerrainFlowTracingWGSL(grid.width, grid.height, props.directionEncoding);
    const declarations = `${tracing}
fn isStream(cell:u32)->bool {return streams[streamsOffset+cell]!=0u;}
fn incomingCount(cell:u32)->u32 {
 var incoming=0u; let column=i32(cell%GRID_WIDTH); let row=i32(cell/GRID_WIDTH);
 for(var rowOffset=-1;rowOffset<=1;rowOffset++){for(var columnOffset=-1;columnOffset<=1;columnOffset++){
 let otherColumn=column+columnOffset;let otherRow=row+rowOffset;
 if(otherColumn<0||otherRow<0||otherColumn>=i32(GRID_WIDTH)||otherRow>=i32(GRID_HEIGHT)||(columnOffset==0&&rowOffset==0)){continue;}
 let other=u32(otherRow)*GRID_WIDTH+u32(otherColumn);
 if(isStream(other)){
  let word=streams[streamsOffset+other];
  let code=getFlowDirectionCode(directions[directionsOffset+other]);
  let receiver=getFlowReceiver(other,code);
  // An unknown direction may donate to any neighbour. An unknown stream word with a known
  // receiver may or may not contribute that donor, so it cannot certify a junction count.
  if(code==UNRESOLVED_DIRECTION||(word>=UNRESOLVED_DIRECTION&&receiver==cell)){return UNRESOLVED_DIRECTION;}
  if(receiver==cell){incoming++;}
 }
 }}return incoming;
}`;
    const streamBindings = [
      {
        name: 'directions',
        view: props.flowDirections,
        type: 'u32' as const,
        access: 'read' as const
      },
      {name: 'streams', view: props.streams, type: 'u32' as const, access: 'read' as const}
    ];
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-seed`,
        operation,
        invocationCount: count,
        bindings: [
          ...streamBindings,
          {name: 'state', view: states[0], type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `var nextCell=NO_CELL;var steps=1u;var outcome=0u;var reachNext=NO_CELL;var reachHops=0u;
if(isStream(index)){
 let stream=streams[streamsOffset+index];
 let code=getFlowDirectionCode(directions[directionsOffset+index]);
 let receiver=getFlowReceiver(index,code);
 let leaves=receiver==NO_CELL||!isStream(receiver);
 if(stream==NO_CELL||stream==UNRESOLVED_DIRECTION||(code!=0u&&receiver==NO_CELL)||incomingCount(index)==UNRESOLVED_DIRECTION){outcome=2u;}
 else if(leaves){outcome=1u;}
 else{nextCell=receiver;}
 // The reach continues through the receiver only while it is a stream cell with exactly one donor.
 if(!leaves){reachHops=1u;if(incomingCount(receiver)==1u){reachNext=receiver;}}
}
let offset=stateOffset+index*5u;
state[offset]=nextCell;state[offset+1u]=steps;state[offset+2u]=outcome;state[offset+3u]=reachNext;state[offset+4u]=reachHops;`
      }),
      ...createRasterJacobiLoopNodes(graph, {
        id: `${id}-jump`,
        operation,
        invocationCount: count,
        roundCount: rounds,
        status: {converged},
        createRound: (
          {nodeId, iteration, slotBinding, markChangedWGSL, condition, extraResources},
          status
        ) =>
          createWGSLKernelNode(graph, {
            id: nodeId,
            operation,
            variant: 'jump',
            invocationCount: count,
            bindings: [
              {name: 'previous', view: states[iteration % 2], type: 'u32', access: 'read'},
              {name: 'next', view: states[(iteration + 1) % 2], type: 'u32', access: 'read_write'},
              {name: 'status', view: status, type: 'atomic<u32>', access: 'read_write'},
              slotBinding
            ],
            declarations: 'const NO_CELL: u32 = 0xffffffffu;',
            condition,
            extraResources,
            body: `let offset=previousOffset+index*5u;let destination=nextOffset+index*5u;
for(var component=0u;component<5u;component++){next[destination+component]=previous[offset+component];}
let pathNext=previous[offset];
if(pathNext!=NO_CELL){
 let ancestor=previousOffset+pathNext*5u;
 next[destination]=previous[ancestor];
 next[destination+1u]=min(${count + 1}u,previous[offset+1u]+previous[ancestor+1u]);
 next[destination+2u]=previous[ancestor+2u];
 ${markChangedWGSL}
}
let reachNext=previous[offset+3u];
if(reachNext!=NO_CELL){
 let ancestor=previousOffset+reachNext*5u;
 next[destination+3u]=previous[ancestor+3u];
 next[destination+4u]=min(${count + 1}u,previous[offset+4u]+previous[ancestor+4u]);
 ${markChangedWGSL}
}`
          })
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-reaches`,
        operation,
        invocationCount: count,
        bindings: [
          ...streamBindings,
          {name: 'state', view: states[rounds % 2], type: 'u32', access: 'read'},
          {name: 'vertexCounts', view: vertexCounts, type: 'u32', access: 'read_write'},
          {name: 'unresolvedFlags', view: unresolvedFlags, type: 'u32', access: 'read_write'},
          {name: 'reachFlags', view: reachFlags, type: 'u32', access: 'read_write'},
          {name: 'cellIds', view: cellIds, type: 'u32', access: 'read_write'}
        ],
        declarations,
        body: `var vertices=0u;var unresolved=0u;
if(isStream(index)){
 let offset=stateOffset+index*5u;
 let complete=state[offset]==NO_CELL&&state[offset+2u]==1u&&state[offset+1u]<=${maximumSteps}u;
 unresolved=u32(!complete);
 if(complete&&incomingCount(index)!=1u){
  vertices=1u+state[offset+4u];
  if(vertices<2u){vertices=0u;}
 }
}
vertexCounts[vertexCountsOffset+index]=vertices;unresolvedFlags[unresolvedFlagsOffset+index]=unresolved;
reachFlags[reachFlagsOffset+index]=u32(vertices>0u);cellIds[cellIdsOffset+index]=index;`
      }),
      ...new GPUReduction({
        id: `${id}-unresolved-sum`,
        input: unresolvedFlags,
        output: output.unresolvedCount,
        operation: 'sum'
      }).getCommandNodes(graph),
      ...new GPUCompaction({
        id: `${id}-compaction`,
        input: cellIds,
        flags: reachFlags,
        output: compactCells,
        count: reachTotal
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-gather`,
        operation,
        invocationCount: count,
        bindings: [
          {name: 'compactCells', view: compactCells, type: 'u32', access: 'read'},
          {name: 'reachTotal', view: reachTotal, type: 'u32', access: 'read'},
          {name: 'vertexCounts', view: vertexCounts, type: 'u32', access: 'read'},
          {name: 'rowVertices', view: rowVertices, type: 'u32', access: 'read_write'}
        ],
        body: `rowVertices[rowVerticesOffset+index]=select(0u,vertexCounts[vertexCountsOffset+compactCells[compactCellsOffset+index]],index<reachTotal[reachTotalOffset]);`
      }),
      ...new GPUScan({
        id: `${id}-vertex-scan`,
        input: rowVertices,
        output: rowEnds,
        mode: 'inclusive'
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-publish-clear`,
        operation,
        invocationCount: 1,
        bindings: [
          {name: 'reachTotal', view: reachTotal, type: 'u32', access: 'read'},
          {name: 'offsets', view: output.vertexOffsets, type: 'u32', access: 'read_write'},
          {name: 'reachCount', view: output.reachCount, type: 'u32', access: 'read_write'},
          {name: 'vertexCount', view: output.vertexCount, type: 'u32', access: 'read_write'},
          {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'}
        ],
        // Without a kept reach, any reach found overflowed; the last kept reach rewrites these words.
        body: `offsets[offsetsOffset]=0u;reachCount[reachCountOffset]=0u;vertexCount[vertexCountOffset]=0u;overflow[overflowOffset]=u32(reachTotal[reachTotalOffset]>0u);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-publish`,
        operation,
        invocationCount: count,
        bindings: [
          {name: 'compactCells', view: compactCells, type: 'u32', access: 'read'},
          {name: 'reachTotal', view: reachTotal, type: 'u32', access: 'read'},
          {name: 'rowEnds', view: rowEnds, type: 'u32', access: 'read'},
          {name: 'sources', view: output.sourceCells, type: 'u32', access: 'read_write'},
          {name: 'offsets', view: output.vertexOffsets, type: 'u32', access: 'read_write'},
          {name: 'reachCount', view: output.reachCount, type: 'u32', access: 'read_write'},
          {name: 'vertexCount', view: output.vertexCount, type: 'u32', access: 'read_write'},
          {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'}
        ],
        // Reaches are kept in order while they fit both capacities, so the kept rows are a prefix
        // and exactly one row, the last kept, publishes the counts.
        body: `let total=reachTotal[reachTotalOffset];
if(index>=total){return;}
let end=rowEnds[rowEndsOffset+index];
if(index>=${output.sourceCells.length}u||end>${output.positions.length}u){return;}
sources[sourcesOffset+index]=compactCells[compactCellsOffset+index];offsets[offsetsOffset+index+1u]=end;
var nextKept=false;
if(index+1u<total){nextKept=index+1u<${output.sourceCells.length}u&&rowEnds[rowEndsOffset+index+1u]<=${output.positions.length}u;}
if(!nextKept){reachCount[reachCountOffset]=index+1u;vertexCount[vertexCountOffset]=end;overflow[overflowOffset]=u32(index+1u<total);}`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-geometry`,
        operation,
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
        // The translation is the affine's minus localOrigin, formed in float64 here: with the grid
        // origin as localOrigin it is zero and the float32 products keep the cell centre exactly.
        declarations: `${tracing}
${getTerrainGroundMeasureWGSL(grid, props.cellSizeMode)}
fn position(cell:u32)->vec2<f32>{let column=f32(cell%GRID_WIDTH)+${grid.pixelInterpretation === 'area' ? '0.5' : '0.0'};let row=f32(cell/GRID_WIDTH)+${grid.pixelInterpretation === 'area' ? '0.5' : '0.0'};return vec2<f32>(${getWGSLFloatLiteral(grid.affine[0])}*column+${getWGSLFloatLiteral(grid.affine[1])}*row+${getWGSLFloatLiteral(grid.affine[2] - this.localOrigin[0])},${getWGSLFloatLiteral(grid.affine[3])}*column+${getWGSLFloatLiteral(grid.affine[4])}*row+${getWGSLFloatLiteral(grid.affine[5] - this.localOrigin[1])});}`,
        body: `if(index>=reachCount[reachCountOffset]){return;}var current=sources[sourcesOffset+index];var total=0.0;var previousCell=current;
for(var vertex=offsets[offsetsOffset+index];vertex<offsets[offsetsOffset+index+1u];vertex++){
 let point=position(current);positions[positionsOffset+vertex*2u]=point.x;positions[positionsOffset+vertex*2u+1u]=point.y;
 // Measure the displacement between cells before adding the origin, which can round away small edges.
 total+=getGroundEdgeLength(previousCell,current);previousCell=current;
 if(vertex+1u<offsets[offsetsOffset+index+1u]){current=getFlowReceiver(current,getFlowDirectionCode(directions[directionsOffset+current]));}
}targets[targetsOffset+index]=current;lengths[lengthsOffset+index]=total;`
      })
    ];
  }
}
