// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {dggs} from '@luma.gl/shaders';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateSinglePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUH3CellIdentities,
  H3_VERTEX_IDENTITY_WGSL
} from '../cell-topology/gpu-h3-cell-identities';
import {H3_NEIGHBOR_WGSL} from '../cell-topology/h3-neighbor-wgsl';
import {H3_BOUNDARY_WGSL} from '../cell-indexing/h3-boundary-wgsl';
import {getOrCreateView, RecipeBuilder} from './recipe-utils';

export type GPUCellFlowTopologyOutput = {
  outgoingTotals: GraphDataView<'float32'>;
  incomingTotals: GraphDataView<'float32'>;
  /** Six direction slots per cell, H3 directed-edge IDs; pentagon missing direction is zero. */
  directedEdges: GraphDataView<'uint32x2'>;
  neighboringCells: GraphDataView<'uint32x2'>;
  edgeFlows: GraphDataView<'float32'>;
  /** Three longitude/latitude vertices per edge slot; first boundaryCounts entries are used.
   * Includes H3 face-distortion vertices. Longitude wraps at the antimeridian.
   */
  boundaries: GraphDataView<'float32x2'>;
  boundaryCounts: GraphDataView<'uint32'>;
  /** One flag per original pair; rejected rows contribute to neither totals nor edge flows. */
  validity: GraphDataView<'uint32'>;
  status: {invalidCount: GraphDataView<'uint32'>};
};
export type GPUCellFlowTopologyRecipeProps = {
  id?: string;
  /** Distinct little-endian H3 cells, all at the same resolution. */
  cells: GraphDataView<'uint32x2'>;
  /** Origin and destination row IDs into cells; arbitrary nonneighbor pairs remain in pairTable. */
  origins: GraphDataView<'uint32'>;
  destinations: GraphDataView<'uint32'>;
  /** Nonnegative finite flow per pair. Repeated pairs are summed deterministically in input order. */
  values: GraphDataView<'float32'>;
  outputs?: Partial<GPUCellFlowTopologyOutput>;
};

/** R11: preserves the original OD table and separately aggregates actual neighboring-cell flows.
 * Cell totals include every valid OD pair (including self and nonneighbor pairs). Edge totals
 * include only immediate H3 neighbors. Therefore sum(outgoing) == sum(incoming) == valid pair sum,
 * while the edge sum can be smaller. No route inference is performed. The bounded direct scan
 * is O(cellCount * pairCount); use preaggregated pairs for large workloads. H3 keys are exact;
 * shared-boundary coordinates inherit the existing f32 H3 geometry precision limits.
 */
export function addCellFlowTopologyRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUCellFlowTopologyRecipeProps
) {
  const id = props.id ?? 'cell-flow-topology';
  const cellCount = props.cells.length;
  const pairCount = props.values.length;
  const edgeCount = cellCount * 6;
  for (const [view, format] of [
    [props.cells, 'uint32x2'],
    [props.origins, 'uint32'],
    [props.destinations, 'uint32'],
    [props.values, 'float32']
  ] as const)
    validateSinglePackedView(view, [format], id);
  if (
    props.origins.length !== pairCount ||
    props.destinations.length !== pairCount ||
    cellCount < 1 ||
    pairCount < 1
  )
    throw new Error('Cell flow requires aligned nonempty inputs');
  const outputs: GPUCellFlowTopologyOutput = {
    outgoingTotals: getOrCreateView(
      graph,
      `${id}-outgoing`,
      'float32',
      cellCount,
      props.outputs?.outgoingTotals
    ),
    incomingTotals: getOrCreateView(
      graph,
      `${id}-incoming`,
      'float32',
      cellCount,
      props.outputs?.incomingTotals
    ),
    directedEdges: getOrCreateView(
      graph,
      `${id}-edges`,
      'uint32x2',
      edgeCount,
      props.outputs?.directedEdges
    ),
    neighboringCells: getOrCreateView(
      graph,
      `${id}-neighbors`,
      'uint32x2',
      edgeCount,
      props.outputs?.neighboringCells
    ),
    edgeFlows: getOrCreateView(
      graph,
      `${id}-flows`,
      'float32',
      edgeCount,
      props.outputs?.edgeFlows
    ),
    boundaries: getOrCreateView(
      graph,
      `${id}-boundaries`,
      'float32x2',
      edgeCount * 3,
      props.outputs?.boundaries
    ),
    boundaryCounts: getOrCreateView(
      graph,
      `${id}-boundary-counts`,
      'uint32',
      edgeCount,
      props.outputs?.boundaryCounts
    ),
    validity: getOrCreateView(
      graph,
      `${id}-validity`,
      'uint32',
      pairCount,
      props.outputs?.validity
    ),
    status: {
      invalidCount: getOrCreateView(
        graph,
        `${id}-invalid-count`,
        'uint32',
        1,
        props.outputs?.status?.invalidCount
      )
    }
  };
  for (const [name, view] of Object.entries(getOutputColumns(outputs))) {
    const format =
      name === 'directedEdges' || name === 'neighboringCells'
        ? 'uint32x2'
        : name === 'boundaries'
          ? 'float32x2'
          : name === 'boundaryCounts' || name === 'validity' || name === 'invalidCount'
            ? 'uint32'
            : 'float32';
    const length =
      name === 'outgoingTotals' || name === 'incomingTotals'
        ? cellCount
        : name === 'boundaries'
          ? edgeCount * 3
          : name === 'validity'
            ? pairCount
            : name === 'invalidCount'
              ? 1
              : edgeCount;
    validateSinglePackedView(view, [format], `${id} ${name}`);
    if (view.length !== length) throw new Error('Cell flow output length mismatch');
  }
  const outputBuffers = Object.values(getOutputColumns(outputs)).map(view => view.buffer);
  if (new Set(outputBuffers).size !== outputBuffers.length)
    throw new Error('Cell flow outputs must use distinct buffers');
  const inputs = [props.cells, props.origins, props.destinations, props.values];
  validateGraphViewsBelongToGraph(id, graph, [
    ...inputs,
    ...Object.values(getOutputColumns(outputs))
  ]);
  validateGraphOutputsDisjointFromInputs(id, Object.values(getOutputColumns(outputs)), inputs);
  const builder = new RecipeBuilder(graph);
  const addNode = (node: GPUCommandNode<Parameters>) => builder.addGlue(node.id, node);
  builder.add(
    new GPUH3CellIdentities({
      id: `${id}-identities`,
      cells: props.cells,
      operation: 'directed-edges',
      output: outputs.directedEdges
    })
  );
  const constants = `const CELL_COUNT: u32 = ${cellCount}u; const PAIR_COUNT: u32 = ${pairCount}u;`;
  const cellBinding = {name: 'cells', view: props.cells, type: 'u32', access: 'read'} as const;
  const pairBindings = [
    {name: 'origins', view: props.origins, type: 'u32', access: 'read'},
    {name: 'destinations', view: props.destinations, type: 'u32', access: 'read'},
    {name: 'values', view: props.values, type: 'f32', access: 'read'}
  ] as const;
  addNode(
    createWGSLKernelNode(graph, {
      id: `${id}-validate`,
      operation: 'CellFlowTopologyRecipe',
      variant: 'validate',
      invocationCount: pairCount,
      bindings: [
        cellBinding,
        ...pairBindings,
        {name: 'validity', view: outputs.validity, type: 'u32', access: 'read_write'}
      ],
      declarations: `${constants}\n${dggs.source}`,
      body: `let origin=origins[originsOffset+index]; let destination=destinations[destinationsOffset+index]; let value=values[valuesOffset+index];
  var valid=origin<CELL_COUNT && destination<CELL_COUNT && value>=0.0 && (bitcast<u32>(value)&0x7f800000u)!=0x7f800000u;
  if(valid){let first=vec2u(cells[cellsOffset+origin*2u+1u],cells[cellsOffset+origin*2u]);let second=vec2u(cells[cellsOffset+destination*2u+1u],cells[cellsOffset+destination*2u]);valid=dggs_h3_is_valid_cell_id(first)&&dggs_h3_is_valid_cell_id(second)&&dggs_h3_get_resolution(first)==dggs_h3_get_resolution(second);}
  validity[validityOffset+index]=select(0u,1u,valid);`
    })
  );
  addNode(
    createWGSLKernelNode(graph, {
      id: `${id}-cell-totals`,
      operation: 'CellFlowTopologyRecipe',
      variant: 'cell-totals',
      invocationCount: cellCount,
      bindings: [
        ...pairBindings,
        {name: 'validity', view: outputs.validity, type: 'u32', access: 'read'},
        {name: 'outgoing', view: outputs.outgoingTotals, type: 'f32', access: 'read_write'},
        {name: 'incoming', view: outputs.incomingTotals, type: 'f32', access: 'read_write'},
        {name: 'invalidCount', view: outputs.status.invalidCount, type: 'u32', access: 'read_write'}
      ],
      declarations: constants,
      body: `var outgoingTotal=0.0;var incomingTotal=0.0;var invalid=0u;
  for(var pair=0u;pair<PAIR_COUNT;pair++){if(validity[validityOffset+pair]==0u){invalid++;continue;}let value=values[valuesOffset+pair];if(origins[originsOffset+pair]==index){outgoingTotal+=value;}if(destinations[destinationsOffset+pair]==index){incomingTotal+=value;}}
  outgoing[outgoingOffset+index]=outgoingTotal;incoming[incomingOffset+index]=incomingTotal;if(index==0u){invalidCount[invalidCountOffset]=invalid;}`
    })
  );
  addNode(
    createWGSLKernelNode(graph, {
      id: `${id}-boundaries`,
      operation: 'CellFlowTopologyRecipe',
      variant: 'boundaries',
      invocationCount: edgeCount,
      bindings: [
        cellBinding,
        {name: 'neighbors', view: outputs.neighboringCells, type: 'u32', access: 'read_write'},
        {name: 'boundaries', view: outputs.boundaries, type: 'f32', access: 'read_write'},
        {name: 'counts', view: outputs.boundaryCounts, type: 'u32', access: 'read_write'}
      ],
      declarations: `${dggs.source}\n${H3_NEIGHBOR_WGSL}\n${H3_VERTEX_IDENTITY_WGSL}\n${H3_BOUNDARY_WGSL}`,
      body: `let row=index/6u;let direction=index%6u+1u;let cell=vec2u(cells[cellsOffset+row*2u+1u],cells[cellsOffset+row*2u]);let neighbor=cellTopologyH3Neighbor(cell,direction);
  neighbors[neighborsOffset+index*2u]=neighbor.y;neighbors[neighborsOffset+index*2u+1u]=neighbor.x;
  var count=0u;var points:array<vec2f,3>;
  if(any(neighbor!=vec2u(0u))){let boundary=cellIndexH3GetBoundary(cell);let vertex=identityVertexNumber(cell,direction);let total=select(6u,5u,cellTopologyH3IsPentagon(cell));
    let first=boundary.vertexOffsets[vertex];let last=boundary.vertexOffsets[(vertex+1u)%total];
    if(boundary.count>0u){count=(last+boundary.count-first)%boundary.count+1u;count=min(count,3u);for(var slot=0u;slot<count;slot++){points[slot]=boundary.points[(first+slot)%boundary.count];}}
  }
  counts[countsOffset+index]=count;for(var slot=0u;slot<3u;slot++){boundaries[boundariesOffset+index*6u+slot*2u]=points[slot].x;boundaries[boundariesOffset+index*6u+slot*2u+1u]=points[slot].y;}`
    })
  );
  addNode(
    createWGSLKernelNode(graph, {
      id: `${id}-edge-flows`,
      operation: 'CellFlowTopologyRecipe',
      variant: 'edge-flows',
      invocationCount: edgeCount,
      bindings: [
        cellBinding,
        ...pairBindings,
        {name: 'validity', view: outputs.validity, type: 'u32', access: 'read'},
        {name: 'neighbors', view: outputs.neighboringCells, type: 'u32', access: 'read'},
        {name: 'flows', view: outputs.edgeFlows, type: 'f32', access: 'read_write'}
      ],
      declarations: constants,
      body: `let origin=index/6u;let neighbor=vec2u(neighbors[neighborsOffset+index*2u],neighbors[neighborsOffset+index*2u+1u]);var total=0.0;
  if(any(neighbor!=vec2u(0u))){for(var pair=0u;pair<PAIR_COUNT;pair++){if(validity[validityOffset+pair]==0u||origins[originsOffset+pair]!=origin){continue;}let destination=destinations[destinationsOffset+pair];let key=vec2u(cells[cellsOffset+destination*2u],cells[cellsOffset+destination*2u+1u]);if(all(key==neighbor)){total+=values[valuesOffset+pair];}}}
  flows[flowsOffset+index]=total;`
    })
  );
  return {
    contributors: builder.contributors,
    outputs,
    pairTable: {
      origins: props.origins,
      destinations: props.destinations,
      values: props.values,
      validity: outputs.validity
    },
    intermediates: {cells: props.cells},
    status: {stages: [{stage: 'flow-input', status: {invalidCount: outputs.status.invalidCount}}]}
  };
}

function getOutputColumns(output: GPUCellFlowTopologyOutput) {
  const {status, ...columns} = output;
  return {...columns, invalidCount: status.invalidCount};
}
