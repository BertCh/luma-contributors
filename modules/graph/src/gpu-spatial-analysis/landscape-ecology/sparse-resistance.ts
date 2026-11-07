// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import {
  GPUJacobiPCG,
  GPUSort,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';
import type {GPUResistanceConnectivityProps} from './resistance-connectivity';
import {
  addSerialSparseResistanceComponents,
  addSparseResistanceComponents
} from './sparse-resistance-components';

/** Sparse grounded Laplacian with GPU union-find, sorted CSR and reusable Jacobi-PCG. */
export function addSparseResistanceConnectivityRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUResistanceConnectivityProps
) {
  const id = props.id ?? 'sparse-resistance';
  const {vertexCount, source, sink} = props;
  const edgeCount = props.edges.length;
  const maximumIterations = props.maximumIterations ?? 256;
  const tolerance = props.tolerance ?? 1e-5;
  assert(Number.isSafeInteger(vertexCount) && vertexCount >= 2 && vertexCount <= 16777216);
  assert(Number.isSafeInteger(source) && source >= 0 && source < vertexCount);
  assert(Number.isSafeInteger(sink) && sink >= 0 && sink < vertexCount && source !== sink);
  assert(
    edgeCount > 0 &&
      props.conductances.length === edgeCount &&
      4 * edgeCount + vertexCount < 0x80000000
  );
  assert(
    Number.isSafeInteger(maximumIterations) && maximumIterations > 0 && maximumIterations <= 4096
  );
  assert(Number.isFinite(tolerance) && tolerance > 0);
  const view = <Format extends 'uint32' | 'uint32x4' | 'float32' | 'float32x4'>(
    name: string,
    format: Format,
    length: number
  ) => getRecipeView(graph, `${id}-${name}`, format, length);
  const components = view('components', 'uint32', vertexCount);
  const status = view('status', 'uint32x4', 1);
  const diagnostics = view('diagnostics', 'float32x4', 1);
  const potentials = view('potentials', 'float32', vertexCount);
  const currents = view('currents', 'float32', edgeCount);
  const entries = 4 * edgeCount + vertexCount;
  const keys = view('keys', 'uint32', entries);
  const ordering = view('ordering', 'uint32', entries);
  const sortedKeys = view('sorted-keys', 'uint32', entries);
  const sortedOrdering = view('sorted-ordering', 'uint32', entries);
  const rowOffsets = view('row-offsets', 'uint32', vertexCount + 1);
  const columns = view('columns', 'uint32', entries);
  const values = view('values', 'float32', entries);
  const rhs = view('rhs', 'float32', vertexCount);
  const residual = view('true-residual', 'float32', vertexCount);
  // Small graphs retain the original single-pass discovery and scheduling.
  const useSerialComponents = vertexCount <= 256 && edgeCount <= 256;
  const addComponents = useSerialComponents
    ? addSerialSparseResistanceComponents
    : addSparseResistanceComponents;
  const componentPreparation = addComponents(graph, {
    id,
    edges: props.edges,
    conductances: props.conductances,
    components,
    status,
    vertexCount,
    source,
    sink,
    maximumIterations
  });
  graph.add({
    // Finish preparation before sorting so long-lived solver buffers do not occupy sort scratch.
    dependsOn: useSerialComponents ? undefined : [componentPreparation.completionNodeId],
    ...createRecipeKernelNode(graph, {
      id: `${id}-coo-rows`,
      operation: id,
      invocationCount: entries,
      bindings: [
        {name: 'edges', view: props.edges, type: 'u32', access: 'read'},
        {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
        {name: 'ordering', view: ordering, type: 'u32', access: 'read_write'}
      ],
      body: `var row=index-${4 * edgeCount}u;if(index<${4 * edgeCount}u){let edge=index/4u;row=min(edges[edgesOffset+2u*edge+(index%2u)],${vertexCount - 1}u);}keys[keysOffset+index]=row;ordering[orderingOffset+index]=index;`
    })
  });
  graph.add(
    new GPUSort({
      id: `${id}-sort`,
      keys,
      values: ordering,
      outputKeys: sortedKeys,
      outputValues: sortedOrdering,
      keyBits: Math.ceil(Math.log2(vertexCount))
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-csr-values`,
      operation: id,
      invocationCount: entries,
      bindings: [
        {name: 'ordering', view: sortedOrdering, type: 'u32', access: 'read'},
        {name: 'edges', view: props.edges, type: 'u32', access: 'read'},
        {name: 'weights', view: props.conductances, type: 'f32', access: 'read'},
        {name: 'parents', view: components, type: 'u32', access: 'read'},
        {name: 'status', view: status, type: 'u32', access: 'read'},
        {name: 'columns', view: columns, type: 'u32', access: 'read_write'},
        {name: 'values', view: values, type: 'f32', access: 'read_write'}
      ],
      body: `let original=ordering[orderingOffset+index];var column=original-${4 * edgeCount}u;var value=0.0;
if(original<${4 * edgeCount}u){let edge=original/4u;let part=original%4u;let first=min(edges[edgesOffset+2u*edge],${vertexCount - 1}u);let second=min(edges[edgesOffset+2u*edge+1u],${vertexCount - 1}u);let row=select(first,second,part%2u==1u);column=select(row,select(second,first,part%2u==1u),part>=2u);
 if(status[statusOffset+2u]!=0u&&parents[parentsOffset+row]==parents[parentsOffset+${sink}u]&&row!=${sink}u&&column!=${sink}u){value=weights[weightsOffset+edge]*select(1.0,-1.0,part>=2u);}
 // Grounding removes only the off-diagonal sink coupling; its edge still contributes to the other endpoint's diagonal.
}else{value=f32(column==${sink}u||parents[parentsOffset+column]!=parents[parentsOffset+${sink}u]||status[statusOffset+2u]==0u);}
columns[columnsOffset+index]=column;values[valuesOffset+index]=value;`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-csr-offsets`,
      operation: id,
      invocationCount: vertexCount + 1,
      bindings: [
        {name: 'keys', view: sortedKeys, type: 'u32', access: 'read'},
        {name: 'offsets', view: rowOffsets, type: 'u32', access: 'read_write'}
      ],
      body: `var low=0u;var high=${entries}u;loop{if(low>=high){break;}let middle=low+(high-low)/2u;if(keys[keysOffset+middle]<index){low=middle+1u;}else{high=middle;}}offsets[offsetsOffset+index]=low;`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-initialize`,
      operation: id,
      invocationCount: vertexCount,
      bindings: [
        {name: 'status', view: status, type: 'u32', access: 'read'},
        {name: 'rhs', view: rhs, type: 'f32', access: 'read_write'},
        {name: 'potentials', view: potentials, type: 'f32', access: 'read_write'}
      ],
      body: `rhs[rhsOffset+index]=f32(index==${source}u&&status[statusOffset+2u]!=0u);potentials[potentialsOffset+index]=0.0;`
    })
  );
  const solver = new GPUJacobiPCG({
    id: `${id}-pcg`,
    rowOffsets,
    columnIndices: columns,
    values,
    rhs,
    solution: potentials,
    columns: vertexCount,
    iterations: maximumIterations,
    toleranceSquared: tolerance * tolerance * 0.01
  });
  graph.add(solver);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-true-residual`,
      operation: id,
      invocationCount: vertexCount,
      bindings: [
        {name: 'offsets', view: rowOffsets, type: 'u32', access: 'read'},
        {name: 'columns', view: columns, type: 'u32', access: 'read'},
        {name: 'values', view: values, type: 'f32', access: 'read'},
        {name: 'potentials', view: potentials, type: 'f32', access: 'read'},
        {name: 'rhs', view: rhs, type: 'f32', access: 'read'},
        {name: 'residual', view: residual, type: 'f32', access: 'read_write'}
      ],
      body: `var product=0.0;for(var entry=offsets[offsetsOffset+index];entry<offsets[offsetsOffset+index+1u];entry++){product+=values[valuesOffset+entry]*potentials[potentialsOffset+columns[columnsOffset+entry]];}residual[residualOffset+index]=product-rhs[rhsOffset+index];`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-certificate`,
      operation: id,
      invocationCount: 1,
      workgroupSize: 2,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'residual', view: residual, type: 'f32', access: 'read'},
        {name: 'potentials', view: potentials, type: 'f32', access: 'read'},
        {name: 'edges', view: props.edges, type: 'u32', access: 'read'},
        {name: 'weights', view: props.conductances, type: 'f32', access: 'read'},
        {name: 'status', view: status, type: 'u32', access: 'read_write'},
        {name: 'diagnostics', view: diagnostics, type: 'f32', access: 'read_write'}
      ],
      body: `var maximum=0.0;var squared=0.0;var valid=status[statusOffset+2u]!=0u;var sum=0.0;
for(var vertex=0u;vertex<${vertexCount}u;vertex++){let value=residual[residualOffset+vertex];valid=valid&&isFiniteValue(value);maximum=max(maximum,abs(value));squared+=value*value;sum+=value;}
// The grounded row residual is zero; current balance at the sink is the negative sum of all other residuals.
maximum=max(maximum,abs(sum));var energy=0.0;if(valid){for(var edge=0u;edge<${edgeCount}u;edge++){let difference=potentials[potentialsOffset+edges[edgesOffset+2u*edge]]-potentials[potentialsOffset+edges[edgesOffset+2u*edge+1u]];energy+=weights[weightsOffset+edge]*difference*difference;}}
let converged=valid&&maximum<=${tolerance}&&isFiniteValue(energy);status[statusOffset]=u32(converged);let missing=bitcast<f32>(0x7fc00000u | (index & 0u));diagnostics[diagnosticsOffset]=select(missing,potentials[potentialsOffset+${source}u],converged);diagnostics[diagnosticsOffset+1u]=select(missing,maximum,valid);diagnostics[diagnosticsOffset+2u]=select(missing,energy,valid);diagnostics[diagnosticsOffset+3u]=sqrt(squared);`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-currents`,
      operation: id,
      invocationCount: edgeCount,
      bindings: [
        {name: 'edges', view: props.edges, type: 'u32', access: 'read'},
        {name: 'weights', view: props.conductances, type: 'f32', access: 'read'},
        {name: 'potentials', view: potentials, type: 'f32', access: 'read'},
        {name: 'parents', view: components, type: 'u32', access: 'read'},
        {name: 'status', view: status, type: 'u32', access: 'read'},
        {name: 'currents', view: currents, type: 'f32', access: 'read_write'}
      ],
      body: `var current=bitcast<f32>(0x7fc00000u | (index & 0u));if(status[statusOffset]!=0u){let first=edges[edgesOffset+2u*index];let second=edges[edgesOffset+2u*index+1u];let conductance=weights[weightsOffset+index];if(conductance==0.0){current=0.0;}else if(parents[parentsOffset+first]==parents[parentsOffset+${sink}u]){current=conductance*(potentials[potentialsOffset+first]-potentials[potentialsOffset+second]);}}currents[currentsOffset+index]=current;`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-ungrounded-mask`,
      operation: id,
      invocationCount: vertexCount,
      bindings: [
        {name: 'parents', view: components, type: 'u32', access: 'read'},
        {name: 'status', view: status, type: 'u32', access: 'read'},
        {name: 'potentials', view: potentials, type: 'f32', access: 'read_write'}
      ],
      body: `if(status[statusOffset+2u]==0u||parents[parentsOffset+index]!=parents[parentsOffset+${sink}u]){potentials[potentialsOffset+index]=bitcast<f32>(0x7fc00000u | (index & 0u));}`
    })
  );
  return {
    outputs: {potentials, currents, diagnostics},
    status,
    intermediates: {components, rowOffsets, columns, values, residual},
    provenance: {
      source,
      sink,
      vertexCount,
      tolerance,
      maximumIterations,
      solver: 'sparse-jacobi-pcg' as const,
      statusIterationMeaning: 'configured-budget' as const
    }
  };
}

export type GPURasterResistanceProps = {
  id?: string;
  resistance: GraphDataView<'float32'>;
  width: number;
  height: number;
  cellSize: readonly [number, number];
  connectivity: 4 | 8;
  source: number;
  sink: number;
  maximumIterations?: number;
  tolerance?: number;
};

/** Positive finite resistance cells; other cells are barriers. Harmonic mean conductance / physical edge length. */
export function addRasterResistanceConnectivityRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPURasterResistanceProps
) {
  const id = props.id ?? 'raster-resistance';
  const count = props.width * props.height;
  const directions = props.connectivity === 8 ? 4 : 2;
  assert(
    Number.isSafeInteger(props.width) &&
      props.width > 0 &&
      Number.isSafeInteger(props.height) &&
      props.height > 0 &&
      props.resistance.length === count
  );
  assert(props.connectivity === 4 || props.connectivity === 8);
  assert(props.cellSize.every(value => Number.isFinite(value) && value > 0));
  const edges = getRecipeView(graph, `${id}-edges`, 'uint32x2', count * directions);
  const conductances = getRecipeView(graph, `${id}-conductances`, 'float32', count * directions);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-conductance`,
      operation: id,
      invocationCount: count * directions,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'resistance', view: props.resistance, type: 'f32', access: 'read'},
        {name: 'edges', view: edges, type: 'u32', access: 'read_write'},
        {name: 'conductances', view: conductances, type: 'f32', access: 'read_write'}
      ],
      body: `let cell=index/${directions}u;let direction=index%${directions}u;let column=i32(cell%${props.width}u);let row=i32(cell/${props.width}u);let offsets=array<vec2<i32>,4>(vec2<i32>(1,0),vec2<i32>(0,1),vec2<i32>(1,1),vec2<i32>(-1,1));let next=vec2<i32>(column,row)+offsets[direction];var neighbor=cell;var conductance=0.0;
if(all(next>=vec2<i32>(0))&&next.x<${props.width}&&next.y<${props.height}){neighbor=u32(next.y)*${props.width}u+u32(next.x);let first=resistance[resistanceOffset+cell];let second=resistance[resistanceOffset+neighbor];if(isFiniteValue(first)&&isFiniteValue(second)&&first>0.0&&second>0.0){let displacement=vec2<f32>(offsets[direction])*vec2<f32>(${props.cellSize[0]},${props.cellSize[1]});conductance=2.0/(first+second)/length(displacement);}}
edges[edgesOffset+2u*index]=cell;edges[edgesOffset+2u*index+1u]=neighbor;conductances[conductancesOffset+index]=conductance;`
    })
  );
  const result = addSparseResistanceConnectivityRecipe(graph, {
    id: `${id}-solve`,
    edges,
    conductances,
    vertexCount: count,
    source: props.source,
    sink: props.sink,
    maximumIterations: props.maximumIterations,
    tolerance: props.tolerance
  });
  const currentDensity = getRecipeView(graph, `${id}-current-density`, 'float32', count);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-current-density`,
      operation: id,
      invocationCount: count,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'currents', view: result.outputs.currents, type: 'f32', access: 'read'},
        {name: 'potentials', view: result.outputs.potentials, type: 'f32', access: 'read'},
        {name: 'output', view: currentDensity, type: 'f32', access: 'read_write'}
      ],
      body: `let column=i32(index%${props.width}u);let row=i32(index/${props.width}u);let directions=array<vec2<i32>,4>(vec2<i32>(1,0),vec2<i32>(0,1),vec2<i32>(1,1),vec2<i32>(-1,1));var incoming=0.0;var outgoing=0.0;var valid=isFiniteValue(potentials[potentialsOffset+index]);
for(var direction=0u;direction<${directions}u;direction++){let outward=currents[currentsOffset+index*${directions}u+direction];valid=valid&&isFiniteValue(outward);incoming+=max(-outward,0.0);outgoing+=max(outward,0.0);let previous=vec2<i32>(column,row)-directions[direction];if(all(previous>=vec2<i32>(0))&&previous.x<${props.width}&&previous.y<${props.height}){let current=currents[currentsOffset+(u32(previous.y)*${props.width}u+u32(previous.x))*${directions}u+direction];valid=valid&&isFiniteValue(current);incoming+=max(current,0.0);outgoing+=max(-current,0.0);}}
output[outputOffset+index]=select(bitcast<f32>(0x7fc00000u | (index & 0u)),max(incoming,outgoing)/${props.cellSize[0] * props.cellSize[1]},valid);`
    })
  );
  return {
    ...result,
    outputs: {...result.outputs, currentDensity},
    raster: {
      edges,
      conductances,
      width: props.width,
      height: props.height,
      cellSize: props.cellSize,
      connectivity: props.connectivity,
      conversion: '2/(resistanceA+resistanceB)/edgeLength' as const
    }
  };
}
