// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {captureUrbanLandscapeDestinations} from './urban-landscape-destinations';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {validateProvidedRecipeViews} from './urban-landscape-recipe-validation';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Most vertices: the single solver thread keeps five private arrays of one float or word per vertex. */
export const RESISTANCE_CONNECTIVITY_MAXIMUM_VERTEX_COUNT = 256;

/** Most edges: the single solver thread sweeps all of them in every conjugate gradient iteration. */
export const RESISTANCE_CONNECTIVITY_MAXIMUM_EDGE_COUNT = 65536;

/** Most conjugate gradient iterations: they run inside the single solver thread. */
export const RESISTANCE_CONNECTIVITY_MAXIMUM_ITERATIONS = 4096;

export type GPUResistanceConnectivityOutput = {
  /** `float32`, `vertexCount` rows: potential of every vertex that shares the sink's component. */
  potentials: GraphDataView<'float32'>;
  /** `float32`, one row per edge: signed current from the first to the second endpoint. */
  currents: GraphDataView<'float32'>;
  /** `float32x4`, one row: effective resistance, true residual, dissipated energy, recursive residual. */
  diagnostics: GraphDataView<'float32x4'>;
  /** `uint32x4`, one row: `[converged, iterations, reachable, invalidEdges]`. */
  status: GraphDataView<'uint32x4'>;
};

export type GPUResistanceConnectivityProps = {
  id?: string;
  /** Undirected edges listed exactly once. Parallel conductances add; zero conductance is a barrier. */
  edges: GraphDataView<'uint32x2'>;
  conductances: GraphDataView<'float32'>;
  vertexCount: number;
  source: number;
  sink: number;
  maximumIterations?: number;
  /** Absolute maximum current-conservation residual for unit injection. */
  tolerance?: number;
  /**
   * Caller-owned result views. Any view that is absent is a graph-owned transient view, as before.
   * Formats and lengths are those of {@link GPUResistanceConnectivityOutput}.
   */
  output?: Partial<GPUResistanceConnectivityOutput>;
};

/**
 * Bounded small-network grounded-Laplacian CG solve, with independently recomputed current residual.
 *
 * @deprecated Use {@link addSparseResistanceConnectivityRecipe}. This recipe solves the whole
 * system in one GPU thread, so it is limited to 256 vertices and 65,536 edges. The sparse recipe takes the same edges, conductances,
 * source and sink and also reports per-vertex currents; it is parallel, and its limits are 1,048,576
 * vertices and 2,097,152 edges. Its default tolerance is 1e-4 (this recipe's is 1e-5) and its
 * iterations are capped by `maximumIterations` (default 1,024; this recipe's is 256).
 */
export function addResistanceConnectivityRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUResistanceConnectivityProps
) {
  const id = props.id ?? 'resistance-connectivity';
  validateRequirement(id, !('outputs' in props), 'outputs was removed; use output');
  validateSinglePackedView(props.edges, ['uint32x2'], id + ' edges');
  validateSinglePackedView(props.conductances, ['float32'], id + ' conductances');
  const {vertexCount, source, sink} = props;
  const edgeCount = props.edges.length;
  const maximumIterations = props.maximumIterations ?? 256;
  const tolerance = props.tolerance ?? 1e-5;
  // A reference-size solver: one thread solves the whole system. Use the sparse recipe above these sizes.
  validateCapacity(id, 'vertexCount', vertexCount, 2, RESISTANCE_CONNECTIVITY_MAXIMUM_VERTEX_COUNT);
  validateCapacity(id, 'source', source, 0, vertexCount - 1);
  validateCapacity(id, 'sink', sink, 0, vertexCount - 1);
  validateRequirement(id, source !== sink, 'source and sink must be different vertices');
  validateRequirement(
    id,
    edgeCount > 0 && edgeCount <= RESISTANCE_CONNECTIVITY_MAXIMUM_EDGE_COUNT,
    `edges must have between 1 and ${RESISTANCE_CONNECTIVITY_MAXIMUM_EDGE_COUNT} rows`
  );
  validateViewLength(id, 'conductances', props.conductances, edgeCount);
  validateCapacity(
    id,
    'maximumIterations',
    maximumIterations,
    1,
    RESISTANCE_CONNECTIVITY_MAXIMUM_ITERATIONS
  );
  validateRequirement(
    id,
    Number.isFinite(tolerance) && tolerance > 0,
    'tolerance must be a positive finite number'
  );
  const output = captureUrbanLandscapeDestinations(id, {output: props.output}, [
    'potentials',
    'currents',
    'diagnostics',
    'status'
  ]);
  validateGraphViewsBelongToGraph(id, graph, [props.edges, props.conductances]);
  validateProvidedRecipeViews(graph, [
    [`${id}-potentials`, 'float32', vertexCount, output?.potentials],
    [`${id}-currents`, 'float32', edgeCount, output?.currents],
    [`${id}-status`, 'uint32x4', 1, output?.status],
    [`${id}-diagnostics`, 'float32x4', 1, output?.diagnostics]
  ]);
  const potentials = getRecipeView(
    graph,
    `${id}-potentials`,
    'float32',
    vertexCount,
    output?.potentials
  );
  const currents = getRecipeView(graph, `${id}-currents`, 'float32', edgeCount, output?.currents);
  const status = getRecipeView(graph, `${id}-status`, 'uint32x4', 1, output?.status);
  const diagnostics = getRecipeView(
    graph,
    `${id}-diagnostics`,
    'float32x4',
    1,
    output?.diagnostics
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-solve`,
      operation: id,
      invocationCount: 1,
      workgroupSize: 2,
      declarations:
        IS_FINITE_VALUE_WGSL +
        `
fn readEdges(row:u32)->vec2<u32>{return vec2<u32>(edges[edgesOffset+row*2u+0u],edges[edgesOffset+row*2u+1u]);}
fn readStatus(row:u32)->vec4<u32>{return vec4<u32>(status[statusOffset+row*4u+0u],status[statusOffset+row*4u+1u],status[statusOffset+row*4u+2u],status[statusOffset+row*4u+3u]);}
fn writeStatus(row:u32,value:vec4<u32>){status[statusOffset+row*4u+0u]=value[0u];status[statusOffset+row*4u+1u]=value[1u];status[statusOffset+row*4u+2u]=value[2u];status[statusOffset+row*4u+3u]=value[3u];}
fn readDiagnostics(row:u32)->vec4<f32>{return vec4<f32>(diagnostics[diagnosticsOffset+row*4u+0u],diagnostics[diagnosticsOffset+row*4u+1u],diagnostics[diagnosticsOffset+row*4u+2u],diagnostics[diagnosticsOffset+row*4u+3u]);}
fn writeDiagnostics(row:u32,value:vec4<f32>){diagnostics[diagnosticsOffset+row*4u+0u]=value[0u];diagnostics[diagnosticsOffset+row*4u+1u]=value[1u];diagnostics[diagnosticsOffset+row*4u+2u]=value[2u];diagnostics[diagnosticsOffset+row*4u+3u]=value[3u];}
`,
      bindings: [
        {name: 'edges', view: props.edges, type: 'u32', access: 'read'},
        {name: 'conductances', view: props.conductances, type: 'f32', access: 'read'},
        {name: 'potentials', view: potentials, type: 'f32', access: 'read_write'},
        {name: 'status', view: status, type: 'u32', access: 'read_write'},
        {name: 'diagnostics', view: diagnostics, type: 'f32', access: 'read_write'}
      ],
      body: `var voltage:array<f32,${vertexCount}>;var residual:array<f32,${vertexCount}>;var search:array<f32,${vertexCount}>;var product:array<f32,${vertexCount}>;var connected:array<u32,${vertexCount}>;
var invalid=0u;for(var edge=0u;edge<${edgeCount}u;edge++){let endpoints=readEdges(edge);let conductance=conductances[conductancesOffset+edge];invalid+=u32(any(endpoints>=vec2<u32>(${vertexCount}u))||!isFiniteValue(conductance)||conductance<0.0);}
connected[${sink}u]=1u;
if(invalid==0u){for(var round=0u;round<${vertexCount}u;round++){
 var changed=false;for(var edge=0u;edge<${edgeCount}u;edge++){let endpoints=readEdges(edge);if(conductances[conductancesOffset+edge]<=0.0){continue;}
 if(connected[endpoints.x]!=0u||connected[endpoints.y]!=0u){changed=changed||connected[endpoints.x]==0u||connected[endpoints.y]==0u;connected[endpoints.x]=1u;connected[endpoints.y]=1u;}}
 if(!changed){break;}
}}
let reachable=connected[${source}u]!=0u&&invalid==0u;var iterations=0u;var residualSquared=1.0;
residual[${source}u]=1.0;search[${source}u]=1.0;
if(reachable){for(var iteration=0u;iteration<${maximumIterations}u;iteration++){
 if(residualSquared<=${tolerance * tolerance}){break;}
 for(var vertex=0u;vertex<${vertexCount}u;vertex++){product[vertex]=0.0;}
 for(var edge=0u;edge<${edgeCount}u;edge++){let endpoints=readEdges(edge);let difference=conductances[conductancesOffset+edge]*(search[endpoints.x]-search[endpoints.y]);product[endpoints.x]+=difference;product[endpoints.y]-=difference;}
 product[${sink}u]=0.0;var curvature=0.0;for(var vertex=0u;vertex<${vertexCount}u;vertex++){curvature+=search[vertex]*product[vertex];}
 if(!(curvature>0.0)||!isFiniteValue(curvature)){break;}
 let step=residualSquared/curvature;var nextResidual=0.0;
 for(var vertex=0u;vertex<${vertexCount}u;vertex++){voltage[vertex]+=step*search[vertex];residual[vertex]-=step*product[vertex];nextResidual+=residual[vertex]*residual[vertex];}
 let beta=nextResidual/residualSquared;for(var vertex=0u;vertex<${vertexCount}u;vertex++){search[vertex]=residual[vertex]+beta*search[vertex];}
 residualSquared=nextResidual;iterations=iteration+1u;
}}
for(var vertex=0u;vertex<${vertexCount}u;vertex++){product[vertex]=0.0;}
var energy=0.0;if(reachable){for(var edge=0u;edge<${edgeCount}u;edge++){let endpoints=readEdges(edge);let difference=voltage[endpoints.x]-voltage[endpoints.y];let current=conductances[conductancesOffset+edge]*difference;product[endpoints.x]+=current;product[endpoints.y]-=current;energy+=current*difference;}}
var maximumResidual=0.0;var arithmeticValid=true;
for(var vertex=0u;vertex<${vertexCount}u;vertex++){
 if(connected[vertex]!=0u){let expected=select(0.0,1.0,vertex==${source}u)-select(0.0,1.0,vertex==${sink}u);maximumResidual=max(maximumResidual,abs(product[vertex]-expected));arithmeticValid=arithmeticValid&&isFiniteValue(product[vertex])&&isFiniteValue(voltage[vertex]);}
 potentials[potentialsOffset+vertex]=select(bitcast<f32>(0x7fc00000u | (index & 0u)),voltage[vertex],reachable&&connected[vertex]!=0u);
}
let converged=reachable&&arithmeticValid&&maximumResidual<=${tolerance};
writeStatus(0u,vec4<u32>(u32(converged),iterations,u32(reachable),invalid));
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));writeDiagnostics(0u,vec4<f32>(select(missing,voltage[${source}u],converged),select(missing,maximumResidual,reachable),select(missing,energy,reachable),sqrt(residualSquared)));`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-edge-currents`,
      operation: id,
      invocationCount: edgeCount,
      declarations: `
fn readEdges(row:u32)->vec2<u32>{return vec2<u32>(edges[edgesOffset+row*2u+0u],edges[edgesOffset+row*2u+1u]);}
fn readStatus(row:u32)->vec4<u32>{return vec4<u32>(status[statusOffset+row*4u+0u],status[statusOffset+row*4u+1u],status[statusOffset+row*4u+2u],status[statusOffset+row*4u+3u]);}
`,
      bindings: [
        {name: 'edges', view: props.edges, type: 'u32', access: 'read'},
        {name: 'conductances', view: props.conductances, type: 'f32', access: 'read'},
        {name: 'potentials', view: potentials, type: 'f32', access: 'read'},
        {name: 'status', view: status, type: 'u32', access: 'read'},
        {name: 'output', view: currents, type: 'f32', access: 'read_write'}
      ],
      body: `var current=bitcast<f32>(0x7fc00000u | (index & 0u));let endpoints=readEdges(index);
if(readStatus(0u).x!=0u){let conductance=conductances[conductancesOffset+index];current=0.0;if(conductance>0.0){current=conductance*(potentials[potentialsOffset+endpoints.x]-potentials[potentialsOffset+endpoints.y]);}}
output[outputOffset+index]=current;`
    })
  );
  return {
    outputs: {potentials, currents, diagnostics},
    status,
    provenance: {
      edges: props.edges,
      conductances: props.conductances,
      source,
      sink,
      injection: 1,
      tolerance,
      maximumIterations,
      solver: 'grounded-conjugate-gradient' as const
    }
  };
}
