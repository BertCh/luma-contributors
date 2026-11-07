// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

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
};

/** Bounded small-network grounded-Laplacian CG solve, with independently recomputed current residual. */
export function addResistanceConnectivityRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUResistanceConnectivityProps
) {
  const id = props.id ?? 'resistance-connectivity';
  const {vertexCount, source, sink} = props;
  const edgeCount = props.edges.length;
  const maximumIterations = props.maximumIterations ?? 256;
  const tolerance = props.tolerance ?? 1e-5;
  // The local solver deliberately bounds private storage and work; larger systems are not supported (sparse path dropped, see dev-docs/upstream-first-pending.md).
  assert(Number.isSafeInteger(vertexCount) && vertexCount >= 2 && vertexCount <= 256);
  assert(Number.isSafeInteger(source) && source >= 0 && source < vertexCount);
  assert(Number.isSafeInteger(sink) && sink >= 0 && sink < vertexCount && source !== sink);
  assert(edgeCount > 0 && props.conductances.length === edgeCount && edgeCount <= 65536);
  assert(
    Number.isSafeInteger(maximumIterations) && maximumIterations > 0 && maximumIterations <= 4096
  );
  assert(Number.isFinite(tolerance) && tolerance > 0);
  const potentials = getRecipeView(graph, `${id}-potentials`, 'float32', vertexCount);
  const currents = getRecipeView(graph, `${id}-currents`, 'float32', edgeCount);
  const status = getRecipeView(graph, `${id}-status`, 'uint32x4', 1);
  const diagnostics = getRecipeView(graph, `${id}-diagnostics`, 'float32x4', 1);
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
