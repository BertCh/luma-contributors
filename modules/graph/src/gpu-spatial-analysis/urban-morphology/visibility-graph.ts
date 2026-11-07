// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma-contributors/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';
import {classifyUrbanViewpoints, type UrbanFreeSpaceRegion} from './classify-viewpoints';
import {getUrbanRayDeclarations} from './urban-ray-utils';

export type GPUVisibilityGraphProps = {
  id?: string;
  positions: GraphDataView<'float32x2'>;
  segments: GraphDataView<'float32x4'>;
  maximumDistance: number;
  edgeCapacity: number;
  tolerance?: number;
  /** Optional explicit filled-ring free-space classification. Default checks boundaries only. */
  region?: UrbanFreeSpaceRegion;
};

/** Closed obstacles block sightlines. Full dense visibility drives hop integration and Brandes choice. */
export function addVisibilityGraphRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUVisibilityGraphProps
) {
  const id = props.id ?? 'visibility-graph';
  const count = props.positions.length;
  const tolerance = props.tolerance ?? 1e-6;
  assert(
    count > 0 && count <= 256 && props.segments.length > 0 && props.segments.length <= 16777216
  );
  assert(
    Number.isFinite(props.maximumDistance) &&
      props.maximumDistance > 0 &&
      Number.isFinite(tolerance) &&
      tolerance > 0
  );
  assert(Number.isSafeInteger(props.edgeCapacity) && props.edgeCapacity > 0);
  const viewpointClassifications = classifyUrbanViewpoints(graph, {
    id,
    positions: props.positions,
    segments: props.segments,
    tolerance,
    region: props.region
  });
  const adjacency = getRecipeView(graph, `${id}-adjacency`, 'uint32', count * count);
  const edges = getRecipeView(graph, `${id}-edges`, 'uint32x2', props.edgeCapacity);
  const status = getRecipeView(graph, `${id}-status`, 'uint32x4', 1);
  const metrics = getRecipeView(graph, `${id}-metrics`, 'float32x4', count);
  const contributions = getRecipeView(
    graph,
    `${id}-choice-contributions`,
    'float32',
    count * count
  );
  const choice = getRecipeView(graph, `${id}-choice`, 'float32', count);
  const integration = getRecipeView(graph, `${id}-integration`, 'float32x4', count);
  const maximumDepth = getRecipeView(graph, `${id}-maximum-depth`, 'float32', count);
  const extendedIntegration = getRecipeView(
    graph,
    `${id}-extended-integration`,
    'float32x4',
    count
  );
  const normalizedChoice = getRecipeView(graph, `${id}-normalized-choice`, 'float32', count);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-visibility`,
      operation: id,
      invocationCount: count * count,
      declarations:
        IS_FINITE_VALUE_WGSL + getUrbanRayDeclarations(props.segments.length, tolerance),
      bindings: [
        {name: 'classifications', view: viewpointClassifications, type: 'u32', access: 'read'},
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        {name: 'output', view: adjacency, type: 'u32', access: 'read_write'}
      ],
      body: `let first=index/${count}u;let second=index%${count}u;let origin=vec2<f32>(positions[positionsOffset+2u*first],positions[positionsOffset+2u*first+1u]);let destination=vec2<f32>(positions[positionsOffset+2u*second],positions[positionsOffset+2u*second+1u]);let displacement=destination-origin;let distance=length(displacement);
var valid=classifications[classificationsOffset+first]==0u&&classifications[classificationsOffset+second]==0u&&isFiniteValue(origin.x)&&isFiniteValue(origin.y)&&isFiniteValue(destination.x)&&isFiniteValue(destination.y);
for(var segment=0u;segment<${props.segments.length}u;segment++){let values=readSegment(segment);valid=valid&&isFiniteValue(values.x)&&isFiniteValue(values.y)&&isFiniteValue(values.z)&&isFiniteValue(values.w)&&distanceToSegment(origin,values)>${tolerance}&&distanceToSegment(destination,values)>${tolerance};}
var visible=false;if(valid&&first!=second&&distance>${tolerance}&&distance<=${props.maximumDistance}){let hit=castRay(origin,displacement/distance,distance);visible=hit.y<0.0;}
output[outputOffset+index]=select(u32(visible),0xffffffffu,!valid);`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-compact`,
      operation: id,
      invocationCount: 1,
      bindings: [
        {name: 'adjacency', view: adjacency, type: 'u32', access: 'read'},
        {name: 'edges', view: edges, type: 'u32', access: 'read_write'},
        {name: 'status', view: status, type: 'u32', access: 'read_write'}
      ],
      body: `for(var edge=0u;edge<${props.edgeCapacity * 2}u;edge++){edges[edgesOffset+edge]=0xffffffffu;}
var required=0u;var invalid=0u;for(var first=0u;first<${count}u;first++){invalid+=u32(adjacency[adjacencyOffset+first*${count}u+first]==0xffffffffu);for(var second=first+1u;second<${count}u;second++){if(adjacency[adjacencyOffset+first*${count}u+second]==1u){if(required<${props.edgeCapacity}u){edges[edgesOffset+required*2u]=first;edges[edgesOffset+required*2u+1u]=second;}required++;}}}
status[statusOffset]=min(required,${props.edgeCapacity}u);status[statusOffset+1u]=required;status[statusOffset+2u]=u32(required>${props.edgeCapacity}u);status[statusOffset+3u]=invalid;`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-hop-analysis`,
      operation: id,
      invocationCount: count,
      workgroupSize: 2,
      bindings: [
        {name: 'adjacency', view: adjacency, type: 'u32', access: 'read'},
        {name: 'status', view: status, type: 'u32', access: 'read'},
        {name: 'maximumDepth', view: maximumDepth, type: 'f32', access: 'read_write'},
        {name: 'metrics', view: metrics, type: 'f32', access: 'read_write'},
        {name: 'contributions', view: contributions, type: 'f32', access: 'read_write'}
      ],
      body: `var distances:array<i32,${count}>;var paths:array<f32,${count}>;var dependency:array<f32,${count}>;var queue:array<u32,${count}>;
for(var vertex=0u;vertex<${count}u;vertex++){distances[vertex]=-1;}
queue[0u]=index;distances[index]=0;paths[index]=1.0;var head=0u;var tail=1u;var degree=0.0;var distanceSum=0.0;
for(var vertex=0u;vertex<${count}u;vertex++){degree+=f32(adjacency[adjacencyOffset+index*${count}u+vertex]==1u);}
loop{if(head>=tail){break;}let vertex=queue[head];head++;
for(var neighbor=0u;neighbor<${count}u;neighbor++){if(adjacency[adjacencyOffset+vertex*${count}u+neighbor]!=1u){continue;}
if(distances[neighbor]<0){distances[neighbor]=distances[vertex]+1;queue[tail]=neighbor;tail++;distanceSum+=f32(distances[neighbor]);}
if(distances[neighbor]==distances[vertex]+1){paths[neighbor]+=paths[vertex];}
}}
var reverse=tail;loop{if(reverse==0u){break;}reverse--;let vertex=queue[reverse];for(var predecessor=0u;predecessor<${count}u;predecessor++){if(adjacency[adjacencyOffset+predecessor*${count}u+vertex]==1u&&distances[predecessor]==distances[vertex] - 1){dependency[predecessor]+=paths[predecessor]/paths[vertex]*(1.0+dependency[vertex]);}}}
var maximum=0.0;for(var vertex=0u;vertex<${count}u;vertex++){maximum=max(maximum,f32(distances[vertex]));}maximumDepth[maximumDepthOffset+index]=maximum;
let valid=status[statusOffset+3u]==0u;let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
metrics[metricsOffset+4u*index]=select(missing,degree,valid);metrics[metricsOffset+4u*index+1u]=select(missing,f32(tail - 1u),valid);metrics[metricsOffset+4u*index+2u]=select(missing,distanceSum,valid);metrics[metricsOffset+4u*index+3u]=select(missing,f32(tail - 1u)/distanceSum,valid&&distanceSum>0.0);
for(var vertex=0u;vertex<${count}u;vertex++){contributions[contributionsOffset+index*${count}u+vertex]=select(missing,select(dependency[vertex],0.0,vertex==index),valid);}`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-choice`,
      operation: id,
      invocationCount: count,
      bindings: [
        {name: 'contributions', view: contributions, type: 'f32', access: 'read'},
        {name: 'output', view: choice, type: 'f32', access: 'read_write'}
      ],
      body: `var total=0.0;for(var source=0u;source<${count}u;source++){total+=contributions[contributionsOffset+source*${count}u+index];}output[outputOffset+index]=0.5*total;`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-normalized-analysis`,
      operation: id,
      invocationCount: count,
      bindings: [
        {name: 'maximumDepth', view: maximumDepth, type: 'f32', access: 'read'},
        {name: 'extended', view: extendedIntegration, type: 'f32', access: 'read_write'},
        {name: 'metrics', view: metrics, type: 'f32', access: 'read'},
        {name: 'choice', view: choice, type: 'f32', access: 'read'},
        {name: 'integration', view: integration, type: 'f32', access: 'read_write'},
        {name: 'normalized', view: normalizedChoice, type: 'f32', access: 'read_write'}
      ],
      body: `let reachable=metrics[metricsOffset+4u*index+1u];let total=metrics[metricsOffset+4u*index+2u];let size=reachable+1.0;
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));var mean=missing;var asymmetry=missing;var real=missing;var hillier=missing;
if(reachable>0.0){mean=total/reachable;}
if(size>2.0){asymmetry=2.0*(mean-1.0)/(size-2.0);let normalization=2.0*(size*(log2((size+2.0)/3.0)-1.0)+1.0)/((size-1.0)*(size-2.0));real=asymmetry/normalization;hillier=1.0/real;}
integration[integrationOffset+4u*index]=mean;integration[integrationOffset+4u*index+1u]=asymmetry;integration[integrationOffset+4u*index+2u]=real;integration[integrationOffset+4u*index+3u]=hillier;
let maximum=maximumDepth[maximumDepthOffset+index];var pyramid=missing;var teklenburg=missing;var penn=missing;
if(size>2.0&&mean>1.0){let normalization=2.0*(size-log2(size)-1.0)/((size-1.0)*(size-2.0));pyramid=normalization/asymmetry;if(total-size+1.0>1.0){teklenburg=log(0.5*(size-2.0))/log(total-size+1.0);}let radius=min(maximum,size-1.0);let greatest=radius*(size-0.5*(radius+1.0));let least=size-1.0;if(greatest>least){penn=(greatest-total)/(greatest-least);}}
extended[extendedOffset+4u*index]=pyramid;extended[extendedOffset+4u*index+1u]=teklenburg;extended[extendedOffset+4u*index+2u]=penn;extended[extendedOffset+4u*index+3u]=select(missing,maximum,reachable>=0.0);
normalized[normalizedOffset+index]=select(missing,2.0*choice[choiceOffset+index]/f32(${Math.max(1, (count - 1) * (count - 2))}),${count}>2);`
    })
  );
  return {
    outputs: {
      adjacency,
      edges,
      metrics,
      choice,
      integration,
      normalizedChoice,
      viewpointClassifications,
      extendedIntegration
    },
    status,
    provenance: {
      positions: props.positions,
      obstacles: props.segments,
      maximumDistance: props.maximumDistance,
      tolerance,
      region: props.region,
      integration: 'reachable-component-hillier-hanson' as const,
      choice: 'unnormalized-undirected-betweenness' as const
    }
  };
}
