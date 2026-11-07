// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {measureBuildingStructures, type GPUBuildingStructureGeometry} from './building-structures';
import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma-contributors/graph/gpu-core';
import type {GPUSharedPathsOutput} from '../line-clip/gpu-shared-paths';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';

export type GPUBuildingAdjacencyProps = {
  id?: string;
  /** Existing GPUSharedPaths result between building boundary lines; each unordered pair supplied once. */
  sharedPaths: GPUSharedPathsOutput;
  /** Maps the common boundary-line row order to stable building rows. */
  lineBuildingIds: GraphDataView<'uint32'>;
  /** Axial radians, typically from the existing shape descriptor. NaN preserves missing orientation. */
  orientations: GraphDataView<'float32'>;
  /** Optional explicit Queen/custom neighborhood; absent means positive shared walls. */
  contiguity?: GraphDataView<'uint32x2'>;
  /** Optional true union/exterior-perimeter analysis. Buffering is explicit upstream geometry work. */
  structures?: GPUBuildingStructureGeometry;
};

/** Shared-wall adjacency and axial alignment from the existing shared-path topology contributor. */
export function addBuildingAdjacencyRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUBuildingAdjacencyProps
) {
  const id = props.id ?? 'building-adjacency';
  const count = props.orientations.length;
  const {runs, status} = props.sharedPaths;
  assert(count > 0 && count <= 1024 && props.lineBuildingIds.length > 0);
  assert(
    runs.leftLineIds.length === runs.rightLineIds.length &&
      runs.offsets.length === runs.leftLineIds.length + 1
  );
  const lengths = getRecipeView(graph, `${id}-run-lengths`, 'float32', runs.leftLineIds.length);
  const pairs = getRecipeView(graph, `${id}-pairs`, 'float32x4', count * count);
  const complete = getRecipeView(graph, `${id}-complete`, 'uint32', 1);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-evidence`,
      operation: id,
      invocationCount: 1,
      bindings: [
        {name: 'count', view: status.count, type: 'u32', access: 'read'},
        ...(status.overflow
          ? [
              {
                name: 'overflow',
                view: status.overflow,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        ...(status.uncertainCount
          ? [
              {
                name: 'uncertain',
                view: status.uncertainCount,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'complete', view: complete, type: 'u32', access: 'read_write'}
      ],
      body: `complete[completeOffset]=u32(count[countOffset]<=${runs.leftLineIds.length}u${status.overflow ? '&&overflow[overflowOffset]==0u' : ''}${status.uncertainCount ? '&&uncertain[uncertainOffset]==0u' : ''});`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-measure-paths`,
      operation: id,
      invocationCount: runs.leftLineIds.length,
      bindings: [
        {name: 'positions', view: runs.positions, type: 'f32', access: 'read'},
        {name: 'offsets', view: runs.offsets, type: 'u32', access: 'read'},
        {name: 'count', view: status.count, type: 'u32', access: 'read'},
        {name: 'output', view: lengths, type: 'f32', access: 'read_write'}
      ],
      body: `var length=0.0;if(index<count[countOffset]){let start=offsets[offsetsOffset+index];let end=offsets[offsetsOffset+index+1u];if(start<=end&&end<=${runs.positions.length}u){for(var vertex=start+1u;vertex<end;vertex++){let first=positionsOffset+2u*(vertex - 1u);let second=positionsOffset+2u*vertex;length+=distance(vec2<f32>(positions[first],positions[first+1u]),vec2<f32>(positions[second],positions[second+1u]));}}else{length=bitcast<f32>(0x7fc00000u | (index & 0u));}}output[outputOffset+index]=length;`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-pairs`,
      operation: id,
      invocationCount: count * count,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'left', view: runs.leftLineIds, type: 'u32', access: 'read'},
        {name: 'right', view: runs.rightLineIds, type: 'u32', access: 'read'},
        {name: 'owners', view: props.lineBuildingIds, type: 'u32', access: 'read'},
        {name: 'lengths', view: lengths, type: 'f32', access: 'read'},
        {name: 'orientations', view: props.orientations, type: 'f32', access: 'read'},
        {name: 'complete', view: complete, type: 'u32', access: 'read'},
        {name: 'output', view: pairs, type: 'f32', access: 'read_write'}
      ],
      body: `let first=index/${count}u;let second=index%${count}u;var sharedLength=0.0;var valid=complete[completeOffset]!=0u;
for(var run=0u;run<${runs.leftLineIds.length}u;run++){
 let leftLine=left[leftOffset+run];let rightLine=right[rightOffset+run];if(leftLine==0xffffffffu&&rightLine==0xffffffffu){continue;}
 if(leftLine>=${props.lineBuildingIds.length}u||rightLine>=${props.lineBuildingIds.length}u){valid=false;continue;}
 let leftOwner=owners[ownersOffset+leftLine];let rightOwner=owners[ownersOffset+rightLine];valid=valid&&leftOwner<${count}u&&rightOwner<${count}u&&isFiniteValue(lengths[lengthsOffset+run]);
 if(first!=second&&((leftOwner==first&&rightOwner==second)||(leftOwner==second&&rightOwner==first))){sharedLength+=lengths[lengthsOffset+run];}
}
let firstAngle=orientations[orientationsOffset+first];let secondAngle=orientations[orientationsOffset+second];let difference=abs(firstAngle-secondAngle);let wrapped=difference-3.141592653589793*floor(difference/3.141592653589793);let angle=min(wrapped,3.141592653589793-wrapped);let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
output[outputOffset+4u*index]=select(missing,sharedLength,valid);output[outputOffset+4u*index+1u]=select(missing,angle,valid&&isFiniteValue(firstAngle)&&isFiniteValue(secondAngle));output[outputOffset+4u*index+2u]=select(missing,f32(sharedLength>0.0),valid);output[outputOffset+4u*index+3u]=f32(valid);`
    })
  );
  const components = getRecipeView(graph, `${id}-components`, 'uint32', count);
  const morphology = getRecipeView(graph, `${id}-morphology`, 'float32x4', count);
  const neighbors = getRecipeView(graph, `${id}-neighbors`, 'uint32', count * count);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-neighborhood`,
      operation: id,
      invocationCount: count * count,
      bindings: [
        {name: 'pairs', view: pairs, type: 'f32', access: 'read'},
        ...(props.contiguity
          ? [
              {
                name: 'contiguity',
                view: props.contiguity,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'neighbors', view: neighbors, type: 'u32', access: 'read_write'}
      ],
      body: `let first=index/${count}u;let second=index%${count}u;var adjacent=pairs[pairsOffset+4u*index+2u]>0.0;var neighborhoodValid=pairs[pairsOffset+4u*index+3u]!=0.0;${props.contiguity ? `adjacent=false;for(var edge=0u;edge<${props.contiguity.length}u;edge++){let left=contiguity[contiguityOffset+2u*edge];let right=contiguity[contiguityOffset+2u*edge+1u];neighborhoodValid=neighborhoodValid&&left<${count}u&&right<${count}u;adjacent=adjacent||((left==first&&right==second)||(left==second&&right==first));}` : ''}neighbors[neighborsOffset+index]=select(0xffffffffu,u32(first!=second&&adjacent),neighborhoodValid);`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-components`,
      operation: id,
      invocationCount: 1,
      workgroupSize: 2,
      bindings: [
        {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
        {name: 'components', view: components, type: 'u32', access: 'read_write'}
      ],
      body: `if(neighbors[neighborsOffset]==0xffffffffu){for(var vertex=0u;vertex<${count}u;vertex++){components[componentsOffset+vertex]=0xffffffffu;}return;}for(var vertex=0u;vertex<${count}u;vertex++){components[componentsOffset+vertex]=vertex;}for(var first=0u;first<${count}u;first++){for(var second=first+1u;second<${count}u;second++){if(neighbors[neighborsOffset+first*${count}u+second]==0u){continue;}var left=first;var right=second;loop{let parent=components[componentsOffset+left];if(parent==left){break;}left=parent;}loop{let parent=components[componentsOffset+right];if(parent==right){break;}right=parent;}components[componentsOffset+max(left,right)]=min(left,right);}}for(var vertex=0u;vertex<${count}u;vertex++){var root=vertex;loop{let parent=components[componentsOffset+root];if(root==parent){break;}root=parent;}components[componentsOffset+vertex]=root;}`
    })
  );
  const structures = props.structures
    ? measureBuildingStructures(graph, id, components, props.structures)
    : undefined;
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-cardinal-alignment`,
      operation: id,
      invocationCount: count,
      declarations:
        IS_FINITE_VALUE_WGSL +
        `
fn cardinal(angle:f32)->f32{let quadrant=angle-1.5707963267948966*floor(angle/1.5707963267948966);return min(quadrant,1.5707963267948966-quadrant)*57.29577951308232;}
`,
      bindings: [
        {name: 'orientations', view: props.orientations, type: 'f32', access: 'read'},
        {name: 'pairs', view: pairs, type: 'f32', access: 'read'},
        {name: 'neighbors', view: neighbors, type: 'u32', access: 'read'},
        {name: 'complete', view: complete, type: 'u32', access: 'read'},
        ...(structures
          ? [
              {
                name: 'perimeters',
                view: structures.perimeters,
                type: 'f32' as const,
                access: 'read' as const
              }
            ]
          : []),
        {name: 'output', view: morphology, type: 'f32', access: 'read_write'}
      ],
      body: `let orientation=orientations[orientationsOffset+index];let own=cardinal(orientation);var sharedLength=0.0;var sum=0.0;var neighborCount=0.0;var neighborhoodValid=neighbors[neighborsOffset+index*${count}u+index]!=0xffffffffu;var valid=complete[completeOffset]!=0u&&isFiniteValue(orientation)&&neighborhoodValid;for(var other=0u;other<${count}u;other++){sharedLength+=pairs[pairsOffset+4u*(index*${count}u+other)];if(neighbors[neighborsOffset+index*${count}u+other]==1u){let angle=orientations[orientationsOffset+other];valid=valid&&isFiniteValue(angle);sum+=abs(own-cardinal(angle));neighborCount+=1.0;}}let missing=bitcast<f32>(0x7fc00000u | (index & 0u));output[outputOffset+4u*index]=select(missing,sharedLength,complete[completeOffset]!=0u);output[outputOffset+4u*index+1u]=select(missing,own,isFiniteValue(orientation));output[outputOffset+4u*index+2u]=select(missing,sum/max(1.0,neighborCount),valid);output[outputOffset+4u*index+3u]=${structures ? 'select(missing,perimeters[perimetersOffset+index],complete[completeOffset]!=0u&&neighborhoodValid)' : 'missing'};`
    })
  );
  return {
    outputs: {pairs, lengths, morphology, components},
    structures,
    status: {complete},
    provenance: {
      sharedPaths: props.sharedPaths,
      lineBuildingIds: props.lineBuildingIds,
      orientations: props.orientations,
      orientationPeriod: Math.PI,
      cardinalOrientationUnits: 'degrees' as const,
      contiguity: props.contiguity,
      structures: props.structures
    }
  };
}
