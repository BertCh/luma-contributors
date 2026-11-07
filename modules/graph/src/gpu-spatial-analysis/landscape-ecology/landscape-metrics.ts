// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {reconcileLandscapePatchLabels} from './reconcile-patches';
import {assert} from '@luma.gl/core';
import type {GPUCommandGraph, GraphDataView} from '@luma-contributors/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';

export type GPULandscapeMetricsProps = {
  id?: string;
  /** Complete domain, with stable connected-patch IDs; IDs >= patchClasses.length are background. */
  patches: GraphDataView<'uint32'>;
  patchClasses: GraphDataView<'uint32'>;
  width: number;
  height: number;
  classCount: number;
  /** Positive planar cell width and height. Class adjacency uses four orthogonal sides. */
  cellSize: readonly [number, number];
  /** Connectivity used upstream to label patches; retained as provenance. */
  connectivity: 4 | 8;
  /** Merge adjacent same-class tile IDs on the complete domain before summary. */
  reconcilePatches?: boolean;
  /** Count contacts with the outer domain in patch perimeter (default true). */
  countBoundary?: boolean;
  /** Count contacts with background/excluded classes in perimeter (default true). */
  countBackground?: boolean;
  /** An optional otherwise-valid class excluded from area and adjacency. */
  backgroundClass?: number;
  /** Area-weighted conditional adjacency (default), or landscapemetrics 2.2.1 joint adjacency. */
  contagionWeighting?: 'area' | 'adjacency';
};

/** Named class/landscape metrics from a complete, already labelled categorical raster. */
export function addLandscapeMetricsRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPULandscapeMetricsProps
) {
  const id = props.id ?? 'landscape-metrics';
  const {width, height, classCount, cellSize} = props;
  const cellCount = width * height;
  const patchCount = props.patchClasses.length;
  assert(Number.isSafeInteger(width) && width > 0 && Number.isSafeInteger(height) && height > 0);
  assert(props.patches.length === cellCount && patchCount > 0);
  assert(Number.isSafeInteger(classCount) && classCount > 0 && classCount <= 256);
  assert(cellSize.every(value => Number.isFinite(Math.fround(value)) && value > 0));
  assert(props.connectivity === 4 || props.connectivity === 8);
  // Validation precedes reconcileLandscapePatchLabels, which adds nodes to the graph.
  const reconciled = props.reconcilePatches
    ? reconcileLandscapePatchLabels(graph, {...props, id})
    : undefined;
  const patches = reconciled?.patches ?? props.patches;
  const patchMetrics = getRecipeView(graph, `${id}-patches`, 'float32x4', patchCount);
  const classMetrics = getRecipeView(graph, `${id}-classes`, 'float32x4', classCount);
  const classAdjacency = getRecipeView(graph, `${id}-adjacency`, 'uint32', classCount * classCount);
  const aggregation = getRecipeView(graph, `${id}-aggregation`, 'float32', classCount);
  const landscape = getRecipeView(graph, `${id}-landscape`, 'float32x4', 1);
  const landscapeAggregation = getRecipeView(graph, `${id}-landscape-aggregation`, 'float32', 1);
  const landscapeCohesion = getRecipeView(graph, `${id}-landscape-cohesion`, 'float32', 1);
  const declarations = `
fn getPatch(column: i32, row: i32) -> u32 {
  if(column<0 || row<0 || column>=${width} || row>=${height}){return 0xffffffffu;}
  return patches[patchesOffset+u32(row)*${width}u+u32(column)];
}
fn getClass(patchIndex: u32) -> u32 {
  if(patchIndex>=${patchCount}u){return 0xffffffffu;}
  let category=classes[classesOffset+patchIndex];
  return select(category,0xffffffffu,category==${props.backgroundClass ?? 0xffffffff}u);
}`;
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-patch-summary`,
      operation: id,
      invocationCount: patchCount,
      declarations:
        declarations +
        `
fn readOutput(row:u32)->vec4<f32>{return vec4<f32>(output[outputOffset+row*4u+0u],output[outputOffset+row*4u+1u],output[outputOffset+row*4u+2u],output[outputOffset+row*4u+3u]);}
fn writeOutput(row:u32,value:vec4<f32>){output[outputOffset+row*4u+0u]=value[0u];output[outputOffset+row*4u+1u]=value[1u];output[outputOffset+row*4u+2u]=value[2u];output[outputOffset+row*4u+3u]=value[3u];}
`,
      bindings: [
        {name: 'patches', view: patches, type: 'u32', access: 'read'},
        {name: 'classes', view: props.patchClasses, type: 'u32', access: 'read'},
        {name: 'output', view: patchMetrics, type: 'f32', access: 'read_write'}
      ],
      body: `var cells=0.0;var perimeter=0.0;var likeSides=0.0;
for(var cell=0u;cell<${cellCount}u;cell++){
 if(patches[patchesOffset+cell]!=index){continue;}
 cells+=1.0;let column=i32(cell%${width}u);let row=i32(cell/${width}u);
 let neighbors=array<u32,4>(getPatch(column - 1,row),getPatch(column+1,row),getPatch(column,row - 1),getPatch(column,row+1));
 for(var direction=0u;direction<4u;direction++){
  let exterior=(direction==0u&&column==0)||(direction==1u&&column==${width - 1})||(direction==2u&&row==0)||(direction==3u&&row==${height - 1});
  let background=getClass(neighbors[direction])>=${classCount}u;
  let countContact=select(select(true,${props.countBackground !== false},background),${props.countBoundary !== false},exterior);
  if(neighbors[direction]!=index&&countContact){perimeter+=select(f32(${cellSize[0]}),f32(${cellSize[1]}),direction<2u);}
  likeSides+=f32(getClass(neighbors[direction])==getClass(index));
 }
}
writeOutput(index,vec4<f32>(cells*${cellSize[0] * cellSize[1]},perimeter,cells,likeSides));`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-class-adjacency`,
      operation: id,
      invocationCount: classCount * classCount,
      declarations,
      bindings: [
        {name: 'patches', view: patches, type: 'u32', access: 'read'},
        {name: 'classes', view: props.patchClasses, type: 'u32', access: 'read'},
        {name: 'output', view: classAdjacency, type: 'u32', access: 'read_write'}
      ],
      body: `let sourceClass=index/${classCount}u;let targetClass=index%${classCount}u;var count=0u;
for(var cell=0u;cell<${cellCount}u;cell++){
 if(getClass(patches[patchesOffset+cell])!=sourceClass){continue;}
 let column=i32(cell%${width}u);let row=i32(cell/${width}u);
 count+=u32(getClass(getPatch(column - 1,row))==targetClass)+u32(getClass(getPatch(column+1,row))==targetClass)+u32(getClass(getPatch(column,row - 1))==targetClass)+u32(getClass(getPatch(column,row+1))==targetClass);
}output[outputOffset+index]=count;`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-class-summary`,
      operation: id,
      invocationCount: classCount,
      declarations: `
fn readPatches(row:u32)->vec4<f32>{return vec4<f32>(patches[patchesOffset+row*4u+0u],patches[patchesOffset+row*4u+1u],patches[patchesOffset+row*4u+2u],patches[patchesOffset+row*4u+3u]);}
fn readOutput(row:u32)->vec4<f32>{return vec4<f32>(output[outputOffset+row*4u+0u],output[outputOffset+row*4u+1u],output[outputOffset+row*4u+2u],output[outputOffset+row*4u+3u]);}
fn writeOutput(row:u32,value:vec4<f32>){output[outputOffset+row*4u+0u]=value[0u];output[outputOffset+row*4u+1u]=value[1u];output[outputOffset+row*4u+2u]=value[2u];output[outputOffset+row*4u+3u]=value[3u];}
`,
      bindings: [
        {name: 'patches', view: patchMetrics, type: 'f32', access: 'read'},
        {name: 'classes', view: props.patchClasses, type: 'u32', access: 'read'},
        {name: 'output', view: classMetrics, type: 'f32', access: 'read_write'},
        {name: 'aggregation', view: aggregation, type: 'f32', access: 'read_write'}
      ],
      body: `var area=0.0;var perimeter=0.0;var squaredArea=0.0;var weightedPerimeter=0.0;var cells=0.0;var likeSides=0.0;var totalCells=0.0;
for(var patchIndex=0u;patchIndex<${patchCount}u;patchIndex++){
 let values=readPatches(patchIndex);if(classes[classesOffset+patchIndex]>=${classCount}u||classes[classesOffset+patchIndex]==${props.backgroundClass ?? 0xffffffff}u){continue;}
 totalCells+=values.z;
 if(classes[classesOffset+patchIndex]==index){area+=values.x;perimeter+=values.y;squaredArea+=values.x*values.x;weightedPerimeter+=values.y*sqrt(values.z);cells+=values.z;likeSides+=values.w;}
}
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));var cohesion=missing;var mesh=missing;
if(totalCells>0.0){mesh=squaredArea/(totalCells*${cellSize[0] * cellSize[1]});}
if(weightedPerimeter>0.0&&totalCells>1.0){cohesion=100.0*(1.0-perimeter/weightedPerimeter)/(1.0-inverseSqrt(totalCells));}
let side=floor(sqrt(cells));let remainder=cells-side*side;
let maximum=2.0*cells - 2.0*side-select(0.0,select(2.0,1.0,remainder<=side),remainder>0.0);
var aggregate=missing;if(maximum>0.0){aggregate=100.0*likeSides/(2.0*maximum);}
writeOutput(index,vec4<f32>(area,perimeter,mesh,cohesion));aggregation[aggregationOffset+index]=aggregate;`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-landscape-summary`,
      operation: id,
      invocationCount: 1,
      declarations: `
fn readClasses(row:u32)->vec4<f32>{return vec4<f32>(classes[classesOffset+row*4u+0u],classes[classesOffset+row*4u+1u],classes[classesOffset+row*4u+2u],classes[classesOffset+row*4u+3u]);}
fn readOutput(row:u32)->vec4<f32>{return vec4<f32>(output[outputOffset+row*4u+0u],output[outputOffset+row*4u+1u],output[outputOffset+row*4u+2u],output[outputOffset+row*4u+3u]);}
fn writeOutput(row:u32,value:vec4<f32>){output[outputOffset+row*4u+0u]=value[0u];output[outputOffset+row*4u+1u]=value[1u];output[outputOffset+row*4u+2u]=value[2u];output[outputOffset+row*4u+3u]=value[3u];}
`,
      bindings: [
        {name: 'classes', view: classMetrics, type: 'f32', access: 'read'},
        {name: 'adjacency', view: classAdjacency, type: 'u32', access: 'read'},
        {name: 'output', view: landscape, type: 'f32', access: 'read_write'}
      ],
      body: `var area=0.0;var mesh=0.0;var present=0.0;
for(var category=0u;category<${classCount}u;category++){let values=readClasses(category);area+=values.x;mesh+=values.z;present+=f32(values.x>0.0);}
var entropy=0.0;var adjacencyComplete=true;var totalAdjacency=0.0;for(var pair=0u;pair<${classCount * classCount}u;pair++){totalAdjacency+=f32(adjacency[adjacencyOffset+pair]);}
for(var sourceClass=0u;sourceClass<${classCount}u;sourceClass++){
 var total=0.0;for(var targetClass=0u;targetClass<${classCount}u;targetClass++){total+=f32(adjacency[adjacencyOffset+sourceClass*${classCount}u+targetClass]);}
 let proportion=readClasses(sourceClass).x/area;
 if(proportion>0.0&&total==0.0){adjacencyComplete=false;}
 for(var targetClass=0u;targetClass<${classCount}u;targetClass++){let probability=${props.contagionWeighting === 'adjacency' ? 'f32(adjacency[adjacencyOffset+sourceClass*' + classCount + 'u+targetClass])/totalAdjacency' : 'proportion*f32(adjacency[adjacencyOffset+sourceClass*' + classCount + 'u+targetClass])/total'};if(probability>0.0){entropy+=probability*log(probability);}}
}
var contagion=bitcast<f32>(0x7fc00000u | (index & 0u));if(present>1.0&&adjacencyComplete){contagion=100.0*(1.0+entropy/(2.0*log(present)));}
writeOutput(0u,vec4<f32>(contagion,mesh,area,present));`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-landscape-aggregation-cohesion`,
      operation: id,
      invocationCount: 1,
      bindings: [
        {name: 'patches', view: patchMetrics, type: 'f32', access: 'read'},
        {name: 'patchClasses', view: props.patchClasses, type: 'u32', access: 'read'},
        {name: 'classes', view: classMetrics, type: 'f32', access: 'read'},
        {name: 'aggregation', view: aggregation, type: 'f32', access: 'read'},
        {
          name: 'landscapeAggregation',
          view: landscapeAggregation,
          type: 'f32',
          access: 'read_write'
        },
        {name: 'landscapeCohesion', view: landscapeCohesion, type: 'f32', access: 'read_write'}
      ],
      body: `var cells=0.0;var perimeter=0.0;var weightedPerimeter=0.0;
for(var patchIndex=0u;patchIndex<${patchCount}u;patchIndex++){
 let category=patchClasses[patchClassesOffset+patchIndex];if(category>=${classCount}u||category==${props.backgroundClass ?? 0xffffffff}u){continue;}
 let patchCells=patches[patchesOffset+4u*patchIndex+2u];let patchPerimeter=patches[patchesOffset+4u*patchIndex+1u];cells+=patchCells;perimeter+=patchPerimeter;weightedPerimeter+=patchPerimeter*sqrt(patchCells);
}
var weightedAggregation=0.0;var area=0.0;
for(var category=0u;category<${classCount}u;category++){
 let classArea=classes[classesOffset+4u*category];let aggregate=aggregation[aggregationOffset+category];area+=classArea;
 // The pinned landscapemetrics landscape convention omits undefined single-cell class indices.
 if(classArea>0.0&&aggregate>=0.0){weightedAggregation+=classArea*aggregate;}
}
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));var aggregate=missing;var cohesion=missing;
if(area>0.0){aggregate=weightedAggregation/area;}
if(cells>1.0&&weightedPerimeter>0.0){cohesion=100.0*(1.0-perimeter/weightedPerimeter)/(1.0-inverseSqrt(cells));}
landscapeAggregation[landscapeAggregationOffset]=aggregate;landscapeCohesion[landscapeCohesionOffset]=cohesion;`
    })
  );
  return {
    outputs: {
      patchMetrics,
      classMetrics,
      classAdjacency,
      aggregation,
      landscape,
      landscapeAggregation,
      landscapeCohesion
    },
    provenance: {
      patches: props.patches,
      reconciledPatches: reconciled?.patches,
      patchParents: reconciled?.parents,
      patchClasses: props.patchClasses,
      connectivity: props.connectivity,
      adjacency: 4 as const,
      cellSize,
      countBoundary: props.countBoundary !== false,
      countBackground: props.countBackground !== false,
      backgroundClass: props.backgroundClass,
      contagionWeighting: props.contagionWeighting ?? 'area'
    }
  };
}
