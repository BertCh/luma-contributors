// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {reconcileLandscapePatchLabels} from './reconcile-patches';
import {assert} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import type {GPULandscapeMetricsProps} from './landscape-metrics';

export type GPUPatchProximityProps = Omit<
  GPULandscapeMetricsProps,
  | 'classCount'
  | 'connectivity'
  | 'countBoundary'
  | 'countBackground'
  | 'backgroundClass'
  | 'contagionWeighting'
> & {
  /** Core pixels have center-to-exterior distance strictly greater than this physical depth. */
  edgeDepth: number;
  searchRadius: number;
  /** Connectivity used when reconcilePatches is enabled. */
  connectivity?: 4 | 8;
  /** Rectangle separation by default; cell centers reproduce raster ENN conventions. */
  distanceModel?: 'cell-boundary' | 'cell-center';
  /** Physical depth by default; raster erosion uses an integer number of four-neighbor layers. */
  coreModel?: 'physical' | 'raster-erosion';
  /** Treat domain exterior as an edge (default), or leave it open. */
  coreBoundary?: 'edge' | 'open';
};

/** Exact rectangular-cell boundary distances on a complete raster; background is a core-area barrier. */
export function addPatchProximityRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUPatchProximityProps
) {
  const id = props.id ?? 'patch-proximity';
  const {width, height, cellSize, edgeDepth, searchRadius} = props;
  const cellCount = width * height;
  const patchCount = props.patchClasses.length;
  const reconciled = props.reconcilePatches
    ? reconcileLandscapePatchLabels(graph, {...props, id, connectivity: props.connectivity ?? 4})
    : undefined;
  const patches = reconciled?.patches ?? props.patches;
  assert(Number.isSafeInteger(width) && width > 0 && Number.isSafeInteger(height) && height > 0);
  assert(props.patches.length === cellCount && patchCount > 0);
  assert(cellSize.every(value => Number.isFinite(Math.fround(value)) && value > 0));
  assert(
    Number.isFinite(edgeDepth) &&
      edgeDepth >= 0 &&
      Number.isFinite(searchRadius) &&
      searchRadius >= 0
  );
  const rasterCore = props.coreModel === 'raster-erosion';
  assert(!rasterCore || Number.isSafeInteger(edgeDepth));
  const separation = props.distanceModel === 'cell-center' ? 0 : 1;
  const core = getRecipeView(graph, `${id}-core`, 'uint32', cellCount);
  const distances = getRecipeView(graph, `${id}-distances`, 'float32', patchCount * patchCount);
  const horizontal = getRecipeView(
    graph,
    `${id}-horizontal-distances`,
    'float32',
    patchCount * cellCount
  );
  const fields = getRecipeView(graph, `${id}-distance-fields`, 'float32', patchCount * cellCount);
  const metrics = getRecipeView(graph, `${id}-metrics`, 'float32x4', patchCount);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-core-distance`,
      operation: id,
      invocationCount: cellCount,
      bindings: [
        {name: 'patches', view: patches, type: 'u32', access: 'read'},
        {name: 'output', view: core, type: 'u32', access: 'read_write'}
      ],
      body: rasterCore
        ? `let owner=patches[patchesOffset+index];let column=i32(index%${width}u);let row=i32(index/${width}u);var valid=owner<${patchCount}u;
for(var offsetRow=-${Math.min(height, edgeDepth)};offsetRow<=${Math.min(height, edgeDepth)};offsetRow++){for(var offsetColumn=-${Math.min(width, edgeDepth)};offsetColumn<=${Math.min(width, edgeDepth)};offsetColumn++){
if(abs(offsetRow)+abs(offsetColumn)>${edgeDepth}){continue;}let otherColumn=column+offsetColumn;let otherRow=row+offsetRow;
if(otherColumn<0||otherColumn>=${width}||otherRow<0||otherRow>=${height}){${props.coreBoundary === 'open' ? '' : 'valid=false;'}continue;}
valid=valid&&patches[patchesOffset+u32(otherRow)*${width}u+u32(otherColumn)]==owner;}}
output[outputOffset+index]=u32(valid);`
        : `let owner=patches[patchesOffset+index];let position=vec2<f32>(f32(index%${width}u)+0.5,f32(index/${width}u)+0.5)*vec2<f32>(${cellSize[0]},${cellSize[1]});
var distance=${props.coreBoundary === 'open' ? 'bitcast<f32>(0x7f800000u | (index & 0u))' : `min(min(position.x,${width * cellSize[0]}-position.x),min(position.y,${height * cellSize[1]}-position.y))`};
let column=i32(index%${width}u);let row=i32(index/${width}u);
for(var offsetRow=-${Math.min(height, Math.ceil(edgeDepth / cellSize[1] + 0.5))};offsetRow<=${Math.min(height, Math.ceil(edgeDepth / cellSize[1] + 0.5))};offsetRow++){
for(var offsetColumn=-${Math.min(width, Math.ceil(edgeDepth / cellSize[0] + 0.5))};offsetColumn<=${Math.min(width, Math.ceil(edgeDepth / cellSize[0] + 0.5))};offsetColumn++){
 let otherColumn=column+offsetColumn;let otherRow=row+offsetRow;if(otherColumn<0||otherColumn>=${width}||otherRow<0||otherRow>=${height}){continue;}let cell=u32(otherRow)*${width}u+u32(otherColumn);
 if(patches[patchesOffset+cell]==owner){continue;}
 let other=vec2<f32>(f32(cell%${width}u)+0.5,f32(cell/${width}u)+0.5)*vec2<f32>(${cellSize[0]},${cellSize[1]});
 distance=min(distance,length(max(abs(position-other) - 0.5*vec2<f32>(${cellSize[0]},${cellSize[1]}),vec2<f32>(0.0))));
}
}
output[outputOffset+index]=u32(owner<${patchCount}u&&distance>${edgeDepth});`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-horizontal-transform`,
      operation: id,
      invocationCount: patchCount * cellCount,
      bindings: [
        {name: 'patches', view: patches, type: 'u32', access: 'read'},
        {name: 'output', view: horizontal, type: 'f32', access: 'read_write'}
      ],
      body: `let owner=index/${cellCount}u;let cell=index%${cellCount}u;let row=cell/${width}u;let column=cell%${width}u;var best=bitcast<f32>(0x7f800000u | (index & 0u));
for(var candidate=0u;candidate<${width}u;candidate++){if(patches[patchesOffset+row*${width}u+candidate]==owner){best=min(best,max(abs(f32(column)-f32(candidate))-${separation}.0,0.0)*${cellSize[0]});}}output[outputOffset+index]=best;`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-vertical-transform`,
      operation: id,
      invocationCount: patchCount * cellCount,
      bindings: [
        {name: 'horizontal', view: horizontal, type: 'f32', access: 'read'},
        {name: 'output', view: fields, type: 'f32', access: 'read_write'}
      ],
      body: `let owner=index/${cellCount}u;let cell=index%${cellCount}u;let row=cell/${width}u;let column=cell%${width}u;var best=bitcast<f32>(0x7f800000u | (index & 0u));
for(var candidate=0u;candidate<${height}u;candidate++){let horizontalDistance=horizontal[horizontalOffset+owner*${cellCount}u+candidate*${width}u+column];let verticalDistance=max(abs(f32(row)-f32(candidate))-${separation}.0,0.0)*${cellSize[1]};best=min(best,length(vec2<f32>(horizontalDistance,verticalDistance)));}output[outputOffset+index]=best;`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-boundary-distances`,
      operation: id,
      invocationCount: patchCount * patchCount,
      bindings: [
        {name: 'patches', view: patches, type: 'u32', access: 'read'},
        {name: 'fields', view: fields, type: 'f32', access: 'read'},
        {name: 'output', view: distances, type: 'f32', access: 'read_write'}
      ],
      body: `let source=index/${patchCount}u;let targetPatch=index%${patchCount}u;var distance=bitcast<f32>(0x7f800000u | (index & 0u));
if(source!=targetPatch){for(var cell=0u;cell<${cellCount}u;cell++){if(patches[patchesOffset+cell]==targetPatch){distance=min(distance,fields[fieldsOffset+source*${cellCount}u+cell]);}}}
output[outputOffset+index]=distance;`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-patch-summary`,
      operation: id,
      invocationCount: patchCount,
      declarations: `
fn readOutput(row:u32)->vec4<f32>{return vec4<f32>(output[outputOffset+row*4u+0u],output[outputOffset+row*4u+1u],output[outputOffset+row*4u+2u],output[outputOffset+row*4u+3u]);}
fn writeOutput(row:u32,value:vec4<f32>){output[outputOffset+row*4u+0u]=value[0u];output[outputOffset+row*4u+1u]=value[1u];output[outputOffset+row*4u+2u]=value[2u];output[outputOffset+row*4u+3u]=value[3u];}
`,
      bindings: [
        {name: 'patches', view: patches, type: 'u32', access: 'read'},
        {name: 'classes', view: props.patchClasses, type: 'u32', access: 'read'},
        {name: 'core', view: core, type: 'u32', access: 'read'},
        {name: 'distances', view: distances, type: 'f32', access: 'read'},
        {name: 'output', view: metrics, type: 'f32', access: 'read_write'}
      ],
      body: `var coreArea=0.0;var area=0.0;var nearest=bitcast<f32>(0x7f800000u | (index & 0u));var proximity=0.0;
for(var cell=0u;cell<${cellCount}u;cell++){if(patches[patchesOffset+cell]==index){area+=${cellSize[0] * cellSize[1]};coreArea+=f32(core[coreOffset+cell])*${cellSize[0] * cellSize[1]};}}
for(var other=0u;other<${patchCount}u;other++){
 if(other==index||classes[classesOffset+other]!=classes[classesOffset+index]){continue;}
 let distance=distances[distancesOffset+index*${patchCount}u+other];nearest=min(nearest,distance);
 if(distance<=${searchRadius}){
  var otherArea=0.0;for(var cell=0u;cell<${cellCount}u;cell++){otherArea+=f32(patches[patchesOffset+cell]==other)*${cellSize[0] * cellSize[1]};}
  proximity+=otherArea/(distance*distance);
 }
}
${props.distanceModel === 'cell-center' ? 'if(nearest==bitcast<f32>(0x7f800000u | (index & 0u))){nearest=bitcast<f32>(0x7fc00000u | (index & 0u));}' : ''}
writeOutput(index,vec4<f32>(area,coreArea,nearest,proximity));`
    })
  );
  return {
    outputs: {core, distances, metrics},
    intermediates: {horizontal, fields},
    provenance: {
      patches: props.patches,
      reconciledPatches: reconciled?.patches,
      patchParents: reconciled?.parents,
      patchClasses: props.patchClasses,
      edgeDepth,
      searchRadius,
      cellSize,
      completeDomain: true as const,
      distanceAlgorithm: 'exact-separable-distance-transform' as const,
      distanceModel: props.distanceModel ?? 'cell-boundary',
      coreModel: props.coreModel ?? 'physical',
      coreBoundary: props.coreBoundary ?? 'edge'
    }
  };
}
