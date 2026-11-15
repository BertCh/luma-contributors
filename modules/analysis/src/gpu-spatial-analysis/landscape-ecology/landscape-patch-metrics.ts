// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {GPURasterPatchMetrics} from '../../raster-analysis/raster-patches/gpu-raster-patch-metrics';
import {
  getPatchRunSegmentsBody,
  getPatchSegmentCount
} from '../../raster-analysis/raster-patches/patch-labels';
import {createFillNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {createRecipeKernelNode} from '../recipes/expansion-recipe-utils';
import type {GPULandscapeMetricsProps} from './landscape-metrics';

/** Raster measurements and integer class histograms, without patch-by-cell tables. @internal */
export function addLandscapePatchMetrics<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPULandscapeMetricsProps & {
    id: string;
    patchMetrics: GraphDataView<'float32x4'>;
    classAdjacency: GraphDataView<'uint32'>;
  }
) {
  const {id, width, height, classCount, cellSize, patches, patchClasses} = props;
  const patchCount = patchClasses.length;
  const labels = createTransientView(graph, `${id}-dense-labels`, 'uint32', width * height);
  const pixelCounts = createTransientView(graph, `${id}-pixel-counts`, 'uint32', patchCount);
  const perimeters = createTransientView(graph, `${id}-perimeters`, 'float32', patchCount);
  const patchLikeSides = createTransientView(graph, `${id}-patch-like-sides`, 'uint32', patchCount);
  const classCells = createTransientView(graph, `${id}-class-cells`, 'uint32', classCount);
  const classLikeSides = createTransientView(graph, `${id}-class-like-sides`, 'uint32', classCount);
  const perimeterSides =
    props.countBackground === false
      ? createTransientView(graph, `${id}-perimeter-sides`, 'uint32x2', patchCount)
      : undefined;
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-dense-labels`,
      operation: id,
      invocationCount: width * height,
      bindings: [
        {name: 'patches', view: patches, type: 'u32', access: 'read'},
        {name: 'labels', view: labels, type: 'u32', access: 'read_write'}
      ],
      body: `let owner=patches[patchesOffset+index];
labels[labelsOffset+index]=select(0u,owner+1u,owner<${patchCount}u);`
    })
  );
  graph.add(
    new GPURasterPatchMetrics({
      id: `${id}-raster-patches`,
      grid: {
        width,
        height,
        pixelInterpretation: 'area',
        affine: [cellSize[0], 0, 0, 0, cellSize[1], 0]
      },
      labels,
      countRasterBorder: props.countBoundary,
      output: {pixelCounts, ...(perimeterSides ? {} : {perimeters})}
    }).getCommandNodes(graph)
  );
  for (const [name, view] of [
    ['patch-like-sides', patchLikeSides],
    ['class-cells', classCells],
    ['class-like-sides', classLikeSides],
    ['class-adjacency', props.classAdjacency],
    ...(perimeterSides ? [['perimeter-sides', perimeterSides] as const] : [])
  ] as const) {
    graph.add(
      createFillNode(graph, {
        id: `${id}-${name}-clear`,
        operation: id,
        view,
        type: 'u32',
        value: '0u'
      })
    );
  }
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-raster-side-counts`,
      operation: id,
      invocationCount: getPatchSegmentCount(width, height),
      bindings: [
        {name: 'keys', view: labels, type: 'u32', access: 'read'},
        {name: 'classes', view: patchClasses, type: 'u32', access: 'read'},
        {name: 'patchLikeSides', view: patchLikeSides, type: 'atomic<u32>', access: 'read_write'},
        {name: 'classCells', view: classCells, type: 'atomic<u32>', access: 'read_write'},
        {name: 'classLikeSides', view: classLikeSides, type: 'atomic<u32>', access: 'read_write'},
        {name: 'adjacency', view: props.classAdjacency, type: 'atomic<u32>', access: 'read_write'},
        ...(perimeterSides
          ? [
              {
                name: 'perimeterSides',
                view: perimeterSides,
                type: 'atomic<u32>' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      declarations: `const WIDTH:u32=${width}u;
fn getClass(key:u32)->u32 {
  if(key==0u){return 0xffffffffu;}
  let category=classes[classesOffset+key-1u];
  return select(category,0xffffffffu,category==${props.backgroundClass ?? 0xffffffff}u);
}`,
      // Reuse bounded raster runs (at most 16 cells) to fold same-class atomics locally. Off-diagonal
      // contacts are counted once from each side, just like the public directed adjacency table.
      body: getPatchRunSegmentsBody(width, {
        declareRunState: 'var likeSides=0u;var sides=vec2<u32>(0u);',
        onRunStart: 'likeSides=0u;sides=vec2<u32>(0u);',
        onPixel: `let category=getClass(key);
let directions=array<vec2<i32>,4>(vec2<i32>(-1,0),vec2<i32>(1,0),vec2<i32>(0,-1),vec2<i32>(0,1));
for(var direction=0u;direction<4u;direction++){
  let adjacent=vec2<i32>(i32(column),i32(row))+directions[direction];
  let exterior=any(adjacent < vec2<i32>(0))||adjacent.x>=${width}||adjacent.y>=${height};
  var neighbor=0u;
  if(!exterior){neighbor=keys[keysOffset+u32(adjacent.y)*WIDTH+u32(adjacent.x)];}
  let neighborClass=getClass(neighbor);
  likeSides+=u32(neighborClass==category);
  if(category<${classCount}u&&neighborClass<${classCount}u&&neighborClass!=category){
    atomicAdd(&adjacency[adjacencyOffset+category*${classCount}u+neighborClass],1u);
  }
  ${
    perimeterSides
      ? `let countContact=select(neighborClass<${classCount}u,${props.countBoundary !== false},exterior);
  if(neighbor!=key&&countContact){sides[select(1u,0u,direction<2u)]+=1u;}`
      : ''
  }
}`,
        onRun: `let owner=runKey-1u;
atomicAdd(&patchLikeSides[patchLikeSidesOffset+owner],likeSides);
let category=getClass(runKey);
if(category<${classCount}u){
  atomicAdd(&classCells[classCellsOffset+category],runEnd-runStart);
  atomicAdd(&classLikeSides[classLikeSidesOffset+category],likeSides);
  atomicAdd(&adjacency[adjacencyOffset+category*${classCount}u+category],likeSides);
}
${
  perimeterSides
    ? `atomicAdd(&perimeterSides[perimeterSidesOffset+2u*owner],sides.x);
atomicAdd(&perimeterSides[perimeterSidesOffset+2u*owner+1u],sides.y);`
    : ''
}`
      })
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-patch-summary`,
      operation: id,
      invocationCount: patchCount,
      bindings: [
        {name: 'pixelCounts', view: pixelCounts, type: 'u32', access: 'read'},
        {name: 'patchLikeSides', view: patchLikeSides, type: 'u32', access: 'read'},
        ...(perimeterSides
          ? [
              {
                name: 'perimeterSides',
                view: perimeterSides,
                type: 'u32' as const,
                access: 'read' as const
              }
            ]
          : [
              {name: 'perimeters', view: perimeters, type: 'f32' as const, access: 'read' as const}
            ]),
        {name: 'output', view: props.patchMetrics, type: 'f32', access: 'read_write'}
      ],
      body: `let cells=pixelCounts[pixelCountsOffset+index];
let perimeter=${perimeterSides ? `f32(perimeterSides[perimeterSidesOffset+2u*index])*${cellSize[1]}+f32(perimeterSides[perimeterSidesOffset+2u*index+1u])*${cellSize[0]}` : 'perimeters[perimetersOffset+index]'};
output[outputOffset+4u*index]=f32(cells)*${cellSize[0] * cellSize[1]};
output[outputOffset+4u*index+1u]=perimeter;
output[outputOffset+4u*index+2u]=f32(cells);
output[outputOffset+4u*index+3u]=f32(patchLikeSides[patchLikeSidesOffset+index]);`
    })
  );
  return {classCells, classLikeSides};
}
