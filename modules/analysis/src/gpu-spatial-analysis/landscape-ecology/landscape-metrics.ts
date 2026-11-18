// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {reconcileLandscapePatchLabels} from './reconcile-patches';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {addLandscapePatchMetrics} from './landscape-patch-metrics';
import {getSortedSegmentSumNodes} from '../../analysis-core/kernels/sorted-segment-sums';
import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createRecipeKernelNode,
  getRecipeView,
  validateRecipeBoundary
} from '../recipes/expansion-recipe-utils';
import {
  validateCapacity,
  validateMaximumCount,
  validateGPUStorageBindingByteLength,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/**
 * Most classes: the class adjacency table holds classes squared counts and the landscape summary
 * walks all of them with one thread, so the table stays at most 65,536 entries.
 */
export const LANDSCAPE_METRICS_MAXIMUM_CLASS_COUNT = 256;

/** Four directed sides per cell must fit an exact uint32 histogram counter. */
const LANDSCAPE_METRICS_MAXIMUM_CELL_COUNT = Math.floor(0xffffffff / 4);

export type GPULandscapeMetricsOutput = {
  /** `float32x4`, one row per patch: area, perimeter, cell count, like-adjacent sides. */
  patchMetrics: GraphDataView<'float32x4'>;
  /** `float32x4`, one row per class: area, perimeter, mesh, cohesion. */
  classMetrics: GraphDataView<'float32x4'>;
  /** `uint32`, `classCount` squared rows: orthogonal side counts from class row to class column. */
  classAdjacency: GraphDataView<'uint32'>;
  /** `float32`, one row per class: aggregation index. */
  aggregation: GraphDataView<'float32'>;
  /** `float32x4`, one row: contagion, mesh, area, class richness. */
  landscape: GraphDataView<'float32x4'>;
  /** `float32`, one row: landscape aggregation index. */
  landscapeAggregation: GraphDataView<'float32'>;
  /** `float32`, one row: landscape cohesion. */
  landscapeCohesion: GraphDataView<'float32'>;
  /** `uint32`, one row per cell, only with `reconcilePatches`: merged patch label. */
  reconciledPatches: GraphDataView<'uint32'>;
  /** `uint32`, one row per patch, only with `reconcilePatches`: lowest patch row of its merged patch. */
  patchParents: GraphDataView<'uint32'>;
};

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
  /**
   * Caller-owned result views. Any view that is absent is a graph-owned transient view, as before.
   * Formats and lengths are those of {@link GPULandscapeMetricsOutput}.
   */
  output?: Partial<GPULandscapeMetricsOutput>;
};

/** Named class/landscape metrics from a complete, already labelled categorical raster. */
export function addLandscapeMetricsRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPULandscapeMetricsProps
) {
  props = snapshotConstructionFields(props, [
    'id',
    'patches',
    'patchClasses',
    'width',
    'height',
    'classCount',
    'cellSize',
    'connectivity',
    'reconcilePatches',
    'countBoundary',
    'countBackground',
    'backgroundClass',
    'contagionWeighting',
    'output'
  ]);
  const id = props.id ?? 'landscape-metrics';
  const {width, height, classCount, cellSize} = props;
  const cellCount = width * height;
  const patchCount = props.patchClasses.length;
  validateRequirement(
    id,
    Number.isSafeInteger(width) && width > 0 && Number.isSafeInteger(height) && height > 0,
    'width and height must be positive integers'
  );
  validateRequirement(
    id,
    props.patches.length === cellCount,
    'patches must have width times height rows'
  );
  validateRequirement(id, patchCount > 0, 'patchClasses must have at least 1 row');
  validateCapacity(id, 'classCount', classCount, 1, LANDSCAPE_METRICS_MAXIMUM_CLASS_COUNT);
  validateRequirement(
    id,
    cellSize.every(value => Number.isFinite(Math.fround(value)) && value > 0),
    'cellSize entries must be positive finite float32 numbers'
  );
  validateRequirement(
    id,
    props.connectivity === 4 || props.connectivity === 8,
    'connectivity must be 4 or 8'
  );
  validateMaximumCount(id, 'width times height', cellCount, LANDSCAPE_METRICS_MAXIMUM_CELL_COUNT);
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'raster labels',
    cellCount * 4,
    'reduce the raster size'
  );
  validateGPUStorageBindingByteLength(
    graph.device,
    id,
    'patch metrics',
    patchCount * 16,
    'reduce the patch count'
  );
  const output = snapshotConstructionFields(props.output ?? {}, [
    'patchMetrics',
    'classMetrics',
    'classAdjacency',
    'aggregation',
    'landscape',
    'landscapeAggregation',
    'landscapeCohesion',
    'reconciledPatches',
    'patchParents'
  ]);
  validateRecipeBoundary(
    graph,
    id,
    [props.patches, props.patchClasses],
    Object.values(output ?? {})
  );
  for (const [name, format, length] of [
    ['patchMetrics', 'float32x4', patchCount],
    ['classMetrics', 'float32x4', classCount],
    ['classAdjacency', 'uint32', classCount * classCount],
    ['aggregation', 'float32', classCount],
    ['landscape', 'float32x4', 1],
    ['landscapeAggregation', 'float32', 1],
    ['landscapeCohesion', 'float32', 1],
    ['reconciledPatches', 'uint32', cellCount],
    ['patchParents', 'uint32', patchCount]
  ] as const) {
    const view = output[name];
    if (view) getRecipeView(graph, `${id} output.${name}`, format, length, view);
  }
  // Every supplied output is validated before reconciliation or default allocation.
  const reconciled = props.reconcilePatches
    ? reconcileLandscapePatchLabels(graph, {
        ...props,
        id,
        output: {patches: output?.reconciledPatches, parents: output?.patchParents}
      })
    : undefined;
  const patches = reconciled?.patches ?? props.patches;
  const patchMetrics = getRecipeView(
    graph,
    `${id}-patches`,
    'float32x4',
    patchCount,
    output?.patchMetrics
  );
  const classMetrics = getRecipeView(
    graph,
    `${id}-classes`,
    'float32x4',
    classCount,
    output?.classMetrics
  );
  const classAdjacency = getRecipeView(
    graph,
    `${id}-adjacency`,
    'uint32',
    classCount * classCount,
    output?.classAdjacency
  );
  const aggregation = getRecipeView(
    graph,
    `${id}-aggregation`,
    'float32',
    classCount,
    output?.aggregation
  );
  const landscape = getRecipeView(graph, `${id}-landscape`, 'float32x4', 1, output?.landscape);
  const landscapeAggregation = getRecipeView(
    graph,
    `${id}-landscape-aggregation`,
    'float32',
    1,
    output?.landscapeAggregation
  );
  const landscapeCohesion = getRecipeView(
    graph,
    `${id}-landscape-cohesion`,
    'float32',
    1,
    output?.landscapeCohesion
  );
  const {classCells, classLikeSides} = addLandscapePatchMetrics(graph, {
    ...props,
    id,
    patches,
    patchMetrics,
    classAdjacency
  });
  const totalCellCount = createTransientView(graph, `${id}-valid-cell-count`, 'uint32', 1);
  graph.add(
    new GPUReduction({
      id: `${id}-valid-cell-count`,
      input: classCells,
      output: totalCellCount,
      operation: 'sum'
    }).getCommandNodes(graph)
  );
  const classKeys = createTransientView(graph, `${id}-class-keys`, 'uint32', patchCount);
  const classTerms = ['perimeter', 'squared-area', 'weighted-perimeter'].map(name => ({
    name,
    contributions: createTransientView(graph, `${id}-class-${name}-terms`, 'float32', patchCount),
    output: createTransientView(graph, `${id}-class-${name}-sums`, 'float32', classCount)
  }));
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-class-terms`,
      operation: id,
      invocationCount: patchCount,
      bindings: [
        {name: 'patches', view: patchMetrics, type: 'f32', access: 'read'},
        {name: 'classes', view: props.patchClasses, type: 'u32', access: 'read'},
        {name: 'keys', view: classKeys, type: 'u32', access: 'read_write'},
        ...classTerms.map((term, index) => ({
          name: `terms${index}`,
          view: term.contributions,
          type: 'f32' as const,
          access: 'read_write' as const
        }))
      ],
      body: `let category=classes[classesOffset+index];
keys[keysOffset+index]=select(category,0xffffffffu,category==${props.backgroundClass ?? 0xffffffff}u);
let area=patches[patchesOffset+4u*index];
let perimeter=patches[patchesOffset+4u*index+1u];
let cells=patches[patchesOffset+4u*index+2u];
terms0[terms0Offset+index]=perimeter;
terms1[terms1Offset+index]=area*area;
terms2[terms2Offset+index]=perimeter*sqrt(cells);`
    })
  );
  graph.add(
    getSortedSegmentSumNodes(graph, {
      id: `${id}-class-sums`,
      operation: id,
      segmentCount: classCount,
      segmentKeys: classKeys,
      reductions: classTerms
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-class-summary`,
      operation: id,
      invocationCount: classCount,
      declarations: `
fn writeOutput(row:u32,value:vec4<f32>){output[outputOffset+row*4u+0u]=value[0u];output[outputOffset+row*4u+1u]=value[1u];output[outputOffset+row*4u+2u]=value[2u];output[outputOffset+row*4u+3u]=value[3u];}
`,
      bindings: [
        {name: 'classCells', view: classCells, type: 'u32', access: 'read'},
        {name: 'classLikeSides', view: classLikeSides, type: 'u32', access: 'read'},
        {name: 'totalCellCount', view: totalCellCount, type: 'u32', access: 'read'},
        ...classTerms.map((term, index) => ({
          name: `sums${index}`,
          view: term.output,
          type: 'f32' as const,
          access: 'read' as const
        })),
        {name: 'output', view: classMetrics, type: 'f32', access: 'read_write'},
        {name: 'aggregation', view: aggregation, type: 'f32', access: 'read_write'}
      ],
      // Cell and like-side counts are exact u32 sums. The largest square inside the class is an
      // exact integer square root: float32 sqrt is only the first guess once cells pass 2^24.
      body: `let cells=classCells[classCellsOffset+index];
let likeSides=classLikeSides[classLikeSidesOffset+index];
let totalCells=totalCellCount[totalCellCountOffset];
let area=f32(cells)*${cellSize[0] * cellSize[1]};
let perimeter=sums0[sums0Offset+index];
let squaredArea=sums1[sums1Offset+index];
let weightedPerimeter=sums2[sums2Offset+index];
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));var cohesion=missing;var mesh=missing;
if(totalCells>0u){mesh=squaredArea/(f32(totalCells)*${cellSize[0] * cellSize[1]});}
if(weightedPerimeter>0.0&&totalCells>1u){cohesion=100.0*(1.0-perimeter/weightedPerimeter)/(1.0-inverseSqrt(f32(totalCells)));}
var side=min(u32(sqrt(f32(cells))),65535u);
while(side*side>cells){side-=1u;}
while(side<65535u&&(side+1u)*(side+1u)<=cells){side+=1u;}
let remainder=cells-side*side;
var shortfall=0u;if(remainder>0u){shortfall=select(2u,1u,remainder<=side);}
let maximum=2.0*f32(cells-side)-f32(shortfall);
var aggregate=missing;if(maximum>0.0){aggregate=100.0*f32(likeSides)/(2.0*maximum);}
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
  // Landscape cell, perimeter and perimeter-weighted sums over the patches are reductions; the
  // final node only walks the classes (at most LANDSCAPE_METRICS_MAXIMUM_CLASS_COUNT).
  const cellTerms = createTransientView(graph, `${id}-cell-terms`, 'float32', patchCount);
  const perimeterTerms = createTransientView(graph, `${id}-perimeter-terms`, 'float32', patchCount);
  const weightedTerms = createTransientView(graph, `${id}-weighted-terms`, 'float32', patchCount);
  const cellTotal = createTransientView(graph, `${id}-cell-total`, 'float32', 1);
  const perimeterTotal = createTransientView(graph, `${id}-perimeter-total`, 'float32', 1);
  const weightedTotal = createTransientView(graph, `${id}-weighted-total`, 'float32', 1);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-landscape-patch-terms`,
      operation: id,
      invocationCount: patchCount,
      bindings: [
        {name: 'patches', view: patchMetrics, type: 'f32', access: 'read'},
        {name: 'patchClasses', view: props.patchClasses, type: 'u32', access: 'read'},
        {name: 'cellTerms', view: cellTerms, type: 'f32', access: 'read_write'},
        {name: 'perimeterTerms', view: perimeterTerms, type: 'f32', access: 'read_write'},
        {name: 'weightedTerms', view: weightedTerms, type: 'f32', access: 'read_write'}
      ],
      body: `let category=patchClasses[patchClassesOffset+index];
let counted=category<${classCount}u&&category!=${props.backgroundClass ?? 0xffffffff}u;
let patchCells=patches[patchesOffset+4u*index+2u];
let patchPerimeter=patches[patchesOffset+4u*index+1u];
cellTerms[cellTermsOffset+index]=select(0.0,patchCells,counted);
perimeterTerms[perimeterTermsOffset+index]=select(0.0,patchPerimeter,counted);
weightedTerms[weightedTermsOffset+index]=select(0.0,patchPerimeter*sqrt(patchCells),counted);`
    })
  );
  for (const [name, input, total] of [
    ['cells', cellTerms, cellTotal],
    ['perimeter', perimeterTerms, perimeterTotal],
    ['weighted-perimeter', weightedTerms, weightedTotal]
  ] as const) {
    graph.add(
      new GPUReduction({
        id: `${id}-landscape-${name}-sum`,
        input,
        output: total,
        operation: 'sum'
      }).getCommandNodes(graph)
    );
  }
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-landscape-aggregation-cohesion`,
      operation: id,
      invocationCount: 1,
      bindings: [
        {name: 'cellTotal', view: cellTotal, type: 'f32', access: 'read'},
        {name: 'perimeterTotal', view: perimeterTotal, type: 'f32', access: 'read'},
        {name: 'weightedTotal', view: weightedTotal, type: 'f32', access: 'read'},
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
      body: `let cells=cellTotal[cellTotalOffset];
let perimeter=perimeterTotal[perimeterTotalOffset];
let weightedPerimeter=weightedTotal[weightedTotalOffset];
var weightedAggregation=0.0;
var area=0.0;
for(var category=0u;category<${classCount}u;category++){
  let classArea=classes[classesOffset+4u*category];
  let aggregate=aggregation[aggregationOffset+category];
  area+=classArea;
  // The pinned landscapemetrics landscape convention omits undefined single-cell class indices.
  if(classArea>0.0&&aggregate>=0.0){weightedAggregation+=classArea*aggregate;}
}
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
var aggregate=missing;
var cohesion=missing;
if(area>0.0){aggregate=weightedAggregation/area;}
if(cells>1.0&&weightedPerimeter>0.0){cohesion=100.0*(1.0-perimeter/weightedPerimeter)/(1.0-inverseSqrt(cells));}
landscapeAggregation[landscapeAggregationOffset]=aggregate;
landscapeCohesion[landscapeCohesionOffset]=cohesion;`
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
