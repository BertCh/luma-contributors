// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {captureUrbanLandscapeDestinations} from '../landscape-ecology/urban-landscape-destinations';
import {
  createTransientView,
  GPUCompaction,
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  addVisibilityGraphCSRRecipe,
  VISIBILITY_GRAPH_MAXIMUM_RAY_TOLERANCE,
  type GPUVisibilityGraphAdjacencyCSRProps
} from './visibility-graph-csr';
export type {
  GPUVisibilityGraphAdjacencyCSRProps,
  GPUVisibilityGraphAdjacencyCSROutput,
  GPUVisibilityGraphAdjacencyCSRStatus
} from './visibility-graph-csr';
export {
  getGPUVisibilityGraphCSRParameterValues,
  GPU_VISIBILITY_GRAPH_CSR_PARAMETER_LENGTH
} from './visibility-graph-csr';
import {
  validateUrbanRecipeBindings,
  validateUrbanStorageByteLength
} from './building-tessellation-validation';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {validateProvidedRecipeViews} from '../landscape-ecology/urban-landscape-recipe-validation';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  addUrbanPointGrid,
  getUrbanPointGridShape,
  getUrbanPointGridDeclarations,
  getUrbanPointGridRadiusWalk
} from './visibility-graph-point-grid';
import {classifyUrbanViewpoints, type UrbanFreeSpaceRegion} from './classify-viewpoints';
import {
  URBAN_MAXIMUM_OBSTACLE_SEGMENT_COUNT,
  addUrbanObstacleBVH,
  getUrbanRayBVHBindings,
  getUrbanRayDeclarations,
  resolveUrbanObstacleIndex,
  type UrbanObstacleIndex
} from './urban-ray-utils';

/**
 * Most viewpoints: the adjacency and Brandes choice tables hold viewpoints squared entries, and
 * each source thread keeps four per-viewpoint arrays in private memory.
 */
export const VISIBILITY_GRAPH_MAXIMUM_POINT_COUNT = 256;

export type GPUVisibilityGraphOutput = {
  /** `uint32`, `positions.length` squared rows: 1 visible, 0 blocked, 0xffffffff invalid. */
  adjacency: GraphDataView<'uint32'>;
  /** `uint32x2`, `edgeCapacity` rows: visible pairs (first < second), padded with 0xffffffff. */
  edges: GraphDataView<'uint32x2'>;
  /** `float32x4`, one row per viewpoint: degree, reachable count, total depth, mean depth inverse. NaN rows when any viewpoint is invalid. */
  metrics: GraphDataView<'float32x4'>;
  /** `float32`, one row per viewpoint: unnormalised undirected betweenness. */
  choice: GraphDataView<'float32'>;
  /** `float32x4`, one row per viewpoint: mean depth, asymmetry, real relative asymmetry, Hillier. */
  integration: GraphDataView<'float32x4'>;
  /**
   * `float32`, one row per viewpoint: normalised choice, `2 * choice / ((n - 1) * (n - 2))` with
   * `n = positions.length`. NaN is the contract in two cases: every viewpoint is NaN when any
   * viewpoint is invalid (`status` word 3 is not 0: a viewpoint within tolerance of an obstacle, or
   * non-finite), and every viewpoint is NaN when `n < 3`, where the normalisation is undefined.
   * Otherwise it is finite: a viewpoint that sees nothing scores 0.
   */
  normalizedChoice: GraphDataView<'float32'>;
  /** `uint32`, one row per viewpoint: 0 free, 1 blocked, 2 boundary, 3 non-finite. */
  viewpointClassifications: GraphDataView<'uint32'>;
  /** `float32x4`, one row per viewpoint: pyramid, Teklenburg, Penn, maximum depth. */
  extendedIntegration: GraphDataView<'float32x4'>;
  /** `uint32x4`, one row: `[count, requiredCount, overflow, invalid]`. */
  status: GraphDataView<'uint32x4'>;
};

export type GPUVisibilityGraphProps = {
  id?: string;
  /** Explicit small dense reference; existing calls retain their output layout. */
  mode?: 'reference';
  positions: GraphDataView<'float32x2'>;
  segments: GraphDataView<'float32x4'>;
  maximumDistance: number;
  /**
   * Compile-time viewpoint candidate search. Default 'grid' tests sightlines only for cells
   * meeting maximumDistance; 'brute-force' preserves the exhaustive reference route. The
   * dense adjacency and exact Brandes outputs retain VISIBILITY_GRAPH_MAXIMUM_POINT_COUNT.
   */
  searchStructure?: 'grid' | 'brute-force';
  edgeCapacity: number;
  /** Positive float32; indexed obstacles also require VISIBILITY_GRAPH_MAXIMUM_RAY_TOLERANCE. */
  tolerance?: number;
  /** Optional explicit filled-ring free-space classification. Default checks boundaries only. */
  region?: UrbanFreeSpaceRegion;
  /**
   * How sightlines and the viewpoint classification find obstacle segments. 'brute-force' tests
   * every segment, 'bvh' builds a segment BVH, and the default 'auto' picks brute force up to
   * `URBAN_BRUTE_FORCE_MAXIMUM_SEGMENT_COUNT` segments and the BVH above it. Both give the same
   * result. Compile-time.
   */
  obstacleIndex?: UrbanObstacleIndex;
  /**
   * Caller-owned result views. Any view that is absent is a graph-owned transient view, as before.
   * Formats and lengths are those of {@link GPUVisibilityGraphOutput}.
   */
  output?: Partial<GPUVisibilityGraphOutput>;
};

/** Closed obstacles block sightlines. CSR mode builds adjacency and degree only; import the same recipe name from @luma-contributors/analysis/gpu-network for sampled centrality. The dense reference retains exact integration and choice. */
export function addVisibilityGraphRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUVisibilityGraphAdjacencyCSRProps
): ReturnType<typeof addVisibilityGraphCSRRecipe>;
export function addVisibilityGraphRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUVisibilityGraphProps
): ReturnType<typeof addVisibilityGraphReferenceRecipe>;
export function addVisibilityGraphRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUVisibilityGraphProps | GPUVisibilityGraphAdjacencyCSRProps
):
  | ReturnType<typeof addVisibilityGraphCSRRecipe>
  | ReturnType<typeof addVisibilityGraphReferenceRecipe>;
export function addVisibilityGraphRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUVisibilityGraphProps | GPUVisibilityGraphAdjacencyCSRProps
) {
  validateRequirement(
    props.id ?? 'visibility-graph',
    props.mode === undefined || props.mode === 'reference' || props.mode === 'csr',
    "mode must be 'reference' or 'csr'"
  );
  return props.mode === 'csr'
    ? addVisibilityGraphCSRRecipe(graph, props)
    : addVisibilityGraphReferenceRecipe(graph, props);
}
/** Precise dense reference result, for composing wrappers without widening to the dispatcher union. */
export type GPUVisibilityGraphReferenceRecipeResult = ReturnType<
  typeof addVisibilityGraphReferenceRecipe
>;

function addVisibilityGraphReferenceRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUVisibilityGraphProps
) {
  const id = props.id ?? 'visibility-graph';
  validateRequirement(id, !('outputs' in props), 'outputs was removed; use output');
  validateRequirement(
    id,
    props.obstacleIndex === undefined ||
      (['auto', 'brute-force', 'bvh'] as readonly string[]).includes(props.obstacleIndex),
    `obstacleIndex must be auto or brute-force or bvh; received ${String(props.obstacleIndex)}`
  );
  validateRequirement(
    id,
    props.searchStructure === undefined ||
      (['grid', 'brute-force'] as readonly string[]).includes(props.searchStructure),
    `searchStructure must be grid or brute-force; received ${String(props.searchStructure)}`
  );
  const count = props.positions.length;
  const tolerance = props.tolerance ?? 1e-6;
  validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
  validateSinglePackedView(props.segments, ['float32x4'], `${id} segments`);
  validateRequirement(
    id,
    count > 0 && count <= VISIBILITY_GRAPH_MAXIMUM_POINT_COUNT,
    `positions must have between 1 and ${VISIBILITY_GRAPH_MAXIMUM_POINT_COUNT} rows`
  );
  validateRequirement(
    id,
    props.segments.length > 0 && props.segments.length <= URBAN_MAXIMUM_OBSTACLE_SEGMENT_COUNT,
    `segments must have between 1 and ${URBAN_MAXIMUM_OBSTACLE_SEGMENT_COUNT} rows`
  );
  validateRequirement(
    id,
    Number.isFinite(Math.fround(props.maximumDistance)) && Math.fround(props.maximumDistance) > 0,
    'maximumDistance must be a positive finite float32 number'
  );
  validateRequirement(
    id,
    Number.isFinite(Math.fround(tolerance)) && Math.fround(tolerance) > 0,
    'tolerance must be a positive finite float32 number'
  );
  const usesObstacleBVH =
    resolveUrbanObstacleIndex(props.obstacleIndex, props.segments.length) === 'bvh';
  validateRequirement(
    id,
    !usesObstacleBVH || tolerance <= VISIBILITY_GRAPH_MAXIMUM_RAY_TOLERANCE,
    `tolerance must be at most VISIBILITY_GRAPH_MAXIMUM_RAY_TOLERANCE (${VISIBILITY_GRAPH_MAXIMUM_RAY_TOLERANCE}) so four times tolerance is finite in float32`
  );
  validateRequirement(
    id,
    Number.isSafeInteger(props.edgeCapacity) && props.edgeCapacity > 0,
    'edgeCapacity must be a positive integer'
  );
  const output = captureUrbanLandscapeDestinations(id, {output: props.output}, [
    'adjacency',
    'edges',
    'status',
    'metrics',
    'choice',
    'integration',
    'extendedIntegration',
    'normalizedChoice',
    'viewpointClassifications'
  ]);
  validateUrbanRecipeBindings(
    graph,
    id,
    {positions: props.positions, segments: props.segments},
    output
  );
  let leafCapacity = 2;
  while (leafCapacity < props.segments.length) leafCapacity *= 2;
  // Check every row-proportional binding before adding any nodes or allocating recipe scratch.
  for (const [name, byteLength, hint] of [
    ['BVH node bounds', usesObstacleBVH ? (2 * leafCapacity - 1) * 8 : 4, 'reduce segments length'],
    [
      'point grid',
      props.searchStructure === 'brute-force' ? 4 : getUrbanPointGridShape(count).byteLength,
      'reduce positions length'
    ],
    ['positions', count * 8, 'reduce positions length'],
    ['segments', props.segments.length * 16, 'reduce segments length'],
    ['adjacency and pair scratch', count * count * 4, 'reduce positions length'],
    ['choice contributions', count * count * 4, 'reduce positions length'],
    ['edges', props.edgeCapacity * 8, 'reduce edgeCapacity'],
    ['metrics', count * 16, 'reduce positions length']
  ] as const) {
    validateUrbanStorageByteLength(graph.device, id, name, byteLength, hint);
  }
  // Formats and lengths of supplied outputs, before the obstacle index or classification adds nodes.
  validateProvidedRecipeViews(graph, [
    [`${id}-adjacency`, 'uint32', count * count, output?.adjacency],
    [`${id}-edges`, 'uint32x2', props.edgeCapacity, output?.edges],
    [`${id}-status`, 'uint32x4', 1, output?.status],
    [`${id}-metrics`, 'float32x4', count, output?.metrics],
    [`${id}-choice`, 'float32', count, output?.choice],
    [`${id}-integration`, 'float32x4', count, output?.integration],
    [`${id}-extended-integration`, 'float32x4', count, output?.extendedIntegration],
    [`${id}-normalized-choice`, 'float32', count, output?.normalizedChoice],
    [`${id}-viewpoint-classifications`, 'uint32', count, output?.viewpointClassifications]
  ]);
  const obstacleBVH =
    resolveUrbanObstacleIndex(props.obstacleIndex, props.segments.length) === 'bvh'
      ? addUrbanObstacleBVH(graph, {
          id: `${id}-obstacles`,
          operation: id,
          segments: props.segments,
          tolerance
        })
      : undefined;
  const viewpointClassifications = classifyUrbanViewpoints(graph, {
    id,
    positions: props.positions,
    segments: props.segments,
    tolerance,
    obstacleBVH,
    region: props.region,
    output: output?.viewpointClassifications
  });
  const adjacency = getRecipeView(
    graph,
    `${id}-adjacency`,
    'uint32',
    count * count,
    output?.adjacency
  );
  const edges = getRecipeView(graph, `${id}-edges`, 'uint32x2', props.edgeCapacity, output?.edges);
  const status = getRecipeView(graph, `${id}-status`, 'uint32x4', 1, output?.status);
  const metrics = getRecipeView(graph, `${id}-metrics`, 'float32x4', count, output?.metrics);
  const contributions = getRecipeView(
    graph,
    `${id}-choice-contributions`,
    'float32',
    count * count
  );
  const choice = getRecipeView(graph, `${id}-choice`, 'float32', count, output?.choice);
  const integration = getRecipeView(
    graph,
    `${id}-integration`,
    'float32x4',
    count,
    output?.integration
  );
  const maximumDepth = getRecipeView(graph, `${id}-maximum-depth`, 'float32', count);
  const extendedIntegration = getRecipeView(
    graph,
    `${id}-extended-integration`,
    'float32x4',
    count,
    output?.extendedIntegration
  );
  const normalizedChoice = getRecipeView(
    graph,
    `${id}-normalized-choice`,
    'float32',
    count,
    output?.normalizedChoice
  );
  const pointGrid =
    props.searchStructure === 'brute-force'
      ? undefined
      : addUrbanPointGrid(graph, {
          id: `${id}-viewpoint-grid`,
          positions: props.positions
        });
  if (pointGrid)
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-clear-adjacency`,
        operation: id,
        invocationCount: count * count,
        bindings: [
          {name: 'classifications', view: viewpointClassifications, type: 'u32', access: 'read'},
          {name: 'adjacency', view: adjacency, type: 'u32', access: 'read_write'}
        ],
        body: `let valid=classifications[classificationsOffset+index/${count}u]==0u&&classifications[classificationsOffset+index%${count}u]==0u;
adjacency[adjacencyOffset+index]=select(0xffffffffu,0u,valid);`
      })
    );
  const testSightline = `let origin=vec2<f32>(positions[positionsOffset+2u*first],positions[positionsOffset+2u*first+1u]);
let destination=vec2<f32>(positions[positionsOffset+2u*second],positions[positionsOffset+2u*second+1u]);
let displacement=destination-origin;
let distance=length(displacement);
let valid=classifications[classificationsOffset+first]==0u&&classifications[classificationsOffset+second]==0u&&isFiniteValue(origin.x)&&isFiniteValue(origin.y)&&isFiniteValue(destination.x)&&isFiniteValue(destination.y);
var visible=false;
if(valid&&first!=second&&distance>${getWGSLFloatLiteral(tolerance)}&&distance<=${getWGSLFloatLiteral(props.maximumDistance)}){
  let hit=castRay(origin,displacement/distance,distance);
  visible=hit.y<0.0;
}
output[outputOffset+first*${count}u+second]=select(u32(visible),0xffffffffu,!valid);`;
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-visibility`,
      operation: id,
      invocationCount: pointGrid ? count : count * count,
      declarations:
        IS_FINITE_VALUE_WGSL +
        (pointGrid ? getUrbanPointGridDeclarations(pointGrid) : '') +
        getUrbanRayDeclarations(props.segments.length, tolerance, obstacleBVH),
      bindings: [
        ...(pointGrid
          ? [{name: 'pointGrid', view: pointGrid.data, type: 'u32', access: 'read'} as const]
          : []),
        {name: 'classifications', view: viewpointClassifications, type: 'u32', access: 'read'},
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        ...(obstacleBVH ? getUrbanRayBVHBindings(obstacleBVH) : []),
        {name: 'output', view: adjacency, type: 'u32', access: 'read_write'}
      ],
      // Segment finiteness and clearance of both viewpoints are part of their classification
      // (3 for a non-finite segment, 2 within tolerance of a segment), so the pair test needs no
      // pass over the segments besides the ray itself.
      body: pointGrid
        ? `let first=index;
let center=vec2<f32>(positions[positionsOffset+2u*first],positions[positionsOffset+2u*first+1u]);
if(classifications[classificationsOffset+first]!=0u){return;}
${getUrbanPointGridRadiusWalk(pointGrid, 'center', getWGSLFloatLiteral(props.maximumDistance), 'second', testSightline)}`
        : `let first=index/${count}u;let second=index%${count}u;${testSightline}`
    })
  );
  // Visible pairs (first < second) in row-major order: flag, stable compaction of the pair ids,
  // decode to vertex pairs. The invalid count is a reduction over the diagonal.
  const pairIds = createTransientView(graph, `${id}-pair-ids`, 'uint32', count * count);
  const pairFlags = createTransientView(graph, `${id}-pair-flags`, 'uint32', count * count);
  const compactPairs = createTransientView(graph, `${id}-compact-pairs`, 'uint32', count * count);
  const requiredCount = createTransientView(graph, `${id}-required-count`, 'uint32', 1);
  const invalidFlags = createTransientView(graph, `${id}-invalid-flags`, 'uint32', count);
  const invalidCount = createTransientView(graph, `${id}-invalid-count`, 'uint32', 1);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-pair-flags`,
      operation: id,
      invocationCount: count * count,
      bindings: [
        {name: 'adjacency', view: adjacency, type: 'u32', access: 'read'},
        {name: 'pairIds', view: pairIds, type: 'u32', access: 'read_write'},
        {name: 'pairFlags', view: pairFlags, type: 'u32', access: 'read_write'}
      ],
      body: `pairIds[pairIdsOffset+index]=index;
pairFlags[pairFlagsOffset+index]=u32(index/${count}u<index%${count}u&&adjacency[adjacencyOffset+index]==1u);`
    })
  );
  graph.add(
    new GPUCompaction({
      id: `${id}-compact-pairs`,
      input: pairIds,
      flags: pairFlags,
      output: compactPairs,
      count: requiredCount
    }).getCommandNodes(graph)
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-invalid-flags`,
      operation: id,
      invocationCount: count,
      bindings: [
        {name: 'adjacency', view: adjacency, type: 'u32', access: 'read'},
        {name: 'invalidFlags', view: invalidFlags, type: 'u32', access: 'read_write'}
      ],
      body: `invalidFlags[invalidFlagsOffset+index]=u32(adjacency[adjacencyOffset+index*${count}u+index]==0xffffffffu);`
    })
  );
  graph.add(
    new GPUReduction({
      id: `${id}-invalid-count`,
      input: invalidFlags,
      output: invalidCount,
      operation: 'sum'
    }).getCommandNodes(graph)
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-edges`,
      operation: id,
      invocationCount: props.edgeCapacity,
      bindings: [
        {name: 'compactPairs', view: compactPairs, type: 'u32', access: 'read'},
        {name: 'requiredCount', view: requiredCount, type: 'u32', access: 'read'},
        {name: 'edges', view: edges, type: 'u32', access: 'read_write'}
      ],
      body: `let stored=index<requiredCount[requiredCountOffset];
let pair=compactPairs[compactPairsOffset+min(index,${count * count - 1}u)];
edges[edgesOffset+index*2u]=select(0xffffffffu,pair/${count}u,stored);
edges[edgesOffset+index*2u+1u]=select(0xffffffffu,pair%${count}u,stored);`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-compact`,
      operation: id,
      invocationCount: 1,
      bindings: [
        {name: 'requiredCount', view: requiredCount, type: 'u32', access: 'read'},
        {name: 'invalidCount', view: invalidCount, type: 'u32', access: 'read'},
        {name: 'status', view: status, type: 'u32', access: 'read_write'}
      ],
      body: `let required=requiredCount[requiredCountOffset];
status[statusOffset]=min(required,${props.edgeCapacity}u);
status[statusOffset+1u]=required;
status[statusOffset+2u]=u32(required>${props.edgeCapacity}u);
status[statusOffset+3u]=invalidCount[invalidCountOffset];`
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
for(var vertex=0u;vertex<${count}u*min(arrayLength(&adjacency),1u);vertex++){distances[vertex]=-1;}
queue[0u]=index;distances[index]=0;paths[index]=1.0;var head=0u;var tail=1u;var degree=0.0;var distanceSum=0.0;
for(var vertex=0u;vertex<${count}u;vertex++){degree+=f32(adjacency[adjacencyOffset+index*${count}u+vertex]==1u);}
loop{if(head>=tail){break;}let vertex=queue[head];head++;
for(var neighbor=0u;neighbor<${count}u;neighbor++){if(adjacency[adjacencyOffset+vertex*${count}u+neighbor]!=1u){continue;}
if(distances[neighbor]<0){distances[neighbor]=distances[vertex]+1;queue[tail]=neighbor;tail++;distanceSum+=f32(distances[neighbor]);}
if(distances[neighbor]==distances[vertex]+1){paths[neighbor]+=paths[vertex];}
}}
var reverse=tail;loop{if(reverse==0u){break;}reverse--;let vertex=queue[reverse];for(var predecessor=0u;predecessor<${count}u;predecessor++){if(adjacency[adjacencyOffset+predecessor*${count}u+vertex]==1u&&distances[predecessor]==distances[vertex] - 1){dependency[predecessor]+=paths[predecessor]/paths[vertex]*(1.0+dependency[vertex]);}}}
var maximum=0.0;for(var vertex=0u;vertex<${count}u;vertex++){maximum=max(maximum,f32(distances[vertex]));}maximumDepth[maximumDepthOffset+index]=maximum;
// Contract: one invalid viewpoint (status word 3) invalidates the summary metrics of every viewpoint.
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
      obstacleIndex: obstacleBVH ? ('bvh' as const) : ('brute-force' as const),
      searchStructure: pointGrid ? ('grid' as const) : ('brute-force' as const),
      integration: 'reachable-component-hillier-hanson' as const,
      choice: 'unnormalized-undirected-betweenness' as const
    }
  };
}
