// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {
  getSpatialAnalysisPackedKernelNodes,
  validateSpatialAnalysisPackedOutputAllocations
} from '../geographic-distribution/spatial-analysis-packed-kernel';
import {
  createTransientView,
  GPUReduction,
  GPUScan,
  GPUSort,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getBinarySearchWGSL} from '../../analysis-core/kernels/binary-search-wgsl';
import {PATCH_PROXIMITY_DISTANCE_WGSL} from './patch-proximity-distance';
import {getSortedSegmentOffsetNodes} from '../../analysis-core/kernels/sorted-segment-offsets';
import {
  createSegmentSumNode,
  getSortKeyBits
} from '../../analysis-core/kernels/sorted-segment-sums';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateRequirement,
  validateSinglePackedView,
  validateViewLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {getRecipeView} from '../recipes/expansion-recipe-utils';
import {
  prepareLandscapePatchInput,
  addLandscapePatchInput,
  validateLandscapePatchDestinations,
  suppressUnresolvedLandscapeResults
} from './landscape-patch-input';
import {reconcileLandscapePatchLabels} from './reconcile-patches';
import {validateLandscapeStorage, validateLandscapeViews} from './landscape-validation';
import type {GPUPatchProximityProps} from './patch-proximity';

/** Bounds serial raster-neighbour visits in each boundary/core invocation. */
export const PATCH_PROXIMITY_BOUNDARY_MAXIMUM_CELL_VISITS = 4096;
/** Four words per patch and signed raster-coordinate arithmetic must fit uint32/i32. */
const PATCH_PROXIMITY_BOUNDARY_MAXIMUM_ROWS = 0x3fffffff;
/** Candidate records use three uint32 words, addressed without wrapping. */
const PATCH_PROXIMITY_BOUNDARY_MAXIMUM_CANDIDATES = Math.floor(0xffffffff / 3);
/** Every source cell may append at most one candidate per visited cell; the append counter cannot wrap. */
const PATCH_PROXIMITY_BOUNDARY_MAXIMUM_CANDIDATE_REQUIREMENT = 0xffffffff;
/** Bounds default candidate storage to 12 MiB, independently of patch and raster counts. */
const PATCH_PROXIMITY_BOUNDARY_DEFAULT_CANDIDATE_CAPACITY = 1048576;
/** Conservative upper bound for three published sorts, scans, reductions and fixed raster passes. */
export const PATCH_PROXIMITY_BOUNDARY_COMMAND_NODE_ALLOWANCE = 768;

export type GPUPatchProximityBoundaryOutput = {
  core: GraphDataView<'uint32'>;
  /** Same [area,coreArea,nearestSameClassDistance,proximity] layout as the reference. */
  metrics: GraphDataView<'float32x4'>;
  /** One per patch. Zero when a radius or candidate budget could not certify its result, or its class is missing. */
  validity: GraphDataView<'uint32'>;
  reconciledPatches: GraphDataView<'uint32'>;
  patchParents: GraphDataView<'uint32'>;
  status: {
    /** Number of patches with validity = 1. */
    count: GraphDataView<'uint32'>;
    /** Patch count, not the candidate-pair storage requirement. */
    requiredCount: GraphDataView<'uint32'>;
    /** Always zero: patch output rows are fixed at construction. */
    overflow: GraphDataView<'uint32'>;
    /** One on candidate capacity loss or an uncertified radius result. */
    candidateOverflow: GraphDataView<'uint32'>;
  };
};

/**
 * Explicit sparse boundary route. Compile-time: lengths, radii, cell size, depth and capacities.
 * Per-frame: input contents. It retains core/metrics layouts but has no dense distance matrix or
 * patch-by-cell intermediate fields. The legacy overload retains those public outputs.
 */
export type GPUPatchProximityBoundaryProps = Omit<
  GPUPatchProximityProps,
  'mode' | 'output' | 'outputs'
> & {
  mode: 'boundary';
  /** Radius used to certify nearest neighbours; at least searchRadius. Defaults to searchRadius. */
  candidateRadius?: number;
  /** Global boundary-cell pair capacity; default min(8*cells,1,048,576). */
  candidateCapacity?: number;
  output?: Partial<Omit<GPUPatchProximityBoundaryOutput, 'status'>> & {
    status?: Partial<GPUPatchProximityBoundaryOutput['status']>;
  };
};

/**
 * Boundary-cell grid queries, stable pair sorting and parallel minimum/sum reductions.
 * O(cells + patches + candidateCapacity) storage, with no patches*cells or patches² allocation.
 * A nearest distance found within candidateRadius is globally exact: every omitted pair is farther.
 * If none is found for an occupied patch with another occupied same-class patch, validity=0 and
 * candidateOverflow=1; increase candidateRadius. Capacity failure invalidates all rows. No-data
 * classes (0xffffffff) are invalid. Empty patches and singleton classes have infinity nearest
 * distance (NaN for cell-center, matching the reference) and zero proximity. Core/area stay valid
 * even on uncertified rows. Proximity sums use a fixed tree in target-patch order.
 * Public status.requiredCount is the patch count and status.overflow is always zero.
 * intermediates.candidateRequirement is uint32[4]: word 0 is the unclamped directed
 * boundary-cell candidate count, word 1 is one on candidateCapacity exhaustion. These words
 * distinguish capacity loss (all rows invalid) from insufficient radius (only uncertified rows
 * invalid); status.candidateOverflow combines both. Words 2 and 3 are reserved scratch.
 * @internal Selected by the public addPatchProximityRecipe boundary overload.
 */
export function addPatchProximityBoundaryRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  inputProps: GPUPatchProximityBoundaryProps
) {
  validateRequirement(
    inputProps.id ?? 'patch-proximity',
    !('outputs' in inputProps),
    'outputs was removed; use output'
  );
  const props = prepareLandscapePatchInput(graph, inputProps.id ?? 'patch-proximity', inputProps);
  const id = props.id ?? 'patch-proximity';
  const {width, height, cellSize} = props;
  validateCapacity(id, 'width', width, 1, PATCH_PROXIMITY_BOUNDARY_MAXIMUM_ROWS);
  validateCapacity(id, 'height', height, 1, PATCH_PROXIMITY_BOUNDARY_MAXIMUM_ROWS);
  const cellCount = width * height;
  const patchCount = props.patchClasses.length;
  validateCapacity(id, 'width * height', cellCount, 1, PATCH_PROXIMITY_BOUNDARY_MAXIMUM_ROWS);
  validateCapacity(id, 'patchClasses.length', patchCount, 1, PATCH_PROXIMITY_BOUNDARY_MAXIMUM_ROWS);
  validateSinglePackedView(props.patches, ['uint32'], `${id} patches`);
  validateSinglePackedView(props.patchClasses, ['uint32'], `${id} patchClasses`);
  validateRequirement(
    id,
    props.patches.length === cellCount,
    'patches must have width times height rows'
  );
  for (const [name, value] of [
    ['cellSize[0]', cellSize[0]],
    ['cellSize[1]', cellSize[1]],
    ['edgeDepth', props.edgeDepth],
    ['searchRadius', props.searchRadius],
    ['candidateRadius', props.candidateRadius ?? props.searchRadius]
  ] as const)
    validateRequirement(
      id,
      Number.isFinite(Math.fround(value)) &&
        (name.startsWith('cellSize')
          ? Math.fround(value) > 0
          : value >= 0 && Math.fround(value) >= 0),
      `${name} must be a ${name.startsWith('cellSize') ? 'positive' : 'nonnegative'} finite float32 number`
    );
  const radius = props.candidateRadius ?? props.searchRadius;
  validateRequirement(
    id,
    radius >= props.searchRadius,
    'candidateRadius must be at least searchRadius'
  );
  validateRequirement(
    id,
    Number.isFinite(Math.fround(cellSize[0] * width)) &&
      Number.isFinite(Math.fround(cellSize[1] * height)) &&
      Number.isFinite(Math.fround(cellSize[0] * cellSize[1] * cellCount)),
    'cellSize and raster extent/area must stay finite in float32'
  );
  validateRequirement(
    id,
    props.connectivity === undefined || props.connectivity === 4 || props.connectivity === 8,
    'connectivity must be 4 or 8'
  );
  validateRequirement(
    id,
    props.distanceModel === undefined ||
      props.distanceModel === 'cell-boundary' ||
      props.distanceModel === 'cell-center',
    'distanceModel must be cell-boundary or cell-center'
  );
  validateRequirement(
    id,
    props.coreModel === undefined ||
      props.coreModel === 'physical' ||
      props.coreModel === 'raster-erosion',
    'coreModel must be physical or raster-erosion'
  );
  validateRequirement(
    id,
    props.coreBoundary === undefined ||
      props.coreBoundary === 'edge' ||
      props.coreBoundary === 'open',
    'coreBoundary must be edge or open'
  );
  const erosion = props.coreModel === 'raster-erosion';
  validateRequirement(
    id,
    !erosion || Number.isSafeInteger(props.edgeDepth),
    'edgeDepth must be an integer for raster-erosion'
  );
  const horizontalRadius = Math.min(width - 1, Math.ceil(radius / cellSize[0]) + 1);
  const verticalRadius = Math.min(height - 1, Math.ceil(radius / cellSize[1]) + 1);
  const horizontalDepth = Math.min(
    width,
    erosion ? props.edgeDepth : Math.ceil(props.edgeDepth / cellSize[0] + 0.5)
  );
  const verticalDepth = Math.min(
    height,
    erosion ? props.edgeDepth : Math.ceil(props.edgeDepth / cellSize[1] + 0.5)
  );
  const visits = (2 * horizontalRadius + 1) * (2 * verticalRadius + 1);
  validateMaximumCount(
    id,
    'candidateRadius neighbourhood cells',
    visits,
    PATCH_PROXIMITY_BOUNDARY_MAXIMUM_CELL_VISITS
  );
  validateMaximumCount(
    id,
    'edgeDepth neighbourhood cells',
    (2 * horizontalDepth + 1) * (2 * verticalDepth + 1),
    PATCH_PROXIMITY_BOUNDARY_MAXIMUM_CELL_VISITS
  );
  validateMaximumCount(
    id,
    'width * height * candidate neighbourhood cells',
    cellCount * visits,
    PATCH_PROXIMITY_BOUNDARY_MAXIMUM_CANDIDATE_REQUIREMENT
  );
  const candidateCapacity =
    props.candidateCapacity ??
    Math.min(Math.max(1, cellCount * 8), PATCH_PROXIMITY_BOUNDARY_DEFAULT_CANDIDATE_CAPACITY);
  validateCapacity(
    id,
    'candidateCapacity',
    candidateCapacity,
    1,
    PATCH_PROXIMITY_BOUNDARY_MAXIMUM_CANDIDATES
  );
  validateCommandNodeBudget(id, PATCH_PROXIMITY_BOUNDARY_COMMAND_NODE_ALLOWANCE);
  const destinations =
    props.output &&
    snapshotConstructionFields(props.output, [
      'core',
      'metrics',
      'validity',
      'reconciledPatches',
      'patchParents',
      'status'
    ]);
  const provided = destinations && {
    ...destinations,
    status:
      destinations.status &&
      snapshotConstructionFields(destinations.status, [
        'count',
        'requiredCount',
        'overflow',
        'candidateOverflow'
      ])
  };
  const usedOutputs = {
    core: provided?.core,
    metrics: provided?.metrics,
    validity: provided?.validity,
    ...(props.reconcilePatches
      ? {reconciledPatches: provided?.reconciledPatches, patchParents: provided?.patchParents}
      : {}),
    ...provided?.status
  };
  validateLandscapePatchDestinations(graph, id, props, Object.values(usedOutputs));
  const inputs = {patches: props.patches, patchClasses: props.patchClasses};
  validateLandscapeViews(id, inputs, usedOutputs, [
    {
      reads: {patches: props.reconcilePatches ? provided?.reconciledPatches : props.patches},
      writes: {core: provided?.core}
    },
    {
      reads: {patchClasses: props.patchClasses},
      writes: {metrics: provided?.metrics, validity: provided?.validity}
    },
    {
      reads: {},
      writes: {requiredCount: provided?.status?.requiredCount, overflow: provided?.status?.overflow}
    }
  ]);
  if (props.reconcilePatches)
    validateLandscapeViews(id, inputs, usedOutputs, [
      {reads: inputs, writes: {parents: provided?.patchParents}},
      {
        reads: {patches: props.patches, parents: provided?.patchParents},
        writes: {reconciled: provided?.reconciledPatches}
      }
    ]);
  validateSpatialAnalysisPackedOutputAllocations(graph, id, usedOutputs);
  validateGraphChunkBindingRanges(graph, id, {...inputs, ...usedOutputs});
  for (const [name, format, length, view] of [
    ['core', 'uint32', cellCount, provided?.core],
    ['metrics', 'float32x4', patchCount, provided?.metrics],
    ['validity', 'uint32', patchCount, provided?.validity],
    ...(props.reconcilePatches
      ? ([
          ['reconciledPatches', 'uint32', cellCount, provided?.reconciledPatches],
          ['patchParents', 'uint32', patchCount, provided?.patchParents]
        ] as const)
      : []),
    ...Object.entries(provided?.status ?? {}).map(
      ([name, view]) => [name, 'uint32', 1, view] as const
    )
  ] as const) {
    if (view) {
      validateSinglePackedView(view, [format], `${id} output.${name}`);
      validateViewLength(id, `output.${name}`, view, length);
    }
  }
  for (const [name, bytes] of [
    ['cell flags', cellCount * 4],
    ['patch records', patchCount * 16],
    ['patch class offsets', (patchCount + 1) * 4],
    ['candidate records', candidateCapacity * 12],
    ['candidate pair keys', candidateCapacity * 8],
    ['candidate scan/sort scratch', candidateCapacity * 4],
    ['control scratch', 16]
  ] as const)
    validateLandscapeStorage(graph.device, id, name, bytes);
  const view = <Format extends 'uint32' | 'float32' | 'uint32x2' | 'float32x4'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${id}-boundary-${name}`, format, length);
  const kernel = (
    name: string,
    count: number,
    bindings: WGSLKernelBinding[],
    body: string,
    declarations = ''
  ) =>
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-boundary-${name}`,
      operation: 'PatchProximityBoundary',
      invocationCount: count,
      bindings,
      body,
      declarations
    });
  const read = (
    name: string,
    view: GraphDataView,
    type: 'u32' | 'f32' = 'u32'
  ): WGSLKernelBinding => ({name, view, type, access: 'read'});
  const write = (
    name: string,
    view: GraphDataView,
    type: 'u32' | 'f32' | 'atomic<u32>' = 'u32'
  ): WGSLKernelBinding => ({name, view, type, access: 'read_write'});
  const core = getRecipeView(graph, `${id}-boundary-core`, 'uint32', cellCount, provided?.core);
  const metrics = getRecipeView(
    graph,
    `${id}-boundary-metrics`,
    'float32x4',
    patchCount,
    provided?.metrics
  );
  const validity = getRecipeView(
    graph,
    `${id}-boundary-validity`,
    'uint32',
    patchCount,
    provided?.validity
  );
  const scalar = (name: keyof GPUPatchProximityBoundaryOutput['status']) =>
    getRecipeView(graph, `${id}-boundary-${name}`, 'uint32', 1, provided?.status?.[name]);
  const status = {
    count: scalar('count'),
    requiredCount: scalar('requiredCount'),
    overflow: scalar('overflow'),
    candidateOverflow: scalar('candidateOverflow')
  };
  const inputPatches = addLandscapePatchInput(graph, id, props);
  const reconciled = props.reconcilePatches
    ? reconcileLandscapePatchLabels(graph, {
        ...props,
        patches: inputPatches,
        id: `${id}-boundary`,
        connectivity: props.connectivity ?? 4,
        output: {patches: provided?.reconciledPatches, parents: provided?.patchParents}
      })
    : undefined;
  const patches = reconciled?.patches ?? inputPatches;
  const flags = view('boundary-flags', 'uint32', cellCount);
  const statistics = view('patch-records', 'uint32', 4 * patchCount);
  const control = view('control', 'uint32', 4);
  const records = view('candidates', 'uint32', 3 * candidateCapacity);
  const constants = `${PATCH_PROXIMITY_DISTANCE_WGSL}
const CELL_AREA:f32=${getWGSLFloatLiteral(cellSize[0] * cellSize[1])};
const WIDTH:u32=${width}u;const HEIGHT:u32=${height}u;const PATCHES:u32=${patchCount}u;const CAPACITY:u32=${candidateCapacity}u;
const CELL_SIZE:vec2<f32>=vec2<f32>(${getWGSLFloatLiteral(cellSize[0])},${getWGSLFloatLiteral(cellSize[1])});
const RADIUS:f32=${getWGSLFloatLiteral(radius)};const SEARCH_RADIUS:f32=${getWGSLFloatLiteral(props.searchRadius)};const DEPTH:f32=${getWGSLFloatLiteral(props.edgeDepth)};
fn infinity(row:u32)->f32{return bitcast<f32>(0x7f800000u|(row&0u));}
fn missing(row:u32)->f32{return bitcast<f32>(0x7fc00000u|(row&0u));}`;
  graph.add(
    kernel(
      'initialize',
      Math.max(patchCount, candidateCapacity, 4),
      [write('statistics', statistics), write('records', records), write('control', control)],
      `if(index<PATCHES){let base=statisticsOffset+4u*index;statistics[base]=0u;statistics[base+1u]=0u;statistics[base+2u]=0x7f800000u;statistics[base+3u]=0u;}
if(index<CAPACITY){let base=recordsOffset+3u*index;records[base]=PATCHES;records[base+1u]=PATCHES;records[base+2u]=0x7f800000u;}
if(index<4u){control[controlOffset+index]=0u;}`,
      constants
    )
  );
  graph.add(
    kernel(
      'core',
      cellCount,
      [
        read('patches', patches),
        write('core', core),
        write('flags', flags),
        write('statistics', statistics, 'atomic<u32>')
      ],
      `let owner=patches[patchesOffset+index];core[coreOffset+index]=0u;flags[flagsOffset+index]=0u;if(owner>=PATCHES){return;}
let column=i32(index%WIDTH);let row=i32(index/WIDTH);var interior=true;var boundary=false;
let directions=array<vec2<i32>,4>(vec2<i32>(-1,0),vec2<i32>(1,0),vec2<i32>(0,-1),vec2<i32>(0,1));
for(var direction=0u;direction<4u;direction++){let adjacent=vec2<i32>(column,row)+directions[direction];if(any(adjacent < vec2<i32>(0))||adjacent.x>=i32(WIDTH)||adjacent.y>=i32(HEIGHT)){boundary=true;}else if(patches[patchesOffset+u32(adjacent.y)*WIDTH+u32(adjacent.x)]!=owner){boundary=true;}}
${
  erosion
    ? `for(var offsetRow=-${verticalDepth};offsetRow<=${verticalDepth};offsetRow++){for(var offsetColumn=-${horizontalDepth};offsetColumn<=${horizontalDepth};offsetColumn++){
 if(f32(abs(offsetRow)+abs(offsetColumn))>DEPTH){continue;}let other=vec2<i32>(column+offsetColumn,row+offsetRow);
 if(any(other < vec2<i32>(0))||other.x>=i32(WIDTH)||other.y>=i32(HEIGHT)){${props.coreBoundary === 'open' ? '' : 'interior=false;'}continue;}
 interior=interior&&patches[patchesOffset+u32(other.y)*WIDTH+u32(other.x)]==owner;}}`
    : `let position=(vec2<f32>(f32(column),f32(row))+vec2<f32>(0.5))*CELL_SIZE;
var closest=${props.coreBoundary === 'open' ? 'infinity(index)' : 'min(min(position.x,f32(WIDTH)*CELL_SIZE.x-position.x),min(position.y,f32(HEIGHT)*CELL_SIZE.y-position.y))'};
for(var offsetRow=-${verticalDepth};offsetRow<=${verticalDepth};offsetRow++){for(var offsetColumn=-${horizontalDepth};offsetColumn<=${horizontalDepth};offsetColumn++){
 let other=vec2<i32>(column+offsetColumn,row+offsetRow);if(any(other < vec2<i32>(0))||other.x>=i32(WIDTH)||other.y>=i32(HEIGHT)){continue;}
 if(patches[patchesOffset+u32(other.y)*WIDTH+u32(other.x)]==owner){continue;}
 let displacement=abs(vec2<f32>(f32(offsetColumn),f32(offsetRow)))*CELL_SIZE;
 closest=min(closest,getPatchProximityDistance(max(displacement-0.5*CELL_SIZE,vec2<f32>(0.0))));}}
interior=closest>DEPTH;`
}
core[coreOffset+index]=u32(interior);flags[flagsOffset+index]=u32(boundary);
atomicAdd(&statistics[statisticsOffset+4u*owner],1u);atomicAdd(&statistics[statisticsOffset+4u*owner+1u],u32(interior));`,
      constants
    )
  );
  graph.add(
    kernel(
      'candidates',
      cellCount,
      [
        read('patches', patches),
        read('classes', props.patchClasses),
        read('flags', flags),
        write('records', records),
        write('control', control, 'atomic<u32>')
      ],
      `let owner=patches[patchesOffset+index];if(owner>=PATCHES||flags[flagsOffset+index]==0u){return;}let category=classes[classesOffset+owner];if(category==0xffffffffu){return;}
let column=i32(index%WIDTH);let row=i32(index/WIDTH);
for(var offsetRow=-${verticalRadius};offsetRow<=${verticalRadius};offsetRow++){for(var offsetColumn=-${horizontalRadius};offsetColumn<=${horizontalRadius};offsetColumn++){
 let other=vec2<i32>(column+offsetColumn,row+offsetRow);if(any(other < vec2<i32>(0))||other.x>=i32(WIDTH)||other.y>=i32(HEIGHT)){continue;}
 let cell=u32(other.y)*WIDTH+u32(other.x);let partner=patches[patchesOffset+cell];if(partner>=PATCHES||partner==owner||flags[flagsOffset+cell]==0u){continue;}
 if(classes[classesOffset+partner]!=category){continue;}
 let displacement=max(abs(vec2<f32>(f32(offsetColumn),f32(offsetRow)))-vec2<f32>(${props.distanceModel === 'cell-center' ? '0.0' : '1.0'}),vec2<f32>(0.0))*CELL_SIZE;
 let distance=getPatchProximityDistance(displacement);if(distance>RADIUS){continue;}
 let slot=atomicAdd(&control[controlOffset],1u);if(slot>=CAPACITY){atomicStore(&control[controlOffset+1u],1u);continue;}
 let base=recordsOffset+3u*slot;records[base]=owner;records[base+1u]=partner;records[base+2u]=bitcast<u32>(distance);
}}`,
      constants
    )
  );
  const keys = view('sort-keys', 'uint32', candidateCapacity);
  const rows = view('sort-rows', 'uint32', candidateCapacity);
  const sortedTargets = view('sorted-targets', 'uint32', candidateCapacity);
  const targetRows = view('target-rows', 'uint32', candidateCapacity);
  const sourceKeys = view('source-keys', 'uint32', candidateCapacity);
  const sortedSources = view('sorted-sources', 'uint32', candidateCapacity);
  const sortedRows = view('sorted-rows', 'uint32', candidateCapacity);
  graph.add(
    kernel(
      'target-keys',
      candidateCapacity,
      [read('records', records), write('keys', keys), write('rows', rows)],
      'keys[keysOffset+index]=records[recordsOffset+3u*index+1u];rows[rowsOffset+index]=index;'
    )
  );
  graph.add(
    new GPUSort({
      id: `${id}-boundary-target-sort`,
      keys,
      values: rows,
      outputKeys: sortedTargets,
      outputValues: targetRows,
      keyBits: getSortKeyBits(patchCount)
    }).getCommandNodes(graph)
  );
  graph.add(
    kernel(
      'source-keys',
      candidateCapacity,
      [read('records', records), read('rows', targetRows), write('keys', sourceKeys)],
      'keys[keysOffset+index]=records[recordsOffset+3u*rows[rowsOffset+index]];'
    )
  );
  graph.add(
    new GPUSort({
      id: `${id}-boundary-source-sort`,
      keys: sourceKeys,
      values: targetRows,
      outputKeys: sortedSources,
      outputValues: sortedRows,
      keyBits: getSortKeyBits(patchCount)
    }).getCommandNodes(graph)
  );
  const pairKeys = view('pair-keys', 'uint32', 2 * candidateCapacity);
  const pairMinimum = view('pair-minimum', 'uint32', candidateCapacity);
  graph.add(
    kernel(
      'pair-keys',
      candidateCapacity,
      [
        read('records', records),
        read('rows', sortedRows),
        write('pairs', pairKeys),
        write('minimum', pairMinimum)
      ],
      `let base=recordsOffset+3u*rows[rowsOffset+index];pairs[pairsOffset+2u*index]=records[base+1u];pairs[pairsOffset+2u*index+1u]=records[base];minimum[minimumOffset+index]=0x7f800000u;`
    )
  );
  const pairSearch = getBinarySearchWGSL('pairs', 'u64');
  graph.add(
    kernel(
      'pair-minimum',
      candidateCapacity,
      [
        read('records', records),
        read('rows', sortedRows),
        read('pairs', pairKeys),
        write('minimum', pairMinimum, 'atomic<u32>')
      ],
      `let key=vec2<u32>(pairs[pairsOffset+2u*index+1u],pairs[pairsOffset+2u*index]);if(key.x>=PATCHES){return;}let head=lowerBoundPairs(0u,${candidateCapacity}u,key);
atomicMin(&minimum[minimumOffset+head],records[recordsOffset+3u*rows[rowsOffset+index]+2u]);`,
      constants + pairSearch
    )
  );
  const terms = view('proximity-terms', 'float32', candidateCapacity);
  const sourceOffsets = view('source-offsets', 'uint32', patchCount + 1);
  const proximity = view('proximity', 'float32', patchCount);
  graph.add(
    kernel(
      'pair-terms',
      candidateCapacity,
      [
        read('pairs', pairKeys),
        read('minimum', pairMinimum),
        write('statistics', statistics, 'atomic<u32>'),
        write('terms', terms, 'f32')
      ],
      `terms[termsOffset+index]=0.0;let owner=pairs[pairsOffset+2u*index+1u];let partner=pairs[pairsOffset+2u*index];if(owner>=PATCHES){return;}
if(index>0u&&owner==pairs[pairsOffset+2u*index-1u]&&partner==pairs[pairsOffset+2u*index-2u]){return;}
let key=minimum[minimumOffset+index];atomicMin(&statistics[statisticsOffset+4u*owner+2u],key);
let distance=bitcast<f32>(key);if(distance<=SEARCH_RADIUS){let area=f32(atomicLoad(&statistics[statisticsOffset+4u*partner]))*CELL_AREA;
if(distance==0.0){terms[termsOffset+index]=select(infinity(index),missing(index),area==0.0);}else{terms[termsOffset+index]=getScaledQuotient(area,distance,distance,0);}}`,
      constants
    )
  );
  graph.add(
    getSortedSegmentOffsetNodes(graph, {
      id: `${id}-boundary-source-offsets`,
      operation: 'PatchProximityBoundary',
      groupCount: patchCount,
      sortedGroupKeys: sortedSources,
      segmentOffsets: sourceOffsets
    })
  );
  graph.add(
    createSegmentSumNode(graph, {
      id: `${id}-boundary-proximity`,
      operation: 'PatchProximityBoundary',
      segmentCount: patchCount,
      input: terms,
      segmentOffsets: sourceOffsets,
      output: proximity
    })
  );
  const classKeys = view('class-keys', 'uint32', patchCount);
  const classRows = view('class-rows', 'uint32', patchCount);
  const sortedClasses = view('sorted-classes', 'uint32', patchCount);
  const sortedClassRows = view('sorted-class-rows', 'uint32', patchCount);
  const presence = view('class-presence', 'uint32', patchCount);
  const presencePrefix = view('class-prefix', 'uint32', patchCount);
  graph.add(
    kernel(
      'class-keys',
      patchCount,
      [read('classes', props.patchClasses), write('keys', classKeys), write('rows', classRows)],
      'keys[keysOffset+index]=classes[classesOffset+index];rows[rowsOffset+index]=index;'
    )
  );
  graph.add(
    new GPUSort({
      id: `${id}-boundary-class-sort`,
      keys: classKeys,
      values: classRows,
      outputKeys: sortedClasses,
      outputValues: sortedClassRows
    }).getCommandNodes(graph)
  );
  graph.add(
    kernel(
      'class-presence',
      patchCount,
      [read('statistics', statistics), read('rows', sortedClassRows), write('presence', presence)],
      'presence[presenceOffset+index]=u32(statistics[statisticsOffset+4u*rows[rowsOffset+index]]>0u);'
    )
  );
  graph.add(
    new GPUScan({
      id: `${id}-boundary-class-scan`,
      input: presence,
      output: presencePrefix,
      mode: 'inclusive'
    }).getCommandNodes(graph)
  );
  const uncertified = view('uncertified', 'uint32', patchCount);
  graph.add(
    kernel(
      'summary',
      patchCount,
      [
        read('classes', props.patchClasses),
        read('sortedClasses', sortedClasses),
        read('prefix', presencePrefix),
        read('statistics', statistics),
        read('control', control),
        read('proximity', proximity, 'f32'),
        write('metrics', metrics, 'f32'),
        write('validity', validity)
      ],
      `let category=classes[classesOffset+index];let first=lowerBoundSortedClasses(0u,PATCHES,category);let end=upperBoundSortedClasses(0u,PATCHES,category);
var population=0u;if(end>first){population=prefix[prefixOffset+end-1u];if(first>0u){population-=prefix[prefixOffset+first-1u];}}
let base=statisticsOffset+4u*index;let cells=statistics[base];let nearestWord=statistics[base+2u];let nearest=bitcast<f32>(nearestWord);let missingNeighbor=nearestWord==0x7f800000u;
let failed=control[controlOffset+1u]!=0u||(cells>0u&&population>1u&&missingNeighbor);let valid=!failed&&category!=0xffffffffu;
validity[validityOffset+index]=u32(valid);let destination=metricsOffset+4u*index;
metrics[destination]=f32(cells)*CELL_AREA;metrics[destination+1u]=f32(statistics[base+1u])*CELL_AREA;
metrics[destination+2u]=select(missing(index),${props.distanceModel === 'cell-center' ? 'select(nearest,missing(index),missingNeighbor)' : 'nearest'},valid);
metrics[destination+3u]=select(missing(index),proximity[proximityOffset+index],valid);`,
      constants + getBinarySearchWGSL('sortedClasses')
    )
  );
  graph.add(
    kernel(
      'uncertified',
      patchCount,
      [
        read('classes', props.patchClasses),
        read('validity', validity),
        write('flags', uncertified)
      ],
      'flags[flagsOffset+index]=u32(classes[classesOffset+index]!=0xffffffffu&&validity[validityOffset+index]==0u);'
    )
  );
  // Published reduction requires distinct buffer handles for input/output, including disjoint
  // caller subviews. Reduce validity into private scratch, then publish the caller scalar.
  const validCount = view('valid-count', 'uint32', 1);
  for (const [name, input, output, operation] of [
    ['count', validity, validCount, 'sum'],
    ['candidateOverflow', uncertified, status.candidateOverflow, 'max']
  ] as const)
    graph.add(
      new GPUReduction({
        id: `${id}-boundary-${name}-reduce`,
        input,
        output,
        operation
      }).getCommandNodes(graph)
    );
  graph.add(
    kernel(
      'publish-count',
      1,
      [read('count', validCount), write('output', status.count)],
      'output[outputOffset]=count[countOffset];'
    )
  );
  graph.add(
    kernel(
      'status',
      1,
      [write('requiredCount', status.requiredCount), write('overflow', status.overflow)],
      `requiredCount[requiredCountOffset]=${patchCount}u;overflow[overflowOffset]=0u;`
    )
  );
  suppressUnresolvedLandscapeResults(
    graph,
    id,
    props,
    [
      core,
      metrics,
      validity,
      status.count,
      ...(reconciled ? [reconciled.patches, reconciled.parents] : [])
    ],
    [
      {view: core, value: 0},
      {view: validity, value: 0},
      {view: status.count, value: 0}
    ]
  );
  return {
    inputStatus: props.denseLabels?.status,
    outputs: {core, metrics, validity},
    status,
    intermediates: {
      candidateRecords: records,
      /**
       * uint32[4]: [unclamped directed candidate count, capacity-exhaustion flag, reserved, reserved].
       * Public requiredCount counts patches; use these first two words to size candidateCapacity.
       */
      candidateRequirement: control,
      patchRecords: statistics
    },
    estimatedCommandNodeCount: PATCH_PROXIMITY_BOUNDARY_COMMAND_NODE_ALLOWANCE,
    provenance: {
      patches: inputPatches,
      denseLabels: props.denseLabels,
      patchClasses: props.patchClasses,
      reconciledPatches: reconciled?.patches,
      patchParents: reconciled?.parents,
      mode: 'boundary' as const,
      candidateRadius: radius,
      candidateCapacity,
      searchRadius: props.searchRadius,
      cellSize,
      completeDomain: true as const
    }
  };
}
