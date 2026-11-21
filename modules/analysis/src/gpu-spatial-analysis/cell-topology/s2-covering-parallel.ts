// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUScan,
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateCellDispatchBindings} from '../cell-indexing/cell-dispatch-validation';
import {
  GPU_S2_POLYGON_COVERING_DEFAULT_VISITED_CELLS,
  type GPUS2PolygonCoveringProps
} from './gpu-s2-polygon-covering';
import {getScanCommandNodeCount} from './s2-covering-support';
import {
  getS2CoveringParallelWGSL,
  MAXIMUM_S2_COVERING_INPUT_EDGES_PER_INVOCATION,
  MAXIMUM_S2_COVERING_RANGE_SEARCH_STEPS
} from './s2-covering-parallel-wgsl';

/** WebGPU guarantees a 128 MiB storage binding; winding scratch never exceeds this per binding. */
const MAXIMUM_WINDING_SCRATCH_BYTES = 128 * 1024 * 1024;
const MAXIMUM_INVOCATION_COUNT = 0xffffffff;
const ROOT_COUNT = 6;
const CONTROL_WORD_COUNT = 8;
// Each two-word round descriptor starts at a portable storage-binding alignment.
const ROUND_PARAMETER_SLOT_WORD_COUNT = 64;

function getLayout(props: GPUS2PolygonCoveringProps) {
  const positionCount = props.positions.length;
  const ringCount = props.ringOffsets.length - 1;
  const budget = props.maximumVisitedCells ?? GPU_S2_POLYGON_COVERING_DEFAULT_VISITED_CELLS;
  const rounds = props.maximumLevel + 1;
  const tile = MAXIMUM_S2_COVERING_INPUT_EDGES_PER_INVOCATION;
  const maximumLongRingCount = Math.max(
    0,
    Math.min(ringCount, Math.floor((positionCount - 3 * ringCount) / (tile - 2)))
  );
  const longRingCapacity = Math.max(1, maximumLongRingCount);
  const tileCapacity = Math.max(1, maximumLongRingCount + Math.ceil(positionCount / tile));
  const batchCapacity = Math.min(
    budget,
    Math.max(1, Math.floor(MAXIMUM_WINDING_SCRATCH_BYTES / (16 * longRingCapacity)))
  );
  const rowChunks = Math.max(
    1,
    Math.floor(MAXIMUM_INVOCATION_COUNT / Math.max(1, Math.ceil(positionCount / tile)))
  );
  const pairChunks = Math.ceil(Math.max(positionCount, 1) / rowChunks);
  const reductionStages = Math.ceil(Math.log2(Math.max(positionCount, 1)));
  return {
    positionCount,
    ringCount,
    budget,
    rounds,
    tile,
    longRingCapacity,
    tileCapacity,
    batchCapacity,
    rowChunks,
    pairChunks,
    reductionStages
  };
}
function getProductChunkCount(rows: number, columns: number): number {
  return Math.ceil(rows / Math.max(1, Math.floor(MAXIMUM_INVOCATION_COUNT / columns)));
}

/** Exact command-node count, including input reduction stages and bounded predicate batches. @internal */
export function getS2CoveringParallelNodeCount(props: GPUS2PolygonCoveringProps): number {
  if (props.positions.length < 3 * (props.ringOffsets.length - 1)) return 2;
  const layout = getLayout(props);
  const {budget, batchCapacity, ringCount, tileCapacity, longRingCapacity, rounds} = layout;
  const fullBatches = Math.floor(budget / batchCapacity);
  const tail = budget % batchCapacity;
  const batchNodes = (count: number) =>
    2 +
    Math.ceil(count / Math.max(1, Math.floor(MAXIMUM_INVOCATION_COUNT / ringCount))) +
    getProductChunkCount(count, tileCapacity) +
    getProductChunkCount(count, longRingCapacity);
  const predicateNodes = fullBatches * batchNodes(batchCapacity) + (tail ? batchNodes(tail) : 0);
  const preparation =
    10 + 2 * layout.reductionStages + 2 * getScanCommandNodeCount(ringCount) + layout.pairChunks;
  return (
    preparation +
    rounds *
      (predicateNodes +
        getScanCommandNodeCount(budget) +
        3 +
        (props.refinement === 'level-synchronous' ? 1 : 0)) +
    1
  );
}

/** Parallel preparation and predicates, followed by exact ordered prefix publication. @internal */
export function createS2CoveringParallelNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUS2PolygonCoveringProps,
  id: string
): GPUCommandNode<Parameters>[] {
  if (props.positions.length < 3 * (props.ringOffsets.length - 1)) {
    validateCellDispatchBindings(
      graph,
      id,
      {positions: props.positions, ringOffsets: props.ringOffsets, ...props.output},
      []
    );
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-invalid-cells`,
        operation: 'GPUS2PolygonCovering',
        variant: 'invalid-shape',
        invocationCount: props.capacity,
        bindings: [{name: 'cells', view: props.output.cells, type: 'u32', access: 'read_write'}],
        body: 'cells[cellsOffset+index*2u]=0u;cells[cellsOffset+index*2u+1u]=0u;'
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-invalid-status`,
        operation: 'GPUS2PolygonCovering',
        variant: 'invalid-shape',
        invocationCount: 1,
        bindings: [
          {name: 'counts', view: props.output.counts, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'},
          ...(props.output.visitedCount
            ? [
                {
                  name: 'visited',
                  view: props.output.visitedCount,
                  type: 'u32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        body: `counts[countsOffset]=0u;counts[countsOffset+1u]=0u;status[statusOffset]=1u;${props.output.visitedCount ? 'visited[visitedOffset]=0u;' : ''}`
      })
    ];
  }
  const layout = getLayout(props);
  const {
    positionCount,
    ringCount,
    budget,
    rounds,
    tile,
    longRingCapacity,
    tileCapacity,
    batchCapacity
  } = layout;
  validateCellDispatchBindings(
    graph,
    id,
    {positions: props.positions, ringOffsets: props.ringOffsets, ...props.output},
    [
      ['vertex reduction scratch', Math.max(positionCount, 1) * 12],
      ['ring center scratch', ringCount * 12],
      ['tile metadata scratch', tileCapacity * 16],
      ['winding scratch', batchCapacity * longRingCapacity * 16],
      ['tree cells and counts scratch', budget * 8],
      ['cell predicate scratch', batchCapacity * 16]
    ]
  );
  const nodes: GPUCommandNode<Parameters>[] = [];
  const allocate = <Format extends 'uint32' | 'uint32x2' | 'uint32x4' | 'float32' | 'float32x3'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${id}-${name}`, format, length);
  const control = allocate('control', 'uint32', CONTROL_WORD_COUNT);
  const totals = allocate('level-counts', 'uint32', rounds + 1);
  const owners = allocate('vertex-rings', 'uint32', Math.max(positionCount, 1));
  const sums = [
    allocate('vertex-sums-a', 'float32x3', Math.max(positionCount, 1)),
    allocate('vertex-sums-b', 'float32x3', Math.max(positionCount, 1))
  ];
  const areas = [
    allocate('vertex-areas-a', 'float32', Math.max(positionCount, 1)),
    allocate('vertex-areas-b', 'float32', Math.max(positionCount, 1))
  ];
  const centers = allocate('ring-centers', 'float32x3', ringCount);
  const orientation = allocate('ring-orientation', 'uint32', ringCount);
  const longCounts = allocate('long-ring-counts', 'uint32', ringCount);
  const longOffsets = allocate('long-ring-offsets', 'uint32', ringCount);
  const tileCounts = allocate('tile-counts', 'uint32', ringCount);
  const tileOffsets = allocate('tile-offsets', 'uint32', ringCount);
  const longRings = allocate('long-rings', 'uint32', longRingCapacity);
  const tiles = allocate('tiles', 'uint32x4', tileCapacity);
  const positions = allocate('oriented-positions', 'float32x3', Math.max(positionCount, 1));
  const state = allocate('cell-state', 'uint32x4', batchCapacity);
  const winding = allocate('cell-ring-winding', 'uint32x4', batchCapacity * longRingCapacity);
  const levels = Array.from({length: rounds}, (_, level) => ({
    cells: allocate(`tree-cells-${level}`, 'uint32x2', budget),
    parents: allocate(`tree-parents-${level}`, 'uint32', budget),
    codes: allocate(`tree-codes-${level}`, 'uint32', budget),
    counts: allocate(`tree-counts-${level}`, 'uint32', budget),
    offsets: allocate(`tree-offsets-${level}`, 'uint32', budget),
    subtree: allocate(`tree-subtree-${level}`, 'uint32x2', budget),
    ranks: allocate(`tree-ranks-${level}`, 'uint32x2', budget)
  }));
  const roundParameters = allocate(
    'round-parameters',
    'uint32',
    rounds * ROUND_PARAMETER_SLOT_WORD_COUNT
  );
  const roundSlots = levels.map((_, level) =>
    graph.createDataView(roundParameters.buffer, {
      format: 'uint32',
      length: 2,
      byteOffset: level * ROUND_PARAMETER_SLOT_WORD_COUNT * Uint32Array.BYTES_PER_ELEMENT
    })
  );
  const bind = (
    name: string,
    view: GraphDataView,
    type: 'u32' | 'f32' | 'atomic<u32>' = 'u32',
    access: 'read' | 'read_write' = 'read'
  ): WGSLKernelBinding => ({name, view, type, access});
  const controlAtomic = () => bind('control', control, 'atomic<u32>', 'read_write');
  const controlRead = () => bind('control', control);
  const originalPosition = () => bind('positions', props.positions, 'f32');
  const preparedPosition = () => bind('positions', positions, 'f32');
  const ranges = () => bind('ranges', props.ringOffsets);
  const geometry = getS2CoveringParallelWGSL(props.tolerance ?? 1e-6);
  const geometryWithoutPositions = getS2CoveringParallelWGSL(props.tolerance ?? 1e-6, false);
  const add = (
    name: string,
    count: number,
    bindings: WGSLKernelBinding[],
    body: string,
    declarations = ''
  ) => {
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-${name}`,
        operation: 'GPUS2PolygonCovering',
        variant: props.refinement ?? 'serial',
        invocationCount: count,
        bindings,
        body,
        declarations
      })
    );
  };
  const addRows = (
    name: string,
    rows: number,
    columns: number,
    bindings: WGSLKernelBinding[],
    body: (rowBase: number) => string,
    declarations = ''
  ) => {
    const rowChunk = Math.max(1, Math.floor(MAXIMUM_INVOCATION_COUNT / columns));
    for (let rowBase = 0; rowBase < rows; rowBase += rowChunk)
      add(
        `${name}-${rowBase}`,
        Math.min(rowChunk, rows - rowBase) * columns,
        bindings,
        body(rowBase),
        declarations
      );
  };
  add(
    'initialize',
    Math.max(props.capacity, ROOT_COUNT, CONTROL_WORD_COUNT, rounds + 1),
    [
      bind('cells', props.output.cells, 'u32', 'read_write'),
      bind('control', control, 'u32', 'read_write'),
      bind('totals', totals, 'u32', 'read_write'),
      bind('roots', levels[0].cells, 'u32', 'read_write'),
      bind('roundParameters', roundParameters, 'u32', 'read_write')
    ],
    `if(index<${props.capacity}u){cells[cellsOffset+index*2u]=0u;cells[cellsOffset+index*2u+1u]=0u;}
if(index<${CONTROL_WORD_COUNT}u){control[controlOffset+index]=0u;}
if(index<${rounds}u){let offset=roundParametersOffset+index*${ROUND_PARAMETER_SLOT_WORD_COUNT}u;roundParameters[offset]=index;roundParameters[offset+1u]=${props.maximumLevel}u;}
if(index<${rounds + 1}u){totals[totalsOffset+index]=0u;}
if(index==0u){totals[totalsOffset]=${ROOT_COUNT}u;}
if(index<${ROOT_COUNT}u){roots[rootsOffset+index*2u]=0u;roots[rootsOffset+index*2u+1u]=(2u*index+1u)<<28u;}`
  );
  add(
    'validate-ranges',
    ringCount,
    [ranges(), controlAtomic()],
    `let begin=ranges[rangesOffset+index];let end=ranges[rangesOffset+index+1u];
if(begin>end||end>${positionCount}u||end-begin<3u){atomicOr(&control[controlOffset],1u);}`
  );
  add(
    'prepare-vertices',
    Math.max(positionCount, 1),
    [
      originalPosition(),
      ranges(),
      bind('owners', owners, 'u32', 'read_write'),
      bind('sums', sums[0], 'f32', 'read_write'),
      controlAtomic()
    ],
    `
if(index>=${positionCount}u){return;}
var owner=0u;var point=vec3f(0.0);
if(atomicLoad(&control[controlOffset])==0u){
  var lower=0u;var upper=${ringCount}u;
  for(var step=0u;step<${MAXIMUM_S2_COVERING_RANGE_SEARCH_STEPS}u&&lower<upper;step++){
    let middle=lower+(upper-lower)/2u;
    if(ranges[rangesOffset+middle+1u]<=index){lower=middle+1u;}else{upper=middle;}
  }
  if(lower<${ringCount}u&&index>=ranges[rangesOffset+lower]&&index<ranges[rangesOffset+lower+1u]){
    owner=lower+1u;point=coverPosition(index);let squared=dot(point,point);
    if((bitcast<u32>(squared)&0x7f800000u)==0x7f800000u||abs(squared-1.0)>1e-4){atomicOr(&control[controlOffset],1u);point=vec3f(0.0);}
  }
}
owners[ownersOffset+index]=owner;let offset=sumsOffset+index*3u;
sums[offset]=point.x;sums[offset+1u]=point.y;sums[offset+2u]=point.z;`,
    geometry
  );
  let sumIndex = 0;
  for (let distance = 1; distance < positionCount; distance *= 2) {
    const source = sums[sumIndex];
    sumIndex = 1 - sumIndex;
    add(
      `reduce-center-${distance}`,
      positionCount,
      [
        bind('source', source, 'f32'),
        bind('destination', sums[sumIndex], 'f32', 'read_write'),
        bind('owners', owners)
      ],
      `
let offset=sourceOffset+index*3u;var value=vec3f(source[offset],source[offset+1u],source[offset+2u]);
if(index>=${distance}u&&owners[ownersOffset+index]==owners[ownersOffset+index-${distance}u]){
  let previous=sourceOffset+(index-${distance}u)*3u;value+=vec3f(source[previous],source[previous+1u],source[previous+2u]);}
let destinationIndex=destinationOffset+index*3u;destination[destinationIndex]=value.x;destination[destinationIndex+1u]=value.y;destination[destinationIndex+2u]=value.z;`
    );
  }
  add(
    'prepare-centers',
    ringCount,
    [
      ranges(),
      bind('sums', sums[sumIndex], 'f32'),
      bind('centers', centers, 'f32', 'read_write'),
      controlAtomic()
    ],
    `
if(atomicLoad(&control[controlOffset])!=0u){return;}
let end=ranges[rangesOffset+index+1u];let offset=sumsOffset+(end-1u)*3u;
var pole=vec3f(sums[offset],sums[offset+1u],sums[offset+2u]);
${props.hemisphereCenter ? `pole=vec3f(${props.hemisphereCenter.join(',')});` : ''}
if(length(pole)<=${props.tolerance ?? 1e-6}){atomicOr(&control[controlOffset],1u);return;}
pole=normalize(pole);let destinationIndex=centersOffset+index*3u;centers[destinationIndex]=pole.x;centers[destinationIndex+1u]=pole.y;centers[destinationIndex+2u]=pole.z;`
  );
  add(
    'prepare-area',
    Math.max(positionCount, 1),
    [
      originalPosition(),
      ranges(),
      bind('owners', owners),
      bind('centers', centers, 'f32'),
      bind('areas', areas[0], 'f32', 'read_write'),
      controlRead()
    ],
    `
if(index>=${positionCount}u){return;}var area=0.0;let owner=owners[ownersOffset+index];
if(control[controlOffset]==0u&&owner!=0u){
  let ring=owner-1u;let begin=ranges[rangesOffset+ring];let end=ranges[rangesOffset+ring+1u];
  if(index>begin&&index+1u<end){let offset=centersOffset+ring*3u;let pole=vec3f(centers[offset],centers[offset+1u],centers[offset+2u]);
    let anchor=coverPosition(begin);let first=coverPosition(index);let second=coverPosition(index+1u);
    let firstDifference=first-anchor;let secondDifference=second-anchor;
    let firstDelta=(firstDifference-anchor*(dot(pole,firstDifference)/dot(pole,anchor)))/dot(pole,first);
    let secondDelta=(secondDifference-anchor*(dot(pole,secondDifference)/dot(pole,anchor)))/dot(pole,second);
    area=dot(pole,cross(firstDelta,secondDelta));
  }
}areas[areasOffset+index]=area;`,
    geometry
  );
  let areaIndex = 0;
  for (let distance = 1; distance < positionCount; distance *= 2) {
    const source = areas[areaIndex];
    areaIndex = 1 - areaIndex;
    add(
      `reduce-area-${distance}`,
      positionCount,
      [
        bind('source', source, 'f32'),
        bind('destination', areas[areaIndex], 'f32', 'read_write'),
        bind('owners', owners)
      ],
      `
var value=source[sourceOffset+index];
if(index>=${distance}u&&owners[ownersOffset+index]==owners[ownersOffset+index-${distance}u]){value+=source[sourceOffset+index-${distance}u];}
destination[destinationOffset+index]=value;`
    );
  }
  add(
    'orient-rings',
    ringCount,
    [
      ranges(),
      bind('areas', areas[areaIndex], 'f32'),
      bind('orientation', orientation, 'u32', 'read_write'),
      bind('longCounts', longCounts, 'u32', 'read_write'),
      bind('tileCounts', tileCounts, 'u32', 'read_write'),
      controlAtomic()
    ],
    `
orientation[orientationOffset+index]=0u;longCounts[longCountsOffset+index]=0u;tileCounts[tileCountsOffset+index]=0u;
if(atomicLoad(&control[controlOffset])!=0u){return;}
let begin=ranges[rangesOffset+index];let end=ranges[rangesOffset+index+1u];let area=areas[areasOffset+end-1u];
if((bitcast<u32>(area)&0x7f800000u)==0x7f800000u||area==0.0){atomicOr(&control[controlOffset],1u);return;}
orientation[orientationOffset+index]=select(0u,1u,area<0.0);
if(end-begin>${tile}u){longCounts[longCountsOffset+index]=1u;tileCounts[tileCountsOffset+index]=(end-begin+${tile - 1}u)/${tile}u;}`
  );
  nodes.push(
    ...new GPUScan({
      id: `${id}-scan-long-rings`,
      input: longCounts,
      output: longOffsets,
      mode: 'exclusive'
    }).getCommandNodes(graph)
  );
  nodes.push(
    ...new GPUScan({
      id: `${id}-scan-input-tiles`,
      input: tileCounts,
      output: tileOffsets,
      mode: 'exclusive'
    }).getCommandNodes(graph)
  );
  add(
    'index-long-rings',
    ringCount,
    [
      bind('longCounts', longCounts),
      bind('longOffsets', longOffsets),
      bind('tileCounts', tileCounts),
      bind('tileOffsets', tileOffsets),
      bind('longRings', longRings, 'u32', 'read_write'),
      bind('control', control, 'u32', 'read_write')
    ],
    `
if(longCounts[longCountsOffset+index]!=0u){longRings[longRingsOffset+longOffsets[longOffsetsOffset+index]]=index;}
if(index==${ringCount - 1}u){control[controlOffset+5u]=longOffsets[longOffsetsOffset+index]+longCounts[longCountsOffset+index];control[controlOffset+6u]=tileOffsets[tileOffsetsOffset+index]+tileCounts[tileCountsOffset+index];}`
  );
  add(
    'describe-input-tiles',
    tileCapacity,
    [
      ranges(),
      bind('tileOffsets', tileOffsets),
      bind('longOffsets', longOffsets),
      bind('tiles', tiles, 'u32', 'read_write'),
      controlRead()
    ],
    `
let destinationIndex=tilesOffset+index*4u;tiles[destinationIndex]=0xffffffffu;tiles[destinationIndex+1u]=0u;tiles[destinationIndex+2u]=0u;tiles[destinationIndex+3u]=0u;
if(control[controlOffset]!=0u||index>=control[controlOffset+6u]){return;}
var lower=0u;var upper=${ringCount}u;
for(var step=0u;step<${MAXIMUM_S2_COVERING_RANGE_SEARCH_STEPS}u&&lower<upper;step++){
  let middle=lower+(upper-lower)/2u;
  if(tileOffsets[tileOffsetsOffset+middle]<=index){lower=middle+1u;}else{upper=middle;}
}
let ring=lower-1u;tiles[destinationIndex]=ring;tiles[destinationIndex+1u]=longOffsets[longOffsetsOffset+ring];
tiles[destinationIndex+2u]=ranges[rangesOffset+ring]+(index-tileOffsets[tileOffsetsOffset+ring])*${tile}u;`
  );
  add(
    'copy-oriented',
    Math.max(positionCount, 1),
    [
      originalPosition(),
      ranges(),
      bind('owners', owners),
      bind('orientation', orientation),
      bind('oriented', positions, 'f32', 'read_write')
    ],
    `
if(index>=${positionCount}u){return;}var source=index;let owner=owners[ownersOffset+index];
if(owner!=0u){let ring=owner-1u;if(orientation[orientationOffset+ring]!=0u){let begin=ranges[rangesOffset+ring];let end=ranges[rangesOffset+ring+1u];source=end-1u-(index-begin);}}
let point=coverPosition(source);let offset=orientedOffset+index*3u;oriented[offset]=point.x;oriented[offset+1u]=point.y;oriented[offset+2u]=point.z;`,
    geometry
  );
  add(
    'validate-edges',
    Math.max(positionCount, 1),
    [
      preparedPosition(),
      ranges(),
      bind('owners', owners),
      bind('centers', centers, 'f32'),
      controlAtomic()
    ],
    `
if(index>=${positionCount}u||atomicLoad(&control[controlOffset])!=0u){return;}let owner=owners[ownersOffset+index];if(owner==0u){return;}
let ring=owner-1u;let begin=ranges[rangesOffset+ring];let end=ranges[rangesOffset+ring+1u];let offset=centersOffset+ring*3u;
let center=vec3f(centers[offset],centers[offset+1u],centers[offset+2u]);let first=coverPosition(index);let second=coverPosition(select(index+1u,begin,index+1u==end));
if(dot(center,first)<=COVER_TOLERANCE||length(coverEdgeCross(first,second))<=COVER_TOLERANCE){atomicOr(&control[controlOffset],1u);}`,
    geometry
  );
  const comparisonTiles = Math.max(1, Math.ceil(positionCount / tile));
  for (let firstBase = 0; firstBase < Math.max(positionCount, 1); firstBase += layout.rowChunks) {
    const rows = Math.min(layout.rowChunks, Math.max(positionCount, 1) - firstBase);
    add(
      `validate-pairs-${firstBase}`,
      rows * comparisonTiles,
      [preparedPosition(), ranges(), bind('owners', owners), controlAtomic()],
      `
let vertex=${firstBase}u+index/${comparisonTiles}u;
if(vertex>=${positionCount}u||atomicLoad(&control[controlOffset])!=0u){return;}
let owner=owners[ownersOffset+vertex];if(owner==0u){return;}let ring=owner-1u;
let begin=ranges[rangesOffset+ring];let end=ranges[rangesOffset+ring+1u];let next=select(vertex+1u,begin,vertex+1u==end);
let first=coverPosition(vertex);let second=coverPosition(next);let start=(index%${comparisonTiles}u)*${tile}u;
for(var slot=0u;slot<${tile}u;slot++){
  let other=start+slot;if(other>=end){break;}if(other<=vertex||other<begin){continue;}
  let otherNext=select(other+1u,begin,other+1u==end);if(other==next||otherNext==vertex){continue;}
  if(coverArcsIntersect(first,second,coverPosition(other),coverPosition(otherNext))){atomicOr(&control[controlOffset],1u);}
}`,
      geometry
    );
  }
  for (let level = 0; level < rounds; level++) {
    const current = levels[level];
    if (props.refinement === 'level-synchronous')
      add(
        `gate-round-${level}`,
        1,
        [bind('control', control, 'u32', 'read_write'), bind('totals', totals)],
        `
if(control[controlOffset]!=0u){return;}let count=totals[totalsOffset+${level}u];
if(count+control[controlOffset+2u]>${budget}u){control[controlOffset]|=4u;return;}
if(count+control[controlOffset+1u]>${budget}u){control[controlOffset]|=4u;control[controlOffset+1u]=${budget}u;return;}
control[controlOffset+1u]+=count;`
      );
    for (let batchStart = 0; batchStart < budget; batchStart += batchCapacity) {
      const batchCount = Math.min(batchCapacity, budget - batchStart);
      const prefix = `round-${level}-batch-${batchStart}`;
      const live = `control[controlOffset]==0u&&${batchStart}u+row<min(totals[totalsOffset+${level}u],${budget}u)`;
      const cellRead = `let cellOffset=cellsOffset+(${batchStart}u+row)*2u;let cell=vec2u(cells[cellOffset+1u],cells[cellOffset]);`;
      const centerRead = `let centerOffset=centersOffset+ring*3u;let pole=vec3f(centers[centerOffset],centers[centerOffset+1u],centers[centerOffset+2u]);`;
      add(
        `${prefix}-clear`,
        Math.max(batchCount, batchCount * longRingCapacity),
        [bind('state', state, 'u32', 'read_write'), bind('winding', winding, 'u32', 'read_write')],
        `if(index<${batchCount}u){let offset=stateOffset+index*4u;state[offset]=0u;state[offset+1u]=0u;state[offset+2u]=0u;state[offset+3u]=0u;}
let offset=windingOffset+index*4u;winding[offset]=0u;winding[offset+1u]=0u;winding[offset+2u]=0u;winding[offset+3u]=0u;`
      );
      const shortRowChunk = Math.max(1, Math.floor(MAXIMUM_INVOCATION_COUNT / ringCount));
      for (let rowBase = 0; rowBase < batchCount; rowBase += shortRowChunk) {
        add(
          `${prefix}-short-rings-${rowBase}`,
          Math.min(shortRowChunk, batchCount - rowBase) * ringCount,
          [
            preparedPosition(),
            ranges(),
            bind('centers', centers, 'f32'),
            bind('cells', current.cells),
            bind('state', state, 'atomic<u32>', 'read_write'),
            controlRead(),
            bind('totals', totals)
          ],
          `
let row=${rowBase}u+index/${ringCount}u;let ring=index%${ringCount}u;
if(!(${live})){return;}let begin=ranges[rangesOffset+ring];let end=ranges[rangesOffset+ring+1u];if(end-begin>${tile}u){return;}
${cellRead}${centerRead}
let contribution=coverRingContribution(cell,begin,end,begin,end);let mask=coverPositiveWindingMask(cell,pole,contribution.winding);
let offset=stateOffset+row*4u;atomicOr(&state[offset+select(1u,0u,ring==0u)],mask);atomicOr(&state[offset+2u],contribution.boundary);atomicOr(&state[offset+3u],contribution.touched);`,
          geometry
        );
      }
      addRows(
        `${prefix}-long-tiles`,
        batchCount,
        tileCapacity,
        [
          preparedPosition(),
          ranges(),
          bind('tiles', tiles),
          bind('cells', current.cells),
          bind('state', state, 'atomic<u32>', 'read_write'),
          bind('winding', winding, 'atomic<u32>', 'read_write'),
          controlRead(),
          bind('totals', totals)
        ],
        base => `
let row=${base}u+index/${tileCapacity}u;let descriptor=tilesOffset+(index%${tileCapacity}u)*4u;
if(!(${live})){return;}let ring=tiles[descriptor];if(ring==0xffffffffu){return;}
let longIndex=tiles[descriptor+1u];let begin=ranges[rangesOffset+ring];let end=ranges[rangesOffset+ring+1u];let start=tiles[descriptor+2u];
${cellRead}
let contribution=coverRingContribution(cell,begin,end,start,min(end,start+${tile}u));let offset=stateOffset+row*4u;
atomicOr(&state[offset+2u],contribution.boundary);atomicOr(&state[offset+3u],contribution.touched);
let destinationIndex=windingOffset+(row*${longRingCapacity}u+longIndex)*4u;
for(var corner=0u;corner<4u;corner++){atomicAdd(&winding[destinationIndex+corner],bitcast<u32>(contribution.winding[corner]));}`,
        geometry
      );
      addRows(
        `${prefix}-finish-long-rings`,
        batchCount,
        longRingCapacity,
        [
          bind('longRings', longRings),
          bind('centers', centers, 'f32'),
          bind('cells', current.cells),
          bind('winding', winding),
          bind('state', state, 'atomic<u32>', 'read_write'),
          controlRead(),
          bind('totals', totals)
        ],
        base => `
let row=${base}u+index/${longRingCapacity}u;let longIndex=index%${longRingCapacity}u;
if(!(${live})||longIndex>=control[controlOffset+5u]){return;}let ring=longRings[longRingsOffset+longIndex];
${cellRead}${centerRead}
let offset=windingOffset+(row*${longRingCapacity}u+longIndex)*4u;
let degree=vec4i(bitcast<i32>(winding[offset]),bitcast<i32>(winding[offset+1u]),bitcast<i32>(winding[offset+2u]),bitcast<i32>(winding[offset+3u]));
let mask=coverPositiveWindingMask(cell,pole,degree);atomicOr(&state[stateOffset+row*4u+select(1u,0u,ring==0u)],mask);`,
        geometryWithoutPositions
      );
      add(
        `${prefix}-classify`,
        batchCount,
        [
          bind('state', state),
          bind('codes', current.codes, 'u32', 'read_write'),
          bind('counts', current.counts, 'u32', 'read_write'),
          controlAtomic(),
          bind('totals', totals),
          bind('roundParameters', roundSlots[level])
        ],
        `
let currentLevel=roundParameters[roundParametersOffset];let maximumLevel=roundParameters[roundParametersOffset+1u];
let row=index;let destinationIndex=${batchStart}u+row;var code=0u;
if(atomicLoad(&control[controlOffset])==0u&&${batchStart}u+row<min(totals[totalsOffset+currentLevel],${budget}u)){
  let offset=stateOffset+row*4u;var inside=state[offset]&(~state[offset+1u])&15u;
  ${props.complement ? 'inside^=15u;' : ''}
  let boundary=state[offset+2u];let touched=state[offset+3u]!=0u;
  let intersects=inside!=0u||boundary!=0u||touched;let contained=inside==15u&&boundary==0u&&!touched;
  if(intersects){
    if(currentLevel>=${props.minimumLevel ?? 0}u&&(contained||(${props.mode !== 'interior'}&&currentLevel==maximumLevel))){code=1u;${props.refinement === 'level-synchronous' ? 'atomicAdd(&control[controlOffset+2u],1u);' : ''}}
    else if(currentLevel<maximumLevel){code=4u;}
  }
}
codes[codesOffset+destinationIndex]=code;counts[countsOffset+destinationIndex]=select(0u,4u,code==4u);`
      );
    }
    nodes.push(
      ...new GPUScan({
        id: `${id}-tree-scan-${level}`,
        input: current.counts,
        output: current.offsets,
        mode: 'exclusive'
      }).getCommandNodes(graph)
    );
    const next = levels[level + 1];
    add(
      `expand-tree-${level}`,
      budget,
      [
        bind('cells', current.cells),
        bind('counts', current.counts),
        bind('offsets', current.offsets),
        bind('totals', totals, 'u32', 'read_write'),
        ...(next
          ? [
              bind('children', next.cells, 'u32', 'read_write'),
              bind('parents', next.parents, 'u32', 'read_write')
            ]
          : [])
      ],
      `
let count=counts[countsOffset+index];let start=offsets[offsetsOffset+index];
if(index==${budget - 1}u){totals[totalsOffset+${level + 1}u]=start+count;}
${
  next
    ? `if(count==4u){
 let offset=cellsOffset+index*2u;let cell=vec2u(cells[offset+1u],cells[offset]);
 let marker=dggs_u64_shift_left(vec2u(0u,1u),${60 - 2 * level}u);let childMarker=dggs_u64_shift_right(marker,2u);
 var child=dggs_u64_add(dggs_u64_subtract(cell,marker),childMarker);let step=dggs_u64_shift_left(childMarker,1u);
 for(var slot=0u;slot<4u;slot++){
   if(start+slot<${budget}u){let destinationIndex=childrenOffset+(start+slot)*2u;children[destinationIndex]=child.y;children[destinationIndex+1u]=child.x;parents[parentsOffset+start+slot]=index;}
   child=dggs_u64_add(child,step);
 }
}`
    : ''
}`,
      geometryWithoutPositions
    );
  }
  for (let level = rounds - 1; level >= 0; level--) {
    const current = levels[level];
    const next = levels[level + 1];
    add(
      `subtree-${level}`,
      budget,
      [
        bind('codes', current.codes),
        bind('offsets', current.offsets),
        bind('totals', totals),
        bind('subtree', current.subtree, 'u32', 'read_write'),
        controlRead(),
        ...(next ? [bind('children', next.subtree)] : [])
      ],
      `
var count=0u;var emitted=0u;
if(control[controlOffset]==0u&&index<min(totals[totalsOffset+${level}u],${budget}u)){
 count=1u;let code=codes[codesOffset+index];if(code==1u){emitted=1u;}
 ${
   next
     ? `if(code==4u){let start=offsets[offsetsOffset+index];
   if(start+4u>${budget}u){count=${budget + 1}u;}
   else{for(var slot=0u;slot<4u;slot++){let offset=childrenOffset+(start+slot)*2u;count=min(${budget + 1}u,count+children[offset]);emitted+=children[offset+1u];}}
 }`
     : ''
 }
}
subtree[subtreeOffset+index*2u]=count;subtree[subtreeOffset+index*2u+1u]=emitted;`
    );
  }
  for (let level = 0; level < rounds; level++) {
    const current = levels[level];
    const previous = levels[level - 1];
    add(
      `publish-prefix-${level}`,
      budget,
      [
        bind('cells', current.cells),
        bind('subtree', current.subtree),
        bind('ranks', current.ranks, 'u32', 'read_write'),
        bind('outputCells', props.output.cells, 'u32', 'read_write'),
        controlAtomic(),
        bind('parents', current.parents),
        bind('previousRanks', previous?.ranks ?? current.subtree),
        bind('roundParameters', roundSlots[level])
      ],
      `
if(atomicLoad(&control[controlOffset])!=0u||subtree[subtreeOffset+index*2u]==0u){return;}
var visitRank=0u;var emitRank=0u;
let currentLevel=roundParameters[roundParametersOffset];
if(currentLevel==0u){
 for(var root=0u;root<${ROOT_COUNT}u;root++){if(root<index){visitRank=min(${budget + 1}u,visitRank+subtree[subtreeOffset+root*2u]);emitRank+=subtree[subtreeOffset+root*2u+1u];}}
 atomicAdd(&control[controlOffset+4u],subtree[subtreeOffset+index*2u]);
}else{
 let parent=parents[parentsOffset+index];visitRank=previousRanks[previousRanksOffset+parent*2u]+1u;emitRank=previousRanks[previousRanksOffset+parent*2u+1u];let sibling=index%4u;
 for(var slot=0u;slot<4u;slot++){if(slot<sibling){let offset=subtreeOffset+(index-sibling+slot)*2u;visitRank=min(${budget + 1}u,visitRank+subtree[offset]);emitRank+=subtree[offset+1u];}}
}
ranks[ranksOffset+index*2u]=visitRank;ranks[ranksOffset+index*2u+1u]=emitRank;
// A one-node subtree emitting one cell is exactly a selected leaf (code 1).
if(visitRank<${budget}u&&subtree[subtreeOffset+index*2u]==1u&&subtree[subtreeOffset+index*2u+1u]==1u){
 atomicAdd(&control[controlOffset+3u],1u);
 if(emitRank<${props.capacity}u){outputCells[outputCellsOffset+emitRank*2u]=cells[cellsOffset+index*2u];outputCells[outputCellsOffset+emitRank*2u+1u]=cells[cellsOffset+index*2u+1u];}
}`
    );
  }
  add(
    'publish',
    1,
    [
      controlRead(),
      bind('counts', props.output.counts, 'u32', 'read_write'),
      bind('status', props.output.status, 'u32', 'read_write'),
      ...(props.output.visitedCount
        ? [bind('visited', props.output.visitedCount, 'u32', 'read_write')]
        : [])
    ],
    `
var failure=control[controlOffset];var required=0u;var visitedCount=0u;
if(failure==0u){required=control[controlOffset+3u];${props.refinement === 'level-synchronous' ? 'visitedCount=control[controlOffset+1u];' : `visitedCount=min(control[controlOffset+4u],${budget}u);if(control[controlOffset+4u]>${budget}u){failure|=4u;}`}
 if(required>${props.capacity}u){failure|=2u;}
}else{visitedCount=control[controlOffset+1u];}
counts[countsOffset]=min(required,${props.capacity}u);counts[countsOffset+1u]=required;status[statusOffset]=failure;
${props.output.visitedCount ? 'visited[visitedOffset]=visitedCount;' : ''}`
  );
  validateRequirement(
    id,
    nodes.length === getS2CoveringParallelNodeCount(props),
    `parallel command-node estimate ${getS2CoveringParallelNodeCount(props)} must equal emitted ${nodes.length}`
  );
  return nodes;
}
