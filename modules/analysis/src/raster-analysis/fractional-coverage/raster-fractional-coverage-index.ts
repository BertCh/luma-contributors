// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  GPUCompaction,
  type GPUCommandGraph,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {getGraphOutputViews} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPUPolygonTriangulation,
  getPolygonTriangulationIndexCount,
  getPolygonTriangulationNodeCount,
  validatePolygonTriangulationScratch
} from '../../analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import {createSegmentSumNode} from '../../analysis-core/kernels/sorted-segment-sums';
import {
  createSaturatedCountScanNodes,
  SCAN_TOTAL_WGSL
} from '../../analysis-core/kernels/scan-total';
import {createWGSLKernelNode, createFillNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {validateRasterExpansionGraph} from '../raster-expansion-utils';
import {
  validateRasterFractionalBindingByteLength,
  createRasterFractionalTileNode
} from './raster-fractional-tile-kernel';
import type {GPURasterFractionalCoverage} from '../gpu-raster-fractional-coverage';
import {RASTER_FRACTIONAL_INVERSE_WGSL} from './raster-fractional-inverse-wgsl';

/** Upper bound on clipping scratch; final input tiles bind and dispatch only their actual rows. */
export const RASTER_FRACTIONAL_CANDIDATE_TILE_SIZE = 262144;

/** Preflight every caller binding and derived row-sized allocation before graph mutation. @internal */
export function validateFractionalCoverageBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPURasterFractionalCoverage
): void {
  const {id, props} = contributor;
  const {zones} = props;
  for (const view of [
    ...Object.values(zones),
    props.parameters,
    ...getGraphOutputViews(props.output)
  ]) {
    validateRasterFractionalBindingByteLength(
      graph.device,
      id,
      view.buffer.id,
      getViewBindingRange(view).size,
      'split the source geometry or output into disjoint cores',
      view.buffer.byteLength
    );
  }
  const polygonCount = zones.polygonOffsets.length - 1;
  const zoneCount = zones.featureOffsets.length - 1;
  const vertexCount = zones.positions.length;
  validatePolygonTriangulationScratch(
    graph.device,
    id,
    vertexCount,
    zones.ringOffsets.length - 1,
    polygonCount
  );
  for (const [name, bytes] of [
    [
      'indices',
      getPolygonTriangulationIndexCount(vertexCount, zones.ringOffsets.length - 1, polygonCount) * 4
    ],
    ['triangle offsets', (polygonCount + 1) * 4],
    ['polygon validity', polygonCount * 4],
    [
      'triangulation scratch',
      Math.max(
        7 *
          getPolygonTriangulationNodeCount(vertexCount, zones.ringOffsets.length - 1, polygonCount),
        4
      ) * 4
    ],
    ['pixel coordinates', props.searchStructure === 'brute-force' ? 0 : vertexCount * 4],
    ['zone offsets and scan', props.searchStructure === 'brute-force' ? 0 : (zoneCount + 1) * 4],
    ['zone bounds', props.searchStructure === 'brute-force' ? 0 : zoneCount * 16],
    [
      'candidate tile',
      props.searchStructure === 'brute-force' ? 0 : RASTER_FRACTIONAL_CANDIDATE_TILE_SIZE * 8
    ],
    [
      'dense pairs',
      props.searchStructure === 'brute-force'
        ? zoneCount * props.grid.width * props.grid.height * 4
        : 0
    ]
  ] as const) {
    validateRasterFractionalBindingByteLength(
      graph.device,
      id,
      name,
      Math.max(4, bytes),
      'split the source geometry into disjoint cores'
    );
  }
}

/** Shared exact clipping and affine inversion used by both coverage routes. @internal */
export const RASTER_FRACTIONAL_CLIPPING_WGSL = `${RASTER_FRACTIONAL_INVERSE_WGSL}
fn setting(slot:u32)->f32{return settings[settingsOffset+slot];}
// A non-identity pixel coordinate within four float32 steps of an integer is that integer,
// capped at 1e-5 pixel so representable subpixel polygons survive large pixel indices. The
// decision uses the displacement from the nearest lattice point, a local subpixel coordinate.
// A vertex meant to
// lie on a cell boundary does not come out of the inverse affine as an integer: the vertex and
// the cell size are each rounded to float32 (float32(0.9) / float32(0.3) is 2.9999998 in exact
// arithmetic), the products and the quotient round again, and GPU division need not be correctly
// rounded. Unsnapped, such a vertex leaves a sliver of coverage (of the order of 1e-7 of a cell)
// in the neighbouring cell, which gives a zone support in a cell it does not touch. The snap moves no
// vertex by more than 1e-5 pixel. Exact identity transforms bypass snapping entirely. It cannot absorb the
// rounding of 'position - origin', which is relative to the origin: keep the origin near the
// geometry (local coordinates).
fn snapToLattice(pixel:vec2<f32>)->vec2<f32>{let nearest=round(pixel-vec2<f32>(FRACTIONAL_BOUNDARY_SHIFT))+vec2<f32>(FRACTIONAL_BOUNDARY_SHIFT);let tolerance=min(vec2<f32>(1.0e-5),4.76837158203125e-7*max(abs(nearest),vec2<f32>(1.0)));return select(pixel,nearest,abs(pixel-nearest)<=tolerance);}
// Pixel coordinates of a vertex. On an axis-aligned grid (b = d = 0) each axis is one division,
// which rounds less than the general form (two products, a difference and a quotient).
// Word 6 retains the original-grid area determinant. Invert the actual packed axes instead.
fn point(vertex:u32)->vec2<f32>{let x=positions[positionsOffset+vertex*2u]-setting(0u);let y=positions[positionsOffset+vertex*2u+1u]-setting(1u);if(setting(0u)==0.0&&setting(1u)==0.0&&setting(2u)==1.0&&setting(3u)==0.0&&setting(4u)==0.0&&setting(5u)==1.0){return vec2<f32>(x,y);}if(setting(3u)==0.0&&setting(4u)==0.0){return snapToLattice(vec2<f32>(fractionalDivide(fractionalValue(x),fractionalValue(setting(2u))),fractionalDivide(fractionalValue(y),fractionalValue(setting(5u)))));}let determinant=fractionalProductDifference(setting(2u),setting(5u),setting(3u),setting(4u));let column=fractionalProductDifference(setting(5u),x,setting(3u),y);let row=fractionalProductDifference(setting(2u),y,setting(4u),x);return snapToLattice(vec2<f32>(fractionalDivide(column,determinant),fractionalDivide(row,determinant)));}
fn clip(vertices:ptr<function,array<vec2<f32>,8>>,count:u32,axis:u32,bound:f32,lower:bool)->u32{
 var result:array<vec2<f32>,8>;var size=0u;if(count==0u){return 0u;}var previous=(*vertices)[count-1u];var previousInside=select(previous[axis]<=bound,previous[axis]>=bound,lower);
 for(var cursor=0u;cursor<count;cursor++){let current=(*vertices)[cursor];let inside=select(current[axis]<=bound,current[axis]>=bound,lower);if(inside!=previousInside){let fraction=(bound-previous[axis])/(current[axis]-previous[axis]);var intersection=mix(previous,current,fraction);intersection[axis]=bound;result[size]=intersection;size++;}if(inside){result[size]=current;size++;}previous=current;previousInside=inside;}
 for(var cursor=0u;cursor<size;cursor++){(*vertices)[cursor]=result[cursor];}return size;
}`;

/** Selects the cell-boundary lattice while keeping exact identity coordinates untouched. @internal */
export function getRasterFractionalClippingWGSL(pixelInterpretation: 'area' | 'point'): string {
  return `const FRACTIONAL_BOUNDARY_SHIFT: f32 = ${pixelInterpretation === 'point' ? '-0.5' : '0.0'};\n${RASTER_FRACTIONAL_CLIPPING_WGSL}`;
}

/** Bounds-indexed exact clipping with stable, reusable candidate and compaction tiles. @internal */
export function getIndexedRasterFractionalCoverageNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPURasterFractionalCoverage
): GPUCommandNode<Parameters>[] {
  const {id, props, candidateCapacity} = contributor;
  const {zones, grid, output} = props;
  const zoneCount = zones.featureOffsets.length - 1;
  const polygonCount = zones.polygonOffsets.length - 1;
  const operation = 'GPURasterFractionalCoverage';
  validateRasterExpansionGraph(id, graph, [
    ...Object.values(zones),
    props.parameters,
    ...getGraphOutputViews(output)
  ]);
  const view = <Format extends 'uint32' | 'uint32x2' | 'uint32x4' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => createTransientView(graph, `${id}-${name}`, format, length);
  const indices = view(
    'indices',
    'uint32',
    getPolygonTriangulationIndexCount(
      zones.positions.length,
      zones.ringOffsets.length - 1,
      polygonCount
    )
  );
  const triangleOffsets = view('triangle-offsets', 'uint32', polygonCount + 1);
  const polygonValidity = view('polygon-validity', 'uint32', polygonCount);
  const polygonValidityFloat = view('polygon-validity-float', 'float32', polygonCount);
  const zoneValidityFloat = view('zone-validity-float', 'float32', zoneCount);
  const vertexOffsets = view('vertex-offsets', 'uint32', zoneCount + 1);
  const pixelX = view('pixel-x', 'float32', zones.positions.length);
  const pixelY = view('pixel-y', 'float32', zones.positions.length);
  const minimumX = view('minimum-x', 'float32', zoneCount);
  const minimumY = view('minimum-y', 'float32', zoneCount);
  const maximumX = view('maximum-x', 'float32', zoneCount);
  const maximumY = view('maximum-y', 'float32', zoneCount);
  const bounds = view('bounds', 'uint32x4', zoneCount);
  const counts = view('candidate-counts', 'uint32', zoneCount);
  const offsets = view('candidate-offsets', 'uint32', zoneCount);
  const candidateTotal = view('candidate-total', 'uint32', 1);
  // State words: next candidate base and total positive support processed so far (saturated).
  const state = view('tile-state', 'uint32', 2);
  const pairs = view('tile-pairs', 'uint32x2', RASTER_FRACTIONAL_CANDIDATE_TILE_SIZE);
  const fractions = view('tile-fractions', 'float32', RASTER_FRACTIONAL_CANDIDATE_TILE_SIZE);
  const pairIds = view('tile-ids', 'uint32', RASTER_FRACTIONAL_CANDIDATE_TILE_SIZE);
  const flags = view('tile-flags', 'uint32', RASTER_FRACTIONAL_CANDIDATE_TILE_SIZE);
  const compact = view('tile-compact', 'uint32', RASTER_FRACTIONAL_CANDIDATE_TILE_SIZE);
  const tileCount = view('tile-count', 'uint32', 1);
  const nodes: GPUCommandNode<Parameters>[] = [
    ...new GPUPolygonTriangulation({
      ...zones,
      id: `${id}-triangulate`,
      output: {indices, triangleOffsets, valid: polygonValidity}
    }).getCommandNodes(graph),
    createWGSLKernelNode(graph, {
      id: `${id}-polygon-validity-float`,
      operation,
      invocationCount: polygonCount,
      bindings: [
        {name: 'validity', view: polygonValidity, type: 'u32', access: 'read'},
        {name: 'floatValidity', view: polygonValidityFloat, type: 'f32', access: 'read_write'}
      ],
      body: 'floatValidity[floatValidityOffset+index]=f32(validity[validityOffset+index]);'
    }),
    createSegmentSumNode(graph, {
      id: `${id}-feature-validity`,
      operation,
      input: polygonValidityFloat,
      output: zoneValidityFloat,
      segmentCount: zoneCount,
      segmentOffsets: zones.featureOffsets,
      reduction: 'min'
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-vertex-offsets`,
      operation,
      invocationCount: zoneCount + 1,
      bindings: [
        {name: 'features', view: zones.featureOffsets, type: 'u32', access: 'read'},
        {name: 'polygons', view: zones.polygonOffsets, type: 'u32', access: 'read'},
        {name: 'rings', view: zones.ringOffsets, type: 'u32', access: 'read'},
        {name: 'offsets', view: vertexOffsets, type: 'u32', access: 'read_write'}
      ],
      body: 'offsets[offsetsOffset+index]=rings[ringsOffset+polygons[polygonsOffset+features[featuresOffset+index]]];'
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-pixel-coordinates`,
      operation,
      invocationCount: zones.positions.length,
      bindings: [
        {name: 'positions', view: zones.positions, type: 'f32', access: 'read'},
        {name: 'settings', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'pixelX', view: pixelX, type: 'f32', access: 'read_write'},
        {name: 'pixelY', view: pixelY, type: 'f32', access: 'read_write'}
      ],
      declarations: getRasterFractionalClippingWGSL(grid.pixelInterpretation),
      body: 'let pixel=point(index);pixelX[pixelXOffset+index]=pixel.x;pixelY[pixelYOffset+index]=pixel.y;'
    })
  ];
  for (const [name, input, result, reduction] of [
    ['minimum-x', pixelX, minimumX, 'min'],
    ['minimum-y', pixelY, minimumY, 'min'],
    ['maximum-x', pixelX, maximumX, 'max'],
    ['maximum-y', pixelY, maximumY, 'max']
  ] as const) {
    nodes.push(
      createSegmentSumNode(graph, {
        id: `${id}-${name}`,
        operation,
        segmentCount: zoneCount,
        input,
        segmentOffsets: vertexOffsets,
        output: result,
        reduction
      })
    );
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-bounds`,
      operation,
      invocationCount: zoneCount,
      bindings: [
        {name: 'minimumX', view: minimumX, type: 'f32', access: 'read'},
        {name: 'minimumY', view: minimumY, type: 'f32', access: 'read'},
        {name: 'maximumX', view: maximumX, type: 'f32', access: 'read'},
        {name: 'maximumY', view: maximumY, type: 'f32', access: 'read'},
        {name: 'vertices', view: vertexOffsets, type: 'u32', access: 'read'},
        {name: 'bounds', view: bounds, type: 'u32', access: 'read_write'},
        {name: 'counts', view: counts, type: 'u32', access: 'read_write'},
        {name: 'validity', view: zoneValidityFloat, type: 'f32', access: 'read'}
      ],
      // Expand by one boundary cell on both sides. Clipping decides positive support, including
      // the legacy float32 slivers; the index must never decide whether a boundary tuple exists.
      declarations: `${SCAN_TOTAL_WGSL}
// Convert only representable uint32 values. 2^32 is an exactly representable f32 sentinel;
// comparing the dimensions as integers prevents f32(width) from narrowing or widening the grid.
const UINT32_CONVERSION_BOUNDARY:f32=4294967296.0;
fn clampPixelBound(value:f32,dimension:u32)->u32{
  if(value<=0.0){return 0u;}
  if(!(value<UINT32_CONVERSION_BOUNDARY)){return dimension;}
  return min(u32(value),dimension);
}
fn lowerPixelBound(value:f32,dimension:u32)->u32{
  let bound=clampPixelBound(floor(value),dimension);return bound-min(bound,1u);
}
fn upperPixelBound(value:f32,dimension:u32)->u32{
  let bound=clampPixelBound(ceil(value),dimension);return bound+min(dimension-bound,1u);
}`,
      body: `let shift=${grid.pixelInterpretation === 'point' ? '0.5' : '0.0'};
let left=lowerPixelBound(minimumX[minimumXOffset+index]+shift,${grid.width}u);
let bottom=lowerPixelBound(minimumY[minimumYOffset+index]+shift,${grid.height}u);
let right=max(left,upperPixelBound(maximumX[maximumXOffset+index]+shift,${grid.width}u));
let top=max(bottom,upperPixelBound(maximumY[maximumYOffset+index]+shift,${grid.height}u));
bounds[boundsOffset+index*4u]=left;bounds[boundsOffset+index*4u+1u]=bottom;
bounds[boundsOffset+index*4u+2u]=right-left;bounds[boundsOffset+index*4u+3u]=top-bottom;
counts[countsOffset+index]=select(getSaturatedScanProduct(right-left,top-bottom),0u,vertices[verticesOffset+index]==vertices[verticesOffset+index+1u]||validity[validityOffset+index]==0.0);`
    }),
    ...createSaturatedCountScanNodes(graph, {
      id: `${id}-candidate-scan`,
      operation,
      counts,
      offsets,
      rowCount: zoneCount,
      total: candidateTotal
    }),
    createFillNode(graph, {
      id: `${id}-reset-tiles`,
      operation,
      view: state,
      type: 'u32',
      value: '0u',
      componentCount: 2
    })
  );
  for (
    let tile = 0;
    tile < Math.ceil(candidateCapacity / RASTER_FRACTIONAL_CANDIDATE_TILE_SIZE);
    tile++
  ) {
    const tileId = `${id}-tile-${tile}`;
    const tileLength = Math.min(
      RASTER_FRACTIONAL_CANDIDATE_TILE_SIZE,
      candidateCapacity - tile * RASTER_FRACTIONAL_CANDIDATE_TILE_SIZE
    );
    const tilePairs = graph.createDataView(pairs.buffer, {format: 'uint32x2', length: tileLength});
    const tileFractions = graph.createDataView(fractions.buffer, {
      format: 'float32',
      length: tileLength
    });
    const tilePairIds = graph.createDataView(pairIds.buffer, {
      format: 'uint32',
      length: tileLength
    });
    const tileFlags = graph.createDataView(flags.buffer, {format: 'uint32', length: tileLength});
    const tileCompact = graph.createDataView(compact.buffer, {
      format: 'uint32',
      length: tileLength
    });
    nodes.push(
      // The public compaction primitive scans fixed scratch. Clear its unused suffix each tile;
      // lookup, clipping, classification and output publication dispatch only actual input rows.
      createFillNode(graph, {
        id: `${tileId}-clear-flags`,
        operation,
        view: flags,
        type: 'u32',
        value: '0u',
        componentCount: RASTER_FRACTIONAL_CANDIDATE_TILE_SIZE
      }),
      createRasterFractionalTileNode(graph, {
        id: `${tileId}-lookup`,
        operation,
        invocationCount: tileLength,
        rowCount: 'arrayLength(&pairs)/2u',
        bindings: [
          {name: 'bounds', view: bounds, type: 'u32', access: 'read'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read'},
          {name: 'total', view: candidateTotal, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'pairs', view: tilePairs, type: 'u32', access: 'read_write'}
        ],
        body: `let candidate=state[stateOffset]+index;
pairs[pairsOffset+index*2u]=${zoneCount}u;pairs[pairsOffset+index*2u+1u]=0u;
if(candidate>=min(total[totalOffset],${candidateCapacity}u)){return;}
var low=0u;var high=${zoneCount}u;
loop{if(low>=high){break;}let middle=low+(high-low)/2u;if(offsets[offsetsOffset+middle]<=candidate){low=middle+1u;}else{high=middle;}}
let zone=low-1u;let local=candidate-offsets[offsetsOffset+zone];let width=bounds[boundsOffset+zone*4u+2u];
let column=bounds[boundsOffset+zone*4u]+local%width;let row=bounds[boundsOffset+zone*4u+1u]+local/width;
pairs[pairsOffset+index*2u]=zone;pairs[pairsOffset+index*2u+1u]=row*${grid.width}u+column;`
      }),
      createRasterFractionalTileNode(graph, {
        id: `${tileId}-clip`,
        operation,
        invocationCount: tileLength,
        rowCount: 'arrayLength(&pairs)/2u',
        bindings: [
          {name: 'positions', view: zones.positions, type: 'f32', access: 'read'},
          {name: 'features', view: zones.featureOffsets, type: 'u32', access: 'read'},
          {name: 'offsets', view: triangleOffsets, type: 'u32', access: 'read'},
          {name: 'indices', view: indices, type: 'u32', access: 'read'},
          {name: 'valid', view: polygonValidity, type: 'u32', access: 'read'},
          {name: 'settings', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'pairs', view: tilePairs, type: 'u32', access: 'read'},
          {name: 'coverage', view: tileFractions, type: 'f32', access: 'read_write'}
        ],
        declarations: getRasterFractionalClippingWGSL(grid.pixelInterpretation),
        body: `coverage[coverageOffset+index]=-1.0;
let zone=pairs[pairsOffset+index*2u];if(zone>=${zoneCount}u){return;}
let cell=pairs[pairsOffset+index*2u+1u];let column=f32(cell%${grid.width}u)${grid.pixelInterpretation === 'point' ? '-0.5' : ''};let row=f32(cell/${grid.width}u)${grid.pixelInterpretation === 'point' ? '-0.5' : ''};
var area=0.0;var complete=true;
for(var polygon=features[featuresOffset+zone];polygon<features[featuresOffset+zone+1u];polygon++){
 if(valid[validOffset+polygon]==0u){complete=false;break;}
 for(var triangle=offsets[offsetsOffset+polygon];triangle<offsets[offsetsOffset+polygon+1u];triangle++){
  let cellOrigin=vec2<f32>(column,row);
  var vertices:array<vec2<f32>,8>;vertices[0]=point(indices[indicesOffset+triangle*3u])-cellOrigin;vertices[1]=point(indices[indicesOffset+triangle*3u+1u])-cellOrigin;vertices[2]=point(indices[indicesOffset+triangle*3u+2u])-cellOrigin;
  var size=clip(&vertices,3u,0u,0.0,true);size=clip(&vertices,size,0u,1.0,false);size=clip(&vertices,size,1u,0.0,true);size=clip(&vertices,size,1u,1.0,false);
  var doubledArea=0.0;for(var vertex=0u;vertex<size;vertex++){let first=vertices[vertex];let second=vertices[(vertex+1u)%size];doubledArea+=first.x*second.y-first.y*second.x;}area+=abs(doubledArea)*0.5;
 }
}coverage[coverageOffset+index]=select(-1.0,min(area,1.0),complete);`
      }),
      createRasterFractionalTileNode(graph, {
        id: `${tileId}-flags`,
        operation,
        invocationCount: tileLength,
        rowCount: 'arrayLength(&coverage)',
        bindings: [
          {name: 'coverage', view: tileFractions, type: 'f32', access: 'read'},
          {name: 'pairIds', view: tilePairIds, type: 'u32', access: 'read_write'},
          {name: 'flags', view: tileFlags, type: 'u32', access: 'read_write'}
        ],
        body: 'pairIds[pairIdsOffset+index]=index;flags[flagsOffset+index]=u32(!(coverage[coverageOffset+index]<=0.0));'
      }),
      ...new GPUCompaction({
        id: `${tileId}-compact`,
        input: pairIds,
        flags,
        output: compact,
        count: tileCount
      }).getCommandNodes(graph),
      createRasterFractionalTileNode(graph, {
        id: `${tileId}-append`,
        operation,
        invocationCount: tileLength,
        rowCount: 'arrayLength(&compact)',
        bindings: [
          {name: 'compact', view: tileCompact, type: 'u32', access: 'read'},
          {name: 'pairs', view: tilePairs, type: 'u32', access: 'read'},
          {name: 'coverage', view: tileFractions, type: 'f32', access: 'read'},
          {name: 'tileCount', view: tileCount, type: 'u32', access: 'read'},
          {name: 'state', view: state, type: 'u32', access: 'read'},
          {name: 'zones', view: output.zoneIds, type: 'u32', access: 'read_write'},
          {name: 'cells', view: output.cellIds, type: 'u32', access: 'read_write'},
          {name: 'fractions', view: output.fractions, type: 'f32', access: 'read_write'}
        ],
        body: `let base=state[stateOffset+1u];if(index>=tileCount[tileCountOffset]||base>=${output.zoneIds.length}u||index>=${output.zoneIds.length}u-base){return;}
let source=compact[compactOffset+index];let destination=base+index;
zones[zonesOffset+destination]=pairs[pairsOffset+source*2u];cells[cellsOffset+destination]=pairs[pairsOffset+source*2u+1u];fractions[fractionsOffset+destination]=coverage[coverageOffset+source];`
      }),
      createWGSLKernelNode(graph, {
        id: `${tileId}-advance`,
        operation,
        invocationCount: 1,
        bindings: [
          {name: 'state', view: state, type: 'u32', access: 'read_write'},
          {name: 'tileCount', view: tileCount, type: 'u32', access: 'read'}
        ],
        body: `state[stateOffset]+=min(${RASTER_FRACTIONAL_CANDIDATE_TILE_SIZE}u,0xffffffffu-state[stateOffset]);state[stateOffset+1u]+=min(tileCount[tileCountOffset],0xffffffffu-state[stateOffset+1u]);`
      })
    );
  }
  nodes.push(
    createWGSLKernelNode(graph, {
      id: `${id}-zone-validity`,
      operation,
      invocationCount: zoneCount,
      bindings: [
        {name: 'features', view: zones.featureOffsets, type: 'u32', access: 'read'},
        {name: 'featureValidity', view: zoneValidityFloat, type: 'f32', access: 'read'},
        {name: 'candidateTotal', view: candidateTotal, type: 'u32', access: 'read'},
        {name: 'validity', view: output.zoneValidity, type: 'u32', access: 'read_write'}
      ],
      body: `let complete=candidateTotal[candidateTotalOffset]<=${candidateCapacity}u;
let valid=featureValidity[featureValidityOffset+index]!=0.0||features[featuresOffset+index]==features[featuresOffset+index+1u];
validity[validityOffset+index]=u32(complete&&valid);`
    }),
    createWGSLKernelNode(graph, {
      id: `${id}-status`,
      operation,
      invocationCount: 1,
      bindings: [
        {name: 'state', view: state, type: 'u32', access: 'read'},
        {name: 'candidateTotal', view: candidateTotal, type: 'u32', access: 'read'},
        {name: 'count', view: output.status.count, type: 'u32', access: 'read_write'},
        {name: 'required', view: output.status.requiredCount, type: 'u32', access: 'read_write'},
        {name: 'overflow', view: output.status.overflow, type: 'u32', access: 'read_write'},
        ...(output.status.candidateOverflow
          ? [
              {
                name: 'candidateOverflow',
                view: output.status.candidateOverflow,
                type: 'u32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      body: `let exhausted=candidateTotal[candidateTotalOffset]>${candidateCapacity}u;let total=state[stateOffset+1u];
count[countOffset]=min(total,${output.zoneIds.length}u);required[requiredOffset]=select(total,0xffffffffu,exhausted);overflow[overflowOffset]=u32(total>${output.zoneIds.length}u);
${output.status.candidateOverflow ? 'candidateOverflow[candidateOverflowOffset]=u32(exhausted);' : ''}`
    })
  );
  return nodes;
}
