// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getSpatialAnalysisPackedKernelNodes} from '../geographic-distribution/spatial-analysis-packed-kernel';
import type {VertexFormat} from '@luma.gl/core';
import {
  addLocalBuildingTessellationRecipe,
  type GPUBuildingTessellationLocalProps
} from './building-tessellation-local';
export type {
  GPUBuildingTessellationLocalProps,
  GPUBuildingTessellationLocalOutput,
  GPUBuildingTessellationLocalStatus
} from './building-tessellation-local';
export {
  getGPUBuildingTessellationLocalParameterValues,
  GPU_BUILDING_TESSELLATION_LOCAL_PARAMETER_LENGTH
} from './building-tessellation-local';
import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateUrbanRecipeBindings,
  validateUrbanStorageByteLength
} from './building-tessellation-validation';
import {getRecipeView} from '../recipes/expansion-recipe-utils';
import {GPUPolygonOverlay, type GPUPolygonOverlayOutput} from '../polygon-overlay/index';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createSaturatedCountScanNodes} from '../../analysis-core/kernels/scan-total';
import {
  addUrbanPointGrid,
  getUrbanPointGridShape,
  getUrbanPointGridDeclarations,
  getUrbanPointGridVoronoiWalk
} from './visibility-graph-point-grid';

/** Most buildings: each one adds a `GPUPolygonOverlay` dissolve (and a clip) to the graph. */
export const BUILDING_TESSELLATION_MAXIMUM_BUILDING_COUNT = 64;

/** Most enclosure vertices: each sample clips a thread-private polygon that starts as the enclosure. */
export const BUILDING_TESSELLATION_MAXIMUM_ENCLOSURE_VERTEX_COUNT = 32;

/**
 * Most boundary samples: thread-private clipping arrays and the public fragment table still
 * hold samples + enclosure vertices per sample. The grid only changes candidate discovery.
 */
export const BUILDING_TESSELLATION_MAXIMUM_SAMPLE_CAPACITY = 128;

/** Most samples one boundary segment may ask for: exact float32 integer row counts (2^24). */
const BUILDING_TESSELLATION_MAXIMUM_SAMPLES_PER_SEGMENT = 16777216;

export type GPUBuildingTessellationOutput = {
  /**
   * One overlay output per building, holding the final polygons: the dissolved cells, or their
   * intersection with the enclosure in 'simple' mode. Each entry is the `output` of a
   * `GPUPolygonOverlay` with the recipe's `capacity`; the entry index is the building row.
   */
  overlays: readonly (GPUPolygonOverlayOutput | undefined)[];
  /** `float32x2`, `sampleCapacity` rows: boundary sample positions. */
  samples: GraphDataView<'float32x2'>;
  /** `uint32`, `sampleCapacity` rows: building row of each sample, or 0xffffffff. */
  owners: GraphDataView<'uint32'>;
  /** `float32x2`, `sampleCapacity * (sampleCapacity + enclosure.length)` rows: clipped Voronoi cells. */
  fragments: GraphDataView<'float32x2'>;
  /** `uint32`, `sampleCapacity` rows: vertices of each cell, or 0xffffffff when invalid. */
  fragmentCounts: GraphDataView<'uint32'>;
  /** `uint32x4`, one row: `[count, requiredCount, overflow, invalid]` of the boundary sampling. */
  samplingStatus: GraphDataView<'uint32x4'>;
  /** `uint32`, one row per building: one when its tessellation is complete. */
  complete: GraphDataView<'uint32'>;
};

export type GPUBuildingTessellationProps = {
  id?: string;
  /** Explicit legacy overlay route. Existing calls without candidateRadius keep this output layout. */
  mode?: 'reference';
  /** Closed building boundaries as directed segments, without duplicate segments. */
  segments: GraphDataView<'float32x4'>;
  segmentBuildingIds: GraphDataView<'uint32'>;
  buildingCount: number;
  /** One convex, counterclockwise enclosure without a repeated closing vertex. Run distinct enclosures independently. */
  enclosure: GraphDataView<'float32x2'>;
  /** Simple nonconvex rings are clipped by robust polygon overlay after the cell dissolve. */
  enclosureMode?: 'convex' | 'simple';
  /** Optional exterior/hole offsets into enclosure; requires simple overlay mode. */
  enclosureRingOffsets?: GraphDataView<'uint32'>;
  sampleSpacing: number;
  sampleCapacity: number;
  /**
   * Compile-time candidate search. Default 'brute-force' preserves exhaustive input-row order
   * and the public fragments, including repeated vertices. Opt-in 'grid' prunes cells whose
   * bisectors cannot cut the current polygon. It preserves the represented region within
   * vertexTolerance but can change repeated-vertex counts and cyclic vertex order. Both retain
   * the overlay and fragment-table limits.
   */
  searchStructure?: 'grid' | 'brute-force';
  /** Overlay lattice tolerance; all resulting building cells are dissolved on this common grid. */
  vertexTolerance: number;
  capacity: {intersections: number; nodedSegments: number; vertices: number; rings: number};
  /**
   * Caller-owned result views. Any view that is absent is a graph-owned transient view, as before.
   * Formats and lengths are those of {@link GPUBuildingTessellationOutput}.
   */
  output?: Partial<GPUBuildingTessellationOutput>;
};

/** Bounded local route, selected by candidateRadius, or the explicit reference overlay route. */
export function addBuildingTessellationRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUBuildingTessellationLocalProps
): ReturnType<typeof addLocalBuildingTessellationRecipe>;
export function addBuildingTessellationRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUBuildingTessellationProps
): ReturnType<typeof addBuildingTessellationReferenceRecipe>;
export function addBuildingTessellationRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUBuildingTessellationProps | GPUBuildingTessellationLocalProps
) {
  if (props.mode === 'local' || ('candidateRadius' in props && props.mode !== 'reference')) {
    return addLocalBuildingTessellationRecipe(graph, props as GPUBuildingTessellationLocalProps);
  }
  return addBuildingTessellationReferenceRecipe(graph, props as GPUBuildingTessellationProps);
}

/** Boundary-sampled, enclosure-clipped reference Voronoi fragments dissolved per building. */
function addBuildingTessellationReferenceRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUBuildingTessellationProps
) {
  const id = props.id ?? 'building-tessellation';
  validateSinglePackedView(props.segments, ['float32x4'], `${id} segments`);
  validateSinglePackedView(props.segmentBuildingIds, ['uint32'], `${id} segmentBuildingIds`);
  validateSinglePackedView(props.enclosure, ['float32x2'], `${id} enclosure`);
  if (props.enclosureRingOffsets)
    validateSinglePackedView(props.enclosureRingOffsets, ['uint32'], `${id} enclosureRingOffsets`);
  const segmentCount = props.segments.length;
  const enclosureCount = props.enclosure.length;
  const simpleEnclosure = props.enclosureMode === 'simple';
  const sampleCapacity = props.sampleCapacity;
  const vertexCapacity = sampleCapacity + enclosureCount;
  validateRequirement(
    id,
    segmentCount > 0 && props.segmentBuildingIds.length === segmentCount,
    'segments must have at least 1 row and segmentBuildingIds one row per segment'
  );
  validateCapacity(
    id,
    'buildingCount',
    props.buildingCount,
    1,
    BUILDING_TESSELLATION_MAXIMUM_BUILDING_COUNT
  );
  validateRequirement(
    id,
    enclosureCount >= 3 && enclosureCount <= BUILDING_TESSELLATION_MAXIMUM_ENCLOSURE_VERTEX_COUNT,
    `enclosure must have between 3 and ${BUILDING_TESSELLATION_MAXIMUM_ENCLOSURE_VERTEX_COUNT} rows`
  );
  validateCapacity(
    id,
    'sampleCapacity',
    sampleCapacity,
    1,
    BUILDING_TESSELLATION_MAXIMUM_SAMPLE_CAPACITY
  );
  validateRequirement(
    id,
    Number.isFinite(Math.fround(props.sampleSpacing)) && Math.fround(props.sampleSpacing) > 0,
    'sampleSpacing must be a positive finite float32 number'
  );
  validateRequirement(
    id,
    Number.isFinite(Math.fround(props.vertexTolerance)) && Math.fround(props.vertexTolerance) > 0,
    'vertexTolerance must be a positive finite float32 number'
  );
  validateRequirement(
    id,
    Object.values(props.capacity).every(value => Number.isSafeInteger(value) && value > 0),
    'capacity entries (intersections, nodedSegments, vertices, rings) must be positive integers'
  );
  validateRequirement(
    id,
    !props.enclosureRingOffsets || (simpleEnclosure && props.enclosureRingOffsets.length >= 2),
    "enclosureRingOffsets needs enclosureMode 'simple' and at least 2 rows"
  );
  validateUrbanRecipeBindings(
    graph,
    id,
    {
      segments: props.segments,
      segmentBuildingIds: props.segmentBuildingIds,
      enclosure: props.enclosure,
      enclosureRingOffsets: props.enclosureRingOffsets
    },
    props.output
  );
  const maximumOverlayVertexCount = Math.max(
    sampleCapacity * (vertexCapacity + 1),
    simpleEnclosure
      ? props.capacity.vertices +
          enclosureCount +
          props.capacity.rings +
          (props.enclosureRingOffsets?.length ?? 2) -
          1
      : 0
  );
  const overlayHotPixelCapacity = maximumOverlayVertexCount + props.capacity.intersections;
  const overlayIncidenceCapacity = Math.max(
    props.capacity.nodedSegments + maximumOverlayVertexCount,
    2 * overlayHotPixelCapacity
  );
  // Recipe declaration is the allocation boundary (there is no separate getCommandNodes).
  for (const [name, byteLength, hint] of [
    [
      'point grid',
      props.searchStructure !== 'grid' ? 4 : getUrbanPointGridShape(sampleCapacity).byteLength,
      'reduce sampleCapacity'
    ],
    ['segments', segmentCount * 16, 'reduce segments length'],
    ['segmentBuildingIds', segmentCount * 4, 'reduce segments length'],
    ['sample scratch', sampleCapacity * 8, 'reduce sampleCapacity'],
    ['segment scratch', segmentCount * 4, 'reduce segments length'],
    [
      'fragments and per-building packed positions',
      sampleCapacity * vertexCapacity * 8,
      'reduce sampleCapacity'
    ],
    ['per-building offsets', (sampleCapacity + 1) * 4, 'reduce sampleCapacity'],
    ['complete', props.buildingCount * 4, 'reduce buildingCount'],
    ['aligned building parameters', props.buildingCount * 256, 'reduce buildingCount'],
    [
      'overlay hot-pixel scratch',
      overlayHotPixelCapacity * 128,
      'reduce capacity.intersections or the overlay input'
    ],
    [
      'overlay candidate scratch',
      overlayIncidenceCapacity * 8,
      'reduce capacity.intersections or capacity.nodedSegments'
    ],
    ['overlay boundary', props.capacity.nodedSegments * 16, 'reduce capacity.nodedSegments'],
    ['overlay positions', props.capacity.vertices * 8, 'reduce capacity.vertices'],
    ['overlay offsets', (props.capacity.rings + 1) * 4, 'reduce capacity.rings']
  ] as const) {
    validateUrbanStorageByteLength(graph.device, id, name, byteLength, hint);
  }
  const view = <Format extends VertexFormat>(name: string, format: Format, length: number) =>
    getRecipeView(graph, `${id}-${name}`, format, length);
  const output = props.output;
  const samples = getRecipeView(
    graph,
    `${id}-samples`,
    'float32x2',
    sampleCapacity,
    output?.samples
  );
  const owners = getRecipeView(graph, `${id}-owners`, 'uint32', sampleCapacity, output?.owners);
  const samplingStatus = getRecipeView(
    graph,
    `${id}-sampling-status`,
    'uint32x4',
    1,
    output?.samplingStatus
  );
  const fragments = getRecipeView(
    graph,
    `${id}-fragments`,
    'float32x2',
    sampleCapacity * vertexCapacity,
    output?.fragments
  );
  const enclosureOffsets = view('enclosure-offsets', 'uint32', 2);
  const enclosureIndices = view('enclosure-indices', 'uint32', 2);
  const enclosurePolygons = view('enclosure-polygons', 'uint32', 2);
  if (simpleEnclosure)
    graph.add(
      getSpatialAnalysisPackedKernelNodes(graph, {
        id: `${id}-enclosure-layout`,
        operation: id,
        invocationCount: 1,
        bindings: [
          {name: 'offsets', view: enclosureOffsets, type: 'u32', access: 'read_write'},
          {name: 'indices', view: enclosureIndices, type: 'u32', access: 'read_write'},
          {name: 'polygons', view: enclosurePolygons, type: 'u32', access: 'read_write'}
        ],
        body: `offsets[offsetsOffset]=0u;offsets[offsetsOffset+1u]=${enclosureCount}u;indices[indicesOffset]=0u;indices[indicesOffset+1u]=1u;polygons[polygonsOffset]=0u;polygons[polygonsOffset+1u]=${props.enclosureRingOffsets ? props.enclosureRingOffsets.length - 1 : 1}u;`
      })
    );
  const fragmentCounts = getRecipeView(
    graph,
    `${id}-fragment-counts`,
    'uint32',
    sampleCapacity,
    output?.fragmentCounts
  );
  // Boundary sampling is segment parallel: sample counts per segment, an exclusive scan for the
  // first sample of each segment, then one thread per segment writes its samples.
  const sampleCounts = createTransientView(graph, `${id}-sample-counts`, 'uint32', segmentCount);
  const sampleOffsets = createTransientView(graph, `${id}-sample-offsets`, 'uint32', segmentCount);
  const invalidSegments = createTransientView(
    graph,
    `${id}-invalid-segments`,
    'uint32',
    segmentCount
  );
  const requiredSamples = createTransientView(graph, `${id}-required-samples`, 'uint32', 1);
  const invalidCount = createTransientView(graph, `${id}-invalid-segment-count`, 'uint32', 1);
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-sample-counts`,
      operation: id,
      invocationCount: segmentCount,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        {name: 'buildingIds', view: props.segmentBuildingIds, type: 'u32', access: 'read'},
        {name: 'counts', view: sampleCounts, type: 'u32', access: 'read_write'},
        {name: 'invalidFlags', view: invalidSegments, type: 'u32', access: 'read_write'}
      ],
      body: `let base=segmentsOffset+4u*index;
let start=vec2<f32>(segments[base],segments[base+1u]);
let end=vec2<f32>(segments[base+2u],segments[base+3u]);
let owner=buildingIds[buildingIdsOffset+index];
let length=distance(start,end);
var count=0u;
var invalid=!isFiniteValue(start.x)||!isFiniteValue(start.y)||!isFiniteValue(end.x)||!isFiniteValue(end.y)||!isFiniteValue(length)||owner>=${props.buildingCount}u||length==0.0;
if(!invalid){
  let countValue=ceil(length/${getWGSLFloatLiteral(props.sampleSpacing)});
  if(!isFiniteValue(countValue)||countValue>${BUILDING_TESSELLATION_MAXIMUM_SAMPLES_PER_SEGMENT}.0){invalid=true;}else{count=u32(countValue);}
}
counts[countsOffset+index]=count;
invalidFlags[invalidFlagsOffset+index]=u32(invalid);`
    })
  );
  graph.add(
    createSaturatedCountScanNodes(graph, {
      id: `${id}-sample-total`,
      operation: id,
      counts: sampleCounts,
      offsets: sampleOffsets,
      rowCount: segmentCount,
      total: requiredSamples
    })
  );
  graph.add(
    new GPUReduction({
      id: `${id}-invalid-segment-total`,
      input: invalidSegments,
      output: invalidCount,
      operation: 'sum'
    }).getCommandNodes(graph)
  );
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-clear-samples`,
      operation: id,
      invocationCount: sampleCapacity,
      bindings: [
        {name: 'samples', view: samples, type: 'f32', access: 'read_write'},
        {name: 'owners', view: owners, type: 'u32', access: 'read_write'}
      ],
      body: `owners[ownersOffset+index]=0xffffffffu;
samples[samplesOffset+2u*index]=0.0;
samples[samplesOffset+2u*index+1u]=0.0;`
    })
  );
  // Each thread writes at most sampleCapacity samples: the loop is over a capacity.
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-write-samples`,
      operation: id,
      invocationCount: segmentCount,
      bindings: [
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        {name: 'buildingIds', view: props.segmentBuildingIds, type: 'u32', access: 'read'},
        {name: 'counts', view: sampleCounts, type: 'u32', access: 'read'},
        {name: 'offsets', view: sampleOffsets, type: 'u32', access: 'read'},
        {name: 'samples', view: samples, type: 'f32', access: 'read_write'},
        {name: 'owners', view: owners, type: 'u32', access: 'read_write'}
      ],
      body: `let count=counts[countsOffset+index];
let first=offsets[offsetsOffset+index];
let base=segmentsOffset+4u*index;
let start=vec2<f32>(segments[base],segments[base+1u]);
let end=vec2<f32>(segments[base+2u],segments[base+3u]);
for(var part=0u;part<min(count,${sampleCapacity}u);part++){
  if(first>=${sampleCapacity}u||part>=${sampleCapacity}u-first){break;}
  let position=mix(start,end,f32(part)/f32(count));
  let sampleRow=first+part;
  samples[samplesOffset+2u*sampleRow]=position.x;
  samples[samplesOffset+2u*sampleRow+1u]=position.y;
  owners[ownersOffset+sampleRow]=buildingIds[buildingIdsOffset+index];
}`
    })
  );
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-sampling-status`,
      operation: id,
      invocationCount: 1,
      bindings: [
        {name: 'required', view: requiredSamples, type: 'u32', access: 'read'},
        {name: 'invalid', view: invalidCount, type: 'u32', access: 'read'},
        {name: 'status', view: samplingStatus, type: 'u32', access: 'read_write'}
      ],
      body: `let requiredTotal=required[requiredOffset];
status[statusOffset]=min(requiredTotal,${sampleCapacity}u);
status[statusOffset+1u]=requiredTotal;
status[statusOffset+2u]=u32(requiredTotal>${sampleCapacity}u);
status[statusOffset+3u]=invalid[invalidOffset];`
    })
  );
  const pointGrid =
    props.searchStructure !== 'grid'
      ? undefined
      : addUrbanPointGrid(graph, {
          id: `${id}-sample-grid`,
          positions: samples,
          count: samplingStatus
        });
  const clipSample = ` if(other==index||size==0u){continue;}let displacement=readSample(other)-site;let squared=dot(displacement,displacement);
 if(squared==0.0){if(owners[ownersOffset+other]!=owners[ownersOffset+index]){valid=false;}else if(other<index){size=0u;}continue;}
 let threshold=0.5*squared;var outputSize=0u;
 for(var vertex=0u;vertex<size;vertex++){let start=polygon[vertex];let end=polygon[(vertex+1u)%size];let startProducts=(start-site)*displacement;let endProducts=(end-site)*displacement;
 let startSide=startProducts.x+startProducts.y-threshold;let endSide=endProducts.x+endProducts.y-threshold;
 if(!isFiniteValue(startProducts.x)||!isFiniteValue(startProducts.y)||!isFiniteValue(endProducts.x)||!isFiniteValue(endProducts.y)){valid=false;break;}let inside=startSide<=0.0;let nextInside=endSide<=0.0;
 if(inside){if(outputSize<${vertexCapacity}u){scratch[outputSize]=start;}outputSize++;}
 if(!isFiniteValue(startSide)||!isFiniteValue(endSide)||!isFiniteValue(squared)){valid=false;break;}
 if(inside!=nextInside){let sideScale=max(abs(startSide),abs(endSide));let scaledStart=startSide/sideScale;let denominator=scaledStart-endSide/sideScale;let fraction=scaledStart/denominator;
 let intersection=(1.0-fraction)*start+fraction*end;
 valid=valid&&isFiniteValue(fraction)&&isFiniteValue(denominator)&&denominator!=0.0&&isFiniteValue(intersection.x)&&isFiniteValue(intersection.y);
 if(outputSize<${vertexCapacity}u){scratch[outputSize]=intersection;}outputSize++;}
 }
 if(outputSize>${vertexCapacity}u){valid=false;break;}size=outputSize;for(var vertex=0u;vertex<size;vertex++){polygon[vertex]=scratch[vertex];}`;
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-clip-cells`,
      operation: id,
      invocationCount: sampleCapacity,
      workgroupSize: 2,
      declarations:
        IS_FINITE_VALUE_WGSL +
        (pointGrid ? getUrbanPointGridDeclarations(pointGrid) : '') +
        `
fn readSample(row:u32)->vec2<f32>{return vec2<f32>(samples[samplesOffset+2u*row],samples[samplesOffset+2u*row+1u]);}
fn readEnclosure(row:u32)->vec2<f32>{return vec2<f32>(enclosure[enclosureOffset+2u*row],enclosure[enclosureOffset+2u*row+1u]);}
fn crossVectors(first:vec2<f32>,second:vec2<f32>)->f32{return first.x*second.y-first.y*second.x;}`,
      bindings: [
        ...(pointGrid
          ? [{name: 'pointGrid', view: pointGrid.data, type: 'u32', access: 'read'} as const]
          : []),
        {name: 'samples', view: samples, type: 'f32', access: 'read'},
        {name: 'owners', view: owners, type: 'u32', access: 'read'},
        {name: 'status', view: samplingStatus, type: 'u32', access: 'read'},
        {name: 'enclosure', view: props.enclosure, type: 'f32', access: 'read'},
        {name: 'fragments', view: fragments, type: 'f32', access: 'read_write'},
        {name: 'counts', view: fragmentCounts, type: 'u32', access: 'read_write'}
      ],
      body: `counts[countsOffset+index]=0u;if(index>=status[statusOffset]||status[statusOffset+2u]!=0u||status[statusOffset+3u]!=0u){return;}
var polygon:array<vec2<f32>,${vertexCapacity}>;var scratch:array<vec2<f32>,${vertexCapacity}>;var size=${enclosureCount}u;let site=readSample(index);var valid=true;var area=0.0;
for(var vertex=0u;vertex<${enclosureCount}u;vertex++){let position=readEnclosure(vertex);let next=readEnclosure((vertex+1u)%${enclosureCount}u);let after=readEnclosure((vertex+2u)%${enclosureCount}u);polygon[vertex]=position;valid=valid&&isFiniteValue(position.x)&&isFiniteValue(position.y)&&crossVectors(next-position,after-next)>=0.0;area+=crossVectors(position,next);}
valid=valid&&area>0.0;
${
  simpleEnclosure
    ? `valid=abs(area)>0.0;var lower=vec2<f32>(3.402823e38);var upper=vec2<f32>(-3.402823e38);
for(var vertex=0u;vertex<${enclosureCount}u;vertex++){let point=readEnclosure(vertex);valid=valid&&isFiniteValue(point.x)&&isFiniteValue(point.y);lower=min(lower,point);upper=max(upper,point);}
polygon[0u]=lower;polygon[1u]=vec2<f32>(upper.x,lower.y);polygon[2u]=upper;polygon[3u]=vec2<f32>(lower.x,upper.y);size=4u;`
    : ''
}
${pointGrid ? getUrbanPointGridVoronoiWalk(pointGrid, clipSample) : `for(var other=0u;other<status[statusOffset];other++){${clipSample}}`}

if(!valid){counts[countsOffset+index]=0xffffffffu;return;}
if(size<3u){size=0u;}counts[countsOffset+index]=size;
for(var vertex=0u;vertex<size;vertex++){let destination=fragmentsOffset+2u*(index*${vertexCapacity}u+vertex);fragments[destination]=polygon[vertex].x;fragments[destination+1u]=polygon[vertex].y;}`
    })
  );
  const complete = getRecipeView(
    graph,
    `${id}-complete`,
    'uint32',
    props.buildingCount,
    output?.complete
  );
  const buildingParameters = view('building-parameters', 'uint32', props.buildingCount * 64);
  graph.add(
    getSpatialAnalysisPackedKernelNodes(graph, {
      id: `${id}-building-parameters`,
      operation: id,
      invocationCount: props.buildingCount,
      bindings: [{name: 'rows', view: buildingParameters, type: 'u32', access: 'read_write'}],
      body: 'rows[rowsOffset+64u*index]=index;'
    })
  );
  const outputs: GPUPolygonOverlayOutput[] = [];
  // Each building's pack and evidence kernel is one thread looping over sampleCapacity (at most
  // BUILDING_TESSELLATION_MAXIMUM_SAMPLE_CAPACITY) samples and their cell vertices.
  for (let building = 0; building < props.buildingCount; building++) {
    const prefix = `building-${building}`;
    const buildingParameter = graph.createDataView(buildingParameters.buffer, {
      format: 'uint32',
      byteOffset: building * 256,
      length: 1
    });
    const positions = view(
      `${prefix}-input-positions`,
      'float32x2',
      sampleCapacity * vertexCapacity
    );
    const offsets = view(`${prefix}-input-offsets`, 'uint32', sampleCapacity + 1);
    const indices = view(`${prefix}-input-indices`, 'uint32', sampleCapacity + 1);
    graph.add(
      getSpatialAnalysisPackedKernelNodes(graph, {
        id: `${id}-${prefix}-pack`,
        operation: id,
        invocationCount: 1,
        bindings: [
          {name: 'fragments', view: fragments, type: 'f32', access: 'read'},
          {name: 'counts', view: fragmentCounts, type: 'u32', access: 'read'},
          {name: 'owners', view: owners, type: 'u32', access: 'read'},
          {name: 'positions', view: positions, type: 'f32', access: 'read_write'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read_write'},
          {name: 'indices', view: indices, type: 'u32', access: 'read_write'},
          {name: 'buildingParameter', view: buildingParameter, type: 'u32', access: 'read'}
        ],
        body: `var valid=true;for(var sampleRow=0u;sampleRow<${sampleCapacity}u;sampleRow++){valid=valid&&counts[countsOffset+sampleRow]!=0xffffffffu;}
var written=0u;offsets[offsetsOffset]=0u;indices[indicesOffset]=0u;
for(var sampleRow=0u;sampleRow<${sampleCapacity}u;sampleRow++){if(valid&&owners[ownersOffset+sampleRow]==buildingParameter[buildingParameterOffset]){for(var vertex=0u;vertex<counts[countsOffset+sampleRow];vertex++){let source=fragmentsOffset+2u*(sampleRow*${vertexCapacity}u+vertex);positions[positionsOffset+2u*written]=fragments[source];positions[positionsOffset+2u*written+1u]=fragments[source+1u];written++;}}offsets[offsetsOffset+sampleRow+1u]=written;indices[indicesOffset+sampleRow+1u]=sampleRow+1u;}`
      })
    );
    const makeStatus = (name: string) => ({
      count: view(`${prefix}-${name}-count`, 'uint32', 1),
      overflow: view(`${prefix}-${name}-overflow`, 'uint32', 1),
      candidateOverflow: view(`${prefix}-${name}-candidates`, 'uint32', 1)
    });
    // The overlay geometry has no feature requirement (contract rule 8): its unclamped ring count
    // sizes `capacity.rings`. The boundary requirement is `requiredCount`.
    const makeGeometryStatus = (name: string) => ({
      ...makeStatus(name),
      requiredRingCount: view(`${prefix}-${name}-required`, 'uint32', 1)
    });
    const makeBoundaryStatus = (name: string) => ({
      ...makeStatus(name),
      requiredCount: view(`${prefix}-${name}-required`, 'uint32', 1)
    });
    // Transient overlay result views; `infix` is '' for the dissolve and 'clip-' for the clip.
    const makeOverlayOutput = (infix: string): GPUPolygonOverlayOutput => ({
      geometry: {
        kind: 'polygons',
        positions: view(`${prefix}-${infix}positions`, 'float32x2', props.capacity.vertices),
        featureOffsets: view(`${prefix}-${infix}features`, 'uint32', props.capacity.rings + 1),
        polygonOffsets: view(`${prefix}-${infix}polygons`, 'uint32', props.capacity.rings + 1),
        ringOffsets: view(`${prefix}-${infix}rings`, 'uint32', props.capacity.rings + 1)
      },
      status: makeGeometryStatus(`${infix}geometry`),
      boundary: {
        endpoints: view(`${prefix}-${infix}boundary`, 'float32x4', props.capacity.nodedSegments),
        operandIds: view(`${prefix}-${infix}operands`, 'uint32', props.capacity.nodedSegments),
        sourceFeatureIds: view(
          `${prefix}-${infix}boundary-sources`,
          'uint32',
          props.capacity.nodedSegments
        ),
        status: makeBoundaryStatus(`${infix}boundary`)
      }
    });
    const providedOverlay = output?.overlays?.[building];
    let overlay: GPUPolygonOverlayOutput =
      !simpleEnclosure && providedOverlay ? providedOverlay : makeOverlayOutput('');

    graph.add(
      new GPUPolygonOverlay({
        id: `${id}-${prefix}-dissolve`,
        left: {
          kind: 'polygons',
          positions,
          featureOffsets: indices,
          polygonOffsets: indices,
          ringOffsets: offsets
        },
        operation: 'dissolve',
        capacity: props.capacity,
        vertexTolerance: props.vertexTolerance,
        output: overlay
      })
    );
    const dissolvedOutput = overlay;
    if (simpleEnclosure) {
      overlay = providedOverlay ?? makeOverlayOutput('clip-');
      graph.add(
        new GPUPolygonOverlay({
          id: `${id}-${prefix}-clip-enclosure`,
          left: dissolvedOutput.geometry,
          right: {
            kind: 'polygons',
            positions: props.enclosure,
            featureOffsets: enclosureIndices,
            polygonOffsets: enclosurePolygons,
            ringOffsets: props.enclosureRingOffsets ?? enclosureOffsets
          },
          operation: 'intersection',
          capacity: props.capacity,
          vertexTolerance: props.vertexTolerance,
          output: overlay
        })
      );
    }
    const finalOverlay = overlay;
    outputs.push(finalOverlay);
    graph.add(
      getSpatialAnalysisPackedKernelNodes(graph, {
        id: `${id}-${prefix}-evidence`,
        operation: id,
        invocationCount: 1,
        bindings: [
          {name: 'sampling', view: samplingStatus, type: 'u32', access: 'read'},
          {name: 'counts', view: fragmentCounts, type: 'u32', access: 'read'},
          {name: 'overflow', view: finalOverlay.status.overflow!, type: 'u32', access: 'read'},
          {
            name: 'candidateOverflow',
            view: finalOverlay.status.candidateOverflow,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'dissolveOverflow',
            view: dissolvedOutput.status.overflow!,
            type: 'u32',
            access: 'read'
          },
          {
            name: 'dissolveCandidates',
            view: dissolvedOutput.status.candidateOverflow,
            type: 'u32',
            access: 'read'
          },
          {name: 'complete', view: complete, type: 'u32', access: 'read_write'},
          {name: 'buildingParameter', view: buildingParameter, type: 'u32', access: 'read'}
        ],
        body: `var valid=dissolveOverflow[dissolveOverflowOffset]==0u&&dissolveCandidates[dissolveCandidatesOffset]==0u&&sampling[samplingOffset+2u]==0u&&sampling[samplingOffset+3u]==0u&&overflow[overflowOffset]==0u&&candidateOverflow[candidateOverflowOffset]==0u;for(var sampleRow=0u;sampleRow<${sampleCapacity}u;sampleRow++){valid=valid&&counts[countsOffset+sampleRow]!=0xffffffffu;}complete[completeOffset+buildingParameter[buildingParameterOffset]]=u32(valid);`
      })
    );
  }
  return {
    outputs,
    intermediates: {samples, owners, fragments, fragmentCounts},
    status: {sampling: samplingStatus, complete},
    provenance: {
      segments: props.segments,
      segmentBuildingIds: props.segmentBuildingIds,
      enclosure: props.enclosure,
      sampleSpacing: props.sampleSpacing,
      vertexTolerance: props.vertexTolerance,
      enclosurePolicy: props.enclosureMode ?? 'convex',
      mode: 'reference' as const,
      searchStructure: pointGrid ? ('grid' as const) : ('brute-force' as const)
    }
  };
}
