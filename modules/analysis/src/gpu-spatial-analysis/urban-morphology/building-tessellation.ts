// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {assert, type VertexFormat} from '@luma.gl/core';
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {GPUPolygonOverlay, type GPUPolygonOverlayOutput} from '../polygon-overlay/index';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

export type GPUBuildingTessellationProps = {
  id?: string;
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
  /** Overlay lattice tolerance; all resulting building cells are dissolved on this common grid. */
  vertexTolerance: number;
  capacity: {intersections: number; nodedSegments: number; vertices: number; rings: number};
};

/** Boundary-sampled, enclosure-clipped Voronoi fragments dissolved into stable building-owned polygons. */
export function addBuildingTessellationRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUBuildingTessellationProps
) {
  const id = props.id ?? 'building-tessellation';
  const segmentCount = props.segments.length;
  const enclosureCount = props.enclosure.length;
  const simpleEnclosure = props.enclosureMode === 'simple';
  const sampleCapacity = props.sampleCapacity;
  const vertexCapacity = sampleCapacity + enclosureCount;
  assert(segmentCount > 0 && props.segmentBuildingIds.length === segmentCount);
  assert(
    Number.isSafeInteger(props.buildingCount) &&
      props.buildingCount > 0 &&
      props.buildingCount <= 64
  );
  assert(enclosureCount >= 3 && enclosureCount <= 32);
  assert(Number.isSafeInteger(sampleCapacity) && sampleCapacity > 0 && sampleCapacity <= 128);
  assert(Number.isFinite(props.sampleSpacing) && props.sampleSpacing > 0);
  assert(Number.isFinite(props.vertexTolerance) && props.vertexTolerance > 0);
  assert(Object.values(props.capacity).every(value => Number.isSafeInteger(value) && value > 0));
  const view = <Format extends VertexFormat>(name: string, format: Format, length: number) =>
    getRecipeView(graph, `${id}-${name}`, format, length);
  const samples = view('samples', 'float32x2', sampleCapacity);
  const owners = view('owners', 'uint32', sampleCapacity);
  const samplingStatus = view('sampling-status', 'uint32x4', 1);
  const fragments = view('fragments', 'float32x2', sampleCapacity * vertexCapacity);
  const enclosureOffsets = view('enclosure-offsets', 'uint32', 2);
  const enclosureIndices = view('enclosure-indices', 'uint32', 2);
  const enclosurePolygons = view('enclosure-polygons', 'uint32', 2);
  assert(
    !props.enclosureRingOffsets || (simpleEnclosure && props.enclosureRingOffsets.length >= 2)
  );
  if (simpleEnclosure)
    graph.add(
      createRecipeKernelNode(graph, {
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
  const fragmentCounts = view('fragment-counts', 'uint32', sampleCapacity);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-sample-boundaries`,
      operation: id,
      invocationCount: 1,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        {name: 'buildingIds', view: props.segmentBuildingIds, type: 'u32', access: 'read'},
        {name: 'samples', view: samples, type: 'f32', access: 'read_write'},
        {name: 'owners', view: owners, type: 'u32', access: 'read_write'},
        {name: 'status', view: samplingStatus, type: 'u32', access: 'read_write'}
      ],
      body: `var required=0u;var invalid=0u;for(var sample=0u;sample<${sampleCapacity}u;sample++){owners[ownersOffset+sample]=0xffffffffu;samples[samplesOffset+2u*sample]=0.0;samples[samplesOffset+2u*sample+1u]=0.0;}
for(var segment=0u;segment<${segmentCount}u;segment++){
 let base=segmentsOffset+4u*segment;let start=vec2<f32>(segments[base],segments[base+1u]);let end=vec2<f32>(segments[base+2u],segments[base+3u]);let owner=buildingIds[buildingIdsOffset+segment];let length=distance(start,end);
 if(!isFiniteValue(start.x)||!isFiniteValue(start.y)||!isFiniteValue(end.x)||!isFiniteValue(end.y)||!isFiniteValue(length)||owner>=${props.buildingCount}u||length==0.0){invalid++;continue;}
 let countValue=ceil(length/${props.sampleSpacing});if(!isFiniteValue(countValue)||countValue>16777216.0){invalid++;continue;}let count=u32(countValue);
 for(var part=0u;part<min(count,${sampleCapacity}u);part++){if(required+part>=${sampleCapacity}u){break;}let position=mix(start,end,f32(part)/f32(count));let sample=required+part;samples[samplesOffset+2u*sample]=position.x;samples[samplesOffset+2u*sample+1u]=position.y;owners[ownersOffset+sample]=owner;}
 required=min(0xffffffffu-count,required)+count;
}
status[statusOffset]=min(required,${sampleCapacity}u);status[statusOffset+1u]=required;status[statusOffset+2u]=u32(required>${sampleCapacity}u);status[statusOffset+3u]=invalid;`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-clip-cells`,
      operation: id,
      invocationCount: sampleCapacity,
      workgroupSize: 2,
      declarations:
        IS_FINITE_VALUE_WGSL +
        `
fn readSample(row:u32)->vec2<f32>{return vec2<f32>(samples[samplesOffset+2u*row],samples[samplesOffset+2u*row+1u]);}
fn readEnclosure(row:u32)->vec2<f32>{return vec2<f32>(enclosure[enclosureOffset+2u*row],enclosure[enclosureOffset+2u*row+1u]);}
fn crossVectors(first:vec2<f32>,second:vec2<f32>)->f32{return first.x*second.y-first.y*second.x;}`,
      bindings: [
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
for(var other=0u;other<status[statusOffset];other++){
 if(other==index||size==0u){continue;}let displacement=readSample(other)-site;let squared=dot(displacement,displacement);
 if(squared==0.0){if(owners[ownersOffset+other]!=owners[ownersOffset+index]){valid=false;}else if(other<index){size=0u;}continue;}
 let threshold=0.5*squared;var outputSize=0u;
 for(var vertex=0u;vertex<size;vertex++){let start=polygon[vertex];let end=polygon[(vertex+1u)%size];let startSide=dot(start-site,displacement)-threshold;let endSide=dot(end-site,displacement)-threshold;let inside=startSide<=0.0;let nextInside=endSide<=0.0;
 if(inside){if(outputSize<${vertexCapacity}u){scratch[outputSize]=start;}outputSize++;}
 if(inside!=nextInside){if(outputSize<${vertexCapacity}u){scratch[outputSize]=mix(start,end,startSide/(startSide-endSide));}outputSize++;}
 }
 if(outputSize>${vertexCapacity}u){valid=false;break;}size=outputSize;for(var vertex=0u;vertex<size;vertex++){polygon[vertex]=scratch[vertex];}
}
if(!valid){counts[countsOffset+index]=0xffffffffu;return;}
if(size<3u){size=0u;}counts[countsOffset+index]=size;
for(var vertex=0u;vertex<size;vertex++){let destination=fragmentsOffset+2u*(index*${vertexCapacity}u+vertex);fragments[destination]=polygon[vertex].x;fragments[destination+1u]=polygon[vertex].y;}`
    })
  );
  const complete = view('complete', 'uint32', props.buildingCount);
  const outputs: GPUPolygonOverlayOutput[] = [];
  for (let building = 0; building < props.buildingCount; building++) {
    const prefix = `building-${building}`;
    const positions = view(
      `${prefix}-input-positions`,
      'float32x2',
      sampleCapacity * vertexCapacity
    );
    const offsets = view(`${prefix}-input-offsets`, 'uint32', sampleCapacity + 1);
    const indices = view(`${prefix}-input-indices`, 'uint32', sampleCapacity + 1);
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-${prefix}-pack`,
        operation: id,
        invocationCount: 1,
        bindings: [
          {name: 'fragments', view: fragments, type: 'f32', access: 'read'},
          {name: 'counts', view: fragmentCounts, type: 'u32', access: 'read'},
          {name: 'owners', view: owners, type: 'u32', access: 'read'},
          {name: 'positions', view: positions, type: 'f32', access: 'read_write'},
          {name: 'offsets', view: offsets, type: 'u32', access: 'read_write'},
          {name: 'indices', view: indices, type: 'u32', access: 'read_write'}
        ],
        body: `var valid=true;for(var sample=0u;sample<${sampleCapacity}u;sample++){valid=valid&&counts[countsOffset+sample]!=0xffffffffu;}
var written=0u;offsets[offsetsOffset]=0u;indices[indicesOffset]=0u;
for(var sample=0u;sample<${sampleCapacity}u;sample++){if(valid&&owners[ownersOffset+sample]==${building}u){for(var vertex=0u;vertex<counts[countsOffset+sample];vertex++){let source=fragmentsOffset+2u*(sample*${vertexCapacity}u+vertex);positions[positionsOffset+2u*written]=fragments[source];positions[positionsOffset+2u*written+1u]=fragments[source+1u];written++;}}offsets[offsetsOffset+sample+1u]=written;indices[indicesOffset+sample+1u]=sample+1u;}`
      })
    );
    const makeStatus = (name: string) => ({
      count: view(`${prefix}-${name}-count`, 'uint32', 1),
      requiredCount: view(`${prefix}-${name}-required`, 'uint32', 1),
      overflow: view(`${prefix}-${name}-overflow`, 'uint32', 1),
      candidateOverflow: view(`${prefix}-${name}-candidates`, 'uint32', 1)
    });
    let output: GPUPolygonOverlayOutput = {
      geometry: {
        kind: 'polygons',
        positions: view(`${prefix}-positions`, 'float32x2', props.capacity.vertices),
        featureOffsets: view(`${prefix}-features`, 'uint32', props.capacity.rings + 1),
        polygonOffsets: view(`${prefix}-polygons`, 'uint32', props.capacity.rings + 1),
        ringOffsets: view(`${prefix}-rings`, 'uint32', props.capacity.rings + 1),
        sourceIds: view(`${prefix}-sources`, 'uint32', props.capacity.rings)
      },
      status: makeStatus('geometry'),
      boundary: {
        endpoints: view(`${prefix}-boundary`, 'float32x4', props.capacity.nodedSegments),
        operandIds: view(`${prefix}-operands`, 'uint32', props.capacity.nodedSegments),
        sourceFeatureIds: view(
          `${prefix}-boundary-sources`,
          'uint32',
          props.capacity.nodedSegments
        ),
        status: makeStatus('boundary')
      }
    };

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
        output
      })
    );
    const dissolvedOutput = output;
    if (simpleEnclosure) {
      output = {
        geometry: {
          kind: 'polygons',
          positions: view(`${prefix}-clip-positions`, 'float32x2', props.capacity.vertices),
          featureOffsets: view(`${prefix}-clip-features`, 'uint32', props.capacity.rings + 1),
          polygonOffsets: view(`${prefix}-clip-polygons`, 'uint32', props.capacity.rings + 1),
          ringOffsets: view(`${prefix}-clip-rings`, 'uint32', props.capacity.rings + 1),
          sourceIds: view(`${prefix}-clip-sources`, 'uint32', props.capacity.rings)
        },
        status: makeStatus('clip-geometry'),
        boundary: {
          endpoints: view(`${prefix}-clip-boundary`, 'float32x4', props.capacity.nodedSegments),
          operandIds: view(`${prefix}-clip-operands`, 'uint32', props.capacity.nodedSegments),
          sourceFeatureIds: view(
            `${prefix}-clip-boundary-sources`,
            'uint32',
            props.capacity.nodedSegments
          ),
          status: makeStatus('clip-boundary')
        }
      };
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
          output
        })
      );
    }
    outputs.push(output);
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-${prefix}-building-identity`,
        operation: id,
        invocationCount: props.capacity.rings,
        bindings: [
          {name: 'count', view: output.status.count, type: 'u32', access: 'read'},
          {name: 'sourceIds', view: output.geometry.sourceIds, type: 'u32', access: 'read_write'}
        ],
        body: `sourceIds[sourceIdsOffset+index]=select(0xffffffffu,${building}u,index<count[countOffset]);`
      })
    );
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-${prefix}-evidence`,
        operation: id,
        invocationCount: 1,
        bindings: [
          {name: 'sampling', view: samplingStatus, type: 'u32', access: 'read'},
          {name: 'counts', view: fragmentCounts, type: 'u32', access: 'read'},
          {name: 'overflow', view: output.status.overflow!, type: 'u32', access: 'read'},
          {
            name: 'candidateOverflow',
            view: output.status.candidateOverflow,
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
          {name: 'complete', view: complete, type: 'u32', access: 'read_write'}
        ],
        body: `var valid=dissolveOverflow[dissolveOverflowOffset]==0u&&dissolveCandidates[dissolveCandidatesOffset]==0u&&sampling[samplingOffset+2u]==0u&&sampling[samplingOffset+3u]==0u&&overflow[overflowOffset]==0u&&candidateOverflow[candidateOverflowOffset]==0u;for(var sample=0u;sample<${sampleCapacity}u;sample++){valid=valid&&counts[countsOffset+sample]!=0xffffffffu;}complete[completeOffset+${building}u]=u32(valid);`
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
      enclosurePolicy: props.enclosureMode ?? 'convex'
    }
  };
}
