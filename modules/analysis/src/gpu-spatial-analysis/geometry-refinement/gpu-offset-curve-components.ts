// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateOutlineOutputPasses} from '../outline-geometry/validate-outline-output-passes';
import {
  validateRefinementScratchByteLengths,
  validateRefinementLineNodingScratch,
  validateRefinementShaderNumber
} from './refinement-storage-validation';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPULineGeometryFields} from '../../analysis-core/contracts/index';
import {validateGPUMultipartGeometryPort} from '../../analysis-core/contracts/index';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {GPULineNoding} from '../geometry-topology/gpu-line-noding';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  getPackedUint32Values
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createOffsetComponentAssemblyNodes} from './offset-component-assembly';
import {OFFSET_ENDPOINT_TANGENCY_WGSL} from './offset-endpoint-tangency-wgsl';
import {getReadPointWGSL} from '../shared/spatial-wgsl-point-helpers';

/** Fragment metadata has four uint32 words; every scalar index must remain addressable. */
const MAXIMUM_FRAGMENT_WORD_COUNT = 0xffffffff;
/** Small reference route only; larger fragment tables use parallel deterministic continuation. */
export const MAXIMUM_OFFSET_REFERENCE_FRAGMENT_COUNT = 256;
/** Enough bisections to reduce any uint32 feature-offset span to one part. */
const MAXIMUM_OFFSET_FEATURE_LOOKUP_ITERATIONS = 32;

type OffsetCurveLines = Omit<GPULineGeometryFields, 'featureOffsets'> & {
  /** Optional feature-to-part offsets; original and raw must use identical grouping. */
  featureOffsets?: GraphDataView<'uint32'>;
};

export type GPUOffsetCurveComponentsProps = {
  id?: string;
  original: OffsetCurveLines;
  /** Raw offset of each original line, in the same source line order and direction. */
  raw: OffsetCurveLines;
  distance: number;
  /** Absolute geometric slack, including the chord error of a discretized round join. */
  tolerance: number;
  /** GPUOffsetCurve output rows per input vertex, for original edge provenance. */
  rowsPerVertex?: number;
  intersectionCapacity: number;
  segmentCapacity: number;
  output: {
    positions: GraphDataView<'float32x2'>;
    /** [firstVertex, vertexCount, sourceLine, sourceFeature] per component; sourceFeature is 0 without featureOffsets. sourceLine remains the input part row. */
    components: GraphDataView<'uint32x4'>;
    sourceEdges: GraphDataView<'uint32'>;
    /** [writtenComponents,writtenVertices,requiredComponents,requiredVertices], [overflow,incompleteNoding,iterationLimit,0]. */
    status: GraphDataView<'uint32x4'>;
  };
};

/** Decodes the two packed status rows shared by the GPU and CPU component routes. */
export function decodeGPUOffsetCurveComponentsStatus(
  input: Uint32Array | Uint8Array | ArrayBuffer
) {
  const words = getPackedUint32Values(input, 8);
  return {
    writtenComponents: words[0],
    writtenVertices: words[1],
    requiredComponents: words[2],
    requiredVertices: words[3],
    overflow: words[4] !== 0,
    incompleteNoding: words[5] !== 0,
    iterationLimit: words[6] !== 0,
    complete: (words[4] | words[5] | words[6]) === 0
  };
}

/**
 * Node raw offsets, trim source-distance bands and compact components in deterministic greedy order.
 * Assembly uses a reference route for at most 256 reserved fragments. Larger tables compact live
 * rows, cancel and select successors in parallel, and commit one fragment per continuation round.
 * Tables with 8,192 through 65,536 fragment slots run up to 32 rounds per workgroup dispatch.
 * Incoming-edge census visits at most 256 fragments per invocation/dispatch.
 * Assembly compute passes each record at most 2,048 dispatches. Each composite command declares at
 * most MAXIMUM_OFFSET_ASSEMBLY_SEQUENCE_INVOCATION_COUNT = 4,294,967,295 invocations.
 * If clipping or traversal cannot finish within its bounds, iterationLimit is set and the readable
 * component and vertex counts are zero.
 * Capacities remain allocation limits; there is no new total fragment cap or CPU readback.
 */
export class GPUOffsetCurveComponents implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUOffsetCurveComponentsProps;
  constructor(props: GPUOffsetCurveComponentsProps) {
    const snapshot = snapshotConstructionFields(props, [
      'id',
      'original',
      'raw',
      'distance',
      'tolerance',
      'rowsPerVertex',
      'intersectionCapacity',
      'segmentCapacity',
      'output'
    ]);
    props = Object.freeze({
      ...snapshot,
      original: snapshotConstructionFields(snapshot.original, [
        'positions',
        'lineOffsets',
        'featureOffsets'
      ]),
      raw: snapshotConstructionFields(snapshot.raw, ['positions', 'lineOffsets', 'featureOffsets']),
      output: snapshotConstructionFields(snapshot.output, [
        'positions',
        'components',
        'sourceEdges',
        'status'
      ])
    });
    this.id = props.id ?? 'offset-curve-components';
    this.props = props;
    validateRefinementShaderNumber(this.id, 'distance magnitude', Math.abs(props.distance));
    validateRefinementShaderNumber(this.id, 'tolerance', props.tolerance);
    validateCapacity(
      this.id,
      'fragment metadata words',
      props.segmentCapacity * (props.original.positions.length * 6 + 1) * 4,
      1,
      MAXIMUM_FRAGMENT_WORD_COUNT
    );
    validateCapacity(
      this.id,
      'cut parameter and coordinate words',
      props.segmentCapacity * (props.original.positions.length * 6 + 2) * 3,
      1,
      MAXIMUM_FRAGMENT_WORD_COUNT
    );
    validateCapacity(
      this.id,
      'output component words',
      props.output.components.length * 4,
      0,
      MAXIMUM_FRAGMENT_WORD_COUNT
    );
    validateCapacity(
      this.id,
      'output position words',
      props.output.positions.length * 2,
      0,
      MAXIMUM_FRAGMENT_WORD_COUNT
    );
    for (const geometry of [props.original, props.raw]) {
      validateSinglePackedView(geometry.positions, ['float32x2'], this.id);
      validateSinglePackedView(geometry.lineOffsets, ['uint32'], this.id);
      if (geometry.featureOffsets) {
        validateGPUMultipartGeometryPort(this.id, {
          ...geometry,
          kind: 'lines',
          featureOffsets: geometry.featureOffsets
        });
        validateSinglePackedView(geometry.featureOffsets, ['uint32'], `${this.id} featureOffsets`);
      }
    }
    if (props.original.featureOffsets?.length !== props.raw.featureOffsets?.length) {
      throw new Error(
        `${this.id} original.featureOffsets and raw.featureOffsets must both be absent or have matching lengths; use identical feature-to-part grouping on aligned source lines`
      );
    }
    validateCapacity(this.id, 'segmentCapacity', props.segmentCapacity, 1);
    validateCapacity(this.id, 'intersectionCapacity', props.intersectionCapacity, 1);
    validateCapacity(this.id, 'rowsPerVertex', props.rowsPerVertex ?? 1, 1);
    if (
      props.original.lineOffsets.length !== props.raw.lineOffsets.length ||
      !Number.isFinite(props.distance) ||
      !(Number.isFinite(props.tolerance) && props.tolerance >= 0)
    )
      throw new Error('Invalid offset component contract');
    validateSinglePackedView(props.output.positions, ['float32x2'], this.id);
    validateSinglePackedView(props.output.components, ['uint32x4'], this.id);
    validateSinglePackedView(props.output.sourceEdges, ['uint32'], this.id);
    validateSinglePackedView(props.output.status, ['uint32x4'], this.id);
    if (
      props.output.sourceEdges.length !== props.output.positions.length ||
      props.output.status.length !== 2
    )
      throw new Error('Invalid offset output size');
    validateOutlineOutputPasses(this.id, props.output, [Object.keys(props.output)]);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.original.positions,
      props.original.lineOffsets,
      props.original.featureOffsets,
      props.raw.positions,
      props.raw.lineOffsets,
      props.raw.featureOffsets
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props, id} = this;
    const capacity = props.segmentCapacity;
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(
        [
          props.original.positions,
          props.original.lineOffsets,
          props.original.featureOffsets,
          props.raw.positions,
          props.raw.lineOffsets,
          props.raw.featureOffsets,
          ...Object.values(props.output)
        ].map((view, index) => [`binding.${index}`, view])
      )
    );
    validateRefinementLineNodingScratch(
      graph,
      id,
      props.raw.positions.length,
      props.raw.lineOffsets.length - 1,
      props.intersectionCapacity
    );
    const requiredCutCapacity = props.original.positions.length * 6 + 2;
    validateRefinementScratchByteLengths(graph, id, {
      fragmentsAndMetadata: capacity * (requiredCutCapacity - 1) * 16,
      cuts: capacity * requiredCutCapacity * 12,
      nodedEndpoints: capacity * 16,
      assemblyFlagsRanksRowsIncoming: capacity * (requiredCutCapacity - 1) * 4
    });
    const transient = <Format extends 'uint32' | 'float32' | 'float32x4'>(
      name: string,
      format: Format,
      length: number
    ) => createTransientView(graph, `${id}-${name}`, format, length);
    const status = {
      count: transient('count', 'uint32', 1),
      requiredCount: transient('required', 'uint32', 1),
      overflow: transient('overflow', 'uint32', 1),
      candidateOverflow: transient('candidate-overflow', 'uint32', 1)
    };
    const noded = {
      endpoints: transient('endpoints', 'float32x4', capacity),
      sourceFeatureIds: transient('source-lines', 'uint32', capacity),
      sourceRingIds: transient('source-rings', 'uint32', capacity),
      sourceEdgeIds: transient('source-edges', 'uint32', capacity),
      sourceStartParameters: transient('starts', 'float32', capacity),
      sourceEndParameters: transient('ends', 'float32', capacity),
      status
    };
    const cutCapacity = props.original.positions.length * 6 + 2;
    const fragmentCapacity = capacity * (cutCapacity - 1);
    // [parameter, x, y] per cut; one binding keeps the clip kernel within eight storage buffers.
    const cuts = transient('cuts', 'float32', capacity * cutCapacity * 3);
    const fragments = transient('fragments', 'float32x4', fragmentCapacity);
    const metadata = createTransientView(
      graph,
      `${id}-fragment-metadata`,
      'uint32',
      fragmentCapacity * 4
    );
    return [
      ...new GPULineNoding({
        id: `${id}-node`,
        // Noding provenance deliberately stays at the part level. Feature ownership is restored
        // when publishing components; GEOS offsets each multiline part independently.
        lines: {kind: 'lines', positions: props.raw.positions, lineOffsets: props.raw.lineOffsets},
        intersectionCapacity: props.intersectionCapacity,
        output: noded
      }).getCommandNodes(graph),
      createWGSLKernelNode(graph, {
        id: `${id}-clip-distance-band`,
        operation: 'GPUOffsetCurveComponents',
        invocationCount: capacity,
        bindings: [
          {name: 'positions', view: props.original.positions, type: 'f32', access: 'read'},
          {name: 'offsets', view: props.original.lineOffsets, type: 'u32', access: 'read'},
          {name: 'endpoints', view: noded.endpoints, type: 'f32', access: 'read'},
          {name: 'lines', view: noded.sourceFeatureIds, type: 'u32', access: 'read'},
          {name: 'edges', view: noded.sourceEdgeIds, type: 'u32', access: 'read'},
          {name: 'cuts', view: cuts, type: 'f32', access: 'read_write'},
          {name: 'fragments', view: fragments, type: 'f32', access: 'read_write'},
          {name: 'metadata', view: metadata, type: 'u32', access: 'read_write'}
        ],
        declarations: `const CUTS:u32=${cutCapacity}u;
const ORIGINAL_POSITION_COUNT:u32=${props.original.positions.length}u;
const DISTANCE:f32=${getWGSLFloatLiteral(Math.abs(props.distance))};
const TOLERANCE:f32=${getWGSLFloatLiteral(props.tolerance)};
${getReadPointWGSL('point', 'positions')}
${OFFSET_ENDPOINT_TANGENCY_WGSL}
fn insertCut(base:u32,count:u32,value:f32,position:vec2f)->u32 {
 if(!(value>0.0 && value<1.0) || count>=CUTS) {return count;}
 var destination=count;
 for(var slot=count;slot>0u;slot--) {
  if(cuts[cutsOffset+(base+slot-1u)*3u]<=value) {break;}
  destination=slot-1u;
 }
 if(destination>0u && cuts[cutsOffset+(base+destination-1u)*3u]==value) {return count;}
 for(var slot=count;slot>destination;slot--) {for(var component=0u;component<3u;component++) {cuts[cutsOffset+(base+slot)*3u+component]=cuts[cutsOffset+(base+slot-1u)*3u+component];}}
 cuts[cutsOffset+(base+destination)*3u]=value;cuts[cutsOffset+(base+destination)*3u+1u]=position.x;cuts[cutsOffset+(base+destination)*3u+2u]=position.y;return count+1u;
}
fn separation(query:vec2f,firstEdge:u32,endEdge:u32,owner:u32,outerJoin:bool,roundingSign:f32)->f32 {
 var minimum=3.0e38;
 for(var vertex=firstEdge;vertex<endEdge;vertex++) {
  if(outerJoin && (vertex==owner || vertex+1u==owner)) {continue;}
  let start=point(vertex);let direction=point(vertex+1u)-start;let denominator=dot(direction,direction);if(denominator==0.0) {continue;}
  let delta=query-start;let projection=dot(delta,direction)/denominator;
  var normal=vec2f(-direction.y,direction.x)/sqrt(denominator);
  var separation=abs(delta.x*normal.x+delta.y*normal.y);
  if(projection<=0.0) {separation=length(delta);if(separation>0.0) {normal=delta/separation;}}
  if(projection>=1.0) {let residual=query-point(vertex+1u);separation=length(residual);if(separation>0.0) {normal=residual/separation;}}
  let roundingFloor=1.1920928955078125e-7*(DISTANCE+abs(query.x*normal.x)+abs(query.y*normal.y));
  minimum=min(minimum,separation+roundingSign*roundingFloor);
 }
 return minimum;
}`,
        body: `let base=index*CUTS;let fragmentBase=index*(CUTS-1u);
for(var slot=0u;slot<CUTS-1u;slot++) {metadata[metadataOffset+(fragmentBase+slot)*4u+2u]=0u;}
let lossWord=metadataOffset+fragmentBase*4u+3u;metadata[lossWord]=0u;
let line=lines[linesOffset+index];if(line>=${props.original.lineOffsets.length - 1}u) {return;}
let firstVertex=offsets[offsetsOffset+line];let endVertex=offsets[offsetsOffset+line+1u];
if(firstVertex>endVertex || endVertex>ORIGINAL_POSITION_COUNT) {metadata[lossWord]=1u;return;}
let endEdge=max(endVertex,1u)-1u;
let first=vec2f(endpoints[endpointsOffset+index*4u],endpoints[endpointsOffset+index*4u+1u]);let last=vec2f(endpoints[endpointsOffset+index*4u+2u],endpoints[endpointsOffset+index*4u+3u]);let direction=last-first;let squaredLength=dot(direction,direction);if(squaredLength==0.0) {return;}
let rawEdge=edges[edgesOffset+index];let owner=rawEdge/${props.rowsPerVertex ?? 1}u;var outerJoin=false;
if(rawEdge%${props.rowsPerVertex ?? 1}u+1u<${props.rowsPerVertex ?? 1}u && owner>firstVertex && owner<endEdge) {
 let incoming=point(owner)-point(owner-1u);let outgoing=point(owner+1u)-point(owner);
 outerJoin=(incoming.x*outgoing.y-incoming.y*outgoing.x)*${getWGSLFloatLiteral(props.distance)}<0.0;
}
cuts[cutsOffset+base*3u]=0.0;cuts[cutsOffset+base*3u+1u]=first.x;cuts[cutsOffset+base*3u+2u]=first.y;
cuts[cutsOffset+(base+1u)*3u]=1.0;cuts[cutsOffset+(base+1u)*3u+1u]=last.x;cuts[cutsOffset+(base+1u)*3u+2u]=last.y;var count=2u;
for(var vertex=firstVertex;vertex<endEdge;vertex++) {
 if(outerJoin && (vertex==owner || vertex+1u==owner)) {continue;}
 let start=point(vertex);let finish=point(vertex+1u);let edge=finish-start;let lengthSquared=dot(edge,edge);if(lengthSquared==0.0) {continue;}
 let normal=vec2f(-edge.y,edge.x)/sqrt(lengthSquared);let denominator=dot(direction,normal);
 if(abs(denominator)>1e-20) {for(var side=0u;side<2u;side++) {
  let bandDistance=select(-DISTANCE,DISTANCE,side==1u);
  let fraction=(bandDistance-dot(first-start,normal))/denominator;var position=first+fraction*direction;
  if(normal.x==0.0) {position.y=start.y+bandDistance*normal.y;}
  if(normal.y==0.0) {position.x=start.x+bandDistance*normal.x;}
  let projection=dot(position-start,edge)/lengthSquared;
  if(projection>=0.0 && projection<=1.0) {count=insertCut(base,count,fraction,position);}
 }}
 for(var endpoint=0u;endpoint<2u;endpoint++) {
  let center=select(start,finish,endpoint==1u);
  if(endpointTangent(first-center,direction,DISTANCE) || endpointTangent(last-center,direction,DISTANCE)) {continue;}
  let delta=first-center;var segmentLength=sqrt(squaredLength);var unit=direction/segmentLength;
  if(direction.y==0.0) {segmentLength=abs(direction.x);unit=vec2f(sign(direction.x),0.0);}
  else if(direction.x==0.0) {segmentLength=abs(direction.y);unit=vec2f(0.0,sign(direction.y));}
  let along=dot(delta,unit);let across=delta.x*unit.y-delta.y*unit.x;
  let discriminant=(DISTANCE-abs(across))*(DISTANCE+abs(across));
  if(discriminant>=0.0) {
   let root=sqrt(discriminant);
   // Construct the geometric circle intersection directly. Reconstructing a tangent via a
   // rounded fraction can turn x=3 into 3.000000238 and prevent a reversed spur cancelling.
   let foot=center+across*vec2f(unit.y,-unit.x);
   count=insertCut(base,count,(-along-root)/segmentLength,foot-root*unit);
   count=insertCut(base,count,(-along+root)/segmentLength,foot+root*unit);
  }
 }
}
// Cut coordinates are stored once at insertion, and moved with their parameters while sorting.
let outward=vec2f(-direction.y,direction.x)/sqrt(squaredLength);
for(var slot=0u;slot+1u<count;slot++) {
 let startParameter=cuts[cutsOffset+(base+slot)*3u];let endParameter=cuts[cutsOffset+(base+slot+1u)*3u];if(endParameter<=startParameter) {continue;}
 let start=vec2f(cuts[cutsOffset+(base+slot)*3u+1u],cuts[cutsOffset+(base+slot)*3u+2u]);
 let finish=vec2f(cuts[cutsOffset+(base+slot+1u)*3u+1u],cuts[cutsOffset+(base+slot+1u)*3u+2u]);
 if(all(start==finish)) {continue;}
 let middle=first+0.5*(startParameter+endParameter)*direction;
 // Probe far enough to move along the normal (four relative ULPs of coordinate magnitude), independently
 // of endpoint identity. A 1e-5-unit probe cannot move x at projected-coordinate magnitudes.
 let roundingFloor=1.1920928955078125e-7*max(DISTANCE,max(abs(middle.x*outward.x),abs(middle.y*outward.y)));
 let roundingStep=4.0*roundingFloor;
 let epsilon=max(roundingStep,max(1e-5,max(TOLERANCE*2.0,DISTANCE*1e-5)));
 if(separation(middle,firstVertex,endEdge,owner,outerJoin,1.0)+TOLERANCE<DISTANCE) {continue;}
 // A probe longer than this interval cannot decide its local two-sided coverage. Retain the
 // already distance-band-selected interval rather than removing a tiny connector.
 if(distance(start,finish)>epsilon && max(separation(middle+epsilon*outward,firstVertex,endEdge,owner,outerJoin,-1.0),separation(middle-epsilon*outward,firstVertex,endEdge,owner,outerJoin,-1.0))<=DISTANCE+epsilon*0.05) {continue;}
 let fragment=fragmentBase+slot;
 fragments[fragmentsOffset+fragment*4u]=start.x;fragments[fragmentsOffset+fragment*4u+1u]=start.y;fragments[fragmentsOffset+fragment*4u+2u]=finish.x;fragments[fragmentsOffset+fragment*4u+3u]=finish.y;
 metadata[metadataOffset+fragment*4u]=line;metadata[metadataOffset+fragment*4u+1u]=owner;metadata[metadataOffset+fragment*4u+2u]=1u;
}`
      }),
      ...(fragmentCapacity <= MAXIMUM_OFFSET_REFERENCE_FRAGMENT_COUNT
        ? [
            createWGSLKernelNode(graph, {
              id: `${id}-clear-components`,
              operation: 'GPUOffsetCurveComponents',
              invocationCount: props.output.components.length * 4,
              bindings: [
                {
                  name: 'components',
                  view: props.output.components,
                  type: 'u32',
                  access: 'read_write'
                }
              ],
              body: 'components[componentsOffset+index]=0u;'
            }),
            createWGSLKernelNode(graph, {
              id: `${id}-components`,
              operation: 'GPUOffsetCurveComponents',
              // contract-allow: single-invocation reference route bounded by MAXIMUM_OFFSET_REFERENCE_FRAGMENT_COUNT = 256 fragments.
              invocationCount: 1,
              bindings: [
                {name: 'fragments', view: fragments, type: 'f32', access: 'read'},
                {name: 'metadata', view: metadata, type: 'u32', access: 'read_write'},
                {
                  name: 'positions',
                  view: props.output.positions,
                  type: 'f32',
                  access: 'read_write'
                },
                {
                  name: 'components',
                  view: props.output.components,
                  type: 'u32',
                  access: 'read_write'
                },
                {
                  name: 'sources',
                  view: props.output.sourceEdges,
                  type: 'u32',
                  access: 'read_write'
                },
                {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
              ],
              declarations: `const CAPACITY:u32=${fragmentCapacity}u;
fn startPoint(fragment:u32)->vec2f {return vec2f(fragments[fragmentsOffset+fragment*4u],fragments[fragmentsOffset+fragment*4u+1u]);}
fn endPoint(fragment:u32)->vec2f {return vec2f(fragments[fragmentsOffset+fragment*4u+2u],fragments[fragmentsOffset+fragment*4u+3u]);}
// Shared noded/clipped endpoints establish topology; proximity must never bridge a clipped gap.
fn matches(left:vec2f,right:vec2f)->bool {return all(left==right);}
fn writeVertex(index:u32,value:vec2f,source:u32) {if(index<${props.output.positions.length}u) {positions[positionsOffset+index*2u]=value.x;positions[positionsOffset+index*2u+1u]=value.y;sources[sourcesOffset+index]=source;}}`,
              body: `var componentCount=0u;var vertexCount=0u;var writtenComponents=0u;var writtenVertices=0u;
// Opposite coincident fragments bound no offset component; cancel collapsed spurs.
for(var candidate=0u;candidate<CAPACITY;candidate++) {if(metadata[metadataOffset+candidate*4u+2u]!=1u) {continue;}
 for(var other=candidate+1u;other<CAPACITY;other++) {if(metadata[metadataOffset+other*4u+2u]!=1u || metadata[metadataOffset+other*4u]!=metadata[metadataOffset+candidate*4u]) {continue;}
  if(matches(startPoint(candidate),endPoint(other)) && matches(endPoint(candidate),startPoint(other))) {metadata[metadataOffset+candidate*4u+2u]=0u;metadata[metadataOffset+other*4u+2u]=0u;break;}
  if(matches(startPoint(candidate),startPoint(other)) && matches(endPoint(candidate),endPoint(other))) {metadata[metadataOffset+other*4u+2u]=0u;}
 }
}
var consumed=0u;var iterationLimit=0u;
for(var componentRound=0u;componentRound<CAPACITY && consumed<CAPACITY;componentRound++) {
 var seed=0xffffffffu;
 for(var candidate=0u;candidate<CAPACITY;candidate++) {if(metadata[metadataOffset+candidate*4u+2u]!=1u) {continue;}
  if(seed==0xffffffffu) {seed=candidate;}
  var incoming=false;for(var other=0u;other<CAPACITY;other++) {if(other!=candidate && metadata[metadataOffset+other*4u+2u]==1u && metadata[metadataOffset+other*4u]==metadata[metadataOffset+candidate*4u] && matches(endPoint(other),startPoint(candidate))) {incoming=true;break;}}
  if(!incoming) {seed=candidate;break;}
 }
 if(seed==0xffffffffu) {break;}
 let componentStart=vertexCount;let line=metadata[metadataOffset+seed*4u];var fragment=seed;
 writeVertex(vertexCount,startPoint(seed),metadata[metadataOffset+seed*4u+1u]);vertexCount++;
 for(var step=0u;step<CAPACITY;step++) {
  if(consumed==CAPACITY || fragment==0xffffffffu) {break;}
  consumed++;
  metadata[metadataOffset+fragment*4u+2u]=2u;let finish=endPoint(fragment);writeVertex(vertexCount,finish,metadata[metadataOffset+fragment*4u+1u]);vertexCount++;
  var next=0xffffffffu;for(var candidate=0u;candidate<CAPACITY;candidate++) {if(metadata[metadataOffset+candidate*4u+2u]==1u && metadata[metadataOffset+candidate*4u]==line && matches(finish,startPoint(candidate))) {next=candidate;break;}}
  fragment=next;
 }
 if(fragment!=0xffffffffu) {iterationLimit=1u;break;}
 if(componentCount<${props.output.components.length}u && vertexCount<=${props.output.positions.length}u) {components[componentsOffset+componentCount*4u]=componentStart;components[componentsOffset+componentCount*4u+1u]=vertexCount-componentStart;components[componentsOffset+componentCount*4u+2u]=line;writtenComponents++;writtenVertices=vertexCount;}componentCount++;
}
let overflow=componentCount>${props.output.components.length}u || vertexCount>${props.output.positions.length}u;
status[statusOffset]=writtenComponents;status[statusOffset+1u]=writtenVertices;status[statusOffset+2u]=componentCount;status[statusOffset+3u]=vertexCount;status[statusOffset+4u]=select(0u,1u,overflow);status[statusOffset+6u]=iterationLimit;status[statusOffset+7u]=0u;`
            })
          ]
        : createOffsetComponentAssemblyNodes(graph, {
            id: `${id}-assembly`,
            capacity: fragmentCapacity,
            fragments,
            metadata,
            output: props.output
          })),
      createWGSLKernelNode(graph, {
        id: `${id}-assembly-completeness`,
        operation: 'GPUOffsetCurveComponents',
        invocationCount: fragmentCapacity,
        bindings: [
          {name: 'metadata', view: metadata, type: 'u32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'atomic<u32>', access: 'read_write'}
        ],
        // The fourth metadata word is written at the first fragment of each noded segment.
        body: `let base=metadataOffset+index*4u;
if(metadata[base+2u]==1u || (index%${cutCapacity - 1}u==0u && metadata[base+3u]!=0u)) {
 atomicOr(&status[statusOffset+6u],1u);
}`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-noding-status`,
        operation: 'GPUOffsetCurveComponents',
        invocationCount: 1,
        bindings: [
          {name: 'overflow', view: status.overflow, type: 'u32', access: 'read'},
          {name: 'candidate', view: status.candidateOverflow, type: 'u32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        body: `status[statusOffset+5u]=overflow[overflowOffset]|candidate[candidateOffset];
if(status[statusOffset+6u]!=0u) {status[statusOffset]=0u;status[statusOffset+1u]=0u;}`
      }),
      ...(props.original.featureOffsets
        ? [
            createWGSLKernelNode(graph, {
              id: `${id}-component-features`,
              operation: 'GPUOffsetCurveComponents',
              invocationCount: props.output.components.length,
              bindings: [
                {
                  name: 'features',
                  view: props.original.featureOffsets,
                  type: 'u32',
                  access: 'read'
                },
                {name: 'status', view: props.output.status, type: 'u32', access: 'read'},
                {
                  name: 'components',
                  view: props.output.components,
                  type: 'u32',
                  access: 'read_write'
                }
              ],
              body: `if(index>=status[statusOffset]) {return;}
let base=componentsOffset+index*4u;let line=components[base+2u];
var low=0u;var high=${props.original.featureOffsets.length - 1}u;
for(var step=0u;step<${MAXIMUM_OFFSET_FEATURE_LOOKUP_ITERATIONS}u && high-low>1u;step++) {
 let middle=low+(high-low)/2u;if(features[featuresOffset+middle]<=line) {low=middle;} else {high=middle;}
}
components[base+3u]=low;`
            })
          ]
        : [])
    ];
  }
}
