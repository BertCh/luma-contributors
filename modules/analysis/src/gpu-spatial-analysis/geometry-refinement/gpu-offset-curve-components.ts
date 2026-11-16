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
import {GPULineNoding} from '../geometry-topology/gpu-line-noding';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getReadPointWGSL} from '../shared/spatial-wgsl-point-helpers';

/** Fragment metadata has four uint32 words; every scalar index must remain addressable. */
const MAXIMUM_FRAGMENT_WORD_COUNT = 0xffffffff;

export type GPUOffsetCurveComponentsProps = {
  id?: string;
  original: GPULineGeometryFields;
  /** Raw offset of each original line, in the same source line order and direction. */
  raw: GPULineGeometryFields;
  distance: number;
  /** Absolute geometric slack, including the chord error of a discretized round join. */
  tolerance: number;
  /** GPUOffsetCurve output rows per input vertex, for original edge provenance. */
  rowsPerVertex?: number;
  intersectionCapacity: number;
  segmentCapacity: number;
  output: {
    positions: GraphDataView<'float32x2'>;
    /** [firstVertex, vertexCount, sourceLine, 0] per connected selected component. */
    components: GraphDataView<'uint32x4'>;
    sourceEdges: GraphDataView<'uint32'>;
    /** [writtenComponents,writtenVertices,requiredComponents,requiredVertices], [overflow,incompleteNoding,0,0]. */
    status: GraphDataView<'uint32x4'>;
  };
};

/** Node raw offsets, remove portions inside the source-distance band, select the requested side and compact multipart components. */
export class GPUOffsetCurveComponents implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUOffsetCurveComponentsProps;
  constructor(props: GPUOffsetCurveComponentsProps) {
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
      props.raw.positions,
      props.raw.lineOffsets
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
          props.raw.positions,
          props.raw.lineOffsets,
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
      cuts: capacity * requiredCutCapacity * 4,
      fragmentsAndMetadata: capacity * (requiredCutCapacity - 1) * 16,
      nodedEndpoints: capacity * 16
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
    const cuts = transient('cuts', 'float32', capacity * cutCapacity);
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
        lines: {kind: 'lines', ...props.raw},
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
const DISTANCE:f32=${getWGSLFloatLiteral(Math.abs(props.distance))};
const TOLERANCE:f32=${getWGSLFloatLiteral(props.tolerance)};
${getReadPointWGSL('point', 'positions')}
fn insertCut(base:u32,count:u32,value:f32)->u32 {
 if(!(value>0.0 && value<1.0) || count>=CUTS) {return count;}
 var destination=count;
 loop {if(destination==0u || cuts[cutsOffset+base+destination-1u]<=value) {break;} cuts[cutsOffset+base+destination]=cuts[cutsOffset+base+destination-1u];destination--;}
 cuts[cutsOffset+base+destination]=value;return count+1u;
}
fn separation(query:vec2f,line:u32,owner:u32,outerJoin:bool)->f32 {
 var minimum=3.0e38;
 for(var vertex=offsets[offsetsOffset+line];vertex+1u<offsets[offsetsOffset+line+1u];vertex++) {
  if(outerJoin && (vertex==owner || vertex+1u==owner)) {continue;}
  let start=point(vertex);let direction=point(vertex+1u)-start;let denominator=dot(direction,direction);if(denominator==0.0) {continue;}
  let closest=start+clamp(dot(query-start,direction)/denominator,0.0,1.0)*direction;minimum=min(minimum,distance(query,closest));
 }
 return minimum;
}`,
        body: `let base=index*CUTS;let fragmentBase=index*(CUTS-1u);
for(var slot=0u;slot<CUTS-1u;slot++) {metadata[metadataOffset+(fragmentBase+slot)*4u+2u]=0u;}
let line=lines[linesOffset+index];if(line>=${props.original.lineOffsets.length - 1}u) {return;}
let first=vec2f(endpoints[endpointsOffset+index*4u],endpoints[endpointsOffset+index*4u+1u]);let last=vec2f(endpoints[endpointsOffset+index*4u+2u],endpoints[endpointsOffset+index*4u+3u]);let direction=last-first;let squaredLength=dot(direction,direction);if(squaredLength==0.0) {return;}
let rawEdge=edges[edgesOffset+index];let owner=rawEdge/${props.rowsPerVertex ?? 1}u;var outerJoin=false;
if(rawEdge%${props.rowsPerVertex ?? 1}u+1u<${props.rowsPerVertex ?? 1}u && owner>offsets[offsetsOffset+line] && owner+1u<offsets[offsetsOffset+line+1u]) {
 let incoming=point(owner)-point(owner-1u);let outgoing=point(owner+1u)-point(owner);
 outerJoin=(incoming.x*outgoing.y-incoming.y*outgoing.x)*${getWGSLFloatLiteral(props.distance)}<0.0;
}
cuts[cutsOffset+base]=0.0;cuts[cutsOffset+base+1u]=1.0;var count=2u;
for(var vertex=offsets[offsetsOffset+line];vertex+1u<offsets[offsetsOffset+line+1u];vertex++) {
 let start=point(vertex);let finish=point(vertex+1u);let edge=finish-start;let lengthSquared=dot(edge,edge);if(lengthSquared==0.0) {continue;}
 let normal=vec2f(-edge.y,edge.x)/sqrt(lengthSquared);let denominator=dot(direction,normal);
 if(abs(denominator)>1e-20) {for(var side=0u;side<2u;side++) {
  let fraction=(select(-DISTANCE,DISTANCE,side==1u)-dot(first-start,normal))/denominator;let projection=dot(first+fraction*direction-start,edge)/lengthSquared;
  if(projection>=0.0 && projection<=1.0) {count=insertCut(base,count,fraction);}
 }}
 for(var endpoint=0u;endpoint<2u;endpoint++) {
  let delta=first-select(start,finish,endpoint==1u);let linear=dot(delta,direction);let constant=dot(delta,delta)-DISTANCE*DISTANCE;let discriminant=linear*linear-squaredLength*constant;
  if(discriminant>=0.0) {let root=sqrt(discriminant);count=insertCut(base,count,(-linear-root)/squaredLength);count=insertCut(base,count,(-linear+root)/squaredLength);}
 }
}
let outward=vec2f(-direction.y,direction.x)/sqrt(squaredLength);let epsilon=max(1e-5,max(TOLERANCE*2.0,DISTANCE*1e-5));
for(var slot=0u;slot+1u<count;slot++) {
 let startParameter=cuts[cutsOffset+base+slot];let endParameter=cuts[cutsOffset+base+slot+1u];if(endParameter-startParameter<=1e-7) {continue;}
 let middle=first+0.5*(startParameter+endParameter)*direction;
 if(separation(middle,line,owner,outerJoin)+TOLERANCE<DISTANCE) {continue;}
 if(max(separation(middle+epsilon*outward,line,owner,outerJoin),separation(middle-epsilon*outward,line,owner,outerJoin))<=DISTANCE+epsilon*0.05) {continue;}
 let start=first+startParameter*direction;let finish=first+endParameter*direction;let fragment=fragmentBase+slot;
 fragments[fragmentsOffset+fragment*4u]=start.x;fragments[fragmentsOffset+fragment*4u+1u]=start.y;fragments[fragmentsOffset+fragment*4u+2u]=finish.x;fragments[fragmentsOffset+fragment*4u+3u]=finish.y;
 metadata[metadataOffset+fragment*4u]=line;metadata[metadataOffset+fragment*4u+1u]=owner;metadata[metadataOffset+fragment*4u+2u]=1u;
}`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-components`,
        operation: 'GPUOffsetCurveComponents',
        invocationCount: 1,
        bindings: [
          {name: 'fragments', view: fragments, type: 'f32', access: 'read'},
          {name: 'metadata', view: metadata, type: 'u32', access: 'read_write'},
          {name: 'positions', view: props.output.positions, type: 'f32', access: 'read_write'},
          {name: 'components', view: props.output.components, type: 'u32', access: 'read_write'},
          {name: 'sources', view: props.output.sourceEdges, type: 'u32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `const CAPACITY:u32=${fragmentCapacity}u;
fn startPoint(fragment:u32)->vec2f {return vec2f(fragments[fragmentsOffset+fragment*4u],fragments[fragmentsOffset+fragment*4u+1u]);}
fn endPoint(fragment:u32)->vec2f {return vec2f(fragments[fragmentsOffset+fragment*4u+2u],fragments[fragmentsOffset+fragment*4u+3u]);}
fn matches(left:vec2f,right:vec2f)->bool {return distance(left,right)<=${getWGSLFloatLiteral(Math.max(1e-5, Math.abs(props.distance) * 1e-5))};}
fn writeVertex(index:u32,value:vec2f,source:u32) {if(index<${props.output.positions.length}u) {positions[positionsOffset+index*2u]=value.x;positions[positionsOffset+index*2u+1u]=value.y;sources[sourcesOffset+index]=source;}}`,
        body: `var componentCount=0u;var vertexCount=0u;var writtenComponents=0u;var writtenVertices=0u;
for(var slot=0u;slot<${props.output.components.length * 4}u;slot++) {components[componentsOffset+slot]=0u;}
// Opposite coincident fragments bound no offset component; cancel collapsed spurs.
for(var candidate=0u;candidate<CAPACITY;candidate++) {if(metadata[metadataOffset+candidate*4u+2u]!=1u) {continue;}
 for(var other=candidate+1u;other<CAPACITY;other++) {if(metadata[metadataOffset+other*4u+2u]!=1u || metadata[metadataOffset+other*4u]!=metadata[metadataOffset+candidate*4u]) {continue;}
  if(matches(startPoint(candidate),endPoint(other)) && matches(endPoint(candidate),startPoint(other))) {metadata[metadataOffset+candidate*4u+2u]=0u;metadata[metadataOffset+other*4u+2u]=0u;break;}
  if(matches(startPoint(candidate),startPoint(other)) && matches(endPoint(candidate),endPoint(other))) {metadata[metadataOffset+other*4u+2u]=0u;}
 }
}
loop {
 var seed=0xffffffffu;
 for(var candidate=0u;candidate<CAPACITY;candidate++) {if(metadata[metadataOffset+candidate*4u+2u]!=1u) {continue;}
  if(seed==0xffffffffu) {seed=candidate;}
  var incoming=false;for(var other=0u;other<CAPACITY;other++) {if(other!=candidate && metadata[metadataOffset+other*4u+2u]==1u && metadata[metadataOffset+other*4u]==metadata[metadataOffset+candidate*4u] && matches(endPoint(other),startPoint(candidate))) {incoming=true;break;}}
  if(!incoming) {seed=candidate;break;}
 }
 if(seed==0xffffffffu) {break;}
 let componentStart=vertexCount;let line=metadata[metadataOffset+seed*4u];var fragment=seed;
 writeVertex(vertexCount,startPoint(seed),metadata[metadataOffset+seed*4u+1u]);vertexCount++;
 loop {
  metadata[metadataOffset+fragment*4u+2u]=2u;let finish=endPoint(fragment);writeVertex(vertexCount,finish,metadata[metadataOffset+fragment*4u+1u]);vertexCount++;
  var next=0xffffffffu;for(var candidate=0u;candidate<CAPACITY;candidate++) {if(metadata[metadataOffset+candidate*4u+2u]==1u && metadata[metadataOffset+candidate*4u]==line && matches(finish,startPoint(candidate))) {next=candidate;break;}}
  if(next==0xffffffffu) {break;}fragment=next;
 }
 if(componentCount<${props.output.components.length}u && vertexCount<=${props.output.positions.length}u) {components[componentsOffset+componentCount*4u]=componentStart;components[componentsOffset+componentCount*4u+1u]=vertexCount-componentStart;components[componentsOffset+componentCount*4u+2u]=line;writtenComponents++;writtenVertices=vertexCount;}componentCount++;
}
let overflow=componentCount>${props.output.components.length}u || vertexCount>${props.output.positions.length}u;
status[statusOffset]=writtenComponents;status[statusOffset+1u]=writtenVertices;status[statusOffset+2u]=componentCount;status[statusOffset+3u]=vertexCount;status[statusOffset+4u]=select(0u,1u,overflow);status[statusOffset+6u]=0u;status[statusOffset+7u]=0u;`
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
        body: `status[statusOffset+5u]=overflow[overflowOffset]|candidate[candidateOffset];`
      })
    ];
  }
}
