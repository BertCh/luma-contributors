// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUPolygonGeometryPort} from '../../analysis-core/contracts/ports';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  type GPUSpatialWeights,
  type GPUSpatialWeightsOutput,
  describeGPUSpatialWeights,
  validateGPUSpatialWeightsOutput,
  getGPUSpatialWeightsViews,
  getGPUSpatialWeightsStatusViews
} from './spatial-weights';

export type GPUFuzzyContiguityProps = {
  id?: string;
  polygons: Omit<GPUPolygonGeometryPort, 'positions'> & {positions: GraphDataView<'float32x2'>};
  /** Planar minimum polygon separation. Equivalent to intersecting exact round buffers of radius tolerance/2. */
  tolerance: number;
  output: GPUSpatialWeightsOutput & {
    /** Per-feature: 0 valid; 1 empty; 2 malformed/nonfinite. Invalid features produce no edges. */
    featureStatus: GraphDataView<'uint32'>;
    /** 1 if edge exists only because of positive separation tolerance; 0 for exact intersection. */
    added: GraphDataView<'uint32'>;
  };
};

/**
 * Distance-based fuzzy polygon contiguity, including multipart geometry and holes. Exhaustive
 * planar segment distances plus even-odd containment establish filled-polygon separation; a
 * vertex grid snap is never substituted. Rings close implicitly. Valid simple polygon rings are
 * a precondition; self intersections and f32 near-degeneracy are not repaired. Empty/malformed
 * features remain explicit. Symmetric binary sorted CSR, exact required slot count and overflow.
 * This bounded baseline stores a dense <=512² distance table; no geographic coordinates/buffering
 * polygon output or general GEOS topological robustness is claimed.
 */
export class GPUFuzzyContiguity implements GPUCommandNodeProducer {
  readonly id: string;
  readonly outputWeights: GPUSpatialWeights;
  constructor(readonly props: GPUFuzzyContiguityProps) {
    this.id = props.id ?? 'fuzzy-contiguity';
    const rows = props.polygons.featureOffsets.length - 1;
    validateCapacity(this.id, 'features', rows, 1, 512);
    validateCapacity(this.id, 'vertices', props.polygons.positions.length, 1, 65536);
    if (!Number.isFinite(props.tolerance) || props.tolerance < 0 || props.tolerance > 1e18)
      throw new Error(`${this.id}: invalid tolerance`);
    if (
      validateGPUSpatialWeightsOutput(this.id, props.output) !== rows ||
      props.output.featureStatus.length !== rows ||
      props.output.added.length !== props.output.neighbors.length
    )
      throw new Error(`${this.id}: shape mismatch`);
    validatePackedView(props.polygons.positions, ['float32x2'], this.id);
    for (const view of [
      props.polygons.featureOffsets,
      props.polygons.polygonOffsets,
      props.polygons.ringOffsets,
      props.output.featureStatus,
      props.output.added
    ])
      validatePackedView(view, ['uint32'], this.id);
    if (props.polygons.polygonOffsets.length < 1 || props.polygons.ringOffsets.length < 1)
      throw new Error(`${this.id}: terminal offsets required`);
    validateGraphOutputsDisjointFromInputs(
      this.id,
      [
        ...getGPUSpatialWeightsViews(props.output),
        ...getGPUSpatialWeightsStatusViews(props.output),
        props.output.featureStatus,
        props.output.added
      ],
      [
        props.polygons.positions,
        props.polygons.featureOffsets,
        props.polygons.polygonOffsets,
        props.polygons.ringOffsets
      ]
    );
    this.outputWeights = describeGPUSpatialWeights(props.output, {
      information: {
        rowCount: rows,
        transform: 'binary',
        symmetric: true,
        hasSelfWeights: false,
        provenance: {
          source: 'GPUFuzzyContiguity',
          criterion: 'polygon-distance',
          radius: props.tolerance
        }
      },
      status: props.output.status
    });
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const rows = props.polygons.featureOffsets.length - 1;
    const capacity = props.output.neighbors.length;
    const polygons = props.polygons;
    const distances = createTransientView(graph, `${id}-distances`, 'float32', rows * rows);
    const binding = (
      name: string,
      view: GraphDataView,
      access: 'read' | 'read_write' = 'read',
      type: 'f32' | 'u32' = 'u32'
    ): WGSLKernelBinding => ({name, view, access, type});
    const geometry = [
      binding('positions', polygons.positions, 'read', 'f32'),
      binding('features', polygons.featureOffsets),
      binding('polygons', polygons.polygonOffsets),
      binding('rings', polygons.ringOffsets)
    ];
    const common = `${GET_NAN_WGSL}\n${IS_FINITE_BITS_WGSL}
fn position(vertex:u32)->vec2<f32>{return vec2<f32>(positions[positionsOffset+vertex*2u],positions[positionsOffset+vertex*2u+1u]);}
fn cross(first:vec2<f32>,second:vec2<f32>)->f32{return first.x*second.y-first.y*second.x;}
fn pointDistance(point:vec2<f32>,first:vec2<f32>,second:vec2<f32>)->f32{let direction=second-first;let square=dot(direction,direction);var fraction=0.0;if(square>0.0){fraction=clamp(dot(point-first,direction)/square,0.0,1.0);}return distance(point,first+fraction*direction);}
fn segmentDistance(first:vec2<f32>,second:vec2<f32>,third:vec2<f32>,fourth:vec2<f32>)->f32{
 let direction=second-first;let other=fourth-third;let determinant=cross(direction,other);
 if(determinant!=0.0){let firstFraction=cross(third-first,other)/determinant;let secondFraction=cross(third-first,direction)/determinant;if(firstFraction>=0.0&&firstFraction<=1.0&&secondFraction>=0.0&&secondFraction<=1.0){return 0.0;}}
 return min(min(pointDistance(first,third,fourth),pointDistance(second,third,fourth)),min(pointDistance(third,first,second),pointDistance(fourth,first,second)));
}
fn contains(feature:u32,point:vec2<f32>)->bool{
 for(var polygon=features[featuresOffset+feature];polygon<features[featuresOffset+feature+1u];polygon++){var inside=false;
 for(var ring=polygons[polygonsOffset+polygon];ring<polygons[polygonsOffset+polygon+1u];ring++){let begin=rings[ringsOffset+ring];let end=rings[ringsOffset+ring+1u];for(var vertex=begin;vertex<end;vertex++){let next=select(vertex+1u,begin,vertex+1u==end);let first=position(vertex);let second=position(next);if((first.y>point.y)!=(second.y>point.y)){let intersection=first.x+(point.y-first.y)*(second.x-first.x)/(second.y-first.y);if(point.x<intersection){inside=!inside;}}}}
 if(inside){return true;}}
 return false;
}`;
    const node = (
      name: string,
      invocationCount: number,
      bindings: WGSLKernelBinding[],
      body: string,
      declarations = ''
    ) =>
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-${name}`,
        operation: 'GPUFuzzyContiguity',
        variant: name,
        invocationCount,
        bindings,
        body,
        declarations
      });
    validateGraphViewsBelongToGraph(id, graph, [
      polygons.positions,
      polygons.featureOffsets,
      polygons.polygonOffsets,
      polygons.ringOffsets,
      ...getGPUSpatialWeightsViews(props.output),
      ...getGPUSpatialWeightsStatusViews(props.output),
      props.output.featureStatus,
      props.output.added
    ]);
    return [
      node(
        'validate',
        rows,
        [...geometry, binding('status', props.output.featureStatus, 'read_write')],
        `let first=features[featuresOffset+index];let last=features[featuresOffset+index+1u];var code=0u;
if(first>last||last>=${polygons.polygonOffsets.length}u){code=2u;}else if(first==last){code=1u;}else{
for(var polygon=first;polygon<last;polygon++){let firstRing=polygons[polygonsOffset+polygon];let lastRing=polygons[polygonsOffset+polygon+1u];if(firstRing>=lastRing||lastRing>=${polygons.ringOffsets.length}u){code=2u;break;}
for(var ring=firstRing;ring<lastRing;ring++){let begin=rings[ringsOffset+ring];let end=rings[ringsOffset+ring+1u];if(begin>end||end>${polygons.positions.length}u||end-begin<3u){code=2u;break;}for(var vertex=begin;vertex<end;vertex++){let value=position(vertex);if(!isFiniteBits(value.x)||!isFiniteBits(value.y)){code=2u;break;}}}}
}status[statusOffset+index]=code;`,
        common
      ),
      node(
        'distance',
        rows * rows,
        [
          ...geometry,
          binding('status', props.output.featureStatus),
          binding('distances', distances, 'read_write', 'f32')
        ],
        `let first=index/${rows}u;let second=index%${rows}u;var best=getNaN();
if(first!=second&&status[statusOffset+first]==0u&&status[statusOffset+second]==0u){best=3.402823e38;
for(var firstPolygon=features[featuresOffset+first];firstPolygon<features[featuresOffset+first+1u];firstPolygon++){
for(var firstRing=polygons[polygonsOffset+firstPolygon];firstRing<polygons[polygonsOffset+firstPolygon+1u];firstRing++){let begin=rings[ringsOffset+firstRing];let end=rings[ringsOffset+firstRing+1u];
for(var vertex=begin;vertex<end;vertex++){let next=select(vertex+1u,begin,vertex+1u==end);let point=position(vertex);if(contains(second,point)){best=0.0;}
for(var secondPolygon=features[featuresOffset+second];secondPolygon<features[featuresOffset+second+1u];secondPolygon++){
for(var secondRing=polygons[polygonsOffset+secondPolygon];secondRing<polygons[polygonsOffset+secondPolygon+1u];secondRing++){let otherBegin=rings[ringsOffset+secondRing];let otherEnd=rings[ringsOffset+secondRing+1u];
for(var other=otherBegin;other<otherEnd;other++){let otherNext=select(other+1u,otherBegin,other+1u==otherEnd);let otherPoint=position(other);if(contains(first,otherPoint)){best=0.0;}best=min(best,segmentDistance(point,position(next),otherPoint,position(otherNext)));}}}
}}}
}distances[distancesOffset+index]=best;`,
        common
      ),
      node(
        'offsets',
        1,
        [
          binding('distances', distances, 'read', 'f32'),
          binding('offsets', props.output.offsets, 'read_write'),
          binding('overflow', props.output.status.overflow, 'read_write'),
          ...(props.output.status.requiredCount
            ? [binding('required', props.output.status.requiredCount, 'read_write')]
            : [])
        ],
        `var total=0u;for(var row=0u;row<${rows}u;row++){offsets[offsetsOffset+row]=min(total,${capacity}u);for(var other=0u;other<${rows}u;other++){if(distances[distancesOffset+row*${rows}u+other]<=${getWGSLFloatLiteral(props.tolerance)}){total++;}}}offsets[offsetsOffset+${rows}u]=min(total,${capacity}u);overflow[overflowOffset]=select(0u,1u,total>${capacity}u);${props.output.status.requiredCount ? 'required[requiredOffset]=total;' : ''}`
      ),
      node(
        'write',
        rows,
        [
          binding('distances', distances, 'read', 'f32'),
          binding('offsets', props.output.offsets),
          binding('neighbors', props.output.neighbors, 'read_write'),
          binding('weights', props.output.weights, 'read_write', 'f32'),
          binding('added', props.output.added, 'read_write'),
          ...(props.output.distances
            ? [binding('outputDistances', props.output.distances, 'read_write', 'f32')]
            : [])
        ],
        `var slot=offsets[offsetsOffset+index];let end=offsets[offsetsOffset+index+1u];for(var other=0u;other<${rows}u&&slot<end;other++){let distance=distances[distancesOffset+index*${rows}u+other];if(distance<=${getWGSLFloatLiteral(props.tolerance)}){neighbors[neighborsOffset+slot]=other;weights[weightsOffset+slot]=1.0;added[addedOffset+slot]=select(0u,1u,distance>0.0);${props.output.distances ? 'outputDistances[outputDistancesOffset+slot]=distance;' : ''}slot++;}}`
      )
    ];
  }
}
