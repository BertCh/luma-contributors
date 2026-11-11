// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {getSegmentBVHNodes} from '../segment-intersection/segment-bvh';
import type {GPUPolygonGeometryPort} from '../../analysis-core/contracts/ports';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createCSRStages} from './spatial-weights-algebra-wgsl';
import {
  type GPUSpatialWeights,
  type GPUSpatialWeightsOutput,
  describeGPUSpatialWeights,
  validateGPUSpatialWeightsOutput,
  getGPUSpatialWeightsViews,
  getGPUSpatialWeightsStatusViews
} from './spatial-weights';

/**
 * Largest feature count: an emitted pair is sorted by one uint32 key of 16 row bits and 16
 * neighbor bits, and per-row neighbor counts and the required total stay within uint32.
 */
const FUZZY_CONTIGUITY_MAXIMUM_FEATURES = 65536;

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
 * Candidate pairs come from a BVH over feature bounds expanded by the tolerance; only candidates
 * run the exact distance test, so no feature-pair table is stored. The BVH leaves are sorted along
 * a Morton curve of the bound centers, so each probe visits nodes near its own feature whatever
 * the source order; emitted pairs are then radix-sorted by (row, neighbor), so every row lists
 * its neighbors in ascending order. When the slots overflow, the row in which the capacity runs
 * out keeps an unspecified subset of its neighbors. No geographic coordinates, buffered polygon
 * output or general GEOS topological robustness is claimed.
 */
export class GPUFuzzyContiguity implements GPUCommandNodeProducer {
  readonly id: string;
  readonly outputWeights: GPUSpatialWeights;
  constructor(readonly props: GPUFuzzyContiguityProps) {
    this.id = props.id ?? 'fuzzy-contiguity';
    const rows = props.polygons.featureOffsets.length - 1;
    validateCapacity(this.id, 'features', rows, 1, FUZZY_CONTIGUITY_MAXIMUM_FEATURES);
    validateCapacity(this.id, 'vertices', props.polygons.positions.length, 1, 65536);
    if (!Number.isFinite(props.tolerance) || props.tolerance < 0 || props.tolerance > 1e18)
      throw new Error(`${this.id}: invalid tolerance`);
    if (
      validateGPUSpatialWeightsOutput(this.id, props.output) !== rows ||
      props.output.featureStatus.length !== rows ||
      props.output.added.length !== props.output.neighbors.length
    )
      throw new Error(`${this.id}: shape mismatch`);
    validateSinglePackedView(props.polygons.positions, ['float32x2'], this.id);
    for (const view of [
      props.polygons.featureOffsets,
      props.polygons.polygonOffsets,
      props.polygons.ringOffsets,
      props.output.featureStatus,
      props.output.added
    ])
      validateSinglePackedView(view, ['uint32'], this.id);
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
    const leafCapacity = getFuzzyContiguityLeafCapacity(rows);
    const nodeCount = 2 * leafCapacity - 1;
    const featureMinima = createTransientView(graph, `${id}-feature-minima`, 'float32x2', rows);
    const featureMaxima = createTransientView(graph, `${id}-feature-maxima`, 'float32x2', rows);
    // Morton-sorted leaves: `leafIds` maps each leaf slot back to its feature row.
    const {bvh, nodes: bvhNodes} = getSegmentBVHNodes<Parameters>(graph, {
      id,
      operation: 'GPUFuzzyContiguity',
      minima: featureMinima,
      maxima: featureMaxima,
      leafCapacity,
      spatialSort: true
    });
    // Node bounds, then each row's own bounds, as (minimum.x, minimum.y, maximum.x, maximum.y):
    // one binding, so the traversal passes stay within eight storage buffers.
    const searchBounds = createTransientView(
      graph,
      `${id}-search-bounds`,
      'float32',
      (nodeCount + rows) * 4
    );
    // Pairs are emitted in traversal order with key (row << shift) | neighbor, then radix-sorted
    // into CSR order. Unused slots keep the largest key, the never-emitted self pair of the last
    // possible row, so they sort last.
    const pairKeyShift = Math.max(1, Math.ceil(Math.log2(rows)));
    const pairKeyBits = 2 * pairKeyShift;
    const pairKeys = createTransientView(graph, `${id}-pair-keys`, 'uint32', capacity);
    const pairDistanceBits = createTransientView(
      graph,
      `${id}-pair-distance-bits`,
      'uint32',
      capacity
    );
    const sortedPairKeys = createTransientView(graph, `${id}-sorted-pair-keys`, 'uint32', capacity);
    const sortedPairDistanceBits = createTransientView(
      graph,
      `${id}-sorted-pair-distance-bits`,
      'uint32',
      capacity
    );
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
    const tolerance = getWGSLFloatLiteral(props.tolerance);
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
    // Filled-polygon separation of two valid features: 0 when a vertex of either lies inside the
    // other, else the smallest distance between their ring segments.
    const featureDistance = `const TOLERANCE: f32 = ${tolerance};
const INTERNAL_NODE_COUNT: u32 = ${leafCapacity - 1}u;
const NODE_COUNT: u32 = ${nodeCount}u;
fn searchBox(entry:u32)->vec4f{let base=entry*4u;return vec4f(searchBounds[searchBoundsOffset+base],searchBounds[searchBoundsOffset+base+1u],searchBounds[searchBoundsOffset+base+2u],searchBounds[searchBoundsOffset+base+3u]);}
fn nodeOverlaps(node:u32,queryMinimum:vec2f,queryMaximum:vec2f)->bool{let box=searchBox(node);return all(box.xy<=queryMaximum)&&all(queryMinimum<=box.zw);}
fn getFeatureDistance(first:u32,second:u32)->f32{
 var best=3.402823e38;
 for(var firstPolygon=features[featuresOffset+first];firstPolygon<features[featuresOffset+first+1u];firstPolygon++){
 for(var firstRing=polygons[polygonsOffset+firstPolygon];firstRing<polygons[polygonsOffset+firstPolygon+1u];firstRing++){let begin=rings[ringsOffset+firstRing];let end=rings[ringsOffset+firstRing+1u];
 for(var vertex=begin;vertex<end;vertex++){let point=position(vertex);if(contains(second,point)){return 0.0;}let nextPoint=position(select(vertex+1u,begin,vertex+1u==end));
 for(var secondPolygon=features[featuresOffset+second];secondPolygon<features[featuresOffset+second+1u];secondPolygon++){
 for(var secondRing=polygons[polygonsOffset+secondPolygon];secondRing<polygons[polygonsOffset+secondPolygon+1u];secondRing++){let otherBegin=rings[ringsOffset+secondRing];let otherEnd=rings[ringsOffset+secondRing+1u];
 for(var other=otherBegin;other<otherEnd;other++){best=min(best,segmentDistance(point,nextPoint,position(other),position(select(other+1u,otherBegin,other+1u==otherEnd))));}}}}}}
 for(var secondPolygon=features[featuresOffset+second];secondPolygon<features[featuresOffset+second+1u];secondPolygon++){
 for(var secondRing=polygons[polygonsOffset+secondPolygon];secondRing<polygons[polygonsOffset+secondPolygon+1u];secondRing++){
 for(var other=rings[ringsOffset+secondRing];other<rings[ringsOffset+secondRing+1u];other++){if(contains(first,position(other))){return 0.0;}}}}
 return best;
}`;
    // Visits the features whose bounds lie within the tolerance of this row's bounds, in Morton
    // leaf order, and keeps those within the tolerance. Invalid features have empty bounds, so they
    // neither probe nor match; padded leaf slots are skipped. The query box is widened by
    // a slack of 1e-5 (about 80 float32 steps) relative to the largest magnitude involved, which is
    // this row's largest coordinate plus the tolerance: any feature the distance test can accept
    // has coordinates within that magnitude, so its float32 distance error, and the rounding of
    // `TOLERANCE + slack` and of the widened box, each stay a few steps of it. Scaling the slack
    // with this row's coordinates alone let a large tolerance round the box inward, so a pair at the
    // tolerance was found from one side only and the CSR lost its symmetry.
    const visit = (onItem: string) => `let ownBox=searchBox(NODE_COUNT+index);
 let boxMinimum=ownBox.xy;
 let boxMaximum=ownBox.zw;
 if(all(boxMinimum<=boxMaximum)){
  let slack=(max(max(abs(boxMinimum.x),abs(boxMinimum.y)),max(abs(boxMaximum.x),abs(boxMaximum.y)))+TOLERANCE)*1.0e-5;
  let queryMinimum=boxMinimum-vec2f(TOLERANCE+slack);let queryMaximum=boxMaximum+vec2f(TOLERANCE+slack);
  var node=0u;
  loop{
   if(nodeOverlaps(node,queryMinimum,queryMaximum)){
    if(node<INTERNAL_NODE_COUNT){node=node*2u+1u;continue;}
    let slot=node-INTERNAL_NODE_COUNT;
    var other=ROWS;if(slot<ROWS){other=leafIds[leafIdsOffset+slot];}
    if(other<ROWS&&other!=index){let distance=getFeatureDistance(index,other);if(distance<=TOLERANCE){${onItem}}}
   }
   loop{if(node==0u||(node&1u)==1u){break;}node=(node-1u)/2u;}
   if(node==0u){break;}
   node=node+1u;
  }
 }`;
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
    return [
      node(
        'validate',
        rows,
        [
          ...geometry,
          binding('status', props.output.featureStatus, 'read_write'),
          binding('featureMinima', featureMinima, 'read_write', 'f32'),
          binding('featureMaxima', featureMaxima, 'read_write', 'f32')
        ],
        `let first=features[featuresOffset+index];let last=features[featuresOffset+index+1u];var code=0u;
if(first>last||last>=${polygons.polygonOffsets.length}u){code=2u;}else if(first==last){code=1u;}else{
for(var polygon=first;polygon<last;polygon++){let firstRing=polygons[polygonsOffset+polygon];let lastRing=polygons[polygonsOffset+polygon+1u];if(firstRing>=lastRing||lastRing>=${polygons.ringOffsets.length}u){code=2u;break;}
for(var ring=firstRing;ring<lastRing;ring++){let begin=rings[ringsOffset+ring];let end=rings[ringsOffset+ring+1u];if(begin>end||end>${polygons.positions.length}u||end-begin<3u){code=2u;break;}for(var vertex=begin;vertex<end;vertex++){let value=position(vertex);if(!isFiniteBits(value.x)||!isFiniteBits(value.y)){code=2u;break;}}}}
}status[statusOffset+index]=code;
var minimum=vec2<f32>(3.402823e38);var maximum=vec2<f32>(-3.402823e38);
if(code==0u){for(var polygon=first;polygon<last;polygon++){for(var ring=polygons[polygonsOffset+polygon];ring<polygons[polygonsOffset+polygon+1u];ring++){for(var vertex=rings[ringsOffset+ring];vertex<rings[ringsOffset+ring+1u];vertex++){let value=position(vertex);minimum=min(minimum,value);maximum=max(maximum,value);}}}}
featureMinima[featureMinimaOffset+index*2u]=minimum.x;featureMinima[featureMinimaOffset+index*2u+1u]=minimum.y;
featureMaxima[featureMaximaOffset+index*2u]=maximum.x;featureMaxima[featureMaximaOffset+index*2u+1u]=maximum.y;`,
        common
      ),
      ...bvhNodes,
      node(
        'search-bounds',
        nodeCount + rows,
        [
          binding('nodeMinima', bvh.nodeMinima, 'read', 'f32'),
          binding('nodeMaxima', bvh.nodeMaxima, 'read', 'f32'),
          binding('featureMinima', featureMinima, 'read', 'f32'),
          binding('featureMaxima', featureMaxima, 'read', 'f32'),
          binding('searchBounds', searchBounds, 'read_write', 'f32')
        ],
        `var box=vec4f(0.0);
if(index<${nodeCount}u){box=vec4f(nodeMinima[nodeMinimaOffset+index*2u],nodeMinima[nodeMinimaOffset+index*2u+1u],nodeMaxima[nodeMaximaOffset+index*2u],nodeMaxima[nodeMaximaOffset+index*2u+1u]);}
else{let row=index-${nodeCount}u;box=vec4f(featureMinima[featureMinimaOffset+row*2u],featureMinima[featureMinimaOffset+row*2u+1u],featureMaxima[featureMaximaOffset+row*2u],featureMaxima[featureMaximaOffset+row*2u+1u]);}
let base=index*4u;searchBounds[searchBoundsOffset+base]=box.x;searchBounds[searchBoundsOffset+base+1u]=box.y;searchBounds[searchBoundsOffset+base+2u]=box.z;searchBounds[searchBoundsOffset+base+3u]=box.w;`
      ),
      node(
        'clear-pair-keys',
        capacity,
        [binding('pairKeys', pairKeys, 'read_write')],
        'pairKeys[pairKeysOffset+index]=0xffffffffu;'
      ),
      ...createCSRStages<Parameters>(graph, {
        id: `${id}-edges`,
        operation: 'GPUFuzzyContiguity',
        rows,
        targetOffsets: props.output.offsets,
        capacity,
        overflow: props.output.status.overflow,
        initializesOverflow: true,
        totalNeighbors: props.output.status.requiredCount,
        bindings: [
          ...geometry,
          binding('searchBounds', searchBounds, 'read', 'f32'),
          binding('leafIds', bvh.leafIds)
        ],
        declarations: `${common}\n${featureDistance}`,
        visit,
        attributes: [
          {
            name: 'pair-keys',
            output: pairKeys,
            type: 'u32',
            value: `(index<<${pairKeyShift}u)|other`
          },
          {
            name: 'pair-distance-bits',
            output: pairDistanceBits,
            type: 'u32',
            value: 'bitcast<u32>(distance)'
          }
        ]
      }),
      ...new GPUSort({
        id: `${id}-pair-sort`,
        keys: pairKeys,
        values: pairDistanceBits,
        outputKeys: sortedPairKeys,
        outputValues: sortedPairDistanceBits,
        algorithm: 'radix',
        keyBits: pairKeyBits
      }).getCommandNodes(graph),
      node(
        'finish',
        capacity,
        [
          binding('offsets', props.output.offsets),
          binding('sortedPairKeys', sortedPairKeys),
          binding('sortedPairDistanceBits', sortedPairDistanceBits),
          binding('neighbors', props.output.neighbors, 'read_write'),
          binding('weights', props.output.weights, 'read_write', 'f32'),
          binding('added', props.output.added, 'read_write'),
          ...(props.output.distances
            ? [binding('distances', props.output.distances, 'read_write', 'f32')]
            : [])
        ],
        `if(index<offsets[offsetsOffset+${rows}u]){
let distance=bitcast<f32>(sortedPairDistanceBits[sortedPairDistanceBitsOffset+index]);
neighbors[neighborsOffset+index]=sortedPairKeys[sortedPairKeysOffset+index]&${2 ** pairKeyShift - 1}u;
weights[weightsOffset+index]=1.0;added[addedOffset+index]=select(0u,1u,distance>0.0);
${props.output.distances ? 'distances[distancesOffset+index]=distance;' : ''}}`
      )
    ];
  }
}

/** Power-of-two BVH leaf capacity for `rows` features. */
function getFuzzyContiguityLeafCapacity(rows: number): number {
  return 2 ** Math.ceil(Math.log2(Math.max(rows, 1)));
}
