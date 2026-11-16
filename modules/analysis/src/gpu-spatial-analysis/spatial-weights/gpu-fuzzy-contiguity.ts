// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  doGraphDataViewsOverlap,
  getViewBindingRange,
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
  validateMaximumCount,
  validateRequirement,
  validateGPUStorageBindingByteLength,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {GET_NAN_WGSL, IS_FINITE_BITS_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getBinarySearchWGSL} from '../../analysis-core/kernels/binary-search-wgsl';
import {getFuzzyContiguityCSRNodes} from './gpu-fuzzy-contiguity-csr';
import {getFuzzyContiguityPairSortNodes} from './gpu-fuzzy-contiguity-pair-sort';
import {
  type GPUSpatialWeights,
  type GPUSpatialWeightsOutput,
  describeGPUSpatialWeights,
  validateGPUSpatialWeightsOutput,
  getGPUSpatialWeightsViews,
  getGPUSpatialWeightsStatusViews
} from './spatial-weights';

import {validateSpatialStatisticsBindings} from './spatial-statistics-bindings';

/** Feature offsets need a terminal entry; row IDs and offset lengths are uint32. */
const FUZZY_CONTIGUITY_MAXIMUM_FEATURES = 0xfffffffe;
/** Positions are accessed as two scalar words per vertex with uint32 indices. */
const FUZZY_CONTIGUITY_MAXIMUM_VERTICES = 0x7fffffff;
/** Pair slots and CSR offsets are uint32; larger required pair counts saturate. */
const FUZZY_CONTIGUITY_MAXIMUM_PAIRS = 0xffffffff;
/** Packed bounds and geometry offsets are addressed with uint32 scalar indices. */
const FUZZY_CONTIGUITY_MAXIMUM_WORDS = 0xffffffff;
/** Published GPUBVH refits use 256 threads and a one-dimensional dispatch per tree level. */
const FUZZY_CONTIGUITY_BVH_REFIT_WORKGROUP_SIZE = 256;
/** Published radix partitions have at most 16 histogram buckets per 256-item workgroup. */
const FUZZY_CONTIGUITY_RADIX_BUCKET_COUNT = 16;
/** Published radix histogram scratch uses one bucket group per 256 pair slots. */
const FUZZY_CONTIGUITY_RADIX_WORKGROUP_SIZE = 256;

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
 * Inputs are single packed views. Feature and pair IDs use separate uint32 words; narrow IDs
 * share a radix pass, and wider IDs use stable neighbor-then-row passes. Scratch and packed
 * inputs must fit the device storage binding size, and the BVH refit must fit one dispatch
 * dimension. Required pair counts and every CSR prefix saturate at 0xffffffff; overflow is set
 * and the bounded prefix stays valid even when more than uint32 pairs are required. Candidate
 * polygon complexity still determines the cost of each exact distance test.
 */
export class GPUFuzzyContiguity implements GPUCommandNodeProducer {
  readonly id: string;
  readonly outputWeights: GPUSpatialWeights;
  constructor(readonly props: GPUFuzzyContiguityProps) {
    this.id = props.id ?? 'fuzzy-contiguity';
    const rows = props.polygons.featureOffsets.length - 1;
    validateCapacity(this.id, 'features', rows, 1, FUZZY_CONTIGUITY_MAXIMUM_FEATURES);
    validateCapacity(
      this.id,
      'vertices',
      props.polygons.positions.length,
      1,
      FUZZY_CONTIGUITY_MAXIMUM_VERTICES
    );
    validateCapacity(
      this.id,
      'output.neighbors.length',
      props.output.neighbors.length,
      1,
      FUZZY_CONTIGUITY_MAXIMUM_PAIRS
    );
    const nodeCount = 2 * getFuzzyContiguityLeafCapacity(rows) - 1;
    validateMaximumCount(
      this.id,
      'features search bounds words',
      (nodeCount + rows) * 4,
      FUZZY_CONTIGUITY_MAXIMUM_WORDS
    );
    for (const [name, view] of [
      ['polygons.polygonOffsets.length', props.polygons.polygonOffsets],
      ['polygons.ringOffsets.length', props.polygons.ringOffsets]
    ] as const) {
      validateCapacity(this.id, name, view.length, 1, FUZZY_CONTIGUITY_MAXIMUM_WORDS);
    }
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
    const outputs = [
      ['output.offsets', props.output.offsets],
      ['output.neighbors', props.output.neighbors],
      ['output.weights', props.output.weights],
      ['output.distances', props.output.distances],
      ['output.featureStatus', props.output.featureStatus],
      ['output.added', props.output.added],
      ['output.status.overflow', props.output.status.overflow],
      ['output.status.requiredCount', props.output.status.requiredCount],
      ['output.status.candidateOverflow', props.output.status.candidateOverflow]
    ] as const;
    for (const [outputIndex, [name, view]] of outputs.entries()) {
      if (!view) continue;
      for (const [otherName, other] of outputs.slice(0, outputIndex)) {
        if (other) {
          validateRequirement(
            this.id,
            !doGraphDataViewsOverlap(view, other),
            `${name} must not overlap writable bytes of ${otherName}`
          );
        }
      }
    }
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
    validateSpatialStatisticsBindings(graph, id, props);
    const rows = props.polygons.featureOffsets.length - 1;
    const capacity = props.output.neighbors.length;
    const polygons = props.polygons;
    const leafCapacity = getFuzzyContiguityLeafCapacity(rows);
    const nodeCount = 2 * leafCapacity - 1;
    // Largest power-of-two leaf capacity whose widest refit level fits one dispatch dimension.
    const FUZZY_CONTIGUITY_MAXIMUM_BVH_LEAF_CAPACITY =
      2 **
      (Math.floor(
        Math.log2(
          graph.device.limits.maxComputeWorkgroupsPerDimension *
            FUZZY_CONTIGUITY_BVH_REFIT_WORKGROUP_SIZE
        )
      ) +
        1);
    validateMaximumCount(
      id,
      'features BVH leaf capacity',
      leafCapacity,
      FUZZY_CONTIGUITY_MAXIMUM_BVH_LEAF_CAPACITY
    );
    // Check every row/pair-proportional allocation before declaring any transient or node.
    for (const [name, byteLength, hint] of [
      ['pair scratch', capacity * 4, 'reduce output.neighbors.length'],
      [
        'pair radix histogram scratch',
        FUZZY_CONTIGUITY_RADIX_BUCKET_COUNT *
          Math.ceil(capacity / FUZZY_CONTIGUITY_RADIX_WORKGROUP_SIZE) *
          4,
        'reduce output.neighbors.length'
      ],
      ['feature bounds scratch', rows * 8, 'use fewer features'],
      ['BVH node bounds scratch', nodeCount * 8, 'use fewer features'],
      ['search bounds scratch', (nodeCount + rows) * 16, 'use fewer features'],
      ['CSR scratch', rows * 4, 'use fewer features']
    ] as const) {
      validateGPUStorageBindingByteLength(graph.device, id, name, byteLength, hint);
    }
    for (const [name, view] of [
      ['polygons.positions', polygons.positions],
      ['polygons.featureOffsets', polygons.featureOffsets],
      ['polygons.polygonOffsets', polygons.polygonOffsets],
      ['polygons.ringOffsets', polygons.ringOffsets],
      ...getGPUSpatialWeightsViews(props.output).map(view => ['output weights', view] as const),
      ...getGPUSpatialWeightsStatusViews(props.output).map(
        view => ['output status', view] as const
      ),
      ['output.featureStatus', props.output.featureStatus],
      ['output.added', props.output.added]
    ] as const) {
      if (view) {
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          name,
          getViewBindingRange(view).size,
          `use a smaller packed ${name} view`
        );
      }
    }
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
    // Separate words preserve every row/neighbor bit. All-ones unused keys sort last, including
    // when the shared sorter packs narrow columns into the former single-word key.
    const pairRows = createTransientView(graph, `${id}-pair-rows`, 'uint32', capacity);
    const pairNeighbors = createTransientView(graph, `${id}-pair-neighbors`, 'uint32', capacity);
    const pairDistanceBits = createTransientView(
      graph,
      `${id}-pair-distance-bits`,
      'uint32',
      capacity
    );
    const pairSort = getFuzzyContiguityPairSortNodes(graph, {
      id: `${id}-pair-sort`,
      rows,
      pairRows,
      pairNeighbors
    });
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
fn getFeatureCount()->u32{return arrayLength(&features)-featuresOffset-1u;}
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
fn getLeafCapacity()->u32{return arrayLength(&leafIds)-leafIdsOffset;}
fn getNodeCount()->u32{return getLeafCapacity()*2u-1u;}
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
    const visit = (onItem: string) => `let rowCount=getFeatureCount();
 let internalNodeCount=getLeafCapacity()-1u;
 let ownBox=searchBox(getNodeCount()+index);
 let boxMinimum=ownBox.xy;
 let boxMaximum=ownBox.zw;
 if(all(boxMinimum<=boxMaximum)){
  let slack=(max(max(abs(boxMinimum.x),abs(boxMinimum.y)),max(abs(boxMaximum.x),abs(boxMaximum.y)))+TOLERANCE)*1.0e-5;
  let queryMinimum=boxMinimum-vec2f(TOLERANCE+slack);let queryMaximum=boxMaximum+vec2f(TOLERANCE+slack);
  var node=0u;
  loop{
   if(nodeOverlaps(node,queryMinimum,queryMaximum)){
    if(node<internalNodeCount){node=node*2u+1u;continue;}
    let slot=node-internalNodeCount;
    var other=rowCount;if(slot<rowCount){other=leafIds[leafIdsOffset+slot];}
    if(other<rowCount&&other!=index){let distance=getFeatureDistance(index,other);if(distance<=TOLERANCE){${onItem}}}
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
if(first>last||last>=arrayLength(&polygons)-polygonsOffset){code=2u;}else if(first==last){code=1u;}else{
for(var polygon=first;polygon<last;polygon++){let firstRing=polygons[polygonsOffset+polygon];let lastRing=polygons[polygonsOffset+polygon+1u];if(firstRing>=lastRing||lastRing>=arrayLength(&rings)-ringsOffset){code=2u;break;}
for(var ring=firstRing;ring<lastRing;ring++){let begin=rings[ringsOffset+ring];let end=rings[ringsOffset+ring+1u];if(begin>end||end>(arrayLength(&positions)-positionsOffset)/2u||end-begin<3u){code=2u;break;}for(var vertex=begin;vertex<end;vertex++){let value=position(vertex);if(!isFiniteBits(value.x)||!isFiniteBits(value.y)){code=2u;break;}}}}
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
        `let nodeCount=(arrayLength(&nodeMinima)-nodeMinimaOffset)/2u;
var box=vec4f(0.0);
if(index<nodeCount){box=vec4f(nodeMinima[nodeMinimaOffset+index*2u],nodeMinima[nodeMinimaOffset+index*2u+1u],nodeMaxima[nodeMaximaOffset+index*2u],nodeMaxima[nodeMaximaOffset+index*2u+1u]);}
else{let row=index-nodeCount;box=vec4f(featureMinima[featureMinimaOffset+row*2u],featureMinima[featureMinimaOffset+row*2u+1u],featureMaxima[featureMaximaOffset+row*2u],featureMaxima[featureMaximaOffset+row*2u+1u]);}
let base=index*4u;searchBounds[searchBoundsOffset+base]=box.x;searchBounds[searchBoundsOffset+base+1u]=box.y;searchBounds[searchBoundsOffset+base+2u]=box.z;searchBounds[searchBoundsOffset+base+3u]=box.w;`
      ),
      node(
        'clear-pair-keys',
        capacity,
        [binding('pairNeighbors', pairNeighbors, 'read_write')],
        'pairNeighbors[pairNeighborsOffset+index]=0xffffffffu;'
      ),
      ...getFuzzyContiguityCSRNodes<Parameters>(graph, {
        id: `${id}-edges`,
        rows,
        offsets: props.output.offsets,
        slots: pairNeighbors,
        overflow: props.output.status.overflow,
        requiredCount: props.output.status.requiredCount,
        bindings: [
          ...geometry,
          binding('searchBounds', searchBounds, 'read', 'f32'),
          binding('leafIds', bvh.leafIds)
        ],
        declarations: `${common}\n${featureDistance}`,
        visit,
        attributes: [
          {
            name: 'pair-neighbors',
            output: pairNeighbors,
            type: 'u32',
            value: 'other'
          },
          {
            name: 'pair-distance-bits',
            output: pairDistanceBits,
            type: 'u32',
            value: 'bitcast<u32>(distance)'
          }
        ]
      }),
      // Recover each emitted slot's row from the clamped CSR offsets, avoiding another full
      // geometry traversal just to write the primary sort key. Upper bound skips empty rows.
      node(
        'pair-rows',
        capacity,
        [binding('offsets', props.output.offsets), binding('pairRows', pairRows, 'read_write')],
        `var row=0xffffffffu;
let offsetCount=arrayLength(&offsets)-offsetsOffset;
if(index<offsets[offsetsOffset+offsetCount-1u]){row=upperBoundOffsets(0u,offsetCount,index)-1u;}
pairRows[pairRowsOffset+index]=row;`,
        getBinarySearchWGSL('offsets')
      ),
      ...pairSort.nodes,
      node(
        'finish',
        capacity,
        [
          binding('offsets', props.output.offsets),
          binding('sortedPairIndices', pairSort.sortedIndices),
          binding('pairNeighbors', pairNeighbors),
          binding('pairDistanceBits', pairDistanceBits),
          binding('neighbors', props.output.neighbors, 'read_write'),
          binding('weights', props.output.weights, 'read_write', 'f32'),
          binding('added', props.output.added, 'read_write'),
          ...(props.output.distances
            ? [binding('distances', props.output.distances, 'read_write', 'f32')]
            : [])
        ],
        `if(index<offsets[arrayLength(&offsets)-1u]){
let pair=sortedPairIndices[sortedPairIndicesOffset+index];
let distance=bitcast<f32>(pairDistanceBits[pairDistanceBitsOffset+pair]);
neighbors[neighborsOffset+index]=pairNeighbors[pairNeighborsOffset+pair];
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
