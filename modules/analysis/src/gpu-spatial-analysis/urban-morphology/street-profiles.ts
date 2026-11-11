// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getWGSLFloatLiteral} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  URBAN_MAXIMUM_OBSTACLE_SEGMENT_COUNT,
  URBAN_RAY_STACK_CAPACITY,
  addUrbanObstacleBVH,
  getUrbanRayBVHBindings,
  getUrbanRayDeclarations,
  resolveUrbanObstacleIndex,
  type UrbanObstacleIndex,
  type UrbanRayBVH
} from './urban-ray-utils';

/**
 * Most obstacle segments when `streetSegment` asks for exact facade lengths: every segment owns a
 * thread that sorts about 3 events per segment, and the event table holds 3 * S + 8 floats per
 * segment, so the table grows with S squared (about 12 MiB at this limit).
 */
export const STREET_PROFILES_MAXIMUM_FACADE_SEGMENT_COUNT = 1024;

/** Events a facade interval keeps besides the three it adds per obstacle segment. */
const FACADE_EVENT_FIXED_COUNT = 8;

/**
 * Relative part of the two facade tolerances, about 100 float32 rounding steps. A wall whose
 * extent along the street is within `tolerance + slack * wall length` is edge-on and shows no
 * length. The walls whose sightline distance is within
 * `tolerance + slack * max(street length, maximumDistance)` of the nearest hit are the same stretch
 * of wall, and the lowest row among them owns it.
 */
const FACADE_RELATIVE_SLACK = 1e-5;

/**
 * WGSL of the facade visibility predicate. The facade point is at `facadeDistance` along the
 * sightline by construction, so its own row is never tested. `nearest` is the smallest hit
 * distance on the sightline, the wall's own included. The wall owns the stretch when it is within
 * `slack` of `nearest` and no lower row is. Ownership is decided against the one nearest hit, so
 * it is transitive: the set of walls within `slack` of `nearest` is never empty and has one lowest
 * row, whatever their number. No two separately rounded distances are compared for equality.
 *
 * Both obstacle indices give the same result bit for bit. The decision depends only on the hits
 * within `slack` of the nearest one, minima do not depend on the visiting order, and the BVH never
 * prunes such a segment: a segment that blocks at distance `d` (0 when it is within `tolerance` of
 * the sightline origin, even behind it) has a point within `2 * tolerance` of the sightline
 * between 0 and `d`, and its box is padded by `2 * tolerance` plus the float32 rounding of the
 * coordinates (`addUrbanObstacleBVH`), so the sightline enters the box no later than `d`.
 */
function getFacadeVisibilityDeclarations(
  segmentCount: number,
  maximumDistance: number,
  tolerance: number,
  obstacleBVH?: UrbanRayBVH
): string {
  const hitTest = `let blocked=getSegmentCandidate(origin,direction,${getWGSLFloatLiteral(maximumDistance)},other);
  if(blocked.y>0.0){nearest=min(nearest,blocked.x);if(other<row){nearestLowerRow=min(nearestLowerRow,blocked.x);}}`;
  const decision = `let reach=nearest+slack;
 return facadeDistance<=reach&&nearestLowerRow>reach;`;
  if (!obstacleBVH) {
    return `
fn isFacadeIntervalVisible(origin:vec2<f32>,direction:vec2<f32>,facadeDistance:f32,slack:f32,row:u32)->bool{
 var nearest=facadeDistance;var nearestLowerRow=3.0e38;
 for(var other=0u;other<${segmentCount}u;other++){
  if(other==row){continue;}
  ${hitTest}
 }
 ${decision}
}`;
  }
  return `
fn isFacadeIntervalVisible(origin:vec2<f32>,direction:vec2<f32>,facadeDistance:f32,slack:f32,row:u32)->bool{
 var nearest=facadeDistance;var nearestLowerRow=3.0e38;
 var stack:array<u32,${URBAN_RAY_STACK_CAPACITY}>;var depth=1u;stack[0u]=0u;
 loop{
  if(depth==0u){break;}
  depth--;let node=stack[depth];
  // Only a hit within slack of the nearest one can change the result. Rounding slack on the
  // pruning distance keeps every such segment in the search.
  let pruning=nearest+slack;
  let limit=pruning+pruning*1.0e-5+${getWGSLFloatLiteral(4 * tolerance)};
  if(getBoxEntry(node,origin,direction,limit)>=3.0e38){continue;}
  if(node<${obstacleBVH.internalNodeCount}u){stack[depth]=node*2u+2u;stack[depth+1u]=node*2u+1u;depth+=2u;continue;}
  let other=leafIds[leafIdsOffset+node-${obstacleBVH.internalNodeCount}u];
  if(other>=${segmentCount}u||other==row){continue;}
  ${hitTest}
 }
 ${decision}
}`;
}

export type GPUStreetProfilesOutput = {
  /** `float32x4`, one row per sample: left and right distance, left and right height. */
  profiles: GraphDataView<'float32x4'>;
  /** `uint32x2`, one row per sample: left and right building row, or 0xffffffff. */
  buildings: GraphDataView<'uint32x2'>;
  /** `uint32`, one row per sample: bit 0 left hit, bit 1 right hit, bit 2 touching, bit 3 invalid. */
  validity: GraphDataView<'uint32'>;
  /** `float32x4`, one row per sample: width, mean height, height to width, uncensored width. */
  canyon: GraphDataView<'float32x4'>;
  /** `float32x2`, one row per building: left and right sampled frontage length. */
  frontage: GraphDataView<'float32x2'>;
  /** `float32x2`, one row per segment, only with `streetSegment`: visible facade length per side. */
  facadeLengths: GraphDataView<'float32x2'>;
  /** `float32x2`, one row per building, only with `streetSegment`: exact facade frontage. */
  facadeFrontage: GraphDataView<'float32x2'>;
};

export type GPUStreetProfilesProps = {
  id?: string;
  /** Caller samples street centerlines with explicit spacing; normals point to the left side. */
  positions: GraphDataView<'float32x2'>;
  normals: GraphDataView<'float32x2'>;
  segments: GraphDataView<'float32x4'>;
  segmentBuildingIds: GraphDataView<'uint32'>;
  buildingHeights: GraphDataView<'float32'>;
  maximumDistance: number;
  sampleSpacing: number;
  tolerance?: number;
  /**
   * Optional straight street start/end segment for exact visible facade arc length in its
   * rectangular corridor. A stretch of wall is visible when no other segment crosses the
   * perpendicular from the street axis before it. The kernel works relative to the street start,
   * so projected-metre coordinates lose nothing beyond their own float32 rounding. A wall
   * perpendicular to the street (within `tolerance` plus 1e-5 of its length) is edge-on and shows
   * no length. Where segments coincide (their sightline distances are within `tolerance` plus 1e-5
   * of the larger of the street length and `maximumDistance` of the nearest one), the lowest row
   * among them owns the stretch and the others show no length there.
   */
  streetSegment?: GraphDataView<'float32x4'>;
  /**
   * How rays find obstacle segments. 'brute-force' tests every segment for every ray, 'bvh' builds
   * a segment BVH, and the default 'auto' picks brute force up to
   * `URBAN_BRUTE_FORCE_MAXIMUM_SEGMENT_COUNT` segments and the BVH above it. Both give the same
   * result. Compile-time.
   */
  obstacleIndex?: UrbanObstacleIndex;
  /**
   * Caller-owned result views. Any view that is absent is a graph-owned transient view, as before.
   * Formats and lengths are those of {@link GPUStreetProfilesOutput}.
   */
  output?: Partial<GPUStreetProfilesOutput>;
};

/** Side-aware first-building profiles; missing sides remain range-censored and never become observed walls. */
export function addStreetProfilesRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUStreetProfilesProps
) {
  const id = props.id ?? 'street-profiles';
  const count = props.positions.length;
  const segmentCount = props.segments.length;
  const buildingCount = props.buildingHeights.length;
  const tolerance = props.tolerance ?? 1e-6;
  validateRequirement(id, count > 0, 'positions must have at least 1 row');
  validateRequirement(id, props.normals.length === count, 'normals must have one row per position');
  validateRequirement(
    id,
    segmentCount > 0 && segmentCount <= URBAN_MAXIMUM_OBSTACLE_SEGMENT_COUNT,
    `segments must have between 1 and ${URBAN_MAXIMUM_OBSTACLE_SEGMENT_COUNT} rows`
  );
  validateRequirement(
    id,
    props.segmentBuildingIds.length === segmentCount,
    'segmentBuildingIds must have one row per segment'
  );
  validateRequirement(id, buildingCount > 0, 'buildingHeights must have at least 1 row');
  validateRequirement(
    id,
    Number.isFinite(props.maximumDistance) && props.maximumDistance > 0,
    'maximumDistance must be a positive finite number'
  );
  validateRequirement(
    id,
    Number.isFinite(props.sampleSpacing) && props.sampleSpacing > 0,
    'sampleSpacing must be a positive finite number'
  );
  validateRequirement(
    id,
    Number.isFinite(tolerance) && tolerance > 0,
    'tolerance must be a positive finite number'
  );
  if (props.streetSegment) {
    validateRequirement(id, props.streetSegment.length === 1, 'streetSegment must have 1 row');
    validateRequirement(
      id,
      segmentCount <= STREET_PROFILES_MAXIMUM_FACADE_SEGMENT_COUNT,
      `segments must have at most ${STREET_PROFILES_MAXIMUM_FACADE_SEGMENT_COUNT} rows when streetSegment is given`
    );
  }
  const output = props.output;
  const profiles = getRecipeView(graph, `${id}-profiles`, 'float32x4', count, output?.profiles);
  const buildings = getRecipeView(graph, `${id}-buildings`, 'uint32x2', count, output?.buildings);
  const validity = getRecipeView(graph, `${id}-validity`, 'uint32', count, output?.validity);
  const canyon = getRecipeView(graph, `${id}-canyon`, 'float32x4', count, output?.canyon);
  const frontage = getRecipeView(
    graph,
    `${id}-frontage`,
    'float32x2',
    buildingCount,
    output?.frontage
  );

  // One pass over the segments counts the non-finite ones and the ones whose owner is not a
  // building row. Every ray then checks two numbers instead of re-reading every segment.
  const nonFiniteFlags = createTransientView(
    graph,
    `${id}-non-finite-flags`,
    'uint32',
    segmentCount
  );
  const badOwnerFlags = createTransientView(graph, `${id}-bad-owner-flags`, 'uint32', segmentCount);
  const nonFiniteCount = createTransientView(graph, `${id}-non-finite-count`, 'uint32', 1);
  const badOwnerCount = createTransientView(graph, `${id}-bad-owner-count`, 'uint32', 1);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-segment-flags`,
      operation: id,
      invocationCount: segmentCount,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        {name: 'owners', view: props.segmentBuildingIds, type: 'u32', access: 'read'},
        {name: 'nonFiniteFlags', view: nonFiniteFlags, type: 'u32', access: 'read_write'},
        {name: 'badOwnerFlags', view: badOwnerFlags, type: 'u32', access: 'read_write'}
      ],
      body: `let base=segmentsOffset+4u*index;
let finiteSegment=isFiniteValue(segments[base])&&isFiniteValue(segments[base+1u])&&isFiniteValue(segments[base+2u])&&isFiniteValue(segments[base+3u]);
nonFiniteFlags[nonFiniteFlagsOffset+index]=u32(!finiteSegment);
badOwnerFlags[badOwnerFlagsOffset+index]=u32(owners[ownersOffset+index]>=${buildingCount}u);`
    })
  );
  graph.add(
    new GPUReduction({
      id: `${id}-non-finite-segments`,
      input: nonFiniteFlags,
      output: nonFiniteCount,
      operation: 'sum'
    }).getCommandNodes(graph)
  );
  graph.add(
    new GPUReduction({
      id: `${id}-bad-owner-segments`,
      input: badOwnerFlags,
      output: badOwnerCount,
      operation: 'sum'
    }).getCommandNodes(graph)
  );

  const obstacleBVH =
    resolveUrbanObstacleIndex(props.obstacleIndex, segmentCount) === 'bvh'
      ? addUrbanObstacleBVH(graph, {
          id: `${id}-obstacles`,
          operation: id,
          segments: props.segments,
          tolerance
        })
      : undefined;
  const rayDeclarations =
    IS_FINITE_VALUE_WGSL + getUrbanRayDeclarations(segmentCount, tolerance, obstacleBVH);
  const bvhBindings = obstacleBVH ? getUrbanRayBVHBindings(obstacleBVH) : [];

  // Raw nearest hit of both side rays: (left distance, right distance, left row, right row).
  const rayHits = createTransientView(graph, `${id}-ray-hits`, 'float32x4', count);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-rays`,
      operation: id,
      invocationCount: count,
      declarations: rayDeclarations,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'normals', view: props.normals, type: 'f32', access: 'read'},
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        ...bvhBindings,
        {name: 'rayHits', view: rayHits, type: 'f32', access: 'read_write'},
        {name: 'validity', view: validity, type: 'u32', access: 'read_write'}
      ],
      // Validity holds 8 for a sample whose own position or normal is unusable, else 0.
      body: `let origin=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
let normal=vec2<f32>(normals[normalsOffset+2u*index],normals[normalsOffset+2u*index+1u]);
let inputValid=isFiniteValue(origin.x)&&isFiniteValue(origin.y)&&isFiniteValue(normal.x)&&isFiniteValue(normal.y)&&isFiniteValue(length(normal))&&length(normal)>0.0;
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
for(var side=0u;side<2u;side++){
  var hit=vec2<f32>(missing,-1.0);
  if(inputValid){hit=castRay(origin,normalize(normal)*select(1.0,-1.0,side==1u),${props.maximumDistance});}
  rayHits[rayHitsOffset+4u*index+side]=hit.x;
  rayHits[rayHitsOffset+4u*index+2u+side]=hit.y;
}
validity[validityOffset+index]=select(8u,0u,inputValid);`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-profile-flags`,
      operation: id,
      invocationCount: count,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'nonFiniteCount', view: nonFiniteCount, type: 'u32', access: 'read'},
        {name: 'badOwnerCount', view: badOwnerCount, type: 'u32', access: 'read'},
        {name: 'owners', view: props.segmentBuildingIds, type: 'u32', access: 'read'},
        {name: 'heights', view: props.buildingHeights, type: 'f32', access: 'read'},
        {name: 'rayHits', view: rayHits, type: 'f32', access: 'read'},
        {name: 'profiles', view: profiles, type: 'f32', access: 'read_write'},
        {name: 'buildings', view: buildings, type: 'u32', access: 'read_write'},
        {name: 'validity', view: validity, type: 'u32', access: 'read_write'}
      ],
      body: `let valid=validity[validityOffset+index]==0u&&nonFiniteCount[nonFiniteCountOffset]==0u&&badOwnerCount[badOwnerCountOffset]==0u;
var flags=0u;
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
for(var side=0u;side<2u;side++){
  var distance=missing;
  var height=missing;
  var owner=0xffffffffu;
  if(valid){
    distance=rayHits[rayHitsOffset+4u*index+side];
    let hitRow=rayHits[rayHitsOffset+4u*index+2u+side];
    if(hitRow>=0.0){
      owner=owners[ownersOffset+u32(hitRow)];
      height=heights[heightsOffset+owner];
      if(!isFiniteValue(height)||height<0.0){height=missing;}
      flags|=1u<<side;
    }
    if(distance<=${tolerance}){flags|=4u;}
  }
  profiles[profilesOffset+4u*index+side]=distance;
  profiles[profilesOffset+4u*index+side+2u]=height;
  buildings[buildingsOffset+2u*index+side]=owner;
}
if(!valid){flags|=8u;}
validity[validityOffset+index]=flags;`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-canyon`,
      operation: id,
      invocationCount: count,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'profiles', view: profiles, type: 'f32', access: 'read'},
        {name: 'validity', view: validity, type: 'u32', access: 'read'},
        {name: 'output', view: canyon, type: 'f32', access: 'read_write'}
      ],
      body: `let base=profilesOffset+4u*index;
let flags=validity[validityOffset+index];
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
let complete=flags==3u;
let width=profiles[base]+profiles[base+1u];
let meanHeight=0.5*(profiles[base+2u]+profiles[base+3u]);
output[outputOffset+4u*index]=select(missing,width,complete);
output[outputOffset+4u*index+1u]=select(missing,meanHeight,complete);
output[outputOffset+4u*index+2u]=select(missing,meanHeight/width,complete&&width>0.0&&isFiniteValue(meanHeight));
output[outputOffset+4u*index+3u]=select(missing,width,(flags&12u)==0u);`
    })
  );
  // One thread per building scans the samples: building rows times samples, no shared state.
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-frontage`,
      operation: id,
      invocationCount: buildingCount,
      bindings: [
        {name: 'buildings', view: buildings, type: 'u32', access: 'read'},
        {name: 'validity', view: validity, type: 'u32', access: 'read'},
        {name: 'output', view: frontage, type: 'f32', access: 'read_write'}
      ],
      body: `for(var side=0u;side<2u;side++){
  var length=0.0;
  for(var sample=0u;sample<${count}u;sample++){
    let flags=validity[validityOffset+sample];
    if((flags&12u)==0u&&(flags&(1u<<side))!=0u&&buildings[buildingsOffset+2u*sample+side]==index){length+=${props.sampleSpacing};}
  }
  output[outputOffset+2u*index+side]=length;
}`
    })
  );
  let facades:
    | {lengths: GraphDataView<'float32x2'>; frontage: GraphDataView<'float32x2'>}
    | undefined;
  if (props.streetSegment) {
    const eventStride = 3 * segmentCount + FACADE_EVENT_FIXED_COUNT;
    const facadeEvents = getRecipeView(
      graph,
      `${id}-facade-events`,
      'float32',
      segmentCount * eventStride
    );
    const lengths = getRecipeView(
      graph,
      `${id}-facade-lengths`,
      'float32x2',
      segmentCount,
      output?.facadeLengths
    );
    const facadeFrontage = getRecipeView(
      graph,
      `${id}-facade-frontage`,
      'float32x2',
      buildingCount,
      output?.facadeFrontage
    );
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-facade-intervals`,
        operation: id,
        invocationCount: segmentCount,
        workgroupSize: 2,
        // The facade kernel computes its sightline origins, so it works relative to the street
        // start: the same geometry gives the same lengths at any distance from the origin.
        declarations: `${IS_FINITE_VALUE_WGSL}${getUrbanRayDeclarations(segmentCount, tolerance, obstacleBVH, 'local')}
${getFacadeVisibilityDeclarations(segmentCount, props.maximumDistance, tolerance, obstacleBVH)}
const EVENT_STRIDE:u32=${eventStride}u;
fn setEvent(row:u32,slot:u32,value:f32){facadeEvents[facadeEventsOffset+row*EVENT_STRIDE+slot]=value;}
fn getEvent(row:u32,slot:u32)->f32{return facadeEvents[facadeEventsOffset+row*EVENT_STRIDE+slot];}`,
        bindings: [
          {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
          {name: 'street', view: props.streetSegment, type: 'f32', access: 'read'},
          {name: 'nonFiniteCount', view: nonFiniteCount, type: 'u32', access: 'read'},
          ...bvhBindings,
          {name: 'output', view: lengths, type: 'f32', access: 'read_write'},
          {name: 'facadeEvents', view: facadeEvents, type: 'f32', access: 'read_write'}
        ],
        // Each segment owns a thread that sorts its own events: cost grows with segments squared,
        // which is why streetSegment is limited to STREET_PROFILES_MAXIMUM_FACADE_SEGMENT_COUNT.
        body: `urbanFrameOrigin=vec2<f32>(street[streetOffset],street[streetOffset+1u]);
let streetVector=vec2<f32>(street[streetOffset+2u],street[streetOffset+3u])-urbanFrameOrigin;
let streetLength=length(streetVector);
let axis=streetVector/streetLength;
let normal=vec2<f32>(-axis.y,axis.x);
// Every coordinate below is relative to the street start.
let segment=readSegment(index);
let start=segment.xy;
let end=segment.zw;
let displacement=end-start;
let facadeLength=length(displacement);
let projectedStart=dot(start,axis);
let projectedDelta=dot(displacement,axis);
let sideStart=dot(start,normal);
let sideDelta=dot(displacement,normal);
// A wall seen edge-on from the street has no visible length.
let edgeOn=abs(projectedDelta)<=${tolerance}+${FACADE_RELATIVE_SLACK}*facadeLength;
let slack=${tolerance}+${FACADE_RELATIVE_SLACK}*max(streetLength,${getWGSLFloatLiteral(props.maximumDistance)});
setEvent(index,0u,0.0);
setEvent(index,1u,1.0);
var size=2u;
if(!edgeOn){
  setEvent(index,size,clamp(-projectedStart/projectedDelta,0.0,1.0));size++;
  setEvent(index,size,clamp((streetLength-projectedStart)/projectedDelta,0.0,1.0));size++;
}
if(abs(sideDelta)>${tolerance}){
  for(var boundary=0u;boundary<3u;boundary++){
    setEvent(index,size,clamp((f32(i32(boundary)-1)*${props.maximumDistance}-sideStart)/sideDelta,0.0,1.0));size++;
  }
}
for(var other=0u;other<${segmentCount}u;other++){
  let obstacle=readSegment(other);
  if(!edgeOn){
    setEvent(index,size,clamp((dot(obstacle.xy,axis)-projectedStart)/projectedDelta,0.0,1.0));size++;
    setEvent(index,size,clamp((dot(obstacle.zw,axis)-projectedStart)/projectedDelta,0.0,1.0));size++;
  }
  let otherDelta=obstacle.zw-obstacle.xy;
  let denominator=displacement.x*otherDelta.y-displacement.y*otherDelta.x;
  if(abs(denominator)>${tolerance}){
    let difference=obstacle.xy-start;
    let crossing=(difference.x*otherDelta.y-difference.y*otherDelta.x)/denominator;
    setEvent(index,size,clamp(crossing,0.0,1.0));size++;
  }
}
for(var event=1u;event<size;event++){
  let value=getEvent(index,event);
  var insertion=event;
  loop{
    if(insertion==0u||getEvent(index,insertion-1u)<=value){break;}
    setEvent(index,insertion,getEvent(index,insertion-1u));
    insertion--;
  }
  setEvent(index,insertion,value);
}
var total=vec2<f32>(0.0);
for(var event=1u;event<size;event++){
  let middle=0.5*(getEvent(index,event)+getEvent(index,event-1u));
  let longitudinal=projectedStart+middle*projectedDelta;
  let lateral=sideStart+middle*sideDelta;
  if(edgeOn||longitudinal<0.0||longitudinal>streetLength||abs(lateral)<=${tolerance}||abs(lateral)>${props.maximumDistance}){continue;}
  // The sightline leaves the street axis at the foot of the perpendicular through the interval
  // midpoint, so it reaches this wall at abs(lateral). The lowest row within a slack of the
  // nearest hit owns the stretch; the slack scales with the corridor, and no two distances are
  // compared for equality.
  if(isFacadeIntervalVisible(longitudinal*axis,normal*sign(lateral),abs(lateral),slack,index)){
    let side=u32(lateral<0.0);
    total[side]+=(getEvent(index,event)-getEvent(index,event-1u))*facadeLength;
  }
}
let valid=isFiniteValue(streetLength)&&streetLength>${tolerance}&&nonFiniteCount[nonFiniteCountOffset]==0u;
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
output[outputOffset+2u*index]=select(missing,total.x,valid);
output[outputOffset+2u*index+1u]=select(missing,total.y,valid);`
      })
    );
    // One thread per building sums the lengths of the segments it owns (building rows times segments).
    graph.add(
      createRecipeKernelNode(graph, {
        id: `${id}-facade-summary`,
        operation: id,
        invocationCount: buildingCount,
        bindings: [
          {name: 'owners', view: props.segmentBuildingIds, type: 'u32', access: 'read'},
          {name: 'lengths', view: lengths, type: 'f32', access: 'read'},
          {name: 'output', view: facadeFrontage, type: 'f32', access: 'read_write'}
        ],
        body: `var total=vec2<f32>(0.0);
for(var segment=0u;segment<${segmentCount}u;segment++){
  if(owners[ownersOffset+segment]==index){total+=vec2<f32>(lengths[lengthsOffset+2u*segment],lengths[lengthsOffset+2u*segment+1u]);}
}
output[outputOffset+2u*index]=total.x;
output[outputOffset+2u*index+1u]=total.y;`
      })
    );
    facades = {lengths, frontage: facadeFrontage};
  }
  return {
    outputs: {profiles, buildings, validity, canyon, frontage},
    facades,
    provenance: {
      positions: props.positions,
      normals: props.normals,
      segments: props.segments,
      segmentBuildingIds: props.segmentBuildingIds,
      maximumDistance: props.maximumDistance,
      sampleSpacing: props.sampleSpacing,
      tolerance,
      obstacleIndex: obstacleBVH ? ('bvh' as const) : ('brute-force' as const)
    }
  };
}
