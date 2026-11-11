// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {GUARDED_ATAN2_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';
import {classifyUrbanViewpoints, type UrbanFreeSpaceRegion} from './classify-viewpoints';
import {
  URBAN_MAXIMUM_OBSTACLE_SEGMENT_COUNT,
  addUrbanObstacleBVH,
  getUrbanRayBVHBindings,
  getUrbanRayDeclarations,
  resolveUrbanObstacleIndex,
  type UrbanObstacleIndex
} from './urban-ray-utils';

/**
 * Most rays per viewpoint when `sampling` is 'vertices': each viewpoint sorts its angles with one
 * thread, so the per-thread insertion sort must stay short.
 */
export const ISOVISTS_MAXIMUM_VERTEX_SAMPLED_RAY_COUNT = 4096;

/** Most rays per viewpoint: the measures kernel walks one viewpoint's ring in a single thread. */
export const ISOVISTS_MAXIMUM_RAY_COUNT = 65536;

/** Fewest rays per viewpoint that close a polygon. */
export const ISOVISTS_MINIMUM_RAY_COUNT = 3;

/** Rays sampled just before, at and after each obstacle endpoint when `sampling` is 'vertices'. */
const ISOVISTS_RAYS_PER_ENDPOINT = 3;

/** Largest accepted `angularEpsilon` in radians: a larger offset no longer brackets an endpoint. */
const ISOVISTS_MAXIMUM_ANGULAR_EPSILON = 0.01;

export type GPUIsovistsOutput = {
  /** `float32x2`, `positions.length * rayCount` rows: ray end points. */
  endpoints: GraphDataView<'float32x2'>;
  /** `float32`, `positions.length * rayCount` rows: ray lengths. */
  distances: GraphDataView<'float32'>;
  /** `uint32`, `positions.length * rayCount` rows: hit obstacle segment rows, or 0xffffffff. */
  segments: GraphDataView<'uint32'>;
  /** `uint32`, `positions.length * rayCount` rows: one when the ray is valid. */
  validity: GraphDataView<'uint32'>;
  /** `float32x4`, one row per viewpoint: area, perimeter, maximum distance, validity. */
  measures: GraphDataView<'float32x4'>;
  /** `float32`, `positions.length * rayCount` rows: ray angles, ascending per viewpoint. */
  angles: GraphDataView<'float32'>;
  /** `uint32`, one row per viewpoint: 0 free, 1 blocked, 2 boundary, 3 non-finite. */
  viewpointClassifications: GraphDataView<'uint32'>;
};

export type GPUIsovistsProps = {
  id?: string;
  /** Free-space viewpoints. Points inside opaque obstacle interiors are not meaningful viewpoints. */
  positions: GraphDataView<'float32x2'>;
  segments: GraphDataView<'float32x4'>;
  rayCount: number;
  /** Add rays just before/at/after each obstacle endpoint, then sort by angle on GPU. */
  sampling?: 'uniform' | 'vertices';
  angularEpsilon?: number;
  maximumDistance: number;
  tolerance?: number;
  /** Optional explicit filled-ring free-space classification. Default checks boundaries only. */
  region?: UrbanFreeSpaceRegion;
  /**
   * How rays find obstacle segments. 'brute-force' tests every segment for every ray, 'bvh' builds
   * a segment BVH, and the default 'auto' picks brute force up to
   * `URBAN_BRUTE_FORCE_MAXIMUM_SEGMENT_COUNT` segments and the BVH above it. Both give the same
   * result. Compile-time.
   */
  obstacleIndex?: UrbanObstacleIndex;
  /**
   * Caller-owned result views. Any view that is absent is a graph-owned transient view, as before.
   * Formats and lengths are those of {@link GPUIsovistsOutput}.
   */
  output?: Partial<GPUIsovistsOutput>;
};

/** Angularly sampled obstacle isovist polygons, counterclockwise from +X, with an explicit circular range boundary. */
export function addIsovistsRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUIsovistsProps
) {
  const id = props.id ?? 'isovists';
  const maximumDistance = props.maximumDistance;
  const vertexSampling = props.sampling === 'vertices';
  const angularEpsilon = props.angularEpsilon ?? 1e-5;
  const rayCount =
    props.rayCount + (vertexSampling ? 2 * ISOVISTS_RAYS_PER_ENDPOINT * props.segments.length : 0);
  const tolerance = props.tolerance ?? 1e-6;
  const viewpointCount = props.positions.length;
  const segmentCount = props.segments.length;
  validateRequirement(
    id,
    !vertexSampling || rayCount <= ISOVISTS_MAXIMUM_VERTEX_SAMPLED_RAY_COUNT,
    `rayCount plus ${2 * ISOVISTS_RAYS_PER_ENDPOINT} rays per obstacle segment must be at most ${ISOVISTS_MAXIMUM_VERTEX_SAMPLED_RAY_COUNT} when sampling is 'vertices'`
  );
  validateRequirement(
    id,
    Number.isFinite(angularEpsilon) &&
      angularEpsilon > 0 &&
      angularEpsilon < ISOVISTS_MAXIMUM_ANGULAR_EPSILON,
    `angularEpsilon must be a finite number in (0, ${ISOVISTS_MAXIMUM_ANGULAR_EPSILON})`
  );
  validateRequirement(id, viewpointCount > 0, 'positions must have at least 1 row');
  validateRequirement(
    id,
    segmentCount > 0 && segmentCount <= URBAN_MAXIMUM_OBSTACLE_SEGMENT_COUNT,
    `segments must have between 1 and ${URBAN_MAXIMUM_OBSTACLE_SEGMENT_COUNT} rows`
  );
  validateRequirement(
    id,
    Number.isSafeInteger(rayCount) &&
      rayCount >= ISOVISTS_MINIMUM_RAY_COUNT &&
      rayCount <= ISOVISTS_MAXIMUM_RAY_COUNT,
    `rayCount must be an integer, and the total rays per viewpoint must be in [${ISOVISTS_MINIMUM_RAY_COUNT}, ${ISOVISTS_MAXIMUM_RAY_COUNT}]`
  );
  validateRequirement(
    id,
    Number.isFinite(maximumDistance) && maximumDistance > 0,
    'maximumDistance must be a positive finite number'
  );
  validateRequirement(
    id,
    Number.isFinite(tolerance) && tolerance > 0,
    'tolerance must be a positive finite number'
  );
  const output = props.output;
  const rayRows = viewpointCount * rayCount;
  const obstacleBVH =
    resolveUrbanObstacleIndex(props.obstacleIndex, segmentCount) === 'bvh'
      ? addUrbanObstacleBVH(graph, {
          id: `${id}-obstacles`,
          operation: id,
          segments: props.segments,
          tolerance
        })
      : undefined;
  const viewpointClassifications = classifyUrbanViewpoints(graph, {
    id,
    positions: props.positions,
    segments: props.segments,
    tolerance,
    obstacleBVH,
    region: props.region,
    output: output?.viewpointClassifications
  });
  const angles = getRecipeView(graph, `${id}-angles`, 'float32', rayRows, output?.angles);
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-angular-events`,
      operation: id,
      invocationCount: viewpointCount,
      workgroupSize: 2,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        {name: 'angles', view: angles, type: 'f32', access: 'read_write'}
      ],
      declarations: GUARDED_ATAN2_WGSL,
      // Per-viewpoint loops run over the ray capacity (at most 4096 for 'vertices' sampling).
      body: vertexSampling
        ? `let origin=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
for(var ray=0u;ray<${props.rayCount}u;ray++){
  angles[anglesOffset+index*${rayCount}u+ray]=6.283185307179586*f32(ray)/${props.rayCount}.0;
}
for(var endpoint=0u;endpoint<${props.segments.length * 2}u;endpoint++){
  let point=vec2<f32>(segments[segmentsOffset+2u*endpoint],segments[segmentsOffset+2u*endpoint+1u])-origin;
  // The second argument is a coordinate difference, which can be -0.0 (contract section 11).
  let center=guardedAtan2(point.y,point.x);
  for(var side=0u;side<3u;side++){
    var angle=center+(f32(side)-1.0)*${angularEpsilon};
    angle=angle-floor(angle/6.283185307179586)*6.283185307179586;
    angles[anglesOffset+index*${rayCount}u+${props.rayCount}u+3u*endpoint+side]=angle;
  }
}
for(var position=1u;position<${rayCount}u;position++){
  let value=angles[anglesOffset+index*${rayCount}u+position];
  var insertion=position;
  loop{
    if(insertion==0u||angles[anglesOffset+index*${rayCount}u+insertion-1u]<=value){break;}
    angles[anglesOffset+index*${rayCount}u+insertion]=angles[anglesOffset+index*${rayCount}u+insertion-1u];
    insertion--;
  }
  angles[anglesOffset+index*${rayCount}u+insertion]=value;
}`
        : `for(var ray=0u;ray<${rayCount}u;ray++){
  angles[anglesOffset+index*${rayCount}u+ray]=6.283185307179586*f32(ray)/${rayCount}.0;
}`
    })
  );
  const endpoints = getRecipeView(
    graph,
    `${id}-endpoints`,
    'float32x2',
    rayRows,
    output?.endpoints
  );
  const distances = getRecipeView(graph, `${id}-distances`, 'float32', rayRows, output?.distances);
  const segments = getRecipeView(graph, `${id}-segments`, 'uint32', rayRows, output?.segments);
  const validity = getRecipeView(graph, `${id}-validity`, 'uint32', rayRows, output?.validity);
  const measures = getRecipeView(
    graph,
    `${id}-measures`,
    'float32x4',
    viewpointCount,
    output?.measures
  );
  // Raw nearest hit per ray. Segment finiteness is already part of the viewpoint classification
  // (a non-finite segment classifies every viewpoint as 3), so the rays do not re-check it.
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-cast`,
      operation: id,
      invocationCount: rayRows,
      declarations: getUrbanRayDeclarations(segmentCount, tolerance, obstacleBVH),
      bindings: [
        {name: 'angles', view: angles, type: 'f32', access: 'read'},
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'segments', view: props.segments, type: 'f32', access: 'read'},
        ...(obstacleBVH ? getUrbanRayBVHBindings(obstacleBVH) : []),
        {name: 'distances', view: distances, type: 'f32', access: 'read_write'},
        {name: 'hits', view: segments, type: 'u32', access: 'read_write'}
      ],
      body: `let viewpoint=index/${rayCount}u;
let angle=angles[anglesOffset+index];
let direction=vec2<f32>(cos(angle),sin(angle));
let origin=vec2<f32>(positions[positionsOffset+2u*viewpoint],positions[positionsOffset+2u*viewpoint+1u]);
let hit=castRay(origin,direction,${maximumDistance});
distances[distancesOffset+index]=hit.x;
hits[hitsOffset+index]=select(0xffffffffu,u32(max(hit.y,0.0)),hit.y>=0.0);`
    })
  );
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-cast-validity`,
      operation: id,
      invocationCount: rayRows,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'classifications', view: viewpointClassifications, type: 'u32', access: 'read'},
        {name: 'angles', view: angles, type: 'f32', access: 'read'},
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'endpoints', view: endpoints, type: 'f32', access: 'read_write'},
        {name: 'distances', view: distances, type: 'f32', access: 'read_write'},
        {name: 'hits', view: segments, type: 'u32', access: 'read_write'},
        {name: 'validity', view: validity, type: 'u32', access: 'read_write'}
      ],
      body: `let viewpoint=index/${rayCount}u;
let angle=angles[anglesOffset+index];
let direction=vec2<f32>(cos(angle),sin(angle));
let origin=vec2<f32>(positions[positionsOffset+2u*viewpoint],positions[positionsOffset+2u*viewpoint+1u]);
let distanceValue=distances[distancesOffset+index];
let hitRow=hits[hitsOffset+index];
let valid=classifications[classificationsOffset+viewpoint]==0u&&isFiniteValue(origin.x)&&isFiniteValue(origin.y)&&distanceValue>${tolerance};
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
let endpoint=origin+direction*distanceValue;
endpoints[endpointsOffset+2u*index]=select(missing,endpoint.x,valid);
endpoints[endpointsOffset+2u*index+1u]=select(missing,endpoint.y,valid);
distances[distancesOffset+index]=select(missing,distanceValue,valid);
hits[hitsOffset+index]=select(0xffffffffu,hitRow,hitRow!=0xffffffffu&&valid);
validity[validityOffset+index]=u32(valid);`
    })
  );
  // One thread per viewpoint walks its own ring of at most ISOVISTS_MAXIMUM_RAY_COUNT rays.
  graph.add(
    createRecipeKernelNode(graph, {
      id: `${id}-measures`,
      operation: id,
      invocationCount: viewpointCount,
      bindings: [
        {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
        {name: 'endpoints', view: endpoints, type: 'f32', access: 'read'},
        {name: 'validity', view: validity, type: 'u32', access: 'read'},
        {name: 'output', view: measures, type: 'f32', access: 'read_write'}
      ],
      body: `let origin=vec2<f32>(positions[positionsOffset+2u*index],positions[positionsOffset+2u*index+1u]);
var area=0.0;
var perimeter=0.0;
var maximum=0.0;
var valid=true;
for(var ray=0u;ray<${rayCount}u;ray++){
  let first=index*${rayCount}u+ray;
  let second=index*${rayCount}u+(ray+1u)%${rayCount}u;
  valid=valid&&validity[validityOffset+first]!=0u;
  let start=vec2<f32>(endpoints[endpointsOffset+2u*first],endpoints[endpointsOffset+2u*first+1u])-origin;
  let end=vec2<f32>(endpoints[endpointsOffset+2u*second],endpoints[endpointsOffset+2u*second+1u])-origin;
  area+=0.5*(start.x*end.y-start.y*end.x);
  perimeter+=distance(start,end);
  maximum=max(maximum,length(start));
}
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
output[outputOffset+4u*index]=select(missing,area,valid);
output[outputOffset+4u*index+1u]=select(missing,perimeter,valid);
output[outputOffset+4u*index+2u]=select(missing,maximum,valid);
output[outputOffset+4u*index+3u]=f32(valid);`
    })
  );
  return {
    outputs: {endpoints, distances, segments, validity, measures, angles, viewpointClassifications},
    provenance: {
      positions: props.positions,
      obstacles: props.segments,
      rayCount,
      maximumDistance,
      angularSpacing: (2 * Math.PI) / props.rayCount,
      sampling: props.sampling ?? 'uniform',
      angularEpsilon,
      tolerance,
      obstacleIndex: obstacleBVH ? ('bvh' as const) : ('brute-force' as const)
    }
  };
}
