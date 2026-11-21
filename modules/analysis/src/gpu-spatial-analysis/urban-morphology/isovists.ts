// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {captureUrbanLandscapeDestinations} from '../landscape-ecology/urban-landscape-destinations';
import type {GPUCommandGraph, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createRecipeKernelNode, getRecipeView} from '../recipes/expansion-recipe-utils';
import {GUARDED_ATAN2_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateCapacity,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {classifyUrbanViewpoints, type UrbanFreeSpaceRegion} from './classify-viewpoints';
import {validateUrbanLandscapeRecipeViews} from '../landscape-ecology/urban-landscape-recipe-validation';
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

/**
 * Total rays per viewpoint. Uniform sampling uses `baseRayCount`; vertex sampling adds six
 * rays per obstacle segment. Multiply this count by `positions.length` for every per-ray output.
 * The base count must be an integer in [3, 65536], segmentCount in [1, 16777216], and the expanded
 * vertex-sampled count must be at most 4096.
 */
export function getGPUIsovistRayCount(
  baseRayCount: number,
  segmentCount: number,
  sampling: 'uniform' | 'vertices' = 'uniform'
): number {
  return calculateIsovistRayCount('isovists', baseRayCount, segmentCount, sampling);
}

function calculateIsovistRayCount(
  id: string,
  baseRayCount: number,
  segmentCount: number,
  sampling: 'uniform' | 'vertices'
): number {
  validateCapacity(
    id,
    'rayCount',
    baseRayCount,
    ISOVISTS_MINIMUM_RAY_COUNT,
    ISOVISTS_MAXIMUM_RAY_COUNT
  );
  validateRequirement(
    id,
    Number.isSafeInteger(segmentCount) &&
      segmentCount > 0 &&
      segmentCount <= URBAN_MAXIMUM_OBSTACLE_SEGMENT_COUNT,
    `segments must have between 1 and ${URBAN_MAXIMUM_OBSTACLE_SEGMENT_COUNT} rows`
  );
  validateRequirement(
    id,
    sampling === 'uniform' || sampling === 'vertices',
    "sampling must be 'uniform' or 'vertices'"
  );
  const rayCount =
    baseRayCount + (sampling === 'vertices' ? 2 * ISOVISTS_RAYS_PER_ENDPOINT * segmentCount : 0);
  validateRequirement(
    id,
    sampling !== 'vertices' || rayCount <= ISOVISTS_MAXIMUM_VERTEX_SAMPLED_RAY_COUNT,
    `rayCount plus ${2 * ISOVISTS_RAYS_PER_ENDPOINT} rays per obstacle segment must be at most ${ISOVISTS_MAXIMUM_VERTEX_SAMPLED_RAY_COUNT} when sampling is 'vertices'`
  );
  return rayCount;
}

export type GPUIsovistsOutput = {
  /** `float32x2`, `positions.length * getGPUIsovistRayCount(rayCount, segments.length, sampling)` rows: ray end points. */
  endpoints: GraphDataView<'float32x2'>;
  /** `float32`, the same expanded per-ray row count as endpoints: ray lengths. */
  distances: GraphDataView<'float32'>;
  /** `uint32`, the same expanded per-ray row count as endpoints: hit obstacle segment rows, or 0xffffffff. */
  segments: GraphDataView<'uint32'>;
  /** `uint32`, the same expanded per-ray row count as endpoints: one when the ray is valid. */
  validity: GraphDataView<'uint32'>;
  /** `float32x4`, one row per viewpoint: local area (coordinate units squared), perimeter and maximum distance (coordinate units), validity. */
  localMeasures: GraphDataView<'float32x4'>;
  /** `float32`, the same expanded per-ray row count as endpoints: ray angles, ascending per viewpoint. */
  angles: GraphDataView<'float32'>;
  /** `uint32`, one row per viewpoint: 0 free, 1 blocked, 2 boundary, 3 non-finite. */
  viewpointClassifications: GraphDataView<'uint32'>;
};

export type GPUIsovistsProps = {
  id?: string;
  /** Free-space viewpoints. Points inside opaque obstacle interiors are not meaningful viewpoints. */
  positions: GraphDataView<'float32x2'>;
  segments: GraphDataView<'float32x4'>;
  /** Base uniform rays per viewpoint, an integer in [3, 65536]. Use getGPUIsovistRayCount for output sizing. */
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
  validateRequirement(id, !('outputs' in props), 'outputs was removed; use output');
  validateRequirement(
    id,
    props.obstacleIndex === undefined ||
      (['auto', 'brute-force', 'bvh'] as readonly string[]).includes(props.obstacleIndex),
    `obstacleIndex must be auto or brute-force or bvh; received ${String(props.obstacleIndex)}`
  );
  validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
  validateSinglePackedView(props.segments, ['float32x4'], `${id} segments`);
  const maximumDistance = props.maximumDistance;
  const vertexSampling = props.sampling === 'vertices';
  const baseRayCount = props.rayCount;
  const angularEpsilon = props.angularEpsilon ?? 1e-5;
  const rayCount = calculateIsovistRayCount(
    id,
    baseRayCount,
    props.segments.length,
    props.sampling ?? 'uniform'
  );
  const tolerance = props.tolerance ?? 1e-6;
  const viewpointCount = props.positions.length;
  const segmentCount = props.segments.length;
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
    Number.isFinite(maximumDistance) && maximumDistance > 0,
    'maximumDistance must be a positive finite number'
  );
  validateRequirement(
    id,
    Number.isFinite(tolerance) && tolerance > 0,
    'tolerance must be a positive finite number'
  );
  const suppliedOutput = props.output;
  validateRequirement(
    id,
    !suppliedOutput || !('measures' in suppliedOutput),
    'output.measures was removed; use output.localMeasures'
  );
  const output = captureUrbanLandscapeDestinations(id, {output: suppliedOutput}, [
    'endpoints',
    'distances',
    'segments',
    'validity',
    'localMeasures',
    'angles',
    'viewpointClassifications'
  ]);
  const rayRows = viewpointCount * rayCount;
  const inputs = {positions: props.positions, segments: props.segments};
  validateUrbanLandscapeRecipeViews(
    graph,
    id,
    inputs,
    {
      endpoints: {view: output?.endpoints, format: 'float32x2', length: rayRows},
      distances: {view: output?.distances, format: 'float32', length: rayRows},
      segments: {view: output?.segments, format: 'uint32', length: rayRows},
      validity: {view: output?.validity, format: 'uint32', length: rayRows},
      localMeasures: {view: output?.localMeasures, format: 'float32x4', length: viewpointCount},
      angles: {view: output?.angles, format: 'float32', length: rayRows},
      viewpointClassifications: {
        view: output?.viewpointClassifications,
        format: 'uint32',
        length: viewpointCount
      }
    },
    [
      {reads: inputs, writes: {viewpointClassifications: output?.viewpointClassifications}},
      {reads: inputs, writes: {angles: output?.angles}},
      {
        reads: {...inputs, angles: output?.angles},
        writes: {distances: output?.distances, segments: output?.segments}
      },
      {
        reads: {
          positions: props.positions,
          angles: output?.angles,
          viewpointClassifications: output?.viewpointClassifications
        },
        writes: {
          endpoints: output?.endpoints,
          distances: output?.distances,
          segments: output?.segments,
          validity: output?.validity
        }
      },
      {
        reads: {
          angles: output?.angles,
          distances: output?.distances,
          validity: output?.validity
        },
        writes: {localMeasures: output?.localMeasures}
      }
    ]
  );
  // Resolve all overrides before classification or BVH construction appends any commands.
  const angles = getRecipeView(graph, `${id}-angles`, 'float32', rayRows, output?.angles);
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
    output?.localMeasures
  );
  const classificationOutput = getRecipeView(
    graph,
    `${id}-viewpoint-classifications`,
    'uint32',
    viewpointCount,
    output?.viewpointClassifications
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
  const viewpointClassifications = classifyUrbanViewpoints(graph, {
    id,
    positions: props.positions,
    segments: props.segments,
    tolerance,
    obstacleBVH,
    region: props.region,
    output: classificationOutput
  });
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
for(var ray=0u;ray<${baseRayCount}u;ray++){
  angles[anglesOffset+index*${rayCount}u+ray]=6.283185307179586*f32(ray)/${baseRayCount}.0;
}
for(var endpoint=0u;endpoint<${props.segments.length * 2}u;endpoint++){
  let point=vec2<f32>(segments[segmentsOffset+2u*endpoint],segments[segmentsOffset+2u*endpoint+1u])-origin;
  // The second argument is a coordinate difference, which can be -0.0 (contract section 11).
  let center=guardedAtan2(point.y,point.x);
  for(var side=0u;side<3u;side++){
    var angle=center+(f32(side)-1.0)*${angularEpsilon};
    angle=angle-floor(angle/6.283185307179586)*6.283185307179586;
    angles[anglesOffset+index*${rayCount}u+${baseRayCount}u+3u*endpoint+side]=angle;
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
        {name: 'angles', view: angles, type: 'f32', access: 'read'},
        {name: 'distances', view: distances, type: 'f32', access: 'read'},
        {name: 'validity', view: validity, type: 'u32', access: 'read'},
        {name: 'output', view: measures, type: 'f32', access: 'read_write'}
      ],
      body: `var area=0.0;
var perimeter=0.0;
var maximum=0.0;
var valid=true;
for(var ray=0u;ray<${rayCount}u;ray++){
  let first=index*${rayCount}u+ray;
  let second=index*${rayCount}u+(ray+1u)%${rayCount}u;
  valid=valid&&validity[validityOffset+first]!=0u;
  let firstAngle=angles[anglesOffset+first];
  let secondAngle=angles[anglesOffset+second];
  let firstDistance=distances[distancesOffset+first];
  let start=vec2<f32>(cos(firstAngle),sin(firstAngle))*firstDistance;
  let end=vec2<f32>(cos(secondAngle),sin(secondAngle))*distances[distancesOffset+second];
  area+=0.5*(start.x*end.y-start.y*end.x);
  perimeter+=distance(start,end);
  maximum=max(maximum,firstDistance);
}
let missing=bitcast<f32>(0x7fc00000u | (index & 0u));
output[outputOffset+4u*index]=select(missing,area,valid);
output[outputOffset+4u*index+1u]=select(missing,perimeter,valid);
output[outputOffset+4u*index+2u]=select(missing,maximum,valid);
output[outputOffset+4u*index+3u]=f32(valid);`
    })
  );
  return {
    outputs: {
      endpoints,
      distances,
      segments,
      validity,
      localMeasures: measures,
      angles,
      viewpointClassifications
    },
    provenance: {
      positions: props.positions,
      obstacles: props.segments,
      rayCount,
      baseRayCount,
      maximumDistance,
      angularSpacing: (2 * Math.PI) / baseRayCount,
      sampling: props.sampling ?? 'uniform',
      angularEpsilon,
      tolerance,
      obstacleIndex: obstacleBVH ? ('bvh' as const) : ('brute-force' as const)
    }
  };
}
