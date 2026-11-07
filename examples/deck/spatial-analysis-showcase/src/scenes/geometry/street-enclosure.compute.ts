// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {COORDINATE_SYSTEM, type Layer} from '@deck.gl/core';
import type {CommandEncoder} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {
  addIsovistsRecipe,
  addStreetProfilesRecipe
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneFrame, SceneInstance} from '../scene';
import {buildStreetEnclosureGeometry} from './street-enclosure-geometry';
import {toRadians} from '@math.gl/core';

type Options = {
  view: 'footprints' | 'isovist' | 'profile' | 'limits';
  observerEast: number;
  observerNorth: number;
  profileBearing: number;
};
const OBSERVER_COUNT = 1;
const RAY_COUNT = 128;
const MAXIMUM_DISTANCE = 250;

/** Real Loop footprints, with GPU isovist and street-profile recipes over a bounded local subset. */
export async function createStreetEnclosure(
  ctx: SceneContext<Options>
): Promise<SceneInstance<Options>> {
  const dataset = ctx.datasets.get('chicago-buildings');
  const origin: [number, number] = [-87.635, 41.8825];
  const geometry = buildStreetEnclosureGeometry(dataset, origin);
  let observerPosition = Float32Array.of(0, 0);
  let sampleNormal = Float32Array.of(1, 0);
  const resources = new SpatialAnalysisResources(ctx.device, 'street-enclosure');
  const buildingSegments = resources.createBuffer('building-segments', geometry.buildingSegments);
  const analysisSegmentBuffer = resources.createBuffer(
    'analysis-segments',
    geometry.analysisSegments
  );
  const outputSegmentBuffer = resources.createBuffer(
    'isovist-segments',
    OBSERVER_COUNT * RAY_COUNT * 16,
    0
  );
  const profileHitBuffer = resources.createBuffer('profile-hit-segments', 32);
  const observerBuffer = resources.createBuffer('observer', observerPosition);
  const normalBuffer = resources.createBuffer('profile-normal', sampleNormal);

  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'street-enclosure-analysis'});
  const observerView = importGraphBuffer(
    graph,
    'observer',
    observerBuffer,
    'float32x2',
    OBSERVER_COUNT
  );
  const normalView = importGraphBuffer(
    graph,
    'profile-normal',
    normalBuffer,
    'float32x2',
    OBSERVER_COUNT
  );
  const segmentsView = importGraphBuffer(
    graph,
    'analysis-segments',
    analysisSegmentBuffer,
    'float32x4',
    geometry.analysisSegments.length / 4
  );
  const ownerView = importGraphBuffer(
    graph,
    'segment-owners',
    resources.createBuffer('segment-owners', geometry.analysisBuildingIds),
    'uint32',
    geometry.analysisBuildingIds.length
  );
  const heightData = geometry.analysisBuildingHeights.length
    ? geometry.analysisBuildingHeights
    : Float32Array.of(0);
  const heightView = importGraphBuffer(
    graph,
    'building-heights',
    resources.createBuffer('building-heights', heightData),
    'float32',
    heightData.length
  );
  const isovist = addIsovistsRecipe(graph, {
    id: 'street-enclosure-isovist',
    positions: observerView,
    segments: segmentsView,
    rayCount: RAY_COUNT,
    maximumDistance: MAXIMUM_DISTANCE,
    sampling: 'uniform'
  });
  // The source has incomplete story counts. Zero is only a placeholder to meet the recipe's
  // required height column; the profile's plan distances remain meaningful and height outputs are
  // deliberately omitted from the map and interpretation.
  const profile = addStreetProfilesRecipe(graph, {
    id: 'street-enclosure-profile',
    positions: observerView,
    normals: normalView,
    segments: segmentsView,
    segmentBuildingIds: ownerView,
    buildingHeights: heightView,
    maximumDistance: MAXIMUM_DISTANCE,
    sampleSpacing: 1
  });
  const profileSummaryBuffer = resources.createBuffer('profile-summary', 16);
  const profileSummaryView = importGraphBuffer(
    graph,
    'profile-summary',
    profileSummaryBuffer,
    'float32x4',
    1
  );
  addKernelPass(graph, {
    id: 'profile-summary-copy',
    invocationCount: 1,
    bindings: [
      {name: 'source', view: profile.outputs.canyon, type: 'f32', access: 'read'},
      {name: 'destination', view: profileSummaryView, type: 'f32', access: 'read_write'}
    ],
    body: 'for (var component = 0u; component < 4u; component++) { destination[destinationOffset + component] = source[sourceOffset + component]; }'
  });
  const profileHitView = importGraphBuffer(
    graph,
    'profile-hit-segments',
    profileHitBuffer,
    'float32x4',
    2
  );
  addKernelPass(graph, {
    id: 'profile-hits-to-segments',
    invocationCount: 2,
    bindings: [
      {name: 'observers', view: observerView, type: 'f32', access: 'read'},
      {name: 'normals', view: normalView, type: 'f32', access: 'read'},
      {name: 'profiles', view: profile.outputs.profiles, type: 'f32', access: 'read'},
      {name: 'validity', view: profile.outputs.validity, type: 'u32', access: 'read'},
      {name: 'segments', view: profileHitView, type: 'f32', access: 'read_write'}
    ],
    body: `let side=index;let flags=validity[validityOffset];let seen=(flags&(1u<<side))!=0u;let origin=vec2<f32>(observers[observersOffset],observers[observersOffset+1u]);let normal=normalize(vec2<f32>(normals[normalsOffset],normals[normalsOffset+1u]));let sign=select(1.0,-1.0,side==1u);let distance=profiles[profilesOffset+side];let missing=bitcast<f32>(0x7fc00000u | (index & 0u));let end=origin+normal*(sign*distance);segments[segmentsOffset+4u*index]=select(missing,origin.x,seen);segments[segmentsOffset+4u*index+1u]=select(missing,origin.y,seen);segments[segmentsOffset+4u*index+2u]=select(missing,end.x,seen);segments[segmentsOffset+4u*index+3u]=select(missing,end.y,seen);`
  });
  const outputSegmentsView = importGraphBuffer(
    graph,
    'isovist-draw-segments',
    outputSegmentBuffer,
    'float32x4',
    OBSERVER_COUNT * RAY_COUNT
  );
  addKernelPass(graph, {
    id: 'isovist-rays-to-segments',
    invocationCount: OBSERVER_COUNT * RAY_COUNT,
    bindings: [
      {name: 'observers', view: observerView, type: 'f32', access: 'read'},
      {name: 'endpoints', view: isovist.outputs.endpoints, type: 'f32', access: 'read'},
      {name: 'segments', view: outputSegmentsView, type: 'f32', access: 'read_write'}
    ],
    declarations: `const RAY_COUNT: u32 = ${RAY_COUNT}u;`,
    body: `let endpoint=index*2u;let observer=index/RAY_COUNT;segments[segmentsOffset+4u*index]=observers[observersOffset+2u*observer];segments[segmentsOffset+4u*index+1u]=observers[observersOffset+2u*observer+1u];segments[segmentsOffset+4u*index+2u]=endpoints[endpointsOffset+endpoint];segments[segmentsOffset+4u*index+3u]=endpoints[endpointsOffset+endpoint+1u];`
  });
  const compiled = resources.track(await graph.compileAsync());
  ctx.setReadout('buildingSample', geometry.buildingCount.toLocaleString());
  ctx.setReadout('analysisInputs', `${geometry.analysisSegments.length / 4} segments`);
  ctx.setReadout('analysisRange', `${MAXIMUM_DISTANCE} m`);
  let profileRevision = 0;
  let profileReadbackRevision = 0;
  const profileReader = new SummaryReader(
    resources,
    'street-profile-width',
    [{buffer: profileSummaryBuffer, size: 16}],
    bytes => {
      if (profileReadbackRevision !== profileRevision) return;
      const canyon = new Float32Array(bytes);
      ctx.setReadout(
        'profileWidth',
        Number.isFinite(canyon[0]) ? `${canyon[0].toFixed(0)} m` : 'Range-censored'
      );
    }
  );

  // The origin is always put in free space first; offset controls then reject footprint interiors.
  let safeBase: [number, number] | null = null;
  for (let north = -220; north <= 220 && !safeBase; north += 10) {
    for (let east = -220; east <= 220; east += 10) {
      if (!geometry.isInsideFootprint(east, north)) {
        safeBase = [east, north];
        break;
      }
    }
  }
  if (!safeBase) throw new Error('Chicago footprint sample contains no clear observer location');
  observerPosition = Float32Array.of(safeBase[0], safeBase[1]);
  observerBuffer.write(observerPosition);
  const updateObserver = (options: Options) => {
    const east = safeBase![0] + options.observerEast;
    const north = safeBase![1] + options.observerNorth;
    if (geometry.isInsideFootprint(east, north)) {
      ctx.setStatus(
        'Observer position is inside a building footprint; move it into the street space.'
      );
      return;
    }
    observerPosition = Float32Array.of(east, north);
    observerBuffer.write(observerPosition);
    ctx.setStatus('Observer is in mapped street space.');
  };
  const updateProfileDirection = (options: Options) => {
    const angle = toRadians(options.profileBearing);
    sampleNormal = Float32Array.of(Math.cos(angle), Math.sin(angle));
    normalBuffer.write(sampleNormal);
  };
  updateObserver(ctx.options);
  updateProfileDirection(ctx.options);
  ctx.setReadout('observerPosition', 'Clear street-space point');

  const getLayers = (): Layer[] => {
    const common = {
      coordinateSystem: COORDINATE_SYSTEM.METER_OFFSETS,
      coordinateOrigin: [origin[0], origin[1], 0] as [number, number, number],
      pickable: false
    };
    const layers: Layer[] = [
      new SpatialAnalysisSegmentLayer({
        ...common,
        id: 'street-enclosure-building-footprints',
        segments: buildingSegments,
        instanceCount: geometry.buildingSegments.length / 4,
        color: [100, 111, 118, 148],
        widthPixels: 1.15
      })
    ];
    layers.push(
      new SpatialAnalysisPointLayer({
        ...common,
        id: 'street-enclosure-observer',
        positions: observerBuffer,
        instanceCount: OBSERVER_COUNT,
        color: [32, 133, 139, 255],
        radiusPixels: 6,
        outlineColor: [255, 255, 255, 255],
        outlineWidthPixels: 1.5
      })
    );
    if (ctx.options.view === 'isovist') {
      layers.push(
        new SpatialAnalysisSegmentLayer({
          ...common,
          id: 'street-enclosure-isovist-rays',
          segments: outputSegmentBuffer,
          instanceCount: OBSERVER_COUNT * RAY_COUNT,
          color: [18, 133, 139, 220],
          widthPixels: 1
        })
      );
    }
    if (ctx.options.view === 'profile') {
      layers.push(
        new SpatialAnalysisSegmentLayer({
          ...common,
          id: 'street-enclosure-profile-cross-section',
          segments: profileHitBuffer,
          instanceCount: 2,
          color: [205, 113, 48, 230],
          widthPixels: 3
        })
      );
    }
    return layers;
  };

  return {
    getCompiledGraphs: () => [compiled] as CompiledGPUCommandGraph<never>[],
    encode(commandEncoder: CommandEncoder, _frame: SceneFrame) {
      compiled.encode(commandEncoder, {parameters: undefined});
      if (!profileReader.isPending) profileReadbackRevision = profileRevision;
      profileReader.request(commandEncoder);
      profileReader.flush(commandEncoder);
    },
    getLayers,
    setOption(id, _value, options) {
      if (id === 'observerEast' || id === 'observerNorth') updateObserver(options);
      if (id === 'profileBearing') updateProfileDirection(options);
      if (id === 'observerEast' || id === 'observerNorth' || id === 'profileBearing') {
        profileRevision++;
        ctx.setReadout('profileWidth', 'Updating for the current observer and direction');
        profileReader.markStale();
      }
      ctx.requestLayers();
    },
    onThemeChange() {
      ctx.requestLayers();
    },
    destroy() {
      profileReader.stop();
      resources.destroy();
    }
  };
}
