// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {
  GPUHaversineDistance,
  GPUPointSpatialQuery,
  GPUPairwisePointDistance,
  GPUPairwisePointInPolygon,
  GPUPairwisePointLinestringNearest,
  GPUPairwisePointSegmentDistance,
  GPU_POINT_IN_POLYGON_CLASSIFICATION
} from '@luma-contributors/graph/geospatial';
import {GPU_EARTH_MEAN_RADIUS_METERS} from '@luma-contributors/graph/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {GraphSwap} from '../../engine/graph-swap';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';

export type RouteFootpointsOptions = {
  route: 'Red' | 'Blue' | 'Brn' | 'G';
  sample: number;
  framing: 'overview' | 'nearest' | 'reference' | 'workframe' | 'query';
  queryRadius: number;
  query: 'radius' | 'bounds' | 'polygon';
  haversineUnits: 'meters' | 'kilometers';
};
type RouteRow = {id: string; name: string; type: string; color: string};
type Variant = {
  key: string;
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  stations: Float32Array;
  referencePositions: Float32Array;
  sourceRows: number[];
  routeName: string;
  routeColor: [number, number, number, number];
  nearest: ReturnType<SpatialAnalysisResources['createBuffer']>;
  nearestFootpointCount: number;
  distance: ReturnType<SpatialAnalysisResources['createBuffer']>;
  queryIds: ReturnType<SpatialAnalysisResources['createBuffer']>;
  queryCount: ReturnType<SpatialAnalysisResources['createBuffer']>;
  queryOverflow: ReturnType<SpatialAnalysisResources['createBuffer']>;
  category: ReturnType<SpatialAnalysisResources['createBuffer']>;
  chartDistances: number[];
  summaryIdentity: number;
  stationBuffer: ReturnType<SpatialAnalysisResources['createBuffer']>;
  routeSegments: ReturnType<SpatialAnalysisResources['createBuffer']>;
  routeSegmentCount: number;
  footpointLinks: ReturnType<SpatialAnalysisResources['createBuffer']>;
  footpointLinkCount: number;
  selectedSegment: ReturnType<SpatialAnalysisResources['createBuffer']>;
  selectedSegmentCount: number;
  workframeSegments: ReturnType<SpatialAnalysisResources['createBuffer']>;
  workframeSegmentCount: number;
  querySupport: ReturnType<SpatialAnalysisResources['createBuffer']>;
  querySupportSegmentCount: number;
  queryMatchCount: number;
};
type SegmentVariant = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
};

const getVariantKey = (state: RouteFootpointsOptions): string =>
  `${state.route}|${state.query}|${state.queryRadius}|${state.haversineUnits}`;

/** Build one honest, route-aligned GPU geometry investigation. */
export async function createRouteFootpoints(
  ctx: SceneContext<RouteFootpointsOptions>
): Promise<SceneInstance<RouteFootpointsOptions>> {
  const transit = ctx.datasets.get('cta-transit');
  const origin = transit.defaultOrigin;
  const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
  const projection = transit.getProjection(origin);
  const positions = transit.projectColumn('stopPosition', origin);
  const stopMode = transit.column<Uint8Array>('stopMode');
  const stopRoutes = transit.column<Uint16Array>('stopRoutes');
  const stopRouteOffsets = transit.column<Uint32Array>('stopRouteOffsets');
  const routeRows = (transit.properties.routes as RouteRow[]) ?? [];
  const shapes = transit.projectColumn('shapeVertices', origin);
  const shapeOffsets = transit.column<Uint32Array>('shapePathOffsets');
  const shapeRoutes = transit.column<Uint16Array>('shapeRoute');
  const shapeDirections = transit.column<Uint8Array>('shapeDirection');
  const resources = new SpatialAnalysisResources(ctx.device, 'route-footpoints');
  const graphSwap = resources.track(new GraphSwap());
  const segmentGraphSwap = resources.track(new GraphSwap());
  let destroyed = false;
  let requestIdentity = 0;
  let variant: Variant;
  let segmentVariant: SegmentVariant | null = null;
  let selectedStation = 0;
  let frameCounter = 0;

  function fitProjectedBounds(coordinates: readonly number[], minimumSpan: number): void {
    if (coordinates.length < 2) return;
    let minimumX = Infinity;
    let minimumY = Infinity;
    let maximumX = -Infinity;
    let maximumY = -Infinity;
    for (let index = 0; index + 1 < coordinates.length; index += 2) {
      const positionX = coordinates[index];
      const positionY = coordinates[index + 1];
      if (!Number.isFinite(positionX) || !Number.isFinite(positionY)) return;
      minimumX = Math.min(minimumX, positionX);
      minimumY = Math.min(minimumY, positionY);
      maximumX = Math.max(maximumX, positionX);
      maximumY = Math.max(maximumY, positionY);
    }
    const centerX = (minimumX + maximumX) / 2;
    const centerY = (minimumY + maximumY) / 2;
    const halfWidth = Math.max(minimumSpan, maximumX - minimumX) / 2;
    const halfHeight = Math.max(minimumSpan, maximumY - minimumY) / 2;
    const southwest = projection.unproject(centerX - halfWidth, centerY - halfHeight);
    const northeast = projection.unproject(centerX + halfWidth, centerY + halfHeight);
    ctx.fitBounds([southwest[0], southwest[1], northeast[0], northeast[1]], {
      transitionMs: 900,
      maxZoom: 19.5,
      padding: 56
    });
  }

  function fitSelectedEvidence(
    state: RouteFootpointsOptions,
    row: number,
    nearestX: number,
    nearestY: number,
    hasNearest: boolean
  ): void {
    const stationX = variant.stations[row * 2];
    const stationY = variant.stations[row * 2 + 1];
    if (!Number.isFinite(stationX) || !Number.isFinite(stationY)) return;
    if (state.framing === 'nearest') {
      if (hasNearest) fitProjectedBounds([stationX, stationY, nearestX, nearestY], 40);
    } else if (state.framing === 'reference') {
      fitProjectedBounds(
        [
          stationX,
          stationY,
          variant.referencePositions[row * 2],
          variant.referencePositions[row * 2 + 1]
        ],
        80
      );
    } else if (state.framing === 'workframe') {
      const firstStationX = variant.stations[0];
      const firstStationY = variant.stations[1];
      fitProjectedBounds(
        [firstStationX - 400, firstStationY - 400, firstStationX + 400, firstStationY + 400],
        800
      );
    } else if (state.framing === 'query') {
      const firstStationX = variant.stations[0];
      const firstStationY = variant.stations[1];
      const halfExtent = state.query === 'polygon' ? 400 : state.queryRadius;
      fitProjectedBounds(
        [
          firstStationX - halfExtent,
          firstStationY - halfExtent,
          firstStationX + halfExtent,
          firstStationY + halfExtent
        ],
        40
      );
    }
  }

  function buildStationSubset(routeIndex: number): {
    stationRows: number[];
    stationPositions: Float32Array;
    parts: Float32Array[];
    routeName: string;
  } {
    const stationRows: number[] = [];
    for (let row = 0; row < stopMode.length; row++) {
      if (stopMode[row] !== 1) continue;
      for (let slot = stopRouteOffsets[row]; slot < stopRouteOffsets[row + 1]; slot++)
        if (stopRoutes[slot] === routeIndex) {
          stationRows.push(row);
          break;
        }
    }
    const stationPositions = new Float32Array(stationRows.length * 2);
    stationRows.forEach((row, index) =>
      stationPositions.set(positions.subarray(row * 2, row * 2 + 2), index * 2)
    );
    const parts: Float32Array[] = [];
    for (let shape = 0; shape < shapeRoutes.length; shape++) {
      if (shapeRoutes[shape] !== routeIndex || shapeDirections[shape] !== 0) continue;
      const start = shapeOffsets[shape];
      const end = shapeOffsets[shape + 1];
      if (end - start >= 2) parts.push(shapes.slice(start * 2, end * 2));
    }
    const sourceInfo = routeRows[routeIndex];
    return {
      stationRows,
      stationPositions,
      parts,
      routeName: sourceInfo?.name ?? `Route ${routeIndex}`
    };
  }

  async function build(state: RouteFootpointsOptions, identity: number): Promise<Variant> {
    const nextRoute = state.route;
    const routeIndex = routeRows.findIndex(row => row.type === 'rail' && row.id === nextRoute);
    const subset = buildStationSubset(routeIndex);
    const sourceRouteColor = routeRows[routeIndex]?.color ?? '#64748b';
    const routeColor: [number, number, number, number] = [
      Number.parseInt(sourceRouteColor.slice(1, 3), 16),
      Number.parseInt(sourceRouteColor.slice(3, 5), 16),
      Number.parseInt(sourceRouteColor.slice(5, 7), 16),
      190
    ];
    const parts = subset.parts;
    const totalVertices = parts.reduce((sum, part) => sum + part.length / 2, 0);
    const flatVertices = new Float32Array(totalVertices * 2);
    const lineOffsets = new Uint32Array(parts.length + 1);
    let vertexOffset = 0;
    parts.forEach((part, index) => {
      flatVertices.set(part, vertexOffset * 2);
      vertexOffset += part.length / 2;
      lineOffsets[index + 1] = vertexOffset;
    });
    const pairedFeatureOffsets = new Uint32Array(subset.stationRows.length + 1);
    const pairedLineOffsets: number[] = [0];
    const pairedVertices: number[] = [];
    // Each station row receives every route part, preserving part boundaries and tie order.
    for (let row = 0; row < subset.stationRows.length; row++) {
      for (const part of parts) {
        pairedVertices.push(...part);
        pairedLineOffsets.push(pairedVertices.length / 2);
      }
      pairedFeatureOffsets[row + 1] = pairedLineOffsets.length - 1;
    }
    const pairedPositions = Float32Array.from(pairedVertices);
    const pairedLineOffsetArray = Uint32Array.from(pairedLineOffsets);
    const safeCount = Math.max(1, subset.stationRows.length);
    const localResources = new SpatialAnalysisResources(
      ctx.device,
      `route-footpoints-${nextRoute}-${identity}`
    );
    try {
      const stationBuffer = localResources.createBuffer(
        'station-positions',
        subset.stationPositions
      );
      const pairedPathBuffer = localResources.createBuffer('paired-path-vertices', pairedPositions);
      const featureOffsetBuffer = localResources.createBuffer(
        'feature-offsets',
        pairedFeatureOffsets
      );
      const lineOffsetBuffer = localResources.createBuffer('line-offsets', pairedLineOffsetArray);
      const nearestBuffer = localResources.createBuffer('nearest-positions', safeCount * 8);
      const distanceBuffer = localResources.createBuffer('nearest-distances', safeCount * 4);
      const lineIndexBuffer = localResources.createBuffer('nearest-line-indices', safeCount * 4);
      const segmentIndexBuffer = localResources.createBuffer(
        'nearest-segment-indices',
        safeCount * 4
      );
      const pointDistanceBuffer = localResources.createBuffer(
        'reference-point-distance',
        safeCount * 4
      );
      const footpointDistanceBuffer = localResources.createBuffer(
        'footpoint-point-distance',
        safeCount * 4
      );
      const haversineBuffer = localResources.createBuffer('haversine-distance', safeCount * 4);
      const polygonClassBuffer = localResources.createBuffer(
        'polygon-classification',
        safeCount * 4
      );
      const referenceBufferData = new Float32Array(safeCount * 2);
      const referenceLngLat = new Float32Array(safeCount * 2);
      // Pair each station with the first vertex of its first source part as a declared reference point.
      for (let row = 0; row < subset.stationRows.length; row++) {
        const part = parts[0];
        const refX = part?.[0] ?? subset.stationPositions[row * 2];
        const refY = part?.[1] ?? subset.stationPositions[row * 2 + 1];
        referenceBufferData.set([refX, refY], row * 2);
        const refLonLat = projection.unproject(refX, refY);
        referenceLngLat.set(refLonLat, row * 2);
      }
      const referenceBuffer = localResources.createBuffer('reference-point', referenceBufferData);
      const stationLngLat = new Float32Array(safeCount * 2);
      for (let row = 0; row < subset.stationRows.length; row++) {
        const stopRow = subset.stationRows[row];
        stationLngLat.set(
          transit.column<Float32Array>('stopPosition').subarray(stopRow * 2, stopRow * 2 + 2),
          row * 2
        );
      }
      const stationLngLatBuffer = localResources.createBuffer(
        'station-longitude-latitude',
        stationLngLat
      );
      const polygonVertices = new Float32Array(safeCount * 8);
      const polygonRingOffsetValues = new Uint32Array(safeCount + 1);
      const polygonOffsetValues = new Uint32Array(safeCount + 1);
      const firstStation =
        subset.stationPositions.length >= 2
          ? subset.stationPositions.slice(0, 2)
          : Float32Array.of(0, 0);
      for (let row = 0; row < safeCount; row++) {
        polygonVertices.set(
          [
            firstStation[0] - 400,
            firstStation[1] - 400,
            firstStation[0] + 400,
            firstStation[1] - 400,
            firstStation[0] + 400,
            firstStation[1] + 400,
            firstStation[0] - 400,
            firstStation[1] + 400
          ],
          row * 8
        );
        polygonRingOffsetValues[row + 1] = (row + 1) * 4;
        polygonOffsetValues[row + 1] = row + 1;
      }
      const polygonBuffer = localResources.createBuffer('synthetic-polygon', polygonVertices);
      const queryPolygonBuffer = localResources.createBuffer(
        'query-synthetic-polygon',
        Float32Array.of(
          firstStation[0] - 400,
          firstStation[1] - 400,
          firstStation[0] + 400,
          firstStation[1] - 400,
          firstStation[0] + 400,
          firstStation[1] + 400,
          firstStation[0] - 400,
          firstStation[1] + 400
        )
      );
      const queryRingOffsetsBuffer = localResources.createBuffer(
        'query-synthetic-ring-offsets',
        Uint32Array.of(0, 4)
      );
      const polygonRingOffsets = localResources.createBuffer(
        'synthetic-ring-offsets',
        polygonRingOffsetValues
      );
      const polygonOffsets = localResources.createBuffer(
        'synthetic-polygon-offsets',
        polygonOffsetValues
      );
      const polygonFeatures = localResources.createBuffer(
        'synthetic-feature-offsets',
        Uint32Array.from({length: safeCount + 1}, (_value, index) => index)
      );
      const queryValues =
        state.query === 'polygon'
          ? Float32Array.of(
              firstStation[0] - 400,
              firstStation[1] - 400,
              firstStation[0] + 400,
              firstStation[1] + 400
            )
          : state.query === 'bounds'
            ? Float32Array.of(
                firstStation[0] - state.queryRadius,
                firstStation[1] - state.queryRadius,
                firstStation[0] + state.queryRadius,
                firstStation[1] + state.queryRadius
              )
            : state.query === 'radius'
              ? Float32Array.of(firstStation[0], firstStation[1], state.queryRadius)
              : Float32Array.of(
                  firstStation[0] - state.queryRadius,
                  firstStation[1] - state.queryRadius,
                  firstStation[0] + state.queryRadius,
                  firstStation[1] + state.queryRadius
                );
      const queryParameters = localResources.createParameterBuffer(
        'query-parameters',
        'float32',
        queryValues.length,
        queryValues
      );
      const queryIds = localResources.createBuffer('query-ids', safeCount * 4);
      const queryCount = localResources.createBuffer('query-count', 4);
      const queryRequired = localResources.createBuffer('query-required-count', 4);
      const queryOverflow = localResources.createBuffer('query-overflow', 4);
      const linkBuffer = localResources.createBuffer('selected-footpoint-link', 16);
      const selectedSegmentBuffer = localResources.createBuffer('selected-nearest-segment', 16);
      const workframeSegments = localResources.createBuffer(
        'workframe-polygon-outline',
        Float32Array.of(
          firstStation[0] - 400,
          firstStation[1] - 400,
          firstStation[0] + 400,
          firstStation[1] - 400,
          firstStation[0] + 400,
          firstStation[1] - 400,
          firstStation[0] + 400,
          firstStation[1] + 400,
          firstStation[0] + 400,
          firstStation[1] + 400,
          firstStation[0] - 400,
          firstStation[1] + 400,
          firstStation[0] - 400,
          firstStation[1] + 400,
          firstStation[0] - 400,
          firstStation[1] - 400
        )
      );
      const querySupportSegments: number[] = [];
      if (state.query === 'radius') {
        const sampleCount = 48;
        for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex++) {
          const firstAngle = (sampleIndex / sampleCount) * Math.PI * 2;
          const secondAngle = ((sampleIndex + 1) / sampleCount) * Math.PI * 2;
          querySupportSegments.push(
            firstStation[0] + Math.cos(firstAngle) * state.queryRadius,
            firstStation[1] + Math.sin(firstAngle) * state.queryRadius,
            firstStation[0] + Math.cos(secondAngle) * state.queryRadius,
            firstStation[1] + Math.sin(secondAngle) * state.queryRadius
          );
        }
      } else {
        const bounds =
          state.query === 'polygon'
            ? [
                firstStation[0] - 400,
                firstStation[1] - 400,
                firstStation[0] + 400,
                firstStation[1] + 400
              ]
            : [
                firstStation[0] - state.queryRadius,
                firstStation[1] - state.queryRadius,
                firstStation[0] + state.queryRadius,
                firstStation[1] + state.queryRadius
              ];
        querySupportSegments.push(
          bounds[0],
          bounds[1],
          bounds[2],
          bounds[1],
          bounds[2],
          bounds[1],
          bounds[2],
          bounds[3],
          bounds[2],
          bounds[3],
          bounds[0],
          bounds[3],
          bounds[0],
          bounds[3],
          bounds[0],
          bounds[1]
        );
      }
      const querySupport = localResources.createBuffer(
        'query-support',
        Float32Array.from(querySupportSegments)
      );
      const routeSegmentValues: number[] = [];
      for (const part of parts)
        for (let index = 0; index + 3 < part.length; index += 2)
          routeSegmentValues.push(part[index], part[index + 1], part[index + 2], part[index + 3]);
      const routeSegmentBuffer = localResources.createBuffer(
        'route-source-segments',
        Float32Array.from(routeSegmentValues)
      );
      const routeSegmentCount = routeSegmentValues.length / 4;
      const graph = new GPUCommandGraph<void>(ctx.device, {id: `route-footpoints-${nextRoute}`});
      const stationView = importGraphBuffer(
        graph,
        'station-positions',
        stationBuffer,
        'float32x2',
        subset.stationRows.length
      );
      const pathView = importGraphBuffer(
        graph,
        'paired-path-vertices',
        pairedPathBuffer,
        'float32x2',
        pairedPositions.length / 2
      );
      const featureOffsetsView = importGraphBuffer(
        graph,
        'feature-offsets',
        featureOffsetBuffer,
        'uint32',
        pairedFeatureOffsets.length
      );
      const lineOffsetsView = importGraphBuffer(
        graph,
        'line-offsets',
        lineOffsetBuffer,
        'uint32',
        pairedLineOffsetArray.length
      );
      const nearestView = importGraphBuffer(
        graph,
        'nearest-positions',
        nearestBuffer,
        'float32x2',
        subset.stationRows.length
      );
      const nearestDistanceView = importGraphBuffer(
        graph,
        'nearest-distances',
        distanceBuffer,
        'float32',
        subset.stationRows.length
      );
      const nearestLineView = importGraphBuffer(
        graph,
        'nearest-line-indices',
        lineIndexBuffer,
        'uint32',
        subset.stationRows.length
      );
      const nearestSegmentView = importGraphBuffer(
        graph,
        'nearest-segment-indices',
        segmentIndexBuffer,
        'uint32',
        subset.stationRows.length
      );
      const referenceView = importGraphBuffer(
        graph,
        'reference-point',
        referenceBuffer,
        'float32x2',
        subset.stationRows.length
      );
      const pointDistanceView = importGraphBuffer(
        graph,
        'reference-point-distance',
        pointDistanceBuffer,
        'float32',
        subset.stationRows.length
      );
      const footpointDistanceView = importGraphBuffer(
        graph,
        'footpoint-point-distance',
        footpointDistanceBuffer,
        'float32',
        subset.stationRows.length
      );
      const stationLonLatView = importGraphBuffer(
        graph,
        'station-longitude-latitude',
        stationLngLatBuffer,
        'float32x2',
        subset.stationRows.length
      );
      const referenceLonLatBuffer = localResources.createBuffer(
        'reference-longitude-latitude',
        referenceLngLat
      );
      const referenceLonLatView = importGraphBuffer(
        graph,
        'reference-longitude-latitude',
        referenceLonLatBuffer,
        'float32x2',
        subset.stationRows.length
      );
      const haversineView = importGraphBuffer(
        graph,
        'haversine-distance',
        haversineBuffer,
        'float32',
        subset.stationRows.length
      );
      const polygonClassView = importGraphBuffer(
        graph,
        'polygon-classification',
        polygonClassBuffer,
        'uint32',
        subset.stationRows.length
      );
      const polygonView = importGraphBuffer(
        graph,
        'synthetic-polygon',
        polygonBuffer,
        'float32x2',
        safeCount * 4
      );
      const polygonRingView = importGraphBuffer(
        graph,
        'synthetic-ring-offsets',
        polygonRingOffsets,
        'uint32',
        polygonRingOffsetValues.length
      );
      const polygonOffsetView = importGraphBuffer(
        graph,
        'synthetic-polygon-offsets',
        polygonOffsets,
        'uint32',
        polygonOffsetValues.length
      );
      const polygonFeatureView = importGraphBuffer(
        graph,
        'synthetic-feature-offsets',
        polygonFeatures,
        'uint32',
        safeCount + 1
      );
      const queryPolygonView = importGraphBuffer(
        graph,
        'query-synthetic-polygon',
        queryPolygonBuffer,
        'float32x2',
        4
      );
      const queryRingOffsetsView = importGraphBuffer(
        graph,
        'query-synthetic-ring-offsets',
        queryRingOffsetsBuffer,
        'uint32',
        2
      );
      const queryParameterView = queryParameters.importToGraph(graph);
      const queryIdsView = importGraphBuffer(graph, 'query-ids', queryIds, 'uint32', safeCount);
      const queryCountView = importGraphBuffer(graph, 'query-count', queryCount, 'uint32', 1);
      const queryRequiredView = importGraphBuffer(
        graph,
        'query-required-count',
        queryRequired,
        'uint32',
        1
      );
      const queryOverflowView = importGraphBuffer(
        graph,
        'query-overflow',
        queryOverflow,
        'uint32',
        1
      );
      const spatialContext = projection.spatialContext;
      graph.add(
        new GPUPairwisePointLinestringNearest({
          queryPositions: stationView,
          positions: pathView,
          featureOffsets: featureOffsetsView,
          lineOffsets: lineOffsetsView,
          output: {
            distances: nearestDistanceView,
            nearestPositions: nearestView,
            lineIndices: nearestLineView,
            segmentIndices: nearestSegmentView
          },
          spatialContext
        })
      );
      graph.add(
        new GPUPairwisePointDistance({
          id: 'route-reference-point-distance',
          left: stationView,
          right: referenceView,
          output: pointDistanceView,
          spatialContext
        })
      );
      graph.add(
        new GPUPairwisePointDistance({
          id: 'route-footpoint-distance',
          left: stationView,
          right: nearestView,
          output: footpointDistanceView,
          spatialContext
        })
      );
      graph.add(
        new GPUHaversineDistance({
          left: stationLonLatView,
          right: referenceLonLatView,
          output: haversineView,
          units: state.haversineUnits,
          sphereRadius:
            state.haversineUnits === 'meters'
              ? GPU_EARTH_MEAN_RADIUS_METERS
              : GPU_EARTH_MEAN_RADIUS_METERS / 1000
        })
      );
      graph.add(
        new GPUPairwisePointInPolygon({
          queryPositions: stationView,
          positions: polygonView,
          featureOffsets: polygonFeatureView,
          polygonOffsets: polygonOffsetView,
          ringOffsets: polygonRingView,
          output: polygonClassView,
          spatialContext
        })
      );
      graph.add(
        new GPUPointSpatialQuery({
          positions: stationView,
          kind: state.query,
          parameters: queryParameterView,
          polygon:
            state.query === 'polygon'
              ? {positions: queryPolygonView, ringOffsets: queryRingOffsetsView}
              : undefined,
          spatialContext,
          output: {
            ids: queryIdsView,
            status: {
              count: queryCountView,
              requiredCount: queryRequiredView,
              overflow: queryOverflowView
            }
          }
        })
      );
      const compiled = await graph.compileAsync();
      localResources.track(compiled);
      let liveSummaryIdentity = identity;
      async function buildSegmentDistance(
        row: number,
        localLineIndex: number,
        localSegmentIndex: number,
        nearestDistance: number
      ): Promise<SegmentVariant | null> {
        const globalLineIndex = row * parts.length + localLineIndex;
        if (globalLineIndex >= pairedLineOffsets.length - 1) return null;
        const firstVertex = pairedLineOffsets[globalLineIndex] + localSegmentIndex;
        if (firstVertex + 1 >= pairedLineOffsets[globalLineIndex + 1]) return null;
        const segmentResources = new SpatialAnalysisResources(
          ctx.device,
          `route-footpoints-segment-${identity}-${row}`
        );
        try {
          const point = Float32Array.of(
            subset.stationPositions[row * 2],
            subset.stationPositions[row * 2 + 1]
          );
          const segmentStarts = Float32Array.of(
            pairedPositions[firstVertex * 2],
            pairedPositions[firstVertex * 2 + 1]
          );
          const segmentEnds = Float32Array.of(
            pairedPositions[(firstVertex + 1) * 2],
            pairedPositions[(firstVertex + 1) * 2 + 1]
          );
          const pointBuffer = segmentResources.createBuffer('point', point);
          const startBuffer = segmentResources.createBuffer('segment-start', segmentStarts);
          const endBuffer = segmentResources.createBuffer('segment-end', segmentEnds);
          const resultBuffer = segmentResources.createBuffer('segment-distance', 4);
          const segmentGraph = new GPUCommandGraph<void>(ctx.device, {
            id: `route-footpoints-segment-${identity}-${row}`
          });
          const pointView = importGraphBuffer(segmentGraph, 'point', pointBuffer, 'float32x2', 1);
          const startView = importGraphBuffer(
            segmentGraph,
            'segment-start',
            startBuffer,
            'float32x2',
            1
          );
          const endView = importGraphBuffer(segmentGraph, 'segment-end', endBuffer, 'float32x2', 1);
          const resultView = importGraphBuffer(
            segmentGraph,
            'segment-distance',
            resultBuffer,
            'float32',
            1
          );
          segmentGraph.add(
            new GPUPairwisePointSegmentDistance({
              positions: pointView,
              segmentStarts: startView,
              segmentEnds: endView,
              output: resultView,
              spatialContext: projection.spatialContext
            })
          );
          const compiled = await segmentGraph.compileAsync();
          segmentResources.track(compiled);
          const reader = new SummaryReader(
            segmentResources,
            'nearest-segment-check',
            [{buffer: resultBuffer, size: 4}],
            bytes => {
              if (
                destroyed ||
                requestIdentity !== identity ||
                variant?.summaryIdentity !== identity ||
                selectedStation !== row ||
                getVariantKey(ctx.options) !== getVariantKey(state)
              )
                return;
              const segmentDistance = new Float32Array(bytes)[0];
              ctx.setReadout(
                'segmentDistanceCheck',
                Number.isFinite(segmentDistance)
                  ? `${segmentDistance.toFixed(3)} m · nearest linestring ${nearestDistance.toFixed(3)} m`
                  : 'unavailable'
              );
            }
          );
          return {resources: segmentResources, compiled, reader};
        } catch (error) {
          segmentResources.destroy();
          throw error;
        }
      }
      const reader = new SummaryReader(
        localResources,
        'route-summary',
        [
          {buffer: distanceBuffer, size: subset.stationRows.length * 4},
          {buffer: nearestBuffer, size: subset.stationRows.length * 8},
          {buffer: lineIndexBuffer, size: subset.stationRows.length * 4},
          {buffer: segmentIndexBuffer, size: subset.stationRows.length * 4},
          {buffer: pointDistanceBuffer, size: subset.stationRows.length * 4},
          {buffer: footpointDistanceBuffer, size: subset.stationRows.length * 4},
          {buffer: haversineBuffer, size: subset.stationRows.length * 4},
          {buffer: polygonClassBuffer, size: subset.stationRows.length * 4},
          {buffer: queryCount, size: 4},
          {buffer: queryRequired, size: 4},
          {buffer: queryOverflow, size: 4},
          {buffer: queryIds, size: safeCount * 4}
        ],
        bytes => {
          if (
            destroyed ||
            variant?.summaryIdentity !== liveSummaryIdentity ||
            identity !== requestIdentity ||
            getVariantKey(ctx.options) !== getVariantKey(state)
          )
            return;
          const data = new DataView(bytes);
          const distances = new Float32Array(bytes, 0, subset.stationRows.length);
          const nearestPositions = new Float32Array(
            bytes,
            subset.stationRows.length * 4,
            subset.stationRows.length * 2
          );
          const row = Math.max(0, Math.min(selectedStation, subset.stationRows.length - 1));
          const distance = distances[row];
          const nearestX = nearestPositions[row * 2];
          const nearestY = nearestPositions[row * 2 + 1];
          const lineIndex = data.getUint32(subset.stationRows.length * 12 + row * 4, true);
          const segmentIndex = data.getUint32(subset.stationRows.length * 16 + row * 4, true);
          const pointDistanceStart = subset.stationRows.length * 20;
          const pointDistance = data.getFloat32(pointDistanceStart + row * 4, true);
          const footpointDistance = data.getFloat32(
            pointDistanceStart + subset.stationRows.length * 4 + row * 4,
            true
          );
          const haversine = data.getFloat32(
            pointDistanceStart + subset.stationRows.length * 8 + row * 4,
            true
          );
          const polygonStart = pointDistanceStart + subset.stationRows.length * 12;
          const classification = data.getUint32(polygonStart + row * 4, true);
          const queryStart = polygonStart + subset.stationRows.length * 4;
          const queryResultCount = data.getUint32(queryStart, true);
          const requiredCount = data.getUint32(queryStart + 4, true);
          const overflow = data.getUint32(queryStart + 8, true);
          const clampedQueryResultCount = Math.min(queryResultCount, subset.stationRows.length);
          const queryMatches = new Uint32Array(bytes, queryStart + 12, clampedQueryResultCount);
          const querySourceRows = Array.from(
            queryMatches.slice(0, 6),
            id => subset.stationRows[id]
          );
          const good = Number.isFinite(distance);
          variant.nearestFootpointCount = subset.stationRows.length;
          variant.queryMatchCount = overflow === 0 ? clampedQueryResultCount : 0;
          variant.footpointLinkCount = good ? 1 : 0;
          variant.selectedSegmentCount = 0;
          if (good)
            linkBuffer.write(
              Float32Array.of(
                subset.stationPositions[row * 2],
                subset.stationPositions[row * 2 + 1],
                nearestX,
                nearestY
              )
            );
          if (good && lineIndex !== 0xffffffff && segmentIndex !== 0xffffffff) {
            const globalLineIndex = row * parts.length + lineIndex;
            if (globalLineIndex < pairedLineOffsets.length - 1) {
              const firstVertex = pairedLineOffsets[globalLineIndex] + segmentIndex;
              if (firstVertex + 1 < pairedLineOffsets[globalLineIndex + 1]) {
                selectedSegmentBuffer.write(
                  Float32Array.of(
                    pairedPositions[firstVertex * 2],
                    pairedPositions[firstVertex * 2 + 1],
                    pairedPositions[(firstVertex + 1) * 2],
                    pairedPositions[(firstVertex + 1) * 2 + 1]
                  )
                );
                variant.selectedSegmentCount = 1;
              }
            }
          }
          if (lineIndex === 0xffffffff || segmentIndex === 0xffffffff) {
            ctx.setReadout('segmentDistanceCheck', 'No valid nearest segment for this station.');
          } else {
            ctx.setReadout('segmentDistanceCheck', 'Computing selected GPU nearest segment…');
            const segmentBuildIdentity = identity;
            const segmentBuildRow = row;
            const segmentBuildKey = getVariantKey(state);
            void segmentGraphSwap
              .request({
                build: () => buildSegmentDistance(row, lineIndex, segmentIndex, distance),
                commit: next => {
                  if (!next) return;
                  if (
                    destroyed ||
                    requestIdentity !== segmentBuildIdentity ||
                    selectedStation !== segmentBuildRow ||
                    variant?.summaryIdentity !== segmentBuildIdentity ||
                    getVariantKey(ctx.options) !== segmentBuildKey
                  ) {
                    next.reader.stop();
                    next.resources.destroy();
                    return;
                  }
                  const previous = segmentVariant;
                  segmentVariant = next;
                  if (previous) {
                    previous.reader.stop();
                    requestAnimationFrame(() =>
                      requestAnimationFrame(() => previous.resources.destroy())
                    );
                  }
                  ctx.requestLayers();
                },
                discard: next => {
                  next?.reader.stop();
                  next?.resources.destroy();
                }
              })
              .then(committed => {
                if (committed && segmentVariant) segmentVariant.reader.markStale();
              })
              .catch(() => {
                if (!destroyed && requestIdentity === identity)
                  ctx.setReadout(
                    'segmentDistanceCheck',
                    'Segment check could not be built; nearest-linestring result remains available.'
                  );
              });
          }
          const distancesValid = distances.filter(Number.isFinite);
          const max = Math.max(1, ...distancesValid);
          const histogram = new Array<number>(10).fill(0);
          for (const value of distancesValid)
            histogram[Math.min(9, Math.floor((value / max) * 10))]++;
          ctx.setChart('offsetDistribution', {
            kind: 'bars',
            values: histogram,
            labels: histogram.map(
              (_item, index) =>
                `${Math.round((index / 10) * max)}–${Math.round(((index + 1) / 10) * max)} m`
            ),
            highlight: good ? [Math.min(9, Math.floor((distance / max) * 10))] : []
          });
          ctx.setReadout(
            'stationDetail',
            `source row ${subset.stationRows[row]} · ${good ? `${distance.toFixed(1)} m` : 'no segment'}\nnearest part ${lineIndex === 0xffffffff ? 'none' : lineIndex} · segment ${segmentIndex === 0xffffffff ? 'none' : segmentIndex}\nfootpoint ${good ? `${nearestX.toFixed(1)}, ${nearestY.toFixed(1)} m` : 'unavailable'}`
          );
          ctx.setReadout(
            'routeCoverage',
            `${distancesValid.length} of ${subset.stationRows.length} GPU rows returned a valid nearest segment; ${subset.stationRows.length - distancesValid.length} had no segment. Source geometry is clipped to the city bbox + 1 km, so a finite result can be nearest to a clipped path end.`
          );
          ctx.setReadout(
            'referenceDistance',
            Number.isFinite(pointDistance)
              ? `${pointDistance.toFixed(1)} m to route reference point`
              : 'unavailable'
          );
          ctx.setReadout(
            'footpointDistanceCheck',
            Number.isFinite(footpointDistance)
              ? `${footpointDistance.toFixed(3)} m · nearest linestring ${distance.toFixed(3)} m`
              : 'unavailable'
          );
          ctx.setReadout(
            'haversineDistance',
            Number.isFinite(haversine)
              ? state.haversineUnits === 'meters'
                ? `${haversine.toFixed(1)} m great-circle to the same reference`
                : `${haversine.toFixed(6)} km great-circle to the same reference (${(haversine * 1000).toFixed(1)} m normalized)`
              : 'unavailable'
          );
          ctx.setReadout(
            'polygonStatus',
            classification === GPU_POINT_IN_POLYGON_CLASSIFICATION.inside
              ? 'inside source-anchored synthetic ±400 m workframe'
              : classification === GPU_POINT_IN_POLYGON_CLASSIFICATION.boundary
                ? 'boundary'
                : classification === GPU_POINT_IN_POLYGON_CLASSIFICATION.uncertain
                  ? 'uncertain'
                  : 'outside source-anchored synthetic ±400 m workframe'
          );
          ctx.setReadout(
            'queryStatus',
            `${queryResultCount} returned · ${requiredCount} required · overflow ${overflow ? 'yes' : 'no'}\nsource station rows ${querySourceRows.join(', ') || 'none'}${overflow ? '\nMap marks hidden because the result overflowed.' : ''}`
          );
          ctx.setReadout(
            'selectionStatus',
            `route subset row ${row + 1} of ${subset.stationRows.length} · source row ${subset.stationRows[row]}`
          );
          fitSelectedEvidence(
            {...state, framing: ctx.options.framing},
            row,
            nearestX,
            nearestY,
            good
          );
          ctx.setHighlight(
            good
              ? {
                  kind: 'point',
                  coordinate: projection.unproject(
                    subset.stationPositions[row * 2],
                    subset.stationPositions[row * 2 + 1]
                  ),
                  radiusPixels: 14,
                  tone: 'signal',
                  pulse: true
                }
              : null
          );
          variant.chartDistances = Array.from(distancesValid);
          ctx.requestLayers();
        }
      );
      liveSummaryIdentity = identity;
      return {
        key: getVariantKey(state),
        resources: localResources,
        compiled,
        reader,
        stations: subset.stationPositions,
        referencePositions: referenceBufferData,
        sourceRows: subset.stationRows,
        routeName: subset.routeName,
        routeColor,
        nearest: nearestBuffer,
        nearestFootpointCount: 0,
        distance: distanceBuffer,
        queryIds,
        queryCount,
        queryOverflow,
        category: polygonClassBuffer,
        chartDistances: [],
        summaryIdentity: identity,
        stationBuffer,
        routeSegments: routeSegmentBuffer,
        routeSegmentCount,
        footpointLinks: linkBuffer,
        footpointLinkCount: 0,
        selectedSegment: selectedSegmentBuffer,
        selectedSegmentCount: 0,
        workframeSegments,
        workframeSegmentCount: 4,
        querySupport,
        querySupportSegmentCount: querySupportSegments.length / 4,
        queryMatchCount: 0
      };
    } catch (error) {
      localResources.destroy();
      throw error;
    }
  }

  variant = await build(ctx.options, ++requestIdentity);
  selectedStation = Math.max(0, Math.min(variant.sourceRows.length - 1, ctx.options.sample));
  if (selectedStation !== ctx.options.sample) ctx.setOptions({sample: selectedStation});
  const selectedStationId = resources.createBuffer('selected-station-id', Uint32Array.of(0));
  selectedStationId.write(Uint32Array.of(selectedStation));
  let dirty = true;
  const SETTLE_FRAMES = 3;

  function clearSegmentVariant(): void {
    const previousSegment = segmentVariant;
    if (!previousSegment) return;
    segmentVariant = null;
    previousSegment.reader.stop();
    requestAnimationFrame(() => requestAnimationFrame(() => previousSegment.resources.destroy()));
  }

  function requestRoute(state: RouteFootpointsOptions): void {
    const identity = ++requestIdentity;
    clearSegmentVariant();
    clearSummaries('Rebuilding the requested contributor variant…');
    void graphSwap
      .request({
        build: () => build(state, identity),
        commit: next => {
          const previous = variant;
          variant = next;
          dirty = true;
          selectedStation = 0;
          selectedStationId.write(Uint32Array.of(0));
          ctx.setOptions({sample: 0});
          ctx.requestLayers();
          clearSegmentVariant();
          if (previous) {
            previous.reader.stop();
            requestAnimationFrame(() => requestAnimationFrame(() => previous.resources.destroy()));
          }
        },
        discard: next => {
          next.reader.stop();
          next.resources.destroy();
        }
      })
      .catch(() => {
        if (!destroyed && identity === requestIdentity)
          ctx.setReadout('stationDetail', 'Variant build failed; the prior graph remains active.');
      });
  }

  function clearSummaries(message: string): void {
    ctx.setReadout('stationDetail', message);
    ctx.setReadout('selectionStatus', 'Waiting for current route selection…');
    ctx.setReadout('segmentDistanceCheck', 'Waiting for current selected segment…');
    ctx.setReadout('footpointDistanceCheck', 'Waiting for current route output…');
    ctx.setReadout('routeCoverage', 'Waiting for current route output…');
    ctx.setReadout('referenceDistance', 'Waiting for current route output…');
    ctx.setReadout('haversineDistance', 'Waiting for current route output…');
    ctx.setReadout('polygonStatus', 'Waiting for current route output…');
    ctx.setReadout('queryStatus', 'Waiting for current route output…');
    ctx.setChart('offsetDistribution', null);
    ctx.setHighlight(null);
    variant.queryMatchCount = 0;
    variant.footpointLinkCount = 0;
    variant.selectedSegmentCount = 0;
    variant.footpointLinks.write(new Float32Array(4));
    variant.selectedSegment.write(new Float32Array(4));
  }

  function getLayers(): Layer[] {
    if (
      destroyed ||
      variant.summaryIdentity !== requestIdentity ||
      variant.key !== getVariantKey(ctx.options)
    )
      return [];
    return [
      new SpatialAnalysisSegmentLayer({
        id: 'cta-route-shapes',
        segments: variant.routeSegments,
        instanceCount: variant.routeSegmentCount,
        coordinateOrigin,
        widthPixels: 1.8,
        color: variant.routeColor,
        colormap: 'uniform'
      }),
      new SpatialAnalysisSegmentLayer({
        id: 'synthetic-workframe',
        segments: variant.workframeSegments,
        instanceCount: variant.workframeSegmentCount,
        coordinateOrigin,
        widthPixels: 1.25,
        color: [95, 109, 118, 190],
        colormap: 'uniform',
        dashArray: [4, 3],
        cap: 'butt'
      }),
      new SpatialAnalysisSegmentLayer({
        id: 'spatial-query-support',
        segments: variant.querySupport,
        instanceCount: variant.querySupportSegmentCount,
        coordinateOrigin,
        widthPixels: 1.1,
        color: [71, 143, 177, 145],
        colormap: 'uniform',
        dashArray: [5, 4],
        cap: 'butt'
      }),
      new SpatialAnalysisPointLayer({
        id: 'route-stations',
        positions: variant.stationBuffer,
        instanceCount: variant.sourceRows.length,
        values: variant.category,
        valueFormat: 'uint32',
        colormap: 'category',
        palette: [
          [133, 143, 150, 220],
          [39, 153, 194, 255],
          [248, 190, 76, 255],
          [166, 112, 190, 255]
        ],
        coordinateOrigin,
        radiusPixels: 3.6
      }),
      new SpatialAnalysisPointLayer({
        id: 'gpu-query-matches',
        positions: variant.stationBuffer,
        ids: variant.queryIds,
        instanceCount: variant.queryMatchCount,
        coordinateOrigin,
        shape: 'ring',
        radiusPixels: 7,
        outlineWidthPixels: 2,
        color: [211, 55, 142, 255],
        colormap: 'uniform'
      }),
      new SpatialAnalysisSegmentLayer({
        id: 'gpu-returned-nearest-segment',
        segments: variant.selectedSegment,
        instanceCount: variant.selectedSegmentCount,
        coordinateOrigin,
        widthPixels: 4,
        color: [35, 143, 183, 255],
        colormap: 'uniform',
        cap: 'round'
      }),
      new SpatialAnalysisPointLayer({
        id: 'gpu-nearest-footpoints',
        positions: variant.nearest,
        instanceCount: variant.nearestFootpointCount,
        coordinateOrigin,
        radiusPixels: 4,
        color: [39, 153, 194, 255],
        colormap: 'uniform'
      }),
      new SpatialAnalysisSegmentLayer({
        id: 'selected-footpoint-link',
        segments: variant.footpointLinks,
        instanceCount: variant.footpointLinkCount,
        coordinateOrigin,
        widthPixels: 2,
        color: [248, 158, 76, 230],
        colormap: 'uniform',
        cap: 'round'
      }),
      new SpatialAnalysisPointLayer({
        id: 'selected-route-station',
        positions: variant.stationBuffer,
        ids: selectedStationId,
        instanceCount: variant.sourceRows.length > 0 ? 1 : 0,
        coordinateOrigin,
        radiusPixels: 7,
        color: [248, 158, 76, 255],
        colormap: 'uniform',
        outlineColor: [255, 255, 255, 255]
      })
    ];
  }

  return {
    getCompiledGraphs: () => [
      variant.compiled,
      ...(segmentVariant ? [segmentVariant.compiled] : [])
    ],
    getLayers,
    encode(commandEncoder, frame) {
      variant.compiled.encode(commandEncoder, {parameters: undefined});
      segmentVariant?.compiled.encode(commandEncoder, {parameters: undefined});
      segmentVariant?.reader.flush(commandEncoder);
      if (dirty && ++frameCounter >= SETTLE_FRAMES) {
        variant.reader.request(commandEncoder);
        dirty = false;
        frameCounter = 0;
      } else variant.reader.flush(commandEncoder);
    },
    setOption(id, _value, state) {
      if (id === 'route' && getVariantKey(state) !== variant.key) requestRoute(state);
      if (id === 'sample') {
        selectedStation =
          variant.sourceRows.length > 0
            ? Math.max(0, Math.min(variant.sourceRows.length - 1, state.sample))
            : 0;
        if (selectedStation !== state.sample) ctx.setOptions({sample: selectedStation});
        selectedStationId.write(Uint32Array.of(selectedStation));
        dirty = true;
        clearSegmentVariant();
        clearSummaries('Selection changed; waiting for the current GPU output…');
        ctx.refreshTooltip();
      }
      if (id === 'framing') {
        if (state.framing === 'overview') {
          ctx.flyTo({longitude: -87.66, latitude: 41.87, zoom: 10.4}, {transitionMs: 900});
        } else {
          dirty = true;
        }
      }
      if (id === 'queryRadius' || id === 'query' || id === 'haversineUnits') requestRoute(state);
      ctx.requestLayers();
    },
    onClick(event) {
      if (!event.coordinate) return false;
      const [positionX, positionY] = projection.project(event.coordinate[0], event.coordinate[1]);
      let nearest = -1;
      let best = 140;
      for (let row = 0; row < variant.stations.length / 2; row++) {
        const distance = Math.hypot(
          positionX - variant.stations[row * 2],
          positionY - variant.stations[row * 2 + 1]
        );
        if (distance < best) {
          best = distance;
          nearest = row;
        }
      }
      if (nearest >= 0) {
        selectedStation = nearest;
        selectedStationId.write(Uint32Array.of(selectedStation));
        dirty = true;
        clearSegmentVariant();
        ctx.setOptions({sample: nearest});
        clearSummaries('Selection changed; waiting for the current GPU output…');
        ctx.refreshTooltip();
        return true;
      }
      return false;
    },
    getTooltip(event) {
      if (!event.coordinate) return null;
      const [positionX, positionY] = projection.project(event.coordinate[0], event.coordinate[1]);
      let nearest = -1;
      let best = 140;
      for (let row = 0; row < variant.stations.length / 2; row++) {
        const distance = Math.hypot(
          positionX - variant.stations[row * 2],
          positionY - variant.stations[row * 2 + 1]
        );
        if (distance < best) {
          best = distance;
          nearest = row;
        }
      }
      return nearest < 0
        ? null
        : `CTA parent-merged rail station · source stop row ${variant.sourceRows[nearest]} · ${variant.routeName}`;
    },
    destroy() {
      destroyed = true;
      variant.reader.stop();
      segmentVariant?.reader.stop();
      segmentGraphSwap.destroy();
      resources.destroy();
      variant.resources.destroy();
      segmentVariant?.resources.destroy();
    }
  };
}
