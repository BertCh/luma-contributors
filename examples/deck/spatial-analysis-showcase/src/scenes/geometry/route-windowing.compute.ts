// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PathLayer, PolygonLayer, TextLayer} from '@deck.gl/layers';
import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  GPULineClipByPolygon,
  GPUMinimumBounds,
  GPUSubdividePolygons
} from '@luma-contributors/graph/gpu-spatial-analysis';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {LocalMetricProjection, projectLngLatArray} from '../../engine/projection';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import type {RouteWindowingOptions} from './route-windowing.types';

const RED_ROUTE_INDEX = 124;
const RED_SHAPE_ROW = 16958;
const LOOP_FEATURE_ROW = 31;
const SOURCE_COLOR: [number, number, number, number] = [145, 162, 173, 190];
const LOOP_COLOR: [number, number, number, number] = [53, 98, 121, 75];
const CLIP_COLOR: [number, number, number, number] = [248, 170, 87, 245];
const SUBDIVISION_COLORS: [number, number, number, number][] = [
  [54, 149, 159, 150],
  [237, 161, 90, 155],
  [103, 127, 193, 155],
  [187, 115, 135, 155]
];
const ROUTE_VIEW = {longitude: -87.65, latitude: 41.88, zoom: 10.2};
const LOOP_VIEW = {longitude: -87.6298, latitude: 41.8781, zoom: 12.4};
const INTERSECTION_CAPACITY = 256;
const CLIP_VERTEX_CAPACITY = 96 + 2 * INTERSECTION_CAPACITY;
const CLIP_PIECE_CAPACITY = 96 + 4 * INTERSECTION_CAPACITY;
const SUBDIVISION_PIECE_CAPACITY = 256;
const SUBDIVISION_MAXIMUM_VERTICES = 16;

type AreaRecord = {id?: number; name?: string};
type ClipBuffers = {
  positions: Buffer;
  offsets: Buffer;
  count: Buffer;
  vertexCount: Buffer;
  overflow: Buffer;
  required: Buffer;
  uncertain: Buffer;
  candidateOverflow: Buffer;
};
type WindowGraph = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  minimumCorners: Buffer;
  minimumSizes: Buffer;
  longestLine: Buffer;
  diameter: Buffer;
  hullSize: Buffer;
  minimumOverflow: Buffer;
  subdivisionPositions: Buffer;
  subdivisionPieces: Buffer;
  subdivisionStatus: Buffer;
  rectilinearPositions: Buffer;
  rectilinearPieces: Buffer;
  rectilinearStatus: Buffer;
  inside: ClipBuffers;
  outside: ClipBuffers;
};
type ClipResult = {
  positions: Float32Array;
  offsets: Uint32Array;
  count: number;
  vertexCount: number;
  overflow: number;
  required: number;
  uncertain: number;
  candidateOverflow: number;
};
type WindowResult = {
  corners: Float32Array;
  sizes: Float32Array;
  longestLine: Float32Array;
  diameter: number;
  hullSize: number;
  minimumOverflow: number;
  subdivisionPositions: Float32Array;
  subdivisionPieces: Uint32Array;
  subdivisionStatus: Uint32Array;
  rectilinearPositions: Float32Array;
  rectilinearPieces: Uint32Array;
  rectilinearStatus: Uint32Array;
  inside: ClipResult;
  outside: ClipResult;
};

function readFloat32(view: DataView, cursor: {value: number}, count: number): Float32Array {
  const values = new Float32Array(count);
  for (let index = 0; index < count; index++)
    values[index] = view.getFloat32(cursor.value + index * 4, true);
  cursor.value += count * 4;
  return values;
}

function readUint32(view: DataView, cursor: {value: number}, count: number): Uint32Array {
  const values = new Uint32Array(count);
  for (let index = 0; index < count; index++)
    values[index] = view.getUint32(cursor.value + index * 4, true);
  cursor.value += count * 4;
  return values;
}

function readClip(view: DataView, cursor: {value: number}): ClipResult {
  const positions = readFloat32(view, cursor, CLIP_VERTEX_CAPACITY * 2);
  const offsets = readUint32(view, cursor, CLIP_PIECE_CAPACITY + 1);
  const [count, vertexCount, overflow, required, uncertain, candidateOverflow] = readUint32(
    view,
    cursor,
    6
  );
  return {positions, offsets, count, vertexCount, overflow, required, uncertain, candidateOverflow};
}

function getWindowResult(
  graph: WindowGraph,
  bytes: ArrayBuffer,
  method: 'convex' | 'rectilinear'
): WindowResult {
  const view = new DataView(bytes);
  const cursor = {value: 0};
  const corners = readFloat32(view, cursor, 8);
  const sizes = readFloat32(view, cursor, 4);
  const longestLine = readFloat32(view, cursor, 4);
  const diameter = readFloat32(view, cursor, 1)[0];
  const hullSize = readUint32(view, cursor, 1)[0];
  const minimumOverflow = readUint32(view, cursor, 1)[0];
  const subdivisionPositions = readFloat32(
    view,
    cursor,
    SUBDIVISION_PIECE_CAPACITY * SUBDIVISION_MAXIMUM_VERTICES * 2
  );
  const subdivisionPieces = readUint32(view, cursor, SUBDIVISION_PIECE_CAPACITY * 4);
  const subdivisionStatus = readUint32(view, cursor, 4);
  return {
    corners,
    sizes,
    longestLine,
    diameter,
    hullSize,
    minimumOverflow,
    subdivisionPositions: method === 'convex' ? subdivisionPositions : new Float32Array(0),
    subdivisionPieces: method === 'convex' ? subdivisionPieces : new Uint32Array(0),
    subdivisionStatus: method === 'convex' ? subdivisionStatus : new Uint32Array(4),
    rectilinearPositions: method === 'rectilinear' ? subdivisionPositions : new Float32Array(0),
    rectilinearPieces: method === 'rectilinear' ? subdivisionPieces : new Uint32Array(0),
    rectilinearStatus: method === 'rectilinear' ? subdivisionStatus : new Uint32Array(4),
    inside: readClip(view, cursor),
    outside: readClip(view, cursor)
  };
}

function makePath(
  local: Float32Array,
  projection: LocalMetricProjection,
  count: number
): [number, number][] {
  const path: [number, number][] = [];
  for (let index = 0; index < count; index++)
    path.push(projection.unproject(local[index * 2], local[index * 2 + 1]));
  return path;
}

function getClipPaths(result: ClipResult, projection: LocalMetricProjection): [number, number][][] {
  const paths: [number, number][][] = [];
  for (let piece = 0; piece < Math.min(result.count, result.offsets.length - 1); piece++) {
    const start = result.offsets[piece];
    const end = result.offsets[piece + 1];
    if (end > start && end <= result.vertexCount)
      paths.push(makePath(result.positions.subarray(start * 2, end * 2), projection, end - start));
  }
  return paths;
}

function getSubdivisionPolygons(
  result: WindowResult,
  projection: LocalMetricProjection,
  method: 'convex' | 'rectilinear'
): [number, number][][] {
  const polygons: [number, number][][] = [];
  const positions = method === 'convex' ? result.subdivisionPositions : result.rectilinearPositions;
  const pieces = method === 'convex' ? result.subdivisionPieces : result.rectilinearPieces;
  const status = method === 'convex' ? result.subdivisionStatus : result.rectilinearStatus;
  if (status[2] !== 0 || status[3] !== 0) return polygons;
  const count = Math.min(status[0], SUBDIVISION_PIECE_CAPACITY);
  for (let piece = 0; piece < count; piece++) {
    const vertexCount = Math.min(pieces[piece * 4], SUBDIVISION_MAXIMUM_VERTICES);
    const start = piece * SUBDIVISION_MAXIMUM_VERTICES * 2;
    if (vertexCount >= 3)
      polygons.push(
        makePath(positions.subarray(start, start + vertexCount * 2), projection, vertexCount)
      );
  }
  return polygons;
}

function getLocalLineLength(
  positions: Float32Array,
  startVertex: number,
  endVertex: number
): number {
  let total = 0;
  for (let vertex = startVertex + 1; vertex < endVertex; vertex++) {
    total += Math.hypot(
      positions[vertex * 2] - positions[(vertex - 1) * 2],
      positions[vertex * 2 + 1] - positions[(vertex - 1) * 2 + 1]
    );
  }
  return total;
}

function getLocalPolygonArea(positions: Float32Array, vertexCount: number): number {
  let twiceArea = 0;
  for (let vertex = 0; vertex < vertexCount; vertex++) {
    const next = (vertex + 1) % vertexCount;
    twiceArea +=
      positions[vertex * 2] * positions[next * 2 + 1] -
      positions[next * 2] * positions[vertex * 2 + 1];
  }
  return Math.abs(twiceArea) / 2;
}

function getClipLocalLength(result: ClipResult): number {
  let total = 0;
  for (let piece = 0; piece < Math.min(result.count, result.offsets.length - 1); piece++) {
    const start = result.offsets[piece];
    const end = result.offsets[piece + 1];
    if (end > start && end <= result.vertexCount)
      total += getLocalLineLength(result.positions, start, end);
  }
  return total;
}

function getSubdivisionLocalArea(result: WindowResult, method: 'convex' | 'rectilinear'): number {
  let total = 0;
  const positions = method === 'convex' ? result.subdivisionPositions : result.rectilinearPositions;
  const pieces = method === 'convex' ? result.subdivisionPieces : result.rectilinearPieces;
  const status = method === 'convex' ? result.subdivisionStatus : result.rectilinearStatus;
  if (status[2] !== 0 || status[3] !== 0) return NaN;
  for (let piece = 0; piece < Math.min(status[0], SUBDIVISION_PIECE_CAPACITY); piece++) {
    const vertexCount = Math.min(pieces[piece * 4], SUBDIVISION_MAXIMUM_VERTICES);
    if (vertexCount < 3) continue;
    const start = piece * SUBDIVISION_MAXIMUM_VERTICES * 2;
    total += getLocalPolygonArea(positions.subarray(start, start + vertexCount * 2), vertexCount);
  }
  return total;
}

export async function createRouteWindowing(
  ctx: SceneContext<RouteWindowingOptions>
): Promise<SceneInstance<RouteWindowingOptions>> {
  const cta = ctx.datasets.get('cta-transit');
  const areaDataset = ctx.datasets.get('chicago-community-areas');
  const routeRecords = cta.properties.routes as {id: string; name: string}[];
  if (routeRecords[RED_ROUTE_INDEX]?.id !== 'Red')
    throw new Error('CTA Red Line source row 124 is unavailable.');
  const shapeOffsets = cta.column<Uint32Array>('shapePathOffsets');
  const shapeRoute = cta.column<Uint16Array>('shapeRoute');
  if (shapeRoute[RED_SHAPE_ROW] !== RED_ROUTE_INDEX)
    throw new Error('CTA Red Line source shape row 16958 is unavailable.');
  const sourceVertices = cta.column<Float32Array>('shapeVertices');
  const sourceStart = shapeOffsets[RED_SHAPE_ROW];
  const sourceEnd = shapeOffsets[RED_SHAPE_ROW + 1];
  const sourceLngLat = sourceVertices.slice(sourceStart * 2, sourceEnd * 2);
  const projection = new LocalMetricProjection([-87.6298, 41.8781]);
  const sourceLocal = projectLngLatArray(projection, sourceLngLat);
  const areas = areaDataset.geojson?.features ?? [];
  const loopFeature = areas[LOOP_FEATURE_ROW];
  const loopProperties = loopFeature?.properties as AreaRecord | undefined;
  if (loopProperties?.id !== 32 || loopProperties.name !== 'Loop')
    throw new Error('Chicago community area row 31 no longer identifies the Loop.');
  const verticesLngLat = areaDataset.column<Float32Array>('vertices');
  const ringOffsets = areaDataset.column<Uint32Array>('ringOffsets');
  const polygonRingOffsets = areaDataset.column<Uint32Array>('polygonRingOffsets');
  const partFeature = areaDataset.column<Uint32Array>('partFeature');
  let loopPart = -1;
  for (let part = 0; part < partFeature.length; part++)
    if (partFeature[part] === LOOP_FEATURE_ROW) {
      loopPart = part;
      break;
    }
  if (loopPart < 0) throw new Error('Loop polygon part is unavailable.');
  const firstRing = polygonRingOffsets[loopPart];
  const lastRing = polygonRingOffsets[loopPart + 1];
  if (lastRing - firstRing !== 1) throw new Error('Loop polygon no longer has one exterior ring.');
  const firstVertex = ringOffsets[firstRing];
  let lastVertex = ringOffsets[firstRing + 1];
  if (
    lastVertex - firstVertex > 1 &&
    verticesLngLat[firstVertex * 2] === verticesLngLat[(lastVertex - 1) * 2] &&
    verticesLngLat[firstVertex * 2 + 1] === verticesLngLat[(lastVertex - 1) * 2 + 1]
  )
    lastVertex--;
  const loopLngLat = verticesLngLat.slice(firstVertex * 2, lastVertex * 2);
  const loopLocal = projectLngLatArray(projection, loopLngLat);
  const sourceOffsetsData = Uint32Array.of(0, sourceLocal.length / 2);
  const polygonRingOffsetsData = Uint32Array.of(0, loopLocal.length / 2);
  const polygonOffsetsData = Uint32Array.of(0, 1);
  const featureOffsetsData = Uint32Array.of(0, 1);
  const resources = new SpatialAnalysisResources(ctx.device, 'route-windowing');
  const sourceBuffer = resources.createBuffer('red-line-local', sourceLocal);
  const sourceOffsetsBuffer = resources.createBuffer('red-line-offsets', sourceOffsetsData);
  const labelsBuffer = resources.createBuffer(
    'route-labels',
    new Uint32Array(sourceLocal.length / 2)
  );
  const loopBuffer = resources.createBuffer('loop-local', loopLocal);
  const ringBuffer = resources.createBuffer('loop-ring-offsets', polygonRingOffsetsData);
  const polygonBuffer = resources.createBuffer('loop-polygon-offsets', polygonOffsetsData);
  const featureBuffer = resources.createBuffer('loop-feature-offsets', featureOffsetsData);
  const bounds = resources.createBuffer('minimum-bounds', 16);
  const rectangleCorners = resources.createBuffer('minimum-rectangle-corners', 32);
  const rectangleSizes = resources.createBuffer('minimum-rectangle-sizes', 16);
  const longestLine = resources.createBuffer('minimum-longest-line', 16);
  const diameter = resources.createBuffer('minimum-diameter', 4);
  const hullSize = resources.createBuffer('minimum-hull-size', 4);
  const minimumOverflow = resources.createBuffer('minimum-overflow', 4);
  const subdivisionPositions = resources.createBuffer(
    'loop-subdivision-positions',
    SUBDIVISION_PIECE_CAPACITY * SUBDIVISION_MAXIMUM_VERTICES * 8
  );
  const subdivisionPieces = resources.createBuffer(
    'loop-subdivision-pieces',
    SUBDIVISION_PIECE_CAPACITY * 16
  );
  const subdivisionStatus = resources.createBuffer('loop-subdivision-status', 16);
  const rectilinearPositions = resources.createBuffer(
    'loop-rectilinear-positions',
    SUBDIVISION_PIECE_CAPACITY * SUBDIVISION_MAXIMUM_VERTICES * 8
  );
  const rectilinearPieces = resources.createBuffer(
    'loop-rectilinear-pieces',
    SUBDIVISION_PIECE_CAPACITY * 16
  );
  const rectilinearStatus = resources.createBuffer('loop-rectilinear-status', 16);
  const makeClipBuffers = (prefix: string): ClipBuffers => ({
    positions: resources.createBuffer(`${prefix}-positions`, CLIP_VERTEX_CAPACITY * 8),
    offsets: resources.createBuffer(`${prefix}-offsets`, (CLIP_PIECE_CAPACITY + 1) * 4),
    count: resources.createBuffer(`${prefix}-count`, 4),
    vertexCount: resources.createBuffer(`${prefix}-vertex-count`, 4),
    overflow: resources.createBuffer(`${prefix}-overflow`, 4),
    required: resources.createBuffer(`${prefix}-required`, 4),
    uncertain: resources.createBuffer(`${prefix}-uncertain`, 4),
    candidateOverflow: resources.createBuffer(`${prefix}-candidate-overflow`, 4)
  });
  const inside = makeClipBuffers('inside-loop');
  const outside = makeClipBuffers('outside-loop');
  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'route-windowing-analysis'});
  const imp = <Format extends 'float32x2' | 'uint32' | 'uint32x4' | 'float32x4' | 'float32'>(
    name: string,
    buffer: Buffer,
    format: Format,
    rows: number
  ) => importGraphBuffer(graph, name, buffer, format, rows);
  const routePositions = imp('red-line-local', sourceBuffer, 'float32x2', sourceLocal.length / 2);
  const routeOffsets = imp('red-line-offsets', sourceOffsetsBuffer, 'uint32', 2);
  const routeLabels = imp('route-labels', labelsBuffer, 'uint32', sourceLocal.length / 2);
  const loopPositions = imp('loop-local', loopBuffer, 'float32x2', loopLocal.length / 2);
  const loopRingOffsets = imp('loop-ring-offsets', ringBuffer, 'uint32', 2);
  const loopPolygonOffsets = imp('loop-polygon-offsets', polygonBuffer, 'uint32', 2);
  const loopFeatureOffsets = imp('loop-feature-offsets', featureBuffer, 'uint32', 2);
  const loopPolygons = {
    kind: 'polygons' as const,
    positions: loopPositions,
    ringOffsets: loopRingOffsets,
    polygonOffsets: loopPolygonOffsets,
    featureOffsets: loopFeatureOffsets
  };
  graph.add(
    new GPUMinimumBounds({
      id: 'red-route-minimum-bounds',
      spatialContext: projection.spatialContext,
      positions: routePositions,
      labels: routeLabels,
      groupCount: 1,
      maximumHullVertices: 256,
      output: {
        bounds: imp('minimum-bounds', bounds, 'float32x4', 1),
        rectangleCorners: imp('minimum-rectangle-corners', rectangleCorners, 'float32x2', 4),
        rectangleSizes: imp('minimum-rectangle-sizes', rectangleSizes, 'float32x4', 1),
        longestLines: imp('minimum-longest-line', longestLine, 'float32x4', 1),
        diameters: imp('minimum-diameter', diameter, 'float32', 1),
        hullSizes: imp('minimum-hull-size', hullSize, 'uint32', 1),
        status: {overflow: imp('minimum-overflow', minimumOverflow, 'uint32', 1)}
      }
    })
  );
  graph.add(
    new GPUSubdividePolygons({
      id: 'loop-convex-subdivision',
      spatialContext: projection.spatialContext,
      positions: loopPositions,
      ringOffsets: loopRingOffsets,
      polygonOffsets: loopPolygonOffsets,
      featureOffsets: loopFeatureOffsets,
      maximumVertices: SUBDIVISION_MAXIMUM_VERTICES,
      maximumWork: 65536,
      output: {
        positions: imp(
          'loop-subdivision-positions',
          subdivisionPositions,
          'float32x2',
          SUBDIVISION_PIECE_CAPACITY * SUBDIVISION_MAXIMUM_VERTICES
        ),
        pieces: imp(
          'loop-subdivision-pieces',
          subdivisionPieces,
          'uint32x4',
          SUBDIVISION_PIECE_CAPACITY
        ),
        status: imp('loop-subdivision-status', subdivisionStatus, 'uint32x4', 1)
      }
    })
  );
  graph.add(
    new GPUSubdividePolygons({
      id: 'loop-rectilinear-subdivision',
      spatialContext: projection.spatialContext,
      positions: loopPositions,
      ringOffsets: loopRingOffsets,
      polygonOffsets: loopPolygonOffsets,
      featureOffsets: loopFeatureOffsets,
      method: 'rectilinear',
      maximumSplits: 256,
      maximumVertices: SUBDIVISION_MAXIMUM_VERTICES,
      maximumWork: 65536,
      output: {
        positions: imp(
          'loop-rectilinear-positions',
          rectilinearPositions,
          'float32x2',
          SUBDIVISION_PIECE_CAPACITY * SUBDIVISION_MAXIMUM_VERTICES
        ),
        pieces: imp(
          'loop-rectilinear-pieces',
          rectilinearPieces,
          'uint32x4',
          SUBDIVISION_PIECE_CAPACITY
        ),
        status: imp('loop-rectilinear-status', rectilinearStatus, 'uint32x4', 1)
      }
    })
  );
  for (const [mode, output] of [
    ['inside', inside],
    ['outside', outside]
  ] as const) {
    graph.add(
      new GPULineClipByPolygon({
        id: `red-line-${mode}-loop`,
        spatialContext: projection.spatialContext,
        lines: {kind: 'lines', positions: routePositions, lineOffsets: routeOffsets},
        polygons: loopPolygons,
        mode,
        intersectionCapacity: INTERSECTION_CAPACITY,
        candidateCapacity: 2048,
        output: {
          pieces: {
            geometry: {
              kind: 'lines',
              positions: imp(
                `${mode}-positions`,
                output.positions,
                'float32x2',
                CLIP_VERTEX_CAPACITY
              ),
              lineOffsets: imp(`${mode}-offsets`, output.offsets, 'uint32', CLIP_PIECE_CAPACITY + 1)
            },
            sourceIds: imp(
              `${mode}-source-ids`,
              resources.createBuffer(`${mode}-source-ids`, CLIP_PIECE_CAPACITY * 4),
              'uint32',
              CLIP_PIECE_CAPACITY
            ),
            status: {
              count: imp(`${mode}-count`, output.count, 'uint32', 1),
              overflow: imp(`${mode}-overflow`, output.overflow, 'uint32', 1),
              requiredCount: imp(`${mode}-required`, output.required, 'uint32', 1),
              vertexCount: imp(`${mode}-vertex-count`, output.vertexCount, 'uint32', 1),
              uncertainCount: imp(`${mode}-uncertain`, output.uncertain, 'uint32', 1),
              candidateOverflow: imp(
                `${mode}-candidate-overflow`,
                output.candidateOverflow,
                'uint32',
                1
              )
            }
          }
        }
      })
    );
  }
  const compiled = resources.track(await graph.compileAsync());
  let activeGraph: WindowGraph;
  let latestResult: WindowResult | null = null;
  let graphDirty = true;
  const commonSummarySources = [
    {buffer: rectangleCorners, size: 32},
    {buffer: rectangleSizes, size: 16},
    {buffer: longestLine, size: 16},
    {buffer: diameter, size: 4},
    {buffer: hullSize, size: 4},
    {buffer: minimumOverflow, size: 4}
  ];
  const clipSummarySources = [inside, outside].flatMap(output => [
    {buffer: output.positions, size: CLIP_VERTEX_CAPACITY * 8},
    {buffer: output.offsets, size: (CLIP_PIECE_CAPACITY + 1) * 4},
    {buffer: output.count, size: 4},
    {buffer: output.vertexCount, size: 4},
    {buffer: output.overflow, size: 4},
    {buffer: output.required, size: 4},
    {buffer: output.uncertain, size: 4},
    {buffer: output.candidateOverflow, size: 4}
  ]);
  const makeSummarySources = (method: 'convex' | 'rectilinear') => [
    ...commonSummarySources,
    {
      buffer: method === 'convex' ? subdivisionPositions : rectilinearPositions,
      size: SUBDIVISION_PIECE_CAPACITY * SUBDIVISION_MAXIMUM_VERTICES * 8
    },
    {
      buffer: method === 'convex' ? subdivisionPieces : rectilinearPieces,
      size: SUBDIVISION_PIECE_CAPACITY * 16
    },
    {buffer: method === 'convex' ? subdivisionStatus : rectilinearStatus, size: 16},
    ...clipSummarySources
  ];
  const makeReader = (method: 'convex' | 'rectilinear') =>
    new SummaryReader(
      resources,
      `route-windowing-summary-${method}`,
      makeSummarySources(method),
      bytes => {
        if (ctx.options.method !== method) return;
        const result = getWindowResult(activeGraph, bytes, method);
        latestResult = result;
        publishResult(result);
      }
    );
  const readers = {convex: makeReader('convex'), rectilinear: makeReader('rectilinear')};
  let selectedReader = readers[ctx.options.method];
  activeGraph = {
    resources,
    compiled,
    reader: selectedReader,
    minimumCorners: rectangleCorners,
    minimumSizes: rectangleSizes,
    longestLine,
    diameter,
    hullSize,
    minimumOverflow,
    subdivisionPositions,
    subdivisionPieces,
    subdivisionStatus,
    rectilinearPositions,
    rectilinearPieces,
    rectilinearStatus,
    inside,
    outside
  };
  const sourcePath = makePath(sourceLocal, projection, sourceLocal.length / 2);
  const loopPath = makePath(loopLocal, projection, loopLocal.length / 2);
  const sourceLoopArea = getLocalPolygonArea(loopLocal, loopLocal.length / 2);
  const sourceLength = getLocalLineLength(sourceLocal, 0, sourceLocal.length / 2);
  function publishResult(result: WindowResult): void {
    const method = ctx.options.method;
    const subdivisionStatus =
      method === 'convex' ? result.subdivisionStatus : result.rectilinearStatus;
    const subdivisionArea = getSubdivisionLocalArea(result, method);
    ctx.setReadout('sourceVertices', sourceLocal.length / 2);
    ctx.setReadout('routeLength', sourceLength / 1000);
    ctx.setReadout('routeDiameter', result.diameter / 1000);
    ctx.setReadout('routeHullVertices', result.hullSize);
    ctx.setReadout(
      'minimumBox',
      result.minimumOverflow
        ? 'Hull capacity exceeded'
        : `${(result.sizes[0] / 1000).toFixed(2)} × ${(result.sizes[1] / 1000).toFixed(2)} km`
    );
    ctx.setReadout('loopSourceArea', sourceLoopArea / 1e6);
    ctx.setReadout('subdivisionPieces', subdivisionStatus[0]);
    ctx.setReadout('subdivisionRequired', subdivisionStatus[1]);
    ctx.setReadout(
      'subdivisionArea',
      Number.isFinite(subdivisionArea) ? subdivisionArea / 1e6 : 'Incomplete output'
    );
    ctx.setReadout(
      'subdivisionAreaDifference',
      Number.isFinite(subdivisionArea)
        ? (subdivisionArea - sourceLoopArea) / 1e6
        : 'Incomplete output'
    );
    ctx.setReadout(
      'subdivisionStatus',
      `${method}; overflow ${subdivisionStatus[2]}, invalid/incomplete ${subdivisionStatus[3]}`
    );
    ctx.setReadout('insidePieces', result.inside.count);
    const insideComplete =
      result.inside.overflow === 0 &&
      result.inside.candidateOverflow === 0 &&
      result.inside.uncertain === 0;
    ctx.setReadout(
      'insideLength',
      insideComplete ? getClipLocalLength(result.inside) / 1000 : 'Incomplete: inspect clip status'
    );
    ctx.setReadout(
      'insideStatus',
      `overflow ${result.inside.overflow}; uncertain ${result.inside.uncertain}; candidate overflow ${result.inside.candidateOverflow}`
    );
    ctx.setReadout('outsidePieces', result.outside.count);
    const outsideComplete =
      result.outside.overflow === 0 &&
      result.outside.candidateOverflow === 0 &&
      result.outside.uncertain === 0;
    ctx.setReadout(
      'outsideLength',
      outsideComplete
        ? getClipLocalLength(result.outside) / 1000
        : 'Incomplete: inspect clip status'
    );
    ctx.setReadout(
      'outsideStatus',
      `overflow ${result.outside.overflow}; uncertain ${result.outside.uncertain}; candidate overflow ${result.outside.candidateOverflow}`
    );
    ctx.requestLayers();
  }
  selectedReader.markStale();
  return {
    getCompiledGraphs: () => [compiled],
    setOption(id, _value, state) {
      if (id === 'focus')
        ctx.flyTo(state.focus === 'loop' ? LOOP_VIEW : ROUTE_VIEW, {
          transitionMs: ctx.reducedMotion() ? 0 : 700
        });
      if (id === 'method') {
        selectedReader = readers[state.method];
        activeGraph.reader = selectedReader;
        latestResult = null;
        selectedReader.markStale();
      }
      if (id === 'view' || id === 'method') {
        if (latestResult) publishResult(latestResult);
        ctx.requestLayers();
      }
    },
    encode(commandEncoder) {
      if (graphDirty) {
        compiled.encode(commandEncoder, {parameters: undefined});
        selectedReader.request(commandEncoder);
        graphDirty = false;
      } else selectedReader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      const result = latestResult;
      const visibleLoop = [{polygon: loopPath}];
      const layers: Layer[] = [
        new PolygonLayer({
          id: 'loop-window-mask',
          data: visibleLoop,
          getPolygon: item => item.polygon,
          getFillColor: LOOP_COLOR,
          getLineColor: [56, 97, 119, 210],
          lineWidthMinPixels: 1,
          stroked: true,
          filled: true,
          pickable: false
        }),
        new PathLayer({
          id: 'red-line-source',
          data: [{path: sourcePath}],
          getPath: item => item.path,
          getColor: SOURCE_COLOR,
          getWidth: 4,
          widthUnits: 'pixels',
          widthMinPixels: 2,
          pickable: false
        })
      ];
      if (ctx.options.view === 'bounds' && result && !result.minimumOverflow) {
        const rectangle: [number, number][] = [];
        for (let index = 0; index < 4; index++)
          rectangle.push(
            projection.unproject(result.corners[index * 2], result.corners[index * 2 + 1])
          );
        rectangle.push(rectangle[0]);
        const diameterPath: [number, number][] = [
          projection.unproject(result.longestLine[0], result.longestLine[1]),
          projection.unproject(result.longestLine[2], result.longestLine[3])
        ];
        layers.push(
          new PathLayer({
            id: 'minimum-rotated-box',
            data: [{path: rectangle}],
            getPath: item => item.path,
            getColor: CLIP_COLOR,
            getWidth: 5,
            widthUnits: 'pixels',
            widthMinPixels: 3,
            pickable: false
          })
        );
        layers.push(
          new PathLayer({
            id: 'route-diameter',
            data: [{path: diameterPath}],
            getPath: item => item.path,
            getColor: [239, 234, 213, 220],
            getWidth: 2,
            widthUnits: 'pixels',
            widthMinPixels: 1,
            pickable: false
          })
        );
      }
      if (ctx.options.view === 'subdivision' && result) {
        const method = ctx.options.method;
        const subdivisionStatus =
          method === 'convex' ? result.subdivisionStatus : result.rectilinearStatus;
        const polygons = getSubdivisionPolygons(result, projection, method).map(
          (polygon, index) => ({polygon, index})
        );
        if (subdivisionStatus[2] !== 0 || subdivisionStatus[3] !== 0) return layers;
        layers.push(
          new PolygonLayer({
            id: 'loop-convex-pieces',
            data: polygons,
            getPolygon: item => item.polygon,
            getFillColor: item => SUBDIVISION_COLORS[item.index % SUBDIVISION_COLORS.length],
            getLineColor: [30, 57, 68, 220],
            lineWidthMinPixels: 1,
            stroked: true,
            filled: true,
            pickable: false
          })
        );
      }
      if (result && (ctx.options.view === 'inside' || ctx.options.view === 'outside')) {
        const selected = ctx.options.view === 'inside' ? result.inside : result.outside;
        if (selected.overflow !== 0 || selected.candidateOverflow !== 0 || selected.uncertain !== 0)
          return layers;
        const paths = getClipPaths(selected, projection);
        layers.push(
          new PathLayer({
            id: 'red-line-clipped',
            data: paths.map(path => ({path})),
            getPath: item => item.path,
            getColor: CLIP_COLOR,
            getWidth: 7,
            widthUnits: 'pixels',
            widthMinPixels: 4,
            pickable: false
          })
        );
      }
      layers.push(
        new TextLayer({
          id: 'loop-place-label',
          data: [{position: [-87.6298, 41.8781], text: 'THE LOOP'}],
          getPosition: item => item.position,
          getText: item => item.text,
          getColor: [28, 49, 61, 255],
          getSize: 12,
          sizeUnits: 'pixels',
          sizeMinPixels: 11,
          characterSet: 'auto',
          fontSettings: {sdf: true},
          outlineColor: [246, 242, 229, 235],
          outlineWidth: 3,
          pickable: false
        })
      );
      return layers;
    },
    destroy() {
      readers.convex.release();
      readers.rectilinear.release();
      resources.destroy();
    }
  };
}
