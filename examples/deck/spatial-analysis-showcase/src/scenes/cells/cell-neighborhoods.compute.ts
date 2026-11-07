// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {PathLayer, PolygonLayer, ScatterplotLayer, TextLayer} from '@deck.gl/layers';
import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {cellToBoundary, cellToLatLng} from 'h3-js';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUCellGridPath,
  GPUCellMeasures,
  GPUH3CellIdentities,
  GPUPointToCell
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GraphSwap} from '../../engine/graph-swap';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {LineChartData} from '../chart-types';
import type {SceneContext, SceneInstance} from '../scene';
import {loadAirportTable} from '../flows/b11-geography';
import {
  findOpposedEdgePair,
  keyToHex,
  makeVertexPoints,
  sharedVertexKeys,
  toCellIndex
} from './cell-neighborhoods-geography';
import type {CellNeighborhoodOptions} from './cell-neighborhoods.types';
import type {ScenePointerEvent} from '../../engine/types';

type Airport = {
  iata: string;
  name: string;
  city: string;
  longitude: number;
  latitude: number;
};
type CellPolygon = {polygon: [number, number][][]; index: number};
type AirportMark = {
  code: string;
  name: string;
  city: string;
  coordinates: [number, number];
};
type Settings = Pick<CellNeighborhoodOptions, 'destination' | 'resolution' | 'pathCapacity'>;
type CellNeighborhoodGraph = {
  resources: SpatialAnalysisResources;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  outputs: {
    originKey: Buffer;
    originValidity: Buffer;
    destinationKey: Buffer;
    destinationValidity: Buffer;
    distance: Buffer;
    pathCount: Buffer;
    pathCells: Buffer;
    areas: Buffer;
    perimeters: Buffer;
    edgeCounts: Buffer;
    pentagons: Buffer;
    directedEdges: Buffer;
    vertices: Buffer;
  };
  settings: Settings;
  capacity: number;
};
type CellNeighborhoodResult = {
  originKey: Uint32Array;
  originValidity: Uint32Array;
  destinationKey: Uint32Array;
  destinationValidity: Uint32Array;
  distance: number;
  pathCount: number;
  pathCells: Uint32Array;
  areas: Float32Array;
  perimeters: Float32Array;
  edgeCounts: Uint32Array;
  pentagons: Uint32Array;
  directedEdges: Uint32Array;
  vertices: Uint32Array;
};

const ORIGIN_CODE = 'ORD';
const UNDEFINED_DISTANCE = 0xffffffff;
const CELL_COLOR = [63, 164, 194, 160] as const;
const CELL_LINE_COLOR = [87, 224, 222, 230] as const;
const PATH_COLOR = [255, 194, 103, 255] as const;
const AIRPORT_COLORS: Record<string, [number, number, number, number]> = {
  ORD: [245, 222, 120, 255],
  destination: [240, 134, 99, 255]
};

function readResult(capacity: number, bytes: ArrayBuffer): CellNeighborhoodResult {
  let byteOffset = 0;
  const readUint32 = (length: number) => {
    const values = new Uint32Array(bytes, byteOffset, length);
    byteOffset += length * Uint32Array.BYTES_PER_ELEMENT;
    return values;
  };
  const readFloat32 = (length: number) => {
    const values = new Float32Array(bytes, byteOffset, length);
    byteOffset += length * Float32Array.BYTES_PER_ELEMENT;
    return values;
  };
  const originKey = readUint32(2);
  const originValidity = readUint32(1);
  const destinationKey = readUint32(2);
  const destinationValidity = readUint32(1);
  const distance = readUint32(1)[0];
  const pathCount = readUint32(1)[0];
  const pathCells = readUint32(capacity * 2);
  const areas = readFloat32(capacity);
  const perimeters = readFloat32(capacity);
  const edgeCounts = readUint32(capacity);
  const pentagons = readUint32(capacity);
  const directedEdges = readUint32(capacity * 6 * 2);
  const vertices = readUint32(capacity * 6 * 2);
  return {
    originKey,
    originValidity,
    destinationKey,
    destinationValidity,
    distance,
    pathCount,
    pathCells,
    areas,
    perimeters,
    edgeCounts,
    pentagons,
    directedEdges,
    vertices
  };
}

function makeCellPath(cellKeys: Uint32Array, storedCount: number): [number, number][][][] {
  const polygons: [number, number][][][] = [];
  for (let row = 0; row < storedCount; row++) {
    const cell = keyToHex(cellKeys, row);
    if (!cell) continue;
    try {
      const boundary = cellToBoundary(cell);
      polygons.push([
        boundary.map(([latitude, longitude]) => [longitude, latitude] as [number, number])
      ]);
    } catch {
      // GPU keys are untrusted display input; a failed decoder hides only this outline.
    }
  }
  return polygons;
}

function makeCellCenters(cellKeys: Uint32Array, storedCount: number): [number, number][] {
  const centers: [number, number][] = [];
  for (let row = 0; row < storedCount; row++) {
    const cell = keyToHex(cellKeys, row);
    if (!cell) continue;
    try {
      const [latitude, longitude] = cellToLatLng(cell);
      centers.push([longitude, latitude]);
    } catch {
      // A malformed GPU key has no display center.
    }
  }
  return centers;
}

function getPathStatus(result: CellNeighborhoodResult, capacity: number): string {
  if (!result.originValidity[0] || !result.destinationValidity[0])
    return 'An endpoint did not receive a valid H3 cell key.';
  if (result.distance === UNDEFINED_DISTANCE)
    return 'H3 grid distance is undefined for this endpoint pair; this says nothing about whether a journey exists.';
  if (result.pathCount === 0)
    return 'Grid distance is defined, but this pair produced no local IJ path; no line is drawn.';
  if (result.pathCount > capacity)
    return `Path truncated: showing the stored prefix (${capacity} of ${result.pathCount} cells); its last cell is not the destination.`;
  return `Complete H3 grid path: ${result.pathCount} cells for ${result.distance} adjacent-cell hops.`;
}

export async function createCellNeighborhoods(
  ctx: SceneContext<CellNeighborhoodOptions>
): Promise<SceneInstance<CellNeighborhoodOptions>> {
  const dataset = ctx.datasets.get('openflights');
  const positions = dataset.column<Float32Array>('locations');
  const airportRecords = await loadAirportTable('openflights', ctx.signal);
  const airports: Airport[] = airportRecords.map((record, index) => ({
    ...record,
    longitude: positions[index * 2],
    latitude: positions[index * 2 + 1]
  }));
  const airportByCode = new Map(airports.map(airport => [airport.iata, airport]));
  const originAirport = airportByCode.get(ORIGIN_CODE);
  if (!originAirport) throw new Error('OpenFlights source is missing ORD airport coordinates.');
  const baseResources = new SpatialAnalysisResources(ctx.device, 'cell-neighborhoods');
  const graphSwap = baseResources.track(new GraphSwap());
  let activeGraph: CellNeighborhoodGraph;
  let graphDirty = true;
  let latestResult: CellNeighborhoodResult | null = null;
  let latestSettings: Settings = {
    destination: ctx.options.destination,
    resolution: ctx.options.resolution,
    pathCapacity: ctx.options.pathCapacity
  };
  const destinationView = (destination: string) => {
    if (destination === 'MDW') return {longitude: -87.75, latitude: 41.78, zoom: 10.1};
    if (destination === 'JFK') return {longitude: -80.5, latitude: 40.8, zoom: 4.8};
    if (destination === 'LAX') return {longitude: -103, latitude: 38, zoom: 3.4};
    return {longitude: 151.2, latitude: -33.9, zoom: 5.2};
  };
  const getBoundaryView = (settings: Settings, result: CellNeighborhoodResult | null) => {
    const centers = result
      ? makeCellCenters(
          result.pathCells,
          Math.min(
            result.pathCount,
            settings.pathCapacity === '16' ? 16 : settings.pathCapacity === '64' ? 64 : 256
          )
        )
      : [];
    const longitude =
      centers.length > 1 ? (centers[0][0] + centers[1][0]) / 2 : originAirport.longitude;
    const latitude =
      centers.length > 1 ? (centers[0][1] + centers[1][1]) / 2 : originAirport.latitude;
    const zoomByResolution: Record<string, number> = {'3': 6.2, '4': 7.6, '5': 9};
    return {longitude, latitude, zoom: zoomByResolution[settings.resolution] ?? 7.6};
  };

  const compileGraph = async (settings: Settings): Promise<CellNeighborhoodGraph> => {
    const destinationAirport = airportByCode.get(settings.destination);
    if (!destinationAirport)
      throw new Error(`OpenFlights source is missing ${settings.destination}.`);
    const resolution = Number(settings.resolution);
    const capacity = Number(settings.pathCapacity);
    const owned = new SpatialAnalysisResources(
      ctx.device,
      `cell-neighborhoods-r${resolution}-${capacity}-${settings.destination}`
    );
    const originPosition = owned.createBuffer(
      'origin-longitude-latitude',
      Float32Array.of(originAirport.longitude, originAirport.latitude)
    );
    const destinationPosition = owned.createBuffer(
      'destination-longitude-latitude',
      Float32Array.of(destinationAirport.longitude, destinationAirport.latitude)
    );
    const originKey = owned.createBuffer('origin-h3-key', 8);
    const originValidity = owned.createBuffer('origin-h3-validity', 4);
    const destinationKey = owned.createBuffer('destination-h3-key', 8);
    const destinationValidity = owned.createBuffer('destination-h3-validity', 4);
    const distance = owned.createBuffer('grid-distance', 4);
    const pathCount = owned.createBuffer('path-required-cell-count', 4);
    const pathCells = owned.createBuffer('bounded-path-cells', capacity * 8);
    const areas = owned.createBuffer('path-cell-areas-km2', capacity * 4);
    const perimeters = owned.createBuffer('path-cell-perimeters-km', capacity * 4);
    const edgeCounts = owned.createBuffer('path-cell-edge-counts', capacity * 4);
    const pentagons = owned.createBuffer('path-cell-pentagon-flags', capacity * 4);
    const directedEdges = owned.createBuffer('path-directed-edge-identities', capacity * 6 * 8);
    const vertices = owned.createBuffer('path-canonical-vertex-identities', capacity * 6 * 8);
    const graph = new GPUCommandGraph<void>(ctx.device, {
      id: `cell-neighborhoods-${settings.destination}-r${resolution}-capacity${capacity}`
    });
    const originPositionView = importGraphBuffer(
      graph,
      'origin-position',
      originPosition,
      'float32x2',
      1
    );
    const destinationPositionView = importGraphBuffer(
      graph,
      'destination-position',
      destinationPosition,
      'float32x2',
      1
    );
    const originKeyView = importGraphBuffer(graph, 'origin-key', originKey, 'uint32x2', 1);
    const originValidityView = importGraphBuffer(
      graph,
      'origin-validity',
      originValidity,
      'uint32',
      1
    );
    const destinationKeyView = importGraphBuffer(
      graph,
      'destination-key',
      destinationKey,
      'uint32x2',
      1
    );
    const destinationValidityView = importGraphBuffer(
      graph,
      'destination-validity',
      destinationValidity,
      'uint32',
      1
    );
    const distanceView = importGraphBuffer(graph, 'grid-distance', distance, 'uint32', 1);
    const pathCountView = importGraphBuffer(graph, 'path-count', pathCount, 'uint32', 1);
    const pathCellsView = importGraphBuffer(graph, 'path-cells', pathCells, 'uint32x2', capacity);
    const areaView = importGraphBuffer(graph, 'path-areas', areas, 'float32', capacity);
    const perimeterView = importGraphBuffer(
      graph,
      'path-perimeters',
      perimeters,
      'float32',
      capacity
    );
    const edgeCountView = importGraphBuffer(
      graph,
      'path-edge-counts',
      edgeCounts,
      'uint32',
      capacity
    );
    const pentagonView = importGraphBuffer(graph, 'path-pentagons', pentagons, 'uint32', capacity);
    const directedEdgesView = importGraphBuffer(
      graph,
      'path-directed-edges',
      directedEdges,
      'uint32x2',
      capacity * 6
    );
    const verticesView = importGraphBuffer(
      graph,
      'path-canonical-vertices',
      vertices,
      'uint32x2',
      capacity * 6
    );
    graph.add(
      new GPUPointToCell({
        id: 'index-ORD',
        family: 'h3',
        resolution,
        positions: originPositionView,
        output: {cells: originKeyView, validity: originValidityView}
      })
    );
    graph.add(
      new GPUPointToCell({
        id: `index-${settings.destination}`,
        family: 'h3',
        resolution,
        positions: destinationPositionView,
        output: {cells: destinationKeyView, validity: destinationValidityView}
      })
    );
    graph.add(
      new GPUCellGridPath({
        id: 'airport-grid-path',
        family: 'h3',
        origins: originKeyView,
        destinations: destinationKeyView,
        maximumPathLength: capacity,
        output: {distances: distanceView, cells: pathCellsView, counts: pathCountView}
      })
    );
    graph.add(
      new GPUCellMeasures({
        id: 'path-cell-measures',
        family: 'h3',
        units: 'kilometers',
        maximumEdgeCount: 10,
        cells: pathCellsView,
        output: {
          areas: areaView,
          perimeters: perimeterView,
          edgeCounts: edgeCountView,
          pentagons: pentagonView
        }
      })
    );
    graph.add(
      new GPUH3CellIdentities({
        id: 'path-directed-edge-ids',
        operation: 'directed-edges',
        cells: pathCellsView,
        output: directedEdgesView
      })
    );
    graph.add(
      new GPUH3CellIdentities({
        id: 'path-canonical-vertex-ids',
        operation: 'vertices',
        cells: pathCellsView,
        output: verticesView
      })
    );
    const compiled = owned.track(await graph.compileAsync());
    const outputs = {
      originKey,
      originValidity,
      destinationKey,
      destinationValidity,
      distance,
      pathCount,
      pathCells,
      areas,
      perimeters,
      edgeCounts,
      pentagons,
      directedEdges,
      vertices
    };
    const makeReader = (onResult: (bytes: ArrayBuffer) => void) =>
      new SummaryReader(
        owned,
        'cell-neighborhoods-summary',
        [
          {buffer: originKey, size: 8},
          {buffer: originValidity, size: 4},
          {buffer: destinationKey, size: 8},
          {buffer: destinationValidity, size: 4},
          {buffer: distance, size: 4},
          {buffer: pathCount, size: 4},
          {buffer: pathCells, size: capacity * 8},
          {buffer: areas, size: capacity * 4},
          {buffer: perimeters, size: capacity * 4},
          {buffer: edgeCounts, size: capacity * 4},
          {buffer: pentagons, size: capacity * 4},
          {buffer: directedEdges, size: capacity * 6 * 8},
          {buffer: vertices, size: capacity * 6 * 8}
        ],
        onResult
      );
    const reader = makeReader(bytes => {
      const result = readResult(capacity, bytes);
      latestResult = result;
      publishResult(activeGraph, result);
    });
    return {resources: owned, compiled, reader, outputs, settings, capacity};
  };

  const getDestination = (settings: Settings) => {
    const airport = airportByCode.get(settings.destination);
    if (!airport) throw new Error(`OpenFlights source is missing ${settings.destination}.`);
    return airport;
  };
  const publishResult = (graph: CellNeighborhoodGraph, result: CellNeighborhoodResult) => {
    const destinationAirport = getDestination(graph.settings);
    const storedCount = Math.min(result.pathCount, graph.capacity);
    const originKey = keyToHex(result.originKey, 0);
    const destinationKey = keyToHex(result.destinationKey, 0);
    const status = getPathStatus(result, graph.capacity);
    ctx.setReadout('originKey', toCellIndex(originKey));
    ctx.setReadout('destinationKey', toCellIndex(destinationKey));
    ctx.setReadout(
      'gridDistance',
      result.distance === UNDEFINED_DISTANCE ? 'Undefined' : result.distance
    );
    ctx.setReadout('requiredCells', result.pathCount);
    ctx.setReadout('storedCells', storedCount);
    ctx.setReadout('pathStatus', status);
    ctx.setReadout(
      'pentagonCells',
      result.pentagons.slice(0, storedCount).reduce((sum, value) => sum + (value ? 1 : 0), 0)
    );
    ctx.setReadout('firstCellArea', storedCount ? result.areas[0] : null);
    ctx.setReadout('lastCellArea', storedCount ? result.areas[storedCount - 1] : null);
    ctx.setReadout('firstCellPerimeter', storedCount ? result.perimeters[0] : null);
    const areas = Array.from(result.areas.slice(0, storedCount)).filter(Number.isFinite);
    if (areas.length) {
      ctx.setChart('cellAreaChart', {
        kind: 'line',
        title: 'Measured area of each stored path cell',
        xLabel: 'H3 path cell index',
        yLabel: 'spherical area (km²)',
        series: [
          {
            label: 'Cell area',
            x: Array.from({length: areas.length}, (_, index) => index),
            y: areas,
            color: 1,
            points: areas.length <= 32
          }
        ],
        description:
          'GPUCellMeasures area for each GPU-produced path cell, in path order. A truncated result shows only the stored prefix.'
      } satisfies LineChartData);
    } else ctx.setChart('cellAreaChart', null);
    const firstPairAvailable = storedCount > 1;
    const shared = firstPairAvailable ? sharedVertexKeys(result.vertices, 0, 1) : [];
    ctx.setReadout('sharedVertices', firstPairAvailable ? shared.length : 'Not available');
    ctx.setReadout(
      'sharedVertexKeys',
      shared.length ? shared.map(key => `0x${key}`).join(', ') : 'No consecutive pair'
    );
    const firstCellKey = keyToHex(result.pathCells, 0);
    const secondCellKey = keyToHex(result.pathCells, 1);
    ctx.setReadout(
      'opposedDirectedEdges',
      firstPairAvailable
        ? findOpposedEdgePair(result.directedEdges, firstCellKey, secondCellKey)
        : 'No consecutive pair'
    );
    const firstCellEdges: string[] = [];
    for (let slot = 0; slot < 6; slot++) {
      const edgeKey = keyToHex(result.directedEdges, slot);
      if (edgeKey) firstCellEdges.push(`0x${edgeKey}`);
    }
    ctx.setReadout(
      'firstCellDirectedEdges',
      firstCellEdges.length ? firstCellEdges.join(' · ') : 'No valid cell edge keys'
    );
    ctx.setReadout(
      'airportPair',
      `${ORIGIN_CODE} ${originAirport.city} → ${destinationAirport.iata} ${destinationAirport.city}`
    );
    latestSettings = graph.settings;
    if (ctx.options.focus === 'boundary') {
      ctx.flyTo(getBoundaryView(graph.settings, result), {
        transitionMs: ctx.reducedMotion() ? 0 : 800
      });
    }
    ctx.requestLayers();
  };
  const releaseGraph = (graph: CellNeighborhoodGraph) => {
    graph.reader.release();
    graph.resources.destroy();
  };
  const requestGraph = (settings: Settings) => {
    ctx.setReadout('pathStatus', 'Rebuilding path graph…');
    void graphSwap
      .request({
        build: () => compileGraph(settings),
        commit: next => {
          const previous = activeGraph;
          activeGraph = next;
          latestSettings = next.settings;
          latestResult = null;
          graphDirty = true;
          next.reader.markStale();
          for (const id of [
            'gridDistance',
            'requiredCells',
            'storedCells',
            'firstCellArea',
            'lastCellArea',
            'firstCellPerimeter',
            'sharedVertices',
            'sharedVertexKeys',
            'firstCellDirectedEdges',
            'opposedDirectedEdges'
          ])
            ctx.setReadout(id, 'Recomputing');
          ctx.setChart('cellAreaChart', null);
          releaseGraph(previous);
          ctx.flyTo(
            ctx.options.focus === 'boundary'
              ? getBoundaryView(next.settings, null)
              : destinationView(next.settings.destination),
            {
              transitionMs: ctx.reducedMotion() ? 0 : 800
            }
          );
          ctx.requestLayers();
        },
        discard: releaseGraph
      })
      .catch(error => ctx.setStatus(`Cell path graph failed: ${String(error)}`));
  };

  activeGraph = await compileGraph(latestSettings);
  graphDirty = true;
  activeGraph.reader.markStale();
  return {
    getCompiledGraphs: () => [activeGraph.compiled],
    setOption(id, _value, state) {
      if (id === 'destination' || id === 'resolution' || id === 'pathCapacity') {
        requestGraph({
          destination: state.destination,
          resolution: state.resolution,
          pathCapacity: state.pathCapacity
        });
      } else if (id === 'focus') {
        ctx.flyTo(
          state.focus === 'boundary'
            ? getBoundaryView(latestSettings, latestResult)
            : destinationView(latestSettings.destination),
          {
            transitionMs: ctx.reducedMotion() ? 0 : 800
          }
        );
        ctx.requestLayers();
      }
    },
    encode(commandEncoder) {
      if (graphDirty) {
        activeGraph.compiled.encode(commandEncoder, {parameters: undefined});
        activeGraph.reader.request(commandEncoder);
        graphDirty = false;
      } else activeGraph.reader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      const destinationAirport = getDestination(latestSettings);
      const result = latestResult;
      const storedCount = result
        ? Math.min(result.pathCount, Number(latestSettings.pathCapacity))
        : 0;
      const cells = result ? makeCellPath(result.pathCells, storedCount) : [];
      const centers = result ? makeCellCenters(result.pathCells, storedCount) : [];
      const boundaryFocus = ctx.options.focus === 'boundary';
      const firstPairCells = boundaryFocus
        ? cells.slice(0, 2).map((polygon, index): CellPolygon => ({polygon, index}))
        : [];
      const shared = result && storedCount > 1 ? sharedVertexKeys(result.vertices, 0, 1) : [];
      const sharedVertices = boundaryFocus ? makeVertexPoints(shared) : [];
      const endpoints: AirportMark[] = [originAirport, destinationAirport].map(airport => ({
        code: airport.iata,
        name: airport.name,
        city: airport.city,
        coordinates: [airport.longitude, airport.latitude] as [number, number]
      }));
      const layers: Layer[] = [
        new PolygonLayer({
          id: 'h3-path-cells',
          data: cells.map((polygon, index): CellPolygon => ({polygon, index})),
          getPolygon: (item: CellPolygon) => item.polygon,
          getFillColor: CELL_COLOR,
          getLineColor: CELL_LINE_COLOR,
          lineWidthUnits: 'pixels',
          getLineWidth: 1,
          lineWidthMinPixels: 0.7,
          stroked: true,
          filled: true,
          pickable: false,
          parameters: {depthCompare: 'always', depthWriteEnabled: false}
        }),
        new PathLayer({
          id: 'h3-cell-center-path',
          data: centers.length > 1 ? [{path: centers}] : [],
          getPath: (item: {path: [number, number][]}) => item.path,
          getColor: PATH_COLOR,
          getWidth: 3,
          widthUnits: 'pixels',
          widthMinPixels: 2,
          capRounded: true,
          jointRounded: true,
          pickable: false,
          parameters: {depthCompare: 'always', depthWriteEnabled: false}
        }),
        new ScatterplotLayer({
          id: 'airport-endpoints',
          data: endpoints,
          getPosition: (item: AirportMark) => item.coordinates,
          getRadius: 7,
          radiusUnits: 'pixels',
          radiusMinPixels: 6,
          getFillColor: (item: AirportMark) =>
            item.code === 'ORD' ? AIRPORT_COLORS.ORD : AIRPORT_COLORS.destination,
          getLineColor: [20, 27, 37, 255],
          lineWidthMinPixels: 1.5,
          stroked: true,
          pickable: true
        }),
        new TextLayer({
          id: 'airport-labels',
          data: endpoints,
          getPosition: (item: AirportMark) => item.coordinates,
          getText: (item: AirportMark) => `${item.code} - ${item.city}`,
          getColor: [250, 247, 236, 255],
          getSize: 13,
          sizeUnits: 'pixels',
          sizeMinPixels: 12,
          getPixelOffset: [12, -10],
          getTextAnchor: 'start',
          getAlignmentBaseline: 'center',
          fontFamily: 'sans-serif',
          characterSet: 'auto',
          fontSettings: {sdf: true},
          outlineColor: [18, 24, 32, 230],
          outlineWidth: 2,
          pickable: false,
          parameters: {depthCompare: 'always', depthWriteEnabled: false}
        })
      ];
      if (firstPairCells.length) {
        layers.push(
          new PolygonLayer({
            id: 'h3-first-adjacent-pair',
            data: firstPairCells,
            getPolygon: (item: CellPolygon) => item.polygon,
            getFillColor: [242, 130, 74, 80],
            getLineColor: [255, 210, 150, 255],
            getLineWidth: 3,
            lineWidthUnits: 'pixels',
            lineWidthMinPixels: 2,
            stroked: true,
            filled: true,
            pickable: false,
            parameters: {depthCompare: 'always', depthWriteEnabled: false}
          })
        );
        layers.push(
          new ScatterplotLayer({
            id: 'h3-shared-canonical-vertices',
            data: sharedVertices,
            getPosition: (position: [number, number]) => position,
            getRadius: 5,
            radiusUnits: 'pixels',
            radiusMinPixels: 5,
            getFillColor: [255, 247, 161, 255],
            getLineColor: [28, 30, 34, 255],
            lineWidthMinPixels: 1.5,
            stroked: true,
            pickable: false,
            parameters: {depthCompare: 'always', depthWriteEnabled: false}
          })
        );
      }
      return layers;
    },
    getTooltip(event: ScenePointerEvent) {
      const picked = event.info.object as AirportMark | undefined;
      return picked ? `${picked.code} · ${picked.name}` : null;
    },
    destroy() {
      graphSwap.destroy();
      releaseGraph(activeGraph);
      baseResources.destroy();
    }
  };
}
