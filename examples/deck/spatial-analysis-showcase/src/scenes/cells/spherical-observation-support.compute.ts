// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {PathLayer, ScatterplotLayer} from '@deck.gl/layers';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUCellGeometry,
  GPUPointToCell,
  GPUNeighborSearch,
  GPUS2CapCovering,
  GPUS2CellHierarchy,
  GPUS2CellMeasures,
  getGPUNeighborSearchParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {GPU_EARTH_MEAN_RADIUS_METERS} from '@luma-contributors/analysis/analysis-core';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisResources, formatCount} from '../../engine/resources';
import {GraphSwap} from '../../engine/graph-swap';
import {SummaryReader} from '../../engine/summary-reader';
import {geodesicCircle} from '../../cartography/reference-geometry';
import type {SceneContext, SceneInstance} from '../scene';
import {toRadians, toDegrees} from '@math.gl/core';

const EARTH_RADIUS_METERS = GPU_EARTH_MEAN_RADIUS_METERS;
const EVENT_SAMPLE_LIMIT = 2_048;

export type SphericalObservationOptions = {
  level: string;
  coverMode: 'exterior' | 'interior';
  radiusKm: number;
  center: 'Japan' | 'Türkiye' | 'Andes' | 'Sunda';
  showParents: boolean;
};

type Variant = {
  key: string;
  level: number;
  mode: 'exterior' | 'interior';
  graph: CompiledGPUCommandGraph<void>;
  resources: SpatialAnalysisResources;
  queryPosition: Buffer;
  neighborParameters: Buffer;
  centerPosition: Buffer;
  reader: {
    markArea: () => void;
    markCover: () => void;
    flush: (encoder: Parameters<SummaryReader['flush']>[0]) => void;
  };
  cap: Buffer;
  parentBoundaries: Buffer;
  parentVertexCounts: Buffer;
  coverBoundaries: Buffer;
  coverVertexCounts: Buffer;
  coverCounts: Buffer;
  coverCapacity: number;
  coverPaths: [number, number][][];
  parentPaths: [number, number][][];
  cachedReadouts: Record<string, string>;
  dirty: boolean;
  areaRead: boolean;
};

const CENTERS: Record<SphericalObservationOptions['center'], [number, number]> = {
  Japan: [140, 35],
  Türkiye: [38, 36],
  Andes: [-72, -24],
  Sunda: [107.5, -1]
};

function makeGeodesicCellPaths(
  boundaries: Float32Array,
  vertexCounts: Uint32Array,
  rows: number,
  originLongitude: number
): [number, number][][] {
  const paths: [number, number][][] = [];
  for (let row = 0; row < rows; row++) {
    const vertexCount = vertexCounts[row];
    if (vertexCount < 3) continue;
    const path: [number, number][] = [];
    const vertices = Array.from({length: vertexCount}, (_, vertex) => {
      const offset = (row * 4 + vertex) * 2;
      const longitude = toRadians(boundaries[offset]);
      const latitude = toRadians(boundaries[offset + 1]);
      const cosine = Math.cos(latitude);
      return [cosine * Math.cos(longitude), cosine * Math.sin(longitude), Math.sin(latitude)];
    });
    for (let edge = 0; edge < vertexCount; edge++) {
      const start = vertices[edge];
      const end = vertices[(edge + 1) % vertexCount];
      const dot = Math.max(
        -1,
        Math.min(1, start[0] * end[0] + start[1] * end[1] + start[2] * end[2])
      );
      const angle = Math.acos(dot);
      const sine = Math.sin(angle);
      for (let step = 0; step < 8; step++) {
        const fraction = step / 8;
        const firstWeight = sine < 1e-8 ? 1 - fraction : Math.sin((1 - fraction) * angle) / sine;
        const secondWeight = sine < 1e-8 ? fraction : Math.sin(fraction * angle) / sine;
        const positionX = firstWeight * start[0] + secondWeight * end[0];
        const positionY = firstWeight * start[1] + secondWeight * end[1];
        const positionZ = firstWeight * start[2] + secondWeight * end[2];
        let longitude = toDegrees(Math.atan2(positionY, positionX));
        while (longitude - originLongitude > 180) longitude -= 360;
        while (longitude - originLongitude < -180) longitude += 360;
        path.push([longitude, toDegrees(Math.atan2(positionZ, Math.hypot(positionX, positionY)))]);
      }
    }
    if (path.length) path.push(path[0]);
    paths.push(path);
  }
  return paths;
}

function makeCap(longitude: number, latitude: number, radiusKm: number): Float32Array {
  const longitudeRadians = toRadians(longitude);
  const latitudeRadians = toRadians(latitude);
  const cosineLatitude = Math.cos(latitudeRadians);
  return Float32Array.of(
    cosineLatitude * Math.cos(longitudeRadians),
    cosineLatitude * Math.sin(longitudeRadians),
    Math.sin(latitudeRadians),
    (radiusKm * 1000) / EARTH_RADIUS_METERS
  );
}

export async function createSphericalObservationSupport(
  ctx: SceneContext<SphericalObservationOptions>
): Promise<SceneInstance<SphericalObservationOptions>> {
  const dataset = ctx.datasets.get('poopdeck-earthquakes');
  const allPositions = dataset.column<Float32Array>('position');
  const eventTypes = dataset.column<Uint8Array>('type');
  const kept: number[] = [];
  for (let index = 0; index < eventTypes.length; index++)
    if (eventTypes[index] === 0) kept.push(index);
  const eventCount = kept.length;
  const sampleCount = Math.min(EVENT_SAMPLE_LIMIT, eventCount);
  const sampledPositions = new Float32Array(sampleCount * 2);
  for (let index = 0; index < sampleCount; index++) {
    const sourceIndex = kept[Math.floor((index * eventCount) / sampleCount)];
    sampledPositions.set(allPositions.subarray(sourceIndex * 2, sourceIndex * 2 + 2), index * 2);
  }

  const resources = new SpatialAnalysisResources(ctx.device, 'spherical-observation-support');
  const graphSwap = resources.track(new GraphSwap());
  const eventPositions = new Float32Array(eventCount * 2);
  for (let index = 0; index < eventCount; index++) {
    eventPositions[index * 2] = allPositions[kept[index] * 2];
    eventPositions[index * 2 + 1] = allPositions[kept[index] * 2 + 1];
  }
  const positions = resources.createBuffer('all-earthquake-positions', eventPositions);
  const variants = new Map<string, Variant>();
  let active: Variant | null = null;
  let destroyed = false;
  let summaryRevision = 0;

  const makeVariant = async (level: number, mode: 'exterior' | 'interior'): Promise<Variant> => {
    const key = `${level}-${mode}`;
    const owned = new SpatialAnalysisResources(ctx.device, `spherical-observation-support-${key}`);
    const cap = owned.createBuffer(
      'cap-input',
      makeCap(...CENTERS[ctx.options.center], ctx.options.radiusKm)
    );
    const queryPosition = owned.createBuffer(
      'search-query-position',
      Float32Array.of(...CENTERS[ctx.options.center])
    );
    const neighborParameters = owned.createBuffer(
      'neighbor-parameters',
      getGPUNeighborSearchParameterValues({
        bounds: [-180, -90, 180, 90],
        radius: ctx.options.radiusKm * 1000,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'meters',
          sphereRadius: EARTH_RADIUS_METERS
        }
      })
    );
    const neighborOffsets = owned.createBuffer('neighbor-offsets', 8);
    const neighborRows = owned.createBuffer('neighbor-rows', eventCount * 4);
    const neighborWeights = owned.createBuffer('neighbor-weights', eventCount * 4);
    const neighborOverflow = owned.createBuffer('neighbor-overflow', 4);
    const neighborRequired = owned.createBuffer('neighbor-required', 4);
    const eventCells = owned.createBuffer('event-cells', eventCount * 8);
    const centerPosition = owned.createBuffer(
      'center-position',
      Float32Array.of(...CENTERS[ctx.options.center])
    );
    const centerCellBuffer = owned.createBuffer('center-cell', 8);
    const parentCells = owned.createBuffer('parent-cell', 8);
    const hierarchyStatus = owned.createBuffer('hierarchy-status', 4);
    const eventMeasures = owned.createBuffer('event-measures', eventCount * 8);
    const eventMeasureStatus = owned.createBuffer('event-measure-status', eventCount * 4);
    const parentBoundaries = owned.createBuffer('parent-boundaries', 4 * 8);
    const parentVertexCounts = owned.createBuffer('parent-vertex-counts', 4);
    const coverCapacity = 6 * 4 ** level;
    const coverCells = owned.createBuffer('cover-cells', coverCapacity * 8);
    const coverCounts = owned.createBuffer('cover-counts', 8);
    const coverStatus = owned.createBuffer('cover-status', 4);
    const coverMeasures = owned.createBuffer('cover-measures', coverCapacity * 8);
    const coverMeasureStatus = owned.createBuffer('cover-measure-status', coverCapacity * 4);
    const coverBoundaries = owned.createBuffer('cover-boundaries', coverCapacity * 4 * 8);
    const coverVertexCounts = owned.createBuffer('cover-vertex-counts', coverCapacity * 4);
    const graph = new GPUCommandGraph<void>(ctx.device, {id: `s2-observation-${key}`});
    const view = <Format extends Parameters<typeof importGraphBuffer>[3]>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => importGraphBuffer(graph, `import-${name}`, buffer, format, length);
    const allCells = view('all-cells', eventCells, 'uint32x2', eventCount);
    const allPositions = view('all-positions', positions, 'float32x2', eventCount);
    graph.add(
      new GPUPointToCell({
        family: 's2',
        resolution: level,
        positions: allPositions,
        output: {cells: allCells}
      })
    );
    graph.add(
      new GPUS2CellMeasures({
        id: 'event-cell-measures',
        cells: allCells,
        output: {
          measures: view('event-measures', eventMeasures, 'float32x2', eventCount),
          status: view('event-measure-status', eventMeasureStatus, 'uint32', eventCount)
        }
      })
    );
    const centerCell = view('center-cell', centerCellBuffer, 'uint32x2', 1);
    graph.add(
      new GPUPointToCell({
        id: 'center-point-to-cell',
        family: 's2',
        resolution: level,
        positions: view('center-position', centerPosition, 'float32x2', 1),
        output: {cells: centerCell}
      })
    );
    const parentCellsView = view('parent-cells', parentCells, 'uint32x2', 1);
    graph.add(
      new GPUS2CellHierarchy({
        operation: 'parent',
        level: Math.max(0, level - 1),
        cells: centerCell,
        output: {
          cells: parentCellsView,
          status: view('hierarchy-status', hierarchyStatus, 'uint32', 1)
        }
      })
    );
    graph.add(
      new GPUCellGeometry({
        id: 'parent-cell-geometry',
        family: 's2',
        cells: parentCellsView,
        output: {
          boundaries: view('parent-boundaries', parentBoundaries, 'float32x2', 4),
          vertexCounts: view('parent-vertex-counts', parentVertexCounts, 'uint32', 1)
        }
      })
    );
    const capView = view('cap', cap, 'float32x4', 1);
    graph.add(
      new GPUNeighborSearch({
        id: 'earthquakes-within-cap',
        mode: 'radius',
        positions: allPositions,
        queryPositions: view('query-position', queryPosition, 'float32x2', 1),
        parameters: view('neighbor-parameters', neighborParameters, 'float32', 12),
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'meters',
          sphereRadius: EARTH_RADIUS_METERS
        },
        gridSize: [180, 90],
        output: {
          offsets: view('neighbor-offsets', neighborOffsets, 'uint32', 2),
          neighbors: view('neighbor-rows', neighborRows, 'uint32', eventCount),
          weights: view('neighbor-weights', neighborWeights, 'float32', eventCount),
          status: {
            overflow: view('neighbor-overflow', neighborOverflow, 'uint32', 1),
            requiredCount: view('neighbor-required', neighborRequired, 'uint32', 1)
          }
        }
      })
    );
    const coverCellsView = view('cover-cells', coverCells, 'uint32x2', coverCapacity);
    graph.add(
      new GPUS2CapCovering({
        level,
        mode,
        capacity: coverCapacity,
        caps: capView,
        output: {
          cells: coverCellsView,
          counts: view('cover-counts', coverCounts, 'uint32x2', 1),
          status: view('cover-status', coverStatus, 'uint32', 1)
        }
      })
    );
    graph.add(
      new GPUS2CellMeasures({
        id: 'cover-cell-measures',
        cells: coverCellsView,
        output: {
          measures: view('cover-measures', coverMeasures, 'float32x2', coverCapacity),
          status: view('cover-measure-status', coverMeasureStatus, 'uint32', coverCapacity)
        }
      })
    );
    graph.add(
      new GPUCellGeometry({
        id: 'cover-cell-geometry',
        family: 's2',
        cells: coverCellsView,
        output: {
          boundaries: view('cover-boundaries', coverBoundaries, 'float32x2', coverCapacity * 4),
          vertexCounts: view('cover-vertex-counts', coverVertexCounts, 'uint32', coverCapacity)
        }
      })
    );
    let compiled: CompiledGPUCommandGraph<void>;
    try {
      compiled = owned.track(await graph.compileAsync());
    } catch (error) {
      owned.destroy();
      throw error;
    }
    let areaRevisionAtSubmission = 0;
    let coverRevisionAtSubmission = 0;
    let variant: Variant;
    const areaReader = new SummaryReader(
      owned,
      `event-area-${key}`,
      [
        {buffer: eventCells, size: eventCount * 8},
        {buffer: eventMeasures, size: eventCount * 8},
        {buffer: eventMeasureStatus, size: eventCount * 4}
      ],
      bytes => {
        if (destroyed || active?.key !== key || areaRevisionAtSubmission !== summaryRevision) {
          if (!destroyed) variant.areaRead = false;
          return;
        }
        const values = new DataView(bytes);
        let uniqueAreaSteradians = 0;
        let validRows = 0;
        const unique = new Map<string, number>();
        for (let index = 0; index < eventCount; index++) {
          const failure = values.getUint32(eventCount * 16 + index * 4, true);
          if (failure) continue;
          validRows++;
          const low = values.getUint32(index * 8, true);
          const high = values.getUint32(index * 8 + 4, true);
          const area = values.getFloat32(eventCount * 8 + index * 8, true);
          const cellKey = `${high}:${low}`;
          if (!unique.has(cellKey)) unique.set(cellKey, area);
        }
        for (const area of unique.values()) uniqueAreaSteradians += area;
        const squareRadius = EARTH_RADIUS_METERS ** 2;
        const uniqueAreaKm2 = (uniqueAreaSteradians * squareRadius) / 1e6;
        variant.cachedReadouts.events = `${formatCount(validRows)} valid event cells / ${formatCount(eventCount)} events`;
        variant.cachedReadouts.uniqueCells = `${formatCount(unique.size)} unique S2 cells`;
        variant.cachedReadouts.supportArea = `${formatCount(uniqueAreaKm2)} km² summed cell area`;
        variant.cachedReadouts.density = `${(validRows / Math.max(uniqueAreaKm2, 1e-9)).toExponential(2)} events / km²`;
        if (active?.key === key)
          for (const [id, value] of Object.entries(variant.cachedReadouts))
            ctx.setReadout(id, value);
      }
    );
    const coverReader = new SummaryReader(
      owned,
      `cap-area-${key}`,
      [
        {buffer: neighborOffsets, size: 8},
        {buffer: neighborOverflow, size: 4},
        {buffer: neighborRequired, size: 4},
        {buffer: coverCounts, size: 8},
        {buffer: coverStatus, size: 4},
        {buffer: coverMeasures, size: coverCapacity * 8},
        {buffer: coverMeasureStatus, size: coverCapacity * 4},
        {buffer: coverBoundaries, size: coverCapacity * 4 * 8},
        {buffer: coverVertexCounts, size: coverCapacity * 4},
        {buffer: parentBoundaries, size: 4 * 8},
        {buffer: parentVertexCounts, size: 4},
        {buffer: hierarchyStatus, size: 4}
      ],
      bytes => {
        if (destroyed || active?.key !== key || coverRevisionAtSubmission !== summaryRevision)
          return;
        const values = new DataView(bytes);
        const capEvents = values.getUint32(4, true);
        const neighborFailure = values.getUint32(8, true);
        const capEventRequirement = values.getUint32(12, true);
        const coverOffset = 16;
        const written = values.getUint32(coverOffset, true);
        const required = values.getUint32(coverOffset + 4, true);
        const coverFailure = values.getUint32(coverOffset + 8, true);
        let coverAreaSteradians = 0;
        const coverAreaOffset = coverOffset + 12;
        for (let index = 0; index < Math.min(written, coverCapacity); index++) {
          if (values.getUint32(coverAreaOffset + coverCapacity * 8 + index * 4, true) === 0) {
            coverAreaSteradians += values.getFloat32(coverAreaOffset + index * 8, true);
          }
        }
        const squareRadius = EARTH_RADIUS_METERS ** 2;
        const coverAreaKm2 = (coverAreaSteradians * squareRadius) / 1e6;
        const capAreaKm2 =
          (2 *
            Math.PI *
            squareRadius *
            (1 - Math.cos((ctx.options.radiusKm * 1000) / EARTH_RADIUS_METERS))) /
          1e6;
        variant.cachedReadouts.capEvents = `${formatCount(capEvents)} of ${formatCount(eventCount)} earthquakes within ${ctx.options.radiusKm} km`;
        variant.cachedReadouts.capDensity = `${(capEvents / Math.max(capAreaKm2, 1e-9)).toExponential(2)} events / km² of cap area`;
        variant.cachedReadouts.coverCells = `${formatCount(written)} cells (${formatCount(required)} required)`;
        variant.cachedReadouts.coverArea = `${formatCount(coverAreaKm2)} km² / ${formatCount(capAreaKm2)} km² cap`;
        variant.cachedReadouts.overcover = `${(coverAreaKm2 / Math.max(capAreaKm2, 1e-9)).toFixed(2)}× cap area`;
        variant.cachedReadouts.status =
          coverFailure || neighborFailure
            ? `GPU status: cover ${coverFailure}, search overflow ${neighborFailure}, required ${capEventRequirement}`
            : 'GPU cover and cap-event search complete';
        ctx.setChart('areaComparison', {
          kind: 'bars',
          values: [capAreaKm2, coverAreaKm2],
          labels: ['analytic cap', 'S2 cover'],
          height: 120,
          yLabel: 'area (km²)',
          formatY: value => formatCount(value),
          description:
            'Analytic area of the chosen spherical cap compared with the summed spherical areas returned by GPUS2CellMeasures for the GPU-selected cover cells.'
        });
        const coverBoundaryOffset = coverOffset + 12 + coverCapacity * 12;
        const coverBoundaries = new Float32Array(bytes, coverBoundaryOffset, coverCapacity * 8);
        const coverCountsOffset = coverBoundaryOffset + coverCapacity * 32;
        const coverVertexCounts = new Uint32Array(bytes, coverCountsOffset, coverCapacity);
        variant.coverPaths = makeGeodesicCellPaths(
          coverBoundaries,
          coverVertexCounts,
          written,
          CENTERS[ctx.options.center][0]
        );
        const parentBoundaryOffset = coverCountsOffset + coverCapacity * 4;
        const parentBoundaries = new Float32Array(bytes, parentBoundaryOffset, 8);
        const parentCountsOffset = parentBoundaryOffset + 32;
        const parentVertexCounts = new Uint32Array(bytes, parentCountsOffset, 1);
        variant.parentPaths = makeGeodesicCellPaths(
          parentBoundaries,
          parentVertexCounts,
          1,
          CENTERS[ctx.options.center][0]
        );
        const hierarchyStatusOffset = parentCountsOffset + 4;
        const hierarchyFailure = values.getUint32(hierarchyStatusOffset, true);
        if (hierarchyFailure)
          variant.cachedReadouts.status += `; center-cell parent status ${hierarchyFailure}`;
        if (active?.key === key)
          for (const [id, value] of Object.entries(variant.cachedReadouts))
            ctx.setReadout(id, value);
        ctx.requestLayers();
      }
    );
    const reader = {
      markArea: () => areaReader.markStale(),
      markCover: () => coverReader.markStale(),
      flush: (encoder: Parameters<SummaryReader['flush']>[0]) => {
        if (!areaReader.isPending) areaRevisionAtSubmission = summaryRevision;
        if (!coverReader.isPending) coverRevisionAtSubmission = summaryRevision;
        areaReader.flush(encoder);
        coverReader.flush(encoder);
      }
    };
    variant = {
      key,
      level,
      mode,
      graph: compiled,
      resources: owned,
      reader,
      cap,
      queryPosition,
      neighborParameters,
      centerPosition,
      parentBoundaries,
      parentVertexCounts,
      coverBoundaries,
      coverVertexCounts,
      coverCounts,
      coverCapacity,
      coverPaths: [],
      parentPaths: [],
      cachedReadouts: {},
      dirty: true,
      areaRead: false
    };
    return variant;
  };

  const updateActive = () => {
    const level = Number(ctx.options.level);
    active = variants.get(`${level}-${ctx.options.coverMode}`) ?? active;
    if (!active) return;
    for (const [id, value] of Object.entries(active.cachedReadouts)) ctx.setReadout(id, value);
    active.cap.write(makeCap(...CENTERS[ctx.options.center], ctx.options.radiusKm));
    active.queryPosition.write(Float32Array.of(...CENTERS[ctx.options.center]));
    active.centerPosition.write(Float32Array.of(...CENTERS[ctx.options.center]));
    active.neighborParameters.write(
      getGPUNeighborSearchParameterValues({
        bounds: [-180, -90, 180, 90],
        radius: ctx.options.radiusKm * 1000,
        spatialContext: {
          coordinateSpace: 'longitude-latitude',
          metric: 'great-circle',
          units: 'meters',
          sphereRadius: EARTH_RADIUS_METERS
        }
      })
    );
    active.dirty = true;
    summaryRevision++;
  };
  const initialLevel = Number(ctx.options.level);
  const initialMode = ctx.options.coverMode;
  variants.set(`${initialLevel}-${initialMode}`, await makeVariant(initialLevel, initialMode));
  updateActive();
  const requestVariant = (level: number, mode: 'exterior' | 'interior') => {
    const key = `${level}-${mode}`;
    const cached = variants.get(key);
    void graphSwap
      .request({
        build: () => (cached ? Promise.resolve(cached) : makeVariant(level, mode)),
        commit: variant => {
          if (!cached) variants.set(key, variant);
          if (Number(ctx.options.level) === level && ctx.options.coverMode === mode) {
            active = variant;
            updateActive();
            ctx.requestLayers();
          }
        },
        discard: variant => {
          if (!cached) variant.resources.destroy();
        }
      })
      .catch(error => ctx.setStatus(`S2 graph build failed: ${String(error)}`));
  };
  return {
    getCompiledGraphs: () => [...variants.values()].map(variant => variant.graph),
    encode(commandEncoder) {
      if (!active || destroyed) return;
      if (active.dirty) {
        active.graph.encode(commandEncoder, {parameters: undefined});
        active.dirty = false;
        active.reader.markCover();
      }
      if (!active.areaRead) {
        active.reader.markArea();
        active.areaRead = true;
      }
      active.reader.flush(commandEncoder);
    },
    getLayers() {
      if (!active) return [];
      const layers: Layer[] = [
        ...(ctx.options.showParents
          ? [
              new PathLayer({
                id: 's2-parent-cells',
                data: active.parentPaths,
                getPath: path => path,
                getColor: [93, 198, 186, 145],
                getWidth: 1,
                widthUnits: 'pixels',
                pickable: false
              })
            ]
          : []),
        new PathLayer({
          id: 's2-cap-cells',
          data: active.coverPaths,
          getPath: path => path,
          getColor: active.mode === 'exterior' ? [58, 137, 178, 205] : [44, 105, 161, 205],
          getWidth: 1.15,
          widthUnits: 'pixels',
          pickable: false
        }),
        new PathLayer({
          id: 's2-search-radius',
          data: geodesicCircle(CENTERS[ctx.options.center], ctx.options.radiusKm * 1000, 128),
          getPath: path => path,
          getColor: [145, 207, 224, 230],
          getWidth: 1.7,
          widthUnits: 'pixels',
          pickable: false
        }),
        new ScatterplotLayer({
          id: 's2-search-center',
          data: [CENTERS[ctx.options.center]],
          getPosition: point => point,
          getRadius: 5,
          radiusUnits: 'pixels',
          getFillColor: [240, 228, 190, 240],
          stroked: true,
          getLineColor: [20, 35, 45, 255],
          lineWidthUnits: 'pixels',
          getLineWidth: 1,
          pickable: false
        }),
        new ScatterplotLayer({
          id: 's2-earthquake-sample',
          data: Array.from(
            {length: sampleCount},
            (_, index) =>
              [sampledPositions[index * 2], sampledPositions[index * 2 + 1]] as [number, number]
          ),
          getPosition: point => point,
          getRadius: 2.2,
          radiusUnits: 'pixels',
          getFillColor: [235, 112, 57, 170],
          stroked: false,
          pickable: false
        })
      ];
      return layers;
    },
    setOption(id) {
      if (id === 'level' || id === 'coverMode') {
        const level = Number(ctx.options.level);
        const mode = ctx.options.coverMode;
        const key = `${level}-${mode}`;
        if (variants.has(key)) {
          updateActive();
          ctx.requestLayers();
        }
        requestVariant(level, mode);
      } else if (id === 'radiusKm' || id === 'center') {
        updateActive();
        ctx.requestLayers();
      }
      if (id === 'showParents') ctx.requestLayers();
    },
    destroy() {
      destroyed = true;
      resources.destroy();
      for (const variant of variants.values()) variant.resources.destroy();
    }
  };
}
