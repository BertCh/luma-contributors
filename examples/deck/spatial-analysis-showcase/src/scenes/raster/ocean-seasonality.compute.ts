// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  GPURasterSeasonalBreak,
  GPURasterTemporal
} from '@luma-contributors/analysis/raster-analysis';
import {SpatialAnalysisPointLayer, SpatialAnalysisPolygonLayer} from '../../engine/layers';
import type {RampName} from '../../engine/ramps';
import {createPolygonMeshBuffers} from '../../engine/polygon-buffers';
import {createAnalysisFrameProjection, type AnalysisFrame} from '../../engine/analysis-frame';
import {GraphSwap} from '../../engine/graph-swap';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {fetchBytes, fetchJson, getDataFileUrl, type GeoJsonCollection} from '../../data/loaders';
import {buildPolygonMesh} from '../../cartography/polygon-mesh';
import {OCEAN_MODEL_STATUS_COLORS} from './ocean-seasonality-data';
import {toRadians} from '@math.gl/core';

export type OceanSeasonalityOptions = {
  series: 'anomaly' | 'sst';
  layer:
    | 'observed'
    | 'smoothed'
    | 'trend'
    | 'seasonality'
    | 'residual'
    | 'break-month'
    | 'bic-gain'
    | 'fit-status';
  month: number;
  ramp: RampName;
  opacity: number;
  smoothWindow: number;
  polynomialOrder: number;
  edgeMode: 'interp' | 'truncate';
  periodModel: 'harmonic' | 'dummy';
  fitMethod: 'joint' | 'iterative';
  missingPolicy: 'omit' | 'reject' | 'interpolate';
  minimumSegmentLength: number;
  minimumBICGain: number;
  maximumIterations: number;
  tolerance: number;
};

type OceanManifest = {
  bbox: [number, number, number, number];
  raster: {
    width: number;
    height: number;
    depth: number;
    bounds: [number, number, number, number];
    cellSizeDegrees: number;
  };
  properties: {
    dates: string[];
    period: number;
    anomalyFile: string;
    sourceVersion: string;
    attribution: string;
    doi: string;
    limitations: string[];
  };
};

const DATASET_ID = 'northeast-pacific-sst';
const FRAME: AnalysisFrame = {
  kind: 'albers-equal-area',
  origin: [-135, 40],
  standardParallels: [25, 50]
};
type ModelBuffers = {
  coefficients: Buffer;
  components: Buffer;
  breakIndex: Buffer;
  improvement: Buffer;
  status: Buffer;
  diagnostics: Buffer;
};

async function loadFloatStack(
  file: string,
  cellCount: number,
  sliceCount: number,
  signal: AbortSignal
): Promise<Float32Array> {
  const buffer = await fetchBytes(getDataFileUrl(DATASET_ID, file), signal);
  if (buffer.byteLength !== cellCount * sliceCount * 4)
    throw new Error(
      `${file} has ${buffer.byteLength} bytes; expected ${cellCount * sliceCount * 4}`
    );
  const view = new DataView(buffer);
  const values = new Float32Array(cellCount * sliceCount);
  for (let index = 0; index < values.length; index++)
    values[index] = view.getFloat32(index * 4, true);
  return values;
}

function makeOceanGrid(
  width: number,
  height: number,
  bounds: readonly [number, number, number, number]
) {
  const [west, south, east, north] = bounds;
  const stepX = (east - west) / width;
  const stepY = (north - south) / height;
  const features: GeoJsonCollection['features'] = [];
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const westEdge = west + column * stepX;
      const eastEdge = westEdge + stepX;
      const northEdge = north - row * stepY;
      const southEdge = northEdge - stepY;
      features.push({
        type: 'Feature',
        id: row * width + column,
        properties: {},
        geometry: {
          type: 'Polygon',
          coordinates: [
            [
              [westEdge, southEdge],
              [eastEdge, southEdge],
              [eastEdge, northEdge],
              [westEdge, northEdge],
              [westEdge, southEdge]
            ]
          ]
        }
      });
    }
  }
  const projection = createAnalysisFrameProjection(FRAME);
  return {
    mesh: buildPolygonMesh({type: 'FeatureCollection', features}, (longitude, latitude) =>
      projection.project(longitude, latitude)
    ),
    projection
  };
}

/** NOAA ERSSTv5 monthly maps, seasonal smoothing and bounded single-break diagnostics. */
export async function createOceanSeasonality(
  ctx: SceneContext<OceanSeasonalityOptions>
): Promise<SceneInstance<OceanSeasonalityOptions>> {
  const manifest = await fetchJson<OceanManifest>(
    getDataFileUrl(DATASET_ID, 'manifest.json'),
    ctx.signal
  );
  const {width, height, depth: sliceCount} = manifest.raster;
  const cellCount = width * height;
  const [sst, anomaly] = await Promise.all([
    loadFloatStack('sst.bin', cellCount, sliceCount, ctx.signal),
    loadFloatStack(manifest.properties.anomalyFile, cellCount, sliceCount, ctx.signal)
  ]);
  ctx.signal.throwIfAborted();
  const resources = new SpatialAnalysisResources(ctx.device, 'ocean-seasonality');
  const grid = makeOceanGrid(width, height, manifest.raster.bounds);
  const polygon = createPolygonMeshBuffers(resources, grid.mesh, 'ocean-cells');
  const marker = resources.createBuffer('selected-cell-marker', 8);
  const input = new Float32Array(ctx.options.series === 'sst' ? sst : anomaly);
  const inputBuffer = resources.createBuffer('series-input', input);
  const smoothValues = resources.createBuffer('smoothed-monthly', cellCount * sliceCount * 4);
  const smoothCounts = resources.createBuffer('smooth-support', cellCount * sliceCount * 4);
  const smoothChosen = resources.createBuffer('smooth-source', cellCount * sliceCount * 4);
  const smoothStatus = resources.createBuffer('smooth-status', cellCount * sliceCount * 4);
  const modelBuffers: ModelBuffers = {
    coefficients: resources.createBuffer('model-coefficients', cellCount * 14 * 4),
    components: resources.createBuffer('model-components', cellCount * sliceCount * 3 * 4),
    breakIndex: resources.createBuffer('model-break-index', cellCount * 4),
    improvement: resources.createBuffer('model-bic-gain', cellCount * 4),
    status: resources.createBuffer('model-status', cellCount * 4),
    diagnostics: resources.createBuffer('model-diagnostics', cellCount * 6 * 4)
  };
  const valueIndices = resources.createBuffer('month-value-indices', new Uint32Array(cellCount));
  let revision = 0;
  let requestRevision = -1;
  let acceptedRevision = -1;
  let selectedCell = 137;
  const initialColumn = selectedCell % width;
  const initialRow = Math.floor(selectedCell / width);
  const initialLongitude =
    manifest.raster.bounds[0] + (initialColumn + 0.5) * manifest.raster.cellSizeDegrees;
  const initialLatitude =
    manifest.raster.bounds[3] - (initialRow + 0.5) * manifest.raster.cellSizeDegrees;
  marker.write(Float32Array.of(...grid.projection.project(initialLongitude, initialLatitude)));
  let summary: {
    smooth: Float32Array;
    smoothCounts: Uint32Array;
    smoothStatus: Uint32Array;
    components: Float32Array;
    breakIndex: Uint32Array;
    improvement: Float32Array;
    status: Uint32Array;
    diagnostics: Float32Array;
  } | null = null;
  const summaryReader = new SummaryReader(
    resources,
    'ocean-seasonality-read',
    [
      {buffer: smoothValues, size: cellCount * sliceCount * 4},
      {buffer: smoothCounts, size: cellCount * sliceCount * 4},
      {buffer: smoothStatus, size: cellCount * sliceCount * 4},
      {buffer: modelBuffers.components, size: cellCount * sliceCount * 3 * 4},
      {buffer: modelBuffers.breakIndex, size: cellCount * 4},
      {buffer: modelBuffers.improvement, size: cellCount * 4},
      {buffer: modelBuffers.status, size: cellCount * 4},
      {buffer: modelBuffers.diagnostics, size: cellCount * 6 * 4}
    ],
    bytes => {
      if (requestRevision !== revision) return;
      const seriesByteLength = cellCount * sliceCount * 4;
      const cellByteLength = cellCount * 4;
      const modelByteOffset = seriesByteLength * 6;
      summary = {
        smooth: new Float32Array(bytes, 0, cellCount * sliceCount),
        smoothCounts: new Uint32Array(bytes, seriesByteLength, cellCount * sliceCount),
        smoothStatus: new Uint32Array(bytes, seriesByteLength * 2, cellCount * sliceCount),
        components: new Float32Array(bytes, seriesByteLength * 3, cellCount * sliceCount * 3),
        breakIndex: new Uint32Array(bytes, modelByteOffset, cellCount),
        improvement: new Float32Array(bytes, modelByteOffset + cellByteLength, cellCount),
        status: new Uint32Array(bytes, modelByteOffset + cellByteLength * 2, cellCount),
        diagnostics: new Float32Array(bytes, modelByteOffset + cellByteLength * 3, cellCount * 6)
      };
      acceptedRevision = revision;
      publishStatus();
      if (selectedCell >= 0) publishPixel();
    }
  );

  const graphViewer = (graph: GPUCommandGraph<void>) => {
    const handles = new Map<Buffer, ReturnType<GPUCommandGraph<void>['importBuffer']>>();
    return <Format extends 'float32' | 'uint32'>(
      id: string,
      buffer: Buffer,
      format: Format,
      length: number
    ): GraphDataView<Format> => {
      let handle = handles.get(buffer);
      if (!handle) {
        handle = graph.importBuffer(
          {id, byteLength: buffer.byteLength, usage: buffer.usage},
          buffer
        );
        handles.set(buffer, handle);
      }
      return graph.createDataView(handle, {format, length});
    };
  };
  const makeTemporal = async (
    options: OceanSeasonalityOptions
  ): Promise<CompiledGPUCommandGraph<void>> => {
    const graph = new GPUCommandGraph<void>(ctx.device, {
      id: `ocean-sg-${options.smoothWindow}-${options.polynomialOrder}-${options.edgeMode}`
    });
    const view = graphViewer(graph);
    graph.add(
      new GPURasterTemporal({
        id: 'seasonal-smoother',
        cellCount,
        sliceCount,
        values: view('values', inputBuffer, 'float32', cellCount * sliceCount),
        operation: 'savitzky-golay',
        windowLength: options.smoothWindow,
        polynomialOrder: options.polynomialOrder,
        edgeMode: options.edgeMode,
        output: {
          values: view('smooth', smoothValues, 'float32', cellCount * sliceCount),
          counts: view('counts', smoothCounts, 'uint32', cellCount * sliceCount),
          chosenSlice: view('chosen', smoothChosen, 'uint32', cellCount * sliceCount),
          status: view('status', smoothStatus, 'uint32', cellCount * sliceCount)
        }
      })
    );
    return await graph.compileAsync();
  };
  const makeModel = async (
    options: OceanSeasonalityOptions
  ): Promise<CompiledGPUCommandGraph<void>> => {
    const graph = new GPUCommandGraph<void>(ctx.device, {
      id: `ocean-break-${options.periodModel}-${options.fitMethod}-${options.missingPolicy}-${options.minimumSegmentLength}-${options.minimumBICGain}`
    });
    const view = graphViewer(graph);
    graph.add(
      new GPURasterSeasonalBreak({
        id: 'single-seasonal-break',
        cellCount,
        sliceCount,
        values: view('values', inputBuffer, 'float32', cellCount * sliceCount),
        period: manifest.properties.period,
        minimumSegmentLength: options.minimumSegmentLength,
        minimumBICGain: options.minimumBICGain,
        seasonality: options.periodModel,
        fitMethod: options.fitMethod,
        missingPolicy: options.missingPolicy,
        maximumIterations: options.maximumIterations,
        tolerance: options.tolerance,
        output: {
          coefficients: view('coefficients', modelBuffers.coefficients, 'float32', cellCount * 14),
          components: view(
            'components',
            modelBuffers.components,
            'float32',
            cellCount * sliceCount * 3
          ),
          breakIndex: view('break-index', modelBuffers.breakIndex, 'uint32', cellCount),
          improvement: view('improvement', modelBuffers.improvement, 'float32', cellCount),
          status: view('model-status', modelBuffers.status, 'uint32', cellCount),
          diagnostics: view('diagnostics', modelBuffers.diagnostics, 'float32', cellCount * 6)
        }
      })
    );
    return await graph.compileAsync();
  };

  let temporalGraph = resources.track(await makeTemporal(ctx.options));
  let modelGraph = resources.track(await makeModel(ctx.options));
  const temporalSwap = resources.track(new GraphSwap());
  const modelSwap = resources.track(new GraphSwap());
  let dirty = true;
  const dateDays = manifest.properties.dates.map(date => Date.parse(date) / 86400000);
  const dateTicks = manifest.properties.dates;

  function writeSeries(): void {
    const data = ctx.options.series === 'sst' ? sst : anomaly;
    input.set(data);
    inputBuffer.write(input);
  }
  function writeMonthIndices(): void {
    const month = Math.max(0, Math.min(sliceCount - 1, Math.round(ctx.options.month)));
    const component =
      ctx.options.layer === 'trend'
        ? 0
        : ctx.options.layer === 'seasonality'
          ? 1
          : ctx.options.layer === 'residual'
            ? 2
            : -1;
    valueIndices.write(
      Uint32Array.from({length: cellCount}, (_, cell) =>
        component < 0 ? cell * sliceCount + month : (cell * sliceCount + month) * 3 + component
      )
    );
  }
  function getCell(coordinate: readonly [number, number] | null): number {
    if (!coordinate) return -1;
    const [west, , , north] = manifest.raster.bounds;
    const column = Math.floor((coordinate[0] - west) / manifest.raster.cellSizeDegrees);
    const row = Math.floor((north - coordinate[1]) / manifest.raster.cellSizeDegrees);
    return column < 0 || row < 0 || column >= width || row >= height ? -1 : row * width + column;
  }
  function publishStatus(): void {
    if (!summary) return;
    const counts = new Map<number, number>();
    for (const status of summary.status) counts.set(status, (counts.get(status) ?? 0) + 1);
    const diagnostics = summary.diagnostics;
    let smoothInsufficient = 0;
    let smoothRankFailure = 0;
    for (const status of summary.smoothStatus) {
      if (status === 1) smoothInsufficient++;
      else if (status === 2) smoothRankFailure++;
    }
    let converged = 0;
    let support = 0;
    let averageIterations = 0;
    for (let cell = 0; cell < cellCount; cell++) {
      if (summary.status[cell] === 0) support++;
      if (diagnostics[cell * 6 + 1] === 1) converged++;
      averageIterations += diagnostics[cell * 6];
    }
    ctx.setReadout(
      'fitStatus',
      `fit ${formatCount(counts.get(0) ?? 0)} · insufficient ${formatCount(counts.get(1) ?? 0)} · numerical ${formatCount(counts.get(2) ?? 0)} · rejected gap ${formatCount(counts.get(4) ?? 0)} · iteration limit ${formatCount(counts.get(8) ?? 0)}`
    );
    ctx.setReadout(
      'fitDiagnostics',
      `${formatCount(support)} supported cells · ${formatCount(converged)} converged · mean ${(averageIterations / cellCount || 0).toFixed(1)} iterations · observed support, not filled sample count`
    );
    ctx.setReadout(
      'sgStatus',
      `Savitzky–Golay insufficient ${formatCount(smoothInsufficient)} slots · rank failures ${formatCount(smoothRankFailure)} slots`
    );
  }
  function publishPixel(): void {
    if (!summary || selectedCell < 0) {
      ctx.setReadout('selectedCell', 'click a Pacific cell');
      ctx.setChart('pixelSeries', null);
      return;
    }
    const [west, , , north] = manifest.raster.bounds;
    const column = selectedCell % width;
    const row = Math.floor(selectedCell / width);
    const longitude = west + (column + 0.5) * manifest.raster.cellSizeDegrees;
    const latitude = north - (row + 0.5) * manifest.raster.cellSizeDegrees;
    const [markerX, markerY] = grid.projection.project(longitude, latitude);
    marker.write(Float32Array.of(markerX, markerY));
    const offset = selectedCell * sliceCount;
    const observed = input.subarray(offset, offset + sliceCount);
    const smooth = summary.smooth.subarray(offset, offset + sliceCount);
    const trend = new Float32Array(sliceCount);
    const seasonal = new Float32Array(sliceCount);
    const residual = new Float32Array(sliceCount);
    for (let month = 0; month < sliceCount; month++) {
      const index = (offset + month) * 3;
      trend[month] = summary.components[index];
      seasonal[month] = summary.components[index + 1];
      residual[month] = summary.components[index + 2];
    }
    const breakIndex = summary.breakIndex[selectedCell];
    const diagnostics = summary.diagnostics.subarray(selectedCell * 6, selectedCell * 6 + 6);
    const selectedMonth = Math.max(0, Math.min(sliceCount - 1, Math.round(ctx.options.month)));
    const smoothOffset = offset + selectedMonth;
    ctx.setReadout(
      'selectedCell',
      `${Math.abs(longitude).toFixed(1)}°W, ${latitude.toFixed(1)}°N · ${ctx.options.series === 'sst' ? 'SST' : 'NOAA anomaly'} · status ${summary.status[selectedCell]} · BIC gain ${summary.improvement[selectedCell].toFixed(2)} · break ${breakIndex === 0xffffffff ? 'none selected' : (dateTicks[breakIndex]?.slice(0, 7) ?? 'unknown')} · iterations ${diagnostics[0].toFixed(0)}, converged ${diagnostics[1] === 1 ? 'yes' : 'no'}, support ${diagnostics[2].toFixed(0)} · SG support ${summary.smoothCounts[smoothOffset]}, SG status ${summary.smoothStatus[smoothOffset]}`
    );
    ctx.setChart('pixelSeries', {
      kind: 'line',
      title: 'Observed monthly data and GPU fits',
      xLabel: 'month',
      yLabel: ctx.options.series === 'sst' ? '°C' : '°C anomaly',
      xDomain: [dateDays[0], dateDays[dateDays.length - 1]],
      formatX: value => new Date(value * 86400000).toISOString().slice(0, 7),
      formatY: value => value.toFixed(1),
      description:
        'NOAA monthly reconstructed sea-surface temperature, Savitzky–Golay smoothing and fitted seasonal-break components at the clicked grid cell.',
      series: [
        {label: 'Observed', x: dateDays, y: observed, color: 0, points: true},
        {label: 'SG smooth', x: dateDays, y: smooth, color: 1},
        {label: 'Trend + jump', x: dateDays, y: trend, color: 2},
        {label: 'Seasonality', x: dateDays, y: seasonal, color: 3},
        {label: 'Residual', x: dateDays, y: residual, color: 4}
      ]
    });
  }

  writeMonthIndices();
  ctx.setReadout(
    'source',
    `${manifest.properties.sourceVersion} · 26 × 21 cells at 2° · monthly 2012–2017 · 423 ocean cells, 123 land/missing`
  );
  const annualMeans = Array.from({length: 6}, (_, yearIndex) => {
    let total = 0;
    let totalWeight = 0;
    let count = 0;
    for (let cell = 0; cell < cellCount; cell++) {
      const row = Math.floor(cell / width);
      const latitude = manifest.raster.bounds[3] - (row + 0.5) * manifest.raster.cellSizeDegrees;
      const weight = Math.cos(toRadians(latitude));
      for (let month = yearIndex * 12; month < (yearIndex + 1) * 12; month++) {
        const value = anomaly[cell * sliceCount + month];
        if (Number.isFinite(value)) {
          total += value * weight;
          totalWeight += weight;
          count++;
        }
      }
    }
    const mean = totalWeight > 0 ? total / totalWeight : Number.NaN;
    return `${2012 + yearIndex} ${count ? (mean >= 0 ? '+' : '') + mean.toFixed(3) : 'n/a'}°C`;
  });
  ctx.setReadout(
    'annualEvidence',
    `cos(latitude)-weighted ocean cells · ${annualMeans.join(' · ')}`
  );
  ctx.setReadout('selectedCell', 'click a Pacific cell');
  ctx.setStatus('');

  return {
    getCompiledGraphs: () => [temporalGraph, modelGraph],
    setOption(id) {
      revision++;
      dirty = true;
      summary = null;
      if (id === 'series') writeSeries();
      if (id === 'month' || id === 'layer') writeMonthIndices();
      if (['smoothWindow', 'polynomialOrder', 'edgeMode'].includes(id)) {
        ctx.setStatus('Compiling the selected smoother');
        const options = {...ctx.options};
        void temporalSwap
          .request({
            build: () => makeTemporal(options),
            commit: next => {
              const previous = temporalGraph;
              temporalGraph = resources.track(next);
              resources.release(previous);
              revision++;
              requestRevision = -1;
              acceptedRevision = -1;
              summary = null;
              dirty = true;
              ctx.setStatus('');
              ctx.requestLayers();
            },
            discard: next => next.destroy()
          })
          .catch(error => ctx.setStatus(`Smoother compile failed: ${String(error)}`));
      }
      if (
        [
          'periodModel',
          'fitMethod',
          'missingPolicy',
          'minimumSegmentLength',
          'minimumBICGain',
          'maximumIterations',
          'tolerance'
        ].includes(id)
      ) {
        ctx.setStatus('Compiling the selected seasonal-break model');
        const options = {...ctx.options};
        void modelSwap
          .request({
            build: () => makeModel(options),
            commit: next => {
              const previous = modelGraph;
              modelGraph = resources.track(next);
              resources.release(previous);
              revision++;
              requestRevision = -1;
              acceptedRevision = -1;
              summary = null;
              dirty = true;
              ctx.setStatus('');
              ctx.requestLayers();
            },
            discard: next => next.destroy()
          })
          .catch(error => ctx.setStatus(`Model compile failed: ${String(error)}`));
      }
      ctx.requestLayers();
    },
    encode(commandEncoder) {
      if (temporalSwap.pending || modelSwap.pending) {
        summaryReader.flush(commandEncoder);
        return;
      }
      if (dirty) {
        temporalGraph.encode(commandEncoder, {parameters: undefined});
        modelGraph.encode(commandEncoder, {parameters: undefined});
        dirty = false;
      }
      if (
        acceptedRevision !== revision &&
        requestRevision !== revision &&
        !summaryReader.isPending
      ) {
        requestRevision = revision;
        summaryReader.request(commandEncoder);
      }
      summaryReader.flush(commandEncoder);
    },
    getLayers() {
      const state = ctx.options;
      let values: Buffer;
      let valueIndicesBuffer: Buffer | undefined;
      let valueFormat: 'float32' | 'uint32' = 'float32';
      let valueRange: readonly [number, number] = state.series === 'sst' ? [-3, 32] : [-3, 3];
      let colormap: RampName | 'category' = state.ramp;
      if (state.layer === 'observed') {
        values = inputBuffer;
        valueIndicesBuffer = valueIndices;
      } else if (state.layer === 'smoothed') {
        values = smoothValues;
        valueIndicesBuffer = valueIndices;
      } else if (
        state.layer === 'trend' ||
        state.layer === 'seasonality' ||
        state.layer === 'residual'
      ) {
        values = modelBuffers.components;
        valueIndicesBuffer = valueIndices;
        valueRange =
          state.layer === 'seasonality' || state.layer === 'residual'
            ? [-2, 2]
            : state.series === 'sst'
              ? [-3, 32]
              : [-3, 3];
      } else if (state.layer === 'break-month') {
        values = modelBuffers.breakIndex;
        valueFormat = 'uint32';
        valueRange = [0, sliceCount - 1];
      } else if (state.layer === 'bic-gain') {
        values = modelBuffers.improvement;
        valueRange = [0, 20];
      } else {
        values = modelBuffers.status;
        valueFormat = 'uint32';
        valueRange = [0, 8];
        colormap = 'category';
      }
      const signed =
        (state.series === 'anomaly' && ['observed', 'smoothed', 'trend'].includes(state.layer)) ||
        ['seasonality', 'residual'].includes(state.layer);
      if (signed) {
        valueRange =
          state.layer === 'seasonality' || state.layer === 'residual' ? [-2, 2] : [-3, 3];
        colormap = 'diverging';
      }
      return [
        new SpatialAnalysisPolygonLayer({
          id: `ocean-seasonality-${state.layer}-${state.month}`,
          analysisFrame: FRAME,
          triangles: polygon.triangles,
          features: polygon.triangleFeatures,
          vertexCount: polygon.vertexCount,
          values,
          valueIndices: valueIndicesBuffer,
          valueFormat,
          colormap,
          valueRange,
          palette: state.layer === 'fit-status' ? OCEAN_MODEL_STATUS_COLORS : undefined,
          noDataValue: 0xffffffff,
          opacity: state.opacity
        }),
        new SpatialAnalysisPointLayer({
          id: 'ocean-seasonality-selected-cell',
          analysisFrame: FRAME,
          positions: marker,
          instanceCount: 1,
          radiusPixels: 7,
          color: [255, 238, 92, 255],
          outlineColor: [18, 27, 43, 255],
          outlineWidthPixels: 1.5
        })
      ];
    },
    getTooltip(event) {
      const cell = getCell(event.coordinate);
      if (cell < 0) return null;
      if (!summary) return 'Loading the fitted monthly series…';
      const month = Math.round(ctx.options.month);
      const index = cell * sliceCount + month;
      const breakIndex = summary.breakIndex[cell];
      return `${dateTicks[month].slice(0, 7)} · ${ctx.options.series === 'sst' ? 'SST' : 'anomaly'} ${input[index].toFixed(2)} °C · SG ${summary.smooth[index].toFixed(2)} °C\n${ctx.options.layer === 'trend' ? `Trend ${summary.components[index * 3].toFixed(2)} °C` : `Model status ${summary.status[cell]}`} · BIC gain ${summary.improvement[cell].toFixed(2)} · break ${breakIndex === 0xffffffff ? 'none' : dateTicks[breakIndex].slice(0, 7)}`;
    },
    onClick(event) {
      const cell = getCell(event.coordinate);
      if (cell < 0) return false;
      selectedCell = cell;
      const column = cell % width;
      const row = Math.floor(cell / width);
      const [west, , , north] = manifest.raster.bounds;
      const longitude = west + (column + 0.5) * manifest.raster.cellSizeDegrees;
      const latitude = north - (row + 0.5) * manifest.raster.cellSizeDegrees;
      marker.write(Float32Array.of(...grid.projection.project(longitude, latitude)));
      publishPixel();
      ctx.requestLayers();
      return true;
    },
    destroy() {
      summaryReader.stop();
      resources.destroy();
    }
  };
}
