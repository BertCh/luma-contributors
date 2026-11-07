// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma-contributors/graph/gpu-core';
import {GPUTerrainDEMCoregistration} from '@luma-contributors/graph/gpu-terrain';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {GraphSwap} from '../../engine/graph-swap';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import type {BarChartData} from '../chart-types';
import {ColorRasterLayer} from './b14a-layers';
import type {SurveyFrameSensitivityOptions} from './survey-frame-sensitivity.types';
import {clamp} from '@math.gl/core';

type DemGrid = {
  width: number;
  height: number;
  affine: readonly [number, number, number, number, number, number];
  pixelInterpretation: 'area' | 'point';
  bounds: readonly [number, number, number, number];
  cellSizeMeters: number;
};
type FitRegion = SurveyFrameSensitivityOptions['controlRegion'];
type Settings = Pick<SurveyFrameSensitivityOptions, 'assumedDifferenceSd' | 'correlationLength'>;
type MaskOutput = {
  region: FitRegion;
  mask: Buffer;
  fit: Buffer;
  differences: Buffer;
  validity: Buffer;
  volume: Buffer;
  controlCount: number;
};
type ActiveGraph = {
  compiled: CompiledGPUCommandGraph<void>;
  results: MaskOutput[];
  settings: Settings;
  ownedBuffers: Buffer[];
};
type ResultSummary = {
  fit: Float32Array;
  differences: Float32Array;
  validity: Uint32Array;
  volume: Float32Array;
};

const FIT_REGIONS: readonly FitRegion[] = ['west', 'east', 'north', 'all-paired'];
const FIT_LABELS: Record<FitRegion, string> = {
  west: 'West',
  east: 'East',
  north: 'North',
  'all-paired': 'All paired cells'
};
const DIFFERENCE_COLORS = [
  [57, 116, 158],
  [152, 190, 194],
  [232, 226, 207],
  [218, 154, 123],
  [167, 75, 75]
] as const;
const UNCERTAINTY_COLORS = [
  [34, 66, 90],
  [81, 125, 135],
  [190, 185, 131],
  [239, 202, 126]
] as const;
const CONTROL_COLOR = [255, 205, 120, 235] as const;
const NO_DATA_HATCH = [194, 203, 211, 145] as const;

function interpolateColor(
  value: number,
  minimum: number,
  maximum: number,
  colors: readonly (readonly [number, number, number])[]
): readonly [number, number, number] {
  const normalized = clamp((value - minimum) / (maximum - minimum), 0, 1);
  const scaled = normalized * (colors.length - 1);
  const start = Math.min(Math.floor(scaled), colors.length - 2);
  const fraction = scaled - start;
  return [0, 1, 2].map(axis =>
    Math.round(colors[start][axis] * (1 - fraction) + colors[start + 1][axis] * fraction)
  ) as [number, number, number];
}

function packRgba(red: number, green: number, blue: number, alpha: number): number {
  return (red | (green << 8) | (blue << 16) | (alpha << 24)) >>> 0;
}

function getRegionMask(region: FitRegion, index: number, width: number, height: number): boolean {
  const column = index % width;
  const row = Math.floor(index / width);
  switch (region) {
    case 'west':
      return column < width / 2;
    case 'east':
      return column >= width / 2;
    case 'north':
      return row < height / 2;
    case 'all-paired':
      return true;
  }
}

export async function createSurveyFrameSensitivity(
  ctx: SceneContext<SurveyFrameSensitivityOptions>
): Promise<SceneInstance<SurveyFrameSensitivityOptions>> {
  const dataset = ctx.datasets.get('ahk-rock-glacier');
  const demGrid = dataset.manifest.properties?.demGrid as DemGrid;
  const {width, height} = demGrid;
  const cellCount = width * height;
  const reference = dataset.column<Float32Array>('demReference');
  const moving = dataset.column<Float32Array>('demComparison');
  const referenceCounts = dataset.column<Uint32Array>('demReferenceCount');
  const movingCounts = dataset.column<Uint32Array>('demComparisonCount');
  const origin = dataset.defaultOrigin;
  const projection = dataset.getDisplayProjection(origin);
  const projectedBounds = [
    projection.project(demGrid.bounds[0], demGrid.bounds[3]),
    projection.project(demGrid.bounds[2], demGrid.bounds[1])
  ];
  const bounds: [number, number, number, number] = [
    Math.min(projectedBounds[0][0], projectedBounds[1][0]),
    Math.min(projectedBounds[0][1], projectedBounds[1][1]),
    Math.max(projectedBounds[0][0], projectedBounds[1][0]),
    Math.max(projectedBounds[0][1], projectedBounds[1][1])
  ];
  const scalarOrigin = [origin[0], origin[1], 0] as [number, number, number];
  const paired = new Uint32Array(cellCount);
  const rawDifferences = new Float32Array(cellCount);
  let pairedCells = 0;
  let rawVolume = 0;
  for (let cell = 0; cell < cellCount; cell++) {
    const valid =
      referenceCounts[cell] >= 3 &&
      movingCounts[cell] >= 3 &&
      Number.isFinite(reference[cell]) &&
      Number.isFinite(moving[cell]);
    if (!valid) continue;
    paired[cell] = 1;
    pairedCells++;
    rawDifferences[cell] = moving[cell] - reference[cell];
    rawVolume += rawDifferences[cell] * demGrid.cellSizeMeters ** 2;
  }
  const {device} = ctx;
  const resources = new SpatialAnalysisResources(device, 'survey-frame-sensitivity');
  const referenceBuffer = resources.createBuffer('dem-reference', reference);
  const movingBuffer = resources.createBuffer('dem-comparison', moving);
  const colorBuffer = resources.createBuffer('result-colors', new Uint32Array(cellCount));
  const controlColorBuffer = resources.createBuffer(
    'control-boundary-colors',
    new Uint32Array(cellCount)
  );
  const graphSwap = resources.track(new GraphSwap());
  let graphDirty = true;
  let activeGraph: ActiveGraph;
  let latestSummaries: ResultSummary[] = [];
  let summaryReader: SummaryReader;

  const compileGraph = async (settings: Settings): Promise<ActiveGraph> => {
    const graph = new GPUCommandGraph<void>(device, {
      id: `survey-frame-sensitivity-sd${settings.assumedDifferenceSd}-corr${settings.correlationLength}`
    });
    const referenceView = importGraphBuffer(
      graph,
      'dem-reference',
      referenceBuffer,
      'float32',
      cellCount
    );
    const movingView = importGraphBuffer(
      graph,
      'dem-comparison',
      movingBuffer,
      'float32',
      cellCount
    );
    const varianceBuffer = resources.createBuffer(
      `assumed-difference-variance-${settings.assumedDifferenceSd}`,
      new Float32Array(cellCount).fill(settings.assumedDifferenceSd ** 2)
    );
    const varianceView = importGraphBuffer(
      graph,
      'difference-variance',
      varianceBuffer,
      'float32',
      cellCount
    );
    const results = FIT_REGIONS.map(region => {
      const maskValues = new Uint32Array(cellCount);
      let controlCount = 0;
      for (let cell = 0; cell < cellCount; cell++) {
        const selected = paired[cell] === 1 && getRegionMask(region, cell, width, height);
        if (selected) {
          maskValues[cell] = 1;
          controlCount++;
        }
      }
      const mask = resources.createBuffer(`${region}-hypothetical-control-mask`, maskValues);
      const fit = resources.createBuffer(`${region}-vertical-fit`, 16);
      const differences = resources.createBuffer(`${region}-corrected-differences`, cellCount * 8);
      const validity = resources.createBuffer(`${region}-corrected-validity`, cellCount * 4);
      const volume = resources.createBuffer(`${region}-corrected-volume`, 12);
      const maskView = importGraphBuffer(
        graph,
        `${region}-control-mask`,
        mask,
        'uint32',
        cellCount
      );
      const fitView = importGraphBuffer(graph, `${region}-fit`, fit, 'float32', 4);
      const differencesView = importGraphBuffer(
        graph,
        `${region}-corrected-differences`,
        differences,
        'float32',
        cellCount * 2
      );
      const validityView = importGraphBuffer(
        graph,
        `${region}-validity`,
        validity,
        'uint32',
        cellCount
      );
      const volumeView = importGraphBuffer(graph, `${region}-volume`, volume, 'float32', 3);
      graph.add(
        new GPUTerrainDEMCoregistration({
          id: `vertical-coregistration-${region}`,
          grid: demGrid,
          reference: referenceView,
          moving: movingView,
          stableMask: maskView,
          differenceVariance: varianceView,
          correlationLength: settings.correlationLength,
          output: {
            fit: fitView,
            differences: differencesView,
            validity: validityView,
            volume: volumeView
          }
        })
      );
      return {region, mask, fit, differences, validity, volume, controlCount};
    });
    const compiled = resources.track(await graph.compileAsync());
    return {
      compiled,
      results,
      settings,
      ownedBuffers: [
        varianceBuffer,
        ...results.flatMap(result => [
          result.mask,
          result.fit,
          result.differences,
          result.validity,
          result.volume
        ])
      ]
    };
  };

  const releaseGraph = (graph: ActiveGraph) => {
    resources.release(graph.compiled);
    for (const buffer of graph.ownedBuffers) resources.release(buffer);
  };
  const presentSummary = (graph: ActiveGraph, summaries: ResultSummary[]) => {
    latestSummaries = summaries;
    const selectedIndex = FIT_REGIONS.indexOf(ctx.options.controlRegion);
    const selected = graph.results[selectedIndex];
    const summary = summaries[selectedIndex];
    const view = ctx.options.view;
    const values = new Float32Array(cellCount);
    if (view === 'raw') values.set(rawDifferences);
    else {
      for (let cell = 0; cell < cellCount; cell++) {
        if (summary.validity[cell] === 0) continue;
        values[cell] = summary.differences[cell * 2 + (view === 'uncertainty' ? 1 : 0)];
      }
    }
    const packedColors = new Uint32Array(cellCount);
    const controlEdges = new Uint32Array(cellCount);
    for (let cell = 0; cell < cellCount; cell++) {
      const isValid = view === 'raw' ? paired[cell] === 1 : summary.validity[cell] === 1;
      if (isValid) {
        const [red, green, blue] =
          view === 'uncertainty'
            ? interpolateColor(values[cell], 0, 0.6, UNCERTAINTY_COLORS)
            : interpolateColor(values[cell], -1, 1, DIFFERENCE_COLORS);
        packedColors[cell] = packRgba(red, green, blue, 255);
      }
      if (paired[cell] === 1 && getRegionMask(selected.region, cell, width, height)) {
        const column = cell % width;
        const row = Math.floor(cell / width);
        const northEdge = row === 0 || !getRegionMask(selected.region, cell - width, width, height);
        const southEdge =
          row === height - 1 || !getRegionMask(selected.region, cell + width, width, height);
        const westEdge = column === 0 || !getRegionMask(selected.region, cell - 1, width, height);
        const eastEdge =
          column === width - 1 || !getRegionMask(selected.region, cell + 1, width, height);
        if (northEdge || southEdge || westEdge || eastEdge)
          controlEdges[cell] = packRgba(...CONTROL_COLOR);
      }
    }
    colorBuffer.write(packedColors);
    controlColorBuffer.write(controlEdges);

    const fitValues = Array.from(summary.fit);
    const validDifferences: number[] = [];
    for (let cell = 0; cell < cellCount; cell++)
      if (summary.validity[cell] === 1) validDifferences.push(summary.differences[cell * 2]);
    const scenarioVolumes = summaries.map(entry => entry.volume[0]);
    ctx.setReadout('pairedCells', pairedCells);
    ctx.setReadout('controlCells', Number.isFinite(fitValues[2]) ? fitValues[2] : 0);
    ctx.setReadout('rawVolume', rawVolume);
    ctx.setReadout('rawBias', Number.isFinite(fitValues[0]) ? fitValues[0] : null);
    ctx.setReadout('residualNmad', Number.isFinite(fitValues[1]) ? fitValues[1] : null);
    ctx.setReadout('biasStandardError', Number.isFinite(fitValues[3]) ? fitValues[3] : null);
    ctx.setReadout(
      'correctedVolume',
      Number.isFinite(summary.volume[0]) ? summary.volume[0] : null
    );
    ctx.setReadout(
      'volumeStandardError',
      Number.isFinite(summary.volume[1]) ? summary.volume[1] : null
    );
    ctx.setReadout('validArea', Number.isFinite(summary.volume[2]) ? summary.volume[2] : null);
    ctx.setReadout('assumedDifferenceSd', graph.settings.assumedDifferenceSd);
    ctx.setReadout('correlationLengthReadout', graph.settings.correlationLength);
    ctx.setChart('maskVolumeChart', {
      kind: 'bars',
      title: 'Corrected vertical surface volume by control scenario',
      labels: ['Raw, no shift', ...FIT_REGIONS.map(region => FIT_LABELS[region])],
      values: [rawVolume, ...scenarioVolumes],
      color: 2,
      yLabel: 'm³',
      description:
        'Each bar is an independent GPU median vertical correction using the named hypothetical mask and the current uncertainty assumptions.'
    } satisfies BarChartData);
    ctx.setChart('residualHistogram', {
      kind: 'histogram',
      title: 'Corrected cell differences',
      xDomain: [-1, 1],
      values: (() => {
        const bins = new Array<number>(20).fill(0);
        for (const difference of validDifferences) {
          const index = clamp(Math.floor(((difference + 1) / 2) * 20), 0, 19);
          bins[index]++;
        }
        return bins;
      })(),
      breaks: Array.from({length: 21}, (_, index) => -1 + index * 0.1),
      xLabel: 'Corrected moving − reference (m)',
      yLabel: 'cells',
      description:
        'Histogram of GPU-corrected paired-cell differences for the selected hypothetical control region. Twenty fixed bins span -1 m to +1 m; values outside that range are counted in the end bins.'
    } satisfies BarChartData);
    ctx.requestLayers();
  };

  const setSummary = (graph: ActiveGraph, bytes: ArrayBuffer) => {
    let offset = 0;
    const summaries = graph.results.map(() => {
      const fit = new Float32Array(bytes, offset, 4);
      offset += 16;
      const differences = new Float32Array(bytes, offset, cellCount * 2);
      offset += cellCount * 8;
      const validity = new Uint32Array(bytes, offset, cellCount);
      offset += cellCount * 4;
      const volume = new Float32Array(bytes, offset, 3);
      offset += 12;
      return {fit, differences, validity, volume};
    });
    presentSummary(graph, summaries);
  };

  activeGraph = await compileGraph({
    assumedDifferenceSd: ctx.options.assumedDifferenceSd,
    correlationLength: ctx.options.correlationLength
  });
  const makeSummaryReader = (graph: ActiveGraph) =>
    new SummaryReader(
      resources,
      'survey-frame-sensitivity-summary',
      graph.results.flatMap(result => [
        {buffer: result.fit, size: 16},
        {buffer: result.differences, size: cellCount * 8},
        {buffer: result.validity, size: cellCount * 4},
        {buffer: result.volume, size: 12}
      ]),
      bytes => setSummary(graph, bytes)
    );
  summaryReader = makeSummaryReader(activeGraph);

  const getSettings = (state: SurveyFrameSensitivityOptions): Settings => ({
    assumedDifferenceSd: state.assumedDifferenceSd,
    correlationLength: state.correlationLength
  });
  const requestGraph = (settings: Settings) => {
    void graphSwap
      .request({
        build: () => compileGraph(settings),
        commit: next => {
          const previous = activeGraph;
          summaryReader.release();
          activeGraph = next;
          summaryReader = makeSummaryReader(next);
          releaseGraph(previous);
          graphDirty = true;
          latestSummaries = [];
          colorBuffer.write(new Uint32Array(cellCount));
          controlColorBuffer.write(new Uint32Array(cellCount));
          ctx.setReadout('rawBias', 'Recomputing');
          ctx.setReadout('volumeStandardError', 'Recomputing');
          ctx.requestLayers();
        },
        discard: next => releaseGraph(next)
      })
      .catch(error => ctx.setStatus(`Graph rebuild failed: ${String(error)}`));
  };

  return {
    getCompiledGraphs: () => [activeGraph.compiled],
    setOption(id, _value, state) {
      if (id === 'view' || id === 'controlRegion') {
        if (latestSummaries.length) presentSummary(activeGraph, latestSummaries);
        else ctx.requestLayers();
      }
      if (id === 'assumedDifferenceSd' || id === 'correlationLength')
        requestGraph(getSettings(state));
    },
    encode(commandEncoder) {
      if (graphDirty) {
        activeGraph.compiled.encode(commandEncoder, {parameters: undefined});
        summaryReader.request(commandEncoder);
        graphDirty = false;
      } else summaryReader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      return [
        new ColorRasterLayer({
          id: 'survey-frame-difference-grid',
          coordinateOrigin: scalarOrigin,
          gridSize: [width, height],
          bounds,
          colors: colorBuffer,
          noDataHatch: NO_DATA_HATCH,
          alpha: 0.92
        }),
        new ColorRasterLayer({
          id: 'survey-frame-control-boundary',
          coordinateOrigin: scalarOrigin,
          gridSize: [width, height],
          bounds,
          colors: controlColorBuffer,
          alpha: 1
        })
      ];
    },
    destroy() {
      summaryReader.release();
      graphSwap.destroy();
      releaseGraph(activeGraph);
      resources.destroy();
    }
  };
}
