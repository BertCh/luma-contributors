// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer, type CommandEncoder} from '@luma.gl/core';
import type {LoadedDataset} from '../../data/catalog';
import {
  GPUCrossVariogram,
  GPUSpaceTimeVariogram
} from '@luma-contributors/analysis/dataframe-analysis';
import {
  GPUCommandGraph,
  type CompiledGPUCommandGraph,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createPolygonMeshBuffers} from '../../engine/polygon-buffers';
import {buildPolygonMesh} from '../../cartography/polygon-mesh';
import {SpatialAnalysisPolygonLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createAnalysisFrameProjection} from '../../engine/analysis-frame';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {getCountyDorlingAuthoredVertexMeanAnchor} from './county-dorling-anchor';
import type {SceneContext, SceneInstance} from '../scene';

export type CountyCrossVariogramOptions = {
  analysis: 'single-period' | 'space-time';
  mode: 'unemployment' | 'unemployment-poverty';
  period: '1970' | '1980' | '1990';
  mapField: 'unemployment' | 'poverty';
  cutoff: number;
  spatialLag: number;
  temporalLag: number;
};

const COUNTY_COUNT = 21;
const PERIOD_COUNT = 3;
const RECORD_COUNT = COUNTY_COUNT * PERIOD_COUNT;
const SPATIAL_LAG_COUNT = 6;
const TEMPORAL_LAG_COUNT = 3;
const PERIODS = ['1970', '1980', '1990'] as const;
const PERIOD_INDEX: Record<CountyCrossVariogramOptions['period'], number> = {
  '1970': 0,
  '1980': 1,
  '1990': 2
};
const PERIOD_LABELS = ['1970', '1980', '1990'] as const;
const ACTUAL_TIME_LAGS = [0, 10, 20] as const;
const TEMPORAL_INTERVALS = ['[0, 6.667)', '[6.667, 13.333)', '[13.333, 20]'] as const;

type VariogramKey = `${'single' | 'space-time'}:${CountyCrossVariogramOptions['mode']}:${string}`;
type VariogramRun = {
  key: VariogramKey;
  compiled: CompiledGPUCommandGraph<void>;
  reader: SummaryReader;
  requestGeneration: Buffer;
};

function sourceField(
  dataset: LoadedDataset,
  field: 'unemployment' | 'poverty',
  period: number
): Float32Array {
  const sourceName =
    field === 'unemployment'
      ? `unemployment${PERIOD_LABELS[period]}`
      : `familyPoverty${['1969', '1979', '1989'][period]}`;
  return dataset.column<Float32Array>(sourceName);
}

/** GPU descriptive pair-bin summaries for the historical New Jersey common-support units. */
export async function createCountyCrossVariogram(
  ctx: SceneContext<CountyCrossVariogramOptions>
): Promise<SceneInstance<CountyCrossVariogramOptions>> {
  const dataset = ctx.datasets.get('ncovr-new-jersey');
  const geojson = dataset.geojson;
  if (!geojson || geojson.features.length !== COUNTY_COUNT)
    throw new Error('Expected all 21 NCOVR New Jersey historical county units');

  const frame = {kind: 'azimuthal-equidistant' as const, origin: [-74.55, 40.05] as const};
  const projection = createAnalysisFrameProjection(frame);
  const resources = new SpatialAnalysisResources(ctx.device, 'county-cross-variogram');
  const polygonBuffers = createPolygonMeshBuffers(
    resources,
    buildPolygonMesh(geojson, projection.project),
    'counties'
  );
  const anchors = new Float32Array(RECORD_COUNT * 2);
  for (let period = 0; period < PERIOD_COUNT; period++) {
    for (let county = 0; county < COUNTY_COUNT; county++) {
      const [longitude, latitude] = getCountyDorlingAuthoredVertexMeanAnchor(
        geojson.features[county]!
      );
      anchors.set(projection.project(longitude, latitude), (period * COUNTY_COUNT + county) * 2);
    }
  }
  const singlePositions = anchors.slice(0, COUNTY_COUNT * 2);
  const positionsBuffer = resources.createBuffer('county-anchors-63-records', anchors);
  const singlePositionsBuffer = resources.createBuffer(
    'county-anchors-21-records',
    singlePositions
  );
  const times = Float32Array.from({length: RECORD_COUNT}, (_, row) =>
    PERIODS[Math.floor(row / COUNTY_COUNT)] === '1970'
      ? 0
      : PERIODS[Math.floor(row / COUNTY_COUNT)] === '1980'
        ? 10
        : 20
  );
  const timesBuffer = resources.createBuffer('observed-decennial-times-years', times);
  const cutoffParameter = resources.createParameterBuffer(
    'lag-cutoffs-meters-years',
    'float32',
    2,
    new Float32Array([200_000, 20])
  );

  const fields = {
    unemployment: PERIODS.map((_period, index) => sourceField(dataset, 'unemployment', index)),
    poverty: PERIODS.map((_period, index) => sourceField(dataset, 'poverty', index))
  };
  const sourceIds = dataset.column<Uint32Array>('fips');
  const unemploymentRecords = new Float32Array(RECORD_COUNT);
  const povertyRecords = new Float32Array(RECORD_COUNT);
  for (let period = 0; period < PERIOD_COUNT; period++) {
    unemploymentRecords.set(fields.unemployment[period]!, period * COUNTY_COUNT);
    povertyRecords.set(fields.poverty[period]!, period * COUNTY_COUNT);
  }
  const outputSlots = new Map<VariogramKey, VariogramRun>();
  const readers: SummaryReader[] = [];
  let graphNumber = 0;

  const buildRun = async (
    analysis: 'single' | 'space-time',
    mode: CountyCrossVariogramOptions['mode'],
    period?: number
  ): Promise<VariogramRun> => {
    const key: VariogramKey =
      analysis === 'single' ? `single:${mode}:${PERIODS[period!]}` : `space-time:${mode}:all`;
    const graph = new GPUCommandGraph<void>(ctx.device, {
      id: `ncovr-cross-variogram-${++graphNumber}-${key}`
    });
    const imported = new Map<Buffer, GraphDataView>();
    const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
      name: string,
      buffer: Buffer,
      format: Format,
      length: number
    ) => {
      const cached = imported.get(buffer);
      if (cached) return cached as GraphDataView<Format>;
      const result = importGraphBuffer(graph, name, buffer, format, length);
      imported.set(buffer, result);
      return result;
    };
    const rows = analysis === 'single' ? COUNTY_COUNT : RECORD_COUNT;
    const binCount = SPATIAL_LAG_COUNT * (analysis === 'single' ? 1 : TEMPORAL_LAG_COUNT);
    const firstField = fields.unemployment;
    const firstValues = analysis === 'single' ? firstField[period!] : unemploymentRecords;
    const secondValues =
      mode === 'unemployment-poverty'
        ? analysis === 'single'
          ? fields.poverty[period!]
          : povertyRecords
        : null;
    const firstBuffer = resources.createBuffer(`${key}-unemployment-values`, firstValues);
    const secondBuffer = secondValues
      ? resources.createBuffer(`${key}-family-poverty-values`, secondValues)
      : null;
    const outputMeans = resources.createBuffer(`${key}-means`, binCount * 4);
    const outputCounts = resources.createBuffer(`${key}-pair-counts`, binCount * 4);
    const outputStatuses = resources.createBuffer(`${key}-status`, binCount * 4);
    const outputGeneration = resources.createBuffer(`${key}-request-generation`, 4);
    const commonProps = {
      id: key,
      spatialContext: projection.spatialContext,
      positions: view(
        'positions',
        analysis === 'single' ? singlePositionsBuffer : positionsBuffer,
        'float32x2',
        rows
      ),
      values: view('unemployment-values', firstBuffer, 'float32', rows),
      ...(secondBuffer
        ? {secondValues: view('family-poverty-values', secondBuffer, 'float32', rows)}
        : {}),
      parameters: cutoffParameter.importToGraph(graph, 'parameters'),
      lagCount: SPATIAL_LAG_COUNT,
      output: {
        semivariances: view('pair-bin-means', outputMeans, 'float32', binCount),
        pairCounts: view('pair-counts', outputCounts, 'uint32', binCount),
        status: view('bin-status', outputStatuses, 'uint32', binCount)
      }
    };
    if (analysis === 'space-time') {
      graph.add(
        new GPUSpaceTimeVariogram({
          ...commonProps,
          times: view('times', timesBuffer, 'float32', rows),
          temporalLagCount: TEMPORAL_LAG_COUNT
        })
      );
    } else {
      graph.add(new GPUCrossVariogram(commonProps));
    }
    const compiled = resources.track(await graph.compileAsync());
    const reader = new SummaryReader(
      resources,
      `pair-summary-${key}`,
      [
        {buffer: outputMeans, size: binCount * 4},
        {buffer: outputCounts, size: binCount * 4},
        {buffer: outputStatuses, size: binCount * 4},
        {buffer: outputGeneration, size: 4}
      ],
      bytes => {
        const resultGeneration = new Uint32Array(bytes, binCount * 12, 1)[0]!;
        if (active?.key !== key || activeGeneration !== resultGeneration) return;
        const values = new Float32Array(bytes, 0, binCount);
        const pairCounts = new Uint32Array(bytes, binCount * 4, binCount);
        const statusCodes = new Uint32Array(bytes, binCount * 8, binCount);
        publishResults(analysis, values, pairCounts, statusCodes);
        ctx.requestLayers();
      }
    );
    readers.push(reader);
    return {key, compiled, reader, requestGeneration: outputGeneration};
  };

  const runs: VariogramRun[] = [];
  for (const mode of ['unemployment', 'unemployment-poverty'] as const) {
    for (let period = 0; period < PERIOD_COUNT; period++)
      runs.push(await buildRun('single', mode, period));
    runs.push(await buildRun('space-time', mode));
  }
  for (const run of runs) outputSlots.set(run.key, run);

  const mapValues = resources.createBuffer('mapped-source-observation', fields.unemployment[0]!);
  const linkBuffer = resources.createBuffer(
    'selected-pair-support-links',
    ((RECORD_COUNT * (RECORD_COUNT - 1)) / 2) * 4 * 4,
    Buffer.VERTEX
  );
  let active: VariogramRun | undefined;
  let activeGeneration = 0;
  let generation = 0;
  let dirty = true;
  let linkCount = 0;
  let currentKey: VariogramKey | null = null;

  const formatBinDetails = (
    means: Float32Array,
    counts: Uint32Array,
    statuses: Uint32Array,
    temporal: boolean
  ) => {
    const details: string[] = [];
    for (let spatial = 0; spatial < SPATIAL_LAG_COUNT; spatial++) {
      const lowerDistanceKm = (spatial * Number(ctx.options.cutoff)) / SPATIAL_LAG_COUNT / 1000;
      const upperDistanceKm =
        ((spatial + 1) * Number(ctx.options.cutoff)) / SPATIAL_LAG_COUNT / 1000;
      const distanceInterval = `[${lowerDistanceKm.toFixed(1)}, ${upperDistanceKm.toFixed(1)}${spatial === SPATIAL_LAG_COUNT - 1 ? ']' : ')'} km`;
      for (let time = 0; time < (temporal ? TEMPORAL_LAG_COUNT : 1); time++) {
        const index = spatial * (temporal ? TEMPORAL_LAG_COUNT : 1) + time;
        const status = statuses[index];
        const estimate = status === 0 && counts[index] ? means[index]!.toFixed(3) : 'gap';
        const temporalLabel = temporal
          ? ` ${ACTUAL_TIME_LAGS[time]}y ${TEMPORAL_INTERVALS[time]}`
          : '';
        details.push(
          `${distanceInterval}${temporalLabel}: ${estimate}; n=${counts[index]}; status=${status}`
        );
      }
    }
    return details.join('\n');
  };

  function publishResults(
    analysis: 'single' | 'space-time',
    means: Float32Array,
    counts: Uint32Array,
    statuses: Uint32Array
  ): void {
    const temporal = analysis === 'space-time';
    const temporalCount = temporal ? TEMPORAL_LAG_COUNT : 1;
    const series = temporal
      ? ACTUAL_TIME_LAGS.map((lag, time) => ({
          label: `${lag}-year lag ${TEMPORAL_INTERVALS[time]}`,
          color: time,
          points: true,
          x: Float32Array.from(
            {length: SPATIAL_LAG_COUNT},
            (_, spatial) =>
              ((spatial + 0.5) * Number(ctx.options.cutoff)) / SPATIAL_LAG_COUNT / 1000
          ),
          y: Float32Array.from({length: SPATIAL_LAG_COUNT}, (_, spatial) => {
            const index = spatial * temporalCount + time;
            return statuses[index] === 0 && counts[index] > 0 ? means[index]! : Number.NaN;
          })
        }))
      : [
          {
            label:
              ctx.options.mode === 'unemployment'
                ? 'Unemployment semivariance'
                : 'Signed unemployment–poverty cross-semivariance',
            color: 0,
            points: true,
            x: Float32Array.from(
              {length: SPATIAL_LAG_COUNT},
              (_, spatial) =>
                ((spatial + 0.5) * Number(ctx.options.cutoff)) / SPATIAL_LAG_COUNT / 1000
            ),
            y: Float32Array.from({length: SPATIAL_LAG_COUNT}, (_, spatial) =>
              statuses[spatial] === 0 && counts[spatial] > 0 ? means[spatial]! : Number.NaN
            )
          }
        ];
    const cross = ctx.options.mode === 'unemployment-poverty';
    let summaryDomain = 0;
    for (let index = 0; index < means.length; index++) {
      if (statuses[index] === 0 && counts[index] > 0 && Number.isFinite(means[index]!))
        summaryDomain = Math.max(summaryDomain, Math.abs(means[index]!));
    }
    summaryDomain = summaryDomain || 1;
    ctx.setChart('variogram', {
      kind: 'line',
      title: temporal ? 'Space–time pair-bin summaries' : 'Single-period pair-bin summary',
      series,
      xLabel: 'Spatial lag midpoint (km)',
      yLabel: cross ? 'Signed cross-semivariance (pp²)' : 'Semivariance (pp²)',
      xDomain: [0, Number(ctx.options.cutoff) / 1000],
      yDomain: cross ? [-summaryDomain, summaryDomain] : [0, summaryDomain],
      guides: [{y: 0, label: 'zero'}],
      formatX: value => `${value.toFixed(0)} km`,
      formatY: value => value.toFixed(2),
      description: temporal
        ? 'GPU means for all eighteen spatial and temporal lag bins. The y-axis scales to valid returned bin means and may change with the distance cutoff; missing bins are gaps.'
        : 'GPU mean semivariance for six equal-width distance bins. The y-axis scales to valid returned bin means and may change with the distance cutoff; missing bins are gaps and signed cross-semivariance may be negative.',
      table: true
    });
    const totalPairs = counts.reduce((sum, value) => sum + value, 0);
    const validBins = statuses.reduce((sum, value) => sum + (value === 0 ? 1 : 0), 0);
    ctx.setReadout(
      'pairSummaryStatus',
      `${validBins} / ${statuses.length} nonempty finite bins; ${totalPairs} unordered pairs; GPU status 0=valid, 1=empty/invalid, 2=nonfinite`
    );
    ctx.setReadout('pairBinDetails', formatBinDetails(means, counts, statuses, temporal));
    ctx.setReadout(
      'pairSupport',
      `${linkCount} anchor links in selected spatial${temporal ? ' and temporal' : ''} band${temporal ? '; same-county positive-time pairs have zero-length links and remain in GPU counts' : ''}`
    );
  }

  const getActiveKey = (): VariogramKey => {
    if (ctx.options.analysis === 'space-time') return `space-time:${ctx.options.mode}:all`;
    return `single:${ctx.options.mode}:${ctx.options.period}`;
  };
  const refreshSourceMap = () => {
    const period = PERIOD_INDEX[ctx.options.period];
    mapValues.write(
      ctx.options.mapField === 'unemployment'
        ? fields.unemployment[period]!
        : fields.poverty[period]!
    );
    ctx.setReadout(
      'sourceObservation',
      ctx.options.mapField === 'unemployment'
        ? `unemployment${ctx.options.period}`
        : `familyPoverty${Number(ctx.options.period) - 1} (preceding year)`
    );
  };
  const updateLinks = () => {
    const cutoff = Number(ctx.options.cutoff);
    const temporal = ctx.options.analysis === 'space-time';
    const positions = temporal ? anchors : singlePositions;
    const rowCount = temporal ? RECORD_COUNT : COUNTY_COUNT;
    const segments: number[] = [];
    for (let first = 0; first < rowCount; first++) {
      for (let second = first + 1; second < rowCount; second++) {
        if (!temporal && Math.floor(first / COUNTY_COUNT) !== Math.floor(second / COUNTY_COUNT))
          continue;
        const dx = positions[first * 2]! - positions[second * 2]!;
        const dy = positions[first * 2 + 1]! - positions[second * 2 + 1]!;
        const distance = Math.hypot(dx, dy);
        if (distance > cutoff) continue;
        const spatialLag = Math.min(
          SPATIAL_LAG_COUNT - 1,
          Math.floor((distance / cutoff) * SPATIAL_LAG_COUNT)
        );
        if (spatialLag !== ctx.options.spatialLag) continue;
        const temporalLag = temporal
          ? Math.min(
              TEMPORAL_LAG_COUNT - 1,
              Math.floor((Math.abs(times[first]! - times[second]!) / 20) * TEMPORAL_LAG_COUNT)
            )
          : 0;
        if (temporal && temporalLag !== ctx.options.temporalLag) continue;
        segments.push(
          positions[first * 2]!,
          positions[first * 2 + 1]!,
          positions[second * 2]!,
          positions[second * 2 + 1]!
        );
      }
    }
    linkCount = segments.length / 4;
    const packed = Float32Array.from(segments);
    if (packed.length) linkBuffer.write(packed);
    else linkBuffer.write(new Float32Array([0, 0, 0, 0]));
    ctx.setReadout(
      'pairSupport',
      `${linkCount} anchor links in selected spatial${temporal ? ' and temporal' : ''} band${temporal ? '; same-county positive-time pairs have zero-length links and remain in GPU counts' : ''}`
    );
  };
  const selectRun = () => {
    const nextKey = getActiveKey();
    if (currentKey === nextKey && active) return;
    currentKey = nextKey;
    active = outputSlots.get(nextKey)!;
    generation++;
    activeGeneration = generation;
    dirty = true;
    ctx.setChart('variogram', null);
    ctx.setReadout('pairSummaryStatus', 'Waiting for the selected GPU pair summary');
    ctx.setReadout('pairBinDetails', '');
    ctx.setReadout('pairSupport', 'Waiting for selected pair support');
    updateLinks();
  };
  const updateCutoff = () => {
    cutoffParameter.write(new Float32Array([Number(ctx.options.cutoff), 20]));
    updateLinks();
    if (active) {
      generation++;
      activeGeneration = generation;
      dirty = true;
      ctx.setChart('variogram', null);
      ctx.setReadout('pairSummaryStatus', 'Waiting for the selected GPU pair summary');
      ctx.setReadout('pairBinDetails', '');
      ctx.setReadout('pairSupport', 'Waiting for selected pair support');
    }
  };

  refreshSourceMap();
  selectRun();
  updateCutoff();
  return {
    getCompiledGraphs: () => (active ? [active.compiled] : []),
    encode(commandEncoder: CommandEncoder) {
      selectRun();
      if (!active) return;
      if (dirty) {
        // This token is copied with the summary, so the callback can reject stale output.
        active.requestGeneration.write(new Uint32Array([activeGeneration]));
        active.compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
        active.reader.request(commandEncoder);
      }
      for (const run of runs) run.reader.flush(commandEncoder);
    },
    getLayers(): Layer[] {
      const povertyMap = ctx.options.mapField === 'poverty';
      return [
        new SpatialAnalysisPolygonLayer({
          id: 'county-cross-variogram-source-observations',
          analysisFrame: frame,
          triangles: polygonBuffers.triangles,
          features: polygonBuffers.triangleFeatures,
          vertexCount: polygonBuffers.vertexCount,
          values: mapValues,
          colormap: 'cividis',
          valueRange: povertyMap ? [0, 50] : [0, 20],
          noDataColor: [220, 220, 220, 210],
          opacity: 0.88
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'county-cross-variogram-selected-pair-support',
          analysisFrame: frame,
          segments: linkBuffer,
          instanceCount: linkCount,
          color: [54, 68, 79, 112],
          widthPixels: 0.8
        }),
        new SpatialAnalysisSegmentLayer({
          id: 'county-cross-variogram-county-outlines',
          analysisFrame: frame,
          segments: polygonBuffers.outline,
          instanceCount: polygonBuffers.outlineCount,
          color: [58, 63, 75, 155],
          widthPixels: 0.65
        })
      ];
    },
    getTooltip(event) {
      if (!event.coordinate) return null;
      const [x, y] = projection.project(event.coordinate[0], event.coordinate[1]);
      let nearestRow = -1;
      let nearestDistance = Infinity;
      for (let row = 0; row < COUNTY_COUNT; row++) {
        const distance = Math.hypot(
          singlePositions[row * 2]! - x,
          singlePositions[row * 2 + 1]! - y
        );
        if (distance < nearestDistance) {
          nearestRow = row;
          nearestDistance = distance;
        }
      }
      if (nearestRow < 0 || nearestDistance > 30_000) return null;
      const period = PERIOD_INDEX[ctx.options.period];
      const fieldValues =
        ctx.options.mapField === 'unemployment' ? fields.unemployment : fields.poverty;
      const sourceFieldName =
        ctx.options.mapField === 'unemployment'
          ? `unemployment${ctx.options.period}`
          : `familyPoverty${Number(ctx.options.period) - 1}`;
      const units = ctx.options.mapField === 'unemployment' ? '%' : '% below poverty line';
      return `Nearest authored county anchor: ${String(geojson.features[nearestRow]!.properties?.NAME ?? geojson.features[nearestRow]!.properties?.name ?? 'County unit')} · source FIPS ${sourceIds[nearestRow]}; ${sourceFieldName} ${fieldValues[period]![nearestRow]!.toFixed(2)} ${units}`;
    },
    setOption(id) {
      if (id === 'period' || id === 'mode' || id === 'analysis') selectRun();
      if (id === 'period' || id === 'mapField') refreshSourceMap();
      if (id === 'cutoff') updateCutoff();
      else if (id === 'spatialLag' || id === 'temporalLag') updateLinks();
      ctx.requestLayers();
    },
    destroy() {
      for (const reader of readers) reader.stop();
      resources.destroy();
    }
  };
}
