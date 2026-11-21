// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {ScatterplotLayer} from '@deck.gl/layers';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type CompiledGPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUHDBSCAN, GPUOPTICS} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {CHICAGO, findPlace} from '../../cartography/gazetteer';
import {formatCount, formatDistance} from '../../cartography/live-text';
import type {LngLat} from '../../cartography/types';
import {createAnalysisFrameProjection} from '../../engine/analysis-frame';
import {GraphSwap} from '../../engine/graph-swap';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {BarChartData, LineChartData} from '../chart-types';
import type {SceneContext, SceneInstance, TooltipContent} from '../scene';
import {
  CLUSTER_COLORS,
  NOISE_COLOR,
  UNRESOLVED_COLOR,
  type ObservationHierarchyOptions
} from './observation-hierarchy.shared';
import {buildObservationHierarchyData} from './observation-hierarchy-data';

type PatternResult = {
  labels: Uint32Array;
  coreDistances: Float32Array;
  ordering: Uint32Array | null;
  reachability: Float32Array | null;
  predecessors: Uint32Array | null;
  visitedCount: number;
  complete: boolean;
  invalidRows: number;
  clusterCount: number;
  overflow: number;
  intervals: Uint32Array | null;
};

type PatternVariant = {
  key: string;
  generation: number;
  compiled: CompiledGPUCommandGraph<void>;
  resources: SpatialAnalysisResources;
  reader: SummaryReader;
  buffers: {
    ordering: Buffer;
    coreDistances: Buffer;
    reachability: Buffer;
    predecessors: Buffer;
    labels: Buffer;
    clusterCount: Buffer;
    iterationCount: Buffer;
    converged: Buffer;
    invalidCount: Buffer;
    intervals: Buffer;
    summary: Buffer;
  };
  result: PatternResult | null;
  submitted: boolean;
};

const FRAME: readonly [number, number] = [-87.607, 41.8595];
let variantSerial = 0;

function getVariantKey(options: ObservationHierarchyOptions): string {
  return [
    options.algorithm,
    options.minimumSamples,
    options.minimumClusterSize,
    options.extractionDistance,
    options.xi,
    options.predecessorCorrection
  ].join('|');
}

function decodePatternResult(
  bytes: ArrayBuffer,
  rowCount: number,
  algorithm: ObservationHierarchyOptions['algorithm']
): PatternResult {
  const words = new Uint32Array(bytes);
  const values = new Float32Array(bytes);
  let offset = 0;
  if (algorithm === 'hdbscan') {
    const labels = words.slice(offset, offset + rowCount);
    offset += rowCount;
    const coreDistances = values.slice(offset, offset + rowCount);
    offset += rowCount;
    return {
      labels,
      coreDistances,
      ordering: null,
      reachability: null,
      predecessors: null,
      visitedCount: rowCount,
      complete: true,
      invalidRows: 0,
      clusterCount: words[offset],
      overflow: 0,
      intervals: null
    };
  }
  const ordering = words.slice(offset, offset + rowCount);
  offset += rowCount;
  const coreDistances = values.slice(offset, offset + rowCount);
  offset += rowCount;
  const reachability = values.slice(offset, offset + rowCount);
  offset += rowCount;
  const predecessors = words.slice(offset, offset + rowCount);
  offset += rowCount;
  const labels = words.slice(offset, offset + rowCount);
  offset += rowCount;
  const visitedCount = words[offset];
  const complete = words[offset + 1] === 1;
  const invalidRows = words[offset + 2];
  offset += 3;
  if (algorithm === 'optics-xi') {
    const intervals = words.slice(offset, offset + rowCount * 2);
    offset += rowCount * 2;
    return {
      labels,
      coreDistances,
      ordering,
      reachability,
      predecessors,
      visitedCount,
      complete,
      invalidRows,
      clusterCount: words[offset],
      overflow: words[offset + 1],
      intervals
    };
  }
  return {
    labels,
    coreDistances,
    ordering,
    reachability,
    predecessors,
    visitedCount,
    complete,
    invalidRows,
    clusterCount: 0,
    overflow: 0,
    intervals: null
  };
}

export async function createObservationHierarchy(
  context: SceneContext<ObservationHierarchyOptions>
): Promise<SceneInstance<ObservationHierarchyOptions>> {
  const dataset = context.datasets.get('chicago-nature');
  const projection = createAnalysisFrameProjection({kind: 'azimuthal-equidistant', origin: FRAME});
  const data = buildObservationHierarchyData(dataset, projection);
  const rowCount = data.sourceRows.length;
  const resources = new SpatialAnalysisResources(context.device, 'observation-hierarchy-scene');
  const positionBuffer = resources.createBuffer('northerly-bird-positions', data.positions);
  const graphSwap = resources.track(new GraphSwap());
  let active: PatternVariant | null = null;
  const variantCache = new Map<string, PatternVariant>();
  let displayed: PatternResult | null = null;
  let displayedLabelsValid = false;
  let currentGeneration = 0;
  let destroyed = false;
  const rowPoints = Array.from({length: rowCount}, (_, index) => ({
    row: index,
    position: [data.longitudeLatitude[index * 2], data.longitudeLatitude[index * 2 + 1]] as LngLat,
    multiplicity: data.coordinateMultiplicity[index]
  }));
  const sharedCoordinateMap = new Map<string, (typeof rowPoints)[number]>();
  for (const point of rowPoints) {
    if (point.multiplicity > 1) {
      const coordinateKey = `${point.position[0]},${point.position[1]}`;
      if (!sharedCoordinateMap.has(coordinateKey)) sharedCoordinateMap.set(coordinateKey, point);
    }
  }
  const sharedCoordinatePoints = [...sharedCoordinateMap.values()];

  context.setReadout('archiveRows', `${formatCount(data.archiveRowCount)} archived rows`);
  context.setReadout('frameRows', `${formatCount(rowCount)} bird records in the mapped extent`);
  context.setReadout(
    'coordinateReuse',
    `${formatCount(data.uniqueCoordinateCount)} coordinate pairs; ${formatCount(data.recordsAtSharedCoordinates)} records share a position (max ${data.maximumCoordinateMultiplicity})`
  );
  const northerlyIsland = findPlace(CHICAGO, 'northerly-island');
  context.setAnnotations(
    'place',
    northerlyIsland
      ? [
          {
            kind: 'note',
            id: 'northerly-island-place',
            coordinate: northerlyIsland.lngLat,
            title: northerlyIsland.name,
            tone: 'muted'
          }
        ]
      : null
  );

  function publishResult(result: PatternResult, options: ObservationHierarchyOptions): void {
    displayed = result;
    displayedLabelsValid =
      options.algorithm === 'hdbscan' ||
      (result.complete &&
        result.invalidRows === 0 &&
        (options.algorithm !== 'optics-xi' || result.overflow === 0));
    const clusterSizes = new Map<number, number>();
    let noiseCount = 0;
    if (displayedLabelsValid) {
      for (const label of result.labels) {
        if (label === 0xffffffff) noiseCount++;
        else clusterSizes.set(label, (clusterSizes.get(label) ?? 0) + 1);
      }
    }
    const clusterLabels = [...clusterSizes.keys()].sort((first, second) => first - second);
    context.setLegendData('clusterLabels', clusterLabels);
    context.setLegendData('hasNoise', noiseCount > 0);
    context.setLegendData('labelsComplete', displayedLabelsValid);
    const countLabels = ['Noise', ...clusterLabels.map(label => `Cluster ${label + 1}`)];
    const counts = [noiseCount, ...clusterLabels.map(label => clusterSizes.get(label) ?? 0)];
    const countChart: BarChartData = {
      kind: 'bars',
      labels: countLabels,
      values: counts,
      color: 0,
      height: 125,
      yLabel: 'records',
      description:
        'Actual GPU-returned cluster labels summarized as record counts; noise is shown separately.'
    };
    context.setChart('clusterSizes', displayedLabelsValid ? countChart : null);

    if (result.ordering && result.reachability) {
      const ordering = result.ordering.subarray(0, result.visitedCount);
      const reachabilityValues = Array.from(ordering, row => {
        const value = result.reachability![row];
        return Number.isFinite(value) ? value : Number.NaN;
      });
      const reachabilityChart: LineChartData = {
        kind: 'line',
        title: 'OPTICS reachability ordering',
        xLabel: 'visit rank',
        yLabel: 'reachability distance',
        height: 165,
        series: [{label: 'GPU reachability', y: reachabilityValues, color: 0, width: 1.5}],
        formatX: value => formatCount(value),
        formatY: value => (Number.isFinite(value) ? formatDistance(value) : ''),
        description: displayedLabelsValid
          ? 'GPU OPTICS visit ordering. Infinite root distances are rendered as gaps; the input-row reachability array is reordered only for this chart.'
          : 'Partial GPU OPTICS visit ordering; infinite root distances are gaps. Cluster labels are withheld because completion, validity or interval-capacity status failed.'
      };
      context.setChart('reachabilityChart', reachabilityChart);
      context.setChart('coreDistanceChart', null);
    } else {
      context.setChart('reachabilityChart', null);
      const finiteCoreDistances = Array.from(result.coreDistances)
        .filter(Number.isFinite)
        .sort((firstDistance, secondDistance) => firstDistance - secondDistance);
      const binCount = 12;
      const maximumCoreDistance = finiteCoreDistances.at(-1) ?? 1;
      const binWidth = maximumCoreDistance > 0 ? maximumCoreDistance / binCount : 1;
      const bins = new Uint32Array(binCount);
      finiteCoreDistances.forEach(distance => {
        bins[Math.min(binCount - 1, Math.floor(distance / binWidth))]++;
      });
      const coreChart: BarChartData = {
        kind: 'histogram',
        values: bins,
        xDomain: [0, maximumCoreDistance || 1],
        xLabel: 'core distance',
        yLabel: 'records',
        height: 145,
        color: 1,
        formatX: value => formatDistance(value),
        description:
          'Histogram of the core distances returned by GPU HDBSCAN; non-finite core distances are omitted.'
      };
      context.setChart('coreDistanceChart', coreChart);
    }

    const clusterCount = clusterSizes.size;
    const status =
      options.algorithm === 'hdbscan'
        ? `HDBSCAN EOM completed for ${formatCount(rowCount)} rows; ${formatCount(result.clusterCount)} selected clusters`
        : `OPTICS visited ${formatCount(result.visitedCount)} of ${formatCount(rowCount)} rows; ${result.complete ? 'complete' : 'incomplete'}; ${formatCount(result.invalidRows)} invalid rows${options.algorithm === 'optics-xi' ? `; ${formatCount(result.clusterCount)} xi intervals, ${formatCount(result.overflow)} overflow` : ''}${displayedLabelsValid ? '' : '; labels withheld'}`;
    context.setReadout('algorithmStatus', status);
    context.setReadout(
      'clusterSummary',
      displayedLabelsValid
        ? `${formatCount(clusterCount)} clusters, ${formatCount(noiseCount)} noise records; sizes ${clusterLabels.map(label => formatCount(clusterSizes.get(label) ?? 0)).join(', ') || 'none'}`
        : 'Cluster labels withheld until GPU completion, validity and extraction status are clear.'
    );
    context.requestLayers();
  }

  async function buildVariant(options: ObservationHierarchyOptions): Promise<PatternVariant> {
    const buildNumber = variantSerial++;
    const key = getVariantKey(options);
    const variantResources = new SpatialAnalysisResources(
      context.device,
      `observation-hierarchy-${buildNumber}`
    );
    try {
      const graph = new GPUCommandGraph<void>(context.device, {
        id: `observation-hierarchy-${buildNumber}-${options.algorithm}`
      });
      const positionView = importGraphBuffer(
        graph,
        'input-positions',
        positionBuffer,
        'float32x2',
        rowCount
      );
      const createOutputView = <Format extends 'float32' | 'uint32' | 'uint32x2'>(
        name: string,
        buffer: Buffer,
        format: Format,
        length: number
      ) => importGraphBuffer(graph, name, buffer, format, length);
      const buffers = {
        ordering: variantResources.createBuffer('ordering', rowCount * 4),
        coreDistances: variantResources.createBuffer('core-distances', rowCount * 4),
        reachability: variantResources.createBuffer('reachability', rowCount * 4),
        predecessors: variantResources.createBuffer('predecessors', rowCount * 4),
        labels: variantResources.createBuffer('labels', rowCount * 4),
        clusterCount: variantResources.createBuffer('cluster-count', 4),
        iterationCount: variantResources.createBuffer('optics-iteration-count', 4),
        converged: variantResources.createBuffer('optics-converged', 4),
        invalidCount: variantResources.createBuffer('optics-invalid-count', 4),
        intervals: variantResources.createBuffer('xi-intervals', rowCount * 8),
        summary: variantResources.createBuffer('xi-summary', 2 * 4)
      };
      if (options.algorithm === 'hdbscan') {
        graph.add(
          new GPUHDBSCAN({
            id: 'hdbscan-eom',
            spatialContext: projection.spatialContext,
            positions: positionView,
            minimumSamples: Number(options.minimumSamples),
            minimumClusterSize: Number(options.minimumClusterSize),
            output: {
              labels: createOutputView('labels', buffers.labels, 'uint32', rowCount),
              coreDistances: createOutputView(
                'core-distances',
                buffers.coreDistances,
                'float32',
                rowCount
              ),
              status: {count: createOutputView('cluster-count', buffers.clusterCount, 'uint32', 1)}
            }
          })
        );
      } else {
        const isXi = options.algorithm === 'optics-xi';
        graph.add(
          new GPUOPTICS({
            id: isXi ? 'optics-xi' : 'optics-dbscan',
            positions: positionView,
            minimumSamples: Number(options.minimumSamples),
            maximumDistance: 1000,
            extractionMethod: isXi ? 'xi' : 'dbscan',
            ...(isXi
              ? {
                  xi: Number(options.xi),
                  minimumClusterSize: Number(options.minimumClusterSize),
                  predecessorCorrection: options.predecessorCorrection
                }
              : {extractionDistance: Number(options.extractionDistance)}),
            output: {
              ordering: createOutputView('ordering', buffers.ordering, 'uint32', rowCount),
              coreDistances: createOutputView(
                'core-distances',
                buffers.coreDistances,
                'float32',
                rowCount
              ),
              reachability: createOutputView(
                'reachability',
                buffers.reachability,
                'float32',
                rowCount
              ),
              predecessors: createOutputView(
                'predecessors',
                buffers.predecessors,
                'uint32',
                rowCount
              ),
              labels: createOutputView('labels', buffers.labels, 'uint32', rowCount),
              ports: {
                status: {
                  iterationCount: createOutputView(
                    'optics-iteration-count',
                    buffers.iterationCount,
                    'uint32',
                    1
                  ),
                  converged: createOutputView('optics-converged', buffers.converged, 'uint32', 1),
                  invalidCount: createOutputView(
                    'optics-invalid-count',
                    buffers.invalidCount,
                    'uint32',
                    1
                  )
                }
              },
              ...(isXi
                ? {
                    clusterIntervals: createOutputView(
                      'xi-intervals',
                      buffers.intervals,
                      'uint32x2',
                      rowCount
                    ),
                    clusterSummary: createOutputView('xi-summary', buffers.summary, 'uint32', 2)
                  }
                : {})
            }
          })
        );
      }
      const compiled = variantResources.track(await graph.compileAsync());
      let variant: PatternVariant | null = null;
      const readerBuffers =
        options.algorithm === 'hdbscan'
          ? [
              {buffer: buffers.labels, size: rowCount * 4},
              {buffer: buffers.coreDistances, size: rowCount * 4},
              {buffer: buffers.clusterCount, size: 4}
            ]
          : [
              {buffer: buffers.ordering, size: rowCount * 4},
              {buffer: buffers.coreDistances, size: rowCount * 4},
              {buffer: buffers.reachability, size: rowCount * 4},
              {buffer: buffers.predecessors, size: rowCount * 4},
              {buffer: buffers.labels, size: rowCount * 4},
              {buffer: buffers.iterationCount, size: 4},
              {buffer: buffers.converged, size: 4},
              {buffer: buffers.invalidCount, size: 4},
              ...(options.algorithm === 'optics-xi'
                ? [
                    {buffer: buffers.intervals, size: rowCount * 8},
                    {buffer: buffers.summary, size: 2 * 4}
                  ]
                : [])
            ];
      const reader = new SummaryReader(variantResources, 'output', readerBuffers, bytes => {
        if (!variant) return;
        const result = decodePatternResult(bytes, rowCount, options.algorithm);
        variant.result = result;
        if (!destroyed && active === variant && currentGeneration === variant.generation)
          publishResult(result, options);
      });
      variant = {
        key,
        generation: 0,
        compiled,
        resources: variantResources,
        reader,
        buffers,
        result: null,
        submitted: false
      };
      variantCache.set(key, variant);
      while (variantCache.size > 8) {
        const oldestEntry = [...variantCache.entries()].find(
          ([cachedKey, cachedVariant]) => cachedKey !== key && cachedVariant !== active
        );
        if (!oldestEntry) break;
        variantCache.delete(oldestEntry[0]);
        oldestEntry[1].reader.stop();
        oldestEntry[1].resources.destroy();
      }
      return variant;
    } catch (error) {
      variantResources.destroy();
      throw error;
    }
  }

  async function requestVariant(options: ObservationHierarchyOptions): Promise<boolean> {
    const snapshot = {...options};
    const generation = ++currentGeneration;
    const key = getVariantKey(snapshot);
    context.setReadout('algorithmStatus', 'Compiling GPU clustering variant…');
    context.setLegendData('labelsComplete', false);
    return graphSwap.request({
      build: async () => variantCache.get(key) ?? buildVariant(snapshot),
      commit: next => {
        if (
          destroyed ||
          generation !== currentGeneration ||
          key !== getVariantKey(context.options)
        ) {
          if (active !== next) {
            if (variantCache.get(next.key) === next) variantCache.delete(next.key);
            next.reader.stop();
            next.resources.destroy();
          }
          return;
        }
        active = next;
        next.generation = generation;
        displayed = null;
        if (next.result) {
          publishResult(next.result, snapshot);
        } else {
          context.setReadout('clusterSummary', null);
          context.setChart('reachabilityChart', null);
          context.setChart('clusterSizes', null);
          context.setChart('coreDistanceChart', null);
          context.setReadout(
            'algorithmStatus',
            'GPU graph ready; waiting for its first output readback'
          );
        }
        context.requestLayers();
      },
      discard: next => {
        if (variantCache.get(next.key) !== next) {
          next.reader.stop();
          next.resources.destroy();
        }
      }
    });
  }

  await requestVariant(context.options);

  function getClusterColor(label: number): readonly [number, number, number, number] {
    return label === 0xffffffff ? NOISE_COLOR : CLUSTER_COLORS[label % CLUSTER_COLORS.length];
  }

  return {
    getCompiledGraphs: () => (active ? [active.compiled] : []),
    setOption(_id, _value, state) {
      void requestVariant(state).catch(() => {
        if (!destroyed) {
          context.setReadout(
            'algorithmStatus',
            'GPU graph compilation failed; the previous result remains visible.'
          );
        }
      });
    },
    encode(commandEncoder) {
      const variant = active;
      if (!variant) return;
      variant.reader.flush(commandEncoder);
      if (!variant.submitted) {
        variant.compiled.encode(commandEncoder, {parameters: undefined});
        variant.reader.request(commandEncoder);
        variant.submitted = true;
      }
    },
    getLayers() {
      const result = displayed;
      if (!result) return [];
      const fillColors = Array.from(result.labels, label =>
        displayedLabelsValid ? getClusterColor(label) : UNRESOLVED_COLOR
      );
      const pointLayer = new ScatterplotLayer<(typeof rowPoints)[number]>({
        id: 'northerly-bird-cluster-observations',
        data: rowPoints,
        getPosition: point => point.position,
        getFillColor: point => fillColors[point.row],
        getRadius: 3.1,
        radiusUnits: 'pixels',
        radiusMinPixels: 2.5,
        radiusMaxPixels: 4.5,
        stroked: false,
        pickable: false,
        updateTriggers: {getFillColor: [result]}
      });
      const sharedLocationLayer = new ScatterplotLayer<(typeof rowPoints)[number]>({
        id: 'northerly-shared-observation-coordinates',
        data: sharedCoordinatePoints,
        getPosition: point => point.position,
        getFillColor: [0, 0, 0, 0],
        getLineColor: [34, 77, 85, 210],
        getRadius: 5,
        radiusUnits: 'pixels',
        lineWidthUnits: 'pixels',
        getLineWidth: 1,
        filled: false,
        stroked: true,
        pickable: false
      });
      return [pointLayer, sharedLocationLayer];
    },
    getTooltip(event): TooltipContent | null {
      if (!event.coordinate || !displayed) return null;
      const [projectedX, projectedY] = projection.project(event.coordinate[0], event.coordinate[1]);
      let nearestRow = -1;
      let nearestDistance = Infinity;
      for (let row = 0; row < rowCount; row++) {
        const horizontalDifference = data.positions[row * 2] - projectedX;
        const verticalDifference = data.positions[row * 2 + 1] - projectedY;
        const distance =
          horizontalDifference * horizontalDifference + verticalDifference * verticalDifference;
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearestRow = row;
        }
      }
      if (nearestRow < 0 || nearestDistance > Math.max(25, (context.getMetersPerPixel() * 10) ** 2))
        return null;
      const label = displayed.labels[nearestRow];
      const clusterName = displayedLabelsValid
        ? label === 0xffffffff
          ? 'Noise'
          : `Cluster ${label + 1}`
        : 'Label withheld: incomplete status';
      const reachability = displayed.reachability?.[nearestRow];
      const coreDistance = displayed.coreDistances[nearestRow];
      const orderingRank = displayed.ordering?.indexOf(nearestRow) ?? -1;
      const predecessor = displayed.predecessors?.[nearestRow] ?? 0xffffffff;
      return {
        title: `Bird record ${data.sourceRows[nearestRow]}`,
        subtitle: clusterName,
        rows: [
          {label: 'Category', value: 'Birds'},
          {label: 'Coordinate rows here', value: String(data.coordinateMultiplicity[nearestRow])},
          {
            label: 'Core distance',
            value: Number.isFinite(coreDistance) ? formatDistance(coreDistance) : 'not core'
          },
          ...(typeof reachability === 'number' && Number.isFinite(reachability)
            ? [{label: 'Reachability', value: formatDistance(reachability)}]
            : reachability === Number.POSITIVE_INFINITY
              ? [{label: 'Reachability', value: 'root / no predecessor'}]
              : []),
          ...(orderingRank >= 0
            ? [{label: 'OPTICS visit rank', value: String(orderingRank + 1)}]
            : []),
          ...(displayed.predecessors
            ? [
                {
                  label: 'OPTICS predecessor',
                  value:
                    predecessor === 0xffffffff
                      ? 'none (root)'
                      : `source row ${data.sourceRows[predecessor]}`
                }
              ]
            : [])
        ],
        highlight: {
          kind: 'point',
          coordinate: [
            data.longitudeLatitude[nearestRow * 2],
            data.longitudeLatitude[nearestRow * 2 + 1]
          ]
        }
      };
    },
    destroy() {
      destroyed = true;
      for (const variant of variantCache.values()) {
        variant.reader.stop();
        variant.resources.destroy();
      }
      resources.destroy();
    }
  };
}
