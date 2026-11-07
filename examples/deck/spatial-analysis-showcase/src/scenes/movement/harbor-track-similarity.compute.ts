// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUTrackSimilaritySearch} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {HARBOR_AUDITED_SOURCE_PATHS, loadHarborAuditedWindow} from './harbor-audited-window';

export type HarborSimilarityOptions = {
  metric: 'frechet' | 'dynamic-time-warping' | 'longest-common-subsequence';
  queryTrack: string;
  frechetThresholdMeters: number;
  dtwAccumulatedThresholdMeters: number;
  lcssSimilarityThreshold: number;
};

const TOP_K = 3;
const FRECHET = 'frechet' as const;
const DTW = 'dynamic-time-warping' as const;
const LCSS = 'longest-common-subsequence' as const;
const METRICS = [FRECHET, DTW, LCSS] as const;

type MetricResult = {
  values: Float32Array;
  bounds: Float32Array;
  matched: Uint32Array;
  status: Uint32Array;
  survivors: Uint32Array;
  topKIndices: Uint32Array;
  topKSummary: Uint32Array;
};

/** Search the eight audited harbor paths with bounded exact GPU Fréchet, DTW and LCSS measures. */
export async function createHarborTrackSimilarity(
  ctx: SceneContext<HarborSimilarityOptions>
): Promise<SceneInstance<HarborSimilarityOptions>> {
  const {tracks, mmsi, maximumObservedGaps} = loadHarborAuditedWindow(
    ctx.datasets.get('ais-vessels')
  );
  const {trackCount, vertexCount, segmentCount} = tracks;
  const pairA: number[] = [],
    pairB: number[] = [];
  for (let first = 0; first < trackCount; first++)
    for (let second = 0; second < trackCount; second++)
      if (first !== second) {
        pairA.push(first);
        pairB.push(second);
      }
  const pairCount = pairA.length;
  const resources = new SpatialAnalysisResources(ctx.device, 'harbor-track-similarity');
  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'harbor-track-similarity'});
  const view = createGraphBufferImporter(graph);
  const positionsBuffer = resources.createBuffer('audited-ais-positions', tracks.positions);
  const offsetsBuffer = resources.createBuffer('audited-ais-offsets', tracks.offsets);
  const segmentsBuffer = resources.createBuffer(
    'audited-ais-segments',
    tracks.segments,
    Buffer.VERTEX
  );
  const positionsView = view(positionsBuffer, 'float32x2', vertexCount);
  const offsetsView = view(offsetsBuffer, 'uint32', trackCount + 1);
  const pairABuffer = resources.createBuffer('query-path-rows', Uint32Array.from(pairA));
  const pairBBuffer = resources.createBuffer('candidate-path-rows', Uint32Array.from(pairB));
  const pairAView = view(pairABuffer, 'uint32', pairCount);
  const pairBView = view(pairBBuffer, 'uint32', pairCount);
  const lines = {kind: 'lines' as const, positions: positionsView, lineOffsets: offsetsView};
  const frechetThreshold = resources.createParameterBuffer(
    'frechet-distance-threshold-metres',
    'float32',
    1,
    Float32Array.of(ctx.options.frechetThresholdMeters)
  );
  const dtwThreshold = resources.createParameterBuffer(
    'dtw-accumulated-distance-threshold-metres',
    'float32',
    1,
    Float32Array.of(ctx.options.dtwAccumulatedThresholdMeters)
  );
  const similarityThreshold = resources.createParameterBuffer(
    'lcss-similarity-threshold',
    'float32',
    1,
    Float32Array.of(ctx.options.lcssSimilarityThreshold)
  );
  const frechetThresholdView = frechetThreshold.importToGraph(graph);
  const dtwThresholdView = dtwThreshold.importToGraph(graph);
  const similarityThresholdView = similarityThreshold.importToGraph(graph);
  const metricBuffers = new Map<
    (typeof METRICS)[number],
    {
      value: Buffer;
      bound: Buffer;
      matched: Buffer;
      status: Buffer;
      survivors: Buffer;
      indices: Buffer;
      topStatus: Buffer;
    }
  >();
  for (const metric of METRICS) {
    const buffers = {
      value: resources.createBuffer(`${metric}-value`, pairCount * 4),
      bound: resources.createBuffer(`${metric}-bound`, pairCount * 4),
      matched: resources.createBuffer(`${metric}-matched`, pairCount * 4),
      status: resources.createBuffer(`${metric}-status`, pairCount * 4),
      survivors: resources.createBuffer(`${metric}-survivor-summary`, 12),
      indices: resources.createBuffer(`${metric}-top-k-indices`, trackCount * TOP_K * 4),
      topStatus: resources.createBuffer(`${metric}-top-k-status`, trackCount * 8)
    };
    metricBuffers.set(metric, buffers);
    graph.add(
      new GPUTrackSimilaritySearch({
        id: `search-${metric}`,
        linesA: lines,
        pairA: pairAView,
        pairB: pairBView,
        metric,
        threshold:
          metric === LCSS
            ? similarityThresholdView
            : metric === FRECHET
              ? frechetThresholdView
              : dtwThresholdView,
        maximumFrechetVertexCount: 256,
        lcssTolerance: 150,
        survivorCapacity: pairCount,
        topK: TOP_K,
        output: {
          ...(metric === LCSS
            ? {
                similarity: view(buffers.value, 'float32', pairCount),
                upperBound: view(buffers.bound, 'float32', pairCount)
              }
            : {
                distance: view(buffers.value, 'float32', pairCount),
                lowerBound: view(buffers.bound, 'float32', pairCount)
              }),
          matched: view(buffers.matched, 'uint32', pairCount),
          status: view(buffers.status, 'uint32', pairCount),
          survivorSummary: view(buffers.survivors, 'uint32', 3),
          topKIndices: view(buffers.indices, 'uint32', trackCount * TOP_K),
          topKSummary: view(buffers.topStatus, 'uint32x2', trackCount)
        }
      })
    );
  }
  const compiled = resources.track(await graph.compileAsync());
  const segmentCapacity = Math.max(1, tracks.longestTrack - 1);
  const querySegmentsBuffer = resources.createBuffer(
    'query-track-overlay',
    new Float32Array(segmentCapacity * 4).fill(NaN),
    Buffer.VERTEX
  );
  const candidateSegmentsBuffer = resources.createBuffer(
    'candidate-track-overlay',
    new Float32Array(segmentCapacity * 4).fill(NaN),
    Buffer.VERTEX
  );
  const segmentsByTrack: Float32Array[] = [];
  const segmentCountsByTrack: number[] = [];
  for (let track = 0; track < trackCount; track++) {
    const first = tracks.offsets[track],
      count = tracks.offsets[track + 1] - first - 1;
    const firstSegment = first - track;
    segmentsByTrack.push(tracks.segments.slice(firstSegment * 4, (firstSegment + count) * 4));
    segmentCountsByTrack.push(count);
  }
  let generation = 1,
    pendingGeneration = 0,
    dirty = true,
    destroyed = false;
  let results = new Map<(typeof METRICS)[number], MetricResult>();
  let queryRows = new Float32Array(segmentCapacity * 4).fill(NaN);
  let candidateRows = new Float32Array(segmentCapacity * 4).fill(NaN);
  let ready = false;
  const allBuffers = METRICS.flatMap(metric => {
    const b = metricBuffers.get(metric)!;
    return [b.value, b.bound, b.matched, b.status, b.survivors, b.indices, b.topStatus];
  });
  const reader = new SummaryReader(
    resources,
    'track-similarity-summary',
    allBuffers.map(buffer => ({buffer, size: buffer.byteLength})),
    bytes => {
      if (destroyed || pendingGeneration !== generation) return;
      let offset = 0;
      const floats = (count: number) => {
        const result = new Float32Array(bytes).slice(offset / 4, offset / 4 + count);
        offset += count * 4;
        return result;
      };
      const words = (count: number) => {
        const result = new Uint32Array(bytes).slice(offset / 4, offset / 4 + count);
        offset += count * 4;
        return result;
      };
      results = new Map();
      for (const metric of METRICS) {
        const values = floats(pairCount),
          bounds = floats(pairCount),
          matched = words(pairCount),
          status = words(pairCount),
          survivors = words(3),
          topKIndices = words(trackCount * TOP_K),
          topKSummary = words(trackCount * 2);
        results.set(metric, {values, bounds, matched, status, survivors, topKIndices, topKSummary});
      }
      ready = true;
      publish();
    }
  );
  function drawSelection(bestTrack: number): void {
    queryRows.fill(NaN);
    candidateRows.fill(NaN);
    const query = Number(ctx.options.queryTrack);
    const copyTrack = (track: number, target: Float32Array) => {
      const sourceSegments = segmentsByTrack[track];
      const count = Math.min(segmentCountsByTrack[track], segmentCapacity);
      target.set(sourceSegments.subarray(0, count * 4));
    };
    copyTrack(query, queryRows);
    if (bestTrack >= 0) copyTrack(bestTrack, candidateRows);
    querySegmentsBuffer.write(queryRows);
    candidateSegmentsBuffer.write(candidateRows);
    ctx.requestLayers();
  }
  function publish(): void {
    if (!ready) return;
    const result = results.get(ctx.options.metric);
    if (!result) return;
    const query = Number(ctx.options.queryTrack);
    const firstRow = query * (trackCount - 1);
    const topRows: {target: number; value: number}[] = [];
    const start = query * TOP_K;
    for (let rank = 0; rank < TOP_K; rank++) {
      const row = result.topKIndices[start + rank];
      if (row === 0xffffffff) continue;
      topRows.push({target: pairB[row], value: result.values[row]});
    }
    const values = topRows.map(item => item.value);
    const isLCSS = ctx.options.metric === LCSS;
    const topKComplete = result.topKSummary[query * 2 + 1] !== 0;
    ctx.setReadout('queryPath', `Path ${HARBOR_AUDITED_SOURCE_PATHS[query]} · MMSI ${mmsi[query]}`);
    ctx.setReadout('candidateCount', `${trackCount - 1} candidates · top ${TOP_K} per query`);
    ctx.setReadout(
      'similarityMetric',
      isLCSS
        ? `LCSS · 150 m match tolerance · threshold ${ctx.options.lcssSimilarityThreshold.toFixed(2)}`
        : ctx.options.metric === FRECHET
          ? `discrete Fréchet · threshold ${(ctx.options.frechetThresholdMeters / 1000).toFixed(0)} km`
          : `DTW accumulated distance · threshold ${(ctx.options.dtwAccumulatedThresholdMeters / 1000).toFixed(0)} km`
    );
    ctx.setReadout(
      'searchStatus',
      `${result.survivors[0]} survivors · ${result.survivors[2]} over capacity · top-K ${topKComplete ? 'complete' : 'incomplete'}`
    );
    ctx.setReadout(
      'bestMatches',
      topKComplete
        ? topRows.length
          ? topRows
              .map(
                item =>
                  `Path ${HARBOR_AUDITED_SOURCE_PATHS[item.target]}: ${isLCSS ? item.value.toFixed(4) : `${(item.value / 1000).toFixed(3)} km`}`
              )
              .join(' · ')
          : 'No candidates meet the threshold'
        : 'withheld: incomplete search'
    );
    ctx.setChart(
      'similarityTopK',
      values.length && topKComplete
        ? {
            kind: 'bars',
            height: 122,
            labels: topRows.map(item => `Path ${HARBOR_AUDITED_SOURCE_PATHS[item.target]}`),
            values: isLCSS ? values : values.map(value => value / 1000),
            yLabel: isLCSS
              ? 'LCSS similarity'
              : ctx.options.metric === FRECHET
                ? 'Fréchet distance (km)'
                : 'DTW accumulated sum (km)',
            highlight: [0],
            description: `${ctx.options.metric} top threshold-matched candidates for selected query path.`
          }
        : null
    );
    const statuses = Array.from({length: trackCount - 1}, (_, index) => firstRow + index).map(
      row => result.status[row]
    );
    ctx.setReadout(
      'candidateStatus',
      `${statuses.filter(status => status === 0).length} exact · ${statuses.filter(status => status === 1).length} bound-pruned · ${statuses.filter(status => status > 1).length} invalid/capacity`
    );
    drawSelection(topKComplete ? (topRows[0]?.target ?? -1) : -1);
  }
  function update(): void {
    frechetThreshold.write(Float32Array.of(ctx.options.frechetThresholdMeters));
    dtwThreshold.write(Float32Array.of(ctx.options.dtwAccumulatedThresholdMeters));
    similarityThreshold.write(Float32Array.of(ctx.options.lcssSimilarityThreshold));
    generation++;
    dirty = true;
    ready = false;
    results.clear();
    reader.markStale();
    ctx.setReadout('searchStatus', 'updating GPU metric searches');
    ctx.setReadout('candidateStatus', 'updating');
    ctx.setReadout('bestMatches', 'updating');
    ctx.setChart('similarityTopK', null);
    ctx.requestLayers();
  }
  ctx.setReadout('windowAudit', '8 real passenger paths · 14:00–15:00 UTC · no extrapolation');
  ctx.setReadout(
    'samplingAudit',
    `Largest observed interval ${Math.max(...maximumObservedGaps)} s · 40–55 fixes per path`
  );
  return {
    getCompiledGraphs: () => [compiled],
    encode(encoder) {
      if (dirty) {
        compiled.encode(encoder, {parameters: undefined});
        dirty = false;
        reader.markStale();
      }
      if (!reader.isPending && generation !== pendingGeneration) pendingGeneration = generation;
      reader.flush(encoder);
    },
    setOption(id) {
      if (
        id === 'frechetThresholdMeters' ||
        id === 'dtwAccumulatedThresholdMeters' ||
        id === 'lcssSimilarityThreshold'
      )
        update();
      else if (id === 'metric') publish();
      else if (id === 'queryTrack') publish();
    },
    getLayers() {
      const context = new SpatialAnalysisSegmentLayer({
        id: 'similarity-all-audited-routes',
        analysisFrame: tracks.analysisFrame,
        segments: segmentsBuffer,
        instanceCount: segmentCount,
        widthPixels: 1.5,
        color: [114, 131, 144, 70]
      });
      const query = Number(ctx.options.queryTrack);
      const queryLayer = ready
        ? new SpatialAnalysisSegmentLayer({
            id: 'similarity-query-track',
            analysisFrame: tracks.analysisFrame,
            segments: querySegmentsBuffer,
            instanceCount: segmentCountsByTrack[query],
            widthPixels: 4.8,
            color: [250, 194, 71, 245],
            outlineColor: [17, 25, 30, 200],
            outlineWidthPixels: 0.8
          })
        : null;
      const candidate =
        ready &&
        results.get(ctx.options.metric)?.topKSummary[query * 2 + 1] &&
        results.get(ctx.options.metric)?.topKSummary[query * 2]
          ? new SpatialAnalysisSegmentLayer({
              id: 'similarity-best-threshold-match',
              analysisFrame: tracks.analysisFrame,
              segments: candidateSegmentsBuffer,
              instanceCount: Math.max(
                0,
                segmentCountsByTrack[
                  pairB[results.get(ctx.options.metric)!.topKIndices[query * TOP_K]]
                ]
              ),
              widthPixels: 3.6,
              color: [41, 218, 204, 245],
              outlineColor: [17, 25, 30, 185],
              outlineWidthPixels: 0.7
            })
          : null;
      return [
        context,
        ...(queryLayer ? [queryLayer] : []),
        ...(candidate ? [candidate] : [])
      ] as Layer[];
    },
    getTooltip(event) {
      const viewport = ctx.getViewport();
      if (!viewport || !ready) return null;
      let nearest = -1,
        closest = 15 * 15;
      for (let track = 0; track < trackCount; track++) {
        const first = tracks.offsets[track];
        const point = viewport.project([tracks.lngLat[first * 2], tracks.lngLat[first * 2 + 1]]);
        const distance = (point[0] - event.pixel[0]) ** 2 + (point[1] - event.pixel[1]) ** 2;
        if (distance < closest) {
          closest = distance;
          nearest = track;
        }
      }
      if (nearest < 0) return null;
      const result = results.get(ctx.options.metric)!;
      const query = Number(ctx.options.queryTrack);
      const row = pairA.findIndex(
        (candidate, index) => candidate === query && pairB[index] === nearest
      );
      if (!result.topKSummary[query * 2 + 1])
        return {
          title: 'Similarity search pending',
          subtitle: 'Top-K result is incomplete',
          rows: []
        };
      const status = row >= 0 ? result.status[row] : 2;
      const metricLabel =
        ctx.options.metric === LCSS
          ? 'LCSS similarity'
          : ctx.options.metric === FRECHET
            ? 'Discrete Fréchet'
            : 'DTW accumulated distance';
      const value =
        row < 0 || status !== 0
          ? status === 1
            ? 'pruned by threshold bound'
            : 'not evaluated / invalid'
          : ctx.options.metric === LCSS
            ? result.values[row].toFixed(3)
            : `${(result.values[row] / 1000).toFixed(2)} km`;
      return {
        title: `Path ${HARBOR_AUDITED_SOURCE_PATHS[nearest]} · MMSI ${mmsi[nearest]}`,
        subtitle: '14:00–15:00 UTC AIS support',
        rows: [
          {
            label: 'Query path',
            value: `Path ${HARBOR_AUDITED_SOURCE_PATHS[query]} · MMSI ${mmsi[query]}`
          },
          {label: metricLabel, value},
          {
            label: 'Threshold match',
            value: row >= 0 ? (result.matched[row] ? 'yes' : 'no') : 'not a query pair'
          },
          {label: 'Largest observed interval', value: `${maximumObservedGaps[nearest]} s`}
        ],
        note:
          ctx.options.metric === LCSS
            ? 'LCSS counts ordered matches within a 150 m tolerance and normalizes by the shorter path. A larger value is more similar.'
            : 'Fréchet is the maximum leash distance; DTW reports the accumulated sum along its minimum-cost alignment. Lower values are more similar. Both use planar metre coordinates.'
      };
    },
    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
    }
  };
}
