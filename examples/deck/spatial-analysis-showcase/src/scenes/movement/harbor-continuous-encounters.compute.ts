// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  addClockEncounters,
  getGPUTrajectoryClockParameterValues,
  GPUTrajectoryClosestApproach,
  GPUTrajectoryRestriction,
  type GPUParameterBuffer
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {createGraphBufferImporter} from '../../engine/graph-buffers';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import {formatUtcClockSeconds, VESSEL_CATEGORIES, VESSEL_CATEGORY_LABELS} from './b12-tracks';
import {
  HARBOR_AUDITED_SOURCE_PATHS as SOURCE_PATHS,
  HARBOR_AUDITED_WINDOW_END as WINDOW_END,
  HARBOR_AUDITED_WINDOW_START as WINDOW_START,
  loadHarborAuditedWindow
} from './harbor-audited-window';
import type {SceneContext, SceneInstance} from '../scene';

export type HarborContinuousOptions = {
  distanceMeters: number;
  maximumGapSeconds: number;
  clockStep: string;
  timeBoundary: 'closed' | 'open' | 'left-closed' | 'right-closed';
  geometryBoundary: 'closed' | 'open';
};

const CLOCK_STEPS = [60, 300, 600] as const;
const PAIR_CAPACITY = (SOURCE_PATHS.length * (SOURCE_PATHS.length - 1)) / 2;
const INTERVALS_PER_PAIR = 64;
const TIME_BOUNDARIES = ['closed', 'open', 'left-closed', 'right-closed'] as const;
const GEOMETRY_BOUNDARIES = ['closed', 'open'] as const;

type BoundaryKey = `${(typeof TIME_BOUNDARIES)[number]}:${(typeof GEOMETRY_BOUNDARIES)[number]}`;

function makePairRows(count: number): readonly (readonly [number, number])[] {
  const pairs: [number, number][] = [];
  for (let first = 0; first < count; first++)
    for (let second = first + 1; second < count; second++) pairs.push([first, second]);
  return pairs;
}

/** Compare exact linear-segment encounters with common-clock samples on a bounded real AIS hour. */
export async function createHarborContinuousEncounters(
  ctx: SceneContext<HarborContinuousOptions>
): Promise<SceneInstance<HarborContinuousOptions>> {
  const {tracks, categories, mmsi, maximumObservedGaps, firstFixTimes, sourceRows} =
    loadHarborAuditedWindow(ctx.datasets.get('ais-vessels'));
  const {offsets, positions, timestamps: times} = tracks;
  const {trackCount, vertexCount, segmentCount} = tracks;
  const pairs = makePairRows(trackCount);
  const resources = new SpatialAnalysisResources(ctx.device, 'harbor-continuous-encounters');
  const graph = new GPUCommandGraph<void>(ctx.device, {id: 'harbor-continuous-encounters'});
  const view = createGraphBufferImporter(graph);
  const positionsBuffer = resources.createBuffer('ais-hour-positions', positions);
  const timesBuffer = resources.createBuffer('ais-hour-times', times);
  const offsetsBuffer = resources.createBuffer('ais-hour-offsets', offsets);
  const sourceSegmentsBuffer = resources.createBuffer(
    'ais-hour-segments',
    tracks.segments,
    Buffer.VERTEX
  );
  const positionsView = view(positionsBuffer, 'float32x2', vertexCount);
  const timesView = view(timesBuffer, 'float32', vertexCount);
  const offsetsView = view(offsetsBuffer, 'uint32', trackCount + 1);
  const distanceBuffer = resources.createParameterBuffer(
    'encounter-distance',
    'float32',
    1,
    Float32Array.of(ctx.options.distanceMeters)
  );
  const distanceView = distanceBuffer.importToGraph(graph);
  const pairIds = resources.createBuffer('closest-pair-ids', PAIR_CAPACITY * 4);
  const pairPartners = resources.createBuffer('closest-pair-partners', PAIR_CAPACITY * 4);
  const pairMinimum = resources.createBuffer('closest-pair-minimum-distance', PAIR_CAPACITY * 4);
  const pairClosestTimes = resources.createBuffer('closest-pair-closest-time', PAIR_CAPACITY * 4);
  const pairFirstTimes = resources.createBuffer('closest-pair-first-time', PAIR_CAPACITY * 4);
  const pairDurations = resources.createBuffer('closest-pair-duration', PAIR_CAPACITY * 4);
  const pairCount = resources.createBuffer('closest-pair-count', 4);
  const pairRequired = resources.createBuffer('closest-pair-required', 4);
  const pairOverflow = resources.createBuffer('closest-pair-overflow', 4);
  const entryOverflow = resources.createBuffer('closest-entry-overflow', 4);
  const hitOverflow = resources.createBuffer('closest-hit-overflow', 4);
  const sampledOutputs = new Map<
    number,
    {
      clock: GPUParameterBuffer<'float32'>;
      ids: Buffer;
      partners: Buffer;
      buckets: Buffer;
      minimum: Buffer;
      bucketCounts: Buffer;
      count: Buffer;
      required: Buffer;
      overflow: Buffer;
      candidateOverflow: Buffer;
    }
  >();
  for (const step of CLOCK_STEPS) {
    sampledOutputs.set(step, {
      clock: resources.createParameterBuffer(
        `encounter-clock-${step}s`,
        'float32',
        4,
        getGPUTrajectoryClockParameterValues({start: 0, step})
      ),
      ids: resources.createBuffer(`sampled-${step}s-pair-ids`, PAIR_CAPACITY * 4),
      partners: resources.createBuffer(`sampled-${step}s-pair-partners`, PAIR_CAPACITY * 4),
      buckets: resources.createBuffer(`sampled-${step}s-first-buckets`, PAIR_CAPACITY * 4),
      minimum: resources.createBuffer(`sampled-${step}s-minimum-distance`, PAIR_CAPACITY * 4),
      bucketCounts: resources.createBuffer(`sampled-${step}s-bucket-counts`, PAIR_CAPACITY * 4),
      count: resources.createBuffer(`sampled-${step}s-pair-count`, 4),
      required: resources.createBuffer(`sampled-${step}s-pair-required`, 4),
      overflow: resources.createBuffer(`sampled-${step}s-pair-overflow`, 4),
      candidateOverflow: resources.createBuffer(`sampled-${step}s-candidate-overflow`, 4)
    });
  }
  const restrictionParameters = new Float32Array(7);
  restrictionParameters[0] = 0;
  restrictionParameters[1] = WINDOW_END - WINDOW_START;
  restrictionParameters[2] = ctx.options.distanceMeters;
  restrictionParameters[6] = ctx.options.maximumGapSeconds;
  const restrictionParametersBuffer = resources.createParameterBuffer(
    'pairwise-restriction-parameters',
    'float32',
    restrictionParameters.length,
    restrictionParameters
  );
  const restrictionParametersView = restrictionParametersBuffer.importToGraph(graph);
  const restrictionOutputs = new Map<
    BoundaryKey,
    {intervals: Buffer; positions: Buffer; status: Buffer; relationships: Buffer}
  >();
  for (const timeBoundary of TIME_BOUNDARIES)
    for (const geometryBoundary of GEOMETRY_BOUNDARIES) {
      const key: BoundaryKey = `${timeBoundary}:${geometryBoundary}`;
      restrictionOutputs.set(key, {
        intervals: resources.createBuffer(
          `restriction-${key}-intervals`,
          pairs.length * INTERVALS_PER_PAIR * 8
        ),
        positions: resources.createBuffer(
          `restriction-${key}-positions`,
          pairs.length * INTERVALS_PER_PAIR * 16,
          Buffer.VERTEX
        ),
        status: resources.createBuffer(
          `restriction-${key}-status`,
          pairs.length * INTERVALS_PER_PAIR * 4
        ),
        relationships: resources.createBuffer(`restriction-${key}-relationships`, pairs.length * 16)
      });
    }
  const bounds = (() => {
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;
    for (let row = 0; row < vertexCount; row++) {
      const positionX = positions[row * 2],
        positionY = positions[row * 2 + 1];
      minX = Math.min(minX, positionX);
      minY = Math.min(minY, positionY);
      maxX = Math.max(maxX, positionX);
      maxY = Math.max(maxY, positionY);
    }
    return [minX - 1000, minY - 1000, maxX + 1000, maxY + 1000] as const;
  })();
  const closestGraph = new GPUTrajectoryClosestApproach({
    id: 'continuous-exact-pair-approach',
    spatialContext: tracks.spatialContext,
    positions: positionsView,
    timestamps: timesView,
    trackOffsets: offsetsView,
    distance: distanceView,
    cellSize: 1000,
    timeCellSize: 120,
    entryCapacity: 1 << 16,
    hitCapacity: 1 << 18,
    output: {
      pairs: {
        ids: view(pairIds, 'uint32', PAIR_CAPACITY),
        status: {
          count: view(pairCount, 'uint32', 1),
          requiredCount: view(pairRequired, 'uint32', 1),
          overflow: view(pairOverflow, 'uint32', 1)
        },
        partners: view(pairPartners, 'uint32', PAIR_CAPACITY),
        minimumDistances: view(pairMinimum, 'float32', PAIR_CAPACITY),
        closestTimes: view(pairClosestTimes, 'float32', PAIR_CAPACITY),
        firstTimes: view(pairFirstTimes, 'float32', PAIR_CAPACITY),
        durations: view(pairDurations, 'float32', PAIR_CAPACITY)
      },
      status: {
        entryOverflow: view(entryOverflow, 'uint32', 1),
        hitOverflow: view(hitOverflow, 'uint32', 1)
      }
    }
  });
  graph.add(closestGraph);
  for (const [step, output] of sampledOutputs) {
    const clockView = output.clock.importToGraph(graph);
    addClockEncounters(graph, {
      id: `shared-${step}s-clock-sampled-encounters`,
      spatialContext: tracks.spatialContext,
      positions: positionsView,
      timestamps: timesView,
      trackOffsets: offsetsView,
      clock: clockView,
      bucketCount: 3600 / step + 1,
      distance: distanceView,
      cellSize: 1000,
      bounds,
      hitCapacity: 1 << 16,
      output: {
        pairs: {
          ids: view(output.ids, 'uint32', PAIR_CAPACITY),
          status: {
            count: view(output.count, 'uint32', 1),
            requiredCount: view(output.required, 'uint32', 1),
            overflow: view(output.overflow, 'uint32', 1)
          },
          partners: view(output.partners, 'uint32', PAIR_CAPACITY),
          firstBuckets: view(output.buckets, 'uint32', PAIR_CAPACITY),
          minimumDistances: view(output.minimum, 'float32', PAIR_CAPACITY),
          bucketCounts: view(output.bucketCounts, 'uint32', PAIR_CAPACITY)
        },
        status: {candidateOverflow: view(output.candidateOverflow, 'uint32', 1)}
      }
    });
  }
  for (const timeBoundary of TIME_BOUNDARIES)
    for (const geometryBoundary of GEOMETRY_BOUNDARIES) {
      const key: BoundaryKey = `${timeBoundary}:${geometryBoundary}`;
      const output = restrictionOutputs.get(key)!;
      graph.add(
        new GPUTrajectoryRestriction({
          id: `pairwise-distance-restriction-${key}`,
          positions: positionsView,
          times: timesView,
          offsets: offsetsView,
          parameters: restrictionParametersView,
          pairwiseTracks: pairs,
          intervalsPerPair: INTERVALS_PER_PAIR,
          timeBoundary,
          geometryBoundary,
          output: {
            intervals: view(output.intervals, 'float32x2', pairs.length * INTERVALS_PER_PAIR),
            positions: view(output.positions, 'float32x4', pairs.length * INTERVALS_PER_PAIR),
            status: view(output.status, 'uint32', pairs.length * INTERVALS_PER_PAIR),
            relationships: view(output.relationships, 'uint32x4', pairs.length)
          }
        })
      );
    }
  const compiled = resources.track(await graph.compileAsync());
  const selectedRestrictionKey = () =>
    `${ctx.options.timeBoundary}:${ctx.options.geometryBoundary}` as BoundaryKey;
  let generation = 1,
    pendingGeneration = 0,
    dirty = true,
    destroyed = false;
  let summary: {
    exactCount: number;
    sampledCount: number;
    sampledPairs: Uint32Array;
    sampledPartners: Uint32Array;
    exactPairs: Uint32Array;
    exactPartners: Uint32Array;
    exactMinimum: Float32Array;
    exactTimes: Float32Array;
    restrictionStatus: Uint32Array;
    restrictionCount: number;
    restrictionInvalid: number;
    restrictionGaps: number;
    restrictionOverflow: number;
    overflows: Uint32Array;
  } | null = null;
  const summarySources = [
    pairCount,
    pairRequired,
    pairOverflow,
    entryOverflow,
    hitOverflow,
    pairIds,
    pairPartners,
    pairMinimum,
    pairClosestTimes,
    ...Array.from(sampledOutputs.values()).flatMap(output => [
      output.count,
      output.required,
      output.overflow,
      output.candidateOverflow,
      output.ids,
      output.partners
    ]),
    ...Array.from(restrictionOutputs.values()).flatMap(output => [
      output.status,
      output.relationships
    ])
  ];
  const reader = new SummaryReader(
    resources,
    'continuous-encounter-summary',
    summarySources.map(buffer => ({buffer, size: buffer.byteLength})),
    bytes => {
      if (destroyed || pendingGeneration !== generation) return;
      let offset = 0;
      const words = (length: number) => {
        const result = new Uint32Array(bytes).slice(offset / 4, offset / 4 + length);
        offset += length * 4;
        return result;
      };
      const floats = (length: number) => {
        const result = new Float32Array(bytes).slice(offset / 4, offset / 4 + length);
        offset += length * 4;
        return result;
      };
      const exactCount = words(1)[0],
        exactRequired = words(1)[0],
        exactPairOverflow = words(1)[0],
        entry = words(1)[0],
        hit = words(1)[0];
      const exactPairs = words(PAIR_CAPACITY),
        exactPartners = words(PAIR_CAPACITY),
        exactMinimum = floats(PAIR_CAPACITY),
        exactTimes = floats(PAIR_CAPACITY);
      const clockResults = new Map<
        number,
        {
          count: number;
          required: number;
          overflow: number;
          candidateOverflow: number;
          pairs: Uint32Array;
          partners: Uint32Array;
        }
      >();
      for (const step of CLOCK_STEPS) {
        const count = words(1)[0],
          required = words(1)[0],
          overflow = words(1)[0],
          candidateOverflow = words(1)[0];
        clockResults.set(step, {
          count,
          required,
          overflow,
          candidateOverflow,
          pairs: words(PAIR_CAPACITY),
          partners: words(PAIR_CAPACITY)
        });
      }
      let selectedStatus = new Uint32Array(pairs.length * INTERVALS_PER_PAIR),
        selectedRelations = new Uint32Array(pairs.length * 4);
      for (const [key, output] of restrictionOutputs) {
        const status = words(pairs.length * INTERVALS_PER_PAIR),
          relations = words(pairs.length * 4);
        if (key === selectedRestrictionKey()) {
          selectedStatus = status;
          selectedRelations = relations;
        }
        void output;
      }
      const restrictionCount = selectedStatus.reduce(
        (count, flags) => count + Number((flags & 1) !== 0),
        0
      );
      const restrictionInvalid = selectedStatus.reduce(
        (count, flags) => count + Number((flags & 4) !== 0),
        0
      );
      const restrictionGaps = selectedStatus.reduce(
        (count, flags) => count + Number((flags & 8) !== 0),
        0
      );
      const restrictionOverflow = selectedStatus.reduce(
        (count, flags) => count + Number((flags & 256) !== 0),
        0
      );
      const unknownCount = selectedRelations.reduce(
        (total, value, index) => total + (index % 4 === 3 ? value : 0),
        0
      );
      const clockStep = Number(ctx.options.clockStep);
      const selectedClock = clockResults.get(clockStep)!;
      summary = {
        exactCount,
        sampledCount: selectedClock.count,
        sampledPairs: selectedClock.pairs,
        sampledPartners: selectedClock.partners,
        exactPairs,
        exactPartners,
        exactMinimum,
        exactTimes,
        restrictionStatus: selectedStatus,
        restrictionCount,
        restrictionInvalid,
        restrictionGaps,
        restrictionOverflow,
        overflows: Uint32Array.of(
          exactRequired,
          exactPairOverflow,
          entry,
          hit,
          ...Array.from(clockResults.values()).flatMap(result => [
            result.required,
            result.overflow,
            result.candidateOverflow
          ]),
          ...selectedRelations
        )
      };
      ctx.setReadout(
        'encounterCounts',
        `${exactCount} continuous · ${selectedClock.count} at ${clockStep}s samples`
      );
      ctx.setReadout(
        'samplingDifference',
        `${Math.max(0, exactCount - selectedClock.count)} continuous pairs absent from ${clockStep}s samples`
      );
      ctx.setReadout(
        'restrictionResult',
        `${restrictionCount} positive-duration intervals · ${selectedRestrictionKey().replace(':', ' / ')} · ${unknownCount} unsupported pair intervals`
      );
      const selectedOverflow = selectedClock.overflow || selectedClock.candidateOverflow;
      const bounded = exactPairOverflow || entry || hit || selectedOverflow;
      ctx.setReadout(
        'solverStatus',
        `${bounded ? `pair search capacity limit · exact ${exactRequired}, sampled ${selectedClock.required}` : 'pair searches complete'} · restriction invalid ${restrictionInvalid}, overflow ${restrictionOverflow}, long-gap segments ${restrictionGaps}`
      );
      const rows = Array.from({length: exactCount}, (_, row) => ({
        pair: `${sourceRows[exactPairs[row]]}/${sourceRows[exactPartners[row]]}`,
        distance: exactMinimum[row]
      })).sort((a, b) => a.distance - b.distance);
      ctx.setReadout(
        'closestDistances',
        bounded
          ? 'withheld: pair capacity exceeded'
          : rows.length
            ? rows
                .slice(0, 8)
                .map(item => `${item.pair}: ${item.distance.toFixed(2)} m`)
                .join(' · ')
            : 'No pair within threshold'
      );
      ctx.setChart(
        'closestPairs',
        rows.length
          ? {
              kind: 'bars',
              height: 118,
              labels: rows.slice(0, 8).map(item => item.pair),
              values: rows.slice(0, 8).map(item => item.distance),
              yLabel: 'minimum separation (m)',
              highlight: [0],
              description: 'Minimum planar separation between linearly interpolated AIS fixes.'
            }
          : null
      );
      ctx.requestLayers();
    }
  );
  function update(): void {
    distanceBuffer.write(Float32Array.of(ctx.options.distanceMeters));
    restrictionParameters[2] = ctx.options.distanceMeters;
    restrictionParameters[6] = ctx.options.maximumGapSeconds;
    restrictionParametersBuffer.write(restrictionParameters);
    generation++;
    dirty = true;
    summary = null;
    reader.markStale();
    ctx.setReadout('encounterCounts', 'updating GPU searches');
    ctx.setReadout('samplingDifference', 'updating GPU searches');
    ctx.setReadout('restrictionResult', 'updating GPU restriction');
    ctx.setReadout('solverStatus', 'waiting for GPU result');
    ctx.setChart('closestPairs', null);
    ctx.setReadout('closestDistances', 'updating');
  }
  ctx.setReadout(
    'selectedTracks',
    SOURCE_PATHS.map((track, index) => `${track} / ${mmsi[index]}`).join(' · ')
  );
  ctx.setReadout(
    'samplingAudit',
    `8 paths · max observed interval ${Math.max(...maximumObservedGaps)} s · 14:00–15:00 UTC`
  );
  ctx.setReadout('modelAssumption', 'AEQD metres · linear between fixes · no extrapolation');
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
    setOption() {
      update();
    },
    getLayers() {
      const routeLayer = new SpatialAnalysisSegmentLayer({
        id: 'audited-ais-hour-routes',
        analysisFrame: tracks.analysisFrame,
        segments: sourceSegmentsBuffer,
        instanceCount: segmentCount,
        widthPixels: 2,
        color: [64, 103, 124, 105]
      });
      const output = restrictionOutputs.get(selectedRestrictionKey())!;
      const intervalsLayer =
        summary && summary.restrictionOverflow === 0 && summary.restrictionInvalid === 0
          ? new SpatialAnalysisSegmentLayer({
              id: 'pairwise-continuous-restriction-results',
              analysisFrame: tracks.analysisFrame,
              segments: output.positions,
              instanceCount: pairs.length * INTERVALS_PER_PAIR,
              widthPixels: 4,
              color: [35, 210, 203, 225]
            })
          : null;
      const endpoints = new SpatialAnalysisPointLayer({
        id: 'audited-passenger-vessels',
        analysisFrame: tracks.analysisFrame,
        positions: positionsBuffer,
        instanceCount: vertexCount,
        radiusPixels: 2,
        color: [228, 239, 242, 155]
      });
      return [routeLayer, ...(intervalsLayer ? [intervalsLayer] : []), endpoints] as Layer[];
    },
    getTooltip(event) {
      const viewport = ctx.getViewport();
      if (!viewport) return null;
      let nearest = -1,
        closest = 14 * 14;
      for (let track = 0; track < trackCount; track++) {
        const point = viewport.project([
          tracks.lngLat[offsets[track] * 2],
          tracks.lngLat[offsets[track] * 2 + 1]
        ]);
        const distance = (point[0] - event.pixel[0]) ** 2 + (point[1] - event.pixel[1]) ** 2;
        if (distance < closest) {
          closest = distance;
          nearest = track;
        }
      }
      if (nearest < 0) return null;
      const trackPairs = pairs
        .map((pair, index) => ({pair, index}))
        .filter(({pair}) => pair.includes(nearest));
      const exactRows = summary
        ? trackPairs.filter(({index}) => {
            for (let row = 0; row < summary!.exactCount; row++)
              if (
                summary!.exactPairs[row] === pairs[index][0] &&
                summary!.exactPartners[row] === pairs[index][1]
              )
                return true;
            return false;
          }).length
        : 0;
      const sampledRows = summary
        ? trackPairs.filter(({index}) => {
            for (let row = 0; row < summary!.sampledCount; row++)
              if (
                summary!.sampledPairs[row] === pairs[index][0] &&
                summary!.sampledPartners[row] === pairs[index][1]
              )
                return true;
            return false;
          }).length
        : 0;
      const closestRow = summary
        ? Array.from({length: summary.exactCount}, (_, row) => row).find(
            row => summary!.exactPairs[row] === nearest || summary!.exactPartners[row] === nearest
          )
        : undefined;
      const lowerTrack = closestRow === undefined ? -1 : summary!.exactPairs[closestRow];
      const closestUtc =
        closestRow === undefined
          ? '—'
          : formatUtcClockSeconds(
              WINDOW_START + firstFixTimes[lowerTrack] + summary!.exactTimes[closestRow]
            );
      return {
        title: `MMSI ${mmsi[nearest]} · source path ${sourceRows[nearest]}`,
        subtitle: `${VESSEL_CATEGORY_LABELS[VESSEL_CATEGORIES[categories[nearest]]]} · 14:00–15:00 UTC`,
        rows: [
          {label: 'Fixes in window', value: offsets[nearest + 1] - offsets[nearest]},
          {label: 'Largest observed interval', value: `${maximumObservedGaps[nearest]} s`},
          {label: 'Continuous pairs with this path', value: summary ? exactRows : 'updating'},
          {label: 'Minute-sampled pairs', value: summary ? sampledRows : 'updating'},
          {label: 'Example closest approach (UTC)', value: closestUtc},
          {
            label: 'First observed fix',
            value: formatUtcClockSeconds(WINDOW_START + firstFixTimes[nearest])
          }
        ],
        note: 'Closest-approach time is relative to the lower-index path’s first retained sample, converted here to UTC. Continuous results assume straight-line motion between fixes; the exact approach contributor has no maximum-gap filter, so its interpretation is bounded by the audited sampling intervals.'
      };
    },
    destroy() {
      destroyed = true;
      reader.stop();
      resources.destroy();
    }
  };
}
