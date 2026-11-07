// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import {getGPUTimeWindowFilterParameterValues, GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH, GPUTimeWindowFilter} from '@luma-contributors/analysis/dataframe-analysis';
import {
  getGPUTrajectoryPlayheadParameterValues,
  GPUTrajectoryPlayhead,
  GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH,
  GPURoseStatistic,
  GPU_ROSE_STATISTIC_SUMMARY_STRIDE
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {DrawCommandBuffer, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  getClassBreaks,
  getClassCounts,
  getClassIndex,
  getExtent,
  getHistogram
} from '../../cartography/breaks';
import {getClassTableLayerProps} from '../../cartography/class-table';
import {formatCount, formatDistance, formatPercent} from '../../cartography/live-text';
import type {LngLat} from '../../cartography/types';
import {importGraphBuffer} from '../../engine/graph-buffers';
import {addKernelPass} from '../../engine/mode-kernels';
import {SpatialAnalysisPointLayer, SpatialAnalysisSegmentLayer} from '../../engine/layers';
import {createPlaybackClock} from '../../engine/playback';
import {SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {AgeFadeTrailLayer} from './nyc-taxi-trails-layers';
import {
  BACKDROP_INK,
  CHORD_INK,
  getCompassName,
  getFareTable,
  getHeadingColors,
  HEAD_HALO,
  HEAD_INK,
  HEAD_RADIUS_STOPS,
  HEADING_BINS,
  HEADING_RAMP,
  INSPECT_ROUTE_INK,
  INSPECT_TICK_INK,
  MAXIMUM_CHORDS,
  MINIMUM_CHORD_METERS,
  ROUTE_INK,
  TRAIL_INK,
  TRAIL_WIDTH_STOPS,
  type RampMatch
} from './nyc-taxi-trails-style';

/** Option state of the nyc-taxi-trails scene. */
export type NycTaxiTrailsOptions = {
  play: boolean;
  time: number;
  playSpeed: number;
  loop: boolean;
  trailMinutes: number;
  colorBy: 'none' | 'heading' | 'fare';
  roseMode: 'directional' | 'axial';
  roseWeighting: 'equal' | 'segment-length';
  roseGrouping: 'all' | 'fare-class';
  rampMatch: RampMatch;
  compareRamps: boolean;
  showBackdrop: boolean;
  showChords: boolean;
  inspect: boolean;
};

/**
 * The window of the TLC records in the dataset's clock: 08:00 to 08:30 is 60 to 1,860 s after the
 * 07:59:00 time origin. The minute of padding at either end of the archive is not played.
 */
export const TRAILS_FIRST_SECOND = 60;
export const TRAILS_LAST_SECOND = 1860;
/** The ramp-up band of the road chart: the window opens with few cabs already under way. */
const RAMP_UP_END_SECOND = 300;
const STATUS_INTERVAL_FRAMES = 10;
const COMPASS_DEGREES = 360;
const FARE_HISTOGRAM_BINS = 30;
const FARE_HISTOGRAM_MAXIMUM = 60;

function getFareClassLabel(index: number, breaks: readonly number[]): string {
  const format = (value: number) => `$${value.toFixed(0)}`;
  if (index === 0) return `< ${format(breaks[0] ?? FARE_HISTOGRAM_MAXIMUM)}`;
  if (index >= breaks.length) return `≥ ${format(breaks[breaks.length - 1] ?? 0)}`;
  return `${format(breaks[index - 1])}–${format(breaks[index])}`;
}

/**
 * Routed taxi trails over a 30-minute Friday morning window. A `GPUTrajectoryPlayhead` graph
 * interpolates every taxi at the clock, a `GPUTimeWindowFilter` graph selects the route segments of
 * the last few minutes with fade weights and clip fractions, and both are drawn straight from GPU
 * storage with indirect counts. The clock and the trail length are parameter buffers. CPU setup
 * provides route headings, lengths, fares and chart framing; GPURoseStatistic summarizes the
 * active modeled headings by direction, weighting and fare group before bounded readback.
 */
export async function createNycTaxiTrails(
  ctx: SceneContext<NycTaxiTrailsOptions>
): Promise<SceneInstance<NycTaxiTrailsOptions>> {
  const {device} = ctx;
  const dataset = ctx.datasets.get('poopdeck-nyc-taxi-paths');
  const origin = dataset.defaultOrigin;
  const projection = dataset.getProjection(origin);
  const offsets = dataset.column<Uint32Array>('pathOffsets');
  const timestamps = Float32Array.from(dataset.column<Uint32Array>('timestamp'));
  const positions = dataset.projectColumn('vertices', origin);
  const fares = dataset.column<Float32Array>('fare_amount');
  const trackCount = offsets.length - 1;
  const vertexCount = timestamps.length;
  const timeOriginSeconds =
    (dataset.manifest.properties as {timeOriginMs: number}).timeOriginMs / 1000;
  const coordinateOrigin: [number, number, number] = [origin[0], origin[1], 0];
  const resources = new SpatialAnalysisResources(device, 'taxi-trails');

  // ---- Segment tables and per-route facts (CPU, once) --------------------------------------------
  const segmentCount = vertexCount - trackCount;
  const segments = new Float32Array(segmentCount * 4);
  const segmentTracks = new Uint32Array(segmentCount);
  const segmentStartTimes = new Float32Array(segmentCount);
  const segmentEndTimes = new Float32Array(segmentCount);
  /** Heading of each segment as a share of the full circle, clockwise from north (0 to <1). */
  const segmentHeadings = new Float32Array(segmentCount);
  const routeLengths = new Float32Array(trackCount);
  const chordLengths = new Float32Array(trackCount);
  let maximumTrackVertices = 0;
  let timeEnd = 0;
  {
    let row = 0;
    for (let track = 0; track < trackCount; track++) {
      const first = offsets[track];
      const last = offsets[track + 1] - 1;
      maximumTrackVertices = Math.max(maximumTrackVertices, last - first + 1);
      let routeLength = 0;
      let heading = 0;
      for (let vertex = first; vertex < last; vertex++, row++) {
        const x0 = positions[vertex * 2];
        const y0 = positions[vertex * 2 + 1];
        const x1 = positions[vertex * 2 + 2];
        const y1 = positions[vertex * 2 + 3];
        segments.set([x0, y0, x1, y1], row * 4);
        segmentTracks[row] = track;
        segmentStartTimes[row] = timestamps[vertex];
        segmentEndTimes[row] = timestamps[vertex + 1];
        const dx = x1 - x0;
        const dy = y1 - y0;
        routeLength += Math.hypot(dx, dy);
        // A zero-length step keeps the heading before it, so it never flashes a wrong colour.
        if (dx !== 0 || dy !== 0) {
          heading = (((Math.atan2(dx, dy) / (2 * Math.PI)) % 1) + 1) % 1;
        }
        segmentHeadings[row] = heading;
      }
      routeLengths[track] = routeLength;
      chordLengths[track] = Math.hypot(
        positions[last * 2] - positions[first * 2],
        positions[last * 2 + 1] - positions[first * 2 + 1]
      );
      timeEnd = Math.max(timeEnd, timestamps[last]);
    }
  }
  const circuities: number[] = [];
  for (let track = 0; track < trackCount; track++) {
    // A route over a very short chord is mostly a loop around the block, not a detour.
    if (chordLengths[track] > 200) circuities.push(routeLengths[track] / chordLengths[track]);
  }
  circuities.sort((a, b) => a - b);
  const medianCircuity = circuities.length ? circuities[circuities.length >> 1] : 1;
  const detourShare = circuities.length
    ? circuities.filter(value => value > 1.3).length / circuities.length
    : 0;

  // Fare classes: five quantile classes computed once, shared by the trails, the legend and the chart.
  const fareBreaks = getClassBreaks(fares, 5, 'quantile');
  const fareClasses = Uint32Array.from(fares, fare => getClassIndex(fare, fareBreaks));
  const fareExtent = getExtent(fares);
  const fareCounts = getClassCounts(fares, fareBreaks);

  // ---- Buffers -----------------------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', positions);
  const timestampsBuffer = resources.createBuffer('timestamps', timestamps);
  const offsetsBuffer = resources.createBuffer('track-offsets', offsets);
  const segmentsBuffer = resources.createBuffer('segments', segments);
  const segmentTracksBuffer = resources.createBuffer('segment-tracks', segmentTracks);
  const segmentStartTimesBuffer = resources.createBuffer('segment-start-times', segmentStartTimes);
  const segmentEndTimesBuffer = resources.createBuffer('segment-end-times', segmentEndTimes);
  const segmentHeadingBuffer = resources.createBuffer('segment-headings', segmentHeadings);
  const fareBuffer = resources.createBuffer('fare', fares);

  // ---- Playhead graph: every taxi at the clock -----------------------------------------------------
  const currentPositions = resources.createBuffer('current-positions', trackCount * 8);
  const headings = resources.createBuffer('headings', trackCount * 4);
  const speeds = resources.createBuffer('speeds', trackCount * 4);
  const segmentRows = resources.createBuffer('segment-rows', trackCount * 4);
  const status = resources.createBuffer('status', trackCount * 4);
  const activeIds = resources.createBuffer('active-ids', trackCount * 4);
  const activeCount = resources.createBuffer('active-count', 4);
  const activeOverflow = resources.createBuffer('active-overflow', 4);
  const roseStarts = resources.createBuffer('rose-starts', trackCount * 8);
  const roseEnds = resources.createBuffer('rose-ends', trackCount * 8);
  const roseMask = resources.createBuffer('rose-mask', trackCount * 4);
  const roseWeights = resources.createBuffer('rose-current-segment-lengths', trackCount * 4);
  const roseFareClasses = resources.createBuffer('rose-fare-classes', fareClasses);
  const ROSE_CLASS_COUNT = 5;
  const roseConfigurations = (['directional', 'axial'] as const).flatMap(mode =>
    (['equal', 'segment-length'] as const).map(weighting => ({mode, weighting}))
  );
  const roseOutputs = roseConfigurations.map(({mode, weighting}, index) => {
    const groupCount = ROSE_CLASS_COUNT;
    const binCount = mode === 'directional' ? HEADING_BINS : HEADING_BINS / 2;
    return {
      mode,
      weighting,
      groupCount,
      binCount,
      summary: resources.createBuffer(
        `rose-${index}-summary`,
        groupCount * GPU_ROSE_STATISTIC_SUMMARY_STRIDE * 4
      ),
      bins: resources.createBuffer(`rose-${index}-bins`, groupCount * binCount * 4)
    };
  });
  const playheadParameters = resources.createParameterBuffer(
    'playhead',
    'float32',
    GPU_TRAJECTORY_PLAYHEAD_PARAMETER_LENGTH
  );
  const markerDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'taxi-marker-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const playheadGraph = new GPUCommandGraph<void>(device, {id: 'taxi-playhead'});
  const playheadStatusView = importGraphBuffer(
    playheadGraph,
    'status',
    status,
    'uint32',
    trackCount
  );
  const playheadHeadingView = importGraphBuffer(
    playheadGraph,
    'headings',
    headings,
    'float32',
    trackCount
  );
  const playheadSegmentRowsView = importGraphBuffer(
    playheadGraph,
    'segment-rows',
    segmentRows,
    'uint32',
    trackCount
  );
  const positionsView = importGraphBuffer(
    playheadGraph,
    'positions',
    positionsBuffer,
    'float32x2',
    vertexCount
  );
  const fareClassesView = importGraphBuffer(
    playheadGraph,
    'rose-fare-classes',
    roseFareClasses,
    'uint32',
    trackCount
  );
  playheadGraph.add(
    new GPUTrajectoryPlayhead({
      spatialContext: projection.spatialContext,
      id: 'playhead',
      positions: positionsView,
      timestamps: importGraphBuffer(
        playheadGraph,
        'timestamps',
        timestampsBuffer,
        'float32',
        vertexCount
      ),
      trackOffsets: importGraphBuffer(
        playheadGraph,
        'offsets',
        offsetsBuffer,
        'uint32',
        trackCount + 1
      ),
      parameters: playheadParameters.importToGraph(playheadGraph),
      output: {
        currentPositions: importGraphBuffer(
          playheadGraph,
          'current-positions',
          currentPositions,
          'float32x2',
          trackCount
        ),
        headings: playheadHeadingView,
        segmentRows: playheadSegmentRowsView,
        speeds: importGraphBuffer(playheadGraph, 'speeds', speeds, 'float32', trackCount),
        trackStatus: playheadStatusView,
        activeTracks: {
          ids: importGraphBuffer(playheadGraph, 'active-ids', activeIds, 'uint32', trackCount),
          status: {
            count: importGraphBuffer(playheadGraph, 'active-count', activeCount, 'uint32', 1),
            overflow: importGraphBuffer(
              playheadGraph,
              'active-overflow',
              activeOverflow,
              'uint32',
              1
            )
          }
        },
        drawInstanceCount: playheadGraph.importGPUData(
          'marker-draw-count',
          markerDraw.getInstanceCountData(0)
        )
      }
    })
  );
  const roseStartsView = importGraphBuffer(
    playheadGraph,
    'rose-starts',
    roseStarts,
    'float32x2',
    trackCount
  );
  const roseEndsView = importGraphBuffer(
    playheadGraph,
    'rose-ends',
    roseEnds,
    'float32x2',
    trackCount
  );
  const roseMaskView = importGraphBuffer(
    playheadGraph,
    'rose-mask',
    roseMask,
    'uint32',
    trackCount
  );
  const roseWeightsView = importGraphBuffer(
    playheadGraph,
    'rose-current-segment-lengths',
    roseWeights,
    'float32',
    trackCount
  );
  addKernelPass(playheadGraph, {
    id: 'taxi-rose-heading-inputs',
    invocationCount: trackCount,
    bindings: [
      {name: 'headings', view: playheadHeadingView, type: 'f32', access: 'read'},
      {name: 'trackStatus', view: playheadStatusView, type: 'u32', access: 'read'},
      {name: 'segmentRows', view: playheadSegmentRowsView, type: 'u32', access: 'read'},
      {name: 'positions', view: positionsView, type: 'f32', access: 'read'},
      {name: 'starts', view: roseStartsView, type: 'f32', access: 'read_write'},
      {name: 'ends', view: roseEndsView, type: 'f32', access: 'read_write'},
      {name: 'mask', view: roseMaskView, type: 'u32', access: 'read_write'},
      {name: 'weights', view: roseWeightsView, type: 'f32', access: 'read_write'}
    ],
    declarations:
      'const STATUS_ACTIVE: u32 = 1u; const HALF_COMPASS_BIN_RADIANS: f32 = 0.08726646259971647;',
    body: `let heading = headings[headingsOffset + index];
  let segmentRow = segmentRows[segmentRowsOffset + index];
  let isActiveTrack = trackStatus[trackStatusOffset + index] == STATUS_ACTIVE;
  starts[startsOffset + 2u * index] = 0.0;
  starts[startsOffset + 2u * index + 1u] = 0.0;
  // GPURoseStatistic floors angle/binWidth; rotate by half a bin so the chart sectors are centered.
  let binCenteredHeading = heading + HALF_COMPASS_BIN_RADIANS;
  ends[endsOffset + 2u * index] = cos(binCenteredHeading);
  ends[endsOffset + 2u * index + 1u] = sin(binCenteredHeading);
  var segmentLength = 0.0;
  if (isActiveTrack && segmentRow + 1u < ${vertexCount}u) {
    let deltaX = positions[positionsOffset + 2u * (segmentRow + 1u)] - positions[positionsOffset + 2u * segmentRow];
    let deltaY = positions[positionsOffset + 2u * (segmentRow + 1u) + 1u] - positions[positionsOffset + 2u * segmentRow + 1u];
    segmentLength = sqrt(deltaX * deltaX + deltaY * deltaY);
  }
  weights[weightsOffset + index] = segmentLength;
  mask[maskOffset + index] = select(0u, 1u, isActiveTrack && segmentLength > 0.0);`
  });
  for (const [index, roseOutput] of roseOutputs.entries()) {
    playheadGraph.add(
      new GPURoseStatistic({
        id: `taxi-rose-${index}`,
        starts: roseStartsView,
        ends: roseEndsView,
        mask: roseMaskView,
        ...(roseOutput.weighting === 'segment-length' ? {weights: roseWeightsView} : {}),
        groupIds: fareClassesView,
        groupCount: roseOutput.groupCount,
        circular: {
          mode: roseOutput.mode,
          angleUnit: 'radians',
          origin: 'positive-x-counter-clockwise',
          binCount: roseOutput.binCount
        },
        output: {
          summary: importGraphBuffer(
            playheadGraph,
            `rose-${index}-summary`,
            roseOutput.summary,
            'float32',
            roseOutput.groupCount * GPU_ROSE_STATISTIC_SUMMARY_STRIDE
          ),
          bins: importGraphBuffer(
            playheadGraph,
            `rose-${index}-bins`,
            roseOutput.bins,
            'float32',
            roseOutput.groupCount * roseOutput.binCount
          )
        }
      })
    );
  }
  const playheadCompiled = resources.track(await playheadGraph.compileAsync());

  // ---- Trail graph: route segments of the last few minutes -----------------------------------------
  const trailIds = resources.createBuffer('trail-ids', segmentCount * 4);
  const trailCount = resources.createBuffer('trail-count', 4);
  const trailOverflow = resources.createBuffer('trail-overflow', 4);
  const fadeWeights = resources.createBuffer('fade-weights', segmentCount * 4);
  const clipFractions = resources.createBuffer('clip-fractions', segmentCount * 8);
  const trackVisibleCounts = resources.createBuffer('track-visible-counts', trackCount * 4);
  const windowParameters = resources.createParameterBuffer(
    'window',
    'float32',
    GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH
  );
  const trailDraw = resources.track(
    new DrawCommandBuffer(device, {
      id: 'taxi-trail-draw',
      type: 'draw',
      commands: [{vertexCount: 6, instanceCount: 0}]
    })
  );
  const trailGraph = new GPUCommandGraph<void>(device, {id: 'taxi-trails'});
  trailGraph.add(
    new GPUTimeWindowFilter({
      id: 'trail-window',
      timestamps: importGraphBuffer(
        trailGraph,
        'segment-start-times',
        segmentStartTimesBuffer,
        'float32',
        segmentCount
      ),
      endTimestamps: importGraphBuffer(
        trailGraph,
        'segment-end-times',
        segmentEndTimesBuffer,
        'float32',
        segmentCount
      ),
      window: windowParameters.importToGraph(trailGraph),
      trackIds: importGraphBuffer(
        trailGraph,
        'segment-tracks',
        segmentTracksBuffer,
        'uint32',
        segmentCount
      ),
      output: {
        ids: importGraphBuffer(trailGraph, 'trail-ids', trailIds, 'uint32', segmentCount),
        status: {
          count: importGraphBuffer(trailGraph, 'trail-count', trailCount, 'uint32', 1),
          overflow: importGraphBuffer(trailGraph, 'trail-overflow', trailOverflow, 'uint32', 1)
        },
        fadeWeights: importGraphBuffer(
          trailGraph,
          'fade-weights',
          fadeWeights,
          'float32',
          segmentCount
        ),
        clipFractions: importGraphBuffer(
          trailGraph,
          'clip-fractions',
          clipFractions,
          'float32x2',
          segmentCount
        ),
        trackVisibleCounts: importGraphBuffer(
          trailGraph,
          'track-visible-counts',
          trackVisibleCounts,
          'uint32',
          trackCount
        ),
        drawInstanceCount: trailGraph.importGPUData(
          'trail-draw-count',
          trailDraw.getInstanceCountData(0)
        )
      }
    })
  );
  const trailCompiled = resources.track(await trailGraph.compileAsync());

  // ---- Chords and the inspected cab: small buffers rewritten from the active-id readback ---------
  const chordBuffer = resources.createBuffer('chords', MAXIMUM_CHORDS * 16);
  const chordRouteBuffer = resources.createBuffer(
    'chord-routes',
    MAXIMUM_CHORDS * maximumTrackVertices * 16
  );
  const inspectRouteBuffer = resources.createBuffer('inspect-route', maximumTrackVertices * 16);
  const inspectVertexBuffer = resources.createBuffer('inspect-vertices', maximumTrackVertices * 8);
  const inspectIdBuffer = resources.createBuffer('inspect-id', 4);
  let chordCount = 0;
  let chordRouteCount = 0;
  let chordKey = '';
  let inspectedTrack = -1;
  let inspectedVertexCount = 0;
  let pendingPick = false;
  let lastActive = new Uint32Array(0);
  let hasRead = false;

  // ---- Charts that do not change with the clock ---------------------------------------------------
  const BIN_SECONDS = 30;
  const binCount = Math.ceil(timeEnd / BIN_SECONDS) + 1;
  const onRoad = new Float64Array(binCount);
  const startTimes = dataset.column<Uint32Array>('startTime');
  const endTimes = dataset.column<Uint32Array>('endTime');
  for (let track = 0; track < trackCount; track++) {
    const firstBin = Math.floor(startTimes[track] / BIN_SECONDS);
    const lastBin = Math.min(binCount - 1, Math.floor(endTimes[track] / BIN_SECONDS));
    for (let bin = firstBin; bin <= lastBin; bin++) onRoad[bin]++;
  }

  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'playSpeed', loop: 'loop'},
    {range: [TRAILS_FIRST_SECOND, TRAILS_LAST_SECOND], rate: 1, step: 5}
  );

  let destroyed = false;
  const statusReader = new SummaryReader(
    resources,
    'taxi-status',
    [
      {buffer: activeCount, size: 4},
      {buffer: activeOverflow, size: 4},
      {buffer: trailCount, size: 4},
      {buffer: trailOverflow, size: 4},
      {buffer: activeIds, size: trackCount * 4},
      ...roseOutputs.flatMap(output => [
        {buffer: output.bins, size: output.groupCount * output.binCount * 4},
        {
          buffer: output.summary,
          size: output.groupCount * GPU_ROSE_STATISTIC_SUMMARY_STRIDE * 4
        }
      ])
    ],
    bytes => {
      if (!destroyed) processStatus(bytes);
    }
  );

  function formatClock(seconds: number): string {
    const total = Math.round(timeOriginSeconds + seconds);
    const hours = Math.floor((total % 86400) / 3600);
    const minutes = Math.floor((total % 3600) / 60);
    const secs = total % 60;
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }

  function getTrackLngLat(x: number, y: number): LngLat {
    return projection.unproject(x, y);
  }

  // ---- Readback of the active list --------------------------------------------------------------
  function processStatus(bytes: ArrayBuffer): void {
    const words = new Uint32Array(bytes);
    const [active, activeOver, trail, trailOver] = words;
    ctx.setReadout('active', active);
    ctx.setReadout('segments', trail);
    ctx.setReadout(
      'overflow',
      activeOver || trailOver ? 'yes: some taxis or segments are not drawn' : 'no'
    );
    const shown = Math.min(active, trackCount);
    lastActive = words.slice(4, 4 + shown);
    hasRead = true;
    const floats = new Float32Array(bytes);
    let roseOffset = 4 + trackCount;
    let selectedRose: (typeof roseOutputs)[number] | undefined;
    let selectedRoseBinsStart = 0;
    let selectedRoseSummaryStart = 0;
    for (const output of roseOutputs) {
      if (output.mode === ctx.options.roseMode && output.weighting === ctx.options.roseWeighting) {
        selectedRose = output;
        selectedRoseBinsStart = roseOffset;
        selectedRoseSummaryStart = roseOffset + output.groupCount * output.binCount;
      }
      roseOffset += output.groupCount * (output.binCount + GPU_ROSE_STATISTIC_SUMMARY_STRIDE);
    }
    if (ctx.options.colorBy === 'heading') {
      if (selectedRose) {
        publishRose(floats, selectedRose, selectedRoseBinsStart, selectedRoseSummaryStart);
      }
    }
    if (ctx.options.showChords && !ctx.options.play) buildChords();
    if (pendingPick || (ctx.options.inspect && inspectedTrack < 0)) pickTaxi();
  }

  /** Reorders positive-x-CCW GPU bins into the chart's north-clockwise compass convention. */
  function publishRose(
    floats: Float32Array,
    roseOutput: (typeof roseOutputs)[number],
    binsStart: number,
    summaryStart: number
  ): void {
    const degreesPerBin = COMPASS_DEGREES / HEADING_BINS;
    const groupBins = Array.from({length: roseOutput.groupCount}, (_, group) => {
      const gpuBins = Array.from(
        floats.subarray(
          binsStart + group * roseOutput.binCount,
          binsStart + (group + 1) * roseOutput.binCount
        )
      );
      const bins = new Array<number>(HEADING_BINS).fill(0);
      for (let compassSector = 0; compassSector < HEADING_BINS; compassSector++) {
        const mathematicalSector =
          (((HEADING_BINS / 4 - compassSector) % roseOutput.binCount) + roseOutput.binCount) %
          roseOutput.binCount;
        bins[compassSector] =
          roseOutput.mode === 'axial'
            ? gpuBins[mathematicalSector] / 2
            : gpuBins[mathematicalSector];
      }
      return bins;
    });
    const period = roseOutput.mode === 'axial' ? Math.PI : Math.PI * 2;
    const groups = Array.from({length: roseOutput.groupCount}, (_, group) => {
      const start = summaryStart + group * GPU_ROSE_STATISTIC_SUMMARY_STRIDE;
      return {
        count: floats[start],
        meanAngle: (floats[start + 1] - Math.PI / HEADING_BINS + period) % period,
        resultantLength: floats[start + 2]
      };
    });
    const allBins = groupBins[0].map((_, sector) =>
      groupBins.reduce((total, group) => total + group[sector], 0)
    );
    const activeWeight = allBins.reduce((total, value) => total + value, 0);
    let resultantX = 0;
    let resultantY = 0;
    const axialScale = roseOutput.mode === 'axial' ? 2 : 1;
    for (let group = 0; group < roseOutput.groupCount; group++) {
      const groupWeight = groupBins[group].reduce((total, value) => total + value, 0);
      if (groupWeight === 0) continue;
      resultantX +=
        groupWeight *
        groups[group].resultantLength *
        Math.cos(axialScale * groups[group].meanAngle);
      resultantY +=
        groupWeight *
        groups[group].resultantLength *
        Math.sin(axialScale * groups[group].meanAngle);
    }
    const resultantLength =
      activeWeight > 0 ? Math.hypot(resultantX, resultantY) / activeWeight : 0;
    const activeCount = groups.reduce((total, group) => total + group.count, 0);
    ctx.setReadout(
      'roseCoverage',
      roseOutput.weighting === 'segment-length'
        ? `${formatCount(activeCount)} nonzero segment headings · ${formatDistance(activeWeight)} current segments`
        : `${formatCount(activeCount)} nonzero segment headings · equal contribution`
    );
    // The busiest three neighbouring sectors, as a share of the heads on the road.
    let bestStart = 0;
    let bestSum = -1;
    for (let start = 0; start < HEADING_BINS; start++) {
      const sum =
        allBins[start] + allBins[(start + 1) % HEADING_BINS] + allBins[(start + 2) % HEADING_BINS];
      if (sum > bestSum) {
        bestSum = sum;
        bestStart = start;
      }
    }
    const centerDegrees = (bestStart + 1.0) * degreesPerBin;
    ctx.setReadout(
      'dominantHeading',
      activeWeight > 0
        ? `${formatPercent(bestSum / activeWeight)} modeled heading ${roseOutput.weighting === 'segment-length' ? 'length' : 'count'} near ${getCompassName(centerDegrees)} · R=${resultantLength.toFixed(2)}`
        : null
    );
    const buildRoseChart = (values: number[], baselineWeight: number, title?: string) => ({
      kind: 'rose' as const,
      values,
      labels: Array.from({length: HEADING_BINS}, (_, sector) =>
        sector % 9 === 0 ? ['N', 'E', 'S', 'W'][sector / 9] : `${sector * degreesPerBin}°`
      ),
      colors: getHeadingColors(),
      baseline: new Array<number>(HEADING_BINS).fill(baselineWeight / HEADING_BINS),
      height: 160,
      title,
      table: false,
      description: `GPU ${roseOutput.mode} circular statistic of active modeled route headings, binned into compass sectors. ${roseOutput.weighting === 'segment-length' ? 'Each heading is weighted by its current route-segment length in metres.' : 'Each active taxi contributes one equal count.'} Fare groupings are descriptive associations, not causal effects. This is modeled route geometry, not GPS bearing observation.`
    });
    if (ctx.options.roseGrouping === 'fare-class') {
      const groupWeights = groupBins.map(group => group.reduce((total, value) => total + value, 0));
      ctx.setChart('headingRose', {
        kind: 'multiples',
        columns: 2,
        titles: Array.from({length: ROSE_CLASS_COUNT}, (_, group) =>
          getFareClassLabel(group, fareBreaks)
        ),
        description: `Modeled route-heading roses grouped by the trip's fare quantile class. Grouping is descriptive and does not imply fares cause route orientation.`,
        charts: groupBins.map((group, index) =>
          buildRoseChart(group, groupWeights[index], getFareClassLabel(index, fareBreaks))
        )
      });
    } else {
      ctx.setChart('headingRose', {
        ...buildRoseChart(allBins, activeWeight)
      });
    }
  }

  /** The chords of up to 40 cabs now on the road, each with its full route. */
  function buildChords(): void {
    const candidates: number[] = [];
    for (const track of lastActive) {
      if (chordLengths[track] > MINIMUM_CHORD_METERS) candidates.push(track);
    }
    const stride = Math.max(1, Math.floor(candidates.length / MAXIMUM_CHORDS));
    const chosen: number[] = [];
    for (
      let index = 0;
      index < candidates.length && chosen.length < MAXIMUM_CHORDS;
      index += stride
    ) {
      chosen.push(candidates[index]);
    }
    const key = chosen.join(',');
    if (key === chordKey) return;
    chordKey = key;
    const chords = new Float32Array(chosen.length * 4);
    let routeSegments = 0;
    for (const track of chosen) routeSegments += offsets[track + 1] - offsets[track] - 1;
    const routes = new Float32Array(routeSegments * 4);
    let routeRow = 0;
    chosen.forEach((track, index) => {
      const first = offsets[track];
      const last = offsets[track + 1] - 1;
      chords.set(
        [
          positions[first * 2],
          positions[first * 2 + 1],
          positions[last * 2],
          positions[last * 2 + 1]
        ],
        index * 4
      );
      // Segment rows of a track start at `offset - track`: every earlier track owns one vertex more.
      const firstSegment = first - track;
      const length = last - first;
      routes.set(segments.subarray(firstSegment * 4, (firstSegment + length) * 4), routeRow * 4);
      routeRow += length;
    });
    chordCount = chosen.length;
    chordRouteCount = routeSegments;
    if (chordCount > 0) {
      chordBuffer.write(chords);
      chordRouteBuffer.write(routes);
    }
    ctx.requestLayers();
  }

  /**
   * Picks the longest route among the cabs on the road now (skipping `exclude`). Before the first
   * readback there is no list, so the pick waits for it.
   */
  function pickTaxi(exclude = -1): void {
    if (!hasRead) {
      pendingPick = true;
      return;
    }
    let best = -1;
    let bestLength = -1;
    const consider = (track: number) => {
      if (track !== exclude && routeLengths[track] > bestLength) {
        bestLength = routeLengths[track];
        best = track;
      }
    };
    for (const track of lastActive) consider(track);
    // Nothing on the road at this clock (the clock stays inside the window, so this is a guard).
    if (best < 0) for (let track = 0; track < trackCount; track++) consider(track);
    pendingPick = false;
    inspectedTrack = best;
    const first = offsets[best];
    const last = offsets[best + 1] - 1;
    inspectedVertexCount = last - first + 1;
    const firstSegment = first - best;
    inspectRouteBuffer.write(
      segments.subarray(firstSegment * 4, (firstSegment + last - first) * 4)
    );
    inspectVertexBuffer.write(positions.subarray(first * 2, (last + 1) * 2));
    inspectIdBuffer.write(Uint32Array.of(best));
    ctx.setReadout(
      'searchSteps',
      `${Math.ceil(Math.log2(inspectedVertexCount))} steps over ${formatCount(inspectedVertexCount)} vertices`
    );
    ctx.setReadout('inspectRoute', `${formatDistance(routeLengths[best])} of route`);
    let west = Infinity;
    let south = Infinity;
    let east = -Infinity;
    let north = -Infinity;
    for (let vertex = first; vertex <= last; vertex++) {
      const [longitude, latitude] = getTrackLngLat(
        positions[vertex * 2],
        positions[vertex * 2 + 1]
      );
      west = Math.min(west, longitude);
      east = Math.max(east, longitude);
      south = Math.min(south, latitude);
      north = Math.max(north, latitude);
    }
    // The camera moves only while paused: a flight under a running clock would hide the cab.
    if (!ctx.options.play) {
      ctx.fitBounds([west, south, east, north], {transitionMs: 1400, maxZoom: 14.5, padding: 40});
    }
    ctx.requestLayers();
  }

  function clearInspection(): void {
    inspectedTrack = -1;
    inspectedVertexCount = 0;
    pendingPick = false;
    ctx.setReadout('searchSteps', null);
    ctx.setReadout('inspectRoute', null);
  }

  // ---- Charts ------------------------------------------------------------------------------------
  function publishRoadChart(): void {
    const times = Array.from(onRoad, (_, index) => index * BIN_SECONDS);
    ctx.setChart('roadChart', {
      kind: 'line',
      series: [{label: 'cabs on the road', x: times, y: Array.from(onRoad), area: true}],
      xLabel: 'Local time',
      yLabel: 'trips on the road (sample)',
      height: 110,
      xDomain: [TRAILS_FIRST_SECOND, TRAILS_LAST_SECOND],
      formatX: value => formatClock(value).slice(0, 5),
      bands: [{from: TRAILS_FIRST_SECOND, to: RAMP_UP_END_SECOND, label: 'ramp-up'}],
      link: {option: 'time', label: value => formatClock(value).slice(0, 5)},
      description:
        'Trips on the road per half minute in the sample, with the first minutes shaded because the window opens with few cabs already under way.'
    });
  }

  function publishFare(): void {
    const match = ctx.options.compareRamps ? 'matched' : ctx.options.rampMatch;
    const table = getFareTable({breaks: fareBreaks, extent: fareExtent, match});
    ctx.setLegendData('fare', {breaks: fareBreaks, extent: fareExtent, counts: fareCounts});
    ctx.setChart('fareHistogram', {
      kind: 'histogram',
      values: getHistogram(fares, [0, FARE_HISTOGRAM_MAXIMUM], FARE_HISTOGRAM_BINS),
      xDomain: [0, FARE_HISTOGRAM_MAXIMUM],
      breaks: fareBreaks,
      classColors: table.colors,
      xLabel: 'Fare (USD)',
      yLabel: 'trips',
      height: 110,
      formatX: value => `$${value.toFixed(2)}`,
      table: false,
      description:
        'Histogram of the fares of the sampled trips with the four quintile breaks marked and the five classes in the map colours.'
    });
  }

  function getTrailStyle(options: NycTaxiTrailsOptions, match: RampMatch) {
    if (options.colorBy === 'heading') {
      return {
        values: segmentHeadingBuffer,
        valueFormat: 'float32' as const,
        colormap: HEADING_RAMP,
        valueRange: [0, 1] as [number, number]
      };
    }
    if (options.colorBy === 'fare') {
      return {
        values: fareBuffer,
        valueFormat: 'float32' as const,
        valueIndices: segmentTracksBuffer,
        colormap: 'grayscale' as const,
        ...getClassTableLayerProps(getFareTable({breaks: fareBreaks, extent: fareExtent, match}))
      };
    }
    return {color: TRAIL_INK};
  }

  ctx.setReadout('trips', trackCount);
  ctx.setReadout('vertices', vertexCount);
  ctx.setReadout('segmentsTotal', segmentCount);
  ctx.setReadout('circuity', `${medianCircuity.toFixed(2)} times the chord`);
  ctx.setReadout('detourShare', formatPercent(detourShare));
  ctx.setCost({records: segmentCount, passes: 2});
  ctx.setFurniture({
    title: {
      sample: `${formatCount(trackCount)} trips, a seeded half of the routed sample`,
      chips: ['Sample', 'Modelled routes']
    }
  });
  publishRoadChart();
  publishFare();

  return {
    getCompiledGraphs: () => [playheadCompiled, trailCompiled] as never[],

    setOption(id) {
      switch (id) {
        case 'time':
        case 'playSpeed':
        case 'loop':
          return;
        case 'play':
          // Paused: refresh the chords for the frozen moment.
          chordKey = '';
          statusReader.markStale();
          break;
        case 'showChords':
          chordKey = '';
          if (!ctx.options.showChords) chordCount = 0;
          statusReader.markStale();
          break;
        case 'inspect':
          if (ctx.options.inspect) pickTaxi();
          else clearInspection();
          break;
        case 'rampMatch':
        case 'compareRamps':
          publishFare();
          break;
        case 'roseMode':
        case 'roseWeighting':
        case 'roseGrouping':
          statusReader.markStale();
          break;
        default:
      }
      ctx.requestLayers();
    },

    onAction(id) {
      if (id !== 'pickTaxi') return;
      ctx.setOptions({inspect: true}, {notify: false});
      // The next press takes the longest of the other cabs.
      pickTaxi(inspectedTrack);
    },

    onGroundChange() {
      ctx.requestLayers();
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      const options = ctx.options;
      const playhead = clock.advance(frame);
      ctx.setReadout('clock', formatClock(playhead));
      playheadParameters.write(getGPUTrajectoryPlayheadParameterValues({playhead}));
      playheadCompiled.encode(commandEncoder, {parameters: undefined});
      const trailSeconds = options.trailMinutes * 60;
      windowParameters.write(
        getGPUTimeWindowFilterParameterValues({
          start: playhead - trailSeconds,
          end: playhead,
          // Fade across the whole trail: weight 1 at the head, 0 at the tail.
          startFadeDuration: trailSeconds
        })
      );
      trailCompiled.encode(commandEncoder, {parameters: undefined});
      if (frame.frameIndex % STATUS_INTERVAL_FRAMES === 0) statusReader.markStale();
      statusReader.flush(commandEncoder);
    },

    getLayers() {
      const options = ctx.options;
      const layers: Layer[] = [];
      if (options.showBackdrop) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'taxi-backdrop',
            coordinateOrigin,
            segments: segmentsBuffer,
            instanceCount: segmentCount,
            widthPixels: 0.75,
            color: BACKDROP_INK
          })
        );
      }
      const comparing = options.compareRamps && options.colorBy === 'fare';
      const sides: {match: RampMatch; side: 'a' | 'b' | undefined}[] = comparing
        ? [
            {match: 'mismatched', side: 'a'},
            {match: 'matched', side: 'b'}
          ]
        : [{match: options.rampMatch, side: undefined}];
      for (const {match, side} of sides) {
        layers.push(
          new AgeFadeTrailLayer({
            id: `taxi-trails${side ? `-${side}` : ''}`,
            coordinateOrigin,
            segments: segmentsBuffer,
            ids: trailIds,
            drawCommands: trailDraw,
            weights: fadeWeights,
            clipFractions,
            widthPixels: TRAIL_WIDTH_STOPS,
            blending: 'additive',
            opacity: 0.9,
            compareSide: side,
            ...getTrailStyle(options, match)
          })
        );
      }
      if (options.showChords && chordCount > 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'taxi-chord-routes',
            coordinateOrigin,
            segments: chordRouteBuffer,
            instanceCount: chordRouteCount,
            widthPixels: 1.5,
            color: ROUTE_INK
          }),
          new SpatialAnalysisSegmentLayer({
            id: 'taxi-chords',
            coordinateOrigin,
            segments: chordBuffer,
            instanceCount: chordCount,
            widthPixels: 1,
            dashArray: [5, 4],
            cap: 'butt',
            color: CHORD_INK
          })
        );
      }
      if (options.inspect && inspectedTrack >= 0) {
        layers.push(
          new SpatialAnalysisSegmentLayer({
            id: 'taxi-inspect-route',
            coordinateOrigin,
            segments: inspectRouteBuffer,
            instanceCount: inspectedVertexCount - 1,
            widthPixels: 2.5,
            color: INSPECT_ROUTE_INK,
            outlineColor: HEAD_HALO,
            outlineWidthPixels: 1.2
          }),
          new SpatialAnalysisPointLayer({
            id: 'taxi-inspect-vertices',
            coordinateOrigin,
            positions: inspectVertexBuffer,
            instanceCount: inspectedVertexCount,
            radiusPixels: 2.2,
            color: INSPECT_TICK_INK
          })
        );
      }
      layers.push(
        new SpatialAnalysisPointLayer({
          id: 'taxi-heads',
          coordinateOrigin,
          positions: currentPositions,
          ids: activeIds,
          drawCommands: markerDraw,
          radiusPixels: HEAD_RADIUS_STOPS,
          color: HEAD_INK,
          outlineColor: HEAD_HALO,
          outlineWidthPixels: 1
        })
      );
      if (options.inspect && inspectedTrack >= 0) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'taxi-inspect-head',
            coordinateOrigin,
            positions: currentPositions,
            ids: inspectIdBuffer,
            instanceCount: 1,
            shape: 'ring',
            radiusPixels: 10,
            outlineWidthPixels: 2,
            color: HEAD_INK
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      statusReader.stop();
      resources.destroy();
    }
  };
}
