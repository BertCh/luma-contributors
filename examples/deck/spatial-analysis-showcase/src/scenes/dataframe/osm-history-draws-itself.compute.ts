// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Layer} from '@deck.gl/core';
import type {Buffer} from '@luma.gl/core';
import {
  getGPUTemporalReductionParameterValues,
  getGPUTimeWindowFilterParameterValues,
  GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH,
  GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH,
  GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH,
  GPUTemporalReduction,
  GPUTimeWindowFilter,
  getGPUTemporalReductionWordParameterValues,
  getInt64TimeWords
} from '@luma-contributors/analysis/dataframe-analysis';
import {
  DrawCommandBuffer,
  GPUCommandGraph,
  type CompiledGPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {SpatialAnalysisPointLayer, SpatialAnalysisRasterLayer} from '../../engine/layers';
import {addKernelPass} from '../../engine/mode-kernels';
import {createPlaybackClock} from '../../engine/playback';
import type {RampName} from '../../engine/ramps';
import {GraphSwap} from '../../engine/graph-swap';
import {formatCount, SpatialAnalysisResources} from '../../engine/resources';
import {SummaryReader} from '../../engine/summary-reader';
import type {SceneContext, SceneInstance} from '../scene';
import {
  createViewImporter,
  dayToYear,
  formatDate,
  loadOsmHistory,
  OSM_DATASET_ID,
  OSM_KIND_COLORS,
  OSM_KIND_LABELS,
  OSM_KINDS,
  scaleToFull
} from './osm-history-data';

/** Option state of the "city draws itself" scene. */
export type OsmDrawsItselfOptions = {
  play: boolean;
  time: number;
  speed: number;
  loop: boolean;
  showPoints: boolean;
  colorBy: 'year' | 'kind';
  ramp: Exclude<RampName, 'grayscale' | 'diverging'>;
  pointSize: number;
  showGlow: boolean;
  glowDays: number;
  showCells: boolean;
  cellSize: '1000' | '500' | '250';
  bucket: 'quarter' | 'half' | 'year';
  cellMode: 'cumulative' | 'bucket';
  ceiling: number;
  cellOpacity: number;
};

/** Bucket width in days for each `bucket` option; the compiled bucket count covers all of them. */
export const BUCKET_DAYS: Record<OsmDrawsItselfOptions['bucket'], number> = {
  quarter: 365.2425 / 4,
  half: 365.2425 / 2,
  year: 365.2425
};

/** Compile-time bucket count: a quarter-year bucket over the 19 years of data, with headroom. */
const BUCKET_COUNT = 80;
const STATUS_INTERVAL_FRAMES = 12;
const AUDIT_KIND_COUNT = 6;
const AUDIT_BUCKET_COUNT = 1;
const AUDIT_WINDOW_SECONDS = 30 * 24 * 60 * 60;
const AUDIT_SUMMARY_BYTES_PER_PATH = AUDIT_KIND_COUNT * 5 * 4 + AUDIT_KIND_COUNT * 4 + 8;

type CellVariant = {
  cellSize: number;
  columns: number;
  rows: number;
  cellCount: number;
  bounds: [number, number, number, number];
  cellValues: Buffer;
  reduce: CompiledGPUCommandGraph<void>;
  values: CompiledGPUCommandGraph<void>;
  /** Buffers owned by this variant, released after its graphs. */
  buffers: Buffer[];
  reduced: boolean;
  summary: SummaryReader;
  /** Per-bucket totals and occupied-slot count, read once per reduction. */
  totals: SummaryReader;
};

/**
 * The city draws itself. `GPUTemporalReduction` folds the 400k sampled node creations into one
 * record per (cell, time bucket); a kernel then turns the buckets up to the playhead into one
 * density value per cell for a raster layer. Two `GPUTimeWindowFilter` graphs pick the nodes
 * created before the playhead (the ink) and in the last few days (the glow) without recompiling.
 * The cell size is compile-time (it changes the dense slot table); the bucket width, the playhead
 * and every display option are parameter writes.
 */
export async function createOsmDrawsItself(
  ctx: SceneContext<OsmDrawsItselfOptions>
): Promise<SceneInstance<OsmDrawsItselfOptions>> {
  const {device} = ctx;
  const dataset = ctx.datasets.get(OSM_DATASET_ID);
  const history = loadOsmHistory(dataset);
  const count = history.count;
  const sourceSeconds = dataset.column<Uint32Array>('timestamp');
  const epochOriginSeconds = BigInt(Math.floor(Number(dataset.properties.timeOriginMs) / 1000));
  const exactTimestamps = new BigInt64Array(count);
  let timestampsAreSorted = true;
  for (let row = 0; row < count; row++) {
    if (row > 0 && sourceSeconds[row] < sourceSeconds[row - 1]) timestampsAreSorted = false;
    exactTimestamps[row] = epochOriginSeconds + BigInt(sourceSeconds[row]);
  }
  const exactTimestampWords = getInt64TimeWords(exactTimestamps);
  const sourceRowValues = Float32Array.from({length: count}, (_value, row) => row);
  const resources = new SpatialAnalysisResources(device, 'osm-draws');
  const coordinateOrigin: [number, number, number] = [history.origin[0], history.origin[1], 0];
  let destroyed = false;

  // ---- Static inputs ------------------------------------------------------------------------------
  const positionsBuffer = resources.createBuffer('positions', history.positions);
  const daysBuffer = resources.createBuffer('days', history.days);
  const kindBuffer = resources.createBuffer('kind', history.kind);
  const onesBuffer = resources.createBuffer('ones', new Float32Array(count).fill(1));
  const reduceParameters = resources.createParameterBuffer(
    'reduce',
    'float32',
    GPU_TEMPORAL_REDUCTION_PARAMETER_LENGTH
  );
  const valueParameters = resources.createParameterBuffer('cell-values', 'uint32', 4);

  // ---- Playback and window graphs -----------------------------------------------------------------
  const clock = createPlaybackClock(
    ctx,
    {time: 'time', play: 'play', speed: 'speed', loop: 'loop'},
    {range: [0, Math.floor(history.maxDay)], secondsPerLoop: 60, step: 7}
  );

  async function createWindowGraph(name: string) {
    const ids = resources.createBuffer(`${name}-ids`, count * 4);
    const countBuffer = resources.createBuffer(`${name}-count`, 4);
    const overflow = resources.createBuffer(`${name}-overflow`, 4);
    const parameters = resources.createParameterBuffer(
      `${name}-window`,
      'float32',
      GPU_TIME_WINDOW_FILTER_PARAMETER_LENGTH
    );
    const draw = resources.track(
      new DrawCommandBuffer(device, {
        id: `osm-${name}-draw`,
        type: 'draw',
        commands: [{vertexCount: 6, instanceCount: 0}]
      })
    );
    const graph = new GPUCommandGraph<void>(device, {id: `osm-${name}`});
    const v = createViewImporter(graph, name);
    graph.add(
      new GPUTimeWindowFilter({
        id: `${name}-window`,
        timestamps: v('days', daysBuffer, 'float32', count),
        window: parameters.importToGraph(graph),
        output: {
          ids: v('ids', ids, 'uint32', count),
          status: {
            count: v('count', countBuffer, 'uint32', 1),
            overflow: v('overflow', overflow, 'uint32', 1)
          },
          drawInstanceCount: graph.importGPUData(`${name}-draw-count`, draw.getInstanceCountData(0))
        }
      })
    );
    return {
      ids,
      countBuffer,
      parameters,
      draw,
      compiled: resources.track(await graph.compileAsync())
    };
  }
  const ink = await createWindowGraph('ink');
  const glow = await createWindowGraph('glow');

  // ---- Exact 30-day time audit -------------------------------------------------------------------
  const auditParameters = resources.createParameterBuffer(
    'audit-window',
    'uint32',
    GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH
  );
  const auditTimestampsBuffer = resources.createBuffer('audit-timestamps', exactTimestampWords);
  const auditKindBuffer = resources.createBuffer('audit-kinds', history.kind);
  const auditRowValuesBuffer = resources.createBuffer('audit-source-rows', sourceRowValues);
  const auditPaths = [] as {
    sorted: boolean;
    counts: Buffer;
    minimums: Buffer;
    maximums: Buffer;
    first: Buffer;
    last: Buffer;
    occupiedIds: Buffer;
    occupiedCount: Buffer;
    occupiedOverflow: Buffer;
    graph: CompiledGPUCommandGraph<void>;
  }[];

  async function buildAuditPath(name: string, sorted: boolean) {
    const counts = resources.createBuffer(`${name}-counts`, AUDIT_KIND_COUNT * 4);
    const minimums = resources.createBuffer(`${name}-minimums`, AUDIT_KIND_COUNT * 4);
    const maximums = resources.createBuffer(`${name}-maximums`, AUDIT_KIND_COUNT * 4);
    const first = resources.createBuffer(`${name}-first`, AUDIT_KIND_COUNT * 4);
    const last = resources.createBuffer(`${name}-last`, AUDIT_KIND_COUNT * 4);
    const occupiedIds = resources.createBuffer(`${name}-occupied-ids`, AUDIT_KIND_COUNT * 4);
    const occupiedCount = resources.createBuffer(`${name}-occupied-count`, 4);
    const occupiedOverflow = resources.createBuffer(`${name}-occupied-overflow`, 4);
    const graph = new GPUCommandGraph<void>(device, {id: `osm-audit-${name}`});
    const view = createViewImporter(graph, name);
    graph.add(
      new GPUTemporalReduction({
        id: `osm-audit-reduction-${name}`,
        cellIds: view('kinds', auditKindBuffer, 'uint32', count),
        timestamps: view('timestamps', auditTimestampsBuffer, 'uint32x2', count),
        values: view('source-rows', auditRowValuesBuffer, 'float32', count),
        parameters: auditParameters.importToGraph(graph),
        cellCount: AUDIT_KIND_COUNT,
        bucketCount: AUDIT_BUCKET_COUNT,
        sorted,
        output: {
          counts: view('counts', counts, 'uint32', AUDIT_KIND_COUNT),
          minimums: view('minimums', minimums, 'float32', AUDIT_KIND_COUNT),
          maximums: view('maximums', maximums, 'float32', AUDIT_KIND_COUNT),
          first: view('first', first, 'float32', AUDIT_KIND_COUNT),
          last: view('last', last, 'float32', AUDIT_KIND_COUNT),
          occupiedSlots: {
            ids: view('occupied-ids', occupiedIds, 'uint32', AUDIT_KIND_COUNT),
            status: {
              count: view('occupied-count', occupiedCount, 'uint32', 1),
              overflow: view('occupied-overflow', occupiedOverflow, 'uint32', 1)
            }
          }
        }
      })
    );
    return {
      sorted,
      counts,
      minimums,
      maximums,
      first,
      last,
      occupiedIds,
      occupiedCount,
      occupiedOverflow,
      graph: resources.track(await graph.compileAsync())
    };
  }

  if (timestampsAreSorted) {
    auditPaths.push(await buildAuditPath('sorted', true));
    auditPaths.push(await buildAuditPath('full-scan', false));
  }

  const auditMismatchBuffer = resources.createBuffer(
    'audit-mismatches',
    (AUDIT_KIND_COUNT + 1) * 4
  );
  let auditComparison: CompiledGPUCommandGraph<void> | null = null;
  if (auditPaths.length === 2) {
    const [sortedPath, fullScanPath] = auditPaths;
    const comparisonGraph = new GPUCommandGraph<void>(device, {id: 'osm-audit-compare'});
    const compare = createViewImporter(comparisonGraph, 'compare');
    addKernelPass(comparisonGraph, {
      id: 'osm-audit-compare-count-extrema',
      invocationCount: AUDIT_KIND_COUNT,
      bindings: [
        {
          name: 'sortedCounts',
          view: compare('sorted-counts', sortedPath.counts, 'uint32', AUDIT_KIND_COUNT),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'fullCounts',
          view: compare('full-counts', fullScanPath.counts, 'uint32', AUDIT_KIND_COUNT),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'sortedMinimums',
          view: compare('sorted-minimums', sortedPath.minimums, 'float32', AUDIT_KIND_COUNT),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'fullMinimums',
          view: compare('full-minimums', fullScanPath.minimums, 'float32', AUDIT_KIND_COUNT),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'sortedMaximums',
          view: compare('sorted-maximums', sortedPath.maximums, 'float32', AUDIT_KIND_COUNT),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'fullMaximums',
          view: compare('full-maximums', fullScanPath.maximums, 'float32', AUDIT_KIND_COUNT),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'mismatches',
          view: compare('mismatches', auditMismatchBuffer, 'uint32', AUDIT_KIND_COUNT + 1),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: `var mismatch = sortedCounts[sortedCountsOffset + index] != fullCounts[fullCountsOffset + index];
  if (sortedCounts[sortedCountsOffset + index] > 0u) {
    mismatch = mismatch || sortedMinimums[sortedMinimumsOffset + index] != fullMinimums[fullMinimumsOffset + index];
    mismatch = mismatch || sortedMaximums[sortedMaximumsOffset + index] != fullMaximums[fullMaximumsOffset + index];
  }
  mismatches[mismatchesOffset + index] = select(0u, 1u, mismatch);`
    });
    addKernelPass(comparisonGraph, {
      id: 'osm-audit-compare-first-last',
      invocationCount: AUDIT_KIND_COUNT,
      bindings: [
        {
          name: 'sortedCounts',
          view: compare('sorted-counts', sortedPath.counts, 'uint32', AUDIT_KIND_COUNT),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'sortedFirst',
          view: compare('sorted-first', sortedPath.first, 'float32', AUDIT_KIND_COUNT),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'fullFirst',
          view: compare('full-first', fullScanPath.first, 'float32', AUDIT_KIND_COUNT),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'sortedLast',
          view: compare('sorted-last', sortedPath.last, 'float32', AUDIT_KIND_COUNT),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'fullLast',
          view: compare('full-last', fullScanPath.last, 'float32', AUDIT_KIND_COUNT),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'mismatches',
          view: compare('mismatches', auditMismatchBuffer, 'uint32', AUDIT_KIND_COUNT + 1),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: `var mismatch = mismatches[mismatchesOffset + index] != 0u;
  if (sortedCounts[sortedCountsOffset + index] > 0u) {
    mismatch = mismatch || sortedFirst[sortedFirstOffset + index] != fullFirst[fullFirstOffset + index];
    mismatch = mismatch || sortedLast[sortedLastOffset + index] != fullLast[fullLastOffset + index];
  }
  mismatches[mismatchesOffset + index] = select(0u, 1u, mismatch);`
    });
    addKernelPass(comparisonGraph, {
      id: 'osm-audit-compare-occupied',
      invocationCount: 1,
      bindings: [
        {
          name: 'sortedOccupiedIds',
          view: compare('sorted-occupied-ids', sortedPath.occupiedIds, 'uint32', AUDIT_KIND_COUNT),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'fullOccupiedIds',
          view: compare('full-occupied-ids', fullScanPath.occupiedIds, 'uint32', AUDIT_KIND_COUNT),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'sortedOccupiedCount',
          view: compare('sorted-occupied-count', sortedPath.occupiedCount, 'uint32', 1),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'fullOccupiedCount',
          view: compare('full-occupied-count', fullScanPath.occupiedCount, 'uint32', 1),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'sortedOverflow',
          view: compare('sorted-overflow', sortedPath.occupiedOverflow, 'uint32', 1),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'fullOverflow',
          view: compare('full-overflow', fullScanPath.occupiedOverflow, 'uint32', 1),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'mismatches',
          view: compare('mismatches', auditMismatchBuffer, 'uint32', AUDIT_KIND_COUNT + 1),
          type: 'u32',
          access: 'read_write'
        }
      ],
      body: `let sortedCount = sortedOccupiedCount[sortedOccupiedCountOffset];
  let fullCount = fullOccupiedCount[fullOccupiedCountOffset];
  var mismatch = sortedCount != fullCount || sortedOverflow[sortedOverflowOffset] != fullOverflow[fullOverflowOffset];
  let occupiedCount = min(sortedCount, ${AUDIT_KIND_COUNT}u);
  for (var occupied = 0u; occupied < occupiedCount; occupied = occupied + 1u) {
    mismatch = mismatch || sortedOccupiedIds[sortedOccupiedIdsOffset + occupied] != fullOccupiedIds[fullOccupiedIdsOffset + occupied];
  }
  mismatches[mismatchesOffset + ${AUDIT_KIND_COUNT}u] = select(0u, 1u, mismatch);`
    });
    auditComparison = resources.track(await comparisonGraph.compileAsync());
  }

  let currentAuditRangeKey = '';
  let auditSummary: SummaryReader | null = null;
  if (auditPaths.length === 2) {
    auditSummary = new SummaryReader(
      resources,
      'osm-audit-summary',
      [
        ...auditPaths.flatMap(path => [
          {buffer: path.counts, size: AUDIT_KIND_COUNT * 4},
          {buffer: path.minimums, size: AUDIT_KIND_COUNT * 4},
          {buffer: path.maximums, size: AUDIT_KIND_COUNT * 4},
          {buffer: path.first, size: AUDIT_KIND_COUNT * 4},
          {buffer: path.last, size: AUDIT_KIND_COUNT * 4},
          {buffer: path.occupiedIds, size: AUDIT_KIND_COUNT * 4},
          {buffer: path.occupiedCount, size: 4},
          {buffer: path.occupiedOverflow, size: 4}
        ]),
        {buffer: auditMismatchBuffer, size: (AUDIT_KIND_COUNT + 1) * 4},
        {buffer: auditParameters.buffer, size: GPU_TEMPORAL_REDUCTION_WORD_PARAMETER_LENGTH * 4}
      ],
      bytes => {
        if (destroyed || !currentAuditRangeKey || auditPaths.length !== 2) return;
        const words = new Uint32Array(bytes);
        const parameterOffset = (2 * AUDIT_SUMMARY_BYTES_PER_PATH + (AUDIT_KIND_COUNT + 1) * 4) / 4;
        const resultRangeKey = `${words[parameterOffset]}:${words[parameterOffset + 1]}`;
        if (resultRangeKey !== currentAuditRangeKey) return;
        const mismatchOffset = (2 * AUDIT_SUMMARY_BYTES_PER_PATH) / 4;
        const mismatchCount = words
          .subarray(mismatchOffset, mismatchOffset + AUDIT_KIND_COUNT + 1)
          .reduce((sum, value) => sum + value, 0);
        let byteOffset = 0;
        const summaries = auditPaths.map(() => {
          const readWords = (length: number) => {
            const values = new Uint32Array(bytes, byteOffset, length);
            byteOffset += length * 4;
            return values;
          };
          const readFloats = (length: number) => {
            const values = new Float32Array(bytes, byteOffset, length);
            byteOffset += length * 4;
            return values;
          };
          const counts = readWords(AUDIT_KIND_COUNT);
          const minimums = readFloats(AUDIT_KIND_COUNT);
          const maximums = readFloats(AUDIT_KIND_COUNT);
          const first = readFloats(AUDIT_KIND_COUNT);
          const last = readFloats(AUDIT_KIND_COUNT);
          const occupiedIds = readWords(AUDIT_KIND_COUNT);
          const occupiedCount = readWords(1)[0];
          const overflow = readWords(1)[0];
          const lines = Array.from({length: AUDIT_KIND_COUNT}, (_value, kind) => {
            if (counts[kind] === 0) return `${OSM_KIND_LABELS[OSM_KINDS[kind]]}: empty`;
            const firstRow = Math.round(first[kind]);
            const lastRow = Math.round(last[kind]);
            const rowTimestamp = (row: number) =>
              Number(epochOriginSeconds + BigInt(sourceSeconds[row]));
            const describeRow = (row: number) =>
              `row ${row}, contributor #${history.contributor[row] + 1}, ${new Date(rowTimestamp(row) * 1000).toISOString().replace('.000Z', 'Z')}`;
            return `${OSM_KIND_LABELS[OSM_KINDS[kind]]}: n ${formatCount(counts[kind])}; row ${Math.round(minimums[kind])}–${Math.round(maximums[kind])}; first ${describeRow(firstRow)}; last ${describeRow(lastRow)}`;
          });
          return {
            lines,
            occupiedIds: Array.from(
              occupiedIds.subarray(0, Math.min(occupiedCount, AUDIT_KIND_COUNT))
            ),
            occupiedCount,
            overflow
          };
        });
        const sortedOccupied = summaries[0].occupiedIds.map(
          kind => OSM_KIND_LABELS[OSM_KINDS[kind]]
        );
        const fullOccupied = summaries[1].occupiedIds.map(kind => OSM_KIND_LABELS[OSM_KINDS[kind]]);
        ctx.setReadout(
          'auditSorted',
          `${summaries[0].lines.join('\n')}\nOccupied: ${summaries[0].occupiedCount} [${sortedOccupied.join(', ')}], overflow ${summaries[0].overflow}`
        );
        ctx.setReadout(
          'auditFullScan',
          `${summaries[1].lines.join('\n')}\nOccupied: ${summaries[1].occupiedCount} [${fullOccupied.join(', ')}], overflow ${summaries[1].overflow}`
        );
        ctx.setReadout(
          'auditMismatch',
          summaries.some(
            summary => summary.overflow > 0 || summary.occupiedCount > AUDIT_KIND_COUNT
          )
            ? `Fail closed: occupied-slot capacity overflow; ${mismatchCount} mismatches across 6 kind slots and status`
            : `${mismatchCount} mismatches across 6 kind slots and occupied-slot status`
        );
      }
    );
  } else {
    ctx.setReadout(
      'auditSorted',
      'Unavailable: timestamp source is not monotonic; sorted:true was not enabled.'
    );
    ctx.setReadout(
      'auditFullScan',
      'Unavailable: exact sorted-versus-full-scan comparison is fail-closed.'
    );
    ctx.setReadout('auditMismatch', 'Not compared');
  }

  let auditDirty = true;

  // ---- Cell variant (compile-time cell size) ------------------------------------------------------
  let variant: CellVariant | null = null;
  let valuesDirty = true;
  let windowsDirty = true;
  let lastBucket = -1;
  let totalsScaled: Float32Array | null = null;
  let usedBuckets = 0;
  let peakText = '';
  let occupiedText = '';

  async function buildVariant(cellSize: number): Promise<CellVariant> {
    const [minX, minY, maxX, maxY] = history.bounds;
    const columns = Math.ceil((maxX - minX) / cellSize) + 1;
    const rows = Math.ceil((maxY - minY) / cellSize) + 1;
    const cellCount = columns * rows;
    const slotCount = cellCount * BUCKET_COUNT;
    const tag = `${cellSize}`;
    const cellIds = resources.createBuffer(`cell-ids-${tag}`, count * 4);
    const counts = resources.createBuffer(`slot-counts-${tag}`, slotCount * 4);
    const slotMin = resources.createBuffer(`slot-min-${tag}`, slotCount * 4);
    const slotMax = resources.createBuffer(`slot-max-${tag}`, slotCount * 4);
    const slotFirst = resources.createBuffer(`slot-first-${tag}`, slotCount * 4);
    const slotLast = resources.createBuffer(`slot-last-${tag}`, slotCount * 4);
    const occupiedIds = resources.createBuffer(
      `occupied-ids-${tag}`,
      Math.min(slotCount, count) * 4
    );
    const occupiedCount = resources.createBuffer(`occupied-count-${tag}`, 4);
    const occupiedOverflow = resources.createBuffer(`occupied-overflow-${tag}`, 4);
    const bucketTotals = resources.createBuffer(`bucket-totals-${tag}`, BUCKET_COUNT * 4);
    const cellValues = resources.createBuffer(`cell-values-${tag}`, cellCount * 4);

    // Graph 1: node -> cell, then the dense (cell, bucket) reduction and a total per bucket.
    const reduceGraph = new GPUCommandGraph<void>(device, {id: `osm-reduce-${tag}`});
    const r = createViewImporter(reduceGraph, 'r');
    const cellIdsView = r('cell-ids', cellIds, 'uint32', count);
    addKernelPass(reduceGraph, {
      id: `osm-cell-ids-${tag}`,
      invocationCount: count,
      declarations: `const MIN_X: f32 = ${minX.toFixed(3)};
const MIN_Y: f32 = ${minY.toFixed(3)};
const CELL: f32 = ${cellSize.toFixed(1)};
const COLUMNS: u32 = ${columns}u;
const ROWS: u32 = ${rows}u;`,
      bindings: [
        {
          name: 'positions',
          view: r('positions', positionsBuffer, 'float32x2', count),
          type: 'f32',
          access: 'read'
        },
        {name: 'cellIds', view: cellIdsView, type: 'u32', access: 'read_write'}
      ],
      body: `let x = positions[positionsOffset + index * 2u];
  let y = positions[positionsOffset + index * 2u + 1u];
  let column = u32(max(floor((x - MIN_X) / CELL), 0.0));
  let row = u32(max(floor((y - MIN_Y) / CELL), 0.0));
  cellIds[cellIdsOffset + index] = select(0xffffffffu, row * COLUMNS + column, column < COLUMNS && row < ROWS);`
    });
    const countsView = r('counts', counts, 'uint32', slotCount);
    reduceGraph.add(
      new GPUTemporalReduction({
        id: `osm-reduction-${tag}`,
        cellIds: cellIdsView,
        timestamps: r('days', daysBuffer, 'float32', count),
        values: r('ones', onesBuffer, 'float32', count),
        parameters: reduceParameters.importToGraph(reduceGraph),
        cellCount,
        bucketCount: BUCKET_COUNT,
        output: {
          counts: countsView,
          minimums: r('slot-min', slotMin, 'float32', slotCount),
          maximums: r('slot-max', slotMax, 'float32', slotCount),
          first: r('slot-first', slotFirst, 'float32', slotCount),
          last: r('slot-last', slotLast, 'float32', slotCount),
          occupiedSlots: {
            ids: r('occupied-ids', occupiedIds, 'uint32', Math.min(slotCount, count)),
            status: {
              count: r('occupied-count', occupiedCount, 'uint32', 1),
              overflow: r('occupied-overflow', occupiedOverflow, 'uint32', 1)
            }
          }
        }
      })
    );
    addKernelPass(reduceGraph, {
      id: `osm-bucket-totals-${tag}`,
      invocationCount: BUCKET_COUNT,
      declarations: `const CELLS: u32 = ${cellCount}u;
const BUCKETS: u32 = ${BUCKET_COUNT}u;`,
      bindings: [
        {name: 'counts', view: countsView, type: 'u32', access: 'read'},
        {
          name: 'totals',
          view: r('bucket-totals', bucketTotals, 'float32', BUCKET_COUNT),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `var sum = 0u;
  for (var cell = 0u; cell < CELLS; cell = cell + 1u) {
    sum = sum + counts[countsOffset + cell * BUCKETS + index];
  }
  totals[totalsOffset + index] = f32(sum);`
    });

    // Graph 2: one density value per cell from the buckets up to the playhead's bucket.
    const valuesGraph = new GPUCommandGraph<void>(device, {id: `osm-cell-values-${tag}`});
    const w = createViewImporter(valuesGraph, 'w');
    addKernelPass(valuesGraph, {
      id: `osm-cell-sum-${tag}`,
      invocationCount: cellCount,
      declarations: `const BUCKETS: u32 = ${BUCKET_COUNT}u;`,
      bindings: [
        {
          name: 'counts',
          view: w('counts', counts, 'uint32', slotCount),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'parameters',
          view: valueParameters.importToGraph(valuesGraph),
          type: 'u32',
          access: 'read'
        },
        {
          name: 'values',
          view: w('cell-values', cellValues, 'float32', cellCount),
          type: 'f32',
          access: 'read_write'
        }
      ],
      body: `let current = min(parameters[parametersOffset], BUCKETS - 1u);
  let cumulative = parameters[parametersOffset + 1u] == 0u;
  let base = index * BUCKETS;
  var sum = 0u;
  if (cumulative) {
    for (var bucket = 0u; bucket <= current; bucket = bucket + 1u) {
      sum = sum + counts[countsOffset + base + bucket];
    }
  } else {
    sum = counts[countsOffset + base + current];
  }
  values[valuesOffset + index] = f32(sum);`
    });

    const summary = new SummaryReader(
      resources,
      `osm-cells-${tag}`,
      [
        {buffer: ink.countBuffer, size: 4},
        {buffer: glow.countBuffer, size: 4},
        {buffer: cellValues, size: cellCount * 4}
      ],
      bytes => {
        if (destroyed || variant?.cellValues !== cellValues) return;
        const words = new Uint32Array(bytes);
        const floats = new Float32Array(bytes);
        let peak = 0;
        let occupied = 0;
        for (let cell = 0; cell < cellCount; cell++) {
          const value = floats[2 + cell];
          if (value > 0) occupied++;
          if (value > peak) peak = value;
        }
        const areaKm2 = (cellSize / 1000) ** 2;
        ctx.setReadout('shown', `${formatCount(scaleToFull(history, words[0]))} nodes`);
        ctx.setReadout('glowing', `${formatCount(scaleToFull(history, words[1]))} nodes`);
        ctx.setReadout('cells', `${formatCount(occupied)} of ${formatCount(cellCount)}`);
        peakText = `${formatCount(scaleToFull(history, peak))} nodes in one ${cellSize} m cell (${formatCount(scaleToFull(history, peak) / areaKm2)} per km2)`;
        ctx.setReadout('peak', peakText);
      }
    );
    const totalsReader = new SummaryReader(
      resources,
      `osm-totals-${tag}`,
      [
        {buffer: occupiedCount, size: 4},
        {buffer: bucketTotals, size: BUCKET_COUNT * 4}
      ],
      bytes => {
        if (destroyed) return;
        const words = new Uint32Array(bytes, 0, 1);
        const floats = new Float32Array(bytes, 4, BUCKET_COUNT);
        const width = BUCKET_DAYS[ctx.options.bucket];
        usedBuckets = Math.min(BUCKET_COUNT, Math.ceil((history.maxDay + 1) / width));
        totalsScaled = Float32Array.from(
          floats.subarray(0, usedBuckets),
          v => v / history.full.sampleFraction
        );
        occupiedText = `${formatCount(words[0])} (cell, bucket) records from ${formatCount(count)} nodes`;
        ctx.setReadout('records', occupiedText);
        lastBucket = -1;
        publishChart();
      }
    );

    return {
      cellSize,
      columns,
      rows,
      cellCount,
      bounds: [minX, minY, minX + columns * cellSize, minY + rows * cellSize],
      cellValues,
      reduce: resources.track(await reduceGraph.compileAsync()),
      values: resources.track(await valuesGraph.compileAsync()),
      buffers: [
        cellIds,
        counts,
        slotMin,
        slotMax,
        slotFirst,
        slotLast,
        occupiedIds,
        occupiedCount,
        occupiedOverflow,
        bucketTotals,
        cellValues
      ],
      reduced: false,
      summary,
      totals: totalsReader
    };
  }

  function disposeVariant(old: CellVariant): void {
    old.summary.stop();
    old.totals.stop();
    resources.release(old.reduce);
    resources.release(old.values);
    for (const buffer of old.buffers) resources.release(buffer);
  }

  /** Swaps a compiled variant in, then releases the one it replaces. */
  function commitVariant(next: CellVariant): void {
    const old = variant;
    variant = next;
    if (old) disposeVariant(old);
    valuesDirty = true;
    windowsDirty = true;
    ctx.setReadout('grid', `${next.columns} x ${next.rows} cells of ${next.cellSize} m`);
    ctx.requestLayers();
  }

  // The previous variant keeps encoding until the new one has compiled.
  const variantSwap = resources.track(new GraphSwap());
  function setVariant(cellSize: number): Promise<boolean> {
    return variantSwap.request({
      build: () => buildVariant(cellSize),
      commit: commitVariant,
      discard: disposeVariant
    });
  }

  function currentBucket(playhead: number): number {
    return Math.min(
      Math.max(Math.floor(playhead / BUCKET_DAYS[ctx.options.bucket]), 0),
      BUCKET_COUNT - 1
    );
  }

  function publishChart(): void {
    if (!totalsScaled) return;
    const width = BUCKET_DAYS[ctx.options.bucket];
    const bucket = currentBucket(clock.time);
    const startYear = dayToYear(history, 0);
    ctx.setChart('perBucket', {
      kind: 'histogram',
      values: totalsScaled,
      xDomain: [startYear, startYear + (usedBuckets * width) / 365.2425],
      xLabel: 'year',
      yLabel: 'nodes created per bucket (estimate)',
      formatX: value => value.toFixed(0),
      height: 120,
      markers: [{x: dayToYear(history, bucket * width)}],
      description:
        'Nodes created per time bucket across the whole city, scaled from the sample to the full history; the vertical rule is the playhead.'
    });
  }

  ctx.setReadout('rows', `${formatCount(count)} of ${formatCount(history.full.fullCount)} nodes`);
  await setVariant(Number(ctx.options.cellSize));

  return {
    getCompiledGraphs: () =>
      variant
        ? [
            ink.compiled,
            glow.compiled,
            variant.reduce,
            variant.values,
            ...auditPaths.map(path => path.graph),
            ...(auditComparison ? [auditComparison] : [])
          ]
        : [],

    setOption(id, _value, state) {
      switch (id) {
        case 'cellSize':
          void setVariant(Number(state.cellSize));
          break;
        case 'bucket':
          if (variant) variant.reduced = false;
          valuesDirty = true;
          break;
        case 'cellMode':
          valuesDirty = true;
          break;
        case 'glowDays':
        case 'showGlow':
          windowsDirty = true;
          ctx.requestLayers();
          break;
        case 'time':
        case 'play':
        case 'speed':
        case 'loop':
          break;
        default:
          ctx.requestLayers();
      }
    },

    onThemeChange() {
      ctx.requestLayers();
    },

    encode(commandEncoder, frame) {
      if (!variant) return;
      const options = ctx.options;
      const playhead = clock.advance(frame);
      const width = BUCKET_DAYS[options.bucket];
      ctx.setReadout('clock', formatDate(history, playhead));

      if (auditPaths.length === 2) {
        const elapsedSeconds = Math.floor(playhead * 86_400);
        const auditOrigin = epochOriginSeconds + BigInt(elapsedSeconds - AUDIT_WINDOW_SECONDS);
        const auditParametersValue = getGPUTemporalReductionWordParameterValues(
          auditOrigin,
          AUDIT_WINDOW_SECONDS
        );
        const rangeKey = `${auditParametersValue[0]}:${auditParametersValue[1]}`;
        if (rangeKey !== currentAuditRangeKey) {
          currentAuditRangeKey = rangeKey;
          auditParameters.write(auditParametersValue);
          auditDirty = true;
          const rangeStart = new Date(Number(auditOrigin) * 1000).toISOString();
          const rangeEnd = new Date(
            Number(auditOrigin + BigInt(AUDIT_WINDOW_SECONDS)) * 1000
          ).toISOString();
          ctx.setReadout(
            'auditWindow',
            `[${rangeStart}, ${rangeEnd}) UTC — fixed 30 days; timestamp order verified across ${formatCount(count)} source rows`
          );
        }
      }

      if (auditDirty && auditComparison) {
        for (const path of auditPaths) path.graph.encode(commandEncoder, {parameters: undefined});
        auditComparison.encode(commandEncoder, {parameters: undefined});
        auditDirty = false;
        auditSummary?.markStale();
      }
      auditSummary?.flush(commandEncoder);

      if (!variant.reduced) {
        reduceParameters.write(getGPUTemporalReductionParameterValues(0, width));
        variant.reduce.encode(commandEncoder, {parameters: undefined});
        variant.reduced = true;
        valuesDirty = true;
        variant.totals.request(commandEncoder);
      } else {
        variant.totals.flush(commandEncoder);
      }

      const bucket = currentBucket(playhead);
      if (bucket !== lastBucket) {
        lastBucket = bucket;
        valuesDirty = true;
        publishChart();
      }
      if (valuesDirty) {
        valueParameters.write(
          Uint32Array.of(bucket, options.cellMode === 'cumulative' ? 0 : 1, 0, 0)
        );
        variant.values.encode(commandEncoder, {parameters: undefined});
        valuesDirty = false;
        variant.summary.markStale();
      }

      if (clock.moved || windowsDirty) {
        ink.parameters.write(getGPUTimeWindowFilterParameterValues({start: -1, end: playhead}));
        ink.compiled.encode(commandEncoder, {parameters: undefined});
        if (options.showGlow) {
          glow.parameters.write(
            getGPUTimeWindowFilterParameterValues({
              start: playhead - options.glowDays,
              end: playhead
            })
          );
          glow.compiled.encode(commandEncoder, {parameters: undefined});
        } else {
          glow.parameters.write(getGPUTimeWindowFilterParameterValues({start: 1e9, end: 1e9 + 1}));
          glow.compiled.encode(commandEncoder, {parameters: undefined});
        }
        windowsDirty = false;
        variant.summary.markStale();
      }
      if (frame.frameIndex % STATUS_INTERVAL_FRAMES === 0 && clock.moved)
        variant.summary.markStale();
      variant.summary.flush(commandEncoder);
    },

    getLayers() {
      if (!variant) return [];
      const options = ctx.options;
      const dark = ctx.theme() === 'dark';
      const layers: Layer[] = [];
      const areaKm2 = (variant.cellSize / 1000) ** 2;
      if (options.showCells) {
        layers.push(
          new SpatialAnalysisRasterLayer({
            id: `osm-cells-${variant.cellSize}`,
            coordinateOrigin,
            gridSize: [variant.columns, variant.rows],
            bounds: variant.bounds,
            values: variant.cellValues,
            valueFormat: 'float32',
            colormap: options.ramp,
            valueRange: [0, options.ceiling],
            valueScale: 1 / areaKm2,
            sqrtScale: true,
            discardAtOrBelow: 0,
            color: [255, 255, 255, 255],
            opacity: options.cellOpacity
          })
        );
      }
      if (options.showPoints) {
        const byYear = options.colorBy === 'year';
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'osm-ink',
            coordinateOrigin,
            positions: positionsBuffer,
            ids: ink.ids,
            drawCommands: ink.draw,
            values: byYear ? daysBuffer : kindBuffer,
            valueFormat: byYear ? 'float32' : 'uint32',
            colormap: byYear ? options.ramp : 'category',
            valueRange: [0, history.maxDay],
            palette: OSM_KIND_COLORS,
            radiusPixels: options.pointSize,
            opacity: options.showCells ? 0.8 : 0.9
          })
        );
      }
      if (options.showGlow) {
        layers.push(
          new SpatialAnalysisPointLayer({
            id: 'osm-glow',
            coordinateOrigin,
            positions: positionsBuffer,
            ids: glow.ids,
            drawCommands: glow.draw,
            colormap: 'uniform',
            color: dark ? [255, 250, 220, 255] : [30, 30, 40, 255],
            radiusPixels: options.pointSize + 1.6,
            opacity: 0.95
          })
        );
      }
      return layers;
    },

    destroy() {
      destroyed = true;
      variant?.summary.stop();
      variant?.totals.stop();
      auditSummary?.stop();
      resources.destroy();
    }
  };
}
