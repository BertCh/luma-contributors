// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {GPUData, GPUVector} from '@luma.gl/graph/gpu-data';
import {
  DEFAULT_GPU_COLUMN_STATISTICS_BINS,
  MAXIMUM_GPU_COLUMN_STATISTICS_ROWS,
  getGPUColumnStatisticsByteLength,
  readGPUColumnStatistics,
  type GPUColumnStatistics,
  type KeplerArrowDataset,
  type KeplerComputeSession,
  type KeplerDatasetLease,
  type KeplerSessionExplorer
} from '@luma-contributors/kepler';
import type * as arrow from 'apache-arrow';
import type {ColumnStats, ColumnStatsType} from '@kepler.gl/utils';
import {getOriginalFieldType, readOriginalColumnValues} from './original-column-values';
import {widenResidentChartMeasure} from './resident-chart-measure-widening';
import {makeNumericColumnStatsFromGPU} from './resident-gpu-column-statistics';
import {classifyTableSortType} from './resident-table-sort-keys';

const CONVERSION_CHUNK_ROWS = 1 << 20;

export type WideStatisticsKind = 'wide-numeric' | 'dictionary-string';
export type WideStatisticsRoute =
  | {supported: true; kind: WideStatisticsKind; reason: string}
  | {supported: false; reason: string};

export type ResidentWideColumnStatistics = {
  fieldType: string;
  rowCount: number;
  nullCount: number;
  stats: ColumnStats;
  routeReason: string;
};

/**
 * Float32 copy of a 64-bit or Float64 column, expressed relative to the column minimum so a
 * timestamp in milliseconds keeps range/2^24 resolution instead of losing its low digits. Nulls,
 * NaN and infinities are invalid rows (counted as nulls, like the bounded CPU statistics).
 */
export function makeRelativeFloat32Column(
  vector: arrow.Vector,
  rowCount: number
): {values: Float32Array; validity: Uint32Array; base: number; validCount: number} {
  const wide = new Float64Array(rowCount);
  const validity = new Uint32Array(rowCount);
  let base = Infinity;
  let validCount = 0;
  const identity = new Uint32Array(Math.min(rowCount, CONVERSION_CHUNK_ROWS));
  for (let start = 0; start < rowCount; start += CONVERSION_CHUNK_ROWS) {
    const length = Math.min(CONVERSION_CHUNK_ROWS, rowCount - start);
    for (let index = 0; index < length; index++) identity[index] = start + index;
    const values = readOriginalColumnValues(vector, identity.subarray(0, length), false);
    for (let index = 0; index < length; index++) {
      const value = values[index];
      if (typeof value !== 'number' || !Number.isFinite(value)) continue;
      wide[start + index] = value;
      validity[start + index] = 1;
      validCount++;
      if (value < base) base = value;
    }
  }
  if (!validCount) base = 0;
  const values = new Float32Array(rowCount);
  for (let row = 0; row < rowCount; row++) if (validity[row]) values[row] = wide[row] - base;
  return {values, validity, base, validCount};
}

/** Moves the relative-to-minimum profile back to column units. Deviation and histogram are shift-invariant. */
export function shiftGPUColumnStatistics(
  statistics: GPUColumnStatistics,
  base: number
): GPUColumnStatistics {
  return {
    ...statistics,
    minimum: statistics.minimum + base,
    maximum: statistics.maximum + base,
    mean: statistics.mean + base,
    sum: statistics.sum + base * statistics.count
  };
}

/** Native-shaped statistics for the field type; every GPU-derived figure is labelled approximate by the caller. */
export function makeWideColumnStats(
  fieldType: string,
  statistics: GPUColumnStatistics,
  dictionarySize?: number
): ColumnStats {
  const percentNulls =
    statistics.populationRows > 0 ? statistics.nullCount / statistics.populationRows : null;
  if (dictionarySize !== undefined)
    return {
      type: 'categorical' as ColumnStatsType.categorical,
      uniqueValues: Math.min(dictionarySize, Math.round(statistics.distinctEstimate)),
      percentNulls
    };
  const numeric = makeNumericColumnStatsFromGPU(statistics) as Extract<
    ColumnStats,
    {type: ColumnStatsType.numeric}
  >;
  if (fieldType !== 'timestamp' && fieldType !== 'date') return numeric;
  return {
    type: 'time' as ColumnStatsType.time,
    bins: numeric.bins,
    min: Number.isFinite(statistics.minimum) ? statistics.minimum : null,
    max: Number.isFinite(statistics.maximum) ? statistics.maximum : null,
    percentNulls
  };
}

/**
 * Column statistics for the types the Float32 fused profile cannot read directly, beyond the
 * bounded CPU path. Dictionary strings run entirely on the GPU (codes widened to Float32, distinct
 * estimate from the profile). Int64, timestamp and Float64 columns need the retained Arrow
 * values: one CPU conversion pass builds a Float32 copy relative to the column minimum, then the
 * GPU reduces it. GPU-only resident datasets keep no Arrow words and are refused with a reason.
 */
export class ResidentTableWideStatistics {
  private readonly lease: KeplerDatasetLease<KeplerArrowDataset>;
  private tail: Promise<void> = Promise.resolve();
  private requestRevision = 0;
  private destroyed = false;
  private destruction?: Promise<void>;

  constructor(
    private readonly props: {
      session: KeplerComputeSession;
      explorer: KeplerSessionExplorer;
      memoryBudgetBytes: number;
    }
  ) {
    this.lease = props.session.registry.acquireDataset<KeplerArrowDataset>(
      props.explorer.dataset.id,
      {allowIncomplete: props.explorer.allowIncompletePopulation}
    );
  }

  plan(columnName: string): WideStatisticsRoute {
    const value = this.lease.dataset.value;
    const field = value.source.schema.fields.find(candidate => candidate.name === columnName);
    if (!field) return {supported: false, reason: 'Unknown column.'};
    const rowCount = value.source.numRows;
    if (rowCount < 1) return {supported: false, reason: 'The table has no rows.'};
    if (rowCount > MAXIMUM_GPU_COLUMN_STATISTICS_ROWS)
      return {
        supported: false,
        reason: 'GPU statistics support at most 16,777,216 rows; narrow the selection.'
      };
    const orderKind = classifyTableSortType(field.type);
    if (orderKind === 'dictionary-string') {
      return value.columns[columnName] && value.prepared.dictionaries[columnName]
        ? {
            supported: true,
            kind: 'dictionary-string',
            reason:
              'GPU profile of the prepared dictionary codes: null count and a distinct estimate (about 1.6% error).'
          }
        : {supported: false, reason: 'This string field has no prepared dictionary.'};
    }
    if (orderKind === 'wide-int64' || orderKind === 'wide-uint64' || orderKind === 'float64') {
      if (value.resident)
        return {
          supported: false,
          reason:
            'This GPU-only resident dataset does not retain the Arrow values of 64-bit and Float64 columns.'
        };
      return {
        supported: true,
        kind: 'wide-numeric',
        reason:
          'GPU profile of a Float32 copy taken relative to the column minimum (one CPU conversion pass); range, mean and deviation are rounded to Float32 resolution of the range, quantiles come from the histogram.'
      };
    }
    return {supported: false, reason: 'No GPU statistics for this column type.'};
  }

  read(request: {
    column: string;
    population: 'all' | 'selected';
  }): Promise<ResidentWideColumnStatistics | null> {
    const requestRevision = ++this.requestRevision;
    const pending = this.tail.then(() => this.readCurrent(request, requestRevision));
    this.tail = pending.then(
      () => undefined,
      () => undefined
    );
    return pending;
  }

  destroy(): Promise<void> {
    if (this.destruction) return this.destruction;
    this.destroyed = true;
    this.requestRevision++;
    this.destruction = (async () => {
      await this.tail.catch(() => undefined);
      await this.lease.release();
    })();
    return this.destruction;
  }

  private isCurrent(requestRevision: number): boolean {
    return (
      !this.destroyed &&
      requestRevision === this.requestRevision &&
      this.lease.isValid() &&
      this.props.explorer.isCurrent &&
      !this.props.session.deviceLost
    );
  }

  private async readCurrent(
    request: {column: string; population: 'all' | 'selected'},
    requestRevision: number
  ): Promise<ResidentWideColumnStatistics | null> {
    if (!this.isCurrent(requestRevision)) return null;
    const route = this.plan(request.column);
    if (!route.supported) throw new Error(route.reason);
    const selection = await this.props.explorer.readSelectedCount();
    if (!selection || !this.isCurrent(requestRevision)) return null;
    const value = this.lease.dataset.value;
    const rowCount = value.source.numRows;
    const populationRows = request.population === 'selected' ? selection.selectedCount : rowCount;
    const field = value.source.schema.fields.find(candidate => candidate.name === request.column)!;
    const fieldType = getOriginalFieldType(field.type);
    const estimatedBytes =
      getGPUColumnStatisticsByteLength(rowCount, DEFAULT_GPU_COLUMN_STATISTICS_BINS) + rowCount * 8;
    if (estimatedBytes > this.props.memoryBudgetBytes)
      throw new Error('GPU column statistics exceed their working memory budget'); // profiling this many rows needs more GPU scratch memory
    const reservation = this.props.session.registry.reserveMemory(estimatedBytes, 'gpu', 'scratch');
    const owned: {destroy(): void}[] = [];
    try {
      const isCurrent = () => this.isCurrent(requestRevision);
      const common = {
        device: this.props.session.device,
        selection:
          request.population === 'selected' ? this.props.explorer.selectionMask : undefined,
        populationRows,
        isCurrent
      };
      let statistics: GPUColumnStatistics | null;
      let dictionarySize: number | undefined;
      if (route.kind === 'dictionary-string') {
        const widened = await widenResidentChartMeasure({
          session: this.props.session,
          vector: value.columns[request.column] as GPUVector<'uint32'>
        });
        owned.push(widened);
        dictionarySize = value.prepared.dictionaries[request.column]!.values.length;
        statistics = await readGPUColumnStatistics({
          ...common,
          values: widened.vector,
          validity: value.analytics.validity[request.column]
        });
      } else {
        const converted = makeRelativeFloat32Column(
          value.source.getChild(request.column)!,
          rowCount
        );
        if (!isCurrent()) return null;
        const makeVector = (
          name: string,
          data: Float32Array | Uint32Array,
          format: 'float32' | 'uint32'
        ) => {
          const buffer = this.props.session.device.createBuffer({
            id: name,
            byteLength: Math.max(4, data.byteLength),
            usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST,
            data
          });
          const vector = new GPUVector({
            name,
            type: 'data',
            format,
            data: [new GPUData({format, length: rowCount, ownsBuffer: true, buffer})],
            ownsData: true
          });
          owned.push(vector);
          return vector;
        };
        const values = makeVector('wide-statistics-values', converted.values, 'float32');
        const validity = makeVector('wide-statistics-validity', converted.validity, 'uint32');
        const profile = await readGPUColumnStatistics({
          ...common,
          values: values as GPUVector<'float32'>,
          validity: validity as GPUVector<'uint32'>
        });
        statistics = profile && shiftGPUColumnStatistics(profile, converted.base);
      }
      if (!statistics) return null;
      return {
        fieldType,
        rowCount: statistics.populationRows,
        nullCount: statistics.nullCount,
        stats: makeWideColumnStats(fieldType, statistics, dictionarySize),
        routeReason: route.reason
      };
    } finally {
      for (const resource of owned.reverse()) resource.destroy();
      reservation.release();
    }
  }
}
