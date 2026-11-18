// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  getViewBindingRange,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import type {
  GPUCardinalityPort,
  GPUOverflowStatusPort
} from '../../analysis-core/kernels/gpu-contributor-types';
import {
  createGroupStatisticsKernelNode,
  GROUP_DISPATCH_PARAMETER_BYTE_LENGTH
} from './group-statistics-kernel';
import type {Device} from '@luma.gl/core';
import {
  getGraphViewChunks,
  getStorageBindingElementLimit,
  validateCapacity,
  validateCommandNodeBudget,
  validateGPUStorageBindingByteLength,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateGraphViewTopology,
  validateMaximumCount,
  validateRequirement,
  validateSinglePackedView,
  validatePowerOfTwo
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateDisjointWritableViews,
  validateGraphChunkBindingRanges
} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  GPU_CELL_DEFAULT_SUM_SCALE,
  validateSumScale
} from '../../analysis-core/kernels/cell-key-wgsl';
import {readBinding, writeBinding} from './group-statistics-common';
import {getColumnReductionNodes} from './group-statistics-moments';
import {getColumnOrderNodes} from './group-statistics-order';
import {getGroupStructureNodes} from './group-statistics-sort';

import {getTiledGroupStatisticsNodes} from './group-statistics-tiled';
import {
  validateTiledKeyUnionKernelSupport,
  validateTiledKeyUnionRowCount
} from '../../analysis-core/kernels/tiled-key-union';

const OPERATION = 'GPUGroupStatistics';
/** Fixed local sort/scan allowances plus gather passes, moments and pairwise key-union nodes. */
const GROUP_TILE_COMMAND_NODE_ALLOWANCE = 1000;
/** A tile moment tree has at most 29 merge levels, plus gathers and integer reduction passes. */
const GROUP_TILE_COLUMN_COMMAND_NODE_ALLOWANCE = 64;
/** Four scalar moment words per tile row must fit a uint32 storage-array length. */
const GROUP_MAXIMUM_TILE_ROW_COUNT = 0x20000000;

/** Largest number of value columns one contributor instance reduces. */
const MAXIMUM_COLUMN_COUNT = 4;

/** Published GPUSort accepts at most 2^31 rows, including its padding. */
const MAXIMUM_ROW_COUNT = 2 ** 31;

/** Four scalar moment words per group must be addressable by uint32 shader indices. */
const MAXIMUM_GROUP_CAPACITY = Math.floor(0xffffffff / 4);

/** The percentile-count shader literal and uniform row indices are uint32 values. */
const MAXIMUM_PERCENTILE_COUNT = 0xffffffff;

/** The group-major percentile output uses uint32 invocation counts and indices. */
const MAXIMUM_PERCENTILE_TOTAL = 0xffffffff;

type GroupKeyRows = GraphDataView<'uint32' | 'uint32x2'> | GraphVectorView<'uint32' | 'uint32x2'>;
type GroupValueRows = GraphDataView<'float32'> | GraphVectorView<'float32'>;

/**
 * Upper bound per published sort, scan or gather: includes radix histogram scans and their
 * hierarchy at the maximum row count. Their public APIs do not expose node count estimates.
 */
const PRIMITIVE_COMMAND_NODE_ALLOWANCE = 512;

/**
 * Largest group capacity (`output.keys.length`) of {@link GPUGroupStatistics} on `device`: the
 * shifted power sums hold four float32 values (16 bytes) per group in one storage binding, so
 * 8388608 on a default device (128 MiB bindings).
 */
export function getGPUGroupStatisticsMaximumGroupCount(device: Device): number {
  return Math.min(MAXIMUM_GROUP_CAPACITY, getStorageBindingElementLimit(device, 16));
}

/**
 * Largest `capacity * percentiles.length` of {@link GPUGroupStatistics} on `device`: the per-group
 * percentile table holds one float32 per group and fraction in one storage binding, so 33554432 on
 * a default device.
 */
export function getGPUGroupStatisticsMaximumPercentileTotal(device: Device): number {
  return Math.min(MAXIMUM_PERCENTILE_TOTAL, getStorageBindingElementLimit(device, 4));
}

/** Statistics {@link GPUGroupStatistics} can compute per group and value column. */
export type GPUGroupStatistic =
  | 'count'
  | 'sum'
  | 'mean'
  | 'minimum'
  | 'maximum'
  | 'variance'
  | 'standardDeviation'
  | 'skewness'
  | 'kurtosis'
  | 'median'
  | 'percentiles'
  | 'mode'
  | 'uniqueCount'
  | 'zScore';

/**
 * Caller-owned output views of one value column. Provide exactly the views of the statistics that
 * the column requests (`sum` takes `sums`, `sumValues`, or both). Per-group views hold
 * `output.keys.length` rows in ascending key order; rows past `output.status.count` hold zero for
 * `counts`, `sums`, `sumValues` and `uniqueCounts`, and NaN for every other float statistic.
 */
export type GPUGroupStatisticsColumnOutput = {
  /** Finite values per group (`count`). */
  counts?: GraphDataView<'uint32'>;
  /**
   * Fixed-point sums in units of `1 / sumScale`, signed 64-bit `(low, high)` words with the
   * cell-table semantics: the sum of `roundHalfEven(value * sumScale)`. Each value is rounded
   * to `1 / sumScale` first, so these words are exact only for values that are multiples of
   * that unit; a value below half a unit adds nothing. Use `sumValues` for the sum of a column
   * of any magnitude.
   */
  sums?: GraphDataView<'uint32x2'>;
  /**
   * Sum of the finite values as f32 (`sum`): the exact sum in a 96-bit fixed-point accumulator
   * whose unit follows the largest value of the column, rounded once to float32. It does not
   * depend on `sumScale`.
   */
  sumValues?: GraphDataView<'float32'>;
  /**
   * `sumValues` divided by the finite count (`mean`), within a few float32 units in the last
   * place of the true mean at any magnitude; NaN for empty groups.
   */
  means?: GraphDataView<'float32'>;
  /** Minimum finite value (`minimum`); NaN for empty groups. `-0` is reported as `+0`. */
  minimums?: GraphDataView<'float32'>;
  /** Maximum finite value (`maximum`); NaN for empty groups. `-0` is reported as `+0`. */
  maximums?: GraphDataView<'float32'>;
  /** Variance (`variance`), sample or population per {@link GPUGroupStatisticsProps.variance}. */
  variances?: GraphDataView<'float32'>;
  /** Square root of the variance (`standardDeviation`). */
  standardDeviations?: GraphDataView<'float32'>;
  /** Fisher-Pearson skewness `g1` (`skewness`); NaN when the group is constant. */
  skewness?: GraphDataView<'float32'>;
  /** Excess kurtosis `g2` (`kurtosis`); NaN when the group is constant. */
  kurtosis?: GraphDataView<'float32'>;
  /** Median, the linear-interpolated 0.5 quantile (`median`). */
  medians?: GraphDataView<'float32'>;
  /**
   * Quantiles for the `percentiles` fractions, group-major: group `g`, fraction `j` is at row
   * `g * P + j` (`percentiles`). Length is `capacity * P`.
   */
  percentiles?: GraphDataView<'float32'>;
  /** Most frequent finite value; ties resolve to the smallest value (`mode`). */
  modes?: GraphDataView<'float32'>;
  /** Number of distinct finite values (`uniqueCount`). */
  uniqueCounts?: GraphDataView<'uint32'>;
  /**
   * Per-source-row z-score (`zScore`), length equal to the row count: `(v - mean) / sd` with the
   * configured variance kind; 0 when sd is 0 (including one-row sample groups); NaN for masked,
   * invalid-key, non-finite, or capacity-dropped rows.
   */
  zScores?: GraphDataView<'float32'>;
};

/** One value column and the statistics to compute for it. */
export type GPUGroupStatisticsColumn<Values extends GroupValueRows = GraphDataView<'float32'>> = {
  /** Per-row values. Rows with a non-finite value are excluded from this column only. */
  values: Values;
  /** Statistics to compute. This set is topology: changing it needs a new graph. */
  statistics: readonly GPUGroupStatistic[];
  /** Output views, one per requested statistic. */
  output: GPUGroupStatisticsColumnOutput;
};

/**
 * Caller-owned group table of {@link GPUGroupStatistics}. Per-column statistic views live in each
 * column's {@link GPUGroupStatisticsColumnOutput}. Capacity is `keys.length`, at most
 * {@link getGPUGroupStatisticsMaximumGroupCount} (8388608 on a default device).
 */
export type GPUGroupStatisticsOutput<
  KeyFormat extends 'uint32' | 'uint32x2' = 'uint32' | 'uint32x2'
> = {
  /** Distinct keys ascending, same format as the input keys. Empty tail rows hold the all-ones key. */
  keys: GraphDataView<KeyFormat>;
  /** Valid rows per group (mask set and key valid, regardless of value finiteness). */
  counts: GraphDataView<'uint32'>;
  /**
   * One-row status scalars: `count` receives `min(requiredCount, capacity)`, `overflow` receives 1
   * when groups were dropped for lack of capacity, and the optional `requiredCount` receives the
   * unclamped number of groups.
   */
  status: GPUCardinalityPort & GPUOverflowStatusPort;
};

/**
 * Properties for {@link GPUGroupStatistics}.
 *
 * Per-frame (no recompile): the contents of `keys`, `mask`, every `values` column, and the
 * `percentiles` fractions. Topology (needs a new graph): view lengths and chunk boundaries, the
 * capacity, each column's statistic set, `variance`, `sumScale`, and the number of percentiles.
 */
export type GPUGroupStatisticsProps<
  Keys extends GroupKeyRows = GraphDataView<'uint32' | 'uint32x2'>,
  Values extends GroupValueRows = GraphDataView<'float32'>
> = {
  /** Prefix for generated node and transient IDs. Defaults to `'group-statistics'`. */
  id?: string;
  /**
   * Group key per row: `uint32`, or `uint32x2` little-endian `(low, high)` words (Arrow `Uint64`).
   * The all-ones key (`0xffffffff`, or both words) is reserved and means "no key": such rows are
   * skipped.
   */
  keys: Keys;
  /** Optional per-row mask; zero skips the row. Must match the key chunk topology. */
  mask?: Keys extends GraphVectorView ? GraphVectorView<'uint32'> : GraphDataView<'uint32'>;
  /**
   * Dense mode. When set, `keys` must be `uint32` with group keys in `[0, keyCount)` and the
   * output tables are indexed by key: `output.keys.length` and every per-group view must hold
   * exactly `keyCount` rows, row `k` describes key `k`, and keys without rows stay in the table
   * with `counts` 0, NaN float statistics (mean, minimum, maximum, variance, standard deviation,
   * skewness, kurtosis, median, percentiles, mode) and 0 sums and unique counts. `output.keys[k]`
   * is `k`, `output.status.count` and `output.status.requiredCount` are `keyCount` and `output.status.overflow` is 0. Rows whose
   * key is outside the range (or all-ones) are skipped like masked rows. No key sort of the
   * output is needed, so dense choropleth tables line up with feature rows. Topology.
   * Defaults to the compact table of occupied keys.
   */
  keyCount?: number;
  /** Zero to four value columns. */
  columns: readonly GPUGroupStatisticsColumn<Values>[];
  /** `'sample'` (n - 1, default, as d3 and kepler) or `'population'` (n). Stdev and z-scores follow it. */
  variance?: 'sample' | 'population';
  /**
   * Per-frame quantile fractions in [0, 1], `P` entries; a fraction that is NaN or outside
   * `[0, 1]` gives a NaN percentile (it is not clamped). Required when a column requests
   * `percentiles`. Typically a {@link GPUParameterBuffer} view; rewriting it needs no
   * rebuild.
   */
  percentiles?: GraphDataView<'float32'>;
  /**
   * Fixed-point scale of the `sums` words only, a positive float32: one unit is `1 / sumScale`.
   * Defaults to 65536. `sumValues`, `means` and the moments do not use it.
   */
  sumScale?: number;
  /** Explicit reference route: all global order scratch must fit one device binding. */
  scratchMode?: 'auto' | 'packed';
  /** Optional power-of-two tile width. Exact order statistics and z-scores retain the packed route. */
  scratchTileSize?: number;
  /** Caller-owned group table. Capacity is `output.keys.length`. */
  output: GPUGroupStatisticsOutput<Keys['format']>;
};

/** Output field each statistic requires. */
const STATISTIC_OUTPUTS: Record<
  GPUGroupStatistic,
  readonly (keyof GPUGroupStatisticsColumnOutput)[]
> = {
  count: ['counts'],
  sum: ['sums', 'sumValues'],
  mean: ['means'],
  minimum: ['minimums'],
  maximum: ['maximums'],
  variance: ['variances'],
  standardDeviation: ['standardDeviations'],
  skewness: ['skewness'],
  kurtosis: ['kurtosis'],
  median: ['medians'],
  percentiles: ['percentiles'],
  mode: ['modes'],
  uniqueCount: ['uniqueCounts'],
  zScore: ['zScores']
};

const FLOAT_OUTPUTS = [
  'sumValues',
  'means',
  'minimums',
  'maximums',
  'variances',
  'standardDeviations',
  'skewness',
  'kurtosis',
  'medians',
  'percentiles',
  'modes',
  'zScores'
] as const;

/**
 * Groups rows by a 32- or 64-bit key and computes kepler-style statistics per group and column
 * entirely on the GPU, into a capacity-bounded table with ascending keys.
 *
 * Statistics: count, sum, mean, minimum, maximum, variance, standard deviation, skewness,
 * kurtosis, median, percentiles, mode, unique count, and a per-row group z-score.
 *
 * Algorithm: rows are stable-sorted by key (LSD radix passes over the 32-bit words) and group
 * ranges come from head flags and a scan. Counts, fixed-point sums and extremes (order-preserving
 * keys) use integer atomics, so they are independent of thread order. The float sum and the mean
 * use the shared 96-bit fixed-point accumulator (`analysis-core/kernels/fixed-point-accumulator.ts`)
 * with a data-derived unit: a power of two set by the largest finite value of the column among
 * the rows of kept groups and by the bit length of the row count. A value keeps all 24
 * significand bits while its magnitude is at least `max * 2^(h - 71)` (`h` is that bit length;
 * about 1e15 of dynamic range at a million rows), so a column of 1e-4 and a column of 1e6 both
 * sum to float32 accuracy; smaller values round to the unit. The optional `sums` words are a
 * second, 64-bit accumulator in the caller's unit `1 / sumScale`. Moments are corrected two-pass:
 * `d = v - mean` is reduced in a fixed binary tree over sorted row blocks, with O(rows) work and
 * O(log rows) depth even for one dominant group. The central sums about the true mean come from the shifted
 * sums, so the result is bitwise reproducible. The z-score is centred on that same corrected mean.
 * Signed first and third powers use a per-group power-of-two scale before the tree and restore
 * it after cancellation, preventing overflow of same-sign branches. Groups whose values have
 * binary exponent at most 30 need no scaling. Even powers retain float32 range limits.
 *
 * The official `GPUGroupAggregation` of `@luma.gl/graph/gpu-core` publishes count, sum, mean,
 * minimum and maximum for dense `uint32` keys with float compare-exchange sums. This contributor
 * is separate because it needs sparse and 64-bit keys, order statistics and moments, and sums
 * that do not depend on thread order.
 * Order statistics re-sort each requested column by value, then by key, so each group's finite
 * values are ascending; quantiles are direct lookups (`lo + (hi - lo) * frac`, numpy `linear`,
 * the order-statistic policy shared with `GPUColumnQuantiles`: finite values only),
 * modes and unique counts come from run heads, a scan, and integer atomics. Ties resolve to the
 * smallest value. When more groups exist than the capacity, the smallest keys are kept.
 *
 * Keys and values accept packed views or chunked GraphVectorView inputs; every chunk must be
 * packed. Public props default to packed input views, and constructor inference preserves the
 * supplied input and output view types. All outputs stay packed global group tables, including
 * with chunked input; they do not mirror source chunk boundaries. The mask must share the key
 * chunk boundaries; each value column may use different boundaries. Key preparation dispatches
 * per chunk into the shared global key-sort scratch.
 * Counts, sums, means and extremes dispatch per value chunk into shared integer accumulators;
 * the fixed-point scale is selected across every chunk before any sums are added. Moments
 * scatter per-chunk terms into the same sorted positions and fixed tree as packed input, and
 * z-scores write the original logical row order. These paths do not gather value chunks.
 * Median, percentiles, mode and uniqueCount gather chunked values into one packed scratch
 * binding before their global sort. Exact order statistics and z-scores retain that named
 * packed row-scratch bound. Other statistics automatically use fixed-width tile sorts above
 * the binding limit. Global integer accumulators retain the original unit and headroom; tile
 * moment trees share the global shift mean and scale, then merge their centered partials.
 * Moments can differ in their last float32 bits; counts, sums, means and extremes are identical. Chunk dispatch metadata and tree widths use runtime uniforms, so chunk count and
 * tree depth do not multiply the analysis pipelines. Outputs and percentile fractions remain
 * single packed views with disjoint logical ranges; fields bound together must also have
 * disjoint aligned binding ranges. Results are bitwise
 * identical to packed input on the packed scratch route. Not included: HyperLogLog-style approximate distinct counts.
 */
export class GPUGroupStatistics<
  Props extends GPUGroupStatisticsProps<GroupKeyRows, GroupValueRows> = GPUGroupStatisticsProps
> implements GPUCommandNodeProducer
{
  /** Prefix for every node and transient ID. */
  readonly id: string;
  readonly scratchTileSize: number;
  readonly usesTiledScratch: boolean;
  /** Validated properties. */
  readonly props: Props;
  /** Variance kind used by variance, standard deviation and z-scores. */
  readonly variance: 'sample' | 'population';
  /** Fixed-point scale of the `sums` words. */
  readonly sumScale: number;
  /** Upper bound including published primitive allowances and the row-dependent moment tree. */
  readonly estimatedCommandNodeCount: number;

  constructor(props: Props) {
    this.id = props.id ?? 'group-statistics';
    this.props = props;
    const id = this.id;
    this.variance = props.variance ?? 'sample';
    if (this.variance !== 'sample' && this.variance !== 'population') {
      throw new Error(`${id} variance must be 'sample' or 'population'`);
    }
    this.sumScale = props.sumScale ?? GPU_CELL_DEFAULT_SUM_SCALE;
    validateSumScale(id, this.sumScale);

    const {keys, mask, columns, output} = props;
    if (columns.length > MAXIMUM_COLUMN_COUNT) {
      throw new Error(`${id} supports at most ${MAXIMUM_COLUMN_COUNT} columns`);
    }
    for (const chunk of getGraphViewChunks(keys)) {
      validateSinglePackedView(chunk, ['uint32', 'uint32x2'], `${id} keys`);
    }
    validateRequirement(
      id,
      keys.format === 'uint32' || keys.format === 'uint32x2',
      'keys.format must be uint32 or uint32x2'
    );
    const rowCount = keys.length;
    validateCapacity(id, 'keys.length', rowCount, 1, MAXIMUM_ROW_COUNT);
    if (mask) {
      for (const chunk of getGraphViewChunks(mask)) {
        validateSinglePackedView(chunk, ['uint32'], `${id} mask`);
      }
      if (mask.length !== rowCount) {
        throw new Error(`${id} mask must have the same length as keys`);
      }
    }
    validateGraphViewTopology(id, 'mask', keys, mask);
    if (props.keyCount !== undefined) {
      if (!Number.isInteger(props.keyCount) || props.keyCount < 1 || props.keyCount >= 0xffffffff) {
        throw new Error(`${id} keyCount must be an integer in [1, 2^32 - 2]`);
      }
      if (keys.format !== 'uint32') {
        throw new Error(`${id} keyCount needs uint32 keys`);
      }
      if (output.keys.length !== props.keyCount) {
        throw new Error(`${id} output.keys must hold exactly keyCount rows`);
      }
    }
    validateSinglePackedView(
      output.keys,
      [keys.format as 'uint32' | 'uint32x2'],
      `${id} output.keys`
    );
    const capacity = output.keys.length;
    validateCapacity(id, 'output.keys.length', capacity, 1, MAXIMUM_GROUP_CAPACITY);
    validateSinglePackedView(output.counts, ['uint32'], `${id} output.counts`);
    if (output.counts.length !== capacity) {
      throw new Error(`${id} output.counts must have the same length as output.keys`);
    }
    for (const name of ['count', 'overflow', 'requiredCount'] as const) {
      const view = output.status[name];
      if (!view) {
        continue;
      }
      validateSinglePackedView(view, ['uint32'], `${id} output.status.${name}`);
      if (view.length < 1) {
        throw new Error(`${id} output.status.${name} must contain one uint32 row`);
      }
    }

    let percentileColumnCount = 0;
    for (const [columnIndex, column] of columns.entries()) {
      const name = `columns[${columnIndex}]`;
      for (const chunk of getGraphViewChunks(column.values)) {
        validateSinglePackedView(chunk, ['float32'], `${id} ${name}.values`);
      }
      if (column.values.length !== rowCount) {
        throw new Error(`${id} ${name}.values must have the same length as keys`);
      }
      validateRequirement(
        id,
        column.values.format === 'float32',
        `${name}.values.format must be float32`
      );
      const requested = new Set(column.statistics);
      for (const statistic of requested) {
        if (!(statistic in STATISTIC_OUTPUTS)) {
          throw new Error(`${id} ${name} has unknown statistic '${statistic}'`);
        }
      }
      for (const statistic of requested) {
        const fields = STATISTIC_OUTPUTS[statistic];
        if (!fields.some(field => column.output[field])) {
          throw new Error(
            `${id} ${name} statistic '${statistic}' needs output.${fields.join(' or output.')}`
          );
        }
      }
      for (const [field, view] of Object.entries(column.output)) {
        if (!view) {
          continue;
        }
        const owner = (
          Object.entries(STATISTIC_OUTPUTS) as [
            GPUGroupStatistic,
            readonly (keyof GPUGroupStatisticsColumnOutput)[]
          ][]
        ).find(([, fields]) => fields.includes(field as keyof GPUGroupStatisticsColumnOutput));
        if (!owner) {
          throw new Error(`${id} ${name} has unknown output.${field}`);
        }
        if (!requested.has(owner[0])) {
          throw new Error(`${id} ${name} output.${field} requires statistic '${owner[0]}'`);
        }
      }
      const columnOutput = column.output;
      for (const field of ['counts', 'uniqueCounts'] as const) {
        const view = columnOutput[field];
        if (view) {
          validateSinglePackedView(view, ['uint32'], `${id} ${name}.output.${field}`);
        }
      }
      if (columnOutput.sums) {
        validateSinglePackedView(columnOutput.sums, ['uint32x2'], `${id} ${name}.output.sums`);
      }
      for (const field of FLOAT_OUTPUTS) {
        const view = columnOutput[field];
        if (view) {
          validateSinglePackedView(view, ['float32'], `${id} ${name}.output.${field}`);
        }
      }
      const expectedLengths: [keyof GPUGroupStatisticsColumnOutput, number][] = [
        ['zScores', rowCount]
      ];
      for (const field of [
        'counts',
        'sums',
        'sumValues',
        'means',
        'minimums',
        'maximums',
        'variances',
        'standardDeviations',
        'skewness',
        'kurtosis',
        'medians',
        'modes',
        'uniqueCounts'
      ] as const) {
        expectedLengths.push([field, capacity]);
      }
      for (const [field, expectedLength] of expectedLengths) {
        const view = columnOutput[field];
        if (view && view.length !== expectedLength) {
          throw new Error(
            `${id} ${name}.output.${field} must have ${
              field === 'zScores' ? 'the same length as keys' : 'the same length as output.keys'
            }`
          );
        }
      }
      if (requested.has('percentiles')) {
        percentileColumnCount++;
        if (!props.percentiles) {
          throw new Error(`${id} ${name} statistic 'percentiles' needs the percentiles view`);
        }
        const percentileOutput = columnOutput.percentiles;
        if (percentileOutput && percentileOutput.length !== capacity * props.percentiles.length) {
          throw new Error(
            `${id} ${name}.output.percentiles must hold output.keys.length * percentiles.length rows`
          );
        }
      }
    }
    if (props.percentiles) {
      validateSinglePackedView(props.percentiles, ['float32'], `${id} percentiles`);
      validateCapacity(
        id,
        'percentiles.length',
        props.percentiles.length,
        1,
        MAXIMUM_PERCENTILE_COUNT
      );
      validateCapacity(
        id,
        'output.keys.length * percentiles.length',
        capacity * props.percentiles.length,
        1,
        MAXIMUM_PERCENTILE_TOTAL
      );
      if (percentileColumnCount === 0) {
        throw new Error(`${id} percentiles is only used by the 'percentiles' statistic`);
      }
    }
    validateGraphOutputsDisjointFromInputs(id, this.getOutputViews(), this.getInputViews());
    this.validateOutputViews();
    // Structure: at most two sorts, two gathers and one scan, plus 16 local nodes. A column's
    // order statistics add at most three sorts, four gathers and two scans. The other column
    // passes need fewer than 32 nodes, plus the logarithmic moment merge levels. Allow one
    // prepare node per key chunk and five nodes per value chunk: scale, accumulate, moment
    // terms, z-scores and (only for order statistics) a storage-only value gather.
    validateRequirement(
      id,
      props.scratchMode === undefined ||
        props.scratchMode === 'auto' ||
        props.scratchMode === 'packed',
      'scratchMode must be auto or packed'
    );
    const limits = (getGraphViewChunks(keys)[0].buffer.graph as GPUCommandGraph).device.limits;
    const scratchByteLimit = Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize);
    this.scratchTileSize =
      props.scratchTileSize ??
      2 **
        Math.floor(
          Math.log2(Math.max(1, Math.min(GROUP_MAXIMUM_TILE_ROW_COUNT, scratchByteLimit / 16)))
        );
    validateCapacity(id, 'scratchTileSize', this.scratchTileSize, 1, GROUP_MAXIMUM_TILE_ROW_COUNT);
    validatePowerOfTwo(id, 'scratchTileSize', this.scratchTileSize);
    const boundedStatistics = columns.some(column =>
      column.statistics.some(statistic =>
        ['median', 'percentiles', 'mode', 'uniqueCount', 'zScore'].includes(statistic)
      )
    );
    this.usesTiledScratch =
      !boundedStatistics &&
      props.scratchMode !== 'packed' &&
      (props.scratchTileSize !== undefined
        ? rowCount > this.scratchTileSize
        : rowCount * 4 > scratchByteLimit);
    const tileCount = Math.ceil(rowCount / this.scratchTileSize);
    if (this.usesTiledScratch && props.keyCount === undefined) {
      validateTiledKeyUnionRowCount(id, tileCount * this.scratchTileSize);
    }
    this.estimatedCommandNodeCount =
      5 * PRIMITIVE_COMMAND_NODE_ALLOWANCE +
      16 +
      getGraphViewChunks(keys).length +
      columns.reduce((total, column) => {
        const needsOrder = column.statistics.some(statistic =>
          ['median', 'percentiles', 'mode', 'uniqueCount'].includes(statistic)
        );
        const needsMoments = column.statistics.some(statistic =>
          ['variance', 'standardDeviation', 'skewness', 'kurtosis', 'zScore'].includes(statistic)
        );
        return (
          total +
          32 +
          5 * getGraphViewChunks(column.values).length +
          (needsOrder ? 9 * PRIMITIVE_COMMAND_NODE_ALLOWANCE : 0) +
          (needsMoments ? Math.ceil(Math.log2(rowCount)) + 2 : 0)
        );
      }, 0);
    if (this.usesTiledScratch)
      this.estimatedCommandNodeCount =
        (GROUP_TILE_COMMAND_NODE_ALLOWANCE +
          GROUP_TILE_COLUMN_COMMAND_NODE_ALLOWANCE * columns.length) *
          tileCount +
        6 * tileCount * tileCount +
        8 *
          (getGraphViewChunks(keys).length +
            columns.reduce((total, column) => total + getGraphViewChunks(column.values).length, 0));
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
  }

  /** Checks logical separation globally and aligned ranges only for co-bound writable fields. */
  private validateOutputViews(): void {
    const {id, props} = this;
    // Treat output fields as separate logical write chunks for the cross-dispatch check. The
    // borrowed word views are metadata only; no resources or nodes are declared. This also lets
    // the shared validator recognize two imported handles backed by the same physical buffer.
    const chunks = this.getOutputViews().flatMap(view =>
      view
        ? [
            (view.buffer.graph as GPUCommandGraph).createDataView(view.buffer, {
              format: 'uint32',
              length: (view.length * view.rowByteLength) / 4,
              byteOffset: view.byteOffset
            })
          ]
        : []
    );
    const length = chunks.reduce((total, chunk) => total + chunk.length, 0);
    validateDisjointWritableViews(id, {
      'output fields': new GraphVectorView({
        id: `${id}-output-fields`,
        name: 'output fields',
        format: 'uint32',
        length,
        valueLength: length,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data: chunks
      })
    });
    // Keys and group counts share the group-table dispatch; status fields share publish.
    validateDisjointWritableViews(id, {
      'output.keys': props.output.keys,
      'output.counts': props.output.counts
    });
    validateDisjointWritableViews(id, {
      'output.status.count': props.output.status.count,
      'output.status.overflow': props.output.status.overflow,
      'output.status.requiredCount': props.output.status.requiredCount
    });
    for (const [columnIndex, column] of props.columns.entries()) {
      const prefix = `columns[${columnIndex}].output`;
      const output = column.output;
      // These are the caller-owned writable subsets of finish-sums, finish-extremes and
      // finish-moments respectively. Other column dispatches bind at most one output field.
      for (const fields of [
        ['counts', 'sumValues', 'means'],
        ['minimums', 'maximums'],
        ['variances', 'standardDeviations', 'skewness', 'kurtosis']
      ] as const) {
        validateDisjointWritableViews(
          id,
          Object.fromEntries(fields.map(field => [`${prefix}.${field}`, output[field]]))
        );
      }
    }
  }

  private getInputViews(): (GraphDataView | GraphVectorView | undefined)[] {
    const {props} = this;
    return [
      props.keys,
      props.mask,
      props.percentiles,
      ...props.columns.map(column => column.values)
    ];
  }

  private getOutputViews(): (GraphDataView | undefined)[] {
    const {output} = this.props;
    return [
      output.keys,
      output.counts,
      output.status.count,
      output.status.overflow,
      output.status.requiredCount,
      ...this.props.columns.flatMap(column => Object.values(column.output))
    ];
  }

  /** Returns key sort, group structure, per-column reduction, order statistic and publish nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    // Empty input chunks never produce prepare, gather or reduction dispatches. In particular,
    // a trailing empty subview may sit exactly at its buffer's end and has no backing row to
    // bind. Preflight only actual bindings, rather than inventing a one-row empty binding.
    const bindingChunks = [...this.getInputViews(), ...this.getOutputViews()].flatMap(view =>
      view ? getGraphViewChunks(view).filter(chunk => chunk.length > 0) : []
    );
    // Every row-proportional scratch binding is a scalar array, including the four separate
    // moment-power arrays. Check before declaring any command nodes or transients.
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'keys.length row scratch',
      (this.usesTiledScratch ? this.scratchTileSize : props.keys.length) * 4,
      'reduce keys.length (exact order statistics and zScore retain the packed row-scratch limit)'
    );
    // A 64-bit key column may exceed one binding in total, but every physical chunk must fit.
    // Include the binding alignment prefix for nonzero view byte offsets.
    for (const chunk of bindingChunks) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'input/output chunk',
        getViewBindingRange(chunk).size,
        'reduce the chunk length or output capacity'
      );
    }
    validateGraphViewsBelongToGraph(id, graph, [...this.getInputViews(), ...this.getOutputViews()]);
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(bindingChunks.map((chunk, index) => [`bound field ${index}`, chunk]))
    );
    for (const chunk of bindingChunks)
      validateCapacity(
        id,
        'input/output buffer byteLength',
        chunk.buffer.byteLength,
        0,
        graph.device.limits.maxBufferSize
      );
    const capacity = props.output.keys.length;
    validateMaximumCount(
      id,
      'output.keys rows',
      capacity,
      getGPUGroupStatisticsMaximumGroupCount(graph.device)
    );
    if (props.percentiles) {
      validateMaximumCount(
        id,
        'output.keys rows x percentiles',
        capacity * props.percentiles.length,
        getGPUGroupStatisticsMaximumPercentileTotal(graph.device)
      );
    }
    // All transient views start at byte offset zero. Published auto sorts use workgroup-local
    // bitonic storage up to 256 rows; above that, radix histograms and scan summaries are
    // smaller than a row-sized scalar binding. These bounds cover their internal scratch too.
    const scratchLengths: [string, number][] = [
      ['group offsets', capacity + 1],
      ['group counts', capacity]
    ];
    for (const [columnIndex, column] of props.columns.entries()) {
      const needsMoments = column.statistics.some(statistic =>
        ['variance', 'standardDeviation', 'skewness', 'kurtosis', 'zScore'].includes(statistic)
      );
      const needsFloatSums =
        Boolean(column.output.sumValues || column.output.means) || needsMoments;
      if (needsFloatSums)
        scratchLengths.push([`columns[${columnIndex}] fixed-point words`, 3 * capacity + 1]);
      if (needsMoments) {
        scratchLengths.push(
          [`columns[${columnIndex}] mean scratch`, 2 * capacity],
          [`columns[${columnIndex}] moment sums`, 4 * capacity]
        );
      }
    }
    for (const [name, length] of scratchLengths) {
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        length * 4,
        'reduce output capacity'
      );
      validateCapacity(id, `${name} byteLength`, length * 4, 0, graph.device.limits.maxBufferSize);
    }
    validateRequirement(
      id,
      GROUP_DISPATCH_PARAMETER_BYTE_LENGTH <= graph.device.limits.maxUniformBufferBindingSize,
      'dispatch parameters must fit maxUniformBufferBindingSize'
    );
    if (this.usesTiledScratch) {
      for (const [name, byteLength] of [
        ['tile moment scratch', this.scratchTileSize * 16],
        ['tile group offsets', (this.scratchTileSize + 1) * 4],
        ['union keys', capacity * 8]
      ] as const) {
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          name,
          byteLength,
          'reduce scratchTileSize or output capacity'
        );
        validateCapacity(
          id,
          `${name} byteLength`,
          byteLength,
          0,
          graph.device.limits.maxBufferSize
        );
      }
      validateTiledKeyUnionKernelSupport(graph, id);
      return getTiledGroupStatisticsNodes(
        graph,
        props,
        id,
        this.scratchTileSize,
        this.sumScale,
        this.variance
      );
    }
    const {nodes, structure} = getGroupStructureNodes<Parameters>(graph, {
      id,
      operation: OPERATION,
      keys: props.keys,
      mask: props.mask,
      keyCount: props.keyCount,
      output: props.output
    });
    const result: GPUCommandNode<Parameters>[] = [...nodes];
    for (const [columnIndex, column] of props.columns.entries()) {
      const columnId = `${id}-c${columnIndex}`;
      const requested = new Set(column.statistics);
      const columnOutput = column.output;
      // Reinterpret packed float32 storage as words without allocating or copying values.
      // Columns may use chunk boundaries independent of the key column.
      let rowOffset = 0;
      const valueChunks = getGraphViewChunks(column.values).flatMap(chunk => {
        const sourceRowOffset = rowOffset;
        rowOffset += chunk.length;
        return chunk.length === 0
          ? []
          : [
              {
                rowOffset: sourceRowOffset,
                valueBits: graph.createDataView(chunk.buffer, {
                  format: 'uint32',
                  length: chunk.length,
                  byteOffset: chunk.byteOffset
                })
              }
            ];
      });
      const hasMoments = ['variance', 'standardDeviation', 'skewness', 'kurtosis'].some(statistic =>
        requested.has(statistic as GPUGroupStatistic)
      );
      const reduction = getColumnReductionNodes<Parameters>(graph, {
        id: columnId,
        operation: OPERATION,
        structure,
        valueChunks,
        needs: {
          sums: requested.has('sum') || requested.has('mean'),
          minimum: requested.has('minimum'),
          maximum: requested.has('maximum'),
          moments: hasMoments,
          zScore: requested.has('zScore')
        },
        variance: this.variance,
        sumScale: this.sumScale,
        output: columnOutput
      });
      result.push(...reduction.nodes);
      if (
        requested.has('median') ||
        requested.has('percentiles') ||
        requested.has('mode') ||
        requested.has('uniqueCount')
      ) {
        const valueBits =
          valueChunks.length === 1
            ? valueChunks[0].valueBits
            : createTransientView(graph, `${columnId}-value-bits`, 'uint32', structure.rowCount);
        if (valueChunks.length > 1) {
          for (const [chunkIndex, chunk] of valueChunks.entries()) {
            result.push(
              createGroupStatisticsKernelNode<Parameters>(graph, {
                id: `${columnId}-gather-value-bits-${chunkIndex}`,
                operation: OPERATION,
                variant: 'gather-value-bits',
                bindings: [
                  readBinding('source', chunk.valueBits),
                  writeBinding('valueBits', valueBits)
                ],
                invocationCount: chunk.valueBits.length,
                rowOffset: chunk.rowOffset,
                body: `valueBits[valueBitsOffset + index + GROUP_ROW_OFFSET] = source[sourceOffset + index];`
              })
            );
          }
        }
        result.push(
          ...getColumnOrderNodes<Parameters>(graph, {
            id: columnId,
            operation: OPERATION,
            structure,
            valueBits,
            finiteCounts: reduction.finiteCounts,
            percentiles: props.percentiles,
            output: columnOutput
          })
        );
      }
    }
    return result;
  }
}
