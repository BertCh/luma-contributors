// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RequireRasterOutput} from './raster-required-output';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {getFusedStatisticsNodes} from './raster-zonal-fused-statistics';
import {
  getRasterZonalMomentNodes,
  getRasterZonalReductionPlan,
  hasSortedSums
} from './raster-zonal-reductions';
import {getCellWeightWGSL} from './raster-zonal-cell-weight';
import {
  validateRasterProducerCompleteness,
  getRasterProducerCompletenessNodes
} from './raster-producer-completeness';
import type {Device} from '@luma.gl/core';
import {
  createTransientView,
  getViewBindingRange,
  GPUGroupAggregation,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {
  GPURasterBand,
  GPURasterBufferBand,
  GPURasterMetadata
} from '@luma.gl/graph/gpu-raster';
import {
  createFillNode,
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {
  getStorageBindingElementLimit,
  validateCapacity,
  validateMaximumCount,
  validateRequirement,
  getGraphOutputViews,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validateRasterFractionalBindingByteLength,
  validateRasterFractionalWritableViews
} from '../fractional-coverage/raster-fractional-tile-kernel';
import {getRasterZoneDictionaryNodes} from './raster-zone-dictionary';
import {
  getRasterBandViews,
  snapshotRasterBand,
  snapshotRasterGrid,
  getRasterBufferBandNodes,
  validateRasterBandBelongsToGraph,
  validateParameterTarget,
  validateRasterGrid,
  validateRasterParameters
} from '../raster-band-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {GPU_EARTH_AUTHALIC_RADIUS_METERS} from '../../analysis-core/contracts/earth';
import {getSphericalRasterCellArea} from '../../analysis-core/kernels/spherical-cell-area';
import {
  RASTER_GRID_PARAMETER_LENGTH,
  writeRasterGridPrefix
} from '../cost-distance/raster-grid-utils';

/**
 * Cell weighting of {@link GPURasterZonalStatistics}, following the `cellSizeMode` vocabulary of the
 * other raster and terrain contributors.
 * - `'uniform'` (default): every cell has weight one (planar or projected rasters).
 * - `'geographic'`: cells are longitude/latitude degrees and are weighted by their area in square
 *   meters on the WGS84 authalic sphere, `R^2 dLon (sin(top) - sin(bottom))`.
 */
export type GPURasterZonalStatisticsCellSizeMode = 'uniform' | 'geographic';

/** Version 2: legacy prefix, signed row step, origin distance from nearest pole, pole sign. */
export const GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH = 7;

/**
 * Packs the grid words read by {@link GPURasterZonalStatistics} with `cellSizeMode:
 * 'geographic'`: version 2 is `[cellSizeX, cellSizeY, northEdge, southEdge, signedRowStep,
 * originPoleDistance, poleSign]` in degrees (poleSign is +1 or -1), the contributor's default
 * parameterLayoutVersion. The pole-relative latitude preserves thin polar cells when the absolute
 * origin rounds to 90 degrees. A target shorter than seven elements requests only the legacy
 * four-word prefix, which the contributor reads only with an explicit parameterLayoutVersion: 1.
 *
 * @param grid The frame's grid; its width and height must match the contributor's.
 * @param target Optional destination of at least 4 elements; defaults to version 2 (7 words).
 */
export function getGPURasterZonalStatisticsParameterValues(
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH)
): Float32Array {
  validateParameterTarget(
    'getGPURasterZonalStatisticsParameterValues',
    target,
    RASTER_GRID_PARAMETER_LENGTH
  );
  validateRequirement(
    'getGPURasterZonalStatisticsParameterValues',
    grid.affine[1] === 0 && grid.affine[3] === 0,
    'geographic cell areas need an axis-aligned grid'
  );
  writeRasterGridPrefix(target, grid);
  if (target.length >= GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH) {
    target[4] = grid.affine[4];
    const originLatitude =
      grid.affine[5] + (grid.pixelInterpretation === 'point' ? -0.5 * grid.affine[4] : 0);
    target[5] = 90 - Math.abs(originLatitude);
    target[6] = originLatitude < 0 ? -1 : 1;
  }
  return target;
}

/**
 * Largest `zoneCapacity` of {@link GPURasterZonalStatistics} on `device` when no sorted sums run:
 * each output column holds 4 bytes per zone in one storage binding, so 33554432 on a default
 * device (128 MiB bindings).
 */
export function getGPURasterZonalStatisticsMaximumZoneCapacity(device: Device): number {
  return getStorageBindingElementLimit(device, 4);
}

/**
 * Largest `zoneCapacity` of {@link GPURasterZonalStatistics} on `device` when `sumOrder` is
 * `'sorted'` and a sum, mean, variance, standard deviation or weight sum is requested (or
 * `cellSizeMode` is `'geographic'`): the sorted reduction sums each zone with one 256-invocation
 * workgroup, and the total invocation count must fit a uint32 (`floor((2^32 - 1) / 256)`); the
 * `zoneCapacity + 1` uint32 segment offsets must also fit one storage binding. 16777215 on a
 * default device (128 MiB bindings).
 */
export function getGPURasterZonalStatisticsMaximumSortedZoneCapacity(device: Device): number {
  return Math.min(Math.floor(0xffffffff / 256), getStorageBindingElementLimit(device, 4, 1));
}

/** Zone ID that marks a cell as belonging to no zone. Default value of `ignoredZone`. */
export const GPU_RASTER_ZONAL_STATISTICS_NO_ZONE = 0xffffffff;
/** Dictionary-validation violations plus missing source rows must fit one uint32 count. */
const MAXIMUM_ZONE_DICTIONARY_VALIDATION_COUNT = 0xffffffff;

/**
 * Caller-owned per-zone result columns. At least one column is required and every column must hold
 * exactly `zoneCapacity` rows. Zones with no contributing cell yield `0` for counts and sums and
 * NaN for `means`, `minimums`, and `maximums`.
 */
export type GPURasterZonalStatisticsDictionaryOutput = {
  /** With zoneDictionary: original sparse ID for each dense output row. */
  sourceZoneIds?: GraphDataView<'uint32'>;
  /** With zoneDictionary: zero-based dense row, or 0xffffffff for an invalid dictionary. */
  denseZoneRows?: GraphDataView<'uint32'>;
  status?: GPURasterZonalStatisticsStatus & {
    /** Adjacent dictionary ordering violations plus unmapped non-ignored cells. */
    invalidCount?: GraphDataView<'uint32'>;
  };
};

type RasterZonalStatisticsColumns = {
  /** Per zone: cells whose zone ID is in range and not ignored, regardless of value validity. */
  cellCounts?: GraphDataView<'uint32'>;
  /** Per zone: cells with a valid, finite calibrated value. */
  valueCounts?: GraphDataView<'uint32'>;
  /** Per zone: sum of valid calibrated values (0 for empty zones). */
  sums?: GraphDataView<'float32'>;
  /** Per zone: mean of valid calibrated values (NaN for zones without valid values). */
  means?: GraphDataView<'float32'>;
  /** Per zone: minimum valid calibrated value (NaN for zones without valid values). */
  minimums?: GraphDataView<'float32'>;
  /** Per zone: maximum valid calibrated value (NaN for zones without valid values). */
  maximums?: GraphDataView<'float32'>;
  /** Optional one-row status words, rewritten on every encoding. */
  status?: GPURasterZonalStatisticsStatus;
};

/** Legacy six-statistic shape, also used for partial recipe overrides. For the reducer's
 * required terminal selection (including moments), use GPURasterZonalStatisticsProps['output'].
 */
export type GPURasterZonalStatisticsOutput = RasterZonalStatisticsColumns;

/** Complete supported selection of this reducer; the legacy base alias remains six columns so
 * recipes can retain their narrower destination/return contracts and partial overrides.
 */
type RasterZonalStatisticsSelectedOutput = RequireRasterOutput<
  RasterZonalStatisticsColumns & GPURasterZonalStatisticsMomentOutput,
  | 'cellCounts'
  | 'valueCounts'
  | 'sums'
  | 'means'
  | 'minimums'
  | 'maximums'
  | 'weightSums'
  | 'variances'
  | 'populationVariances'
  | 'standardDeviations'
> &
  GPURasterZonalStatisticsDictionaryOutput;

/** One-row status words of {@link GPURasterZonalStatistics}. */
export type GPURasterZonalStatisticsStatus = {
  /** 1 when producerOverflow or any incompleteSources flag is nonzero; results are suppressed. */
  incomplete?: GraphDataView<'uint32'>;
  /**
   * One-row flag: 1 when any cell has a zone ID at or above `zoneCapacity` that is not
   * `ignoredZone`, otherwise 0. Presence is compile-time.
   * With `zoneDictionary`, this checks mapped dense IDs; unmapped source IDs are excluded and
   * reported through `invalidCount` instead.
   */
  overflow?: GraphDataView<'uint32'>;
};

/**
 * Optional weight-sum and dispersion columns of {@link GPURasterZonalStatistics}, with the same
 * length rule as {@link GPURasterZonalStatisticsOutput}. Kept separate so the base type keeps
 * naming exactly the six basic statistics.
 */
export type GPURasterZonalStatisticsMomentOutput = {
  /**
   * Per zone: sum of the cell weights of valid values. Square meters with
   * `cellSizeMode: 'geographic'`, otherwise the valid value count as a float. 0 for empty zones.
   */
  weightSums?: GraphDataView<'float32'>;
  /**
   * Per zone: unbiased sample variance of the valid values. Uniform cells:
   * `sum (v - mean)^2 / (n - 1)`. Geographic cells, whose areas act as reliability weights:
   * `sum w (v - m)^2 / (V1 - V2 / V1)` with `m` the area-weighted mean, `V1 = sum w` and
   * `V2 = sum w^2` (`numpy.cov(v, aweights=w, ddof=1)`), the convention of `GPUZonalStatistics`.
   * The float32 mean is refined with a pass over the deviations around it, then the corrected
   * two-pass sum `sum w (v - m)^2 - (sum w (v - m))^2 / V1` around the refined mean is used, so
   * values far from zero (1e7 + N(0, 1) over 100,000 cells) keep a variance close to the float64
   * one. NaN below two valid values or for a non-positive denominator.
   */
  variances?: GraphDataView<'float32'>;
  /** Per zone: corrected squared deviations divided by sum of weights (population variance).
   * Zero for one positive-weight value; NaN without positive valid support.
   */
  populationVariances?: GraphDataView<'float32'>;
  /**
   * Per zone: `sqrt(variances)`, the sample standard deviation (divisor `n - 1`); NaN where the
   * variance is NaN. `GPURasterCellStatistics` divides by `n`.
   */
  standardDeviations?: GraphDataView<'float32'>;
};

/**
 * How per-zone sums (and means) are accumulated.
 * - `'sorted'` (default): cells are stably sorted by zone and reduced per zone in a fixed tree, so
 *   `sums` and `means` are bitwise reproducible across encodings on one device. Cost is roughly
 *   flat (about 10 to 25 ms at 2048x2048 on an Apple M3 Pro) whatever the zone layout.
 * - `'atomic'`: float compare-exchange atomics. The last bits of `sums` and `means` depend on
 *   accumulation order and can differ between encodings. Only worthwhile for many (1000 or more)
 *   spatially scattered zones, where it is up to about 1.5x faster. It is 10 to 60x slower when
 *   many cells in a wave share a zone, which is the normal case for contiguous administrative
 *   polygons and for few zones, because every lane retries a compare-exchange on the same address.
 */
export type GPURasterZonalStatisticsSumOrder = 'atomic' | 'sorted';

/**
 * Properties for {@link GPURasterZonalStatistics}.
 *
 * Compile-time: `grid` size, `zoneCapacity`, `ignoredZone`, the band format, calibration and
 * storage kind, which output columns and `overflow` exist, and every view length. Per-frame: the
 * contents of `zones` and `values` (and its validity), which callers rewrite between encodings.
 */
export type GPURasterZonalStatisticsProps = {
  /** Prefix for node and transient IDs. Defaults to `'raster-zonal-statistics'`. Compile-time. */
  id?: string;
  /** Raster grid: its width and height size the kernels. */
  grid: GPURasterMetadata;
  /**
   * Packed row-major zone ID per cell, for example rasterized administrative areas. Length is
   * `grid.width * grid.height`. Length is compile-time, contents are per-frame.
   */
  zones: GraphDataView<'uint32'>;
  /** Rasterization overflow. Requires the new output.status.incomplete; failed frames publish NaN floats and 0xffffffff integers. */
  producerOverflow?: GraphDataView<'uint32'>;
  /** Independent one-row nonzero loss flags, ORed with producerOverflow. Requires status.incomplete.
   * For example, polygon overflow plus cost-distance iterationLimitReached. Complete unreachable
   * values do not signal loss. Failed frames suppress all result columns.
   */
  incompleteSources?: readonly GraphDataView<'uint32'>[];
  /**
   * Optional strictly increasing uint32 source IDs, exactly zoneCapacity rows. Sparse source IDs
   * are looked up without float conversion; output row i belongs to zoneDictionary[i]. Requires
   * output.sourceZoneIds, output.denseZoneRows and output.status.invalidCount. Unmapped cells are
   * excluded and counted as invalid. An invalid dictionary excludes every cell. Contents may
   * change between encodings. Omitted preserves the original dense-ID interpretation.
   */
  zoneDictionary?: GraphDataView<'uint32'>;
  /**
   * Value band, buffer or texture backed. Validity, `noDataValue`, `scale`, and `offset` are
   * honored. Band configuration is compile-time, contents are per-frame.
   */
  values: GPURasterBand;
  /**
   * Number of dense zones. Zone IDs in `[0, zoneCapacity)` aggregate. Every output column must
   * have exactly this length. Compile-time. An integer in
   * [1, {@link getGPURasterZonalStatisticsMaximumZoneCapacity}] (33554432 on a default device), or
   * at most {@link getGPURasterZonalStatisticsMaximumSortedZoneCapacity} (16777215) when sorted
   * sums run.
   */
  zoneCapacity: number;
  /**
   * Zone ID silently skipped, for example background `0`. Defaults to
   * {@link GPU_RASTER_ZONAL_STATISTICS_NO_ZONE}. Compile-time.
   */
  ignoredZone?: number;
  /**
   * Cell weighting. Defaults to `'uniform'`. With `'geographic'`, `sums`, `means`, `weightSums` and
   * `variances` weight each cell by its spherical area; counts, minimums and maximums are never
   * weighted. Compile-time.
   */
  cellSizeMode?: GPURasterZonalStatisticsCellSizeMode;
  /**
   * Compile-time layout of `parameters`. Defaults to 2, the layout the default packer writes; the
   * default then requires exactly seven words and rejects any other length. Set 1 explicitly for
   * a legacy four-word view (padding allowed and ignored), or 2 explicitly for a padded view.
   */
  parameterLayoutVersion?: 1 | 2;
  /**
   * Grid words for `cellSizeMode: 'geographic'` (required then, rejected otherwise), see
   * {@link getGPURasterZonalStatisticsParameterValues}:
   * Version 2 (default), seven words: `[cellSizeX, cellSizeY, northEdge, southEdge,
   * signedRowStep, originPoleDistance, poleSign]`. The last two words preserve polar precision.
   * An axis-aligned grid is required.
   * Legacy version 1 (explicit, at least 4 words) retains `[cellSizeX, cellSizeY, northEdge,
   * southEdge]` and derives the step from `(southEdge - northEdge) / height`, which can lose
   * thin-cell area. `cellSizeY` is unused in both versions. Contents are per-frame.
   */
  parameters?: GraphDataView<'float32'>;
  /**
   * Accumulation order for `sums` and `means`. Defaults to `'sorted'`. Compile-time.
   * `'sorted'` makes them bitwise reproducible and avoids same-address atomic contention, at the
   * cost of a radix sort of all cells, a scan, and a gather per encoding. `'atomic'` skips the sort
   * and only wins for many scattered zones. Counts, minimums, and maximums are exact either way.
   */
  sumOrder?: GPURasterZonalStatisticsSumOrder;
  /** Per-zone result columns, rewritten on every encoding. Which columns exist is compile-time. */
  output: RasterZonalStatisticsSelectedOutput;
};

const COLUMN_OPERATIONS = [
  ['cellCounts', 'count'],
  ['valueCounts', 'count'],
  ['sums', 'sum'],
  ['means', 'mean'],
  ['minimums', 'min'],
  ['maximums', 'max']
] as const;

/** Float columns computed by the weighted moment path rather than `GPUGroupAggregation`. */
const MOMENT_COLUMNS = [
  'weightSums',
  'variances',
  'populationVariances',
  'standardDeviations'
] as const;

/**
 * Area in square meters of one cell of row `row` of a geographic raster on the authalic sphere,
 * using the latitude-strip formula R² |Δlongitude (sin(top) - sin(bottom))|. This differs from
 * grid-derived fractional area, which uses WGS84 ellipsoidal meters per degree at the row center.
 * Caller-supplied fractional cellAreas specify full-cell area in caller-declared units independently of either model.
 * This is the CPU twin of the WGSL cell weight of {@link GPURasterZonalStatistics}.
 *
 * @param settings Version 2 (7 words, or at least 7 with an explicit version) or explicit
 *   legacy version 1 (at least 4 words).
 * @param parameterLayoutVersion Layout discriminator; when omitted, version 2 and exactly 7 words,
 *   matching the contributor's default.
 * @internal
 */
export function getRasterZonalCellArea(
  settings: ArrayLike<number>,
  row: number,
  height: number,
  parameterLayoutVersion?: 1 | 2
): number {
  if (
    parameterLayoutVersion === undefined &&
    settings.length !== GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH
  ) {
    throw new Error(
      `getRasterZonalCellArea settings must contain exactly ${GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH} values for the default parameterLayoutVersion 2, got ${settings.length}`
    );
  }
  const minimumLength = (parameterLayoutVersion ?? 2) === 2 ? 7 : 4;
  validateRequirement(
    'getRasterZonalCellArea',
    settings.length >= minimumLength,
    `settings must contain at least ${minimumLength} values for parameterLayoutVersion ${parameterLayoutVersion ?? 2}`
  );
  if ((parameterLayoutVersion ?? 2) === 2) {
    const radians = Math.PI / 180;
    const signedPoleStep = settings[6] * settings[4];
    const rowDisplacement = row * signedPoleStep;
    const originPoleEdgeDistance = settings[5] - rowDisplacement;
    const middlePoleDistance =
      originPoleEdgeDistance > 90
        ? Math.abs(180 + rowDisplacement) - settings[5] + 0.5 * signedPoleStep
        : originPoleEdgeDistance - 0.5 * signedPoleStep;
    return Math.abs(
      2 *
        GPU_EARTH_AUTHALIC_RADIUS_METERS ** 2 *
        settings[0] *
        radians *
        Math.sin(middlePoleDistance * radians) *
        Math.sin(0.5 * settings[4] * radians)
    );
  }
  const rowHeight = (settings[3] - settings[2]) / height;
  return Math.abs(getSphericalRasterCellArea(settings[2], settings[0], rowHeight, row));
}

/**
 * Computes per-zone count, sum, mean, minimum, and maximum of a raster band, where zones are a
 * pre-rasterized grid of dense zone IDs.
 *
 * Cell weights: with the default `cellSizeMode: 'uniform'` every cell weighs one, the right model
 * for planar or projected rasters. Longitude/latitude rasters, whose cells shrink with
 * `cos(latitude)`, set `cellSizeMode: 'geographic'` and `parameters`: `sums` become `sum(a * v)`,
 * `means` `sum(a * v) / sum(a)` and `variances` reliability-weighted, with `a` the cell area in
 * square meters on the WGS84 authalic sphere, R² |Δlongitude (sin(top) - sin(bottom))|.
 * Grid-derived fractional areas instead use WGS84 ellipsoidal meters per degree at the row center;
 * supplied fractional cellAreas retain caller-declared units. Counts, minimums and maximums are never weighted.
 *
 * Variance and standard deviation divisor: `n - 1`, the unbiased sample form (`numpy.var` with
 * `ddof=1`), also for a zone that is a whole population. Two cells with values 1 and 3 give a
 * standard deviation of 1.414. ArcGIS Zonal Statistics, rasterstats `std`, exactextract `stdev`
 * and `GPURasterCellStatistics` use the population form (`n`), which gives 1 for the same two
 * cells. For uniform weights only, multiply a variance by `(n - 1) / n` to convert.
 * For reliability weights use `1 - sum(w²) / sum(w)²`; `weightSums` alone cannot recover this
 * factor. Request `populationVariances` directly for either weighting mode. Published luma.gl raster statistics
 * state no divisor, so each class keeps and documents its own.
 *
 * Variances and standard deviations take two more passes over the cells after the means (one
 * refines the mean, one sums the corrected two-pass terms around it; with `sumOrder: 'sorted'`,
 * each pass also sorts), so they are only computed when requested.
 *
 * The value band is canonicalized (calibration, nodata, validity, non-finite samples). Counts,
 * minimums, and maximums are reduced together (with the optional overflow flag) in one pass over
 * the cells using workgroup-private per-zone tables and one global merge per workgroup, when
 * `zoneCapacity` words per requested column fit in workgroup storage. Otherwise, and for atomic
 * sums and means, a kernel derives zone and value masks and one `GPUGroupAggregation` runs per
 * requested column. Only kernels and masks that a requested column needs are scheduled. Empty zones produce counts of 0, sums of 0, and NaN means, minimums, and
 * maximums. Zones with cells but no valid values count in `cellCounts` only.
 *
 * Determinism: counts, minimums, and maximums are exact. With the default `sumOrder: 'sorted'`
 * the cells are stably sorted by zone ID, scanned into zone offsets, and each zone is reduced with
 * a fixed tree, so `sums` and `means` (sum divided by `valueCounts`) are bitwise reproducible
 * across encodings on one device. This adds a sort over `width * height` cells, a scan, a gather,
 * and one segmented reduction per encoding, but it avoids the same-address contention that makes
 * float atomics 10 to 60x slower on contiguous zones. With `sumOrder: 'atomic'`, sums and means
 * accumulate with float atomics, so their last bits depend on accumulation order and can differ
 * between runs. Results may still differ from a sequential CPU sum in the last bits, and between
 * devices.
 *
 * Non-goals:
 * - No median or percentiles, and no fractional cell coverage of zones.
 * - No dictionary construction or string IDs: sparse uint32 IDs use the caller-supplied sorted
 *   `zoneDictionary`, with explicit source-to-dense output rows.
 * - No polygon rasterization: rasterize zones upstream.
 * - Single tile only: no cross-tile merge, although counts, sums, minimums, and maximums are
 *   mergeable by the caller.
 */
export class GPURasterZonalStatistics implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPURasterZonalStatisticsProps;
  /** Validated sum order. */
  readonly sumOrder: GPURasterZonalStatisticsSumOrder;

  constructor(props: GPURasterZonalStatisticsProps) {
    props = snapshotConstructionFields(props, [
      'id',
      'grid',
      'zones',
      'values',
      'zoneCapacity',
      'ignoredZone',
      'producerOverflow',
      'incompleteSources',
      'zoneDictionary',
      'cellSizeMode',
      'parameterLayoutVersion',
      'parameters',
      'sumOrder',
      'output'
    ]);
    const capturedOutput = snapshotConstructionFields(props.output, [
      'cellCounts',
      'valueCounts',
      'sums',
      'means',
      'minimums',
      'maximums',
      'weightSums',
      'variances',
      'populationVariances',
      'standardDeviations',
      'sourceZoneIds',
      'denseZoneRows',
      'status'
    ]);
    props = {
      ...props,
      grid: snapshotRasterGrid(props.grid),
      values: snapshotRasterBand(props.values),
      incompleteSources: props.incompleteSources ? [...props.incompleteSources] : undefined,
      output: {
        ...capturedOutput,
        ...(capturedOutput.status
          ? {
              status: snapshotConstructionFields(capturedOutput.status, [
                'incomplete',
                'overflow',
                'invalidCount'
              ])
            }
          : {})
      }
    };
    this.id = props.id ?? 'raster-zonal-statistics';
    this.props = props;
    const {id} = this;
    this.sumOrder = props.sumOrder ?? 'sorted';
    if (this.sumOrder !== 'atomic' && this.sumOrder !== 'sorted') {
      throw new Error(`${id} sumOrder must be atomic or sorted`);
    }
    const cellCount = validateRasterGrid(id, props.grid);
    validateSinglePackedView(props.zones, ['uint32'], `${id} zones`);
    if (props.zones.length !== cellCount) {
      throw new Error(`${id} zones must contain one value per cell`);
    }
    if (
      !Number.isSafeInteger(props.zoneCapacity) ||
      props.zoneCapacity < 1 ||
      props.zoneCapacity > 0xffffffff
    ) {
      throw new Error(`${id} zoneCapacity must be a positive integer that fits in uint32`);
    }
    const ignoredZone = props.ignoredZone ?? GPU_RASTER_ZONAL_STATISTICS_NO_ZONE;
    if (!Number.isInteger(ignoredZone) || ignoredZone < 0 || ignoredZone > 0xffffffff) {
      throw new Error(`${id} ignoredZone must be a uint32 integer`);
    }
    const cellSizeMode = props.cellSizeMode ?? 'uniform';
    validateRequirement(
      id,
      props.parameterLayoutVersion === undefined ||
        props.parameterLayoutVersion === 1 ||
        props.parameterLayoutVersion === 2,
      'parameterLayoutVersion must be 1 or 2'
    );
    if (cellSizeMode !== 'uniform' && cellSizeMode !== 'geographic') {
      throw new Error(`${id} cellSizeMode must be uniform or geographic`);
    }
    if (cellSizeMode === 'geographic') {
      validateRequirement(
        id,
        props.grid.affine[1] === 0 && props.grid.affine[3] === 0,
        'geographic cell areas need an axis-aligned grid'
      );
      if (!props.parameters) {
        throw new Error(`${id} geographic cells need parameters`);
      }
      validateRequirement(
        id,
        props.parameterLayoutVersion !== undefined ||
          props.parameters.length === GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH,
        `parameters must contain exactly ${GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH} values ` +
          `(getGPURasterZonalStatisticsParameterValues) for the default parameterLayoutVersion 2, ` +
          `got ${props.parameters.length}; set parameterLayoutVersion: 1 for a legacy ` +
          `four-word view or parameterLayoutVersion: 2 for a padded one`
      );
      validateRasterParameters(
        id,
        props.parameters,
        (props.parameterLayoutVersion ?? 2) === 2
          ? GPU_RASTER_ZONAL_STATISTICS_PARAMETER_LENGTH
          : RASTER_GRID_PARAMETER_LENGTH
      );
    } else if (props.parameters) {
      throw new Error(`${id} parameters need cellSizeMode geographic`);
    }
    const {output} = props;
    validateRequirement(
      id,
      Boolean(props.zoneDictionary) ===
        Boolean(output.sourceZoneIds || output.denseZoneRows || output.status?.invalidCount),
      'zoneDictionary must be present exactly when dictionary outputs are requested'
    );
    if (props.zoneDictionary) {
      validateSinglePackedView(props.zoneDictionary, ['uint32'], `${id} zoneDictionary`);
      validateRequirement(
        id,
        props.zoneDictionary.length === props.zoneCapacity,
        'zoneDictionary must have zoneCapacity entries'
      );
      for (const [name, view] of [
        ['sourceZoneIds', output.sourceZoneIds],
        ['denseZoneRows', output.denseZoneRows]
      ] as const) {
        validateRequirement(id, Boolean(view), `output.${name} is required with zoneDictionary`);
        validateSinglePackedView(view!, ['uint32'], `${id} output.${name}`);
        validateRequirement(
          id,
          view!.length === props.zoneCapacity,
          `output.${name} must have zoneCapacity entries`
        );
      }
      validateRequirement(
        id,
        Boolean(output.status?.invalidCount),
        'output.status.invalidCount is required with zoneDictionary'
      );
      validateSinglePackedView(
        output.status!.invalidCount!,
        ['uint32'],
        `${id} output.status.invalidCount`
      );
      validateRequirement(
        id,
        output.status!.invalidCount!.length === 1,
        'output.status.invalidCount must have one entry'
      );
      validateMaximumCount(
        id,
        'zones.length + zoneCapacity',
        cellCount + props.zoneCapacity,
        MAXIMUM_ZONE_DICTIONARY_VALIDATION_COUNT
      );
    }
    const columns = [
      ...COLUMN_OPERATIONS.filter(([name]) => props.output[name]),
      ...MOMENT_COLUMNS.filter(name => props.output[name])
    ];
    if (columns.length === 0) {
      throw new Error(`${id} requires at least one output`);
    }
    for (const [name, operation] of COLUMN_OPERATIONS) {
      const view = props.output[name];
      if (!view) {
        continue;
      }
      if (operation === 'count') {
        validateSinglePackedView(view, ['uint32'], `${id} output.${name}`);
      } else {
        validateSinglePackedView(view, ['float32'], `${id} output.${name}`);
      }
      if (view.length !== props.zoneCapacity) {
        throw new Error(`${id} output.${name} must contain zoneCapacity rows`);
      }
    }
    for (const name of MOMENT_COLUMNS) {
      const view = props.output[name];
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} output.${name}`);
        if (view.length !== props.zoneCapacity) {
          throw new Error(`${id} output.${name} must contain zoneCapacity rows`);
        }
      }
    }
    validateRasterProducerCompleteness(
      id,
      props.producerOverflow,
      output.status?.incomplete,
      getGraphOutputViews(output),
      props.incompleteSources
    );
    const overflow = output.status?.overflow;
    if (overflow) {
      validateSinglePackedView(overflow, ['uint32'], `${id} output.status.overflow`);
      if (overflow.length !== 1) {
        throw new Error(`${id} output.status.overflow must contain one uint32 row`);
      }
    }
    const sourceGraph = props.zones.buffer.graph as GPUCommandGraph;
    const {weighted, momentPath, sortedSums, fused} = getRasterZonalReductionPlan(
      props,
      this.sumOrder,
      sourceGraph.device.limits.maxComputeWorkgroupStorageSize
    );
    const input = props.values;
    const reusedBandViews = canReuseRasterZonalValues(input, output, cellCount)
      ? getRasterBandViews(input)
      : [];
    const denseZones = props.zoneDictionary ? undefined : props.zones;
    validateRasterFractionalWritableViews(
      id,
      output,
      [
        props.zones,
        props.producerOverflow,
        ...(props.incompleteSources ?? []),
        props.zoneDictionary,
        props.parameters,
        ...getRasterBandViews(input)
      ],
      [
        ...getGraphOutputViews(output).map(view => ({
          writes: [view],
          reads: [props.producerOverflow, ...(props.incompleteSources ?? [])]
        })),
        ...(fused
          ? [
              {
                writes: [
                  output.cellCounts,
                  output.valueCounts,
                  output.minimums,
                  output.maximums,
                  overflow
                ],
                reads: [denseZones, ...reusedBandViews]
              }
            ]
          : []),
        ...COLUMN_OPERATIONS.flatMap(([name]) => {
          if (
            (fused && name !== 'sums' && name !== 'means') ||
            ((sortedSums || momentPath) && (name === 'sums' || name === 'means'))
          )
            return [];
          return [
            {
              writes: [output[name]],
              reads: [denseZones, ...(name === 'cellCounts' ? [] : reusedBandViews)]
            }
          ];
        }),
        ...(sortedSums || momentPath
          ? [
              {writes: [output.sums], reads: sortedSums ? [] : [denseZones]},
              {writes: [output.weightSums], reads: sortedSums ? [] : [denseZones]},
              {
                writes: [output.means],
                reads: [output.sums, output.valueCounts, weighted ? output.weightSums : undefined]
              },
              {
                writes: [output.variances, output.populationVariances, output.standardDeviations],
                reads: [output.valueCounts, weighted ? output.weightSums : undefined]
              }
            ]
          : []),
        {
          writes: [output.sourceZoneIds, output.denseZoneRows, output.status?.invalidCount],
          reads: [props.zoneDictionary]
        },
        {writes: [output.status?.invalidCount], reads: [props.zones, props.zoneDictionary]},
        ...(!fused
          ? [{writes: [overflow], reads: [denseZones, props.parameters, ...reusedBandViews]}]
          : [])
      ]
    );
  }

  /** Returns band canonicalization, overflow reset, mask, and per-column aggregation nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    this.validateBindings(graph);
    if (props.zoneDictionary) {
      const mapping = getRasterZoneDictionaryNodes(graph, {
        id,
        zones: props.zones,
        dictionary: props.zoneDictionary,
        ignoredZone: props.ignoredZone ?? GPU_RASTER_ZONAL_STATISTICS_NO_ZONE,
        sourceZoneIds: props.output.sourceZoneIds!,
        denseZoneRows: props.output.denseZoneRows!,
        invalidCount: props.output.status!.invalidCount!
      });
      const {
        sourceZoneIds: _sourceZoneIds,
        denseZoneRows: _denseZoneRows,
        status,
        ...columns
      } = props.output;
      return [
        ...mapping.nodes,
        ...new GPURasterZonalStatistics({
          ...props,
          id: `${id}-dense`,
          zones: mapping.zones,
          zoneDictionary: undefined,
          producerOverflow: undefined,
          incompleteSources: undefined,
          ignoredZone: props.zoneCapacity,
          output: {...columns, status: status?.overflow ? {overflow: status.overflow} : undefined}
        }).getCommandNodes(graph),
        ...getRasterProducerCompletenessNodes(
          graph,
          id,
          props.producerOverflow,
          props.output.status?.incomplete,
          props.output,
          props.incompleteSources
        )
      ];
    }
    const {zones, zoneCapacity, output} = props;
    const overflow = output.status?.overflow;
    const {width, height} = props.grid;
    const ignoredZone = props.ignoredZone ?? GPU_RASTER_ZONAL_STATISTICS_NO_ZONE;
    validateRasterBandBelongsToGraph(id, graph, props.values, []);
    validateGraphViewsBelongToGraph(id, graph, [
      zones,
      overflow,
      props.parameters,
      ...COLUMN_OPERATIONS.map(([name]) => output[name]),
      ...MOMENT_COLUMNS.map(name => output[name])
    ]);
    // The sorted sums add one 256-invocation workgroup per zone, which lowers the maximum.
    validateCapacity(
      id,
      'zoneCapacity',
      zoneCapacity,
      1,
      hasSortedSums(this.sumOrder, props)
        ? getGPURasterZonalStatisticsMaximumSortedZoneCapacity(graph.device)
        : getGPURasterZonalStatisticsMaximumZoneCapacity(graph.device)
    );
    const cellCount = width * height;
    const {
      weighted,
      needsVariance,
      momentPath,
      needsMeans,
      needsWeightSums,
      sortedSums,
      needsValueCounts,
      fused
    } = getRasterZonalReductionPlan(
      props,
      this.sumOrder,
      graph.device.limits.maxComputeWorkgroupStorageSize
    );
    const needsValueColumn =
      momentPath ||
      Boolean(
        output.valueCounts || output.sums || output.means || output.minimums || output.maximums
      );
    // Counts, minimums and maximums are reduced together in ONE pass with workgroup-private
    // tables (see `getFusedStatisticsNodes`) whenever the tables fit in workgroup memory. Without
    // that pass every column re-reads zones and a mask through its own `GPUGroupAggregation`.
    // Masks only feed the per-column aggregations that the fused pass replaces.
    const needsZoneMask = !fused && Boolean(output.cellCounts);
    const needsValueMask = fused
      ? !sortedSums && (momentPath || Boolean(output.sums || output.means))
      : momentPath || COLUMN_OPERATIONS.some(([name]) => name !== 'cellCounts' && output[name]);
    const nodes: GPUCommandNode<Parameters>[] = [];

    let canonicalValues: GraphDataView<'float32'> | undefined;
    let canonicalValidity: GraphDataView<'uint32'> | undefined;
    if (needsValueColumn) {
      if (canReuseRasterZonalValues(props.values, output, cellCount)) {
        canonicalValues = props.values.storage.values;
        canonicalValidity = props.values.validity;
        if (!canonicalValidity) {
          canonicalValidity = createTransientView(
            graph,
            `${id}-values-validity`,
            'uint32',
            cellCount
          );
          // Every zonal value reader also checks isFiniteValue. An absent caller mask means
          // every row is eligible, not that NaN/Infinity contribute. Use the bounded linear fill
          // instead of a radius-zero 2D neighborhood copy, whose axis dispatch limit is unrelated
          // to the row-major zone reductions (e.g. a 1 x 5,898,240 geographic strip).
          nodes.push(
            createFillNode(graph, {
              id: `${id}-values-validity`,
              operation: 'GPURasterZonalStatistics',
              view: canonicalValidity,
              type: 'u32',
              value: '1u'
            })
          );
        }
      } else {
        const source = getRasterBufferBandNodes(
          graph,
          `${id}-values`,
          props.values,
          props.grid,
          'reuse-canonical'
        );
        nodes.push(...source.nodes);
        canonicalValues = (source.band.storage as {values: GraphDataView<'float32'>}).values;
        canonicalValidity = source.band.validity;
      }
    }

    if (overflow) {
      nodes.push(
        createFillNode<Parameters>(graph, {
          id: `${id}-overflow-reset`,
          operation: 'GPURasterZonalStatistics',
          view: overflow,
          type: 'u32',
          value: '0u'
        })
      );
    }

    const zoneMask = needsZoneMask
      ? createTransientView(graph, `${id}-zone-mask`, 'uint32', cellCount)
      : undefined;
    const valueMask = needsValueMask
      ? createTransientView(graph, `${id}-value-mask`, 'uint32', cellCount)
      : undefined;
    const sortRows = sortedSums
      ? createTransientView(graph, `${id}-sort-rows`, 'uint32', cellCount)
      : undefined;
    const contributions =
      sortedSums || momentPath
        ? createTransientView(graph, `${id}-contributions`, 'float32', cellCount)
        : undefined;
    const weightContributions =
      momentPath && needsWeightSums
        ? createTransientView(graph, `${id}-weight-contributions`, 'float32', cellCount)
        : undefined;
    // The fused pass also raises the overflow flag, so the mask kernel only does when it is absent.
    const maskOverflow = fused ? undefined : overflow;
    // The mask pass binds zones plus every per-cell output it writes. All of them together can
    // reach 10 storage buffers; above the core limit of 8 the zone-only outputs (zone mask, sort
    // rows, overflow flag) get their own pass, which reads only the zones.
    type MaskPart = {
      zoneMask?: GraphDataView<'uint32'>;
      valueMask?: GraphDataView<'uint32'>;
      sortRows?: GraphDataView<'uint32'>;
      contributions?: GraphDataView<'float32'>;
      weightContributions?: GraphDataView<'float32'>;
      maskOverflow?: GraphDataView<'uint32'>;
      /** Whether the pass binds and reads the canonical values and validity. */
      needsValues: boolean;
    };
    const addMaskNode = (nodeId: string, part: MaskPart) => {
      const {needsValues} = part;
      const bindings: WGSLKernelBinding[] = [
        {name: 'zones', view: zones, type: 'u32', access: 'read'}
      ];
      if (needsValues) {
        bindings.push(
          {
            name: 'values',
            view: canonicalValues!,
            type: 'f32',
            access: 'read'
          },
          {
            name: 'validity',
            view: canonicalValidity!,
            type: 'u32',
            access: 'read'
          }
        );
      }
      if (part.valueMask) {
        bindings.push({
          name: 'valueMask',
          view: part.valueMask,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (part.zoneMask) {
        bindings.push({
          name: 'zoneMask',
          view: part.zoneMask,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (part.sortRows) {
        bindings.push({
          name: 'sortRows',
          view: part.sortRows,
          type: 'u32',
          access: 'read_write'
        });
      }
      if (part.contributions) {
        bindings.push({
          name: 'contributions',
          view: part.contributions,
          type: 'f32',
          access: 'read_write'
        });
      }
      if (part.weightContributions) {
        bindings.push({
          name: 'weightContributions',
          view: part.weightContributions,
          type: 'f32',
          access: 'read_write'
        });
      }
      if (weighted && part.contributions) {
        bindings.push({name: 'settings', view: props.parameters!, type: 'f32', access: 'read'});
      }
      if (part.maskOverflow) {
        bindings.push({
          name: 'overflowFlag',
          view: part.maskOverflow,
          type: 'atomic<u32>',
          access: 'read_write'
        });
      }
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: nodeId,
          operation: 'GPURasterZonalStatistics',
          variant: 'mask',
          bindings,
          invocationCount: cellCount,
          declarations: `const CAPACITY: u32 = ${zoneCapacity}u;
const IGNORED: u32 = ${ignoredZone}u;
${IS_FINITE_VALUE_WGSL}
${weighted && part.contributions ? getCellWeightWGSL(width, height, props.parameterLayoutVersion ?? 2) : ''}`,
          body: `let zone = zones[zonesOffset + index];
  let inZone = zone < CAPACITY && zone != IGNORED;
  ${part.zoneMask ? 'zoneMask[zoneMaskOffset + index] = select(0u, 1u, inZone);' : ''}
  ${
    needsValues
      ? `let value = values[valuesOffset + index];
  let isValid = inZone && validity[validityOffset + index] != 0u && isFiniteValue(value);`
      : ''
  }
  ${part.valueMask ? 'valueMask[valueMaskOffset + index] = select(0u, 1u, isValid);' : ''}
  ${part.sortRows ? 'sortRows[sortRowsOffset + index] = select(CAPACITY, zone, inZone);' : ''}
  ${
    part.contributions
      ? weighted
        ? `let weight = getCellWeight(index);
  contributions[contributionsOffset + index] = select(0.0, weight * value, isValid);`
        : 'contributions[contributionsOffset + index] = select(0.0, value, isValid);'
      : ''
  }
  ${
    part.weightContributions
      ? `weightContributions[weightContributionsOffset + index] = select(0.0, ${weighted ? 'weight' : '1.0'}, isValid);`
      : ''
  }
  ${part.maskOverflow ? 'if (zone >= CAPACITY && zone != IGNORED) {\n    atomicStore(&overflowFlag[overflowFlagOffset], 1u);\n  }' : ''}`
        })
      );
    };
    const maskPart: MaskPart = {
      zoneMask,
      valueMask,
      sortRows,
      contributions,
      weightContributions,
      maskOverflow,
      needsValues: Boolean(valueMask || sortRows || contributions)
    };
    const maskBindingCount =
      1 +
      (maskPart.needsValues ? 2 : 0) +
      [zoneMask, valueMask, sortRows, contributions, weightContributions, maskOverflow].filter(
        Boolean
      ).length +
      (weighted && contributions ? 1 : 0);
    if (maskBindingCount > 8) {
      addMaskNode(`${id}-mask-zones`, {zoneMask, sortRows, maskOverflow, needsValues: false});
      addMaskNode(`${id}-mask`, {valueMask, contributions, weightContributions, needsValues: true});
    } else if (zoneMask || valueMask || sortRows || contributions || maskOverflow) {
      addMaskNode(`${id}-mask`, maskPart);
    }

    // Sorted sums find segment sizes by binary search, so cell counts are only built on request;
    // means still need value counts.
    const cellCountsView = output.cellCounts;
    const valueCountsView =
      output.valueCounts ??
      (needsValueCounts
        ? createTransientView(graph, `${id}-value-counts`, 'uint32', zoneCapacity)
        : undefined);
    const countViews: Record<string, GraphDataView<'uint32'> | undefined> = {
      cellCounts: cellCountsView,
      valueCounts: valueCountsView
    };
    if (fused) {
      nodes.push(
        ...getFusedStatisticsNodes<Parameters>(graph, {
          id,
          zones,
          values: canonicalValues,
          validity: canonicalValidity,
          zoneCapacity,
          ignoredZone,
          cellCount,
          cellCounts: cellCountsView,
          valueCounts: valueCountsView,
          minimums: output.minimums,
          maximums: output.maximums,
          overflow
        })
      );
    } else if (sortedSums || momentPath) {
      for (const name of ['cellCounts', 'valueCounts'] as const) {
        if (!output[name] && countViews[name]) {
          nodes.push(
            ...new GPUGroupAggregation({
              id: `${id}-${name}`,
              keys: zones,
              mask: name === 'cellCounts' ? zoneMask! : valueMask!,
              output: countViews[name]!,
              operation: 'count'
            }).getCommandNodes(graph)
          );
        }
      }
    }

    for (const [name, operation] of COLUMN_OPERATIONS) {
      const view = output[name];
      if (!view) {
        continue;
      }
      if ((sortedSums || momentPath) && (name === 'sums' || name === 'means')) {
        continue;
      }
      if (fused && name !== 'sums' && name !== 'means') {
        continue;
      }
      const common = {
        id: `${id}-${name}`,
        keys: zones,
        mask: name === 'cellCounts' ? zoneMask! : valueMask!
      };
      const aggregation =
        operation === 'count'
          ? new GPUGroupAggregation({
              ...common,
              output: view as GraphDataView<'uint32'>,
              operation
            })
          : new GPUGroupAggregation({
              ...common,
              values: canonicalValues!,
              output: view as GraphDataView<'float32'>,
              operation
            });
      nodes.push(...aggregation.getCommandNodes(graph));
    }
    if (sortedSums || momentPath) {
      nodes.push(
        ...getRasterZonalMomentNodes(graph, id, props, {
          weighted,
          needsMeans,
          needsVariance,
          needsWeightSums,
          sortedSums,
          canonicalValues,
          canonicalValidity,
          valueMask,
          sortRows,
          contributions: contributions!,
          weightContributions,
          valueCounts: valueCountsView
        })
      );
    }
    nodes.push(
      ...getRasterProducerCompletenessNodes(
        graph,
        id,
        props.producerOverflow,
        output.status?.incomplete,
        output,
        props.incompleteSources
      )
    );
    return nodes;
  }

  /** Preflights outer and nested dense-analysis storage before any sparse mapping is allocated. */
  private validateBindings<Parameters>(graph: GPUCommandGraph<Parameters>): void {
    const {id, props} = this;
    const sorted = hasSortedSums(this.sumOrder, props);
    validateCapacity(
      id,
      'zoneCapacity',
      props.zoneCapacity,
      1,
      sorted
        ? getGPURasterZonalStatisticsMaximumSortedZoneCapacity(graph.device)
        : getGPURasterZonalStatisticsMaximumZoneCapacity(graph.device)
    );
    const views = [
      props.zones,
      props.zoneDictionary,
      props.producerOverflow,
      ...(props.incompleteSources ?? []),
      props.parameters,
      ...getRasterBandViews(props.values),
      ...getGraphOutputViews(props.output)
    ];
    validateGraphViewsBelongToGraph(id, graph, views);
    validateGraphChunkBindingRanges(
      graph,
      id,
      Object.fromEntries(views.map((view, index) => [`view.${index}`, view]))
    );
    validateRasterBandBelongsToGraph(id, graph, props.values, []);
    for (const view of views)
      if (view)
        validateRasterFractionalBindingByteLength(
          graph.device,
          id,
          view.buffer.id,
          getViewBindingRange(view).size,
          'reduce the packed raster or zone capacity',
          view.buffer.byteLength
        );
    for (const [name, bytes] of [
      ['per-cell planes and sort scratch', props.grid.width * props.grid.height * 4],
      ['zone scratch', props.zoneCapacity * 4],
      ['sorted segment offsets', sorted ? (props.zoneCapacity + 1) * 4 : 0],
      ['dictionary validation', props.zoneDictionary ? 4 : 0]
    ] as const)
      validateRasterFractionalBindingByteLength(
        graph.device,
        id,
        name,
        bytes,
        'reduce grid dimensions or zoneCapacity'
      );
  }
}

/** Reuse values only when conversion is unnecessary and no new input/output binding alias is introduced. */
function canReuseRasterZonalValues(
  band: GPURasterBand,
  output: GPURasterZonalStatisticsProps['output'],
  cellCount: number
): band is GPURasterBufferBand<'float32'> {
  if (
    band.storage.kind !== 'buffer' ||
    band.format !== 'float32' ||
    band.scale !== undefined ||
    band.offset !== undefined ||
    band.noDataValue !== undefined
  )
    return false;
  // Caller-masked canonical bands already used this route. Unmasked bands previously copied;
  // keep that copy for shared output handles so disjoint logical ranges with overlapping aligned
  // bindings remain supported. Separately imported writable aliases are rejected by the existing
  // global raster-view validation, independently of the canonicalization route.
  if (band.validity) return true;
  const values = band.storage.values;
  if (values.format !== 'float32' || values.byteStride !== 4 || values.length < cellCount)
    return false;
  return getGraphOutputViews(output).every(view => view.buffer !== values.buffer);
}
