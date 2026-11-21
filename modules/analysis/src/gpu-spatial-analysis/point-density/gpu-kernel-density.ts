// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUAnalysisRoute} from '../ontology/spatial-analysis-ontology';
import {
  makeDensityExecutionPlan,
  hasDensityShapeRows,
  type GPUDensityDeviceLimits,
  type GPUDensityExecutionPlan
} from './density-execution-plan';
import {getDensityFrameRoute} from './density-auto-route';
import {makeGPURasterBand} from '../../raster-analysis/raster-grid/gpu-raster-grid';
import type {GPURasterBufferBand} from '@luma.gl/graph/gpu-raster';
import {resolveKernelShapeName} from '../shared/kernel-shape-name';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandGraphNodeCondition,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphBufferUse,
  type GraphDataView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {
  getGPUGridIndexMaximumCellCount,
  getGraphViewChunks,
  validateGraphViewTopology,
  validateCapacity,
  validateFiniteNumber,
  validateCommandNodeBudget,
  validateMaximumCount,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext
} from '../../analysis-core/contracts/index';
import {
  resolveGPUStudyWindow,
  type GPUStudyWindow
} from '../../analysis-core/contracts/study-window';
import {getStudyWindowBinding} from '../../analysis-core/kernels/study-window-wgsl';
import {FIXED_POINT_WORD_COUNT} from '../../analysis-core/kernels/fixed-point-accumulator';
import {validateRasterGrid} from '../../raster-analysis/raster-band-utils';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {getRasterGridCorners} from '../spatial-interpolation/spatial-interpolation-parameters';
import {getRectangleScanWGSL} from '../spatial-interpolation/grid-neighbor-scan';
import {createGridCellRowOrderNodes} from '../shared/grid-cell-row-order';
import {createPointNeighborGridIndexNodes} from '../../analysis-core/kernels/point-neighbor-grid-index';
import {GPU_GRID_COORDINATE_WGSL} from '@luma.gl/graph/gpu-core';
import {
  getDensityDomainConstantsWGSL,
  getDensityNeighborBoundsWGSL,
  DENSITY_VALUE_WGSL,
  DENSITY_RECTANGLE_MEMBERSHIP_WGSL
} from './density-neighbor-wgsl';
import {
  GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION,
  type GPUKernelDensityBandwidthRule,
  type GPUKernelDensityKernel,
  type GPUKernelDensityKernelName,
  type GPUKernelDensityMethod,
  type GPUKernelDensityWindow
} from './kernel-density-math';
import {
  createGPUKernelDensityBinnedNodes,
  getGPUKernelDensityBinnedSize,
  getGPUKernelDensityDefaultBinnedPadding,
  GPU_KERNEL_DENSITY_BINNED_PLAN_BYTE_LENGTH,
  GPU_KERNEL_DENSITY_BINNED_MAXIMUM_CELL_COUNT
} from './kernel-density-binned';
import {
  createKernelDensityBandwidthNodes,
  validateKernelDensityBandwidthRule,
  type GPUKernelDensityBandwidthPopulation
} from './kernel-density-bandwidth';
import {
  getKernelDensityResultWGSL,
  getKernelDensityStudyWindowWGSL,
  getKernelValueWGSL,
  getWindowMassWGSL
} from './kernel-density-wgsl';

/** Spatial contexts supported by this area-measuring operation. */
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'area'
} as const);

import {
  createDensityPointPass,
  stageDensityOutputViews,
  validateDensityScratchBindings,
  validateDensityViewBindings,
  validateDensityWritableViews
} from './density-point-passes';

import {
  createDensityScalarScratch,
  validateDensityPointScratch,
  createDensityWeightSumNodes,
  createDensityInvalidCountNodes,
  needsDensitySplitScratch
} from './density-scratch';

/** Five point passes, seven fixed-point totals (two chunk passes each), scale and exclusion status. */
const KERNEL_DENSITY_CHUNK_COMMAND_NODE_ALLOWANCE = 23;
function getKernelDensityChunkCommandAllowance(
  bandwidthRule: GPUKernelDensityBandwidthRule | undefined,
  populationCount: boolean
): number {
  return (
    KERNEL_DENSITY_CHUNK_COMMAND_NODE_ALLOWANCE + (bandwidthRule ? 4 + Number(populationCount) : 0)
  );
}

import {snapshotDensityConstructionProps} from './density-construction-snapshot';

const OPERATION = 'GPUKernelDensity';

/** Runtime row indices and the excluded-row counter are uint32, even with chunked input. */
export const GPU_KERNEL_DENSITY_MAXIMUM_POINT_COUNT = 0xffffffff;

/** Exact gather emits each index dimension as a WGSL uint32 literal. */
const KERNEL_DENSITY_MAXIMUM_INDEX_DIMENSION = 0xffffffff;

/**
 * Binned fixed passes plus an allowance for eight published hierarchical reductions. Each
 * packed u32-indexed reduction needs at most 32 levels; chunk point passes are counted separately.
 */
const KERNEL_DENSITY_BINNED_COMMAND_NODE_ALLOWANCE = 512;

/** Number of float32 elements in a {@link GPUKernelDensity} parameter view. */
export const GPU_KERNEL_DENSITY_PARAMETER_LENGTH = 9;

/** Edge treatment at the boundary of the study window. See {@link GPUKernelDensityProps.edgeCorrection}. */
export type GPUKernelDensityEdgeCorrection = 'none' | 'uniform' | 'diggle';

/** Output scaling. See {@link GPUKernelDensityProps.normalization}. */
export type GPUKernelDensityNormalization = 'intensity' | 'probability';

/** Per-frame settings of {@link GPUKernelDensity}. */
export type GPUKernelDensitySettings = {
  /**
   * Kernel bandwidth `h` in position units (ground units). Positive and finite. Ignored (the GPU
   * computes it) when {@link GPUKernelDensityProps.bandwidthRule} is set; pass any positive value.
   */
  bandwidth: number;
  /**
   * Study window `[minX, minY, maxX, maxY]` of the edge correction. Defaults to the ordered rectangle of the output grid. Ignored
   * without edge correction. Use `'study-window'` with a polygon `studyWindow`; it writes NaN
   * rectangle slots, so a rectangle-mode operator with edge correction yields NaN cells.
   */
  window?: GPUKernelDensityWindow | 'study-window';
};

/**
 * Packs per-frame {@link GPUKernelDensity} parameters.
 *
 * Layout (float32): `[startX, startY, endX, endY, bandwidth, windowMinX, windowMinY, windowMaxX,
 * windowMaxY]`. The first four values are the outer corners of the output grid in grid order:
 * `(startX, startY)` is the outer corner of cell `(0, 0)` and `(endX, endY)` the opposite corner,
 * so a north-up grid (`grid.affine[4] < 0`) has `endY < startY` and row 0 at the top, and a grid
 * whose rows advance north has row 0 at `minY`. The window defaults to the ordered rectangle
 * `[min(startX, endX), min(startY, endY), max(startX, endX), max(startY, endY)]`.
 *
 * @param settings Bandwidth and optional window.
 * @param grid The output grid: axis-aligned, with a nonzero finite cell size; its width and height
 *   must match the contributor's.
 * @param target Optional destination of at least {@link GPU_KERNEL_DENSITY_PARAMETER_LENGTH} elements.
 * @throws If the grid is rotated, sheared or invalid, the target is too short, the bandwidth is
 * not positive and finite, the cell size is zero, or the window is not finite and ordered.
 */
export function getGPUKernelDensityParameterValues(
  settings: GPUKernelDensitySettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_KERNEL_DENSITY_PARAMETER_LENGTH)
): Float32Array {
  const id = 'getGPUKernelDensityParameterValues';
  validateRequirement(
    id,
    target.length >= GPU_KERNEL_DENSITY_PARAMETER_LENGTH,
    `target must hold ${GPU_KERNEL_DENSITY_PARAMETER_LENGTH} elements; received ${target.length}`
  );
  const bandwidth = Math.fround(settings.bandwidth);
  validateRequirement(
    id,
    Number.isFinite(bandwidth) && bandwidth > 0,
    `bandwidth must be positive and finite in float32; received ${settings.bandwidth} encoded as ${bandwidth}`
  );
  const original = getRasterGridCorners(id, grid, false);
  const corners = original.map(Math.fround);
  for (const [axis, startIndex, endIndex, count] of [
    ['x', 0, 2, grid.width],
    ['y', 1, 3, grid.height]
  ] as const) {
    const start = corners[startIndex];
    const end = corners[endIndex];
    const span = Math.fround(end - start);
    const step = Math.fround(span / count);
    validateRequirement(
      id,
      Number.isFinite(start) &&
        Number.isFinite(end) &&
        Number.isFinite(span) &&
        Number.isFinite(step) &&
        step !== 0 &&
        Math.sign(span) === Math.sign(original[endIndex] - original[startIndex]),
      `grid ${axis} corners ${original[startIndex]}, ${original[endIndex]} encoded as ${start}, ${end} must retain an oriented finite nonzero float32 step; subtract a common origin from positions, grid, index bounds and window before float32 conversion`
    );
  }
  const [startX, startY, endX, endY] = corners;
  const window = settings.window ?? [
    Math.min(startX, endX),
    Math.min(startY, endY),
    Math.max(startX, endX),
    Math.max(startY, endY)
  ];
  let encodedWindow: number[] | undefined;
  if (window !== 'study-window') {
    encodedWindow = Array.from(window, Math.fround);
    validateRequirement(
      id,
      encodedWindow.length === 4,
      `window must contain four endpoints; received ${Array.from(window)}`
    );
    for (const [axis, low, high] of [
      ['x', 0, 2],
      ['y', 1, 3]
    ] as const) {
      validateRequirement(
        id,
        Number.isFinite(encodedWindow[low]) &&
          Number.isFinite(encodedWindow[high]) &&
          encodedWindow[low] < encodedWindow[high],
        `window ${axis} endpoints ${window[low]}, ${window[high]} encoded as ${encodedWindow[low]}, ${encodedWindow[high]} must be finite and strictly ordered; subtract a common origin from positions, grid, index bounds and window before float32 conversion`
      );
    }
  }
  // Publish only after validating every encoded word; target tails remain untouched.
  target.set(corners, 0);
  target[4] = bandwidth;
  if (encodedWindow) target.set(encodedWindow, 5);
  else target.fill(Number.NaN, 5, 9);
  return target;
}

/**
 * Caller-owned packed outputs of {@link GPUKernelDensity}. Disjoint aligned fields may share a
 * buffer; those fields use private working buffers and publish their result bits at the end.
 */
export type GPUKernelDensityOutput = {
  /**
   * Density per output cell, at least `grid.width * grid.height` float32 rows, row
   * `y * grid.width + x` in the grid's row order (row 0 is the grid's first row: the top row of a
   * north-up grid, `minY` when rows advance north). Units: `'intensity'` is the sum of weights per unit area, position units to the
   * power -2 (points per square meter for a meter frame); `'probability'` is the same divided by
   * the sum of included weights, yielding a density in position units to the power -2. With uniform edge correction its integral
   * depends on event locations; use diggle correction for mass conservation, allowing for truncation and raster quadrature. NaN where the cell center is outside the edge-correction window, the grid or
   * bandwidth is invalid, or (probability) no weight is included. Rows beyond grid.width * grid.height
   * are unspecified; outputBand exposes only the grid-sized prefix.
   */
  values: GraphDataView<'float32'>;
  /**
   * Optional one-row float32 sum of the included weights (the point count without `weights`),
   * before edge correction: the divisor of `'probability'`.
   */
  weightSum?: GraphDataView<'float32'>;
  /**
   * Optional one-row float32 holding the bandwidth in position units that the density kernels
   * read. Only with {@link GPUKernelDensityProps.bandwidthRule}, which writes it on the GPU each
   * encoding (NaN when the rule is undefined, for example fewer than two included rows, an
   * effective sample size of one or less, or zero spread, in which case every cell is NaN).
   */
  bandwidth?: GraphDataView<'float32'>;
  /** Optional one-row uint32 live row count of bandwidthPopulation, including zero weights. Requires bandwidthRule. */
  bandwidthPopulationCount?: GraphDataView<'uint32'>;
  /** Optional diagnostic scalars. */
  status?: {
    /**
     * Optional one-row uint32 count of input rows that were excluded: a non-finite position or
     * weight, a negative weight, a position outside `indexBounds`, or (with edge correction)
     * outside the window. Zero-weight rows are included.
     */
    invalidCount?: GraphDataView<'uint32'>;
    /**
     * Optional one-row uint32 for `method: 'binned'` and `'auto'`: 1 when `'binned'` ran although
     * `binnedPadding` is smaller than the margin of `2.5 h / cell` cells the blur needs (the
     * estimate then drops near the grid edges), 0 otherwise. Always 0 for `'auto'`, which falls
     * back to the exact gather instead.
     */
    insufficientPadding?: GraphDataView<'uint32'>;
    /** Optional one-row uint32 for binned/auto: 1 when the binned route runs with effective
     * bandwidth below 3 cells on either axis, outside its stated error bound; 0 otherwise.
     * Recomputed on every encoding, including GPU-selected bandwidths.
     */
    bandwidthBelowThreeCells?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUKernelDensity}.
 *
 * Per-frame (no recompile): the contents of `parameters` (bandwidth, window) and of every
 * input buffer. Topology (needs a new graph): the `grid` size, point count, `indexGridSize`,
 * `indexBounds`, `kernel`, `edgeCorrection`, `normalization`, and which optional inputs and outputs
 * are present.
 * Input view kinds are inferred with packed defaults; the output grid remains packed for chunked input.
 */
export type GPUKernelDensityProps<
  Positions extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>,
  Weights extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> = {
  /** Prefix for generated node and transient IDs. Defaults to `'kernel-density'`. */
  id?: string;
  /**
   * Coordinate and metric contract of positions, grid and bandwidth. Planar/native only
   * (omitted means planar/native): the bandwidth is a ground distance, so positions must be in a
   * planar frame with the same unit on both axes. Longitude/latitude throws at construction.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Compile-time study window, default `'rectangle'` (the per-frame parameter rectangle).
   * A prepared polygon must be valid, share this operator's planar frame, and be added to the
   * graph first. Requires `'uniform'` or `'diggle'` edge correction. Window queries use absolute
   * frame coordinates; outside points are excluded and outside cell centers produce NaN.
   */
  studyWindow?: GPUStudyWindow;
  /**
   * Packed point positions in the same units as the grid and bandwidth. `method: 'binned'`
   * also accepts a chunked vector; view lengths and chunk boundaries are compile-time.
   */
  positions: Positions;
  /**
   * Optional per-point weights, for example populations or counts. Non-finite and negative
   * weights exclude the point. Without weights every point has weight one. Chunked weights
   * must have the same chunk boundaries as positions (`method: 'binned'` only).
   */
  weights?: Weights;
  /** Per-frame float32 view of at least 9 values written with {@link getGPUKernelDensityParameterValues}. */
  parameters: GraphDataView<'float32'>;
  /**
   * Output raster grid: axis-aligned, its width and height size the kernels, and its cell centers
   * are evaluated. Pack the matching placement with `getGPUKernelDensityParameterValues`.
   */
  grid: GPURasterMetadata;
  /**
   * Positive uint32 dimensions `[columns, rows]` of the internal `GPUGridIndex` over the points. Aim for a few
   * points per index cell, and index cells not much larger than the bandwidth. At most
   * {@link getGPUGridIndexMaximumCellCount} cells (2^25 - 1 on a default device).
   */
  indexGridSize: readonly [number, number];
  /**
   * Inclusive point domain `[minX, minY, maxX, maxY]` of the internal grid index. It must cover the
   * output grid and every point you want counted: points outside it are ignored (counted in
   * `output.status.invalidCount`), so a point just outside the domain stops contributing to cells within one
   * support radius of it. Bounds are compile-time literals and must be finite in float32.
   */
  indexBounds: readonly [number, number, number, number];
  /**
   * Computes the bandwidth on the GPU every encoding instead of reading `parameters` slot 4, so
   * no read-back is needed. `'scott'` and `'silverman'` are the normal-reference rules of
   * `scipy.stats.gaussian_kde`; in two dimensions both give the factor `n_eff^(-1/6)` and so the
   * same Gaussian-scale value: `h = n_eff^(-1/6) * sqrt((varianceX + varianceY) / 2)`, with the unbiased weighted
   * covariance (`numpy.cov(aweights = w)`) and `n_eff = (sum w)^2 / sum w^2`. Rows with a
   * non-finite position, or a non-finite or negative weight, are skipped (the index domain and
   * the study window are not applied by the default 'all-finite' population). A normal-reference rule oversmooths clustered data. For
   * the compact `'quartic'` and `'epanechnikov'` kernels the Gaussian-scale value is multiplied by
   * their canonical-bandwidth ratio (`getGPUKernelDensityCanonicalBandwidthScale`) so that the
   * support radius smooths as much as the Gaussian rule (ratios 2.40187391 and 2.77919324). The
   * moments use deterministic fixed-point totals in two centered passes, relative to a shared eligible-data minimum on each axis.
   * Topology: changing it needs a new graph.
   */
  bandwidthRule?: GPUKernelDensityBandwidthRule;
  /** Population for automatic bandwidth: all-finite (default) retains legacy behavior; included uses the index domain and active edge window. */
  bandwidthPopulation?: GPUKernelDensityBandwidthPopulation;
  /**
   * Kernel, `K_h(d) = K(d / h) / h^2`, normalised to integrate to one. Defaults to `'gaussian'`.
   * `'bisquare'` is an alias of `'quartic'` (the name neighbor search uses for the same shape).
   */
  kernel?: GPUKernelDensityKernelName;
  /**
   * Treatment of the study window boundary, where an uncorrected estimate loses the kernel mass
   * that falls outside the data region and underestimates the intensity.
   * - `'uniform'` (default): divide each cell by `e(u)`, the kernel mass inside the window
   *   (Diggle 1985; spatstat `density.ppp(edge = TRUE, diggle = FALSE)`). Unbiased in expectation
   *   for a locally uniform intensity, deterministic, and cheap. Its window integral depends on event locations. It is the default because the
   *   uncorrected estimate is systematically low within one bandwidth of the window edge.
   * - `'diggle'`: weight each point by `1 / e(x_i)` (Diggle 1985; Jones 1993, `diggle = TRUE`):
   *   preserves the point total better but is noisier near the edge.
   * - `'none'`: the plain kernel sum; the window is ignored.
   *
   * With `'uniform'` or `'diggle'` the window is the polygon `studyWindow` when supplied,
   * otherwise `parameters.window` (the grid rectangle by default),
   * points outside it are excluded, and cells whose center is outside it are NaN.
   */
  edgeCorrection?: GPUKernelDensityEdgeCorrection;
  /**
   * `'intensity'` (default): `sum_i w_i K_h(u - x_i)`, weights per unit area. `'probability'`:
   * that divided by the sum of included weights.
   */
  normalization?: GPUKernelDensityNormalization;
  /**
   * How the sum is evaluated. Topology: changing it needs a new graph.
   * - `'exact'` (default): every output cell gathers the points inside the kernel support (the
   *   gaussian is truncated at 5 bandwidths). Cost grows with the number of cells inside the
   *   support, `(5 h / cell)^2` per output cell times the local point density.
   * - `'binned'` (gaussian kernel only): each included weight is split over the four nearest cell
   *   centers (linear, or cloud-in-cell, binning, Wand 1994) with the deterministic fixed-point
   *   accumulator, and the bins are blurred along x then y by an almost-gaussian filter: three
   *   running-sum box blurs and one two-sided exponential pass per axis, variance-matched to
   *   `(h / cell)^2` (Wells 1986; Kovesi 2010, with the exponential cancelling the boxes' fourth
   *   cumulant). The cost is a few passes over the grid whatever the bandwidth and does not depend
   *   on the number of points beyond the binning. Result error versus `'exact'`, for
   *   `h >= 3` cells, points inside the binned region and a margin of `2.5 h / cell` cells
   *   (`binnedPadding`), is at most
   *   `getGPUKernelDensityBinnedErrorBound(r) * W / (2 pi h^2)` per cell, where `W` is the sum of
   *   the included weights and `r = max(cellX, cellY) / h`: `r^2 / 4` from the linear binning, 0.016
   *   from the filter shape, and a float32 term, 2e-4 or less up to `h = 200` cells; for example
   *   0.0185 at `h = 10` cells and 0.0161 at `h = 64` cells, that is under 2% of the peak of one
   *   point's kernel. Smoothly varying fields err far less than a lone point. Other kernels throw.
   * - `'auto'`: both paths are built and the GPU chooses every encoding from the bandwidth: exact
   *   while the gaussian support radius `5 h` is at most 32 cells on the larger axis (the
   *   exact gather visits that many cells squared per point), binned above it, but only when
   *   `binnedPadding` gives the blur its margin of `2.5 h / cell` cells (otherwise the exact
   *   gather stays correct, only slower). A non-gaussian kernel resolves to `'exact'`. Both paths'
   *   buffers are allocated.
   */
  method?: GPUKernelDensityMethod;
  /**
   * Cells added on every side of the output grid to the binned array (plus one guard cell), in
   * cells. It sets two things: the blur needs a margin of `2.5 h / cell` cells around the cells you
   * read, because each pass of the cascade drops the mass that leaves the array, so the binned
   * path is exact-to-tolerance only for `h <= binnedPadding * cell / 2.5` (`'binned'` reports a
   * shortfall in `output.status.insufficientPadding`; `'auto'` uses the exact gather instead); and
   * points up to `binnedPadding + 0.5` cells outside the grid are counted (points beyond are not,
   * though they still count in `weightSum` and the exact path), which matters with a `window`
   * larger than the grid or `edgeCorrection: 'none'`. The array holds
   * `(width + 2 binnedPadding + 2) * (height + 2 binnedPadding + 2)` cells of 20 bytes, bound as
   * one storage buffer of 12 bytes per cell. Defaults to the larger grid side (bandwidths up to
   * 40% of it), reduced to fit the buffer. Ignored by `'exact'`.
   */
  binnedPadding?: number;
  /** Caller-owned outputs. */
  output: GPUKernelDensityOutput;
};

/** Numeric shape for route planning; bandwidth and cell size describe one encoded frame. */
export type GPUKernelDensityExecutionShape = {
  pointCount: number;
  gridSize: readonly [number, number];
  indexGridSize: readonly [number, number];
  positionChunkLengths?: readonly number[];
  weightChunkLengths?: readonly number[];
  method?: GPUKernelDensityMethod;
  kernel?: GPUKernelDensityKernelName;
  binnedPadding?: number;
  bandwidthRule?: GPUKernelDensityBandwidthRule;
  bandwidthPopulationCount?: boolean;
  bandwidth?: number;
  cellSize?: readonly [number, number];
};

/**
 * Planar kernel density estimate on a raster, with the bandwidth in ground units.
 *
 * For each output cell center `u` it computes `lambda(u) = sum_i w_i K_h(|u - x_i|)` over the
 * points within the kernel support, where `K_h(d) = K(d / h) / h^2` is a radially symmetric kernel
 * normalised to integrate to one over the plane: `'gaussian'` `exp(-t / 2) / (2 pi h^2)` truncated
 * at 5 bandwidths (lost mass `exp(-12.5)`, 3.7e-6), `'quartic'` (biweight) `3 / pi (1 - t)^2 / h^2`
 * and `'epanechnikov'` `2 / pi (1 - t) / h^2`, with `t = d^2 / h^2 < 1` (Silverman 1986, Density
 * Estimation, Table 3.1). The output is therefore an intensity in weight per unit area (not a count
 * per cell), unchanged by the raster resolution, and `normalization: 'probability'` divides by the
 * included weight to give a probability density. Compare {@link GPUPointDensity}, which bins points
 * and smooths the bin counts with a kernel measured in cells.
 *
 * Edge correction (`edgeCorrection`) divides by, or reweights with, the kernel mass `e(u)` inside
 * the study window. For rectangles the mass is closed form: normal-CDF differences for the
 * gaussian, and an exact inclusion-exclusion of four disk-clipped quadrant integrals for the
 * compact kernels (no quadrature). A prepared polygon `studyWindow` uses an oriented fan
 * decomposition of the kernel integral (2e-5 relative accuracy). It must be valid and in the
 * same planar frame as the KDE; window queries use absolute frame coordinates.
 *
 * Implementation: every encoding rebuilds a `GPUGridIndex` over the points, ranks the IDs inside
 * each index cell ascending, computes per-point weights (exclusion, and `w_i / e(x_i)` for
 * `'diggle'`), and gathers one output cell per invocation by visiting the index rows overlapping the
 * support square in a fixed order (index row, index column, ascending point row). No float atomics,
 * so results are deterministic for a device. Distances are formed relative to the grid start corner to
 * limit f32 cancellation far from the origin, but the input positions themselves are f32.
 *
 * Large bandwidths: the gather costs `(5 h / cell)^2` cells per point, which explodes when the
 * bandwidth spans many cells. `method: 'binned'` (or `'auto'`) replaces it with linear binning and
 * a separable almost-gaussian blur whose cost does not depend on the bandwidth, at the documented
 * error (see {@link GPUKernelDensityProps.method}). The binned error bound requires the frame's
 * effective bandwidth to be at least 3 cells on both axes; optional
 * `output.status.bandwidthBelowThreeCells` reports a binned frame outside this condition. The binned route accepts chunked positions
 * and aligned weights, dispatching each chunk into the same fixed-point grid. Its global scale
 * and total row count make the cell accumulator bit-identical to packed input. The exact and
 * auto routes require packed inputs for indexed random access. Edge correction, normalization and the NaN
 * rules are unchanged. Corrected weights, exclusion flags and bandwidth moment terms follow
 * the input chunks when global scalar scratch cannot fit, with four bytes per point per
 * scratch binding. Packed scratch keeps its established reduction order; split totals use
 * fixed-point accumulation and can differ in the last float32 bits. Totals may exceed one
 * binding; individual input/scratch chunks and the packed grid scratch are checked against
 * the device storage-binding limit before any nodes are created.
 * Total input rows are limited to 4,294,967,295 on every route. Binned and Gaussian auto
 * declarations limit the entire padded grid to 11,184,810 cells (three words per cell in
 * 128 MiB), and the actual scratch must also fit the device's binding and buffer limits.
 * The binned route's estimated command-node budget is 32,768 across all input chunks.
 *
 * Bandwidth: choose it in ground units for the question being asked, or set `bandwidthRule` to
 * have the GPU compute Scott's/Silverman's rule of thumb from the points with no read-back.
 *
 * Output cell `(x, y)` is evaluated at its center `gridStart + (x + 0.5, y + 0.5) * cellSize`,
 * where `gridStart` is the outer corner of cell (0, 0) and `cellSize` is signed, so output rows
 * follow the grid's row order (row 0 at the top of a north-up grid, at `minY` when rows advance
 * north). The estimate is a cell-center sample, not a cell average.
 *
 * Per-frame: input buffer contents and parameter values. Compile-time: input lengths and
 * chunk boundaries, grid dimensions, index bounds and dimensions, kernel, edge correction,
 * normalization, method, padding, bandwidth rule, study window and optional output topology.
 */
export class GPUKernelDensity<
  Positions extends
    | GraphDataView<'float32x2'>
    | GraphVectorView<'float32x2'> = GraphDataView<'float32x2'>,
  Weights extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> implements GPUCommandNodeProducer
{
  /** Source-backed numeric limits; the planner resolves device-dependent byte constraints. */
  static readonly limits: readonly GPUAnalysisRoute[] = Object.freeze([
    {
      name: 'point-count',
      kind: 'hard-limit',
      quantity: 'pointCount',
      unit: 'rows',
      appliesWhen: 'all routes',
      minimum: 1,
      maximum: GPU_KERNEL_DENSITY_MAXIMUM_POINT_COUNT,
      scope: 'domain',
      accuracyImplications: '',
      source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts'
    },
    {
      name: 'auto-support',
      kind: 'auto-crossover',
      quantity: '5 * bandwidth / minimum cell width',
      unit: 'elements',
      appliesWhen: 'Gaussian auto route with sufficient padding',
      maximum: 32,
      scope: 'output',
      alternative: 'binned',
      accuracyImplications:
        'Binned Gaussian approximation; see getGPUKernelDensityBinnedErrorBound.',
      source: 'modules/analysis/src/gpu-spatial-analysis/point-density/kernel-density-binned.ts'
    },
    {
      name: 'binned-cells',
      kind: 'hard-limit',
      quantity: 'padded grid cells',
      unit: 'elements',
      appliesWhen: 'binned or Gaussian auto',
      maximum: GPU_KERNEL_DENSITY_BINNED_MAXIMUM_CELL_COUNT,
      scope: 'binding',
      alternative: 'exact',
      accuracyImplications: 'Exact gathers truncate Gaussian support at five bandwidths.',
      source: 'modules/analysis/src/gpu-spatial-analysis/point-density/kernel-density-binned.ts'
    }
  ]);

  /** Inspect a numeric shape without declaring GPU work. Missing per-frame bandwidth keeps the route 'auto'. */
  static getExecutionPlan(
    shape: GPUKernelDensityExecutionShape,
    deviceLimits: GPUDensityDeviceLimits
  ): GPUDensityExecutionPlan {
    const id = 'GPUKernelDensity.getExecutionPlan';
    for (const [name, value] of [
      ['pointCount', shape.pointCount],
      ['gridSize[0]', shape.gridSize[0]],
      ['gridSize[1]', shape.gridSize[1]],
      ['indexGridSize[0]', shape.indexGridSize[0]],
      ['indexGridSize[1]', shape.indexGridSize[1]]
    ] as const)
      validateCapacity(id, name, value, 1, Number.MAX_SAFE_INTEGER);
    if (shape.binnedPadding !== undefined)
      validateCapacity(id, 'binnedPadding', shape.binnedPadding, 0, Number.MAX_SAFE_INTEGER);
    const kernel = resolveKernelShapeName(
      'GPUKernelDensity.getExecutionPlan',
      shape.kernel ?? 'gaussian',
      ['gaussian', 'quartic', 'epanechnikov']
    );
    const method = shape.method ?? 'exact';
    validateRequirement(
      'GPUKernelDensity.getExecutionPlan',
      ['exact', 'binned', 'auto'].includes(method),
      `method received ${method}; expected exact, binned, or auto`
    );
    const declared = method === 'auto' && kernel !== 'gaussian' ? 'exact' : method;
    const [width, height] = shape.gridSize;
    const padding = shape.binnedPadding ?? getGPUKernelDensityDefaultBinnedPadding(width, height);
    const {paddedWidth, paddedHeight, margin} = getGPUKernelDensityBinnedSize(
      width,
      height,
      padding
    );
    const byteLimit = Math.min(
      deviceLimits.maxBufferSize,
      deviceLimits.maxStorageBufferBindingSize
    );
    let route: string = declared;
    const frame = getDensityFrameRoute(shape.bandwidth ?? 1, shape.cellSize ?? [1, 1], margin);
    if (declared === 'auto' && shape.bandwidth !== undefined && shape.cellSize)
      route = frame.useBinned ? 'binned' : 'exact';
    const chunks = shape.positionChunkLengths ?? [shape.pointCount];
    const weights = shape.weightChunkLengths ?? [];
    const indexCells = shape.indexGridSize[0] * shape.indexGridSize[1];
    const limits: GPUAnalysisRoute[] = [
      this.limits[0],
      {
        ...this.limits[2],
        maximum:
          declared !== 'exact'
            ? GPU_KERNEL_DENSITY_BINNED_MAXIMUM_CELL_COUNT
            : Number.MAX_SAFE_INTEGER
      },
      {
        name: 'grid-scratch',
        kind: 'allocation',
        quantity: 'grid scratch',
        unit: 'bytes',
        appliesWhen: 'all routes',
        maximum: byteLimit,
        byteExpression: 'width * height * 4',
        scope: 'binding',
        accuracyImplications: '',
        source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts'
      },
      {
        name: 'binned-scratch',
        kind: 'allocation',
        quantity: 'fixed-point grid scratch',
        unit: 'bytes',
        appliesWhen: 'binned and auto declarations',
        maximum: byteLimit,
        byteExpression: 'paddedCells * 12',
        scope: 'binding',
        accuracyImplications: '',
        source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts'
      },
      {
        name: 'index-cells',
        kind: 'hard-limit',
        quantity: 'index cells',
        unit: 'elements',
        appliesWhen: 'all routes',
        maximum: getGPUGridIndexMaximumCellCount({limits: deviceLimits} as Parameters<
          typeof getGPUGridIndexMaximumCellCount
        >[0]),
        scope: 'binding',
        accuracyImplications: '',
        source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts'
      },
      {
        name: 'command-budget',
        kind: 'hard-limit',
        quantity: 'estimated command count',
        unit: 'commands',
        appliesWhen: 'binned declaration',
        maximum: 32768,
        scope: 'domain',
        accuracyImplications: '',
        source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts'
      },
      ...chunks.map(
        (_, index): GPUAnalysisRoute => ({
          name: `positions-${index}`,
          kind: 'allocation',
          quantity: 'position chunk',
          unit: 'bytes',
          appliesWhen: 'all routes',
          maximum: byteLimit,
          byteExpression: 'chunkRows * 8',
          scope: 'chunk',
          accuracyImplications: '',
          source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts'
        })
      ),
      ...weights.map(
        (_, index): GPUAnalysisRoute => ({
          name: `weights-${index}`,
          kind: 'allocation',
          quantity: 'weight chunk',
          unit: 'bytes',
          appliesWhen: 'weighted routes',
          maximum: byteLimit,
          byteExpression: 'chunkRows * 4',
          scope: 'chunk',
          accuracyImplications: '',
          source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts'
        })
      ),
      {
        name: 'packed-exact',
        kind: 'hard-limit',
        quantity: 'position chunk count',
        unit: 'elements',
        appliesWhen: 'exact and auto declarations',
        maximum: declared === 'binned' ? Number.MAX_SAFE_INTEGER : 1,
        scope: 'domain',
        accuracyImplications: '',
        source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts'
      },
      {
        name: 'chunk-row-counts',
        kind: 'hard-limit',
        quantity: 'invalid chunk row counts',
        unit: 'elements',
        appliesWhen: 'all routes',
        maximum: 0,
        scope: 'domain',
        accuracyImplications: '',
        source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts'
      },
      {
        name: 'weight-topology',
        kind: 'hard-limit',
        quantity: 'weight topology mismatch',
        unit: 'elements',
        appliesWhen: 'weights supplied',
        maximum: 0,
        scope: 'domain',
        accuracyImplications: '',
        source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts'
      },
      {
        name: 'frame-validity',
        kind: 'hard-limit',
        quantity: 'invalid encoded frame',
        unit: 'elements',
        appliesWhen: 'frame controls supplied',
        maximum: 0,
        scope: 'output',
        accuracyImplications: '',
        source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts'
      },
      {
        name: 'gaussian-binning',
        kind: 'hard-limit',
        quantity: 'unsupported kernel',
        unit: 'elements',
        appliesWhen: 'forced binned',
        maximum: 0,
        scope: 'domain',
        accuracyImplications: '',
        source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-kernel-density.ts'
      }
    ];
    return makeDensityExecutionPlan(
      route,
      limits,
      [
        shape.pointCount,
        paddedWidth * paddedHeight,
        width * height * 4,
        declared === 'exact' ? 0 : paddedWidth * paddedHeight * 12,
        indexCells,
        declared === 'binned'
          ? KERNEL_DENSITY_BINNED_COMMAND_NODE_ALLOWANCE +
            getKernelDensityChunkCommandAllowance(
              shape.bandwidthRule,
              Boolean(shape.bandwidthPopulationCount)
            ) *
              chunks.length
          : 0,
        ...chunks.map(length => length * 8),
        ...weights.map(length => length * 4),
        chunks.length,
        hasDensityShapeRows(shape.pointCount, chunks) &&
        (shape.weightChunkLengths === undefined || hasDensityShapeRows(shape.pointCount, weights))
          ? 0
          : 1,
        shape.weightChunkLengths !== undefined &&
        ((declared !== 'binned' && weights.length !== 1) ||
          weights.length !== chunks.length ||
          weights.some((length, index) => length !== chunks[index]))
          ? 1
          : 0,
        shape.bandwidth !== undefined && shape.cellSize && !frame.valid ? 1 : 0,
        method === 'binned' && kernel !== 'gaussian' ? 1 : 0
      ],
      [
        route === 'binned'
          ? 'Approximate Gaussian convolution; error bound requires bandwidth at least three cells and 2.5 bandwidths of padding.'
          : route === 'auto'
            ? 'Frame-dependent exact or approximate Gaussian route.'
            : 'Exact support gather; Gaussian truncation loses less than 4e-6 mass.'
      ]
    );
  }

  /** Grid-sized view for classification and color scaling; preserves the allocation byte offset. */
  readonly outputBand: GPURasterBufferBand<'float32'>;

  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUKernelDensityProps<Positions, Weights>;
  /** Resolved coordinate and metric contract (planar/native). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved compile-time study window. */
  readonly studyWindow: GPUStudyWindow;
  /** Resolved kernel. */
  /** The canonical kernel name: a `'bisquare'` prop reads back as `'quartic'`. */
  readonly kernel: GPUKernelDensityKernel;
  /** Resolved edge correction. */
  readonly edgeCorrection: GPUKernelDensityEdgeCorrection;
  /** Resolved output scaling. */
  readonly normalization: GPUKernelDensityNormalization;
  /** Resolved method: `'auto'` with a non-gaussian kernel resolves to `'exact'`. */
  readonly method: GPUKernelDensityMethod;
  /** Resolved `binnedPadding`. */
  readonly binnedPadding: number;
  /** Conservative node bound for the chunk-capable binned route. */
  readonly estimatedCommandNodeCount: number | undefined;
  /** `grid.width * grid.height`. */
  readonly cellCount: number;

  constructor(props: GPUKernelDensityProps<Positions, Weights>) {
    props = snapshotDensityConstructionProps(
      props,
      [
        'bandwidthRule',
        'bandwidthPopulation',
        'binnedPadding',
        'edgeCorrection',
        'grid',
        'id',
        'indexBounds',
        'indexGridSize',
        'kernel',
        'method',
        'normalization',
        'output',
        'parameters',
        'positions',
        'spatialContext',
        'studyWindow',
        'weights'
      ],
      ['values', 'weightSum', 'bandwidth', 'bandwidthPopulationCount', 'status'],
      ['invalidCount', 'insufficientPadding', 'bandwidthBelowThreeCells']
    );
    this.id = props.id ?? 'kernel-density';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.studyWindow = resolveGPUStudyWindow(id, props.studyWindow, this.spatialContext);
    this.kernel = resolveKernelShapeName(id, props.kernel ?? 'gaussian', [
      'gaussian',
      'quartic',
      'epanechnikov'
    ]);
    this.edgeCorrection = props.edgeCorrection ?? 'uniform';
    validateRequirement(
      id,
      !(this.studyWindow !== 'rectangle' && this.edgeCorrection === 'none'),
      `studyWindow is ignored without edge correction; set edgeCorrection to 'uniform' or 'diggle'`
    );
    this.normalization = props.normalization ?? 'intensity';
    this.binnedPadding =
      props.binnedPadding ??
      getGPUKernelDensityDefaultBinnedPadding(props.grid.width, props.grid.height);
    const requestedMethod = props.method ?? 'exact';
    validateRequirement(
      id,
      ['gaussian', 'quartic', 'epanechnikov'].includes(this.kernel),
      `kernel must be gaussian, quartic (alias bisquare), or epanechnikov`
    );
    validateRequirement(
      id,
      ['none', 'uniform', 'diggle'].includes(this.edgeCorrection),
      `edgeCorrection must be none, uniform, or diggle`
    );
    validateRequirement(
      id,
      ['intensity', 'probability'].includes(this.normalization),
      `normalization must be intensity or probability`
    );
    validateRequirement(
      id,
      ['exact', 'binned', 'auto'].includes(requestedMethod),
      `method must be exact, binned, or auto`
    );
    validateRequirement(
      id,
      !(requestedMethod === 'binned' && this.kernel !== 'gaussian'),
      `method binned needs the gaussian kernel`
    );
    validateRequirement(
      id,
      !(!Number.isSafeInteger(this.binnedPadding) || this.binnedPadding < 0),
      `binnedPadding must be a nonnegative integer`
    );
    this.method =
      requestedMethod === 'auto' && this.kernel !== 'gaussian' ? 'exact' : requestedMethod;
    validateMaximumCount(
      id,
      'positions.length',
      props.positions.length,
      GPU_KERNEL_DENSITY_MAXIMUM_POINT_COUNT
    );
    this.estimatedCommandNodeCount =
      this.method === 'binned'
        ? KERNEL_DENSITY_BINNED_COMMAND_NODE_ALLOWANCE +
          getKernelDensityChunkCommandAllowance(
            props.bandwidthRule,
            Boolean(props.output.bandwidthPopulationCount)
          ) *
            getGraphViewChunks(props.positions).length
        : undefined;
    if (this.estimatedCommandNodeCount !== undefined) {
      validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    }
    validateKernelDensityBandwidthRule(id, props.bandwidthRule, props.output.bandwidth);
    validateRequirement(
      id,
      ['all-finite', 'included'].includes(props.bandwidthPopulation ?? 'all-finite'),
      `bandwidthPopulation received ${props.bandwidthPopulation}; expected all-finite or included`
    );
    if (props.output.bandwidthPopulationCount) {
      validateRequirement(
        id,
        Boolean(props.bandwidthRule),
        'output.bandwidthPopulationCount needs bandwidthRule'
      );
      validateSinglePackedView(
        props.output.bandwidthPopulationCount,
        ['uint32'],
        `${id} output.bandwidthPopulationCount`
      );
      validateRequirement(
        id,
        props.output.bandwidthPopulationCount.length === 1,
        `output.bandwidthPopulationCount must contain exactly one uint32 row; received ${props.output.bandwidthPopulationCount.length}`
      );
    }
    this.cellCount = validateRasterGrid(id, props.grid);
    validateRequirement(
      id,
      !(props.grid.affine[1] !== 0 || props.grid.affine[3] !== 0),
      `needs an axis-aligned grid`
    );
    if (this.method !== 'exact') {
      const {paddedWidth, paddedHeight} = getGPUKernelDensityBinnedSize(
        props.grid.width,
        props.grid.height,
        this.binnedPadding
      );
      validateRequirement(
        id,
        !(paddedWidth * paddedHeight > GPU_KERNEL_DENSITY_BINNED_MAXIMUM_CELL_COUNT),
        `binned grid cell count ${paddedWidth * paddedHeight} must be <= ${GPU_KERNEL_DENSITY_BINNED_MAXIMUM_CELL_COUNT}`
      );
    }
    for (const [name, size] of [
      ['indexGridSize[0]', props.indexGridSize[0]],
      ['indexGridSize[1]', props.indexGridSize[1]]
    ] as const) {
      validateCapacity(id, name, size, 1, KERNEL_DENSITY_MAXIMUM_INDEX_DIMENSION);
    }
    for (const [index, bound] of props.indexBounds.entries()) {
      validateFiniteNumber(id, `indexBounds[${index}]`, bound);
    }
    const [minX, minY, maxX, maxY] = props.indexBounds;
    validateRequirement(
      id,
      !(
        props.indexBounds.length !== 4 ||
        !props.indexBounds.every(Number.isFinite) ||
        minX > maxX ||
        minY > maxY
      ),
      `indexBounds must be finite and ordered`
    );
    for (const chunk of getGraphViewChunks(props.positions)) {
      validateSinglePackedView(chunk, ['float32x2'], `${id} positions`);
    }
    if (this.method !== 'binned') {
      validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
      if (props.weights) {
        validateSinglePackedView(props.weights, ['float32'], `${id} weights`);
      }
    }
    validateRequirement(id, !(props.positions.length < 1), `needs at least one point`);
    if (props.weights) {
      for (const chunk of getGraphViewChunks(props.weights)) {
        validateSinglePackedView(chunk, ['float32'], `${id} weights`);
      }
      validateRequirement(
        id,
        !(props.weights.length !== props.positions.length),
        `weights length must equal positions length`
      );
      validateGraphViewTopology(id, 'weights', props.positions, props.weights);
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      !(props.parameters.length < GPU_KERNEL_DENSITY_PARAMETER_LENGTH),
      `parameters must hold ${GPU_KERNEL_DENSITY_PARAMETER_LENGTH} float32 values`
    );
    validateSinglePackedView(props.output.values, ['float32'], `${id} output.values`);
    validateRequirement(
      id,
      !(props.output.values.length < this.cellCount),
      `output.values must hold grid.width * grid.height rows`
    );
    this.outputBand = makeGPURasterBand(props.output.values, props.grid, {id: `${id}-output`});
    if (props.output.weightSum) {
      validateSinglePackedView(props.output.weightSum, ['float32'], `${id} output.weightSum`);
      validateRequirement(
        id,
        !(props.output.weightSum.length !== 1),
        `output.weightSum must contain exactly one float32 row`
      );
    }
    if (props.output.status?.invalidCount) {
      validateSinglePackedView(
        props.output.status.invalidCount,
        ['uint32'],
        `${id} output.status.invalidCount`
      );
      validateRequirement(
        id,
        !(props.output.status.invalidCount.length !== 1),
        `output.status.invalidCount must contain exactly one uint32 row`
      );
    }
    if (props.output.status?.insufficientPadding) {
      validateRequirement(
        id,
        !(this.method === 'exact'),
        `output.status.insufficientPadding needs method binned or auto`
      );
      validateSinglePackedView(
        props.output.status.insufficientPadding,
        ['uint32'],
        `${id} output.status.insufficientPadding`
      );
      validateRequirement(
        id,
        !(props.output.status.insufficientPadding.length !== 1),
        `output.status.insufficientPadding must contain exactly one uint32 row`
      );
    }
    if (props.output.status?.bandwidthBelowThreeCells) {
      validateRequirement(
        id,
        !(this.method === 'exact' && props.method !== 'auto'),
        `output.status.bandwidthBelowThreeCells needs method binned or auto`
      );
      validateSinglePackedView(
        props.output.status.bandwidthBelowThreeCells,
        ['uint32'],
        `${id} output.status.bandwidthBelowThreeCells`
      );
      validateRequirement(
        id,
        !(props.output.status.bandwidthBelowThreeCells.length !== 1),
        `output.status.bandwidthBelowThreeCells must contain exactly one uint32 row`
      );
    }
    validateDensityWritableViews(id, {
      'output.values': props.output.values,
      'output.weightSum': props.output.weightSum,
      'output.status.invalidCount': props.output.status?.invalidCount,
      'output.status.insufficientPadding': props.output.status?.insufficientPadding,
      'output.status.bandwidthBelowThreeCells': props.output.status?.bandwidthBelowThreeCells,
      'output.bandwidthPopulationCount': props.output.bandwidthPopulationCount,
      'output.bandwidth': props.output.bandwidth
    });
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.output.values,
        props.output.weightSum,
        props.output.status?.invalidCount,
        props.output.status?.insufficientPadding,
        props.output.status?.bandwidthBelowThreeCells,
        props.output.bandwidth,
        props.output.bandwidthPopulationCount
      ],
      [props.positions, props.weights, props.parameters]
    );
  }

  /**
   * Returns the grid index build and the in-cell sort of the exact gather, with the cell offsets
   * and sorted IDs it reads.
   */
  private getIndexNodes<Parameters>(graph: GPUCommandGraph<Parameters>): {
    nodes: GPUCommandNode<Parameters>[];
    cellOffsets: GraphDataView<'uint32'>;
    sortedIds: GraphDataView<'uint32'>;
  } {
    const {id, props} = this;
    const pointCount = props.positions.length;
    const {indexGridSize, indexBounds} = props;
    const index = createPointNeighborGridIndexNodes<Parameters>(graph, {
      id,
      indexId: `${id}-index`,
      positions: props.positions as GraphDataView<'float32x2'>,
      gridSize: indexGridSize,
      bounds: indexBounds,
      rowCount: pointCount
    });
    const {cellOffsets, sortedRows: objectIds, gridCount: indexCount} = index;
    const nodes: GPUCommandNode<Parameters>[] = [...index.nodes];

    // GPUGridIndex scatters IDs with atomics, so in-cell order varies between runs. A stable sort
    // of the rows by cell restores ascending row order (before any floating-point sum depends on
    // it) in O(rows) work, however many rows share a cell.
    const cellRowOrder = createGridCellRowOrderNodes(graph, {
      id: `${id}-cell-row-order`,
      operation: OPERATION,
      cellOffsets,
      objectIds,
      indexCount,
      rowCount: pointCount
    });
    nodes.push(...cellRowOrder.nodes);
    const sortedIds = cellRowOrder.sortedRows;
    return {nodes, cellOffsets, sortedIds};
  }

  /**
   * Returns the grid index build and the in-cell sort (not for `method: 'binned'`), the per-point
   * weights, the optional weight sum and invalid count reductions, and the per-cell gather
   * (`'exact'`), the binned estimate (`'binned'`), or both behind a GPU gate (`'auto'`), in order.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, kernel, edgeCorrection, normalization, method} = this;
    const {output: callerOutput, grid, indexBounds} = props;
    let output = callerOutput;
    // Preserve the named index capacity refusal before checking its (cellCount + 1) scratch.
    validateMaximumCount(
      id,
      'indexGridSize cells',
      props.indexGridSize[0] * props.indexGridSize[1],
      getGPUGridIndexMaximumCellCount(graph.device)
    );
    validateDensityViewBindings(
      graph,
      id,
      [
        ['positions', props.positions],
        ['weights', props.weights],
        ['parameters', props.parameters],
        ['output.values', output.values],
        ['output.weightSum', output.weightSum],
        ['output.bandwidth', output.bandwidth],
        ['output.bandwidthPopulationCount', output.bandwidthPopulationCount],
        ['output.status.invalidCount', output.status?.invalidCount],
        ['output.status.insufficientPadding', output.status?.insufficientPadding],
        ['output.status.bandwidthBelowThreeCells', output.status?.bandwidthBelowThreeCells]
      ],
      {skipEmptyChunks: true}
    );
    if (method !== 'binned') {
      validateDensityScratchBindings(graph, id, [
        ['index offset scratch', (props.indexGridSize[0] * props.indexGridSize[1] + 1) * 4]
      ]);
    }
    validateDensityPointScratch(graph, id, props.positions, 4);
    validateDensityScratchBindings(graph, id, [['grid scratch', this.cellCount * 4]]);
    validateCapacity(
      id,
      'grid scratch byteLength',
      this.cellCount * 4,
      0,
      graph.device.limits.maxBufferSize
    );
    if (method !== 'exact') {
      validateDensityScratchBindings(graph, id, [
        ['binned blur plan scratch', GPU_KERNEL_DENSITY_BINNED_PLAN_BYTE_LENGTH]
      ]);
      const {paddedWidth, paddedHeight} = getGPUKernelDensityBinnedSize(
        grid.width,
        grid.height,
        this.binnedPadding
      );
      validateDensityScratchBindings(graph, id, [
        ['binned fixed-point grid scratch', paddedWidth * paddedHeight * 4 * FIXED_POINT_WORD_COUNT]
      ]);
      validateCapacity(
        id,
        'binned fixed-point grid scratch byteLength',
        paddedWidth * paddedHeight * 4 * FIXED_POINT_WORD_COUNT,
        0,
        graph.device.limits.maxBufferSize
      );
    }
    // The handle only looks up graph identity; it never invokes parameter-dependent callbacks.
    const studyWindow =
      this.studyWindow === 'rectangle'
        ? undefined
        : this.studyWindow.getViews(graph as unknown as GPUCommandGraph<unknown>);
    if (studyWindow) {
      validateDensityViewBindings(graph, id, [
        ['studyWindow.edges', studyWindow.edges],
        ['studyWindow.summary', studyWindow.summary],
        ['studyWindow.slabs.params', studyWindow.slabs.params],
        ['studyWindow.slabs.slabOffsets', studyWindow.slabs.slabOffsets],
        ['studyWindow.slabs.slabEntries', studyWindow.slabs.slabEntries],
        ['studyWindow.sampleCells', studyWindow.sampleCells],
        ['studyWindow.sampleCellCount', studyWindow.sampleCellCount],
        ['studyWindow.packed', studyWindow.packed]
      ]);
      validateGraphViewsBelongToGraph(id, graph, [
        studyWindow.edges,
        studyWindow.summary,
        studyWindow.slabs.params,
        studyWindow.slabs.slabOffsets,
        studyWindow.slabs.slabEntries,
        studyWindow.sampleCells,
        studyWindow.sampleCellCount,
        studyWindow.packed
      ]);
    }
    const needsExact = method !== 'binned';
    const {width, height} = grid;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.weights,
      props.parameters,
      output.values,
      output.weightSum,
      output.status?.invalidCount,
      output.status?.insufficientPadding,
      output.status?.bandwidthBelowThreeCells,
      output.bandwidth,
      output.bandwidthPopulationCount
    ]);
    const staged = stageDensityOutputViews(
      graph,
      id,
      OPERATION,
      {
        values: output.values,
        weightSum: output.weightSum,
        bandwidth: output.bandwidth,
        bandwidthPopulationCount: output.bandwidthPopulationCount,
        invalidCount: output.status?.invalidCount,
        insufficientPadding: output.status?.insufficientPadding,
        bandwidthBelowThreeCells: output.status?.bandwidthBelowThreeCells
      },
      {values: this.cellCount}
    );
    output = {
      ...callerOutput,
      ...staged.views,
      status: callerOutput.status
        ? {
            invalidCount: staged.views.invalidCount,
            insufficientPadding: staged.views.insufficientPadding,
            bandwidthBelowThreeCells: staged.views.bandwidthBelowThreeCells
          }
        : undefined
    };
    const nodes: GPUCommandNode<Parameters>[] = [];
    if (method === 'exact' && props.method === 'auto' && output.status?.bandwidthBelowThreeCells) {
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-bandwidth-resolution`,
          operation: OPERATION,
          variant: 'bandwidth-resolution-exact',
          invocationCount: 1,
          bindings: [
            {
              name: 'resolution',
              view: output.status.bandwidthBelowThreeCells,
              type: 'u32',
              access: 'read_write'
            }
          ],
          body: 'resolution[resolutionOffset] = 0u;'
        })
      );
    }

    // The binned path never reads the index; only the exact gather does.
    const exactIndex = needsExact ? this.getIndexNodes(graph) : undefined;
    if (exactIndex) {
      nodes.push(...exactIndex.nodes);
    }

    // Bandwidth rule on the GPU: the density kernels read `bandwidthState` instead of parameters[4].
    const bandwidthState = props.bandwidthRule
      ? (output.bandwidth ?? createTransientView(graph, `${id}-bandwidth-state`, 'float32', 1))
      : undefined;
    if (bandwidthState) {
      nodes.push(
        ...createKernelDensityBandwidthNodes(graph, {
          id,
          props,
          kernel,
          bandwidthState,
          populationCount: output.bandwidthPopulationCount,
          studyWindow,
          edgeCorrection
        })
      );
    }
    const bandwidthBindings: WGSLKernelBinding[] = bandwidthState
      ? [{name: 'bandwidthState', view: bandwidthState, type: 'f32', access: 'read'}]
      : [];
    const bandwidthExpression = bandwidthState
      ? 'bandwidthState[bandwidthStateOffset]'
      : 'params[paramsOffset + 4u]';

    // Per-point weights: 0 for excluded rows; w / e(x) under Diggle. The raw (uncorrected) weights
    // feed the weight sum, so they are separate only when Diggle changes them.
    const needsRawWeights =
      edgeCorrection === 'diggle' && (normalization === 'probability' || Boolean(output.weightSum));
    const needsWeightSum = normalization === 'probability' || Boolean(output.weightSum);
    const splitPointScratch = needsDensitySplitScratch(graph, props.positions.length);
    const pointWeights = createDensityScalarScratch(
      graph,
      `${id}-point-weights`,
      props.positions,
      'float32',
      !splitPointScratch
    );
    const rawWeights = needsRawWeights
      ? createDensityScalarScratch(
          graph,
          `${id}-raw-weights`,
          props.positions,
          'float32',
          !splitPointScratch
        )
      : pointWeights;
    const excluded = output.status?.invalidCount
      ? createDensityScalarScratch(
          graph,
          `${id}-excluded`,
          props.positions,
          'uint32',
          !splitPointScratch
        )
      : undefined;
    const pointWeightChunks = getGraphViewChunks(pointWeights);
    const rawWeightChunks = getGraphViewChunks(rawWeights);
    const excludedChunks = excluded ? getGraphViewChunks(excluded) : undefined;
    const domainConstants = getDensityDomainConstantsWGSL(indexBounds);
    const hasEdgeCorrection = edgeCorrection !== 'none';
    const positionChunks = getGraphViewChunks(props.positions);
    const weightChunks = props.weights ? getGraphViewChunks(props.weights) : undefined;
    let rowStart = 0;
    for (const [chunkIndex, positions] of positionChunks.entries()) {
      const start = rowStart;
      rowStart += positions.length;
      if (positions.length === 0) {
        continue;
      }
      const weightBindings: WGSLKernelBinding[] = [
        ...(studyWindow ? [getStudyWindowBinding(studyWindow)] : []),
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        ...(props.weights
          ? [
              {
                name: 'inputWeights',
                view: weightChunks![chunkIndex],
                type: 'f32',
                access: 'read'
              } as const
            ]
          : []),
        {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
        ...bandwidthBindings,
        {
          name: 'pointWeights',
          view: pointWeightChunks[splitPointScratch ? chunkIndex : 0],
          type: 'f32',
          access: 'read_write'
        },
        ...(needsRawWeights
          ? [
              {
                name: 'rawWeights',
                view: rawWeightChunks[splitPointScratch ? chunkIndex : 0],
                type: 'f32',
                access: 'read_write'
              } as const
            ]
          : []),
        ...(excluded
          ? [
              {
                name: 'excluded',
                view: excludedChunks![splitPointScratch ? chunkIndex : 0],
                type: 'u32',
                access: 'read_write'
              } as const
            ]
          : [])
      ];
      nodes.push(
        createDensityPointPass<Parameters>(graph, {
          id: `${id}-point-weights${positionChunks.length === 1 ? '' : `-${chunkIndex}`}`,
          operation: OPERATION,
          variant: `point-weights-${edgeCorrection}`,
          bindings: weightBindings,
          invocationCount: positions.length,
          runtimeConstants: {ROW_START: splitPointScratch ? 0 : start},
          declarations: `${domainConstants}
${DENSITY_VALUE_WGSL}${DENSITY_RECTANGLE_MEMBERSHIP_WGSL}${GPU_GRID_COORDINATE_WGSL}
${studyWindow ? getKernelDensityStudyWindowWGSL(studyWindow, edgeCorrection === 'diggle' ? kernel : undefined) : edgeCorrection === 'diggle' ? getWindowMassWGSL(kernel) : ''}`,
          body: `let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let bandwidth = ${bandwidthExpression};
  let windowMin = vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]) - gridStart;
  let windowMax = vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]) - gridStart;
  let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let weight = ${props.weights ? 'inputWeights[inputWeightsOffset + index]' : '1.0'};
  var included = isFiniteValue(position.x) && isFiniteValue(position.y) &&
    all(position >= DOMAIN_MIN) && all(position <= DOMAIN_MAX) &&
    isFiniteValue(weight) && weight >= 0.0;
  let relative = position - gridStart;
  ${studyWindow ? 'included = included && isInsideStudyWindow(position);' : hasEdgeCorrection ? 'included = included && isInsideDensityRectangle(position, vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]), vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]));' : ''}
  var corrected = select(0.0, weight, included);
  ${rawWeights !== pointWeights ? 'rawWeights[rawWeightsOffset + ROW_START + index] = corrected;' : ''}
  ${
    edgeCorrection === 'diggle'
      ? `if (included) {
    let mass = ${studyWindow ? 'getStudyWindowKernelMass(position, bandwidth)' : 'getWindowMass(relative, windowMin, windowMax, bandwidth)'};
    corrected = select(0.0, weight / mass, mass > 1e-6);
  }`
      : ''
  }
  pointWeights[pointWeightsOffset + ROW_START + index] = corrected;
  ${excluded ? 'excluded[excludedOffset + ROW_START + index] = select(1u, 0u, included);' : ''}`
        })
      );
    }
    const weightSum = needsWeightSum
      ? (output.weightSum ?? createTransientView(graph, `${id}-weight-sum-scratch`, 'float32', 1))
      : undefined;
    if (weightSum) {
      nodes.push(...createDensityWeightSumNodes(graph, `${id}-weight-sum`, rawWeights, weightSum));
    }
    if (excluded && output.status?.invalidCount) {
      nodes.push(
        ...createDensityInvalidCountNodes(
          graph,
          `${id}-invalid-count`,
          excluded,
          output.status.invalidCount
        )
      );
    }

    // Binned path: the plan (and under 'auto' the gate) precede the gather, the blur nodes follow.
    const binned =
      method === 'exact'
        ? undefined
        : createGPUKernelDensityBinnedNodes(graph, {
            id,
            positions: props.positions,
            parameters: props.parameters,
            pointWeights,
            studyWindow,
            bandwidthBindings,
            bandwidthExpression,
            weightSum,
            values: output.values,
            width,
            height,
            padding: this.binnedPadding,
            insufficientPadding: output.status?.insufficientPadding,
            bandwidthBelowThreeCells: output.status?.bandwidthBelowThreeCells,
            edgeCorrection,
            normalization,
            autoGate: method === 'auto'
          });
    if (binned) {
      nodes.push(...binned.planNodes);
    }
    if (exactIndex) {
      // Separate polygon finish keeps the gather within eight storage bindings even with a
      // GPU bandwidth rule and probability normalization.
      const gathered = studyWindow
        ? createTransientView(graph, `${id}-gathered`, 'float32', this.cellCount)
        : undefined;
      nodes.push(
        this.getGatherNode(graph, {
          exactIndex,
          gathered,
          values: output.values,
          pointWeights: pointWeightChunks[0],
          bandwidthBindings,
          bandwidthExpression,
          weightSum,
          gate: binned?.exactGate
        })
      );
      if (studyWindow && gathered) {
        nodes.push(
          createWGSLKernelNode<Parameters>(graph, {
            id: `${id}-polygon-finish`,
            operation: OPERATION,
            variant: `polygon-finish-${kernel}-${edgeCorrection}-${normalization}`,
            bindings: [
              {name: 'gathered', view: gathered, type: 'f32', access: 'read'},
              {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
              ...bandwidthBindings,
              getStudyWindowBinding(studyWindow),
              ...(normalization === 'probability' && weightSum
                ? [{name: 'weightSum', view: weightSum, type: 'f32', access: 'read'} as const]
                : []),
              {name: 'valuesOut', view: output.values, type: 'f32', access: 'read_write'}
            ],
            invocationCount: this.cellCount,
            workgroupSize: 64,
            ...binned?.exactGate,
            declarations: `${DENSITY_VALUE_WGSL}${DENSITY_RECTANGLE_MEMBERSHIP_WGSL}${GPU_GRID_COORDINATE_WGSL}${getKernelDensityStudyWindowWGSL(studyWindow, edgeCorrection === 'uniform' ? kernel : undefined)}`,
            body: `let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let cellSize = (gridEnd - gridStart) / vec2f(${width}.0, ${height}.0);
  let center = gridStart + (vec2f(f32(index % ${width}u), f32(index / ${width}u)) + vec2f(0.5)) * cellSize;
  let bandwidth = ${bandwidthExpression};
  let total = gathered[gatheredOffset + index];
  let valid = isFiniteValue(total) && isStudyWindowUsable();
  ${getKernelDensityResultWGSL(edgeCorrection, normalization, true)}`
          })
        );
      }
    }
    if (binned) {
      nodes.push(...binned.bodyNodes);
    }
    nodes.push(...staged.publishNodes);
    return nodes;
  }

  /** Per-cell exact gather over the sorted grid index, optionally gated by `method: 'auto'`. */
  private getGatherNode<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    inputs: {
      exactIndex: {cellOffsets: GraphDataView<'uint32'>; sortedIds: GraphDataView<'uint32'>};
      pointWeights: GraphDataView<'float32'>;
      gathered?: GraphDataView<'float32'>;
      values: GraphDataView<'float32'>;
      bandwidthBindings: WGSLKernelBinding[];
      bandwidthExpression: string;
      weightSum?: GraphDataView<'float32'>;
      gate?: {
        condition: GPUCommandGraphNodeCondition<Parameters>;
        extraResources: GraphBufferUse[];
      };
    }
  ): GPUCommandNode<Parameters> {
    const {id, props, cellCount, kernel} = this;
    const edgeCorrection = inputs.gathered ? 'none' : this.edgeCorrection;
    const normalization = inputs.gathered ? 'intensity' : this.normalization;
    const {grid, indexGridSize, indexBounds} = props;
    const {width, height} = grid;
    const {pointWeights, bandwidthBindings, bandwidthExpression, weightSum, gate} = inputs;
    const {cellOffsets, sortedIds} = inputs.exactIndex;
    const domainConstants = getDensityDomainConstantsWGSL(indexBounds);
    const gatherBindings: WGSLKernelBinding[] = [
      {
        name: 'positions',
        view: props.positions as GraphDataView<'float32x2'>,
        type: 'f32',
        access: 'read'
      },
      {name: 'pointWeights', view: pointWeights, type: 'f32', access: 'read'},
      {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
      ...bandwidthBindings,
      {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
      {name: 'sortedIds', view: sortedIds, type: 'u32', access: 'read'},
      ...(normalization === 'probability' && weightSum
        ? [{name: 'weightSum', view: weightSum, type: 'f32', access: 'read'} as const]
        : []),
      {name: 'valuesOut', view: inputs.gathered ?? inputs.values, type: 'f32', access: 'read_write'}
    ];
    const supportRadiusFactor = kernel === 'gaussian' ? GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION : 1;
    const visit = `let pointWeight = pointWeights[pointWeightsOffset + sample];
        if (!(pointWeight > 0.0)) {
          continue;
        }
        let position = vec2f(positions[positionsOffset + 2u * sample], positions[positionsOffset + 2u * sample + 1u]);
        // Distances come from local-origin offsets so large absolute coordinates cancel exactly.
        let delta = (position - gridStart) - cellOffset;
        let distanceSquared = dot(delta, delta);
        if (!(distanceSquared < supportSquared)) {
          continue;
        }
        total += pointWeight * getKernelValue(distanceSquared, bandwidth);`;
    return createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-gather`,
      operation: OPERATION,
      variant: `gather-${kernel}-${edgeCorrection}-${normalization}`,
      bindings: gatherBindings,
      invocationCount: cellCount,
      workgroupSize: 64,
      ...gate,
      declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const INDEX_WIDTH: u32 = ${indexGridSize[0]}u;
const INDEX_HEIGHT: u32 = ${indexGridSize[1]}u;
const SUPPORT_FACTOR: f32 = ${getWGSLFloatLiteral(supportRadiusFactor)};
${domainConstants}
${DENSITY_VALUE_WGSL}${DENSITY_RECTANGLE_MEMBERSHIP_WGSL}${GPU_GRID_COORDINATE_WGSL}
${getKernelValueWGSL(kernel)}
${edgeCorrection === 'none' ? '' : getWindowMassWGSL(kernel)}`,
      body: `let column = index % WIDTH;
  let rasterRow = index / WIDTH;
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let bandwidth = ${bandwidthExpression};
  let windowMin = vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]) - gridStart;
  let windowMax = vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]) - gridStart;
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  let cellOffset = (vec2f(f32(column), f32(rasterRow)) + vec2f(0.5)) * cellSize;
  let center = gridStart + cellOffset;
  let radius = SUPPORT_FACTOR * bandwidth;
  let supportSquared = radius * radius;
  let queryMin = center - vec2f(radius);
  let queryMax = center + vec2f(radius);
  let valid = isFiniteValue(center.x) && isFiniteValue(center.y) &&
    isFiniteValue(bandwidth) && bandwidth > 0.0 && isFiniteValue(radius) &&
    all(gridEnd != gridStart);
  var total = 0.0;
  if (valid && all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX)) {
    ${getDensityNeighborBoundsWGSL()}
    ${getRectangleScanWGSL(visit)}
  }
  ${getKernelDensityResultWGSL(edgeCorrection, normalization)}`
    });
  }
}
