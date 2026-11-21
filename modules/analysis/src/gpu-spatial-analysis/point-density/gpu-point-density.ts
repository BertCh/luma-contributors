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
import {
  createTransientView,
  GPUConvolution,
  getGPUConvolutionSupport,
  GPUGridAggregation,
  GPUGridBinning,
  GPUGroupAggregation,
  GPUHistogram,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUConvolutionStrategy,
  type GraphDataView,
  type GraphTextureView,
  type GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {GPURasterBufferToTexture} from '@luma.gl/graph/gpu-raster';
import type {GPUFloat32Positions} from '@luma.gl/graph/geospatial';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {snapshotDensityConstructionProps} from './density-construction-snapshot';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  captureGraphCommandNodes,
  getCoreBindingElementLimit,
  getGraphViewChunks,
  validateCapacity,
  validateCommandNodeBudget,
  validateFiniteNumber,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateRequirement,
  validateSinglePackedView,
  validatePowerOfTwo
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSampledSurfacePort,
  type GPUSpatialContext
} from '../../analysis-core/contracts/index';
import {
  createPointDensityClearOverflowNode,
  createPointDensityFinalizeNode,
  createPointDensityGridKeysNode,
  createPointDensityHexagonKeysNode,
  createPointDensityMaskedPositionsNode,
  createPointDensityWorkgroupSumNode,
  createPointDensityScaleNodes,
  type PointDensityResolvedBounds
} from './point-density-kernels';
import {
  createFixedPointClearNode,
  createFixedPointDecodeNode,
  FIXED_POINT_WORD_COUNT
} from '../../analysis-core/kernels/fixed-point-accumulator';

import {
  stageDensityOutputViews,
  validateDensityScratchBindings,
  validateDensityViewBindings,
  validateDensityWritableViews
} from './density-point-passes';
import {createPointDensityTiles} from './point-density-tiles';
import {
  createDensityScalarScratch,
  getDensityWeightKeySpans,
  validateDensityPointScratch
} from './density-scratch';

/** Total accepted rows must fit a uint32 cell counter, even when every row hits one cell. */
export const GPU_POINT_DENSITY_MAXIMUM_POINT_COUNT = 0xffffffff;
/** Two scalar position words per tile row must fit a uint32 storage-array length. */
const POINT_DENSITY_MAXIMUM_TILE_ROW_COUNT = 0x40000000;
/** Covers padded point/weight/mask gathers, masking, key, count, scale and sum passes per tile. */
const POINT_DENSITY_TILE_COMMAND_NODE_ALLOWANCE = 32;

/** Published convolution kernels represent kernel dimensions and tap indices as uint32. */
const POINT_DENSITY_MAXIMUM_SMOOTHING_KERNEL_DIMENSION = 0xffffffff;

/** Spatial contexts supported by this area-measuring operation. */
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'area'
} as const);

/** Cell shape. `'grid'` is row-major rectangular cells; `'hexagon'` is pointy-top odd-r hexagons. */
export type GPUPointDensityBinning = 'grid' | 'hexagon';

/** Hex assignment: nearest-center cube rounding (default), or d3-hexbin's native rule. */
export type GPUPointDensityHexagonAssignment = 'cube' | 'd3';

/** Statistic written to `output.values`, the field used for smoothing, extent, histogram, and texture. */
export type GPUPointDensityStatistic = 'count' | 'sum' | 'mean';

/**
 * How weighted per-cell sums are accumulated. `'auto'` (default) selects `'workgroup'`, the
 * deterministic path (algorithms roadmap decision 4: reproducible sums are the contract). `'workgroup'` quantizes each
 * weight to a shared data-derived fixed-point unit and pre-aggregates rows that share a cell inside
 * each workgroup, so hotspot cells stay fast (1M points in one cell: tens of milliseconds) and the
 * sums are bitwise reproducible for every run and row order (each sum is the f32 nearest to the
 * exact sum of the quantized weights; weights below `2^(h - 71)` of the largest, with `h` the bit
 * length of the row count, round to the unit). `'atomic'` adds every row with a global
 * compare-exchange float atomic: it skips the key and scale passes but serializes on cells that
 * many points share (seconds at 1M points per cell) and is NOT deterministic: the last float bits
 * depend on the atomic interleaving. Use it only as an explicit opt-in.
 */
export type GPUPointDensitySumAccumulation = 'auto' | 'workgroup' | 'atomic';

/**
 * Largest `gridSize` cell count of {@link GPUPointDensity} when weighted sums/means are requested on the deterministic
 * `'workgroup'` sums (the `'auto'` default): the fixed-point sum table holds three `uint32` words
 * (12 bytes) per cell and must fit one 128 MiB storage binding, so `floor(2^27 / 12)`. Larger
 * weighted grids need `sumAccumulation: 'atomic'`.
 */
export const GPU_POINT_DENSITY_MAXIMUM_FIXED_POINT_CELL_COUNT = getCoreBindingElementLimit(
  4 * FIXED_POINT_WORD_COUNT
);

/**
 * Allowance for published grid/group aggregation, convolution, reduction, histogram and texture
 * nodes with packed cell outputs. Chunk-dependent point passes are counted separately below.
 */
const POINT_DENSITY_PRIMITIVE_COMMAND_NODE_ALLOWANCE = 512;

/** Selects the sum implementation: `'auto'` is the deterministic fixed-point `'workgroup'` path. */
function resolvePointDensitySumAccumulation(
  requested: GPUPointDensitySumAccumulation | undefined
): Exclude<GPUPointDensitySumAccumulation, 'auto'> {
  return requested === 'atomic' ? 'atomic' : 'workgroup';
}

/** Shared numeric sum requirement: count-only routes do not allocate sum scratch. */
function needsPointDensitySums(
  weighted: boolean,
  statistic: GPUPointDensityStatistic | undefined,
  sums: boolean,
  means: boolean
): boolean {
  return weighted && ((statistic ?? 'count') !== 'count' || sums || means);
}

function getPointDensityTileRoute(
  pointCount: number,
  binning: GPUPointDensityBinning,
  masked: boolean,
  requested: number | undefined,
  byteLimit: number
): {tileRows: number; tiled: boolean} {
  const tileRows =
    requested ??
    2 **
      Math.floor(
        Math.log2(Math.max(1, Math.min(POINT_DENSITY_MAXIMUM_TILE_ROW_COUNT, byteLimit / 8)))
      );
  return {
    tileRows,
    tiled:
      requested !== undefined
        ? pointCount > tileRows
        : (binning === 'hexagon' && pointCount * 4 > byteLimit) ||
          (masked && binning === 'grid' && pointCount * 8 > byteLimit)
  };
}

/**
 * Inclusive `[minX, minY, maxX, maxY]` domain.
 *
 * A literal is compile-time topology. A GPU view is per-frame: one `float32x4` row or four packed
 * `float32` rows, the shape of a `GPUParameterBuffer` with `format: 'float32', length: 4`.
 */
export type GPUPointDensityBounds =
  | readonly [number, number, number, number]
  | GraphDataView<'float32x4'>
  | GraphDataView<'float32'>;

/** Optional square-grid smoothing applied to the field with `GPUConvolution` (zero boundary). */
export type GPUPointDensitySmoothing = {
  /**
   * Dense row-major kernel weights, at least `kernelWidth * kernelHeight` rows. Per-frame contents.
   * Required unless `separableKernel` is set; use it only for non-separable kernels.
   */
  kernel?: GraphDataView<'float32'>;
  /**
   * Two 1D kernels whose outer product is the 2D kernel (`weight(x, y) = horizontal[x] * vertical[y]`,
   * for example a Gaussian). Runs two 1D passes, O(cells * (kernelWidth + kernelHeight)) instead of
   * O(cells * kernelWidth * kernelHeight). Matches the dense path within float32 rounding.
   * Takes precedence over `kernel`. Per-frame contents.
   */
  separableKernel?: {
    /** At least `kernelWidth` float32 rows. */
    horizontal: GraphDataView<'float32'>;
    /** At least `kernelHeight` float32 rows. */
    vertical: GraphDataView<'float32'>;
  };
  /** Positive odd uint32 kernel width. Compile-time; its binding must also fit the device. */
  kernelWidth: number;
  /** Positive odd uint32 kernel height. Compile-time; its binding must also fit the device. */
  kernelHeight: number;
  /**
   * Forwarded to `GPUConvolution` for the dense kernel. Defaults to `'auto'`. Ignored with
   * `separableKernel`, whose 1D passes always run the direct strategy.
   */
  strategy?: GPUConvolutionStrategy;
};

function getSmoothingKernelViews(
  smoothing: GPUPointDensitySmoothing | undefined
): GraphDataView<'float32'>[] {
  if (!smoothing) {
    return [];
  }
  if (smoothing.separableKernel) {
    return [smoothing.separableKernel.horizontal, smoothing.separableKernel.vertical];
  }
  return smoothing.kernel ? [smoothing.kernel] : [];
}

/**
 * Caller-owned packed results, with disjoint aligned bindings and buffers separate from inputs.
 * Fields may share an allocation: those fields use private working buffers and publish their bits
 * after computation, avoiding mixed read-only/writable storage usage on one WebGPU buffer.
 */
export type GPUPointDensityOutput = {
  /** `cellCount` float32 rows: the selected statistic, smoothed when `smoothing` is set. */
  values: GraphDataView<'float32'>;
  /** Optional `cellCount` uint32 rows: unweighted number of accepted points per cell. */
  counts?: GraphDataView<'uint32'>;
  /** Optional `cellCount` float32 rows: sum of finite weights per cell. Requires `weights`. */
  sums?: GraphDataView<'float32'>;
  /** Optional `cellCount` float32 rows: `sums / counts`, `0` for empty cells. Requires `weights`. */
  means?: GraphDataView<'float32'>;
  /** Optional two float32 rows `[min, max]` of `values`, `[0, 0]` when nothing qualifies. */
  extent?: GraphDataView<'float32'>;
  /** Optional uint32 bin counts of `values` over the extent. Its length is the bin count. */
  histogram?: GraphDataView<'uint32'>;
  /** Optional `r32float` storage texture with `width = columns` and `height = rows`. */
  texture?: GraphTextureView<'r32float'>;
  /** Optional completeness scalars. */
  status?: {
    /** Optional flag: 1 when an in-bounds point fell outside the hexagon lattice. Always 0 for grids. */
    overflow?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUPointDensity}.
 *
 * Compile-time: `gridSize`, `binning`, `statistic`, smoothing kernel size, which outputs exist,
 * literal bounds, and a literal radius. Per-frame: positions, weights, GPU bounds, GPU radius, and
 * smoothing kernel weights. Input lengths and chunk boundaries are compile-time.
 * Input view kinds are inferred with packed defaults; outputs always describe a packed cell grid.
 */
export type GPUPointDensityProps<
  Positions extends GPUFloat32Positions = GraphDataView<'float32x2'>,
  Weights extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> = {
  /** Prefix for generated node and transient IDs. Defaults to `'point-density'`. */
  id?: string;
  /** Optional power-of-two packed tile width for hexagon keys and masked point scratch. */
  scratchTileSize?: number;
  /**
   * Coordinate and metric contract of positions and bounds. Planar/native only (omitted means
   * planar/native). Longitude/latitude throws at construction: project to a planar frame first, so
   * a cell and a smoothing kernel have a consistent ground size.
   */
  spatialContext?: GPUSpatialContext;
  /** Packed source positions (single view or chunked vector), at most GPU_POINT_DENSITY_MAXIMUM_POINT_COUNT total rows. */
  positions: Positions;
  /** Optional per-point float32 weights with the same logical length as `positions`. */
  weights?: Weights;
  /**
   * Optional per-row uint32 mask with the same logical length as `positions` (single packed view).
   * Rows whose mask is `0` are excluded from the count, sum, mean, smoothing input, extent, and
   * histogram; any nonzero value includes the row, matching the other contributor mask inputs. The
   * contents are per-frame: rewriting them never recompiles the graph. Omitted means every row is
   * included, and the node list is unchanged.
   */
  mask?: GraphDataView<'uint32'>;
  /** Domain covered by the cells. */
  bounds: GPUPointDensityBounds;
  /** `[columns, rows]` positive integers. Compile-time. */
  gridSize: readonly [number, number];
  /** Cell shape. Defaults to `'grid'`. */
  binning?: GPUPointDensityBinning;
  /**
   * Hexagon center-to-vertex radius in position units. Required for `'hexagon'`. A number is
   * compile-time and must remain positive and finite in float32; a one-row float32 view is per-frame.
   */
  hexagonRadius?: number | GraphDataView<'float32'>;
  /** Compile-time assignment rule. d3 includes its normalized-distance and half-tie semantics. */
  hexagonAssignment?: GPUPointDensityHexagonAssignment;
  /**
   * Weighted sum accumulation. Defaults to `'auto'`, which selects the deterministic
   * `'workgroup'` fixed-point path, which supports at most
   * {@link GPU_POINT_DENSITY_MAXIMUM_FIXED_POINT_CELL_COUNT} cells; `'atomic'` is a
   * non-deterministic opt-in. Ignored without `weights`.
   */
  sumAccumulation?: GPUPointDensitySumAccumulation;
  /** Field statistic. Defaults to `'count'`. `'sum'` and `'mean'` require `weights`. */
  statistic?: GPUPointDensityStatistic;
  /** Optional smoothing, grid binning only. */
  smoothing?: GPUPointDensitySmoothing;
  /** Caller-owned results. */
  output: GPUPointDensityOutput;
};

/** Numeric shape for allocation planning, independent of graph or device objects. */
export type GPUPointDensityExecutionShape = {
  pointCount: number;
  gridSize: readonly [number, number];
  weighted?: boolean;
  statistic?: GPUPointDensityStatistic;
  outputSums?: boolean;
  outputMeans?: boolean;
  outputExtent?: boolean;
  histogramBinCount?: number;
  masked?: boolean;
  binning?: GPUPointDensityBinning;
  sumAccumulation?: GPUPointDensitySumAccumulation;
  scratchTileSize?: number;
  /** Defaults to one packed point chunk and, when weighted, one weight chunk. */
  positionChunkLengths?: readonly number[];
  weightChunkLengths?: readonly number[];
  boundsView?: boolean;
};

/**
 * Bins points into a square or hexagonal grid and writes per-cell count, sum, mean, a heatmap
 * field, its extent and histogram, and an optional `r32float` texture, all on the GPU.
 *
 * This is a binned count (or weight sum) per cell, optionally smoothed, and NOT a kernel density
 * estimate: `output.values` is points (or summed weights) per cell, not intensity per unit area
 * (it changes with `gridSize`), and not a probability density. Optional `smoothing` convolves the
 * field with a kernel whose size and sigma are in CELLS (not ground units) using a zero boundary,
 * so mass leaks out at the domain edge and the field is underestimated within about two sigma of
 * the border; on a grid whose cells are not square on the ground the kernel is anisotropic. For
 * intensity per unit area with a bandwidth in ground units, selectable kernels and edge correction,
 * use {@link GPUKernelDensity}.
 *
 * Square grids reuse `GPUGridBinning` and `GPUGridAggregation`; hexagons use one keys kernel and
 * `GPUGroupAggregation`. `GPUConvolution`, `GPUReduction`, `GPUHistogram`, and
 * `GPURasterBufferToTexture` are reused unchanged. Positions and weights accept chunked
 * vectors, including different chunk boundaries. Per-chunk row offsets preserve alignment;
 * the shared fixed-point scale and summed logical row count give bit-identical packed and
 * chunked deterministic sums. Unmasked grid keys use one scratch binding per position chunk,
 * so their total may exceed one storage binding. Hexagon keys and masked positions switch
 * to fixed-width packed tiles when global scratch exceeds one binding; tail points are NaN
 * and tail weights zero. Published aggregation pipelines therefore use one tile shape. Grid scratch and each individual
 * input/scratch chunk must still fit the device binding limit.
 * Total input rows are limited to 4,294,967,295 on every route. Requested weighted workgroup sums/means
 * additionally limit the whole grid to 11,184,810 cells (three words per cell in 128 MiB);
 * larger weighted grids require explicit atomic sums and must still fit the device binding.
 * The estimated command-node budget is 32,768 across all chunks and tiles.
 */
export class GPUPointDensity<
  Positions extends GPUFloat32Positions = GraphDataView<'float32x2'>,
  Weights extends GraphDataView<'float32'> | GraphVectorView<'float32'> = GraphDataView<'float32'>
> implements GPUCommandNodeProducer
{
  /** Source-backed route limits, in the all-entry catalog vocabulary. */
  static readonly limits: readonly GPUAnalysisRoute[] = Object.freeze([
    {
      name: 'point-count',
      kind: 'hard-limit',
      quantity: 'pointCount',
      unit: 'rows',
      appliesWhen: 'all routes',
      maximum: GPU_POINT_DENSITY_MAXIMUM_POINT_COUNT,
      minimum: 0,
      scope: 'domain',
      accuracyImplications: 'Exact uint32 counts.',
      source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-point-density.ts'
    },
    {
      name: 'fixed-point-cells',
      kind: 'hard-limit',
      quantity: 'gridSize[0] * gridSize[1]',
      unit: 'elements',
      appliesWhen: 'weighted workgroup sums',
      maximum: GPU_POINT_DENSITY_MAXIMUM_FIXED_POINT_CELL_COUNT,
      scope: 'binding',
      alternative: "sumAccumulation: 'atomic'",
      accuracyImplications: 'Atomic sums have order-dependent float32 rounding.',
      source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-point-density.ts'
    },
    {
      name: 'point-scratch',
      kind: 'allocation',
      quantity: 'largest scratch chunk',
      unit: 'bytes',
      appliesWhen: 'masked grid or hexagon',
      byteExpression: 'tileRows * (masked grid ? 8 : 4)',
      scope: 'chunk',
      accuracyImplications: 'Tiling preserves the deterministic sum.',
      source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-point-density.ts'
    }
  ]);

  /** Inspect numeric route and storage bounds without creating views, buffers or command nodes.
   * Port layout, formats and graph ownership are checked when the contributor is constructed.
   */
  static getExecutionPlan(
    shape: GPUPointDensityExecutionShape,
    deviceLimits: GPUDensityDeviceLimits
  ): GPUDensityExecutionPlan {
    const id = 'GPUPointDensity.getExecutionPlan';
    validateRequirement(
      id,
      ['auto', 'workgroup', 'atomic'].includes(shape.sumAccumulation ?? 'auto'),
      `sumAccumulation received ${shape.sumAccumulation}; expected auto, workgroup, or atomic`
    );
    validateRequirement(
      id,
      ['grid', 'hexagon'].includes(shape.binning ?? 'grid'),
      `binning received ${shape.binning}; expected grid or hexagon`
    );
    for (const [name, value] of [
      ['pointCount', shape.pointCount],
      ['gridSize[0]', shape.gridSize[0]],
      ['gridSize[1]', shape.gridSize[1]]
    ] as const)
      validateCapacity(id, name, value, name === 'pointCount' ? 0 : 1, Number.MAX_SAFE_INTEGER);
    if (shape.scratchTileSize !== undefined) {
      validateCapacity(
        id,
        'scratchTileSize',
        shape.scratchTileSize,
        1,
        POINT_DENSITY_MAXIMUM_TILE_ROW_COUNT
      );
      validatePowerOfTwo(id, 'scratchTileSize', shape.scratchTileSize);
    }
    validateRequirement(
      id,
      ['count', 'sum', 'mean'].includes(shape.statistic ?? 'count'),
      `statistic received ${shape.statistic}; expected count, sum, or mean`
    );
    validateRequirement(
      id,
      Boolean(shape.weighted) ||
        ((shape.statistic ?? 'count') === 'count' && !shape.outputSums && !shape.outputMeans),
      'statistic or requested sum/mean output requires weights'
    );
    const cells = shape.gridSize[0] * shape.gridSize[1];
    const accumulation = resolvePointDensitySumAccumulation(shape.sumAccumulation);
    const byteLimit = Math.min(
      deviceLimits.maxBufferSize,
      deviceLimits.maxStorageBufferBindingSize
    );
    const binning = shape.binning ?? 'grid';
    const {tileRows, tiled} = getPointDensityTileRoute(
      shape.pointCount,
      binning,
      Boolean(shape.masked),
      shape.scratchTileSize,
      byteLimit
    );
    const needsSums = needsPointDensitySums(
      Boolean(shape.weighted),
      shape.statistic,
      Boolean(shape.outputSums),
      Boolean(shape.outputMeans)
    );
    const positionChunks = shape.positionChunkLengths ?? [shape.pointCount];
    const weightChunks = shape.weighted ? (shape.weightChunkLengths ?? [shape.pointCount]) : [];
    const commands =
      POINT_DENSITY_PRIMITIVE_COMMAND_NODE_ALLOWANCE +
      Number(Boolean(shape.boundsView) && binning === 'grid') +
      8 * (positionChunks.length + weightChunks.length) +
      (tiled
        ? POINT_DENSITY_TILE_COMMAND_NODE_ALLOWANCE * Math.ceil(shape.pointCount / tileRows)
        : 0);
    const limits: GPUAnalysisRoute[] = [
      this.limits[0],
      {
        ...this.limits[1],
        appliesWhen: 'selected accumulation',
        maximum:
          needsSums && accumulation === 'workgroup'
            ? GPU_POINT_DENSITY_MAXIMUM_FIXED_POINT_CELL_COUNT
            : 0xfffffffe
      },
      {...this.limits[2], maximum: byteLimit},
      {
        name: 'cell-scratch',
        kind: 'allocation',
        quantity: 'cell scratch',
        unit: 'bytes',
        appliesWhen: 'all routes',
        maximum: byteLimit,
        byteExpression: 'cells * (weighted workgroup ? 12 : 4)',
        scope: 'binding',
        accuracyImplications: '',
        source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-point-density.ts'
      },
      {
        name: 'command-budget',
        kind: 'hard-limit',
        quantity: 'estimated command count',
        unit: 'commands',
        appliesWhen: 'all routes',
        maximum: 32768,
        scope: 'domain',
        accuracyImplications: '',
        source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-point-density.ts'
      },
      {
        name: 'packed-mask',
        kind: 'allocation',
        quantity: 'mask',
        unit: 'bytes',
        appliesWhen: 'mask requested',
        maximum: byteLimit,
        scope: 'binding',
        accuracyImplications: '',
        source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-point-density.ts'
      },
      {
        name: 'summary-scratch',
        kind: 'allocation',
        quantity: 'extent and histogram scratch',
        unit: 'bytes',
        appliesWhen: 'summary output requested',
        maximum: byteLimit,
        scope: 'binding',
        accuracyImplications: '',
        source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-point-density.ts'
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
        source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-point-density.ts'
      },
      ...positionChunks.map(
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
          source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-point-density.ts'
        })
      ),
      ...weightChunks.map(
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
          source: 'modules/analysis/src/gpu-spatial-analysis/point-density/gpu-point-density.ts'
        })
      )
    ];
    return makeDensityExecutionPlan(
      `${accumulation}${tiled ? '-tiled' : ''}`,
      limits,
      [
        shape.pointCount,
        cells,
        tiled
          ? tileRows * 8
          : binning === 'hexagon'
            ? shape.pointCount * 4
            : shape.masked
              ? shape.pointCount * 8
              : needsSums && accumulation === 'workgroup'
                ? Math.max(0, ...positionChunks) * 4
                : 0,
        cells * (needsSums && accumulation === 'workgroup' ? 12 : 4),
        commands,
        shape.masked ? shape.pointCount * 4 : 0,
        Math.max(
          shape.outputExtent || shape.histogramBinCount ? 8 : 0,
          (shape.histogramBinCount ?? 0) * 4
        ),
        hasDensityShapeRows(shape.pointCount, positionChunks) &&
        (!shape.weighted || hasDensityShapeRows(shape.pointCount, weightChunks))
          ? 0
          : 1,
        ...positionChunks.map(length => length * 8),
        ...weightChunks.map(length => length * 4)
      ],
      [
        accumulation === 'atomic'
          ? 'Order-dependent float32 sums.'
          : 'Deterministic fixed-point sums; contributions below the shared quantization unit may round to zero.'
      ]
    );
  }

  /** Prefix for every node and transient ID. */
  readonly id: string;
  readonly scratchTileSize: number;
  readonly usesTiledScratch: boolean;
  /** Validated properties. */
  readonly props: GPUPointDensityProps<Positions, Weights>;
  /** Resolved coordinate and metric contract (planar/native). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved cell shape. */
  readonly binning: GPUPointDensityBinning;
  /** Resolved field statistic. */
  readonly statistic: GPUPointDensityStatistic;
  /** Resolved weighted-sum implementation. */
  readonly sumAccumulation: Exclude<GPUPointDensitySumAccumulation, 'auto'>;
  /** `gridSize[0] * gridSize[1]`. */
  readonly cellCount: number;
  /** Conservative node bound, including published primitives and per-chunk point passes. */
  readonly estimatedCommandNodeCount: number;
  /** Canonical sampled-surface port for downstream raster and classification contributors. */
  readonly surface: GPUSampledSurfacePort<'float32'>;

  constructor(inputProps: GPUPointDensityProps<Positions, Weights>) {
    const props = snapshotDensityConstructionProps(
      inputProps,
      [
        'id',
        'scratchTileSize',
        'spatialContext',
        'positions',
        'weights',
        'mask',
        'bounds',
        'gridSize',
        'binning',
        'hexagonRadius',
        'hexagonAssignment',
        'sumAccumulation',
        'statistic',
        'smoothing',
        'output'
      ],
      ['values', 'counts', 'sums', 'means', 'extent', 'histogram', 'texture', 'status'],
      ['overflow']
    );
    this.id = props.id ?? 'point-density';
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(
      this.id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.binning = props.binning ?? 'grid';
    this.statistic = props.statistic ?? 'count';
    const id = this.id;
    const [columns, rows] = props.gridSize;
    this.cellCount = columns * rows;
    this.sumAccumulation = resolvePointDensitySumAccumulation(props.sumAccumulation);
    this.surface = {values: props.output.values, columns, rows};
    const {output, weights, smoothing, mask} = props;
    const firstChunk = getGraphViewChunks(props.positions)[0];
    const limits = firstChunk
      ? (firstChunk.buffer.graph as GPUCommandGraph).device.limits
      : undefined;
    const scratchByteLimit = limits
      ? Math.min(limits.maxStorageBufferBindingSize, limits.maxBufferSize)
      : 8;
    const tileRoute = getPointDensityTileRoute(
      props.positions.length,
      this.binning,
      Boolean(mask),
      props.scratchTileSize,
      scratchByteLimit
    );
    this.scratchTileSize = tileRoute.tileRows;
    validateCapacity(
      id,
      'scratchTileSize',
      this.scratchTileSize,
      1,
      POINT_DENSITY_MAXIMUM_TILE_ROW_COUNT
    );
    validatePowerOfTwo(id, 'scratchTileSize', this.scratchTileSize);
    this.usesTiledScratch = tileRoute.tiled;
    this.estimatedCommandNodeCount =
      POINT_DENSITY_PRIMITIVE_COMMAND_NODE_ALLOWANCE +
      Number(this.binning === 'grid' && needsPointDensityBoundsView(props.bounds)) +
      8 *
        (getGraphViewChunks(props.positions).length +
          (weights ? getGraphViewChunks(weights).length : 0));
    if (this.usesTiledScratch)
      this.estimatedCommandNodeCount +=
        POINT_DENSITY_TILE_COMMAND_NODE_ALLOWANCE *
        Math.ceil(props.positions.length / this.scratchTileSize);
    validateMaximumCount(
      id,
      'positions.length',
      props.positions.length,
      GPU_POINT_DENSITY_MAXIMUM_POINT_COUNT
    );
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);

    if (!['grid', 'hexagon'].includes(this.binning)) {
      validateRequirement(id, false, `binning must be grid or hexagon`);
    }
    if (!['cube', 'd3'].includes(props.hexagonAssignment ?? 'cube')) {
      validateRequirement(id, false, `hexagonAssignment must be cube or d3`);
    }
    if (!['count', 'sum', 'mean'].includes(this.statistic)) {
      validateRequirement(id, false, `statistic must be count, sum, or mean`);
    }
    if (!['auto', 'workgroup', 'atomic'].includes(props.sumAccumulation ?? 'auto')) {
      validateRequirement(id, false, `sumAccumulation must be auto, workgroup, or atomic`);
    }
    validateCapacity(id, 'gridSize columns', columns);
    validateCapacity(id, 'gridSize rows', rows);
    if (this.cellCount >= 0xffffffff) {
      validateRequirement(id, false, `gridSize cell count must be below 2^32 - 1`);
    }
    if (
      needsPointDensitySums(
        Boolean(weights),
        this.statistic,
        Boolean(output.sums),
        Boolean(output.means)
      ) &&
      this.sumAccumulation === 'workgroup'
    ) {
      validateMaximumCount(
        id,
        'gridSize cells',
        this.cellCount,
        GPU_POINT_DENSITY_MAXIMUM_FIXED_POINT_CELL_COUNT
      );
    }
    for (const chunk of getGraphViewChunks(props.positions)) {
      validateSinglePackedView(chunk, ['float32x2'], `${id} positions`);
    }
    if (weights) {
      for (const chunk of getGraphViewChunks(weights)) {
        validateSinglePackedView(chunk, ['float32'], `${id} weights`);
      }
      if (weights.length !== props.positions.length) {
        validateRequirement(id, false, `weights length must equal positions length`);
      }
    }
    for (const [needsWeights, name] of [
      [this.statistic !== 'count', `statistic ${this.statistic}`],
      [Boolean(output.sums), 'output.sums'],
      [Boolean(output.means), 'output.means']
    ] as const) {
      if (needsWeights && !weights) {
        validateRequirement(id, false, `${name} requires weights`);
      }
    }
    if (mask) {
      validateSinglePackedView(mask, ['uint32'], `${id} mask`);
      if (mask.length !== props.positions.length) {
        validateRequirement(id, false, `mask length must equal positions length`);
      }
    }
    validatePointDensityBounds(id, props.bounds);
    if (this.binning === 'hexagon') {
      const radius = props.hexagonRadius;
      if (radius === undefined) {
        validateRequirement(id, false, `hexagon binning requires hexagonRadius`);
      }
      if (typeof radius === 'number') {
        validateFiniteNumber(id, 'hexagonRadius', radius, 0);
        validateRequirement(
          id,
          Math.fround(radius) > 0,
          'hexagonRadius must be positive in float32'
        );
      } else {
        validateSinglePackedView(radius, ['float32'], `${id} hexagonRadius`);
        if (radius.length < 1) {
          validateRequirement(id, false, `hexagonRadius must contain one float32 row`);
        }
      }
      if (smoothing) {
        validateRequirement(id, false, `smoothing requires grid binning`);
      }
    } else if (props.hexagonRadius !== undefined) {
      validateRequirement(id, false, `hexagonRadius requires hexagon binning`);
    }
    if (smoothing) {
      validateRequirement(
        id,
        ['auto', 'direct', 'fft'].includes(smoothing.strategy ?? 'auto'),
        `smoothing.strategy received ${smoothing.strategy}; expected auto, direct, or fft`
      );
      for (const [name, size] of [
        ['smoothing.kernelWidth', smoothing.kernelWidth],
        ['smoothing.kernelHeight', smoothing.kernelHeight]
      ] as const) {
        validateCapacity(id, name, size, 1, POINT_DENSITY_MAXIMUM_SMOOTHING_KERNEL_DIMENSION);
        if (size % 2 === 0) {
          validateRequirement(id, false, `smoothing kernel sizes must be positive odd integers`);
        }
      }
      const {separableKernel} = smoothing;
      if (separableKernel) {
        validateSinglePackedView(
          separableKernel.horizontal,
          ['float32'],
          `${id} smoothing.separableKernel.horizontal`
        );
        validateSinglePackedView(
          separableKernel.vertical,
          ['float32'],
          `${id} smoothing.separableKernel.vertical`
        );
        if (
          separableKernel.horizontal.length < smoothing.kernelWidth ||
          separableKernel.vertical.length < smoothing.kernelHeight
        ) {
          validateRequirement(
            id,
            false,
            `smoothing.separableKernel is shorter than kernelWidth or kernelHeight`
          );
        }
      } else {
        if (!smoothing.kernel) {
          validateRequirement(id, false, `smoothing requires kernel or separableKernel`);
        }
        validateSinglePackedView(smoothing.kernel, ['float32'], `${id} smoothing.kernel`);
        if (smoothing.kernel.length < smoothing.kernelWidth * smoothing.kernelHeight) {
          validateRequirement(
            id,
            false,
            `smoothing.kernel is shorter than kernelWidth * kernelHeight`
          );
        }
      }
    }

    const cellOutputs = [
      ['values', output.values, 'float32'],
      ['counts', output.counts, 'uint32'],
      ['sums', output.sums, 'float32'],
      ['means', output.means, 'float32']
    ] as const;
    for (const [name, view, format] of cellOutputs) {
      if (!view) {
        continue;
      }
      validateSinglePackedView(view, [format], `${id} output.${name}`);
      if (view.length !== this.cellCount) {
        validateRequirement(id, false, `output.${name} must have one row per cell`);
      }
    }
    if (output.extent) {
      validateSinglePackedView(output.extent, ['float32'], `${id} output.extent`);
      if (output.extent.length !== 2) {
        validateRequirement(id, false, `output.extent must have two rows`);
      }
    }
    for (const [name, view] of [
      ['histogram', output.histogram],
      ['status.overflow', output.status?.overflow]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['uint32'], `${id} output.${name}`);
        if (view.length < 1) {
          validateRequirement(id, false, `output.${name} must have at least one row`);
        }
      }
    }
    if (
      output.texture &&
      (output.texture.format !== 'r32float' ||
        output.texture.width !== columns ||
        output.texture.height !== rows)
    ) {
      validateRequirement(id, false, `output.texture must be r32float with gridSize extent`);
    }

    validateDensityWritableViews(id, {
      'output.values': output.values,
      'output.counts': output.counts,
      'output.sums': output.sums,
      'output.means': output.means,
      'output.extent': output.extent,
      'output.histogram': output.histogram,
      'output.status.overflow': output.status?.overflow
    });
    const writableBuffers = [
      output.values,
      output.counts,
      output.sums,
      output.means,
      output.extent,
      output.histogram,
      output.status?.overflow
    ]
      .filter(view => view !== undefined)
      .map(view => view.buffer);
    const inputBuffers = [
      ...getGraphViewChunks(props.positions),
      ...(weights ? getGraphViewChunks(weights) : []),
      ...(mask ? [mask] : []),
      ...(Array.isArray(props.bounds) ? [] : [props.bounds as GraphDataView]),
      ...(typeof props.hexagonRadius === 'object' ? [props.hexagonRadius] : []),
      ...getSmoothingKernelViews(smoothing)
    ].map(view => view.buffer);
    if (writableBuffers.some(buffer => inputBuffers.includes(buffer))) {
      validateRequirement(id, false, `outputs must use separate buffers`);
    }
  }

  /** Returns binning, aggregation, finalize, and optional post-processing nodes in order. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, statistic, sumAccumulation, cellCount} = this;
    const {smoothing, output: callerOutput, gridSize} = props;
    let positions: GPUFloat32Positions = props.positions;
    let weights: GraphDataView<'float32'> | GraphVectorView<'float32'> | undefined = props.weights;
    let mask = props.mask;
    let output = callerOutput;
    validateDensityViewBindings(
      graph,
      id,
      [
        ['positions', positions],
        ['weights', weights],
        ['mask', mask],
        ['bounds', Array.isArray(props.bounds) ? undefined : (props.bounds as GraphDataView)],
        [
          'hexagonRadius',
          typeof props.hexagonRadius === 'object' ? props.hexagonRadius : undefined
        ],
        ...getSmoothingKernelViews(smoothing).map(view => ['smoothing kernel', view] as const),
        ['output.values', output.values],
        ['output.counts', output.counts],
        ['output.sums', output.sums],
        ['output.means', output.means],
        ['output.extent', output.extent],
        ['output.histogram', output.histogram],
        ['output.status.overflow', output.status?.overflow]
      ],
      {skipEmptyChunks: true}
    );
    const needsSums = needsPointDensitySums(
      Boolean(weights),
      statistic,
      Boolean(output.sums),
      Boolean(output.means)
    );
    validateDensityScratchBindings(graph, id, [['grid scratch', cellCount * 4]]);
    if (output.histogram || output.extent) {
      validateDensityScratchBindings(graph, id, [['extent scratch', 8]]);
    }
    if (output.histogram) {
      validateDensityScratchBindings(graph, id, [
        ['histogram scratch', output.histogram.length * 4]
      ]);
    }
    if (needsSums && sumAccumulation === 'workgroup') {
      validateDensityScratchBindings(graph, id, [
        ['fixed-point grid scratch', cellCount * 4 * FIXED_POINT_WORD_COUNT]
      ]);
    }
    if (this.usesTiledScratch) {
      validateDensityScratchBindings(graph, id, [['point tile scratch', this.scratchTileSize * 8]]);
      validateCapacity(
        id,
        'point tile scratch byteLength',
        this.scratchTileSize * 8,
        0,
        graph.device.limits.maxBufferSize
      );
    } else if (this.binning === 'hexagon') {
      validateDensityScratchBindings(graph, id, [['point key scratch', positions.length * 4]]);
    } else if (needsSums && sumAccumulation === 'workgroup') {
      validateDensityPointScratch(graph, id, positions, 4);
    }
    if (mask && this.binning === 'grid' && !this.usesTiledScratch) {
      validateDensityScratchBindings(graph, id, [
        ['masked position scratch', positions.length * 8]
      ]);
    }
    if (smoothing && !smoothing.separableKernel) {
      const support = getGPUConvolutionSupport(graph.device, {
        width: gridSize[0],
        height: gridSize[1],
        kernelWidth: smoothing.kernelWidth,
        kernelHeight: smoothing.kernelHeight,
        strategy: smoothing.strategy ?? 'auto',
        boundary: 'zero'
      });
      if (support.strategy === 'fft' || smoothing.strategy === 'fft') {
        validateDensityScratchBindings(graph, id, [
          ['smoothing FFT scratch', support.stats?.fftComplexBufferByteLength ?? 0]
        ]);
      }
      validateRequirement(
        id,
        support.supported,
        `smoothing must fit device limits: ${support.reason ?? ''}`
      );
    }
    const boundsView = Array.isArray(props.bounds) ? undefined : (props.bounds as GraphDataView);
    const radiusView = typeof props.hexagonRadius === 'object' ? props.hexagonRadius : undefined;
    validateGraphViewsBelongToGraph(id, graph, [
      positions,
      weights,
      mask,
      boundsView,
      radiusView,
      ...getSmoothingKernelViews(smoothing),
      output.values,
      output.counts,
      output.sums,
      output.means,
      output.extent,
      output.histogram,
      output.status?.overflow
    ]);
    if (output.texture && output.texture.texture.graph !== graph) {
      throw new Error(`${id} views must belong to the target graph`);
    }

    const staged = stageDensityOutputViews(
      graph,
      id,
      'GPUPointDensity',
      {
        values: output.values,
        counts: output.counts,
        sums: output.sums,
        means: output.means,
        extent: output.extent,
        histogram: output.histogram,
        overflow: output.status?.overflow
      },
      {overflow: 1}
    );
    output = {
      ...callerOutput,
      ...staged.views,
      status: callerOutput.status ? {overflow: staged.views.overflow} : undefined
    };

    const pointTiles = this.usesTiledScratch
      ? createPointDensityTiles(graph, id, positions, weights, mask, this.scratchTileSize)
      : undefined;
    if (pointTiles) {
      positions = pointTiles.positions;
      weights = pointTiles.weights;
      mask = undefined;
    }
    let bounds: PointDensityResolvedBounds =
      boundsView && boundsView.format === 'float32'
        ? graph.createDataView(boundsView.buffer, {
            format: 'float32x4',
            length: 1,
            byteOffset: boundsView.byteOffset
          })
        : (props.bounds as PointDensityResolvedBounds);
    const counts =
      output.counts ?? createTransientView(graph, `${id}-counts-scratch`, 'uint32', cellCount);
    const sums = needsSums
      ? (output.sums ?? createTransientView(graph, `${id}-sums-scratch`, 'float32', cellCount))
      : undefined;
    const field = smoothing
      ? createTransientView(graph, `${id}-unsmoothed-values`, 'float32', cellCount)
      : output.values;
    const extent =
      output.extent ??
      (output.histogram
        ? createTransientView(graph, `${id}-extent-scratch`, 'float32', 2)
        : undefined);

    const nodes: GPUCommandNode<Parameters>[] = [...(pointTiles?.nodes ?? [])];
    if (this.binning === 'grid' && needsPointDensityBoundsView(props.bounds)) {
      const literalBounds = props.bounds as readonly number[];
      const boundsValues = createTransientView(graph, `${id}-grid-bounds`, 'float32x4', 1);
      nodes.push(
        createWGSLKernelNode(graph, {
          id: `${id}-grid-bounds`,
          operation: 'GPUPointDensity',
          variant: 'literal-bounds',
          invocationCount: 4,
          bindings: [{name: 'boundsValues', view: boundsValues, type: 'f32', access: 'read_write'}],
          body: `boundsValues[boundsValuesOffset + index] = array<f32, 4>(${literalBounds.map(getWGSLFloatLiteral).join(', ')})[index];`
        })
      );
      bounds = boundsValues;
    }
    if (output.status?.overflow) {
      nodes.push(
        createPointDensityClearOverflowNode(graph, `${id}-clear-overflow`, output.status.overflow)
      );
    }
    if (this.binning === 'hexagon') {
      const keys = createDensityScalarScratch(
        graph,
        `${id}-hexagon-keys`,
        positions,
        'uint32',
        !this.usesTiledScratch
      );
      const keyChunks = getGraphViewChunks(keys);
      let keyStart = 0;
      for (const [chunkIndex, chunk] of getGraphViewChunks(positions).entries()) {
        if (chunk.length > 0) {
          nodes.push(
            createPointDensityHexagonKeysNode(graph, {
              id: `${id}-hexagon-keys-${chunkIndex}`,
              positions: chunk,
              keys: keyChunks[this.usesTiledScratch ? chunkIndex : 0],
              keyStart: this.usesTiledScratch ? 0 : keyStart,
              gridSize,
              bounds,
              radius: props.hexagonRadius as number | GraphDataView<'float32'>,
              assignment: props.hexagonAssignment,
              mask,
              maskStart: keyStart,
              overflow: output.status?.overflow
            })
          );
        }
        keyStart += chunk.length;
      }
      nodes.push(
        ...new GPUGroupAggregation({id: `${id}-counts`, keys, output: counts}).getCommandNodes(
          graph
        )
      );
      if (sums && weights) {
        if (sumAccumulation === 'atomic') {
          nodes.push(
            ...new GPUGroupAggregation({
              id: `${id}-sums`,
              keys,
              values: weights,
              output: sums,
              operation: 'sum'
            }).getCommandNodes(graph)
          );
        } else {
          nodes.push(
            ...createWorkgroupSumNodes(
              graph,
              `${id}-sums`,
              weights,
              keys,
              sums,
              props.positions.length
            )
          );
        }
      }
    } else {
      // Grid kernels have no mask input: masked rows become NaN positions, which they ignore.
      let gridPositions: GPUFloat32Positions = positions;
      if (mask) {
        const maskedPositions = createTransientView(
          graph,
          `${id}-masked-positions`,
          'float32x2',
          positions.length
        );
        let rowStart = 0;
        for (const [chunkIndex, chunk] of getGraphViewChunks(positions).entries()) {
          if (chunk.length > 0) {
            nodes.push(
              createPointDensityMaskedPositionsNode(graph, {
                id: `${id}-mask-positions-${chunkIndex}`,
                positions: chunk,
                mask,
                maskedPositions,
                rowStart
              })
            );
          }
          rowStart += chunk.length;
        }
        gridPositions = maskedPositions;
      }
      nodes.push(
        ...new GPUGridBinning({
          id: `${id}-counts`,
          positions: gridPositions,
          output: counts,
          gridSize,
          bounds
        }).getCommandNodes(graph)
      );
      if (sums && weights) {
        if (sumAccumulation === 'atomic') {
          nodes.push(
            ...new GPUGridAggregation({
              id: `${id}-sums`,
              positions: gridPositions,
              weights,
              output: sums,
              operation: 'sum',
              gridSize,
              bounds
            }).getCommandNodes(graph)
          );
        } else {
          const keys = createDensityScalarScratch(
            graph,
            `${id}-grid-keys`,
            gridPositions,
            'uint32'
          );
          const keyChunks = getGraphViewChunks(keys);
          for (const [chunkIndex, chunk] of getGraphViewChunks(gridPositions).entries()) {
            if (chunk.length > 0) {
              nodes.push(
                createPointDensityGridKeysNode(graph, {
                  id: `${id}-sums-grid-keys-${chunkIndex}`,
                  positions: chunk,
                  keys: keyChunks[chunkIndex],
                  keyStart: 0,
                  gridSize,
                  bounds
                })
              );
            }
          }
          nodes.push(
            ...createWorkgroupSumNodes(
              graph,
              `${id}-sums`,
              weights,
              keys,
              sums,
              props.positions.length
            )
          );
        }
      }
    }
    nodes.push(
      createPointDensityFinalizeNode(graph, {
        id: `${id}-finalize`,
        statistic,
        counts,
        sums,
        values: field,
        means: output.means
      })
    );
    if (smoothing?.separableKernel) {
      const rowPass = createTransientView(graph, `${id}-smooth-rows`, 'float32', cellCount);
      const sharedProps = {width: gridSize[0], height: gridSize[1], boundary: 'zero'} as const;
      nodes.push(
        ...new GPUConvolution({
          ...sharedProps,
          id: `${id}-smooth-horizontal`,
          input: field,
          kernel: smoothing.separableKernel.horizontal,
          output: rowPass,
          kernelWidth: smoothing.kernelWidth,
          kernelHeight: 1,
          strategy: 'direct'
        }).getCommandNodes(graph),
        ...new GPUConvolution({
          ...sharedProps,
          id: `${id}-smooth-vertical`,
          input: rowPass,
          kernel: smoothing.separableKernel.vertical,
          output: output.values,
          kernelWidth: 1,
          kernelHeight: smoothing.kernelHeight,
          strategy: 'direct'
        }).getCommandNodes(graph)
      );
    } else if (smoothing) {
      nodes.push(
        ...new GPUConvolution({
          id: `${id}-smooth`,
          input: field,
          kernel: smoothing.kernel!,
          output: output.values,
          width: gridSize[0],
          height: gridSize[1],
          kernelWidth: smoothing.kernelWidth,
          kernelHeight: smoothing.kernelHeight,
          boundary: 'zero',
          strategy: smoothing.strategy ?? 'auto'
        }).getCommandNodes(graph)
      );
    }
    // Without smoothing, statistics cover occupied cells only: any nonzero count selects a row.
    const statisticsMask = smoothing ? undefined : counts;
    if (extent) {
      nodes.push(
        ...new GPUReduction({
          id: `${id}-extent`,
          input: output.values,
          mask: statisticsMask,
          output: extent,
          operation: 'extent'
        }).getCommandNodes(graph)
      );
    }
    const histogram = output.histogram;
    if (histogram && extent) {
      nodes.push(
        ...new GPUHistogram({
          id: `${id}-histogram`,
          input: output.values,
          mask: statisticsMask,
          output: histogram,
          domain: extent
        }).getCommandNodes(graph)
      );
    }
    if (output.texture) {
      const texture = output.texture;
      nodes.push(
        ...captureGraphCommandNodes(graph, () =>
          new GPURasterBufferToTexture({
            id: `${id}-texture`,
            input: {
              id: `${id}-texture-band`,
              format: 'float32',
              storage: {kind: 'buffer', values: output.values}
            },
            output: texture
          }).addToGraph(graph)
        )
      );
    }
    nodes.push(...staged.publishNodes);
    return nodes;
  }
}

// Published grid primitives append .0 to integer literals, producing invalid 3e+38.0.
// Keep their public view route for these accepted bounds; ordinary graphs stay unchanged.
function needsPointDensityBoundsView(bounds: GPUPointDensityBounds): boolean {
  return (
    Array.isArray(bounds) &&
    bounds.some(value => Number.isInteger(value) && /e/i.test(String(value)))
  );
}

/** Validates literal or GPU-resident density bounds. */
function validatePointDensityBounds(id: string, bounds: GPUPointDensityBounds): void {
  if (Array.isArray(bounds)) {
    const [minX, minY, maxX, maxY] = bounds as readonly number[];
    if (
      bounds.length !== 4 ||
      !bounds.every(value => Number.isFinite(Math.fround(value))) ||
      minX > maxX ||
      minY > maxY
    ) {
      validateRequirement(
        id,
        false,
        `bounds must be finite float32 ordered [minX, minY, maxX, maxY]; received ${bounds}`
      );
    }
    return;
  }
  const view = bounds as GraphDataView;
  validateSinglePackedView(view, ['float32x4', 'float32'], `${id} bounds`);
  if (
    (view.format === 'float32x4' && view.length !== 1) ||
    (view.format === 'float32' && view.length !== 4)
  ) {
    validateRequirement(
      id,
      false,
      `bounds view must be one float32x4 row or four float32 rows; received ${view.format} length ${view.length}`
    );
  }
}

/**
 * Deterministic weighted sums (algorithms roadmap decision 4): the fixed-point scale over every
 * weight, a clear, one workgroup pre-aggregated fixed-point add per weight chunk, then a decode
 * into `sums`.
 */
function createWorkgroupSumNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  weights: GraphDataView<'float32'> | GraphVectorView<'float32'>,
  keys: GraphDataView<'uint32'> | GraphVectorView<'uint32'>,
  sums: GraphDataView<'float32'>,
  maximumTermCount = weights.length
): GPUCommandNode<Parameters>[] {
  const scale = createTransientView(graph, `${id}-scale`, 'uint32', 1);
  const fixedPointSums = createTransientView(
    graph,
    `${id}-fixed-point`,
    'uint32',
    sums.length * FIXED_POINT_WORD_COUNT
  );
  const nodes = [
    ...createPointDensityScaleNodes<Parameters>(graph, {
      id: `${id}-scale`,
      values: weights,
      scale
    }),
    createFixedPointClearNode<Parameters>(graph, {
      id: `${id}-clear`,
      operation: 'GPUPointDensity',
      words: fixedPointSums
    })
  ];
  for (const [chunkIndex, span] of getDensityWeightKeySpans(graph, weights, keys).entries()) {
    nodes.push(
      createPointDensityWorkgroupSumNode(graph, {
        id: `${id}-workgroup-${chunkIndex}`,
        weights: span.weights,
        keys: span.keys,
        keyStart: 0,
        fixedPointSums,
        scale,
        rowCount: maximumTermCount
      })
    );
  }
  nodes.push(
    createFixedPointDecodeNode<Parameters>(graph, {
      id: `${id}-decode`,
      operation: 'GPUPointDensity',
      words: fixedPointSums,
      scale,
      maximumTermCount,
      output: sums
    })
  );
  return nodes;
}
