// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {resolveKernelShapeName} from '../shared/kernel-shape-name';
import {
  createTransientView,
  GPUReduction,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  getGPUGridIndexMaximumCellCount,
  getStorageBindingElementLimit,
  validateCapacity,
  validateFiniteNumber,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  createWGSLKernelNode,
  getWGSLFloatLiteral
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  resolveGPUSpatialContext,
  type GPUSpatialContext,
  type GPUTemporalContext
} from '../../analysis-core/contracts/index';
import {GET_ORDERED_KEY_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {validateRasterGrid} from '../../raster-analysis/raster-band-utils';
import {GPU_GRID_COORDINATE_WGSL} from '@luma.gl/graph/gpu-core';
import {DENSITY_VALUE_WGSL} from './density-neighbor-wgsl';
import {createGridCellRowOrderNodes} from '../shared/grid-cell-row-order';
import {createPointNeighborGridIndexNodes} from '../../analysis-core/kernels/point-neighbor-grid-index';
import {resolveGPUSpaceTimeTemporalContext} from '../space-time-tests/space-time-bands';
import {
  GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION,
  type GPUKernelDensityKernel,
  type GPUKernelDensityKernelName
} from './kernel-density-math';
import {getKernelValueWGSL, getWindowMassWGSL} from './kernel-density-wgsl';
import {
  GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH,
  type GPUSpaceTimeTemporalKernel,
  type GPUSpaceTimeTemporalKernelName
} from './space-time-kernel-density-math';
import {
  getTemporalKernelValueWGSL,
  getTemporalWindowMassWGSL
} from './space-time-kernel-density-wgsl';

import {
  DENSITY_MAXIMUM_SORT_ROW_COUNT,
  DENSITY_MAXIMUM_INDEX_DIMENSION,
  stageDensityOutputViews,
  validateDensityIndexedScratch,
  validateDensityScratchBindings,
  validateDensityViewBindings,
  validateDensityWritableViews
} from './density-point-passes';

import {snapshotDensityConstructionProps} from './density-construction-snapshot';

const OPERATION = 'GPUSpaceTimeKernelDensity';

/** Spatial contexts supported by this area-measuring operation. */
const SPATIAL_CONTEXT_SUPPORT = Object.freeze({
  ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
  measure: 'area'
} as const);

/**
 * Sort key of a row the density ignores. It is above the ordered key of every finite float32
 * (the largest is 0xff7fffff), so ignored rows sort last.
 */
const IGNORED_ROW_KEY = 0xffffffff;

/**
 * Edge treatment of one axis of {@link GPUSpaceTimeKernelDensity}. The number of voxels is limited
 * by the output binding (4 bytes per row), `getStorageBindingElementLimit(device, 4)`, 2^25 on a
 * default device.
 */
export type GPUSpaceTimeKernelDensityEdgeCorrection = 'none' | 'uniform';

/** Caller-owned outputs of {@link GPUSpaceTimeKernelDensity}. */
export type GPUSpaceTimeKernelDensityOutput = {
  /**
   * Density per voxel, at least `grid.width * grid.height * sliceCount` float32 rows in cell-major
   * order: row `(y * grid.width + x) * sliceCount + slice`, with `y * grid.width + x` in the grid's
   * row order (see `GPUKernelDensity`). Units: `'intensity'` is weight per unit area per time
   * unit; `'probability'` divides by the included weight, a density over space and time that
   * has a window integral depending on event locations under uniform edge correction. NaN where the cell center or slice time is outside an
   * edge-correction window, a bandwidth is invalid, or (probability) no weight is included.
   */
  values: GraphDataView<'float32'>;
  /** Optional one-row float32 sum of the included weights, before edge correction. */
  weightSum?: GraphDataView<'float32'>;
  /** Optional diagnostic scalars. */
  status?: {
    /**
     * Optional one-row uint32 count of input rows that were excluded: a non-finite position,
     * time or weight, a negative weight, a position outside `indexBounds`, or, with an edge
     * correction, a position outside the spatial window or a time outside the time window.
     */
    invalidCount?: GraphDataView<'uint32'>;
  };
};

/**
 * Properties for {@link GPUSpaceTimeKernelDensity}.
 *
 * Per-frame (no recompile): the contents of `parameters` (both bandwidths, grid placement, both
 * windows, slice times) and of every input buffer. Compile-time (needs a new graph): the `grid`
 * size, `sliceCount`, point count, `indexGridSize`, `indexBounds`, both kernels, both edge
 * corrections, `normalization`, and which optional outputs are present.
 */
export type GPUSpaceTimeKernelDensityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'space-time-kernel-density'`. */
  id?: string;
  /**
   * Coordinate and metric contract of positions, grid and spatial bandwidth: planar/native only,
   * as for `GPUKernelDensity`.
   */
  spatialContext?: GPUSpatialContext;
  /**
   * Optional declaration of `times`: encoding `float32-relative`, in the unit of the temporal
   * bandwidth and slice interval, counted from a declared origin. Checked at construction and
   * exposed as {@link GPUSpaceTimeKernelDensity.temporalContext}; nothing is converted.
   */
  temporalContext?: GPUTemporalContext;
  /** Packed point positions in the same units as the grid and spatial bandwidth. */
  positions: GraphDataView<'float32x2'>;
  /**
   * Event time per row, float32, relative to a nearby origin (see `getGPUSpaceTimeRelativeTimes`
   * and `getGPUSpaceTimeGuardedRelativeTimes`), in the unit of the temporal bandwidth. Rows with a
   * non-finite time are excluded.
   */
  times: GraphDataView<'float32'>;
  /** Optional per-point weights. Non-finite and negative weights exclude the point. */
  weights?: GraphDataView<'float32'>;
  /**
   * Per-frame float32 view of at least {@link GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH}
   * values written with `getGPUSpaceTimeKernelDensityParameterValues`.
   */
  parameters: GraphDataView<'float32'>;
  /**
   * Output raster grid: axis-aligned, its width and height size the voxels, and its cell centers
   * are evaluated. Pack the matching placement with `getGPUSpaceTimeKernelDensityParameterValues`.
   */
  grid: GPURasterMetadata;
  /** Number of time slices, at least 1. Slice `k` is evaluated at `firstSliceTime + k * sliceInterval`. */
  sliceCount: number;
  /**
   * Dimensions `[columns, rows]` of the internal `GPUGridIndex` over the points. Aim for a few
   * points per index cell, and index cells not much larger than the spatial bandwidth. At most
   * {@link getGPUGridIndexMaximumCellCount} cells.
   */
  indexGridSize: readonly [number, number];
  /**
   * Inclusive point domain `[minX, minY, maxX, maxY]` of the internal grid index. Points outside it
   * are excluded and counted in `output.status.invalidCount`.
   */
  indexBounds: readonly [number, number, number, number];
  /** Spatial kernel, as for `GPUKernelDensity`. Defaults to `'gaussian'`. */
  spatialKernel?: GPUKernelDensityKernelName;
  /** Temporal kernel. Defaults to `'gaussian'`. */
  temporalKernel?: GPUSpaceTimeTemporalKernelName;
  /**
   * `'uniform'` (default) divides each voxel by the spatial kernel mass inside the spatial window
   * (`parameters.window`, the grid rectangle by default) and excludes points outside it, as
   * `GPUKernelDensity` does; sparr `spattemp.density(sedge = 'uniform')`. `'none'` ignores the window.
   */
  spatialEdgeCorrection?: GPUSpaceTimeKernelDensityEdgeCorrection;
  /**
   * `'uniform'` (default) divides each voxel by the temporal kernel mass inside the time window
   * (`parameters.timeWindow`, the first to the last slice time by default), excludes points whose
   * time is outside it and makes slices outside it NaN; sparr `spattemp.density(tedge = 'uniform')`.
   * `'none'` ignores the window.
   */
  temporalEdgeCorrection?: GPUSpaceTimeKernelDensityEdgeCorrection;
  /** `'intensity'` (default) or `'probability'`: the sum divided by the sum of included weights. */
  normalization?: 'intensity' | 'probability';
  /** Caller-owned outputs. */
  output: GPUSpaceTimeKernelDensityOutput;
};

/**
 * Space-time kernel density on a voxel cube with a fixed product kernel (sparr 2.3-16
 * `spattemp.density`, Fernando and Hazelton 2014; fixed bandwidths).
 *
 * Definition (float64 reference in the spec): for the voxel at cell center `u` and slice time `t`,
 *
 * `f(u, t) = sum_i w_i Ks_h(|u - x_i|) Kt_l(t - t_i) / (es(u) et(t))`
 *
 * with the radially symmetric 2D kernel `Ks_h` of `GPUKernelDensity` (gaussian truncated at 5
 * bandwidths, quartic or epanechnikov), the 1D kernel `Kt_l` of
 * {@link evaluateGPUSpaceTimeTemporalKernel}, and edge factors `es(u)` and `et(t)` that are the
 * kernel masses inside the spatial and time windows under `'uniform'` correction and 1 otherwise.
 * `'probability'` divides by the sum of included weights. Only the `'uniform'` edge correction of
 * sparr is implemented: Diggle-style per-point reweighting, adaptive bandwidths, bandwidth
 * selectors (`OS`, `NS`, `LSCV`, `LIK`, `BOOT`) and tolerance contours are not.
 *
 * Implementation: one thread per voxel over a grid index with a time-sorted run per cell. Rows are
 * stably sorted by time (radix sort of the order-preserving float key), copied in that order into
 * a `GPUGridIndex`, and the row order inside every index cell is restored (`shared/grid-cell-row-order`),
 * which leaves each cell a run of points in ascending time. A voxel visits the index cells under
 * its spatial support in a fixed order (index row, index column), binary-searches each run for the
 * first point inside the temporal support, and walks the run until its time leaves the support.
 * Cost per voxel is the number of index cells under the spatial support times a logarithm, plus the
 * points inside the space-time cylinder; there are no float atomics, so results are deterministic
 * for a device. Distances use offsets from the grid start corner, as `GPUKernelDensity`.
 * The number of points per cell is not capped: a very dense cell costs its points inside the window
 * per voxel.
 *
 * The output is cell-major, `(y * width + x) * sliceCount + slice`, the layout `GPUEmergingHotSpots`
 * and `GPUCubeLayoutTranspose` call `'cell-major'`.
 */
export class GPUSpaceTimeKernelDensity implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUSpaceTimeKernelDensityProps;
  /** Resolved coordinate and metric contract (planar/native). */
  readonly spatialContext: GPUSpatialContext;
  /** Resolved declaration of `times`, when the props gave one. */
  readonly temporalContext?: GPUTemporalContext;
  /** Canonical spatial kernel: a `'bisquare'` prop reads back as `'quartic'`. */
  readonly spatialKernel: GPUKernelDensityKernel;
  /** Resolved temporal kernel. */
  readonly temporalKernel: GPUSpaceTimeTemporalKernel;
  /** Resolved spatial edge correction. */
  readonly spatialEdgeCorrection: GPUSpaceTimeKernelDensityEdgeCorrection;
  /** Resolved temporal edge correction. */
  readonly temporalEdgeCorrection: GPUSpaceTimeKernelDensityEdgeCorrection;
  /** Resolved output scaling. */
  readonly normalization: 'intensity' | 'probability';
  /** `grid.width * grid.height * sliceCount`. */
  readonly voxelCount: number;

  constructor(props: GPUSpaceTimeKernelDensityProps) {
    props = snapshotDensityConstructionProps(
      props,
      [
        'grid',
        'id',
        'indexBounds',
        'indexGridSize',
        'normalization',
        'output',
        'parameters',
        'positions',
        'sliceCount',
        'spatialContext',
        'spatialEdgeCorrection',
        'spatialKernel',
        'temporalContext',
        'temporalEdgeCorrection',
        'temporalKernel',
        'times',
        'weights'
      ],
      ['values', 'weightSum', 'status'],
      ['invalidCount']
    );
    this.id = props.id ?? 'space-time-kernel-density';
    this.props = props;
    const {id} = this;
    this.spatialContext = resolveGPUSpatialContext(
      id,
      props.spatialContext,
      SPATIAL_CONTEXT_SUPPORT
    );
    this.temporalContext = resolveGPUSpaceTimeTemporalContext(id, props.temporalContext);
    for (const [name, kernel] of [
      ['spatialKernel', props.spatialKernel ?? 'gaussian'],
      ['temporalKernel', props.temporalKernel ?? 'gaussian']
    ] as const) {
      validateRequirement(
        id,
        ['gaussian', 'quartic', 'epanechnikov', 'bisquare'].includes(kernel),
        `${name} received ${String(kernel)}; expected one of gaussian, quartic, epanechnikov, bisquare`
      );
    }
    this.spatialKernel = resolveKernelShapeName(id, props.spatialKernel ?? 'gaussian', [
      'gaussian',
      'quartic',
      'epanechnikov'
    ]);
    this.temporalKernel = resolveKernelShapeName(id, props.temporalKernel ?? 'gaussian', [
      'gaussian',
      'quartic',
      'epanechnikov'
    ]);
    this.spatialEdgeCorrection = props.spatialEdgeCorrection ?? 'uniform';
    this.temporalEdgeCorrection = props.temporalEdgeCorrection ?? 'uniform';
    this.normalization = props.normalization ?? 'intensity';
    validateRequirement(
      id,
      ['none', 'uniform'].includes(this.spatialEdgeCorrection) &&
        ['none', 'uniform'].includes(this.temporalEdgeCorrection),
      "edge corrections must be 'none' or 'uniform'"
    );
    validateRequirement(
      id,
      ['intensity', 'probability'].includes(this.normalization),
      'normalization must be intensity or probability'
    );
    const cellCount = validateRasterGrid(id, props.grid);
    validateRequirement(
      id,
      props.grid.affine[1] === 0 && props.grid.affine[3] === 0,
      'needs an axis-aligned grid'
    );
    validateCapacity(id, 'sliceCount', props.sliceCount, 1, DENSITY_MAXIMUM_INDEX_DIMENSION);
    this.voxelCount = cellCount * props.sliceCount;
    for (const [name, size] of [
      ['indexGridSize[0]', props.indexGridSize[0]],
      ['indexGridSize[1]', props.indexGridSize[1]]
    ] as const) {
      validateCapacity(id, name, size, 1, DENSITY_MAXIMUM_INDEX_DIMENSION);
    }
    for (const [index, value] of props.indexBounds.entries()) {
      validateFiniteNumber(id, `indexBounds[${index}]`, value);
    }
    const [minX, minY, maxX, maxY] = props.indexBounds;
    validateRequirement(
      id,
      props.indexBounds.length === 4 &&
        props.indexBounds.every(Number.isFinite) &&
        minX <= maxX &&
        minY <= maxY,
      'indexBounds must be finite and ordered'
    );
    validateSinglePackedView(props.positions, ['float32x2'], `${id} positions`);
    validateSinglePackedView(props.times, ['float32'], `${id} times`);
    validateCapacity(
      id,
      'positions length',
      props.positions.length,
      1,
      DENSITY_MAXIMUM_SORT_ROW_COUNT
    );
    validateRequirement(
      id,
      props.times.length === props.positions.length,
      'times length must equal positions length'
    );
    if (props.weights) {
      validateSinglePackedView(props.weights, ['float32'], `${id} weights`);
      validateRequirement(
        id,
        props.weights.length === props.positions.length,
        'weights length must equal positions length'
      );
    }
    validateSinglePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      props.parameters.length >= GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH,
      `parameters must hold ${GPU_SPACE_TIME_KERNEL_DENSITY_PARAMETER_LENGTH} float32 values`
    );
    const {output} = props;
    validateSinglePackedView(output.values, ['float32'], `${id} output.values`);
    validateRequirement(
      id,
      output.values.length >= this.voxelCount,
      'output.values must hold grid.width * grid.height * sliceCount rows'
    );
    for (const [name, view, format] of [
      ['output.weightSum', output.weightSum, 'float32'],
      ['output.status.invalidCount', output.status?.invalidCount, 'uint32']
    ] as const) {
      if (view) {
        validateSinglePackedView(view, [format], `${id} ${name}`);
        validateRequirement(id, view.length === 1, `${name} must contain exactly one row`);
      }
    }
    const outputViews = [output.values, output.weightSum, output.status?.invalidCount].filter(
      view => view !== undefined
    );
    validateDensityWritableViews(id, {
      values: output.values,
      weightSum: output.weightSum,
      invalidCount: output.status?.invalidCount
    });
    validateGraphOutputsDisjointFromInputs(id, outputViews, [
      props.positions,
      props.times,
      props.weights,
      props.parameters
    ]);
  }

  /**
   * Returns, in order: the row keys, the time sort, the time-ordered copy of the positions, the
   * grid index and in-cell order, the cell-ordered point table, the optional weight and invalid
   * counts, and the voxel gather.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateSpaceTimeKernelDensityBindings(graph, this);
    const {id, props, spatialKernel, temporalKernel, normalization} = this;
    const {grid, indexBounds, indexGridSize, sliceCount} = props;
    const staged = stageDensityOutputViews(
      graph,
      id,
      OPERATION,
      {
        values: props.output.values,
        weightSum: props.output.weightSum,
        invalidCount: props.output.status?.invalidCount
      },
      {values: this.voxelCount}
    );
    const output = {...staged.views, status: {invalidCount: staged.views.invalidCount}};
    const rowCount = props.positions.length;
    const {width, height} = grid;
    const spatialEdge = this.spatialEdgeCorrection === 'uniform';
    const temporalEdge = this.temporalEdgeCorrection === 'uniform';
    const domainConstants = `const DOMAIN_MIN: vec2f = vec2f(${getWGSLFloatLiteral(indexBounds[0])}, ${getWGSLFloatLiteral(indexBounds[1])});
const DOMAIN_MAX: vec2f = vec2f(${getWGSLFloatLiteral(indexBounds[2])}, ${getWGSLFloatLiteral(indexBounds[3])});`;
    const u32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'uint32', Math.max(length, 1));
    const f32 = (name: string, length: number) =>
      createTransientView(graph, `${id}-${name}`, 'float32', Math.max(length, 1));
    const nodes: GPUCommandNode<Parameters>[] = [];

    // 1. Row keys: the ordered time key of an included row, the ignored key otherwise.
    const keys = u32('keys', rowCount);
    const rowIds = u32('row-ids', rowCount);
    const includedWeights = f32('included-weights', rowCount);
    const excluded = output.status?.invalidCount ? u32('excluded', rowCount) : undefined;
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-row-keys`,
        operation: OPERATION,
        variant: `row-keys-${this.spatialEdgeCorrection}-${this.temporalEdgeCorrection}`,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'times', view: props.times, type: 'f32', access: 'read'},
          ...(props.weights
            ? [{name: 'inputWeights', view: props.weights, type: 'f32', access: 'read'} as const]
            : []),
          {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'keys', view: keys, type: 'u32', access: 'read_write'},
          {name: 'rowIds', view: rowIds, type: 'u32', access: 'read_write'},
          {name: 'includedWeights', view: includedWeights, type: 'f32', access: 'read_write'},
          ...(excluded
            ? [{name: 'excluded', view: excluded, type: 'u32', access: 'read_write'} as const]
            : [])
        ],
        invocationCount: rowCount,
        declarations: `${domainConstants}
${IS_FINITE_VALUE_WGSL}
${GET_ORDERED_KEY_WGSL}`,
        body: `let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let windowMin = vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]) - gridStart;
  let windowMax = vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]) - gridStart;
  let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let time = times[timesOffset + index];
  let weight = ${props.weights ? 'inputWeights[inputWeightsOffset + index]' : '1.0'};
  var included = isFiniteValue(position.x) && isFiniteValue(position.y) &&
    all(position >= DOMAIN_MIN) && all(position <= DOMAIN_MAX) &&
    isFiniteValue(time) && isFiniteValue(weight) && weight >= 0.0;
  ${spatialEdge ? 'let relative = position - gridStart;\n  included = included && all(relative >= windowMin) && all(relative <= windowMax);' : ''}
  ${temporalEdge ? 'included = included && time >= params[paramsOffset + 12u] && time <= params[paramsOffset + 13u];' : ''}
  keys[keysOffset + index] = select(${IGNORED_ROW_KEY}u, getOrderedKey(time), included);
  rowIds[rowIdsOffset + index] = index;
  includedWeights[includedWeightsOffset + index] = select(0.0, weight, included);
  ${excluded ? 'excluded[excludedOffset + index] = select(1u, 0u, included);' : ''}`
      })
    );

    // 2. Stable time sort of the row IDs (ties keep row order).
    const sortedKeys = u32('sorted-keys', rowCount);
    const timeOrder = u32('time-order', rowCount);
    nodes.push(
      ...new GPUSort({
        id: `${id}-time-sort`,
        keys,
        values: rowIds,
        outputKeys: sortedKeys,
        outputValues: timeOrder,
        keyBits: 32
      }).getCommandNodes(graph)
    );

    // 3. Positions in time order; ignored rows become NaN, which the grid index skips.
    const sortedPositions = createTransientView(
      graph,
      `${id}-sorted-positions`,
      'float32x2',
      rowCount
    );
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sorted-positions`,
        operation: OPERATION,
        variant: 'sorted-positions',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'timeOrder', view: timeOrder, type: 'u32', access: 'read'},
          {name: 'sortedKeys', view: sortedKeys, type: 'u32', access: 'read'},
          {name: 'sortedPositions', view: sortedPositions, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rowCount,
        declarations:
          'fn getNaNBits() -> f32 { var bits = 0x7fc00000u; return bitcast<f32>(bits); }',
        body: `let row = timeOrder[timeOrderOffset + index];
  let kept = sortedKeys[sortedKeysOffset + index] != ${IGNORED_ROW_KEY}u;
  let nan = getNaNBits();
  sortedPositions[sortedPositionsOffset + 2u * index] = select(nan, positions[positionsOffset + 2u * row], kept);
  sortedPositions[sortedPositionsOffset + 2u * index + 1u] = select(nan, positions[positionsOffset + 2u * row + 1u], kept);`
      })
    );

    // 4. Grid index over the time-ordered positions and ascending time order inside each cell.
    const index = createPointNeighborGridIndexNodes<Parameters>(graph, {
      id,
      indexId: `${id}-index`,
      positions: sortedPositions,
      gridSize: indexGridSize,
      bounds: indexBounds,
      rowCount
    });
    const {cellOffsets, sortedRows: objectIds, gridCount: indexCount} = index;
    nodes.push(...index.nodes);
    // The grid index scatters IDs with atomics, so the order inside a cell varies between runs. A
    // time-ordered copy makes ascending slot order the same as ascending time order.
    const cellRowOrder = createGridCellRowOrderNodes(graph, {
      id: `${id}-cell-row-order`,
      operation: OPERATION,
      cellOffsets,
      objectIds,
      indexCount,
      rowCount
    });
    nodes.push(...cellRowOrder.nodes);

    // 5. Point table in cell order: [x - startX, y - startY, time, weight] per slot.
    const slotPoints = f32('slot-points', 4 * rowCount);
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-slot-points`,
        operation: OPERATION,
        variant: 'slot-points',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'times', view: props.times, type: 'f32', access: 'read'},
          {name: 'includedWeights', view: includedWeights, type: 'f32', access: 'read'},
          {name: 'timeOrder', view: timeOrder, type: 'u32', access: 'read'},
          {name: 'sortedRows', view: cellRowOrder.sortedRows, type: 'u32', access: 'read'},
          {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'slotPoints', view: slotPoints, type: 'f32', access: 'read_write'}
        ],
        invocationCount: rowCount,
        body: `let row = timeOrder[timeOrderOffset + sortedRows[sortedRowsOffset + index]];
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let position = vec2f(positions[positionsOffset + 2u * row], positions[positionsOffset + 2u * row + 1u]);
  let relative = position - gridStart;
  slotPoints[slotPointsOffset + 4u * index] = relative.x;
  slotPoints[slotPointsOffset + 4u * index + 1u] = relative.y;
  slotPoints[slotPointsOffset + 4u * index + 2u] = times[timesOffset + row];
  slotPoints[slotPointsOffset + 4u * index + 3u] = includedWeights[includedWeightsOffset + row];`
      })
    );

    // 6. Weight sum and invalid count.
    const weightSum =
      normalization === 'probability' || output.weightSum
        ? (output.weightSum ?? f32('weight-sum-scratch', 1))
        : undefined;
    if (weightSum) {
      nodes.push(
        ...new GPUReduction({
          id: `${id}-weight-sum`,
          input: includedWeights,
          output: weightSum,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }
    if (excluded && output.status?.invalidCount) {
      nodes.push(
        ...new GPUReduction({
          id: `${id}-invalid-count`,
          input: excluded,
          output: output.status.invalidCount,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    }

    // 7. One thread per voxel.
    const spatialFactor = spatialKernel === 'gaussian' ? GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION : 1;
    const temporalFactor =
      temporalKernel === 'gaussian' ? GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION : 1;
    // The spatial window mass of a gaussian declares erfc; the temporal one must not repeat it.
    const temporalNeedsErfc = !(spatialEdge && spatialKernel === 'gaussian');
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-gather`,
        operation: OPERATION,
        variant: `gather-${spatialKernel}-${temporalKernel}-${this.spatialEdgeCorrection}-${this.temporalEdgeCorrection}-${normalization}`,
        bindings: [
          {name: 'slotPoints', view: slotPoints, type: 'f32', access: 'read'},
          {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
          {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
          ...(normalization === 'probability' && weightSum
            ? [{name: 'weightSum', view: weightSum, type: 'f32', access: 'read'} as const]
            : []),
          {name: 'valuesOut', view: output.values, type: 'f32', access: 'read_write'}
        ],
        invocationCount: this.voxelCount,
        workgroupSize: 64,
        declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const SLICES: u32 = ${sliceCount}u;
const INDEX_WIDTH: u32 = ${indexGridSize[0]}u;
const INDEX_HEIGHT: u32 = ${indexGridSize[1]}u;
const SPATIAL_FACTOR: f32 = ${getWGSLFloatLiteral(spatialFactor)};
const TEMPORAL_FACTOR: f32 = ${getWGSLFloatLiteral(temporalFactor)};
${domainConstants}
${DENSITY_VALUE_WGSL}${GPU_GRID_COORDINATE_WGSL}
${getKernelValueWGSL(spatialKernel)}
${getTemporalKernelValueWGSL(temporalKernel)}
${spatialEdge ? getWindowMassWGSL(spatialKernel) : ''}
${temporalEdge ? getTemporalWindowMassWGSL(temporalKernel, temporalNeedsErfc) : ''}`,
        body: `let slice = index % SLICES;
  let cell = index / SLICES;
  let column = cell % WIDTH;
  let rasterRow = cell / WIDTH;
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let bandwidth = params[paramsOffset + 4u];
  let temporalBandwidth = params[paramsOffset + 9u];
  let sliceTime = fma(f32(slice), params[paramsOffset + 11u], params[paramsOffset + 10u]);
  let windowMin = vec2f(params[paramsOffset + 5u], params[paramsOffset + 6u]) - gridStart;
  let windowMax = vec2f(params[paramsOffset + 7u], params[paramsOffset + 8u]) - gridStart;
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  let cellOffset = (vec2f(f32(column), f32(rasterRow)) + vec2f(0.5)) * cellSize;
  let center = gridStart + cellOffset;
  let radius = SPATIAL_FACTOR * bandwidth;
  let supportSquared = radius * radius;
  let temporalRadius = TEMPORAL_FACTOR * temporalBandwidth;
  let queryMin = center - vec2f(radius);
  let queryMax = center + vec2f(radius);
  var valid = isFiniteValue(center.x) && isFiniteValue(center.y) &&
    isFiniteValue(bandwidth) && bandwidth > 0.0 && isFiniteValue(radius) &&
    isFiniteValue(temporalBandwidth) && temporalBandwidth > 0.0 && isFiniteValue(temporalRadius) &&
    isFiniteValue(sliceTime) && all(gridEnd != gridStart);
  var total = 0.0;
  if (valid && all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX)) {
    let clampedMin = max(queryMin, DOMAIN_MIN);
    let clampedMax = min(queryMax, DOMAIN_MAX);
    // Widen by one index cell per side to absorb f32 rounding in the binning pass; the exact
    // distance test rejects the extra candidates.
    var columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH);
    var rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT);
    columnLow = columnLow - min(columnLow, 1u);
    rowLow = rowLow - min(rowLow, 1u);
    let columnHigh = min(getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH) + 1u, INDEX_WIDTH - 1u);
    let rowHigh = min(getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT) + 1u, INDEX_HEIGHT - 1u);
    let timeLow = sliceTime - temporalRadius;
    let timeHigh = sliceTime + temporalRadius;
    for (var indexRow = rowLow; indexRow <= rowHigh; indexRow++) {
      for (var indexColumn = columnLow; indexColumn <= columnHigh; indexColumn++) {
        let indexCell = indexRow * INDEX_WIDTH + indexColumn;
        let end = cellOffsets[cellOffsetsOffset + indexCell + 1u];
        // First slot of the run whose time is at or after timeLow.
        var low = cellOffsets[cellOffsetsOffset + indexCell];
        var high = end;
        while (low < high) {
          let middle = low + (high - low) / 2u;
          if (slotPoints[slotPointsOffset + 4u * middle + 2u] < timeLow) {
            low = middle + 1u;
          } else {
            high = middle;
          }
        }
        for (var slot = low; slot < end; slot++) {
          let pointTime = slotPoints[slotPointsOffset + 4u * slot + 2u];
          if (pointTime > timeHigh) {
            break;
          }
          let delta = vec2f(slotPoints[slotPointsOffset + 4u * slot], slotPoints[slotPointsOffset + 4u * slot + 1u]) - cellOffset;
          let distanceSquared = dot(delta, delta);
          if (!(distanceSquared < supportSquared)) {
            continue;
          }
          total += slotPoints[slotPointsOffset + 4u * slot + 3u] * getKernelValue(distanceSquared, bandwidth) *
            getTemporalKernelValue(sliceTime - pointTime, temporalBandwidth);
        }
      }
    }
  }
  var result = total;
  ${
    spatialEdge
      ? `if (valid) {
    if (!(all(cellOffset >= windowMin) && all(cellOffset <= windowMax))) {
      valid = false;
    } else {
      let mass = getWindowMass(cellOffset, windowMin, windowMax, bandwidth);
      valid = mass > 1e-6;
      result = result / max(mass, 1e-6);
    }
  }`
      : ''
  }
  ${
    temporalEdge
      ? `if (valid) {
    let windowStart = params[paramsOffset + 12u];
    let windowEnd = params[paramsOffset + 13u];
    if (!(sliceTime >= windowStart && sliceTime <= windowEnd)) {
      valid = false;
    } else {
      let mass = getTemporalWindowMass(sliceTime, windowStart, windowEnd, temporalBandwidth);
      valid = mass > 1e-6;
      result = result / max(mass, 1e-6);
    }
  }`
      : ''
  }
  ${
    normalization === 'probability'
      ? `if (valid) {
    let totalWeight = weightSum[weightSumOffset];
    valid = totalWeight > 0.0;
    result = result / max(totalWeight, 1e-30);
  }`
      : ''
  }
  valuesOut[valuesOutOffset + index] = select(getNaN(), result, valid);`
      })
    );
    return [...nodes, ...staged.publishNodes];
  }
}

/** Checks bindings and nested scratch for internal density composition. Not a barrel export. @internal */
export function validateSpaceTimeKernelDensityBindings<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  contributor: GPUSpaceTimeKernelDensity
): void {
  const {id, props, voxelCount} = contributor;
  const {output, indexGridSize} = props;
  const rowCount = props.positions.length;
  validateGraphViewsBelongToGraph(id, graph, [
    props.positions,
    props.times,
    props.weights,
    props.parameters,
    output.values,
    output.weightSum,
    output.status?.invalidCount
  ]);
  validateMaximumCount(
    id,
    'indexGridSize cells',
    indexGridSize[0] * indexGridSize[1],
    getGPUGridIndexMaximumCellCount(graph.device)
  );
  validateMaximumCount(
    id,
    'voxels (grid.width * grid.height * sliceCount)',
    voxelCount,
    getStorageBindingElementLimit(graph.device, 4)
  );
  // The cell-ordered point table holds four floats per row in one binding.
  validateMaximumCount(
    id,
    'positions length',
    rowCount,
    getStorageBindingElementLimit(graph.device, 16)
  );
  validateDensityViewBindings(
    graph,
    id,
    [
      ['positions', props.positions],
      ['times', props.times],
      ['weights', props.weights],
      ['parameters', props.parameters],
      ['output.values', output.values],
      ['output.weightSum', output.weightSum],
      ['output.status.invalidCount', output.status?.invalidCount]
    ],
    {dispatchMetadata: false}
  );
  validateDensityIndexedScratch(graph, id, rowCount, indexGridSize[0] * indexGridSize[1]);
  validateDensityScratchBindings(graph, id, [['cell-ordered point table scratch', rowCount * 16]]);
}
