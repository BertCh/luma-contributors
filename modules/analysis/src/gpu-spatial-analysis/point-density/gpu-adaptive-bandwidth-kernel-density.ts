// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUGridIndex,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  getGPUGridIndexMaximumCellCount,
  validateCapacity,
  validateFiniteNumber,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateMaximumCount,
  validatePackedView,
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
import {getRectangleScanWGSL} from '../spatial-interpolation/grid-neighbor-scan';
import {createGridCellRowOrderNodes} from '../shared/grid-cell-row-order';
import {GRID_COORDINATE_WGSL} from '../spatial-interpolation/gpu-inverse-distance-weighting';
import {
  getGPUKernelDensityParameterValues,
  GPU_KERNEL_DENSITY_PARAMETER_LENGTH,
  type GPUKernelDensityNormalization
} from './gpu-kernel-density';
import {
  GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION,
  type GPUKernelDensityKernel,
  type GPUKernelDensityKernelName
} from './kernel-density-math';
import {getKernelValueWGSL} from './kernel-density-wgsl';

const OPERATION = 'GPUAdaptiveBandwidthKernelDensity';

/** Number of float32 elements in a {@link GPUAdaptiveBandwidthKernelDensity} parameter view. */
export const GPU_ADAPTIVE_BANDWIDTH_KERNEL_DENSITY_PARAMETER_LENGTH =
  GPU_KERNEL_DENSITY_PARAMETER_LENGTH + 1;

/** Default sensitivity of the Abramson rule: the square root, `alpha = 1/2` (Abramson 1982). */
export const GPU_ADAPTIVE_BANDWIDTH_DEFAULT_SENSITIVITY = 0.5;

/**
 * Default and largest accepted `bandwidthFactorLimit`. The gather of a cell must search as far as
 * the widest point bandwidth, `5 h L` for a Gaussian kernel, so the limit bounds the cost; 16
 * widens the search square 256-fold.
 */
export const GPU_ADAPTIVE_BANDWIDTH_DEFAULT_FACTOR_LIMIT = 4;
export const GPU_ADAPTIVE_BANDWIDTH_MAXIMUM_FACTOR_LIMIT = 16;

/** Per-frame settings of {@link GPUAdaptiveBandwidthKernelDensity}. */
export type GPUAdaptiveBandwidthKernelDensitySettings = {
  /** Pilot and global bandwidth `h` in position units. Positive and finite. */
  bandwidth: number;
  /** Abramson exponent `alpha` in `[0, 1]`; 0 is the fixed bandwidth. Defaults to 1/2. */
  sensitivity?: number;
};

/**
 * Packs per-frame parameters: the layout of `getGPUKernelDensityParameterValues` (slots 0 to 8, the
 * window slots unused) followed by the sensitivity in slot 9.
 *
 * @param settings Bandwidth and sensitivity.
 * @param grid The output grid: axis-aligned with a nonzero finite cell size.
 * @param target Optional destination of at least
 *   {@link GPU_ADAPTIVE_BANDWIDTH_KERNEL_DENSITY_PARAMETER_LENGTH} elements.
 */
export function getGPUAdaptiveBandwidthKernelDensityParameterValues(
  settings: GPUAdaptiveBandwidthKernelDensitySettings,
  grid: GPURasterMetadata,
  target: Float32Array = new Float32Array(GPU_ADAPTIVE_BANDWIDTH_KERNEL_DENSITY_PARAMETER_LENGTH)
): Float32Array {
  if (target.length < GPU_ADAPTIVE_BANDWIDTH_KERNEL_DENSITY_PARAMETER_LENGTH) {
    throw new Error(
      `Adaptive kernel density parameter target must hold ${GPU_ADAPTIVE_BANDWIDTH_KERNEL_DENSITY_PARAMETER_LENGTH} elements`
    );
  }
  const sensitivity = settings.sensitivity ?? GPU_ADAPTIVE_BANDWIDTH_DEFAULT_SENSITIVITY;
  if (!(sensitivity >= 0 && sensitivity <= 1)) {
    throw new Error('Adaptive kernel density sensitivity must be in [0, 1]');
  }
  getGPUKernelDensityParameterValues({bandwidth: settings.bandwidth}, grid, target);
  target[GPU_KERNEL_DENSITY_PARAMETER_LENGTH] = sensitivity;
  return target;
}

/** Properties for {@link GPUAdaptiveBandwidthKernelDensity}. */
export type GPUAdaptiveBandwidthKernelDensityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'adaptive-kernel-density'`. */
  id?: string;
  /** Planar/native only, as {@link GPUKernelDensity}. */
  spatialContext?: GPUSpatialContext;
  /** Packed point positions in the same units as the grid and bandwidth. */
  positions: GraphDataView<'float32x2'>;
  /** Optional non-negative per-point weights; non-finite and negative weights exclude the point. */
  weights?: GraphDataView<'float32'>;
  /**
   * Per-frame float32 view of at least
   * {@link GPU_ADAPTIVE_BANDWIDTH_KERNEL_DENSITY_PARAMETER_LENGTH} values written with
   * {@link getGPUAdaptiveBandwidthKernelDensityParameterValues}.
   */
  parameters: GraphDataView<'float32'>;
  /** Output raster grid, as {@link GPUKernelDensityProps.grid}. */
  grid: GPURasterMetadata;
  /** Dimensions of the internal `GPUGridIndex`, as {@link GPUKernelDensityProps.indexGridSize}. */
  indexGridSize: readonly [number, number];
  /** Inclusive point domain of the grid index; points outside it are excluded. */
  indexBounds: readonly [number, number, number, number];
  /** Kernel, as {@link GPUKernelDensityProps.kernel}. Defaults to `'gaussian'`. */
  kernel?: GPUKernelDensityKernelName;
  /** `'intensity'` (default) or `'probability'`, as {@link GPUKernelDensityProps.normalization}. */
  normalization?: GPUKernelDensityNormalization;
  /**
   * Each point bandwidth is clamped to `[h / L, h L]` with `L` this limit (compile-time: it sets
   * the search radius). Defaults to {@link GPU_ADAPTIVE_BANDWIDTH_DEFAULT_FACTOR_LIMIT}.
   */
  bandwidthFactorLimit?: number;
  /** Caller-owned outputs. */
  output: GPUAdaptiveBandwidthKernelDensityOutput;
};

/** Caller-owned outputs of {@link GPUAdaptiveBandwidthKernelDensity}. */
export type GPUAdaptiveBandwidthKernelDensityOutput = {
  /** Density per output cell (`grid.width * grid.height` rows), NaN where the bandwidth is invalid. */
  values: GraphDataView<'float32'>;
  /** Optional one-row sum of the included weights, the divisor of `'probability'`. */
  weightSum?: GraphDataView<'float32'>;
  /** Optional bandwidth of each point (0 for excluded points), in position units. */
  pointBandwidths?: GraphDataView<'float32'>;
  /** Optional fixed-bandwidth pilot density at each point (0 for excluded points). */
  pilotDensities?: GraphDataView<'float32'>;
};

/** Defines `DOMAIN_MIN` and `DOMAIN_MAX` of the index domain. */
function getDomainConstantsWGSL(bounds: readonly [number, number, number, number]): string {
  return `const DOMAIN_MIN: vec2f = vec2f(${getWGSLFloatLiteral(bounds[0])}, ${getWGSLFloatLiteral(bounds[1])});
const DOMAIN_MAX: vec2f = vec2f(${getWGSLFloatLiteral(bounds[2])}, ${getWGSLFloatLiteral(bounds[3])});`;
}

/**
 * Adaptive-bandwidth (variable-kernel) density estimate on a raster: the bandwidth of each point
 * is wider where the data are sparse and narrower where they are dense.
 *
 * `lambda(u) = sum_i w_i K_{h_i}(|u - x_i|)` with `h_i = h * clamp((f(x_i) / g)^(-alpha), 1/L, L)`,
 * where `f` is the fixed-bandwidth pilot estimate with bandwidth `h` (including the point itself),
 * `g = exp(sum_i w_i ln f(x_i) / sum_i w_i)` is its weighted geometric mean and `alpha` is
 * the sensitivity, 1/2 for the square-root law. With `alpha = 0` this is the fixed-bandwidth
 * estimate up to the truncation of the support.
 *
 * **Reference.** Abramson (1982), "On bandwidth variation in kernel estimates: a square root law",
 * Annals of Statistics 10(4); Silverman (1986), Density Estimation for Statistics and Data
 * Analysis, section 5.3.1 (the two-stage form with a geometric-mean scale). The specs compare
 * with an independent binary64 evaluation of those formulas; no package was run (spatstat's
 * `adaptive.density` is a different, resampling-based scheme and was not used).
 *
 * It reuses the kernel shapes, support radii and the parameter packer of {@link GPUKernelDensity}
 * and the same sorted `GPUGridIndex` gather; {@link GPUKernelDensity} itself is unchanged, so its
 * results do not depend on this class. The clamp `L` is a deliberate approximation that keeps the
 * search radius finite. Not implemented: edge correction, the binned method, polygon windows,
 * bandwidth selection.
 *
 * Per-frame: the contents of `parameters` (bandwidth, sensitivity) and the inputs. Compile-time:
 * grid size, point count, index size and bounds, `kernel`, `normalization`,
 * `bandwidthFactorLimit` and which optional outputs exist. Float32 throughout; the specs measure
 * agreement with binary64.
 */
export class GPUAdaptiveBandwidthKernelDensity implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUAdaptiveBandwidthKernelDensityProps;
  /** Resolved coordinate and metric contract (planar/native). */
  readonly spatialContext: GPUSpatialContext;
  /** Canonical kernel name; `'bisquare'` reads back as `'quartic'`. */
  readonly kernel: GPUKernelDensityKernel;
  /** Resolved output scaling. */
  readonly normalization: GPUKernelDensityNormalization;
  /** Resolved clamp of the point bandwidth factor. */
  readonly bandwidthFactorLimit: number;
  /** `grid.width * grid.height`. */
  readonly cellCount: number;

  constructor(props: GPUAdaptiveBandwidthKernelDensityProps) {
    const id = props.id ?? 'adaptive-kernel-density';
    this.id = id;
    this.props = props;
    this.spatialContext = resolveGPUSpatialContext(id, props.spatialContext, {
      ...GPU_PLANAR_SPATIAL_CONTEXT_SUPPORT,
      measure: 'area'
    } as const);
    this.kernel = props.kernel === 'bisquare' ? 'quartic' : (props.kernel ?? 'gaussian');
    this.normalization = props.normalization ?? 'intensity';
    this.bandwidthFactorLimit =
      props.bandwidthFactorLimit ?? GPU_ADAPTIVE_BANDWIDTH_DEFAULT_FACTOR_LIMIT;
    this.cellCount = props.grid.width * props.grid.height;
    validateRequirement(
      id,
      ['gaussian', 'quartic', 'epanechnikov'].includes(this.kernel),
      'kernel must be gaussian, quartic (alias bisquare) or epanechnikov'
    );
    validateRequirement(
      id,
      ['intensity', 'probability'].includes(this.normalization),
      'normalization must be intensity or probability'
    );
    validateFiniteNumber(id, 'bandwidthFactorLimit', this.bandwidthFactorLimit);
    validateRequirement(
      id,
      this.bandwidthFactorLimit >= 1 &&
        this.bandwidthFactorLimit <= GPU_ADAPTIVE_BANDWIDTH_MAXIMUM_FACTOR_LIMIT,
      `bandwidthFactorLimit must be in [1, ${GPU_ADAPTIVE_BANDWIDTH_MAXIMUM_FACTOR_LIMIT}]`
    );
    for (const [name, size] of [
      ['indexGridSize[0]', props.indexGridSize[0]],
      ['indexGridSize[1]', props.indexGridSize[1]]
    ] as const) {
      validateCapacity(id, name, size, 1, Number.MAX_SAFE_INTEGER);
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
    validatePackedView(props.positions, ['float32x2'], `${id} positions`);
    validateRequirement(id, props.positions.length >= 1, 'needs at least one point');
    const pointCount = props.positions.length;
    if (props.weights) {
      validatePackedView(props.weights, ['float32'], `${id} weights`);
      validateRequirement(
        id,
        props.weights.length === pointCount,
        'weights length must equal positions length'
      );
    }
    validatePackedView(props.parameters, ['float32'], `${id} parameters`);
    validateRequirement(
      id,
      props.parameters.length >= GPU_ADAPTIVE_BANDWIDTH_KERNEL_DENSITY_PARAMETER_LENGTH,
      `parameters must hold ${GPU_ADAPTIVE_BANDWIDTH_KERNEL_DENSITY_PARAMETER_LENGTH} float32 values`
    );
    validatePackedView(props.output.values, ['float32'], `${id} output.values`);
    validateRequirement(
      id,
      props.output.values.length >= this.cellCount,
      'output.values must hold grid.width * grid.height rows'
    );
    if (props.output.weightSum) {
      validatePackedView(props.output.weightSum, ['float32'], `${id} output.weightSum`);
      validateRequirement(
        id,
        props.output.weightSum.length === 1,
        'output.weightSum must contain exactly one float32 row'
      );
    }
    for (const name of ['pointBandwidths', 'pilotDensities'] as const) {
      const view = props.output[name];
      if (view) {
        validatePackedView(view, ['float32'], `${id} output.${name}`);
        validateRequirement(
          id,
          view.length >= pointCount,
          `output.${name} must hold one row per point`
        );
      }
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [
        props.output.values,
        props.output.weightSum,
        props.output.pointBandwidths,
        props.output.pilotDensities
      ],
      [props.positions, props.weights, props.parameters]
    );
  }

  /** Returns the index, pilot, bandwidth and gather nodes. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, kernel, normalization, bandwidthFactorLimit, cellCount} = this;
    const {output, grid, indexGridSize, indexBounds} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.positions,
      props.weights,
      props.parameters,
      output.values,
      output.weightSum,
      output.pointBandwidths,
      output.pilotDensities
    ]);
    validateMaximumCount(
      id,
      'indexGridSize cells',
      indexGridSize[0] * indexGridSize[1],
      getGPUGridIndexMaximumCellCount(graph.device)
    );
    const {width, height} = grid;
    const pointCount = props.positions.length;
    const indexCellCount = indexGridSize[0] * indexGridSize[1];
    const cellOffsets = createTransientView(
      graph,
      `${id}-cell-offsets`,
      'uint32',
      indexCellCount + 1
    );
    const objectIds = createTransientView(graph, `${id}-object-ids`, 'uint32', pointCount);
    const indexCount = createTransientView(graph, `${id}-index-count`, 'uint32', 1);
    const indexOverflow = createTransientView(graph, `${id}-index-overflow`, 'uint32', 1);
    const nodes: GPUCommandNode<Parameters>[] = [
      ...new GPUGridIndex({
        id: `${id}-index`,
        positions: props.positions,
        gridSize: indexGridSize,
        bounds: indexBounds,
        cellOffsets,
        objectIds,
        count: indexCount,
        overflow: indexOverflow
      }).getCommandNodes(graph)
    ];
    // The grid index scatters with atomics; the stable sort restores ascending row order in a
    // cell so that float sums do not depend on thread order.
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

    const pointWeights = createTransientView(graph, `${id}-point-weights`, 'float32', pointCount);
    const pilot =
      output.pilotDensities ?? createTransientView(graph, `${id}-pilot`, 'float32', pointCount);
    const logTerms = createTransientView(graph, `${id}-log-terms`, 'float32', pointCount);
    const bandwidths =
      output.pointBandwidths ??
      createTransientView(graph, `${id}-bandwidths`, 'float32', pointCount);
    const weightSum =
      output.weightSum ?? createTransientView(graph, `${id}-weight-sum`, 'float32', 1);
    const logSum = createTransientView(graph, `${id}-log-sum`, 'float32', 1);
    const domainConstants = getDomainConstantsWGSL(indexBounds);
    const supportFactor = getWGSLFloatLiteral(
      kernel === 'gaussian' ? GPU_KERNEL_DENSITY_GAUSSIAN_TRUNCATION : 1
    );
    const indexBindings: WGSLKernelBinding[] = [
      {name: 'cellOffsets', view: cellOffsets, type: 'u32', access: 'read'},
      {name: 'sortedIds', view: sortedIds, type: 'u32', access: 'read'}
    ];
    const indexDeclarations = `const INDEX_WIDTH: u32 = ${indexGridSize[0]}u;
const INDEX_HEIGHT: u32 = ${indexGridSize[1]}u;
const SUPPORT_FACTOR: f32 = ${supportFactor};
${domainConstants}
${GRID_COORDINATE_WGSL}
${getKernelValueWGSL(kernel)}`;
    // Binds columnLow..rowHigh for the rectangle scan around `center` with search `radius`.
    const getScan = (visit: string) => `let queryMin = center - vec2f(radius);
    let queryMax = center + vec2f(radius);
    if (all(queryMax >= DOMAIN_MIN) && all(queryMin <= DOMAIN_MAX)) {
      let clampedMin = max(queryMin, DOMAIN_MIN);
      let clampedMax = min(queryMax, DOMAIN_MAX);
      // Widened by one index cell per side to absorb f32 rounding of the cell mapping.
      var columnLow = getCoordinate(clampedMin.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH);
      var rowLow = getCoordinate(clampedMin.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT);
      columnLow = columnLow - min(columnLow, 1u);
      rowLow = rowLow - min(rowLow, 1u);
      let columnHigh = min(getCoordinate(clampedMax.x, DOMAIN_MIN.x, DOMAIN_MAX.x, INDEX_WIDTH) + 1u, INDEX_WIDTH - 1u);
      let rowHigh = min(getCoordinate(clampedMax.y, DOMAIN_MIN.y, DOMAIN_MAX.y, INDEX_HEIGHT) + 1u, INDEX_HEIGHT - 1u);
      ${getRectangleScanWGSL(visit)}
    }`;

    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-point-weights`,
        operation: OPERATION,
        variant: 'point-weights',
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          ...(props.weights
            ? [{name: 'inputWeights', view: props.weights, type: 'f32', access: 'read'} as const]
            : []),
          {name: 'pointWeights', view: pointWeights, type: 'f32', access: 'read_write'}
        ],
        invocationCount: pointCount,
        declarations: `${domainConstants}
${GRID_COORDINATE_WGSL}`,
        body: `let position = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let weight = ${props.weights ? 'inputWeights[inputWeightsOffset + index]' : '1.0'};
  let included = isFiniteValue(position.x) && isFiniteValue(position.y) &&
    all(position >= DOMAIN_MIN) && all(position <= DOMAIN_MAX) &&
    isFiniteValue(weight) && weight >= 0.0;
  pointWeights[pointWeightsOffset + index] = select(0.0, weight, included);`
      }),
      ...new GPUReduction({
        id: `${id}-weight-sum`,
        input: pointWeights,
        output: weightSum,
        operation: 'sum'
      }).getCommandNodes(graph),
      // Pilot: the fixed-bandwidth estimate at every included point, and w ln(pilot) for the
      // geometric mean.
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-pilot`,
        operation: OPERATION,
        variant: `pilot-${kernel}`,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'pointWeights', view: pointWeights, type: 'f32', access: 'read'},
          {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
          ...indexBindings,
          {name: 'pilot', view: pilot, type: 'f32', access: 'read_write'},
          {name: 'logTerms', view: logTerms, type: 'f32', access: 'read_write'}
        ],
        invocationCount: pointCount,
        workgroupSize: 64,
        declarations: indexDeclarations,
        body: `let own = pointWeights[pointWeightsOffset + index];
  pilot[pilotOffset + index] = 0.0;
  logTerms[logTermsOffset + index] = 0.0;
  let bandwidth = params[paramsOffset + 4u];
  if (!(own > 0.0) || !(bandwidth > 0.0) || !isFiniteValue(bandwidth)) {
    return;
  }
  let center = vec2f(positions[positionsOffset + 2u * index], positions[positionsOffset + 2u * index + 1u]);
  let radius = SUPPORT_FACTOR * bandwidth;
  let supportSquared = radius * radius;
  var total = 0.0;
  ${getScan(`let pointWeight = pointWeights[pointWeightsOffset + sample];
        if (!(pointWeight > 0.0)) {
          continue;
        }
        let delta = vec2f(positions[positionsOffset + 2u * sample], positions[positionsOffset + 2u * sample + 1u]) - center;
        let distanceSquared = dot(delta, delta);
        if (!(distanceSquared < supportSquared)) {
          continue;
        }
        total += pointWeight * getKernelValue(distanceSquared, bandwidth);`)}
  pilot[pilotOffset + index] = total;
  logTerms[logTermsOffset + index] = select(0.0, own * log(total), total > 0.0);`
      }),
      ...new GPUReduction({
        id: `${id}-log-sum`,
        input: logTerms,
        output: logSum,
        operation: 'sum'
      }).getCommandNodes(graph),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-point-bandwidths`,
        operation: OPERATION,
        variant: 'point-bandwidths',
        bindings: [
          {name: 'pointWeights', view: pointWeights, type: 'f32', access: 'read'},
          {name: 'pilot', view: pilot, type: 'f32', access: 'read'},
          {name: 'logSum', view: logSum, type: 'f32', access: 'read'},
          {name: 'weightSum', view: weightSum, type: 'f32', access: 'read'},
          {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'bandwidths', view: bandwidths, type: 'f32', access: 'read_write'}
        ],
        invocationCount: pointCount,
        workgroupSize: 64,
        declarations: `const FACTOR_LIMIT: f32 = ${getWGSLFloatLiteral(bandwidthFactorLimit)};`,
        body: `let own = pointWeights[pointWeightsOffset + index];
  let pilotValue = pilot[pilotOffset + index];
  let bandwidth = params[paramsOffset + 4u];
  let sensitivity = params[paramsOffset + ${GPU_KERNEL_DENSITY_PARAMETER_LENGTH}u];
  if (!(own > 0.0) || !(pilotValue > 0.0)) {
    bandwidths[bandwidthsOffset + index] = 0.0;
    return;
  }
  let logMean = logSum[logSumOffset] / weightSum[weightSumOffset];
  // (pilot / g)^(-alpha) = exp(-alpha (ln pilot - ln g)) keeps the ratio out of f32 overflow.
  let factor = clamp(exp(-sensitivity * (log(pilotValue) - logMean)), 1.0 / FACTOR_LIMIT, FACTOR_LIMIT);
  bandwidths[bandwidthsOffset + index] = bandwidth * factor;`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-gather`,
        operation: OPERATION,
        variant: `gather-${kernel}-${normalization}`,
        bindings: [
          {name: 'positions', view: props.positions, type: 'f32', access: 'read'},
          {name: 'pointWeights', view: pointWeights, type: 'f32', access: 'read'},
          {name: 'bandwidths', view: bandwidths, type: 'f32', access: 'read'},
          {name: 'params', view: props.parameters, type: 'f32', access: 'read'},
          ...indexBindings,
          ...(normalization === 'probability'
            ? [{name: 'weightSum', view: weightSum, type: 'f32', access: 'read'} as const]
            : []),
          {name: 'valuesOut', view: output.values, type: 'f32', access: 'read_write'}
        ],
        invocationCount: cellCount,
        workgroupSize: 64,
        declarations: `const WIDTH: u32 = ${width}u;
const HEIGHT: u32 = ${height}u;
const FACTOR_LIMIT: f32 = ${getWGSLFloatLiteral(bandwidthFactorLimit)};
${indexDeclarations}`,
        body: `let column = index % WIDTH;
  let rasterRow = index / WIDTH;
  let gridStart = vec2f(params[paramsOffset], params[paramsOffset + 1u]);
  let gridEnd = vec2f(params[paramsOffset + 2u], params[paramsOffset + 3u]);
  let globalBandwidth = params[paramsOffset + 4u];
  let cellSize = (gridEnd - gridStart) / vec2f(f32(WIDTH), f32(HEIGHT));
  let cellOffset = (vec2f(f32(column), f32(rasterRow)) + vec2f(0.5)) * cellSize;
  let center = gridStart + cellOffset;
  let radius = SUPPORT_FACTOR * globalBandwidth * FACTOR_LIMIT;
  let valid = isFiniteValue(center.x) && isFiniteValue(center.y) &&
    isFiniteValue(globalBandwidth) && globalBandwidth > 0.0 && all(gridEnd != gridStart);
  var total = 0.0;
  if (valid) {
    ${getScan(`let pointWeight = pointWeights[pointWeightsOffset + sample];
        if (!(pointWeight > 0.0)) {
          continue;
        }
        let pointBandwidth = bandwidths[bandwidthsOffset + sample];
        let pointRadius = SUPPORT_FACTOR * pointBandwidth;
        // Distances come from local-origin offsets so large absolute coordinates cancel exactly.
        let delta = (vec2f(positions[positionsOffset + 2u * sample], positions[positionsOffset + 2u * sample + 1u]) - gridStart) - cellOffset;
        let distanceSquared = dot(delta, delta);
        if (!(distanceSquared < pointRadius * pointRadius)) {
          continue;
        }
        total += pointWeight * getKernelValue(distanceSquared, pointBandwidth);`)}
  }
  var result = select(getNaN(), total, valid);
  ${
    normalization === 'probability'
      ? `if (valid) {
    let totalWeight = weightSum[weightSumOffset];
    result = select(getNaN(), result / totalWeight, totalWeight > 0.0);
  }`
      : ''
  }
  valuesOut[valuesOutOffset + index] = result;`
      })
    );
    return nodes;
  }
}
