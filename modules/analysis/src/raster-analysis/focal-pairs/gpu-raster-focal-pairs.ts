// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPURasterBand, GPURasterMetadata} from '@luma.gl/graph/gpu-raster';
import {
  getRasterBandViews,
  getRasterBufferBandNodes,
  validateRasterBand,
  validateRasterBandBelongsToGraph,
  validateRasterBuffersDistinct,
  validateRasterGrid
} from '../raster-band-utils';
import {
  validateCapacity,
  validateGraphViewsBelongToGraph,
  validatePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getFocalPairsDeclarations, getFocalPairsBody} from './raster-focal-pairs-wgsl';

/** Window radius: a (2 * radius + 1) squared window (17 by 17) bounds each invocation's work and needs no private sample array. */
export const MAXIMUM_FOCAL_PAIRS_RADIUS = 8;
/** Most paired observations a window can hold at the maximum radius, 17 squared. */
export const MAXIMUM_FOCAL_PAIRS_WINDOW_CELLS = (2 * MAXIMUM_FOCAL_PAIRS_RADIUS + 1) ** 2;
/** Scratch rows: six scalar metrics per pixel, addressed with uint32 indices. */
const FOCAL_PAIRS_METRIC_COUNT = 6;

/**
 * Per-cell diagnostic bits of `GPURasterFocalPairsOutput.status`, combined with bitwise or.
 * Constant-band flags do not invalidate covariance.
 */
export const GPU_RASTER_FOCAL_PAIRS_STATUS = {
  /** Bit 0: fewer than `minimumCount` paired-valid observations; all metrics are NaN. */
  insufficientPairs: 1,
  /** Bit 1: required neighborhood or center invalidity (see `noDataPolicy`, `propagateCenterNoData`); all metrics are NaN. */
  missingPair: 2,
  /** Bit 2: zero predictor centered sum of squares; covariance stays zero, regression and correlation are NaN. */
  constantPredictor: 4,
  /** Bit 3: zero response centered sum of squares; covariance stays zero, correlation and R-squared are NaN. */
  constantResponse: 8,
  /** Bit 4: moment or fit overflow, or other nonfinite arithmetic; all metrics are NaN. */
  nonFiniteResult: 16
} as const;

/** Caller-owned packed outputs, one row per raster cell. Undefined float results are NaN. */
export type GPURasterFocalPairsOutput = {
  /** Paired-valid observations, including repeated samples under clamp/reflect, even on failure. */
  pairedCount: GraphDataView<'uint32'>;
  /**
   * Bitwise combination of `GPU_RASTER_FOCAL_PAIRS_STATUS`; zero means all metrics are defined.
   * Bit 0 (1) insufficientPairs, bit 1 (2) missingPair, bit 2 (4) constantPredictor,
   * bit 3 (8) constantResponse, bit 4 (16) nonFiniteResult.
   */
  status: GraphDataView<'uint32'>;
  covariance?: GraphDataView<'float32'>;
  correlation?: GraphDataView<'float32'>;
  /** OLS coefficient in response = intercept + slope * predictor. */
  slope?: GraphDataView<'float32'>;
  intercept?: GraphDataView<'float32'>;
  /** Descriptive coefficient of determination; undefined for a constant response. */
  rSquared?: GraphDataView<'float32'>;
  residualSumSquares?: GraphDataView<'float32'>;
};

export type GPURasterFocalPairsProps = {
  id?: string;
  /** Both bands must already occupy this same row-major grid; no implicit resampling. */
  grid: GPURasterMetadata;
  predictor: GPURasterBand;
  response: GPURasterBand;
  /** Square source-pixel radius in [0, 8], including the center. */
  radius: number;
  /** Defaults to nodata (clipped windows). Reflect excludes duplicated edge pixels. */
  borderMode?: 'nodata' | 'clamp' | 'reflect';
  /** Defaults to ignore: drop a pair if either calibrated observation is invalid. */
  noDataPolicy?: 'ignore' | 'propagate';
  /** Defaults to false. If true, an invalid pair at the center invalidates all float outputs. */
  propagateCenterNoData?: boolean;
  /** Integer in [2, 289]. Defaults to 2; pairedCount is always written. */
  minimumCount?: number;
  /** Defaults to sample (divide by n - 1); population divides by n. */
  covarianceNormalization?: 'sample' | 'population';
  /** At least one float metric must be requested, in addition to count and status. */
  output: GPURasterFocalPairsOutput;
};

const METRICS = [
  'covariance',
  'correlation',
  'slope',
  'intercept',
  'rSquared',
  'residualSumSquares'
] as const;

/**
 * Unweighted focal covariance, Pearson correlation and simple OLS with an intercept.
 *
 * Native nodata and masks are resolved before each band's calibration. Two centered passes
 * accumulate moments over the same paired-valid observations; a third computes squared
 * residuals directly. Residual-mean corrections retain the centering when shader backends
 * reassociate subtraction into a rounded original-coordinate centroid. No normal-equation subtraction of large uncentered moments is used.
 * A constant predictor yields covariance zero and undefined fit/correlation. A constant
 * response with varying predictor has slope zero, its constant intercept and residual zero,
 * but undefined correlation and R². No inferential statistics or implicit regularization.
 *
 * Required halo is radius. Execute on the halo-expanded source and retain only core outputs;
 * clamp/reflect applies to actual domain borders, never to an interior tile seam.
 * Input values and masks may change between graph executions without recompilation.
 */
export class GPURasterFocalPairs implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPURasterFocalPairsProps;
  readonly requiredHalo: number;

  constructor(props: GPURasterFocalPairsProps) {
    this.id = props.id ?? 'raster-focal-pairs';
    this.props = props;
    this.requiredHalo = props.radius;
    const count = validateRasterGrid(this.id, props.grid);
    validateRasterBand(props.predictor, props.grid, `${this.id} predictor`);
    validateRasterBand(props.response, props.grid, `${this.id} response`);
    validateCapacity(this.id, 'radius', props.radius, 0, MAXIMUM_FOCAL_PAIRS_RADIUS);
    validateCapacity(
      this.id,
      'minimumCount',
      props.minimumCount ?? 2,
      2,
      MAXIMUM_FOCAL_PAIRS_WINDOW_CELLS
    );
    validateRequirement(
      this.id,
      ['nodata', 'clamp', 'reflect'].includes(props.borderMode ?? 'nodata'),
      "borderMode must be 'nodata', 'clamp' or 'reflect'"
    );
    validateRequirement(
      this.id,
      ['ignore', 'propagate'].includes(props.noDataPolicy ?? 'ignore'),
      "noDataPolicy must be 'ignore' or 'propagate'"
    );
    validateRequirement(
      this.id,
      ['sample', 'population'].includes(props.covarianceNormalization ?? 'sample'),
      "covarianceNormalization must be 'sample' or 'population'"
    );
    validateRequirement(
      this.id,
      props.propagateCenterNoData === undefined || typeof props.propagateCenterNoData === 'boolean',
      'propagateCenterNoData must be a boolean'
    );
    validateRequirement(
      this.id,
      METRICS.some(metric => props.output[metric]),
      'output must request at least one float metric'
    );
    validateCapacity(
      this.id,
      'grid cell count',
      count,
      1,
      Math.floor(0xffffffff / FOCAL_PAIRS_METRIC_COUNT)
    );
    for (const [name, view] of Object.entries(props.output)) {
      if (!view) continue;
      validatePackedView(
        view,
        [name === 'pairedCount' || name === 'status' ? 'uint32' : 'float32'],
        this.id
      );
      validateRequirement(
        this.id,
        view.length >= count,
        `output.${name} must hold at least one entry per cell`
      );
    }
    validateRasterBuffersDistinct(this.id, Object.values(props.output), [
      ...getRasterBandViews(props.predictor),
      ...getRasterBandViews(props.response)
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateRasterBandBelongsToGraph(id, graph, props.predictor, []);
    validateRasterBandBelongsToGraph(id, graph, props.response, []);
    validateGraphViewsBelongToGraph(id, graph, Object.values(props.output));
    const predictor = getRasterBufferBandNodes(
      graph,
      `${id}-predictor`,
      props.predictor,
      props.grid,
      true
    );
    const response = getRasterBufferBandNodes(
      graph,
      `${id}-response`,
      props.response,
      props.grid,
      true
    );
    const count = props.grid.width * props.grid.height;
    const metrics = createTransientView(graph, `${id}-metrics`, 'float32', count * METRICS.length);
    // Canonicalization writes NaN for every invalid sample, so the paired scan needs two loads.
    const moments = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments`,
      operation: 'GPURasterFocalPairs',
      invocationCount: count,
      bindings: [
        {name: 'predictor', view: predictor.band.storage.values, type: 'f32', access: 'read'},
        {name: 'response', view: response.band.storage.values, type: 'f32', access: 'read'},
        {name: 'metrics', view: metrics, type: 'f32', access: 'read_write'},
        {name: 'pairedCount', view: props.output.pairedCount, type: 'u32', access: 'read_write'},
        {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
      ],
      declarations: getFocalPairsDeclarations(props),
      body: getFocalPairsBody(props)
    });
    const requested = METRICS.flatMap((name, slot) =>
      props.output[name] ? [{name, slot, view: props.output[name]!}] : []
    );
    const project = createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-outputs`,
      operation: 'GPURasterFocalPairs',
      invocationCount: count,
      bindings: [
        {name: 'metrics', view: metrics, type: 'f32', access: 'read'},
        ...requested.map(({name, view}) => ({
          name,
          view,
          type: 'f32' as const,
          access: 'read_write' as const
        }))
      ],
      body: requested
        .map(
          ({name, slot}) =>
            `${name}[${name}Offset + index] = metrics[metricsOffset + index * 6u + ${slot}u];`
        )
        .join('\n')
    });
    return [...predictor.nodes, ...response.nodes, moments, project];
  }
}
