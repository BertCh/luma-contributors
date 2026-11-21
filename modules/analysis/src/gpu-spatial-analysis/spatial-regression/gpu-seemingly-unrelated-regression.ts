// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotSURIterationOutput} from './sur-iteration-output';
import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateRequirement,
  validateCommandNodeBudget,
  validateSinglePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {getSpatialSURNodes, getSpatialSURAllocationLengths} from './spatial-sur-nodes';
import {validateRegressionAllocations} from './spatial-likelihood-allocations';
import {validateRegressionOutputDispatches} from './regression-output-dispatches';
import {getOrdinarySURNodes, getOrdinarySURScratchLength} from './ordinary-sur-nodes';
import {REGRESSION_TILE_MAXIMUM_ROW_COUNT} from './regression-row-tiles';
import {getGPUSeeminglyUnrelatedRegressionParameterValues} from './seemingly-unrelated-regression-parameters';

/** Tiled counts are stored as exact float32 integers, up to 2^24 observations. */
export const GPU_SUR_MAXIMUM_ROW_COUNT = REGRESSION_TILE_MAXIMUM_ROW_COUNT;
/** Nested OLS and GM row counts remain exact float32 integers; bindings are audited separately. */
export const GPU_SPATIAL_SUR_MAXIMUM_ROW_COUNT = REGRESSION_TILE_MAXIMUM_ROW_COUNT;
/** Private dense error covariance and system solves are bounded to four equations. */
export const GPU_SUR_MAXIMUM_EQUATION_COUNT = 4;
/** Each equation uses the existing eight-column (including intercept) dense fit. */
export const GPU_SUR_MAXIMUM_PREDICTOR_COUNT = 7;
/** The joint GLS Cholesky is limited to sixteen coefficients. */
export const GPU_SUR_MAXIMUM_SYSTEM_COEFFICIENT_COUNT = 16;
/** Upper bound per spatial equation for nested tiled OLS, GM, packing and filtering nodes. */
const SPATIAL_EQUATION_COMMAND_NODE_ALLOWANCE = 256;

/** Bounds four unrolled stages per FGLS round; every stage uses an indirect dispatch gate. */
export const GPU_SUR_MAXIMUM_ITERATIONS = 256;

export type GPUSeeminglyUnrelatedRegressionProps = {
  id?: string;
  equationCount: number;
  predictorCount: number;
  /** Up to 2^24 nonspatial rows (subject to device bindings); both routes use audited device bindings. */
  rowCount: number;
  /** Equation-major, then row-major predictors; each equation may have a different design. */
  predictors: GraphDataView<'float32'>;
  /** Equation-major aligned response observations. */
  responses: GraphDataView<'float32'>;
  /** Common row mask across all equations. Nonfinite in any equation excludes the whole row. */
  mask?: GraphDataView<'uint32'>;
  /** Optional equation-major seed residuals for the joint covariance. Iteration requires a
   * subsequent GLS fit with refreshed residuals before convergence can be reported.
   */
  initialResiduals?: GraphDataView<'float32'>;
  /** Optional per-frame [tolerance] view, packed by getGPUSeeminglyUnrelatedRegressionParameterValues.
   * Overrides iteration.tolerance without recompiling. Requires iteration; invalid runtime values
   * report status 5 with count zero and NaN numeric results.
   */
  parameters?: GraphDataView<'float32'>;
  /** Opt into iterated FGLS. The budget includes the first GLS fit; maximum 256 rounds.
   * Stops when max abs(new-old)/max(1,abs(old)) < tolerance (default 1e-5).
   * GPU tolerance must round to a finite normal positive float32 (at least 2^-126).
   * Spatial mode holds first-stage GM parameters fixed and compares coefficients in the
   * original filtered predictor coordinates, with responses anchored at their first value.
   */
  iteration?: {maximumIterations: number; tolerance?: number};
  /** Kelejian-Prucha spatial-error SUR (spreg SURerrorGM) on complete aligned observations.
   * Requires finite, nonnegative, row-standardized zero-diagonal weights without islands.
   * Per-equation error parameters are bounded to [-0.99,0.99]; masks are not accepted in this mode.
   * Accepted rows within 1e-5 of unit sum are normalized consistently before GM and filtering.
   */
  spatial?: {
    weights: GPUSpatialWeights;
    output: {
      parameters: GraphDataView<'float32'>;
      /** Per equation: 0 valid, upstream GM error code, or 4 invalid spatial weights. */
      status: GraphDataView<'uint32'>;
    };
  };
  output: {
    /** Equation-major, intercept first within each equation. */
    coefficients: GraphDataView<'float32'>;
    /** Full system conditional GLS covariance from the last completed round, in the same order. */
    covariance: GraphDataView<'float32'>;
    /** Residual cross-equation covariance used by the last GLS round (divisor complete-case n). */
    residualCovariance: GraphDataView<'float32'>;
    residuals: GraphDataView<'float32'>;
    iteration?: {
      status: {
        /** Completed GLS fits. Zero if the first fit fails. Optional even without iteration. */
        iterationCount?: GraphDataView<'uint32'>;
      };
    };
    /** 0 successful one-step FGLS or converged iteration; 1 insufficient rows; 2 singular design;
     * 3 singular error covariance; 4 iteration budget exhausted (last fit remains usable);
     * 5 invalid runtime parameters.
     */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Nonspatial one-step feasible SUR: equation OLS -> JOINT residual covariance -> system GLS.
 * Observations must share IDs and order. Complete-case selection is common across equations.
 * Row-tile reductions center predictors and responses before the small norm-scaled solves.
 * A common error-covariance scale is restored after solving the normalized GLS system. Coefficient covariance
 * is the conditional GLS information inverse using the first-stage residual covariance by default.
 * Optional iteration repeats covariance and GLS to a coefficient fixed point (Gaussian ML for
 * nonspatial SUR). Optional spatial mode holds GM parameters fixed. No small-sample inference.
 * Shape, iteration budget and output maps are compile-time; input and parameters contents are per-frame.
 */
export class GPUSeeminglyUnrelatedRegression implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUSeeminglyUnrelatedRegressionProps;
  readonly estimatedCommandNodeCount: number;
  constructor(props: GPUSeeminglyUnrelatedRegressionProps) {
    this.id = props.id ?? 'seemingly-unrelated-regression';
    this.props = {
      ...props,
      iteration: props.iteration ? {...props.iteration} : undefined,
      output: snapshotSURIterationOutput(this.id, props.output),
      spatial: props.spatial
        ? {
            ...props.spatial,
            weights: {
              ...props.spatial.weights,
              status: props.spatial.weights.status ? {...props.spatial.weights.status} : undefined
            },
            output: {...props.spatial.output}
          }
        : undefined
    };
    props = this.props;
    const {iteration, ...outputs} = props.output;
    const outputViews = {...outputs, iterationCount: iteration?.status.iterationCount};
    const equations = props.equationCount;
    const columns = props.predictorCount;
    const rows = props.rowCount;
    const size = equations * (columns + 1);
    if (props.iteration) {
      validateRequirement(
        this.id,
        Number.isFinite(props.iteration.tolerance ?? 1e-5) &&
          (props.iteration.tolerance ?? 1e-5) > 0,
        'iteration.tolerance must be a positive finite JavaScript number'
      );
      validateCapacity(
        this.id,
        'iteration.maximumIterations',
        props.iteration.maximumIterations,
        1,
        GPU_SUR_MAXIMUM_ITERATIONS
      );
      if (!props.parameters) {
        try {
          getGPUSeeminglyUnrelatedRegressionParameterValues(props.iteration);
        } catch (error) {
          throw new Error(
            `${this.id} iteration.${(error as Error).message.replace('getGPUSeeminglyUnrelatedRegressionParameterValues ', '')}`
          );
        }
      }
    }
    validateRequirement(
      this.id,
      !props.parameters || Boolean(props.iteration),
      'parameters requires iteration; omit parameters for one-step SUR'
    );
    this.estimatedCommandNodeCount =
      5 +
      (props.spatial && props.iteration ? 1 + 2 * props.iteration.maximumIterations : 0) +
      4 * (props.iteration?.maximumIterations ?? 1) +
      (props.iteration ? Math.ceil(props.iteration.maximumIterations / 8) : 0) +
      (props.iteration && !props.parameters ? 1 : 0) +
      (props.iteration || props.output.iteration?.status.iterationCount ? 1 : 0) +
      (props.spatial ? 6 + equations * SPATIAL_EQUATION_COMMAND_NODE_ALLOWANCE : 0);
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    if (props.spatial) {
      if (
        props.mask ||
        props.initialResiduals ||
        validateGPUSpatialWeights(this.id, props.spatial.weights) !== rows ||
        props.spatial.output.parameters.length !== equations ||
        props.spatial.output.status.length !== equations
      )
        throw new Error(
          `${this.id}: spatial SUR requires complete aligned observations and equation outputs`
        );
      validateSinglePackedView(props.spatial.output.parameters, ['float32'], this.id);
      validateSinglePackedView(props.spatial.output.status, ['uint32'], this.id);
      validateGraphOutputsDisjointFromInputs(
        this.id,
        [...Object.values(outputViews), ...Object.values(props.spatial.output)],
        [
          props.predictors,
          props.responses,
          props.spatial.weights.offsets,
          props.spatial.weights.neighbors,
          props.spatial.weights.weights,
          props.spatial.weights.status?.overflow,
          props.spatial.weights.status?.candidateOverflow
        ]
      );
    }
    validateCapacity(this.id, 'equationCount', equations, 2, GPU_SUR_MAXIMUM_EQUATION_COUNT);
    validateCapacity(this.id, 'predictorCount', columns, 1, GPU_SUR_MAXIMUM_PREDICTOR_COUNT);
    validateCapacity(
      this.id,
      'system coefficients',
      size,
      1,
      GPU_SUR_MAXIMUM_SYSTEM_COEFFICIENT_COUNT
    );
    validateCapacity(
      this.id,
      'rowCount',
      rows,
      1,
      props.spatial ? GPU_SPATIAL_SUR_MAXIMUM_ROW_COUNT : GPU_SUR_MAXIMUM_ROW_COUNT
    );
    for (const [view, format, length] of [
      [props.predictors, 'float32', equations * rows * columns],
      [props.responses, 'float32', equations * rows],
      [props.mask, 'uint32', rows],
      [props.initialResiduals, 'float32', rows * equations],
      [props.parameters, 'float32', 1],
      [props.output.coefficients, 'float32', size],
      [props.output.covariance, 'float32', size * size],
      [props.output.residualCovariance, 'float32', equations * equations],
      [props.output.residuals, 'float32', equations * rows],
      [props.output.status, 'uint32', 1],
      [props.output.iteration?.status.iterationCount, 'uint32', 1]
    ] as const)
      if (view) {
        validateSinglePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
      }
    validateRegressionOutputDispatches(
      this.id,
      {
        ...outputViews,
        parameters: props.parameters,
        spatialParameters: props.spatial?.output.parameters,
        spatialStatus: props.spatial?.output.status
      },
      props.spatial
        ? [
            {read: ['spatialParameters'], write: ['coefficients', 'covariance']},
            {read: ['spatialParameters'], write: ['residuals']},
            ['residualCovariance', 'status', 'iterationCount'],
            {read: ['parameters'], write: ['residualCovariance', 'status']},
            ['spatialParameters', 'spatialStatus']
          ]
        : [
            ['coefficients', 'covariance', 'residualCovariance', 'status', 'iterationCount'],
            {
              read: ['parameters'],
              write: ['coefficients', 'covariance', 'residualCovariance', 'status']
            },
            {read: ['status'], write: ['residuals']}
          ]
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(outputViews), [
      props.predictors,
      props.responses,
      props.mask,
      props.initialResiduals,
      props.parameters
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>,
    /** Original predictor origins and fixed GM parameters for composed spatial stopping. */
    convergenceCoordinates?: {
      predictors: GraphDataView<'float32'>;
      parameters: GraphDataView<'float32'>;
    }
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {iteration, ...outputs} = props.output;
    const outputViews = {...outputs, iterationCount: iteration?.status.iterationCount};
    if (convergenceCoordinates) {
      if (props.iteration)
        validateCommandNodeBudget(
          id,
          this.estimatedCommandNodeCount + 1 + 2 * props.iteration.maximumIterations
        );
      validateSinglePackedView(convergenceCoordinates.predictors, ['float32'], id);
      validateSinglePackedView(convergenceCoordinates.parameters, ['float32'], id);
      validateRequirement(
        id,
        convergenceCoordinates.predictors.length ===
          props.equationCount * props.rowCount * props.predictorCount,
        `convergenceCoordinates.predictors.length must equal ${props.equationCount * props.rowCount * props.predictorCount}; received ${convergenceCoordinates.predictors.length}`
      );
      validateRequirement(
        id,
        convergenceCoordinates.parameters.length === props.equationCount,
        `convergenceCoordinates.parameters.length must equal equationCount ${props.equationCount}; received ${convergenceCoordinates.parameters.length}`
      );
      validateRequirement(
        id,
        !props.spatial,
        'convergenceCoordinates requires a composed filtered system without spatial'
      );
      validateGraphOutputsDisjointFromInputs(id, Object.values(outputViews), [
        convergenceCoordinates.predictors,
        convergenceCoordinates.parameters
      ]);
    }
    validateGraphChunkBindingRanges(graph, id, {
      convergencePredictors: convergenceCoordinates?.predictors,
      convergenceParameters: convergenceCoordinates?.parameters,
      predictors: props.predictors,
      responses: props.responses,
      mask: props.mask,
      initialResiduals: props.initialResiduals,
      parameters: props.parameters,
      ...outputViews,
      spatialParameters: props.spatial?.output.parameters,
      spatialStatus: props.spatial?.output.status,
      offsets: props.spatial?.weights.offsets,
      neighbors: props.spatial?.weights.neighbors,
      weights: props.spatial?.weights.weights,
      overflow: props.spatial?.weights.status?.overflow,
      candidateOverflow: props.spatial?.weights.status?.candidateOverflow
    });
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'statistics scratch',
      getOrdinarySURScratchLength(props.equationCount, props.predictorCount, props.rowCount) * 4,
      'reduce rowCount or the system coefficient count'
    );
    if (props.spatial) {
      validateRegressionAllocations(
        graph.device,
        id,
        getSpatialSURAllocationLengths(props.rowCount, props.equationCount, props.predictorCount)
      );
      return getSpatialSURNodes(graph, this);
    }
    return getOrdinarySURNodes(graph, this, convergenceCoordinates);
  }
}
