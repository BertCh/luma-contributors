// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
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

/** Tiled counts are stored as exact float32 integers, up to 2^24 observations. */
export const GPU_SUR_MAXIMUM_ROW_COUNT = REGRESSION_TILE_MAXIMUM_ROW_COUNT;
/** Nested OLS and GM row counts remain exact float32 integers; bindings are audited separately. */
export const GPU_SPATIAL_SUR_MAXIMUM_ROW_COUNT = REGRESSION_TILE_MAXIMUM_ROW_COUNT;
/** Private dense error covariance and system solves are bounded to four equations. */
const MAXIMUM_EQUATION_COUNT = 4;
/** Each equation uses the existing eight-column (including intercept) dense fit. */
const MAXIMUM_PREDICTOR_COUNT = 7;
/** The joint GLS Cholesky is limited to sixteen coefficients. */
const MAXIMUM_SYSTEM_COEFFICIENT_COUNT = 16;

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
  /** Optional equation-major first-stage residuals for the joint covariance. */
  initialResiduals?: GraphDataView<'float32'>;
  /** Kelejian-Prucha spatial-error SUR (spreg SURerrorGM) on complete aligned observations.
   * Requires finite, nonnegative, row-standardized zero-diagonal weights without islands.
   * Per-equation error parameters are bounded to [-0.99,0.99]; masks are not accepted in this mode.
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
    /** Full system coefficient covariance in the same order. */
    covariance: GraphDataView<'float32'>;
    /** OLS residual cross-equation covariance used for one-step GLS (divisor n). */
    residualCovariance: GraphDataView<'float32'>;
    residuals: GraphDataView<'float32'>;
    /** 0 successful FGLS; 1 insufficient rows; 2 singular design; 3 singular error covariance. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Nonspatial one-step feasible SUR: equation OLS -> JOINT residual covariance -> system GLS.
 * Observations must share IDs and order. Complete-case selection is common across equations.
 * Row-tile reductions center predictors and responses before the small norm-scaled solves.
 * A common error-covariance scale is restored after solving the normalized GLS system. Coefficient covariance
 * is the conditional GLS information inverse using the first-stage residual covariance; this is
 * not spatial SUR, iterated SUR, small-sample inference or an equation-independent error model.
 */
export class GPUSeeminglyUnrelatedRegression implements GPUCommandNodeProducer {
  readonly id: string;
  constructor(readonly props: GPUSeeminglyUnrelatedRegressionProps) {
    this.id = props.id ?? 'seemingly-unrelated-regression';
    const equations = props.equationCount;
    const columns = props.predictorCount;
    const rows = props.rowCount;
    const size = equations * (columns + 1);
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
        [...Object.values(props.output), ...Object.values(props.spatial.output)],
        [
          props.predictors,
          props.responses,
          props.spatial.weights.offsets,
          props.spatial.weights.neighbors,
          props.spatial.weights.weights,
          props.spatial.weights.status?.overflow
        ]
      );
    }
    validateCapacity(this.id, 'equationCount', equations, 2, MAXIMUM_EQUATION_COUNT);
    validateCapacity(this.id, 'predictorCount', columns, 1, MAXIMUM_PREDICTOR_COUNT);
    validateCapacity(this.id, 'system coefficients', size, 1, MAXIMUM_SYSTEM_COEFFICIENT_COUNT);
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
      [props.output.coefficients, 'float32', size],
      [props.output.covariance, 'float32', size * size],
      [props.output.residualCovariance, 'float32', equations * equations],
      [props.output.residuals, 'float32', equations * rows],
      [props.output.status, 'uint32', 1]
    ] as const)
      if (view) {
        validateSinglePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
      }
    validateRegressionOutputDispatches(
      this.id,
      {
        ...props.output,
        spatialParameters: props.spatial?.output.parameters,
        spatialStatus: props.spatial?.output.status
      },
      props.spatial
        ? [
            {read: ['spatialParameters'], write: ['coefficients', 'covariance']},
            {read: ['coefficients'], write: ['residuals']},
            ['residualCovariance', 'status'],
            ['spatialParameters', 'spatialStatus']
          ]
        : [
            ['coefficients', 'covariance', 'residualCovariance', 'status'],
            {read: ['status'], write: ['residuals']}
          ]
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.predictors,
      props.responses,
      props.mask,
      props.initialResiduals
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphChunkBindingRanges(graph, id, {
      predictors: props.predictors,
      responses: props.responses,
      mask: props.mask,
      initialResiduals: props.initialResiduals,
      ...props.output,
      spatialParameters: props.spatial?.output.parameters,
      spatialStatus: props.spatial?.output.status,
      offsets: props.spatial?.weights.offsets,
      neighbors: props.spatial?.weights.neighbors,
      weights: props.spatial?.weights.weights,
      overflow: props.spatial?.weights.status?.overflow
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
    return getOrdinarySURNodes(graph, this);
  }
}
