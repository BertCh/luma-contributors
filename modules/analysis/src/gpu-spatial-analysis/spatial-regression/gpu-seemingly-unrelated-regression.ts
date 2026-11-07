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
  validatePackedView,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {getSpatialSURNodes} from './spatial-sur-nodes';
import {getOrdinarySURNodes} from './ordinary-sur-nodes';

export type GPUSeeminglyUnrelatedRegressionProps = {
  id?: string;
  equationCount: number;
  predictorCount: number;
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
      validatePackedView(props.spatial.output.parameters, ['float32'], this.id);
      validatePackedView(props.spatial.output.status, ['uint32'], this.id);
      validateGraphOutputsDisjointFromInputs(
        this.id,
        [...Object.values(props.output), ...Object.values(props.spatial.output)],
        [
          props.predictors,
          props.responses,
          props.spatial.weights.offsets,
          props.spatial.weights.neighbors,
          props.spatial.weights.weights
        ]
      );
    }
    validateCapacity(this.id, 'equationCount', equations, 2, 4);
    validateCapacity(this.id, 'predictorCount', columns, 1, 7);
    validateCapacity(this.id, 'system coefficients', size, 1, 16);
    validateCapacity(this.id, 'rowCount', rows, 1, 65536);
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
        validatePackedView(view, [format], this.id);
        if (view.length !== length) throw new Error(`${this.id}: shape mismatch`);
      }
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
    if (props.spatial) return getSpatialSURNodes(graph, this);
    validateGraphViewsBelongToGraph(id, graph, [
      props.predictors,
      props.responses,
      props.mask,
      props.initialResiduals,
      ...Object.values(props.output)
    ]);
    return getOrdinarySURNodes(graph, this);
  }
}
