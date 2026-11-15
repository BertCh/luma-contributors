// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {
  GPUIterationStatusPort,
  GPUInvalidStatusPort
} from '../../analysis-core/contracts/status';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateRequirement,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_COREGIONALIZATION_MAXIMUM_BIN_COUNT,
  validateCoregionalizationFitOptions,
  type CoregionalizationFitOptions
} from './coregionalization-parameters';
import {getCoregionalizationFitNodes} from './coregionalization-fit-nodes';

/** Named scalar diagnostics; validity is zero for unusable bins or numerical failure. */
export type GPUCoregionalizationFitStatus = Required<
  GPUIterationStatusPort & GPUInvalidStatusPort
> & {
  validity: GraphDataView<'uint32'>;
};
/** Fixed output: structure-major, row-major square matrices and square loading factors. */
export type GPUCoregionalizationFitOutput = {
  matrices: GraphDataView<'float32'>;
  /** L with B = L L^T; pass each structure's square factor to local kriging's supplied LMC. */
  loadings: GraphDataView<'float32'>;
  /** [pair-weighted Frobenius squared residual / total weight, relative change, usable bins, total weight]. */
  diagnostics: GraphDataView<'float32x4'>;
  status: GPUCoregionalizationFitStatus;
};
/** GPU-resident complete direct/cross variogram matrices on common lag bins. */
export type GPUCoregionalizationFitProps = CoregionalizationFitOptions & {
  id?: string;
  distances: GraphDataView<'float32'>;
  /** Bin-major upper triangle, rows (0,0),(0,1),...,(1,1),...; signed cross values retained. */
  semivariances: GraphDataView<'float32'>;
  /** One shared pair weight per bin, not per variogram; zero skips the complete lag matrix. */
  pairCounts: GraphDataView<'uint32'>;
  /** Written with getGPUCoregionalizationFitParameterValues. */
  parameters: GraphDataView<'float32'>;
  output: GPUCoregionalizationFitOutput;
};

/**
 * Goulard–Voltz block weighted least squares with a PSD projection after each structure update.
 * Reference tier: one invocation forms sufficient statistics from at most 256 lag matrices;
 * each gated sweep solves at most four symmetric systems of order at most four. Prefer
 * CPUCoregionalizationFitOperation for these small fits when the bins are on the CPU.
 *
 * Compile-time: variableCount, structureCount, maximumIterations (default 128), view shapes.
 * Per-frame: all view contents, including the covariance structures and tolerance in parameters.
 * No coordinates are read. Nonpositive/nonfinite lags, zero pair counts, or a nonfinite entry
 * skip the complete bin. A zero-energy structure or invalid parameters gives validity 0 and NaN
 * outputs. Singular/duplicate structures need not have unique coefficients; the deterministic
 * zero start still returns a PSD fit. A budget exit publishes the current fit with validity 1,
 * converged 0 and iterationLimitReached 1. A zero budget returns the zero model.
 *
 * Reference: Goulard & Voltz (1992), doi:10.1007/BF00893750.
 */
export class GPUCoregionalizationFit implements GPUCommandNodeProducer {
  readonly id: string;
  readonly maximumIterations: number;
  readonly estimatedCommandNodeCount: number;

  constructor(readonly props: GPUCoregionalizationFitProps) {
    this.id = props.id ?? 'coregionalization-fit';
    this.maximumIterations = props.maximumIterations ?? 128;
    validateCoregionalizationFitOptions(this.id, props, true);
    validateCapacity(
      this.id,
      'distances.length',
      props.distances.length,
      0,
      GPU_COREGIONALIZATION_MAXIMUM_BIN_COUNT
    );
    const triangleCount = (props.variableCount * (props.variableCount + 1)) / 2;
    const matrixLength = props.structureCount * props.variableCount ** 2;
    for (const [name, view] of Object.entries({
      distances: props.distances,
      semivariances: props.semivariances,
      pairCounts: props.pairCounts,
      parameters: props.parameters,
      matrices: props.output.matrices,
      loadings: props.output.loadings,
      diagnostics: props.output.diagnostics,
      ...props.output.status
    })) {
      validateSinglePackedView(
        view,
        [
          name === 'diagnostics'
            ? 'float32x4'
            : name === 'pairCounts' || name in props.output.status
              ? 'uint32'
              : 'float32'
        ],
        `${this.id} ${name}`
      );
    }
    validateRequirement(
      this.id,
      props.pairCounts.length === props.distances.length,
      'pairCounts.length must equal distances.length'
    );
    validateRequirement(
      this.id,
      props.semivariances.length === props.distances.length * triangleCount,
      'semivariances.length must equal distances.length * variableCount * (variableCount + 1) / 2'
    );
    validateRequirement(
      this.id,
      props.parameters.length >= 1 + 4 * props.structureCount,
      'parameters.length must be at least 1 + 4 * structureCount'
    );
    validateRequirement(
      this.id,
      props.output.matrices.length >= matrixLength && props.output.loadings.length >= matrixLength,
      'output matrices and loadings lengths must be at least structureCount * variableCount^2'
    );
    validateRequirement(
      this.id,
      props.output.diagnostics.length >= 1 &&
        Object.values(props.output.status).every(view => view.length >= 1),
      'output diagnostics and status views must contain at least one row'
    );
    const outputs = [
      props.output.matrices,
      props.output.loadings,
      props.output.diagnostics,
      ...Object.values(props.output.status)
    ];
    validateRequirement(
      this.id,
      new Set(outputs.map(view => view.buffer)).size === outputs.length,
      'output views must use distinct buffers'
    );
    validateGraphOutputsDisjointFromInputs(this.id, outputs, [
      props.distances,
      props.semivariances,
      props.pairCounts,
      props.parameters
    ]);
    this.estimatedCommandNodeCount = 2 * this.maximumIterations + 3;
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {props} = this;
    validateGraphViewsBelongToGraph(this.id, graph, [
      props.distances,
      props.semivariances,
      props.pairCounts,
      props.parameters,
      props.output.matrices,
      props.output.loadings,
      props.output.diagnostics,
      ...Object.values(props.output.status)
    ]);
    return getCoregionalizationFitNodes(graph, this);
  }
}
