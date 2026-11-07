// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validatePackedView
} from '../../utils/gpu-contributor-utils';
import {getParallelVariogramModelFitNodes} from './variogram-model-fit-nodes';
import type {VariogramModelType, VariogramModelWeighting} from './variogram-model';

/** GPU-resident empirical-bin model search; output row order follows models. */
export type GPUVariogramModelFitProps = {
  id?: string;
  distances: GraphDataView<'float32'>;
  semivariances: GraphDataView<'float32'>;
  pairCounts: GraphDataView<'uint32'>;
  models?: readonly VariogramModelType[];
  weighting?: VariogramModelWeighting;
  rangeSteps?: number;
  maximumIterations?: number;
  output: {
    /** [nugget, partial sill, effective range, weighted residual]. */
    parameters: GraphDataView<'float32x4'>;
    /** [usable bins, unweighted squared residual, AIC (3 parameters), relative log-range bracket]. */
    diagnostics: GraphDataView<'float32x4'>;
    /** 0 converged interior solution; 1 insufficient bins; 2 numerical failure; 3 search boundary/budget. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Constrained nugget/sill fitting over a log-range ladder and golden-section refinement, entirely
 * on GPU. Cressie repeats the search three times after the initial pair-weighted fit. Like the
 * CPU helper, the search domain is [minimum positive lag / 10, maximum lag * 3]. A boundary optimum
 * is returned with status 3 rather than claimed to be an identified range. Nonpositive-count,
 * nonfinite, nonpositive-distance bins are skipped. Range candidates run in parallel. Each model's
 * refinement reduces bins cooperatively, while refinement steps and Cressie rounds remain ordered
 * on the GPU. Floating reductions preserve the fitting tolerance, not serial bit identity. Only
 * public diagnostics need readback. Compare model AIC only for the same empirical bins, and note
 * it is an unweighted Gaussian residual diagnostic, not a spatial likelihood.
 */
export class GPUVariogramModelFit implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUVariogramModelFitProps;
  readonly models: readonly VariogramModelType[];
  readonly rangeSteps: number;
  readonly maximumIterations: number;

  constructor(props: GPUVariogramModelFitProps) {
    this.props = props;
    this.id = props.id ?? 'variogram-model-fit';
    this.models = props.models ?? ['spherical', 'exponential', 'gaussian'];
    this.rangeSteps = props.rangeSteps ?? 200;
    this.maximumIterations = props.maximumIterations ?? 32;
    validateCapacity(this.id, 'rangeSteps', this.rangeSteps, 3);
    validateCapacity(this.id, 'maximumIterations', this.maximumIterations, 0);
    if (
      !this.models.length ||
      this.models.length > 3 ||
      this.models.some(model => !['spherical', 'exponential', 'gaussian'].includes(model)) ||
      !['pairs', 'none', 'cressie'].includes(props.weighting ?? 'cressie') ||
      this.rangeSteps > 1024 ||
      this.maximumIterations > 64 ||
      props.distances.length !== props.semivariances.length ||
      props.distances.length !== props.pairCounts.length
    ) {
      throw new Error(`${this.id} invalid model or bin topology`);
    }
    for (const view of [
      props.distances,
      props.semivariances,
      props.pairCounts,
      props.output.parameters,
      props.output.diagnostics,
      props.output.status
    ]) {
      validatePackedView(view, [view.format], this.id);
    }
    if (
      [props.output.parameters, props.output.diagnostics, props.output.status].some(
        view => view.length < this.models.length
      ) ||
      new Set([
        props.output.parameters.buffer,
        props.output.diagnostics.buffer,
        props.output.status.buffer
      ]).size !== 3
    ) {
      throw new Error(`${this.id} invalid output topology`);
    }
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.distances,
      props.semivariances,
      props.pairCounts
    ]);
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.distances,
      props.semivariances,
      props.pairCounts,
      ...Object.values(props.output)
    ]);
    return getParallelVariogramModelFitNodes(graph, this);
  }
}
