// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateRequirement,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {CRESSIE_ROUND_COUNT, getParallelVariogramModelFitNodes} from './variogram-model-fit-nodes';

import type {VariogramModelType, VariogramModelWeighting} from './variogram-model';
import {validateVariogramModelFitChoices} from './variogram-model-fit-requirements';

/** The ladder is evaluated per model and retained in one bounded search scratch table. */
const MAXIMUM_RANGE_STEPS = 1024;
/** Golden-section refinement remains ordered within each cooperative model workgroup. */
const MAXIMUM_ITERATIONS = 64;
/** Five supported covariance shapes bound the model workgroups and output records. */
const MAXIMUM_MODEL_COUNT = 5;
/** The 64-lane cooperative bin stride must terminate without wrapping a uint32 index. */
const MAXIMUM_BIN_COUNT = 0xffffffc0;

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
 * on GPU. Cressie repeats the search three times after the initial pair-weighted fit; its weights
 * `N / max(fitted^2, 1e-20)` are divided by their total before the normal equations, in the
 * ladder and in the refinement, so the fit scales with the semivariances (fitted values down to
 * 1e-10, where the 1e-20 floor takes over). Like the
 * CPU helper, the search domain is [minimum positive lag / 10, maximum lag * 3]. A boundary optimum
 * is returned with status 3 rather than claimed to be an identified range. Nonpositive-count,
 * nonfinite, nonpositive-distance bins are skipped. Range candidates run in parallel. Each model's
 * refinement reduces bins cooperatively, while refinement steps and Cressie rounds remain ordered
 * on the GPU. Round flags live in aligned parameter slots; every round uses the same candidate
 * and refinement pipelines. Floating reductions preserve the fitting tolerance, not serial bit identity. Only
 * public diagnostics need readback. Compare model AIC only for the same empirical bins, and note
 * it is an unweighted Gaussian residual diagnostic, not a spatial likelihood.
 *
 * Status 2 (numerical failure, NaN outputs) is reported for every model when the objective
 * cannot be evaluated in float32: when no candidate has a finite weighted residual, and, decided
 * before the search, when the float32 resolution of the largest semivariance (`2^-23` of it),
 * squared and summed with the pair weights, is not finite. That is semivariances above about
 * `1.5e26 / sqrt(sum of weights)`; below it the fit of a column of constant semivariances would
 * otherwise be 0 or infinity by rounding alone.
 */
export class GPUVariogramModelFit implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUVariogramModelFitProps;
  readonly models: readonly VariogramModelType[];
  readonly rangeSteps: number;
  readonly maximumIterations: number;
  /**
   * Includes two dispatches per fitting round. Preparation adds an upload and an ordered scan
   * through 1,024 bins, or two dispatches for tile reduction and merge above 1,024 bins.
   */
  readonly estimatedCommandNodeCount: number;

  constructor(props: GPUVariogramModelFitProps) {
    const snapshot = snapshotConstructionFields(props, [
      'id',
      'distances',
      'semivariances',
      'pairCounts',
      'models',
      'weighting',
      'rangeSteps',
      'maximumIterations',
      'output'
    ]);
    props = Object.freeze({
      ...snapshot,
      models: Object.freeze([...(snapshot.models ?? ['spherical', 'exponential', 'gaussian'])]),
      output: snapshotConstructionFields(snapshot.output, ['parameters', 'diagnostics', 'status'])
    });
    this.props = props;
    this.id = props.id ?? 'variogram-model-fit';
    this.models = props.models ?? ['spherical', 'exponential', 'gaussian'];
    this.rangeSteps = props.rangeSteps ?? 200;
    this.maximumIterations = props.maximumIterations ?? 32;
    validateCapacity(this.id, 'distances length', props.distances.length, 0, MAXIMUM_BIN_COUNT);
    validateCapacity(this.id, 'rangeSteps', this.rangeSteps, 3, MAXIMUM_RANGE_STEPS);
    validateCapacity(this.id, 'maximumIterations', this.maximumIterations, 0, MAXIMUM_ITERATIONS);
    validateCapacity(this.id, 'models length', this.models.length, 1, MAXIMUM_MODEL_COUNT);
    validateVariogramModelFitChoices(this.id, this.models, props.weighting ?? 'cressie');
    for (const [name, view] of [
      ['semivariances', props.semivariances],
      ['pairCounts', props.pairCounts]
    ] as const) {
      validateRequirement(
        this.id,
        view.length === props.distances.length,
        `invalid model or bin topology: ${name} bin count ${view.length} must equal distances bin count ${props.distances.length}`
      );
    }
    for (const [name, view, format] of [
      ['distances', props.distances, 'float32'],
      ['semivariances', props.semivariances, 'float32'],
      ['pairCounts', props.pairCounts, 'uint32'],
      ['output.parameters', props.output.parameters, 'float32x4'],
      ['output.diagnostics', props.output.diagnostics, 'float32x4'],
      ['output.status', props.output.status, 'uint32']
    ] as const) {
      validateSinglePackedView(view, [format], `${this.id} ${name}`);
    }
    for (const [name, view] of Object.entries(props.output)) {
      validateRequirement(
        this.id,
        view.length >= this.models.length,
        `output.${name} length must be at least models length (${this.models.length})`
      );
    }
    this.estimatedCommandNodeCount =
      2 + 2 * ((props.weighting ?? 'cressie') === 'cressie' ? CRESSIE_ROUND_COUNT : 1);
    validateCommandNodeBudget(this.id, this.estimatedCommandNodeCount);
    validateDisjointWritableViews(this.id, props.output);
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
