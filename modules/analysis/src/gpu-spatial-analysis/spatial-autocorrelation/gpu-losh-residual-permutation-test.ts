// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {snapshotSpatialAutocorrelationProps} from './spatial-autocorrelation-construction';

import {validateDisjointWritableViews} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateSpatialStatisticsBindings} from '../spatial-weights/spatial-statistics-bindings';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {validateGraphOutputsDisjointFromInputs} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getGPUSpatialWeightsOrChunksViews} from '../spatial-weights/index';
import {
  GPULocalPermutationTest,
  type GPULocalPermutationTestProps,
  type GPULocalPermutationTestOutput
} from '../permutation-inference/index';
import {
  GPULocalSpatialHeteroscedasticity,
  type GPULocalSpatialHeteroscedasticityProps
} from './gpu-local-variability';

/** Conditional randomization of the observed LOSH residual field, with an explicit tail. */
export type GPULOSHResidualPermutationTestProps = Pick<
  GPULocalPermutationTestProps,
  'parameters' | 'maximumPermutations' | 'maximumNeighbors' | 'multipleTesting'
> &
  Omit<GPULocalSpatialHeteroscedasticityProps, 'output'> & {
    /** Required tail; legacy data-selected directed tails are deliberately excluded. */
    alternative: 'greater' | 'lesser' | 'two-sided';
    output: GPULocalSpatialHeteroscedasticityProps['output'] & {
      inference: Pick<
        GPULocalPermutationTestOutput,
        'exceedances' | 'pseudoPValues' | 'adjustedPValues' | 'significant' | 'status'
      >;
    };
  };

/**
 * Computes descriptive LOSH and tests its fixed normalized residual field by conditional
 * sampling without replacement. The focus residual stays fixed; each draw assigns distinct
 * other residual labels to the focus row's active neighbor slots. Residuals/local means are
 * computed once, never refitted within permutations. This is not esda LOSH inference or a
 * raw-observation permutation test. All stages execute on the GPU without readback.
 */
export class GPULOSHResidualPermutationTest implements GPUCommandNodeProducer {
  readonly id: string;

  constructor(readonly props: GPULOSHResidualPermutationTestProps) {
    props = snapshotSpatialAutocorrelationProps(props);
    this.props = props;
    this.id = props.id ?? 'losh-residual-permutation-test';
    if (!['greater', 'lesser', 'two-sided'].includes(props.alternative)) {
      throw new Error(`${this.id}: an explicit greater, lesser or two-sided tail is required`);
    }
    new GPULocalSpatialHeteroscedasticity(props);
    // Validate the inference shape/settings now, before its residual scratch view exists. The
    // residual field has exactly the source values' packed float32 shape; no contents are read.
    new GPULocalPermutationTest({
      id: `${this.id}-conditional`,
      weights: props.weights,
      values: props.values,
      parameters: props.parameters,
      maximumPermutations: props.maximumPermutations,
      maximumNeighbors: props.maximumNeighbors,
      multipleTesting: props.multipleTesting,
      alternative: props.alternative,
      islands: 'keep',
      statistic: 'localG',
      output: props.output.inference
    });
    validateDisjointWritableViews(this.id, {
      statistic: props.output.statistic,
      status: props.output.status,
      normalizedResiduals: props.output.normalizedResiduals,
      residualCount: props.output.residualCount,
      exceedances: props.output.inference.exceedances,
      pseudoPValues: props.output.inference.pseudoPValues,
      adjustedPValues: props.output.inference.adjustedPValues,
      significant: props.output.inference.significant,
      ...props.output.inference.status
    });
    const outputs = [
      props.output.statistic,
      props.output.status,
      props.output.normalizedResiduals,
      props.output.residualCount,
      props.output.inference.exceedances,
      props.output.inference.pseudoPValues,
      props.output.inference.adjustedPValues,
      props.output.inference.significant,
      ...Object.values(props.output.inference.status)
    ].filter(view => view !== undefined);
    for (let index = 0; index < outputs.length; index++) {
      validateGraphOutputsDisjointFromInputs(this.id, [outputs[index]], outputs.slice(index + 1));
    }
    validateGraphOutputsDisjointFromInputs(
      this.id,
      outputs,
      [
        props.values,
        props.parameters,
        typeof props.residualPower === 'object' ? props.residualPower : undefined,
        ...getGPUSpatialWeightsOrChunksViews(props.weights),
        props.mask,
        props.weights.status?.overflow
      ].filter(view => view !== undefined)
    );
  }

  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    validateSpatialStatisticsBindings(graph, this.id, this.props, {
      'row scratch': this.props.values.length * 4,
      'permutation sums': Math.ceil(this.props.values.length / 4096) * 8 * 4
    });
    const {props, id} = this;
    const normalizedResiduals =
      props.output.normalizedResiduals ??
      createTransientView(graph, `${id}-residuals`, 'float32', props.values.length);
    const descriptive = new GPULocalSpatialHeteroscedasticity({
      ...props,
      id: `${id}-descriptive`,
      output: {...props.output, normalizedResiduals}
    });
    const inference = new GPULocalPermutationTest({
      id: `${id}-conditional`,
      weights: props.weights,
      values: normalizedResiduals,
      parameters: props.parameters,
      maximumPermutations: props.maximumPermutations,
      maximumNeighbors: props.maximumNeighbors,
      multipleTesting: props.multipleTesting,
      alternative: props.alternative,
      // Retain every finite residual as a donor, including second-stage islands.
      islands: 'keep',
      statistic: 'localG',
      output: props.output.inference
    });
    return [...descriptive.getCommandNodes(graph), ...inference.getCommandNodes(graph)];
  }
}
