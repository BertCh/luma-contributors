// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Properties for {@link GPUGaussianPredictiveScores}. */
export type GPUGaussianPredictiveScoresProps = {
  /** Prefix for generated node IDs. Defaults to `'gaussian-predictive-scores'`. */
  id?: string;
  /** Predictive mean per site, for example the kriging prediction. */
  means: GraphDataView<'float32'>;
  /**
   * Predictive variance per site in squared value units, for example the kriging variance. Zero
   * is a point forecast and scores `|y - mean|`; negative or NaN variance scores NaN.
   */
  variances: GraphDataView<'float32'>;
  /** Observed value per site. */
  observations: GraphDataView<'float32'>;
  /** Caller-owned outputs, one row per site (the length of `means`). */
  output: {
    /** Closed-form CRPS of the normal predictive distribution, `crps_norm(y, mean, sd)`. */
    crps: GraphDataView<'float32'>;
  };
};

/**
 * Closed-form continuous ranked probability score of a Gaussian predictive distribution, for
 * kriging predictions `N(mean, variance)` (scoringRules 1.1.3 `crps_norm`, Gneiting et al. 2005):
 * with `z = (y - mean) / sd`, `CRPS = sd * (z * (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi))`.
 *
 * `2 Phi(z) - 1 = erf(z / sqrt(2))` is evaluated as the all-positive series
 * `erf(x) = 2 / sqrt(pi) * exp(-x^2) * sum_n 2^n x^(2n+1) / (2n + 1)!!` for `|x| <= 4` (no
 * cancellation in f32) and 1 beyond, which truncates `erf` by less than `1.6e-8`. One thread per
 * site; there are no reductions and no atomics. Lower is better.
 */
export class GPUGaussianPredictiveScores implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUGaussianPredictiveScoresProps;

  constructor(props: GPUGaussianPredictiveScoresProps) {
    this.id = props.id ?? 'gaussian-predictive-scores';
    this.props = props;
    const {id} = this;
    for (const [name, view] of [
      ['means', props.means],
      ['variances', props.variances],
      ['observations', props.observations],
      ['output.crps', props.output.crps]
    ] as const) {
      validateSinglePackedView(view, ['float32'], `${id} ${name}`);
    }
    const count = props.means.length;
    if (
      count < 1 ||
      props.variances.length < count ||
      props.observations.length < count ||
      props.output.crps.length < count
    ) {
      throw new Error(`${id} variances, observations and output.crps need one row per mean`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [props.output.crps],
      [props.means, props.variances, props.observations]
    );
  }

  /** Returns the scoring kernel. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.means,
      props.variances,
      props.observations,
      props.output.crps
    ]);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-crps`,
        operation: 'GPUGaussianPredictiveScores',
        variant: 'crps-normal',
        bindings: [
          {name: 'means', view: props.means, type: 'f32', access: 'read'},
          {name: 'variances', view: props.variances, type: 'f32', access: 'read'},
          {name: 'observations', view: props.observations, type: 'f32', access: 'read'},
          {name: 'crps', view: props.output.crps, type: 'f32', access: 'read_write'}
        ],
        invocationCount: props.means.length,
        declarations: `const INVERSE_SQRT_PI: f32 = 0.5641895835;
const SQRT_TWO_OVER_PI: f32 = 0.7978845608;
// erf(x) for x >= 0: the positive series below 4, 1 above.
fn errorFunction(x: f32) -> f32 {
  if (x > 4.0) { return 1.0; }
  var term = x;
  var sum = x;
  let doubledSquare = 2.0 * x * x;
  for (var n = 1u; n < 160u; n++) {
    term = term * doubledSquare / f32(2u * n + 1u);
    sum += term;
    if (term < 1.0e-9 * sum) { break; }
  }
  return 1.1283791671 * exp(-x * x) * sum;
}`,
        body: `let variance = variances[variancesOffset + index];
  let error = observations[observationsOffset + index] - means[meansOffset + index];
  var score = bitcast<f32>(0x7fc00000u | (index & 0u));
  if (variance == 0.0) {
    score = abs(error);
  } else if (variance > 0.0) {
    let deviation = sqrt(variance);
    let z = abs(error) / deviation;
    score = deviation * (z * errorFunction(z * 0.7071067812) + SQRT_TWO_OVER_PI * exp(-0.5 * z * z) - INVERSE_SQRT_PI);
  }
  crps[crpsOffset + index] = score;`
      })
    ];
  }
}
