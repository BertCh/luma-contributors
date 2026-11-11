// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUReduction,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_NAN_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';

const OPERATION = 'GPUEnsembleScores';
/** Lanes of the per-site CRPS workgroup. */
const CRPS_LANES = 64;

/** Properties for {@link GPUEnsembleScores}. */
export type GPUEnsembleScoresProps = {
  /** Prefix for generated node IDs. Defaults to `'ensemble-scores'`. */
  id?: string;
  /**
   * Ensemble members, member-major: row `member * dimension + site` is the value of `member` at
   * `site`. This is the layout of `GPUConditionalGaussianSimulation` `output.values` (replicate
   * major), so its output can be scored without a copy. At least `memberCount * dimension` rows.
   */
  ensemble: GraphDataView<'float32'>;
  /** Observed value per site, at least `dimension` rows. */
  observations: GraphDataView<'float32'>;
  /** Number of ensemble members `m`. */
  memberCount: number;
  /** Number of sites `d` of one field. */
  dimension: number;
  /**
   * Fair variant of the sample CRPS and the energy score. scoringRules 1.1.3 defines only the
   * non-fair (empirical-distribution) forms, `crps_sample(method = "edf")` and `es_sample`, which
   * divide the member-pair term by `2 m^2`; they are the default here (`fair` false). With `fair`
   * true the pair term is divided by `2 m (m - 1)` (Ferro 2014; Zamo and Naveau 2018), which makes
   * the score unbiased for the CRPS of the member distribution. The fair variant is this package's
   * extension and has no scoringRules oracle. It needs `memberCount >= 2`. The variogram score
   * has no fair variant.
   */
  fair?: boolean;
  /**
   * Order `p` of the variogram score, the `p` of `vs_sample(p = ...)`. Defaults to 0.5 as in the
   * scoringRules vignette. Compile-time. Must be positive and finite.
   */
  variogramOrder?: number;
  /**
   * Ordered site pairs `[i, j]` of the variogram score. Every listed pair contributes one term, so
   * listing both `[i, j]` and `[j, i]` for all pairs reproduces `vs_sample` with its default
   * weights (it sums the full symmetric matrix), while a list of neighbouring pairs restricts the
   * score to them. Required with `output.variogramScore`.
   */
  variogramPairs?: GraphDataView<'uint32x2'>;
  /** Optional nonnegative weight per listed pair (`w_vs` entries); defaults to 1. */
  variogramWeights?: GraphDataView<'float32'>;
  /** Caller-owned outputs. At least one is required. */
  output: {
    /** Sample CRPS per site, `dimension` rows. */
    crps?: GraphDataView<'float32'>;
    /** Energy score of the whole field (Euclidean norm over the sites), one row. */
    energyScore?: GraphDataView<'float32'>;
    /** Variogram score over `variogramPairs`, one row. */
    variogramScore?: GraphDataView<'float32'>;
  };
};

/**
 * Proper scoring rules of an ensemble forecast (for example conditional simulations) against an
 * observed field: the continuous ranked probability score per site, the energy score and the
 * variogram score of the whole field.
 *
 * Definitions follow scoringRules 1.1.3 with members `x_1 .. x_m` and observation `y`:
 * - CRPS, `crps_sample(method = "edf")`: `(1/m) sum_i |x_i - y| - (1/(2 m^2)) sum_i sum_j |x_i - x_j|`.
 * - Energy score, `es_sample` (Gneiting et al. 2008): the same with Euclidean norms of the
 *   `dimension`-vectors, `(1/m) sum_i ||x_i - y|| - (1/(2 m^2)) sum_i sum_j ||x_i - x_j||`.
 * - Variogram score, `vs_sample` (Scheuerer and Hamill 2015):
 *   `sum_(i,j) w_ij (|y_i - y_j|^p - (1/m) sum_k |x_ki - x_kj|^p)^2`.
 * With `fair` the factor `1/(2 m^2)` of the CRPS and energy score becomes `1/(2 m (m - 1))`.
 * Lower is better for all three.
 *
 * Composition: the CRPS runs one 64-lane workgroup per site that adds the member terms in a fixed
 * lane order and a fixed binary tree. The energy and variogram scores write one term per member
 * or pair and the published `GPUReduction` sums them (a hierarchical tree of 256-row workgroups).
 * There are no atomics: the f32 results are repeatable on one device (the reduction may use
 * subgroup collectives where the device offers them, whose order is the device's). The cost is `O(m^2 d)` for the CRPS and energy score and `O(P m)` for the variogram
 * score over `P` pairs. Members, observations and pairs with NaN give NaN scores.
 */
export class GPUEnsembleScores implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUEnsembleScoresProps;

  constructor(props: GPUEnsembleScoresProps) {
    this.id = props.id ?? 'ensemble-scores';
    this.props = props;
    const {id} = this;
    const {output, memberCount, dimension} = props;
    validateCapacity(id, 'memberCount', memberCount, props.fair ? 2 : 1, 0xffff);
    // One 64-lane workgroup per site must stay below 2^32 invocations.
    validateCapacity(id, 'dimension', dimension, 1, Math.floor(0xffffffff / CRPS_LANES));
    // The invocation count of the per-member kernels is memberCount * dimension.
    validateCapacity(id, 'memberCount * dimension', memberCount * dimension, 1, 0xfffffffe);
    validateSinglePackedView(props.ensemble, ['float32'], `${id} ensemble`);
    validateSinglePackedView(props.observations, ['float32'], `${id} observations`);
    if (props.ensemble.length < memberCount * dimension) {
      throw new Error(`${id} ensemble needs memberCount * dimension rows`);
    }
    if (props.observations.length < dimension) {
      throw new Error(`${id} observations needs dimension rows`);
    }
    if (!output.crps && !output.energyScore && !output.variogramScore) {
      throw new Error(`${id} needs crps, energyScore or variogramScore output`);
    }
    if (output.crps) {
      validateSinglePackedView(output.crps, ['float32'], `${id} output.crps`);
      if (output.crps.length < dimension) {
        throw new Error(`${id} output.crps needs dimension rows`);
      }
    }
    if (output.energyScore) {
      validateSinglePackedView(output.energyScore, ['float32'], `${id} output.energyScore`);
      if (output.energyScore.length !== 1) {
        throw new Error(`${id} output.energyScore must have one row`);
      }
    }
    if (Boolean(output.variogramScore) !== Boolean(props.variogramPairs)) {
      throw new Error(`${id} output.variogramScore and variogramPairs go together`);
    }
    if (output.variogramScore && props.variogramPairs) {
      validateSinglePackedView(output.variogramScore, ['float32'], `${id} output.variogramScore`);
      validateSinglePackedView(props.variogramPairs, ['uint32x2'], `${id} variogramPairs`);
      if (output.variogramScore.length !== 1 || props.variogramPairs.length < 1) {
        throw new Error(
          `${id} output.variogramScore must have one row and variogramPairs one pair`
        );
      }
      if (props.variogramWeights) {
        validateSinglePackedView(props.variogramWeights, ['float32'], `${id} variogramWeights`);
        if (props.variogramWeights.length < props.variogramPairs.length) {
          throw new Error(`${id} variogramWeights needs one row per pair`);
        }
      }
    } else if (props.variogramWeights) {
      throw new Error(`${id} variogramWeights needs variogramPairs`);
    }
    const order = props.variogramOrder ?? 0.5;
    if (!(order > 0 && Number.isFinite(Math.fround(order)))) {
      throw new Error(`${id} variogramOrder must be positive and finite`);
    }
    validateGraphOutputsDisjointFromInputs(
      id,
      [output.crps, output.energyScore, output.variogramScore],
      [props.ensemble, props.observations, props.variogramPairs, props.variogramWeights]
    );
  }

  /** Returns the offsets, term and reduction nodes of the requested scores. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const {output, memberCount, dimension} = props;
    validateGraphViewsBelongToGraph(id, graph, [
      props.ensemble,
      props.observations,
      props.variogramPairs,
      props.variogramWeights,
      output.crps,
      output.energyScore,
      output.variogramScore
    ]);
    const nodes: GPUCommandNode<Parameters>[] = [];
    const memberCountText = `${memberCount}u`;
    const dimensionText = `${dimension}u`;
    const pairFactor = 1 / (2 * memberCount * (props.fair ? memberCount - 1 : memberCount));

    const sumAll = (
      name: string,
      terms: GraphDataView<'float32'>,
      result: GraphDataView<'float32'>
    ) => {
      nodes.push(
        ...new GPUReduction({
          id: `${id}-${name}-sum`,
          input: terms,
          output: result,
          operation: 'sum'
        }).getCommandNodes(graph)
      );
    };

    if (output.crps) {
      // One workgroup per site. Lane l adds the terms of members l, l + 64, ... in order, then a
      // fixed binary tree adds the lanes: the f32 sum does not depend on scheduling.
      // Term of member i: |x_i - y| / m - pairFactor * sum_j |x_i - x_j|, j in order.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-crps`,
          operation: OPERATION,
          variant: props.fair ? 'crps-fair' : 'crps',
          bindings: [
            {name: 'ensemble', view: props.ensemble, type: 'f32', access: 'read'},
            {name: 'observations', view: props.observations, type: 'f32', access: 'read'},
            {name: 'crps', view: output.crps, type: 'f32', access: 'read_write'}
          ],
          workgroupSize: CRPS_LANES,
          invocationCount: dimension * CRPS_LANES,
          guardIndex: false,
          declarations: `var<workgroup> partials: array<f32, ${CRPS_LANES}>;`,
          body: `let site = index / ${CRPS_LANES}u;
  let lane = localInvocationIndex;
  var laneSum = 0.0;
  if (site < ${dimensionText}) {
    let observed = observations[observationsOffset + site];
    for (var member = lane; member < ${memberCountText}; member += ${CRPS_LANES}u) {
      let value = ensemble[ensembleOffset + member * ${dimensionText} + site];
      var pairSum = 0.0;
      for (var other = 0u; other < ${memberCountText}; other++) {
        pairSum += abs(value - ensemble[ensembleOffset + other * ${dimensionText} + site]);
      }
      laneSum += abs(value - observed) / f32(${memberCount}) - pairSum * f32(${pairFactor});
    }
  }
  partials[lane] = laneSum;
  workgroupBarrier();
  for (var stride = ${CRPS_LANES / 2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) { partials[lane] = partials[lane] + partials[lane + stride]; }
    workgroupBarrier();
  }
  if (lane == 0u && site < ${dimensionText}) { crps[crpsOffset + site] = partials[0]; }`
        })
      );
    }

    if (output.energyScore) {
      const terms = createTransientView(graph, `${id}-energy-terms`, 'float32', memberCount);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-energy-terms`,
          operation: OPERATION,
          variant: props.fair ? 'energy-terms-fair' : 'energy-terms',
          bindings: [
            {name: 'ensemble', view: props.ensemble, type: 'f32', access: 'read'},
            {name: 'observations', view: props.observations, type: 'f32', access: 'read'},
            {name: 'terms', view: terms, type: 'f32', access: 'read_write'}
          ],
          invocationCount: memberCount,
          body: `var observedSquares = 0.0;
  for (var site = 0u; site < ${dimensionText}; site++) {
    let difference = ensemble[ensembleOffset + index * ${dimensionText} + site] - observations[observationsOffset + site];
    observedSquares += difference * difference;
  }
  var pairSum = 0.0;
  for (var other = 0u; other < ${memberCountText}; other++) {
    var squares = 0.0;
    for (var site = 0u; site < ${dimensionText}; site++) {
      let difference = ensemble[ensembleOffset + index * ${dimensionText} + site] - ensemble[ensembleOffset + other * ${dimensionText} + site];
      squares += difference * difference;
    }
    pairSum += sqrt(squares);
  }
  terms[termsOffset + index] = sqrt(observedSquares) / f32(${memberCount}) - pairSum * f32(${pairFactor});`
        })
      );
      sumAll('energy', terms, output.energyScore);
    }

    if (output.variogramScore && props.variogramPairs) {
      const pairCount = props.variogramPairs.length;
      const order = props.variogramOrder ?? 0.5;
      const terms = createTransientView(graph, `${id}-variogram-terms`, 'float32', pairCount);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-variogram-terms`,
          operation: OPERATION,
          variant: 'variogram-terms',
          bindings: [
            {name: 'ensemble', view: props.ensemble, type: 'f32', access: 'read'},
            {name: 'observations', view: props.observations, type: 'f32', access: 'read'},
            {name: 'pairs', view: props.variogramPairs, type: 'u32', access: 'read'},
            ...(props.variogramWeights
              ? [
                  {
                    name: 'weights',
                    view: props.variogramWeights,
                    type: 'f32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            {name: 'terms', view: terms, type: 'f32', access: 'read_write'}
          ],
          invocationCount: pairCount,
          declarations: `${IS_NAN_VALUE_WGSL}
// |x|^p for x >= 0 with 0^p = 0; NaN stays NaN. The NaN test reads the bits: a float comparison
// with NaN is false under IEEE rules (which sent NaN to the zero branch) and unspecified under
// the finite-math assumption WGSL allows, so neither order of a comparison keeps NaN reliably.
fn powerOfMagnitude(magnitude: f32) -> f32 {
  if (isNanValue(magnitude)) { return magnitude; }
  return select(pow(magnitude, f32(${order})), 0.0, magnitude == 0.0);
}`,
          body: `let first = pairs[pairsOffset + index * 2u];
  let second = pairs[pairsOffset + index * 2u + 1u];
  if (first >= ${dimensionText} || second >= ${dimensionText}) {
    terms[termsOffset + index] = bitcast<f32>(0x7fc00000u | (index & 0u));
    return;
  }
  let observed = powerOfMagnitude(abs(observations[observationsOffset + first] - observations[observationsOffset + second]));
  var forecastSum = 0.0;
  for (var member = 0u; member < ${memberCountText}; member++) {
    forecastSum += powerOfMagnitude(abs(ensemble[ensembleOffset + member * ${dimensionText} + first] - ensemble[ensembleOffset + member * ${dimensionText} + second]));
  }
  let difference = observed - forecastSum / f32(${memberCount});
  terms[termsOffset + index] = ${props.variogramWeights ? 'weights[weightsOffset + index] * ' : ''}difference * difference;`
        })
      );
      sumAll('variogram', terms, output.variogramScore);
    }
    return nodes;
  }
}
