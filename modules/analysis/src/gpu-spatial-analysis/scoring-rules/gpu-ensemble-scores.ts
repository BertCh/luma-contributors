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
import {SCALED_QUOTIENT_WGSL} from '../../analysis-core/kernels/scaled-quotient-wgsl';
import {
  IS_NAN_VALUE_WGSL,
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL
} from '../../analysis-core/kernels/wgsl-common';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs,
  validateFiniteNumber,
  validateRequirement,
  validateGPUStorageBindingByteLength,
  validateSinglePackedView
} from '../../analysis-core/kernels/gpu-contributor-utils';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateRegressionOutputDispatches} from '../spatial-regression/regression-output-dispatches';

import {
  captureEnsembleScoreOutputs,
  resolveEnsembleScoresEstimator,
  type EnsembleScoresEstimator
} from './ensemble-scores-options';

const OPERATION = 'GPUEnsembleScores';
/** The reference scores enumerate member pairs; the member index remains a bounded uint16 count. */
const ENSEMBLE_MAXIMUM_MEMBER_COUNT = 0xffff;
/** CRPS runs 64 invocations per site and keeps the invocation index inside uint32. */
const ENSEMBLE_MAXIMUM_DIMENSION = Math.floor(0xffffffff / 64);
/** Member-major scalar indices reserve the final uint32 value. */
const ENSEMBLE_MAXIMUM_VALUE_COUNT = 0xfffffffe;
/** Pair addresses contain two uint32 words, so multiplication by two cannot wrap. */
const ENSEMBLE_MAXIMUM_VARIOGRAM_PAIR_COUNT = 0x7fffffff;
/** Covers all score kernels and at most four two-pass published reduction levels. */
const ENSEMBLE_COMMAND_NODE_ALLOWANCE = 32;
/** Lanes of the per-site CRPS workgroup. */
const CRPS_LANES = 64;

/** Normalize before overflow, retaining the direct difference when it is representable. */
const ENSEMBLE_DISTANCE_WGSL = `${IS_FINITE_BITS_WGSL}
${SCALED_QUOTIENT_WGSL}
fn getEnsembleWeightedDifference(first: f32, second: f32, weight: f32) -> f32 {
  let difference = first - second;
  if (!isFiniteBits(first) || !isFiniteBits(second)) {
    let firstMagnitudeBits = bitcast<u32>(first) & 0x7fffffffu;
    let secondMagnitudeBits = bitcast<u32>(second) & 0x7fffffffu;
    if (firstMagnitudeBits > 0x7f800000u || secondMagnitudeBits > 0x7f800000u ||
        (firstMagnitudeBits == 0x7f800000u && secondMagnitudeBits == 0x7f800000u)) { return getNaN(); }
    // Publish an infinite observation distance by bits; never ask finite-math arithmetic to
    // manufacture the exceptional value from nonfinite subtraction.
    return bitcast<f32>(max(firstMagnitudeBits, secondMagnitudeBits));
  }
  if (isFiniteBits(difference)) { return abs(difference) * weight; }
  // Keep scaling ahead of subtraction even under reassociation. All score weights are in
  // (0, 1], and a finite overflowing difference needs weight <= 0.5 for a finite score.
  let divisor = 1.0 / weight;
  return abs(getScaledQuotient(first, divisor, 1.0, 0)
    - getScaledQuotient(second, divisor, 1.0, 0));
}`;

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
  /** Estimator for CRPS and energy; defaults to empirical. */
  estimator?: EnsembleScoresEstimator;
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
  /** Optional finite nonnegative weight per pair; defaults to 1. Invalid weights give NaN variogram. */
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

/** Caller-owned score ports, with CRPS in the site row domain and scalar whole-field scores. */
export type GPUEnsembleScoresOutput = GPUEnsembleScoresProps['output'];

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
 * With `estimator: 'fair'` the factor `1/(2 m^2)` of the CRPS and energy score becomes `1/(2 m (m - 1))`.
 * Lower is better for all three.
 *
 * Composition: the CRPS runs one 64-lane workgroup per site that adds the member terms in a fixed
 * lane order and a fixed binary tree. The energy and variogram scores write one term per member
 * or pair and the published `GPUReduction` sums them (a hierarchical tree of 256-row workgroups).
 * There are no atomics: the f32 results are repeatable on one device (the reduction may use
 * subgroup collectives where the device offers them, whose order is the device's). The cost is `O(m^2 d)` for the CRPS and energy score and `O(P m)` for the variogram
 * score over `P` pairs. Fair CRPS and scalar fair energy integrate nonnegative sorted intervals.
 * Variogram endpoints share a scale before subtraction and powers. Invalid pair indices,
 * nonfinite endpoints, and negative or nonfinite weights give NaN only on the variogram output.
 * The GPU route accepts at most 65,535 members, 67,108,863 sites and 4,294,967,294 member-site
 * entries, subject to device binding limits. The variogram route accepts at most 2,147,483,647 pairs.
 * The estimated graph is limited to 32,768 command nodes.
 */
export class GPUEnsembleScores implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUEnsembleScoresProps;
  private readonly estimator: EnsembleScoresEstimator;
  readonly estimatedCommandNodeCount = ENSEMBLE_COMMAND_NODE_ALLOWANCE;

  constructor(props: GPUEnsembleScoresProps) {
    this.id = props.id ?? 'ensemble-scores';
    const {estimator} = props;
    this.estimator = resolveEnsembleScoresEstimator(this.id, 'fair' in props ? props : {estimator});
    const selected = captureEnsembleScoreOutputs(this.id, props.output);
    props = {
      id: props.id,
      ensemble: props.ensemble,
      observations: props.observations,
      memberCount: props.memberCount,
      dimension: props.dimension,
      estimator,
      variogramOrder: props.variogramOrder,
      variogramPairs: props.variogramPairs,
      variogramWeights: props.variogramWeights,
      output: Object.fromEntries(selected)
    };
    const fair = this.estimator === 'fair';
    this.props = props;
    const {id} = this;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    const {output, memberCount, dimension} = props;
    captureEnsembleScoreOutputs(id, output);
    validateCapacity(id, 'memberCount', memberCount, fair ? 2 : 1, ENSEMBLE_MAXIMUM_MEMBER_COUNT);
    // One 64-lane workgroup per site must stay below 2^32 invocations.
    validateCapacity(id, 'dimension', dimension, 1, ENSEMBLE_MAXIMUM_DIMENSION);
    // The invocation count of the per-member kernels is memberCount * dimension.
    validateCapacity(
      id,
      'memberCount * dimension',
      memberCount * dimension,
      1,
      ENSEMBLE_MAXIMUM_VALUE_COUNT
    );
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
      validateCapacity(
        id,
        'variogramPairs.length',
        props.variogramPairs.length,
        1,
        ENSEMBLE_MAXIMUM_VARIOGRAM_PAIR_COUNT
      );
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
    const order = Math.fround(props.variogramOrder ?? 0.5);
    validateFiniteNumber(id, 'variogramOrder', props.variogramOrder ?? 0.5, 0);
    validateRequirement(
      id,
      Math.fround(order) > 0,
      'variogramOrder must be positive when represented as float32'
    );
    validateRegressionOutputDispatches(id, output, []);
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
    const fair = this.estimator === 'fair';
    const {output, memberCount, dimension} = props;
    validateGraphChunkBindingRanges(graph, id, {
      ensemble: props.ensemble,
      observations: props.observations,
      variogramPairs: props.variogramPairs,
      variogramWeights: props.variogramWeights,
      ...output
    });
    for (const [name, length] of [
      ['energy terms', output.energyScore ? memberCount : 0],
      ['variogram terms', output.variogramScore ? props.variogramPairs!.length : 0]
    ] as const) {
      if (!length) continue;
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        name,
        length * 4,
        'reduce members or variogram pairs'
      );
      // Published GPUReduction has one scalar partial per 256 input rows at every level.
      let reductionLength = length;
      while (reductionLength > 256) {
        reductionLength = Math.ceil(reductionLength / 256);
        validateGPUStorageBindingByteLength(
          graph.device,
          id,
          `${name} reduction partials`,
          reductionLength * 4,
          'reduce members or variogram pairs'
        );
      }
    }
    const nodes: GPUCommandNode<Parameters>[] = [];
    const memberCountText = `${memberCount}u`;
    const dimensionText = `${dimension}u`;
    const pairFactor = 1 / (2 * memberCount * (fair ? memberCount - 1 : memberCount));

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

    const scalarEnergy = fair && dimension === 1 ? output.energyScore : undefined;
    // Keep the outputs in separate dispatches: callers may pack their logical ranges in one
    // buffer, where aligned writable binding ranges could overlap in a combined dispatch.
    for (const [scoreName, scoreOutput] of [
      ['crps', output.crps],
      ['scalar-energy', scalarEnergy]
    ] as const) {
      if (!scoreOutput) continue;
      // One workgroup per site. Lane l adds the terms of members l, l + 64, ... in order, then a
      // fixed binary tree adds the lanes: the f32 sum does not depend on scheduling.
      // Term of member i: |x_i - y| / m - pairFactor * sum_j |x_i - x_j|, j in order.
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-${scoreName}`,
          operation: OPERATION,
          variant: fair ? 'crps-fair' : 'crps',
          bindings: [
            {name: 'ensemble', view: props.ensemble, type: 'f32', access: 'read'},
            {name: 'observations', view: props.observations, type: 'f32', access: 'read'},
            {name: 'crps', view: scoreOutput, type: 'f32', access: 'read_write'}
          ],
          workgroupSize: CRPS_LANES,
          invocationCount: dimension * CRPS_LANES,
          guardIndex: false,
          declarations: `${ENSEMBLE_DISTANCE_WGSL}\n${GET_NAN_WGSL}\nvar<workgroup> partials: array<f32, ${CRPS_LANES}>;\nvar<workgroup> invalidMembers: array<u32, ${CRPS_LANES}>;`,
          body: `let site = index / ${CRPS_LANES}u;
  let lane = localInvocationIndex;
  var laneSum = 0.0;
  var laneInvalid = 0u;
  if (site < ${dimensionText}) {
    let observed = observations[observationsOffset + site];
    for (var member = lane; member < ${memberCountText}; member += ${CRPS_LANES}u) {
      let value = ensemble[ensembleOffset + member * ${dimensionText} + site];
      if (!isFiniteBits(value)) { laneInvalid = 1u; continue; }
      ${
        fair
          ? `
      // Rank ties by member index. Each member owns the interval to its successor;
      // tied intervals have length zero. This avoids subtracting large score means.
      var rank = 0u;
      var nextValue = value;
      var hasNext = false;
      for (var other = 0u; other < ${memberCountText}; other++) {
        let otherValue = ensemble[ensembleOffset + other * ${dimensionText} + site];
        if (!isFiniteBits(otherValue)) { laneInvalid = 1u; continue; }
        if (otherValue < value || (otherValue == value && other < member)) { rank += 1u; }
        if (otherValue > value || (otherValue == value && other > member)) {
          if (!hasNext || otherValue < nextValue) { nextValue = otherValue; hasNext = true; }
        }
      }
      if (rank == 0u && observed < value) {
        laneSum += getEnsembleWeightedDifference(value, observed, 1.0);
      }
      if (rank == ${memberCount - 1}u && observed > value) {
        laneSum += getEnsembleWeightedDifference(observed, value, 1.0);
      }
      if (hasNext) {
        let below = f32(rank + 1u) / f32(${memberCount});
        let above = f32(${memberCount - 1}u - rank) / f32(${memberCount});
        if (observed > value && rank > 0u) {
          laneSum += getEnsembleWeightedDifference(min(observed, nextValue), value,
            below * (f32(rank) / f32(${memberCount - 1})));
        }
        if (observed < nextValue && rank < ${memberCount - 2}u) {
          laneSum += getEnsembleWeightedDifference(nextValue, max(observed, value),
            above * (f32(${memberCount - 2}u - rank) / f32(${memberCount - 1})));
        }
      }
      // A NaN observation must not silently skip every interval comparison.
      if ((bitcast<u32>(observed) & 0x7fffffffu) > 0x7f800000u) { laneSum = getNaN(); }
      `
          : `      var pairSum = 0.0;
      for (var other = 0u; other < ${memberCountText}; other++) {
        pairSum += getEnsembleWeightedDifference(value, ensemble[ensembleOffset + other * ${dimensionText} + site], f32(${pairFactor}));
      }
      laneSum += getEnsembleWeightedDifference(value, observed, 1.0 / f32(${memberCount})) - pairSum;`
      }

    }
  }
  partials[lane] = laneSum;
  invalidMembers[lane] = laneInvalid;
  workgroupBarrier();
  for (var stride = ${CRPS_LANES / 2}u; stride > 0u; stride = stride / 2u) {
    if (lane < stride) {
      partials[lane] = partials[lane] + partials[lane + stride];
      invalidMembers[lane] = invalidMembers[lane] | invalidMembers[lane + stride];
    }
    workgroupBarrier();
  }
  if (lane == 0u && site < ${dimensionText}) {
    if (invalidMembers[0] != 0u) { crps[crpsOffset + site] = getNaN(); }
    else { crps[crpsOffset + site] = partials[0]; }
  }`
        })
      );
    }

    if (output.energyScore && !scalarEnergy) {
      const terms = createTransientView(graph, `${id}-energy-terms`, 'float32', memberCount);
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-energy-terms`,
          operation: OPERATION,
          variant: fair ? 'energy-terms-fair' : 'energy-terms',
          bindings: [
            {name: 'ensemble', view: props.ensemble, type: 'f32', access: 'read'},
            {name: 'observations', view: props.observations, type: 'f32', access: 'read'},
            {name: 'terms', view: terms, type: 'f32', access: 'read_write'}
          ],
          invocationCount: memberCount,
          declarations: `${ENSEMBLE_DISTANCE_WGSL}\n${IS_NAN_VALUE_WGSL}\n${GET_NAN_WGSL}
// A raw ratio of near-maximum magnitudes can become multiply-by-zero when Metal
// lowers division to a flushed subnormal reciprocal. Divide bounded mantissas instead.
fn accumulateEnsembleMagnitude(state: vec2<f32>, magnitude: f32) -> vec2<f32> {
  if (isNanValue(magnitude)) { return vec2<f32>(getNaN()); }
  if (!isFiniteBits(state.x)) { return state; }
  if (!isFiniteBits(magnitude)) { return vec2<f32>(magnitude, 1.0); }
  if (magnitude == 0.0) { return state; }
  if (state.x < magnitude) { let ratio = getScaledQuotient(state.x, magnitude, 1.0, 0); return vec2<f32>(magnitude, 1.0 + state.y * ratio * ratio); }
  let ratio = getScaledQuotient(magnitude, state.x, 1.0, 0); return vec2<f32>(state.x, state.y + ratio * ratio);
}`,
          body: `var observedNorm = vec2<f32>(0.0);
  for (var site = 0u; site < ${dimensionText}; site++) {
    let memberValue = ensemble[ensembleOffset + index * ${dimensionText} + site];
    if (!isFiniteBits(memberValue)) { terms[termsOffset + index] = getNaN(); return; }
    let magnitude = getEnsembleWeightedDifference(memberValue, observations[observationsOffset + site], 1.0 / f32(${memberCount}));
    observedNorm = accumulateEnsembleMagnitude(observedNorm, magnitude);
  }
  var pairSum = 0.0;
  for (var other = 0u; other < ${memberCountText}; other++) {
    var pairNorm = vec2<f32>(0.0);
    for (var site = 0u; site < ${dimensionText}; site++) {
      let otherValue = ensemble[ensembleOffset + other * ${dimensionText} + site];
      if (!isFiniteBits(otherValue)) { terms[termsOffset + index] = getNaN(); return; }
      let magnitude = getEnsembleWeightedDifference(ensemble[ensembleOffset + index * ${dimensionText} + site], otherValue, f32(${pairFactor}));
      pairNorm = accumulateEnsembleMagnitude(pairNorm, magnitude);
    }
    pairSum += pairNorm.x * sqrt(pairNorm.y);
  }
  terms[termsOffset + index] = observedNorm.x * sqrt(observedNorm.y) - pairSum;`
        })
      );
      sumAll('energy', terms, output.energyScore);
    }

    if (output.variogramScore && props.variogramPairs) {
      const pairCount = props.variogramPairs.length;
      const order = Math.fround(props.variogramOrder ?? 0.5);
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
          declarations: `${IS_FINITE_BITS_WGSL}
${GET_NAN_WGSL}
struct VariogramGap { fraction: f32, exponent: i32 }
// Decode before arithmetic so a subnormal endpoint can be normalized without being flushed.
fn getVariogramParts(value: f32) -> VariogramGap {
  let bits = bitcast<u32>(value);
  let magnitude = bits & 0x7fffffffu;
  if (magnitude == 0u) { return VariogramGap(0.0, 0); }
  var fractionBits = magnitude;
  var exponent = i32((magnitude >> 23u) & 255u) - 126;
  if ((magnitude & 0x7f800000u) == 0u) {
    let shift = countLeadingZeros(magnitude) - 8u;
    fractionBits = magnitude << shift;
    exponent = -125 - i32(shift);
  }
  return VariogramGap(bitcast<f32>((bits & 0x80000000u) | 0x3f000000u |
    (fractionBits & 0x7fffffu)), exponent);
}
fn getVariogramGap(firstValue: f32, secondValue: f32) -> VariogramGap {
  let firstParts = getVariogramParts(firstValue);
  let secondParts = getVariogramParts(secondValue);
  if (firstParts.fraction == 0.0) { return VariogramGap(abs(secondParts.fraction), secondParts.exponent); }
  if (secondParts.fraction == 0.0) { return VariogramGap(abs(firstParts.fraction), firstParts.exponent); }
  let frame = max(firstParts.exponent, secondParts.exponent);
  let magnitude = abs(ldexp(firstParts.fraction, firstParts.exponent - frame)
    - ldexp(secondParts.fraction, secondParts.exponent - frame));
  if (magnitude == 0.0) { return VariogramGap(0.0, 0); }
  let parts = frexp(magnitude);
  return VariogramGap(parts.fract, frame + parts.exp);
}
fn getLargerVariogramGap(firstGap: VariogramGap, secondGap: VariogramGap) -> VariogramGap {
  if (firstGap.fraction == 0.0) { return secondGap; }
  if (secondGap.fraction == 0.0) { return firstGap; }
  if (secondGap.exponent > firstGap.exponent ||
      (secondGap.exponent == firstGap.exponent && secondGap.fraction > firstGap.fraction)) { return secondGap; }
  return firstGap;
}
fn getVariogramMagnitude(gap: VariogramGap, maximumGap: VariogramGap) -> f32 {
  if (gap.fraction == 0.0) { return 0.0; }
  ${
    order === 1
      ? `return ldexp(gap.fraction, gap.exponent - maximumGap.exponent);`
      : `
  let relativeLogarithm = log2(gap.fraction / maximumGap.fraction)
    + f32(gap.exponent - maximumGap.exponent);
  return exp2(f32(${order}) * relativeLogarithm);`
  }
}
fn getVariogramDiscrepancy(observed: VariogramGap, forecast: VariogramGap, maximumGap: VariogramGap) -> f32 {
  ${
    order === 0.5
      ? `
  // Rationalize close square roots before dividing by the common scale. This preserves a
  // small gap difference instead of subtracting independently rounded nonlinear ratios.
  if (observed.fraction != 0.0 && forecast.fraction != 0.0) {
    let firstScaled = ldexp(observed.fraction, observed.exponent - maximumGap.exponent);
    let secondScaled = ldexp(forecast.fraction, forecast.exponent - maximumGap.exponent);
    let denominator = sqrt(firstScaled) + sqrt(secondScaled);
    if (denominator != 0.0) {
      return ((firstScaled - secondScaled) / denominator) / sqrt(maximumGap.fraction);
    }
  }`
      : ''
  }
  return getVariogramMagnitude(observed, maximumGap) - getVariogramMagnitude(forecast, maximumGap);
}
fn restoreVariogramSquare(difference: f32, scalePower: f32, weight: f32) -> f32 {
  if (difference == 0.0 || (bitcast<u32>(weight) & 0x7fffffffu) == 0u) { return 0.0; }
  let differenceParts = frexp(difference);
  let weightParts = getVariogramParts(weight);
  let productParts = frexp(differenceParts.fract * differenceParts.fract * weightParts.fraction);
  let power = 2.0 * (f32(differenceParts.exp) + scalePower)
    + f32(weightParts.exponent + productParts.exp);
  let wholePower = floor(power);
  if (wholePower > 128.0) { var overflowBits = 0x7f800000u; return bitcast<f32>(overflowBits); }
  if (wholePower < -150.0) { return 0.0; }
  return ldexp(productParts.fract * exp2(power - wholePower), i32(wholePower));
}`,
          body: `let first = pairs[pairsOffset + index * 2u];
  let second = pairs[pairsOffset + index * 2u + 1u];
  if (first >= ${dimensionText} || second >= ${dimensionText}) {
    terms[termsOffset + index] = getNaN();
    return;
  }
  let weight = ${props.variogramWeights ? 'weights[weightsOffset + index]' : '1.0'};
  let weightBits = bitcast<u32>(weight);
  if (!isFiniteBits(weight) || ((weightBits & 0x80000000u) != 0u && (weightBits & 0x7fffffffu) != 0u)) {
    terms[termsOffset + index] = getNaN(); return;
  }
  let observedFirst = observations[observationsOffset + first];
  let observedSecond = observations[observationsOffset + second];
  let observedGap = getVariogramGap(observedFirst, observedSecond);
  var maximumGap = observedGap;
  var valid = isFiniteBits(observedFirst) && isFiniteBits(observedSecond);
  for (var member = 0u; member < ${memberCountText}; member++) {
    let firstValue = ensemble[ensembleOffset + member * ${dimensionText} + first];
    let secondValue = ensemble[ensembleOffset + member * ${dimensionText} + second];
    valid = valid && isFiniteBits(firstValue) && isFiniteBits(secondValue);
    maximumGap = getLargerVariogramGap(maximumGap, getVariogramGap(firstValue, secondValue));
  }
  if (!valid) { terms[termsOffset + index] = getNaN(); return; }
  if (maximumGap.fraction == 0.0) { terms[termsOffset + index] = 0.0; return; }
  let scalePower = ${order === 1 ? 'f32(maximumGap.exponent)' : `(f32(maximumGap.exponent) + log2(maximumGap.fraction)) * f32(${order})`};
  var difference = 0.0;
  for (var member = 0u; member < ${memberCountText}; member++) {
    let forecastGap = getVariogramGap(ensemble[ensembleOffset + member * ${dimensionText} + first],
      ensemble[ensembleOffset + member * ${dimensionText} + second]);
    difference += getVariogramDiscrepancy(observedGap, forecastGap, maximumGap);
  }
  terms[termsOffset + index] = restoreVariogramSquare(difference / f32(${memberCount}), scalePower,
    weight);`
        })
      );
      sumAll('variogram', terms, output.variogramScore);
    }
    return nodes;
  }
}
