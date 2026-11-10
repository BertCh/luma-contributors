// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GPUCommandNodeProducer,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createRankEnvelopeNodes} from './gpu-point-pattern-rank-envelope';
import {validatePointProcessView} from './point-process-utils';

/** Most curve rows: the global deviation tests reduce a curve in one serial invocation. */
const MAXIMUM_CURVE_ROW_COUNT = 4096;
/** Most simulated curves: the pointwise order statistics sort S values per row. */
const MAXIMUM_SIMULATION_COUNT = 4096;
/** Quadratic rank comparison budget of the extreme-rank test; no curves are truncated. */
const MAXIMUM_RANK_COMPARISON_COUNT = 16777216;

/** Resident simulation-curve inference with explicit exchangeability metadata. */
export type GPUPointPatternEnvelopeProps = {
  id?: string;
  observed: GraphDataView<'float32'>;
  /** Replicate-major curves, simulationCount * observed.length values. */
  simulations: GraphDataView<'float32'>;
  /** Fixed, externally specified null expectation, shared by observed and simulated curves. */
  reference: GraphDataView<'float32'>;
  /** Positive integration weights; used by the DCLF statistic. */
  integrationWeights: GraphDataView<'float32'>;
  simulationCount: number;
  /** Pointwise lower/upper order statistic, default 1. */
  rank?: number;
  statistic?: 'maximum-deviation' | 'integrated-squared-deviation' | 'extreme-rank';
  /** Global rank-envelope significance level, default 0.05. */
  significanceLevel?: number;
  /** Explicit provenance; curves are supplied by a simulation graph, never synthesized here. */
  nullModel: string;
  /** Estimated parameters require the same refitting procedure for every simulated pattern. */
  fittedParameters: 'none' | 'fixed-external' | 'refit-each-replicate';
  output: {
    lower: GraphDataView<'float32'>;
    upper: GraphDataView<'float32'>;
    /** [observedStatistic, conservativePValue, pValueResolution, simulationCount]. */
    statistics: GraphDataView<'float32'>;
    /** 0 valid, 1 invalid/nonfinite curve or nonpositive integration weight. */
    status: GraphDataView<'uint32'>;
    /** Extreme-rank p interval [strict, conservative]; available only in extreme-rank mode. */
    pValueInterval?: GraphDataView<'float32'>;
  };
};

/** Selects the zero-based order statistic of one ordinate column by bitwise key selection. */
const getOrderedValueSelectionWGSL = (count: number, simulationCount: number) => `
fn getOrderedFloatKey(value: f32) -> u32 {
  let bits = bitcast<u32>(value);
  return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn selectOrderedValue(column: u32, order: u32) -> f32 {
  var answer = 0u;
  for (var bit = 32u; bit > 0u; bit--) {
    let trial = answer | (1u << (bit - 1u));
    var below = 0u;
    for (var other = 0u; other < ${simulationCount}u; other++) {
      below += select(0u, 1u, getOrderedFloatKey(simulations[simulationsOffset + other * ${count}u + column]) < trial);
    }
    if (below <= order) { answer = trial; }
  }
  return bitcast<f32>(select(~answer, answer ^ 0x80000000u, (answer & 0x80000000u) != 0u));
}
`;

/**
 * Pointwise order-statistic envelopes and a distinct global deviation test, composed over GPU
 * simulation curves. The global p-value is (1 + number simulated >= observed)/(S + 1), retaining
 * ties conservatively. DCLF uses the supplied quadrature measure. No invalid replicate is silently
 * discarded: any invalid curve makes every inferential result NaN. Null calibration depends on
 * caller-provided exchangeable simulations and declared parameter treatment.
 */
export class GPUPointPatternEnvelope implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUPointPatternEnvelopeProps;
  constructor(props: GPUPointPatternEnvelopeProps) {
    this.id = props.id ?? 'point-pattern-envelope';
    this.props = props;
    const count = props.observed.length;
    const rank = props.rank ?? 1;
    const id = this.id;
    validateCapacity(id, 'observed rows', count, 1, MAXIMUM_CURVE_ROW_COUNT);
    validateCapacity(id, 'simulationCount', props.simulationCount, 2, MAXIMUM_SIMULATION_COUNT);
    validateRequirement(
      id,
      Number.isInteger(rank) && rank >= 1 && 2 * rank <= props.simulationCount,
      'rank must be an integer with 1 <= rank and 2 * rank <= simulationCount'
    );
    validateRequirement(id, props.nullModel.trim().length > 0, 'nullModel must be non-empty');
    validateRequirement(
      id,
      ['none', 'fixed-external', 'refit-each-replicate'].includes(props.fittedParameters),
      "fittedParameters must be 'none', 'fixed-external' or 'refit-each-replicate'"
    );
    validateRequirement(
      id,
      props.statistic === undefined ||
        ['maximum-deviation', 'integrated-squared-deviation', 'extreme-rank'].includes(
          props.statistic
        ),
      "statistic must be 'maximum-deviation', 'integrated-squared-deviation' or 'extreme-rank'"
    );
    for (const view of [
      props.observed,
      props.reference,
      props.integrationWeights,
      props.output.lower,
      props.output.upper
    ]) {
      validatePointProcessView(view, 'float32', count, id);
    }
    validatePointProcessView(props.simulations, 'float32', count * props.simulationCount, id);
    validatePointProcessView(props.output.statistics, 'float32', 4, id);
    validatePointProcessView(props.output.status, 'uint32', 1, id);
    if (props.statistic === 'extreme-rank') {
      // The quadratic rank comparison budget is explicit; no curves/radii are truncated.
      validateRequirement(
        id,
        (props.simulationCount + 1) ** 2 * count <= MAXIMUM_RANK_COMPARISON_COUNT,
        `(simulationCount + 1)^2 * observed rows must not exceed ${MAXIMUM_RANK_COMPARISON_COUNT}`
      );
      validateRequirement(
        id,
        (props.significanceLevel ?? 0.05) > 0 && (props.significanceLevel ?? 0.05) < 1,
        'significanceLevel must lie in (0, 1)'
      );
    }
    if (props.output.pValueInterval) {
      validateRequirement(
        id,
        props.statistic === 'extreme-rank',
        "output.pValueInterval needs statistic 'extreme-rank'"
      );
      validatePointProcessView(props.output.pValueInterval, 'float32', 2, id);
    }

    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.observed,
      props.simulations,
      props.reference,
      props.integrationWeights
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    const count = props.observed.length;
    const simulationCount = props.simulationCount;
    const rank = props.rank ?? 1;
    validateGraphViewsBelongToGraph(id, graph, [
      props.observed,
      props.simulations,
      props.reference,
      props.integrationWeights,
      ...Object.values(props.output)
    ]);
    if (props.statistic === 'extreme-rank') return createRankEnvelopeNodes(graph, props, id);
    const deviations = createTransientView(
      graph,
      `${id}-deviations`,
      'float32',
      simulationCount + 1
    );
    return [
      createWGSLKernelNode(graph, {
        id: `${id}-deviations`,
        operation: 'GPUPointPatternEnvelope',
        variant: props.statistic ?? 'maximum-deviation',
        invocationCount: simulationCount + 1,
        bindings: [
          {name: 'observed', view: props.observed, type: 'f32', access: 'read'},
          {name: 'simulations', view: props.simulations, type: 'f32', access: 'read'},
          {name: 'reference', view: props.reference, type: 'f32', access: 'read'},
          {name: 'weights', view: props.integrationWeights, type: 'f32', access: 'read'},
          {name: 'deviations', view: deviations, type: 'f32', access: 'read_write'}
        ],
        declarations: IS_FINITE_FLOAT_WGSL,
        body: `var statistic = 0.0;
  var valid = true;
  for (var radius = 0u; radius < ${count}u; radius++) {
    var value = observed[observedOffset + radius];
    if (index < ${simulationCount}u) { value = simulations[simulationsOffset + index * ${count}u + radius]; }
    let center = reference[referenceOffset + radius];
    let weight = weights[weightsOffset + radius];
    valid = valid && isFiniteFloat(value) && isFiniteFloat(center) && isFiniteFloat(weight) && weight > 0.0;
    let deviation = abs(value - center);
    ${props.statistic === 'integrated-squared-deviation' ? 'statistic += weight * deviation * deviation;' : 'statistic = max(statistic, deviation);'}
  }
  deviations[deviationsOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), statistic, valid && isFiniteFloat(statistic));`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-test`,
        operation: 'GPUPointPatternEnvelope',
        variant: 'global-test',
        invocationCount: 1,
        bindings: [
          {name: 'deviations', view: deviations, type: 'f32', access: 'read'},
          {name: 'statistics', view: props.output.statistics, type: 'f32', access: 'read_write'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: IS_FINITE_FLOAT_WGSL,
        body: `let observed = deviations[deviationsOffset + ${simulationCount}u];
  var valid = isFiniteFloat(observed);
  var exceedances = 0u;
  for (var replicate = 0u; replicate < ${simulationCount}u; replicate++) {
    let value = deviations[deviationsOffset + replicate];
    valid = valid && isFiniteFloat(value);
    exceedances += select(0u, 1u, value >= observed);
  }
  let invalid = bitcast<f32>(0x7fc00000u | (index & 0u));
  statistics[statisticsOffset] = select(invalid, observed, valid);
  statistics[statisticsOffset + 1u] = select(invalid, f32(exceedances + 1u) / ${simulationCount + 1}.0, valid);
  statistics[statisticsOffset + 2u] = 1.0 / ${simulationCount + 1}.0;
  statistics[statisticsOffset + 3u] = ${simulationCount}.0;
  status[statusOffset] = select(1u, 0u, valid);`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-pointwise`,
        operation: 'GPUPointPatternEnvelope',
        variant: 'pointwise',
        invocationCount: count,
        bindings: [
          {name: 'simulations', view: props.simulations, type: 'f32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read'},
          {name: 'lower', view: props.output.lower, type: 'f32', access: 'read_write'},
          {name: 'upper', view: props.output.upper, type: 'f32', access: 'read_write'}
        ],
        declarations: getOrderedValueSelectionWGSL(count, simulationCount),
        // The rank-th smallest and largest ordinate come from a 32-bit radix selection over the
        // monotone integer keys of the finite simulated values: 64 linear passes per ordinate
        // instead of the quadratic pairwise rank (all values are finite when status is zero).
        body: `var lowerValue = bitcast<f32>(0x7fc00000u | (index & 0u));
  var upperValue = lowerValue;
  if (status[statusOffset] == 0u) {
    lowerValue = selectOrderedValue(index, ${rank - 1}u);
    upperValue = selectOrderedValue(index, ${simulationCount - rank}u);
  }
  lower[lowerOffset + index] = lowerValue;
  upper[upperOffset + index] = upperValue;`
      })
    ];
  }
}
