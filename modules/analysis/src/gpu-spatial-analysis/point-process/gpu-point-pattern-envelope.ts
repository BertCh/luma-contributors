// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateSpatialOutputViews} from '../spatial-clustering/spatial-output-validation';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
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
  validateRequirement,
  validateGPUStorageBindingByteLength
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  POINT_PATTERN_EXACT_STATISTIC_WGSL,
  POINT_PATTERN_EXACT_MAXIMUM_WGSL
} from './point-pattern-exact-statistic-wgsl';
import {COMPENSATED_SUM_WGSL} from '../../analysis-core/kernels/compensated-sum-wgsl';
import {IS_FINITE_FLOAT_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {createRankEnvelopeNodes} from './gpu-point-pattern-rank-envelope';
import {
  snapshotEnvelopeConstruction,
  resolveEnvelopeSimulationCount,
  resolveEnvelopeStatistic
} from './point-pattern-envelope-construction';
import {validatePointProcessView} from './point-process-utils';

/** Most curve rows: the global deviation tests reduce a curve in one serial invocation. */
export const MAXIMUM_CURVE_ROW_COUNT = 4096;
/** Most simulated curves: the pointwise order statistics select from S values per row. */
export const MAXIMUM_SIMULATION_COUNT = 4096;
/** Quadratic rank comparison budget of the extreme-rank test; no curves are truncated. */
export const MAXIMUM_RANK_COMPARISON_COUNT = 16777216;

/** Resident simulation-curve inference with explicit exchangeability metadata. */
export type GPUPointPatternEnvelopeProps = {
  id?: string;
  observed: GraphDataView<'float32'>;
  /** Replicate-major curves, simulationCount * observed.length values. */
  simulations: GraphDataView<'float32'>;
  /** Fixed null expectation in curve units; required for deviation statistics, unused by extreme rank. */
  reference?: GraphDataView<'float32'>;
  /** Positive quadrature weights; required only for integrated-squared-deviation (DCLF). */
  integrationWeights?: GraphDataView<'float32'>;
  /** Number of simulated curves; omitted means the exact simulations.length / observed.length quotient. */
  simulationCount?: number;
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
    /** 0 valid, 1 invalid/nonfinite used input (including nonpositive weights in integrated mode). */
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
 * GPU limits: 4096 ordinates per curve, 4096 simulations, and for extreme rank
 * (simulationCount + 1)^2 * ordinateCount <= 16777216 comparisons in total.
 * Maximum and integrated deviations are compared with exact dyadic integer arithmetic on
 * the stored float32 inputs. The displayed statistic remains float32.
 */
export class GPUPointPatternEnvelope implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUPointPatternEnvelopeProps & {simulationCount: number};
  constructor(input: GPUPointPatternEnvelopeProps) {
    const captured = snapshotEnvelopeConstruction(input);
    const props = snapshotEnvelopeConstruction({
      ...captured,
      statistic: resolveEnvelopeStatistic(
        captured.id ?? 'point-pattern-envelope',
        captured.statistic
      ),
      simulationCount: resolveEnvelopeSimulationCount(
        captured.id ?? 'point-pattern-envelope',
        captured.observed.length,
        captured.simulations.length,
        captured.simulationCount
      )
    });
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
    validateRequirement(
      id,
      props.statistic === 'extreme-rank' || props.reference !== undefined,
      'reference is required for maximum-deviation and integrated-squared-deviation'
    );
    validateRequirement(
      id,
      props.statistic !== 'integrated-squared-deviation' || props.integrationWeights !== undefined,
      'integrationWeights is required for integrated-squared-deviation'
    );
    for (const [path, view] of Object.entries({
      observed: props.observed,
      'output.lower': props.output.lower,
      'output.upper': props.output.upper,
      ...(props.statistic === 'extreme-rank' ? {} : {reference: props.reference!}),
      ...(props.statistic === 'integrated-squared-deviation'
        ? {integrationWeights: props.integrationWeights!}
        : {})
    })) {
      validatePointProcessView(id, path, view, ['float32'], count);
    }
    validatePointProcessView(
      id,
      'simulations',
      props.simulations,
      ['float32'],
      count * props.simulationCount
    );
    validatePointProcessView(id, 'output.statistics', props.output.statistics, ['float32'], 4);
    validatePointProcessView(id, 'output.status', props.output.status, ['uint32'], 1);
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
      validatePointProcessView(
        id,
        'output.pValueInterval',
        props.output.pValueInterval,
        ['float32'],
        2
      );
    }

    validateSpatialOutputViews(this.id, props.output, [
      [props.output.statistics, props.output.status, props.output.pValueInterval],
      {read: [props.output.status], write: [props.output.lower, props.output.upper]}
    ]);
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.observed,
      props.simulations,
      ...(props.statistic === 'extreme-rank' ? [] : [props.reference]),
      ...(props.statistic === 'integrated-squared-deviation' ? [props.integrationWeights] : [])
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
      ...(props.statistic === 'extreme-rank' ? [] : [props.reference]),
      ...(props.statistic === 'integrated-squared-deviation' ? [props.integrationWeights] : []),
      ...Object.values(props.output)
    ]);
    validateGraphChunkBindingRanges(graph, id, {
      observed: props.observed,
      simulations: props.simulations,
      ...(props.statistic === 'extreme-rank' ? {} : {reference: props.reference}),
      ...(props.statistic === 'integrated-squared-deviation'
        ? {integrationWeights: props.integrationWeights}
        : {}),
      ...Object.fromEntries(
        Object.entries(props.output).map(([name, view]) => [`output.${name}`, view])
      )
    });
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      'curve deviations/ranks',
      (simulationCount + 1) * 4,
      'reduce simulationCount'
    );
    if (props.statistic === 'extreme-rank') return createRankEnvelopeNodes(graph, props, id);
    const residuals = createTransientView(graph, `${id}-residuals`, 'float32', simulationCount + 1);
    const comparisonFlags = createTransientView(
      graph,
      `${id}-comparison-flags`,
      'float32',
      simulationCount + 1
    );
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
          {name: 'reference', view: props.reference!, type: 'f32', access: 'read'},
          ...(props.statistic === 'integrated-squared-deviation'
            ? [
                {
                  name: 'weights',
                  view: props.integrationWeights!,
                  type: 'f32' as const,
                  access: 'read' as const
                }
              ]
            : []),
          {name: 'residuals', view: residuals, type: 'f32', access: 'read_write'},
          {name: 'deviations', view: deviations, type: 'f32', access: 'read_write'}
        ],
        declarations: `${IS_FINITE_FLOAT_WGSL}\n${COMPENSATED_SUM_WGSL}`,
        body: `var statistic = 0.0;
  var accumulator = vec2f(0.0);
  var valid = true;
  for (var radius = 0u; radius < ${count}u; radius++) {
    var value = observed[observedOffset + radius];
    if (index < ${simulationCount}u) { value = simulations[simulationsOffset + index * ${count}u + radius]; }
    let center = reference[referenceOffset + radius];
    valid = valid && isFiniteFloat(value) && isFiniteFloat(center);
    ${props.statistic === 'integrated-squared-deviation' ? 'let weight = weights[weightsOffset + radius];\n    valid = valid && (bitcast<u32>(weight) > 0u && bitcast<u32>(weight) < 0x7f800000u);' : ''}
    let deviation = abs(value - center);
    ${props.statistic === 'integrated-squared-deviation' ? 'accumulator = addCompensatedValue(accumulator, weight * deviation * deviation); statistic = accumulator.x;' : 'statistic = max(statistic, deviation);'}
  }
  residuals[residualsOffset + index] = accumulator.y;
  deviations[deviationsOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), statistic, valid && isFiniteFloat(statistic));`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-compare`,
        operation: 'GPUPointPatternEnvelope',
        variant: 'residual-comparison',
        invocationCount: simulationCount + 1,
        bindings:
          props.statistic === 'integrated-squared-deviation'
            ? [
                {name: 'observed', view: props.observed, type: 'f32', access: 'read'},
                {name: 'simulations', view: props.simulations, type: 'f32', access: 'read'},
                {name: 'reference', view: props.reference!, type: 'f32', access: 'read'},
                {name: 'weights', view: props.integrationWeights!, type: 'f32', access: 'read'},
                {name: 'deviations', view: deviations, type: 'f32', access: 'read'},
                {name: 'flags', view: comparisonFlags, type: 'f32', access: 'read_write'}
              ]
            : [
                {name: 'observed', view: props.observed, type: 'f32', access: 'read'},
                {name: 'simulations', view: props.simulations, type: 'f32', access: 'read'},
                {name: 'reference', view: props.reference!, type: 'f32', access: 'read'},
                {name: 'deviations', view: deviations, type: 'f32', access: 'read'},
                {name: 'flags', view: comparisonFlags, type: 'f32', access: 'read_write'}
              ],
        declarations: `${IS_FINITE_FLOAT_WGSL}\nconst CURVE_ROWS: u32 = ${count}u;\nconst SIMULATION_COUNT: u32 = ${simulationCount}u;\n${props.statistic === 'integrated-squared-deviation' ? POINT_PATTERN_EXACT_STATISTIC_WGSL : POINT_PATTERN_EXACT_MAXIMUM_WGSL}`,
        body: `let observed = deviations[deviationsOffset + ${simulationCount}u];
  let value = deviations[deviationsOffset + index];
  let exceeds = ${props.statistic === 'integrated-squared-deviation' ? 'envelopeExactAtLeast(getEnvelopeExactStatistic(index), getEnvelopeExactStatistic(SIMULATION_COUNT))' : 'envelopeExactAtLeast(getEnvelopeExactMaximum(index), getEnvelopeExactMaximum(SIMULATION_COUNT))'};
  flags[flagsOffset + index] = select(bitcast<f32>(0x7fc00000u | (index & 0u)), select(0.0, 1.0, exceeds), isFiniteFloat(value) && isFiniteFloat(observed));`
      }),
      // The scalar test reduces 0/1 exceedance flags against an observed sentinel of 1.
      // The following display node replaces its temporary statistic with the actual rounded sum.
      createWGSLKernelNode(graph, {
        id: `${id}-test`,
        operation: 'GPUPointPatternEnvelope',
        variant: 'global-test',
        invocationCount: 1,
        bindings: [
          {name: 'deviations', view: comparisonFlags, type: 'f32', access: 'read'},
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
        id: `${id}-observed-statistic`,
        operation: 'GPUPointPatternEnvelope',
        variant: 'display-statistic',
        invocationCount: 1,
        bindings: [
          {name: 'deviations', view: deviations, type: 'f32', access: 'read'},
          {name: 'residuals', view: residuals, type: 'f32', access: 'read'},
          {name: 'status', view: props.output.status, type: 'u32', access: 'read'},
          {name: 'statistics', view: props.output.statistics, type: 'f32', access: 'read_write'}
        ],
        body: `if (status[statusOffset] == 0u) { statistics[statisticsOffset] = deviations[deviationsOffset + ${simulationCount}u] + residuals[residualsOffset + ${simulationCount}u]; }`
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

/** Caller-owned output map for {@link GPUPointPatternEnvelope}. */
export type GPUPointPatternEnvelopeOutput = GPUPointPatternEnvelopeProps['output'];
