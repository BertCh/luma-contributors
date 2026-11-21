// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getSpatialAnalysisPackedKernelNodes,
  validateSpatialAnalysisPackedOutputAllocations
} from '../geographic-distribution/spatial-analysis-packed-kernel';
import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {snapshotConstructionFields} from '../../analysis-core/kernels/construction-snapshot';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  getAccessibilityWeightsLossChecks,
  validateCatchmentViews,
  validateCatchmentStorage
} from './catchment-accessibility-validation';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {SPATIAL_ANALYSIS_QUOTIENT_WGSL} from '../geographic-distribution/spatial-analysis-quotient';
import {
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  type GPUSpatialWeights,
  validateGPUSpatialWeights
} from '../spatial-weights/spatial-weights';
import {assertCompatibleGPUSpatialContexts} from '../../analysis-core/contracts/spatial-context';

const OPERATION = 'GPUHuffTradeAreas';

/** Trade-area value of a demand location that reaches no attractive facility. */
export const GPU_HUFF_NO_TRADE_AREA = 0xffffffff;

/** Number of `float32` words of the optional per-frame parameter view of {@link GPUHuffTradeAreas}. */
export const GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH = 1;

/** Per-frame settings of {@link GPUHuffTradeAreas}. */
export type GPUHuffTradeAreasSettings = {
  /** Attractiveness exponent `alpha`, positive and finite. Defaults to 1. */
  exponent?: number;
};

/**
 * Packs {@link GPUHuffTradeAreasSettings} into the `GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH` `float32`
 * words `[exponent]` read by {@link GPUHuffTradeAreas}.
 *
 * @throws If `exponent` is not a positive finite number.
 */
export function getGPUHuffTradeAreasParameterValues(
  settings: GPUHuffTradeAreasSettings = {}
): Float32Array {
  const exponent = settings.exponent ?? 1;
  if (!Number.isFinite(Math.fround(exponent)) || !(Math.fround(exponent) > 0)) {
    throw new Error('GPUHuffTradeAreas exponent must be positive and finite');
  }
  return Float32Array.of(exponent);
}

/** Caller-owned outputs of {@link GPUHuffTradeAreas}. */
export type GPUHuffTradeAreasOutput = {
  /**
   * Trade area per demand location: the facility with the highest probability (the lowest ID wins
   * ties), or {@link GPU_HUFF_NO_TRADE_AREA} when none is reachable.
   * Also NO_TRADE_AREA when supplied demand-weights loss flags are nonzero.
   */
  tradeArea: GraphDataView<'uint32'>;
  /**
   * Optional Huff probability per demand-weights slot, aligned with `demandWeights.neighbors`:
   * `p_ij = A_j^alpha w_ij / sum_k A_k^alpha w_ik` (0 for slots past the last row, and for rows
   * whose denominator is 0). Rows sum to 1. A supplied nonzero demand-weights overflow or
   * candidateOverflow suppresses the whole view with NaN, including padding.
   */
  probabilities?: GraphDataView<'float32'>;
  /** Optional probability `p_ij` of the trade-area facility per demand location; NaN after forward weights loss. */
  tradeAreaProbability?: GraphDataView<'float32'>;
  /**
   * Optional expected demand `E_j = sum_i P_i p_ij` captured per facility. Needs `facilityWeights`
   * and `demand`. A demand location whose `demand` is NaN or infinite is missing: it adds nothing
   * to any facility (the rule of `GPUCatchmentAccessibility`) and its `demandValidity` row is 0.
   * Nonzero supplied overflow/candidateOverflow on either weights table suppresses this view
   * with NaN. Contributions incorporate demand scale before rounding a small share to float32.
   */
  expectedDemand?: GraphDataView<'float32'>;
  /**
   * Optional flag per demand location: 1 when its `demand` is finite, 0 when it is NaN or infinite
   * and was left out of `expectedDemand`. Needs `demand`. The trade area and the probabilities of
   * a flagged row do not read `demand` and remain valid.
   */
  demandValidity?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUHuffTradeAreas}.
 *
 * Compile-time: view lengths, and which optional views and outputs exist. Per-frame: every view's
 * contents, including the attractiveness exponent in `parameters`.
 */
export type GPUHuffTradeAreasProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'huff-trade-areas'`. */
  id?: string;
  /**
   * Attractiveness `A_j` (floor area, capacity) per facility, positive and finite to count. A
   * facility whose attractiveness is zero, negative, NaN or infinite is excluded from every row
   * (probability 0, never the trade area, no captured demand).
   */
  attractiveness: GraphDataView<'float32'>;
  /**
   * Demand-to-facility weights: row `i` lists facilities `j` demand location `i` can reach, with
   * the distance-decay value `w_ij = f(c_ij)` as weight. Neighbor IDs index `attractiveness`.
   * A slot whose weight is not positive and finite is excluded like a zero attractiveness.
   */
  demandWeights: GPUSpatialWeights;
  /**
   * Facility-to-demand weights for `expectedDemand`: row `j` lists the demand locations of facility
   * `j` with `w_ji`. Required only when `expectedDemand` is requested; it must be the transpose of
   * `demandWeights`, and when `demand` and `attractiveness` have equal length with symmetric
   * weights `demandWeights` itself may be passed.
   */
  facilityWeights?: GPUSpatialWeights;
  /** Demand `P_i` per demand location. Required with `expectedDemand`. */
  demand?: GraphDataView<'float32'>;
  /**
   * Optional per-frame parameters, `GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH` `float32` words written
   * with {@link getGPUHuffTradeAreasParameterValues}. Without it `alpha = 1`.
   */
  parameters?: GraphDataView<'float32'>;
  /** Caller-owned outputs. */
  output: GPUHuffTradeAreasOutput;
};

/**
 * Huff gravity-model trade areas on any {@link GPUSpatialWeights}: the probability that demand
 * location `i` patronises facility `j`, `p_ij = A_j^alpha w_ij / sum_k A_k^alpha w_ik`, the modal
 * facility per demand location (its trade area), and the expected demand captured by each
 * facility.
 *
 * The weights carry the decay and the catchment cut-off, so a Euclidean kernel from
 * `GPUNeighborSearch` gives the classic distance-decay Huff model. Rows accumulate in slot order,
 * so results are deterministic within f32 rounding. Relative log scores are computed from the
 * original mantissas before exponentiation, preserving close values when attractiveness units change.
 * Facilities with zero attractiveness have probability 0; a slot whose attractiveness or weight is
 * negative, NaN or infinite is invalid and excluded the same way. A non-finite `demand` is missing: it adds
 * nothing to `expectedDemand` and its `demandValidity` row is 0, the no-data rule of
 * `GPUCatchmentAccessibility`.
 *
 * When both weights declare spatial contexts, they must be compatible.
 * Supplied overflow/candidateOverflow on demandWeights suppresses trade areas to NO_TRADE_AREA
 * and their probabilities to NaN. Either weights table's loss flags suppress expectedDemand to
 * NaN. A supplied zero status.complete independently suppresses the same dependent outputs.
 * Demand validity still describes only whether demand is finite. Clear negative loss flags and
 * write nonzero completeness once the input is whole to restore
 * ordinary results on the next submission without recompiling.
 *
 * Spatial context: this operator reads no positions, only the weights, so it has no
 * `spatialContext` prop; distance units and the coordinate space are those of the weights' producer.
 */
export class GPUHuffTradeAreas implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUHuffTradeAreasProps;
  /** Number of facilities. */
  readonly facilityCount: number;
  /** Number of demand locations. */
  readonly demandCount: number;

  constructor(props: GPUHuffTradeAreasProps) {
    const id = props.id ?? 'huff-trade-areas';
    validateRequirement(
      id,
      !('validity' in props.output),
      'output.validity was removed. Use output.demandValidity'
    );
    props = {
      ...props,
      demandWeights: {
        ...props.demandWeights,
        status: props.demandWeights.status ? {...props.demandWeights.status} : undefined
      },
      facilityWeights: props.facilityWeights
        ? {
            ...props.facilityWeights,
            status: props.facilityWeights.status ? {...props.facilityWeights.status} : undefined
          }
        : undefined,
      output: snapshotConstructionFields(props.output, [
        'tradeArea',
        'probabilities',
        'tradeAreaProbability',
        'expectedDemand',
        'demandValidity'
      ])
    };
    this.id = id;
    this.props = props;
    this.demandCount = validateGPUSpatialWeights(id, props.demandWeights, 'demandWeights');
    this.facilityCount = props.attractiveness.length;
    if (props.demandWeights?.spatialContext && props.facilityWeights?.spatialContext) {
      assertCompatibleGPUSpatialContexts(
        id,
        props.demandWeights.spatialContext,
        props.facilityWeights.spatialContext
      );
    }
    const {output} = props;
    validateSinglePackedView(props.attractiveness, ['float32'], `${id} attractiveness`);
    validateSinglePackedView(output.tradeArea, ['uint32'], `${id} output.tradeArea`);
    for (const [name, view] of [
      ['demand', props.demand],
      ['parameters', props.parameters],
      ['output.probabilities', output.probabilities],
      ['output.tradeAreaProbability', output.tradeAreaProbability],
      ['output.expectedDemand', output.expectedDemand]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    if (output.tradeArea.length !== this.demandCount) {
      throw new Error(`${id} output.tradeArea length must equal the demand count`);
    }
    if (output.tradeAreaProbability && output.tradeAreaProbability.length !== this.demandCount) {
      throw new Error(`${id} output.tradeAreaProbability length must equal the demand count`);
    }
    if (
      output.probabilities &&
      output.probabilities.length !== props.demandWeights.neighbors.length
    ) {
      throw new Error(
        `${id} output.probabilities length must equal demandWeights.neighbors length`
      );
    }
    if (props.parameters && props.parameters.length < GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH) {
      throw new Error(`${id} parameters must hold ${GPU_HUFF_TRADE_AREAS_PARAMETER_LENGTH} word`);
    }
    if (output.demandValidity) {
      validateSinglePackedView(output.demandValidity, ['uint32'], `${id} output.demandValidity`);
      validateRequirement(id, props.demand !== undefined, 'output.demandValidity needs demand');
      validateRequirement(
        id,
        props.demand?.length === this.demandCount &&
          output.demandValidity.length === this.demandCount,
        'demand and output.demandValidity must have one row per demand location'
      );
    }
    if (output.expectedDemand) {
      if (!props.facilityWeights || !props.demand) {
        throw new Error(`${id} output.expectedDemand needs facilityWeights and demand`);
      }
      if (output.expectedDemand.length !== this.facilityCount) {
        throw new Error(`${id} output.expectedDemand length must equal the facility count`);
      }
      if (props.demand.length !== this.demandCount) {
        throw new Error(`${id} demand length must equal the demandWeights row count`);
      }
      const facilityRows = validateGPUSpatialWeights(id, props.facilityWeights, 'facilityWeights');
      if (facilityRows !== this.facilityCount) {
        throw new Error(`${id} facilityWeights row count must equal the facility count`);
      }
    }
    const inputs = Object.fromEntries(
      getInputs(props).map((view, index) => [`input.${index}`, view])
    );
    const demandInputs = {
      offsets: props.demandWeights.offsets,
      neighbors: props.demandWeights.neighbors,
      weights: props.demandWeights.weights
    };
    const facilityInputs = props.output.expectedDemand
      ? {
          offsets: props.facilityWeights?.offsets,
          neighbors: props.facilityWeights?.neighbors,
          weights: props.facilityWeights?.weights,
          demand: props.demand
        }
      : {};
    validateCatchmentViews(id, inputs, output, [
      {
        reads: demandInputs,
        writes: {tradeArea: output.tradeArea, tradeAreaProbability: output.tradeAreaProbability}
      },
      {reads: demandInputs, writes: {probabilities: output.probabilities}},
      {reads: facilityInputs, writes: {expectedDemand: output.expectedDemand}},
      {reads: {demand: props.demand}, writes: {demandValidity: output.demandValidity}}
    ]);
  }

  /** Returns attraction, trade-area, probability clearing/scatter and expected-demand nodes as requested. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, facilityCount, demandCount} = this;
    const {attractiveness, demandWeights, parameters} = props;
    const {output} = props;
    validateSpatialAnalysisPackedOutputAllocations(graph, id, output);
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(output)]);
    validateGraphChunkBindingRanges(graph, id, {
      ...Object.fromEntries(getInputs(props).map((view, index) => [`input.${index}`, view])),
      ...output
    });
    validateCatchmentStorage(graph.device, id, 'normalization scratch', demandCount * 4 * 4);
    validateCatchmentStorage(graph.device, id, 'attraction scratch', facilityCount * 4);
    // Per demand row: reference attractiveness, reference weight, maximum log score, scaled total.
    const normalization = createTransientView(
      graph,
      `${id}-normalization`,
      'float32',
      demandCount * 4
    );
    const parameterBinding = parameters
      ? [{name: 'parameters', view: parameters, type: 'f32' as const, access: 'read' as const}]
      : [];
    // Keep the original values: rounding absolute logarithms loses nearby attractiveness values.
    const attraction = createTransientView(graph, `${id}-attraction`, 'float32', facilityCount);
    const declarations = `const FACILITIES: u32 = ${facilityCount}u;
const DEMANDS: u32 = ${demandCount}u;
${IS_FINITE_VALUE_WGSL}
${SPATIAL_ANALYSIS_QUOTIENT_WGSL}
// Mantissas are in [0.5, 1); their difference is exact. The atanh series computes the log
// ratio without rounding away near-equal values or depending on their common power-of-two scale.
fn getHuffLogRatio(value: f32, reference: f32) -> f32 {
  let first = frexp(value);
  let second = frexp(reference);
  var firstFraction = first.fract;
  var differencePower = first.exp - second.exp;
  // Put near-equal values on the same binary scale, including across a power-of-two boundary.
  // The remaining mantissa ratio stays in [0.5, 2], so |difference| <= 1/3.
  if (differencePower < 0 && firstFraction > second.fract) {
    firstFraction *= 0.5;
    differencePower += 1;
  } else if (differencePower > 0 && firstFraction < second.fract) {
    firstFraction *= 2.0;
    differencePower -= 1;
  }
  let difference = (firstFraction - second.fract) / (firstFraction + second.fract);
  let square = difference * difference;
  let series = 1.0 + square * (1.0 / 3.0 + square * (1.0 / 5.0 + square * (1.0 / 7.0 +
    square * (1.0 / 9.0 + square * (1.0 / 11.0 + square * (1.0 / 13.0 + square / 15.0))))));
  return 2.0 * difference * series + f32(differencePower) * 0.6931471805599453;
}
fn getHuffScore(value: f32, weight: f32, reference: f32, referenceWeight: f32, exponent: f32) -> f32 {
  if (exponent == 1.0) {
    // Bounded products preserve exact weighted ties at the default exponent, even if A*w
    // itself would overflow. Multiplying both attractiveness values by 2^k leaves this unchanged.
    let first = frexp(value);
    let second = frexp(weight);
    let firstReference = frexp(reference);
    let secondReference = frexp(referenceWeight);
    return getHuffLogRatio(first.fract * second.fract, firstReference.fract * secondReference.fract) +
      f32(first.exp + second.exp - firstReference.exp - secondReference.exp) * 0.6931471805599453;
  }
  return exponent * getHuffLogRatio(value, reference) + getHuffLogRatio(weight, referenceWeight);
}
// A facility slot counts only with positive finite attractiveness and weight. Infinity would pass a
// positive test and reach frexp, whose result is undefined for it; NaN fails the comparison.
fn isHuffCandidate(value: f32, weight: f32) -> bool {
  return value > 0.0 && weight > 0.0 && isFiniteValue(value) && isFiniteValue(weight);
}
fn getHuffShare(value: f32, weight: f32, reference: f32, referenceWeight: f32, maximum: f32, exponent: f32) -> f32 {
  if (!isHuffCandidate(value, weight)) { return 0.0; }
  return exp(min(getHuffScore(value, weight, reference, referenceWeight, exponent) - maximum, 0.0));
}
fn getHuffContribution(value: f32, weight: f32, reference: f32, referenceWeight: f32, maximum: f32, total: f32, exponent: f32, demand: f32) -> f32 {
  if (!isHuffCandidate(value, weight) || demand == 0.0) { return 0.0; }
  let score = min(getHuffScore(value, weight, reference, referenceWeight, exponent) - maximum, 0.0);
  // Below exp(-256), even the largest finite demand cannot produce a float32 subnormal.
  // This also keeps conversion to an integer exponent finite and in range.
  if (score < -256.0) { return 0.0; }
  let power = i32(floor(score * 1.4426950408889634));
  let fraction = exp(score - f32(power) * 0.6931471805599453);
  let demandParts = frexp(demand);
  return getScaledQuotient(demandParts.fract * fraction, total, 1.0, demandParts.exp + power);
}`;
    const rowLoop = (
      inner: string
    ) => `for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < FACILITIES) {
      let value = attraction[attractionOffset + partner];
      let weight = weights[weightsOffset + slot];
      if (isHuffCandidate(value, weight)) {
      ${inner}
      }
    }
  }`;
    const demandBindings = [
      {name: 'offsets', view: demandWeights.offsets, type: 'u32' as const, access: 'read' as const},
      {
        name: 'neighbors',
        view: demandWeights.neighbors,
        type: 'u32' as const,
        access: 'read' as const
      },
      {name: 'weights', view: demandWeights.weights, type: 'f32' as const, access: 'read' as const},
      {name: 'attraction', view: attraction, type: 'f32' as const, access: 'read' as const},
      ...parameterBinding
    ];
    const nodes: (GPUCommandNode<Parameters> | GPUCommandNode<Parameters>[])[] = [
      getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
        id: `${id}-attraction`,
        operation: OPERATION,
        variant: 'attraction',
        bindings: [
          {name: 'attractiveness', view: attractiveness, type: 'f32', access: 'read'},
          {name: 'attraction', view: attraction, type: 'f32', access: 'read_write'}
        ],
        invocationCount: facilityCount,
        body: 'attraction[attractionOffset + index] = attractiveness[attractivenessOffset + index];'
      }),
      getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
        id: `${id}-trade-area`,
        operation: OPERATION,
        variant: 'trade-area',
        bindings: [
          ...demandBindings,
          {name: 'normalization', view: normalization, type: 'f32', access: 'read_write'},
          {name: 'tradeArea', view: output.tradeArea, type: 'u32', access: 'read_write'},
          ...(output.tradeAreaProbability
            ? [
                {
                  name: 'tradeProbability',
                  view: output.tradeAreaProbability,
                  type: 'f32' as const,
                  access: 'read_write' as const
                }
              ]
            : [])
        ],
        invocationCount: demandCount,
        declarations,
        body: `let exponent = ${parameters ? 'parameters[parametersOffset]' : '1.0'};
  var reference = 0.0;
  var referenceWeight = 0.0;
  ${rowLoop('reference = max(reference, value); referenceWeight = max(referenceWeight, weight);')}
  var maximum = -3.4e38;
  ${rowLoop('maximum = max(maximum, getHuffScore(value, weight, reference, referenceWeight, exponent));')}
  var total = 0.0;
  var best = 0.0;
  var bestFacility = ${GPU_HUFF_NO_TRADE_AREA}u;
  ${rowLoop(`let share = getHuffShare(value, weight, reference, referenceWeight, maximum, exponent);
      total += share;
      if (share > best || (share == best && share > 0.0 && partner < bestFacility)) {
        best = share;
        bestFacility = partner;
      }`)}
  normalization[normalizationOffset + 4u * index] = reference;
  normalization[normalizationOffset + 4u * index + 1u] = referenceWeight;
  normalization[normalizationOffset + 4u * index + 2u] = maximum;
  normalization[normalizationOffset + 4u * index + 3u] = total;
  tradeArea[tradeAreaOffset + index] = select(${GPU_HUFF_NO_TRADE_AREA}u, bestFacility, total > 0.0);
  ${output.tradeAreaProbability ? 'tradeProbability[tradeProbabilityOffset + index] = select(0.0, getSpatialAnalysisQuotient(best,total), total > 0.0);' : ''}`
      })
    ];
    if (output.probabilities) {
      nodes.push(
        getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
          id: `${id}-clear-probabilities`,
          operation: OPERATION,
          variant: 'clear-probabilities',
          bindings: [
            {name: 'probabilities', view: output.probabilities, type: 'f32', access: 'read_write'}
          ],
          invocationCount: output.probabilities.length,
          body: 'probabilities[probabilitiesOffset + index] = 0.0;'
        }),
        getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
          id: `${id}-probabilities`,
          operation: OPERATION,
          variant: 'probabilities',
          bindings: [
            ...demandBindings,
            {name: 'normalization', view: normalization, type: 'f32', access: 'read'},
            {name: 'probabilities', view: output.probabilities, type: 'f32', access: 'read_write'}
          ],
          invocationCount: demandCount,
          declarations,
          body: `let reference = normalization[normalizationOffset + 4u * index];
  let referenceWeight = normalization[normalizationOffset + 4u * index + 1u];
  let maximum = normalization[normalizationOffset + 4u * index + 2u];
  let total = normalization[normalizationOffset + 4u * index + 3u];
  let exponent = ${parameters ? 'parameters[parametersOffset]' : '1.0'};
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    var probability = 0.0;
    if (partner < FACILITIES && total > 0.0) {
      probability = getSpatialAnalysisQuotient(getHuffShare(attraction[attractionOffset + partner], weights[weightsOffset + slot], reference, referenceWeight, maximum, exponent),total);
    }
    probabilities[probabilitiesOffset + slot] = probability;
  }`
        })
      );
    }
    if (output.expectedDemand && props.facilityWeights && props.demand) {
      const facilityWeights = props.facilityWeights;
      nodes.push(
        getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
          id: `${id}-expected-demand`,
          operation: OPERATION,
          variant: 'expected-demand',
          bindings: [
            {name: 'offsets', view: facilityWeights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: facilityWeights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: facilityWeights.weights, type: 'f32', access: 'read'},
            {name: 'attraction', view: attraction, type: 'f32', access: 'read'},
            {name: 'demand', view: props.demand, type: 'f32', access: 'read'},
            {name: 'normalization', view: normalization, type: 'f32', access: 'read'},
            ...parameterBinding,
            {name: 'expectedDemand', view: output.expectedDemand, type: 'f32', access: 'read_write'}
          ],
          invocationCount: facilityCount,
          declarations,
          body: `let exponent = ${parameters ? 'parameters[parametersOffset]' : '1.0'};
  var captured = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < DEMANDS) {
      let reference = normalization[normalizationOffset + 4u * partner];
      let referenceWeight = normalization[normalizationOffset + 4u * partner + 1u];
      let maximum = normalization[normalizationOffset + 4u * partner + 2u];
      let total = normalization[normalizationOffset + 4u * partner + 3u];
      let partnerDemand = demand[demandOffset + partner];
      // A non-finite demand is missing and adds nothing, as in GPUCatchmentAccessibility.
      if (total > 0.0 && isFiniteValue(partnerDemand)) {
        captured += getHuffContribution(attraction[attractionOffset + index], weights[weightsOffset + slot], reference, referenceWeight, maximum, total, exponent, partnerDemand);
      }
    }
  }
  expectedDemand[expectedDemandOffset + index] = captured;`
        })
      );
    }
    if (output.demandValidity && props.demand) {
      nodes.push(
        getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
          id: `${id}-validity`,
          operation: OPERATION,
          variant: 'validity',
          bindings: [
            {name: 'demand', view: props.demand, type: 'f32', access: 'read'},
            {name: 'validity', view: output.demandValidity, type: 'u32', access: 'read_write'}
          ],
          invocationCount: demandCount,
          declarations: IS_FINITE_VALUE_WGSL,
          body: 'validity[validityOffset + index] = u32(isFiniteValue(demand[demandOffset + index]));'
        })
      );
    }
    // Incomplete forward weights invalidate the normalized row and every derived capture.
    // Incomplete reverse weights affect only expected demand. Never publish either partial sum.
    const forwardLoss = getAccessibilityWeightsLossChecks(id, demandWeights);
    const reverseLoss =
      output.expectedDemand && props.facilityWeights
        ? getAccessibilityWeightsLossChecks(id, props.facilityWeights)
        : [];
    for (const [name, view, flags, type, value] of [
      ['tradeArea', output.tradeArea, forwardLoss, 'u32', `${GPU_HUFF_NO_TRADE_AREA}u`],
      ['probabilities', output.probabilities, forwardLoss, 'f32', 'getNaN()'],
      ['tradeAreaProbability', output.tradeAreaProbability, forwardLoss, 'f32', 'getNaN()'],
      ['expectedDemand', output.expectedDemand, [...forwardLoss, ...reverseLoss], 'f32', 'getNaN()']
    ] as const) {
      if (!view || flags.length === 0) continue;
      nodes.push(
        getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
          id: `${id}-suppress-${name}`,
          operation: OPERATION,
          variant: 'incomplete-input',
          bindings: [
            ...flags.map(({flag}, index) => ({
              name: `loss${index}`,
              view: flag,
              type: 'u32' as const,
              access: 'read' as const
            })),
            {name: 'result', view, type, access: 'read_write'}
          ],
          invocationCount: view.length,
          declarations: type === 'f32' ? GET_NAN_WGSL : undefined,
          body: `if (${flags.map(({test}, index) => `loss${index}[loss${index}Offset] ${test}`).join(' || ')}) {
  result[resultOffset + index] = ${value};
}`
        })
      );
    }
    return nodes.flat();
  }
}

/** Returns every caller-owned output view of a Huff contributor. */
function getOutputs(output: GPUHuffTradeAreasOutput): (GraphDataView | undefined)[] {
  return [
    output.probabilities,
    output.tradeArea,
    output.tradeAreaProbability,
    output.expectedDemand,
    output.demandValidity
  ];
}

/** Returns every read-only view of a Huff contributor. */
function getInputs(props: GPUHuffTradeAreasProps): (GraphDataView | undefined)[] {
  return [
    props.attractiveness,
    props.output.expectedDemand || props.output.demandValidity ? props.demand : undefined,
    props.parameters,
    props.demandWeights.offsets,
    props.demandWeights.neighbors,
    props.demandWeights.weights,
    ...getAccessibilityWeightsLossChecks(props.id ?? 'huff-trade-areas', props.demandWeights).map(
      check => check.flag
    ),
    ...(props.output.expectedDemand
      ? [
          props.facilityWeights?.offsets,
          props.facilityWeights?.neighbors,
          props.facilityWeights?.weights,
          ...(props.facilityWeights
            ? getAccessibilityWeightsLossChecks(
                props.id ?? 'huff-trade-areas',
                props.facilityWeights
              ).map(check => check.flag)
            : [])
        ]
      : [])
  ];
}
