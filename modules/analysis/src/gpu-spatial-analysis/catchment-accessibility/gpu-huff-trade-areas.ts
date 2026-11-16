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
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {
  validateCatchmentViews,
  validateCatchmentStorage
} from './catchment-accessibility-validation';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
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
   */
  tradeArea: GraphDataView<'uint32'>;
  /**
   * Optional Huff probability per demand-weights slot, aligned with `demandWeights.neighbors`:
   * `p_ij = A_j^alpha w_ij / sum_k A_k^alpha w_ik` (0 for slots past the last row, and for rows
   * whose denominator is 0). Rows sum to 1.
   */
  probabilities?: GraphDataView<'float32'>;
  /** Optional probability `p_ij` of the trade-area facility per demand location. */
  tradeAreaProbability?: GraphDataView<'float32'>;
  /**
   * Optional expected demand `E_j = sum_i P_i p_ij` captured per facility. Needs `facilityWeights`
   * and `demand`. A demand location whose `demand` is NaN or infinite is missing: it adds nothing
   * to any facility (the rule of `GPUCatchmentAccessibility`) and its `validity` row is 0.
   */
  expectedDemand?: GraphDataView<'float32'>;
  /**
   * Optional flag per demand location: 1 when its `demand` is finite, 0 when it is NaN or infinite
   * and was left out of `expectedDemand`. Needs `demand`. The trade area and the probabilities of
   * a flagged row do not read `demand` and remain valid.
   */
  validity?: GraphDataView<'uint32'>;
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
  /** Attractiveness `A_j` (floor area, capacity) per facility, non-negative. */
  attractiveness: GraphDataView<'float32'>;
  /**
   * Demand-to-facility weights: row `i` lists facilities `j` demand location `i` can reach, with
   * the distance-decay value `w_ij = f(c_ij)` as weight. Neighbor IDs index `attractiveness`.
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
 * so results are deterministic and match a sequential CPU evaluation within f32 rounding.
 * Facilities with zero attractiveness have probability 0. A non-finite `demand` is missing: it adds
 * nothing to `expectedDemand` and its `validity` row is 0, the no-data rule of
 * `GPUCatchmentAccessibility`.
 *
 * When both weights declare spatial contexts, they must be compatible.
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
    if (output.validity) {
      validateSinglePackedView(output.validity, ['uint32'], `${id} output.validity`);
      validateRequirement(id, props.demand !== undefined, 'output.validity needs demand');
      validateRequirement(
        id,
        props.demand?.length === this.demandCount && output.validity.length === this.demandCount,
        'demand and output.validity must have one row per demand location'
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
      {reads: {demand: props.demand}, writes: {validity: output.validity}}
    ]);
  }

  /** Returns `denominators`, `trade-area`, `probabilities` and `expected-demand` nodes as requested. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, facilityCount, demandCount} = this;
    const {attractiveness, demandWeights, parameters, output} = props;
    validateSpatialAnalysisPackedOutputAllocations(graph, id, output);
    validateGraphViewsBelongToGraph(id, graph, [...getInputs(props), ...getOutputs(output)]);
    validateGraphChunkBindingRanges(graph, id, {
      ...Object.fromEntries(getInputs(props).map((view, index) => [`input.${index}`, view])),
      ...output
    });
    validateCatchmentStorage(graph.device, id, 'denominators scratch', demandCount * 4);
    validateCatchmentStorage(graph.device, id, 'attraction scratch', facilityCount * 4);
    const denominators = createTransientView(graph, `${id}-denominators`, 'float32', demandCount);
    const parameterBinding = parameters
      ? [{name: 'parameters', view: parameters, type: 'f32' as const, access: 'read' as const}]
      : [];
    // The attraction pow is evaluated once per facility, not once per table slot.
    const attraction = createTransientView(graph, `${id}-attraction`, 'float32', facilityCount);
    const declarations = `const FACILITIES: u32 = ${facilityCount}u;
const DEMANDS: u32 = ${demandCount}u;
${IS_FINITE_VALUE_WGSL}
${SPATIAL_ANALYSIS_QUOTIENT_WGSL}`;
    const rowLoop = (
      inner: string
    ) => `for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < FACILITIES) {
      let share = attraction[attractionOffset + partner] * weights[weightsOffset + slot];
      ${inner}
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
      {name: 'attraction', view: attraction, type: 'f32' as const, access: 'read' as const}
    ];
    const nodes: (GPUCommandNode<Parameters> | GPUCommandNode<Parameters>[])[] = [
      getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
        id: `${id}-attraction`,
        operation: OPERATION,
        variant: 'attraction',
        bindings: [
          {name: 'attractiveness', view: attractiveness, type: 'f32', access: 'read'},
          ...parameterBinding,
          {name: 'attraction', view: attraction, type: 'f32', access: 'read_write'}
        ],
        invocationCount: facilityCount,
        body: `let value = attractiveness[attractivenessOffset + index];
  let exponent = ${parameters ? 'parameters[parametersOffset]' : '1.0'};
  attraction[attractionOffset + index] = select(0.0, pow(value, exponent), value > 0.0);`
      }),
      getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
        id: `${id}-trade-area`,
        operation: OPERATION,
        variant: 'trade-area',
        bindings: [
          ...demandBindings,
          {name: 'denominators', view: denominators, type: 'f32', access: 'read_write'},
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
        body: `var total = 0.0;
  var best = 0.0;
  var bestFacility = ${GPU_HUFF_NO_TRADE_AREA}u;
  ${rowLoop(`total += share;
      if (share > best) {
        best = share;
        bestFacility = partner;
      }`)}
  denominators[denominatorsOffset + index] = total;
  tradeArea[tradeAreaOffset + index] = select(${GPU_HUFF_NO_TRADE_AREA}u, bestFacility, total > 0.0);
  ${output.tradeAreaProbability ? 'tradeProbability[tradeProbabilityOffset + index] = select(0.0, getSpatialAnalysisQuotient(best,total), total > 0.0);' : ''}`
      })
    ];
    if (output.probabilities) {
      nodes.push(
        getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
          id: `${id}-probabilities`,
          operation: OPERATION,
          variant: 'probabilities',
          bindings: [
            ...demandBindings,
            {name: 'denominators', view: denominators, type: 'f32', access: 'read'},
            {name: 'probabilities', view: output.probabilities, type: 'f32', access: 'read_write'}
          ],
          invocationCount: demandCount,
          declarations,
          body: `let total = denominators[denominatorsOffset + index];
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    var probability = 0.0;
    if (partner < FACILITIES && total > 0.0) {
      probability = getSpatialAnalysisQuotient(attraction[attractionOffset + partner] * weights[weightsOffset + slot],total);
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
            {name: 'denominators', view: denominators, type: 'f32', access: 'read'},
            {name: 'expectedDemand', view: output.expectedDemand, type: 'f32', access: 'read_write'}
          ],
          invocationCount: facilityCount,
          declarations,
          body: `var captured = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < DEMANDS) {
      let total = denominators[denominatorsOffset + partner];
      let partnerDemand = demand[demandOffset + partner];
      // A non-finite demand is missing and adds nothing, as in GPUCatchmentAccessibility.
      if (total > 0.0 && isFiniteValue(partnerDemand)) {
        let share = getSpatialAnalysisQuotient(attraction[attractionOffset + index] * weights[weightsOffset + slot],total);
        captured += partnerDemand * share;
      }
    }
  }
  expectedDemand[expectedDemandOffset + index] = captured;`
        })
      );
    }
    if (output.validity && props.demand) {
      nodes.push(
        getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
          id: `${id}-validity`,
          operation: OPERATION,
          variant: 'validity',
          bindings: [
            {name: 'demand', view: props.demand, type: 'f32', access: 'read'},
            {name: 'validity', view: output.validity, type: 'u32', access: 'read_write'}
          ],
          invocationCount: demandCount,
          declarations: IS_FINITE_VALUE_WGSL,
          body: 'validity[validityOffset + index] = u32(isFiniteValue(demand[demandOffset + index]));'
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
    output.validity
  ];
}

/** Returns every read-only view of a Huff contributor. */
function getInputs(props: GPUHuffTradeAreasProps): (GraphDataView | undefined)[] {
  return [
    props.attractiveness,
    props.output.expectedDemand || props.output.validity ? props.demand : undefined,
    props.parameters,
    props.demandWeights.offsets,
    props.demandWeights.neighbors,
    props.demandWeights.weights,
    ...(props.output.expectedDemand
      ? [
          props.facilityWeights?.offsets,
          props.facilityWeights?.neighbors,
          props.facilityWeights?.weights
        ]
      : [])
  ];
}
