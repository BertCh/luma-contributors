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
import {
  GET_NAN_WGSL,
  IS_FINITE_BITS_WGSL,
  IS_FINITE_VALUE_WGSL
} from '../../analysis-core/kernels/wgsl-common';
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

const OPERATION = 'GPUCatchmentAccessibility';

/** Floating catchment method of {@link GPUCatchmentAccessibility}. */
export type GPUCatchmentMethod = '2sfca' | '3sfca';

/** Caller-owned outputs of {@link GPUCatchmentAccessibility}. */
export type GPUCatchmentAccessibilityOutput = {
  /** Accessibility `A_i` per demand location; NaN throughout after supplied loss on either weights table. */
  accessibility: GraphDataView<'float32'>;
  /**
   * Optional supply-to-demand ratio per facility: `R_j = S_j / sum_i P_i w_ji` for `'2sfca'`, with
   * `P_i / T_i` in place of `P_i` for `'3sfca'`. 0 when no demand is in range.
   * NaN after facility-weights loss, or after demand-weights loss for 3SFCA.
   */
  ratios?: GraphDataView<'float32'>;
  /**
   * Optional count of facilities each demand location reaches: the slots of its demand-weights row
   * whose neighbor is a facility, whatever their weight. 0xffffffff means unknown after supplied
   * demand-weights loss; facility-weights loss alone does not invalidate this forward count.
   */
  reachableFacilities?: GraphDataView<'uint32'>;
  /**
   * Optional flag per demand location: 1 when its `demand` is finite, 0 when it is NaN or infinite
   * and was left out of every facility's ratio. The accessibility of a flagged row does not read
   * its own demand and remains valid. The same output as `GPUHuffTradeAreas`.
   */
  demandValidity?: GraphDataView<'uint32'>;
};

/**
 * Properties for {@link GPUCatchmentAccessibility}.
 *
 * Compile-time: `method`, view lengths and which optional outputs exist. Per-frame: the contents of
 * `supply`, `demand` and both weights.
 */
export type GPUCatchmentAccessibilityProps = {
  /** Prefix for generated node and transient IDs. Defaults to `'catchment-accessibility'`. */
  id?: string;
  /**
   * `'2sfca'` (Luo and Wang 2003) or `'3sfca'` (Wan, Zou and Sternberg 2012, which splits each
   * demand location's population over the facilities it can reach by selection weights).
   */
  method: GPUCatchmentMethod;
  /** Supply (beds, doctors, capacity) per facility, one row per facility. */
  supply: GraphDataView<'float32'>;
  /** Demand (population) per demand location, one row per demand location. */
  demand: GraphDataView<'float32'>;
  /**
   * Facility-to-demand weights: row `j` lists the demand locations `i` inside facility `j`'s
   * catchment, and `w_ji` is the distance-decay value `f(c_ji)` (1 for the classic binary
   * catchment, a kernel value otherwise). Producers such as `GPUNeighborSearch` write it. Its row
   * count is the facility count and its neighbor IDs index `demand`.
   */
  facilityWeights: GPUSpatialWeights;
  /**
   * Demand-to-facility weights: row `i` lists the facilities `j` that demand location `i` can
   * reach, with weight `w_ij`. Defaults to `facilityWeights`, which is correct when facilities and
   * demand locations are the same set of places and the weights are symmetric (a self-join
   * distance band or kernel, `w_ij = w_ji`). Supply a separate cross weights otherwise, for example
   * the transpose or a second `GPUNeighborSearch` from demand to facilities.
   */
  demandWeights?: GPUSpatialWeights;
  /**
   * Optional one-row completeness word of the costs behind both weights tables, for example the
   * `output.status.complete` of `GPUNetworkCostWeights` (fed by the cost matrix
   * `status.converged` and `costLimit`). A finite weights table cannot show a route the matrix
   * did not finish or a horizon below the catchment cutoff, so when this word is 0 the dependent
   * results are suppressed exactly like supplied weights loss: `accessibility` and `ratios` become
   * NaN and `reachableFacilities` 0xffffffff. `demandValidity` is unaffected (NET-101).
   */
  costsComplete?: GraphDataView<'uint32'>;
  /** Caller-owned outputs. */
  output: GPUCatchmentAccessibilityOutput;
};

/**
 * Two-step and three-step floating catchment area accessibility on any {@link GPUSpatialWeights}
 * (Euclidean kernel, network-cost kernel, binary band). This is the library's one floating
 * catchment kernel.
 *
 * With `w` the decay weights, facility supply `S_j` and demand `P_i`:
 * - `'2sfca'`: `R_j = S_j / sum_i P_i w_ji` over the demand in facility `j`'s row (0 when the sum is
 *   0), then `A_i = sum_j w_ij R_j` over the facilities in demand row `i`.
 * - `'3sfca'` (Wan, Zou and Sternberg 2012; PySAL `access` `three_stage_fca`): selection weights
 *   `G_ij = w_ij / T_i` with `T_i = sum_k w_ik` over demand row `i` (the demand weights), applied
 *   together with the decay. The ratio step reads facility row `j` (the facility weights):
 *   `R_j = S_j / sum_i P_i w_ji^2 / T_i`. The score step reads demand row `i`:
 *   `A_i = sum_j (w_ij^2 / T_i) R_j`. Both are `G_ij W_ij` only when the demand weights are the
 *   transpose of the facility weights. Demand locations with `T_i = 0` are skipped.
 *
 * Both methods conserve supply, `sum_i P_i A_i = sum_j S_j` over facilities with demand in range,
 * whenever the demand weights are the transpose of the facility weights, for any decay function.
 *
 * Each output row accumulates its neighbors in slot order in one invocation, so results are
 * deterministic and match a sequential CPU sum within f32 rounding. Neighbor IDs outside the
 * partner row space are skipped, and a non-finite demand or ratio (from a non-finite supply) is
 * missing: it adds nothing to a sum. The optional `demandValidity` output flags finite demand.
 *
 * Network costs: `GPUNetworkCostWeights` turns a network cost matrix into the facility weights and
 * their transpose, which is how the `catchment` option of `GPUNetworkAccessibility` runs.
 *
 * When both weights declare spatial contexts, they must be compatible.
 * Pass each producer's described outputWeights to retain its loss flags. Supplied nonzero
 * overflow/candidateOverflow, or zero status.complete, suppresses dependent float results with
 * NaN and forward counts with 0xffffffff. Completeness is read independently of candidate loss.
 * Demand validity still describes finite demand only. Clear negative loss flags and write nonzero
 * completeness once the input is whole to restore results on the next graph submission.
 *
 * Spatial context: this operator reads no positions, only the weights, so it has no
 * `spatialContext` prop. Distance units and the coordinate space (planar, or great-circle meters)
 * are those of the weights' producer (`GPUNeighborSearch`, a network cost kernel); the operator is
 * unit-free.
 */
export class GPUCatchmentAccessibility implements GPUCommandNodeProducer {
  /** Prefix for every node and transient ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUCatchmentAccessibilityProps;
  /** Number of facilities. */
  readonly facilityCount: number;
  /** Number of demand locations. */
  readonly demandCount: number;

  constructor(props: GPUCatchmentAccessibilityProps) {
    const id = props.id ?? 'catchment-accessibility';
    validateRequirement(
      id,
      !('validity' in props.output),
      'output.validity was removed. Use output.demandValidity'
    );
    props = {
      ...props,
      facilityWeights: {
        ...props.facilityWeights,
        status: props.facilityWeights.status ? {...props.facilityWeights.status} : undefined
      },
      demandWeights: props.demandWeights
        ? {
            ...props.demandWeights,
            status: props.demandWeights.status ? {...props.demandWeights.status} : undefined
          }
        : undefined,
      output: snapshotConstructionFields(props.output, [
        'accessibility',
        'ratios',
        'reachableFacilities',
        'demandValidity'
      ])
    };
    this.id = id;
    this.props = props;
    if (props.method !== '2sfca' && props.method !== '3sfca') {
      throw new Error(`${id} method must be '2sfca' or '3sfca'`);
    }
    this.facilityCount = validateGPUSpatialWeights(id, props.facilityWeights, 'facilityWeights');
    this.demandCount = props.demandWeights
      ? validateGPUSpatialWeights(id, props.demandWeights, 'demandWeights')
      : this.facilityCount;
    if (!props.demandWeights && props.supply.length !== props.demand.length) {
      throw new Error(`${id} needs demandWeights unless supply and demand have equal length`);
    }
    if (props.demandWeights?.spatialContext && props.facilityWeights?.spatialContext) {
      assertCompatibleGPUSpatialContexts(
        id,
        props.demandWeights.spatialContext,
        props.facilityWeights.spatialContext
      );
    }
    const {output} = props;
    for (const [name, view] of [
      ['supply', props.supply],
      ['demand', props.demand],
      ['output.accessibility', output.accessibility],
      ['output.ratios', output.ratios]
    ] as const) {
      if (view) {
        validateSinglePackedView(view, ['float32'], `${id} ${name}`);
      }
    }
    if (output.reachableFacilities) {
      validateSinglePackedView(
        output.reachableFacilities,
        ['uint32'],
        `${id} output.reachableFacilities`
      );
    }
    if (props.supply.length !== this.facilityCount) {
      throw new Error(`${id} supply length must equal the facilityWeights row count`);
    }
    if (props.demand.length !== this.demandCount) {
      throw new Error(`${id} demand length must equal the demandWeights row count`);
    }
    if (output.accessibility.length !== this.demandCount) {
      throw new Error(`${id} output.accessibility length must equal the demand count`);
    }
    if (output.ratios && output.ratios.length !== this.facilityCount) {
      throw new Error(`${id} output.ratios length must equal the facility count`);
    }
    if (output.reachableFacilities && output.reachableFacilities.length !== this.demandCount) {
      throw new Error(`${id} output.reachableFacilities length must equal the demand count`);
    }
    if (props.costsComplete) {
      validateSinglePackedView(props.costsComplete, ['uint32'], `${id} costsComplete`);
      validateRequirement(
        id,
        props.costsComplete.length === 1,
        'costsComplete must contain exactly one row'
      );
    }
    if (output.demandValidity) {
      validateSinglePackedView(output.demandValidity, ['uint32'], `${id} output.demandValidity`);
      validateRequirement(
        id,
        output.demandValidity.length === this.demandCount,
        'output.demandValidity must have one row per demand location'
      );
    }
    const weightViews = getWeightViews(props);
    const inputs = {
      supply: props.supply,
      demand: props.demand,
      ...Object.fromEntries(weightViews.map((view, index) => [`weights.${index}`, view]))
    };
    const outputViews = {...output};
    const facilityInputs = {
      offsets: props.facilityWeights.offsets,
      neighbors: props.facilityWeights.neighbors,
      weights: props.facilityWeights.weights
    };
    const demandWeights = props.demandWeights ?? props.facilityWeights;
    const demandInputs = {
      offsets: demandWeights.offsets,
      neighbors: demandWeights.neighbors,
      weights: demandWeights.weights
    };
    validateCatchmentViews(id, inputs, outputViews, [
      {
        reads: {demand: props.demand, supply: props.supply, ...facilityInputs},
        writes: {ratios: output.ratios}
      },
      {
        reads: {...demandInputs, ratios: output.ratios},
        writes: {accessibility: output.accessibility}
      },
      {
        reads: {offsets: demandWeights.offsets, neighbors: demandWeights.neighbors},
        writes: {reachableFacilities: output.reachableFacilities}
      },
      {reads: {demand: props.demand}, writes: {demandValidity: output.demandValidity}}
    ]);
  }

  /**
   * Returns the nodes in order: `selection` (3SFCA only), `ratios`, `scores`, and `reachable` and
   * `demandValidity` when requested.
   */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, facilityCount, demandCount} = this;
    const {supply, demand, facilityWeights} = props;
    const {output} = props;
    const {accessibility} = output;
    const demandWeights = props.demandWeights ?? facilityWeights;
    validateSpatialAnalysisPackedOutputAllocations(graph, id, output);
    validateGraphViewsBelongToGraph(id, graph, [
      supply,
      demand,
      accessibility,
      output.ratios,
      output.reachableFacilities,
      output.demandValidity,
      ...getWeightViews(props)
    ]);
    validateGraphChunkBindingRanges(graph, id, {
      supply,
      demand,
      ...output,
      ...Object.fromEntries(getWeightViews(props).map((view, index) => [`weights.${index}`, view]))
    });
    if (!output.ratios)
      validateCatchmentStorage(graph.device, id, 'ratios scratch', facilityCount * 4);
    if (props.method === '3sfca')
      validateCatchmentStorage(graph.device, id, 'selection scratch', demandCount * 4);
    const isThreeStep = props.method === '3sfca';
    const ratios =
      output.ratios ?? createTransientView(graph, `${id}-ratios`, 'float32', facilityCount);
    const selection = isThreeStep
      ? createTransientView(graph, `${id}-selection`, 'float32', demandCount)
      : undefined;
    const nodes: (GPUCommandNode<Parameters> | GPUCommandNode<Parameters>[])[] = [];
    if (selection) {
      nodes.push(
        getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
          id: `${id}-selection`,
          operation: OPERATION,
          variant: 'selection',
          bindings: [
            {name: 'offsets', view: demandWeights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: demandWeights.neighbors, type: 'u32', access: 'read'},
            {name: 'weights', view: demandWeights.weights, type: 'f32', access: 'read'},
            {name: 'selection', view: selection, type: 'f32', access: 'read_write'}
          ],
          invocationCount: demandCount,
          declarations: `const PARTNERS: u32 = ${facilityCount}u;`,
          body: `var total = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (neighbors[neighborsOffset + slot] < PARTNERS) {
      total += weights[weightsOffset + slot];
    }
  }
  selection[selectionOffset + index] = total;`
        })
      );
    }
    const selectionBinding = selection
      ? [{name: 'selection', view: selection, type: 'f32' as const, access: 'read' as const}]
      : [];
    nodes.push(
      getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
        id: `${id}-ratios`,
        operation: OPERATION,
        variant: 'ratios',
        bindings: [
          {name: 'offsets', view: facilityWeights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: facilityWeights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: facilityWeights.weights, type: 'f32', access: 'read'},
          {name: 'demand', view: demand, type: 'f32', access: 'read'},
          {name: 'supply', view: supply, type: 'f32', access: 'read'},
          ...selectionBinding,
          {name: 'ratios', view: ratios, type: 'f32', access: 'read_write'}
        ],
        invocationCount: facilityCount,
        declarations: `const PARTNERS: u32 = ${demandCount}u;
${IS_FINITE_BITS_WGSL}
${IS_FINITE_VALUE_WGSL}
${SPATIAL_ANALYSIS_QUOTIENT_WGSL}`,
        body: `var captured = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < PARTNERS) {
      let partnerDemand = demand[demandOffset + partner];
      // A non-finite demand is missing and captures nothing.
      if (isFiniteBits(partnerDemand)) {
        ${
          isThreeStep
            ? `let total = selection[selectionOffset + partner];
        if (total > 0.0) {
          // Wan, Zou and Sternberg (2012): selection weight G = w / T times the decay w.
          let weight = weights[weightsOffset + slot];
          captured += getSpatialAnalysisQuotient(partnerDemand * weight * weight,total);
        }`
            : 'captured += partnerDemand * weights[weightsOffset + slot];'
        }
      }
    }
  }
  ratios[ratiosOffset + index] = select(0.0, getSpatialAnalysisQuotient(supply[supplyOffset + index],captured), captured > 0.0);`
      }),
      getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
        id: `${id}-scores`,
        operation: OPERATION,
        variant: 'scores',
        bindings: [
          {name: 'offsets', view: demandWeights.offsets, type: 'u32', access: 'read'},
          {name: 'neighbors', view: demandWeights.neighbors, type: 'u32', access: 'read'},
          {name: 'weights', view: demandWeights.weights, type: 'f32', access: 'read'},
          {name: 'ratios', view: ratios, type: 'f32', access: 'read'},
          ...selectionBinding,
          {name: 'accessibility', view: accessibility, type: 'f32', access: 'read_write'}
        ],
        invocationCount: demandCount,
        declarations: `const PARTNERS: u32 = ${facilityCount}u;
${IS_FINITE_BITS_WGSL}
${IS_FINITE_VALUE_WGSL}
${SPATIAL_ANALYSIS_QUOTIENT_WGSL}`,
        body: `var score = 0.0;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    let partner = neighbors[neighborsOffset + slot];
    if (partner < PARTNERS) {
      let ratio = ratios[ratiosOffset + partner];
      // A non-finite ratio (a non-finite supply) is missing and adds nothing.
      if (isFiniteBits(ratio)) {
        let weight = weights[weightsOffset + slot];
        score += ${isThreeStep ? 'weight * weight' : 'weight'} * ratio;
      }
    }
  }
  ${
    isThreeStep
      ? `let total = selection[selectionOffset + index];
  score = select(0.0, getSpatialAnalysisQuotient(score,total), total > 0.0);`
      : ''
  }
  accessibility[accessibilityOffset + index] = score;`
      })
    );
    if (output.reachableFacilities) {
      nodes.push(
        getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
          id: `${id}-reachable`,
          operation: OPERATION,
          variant: 'reachable',
          bindings: [
            {name: 'offsets', view: demandWeights.offsets, type: 'u32', access: 'read'},
            {name: 'neighbors', view: demandWeights.neighbors, type: 'u32', access: 'read'},
            {name: 'reachable', view: output.reachableFacilities, type: 'u32', access: 'read_write'}
          ],
          invocationCount: demandCount,
          declarations: `const PARTNERS: u32 = ${facilityCount}u;`,
          body: `var count = 0u;
  for (var slot = offsets[offsetsOffset + index]; slot < offsets[offsetsOffset + index + 1u]; slot++) {
    if (neighbors[neighborsOffset + slot] < PARTNERS) {
      count++;
    }
  }
  reachable[reachableOffset + index] = count;`
        })
      );
    }
    if (output.demandValidity) {
      nodes.push(
        getSpatialAnalysisPackedKernelNodes<Parameters>(graph, {
          id: `${id}-validity`,
          operation: OPERATION,
          variant: 'validity',
          bindings: [
            {name: 'demand', view: demand, type: 'f32', access: 'read'},
            {name: 'validity', view: output.demandValidity, type: 'u32', access: 'read_write'}
          ],
          invocationCount: demandCount,
          declarations: IS_FINITE_BITS_WGSL,
          body: 'validity[validityOffset + index] = u32(isFiniteBits(demand[demandOffset + index]));'
        })
      );
    }
    // Supplied loss flags suppress when nonzero; the costs completeness word when zero.
    const forwardLoss = getAccessibilityWeightsLossChecks(id, demandWeights);
    const reverseLoss = getAccessibilityWeightsLossChecks(id, facilityWeights);
    const incompleteCosts = props.costsComplete ? [{flag: props.costsComplete, test: '== 0u'}] : [];
    for (const [name, view, flags, type, value] of [
      [
        'accessibility',
        output.accessibility,
        [...forwardLoss, ...reverseLoss, ...incompleteCosts],
        'f32',
        'getNaN()'
      ],
      [
        'ratios',
        output.ratios,
        [...reverseLoss, ...(isThreeStep ? forwardLoss : []), ...incompleteCosts],
        'f32',
        'getNaN()'
      ],
      [
        'reachableFacilities',
        output.reachableFacilities,
        [...forwardLoss, ...incompleteCosts],
        'u32',
        '0xffffffffu'
      ]
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

/** Returns every weights view read by a catchment contributor. */
function getWeightViews(props: GPUCatchmentAccessibilityProps): GraphDataView[] {
  const views: GraphDataView[] = [];
  for (const weights of [props.facilityWeights, props.demandWeights]) {
    if (weights) {
      views.push(
        weights.offsets,
        weights.neighbors,
        weights.weights,
        ...getAccessibilityWeightsLossChecks(props.id ?? 'catchment-accessibility', weights).map(
          check => check.flag
        )
      );
    }
  }
  if (props.costsComplete) views.push(props.costsComplete);
  return views;
}
