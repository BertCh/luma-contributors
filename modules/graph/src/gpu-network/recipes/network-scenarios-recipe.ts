// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GraphDataView} from '@luma-contributors/graph/gpu-core';
import {
  GPUNetworkCostMatrix,
  type GPUNetworkCostMatrixProps
} from '../network-accessibility/gpu-network-cost-matrix';
import {GPUNetworkAccessibility} from '../network-accessibility/gpu-network-accessibility';
import {
  createRecipeKernelNode as createWGSLKernelNode,
  getRecipeView as getOrCreateView
} from '../../gpu-spatial-analysis/recipes/expansion-recipe-utils';
import {validateGraphOutputsDisjointFromInputs} from '../../utils/gpu-contributor-utils';
import {getGPUNetworkTopologyViews} from '../../analysis-core/network-topology';
import {IS_FINITE_VALUE_WGSL} from '../../utils/wgsl-common';
import {RecipeBuilder, type GPURecipeResult} from '../../gpu-spatial-analysis/recipes/recipe-utils';

export type GPUNetworkDisruptionOutput = {
  baselineCosts: GraphDataView<'float32'>;
  scenarioCosts: GraphDataView<'float32'>;
  costDifferences: GraphDataView<'float32'>;
  /** Per origin-target pair: 1 only when both searches converged and a reached target became unreachable. */
  newlyUnreachable: GraphDataView<'uint32'>;
  baselineAccessibility: GraphDataView<'float32'>;
  scenarioAccessibility: GraphDataView<'float32'>;
  accessibilityDifferences: GraphDataView<'float32'>;
};
export type GPUNetworkDisruptionRecipeProps = {
  id?: string;
  search: Omit<GPUNetworkCostMatrixProps, 'id' | 'output'>;
  rowCount: number;
  /** Zero closes a CSR edge slot. Original nodes, parallel edges and source slots remain stable. */
  edgeMask?: GraphDataView<'uint32'>;
  replacementCosts?: GraphDataView<'float32'>;
  opportunityWeights: GraphDataView<'float32'>;
  parameters: GraphDataView<'float32'>;
  /** Reuse a caller-retained baseline and its convergence word; caller owns invalidation. */
  baseline?: {costs: GraphDataView<'float32'>; status: {converged: GraphDataView<'uint32'>}};
  outputs?: Partial<GPUNetworkDisruptionOutput>;
};

/** Compare directed routing scenarios without compaction or Infinity - Infinity arithmetic. */
export function addNetworkDisruptionRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUNetworkDisruptionRecipeProps
): GPURecipeResult<GPUNetworkDisruptionOutput> {
  const id = props.id ?? 'network-disruption-recipe';
  const builder = new RecipeBuilder(graph);
  const columnCount = props.search.targetNodes?.length ?? props.search.topology.offsets.length - 1;
  const edgeCount = props.search.topology.weights.length;
  const pairCount = props.rowCount * columnCount;
  if (
    !Number.isSafeInteger(props.rowCount) ||
    props.rowCount < 1 ||
    props.opportunityWeights.length !== columnCount ||
    (props.edgeMask && props.edgeMask.length !== edgeCount) ||
    (props.replacementCosts && props.replacementCosts.length !== edgeCount)
  ) {
    throw new Error(`${id}: inconsistent scenario dimensions`);
  }
  const provided = props.outputs ?? {};
  if (props.baseline && provided.baselineCosts && props.baseline.costs !== provided.baselineCosts) {
    throw new Error(`${id}: retained baseline is the baseline output`);
  }
  const outputs: GPUNetworkDisruptionOutput = {
    baselineCosts: getOrCreateView(
      graph,
      `${id}-baseline-costs`,
      'float32',
      pairCount,
      props.baseline?.costs ?? provided.baselineCosts
    ),
    scenarioCosts: getOrCreateView(
      graph,
      `${id}-scenario-costs`,
      'float32',
      pairCount,
      provided.scenarioCosts
    ),
    costDifferences: getOrCreateView(
      graph,
      `${id}-cost-differences`,
      'float32',
      pairCount,
      provided.costDifferences
    ),
    newlyUnreachable: getOrCreateView(
      graph,
      `${id}-newly-unreachable`,
      'uint32',
      pairCount,
      provided.newlyUnreachable
    ),
    baselineAccessibility: getOrCreateView(
      graph,
      `${id}-baseline-accessibility`,
      'float32',
      props.rowCount,
      provided.baselineAccessibility
    ),
    scenarioAccessibility: getOrCreateView(
      graph,
      `${id}-scenario-accessibility`,
      'float32',
      props.rowCount,
      provided.scenarioAccessibility
    ),
    accessibilityDifferences: getOrCreateView(
      graph,
      `${id}-accessibility-differences`,
      'float32',
      props.rowCount,
      provided.accessibilityDifferences
    )
  };
  validateOutputOwnership(id, Object.values(outputs), [
    ...getGPUNetworkTopologyViews(props.search.topology),
    props.search.startNodes,
    props.search.startRows,
    props.search.startCosts,
    props.search.targetNodes,
    props.search.costLimit,
    props.edgeMask,
    props.replacementCosts,
    props.opportunityWeights,
    props.parameters,
    props.baseline?.status?.converged
  ]);
  if (props.baseline)
    validateGraphOutputsDisjointFromInputs(
      id,
      Object.values(outputs).filter(view => view !== outputs.baselineCosts),
      [props.baseline.costs]
    );
  const baselineConverged = getOrCreateView(
    graph,
    `${id}-baseline-converged`,
    'uint32',
    1,
    props.baseline?.status?.converged
  );
  const scenarioConverged = getOrCreateView(graph, `${id}-scenario-converged`, 'uint32', 1);
  if (!props.baseline)
    builder.add(
      new GPUNetworkCostMatrix({
        ...props.search,
        id: `${id}-baseline`,
        output: {costs: outputs.baselineCosts, status: {converged: baselineConverged}}
      })
    );
  const weights = getOrCreateView(graph, `${id}-weights`, 'float32', edgeCount);
  graph.add(
    createWGSLKernelNode(graph, {
      id: `${id}-closures`,
      operation: id,
      invocationCount: edgeCount,
      bindings: [
        {
          name: 'source',
          view: props.replacementCosts ?? props.search.topology.weights,
          type: 'f32',
          access: 'read'
        },
        {name: 'destination', view: weights, type: 'f32', access: 'read_write'},
        ...(props.edgeMask
          ? [{name: 'mask', view: props.edgeMask, type: 'u32' as const, access: 'read' as const}]
          : [])
      ],
      body: `destination[destinationOffset + index] = ${props.edgeMask ? 'select(-1.0, source[sourceOffset + index], mask[maskOffset + index] != 0u)' : 'source[sourceOffset + index]'};`
    })
  );
  builder.add(
    new GPUNetworkCostMatrix({
      ...props.search,
      id: `${id}-scenario`,
      topology: {...props.search.topology, weights},
      output: {costs: outputs.scenarioCosts, status: {converged: scenarioConverged}}
    })
  );
  for (const [name, costs, cumulative] of [
    ['baseline', outputs.baselineCosts, outputs.baselineAccessibility],
    ['scenario', outputs.scenarioCosts, outputs.scenarioAccessibility]
  ] as const)
    builder.add(
      new GPUNetworkAccessibility({
        id: `${id}-${name}-accessibility`,
        costs,
        orientation: 'origin-rows',
        opportunityWeights: props.opportunityWeights,
        parameters: props.parameters,
        cumulative
      })
    );
  graph.add(
    createWGSLKernelNode(graph, {
      id: `${id}-differences`,
      operation: id,
      invocationCount: pairCount,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'before', view: outputs.baselineCosts, type: 'f32', access: 'read'},
        {name: 'after', view: outputs.scenarioCosts, type: 'f32', access: 'read'},
        {name: 'beforeConverged', view: baselineConverged, type: 'u32', access: 'read'},
        {name: 'afterConverged', view: scenarioConverged, type: 'u32', access: 'read'},
        {name: 'difference', view: outputs.costDifferences, type: 'f32', access: 'read_write'},
        {name: 'unreachable', view: outputs.newlyUnreachable, type: 'u32', access: 'read_write'}
      ],
      body: `let valid = beforeConverged[beforeConvergedOffset] != 0u && afterConverged[afterConvergedOffset] != 0u;
let first = before[beforeOffset + index]; let second = after[afterOffset + index];
var delta = bitcast<f32>(0x7fc00000u | (index & 0u));
if (valid && isFiniteValue(first) && isFiniteValue(second)) { delta = second - first; }
difference[differenceOffset + index] = delta;
unreachable[unreachableOffset + index] = select(0u, 1u, valid && isFiniteValue(first) && second == bitcast<f32>(0x7f800000u | (index & 0u)));`
    })
  );
  graph.add(
    createWGSLKernelNode(graph, {
      id: `${id}-accessibility-delta`,
      operation: id,
      invocationCount: props.rowCount,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'before', view: outputs.baselineAccessibility, type: 'f32', access: 'read'},
        {name: 'after', view: outputs.scenarioAccessibility, type: 'f32', access: 'read'},
        {
          name: 'difference',
          view: outputs.accessibilityDifferences,
          type: 'f32',
          access: 'read_write'
        },
        {name: 'beforeConverged', view: baselineConverged, type: 'u32', access: 'read'},
        {name: 'afterConverged', view: scenarioConverged, type: 'u32', access: 'read'}
      ],
      body: `var delta = bitcast<f32>(0x7fc00000u | (index & 0u)); let first = before[beforeOffset + index]; let second = after[afterOffset + index];
if (beforeConverged[beforeConvergedOffset] != 0u && afterConverged[afterConvergedOffset] != 0u && isFiniteValue(first) && isFiniteValue(second)) { delta = second - first; }
difference[differenceOffset + index] = delta;`
    })
  );
  return {
    contributors: builder.contributors,
    outputs,
    intermediates: {
      weights,
      originNodes: props.search.startNodes,
      targetNodes: props.search.targetNodes,
      topology: props.search.topology
    },
    status: {
      stages: [
        {stage: 'baseline-routing', status: {converged: baselineConverged}},
        {stage: 'scenario-routing', status: {converged: scenarioConverged}}
      ]
    }
  };
}

export type GPUFacilityScenarioOutput = {
  accessibility: GraphDataView<'float32'>;
  nearestCosts: GraphDataView<'float32'>;
  nearestFacilities: GraphDataView<'uint32'>;
  coveredDemand: GraphDataView<'float32'>;
  uncoveredDemand: GraphDataView<'float32'>;
  /** One row: demand omitted because it is negative or nonfinite. */
  invalidDemandCount: GraphDataView<'uint32'>;
};
export type GPUFacilityScenarioRecipeProps = {
  id?: string;
  /** Facility-major costs to demand locations, typically from reverse-network searches. */
  costs: GraphDataView<'float32'>;
  demand: GraphDataView<'float32'>;
  supply: GraphDataView<'float32'>;
  facilityMask: GraphDataView<'uint32'>;
  parameters: GraphDataView<'float32'>;
  outputs?: Partial<GPUFacilityScenarioOutput>;
};
/** Evaluate one live facility scenario against a reusable matrix. Selection is a separate objective. */
export function addFacilityScenarioRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUFacilityScenarioRecipeProps
): GPURecipeResult<GPUFacilityScenarioOutput> {
  const id = props.id ?? 'facility-scenario-recipe';
  const builder = new RecipeBuilder(graph);
  const demandCount = props.demand.length;
  const facilityCount = props.supply.length;
  if (
    props.costs.length !== demandCount * facilityCount ||
    props.facilityMask.length !== facilityCount ||
    !demandCount ||
    !facilityCount
  )
    throw new Error(`${id}: inconsistent matrix dimensions`);
  const provided = props.outputs ?? {};
  const outputs: GPUFacilityScenarioOutput = {
    accessibility: getOrCreateView(
      graph,
      `${id}-accessibility`,
      'float32',
      demandCount,
      provided.accessibility
    ),
    nearestCosts: getOrCreateView(
      graph,
      `${id}-nearest-costs`,
      'float32',
      demandCount,
      provided.nearestCosts
    ),
    nearestFacilities: getOrCreateView(
      graph,
      `${id}-nearest-facilities`,
      'uint32',
      demandCount,
      provided.nearestFacilities
    ),
    coveredDemand: getOrCreateView(graph, `${id}-covered`, 'float32', 1, provided.coveredDemand),
    uncoveredDemand: getOrCreateView(
      graph,
      `${id}-uncovered`,
      'float32',
      1,
      provided.uncoveredDemand
    ),
    invalidDemandCount: getOrCreateView(
      graph,
      `${id}-invalid`,
      'uint32',
      1,
      provided.invalidDemandCount
    )
  };
  validateOutputOwnership(id, Object.values(outputs), [
    props.costs,
    props.demand,
    props.supply,
    props.facilityMask,
    props.parameters
  ]);
  const supply = getOrCreateView(graph, `${id}-active-supply`, 'float32', facilityCount);
  graph.add(
    createWGSLKernelNode(graph, {
      id: `${id}-supply`,
      operation: id,
      invocationCount: facilityCount,
      bindings: [
        {name: 'source', view: props.supply, type: 'f32', access: 'read'},
        {name: 'mask', view: props.facilityMask, type: 'u32', access: 'read'},
        {name: 'destination', view: supply, type: 'f32', access: 'read_write'}
      ],
      body: 'destination[destinationOffset + index] = select(0.0, source[sourceOffset + index], mask[maskOffset + index] != 0u);'
    })
  );
  builder.add(
    new GPUNetworkAccessibility({
      id: `${id}-accessibility`,
      costs: props.costs,
      opportunityWeights: supply,
      parameters: props.parameters,
      gravity: outputs.accessibility
    })
  );
  graph.add(
    createWGSLKernelNode(graph, {
      id: `${id}-nearest`,
      operation: id,
      invocationCount: demandCount,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
        {name: 'mask', view: props.facilityMask, type: 'u32', access: 'read'},
        {name: 'nearest', view: outputs.nearestCosts, type: 'f32', access: 'read_write'},
        {name: 'facility', view: outputs.nearestFacilities, type: 'u32', access: 'read_write'}
      ],
      body: `var minimum = bitcast<f32>(0x7f800000u | (index & 0u)); var selected = 0xffffffffu;
for (var candidate = 0u; candidate < ${facilityCount}u; candidate++) {
  let cost = costs[costsOffset + candidate * ${demandCount}u + index];
  if (mask[maskOffset + candidate] != 0u && isFiniteValue(cost) && cost >= 0.0 && cost < minimum) { minimum = cost; selected = candidate; }
}
nearest[nearestOffset + index] = minimum; facility[facilityOffset + index] = selected;`
    })
  );
  graph.add(
    createWGSLKernelNode(graph, {
      id: `${id}-summary`,
      operation: id,
      invocationCount: 1,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'demand', view: props.demand, type: 'f32', access: 'read'},
        {name: 'nearest', view: outputs.nearestCosts, type: 'f32', access: 'read'},
        {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
        {name: 'covered', view: outputs.coveredDemand, type: 'f32', access: 'read_write'},
        {name: 'uncovered', view: outputs.uncoveredDemand, type: 'f32', access: 'read_write'},
        {name: 'invalid', view: outputs.invalidDemandCount, type: 'u32', access: 'read_write'}
      ],
      body: `var coveredTotal = 0.0; var uncoveredTotal = 0.0; var invalidCount = 0u;
for (var row = 0u; row < ${demandCount}u; row++) {
  let weight = demand[demandOffset + row]; let cost = nearest[nearestOffset + row];
  if (!isFiniteValue(weight) || weight < 0.0) { invalidCount++; continue; }
  if (isFiniteValue(cost) && cost <= parameters[parametersOffset]) { coveredTotal += weight; } else { uncoveredTotal += weight; }
}
covered[coveredOffset] = coveredTotal; uncovered[uncoveredOffset] = uncoveredTotal; invalid[invalidOffset] = invalidCount;`
    })
  );
  return {
    contributors: builder.contributors,
    outputs,
    intermediates: {activeSupply: supply, costs: props.costs},
    status: {
      stages: [{stage: 'demand-summary', status: {invalidCount: outputs.invalidDemandCount}}]
    }
  };
}

/** Ownership is checked across stages, not only for bindings within one arithmetic pass. */
function validateOutputOwnership(
  id: string,
  outputs: readonly GraphDataView[],
  inputs: readonly (GraphDataView | undefined)[]
): void {
  const buffers = outputs.map(view => view.buffer);
  if (new Set(buffers).size !== buffers.length)
    throw new Error(`${id}: outputs must use distinct buffers`);
  validateGraphOutputsDisjointFromInputs(id, outputs, inputs);
}
