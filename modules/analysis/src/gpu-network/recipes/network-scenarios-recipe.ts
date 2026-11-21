// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {addPreparedRecipe} from '../../gpu-spatial-analysis/recipes/prepared-recipe';

import {GPUReduction, type GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  GPUNetworkCostMatrix,
  type GPUNetworkCostMatrixProps
} from '../network-accessibility/gpu-network-cost-matrix';
import {GPUNetworkAccessibility} from '../network-accessibility/gpu-network-accessibility';
import {
  createRecipeKernelNode as createWGSLKernelNode,
  getRecipeView as getOrCreateView,
  validateRecipeBoundary
} from '../../gpu-spatial-analysis/recipes/recipe-utils';
import {
  validateGraphOutputsDisjointFromInputs,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {getGPUNetworkTopologyViews} from '../../analysis-core/contracts/network-topology';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  addRecipeCountNodes,
  RecipeBuilder,
  type GPURecipeResult
} from '../../gpu-spatial-analysis/recipes/recipe-utils';

export type GPUNetworkDisruptionOutput = {
  baselineCosts: GraphDataView<'float32'>;
  scenarioCosts: GraphDataView<'float32'>;
  costDifferences: GraphDataView<'float32'>;
  /** Per pair: 1 when complete source/search evidence shows a reached trip became unreachable within the common (smaller) horizon. */
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
  /**
   * Reuse a retained baseline; caller owns invalidation. Supply its effective costLimit, or null
   * for an unbounded baseline. Omission explicitly binds reuse to search.costLimit and the same
   * search configuration, as in the original API.
   * A retained noded baseline must also retain status.sourceComplete; its source evidence cannot
   * be inferred from a current noding submission.
   */
  baseline?: {
    costs: GraphDataView<'float32'>;
    costLimit?: GraphDataView<'float32'> | null;
    status: {converged: GraphDataView<'uint32'>; sourceComplete?: GraphDataView<'uint32'>};
  };
  outputs?: Partial<GPUNetworkDisruptionOutput>;
};

/** Routing evidence and score certificates for a disruption comparison. */
export type GPUNetworkDisruptionRecipeResult = GPURecipeResult<GPUNetworkDisruptionOutput> & {
  status: GPURecipeResult<GPUNetworkDisruptionOutput>['status'] & {
    baselineRoutingComplete: GraphDataView<'uint32'>;
    scenarioRoutingComplete: GraphDataView<'uint32'>;
    baselineAccessibilityComplete: GraphDataView<'uint32'>;
    scenarioAccessibilityComplete: GraphDataView<'uint32'>;
    accessibilityComplete: GraphDataView<'uint32'>;
  };
};

/** Compare directed routing scenarios without compaction or Infinity - Infinity arithmetic. */
export function addNetworkDisruptionRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUNetworkDisruptionRecipeProps
): GPUNetworkDisruptionRecipeResult {
  return addPreparedRecipe(graph, props, declareNetworkDisruptionRecipe);
}

function declareNetworkDisruptionRecipe<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: GPUNetworkDisruptionRecipeProps
): GPUNetworkDisruptionRecipeResult {
  const id = props.id ?? 'network-disruption-recipe';
  const builder = new RecipeBuilder(graph);
  const columnCount = props.search.targetNodes?.length ?? props.search.topology.offsets.length - 1;
  const edgeCount = props.search.topology.weights.length;
  const pairCount = props.rowCount * columnCount;
  const noded = props.search.nodedTopology;
  validateRequirement(
    id,
    !noded || !props.baseline || Boolean(props.baseline.status.sourceComplete),
    'baseline.status.sourceComplete is required when reusing a retained noded baseline; retain its source build certificate'
  );
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
  validateOutputOwnership(graph, id, Object.values(outputs), [
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
    props.baseline?.status?.converged,
    props.baseline?.status?.sourceComplete,
    noded?.nodes.status.count,
    noded?.pieces.status.overflow,
    noded?.pieces.status.candidateOverflow,
    noded?.status?.overflow,
    noded?.status?.uncertainCount,
    props.baseline?.costLimit ?? undefined
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
  const baselineSourceComplete = props.baseline
    ? props.baseline.status.sourceComplete
    : noded
      ? getOrCreateView(graph, `${id}-baseline-source-complete`, 'uint32', 1)
      : undefined;
  const scenarioSourceComplete = noded
    ? getOrCreateView(graph, `${id}-scenario-source-complete`, 'uint32', 1)
    : undefined;
  if (!props.baseline)
    builder.add(
      new GPUNetworkCostMatrix({
        ...props.search,
        id: `${id}-baseline`,
        output: {
          costs: outputs.baselineCosts,
          status: {converged: baselineConverged, sourceComplete: baselineSourceComplete}
        }
      })
    );
  const weights = getOrCreateView(graph, `${id}-weights`, 'float32', edgeCount);
  builder.addGlue(
    'closures',
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
  const scenarioTopology = {...props.search.topology, weights};
  builder.add(
    new GPUNetworkCostMatrix({
      ...props.search,
      id: `${id}-scenario`,
      topology: scenarioTopology,
      nodedTopology: noded && {...noded, topology: scenarioTopology},
      output: {
        costs: outputs.scenarioCosts,
        status: {converged: scenarioConverged, sourceComplete: scenarioSourceComplete}
      }
    })
  );
  const addRoutingComplete = (
    name: string,
    converged: GraphDataView<'uint32'>,
    sourceComplete: GraphDataView<'uint32'> | undefined
  ) => {
    if (!sourceComplete) return converged;
    const complete = getOrCreateView(graph, `${id}-${name}-routing-complete`, 'uint32', 1);
    builder.addGlue(
      `${name}-routing-complete`,
      createWGSLKernelNode(graph, {
        id: `${id}-${name}-routing-complete`,
        operation: id,
        invocationCount: 1,
        bindings: [
          {name: 'converged', view: converged, type: 'u32', access: 'read'},
          {name: 'sourceComplete', view: sourceComplete, type: 'u32', access: 'read'},
          {name: 'complete', view: complete, type: 'u32', access: 'read_write'}
        ],
        body: 'complete[completeOffset] = u32(converged[convergedOffset] != 0u && sourceComplete[sourceCompleteOffset] != 0u);'
      })
    );
    return complete;
  };
  const baselineRoutingComplete = addRoutingComplete(
    'baseline',
    baselineConverged,
    baselineSourceComplete
  );
  const scenarioRoutingComplete = addRoutingComplete(
    'scenario',
    scenarioConverged,
    scenarioSourceComplete
  );
  const baselineLimit =
    props.baseline?.costLimit !== undefined
      ? (props.baseline.costLimit ?? undefined)
      : props.search.costLimit;
  const baselineAccessibilityComplete = getOrCreateView(
    graph,
    `${id}-baseline-accessibility-complete`,
    'uint32',
    1
  );
  const scenarioAccessibilityComplete = getOrCreateView(
    graph,
    `${id}-scenario-accessibility-complete`,
    'uint32',
    1
  );
  for (const [name, costs, cumulative, converged, complete] of [
    [
      'baseline',
      outputs.baselineCosts,
      outputs.baselineAccessibility,
      baselineRoutingComplete,
      baselineAccessibilityComplete
    ],
    [
      'scenario',
      outputs.scenarioCosts,
      outputs.scenarioAccessibility,
      scenarioRoutingComplete,
      scenarioAccessibilityComplete
    ]
  ] as const)
    builder.add(
      new GPUNetworkAccessibility({
        id: `${id}-${name}-accessibility`,
        costs,
        orientation: 'origin-rows',
        opportunityWeights: props.opportunityWeights,
        parameters: props.parameters,
        cumulative,
        costsComplete: converged,
        costLimit: name === 'baseline' ? baselineLimit : props.search.costLimit,
        status: {complete}
      })
    );
  builder.addGlue(
    'differences',
    createWGSLKernelNode(graph, {
      id: `${id}-differences`,
      operation: id,
      invocationCount: pairCount,
      declarations: IS_FINITE_VALUE_WGSL,
      bindings: [
        {name: 'before', view: outputs.baselineCosts, type: 'f32', access: 'read'},
        {name: 'after', view: outputs.scenarioCosts, type: 'f32', access: 'read'},
        {name: 'beforeConverged', view: baselineRoutingComplete, type: 'u32', access: 'read'},
        {name: 'afterConverged', view: scenarioRoutingComplete, type: 'u32', access: 'read'},
        {name: 'difference', view: outputs.costDifferences, type: 'f32', access: 'read_write'},
        {name: 'unreachable', view: outputs.newlyUnreachable, type: 'u32', access: 'read_write'},
        ...(baselineLimit
          ? [
              {
                name: 'beforeLimit',
                view: baselineLimit,
                type: 'f32' as const,
                access: 'read' as const
              }
            ]
          : []),
        ...(props.search.costLimit
          ? [
              {
                name: 'afterLimit',
                view: props.search.costLimit,
                type: 'f32' as const,
                access: 'read' as const
              }
            ]
          : [])
      ],
      body: `let valid = beforeConverged[beforeConvergedOffset] != 0u && afterConverged[afterConvergedOffset] != 0u;
let first = before[beforeOffset + index]; let second = after[afterOffset + index];
var delta = bitcast<f32>(0x7fc00000u | (index & 0u));
if (valid && isFiniteValue(first) && isFiniteValue(second)) { delta = second - first; }
difference[differenceOffset + index] = delta;
var commonLimit = bitcast<f32>(0x7f800000u | (index & 0u));
var horizonsKnown = true;
${baselineLimit ? 'let beforeHorizon = beforeLimit[beforeLimitOffset]; horizonsKnown = horizonsKnown && (bitcast<u32>(beforeHorizon) & 0x7fffffffu) <= 0x7f800000u; commonLimit = min(commonLimit, beforeHorizon);' : ''}
${props.search.costLimit ? 'let afterHorizon = afterLimit[afterLimitOffset]; horizonsKnown = horizonsKnown && (bitcast<u32>(afterHorizon) & 0x7fffffffu) <= 0x7f800000u; commonLimit = min(commonLimit, afterHorizon);' : ''}
let afterOutside = second == bitcast<f32>(0x7f800000u | (index & 0u)) || (isFiniteValue(second) && second > commonLimit);
unreachable[unreachableOffset + index] = select(0u, 1u, valid && horizonsKnown && isFiniteValue(first) && first <= commonLimit && afterOutside);`
    })
  );
  const accessibilityComplete = getOrCreateView(graph, `${id}-accessibility-complete`, 'uint32', 1);
  builder.addGlue(
    'accessibility-complete',
    createWGSLKernelNode(graph, {
      id: `${id}-accessibility-complete`,
      operation: id,
      invocationCount: 1,
      bindings: [
        {name: 'beforeComplete', view: baselineAccessibilityComplete, type: 'u32', access: 'read'},
        {name: 'afterComplete', view: scenarioAccessibilityComplete, type: 'u32', access: 'read'},
        {name: 'complete', view: accessibilityComplete, type: 'u32', access: 'read_write'}
      ],
      body: 'complete[completeOffset] = u32(beforeComplete[beforeCompleteOffset] != 0u && afterComplete[afterCompleteOffset] != 0u);'
    })
  );
  builder.addGlue(
    'accessibility-delta',
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
        {name: 'beforeComplete', view: baselineAccessibilityComplete, type: 'u32', access: 'read'},
        {name: 'afterComplete', view: scenarioAccessibilityComplete, type: 'u32', access: 'read'}
      ],
      body: `var delta = bitcast<f32>(0x7fc00000u | (index & 0u)); let first = before[beforeOffset + index]; let second = after[afterOffset + index];
if (beforeComplete[beforeCompleteOffset] != 0u && afterComplete[afterCompleteOffset] != 0u && isFiniteValue(first) && isFiniteValue(second)) { delta = second - first; }
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
      baselineRoutingComplete,
      scenarioRoutingComplete,
      baselineAccessibilityComplete,
      scenarioAccessibilityComplete,
      accessibilityComplete,
      stages: [
        {
          stage: 'baseline-routing',
          status: {converged: baselineConverged, complete: baselineRoutingComplete}
        },
        {
          stage: 'scenario-routing',
          status: {converged: scenarioConverged, complete: scenarioRoutingComplete}
        },
        {stage: 'baseline-accessibility', status: {complete: baselineAccessibilityComplete}},
        {stage: 'scenario-accessibility', status: {complete: scenarioAccessibilityComplete}},
        {stage: 'accessibility', status: {complete: accessibilityComplete}}
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
  return addPreparedRecipe(graph, props, declareFacilityScenarioRecipe);
}

function declareFacilityScenarioRecipe<Parameters>(
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
  validateOutputOwnership(graph, id, Object.values(outputs), [
    props.costs,
    props.demand,
    props.supply,
    props.facilityMask,
    props.parameters
  ]);
  const supply = getOrCreateView(graph, `${id}-active-supply`, 'float32', facilityCount);
  builder.addGlue(
    'supply',
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
  builder.addGlue(
    'nearest',
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
  const covered = getOrCreateView(graph, `${id}-covered-values`, 'float32', demandCount);
  const uncovered = getOrCreateView(graph, `${id}-uncovered-values`, 'float32', demandCount);
  addRecipeCountNodes(builder, {
    id: `${id}-summary`,
    operation: id,
    rowCount: demandCount,
    declarations: IS_FINITE_VALUE_WGSL,
    bindings: [
      {name: 'demand', view: props.demand, type: 'f32', access: 'read'},
      {name: 'nearest', view: outputs.nearestCosts, type: 'f32', access: 'read'},
      {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
      {name: 'covered', view: covered, type: 'f32', access: 'read_write'},
      {name: 'uncovered', view: uncovered, type: 'f32', access: 'read_write'}
    ],
    prelude: `let weight = demand[demandOffset + index];
let cost = nearest[nearestOffset + index];
let valid = isFiniteValue(weight) && weight >= 0.0;
let reached = isFiniteValue(cost) && cost <= parameters[parametersOffset];
covered[coveredOffset + index] = select(0.0, weight, valid && reached);
uncovered[uncoveredOffset + index] = select(0.0, weight, valid && !reached);`,
    counts: [{name: 'invalid', output: outputs.invalidDemandCount, value: 'u32(!valid)'}]
  });
  for (const [name, input, output] of [
    ['covered', covered, outputs.coveredDemand],
    ['uncovered', uncovered, outputs.uncoveredDemand]
  ] as const) {
    builder.add(new GPUReduction({id: `${id}-summary-${name}`, input, output, operation: 'sum'}));
  }
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
function validateOutputOwnership<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  outputs: readonly GraphDataView[],
  inputs: readonly (GraphDataView | undefined)[]
): void {
  validateRecipeBoundary(graph, id, inputs, outputs);
}
