// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {
  GPUInvalidStatusPort,
  GPUIterationStatusPort
} from '../../analysis-core/contracts/status';
import {
  validateNetworkOptimizationBindings,
  validateNetworkOptimizationInputOutputBuffers
} from './network-optimization-bindings';

import {
  createNetworkOptimizationStatusNodes,
  getNetworkOptimizationStatusViews,
  validateNetworkOptimizationStatus
} from './network-optimization-status';
import {
  getRationalAccessCycleKernel,
  RATIONAL_ACCESS_WORKGROUP_SIZE
} from './network-rational-access-cycles';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** Largest origin count: one lane per origin in the cooperative cycle workgroup. */
const MAXIMUM_ORIGIN_COUNT = 64;
/** Largest facility count: one lane per facility in the cooperative cycle workgroup. */
const MAXIMUM_FACILITY_COUNT = 32;
/** Largest cycle count: bounds workgroup-local equalization without growing the command graph. */
const MAXIMUM_CYCLES = 4096;
/** Default `maximumCycles`. */
const DEFAULT_MAXIMUM_CYCLES = 150;
/** Default `initialLimitCycles`. */
const DEFAULT_INITIAL_LIMIT_CYCLES = 20;
/** Rows of the `parameters` view: tau, rho, initial step, minimum step, half-life, gap tolerance. */
const PARAMETER_COUNT = 6;
/** Words of the `summary` view: valid, gap within tolerance, completed cycles, maximum gap bits. */
const SUMMARY_WORD_COUNT = 4;

export type GPUNetworkRationalAccessOutput = {
  assignments: GraphDataView<'float32'>;
  scores: GraphDataView<'float32'>;
  unservedDemand: GraphDataView<'float32'>;
  /** [valid, equilibriumGapWithinTolerance, completedCycles, maximumGapBits]. */
  summary: GraphDataView<'uint32'>;
  /** invalidCount is 1 for a rejected problem or nonfinite arithmetic. iterationCount counts
   * completed allocation shifts; the initial gap evaluation is cycle zero. Only the final
   * equilibrium gap certifies convergence; a rounding stall still reaches the cycle limit.
   */
  status?: GPUInvalidStatusPort & GPUIterationStatusPort;
};

export type GPUNetworkRationalAccessProps = {
  id?: string;
  /** Origin-major travel costs; +Infinity is unreachable. */
  costs: GraphDataView<'float32'>;
  /** Nonnegative demand, integral by default; total <= 2^24. */
  demand: GraphDataView<'float32'>;
  supply: GraphDataView<'float32'>;
  /** tau, rho, initial fractional step, minimum fractional step, half-life, equilibrium gap tolerance. */
  parameters: GraphDataView<'float32'>;
  maximumCycles?: number;
  /** Fractional demands retain continuous equalizing shifts. Default integer rounding. */
  demandMode?: 'integral' | 'fractional';
  /** Step parameters cap a fraction of each origin demand, or an absolute demand amount. */
  stepMode?: 'fraction' | 'absolute';
  /** Early-cycle destination inflow cap; default 20, zero disables. */
  initialLimitCycles?: number;
  output: GPUNetworkRationalAccessOutput;
};

/**
 * Bounded integral/fractional rational-agent allocation with RAAM's travel/supply normalization,
 * pairwise congestion equalization and decaying demand-shift cap. Final scores use the final
 * allocation. Rounding-stalled allocation is not convergence; summary reports the cost gap.
 * Parallel origins move simultaneously. Bounds: 64 origins, 32 facilities, 4096 cycles.
 *
 * Reference tier: initialization preserves ordered validation and partial results in one lane.
 * Cycles run cooperatively in one 64-lane workgroup: one lane per origin scores and shifts,
 * one per facility sums loads and inflow in ascending origin order. Ties and float32 summation
 * order are unchanged. Uniform workgroup exits stop at equilibrium or invalid arithmetic.
 * Bounds remain 64 origins, 32 facilities and 4,096 cycles; larger allocations need a separate
 * multi-workgroup design. The command graph has two kernels plus optional named-status publication.
 * Compile-time: view lengths, demandMode, stepMode, maximumCycles, initialLimitCycles and optional
 * status views. Per-frame: costs, demand, supply and parameter contents. Every encoding rewrites
 * the outputs.
 */
export class GPUNetworkRationalAccess implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkRationalAccessProps;
  constructor(props: GPUNetworkRationalAccessProps) {
    this.id = props.id ?? 'network-rational-access';
    this.props = props;
    const {id} = this;
    validateCapacity(id, 'demand rows', props.demand.length, 1, MAXIMUM_ORIGIN_COUNT);
    validateCapacity(id, 'supply rows', props.supply.length, 1, MAXIMUM_FACILITY_COUNT);
    validateCapacity(
      id,
      'maximumCycles',
      props.maximumCycles ?? DEFAULT_MAXIMUM_CYCLES,
      0,
      MAXIMUM_CYCLES
    );
    validateCapacity(
      id,
      'initialLimitCycles',
      props.initialLimitCycles ?? DEFAULT_INITIAL_LIMIT_CYCLES,
      0,
      MAXIMUM_CYCLES
    );
    for (const view of [
      props.costs,
      props.demand,
      props.supply,
      props.parameters,
      props.output.assignments,
      props.output.scores,
      props.output.unservedDemand
    ])
      validateSinglePackedView(view, ['float32'], this.id);
    validateSinglePackedView(props.output.summary, ['uint32'], this.id);
    validateRequirement(
      id,
      props.costs.length === props.demand.length * props.supply.length,
      'costs must hold demand rows times supply rows'
    );
    validateRequirement(
      id,
      props.parameters.length === PARAMETER_COUNT,
      `parameters must hold ${PARAMETER_COUNT} rows`
    );
    validateRequirement(
      id,
      props.output.assignments.length === props.costs.length,
      'output.assignments must hold as many rows as costs'
    );
    validateRequirement(
      id,
      props.output.scores.length === props.demand.length,
      'output.scores must hold one row per demand row'
    );
    validateRequirement(
      id,
      props.output.unservedDemand.length === props.demand.length,
      'output.unservedDemand must hold one row per demand row'
    );
    validateRequirement(
      id,
      props.output.summary.length === SUMMARY_WORD_COUNT,
      `output.summary must hold ${SUMMARY_WORD_COUNT} rows`
    );
    validateNetworkOptimizationStatus(
      id,
      [
        props.output.assignments,
        props.output.scores,
        props.output.unservedDemand,
        props.output.summary
      ],
      props.output.status
    );
    validateNetworkOptimizationInputOutputBuffers(
      this.id,
      getRationalAccessOutputViews(props.output),
      [props.costs, props.demand, props.supply, props.parameters]
    );
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateNetworkOptimizationBindings(
      graph,
      id,
      Object.fromEntries(
        [
          props.costs,
          props.demand,
          props.supply,
          props.parameters,
          ...getRationalAccessOutputViews(props.output)
        ].map((view, index) => [`view ${index}`, view])
      )
    );

    const nodes = [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPUNetworkRationalAccess',
        variant: 'initialize',
        invocationCount: 1,
        workgroupSize: 2,
        bindings: [
          {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
          {name: 'demand', view: props.demand, type: 'f32', access: 'read'},
          {name: 'supply', view: props.supply, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'assignments', view: props.output.assignments, type: 'f32', access: 'read_write'},
          {name: 'scores', view: props.output.scores, type: 'f32', access: 'read_write'},
          {name: 'unserved', view: props.output.unservedDemand, type: 'f32', access: 'read_write'},
          {name: 'summary', view: props.output.summary, type: 'u32', access: 'read_write'}
        ],
        declarations: `const ORIGIN_COUNT: u32 = ${props.demand.length}u; const FACILITY_COUNT: u32 = ${props.supply.length}u;
${IS_FINITE_VALUE_WGSL}
`,
        body: `var nanBits = 0x7fc00000u; let missing = bitcast<f32>(nanBits); var infinityBits = 0x7f800000u; let infinity = bitcast<f32>(infinityBits);
for (var word = 0u; word < 4u; word++) { summary[summaryOffset + word] = 0u; }
for (var entry = 0u; entry < ORIGIN_COUNT * FACILITY_COUNT; entry++) { assignments[assignmentsOffset + entry] = 0.0; }
for (var origin = 0u; origin < ORIGIN_COUNT; origin++) { scores[scoresOffset + origin] = missing; unserved[unservedOffset + origin] = demand[demandOffset + origin]; }
let travelScale = parameters[parametersOffset]; let supplyScale = parameters[parametersOffset + 1u];
let initialStep = parameters[parametersOffset + 2u]; let minimumStep = parameters[parametersOffset + 3u]; let halfLife = parameters[parametersOffset + 4u]; let tolerance = parameters[parametersOffset + 5u];
if (!isFiniteValue(travelScale) || travelScale <= 0.0 || !isFiniteValue(supplyScale) || supplyScale <= 0.0 || !isFiniteValue(initialStep) || initialStep < 0.0 || ${props.stepMode === 'absolute' ? 'false' : 'initialStep > 1.0'} || !isFiniteValue(minimumStep) || minimumStep < 0.0 || ${props.stepMode === 'absolute' ? 'false' : 'minimumStep > 1.0'} || !isFiniteValue(halfLife) || halfLife <= 0.0 || !isFiniteValue(tolerance) || tolerance < 0.0) { return; }
var capacities: array<f32, ${props.supply.length}>;
for (var facility = 0u; facility < FACILITY_COUNT; facility++) { capacities[facility] = supply[supplyOffset + facility] * supplyScale; if (!isFiniteValue(capacities[facility]) || capacities[facility] <= 0.0) { return; } }
var totalDemand = 0.0;
for (var origin = 0u; origin < ORIGIN_COUNT; origin++) {
  let amount = demand[demandOffset + origin]; if (!isFiniteValue(amount) || amount < 0.0 || ${props.demandMode === 'fractional' ? 'false' : 'floor(amount) != amount'} || amount > 16777216.0 - totalDemand) { return; } totalDemand += amount;
  var nearest = 0xffffffffu; var nearestCost = infinity;
  for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
    let cost = costs[costsOffset + origin * FACILITY_COUNT + facility];
    if (cost < 0.0 || (bitcast<u32>(cost) & 0x7fffffffu) > 0x7f800000u) { return; }
    if (cost < nearestCost) { nearest = facility; nearestCost = cost; }
  }
  if (nearest != 0xffffffffu) { assignments[assignmentsOffset + origin * FACILITY_COUNT + nearest] = amount; unserved[unservedOffset + origin] = 0.0; }
}
if (totalDemand > 16777216.0) { return; }
summary[summaryOffset] = 1u;
`
      }),
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-cycles`,
        operation: 'GPUNetworkRationalAccess',
        variant: 'cooperative-cycles',
        invocationCount: RATIONAL_ACCESS_WORKGROUP_SIZE,
        workgroupSize: RATIONAL_ACCESS_WORKGROUP_SIZE,
        guardIndex: false,
        bindings: [
          {name: 'costs', view: props.costs, type: 'f32', access: 'read'},
          {name: 'demand', view: props.demand, type: 'f32', access: 'read'},
          {name: 'supply', view: props.supply, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'assignments', view: props.output.assignments, type: 'f32', access: 'read_write'},
          {name: 'scores', view: props.output.scores, type: 'f32', access: 'read_write'},
          {name: 'unserved', view: props.output.unservedDemand, type: 'f32', access: 'read_write'},
          {name: 'summary', view: props.output.summary, type: 'u32', access: 'read_write'}
        ],
        ...getRationalAccessCycleKernel(props)
      })
    ];
    nodes.push(
      ...createNetworkOptimizationStatusNodes(
        graph,
        id,
        'GPUNetworkRationalAccess',
        props.output.summary,
        props.output.status,
        {
          invalidCount: '1u - summary[summaryOffset]',
          iterationCount: 'summary[summaryOffset + 2u]',
          converged: 'summary[summaryOffset] * summary[summaryOffset + 1u]',
          iterationLimitReached: 'summary[summaryOffset] * (1u - summary[summaryOffset + 1u])'
        }
      )
    );
    return nodes;
  }
}

function getRationalAccessOutputViews(
  output: GPUNetworkRationalAccessProps['output']
): GraphDataView[] {
  return [
    output.assignments,
    output.scores,
    output.unservedDemand,
    output.summary,
    ...getNetworkOptimizationStatusViews(output.status)
  ];
}
