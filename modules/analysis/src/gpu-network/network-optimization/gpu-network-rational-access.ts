// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validateCapacity,
  validateSinglePackedView,
  validateRequirement,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';

/** Largest demand (origin) row count: the single invocation keeps per-origin state in private arrays. */
const MAXIMUM_ORIGIN_COUNT = 64;
/** Largest supply (facility) row count: the single invocation keeps per-facility state in private arrays. */
const MAXIMUM_FACILITY_COUNT = 32;
/** Largest cycle count: bounds the equalization loop of the single invocation. */
const MAXIMUM_CYCLES = 4096;
/** Default `maximumCycles`. */
const DEFAULT_MAXIMUM_CYCLES = 150;
/** Default `initialLimitCycles`. */
const DEFAULT_INITIAL_LIMIT_CYCLES = 20;
/** Rows of the `parameters` view: tau, rho, initial step, minimum step, half-life, gap tolerance. */
const PARAMETER_COUNT = 6;
/** Words of the `status` view: valid, gap within tolerance, completed cycles, maximum gap bits. */
const STATUS_WORD_COUNT = 4;

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
  output: {
    assignments: GraphDataView<'float32'>;
    scores: GraphDataView<'float32'>;
    unservedDemand: GraphDataView<'float32'>;
    /** [valid, equilibriumGapWithinTolerance, completedCycles, maximumGapBits]. */
    status: GraphDataView<'uint32'>;
  };
};

/**
 * Bounded integral/fractional rational-agent allocation with RAAM's travel/supply normalization,
 * pairwise congestion equalization and decaying demand-shift cap. Final scores use the final
 * allocation. Rounding-stalled allocation is not convergence; status reports the cost gap.
 * Parallel origins move simultaneously. Bounds: 64 origins, 32 facilities, 4096 cycles.
 *
 * Reference tier (contributor contract rule 2): the cycles run in one GPU invocation
 * (`invocationCount: 1`); each cycle loops over every origin and facility, O(cycles * origins *
 * facilities), at most about 8.4e6 operations at the limits (64 * 32 * 4,096), enforced at
 * construction. The shift is already a Jacobi update, so a per-cycle row-parallel kernel with
 * facility load reductions would replace it for larger problems.
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
    validateSinglePackedView(props.output.status, ['uint32'], this.id);
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
      props.output.status.length === STATUS_WORD_COUNT,
      `output.status must hold ${STATUS_WORD_COUNT} rows`
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.costs,
      props.demand,
      props.supply,
      props.parameters
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.costs,
      props.demand,
      props.supply,
      props.parameters,
      ...Object.values(props.output)
    ]);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPUNetworkRationalAccess',
        variant: 'integral-fractional-step',
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
          {name: 'status', view: props.output.status, type: 'u32', access: 'read_write'}
        ],
        declarations: `const ORIGIN_COUNT: u32 = ${props.demand.length}u; const FACILITY_COUNT: u32 = ${props.supply.length}u;
${IS_FINITE_VALUE_WGSL}
fn roundEven(value: f32) -> f32 { let lower = floor(value); let fraction = value - lower; return lower + select(0.0, 1.0, fraction > 0.5 || (fraction == 0.5 && u32(lower) % 2u == 1u)); }`,
        body: `var nanBits = 0x7fc00000u; let missing = bitcast<f32>(nanBits); var infinityBits = 0x7f800000u; let infinity = bitcast<f32>(infinityBits);
for (var word = 0u; word < 4u; word++) { status[statusOffset + word] = 0u; }
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
status[statusOffset] = 1u;
var loads: array<f32, ${props.supply.length}>; var incoming: array<f32, ${props.supply.length}>;
var minima: array<u32, ${props.demand.length}>; var maxima: array<u32, ${props.demand.length}>; var shifts: array<f32, ${props.demand.length}>;
for (var cycle = 0u; cycle <= ${props.maximumCycles ?? 150}u; cycle++) {
  for (var facility = 0u; facility < FACILITY_COUNT; facility++) { loads[facility] = 0.0; incoming[facility] = 0.0; }
  for (var origin = 0u; origin < ORIGIN_COUNT; origin++) { for (var facility = 0u; facility < FACILITY_COUNT; facility++) { loads[facility] += assignments[assignmentsOffset + origin * FACILITY_COUNT + facility]; } }
  var maximumGap = 0.0;
  let step = max(minimumStep, initialStep * exp2(-f32(cycle) / halfLife));
  for (var origin = 0u; origin < ORIGIN_COUNT; origin++) {
    shifts[origin] = 0.0; var lowest = infinity; var highest = -1.0; var minimum = 0u; var maximum = 0u; var weightedCost = 0.0;
    for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
      let travel = costs[costsOffset + origin * FACILITY_COUNT + facility]; if (!isFiniteValue(travel)) { continue; }
      let cost = travel / travelScale + loads[facility] / capacities[facility];
      if (!isFiniteValue(cost)) { status[statusOffset] = 0u; return; }
      let amount = assignments[assignmentsOffset + origin * FACILITY_COUNT + facility];
      if (cost < lowest) { lowest = cost; minimum = facility; }
      if (amount > 0.0 && cost > highest) { highest = cost; maximum = facility; }
      weightedCost += cost * amount;
    }
    let amount = demand[demandOffset + origin];
    if (highest < 0.0 || amount == 0.0) { continue; }
    if (!isFiniteValue(weightedCost)) { status[statusOffset] = 0u; return; }
    scores[scoresOffset + origin] = weightedCost / amount;
    maximumGap = max(maximumGap, highest - lowest);
    minima[origin] = minimum; maxima[origin] = maximum;
    if (minimum == maximum) { continue; }
    let held = assignments[assignmentsOffset + origin * FACILITY_COUNT + maximum];
    // Pairwise equalization before simultaneous shifts; normalization is explicit caller input.
    let equalizingShift = (highest - lowest) / (1.0 / capacities[minimum] + 1.0 / capacities[maximum]);
    let boundedShift = max(0.0, min(held, min(equalizingShift, ${props.stepMode === 'absolute' ? 'step' : 'step * amount'})));
    shifts[origin] = ${props.demandMode === 'fractional' ? 'boundedShift' : 'floor(boundedShift)'};
    incoming[minimum] += shifts[origin];
  }
  status[statusOffset + 2u] = cycle; status[statusOffset + 3u] = bitcast<u32>(maximumGap);
  if (maximumGap <= tolerance) { status[statusOffset + 1u] = 1u; break; }
  if (cycle == ${props.maximumCycles ?? 150}u) { break; }
  for (var origin = 0u; origin < ORIGIN_COUNT; origin++) {
    var shift = shifts[origin]; if (shift == 0.0) { continue; }
    if (cycle < ${props.initialLimitCycles ?? 20}u) { let limited = shift / max(1.0, incoming[minima[origin]] / capacities[minima[origin]]); shift = ${props.demandMode === 'fractional' ? 'limited' : 'roundEven(limited)'}; }
    assignments[assignmentsOffset + origin * FACILITY_COUNT + minima[origin]] += shift;
    assignments[assignmentsOffset + origin * FACILITY_COUNT + maxima[origin]] -= shift;
  }
}`
      })
    ];
  }
}
