// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import type {GPUNetworkRationalAccessProps} from './gpu-network-rational-access';

/** One lane per origin/facility; the public caps fit a single portable workgroup. @internal */
export const RATIONAL_ACCESS_WORKGROUP_SIZE = 64;

/** Cooperative Jacobi cycles, preserving the serial solver's addition and publication order. @internal */
export function getRationalAccessCycleKernel(props: GPUNetworkRationalAccessProps) {
  return {
    declarations: `const ORIGIN_COUNT: u32 = ${props.demand.length}u;
const FACILITY_COUNT: u32 = ${props.supply.length}u;
${IS_FINITE_VALUE_WGSL}
var<workgroup> capacities: array<f32, ${props.supply.length}>;
var<workgroup> loads: array<f32, ${props.supply.length}>;
var<workgroup> incoming: array<f32, ${props.supply.length}>;
var<workgroup> minima: array<u32, ${props.demand.length}>;
var<workgroup> maxima: array<u32, ${props.demand.length}>;
var<workgroup> shifts: array<f32, ${props.demand.length}>;
var<workgroup> gaps: array<f32, ${props.demand.length}>;
var<workgroup> nextScores: array<f32, ${props.demand.length}>;
// 0: no score (unserved/zero demand); 1: score ready; 2: nonfinite arithmetic.
var<workgroup> scoreStates: array<u32, ${props.demand.length}>;
var<workgroup> continueCycles: u32;
fn roundEven(value: f32) -> f32 {
  let lower = floor(value); let fraction = value - lower;
  return lower + select(0.0, 1.0, fraction > 0.5 || (fraction == 0.5 && u32(lower) % 2u == 1u));
}`,
    body: `let origin = localInvocationIndex;
let travelScale = parameters[parametersOffset];
let supplyScale = parameters[parametersOffset + 1u];
let initialStep = parameters[parametersOffset + 2u];
let minimumStep = parameters[parametersOffset + 3u];
let halfLife = parameters[parametersOffset + 4u];
let tolerance = parameters[parametersOffset + 5u];
if (localInvocationIndex == 0u) { continueCycles = summary[summaryOffset]; }
if (workgroupUniformLoad(&continueCycles) == 0u) { return; }
if (localInvocationIndex < FACILITY_COUNT) {
  capacities[localInvocationIndex] = supply[supplyOffset + localInvocationIndex] * supplyScale;
}
workgroupBarrier();
for (var cycle = 0u; cycle <= ${props.maximumCycles ?? 150}u; cycle++) {
  // Each facility sums in ascending origin order, exactly as before. There are no float atomics.
  if (localInvocationIndex < FACILITY_COUNT) {
    var load = 0.0;
    for (var row = 0u; row < ORIGIN_COUNT; row++) {
      load += assignments[assignmentsOffset + row * FACILITY_COUNT + localInvocationIndex];
    }
    loads[localInvocationIndex] = load;
  }
  workgroupBarrier();
  if (origin < ORIGIN_COUNT) {
    shifts[origin] = 0.0; gaps[origin] = 0.0; scoreStates[origin] = 0u;
    minima[origin] = 0u; maxima[origin] = 0u;
    var infinityBits = 0x7f800000u; let infinity = bitcast<f32>(infinityBits);
    var lowest = infinity; var highest = -1.0;
    var minimum = 0u; var maximum = 0u; var weightedCost = 0.0;
    for (var facility = 0u; facility < FACILITY_COUNT; facility++) {
      let travel = costs[costsOffset + origin * FACILITY_COUNT + facility];
      if (!isFiniteValue(travel)) { continue; }
      let cost = travel / travelScale + loads[facility] / capacities[facility];
      if (!isFiniteValue(cost)) { scoreStates[origin] = 2u; break; }
      let amount = assignments[assignmentsOffset + origin * FACILITY_COUNT + facility];
      if (cost < lowest) { lowest = cost; minimum = facility; }
      if (amount > 0.0 && cost > highest) { highest = cost; maximum = facility; }
      weightedCost += cost * amount;
    }
    let amount = demand[demandOffset + origin];
    if (scoreStates[origin] != 2u && highest >= 0.0 && amount != 0.0) {
      if (!isFiniteValue(weightedCost)) { scoreStates[origin] = 2u; }
      else {
        nextScores[origin] = weightedCost / amount; scoreStates[origin] = 1u;
        gaps[origin] = highest - lowest;
        minima[origin] = minimum; maxima[origin] = maximum;
        if (minimum != maximum) {
          let held = assignments[assignmentsOffset + origin * FACILITY_COUNT + maximum];
          let equalizingShift = (highest - lowest) / (1.0 / capacities[minimum] + 1.0 / capacities[maximum]);
          let step = max(minimumStep, initialStep * exp2(-f32(cycle) / halfLife));
          let boundedShift = max(0.0, min(held, min(equalizingShift, ${props.stepMode === 'absolute' ? 'step' : 'step * amount'})));
          shifts[origin] = ${props.demandMode === 'fractional' ? 'boundedShift' : 'floor(boundedShift)'};
        }
      }
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    var maximumGap = 0.0;
    // Preserve partial scores through the first invalid origin, including on later cycles.
    for (var row = 0u; row < ORIGIN_COUNT; row++) {
      if (scoreStates[row] == 2u) { summary[summaryOffset] = 0u; continueCycles = 0u; break; }
      if (scoreStates[row] == 1u) { scores[scoresOffset + row] = nextScores[row]; }
      maximumGap = max(maximumGap, gaps[row]);
    }
    if (continueCycles != 0u) {
      summary[summaryOffset + 3u] = bitcast<u32>(maximumGap);
      if (maximumGap <= tolerance) { summary[summaryOffset + 1u] = 1u; continueCycles = 0u; }
      if (cycle == ${props.maximumCycles ?? 150}u) { continueCycles = 0u; }
    }
  }
  if (workgroupUniformLoad(&continueCycles) == 0u) { break; }
  // Facility inflow uses the same ascending-origin sum as the old simultaneous update.
  if (localInvocationIndex < FACILITY_COUNT) {
    var total = 0.0;
    for (var row = 0u; row < ORIGIN_COUNT; row++) {
      if (minima[row] == localInvocationIndex) { total += shifts[row]; }
    }
    incoming[localInvocationIndex] = total;
  }
  workgroupBarrier();
  if (origin < ORIGIN_COUNT) {
    var shift = shifts[origin];
    if (shift != 0.0) {
      if (cycle < ${props.initialLimitCycles ?? 20}u) {
        let limited = shift / max(1.0, incoming[minima[origin]] / capacities[minima[origin]]);
        shift = ${props.demandMode === 'fractional' ? 'limited' : 'roundEven(limited)'};
      }
      assignments[assignmentsOffset + origin * FACILITY_COUNT + minima[origin]] += shift;
      assignments[assignmentsOffset + origin * FACILITY_COUNT + maxima[origin]] -= shift;
    }
  }
  // Every origin finishes its stores before any facility reads the next cycle's allocation.
  storageBarrier();
  if (localInvocationIndex == 0u) { summary[summaryOffset + 2u] = cycle + 1u; }
}`
  };
}
