// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  getRegressionTileLayout,
  getRowTileSumNodes
} from '../spatial-regression/regression-row-tiles';
import {
  HAWKES_DECAY_REFINEMENTS,
  HAWKES_MINIMUM_SCALED_DECAY,
  HAWKES_WINDOW_INTEGRALS_WGSL
} from './hawkes-finite-window';

/** Profile scratch: rate, bracket, best rate/omega/Q, three window moments, active, entering fit. @internal */
export const HAWKES_PROFILE_LENGTH = 16;

/** Tiled concave finite-window profile solve; no invocation scans the entire event catalogue. @internal */
export function createHawkesDecayProfileNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    rowCount: number;
    stateConstants: string;
    state: GraphDataView<'float32'>;
    events: GraphDataView<'float32'>;
    params: GraphDataView<'float32'>;
    profile: GraphDataView<'float32'>;
    partial: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, state, events, params, profile, partial, stateConstants, rowCount} = props;
  const bindings = [
    {name: 'state', view: state, type: 'f32' as const, access: 'read' as const},
    {name: 'events', view: events, type: 'f32' as const, access: 'read' as const},
    {name: 'params', view: params, type: 'f32' as const, access: 'read' as const},
    {name: 'profile', view: profile, type: 'f32' as const, access: 'read' as const}
  ];
  const minimum = `${HAWKES_MINIMUM_SCALED_DECAY}`;
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode(graph, {
      id: `${id}-profile-initialize`,
      operation: 'GPUSpaceTimeHawkes',
      variant: 'profile-initialize',
      invocationCount: 1,
      bindings: [bindings[0]!, bindings[2]!, {...bindings[3]!, access: 'read_write'}],
      declarations: stateConstants,
      body: `let runs = state[stateOffset + S_DONE] == 0.0;
  profile[profileOffset + 9u] = select(0.0, 1.0, runs);
  if (!runs) { return; }
  let duration = params[paramsOffset + 1u];
  let triggered = state[stateOffset + S_SUMS + 1u];
  let lag = state[stateOffset + S_SUMS + 2u];
  let rate = state[stateOffset + S_DECAY_RATE] * duration;
  let branching = state[stateOffset + S_BRANCHING_RATIO];
  // S*T/L bounds the root: the window-weighted mean lag never exceeds 1/rate. Start Newton at
  // the entering decay, which is close to the root after the first updates.
  let upper = max(${minimum}, triggered * duration / max(lag, 1.1754944e-38));
  profile[profileOffset] = clamp(rate, ${minimum}, upper);
  profile[profileOffset + 1u] = ${minimum};
  profile[profileOffset + 2u] = upper;
  profile[profileOffset + 3u] = rate;
  profile[profileOffset + 4u] = branching;
  profile[profileOffset + 5u] = triggered * (log(branching) + log(rate)) - rate * (lag / duration) - branching * state[stateOffset + S_SUMS + 7u];
  profile[profileOffset + 10u] = state[stateOffset + S_BACKGROUND];
  profile[profileOffset + 11u] = branching;
  profile[profileOffset + 12u] = state[stateOffset + S_DECAY_RATE];
  profile[profileOffset + 13u] = state[stateOffset + S_SPATIAL_VARIANCE];
  profile[profileOffset + 15u] = select(1.0, 0.0, triggered > 1e-6 && lag > 0.0 && state[stateOffset + S_SUMS + 3u] > 0.0);`
    })
  ];
  for (let step = 0; step < HAWKES_DECAY_REFINEMENTS; step++) {
    nodes.push(
      ...getRowTileSumNodes(graph, {
        id: `${id}-profile-${step}`,
        operation: 'GPUSpaceTimeHawkes',
        variant: 'profile-window',
        rowCount,
        layout: getRegressionTileLayout(rowCount),
        sumCount: 3,
        bindings: bindings.slice(1),
        partial,
        output: profile,
        outputSlot: 6,
        declarations: `${stateConstants}\n${IS_FINITE_VALUE_WGSL}\n${HAWKES_WINDOW_INTEGRALS_WGSL}
fn accumulateRow(row:u32, terms:ptr<function,array<f32,TERM_COUNT>>) {
  if (profile[profileOffset + 9u] == 0.0 || profile[profileOffset + 15u] != 0.0) { return; }
  let time = events[eventsOffset + 4u*row + 2u];
  if (!isFiniteValue(time)) { return; }
  let remaining = (params[paramsOffset + 1u] - time) / params[paramsOffset + 1u];
  let integrals = hawkesWindowIntegrals(profile[profileOffset], remaining);
  (*terms)[0] += integrals.x;
  (*terms)[1] += integrals.y;
  (*terms)[2] += integrals.z;
}`
      }),
      createWGSLKernelNode(graph, {
        id: `${id}-profile-${step}-refine`,
        operation: 'GPUSpaceTimeHawkes',
        variant: 'profile-refine',
        invocationCount: 1,
        bindings: [bindings[0]!, bindings[2]!, {...bindings[3]!, access: 'read_write'}],
        declarations: stateConstants,
        body: `if (profile[profileOffset + 9u] == 0.0 || profile[profileOffset + 15u] != 0.0) { return; }
  let rate = profile[profileOffset];
  let zeroth = profile[profileOffset + 6u];
  let first = profile[profileOffset + 7u];
  let second = profile[profileOffset + 8u];
  if (!(zeroth > 0.0)) { return; }
  let triggered = state[stateOffset + S_SUMS + 1u];
  let lag = state[stateOffset + S_SUMS + 2u] / params[paramsOffset + 1u];
  let branching = max(triggered / (rate * zeroth), 1e-30);
  // The last evaluated iterate is the fit: the profile score decreases strictly, so the
  // bracketed root is the maximizer, and Q is flat to below float32 rounding near it.
  profile[profileOffset + 3u] = rate;
  profile[profileOffset + 4u] = branching;
  profile[profileOffset + 5u] = triggered * (log(branching) + log(rate)) - rate * lag - branching * rate * zeroth;
  var score = triggered * (first/zeroth) - lag;
  var curvature = -triggered * max(0.0, second/zeroth - (first/zeroth)*(first/zeroth));
  if (branching <= 1e-30) {
    score = triggered/rate - lag - branching * (zeroth-rate*first);
    curvature = -triggered/(rate*rate) + branching*(2.0*first-rate*second);
  }
  if ((rate == ${minimum} && score <= 0.0) || abs(score) <= 1e-7 * lag) {
    profile[profileOffset + 15u] = 1.0;
    return;
  }
  var lower = profile[profileOffset + 1u];
  var upper = profile[profileOffset + 2u];
  if (score > 0.0) { lower = rate; } else { upper = rate; }
  let candidate = rate - score/curvature;
  var next = (lower+upper)*0.5;
  // A Newton step below an untested zero-decay boundary evaluates the boundary itself: a
  // profile that decreases from beta=0 is then recognized in one refinement, not by bisection.
  if (candidate > lower && candidate < upper) { next = candidate; }
  else if (score < 0.0 && lower == ${minimum} && rate != ${minimum}) { next = lower; }
  profile[profileOffset] = next;
  profile[profileOffset + 1u] = lower;
  profile[profileOffset + 2u] = upper;`
      })
    );
  }
  return nodes;
}
