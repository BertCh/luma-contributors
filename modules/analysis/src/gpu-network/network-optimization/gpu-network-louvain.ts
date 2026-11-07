// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {assert} from '@luma.gl/core';
import type {
  GPUCommandGraph,
  GPUCommandNode,
  GPUCommandNodeProducer,
  GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  validatePackedView,
  validateGraphViewsBelongToGraph,
  validateGraphOutputsDisjointFromInputs
} from '../../analysis-core/kernels/gpu-contributor-utils';

export type GPUNetworkLouvainProps = {
  id?: string;
  nodeCount: number;
  /** Directed adjacency uses each arc once, including self loops. Default: undirected. */
  directed?: boolean;
  /** Optional uint32 seed for reproducible Fisher–Yates visitation; omitted uses ascending IDs. */
  seed?: number;
  /** Symmetric nonnegative row-major adjacency; self-loop diagonals contain twice loop weight. */
  weights: GraphDataView<'float32'>;
  /** Resolution gamma and minimum accepted modularity improvement (both nonnegative). */
  parameters: GraphDataView<'float32'>;
  maximumLevels?: number;
  maximumIterations?: number;
  output: {
    communities: GraphDataView<'uint32'>;
    /** Initial objective, then one objective per completed local-move sweep; unused rows are NaN. */
    objectiveTrace: GraphDataView<'float32'>;
    /** [valid, converged, completedLevels, completedSweeps]. */
    summary: GraphDataView<'uint32'>;
  };
};

/**
 * Reproducible weighted directed/undirected Louvain for at most 32 vertices. Coarsening is represented
 * by immutable vertex groups at each level, preserving original self-loop and internal weights.
 * Exact objective reevaluation makes this a bounded correctness baseline, not a large-graph kernel.
 * Community IDs use ascending or seeded shuffled visitation; modularity is heuristic, never a global optimum certificate.
 */
export class GPUNetworkLouvain implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkLouvainProps;
  constructor(props: GPUNetworkLouvainProps) {
    this.id = props.id ?? 'network-louvain';
    this.props = props;
    assert(Number.isInteger(props.nodeCount) && props.nodeCount > 0 && props.nodeCount <= 32);
    assert(
      props.seed === undefined ||
        (Number.isInteger(props.seed) && props.seed >= 0 && props.seed <= 0xffffffff)
    );
    const levels = props.maximumLevels ?? 8;
    const rounds = props.maximumIterations ?? 16;
    assert(
      Number.isInteger(levels) &&
        levels > 0 &&
        levels <= 32 &&
        Number.isInteger(rounds) &&
        rounds > 0 &&
        rounds <= 64
    );
    validatePackedView(props.weights, ['float32'], this.id);
    validatePackedView(props.parameters, ['float32'], this.id);
    validatePackedView(props.output.communities, ['uint32'], this.id);
    validatePackedView(props.output.objectiveTrace, ['float32'], this.id);
    validatePackedView(props.output.summary, ['uint32'], this.id);
    assert(
      props.weights.length === props.nodeCount ** 2 &&
        props.parameters.length === 2 &&
        props.output.communities.length === props.nodeCount &&
        props.output.objectiveTrace.length === 1 + levels * rounds &&
        props.output.summary.length === 4
    );
    validateGraphOutputsDisjointFromInputs(this.id, Object.values(props.output), [
      props.weights,
      props.parameters
    ]);
  }
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props} = this;
    validateGraphViewsBelongToGraph(id, graph, [
      props.weights,
      props.parameters,
      ...Object.values(props.output)
    ]);
    return [
      createWGSLKernelNode<Parameters>(graph, {
        id,
        operation: 'GPUNetworkLouvain',
        variant: 'bounded-coarsened-local-moves',
        invocationCount: 1,
        workgroupSize: 2,
        bindings: [
          {name: 'weights', view: props.weights, type: 'f32', access: 'read'},
          {name: 'parameters', view: props.parameters, type: 'f32', access: 'read'},
          {name: 'communities', view: props.output.communities, type: 'u32', access: 'read_write'},
          {name: 'trace', view: props.output.objectiveTrace, type: 'f32', access: 'read_write'},
          {name: 'summary', view: props.output.summary, type: 'u32', access: 'read_write'}
        ],
        declarations: `${GET_NAN_WGSL}
const NODE_COUNT: u32 = ${props.nodeCount}u;
fn finite(value: f32) -> bool { return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u; }
fn objective(total: f32, resolution: f32) -> f32 {
  var internal = 0.0;
  var strengths: array<f32, ${props.nodeCount}>;
  var incoming: array<f32, ${props.nodeCount}>;
  for (var source = 0u; source < NODE_COUNT; source++) {
    let sourceCommunity = communities[communitiesOffset + source];
    for (var destination = 0u; destination < NODE_COUNT; destination++) {
      let weight = weights[weightsOffset + source * NODE_COUNT + destination];
      strengths[sourceCommunity] += weight;
      incoming[communities[communitiesOffset + destination]] += weight;
      if (sourceCommunity == communities[communitiesOffset + destination]) { internal += weight; }
    }
  }
  var penalty = 0.0;
  for (var community = 0u; community < NODE_COUNT; community++) { penalty += (strengths[community] / total) * (incoming[community] / total); }
  return internal / total - resolution * penalty;
}`,
        body: `for (var word = 0u; word < 4u; word++) { summary[summaryOffset + word] = 0u; }
for (var entry = 0u; entry < ${props.output.objectiveTrace.length}u; entry++) { trace[traceOffset + entry] = getNaN(); }
var groups: array<u32, ${props.nodeCount}>;
var order: array<u32, ${props.nodeCount}>;
for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { order[vertex] = vertex; }
${
  props.seed === undefined
    ? ''
    : `var randomState = ${props.seed}u ^ 0x9e3779b9u;
for (var remaining = NODE_COUNT; remaining > 1u; remaining--) {
  randomState ^= randomState << 13u; randomState ^= randomState >> 17u; randomState ^= randomState << 5u;
  let selected = randomState % remaining; let temporary = order[remaining - 1u]; order[remaining - 1u] = order[selected]; order[selected] = temporary;
}`
}
for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { communities[communitiesOffset + vertex] = vertex; groups[vertex] = vertex; }
let resolution = parameters[parametersOffset];
let tolerance = parameters[parametersOffset + 1u];
if (!finite(resolution) || resolution < 0.0 || !finite(tolerance) || tolerance < 0.0) { return; }
var total = 0.0;
for (var source = 0u; source < NODE_COUNT; source++) {
  for (var destination = 0u; destination < NODE_COUNT; destination++) {
    let weight = weights[weightsOffset + source * NODE_COUNT + destination];
    if (!finite(weight) || weight < 0.0${props.directed ? '' : ' || weight != weights[weightsOffset + destination * NODE_COUNT + source]'}) { return; }
    total += weight;
  }
}
if (!finite(total)) { return; }
summary[summaryOffset] = 1u;
if (total == 0.0) { trace[traceOffset] = 0.0; summary[summaryOffset + 1u] = 1u; return; }
var current = objective(total, resolution);
if (!finite(current)) { summary[summaryOffset] = 0u; return; }
trace[traceOffset] = current;
var sweeps = 0u;
for (var level = 0u; level < ${props.maximumLevels ?? 8}u; level++) {
  var levelChanged = false;
  var stable = false;
  for (var round = 0u; round < ${props.maximumIterations ?? 16}u; round++) {
    var changed = false;
    for (var visit = 0u; visit < NODE_COUNT; visit++) {
      let group = order[visit];
      var present = false;
      var original = 0u;
      for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { if (groups[vertex] == group) { present = true; original = communities[communitiesOffset + vertex]; } }
      if (!present) { continue; }
      var best = original;
      var bestObjective = current;
      for (var candidate = 0u; candidate < NODE_COUNT; candidate++) {
        // Only existing neighboring communities are Louvain local-move candidates.
        var adjacent = false;
        for (var source = 0u; source < NODE_COUNT; source++) {
          if (groups[source] != group) { continue; }
          for (var destination = 0u; destination < NODE_COUNT; destination++) {
            if (groups[destination] != group && communities[communitiesOffset + destination] == candidate && (weights[weightsOffset + source * NODE_COUNT + destination] > 0.0 || weights[weightsOffset + destination * NODE_COUNT + source] > 0.0)) { adjacent = true; }
          }
        }
        if (!adjacent || candidate == original) { continue; }
        for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { if (groups[vertex] == group) { communities[communitiesOffset + vertex] = candidate; } }
        let candidateObjective = objective(total, resolution);
        if (candidateObjective > bestObjective + tolerance) { best = candidate; bestObjective = candidateObjective; }
        for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { if (groups[vertex] == group) { communities[communitiesOffset + vertex] = original; } }
      }
      for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { if (groups[vertex] == group) { communities[communitiesOffset + vertex] = best; } }
      if (best != original) { changed = true; current = bestObjective; }
    }
    sweeps++;
    trace[traceOffset + sweeps] = current;
    levelChanged = levelChanged || changed;
    if (!changed) { stable = true; break; }
  }
  summary[summaryOffset + 2u] = level + 1u;
  summary[summaryOffset + 3u] = sweeps;
  if (!stable) { return; }
  if (!levelChanged) { summary[summaryOffset + 1u] = 1u; return; }
  // An implicit quotient graph preserves every original weighted edge exactly.
  for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { groups[vertex] = communities[communitiesOffset + vertex]; }
}`
      })
    ];
  }
}
