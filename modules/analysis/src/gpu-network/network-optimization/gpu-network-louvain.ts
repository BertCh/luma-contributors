// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GET_NAN_WGSL, IS_FINITE_VALUE_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {PERMUTATION_RANDOM_WGSL} from '../../analysis-core/kernels/permutation-random';
import {PHILOX_PURPOSES} from '../../analysis-core/kernels/philox-random';
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

/** Largest vertex count: the kernel keeps a dense n x n weight matrix and reevaluates it per move. */
const MAXIMUM_NODE_COUNT = 32;
/** Largest `maximumLevels`: bounds the coarsening loop of the single invocation. */
const MAXIMUM_LEVELS = 32;
/** Largest `maximumIterations`: bounds the local-move sweeps per level (and the trace rows). */
const MAXIMUM_ITERATIONS = 64;
/** Rows of the `parameters` view: resolution and minimum modularity improvement. */
const PARAMETER_COUNT = 2;
/** Words of the `summary` view: valid, converged, completed levels, completed sweeps. */
const SUMMARY_WORD_COUNT = 4;

/**
 * WGSL of the seeded visit order of {@link GPUNetworkLouvain}: `declarations` holds the
 * module-scope table `louvainVisitOrder`, initialised to the identity by a literal, and `body`
 * shuffles it with a Fisher-Yates shuffle that draws unbiased bounded integers from the
 * registered Philox stream, key `(seed, purpose)`, counter `(block, 0, 0, 0)`. The caller's
 * declarations must include `PERMUTATION_RANDOM_WGSL`. The private xorshift32 this replaces was
 * stuck at 0 for seed `0x9e3779b9` and reduced its draws with a biased modulo (review
 * network-paths-08).
 *
 * The identity is a literal initializer on purpose. On the macOS test GPU the stores of a loop
 * with a constant trip count into a zero-initialised local `u32` or `i32` table can be lost, and
 * the table then reads back as zeros. The order table of this kernel, when it was a
 * function-scope `var` filled by `order[vertex] = vertex`, was lost from 12 vertices up (a
 * kernel that does nothing else loses such a table at 33 to 96 entries). Module scope does not
 * prevent it: the browser emits a `var<private>` as the same zero-initialised local, and a
 * private table filled by a loop is lost in the reduced shader. A literal initializer leaves no
 * loop whose stores could be lost (contributor contract section 11).
 * `shader-compiler-table-shapes.spec.ts` runs the old kernel and the cures;
 * `gpu-network-louvain-visit-order.spec.ts` reads this source back on the GPU at sizes up to 512.
 *
 * @internal
 */
export function getGPUNetworkLouvainVisitOrderSource(
  vertexCount: number,
  seed: number
): {declarations: string; body: string} {
  const identity = Array.from({length: vertexCount}, (_, vertex) => `${vertex}u`).join(', ');
  return {
    declarations: `var<private> louvainVisitOrder: array<u32, ${vertexCount}> = array<u32, ${vertexCount}>(${identity});`,
    body: `var visitStream = createPhiloxStream(vec2<u32>(${seed}u, ${PHILOX_PURPOSES.networkLouvainVisitOrder.id}u), 0u, 0u, 0u);
for (var remaining = ${vertexCount}u; remaining > 1u; remaining--) {
  let selected = nextPhiloxBelow(&visitStream, remaining);
  let temporary = louvainVisitOrder[remaining - 1u];
  louvainVisitOrder[remaining - 1u] = louvainVisitOrder[selected];
  louvainVisitOrder[selected] = temporary;
}`
  };
}

export type GPUNetworkLouvainProps = {
  id?: string;
  nodeCount: number;
  /** Directed adjacency uses each arc once, including self loops. Default: undirected. */
  directed?: boolean;
  /**
   * Optional uint32 seed for a reproducible Fisher–Yates visit order; omitted uses ascending IDs.
   * The shuffle draws unbiased bounded integers from the Philox stream registered as
   * `networkLouvainVisitOrder` (key `(seed, purpose)`), so every seed, 0 included, gives a
   * uniform order. A seed gives a different order than before the 2026-11-10 review, which used a
   * private xorshift32.
   */
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
 *
 * Reference tier (contributor contract rule 2): the whole algorithm is one GPU invocation
 * (`invocationCount: 1`) over a dense `nodeCount x nodeCount` weight matrix, and every candidate
 * move re-evaluates the full objective, about n^4 operations per local-move sweep. The maximum size
 * is enforced at construction: 32 vertices, at most 32 levels and 64 sweeps per level (a 32-vertex
 * run is about 2.7e8 serial operations, near the watchdog on a slow GPU). For larger graphs use a
 * parallel multilevel Louvain or Leiden on a CSR; none is published yet.
 */
export class GPUNetworkLouvain implements GPUCommandNodeProducer {
  readonly id: string;
  readonly props: GPUNetworkLouvainProps;
  constructor(props: GPUNetworkLouvainProps) {
    this.id = props.id ?? 'network-louvain';
    this.props = props;
    const {id} = this;
    validateCapacity(id, 'nodeCount', props.nodeCount, 1, MAXIMUM_NODE_COUNT);
    validateRequirement(
      id,
      props.seed === undefined ||
        (Number.isInteger(props.seed) && props.seed >= 0 && props.seed <= 0xffffffff),
      'seed must be an integer in [0, 4294967295]'
    );
    const levels = props.maximumLevels ?? 8;
    const rounds = props.maximumIterations ?? 16;
    validateCapacity(id, 'maximumLevels', levels, 1, MAXIMUM_LEVELS);
    validateCapacity(id, 'maximumIterations', rounds, 1, MAXIMUM_ITERATIONS);
    validateSinglePackedView(props.weights, ['float32'], this.id);
    validateSinglePackedView(props.parameters, ['float32'], this.id);
    validateSinglePackedView(props.output.communities, ['uint32'], this.id);
    validateSinglePackedView(props.output.objectiveTrace, ['float32'], this.id);
    validateSinglePackedView(props.output.summary, ['uint32'], this.id);
    validateRequirement(
      id,
      props.weights.length === props.nodeCount ** 2,
      'weights must hold nodeCount squared rows'
    );
    validateRequirement(
      id,
      props.parameters.length === PARAMETER_COUNT,
      `parameters must hold ${PARAMETER_COUNT} rows: resolution and minimum improvement`
    );
    validateRequirement(
      id,
      props.output.communities.length === props.nodeCount,
      'output.communities must hold nodeCount rows'
    );
    validateRequirement(
      id,
      props.output.objectiveTrace.length === 1 + levels * rounds,
      'output.objectiveTrace must hold 1 + maximumLevels * maximumIterations rows'
    );
    validateRequirement(
      id,
      props.output.summary.length === SUMMARY_WORD_COUNT,
      `output.summary must hold ${SUMMARY_WORD_COUNT} rows`
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
${
  props.seed === undefined
    ? ''
    : `${PERMUTATION_RANDOM_WGSL}
${getGPUNetworkLouvainVisitOrderSource(props.nodeCount, props.seed).declarations}`
}
const NODE_COUNT: u32 = ${props.nodeCount}u;
${IS_FINITE_VALUE_WGSL}
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
        // `groups` is also written only by loops with a constant trip count. Both loops touch
        // storage as well, and the table reads right at every size from 2 to 32 with and without
        // a seed (test/analysis-core/table-audit/table-audit-strict.spec.ts); see
        // getGPUNetworkLouvainVisitOrderSource for the fault this is about.
        body: `for (var word = 0u; word < 4u; word++) { summary[summaryOffset + word] = 0u; }
for (var entry = 0u; entry < ${props.output.objectiveTrace.length}u; entry++) { trace[traceOffset + entry] = getNaN(); }
var groups: array<u32, ${props.nodeCount}>;
${
  props.seed === undefined
    ? // Ascending visits need no order table: the visit index is the group. (A function-scope
      // table filled only by an identity loop read back as zeros from 12 vertices up on the
      // macOS test GPU, see getGPUNetworkLouvainVisitOrderSource, so every visit went to vertex
      // 0 and only one move was ever made.)
      ''
    : getGPUNetworkLouvainVisitOrderSource(props.nodeCount, props.seed).body
}
for (var vertex = 0u; vertex < NODE_COUNT; vertex++) { communities[communitiesOffset + vertex] = vertex; groups[vertex] = vertex; }
let resolution = parameters[parametersOffset];
let tolerance = parameters[parametersOffset + 1u];
if (!isFiniteValue(resolution) || resolution < 0.0 || !isFiniteValue(tolerance) || tolerance < 0.0) { return; }
var total = 0.0;
for (var source = 0u; source < NODE_COUNT; source++) {
  for (var destination = 0u; destination < NODE_COUNT; destination++) {
    let weight = weights[weightsOffset + source * NODE_COUNT + destination];
    if (!isFiniteValue(weight) || weight < 0.0${props.directed ? '' : ' || weight != weights[weightsOffset + destination * NODE_COUNT + source]'}) { return; }
    total += weight;
  }
}
if (!isFiniteValue(total)) { return; }
summary[summaryOffset] = 1u;
if (total == 0.0) { trace[traceOffset] = 0.0; summary[summaryOffset + 1u] = 1u; return; }
var current = objective(total, resolution);
if (!isFiniteValue(current)) { summary[summaryOffset] = 0u; return; }
trace[traceOffset] = current;
var sweeps = 0u;
for (var level = 0u; level < ${props.maximumLevels ?? 8}u; level++) {
  var levelChanged = false;
  var stable = false;
  for (var round = 0u; round < ${props.maximumIterations ?? 16}u; round++) {
    var changed = false;
    for (var visit = 0u; visit < NODE_COUNT; visit++) {
      let group = ${props.seed === undefined ? 'visit' : 'louvainVisitOrder[visit]'};
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
