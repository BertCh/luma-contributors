// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  validatePackedUint32View,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_NAN_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {
  validateCapacity,
  validateCommandNodeBudget,
  validateGraphOutputsDisjointFromInputs,
  validateGraphViewsBelongToGraph,
  validateSinglePackedView,
  validateGPUStorageBindingByteLength,
  getGraphOutputViews
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {
  validateGPUStatusPort,
  type GPUIterationStatusPort
} from '../../analysis-core/contracts/index';
import {createTransitionDiagnosticsSummaryNodes} from './transition-diagnostics-summary';

import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';

const OPERATION = 'GPUTransitionDiagnostics';
/** GTH elimination retains a bounded reference state system; condition summaries use parallel reductions. */
export const GPU_TRANSITION_DIAGNOSTICS_MAXIMUM_CLASS_COUNT = 64;

/** Two integer condition reductions (at most four 256-way levels plus publication each), four public sums (at most five nodes each), and at most twelve summary/solver nodes. */
const DIAGNOSTIC_COMMAND_NODE_ALLOWANCE = 64;

/** Default {@link GPUTransitionDiagnosticsProps.minimumTransitions}. */
export const GPU_TRANSITION_DEFAULT_MINIMUM_TRANSITIONS = 5;

/** Row layout of the `homogeneity` output of {@link GPUTransitionDiagnostics}. */
export const GPU_TRANSITION_HOMOGENEITY = {
  /** Chi-square statistic Q (giddy `Spatial_Markov.Q`). */
  Q: 0,
  /** Upper-tail chi-square p-value of Q (`Q_p_value`); NaN when the degrees of freedom are 0. */
  Q_P_VALUE: 1,
  /** Likelihood-ratio statistic LR (`LR`). */
  LR: 2,
  /** Upper-tail chi-square p-value of LR (`LR_p_value`); NaN when the degrees of freedom are 0. */
  LR_P_VALUE: 3,
  /** Degrees of freedom (`dof_hom`), exact in f32. */
  DEGREES_OF_FREEDOM: 4
} as const;

/** Number of rows of the `homogeneity` output. */
export const GPU_TRANSITION_HOMOGENEITY_LENGTH = 5;

/** Row layout of the `stateSparsity` output of {@link GPUTransitionDiagnostics}. */
export const GPU_TRANSITION_STATE_SPARSITY = {
  /** Number of `(condition, from)` states with at least one but fewer than `minimumTransitions` transitions. */
  SPARSE: 0,
  /** Number of `(condition, from)` states with no transitions. */
  EMPTY: 1
} as const;

/** Number of rows of the `stateSparsity` output. */
export const GPU_TRANSITION_STATE_SPARSITY_LENGTH = 2;

/** Caller-owned outputs of {@link GPUTransitionDiagnostics}. */
export type GPUTransitionDiagnosticsOutput = {
  /**
   * Optional homogeneity test of the conditional transition matrices (see
   * {@link GPU_TRANSITION_HOMOGENEITY}), at least {@link GPU_TRANSITION_HOMOGENEITY_LENGTH} rows.
   * Needs `conditionCount > 1`.
   */
  homogeneity?: GraphDataView<'float32'>;
  /**
   * Optional steady-state (ergodic) distributions, `chains * classCount` rows with `chains = 1`
   * for `conditionCount = 1` and `conditionCount + 1` otherwise. Chain 0 is the pooled chain
   * (counts summed over the conditions); chain `1 + c` is the chain of condition `c`. Row
   * `chain * classCount + s` is the long-run probability of state `s`. A chain without any
   * transition, with a state that is entered but never left, or that is reducible after dropping
   * its unused states, is all NaN.
   */
  steadyStates?: GraphDataView<'float32'>;
  /**
   * Optional minimum-cell-count diagnostic (see {@link GPU_TRANSITION_STATE_SPARSITY}), at least
   * {@link GPU_TRANSITION_STATE_SPARSITY_LENGTH} rows.
   */
  stateSparsity?: GraphDataView<'uint32'>;
  /**
   * Optional status. `converged` is one when every chain with at least one transition got a
   * steady state, zero otherwise (see the NaN rows of `steadyStates` for which chain). The
   * solver is direct, so `iterationLimitReached` is never written. Needs `steadyStates`.
   */
  status?: Pick<GPUIterationStatusPort, 'converged'>;
};

/** Properties for {@link GPUTransitionDiagnostics}. */
export type GPUTransitionDiagnosticsProps = {
  /** Prefix for generated node IDs. Defaults to `'transition-diagnostics'`. */
  id?: string;
  /**
   * Exact transition counts, `conditionCount * classCount * classCount` rows, laid out as the
   * `counts` of {@link GPUTransitionMatrix}.
   */
  counts: GraphDataView<'uint32'>;
  /** Number of classes `K`, 1 to 64. */
  classCount: number;
  /** Number of condition classes `C`. Defaults to 1. */
  conditionCount?: number;
  /**
   * Compile-time. A `(condition, from)` state with fewer transitions than this is counted as
   * sparse. Defaults to {@link GPU_TRANSITION_DEFAULT_MINIMUM_TRANSITIONS}, the classic rule of
   * thumb of five observations per cell row for chi-square tests of transition matrices.
   */
  minimumTransitions?: number;
  /** Caller-owned outputs. */
  output: GPUTransitionDiagnosticsOutput;
};

/**
 * Reads each declared scalar and nested port field once (own, inherited or accessor) before any
 * validation; a spread would drop an inherited `minimumTransitions`. @internal
 */
function captureGPUTransitionDiagnosticsProps(
  props: GPUTransitionDiagnosticsProps
): GPUTransitionDiagnosticsProps {
  const {id, counts, classCount, conditionCount, minimumTransitions, output} = props;
  const {homogeneity, steadyStates, stateSparsity, status} = output;
  return {
    id,
    counts,
    classCount,
    conditionCount,
    minimumTransitions,
    output: {
      homogeneity,
      steadyStates,
      stateSparsity,
      status: status ? {converged: status.converged} : undefined
    }
  };
}

/**
 * Diagnostics of a transition-count tensor: homogeneity of the conditional matrices, steady
 * states of the pooled and conditional chains, and a minimum-cell-count count.
 *
 * Homogeneity (Bickenbach and Bode 2003, giddy `homogeneity` behind `Spatial_Markov.Q`, `LR`,
 * `dof_hom`, `Q_p_value`, `LR_p_value`): the null is that every conditional matrix equals the
 * pooled matrix. `Q = sum n_im (p_ijm - p_ij)^2 / p_ij` and `LR = 2 sum n_ijm ln(p_ijm / p_ij)` are
 * asymptotically chi-square with `sum_i (b_i - 1)(A_i - 1)` degrees of freedom, `b_i` the number of
 * conditions with a transition out of state `i` and `A_i` the number of non-zero pooled cells of
 * row `i`. A from-state without any transition adds 0, whereas giddy's expression adds 1 for it
 * (`(0 - 1)(0 - 1)`), so the degrees of freedom differ from giddy's exactly by the number of empty
 * pooled rows. The p-values are upper-tail (`'greater'`) chi-square probabilities from an f32
 * incomplete gamma, see `GPU_UPPER_INCOMPLETE_GAMMA_WGSL`. The test is asymptotic: it is not
 * trustworthy with sparse rows, see `stateSparsity`.
 *
 * Steady states: the stationary distribution of the row-normalized counts, by the
 * Grassmann-Taksar-Heyman elimination (subtraction free, so stable in f32), one invocation per
 * chain. States with no transition in or out are dropped and get probability 0, as giddy's
 * `Markov` drops classes absent from the data; the rest must form one communicating class and
 * every one of them must have an outgoing transition. Otherwise the chain is not ergodic (giddy
 * raises or returns several distributions) and the row is NaN.
 *
 * Counts are exact; the statistics are f32 sums, accurate to about 1e-5 relative.
 */
export class GPUTransitionDiagnostics implements GPUCommandNodeProducer {
  /** Prefix for every node ID. */
  readonly id: string;
  /** Validated properties. */
  readonly props: GPUTransitionDiagnosticsProps;
  /** Resolved condition class count. */
  readonly conditionCount: number;
  /** Number of steady-state chains: 1, or `conditionCount + 1` with conditions. */
  readonly chainCount: number;
  /** Resolved minimum transitions per state. */
  readonly minimumTransitions: number;
  /** Upper bound including public reduction nodes at every supported tensor length. */
  readonly estimatedCommandNodeCount = DIAGNOSTIC_COMMAND_NODE_ALLOWANCE;

  constructor(props: GPUTransitionDiagnosticsProps) {
    props = captureGPUTransitionDiagnosticsProps(props);
    const id = props.id ?? 'transition-diagnostics';
    this.id = id;
    validateCommandNodeBudget(id, this.estimatedCommandNodeCount);
    this.props = props;
    this.conditionCount = props.conditionCount ?? 1;
    this.chainCount = this.conditionCount > 1 ? this.conditionCount + 1 : 1;
    this.minimumTransitions =
      props.minimumTransitions ?? GPU_TRANSITION_DEFAULT_MINIMUM_TRANSITIONS;
    const {classCount, output} = props;
    validateCapacity(
      id,
      'classCount',
      classCount,
      1,
      GPU_TRANSITION_DIAGNOSTICS_MAXIMUM_CLASS_COUNT
    );
    validateCapacity(id, 'conditionCount', this.conditionCount);
    validateCapacity(id, 'minimumTransitions', this.minimumTransitions, 1, 0xffffffff);
    validatePackedUint32View(props.counts, `${id} counts`);
    if (props.counts.length < this.conditionCount * classCount * classCount) {
      throw new Error(`${id} counts is too short`);
    }
    if (output.homogeneity) {
      validateSinglePackedView(output.homogeneity, ['float32'], `${id} output.homogeneity`);
      if (output.homogeneity.length < GPU_TRANSITION_HOMOGENEITY_LENGTH) {
        throw new Error(`${id} output.homogeneity is too short`);
      }
      if (this.conditionCount < 2) {
        // A homogeneity test compares at least two conditional matrices.
        throw new Error(`${id} output.homogeneity needs conditionCount of at least 2`);
      }
    }
    if (output.steadyStates) {
      validateSinglePackedView(output.steadyStates, ['float32'], `${id} output.steadyStates`);
      if (output.steadyStates.length < this.chainCount * classCount) {
        throw new Error(`${id} output.steadyStates is too short`);
      }
    }
    if (output.stateSparsity) {
      validatePackedUint32View(output.stateSparsity, `${id} output.stateSparsity`);
      if (output.stateSparsity.length < GPU_TRANSITION_STATE_SPARSITY_LENGTH) {
        throw new Error(`${id} output.stateSparsity is too short`);
      }
    }
    if (output.status) {
      validateGPUStatusPort(id, output.status);
      if (output.status.converged && !output.steadyStates) {
        throw new Error(`${id} output.status.converged needs output.steadyStates`);
      }
    }
    const outputs = [
      output.homogeneity,
      output.steadyStates,
      output.stateSparsity,
      output.status?.converged
    ];
    validateGraphOutputsDisjointFromInputs(id, outputs, [props.counts]);
    const buffers = outputs.filter(view => view !== undefined).map(view => view.buffer);
    if (new Set(buffers).size !== buffers.length) {
      throw new Error(`${id} outputs must use separate buffers`);
    }
  }

  /** Returns parallel homogeneity/sparsity summaries, status reset and the dense steady-state node. */
  getCommandNodes<Parameters>(
    graph: GPUCommandGraph<Parameters>
  ): readonly GPUCommandNode<Parameters>[] {
    const {id, props, conditionCount, chainCount, minimumTransitions} = this;
    const {counts, classCount, output} = props;
    const converged = output.status?.converged;
    validateGraphViewsBelongToGraph(id, graph, [
      counts,
      output.homogeneity,
      output.steadyStates,
      output.stateSparsity,
      converged
    ]);
    validateGraphChunkBindingRanges(graph, id, {
      counts,
      ...Object.fromEntries(
        getGraphOutputViews({...output, status: {converged: output.status?.converged}}).map(
          (view, index) => [`output[${index}]`, view]
        )
      )
    });
    if (output.steadyStates)
      validateGPUStorageBindingByteLength(
        graph.device,
        id,
        'chainWork',
        chainCount * classCount * classCount * 4,
        'reduce conditionCount or classCount'
      );
    const constants = `const K: u32 = ${classCount}u;
const C: u32 = ${conditionCount}u;
const MINIMUM_TRANSITIONS: u32 = ${minimumTransitions}u;`;
    const nodes: GPUCommandNode<Parameters>[] = [];
    const summary = createTransitionDiagnosticsSummaryNodes(graph, {
      id,
      counts,
      classCount,
      conditionCount,
      minimumTransitions,
      output
    });
    nodes.push(...summary.nodes);
    const pooledCounts = summary.pooledCounts;
    if (output.steadyStates) {
      // The elimination matrix of every chain (K * K floats each) lives in storage: at 64 classes
      // it would take 16 KiB of private memory per invocation, twice the core limit of 8,192 bytes.
      const chainWork = createTransientView(
        graph,
        `${id}-chain-work`,
        'float32',
        chainCount * classCount * classCount
      );
      nodes.push(
        createWGSLKernelNode<Parameters>(graph, {
          id: `${id}-steady-states`,
          operation: OPERATION,
          variant: 'steady-states',
          bindings: [
            {name: 'counts', view: counts, type: 'u32', access: 'read'},
            ...(pooledCounts
              ? [
                  {
                    name: 'pooledCounts',
                    view: pooledCounts,
                    type: 'f32' as const,
                    access: 'read' as const
                  }
                ]
              : []),
            {
              name: 'steadyStates',
              view: output.steadyStates,
              type: 'f32',
              access: 'read_write'
            },
            {name: 'chainWork', view: chainWork, type: 'f32', access: 'read_write'},
            ...(converged
              ? [
                  {
                    name: 'converged',
                    view: converged,
                    type: 'atomic<u32>' as const,
                    access: 'read_write' as const
                  }
                ]
              : [])
          ],
          invocationCount: chainCount,
          declarations: `${constants}
${GET_NAN_WGSL}
// Private memory: 2 * K words (at most 512 bytes), far below the 8,192-byte core limit.
var<private> usedStates: array<u32, ${classCount}>;
var<private> solution: array<f32, ${classCount}>;

// Transition count of one chain: chain 0 is the pooled chain when there are conditions.
fn getChainCount(chain: u32, origin: u32, to: u32) -> f32 {
  ${
    pooledCounts
      ? `if (chain == 0u) {
    return pooledCounts[pooledCountsOffset + origin * K + to];
  }
  return f32(counts[countsOffset + ((chain - 1u) * K + origin) * K + to]);`
      : 'return f32(counts[countsOffset + origin * K + to]);'
  }
}`,
          body: `let chain = index;
  let workBase = chainWorkOffset + chain * K * K;
  let nan = getNaN();
  var activeCount = 0u;
  var solvable = true;
  for (var state = 0u; state < K; state++) {
    var outgoing = 0.0;
    var incoming = 0.0;
    for (var other = 0u; other < K; other++) {
      outgoing += getChainCount(chain, state, other);
      incoming += getChainCount(chain, other, state);
    }
    if (outgoing > 0.0 || incoming > 0.0) {
      usedStates[activeCount] = state;
      activeCount += 1u;
      if (outgoing == 0.0) {
        solvable = false;
      }
    }
  }
  if (activeCount == 0u) {
    // No transitions at all: nothing to solve, and not a failure of the solver.
    for (var state = 0u; state < K; state++) {
      steadyStates[steadyStatesOffset + chain * K + state] = nan;
    }
    return;
  }
  let m = activeCount;
  // GTH escape sums are only a one-sided test: transient higher-numbered states can escape
  // to a lower closed class. Check reachability from state 0 in both support directions.
  // Reuse solution as 0/1 reachability scratch before back-substitution overwrites it.
  for (var direction = 0u; direction < 2u; direction++) {
    for (var stateIndex = 0u; stateIndex < m; stateIndex++) {
      solution[stateIndex] = select(0.0, 1.0, stateIndex == 0u);
    }
    for (var traversal = 0u; traversal < m; traversal++) {
      var changed = false;
      for (var originIndex = 0u; originIndex < m; originIndex++) {
        if (solution[originIndex] == 0.0) { continue; }
        for (var destinationIndex = 0u; destinationIndex < m; destinationIndex++) {
          let origin = select(usedStates[originIndex], usedStates[destinationIndex], direction == 1u);
          let destination = select(usedStates[destinationIndex], usedStates[originIndex], direction == 1u);
          if (getChainCount(chain, origin, destination) > 0.0 && solution[destinationIndex] == 0.0) {
            solution[destinationIndex] = 1.0;
            changed = true;
          }
        }
      }
      if (!changed) { break; }
    }
    for (var stateIndex = 0u; stateIndex < m; stateIndex++) {
      if (solution[stateIndex] == 0.0) { solvable = false; }
    }
  }
  if (solvable) {
    for (var a = 0u; a < m; a++) {
      var rowTotal = 0.0;
      for (var b = 0u; b < m; b++) {
        rowTotal += getChainCount(chain, usedStates[a], usedStates[b]);
      }
      for (var b = 0u; b < m; b++) {
        chainWork[workBase + a * m + b] = getChainCount(chain, usedStates[a], usedStates[b]) / rowTotal;
      }
    }
    // Grassmann-Taksar-Heyman: eliminate the last state repeatedly. The sum of the remaining
    // outgoing probabilities of a state is zero only when it cannot reach the lower states.
    for (var k = m - 1u; k >= 1u; k--) {
      var escape = 0.0;
      for (var j = 0u; j < k; j++) {
        escape += chainWork[workBase + k * m + j];
      }
      if (!(escape > 0.0)) {
        solvable = false;
        break;
      }
      for (var i = 0u; i < k; i++) {
        chainWork[workBase + i * m + k] = chainWork[workBase + i * m + k] / escape;
      }
      for (var i = 0u; i < k; i++) {
        for (var j = 0u; j < k; j++) {
          chainWork[workBase + i * m + j] = chainWork[workBase + i * m + j] + chainWork[workBase + i * m + k] * chainWork[workBase + k * m + j];
        }
      }
    }
  }
  if (solvable) {
    solution[0] = 1.0;
    var total = 1.0;
    for (var k = 1u; k < m; k++) {
      var weight = 0.0;
      for (var i = 0u; i < k; i++) {
        weight += solution[i] * chainWork[workBase + i * m + k];
      }
      solution[k] = weight;
      total += weight;
    }
    // Float32 GTH weights can overflow on a highly biased chain. Do not certify NaN
    // probabilities as converged: publish the existing whole-chain failure convention.
    solvable = total > 0.0 && (bitcast<u32>(total) & 0x7f800000u) != 0x7f800000u;
  }
  if (solvable) {
    var total = 0.0;
    for (var stateIndex = 0u; stateIndex < m; stateIndex++) {
      total += solution[stateIndex];
    }
    for (var state = 0u; state < K; state++) {
      steadyStates[steadyStatesOffset + chain * K + state] = 0.0;
    }
    for (var a = 0u; a < m; a++) {
      steadyStates[steadyStatesOffset + chain * K + usedStates[a]] = solution[a] / total;
    }
  } else {
    for (var state = 0u; state < K; state++) {
      steadyStates[steadyStatesOffset + chain * K + state] = nan;
    }
    ${converged ? 'atomicMin(&converged[convergedOffset], 0u);' : ''}
  }`
        })
      );
    }
    return nodes;
  }
}
