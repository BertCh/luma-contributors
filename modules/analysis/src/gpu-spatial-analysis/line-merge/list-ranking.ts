// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';

/** Pointer value that ends a list. */
export const LIST_RANKING_NONE = 0xffffffff;

/**
 * Rule that combines a value with the value of the node it points to in one jumping round.
 * `'add'` sums (list lengths and prefix sums), `'minimum'` keeps the smaller (cycle leaders), and
 * `'replace'` takes the pointed node's value (the head of the list).
 */
export type ListRankingRule = 'add' | 'minimum' | 'replace';

/** A `uint32` value carried along the pointers of a list-ranking pass. */
export type ListRankingValue = {
  /** Name used in the kernel bindings. */
  name: string;
  /** Ping-pong pair; element 0 holds the initial values. */
  views: readonly [GraphDataView<'uint32'>, GraphDataView<'uint32'>];
  rule: ListRankingRule;
};

/** Properties of one list-ranking round or of a complete pass. */
export type ListRankingProps = {
  /** Prefix for node IDs; rounds append `-${round}`. */
  id: string;
  /** Operation name of the owning contributor. */
  operation: string;
  /** Kernel variant, shared by every round so rounds reuse one pipeline. */
  variant: string;
  /** Number of nodes (kernel invocations). */
  length: number;
  /**
   * Number of rounds of a complete pass, which must reach the end of the longest list: at least
   * `ceil(log2(longest list length))`. Defaults to {@link getListRankingRoundCount} of `length`.
   */
  rounds?: number;
  /** Ping-pong pair of pointers to the previous node (or {@link LIST_RANKING_NONE}); element 0 is the input. */
  pointers: readonly [GraphDataView<'uint32'>, GraphDataView<'uint32'>];
  /** Values carried along the pointers. */
  values: readonly ListRankingValue[];
};

/**
 * Number of Wyllie pointer-jumping rounds that resolve lists of up to `length` nodes: after
 * `ceil(log2(length))` rounds every pointer has reached the end of its list.
 */
export function getListRankingRoundCount(length: number): number {
  return Math.max(1, Math.ceil(Math.log2(Math.max(length, 2))));
}

/**
 * One round of Wyllie pointer jumping over `length` nodes (list ranking in `O(log n)` rounds with
 * no thread walking a list). Node `i` with pointer `p = pointers[from][i]` not equal to
 * {@link LIST_RANKING_NONE} combines its values with those of `p` by each value's rule and takes
 * `pointers[from][p]` as its next pointer; other nodes copy their state. Round `round` reads the
 * pair element `round % 2` and writes the other. A pointer to itself stays in place, which lets a
 * pass resolve "first node satisfying a condition" lists.
 *
 * @internal
 */
export function createListRankingRoundNode<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ListRankingProps,
  round: number
): GPUCommandNode<Parameters> {
  const from = round % 2;
  const to = 1 - from;
  const bindings: WGSLKernelBinding[] = [
    {name: 'pointerIn', view: props.pointers[from], type: 'u32', access: 'read'},
    {name: 'pointerOut', view: props.pointers[to], type: 'u32', access: 'read_write'}
  ];
  let loads = '';
  let combines = '';
  let stores = '';
  for (const value of props.values) {
    bindings.push(
      {name: `${value.name}In`, view: value.views[from], type: 'u32', access: 'read'},
      {name: `${value.name}Out`, view: value.views[to], type: 'u32', access: 'read_write'}
    );
    const input = `${value.name}In[${value.name}InOffset`;
    loads += `\n  var ${value.name} = ${input} + index];`;
    const ahead = `${input} + previous]`;
    combines +=
      value.rule === 'add'
        ? `\n    ${value.name} += ${ahead};`
        : value.rule === 'minimum'
          ? `\n    ${value.name} = min(${value.name}, ${ahead});`
          : `\n    ${value.name} = ${ahead};`;
    stores += `\n  ${value.name}Out[${value.name}OutOffset + index] = ${value.name};`;
  }
  return createWGSLKernelNode<Parameters>(graph, {
    id: `${props.id}-${round}`,
    operation: props.operation,
    variant: props.variant,
    bindings,
    invocationCount: props.length,
    declarations: `const LIST_NONE: u32 = ${LIST_RANKING_NONE}u;`,
    body: `let previous = pointerIn[pointerInOffset + index];${loads}
  var nextPrevious = previous;
  if (previous != LIST_NONE) {${combines}
    nextPrevious = pointerIn[pointerInOffset + previous];
  }
  pointerOut[pointerOutOffset + index] = nextPrevious;${stores}`
  });
}

/**
 * A complete list-ranking pass: {@link getListRankingRoundCount} rounds, so the result is in the
 * pair element `rounds % 2`.
 *
 * @internal
 */
export function createListRankingNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: ListRankingProps
): {nodes: GPUCommandNode<Parameters>[]; rounds: number; finalIndex: number} {
  const rounds = props.rounds ?? getListRankingRoundCount(props.length);
  const nodes: GPUCommandNode<Parameters>[] = [];
  for (let round = 0; round < rounds; round++) {
    nodes.push(createListRankingRoundNode(graph, props, round));
  }
  return {nodes, rounds, finalIndex: rounds % 2};
}
