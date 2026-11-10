// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Typed access to the recorded reachability search (reachability-search.json, written by
// generate-data.ts) and the quantities every figure on the page derives from it.

import gatedJson from './gated-records.json';
import searchJson from './reachability-search.json';

export type ReachabilitySearch = {
  scenario: string;
  stampMeaning: string;
  nodeCount: number;
  directedEdgeCount: number;
  aspectRatio: number;
  /** x, y per node, 0..1, north up. */
  positions: number[];
  /** Undirected edges as index pairs. */
  edges: number[];
  /** Per node: the round whose queue it last joined, right after its cost last improved. */
  lastImprovedRound: number[];
  /** Final cost per node in seconds of travel time, null when unreached. */
  costs: (number | null)[];
  startNodes: number[];
  /** Nodes in each round's queue (the record's frontier-size series). */
  frontier: number[];
  maximumIterations: number;
  localIterations: number;
  relaxRoundCount: number;
  relaxMaximumWorkgroups: number;
  converged: number;
  iterationCount: number;
  /** Dispatch buffers of even and odd rounds, in that order. */
  gateBuffers: string[];
  seedWrites: string[];
  graphNodeCount: number;
  commandNodeBudget: number;
  iterationsLimit: number;
  iterationsMessage: string;
  budgetMessage: string;
};

export type GatedRecord = {id: string; title: string; gated: number; kernels: number};
export type GatedRecords = {
  recordCount: number;
  recordsWithGates: number;
  top: GatedRecord[];
  reachability: GatedRecord | null;
  reachabilityRank: number;
};

export const SEARCH = searchJson as ReachabilitySearch;
export const GATED = gatedJson as GatedRecords;

/**
 * Entries per frontier queue chunk and threads per relax workgroup: every push that opens a new
 * chunk adds one workgroup to the next round
 * (`FRONTIER_WORKGROUP_SIZE`, modules/analysis/src/gpu-network/network-reachability/network-frontier.ts:41).
 */
export const FRONTIER_WORKGROUP_SIZE = 256;

/**
 * Default `maximumIterations` of `GPUNetworkReachability`
 * (modules/analysis/src/gpu-network/network-reachability/gpu-network-reachability.ts:70).
 */
export const DEFAULT_MAXIMUM_ITERATIONS = 64;

/** Workgroups round `round` dispatches: one per started 256-entry chunk of its queue. */
export function getRoundWorkgroups(round: number): number {
  return Math.ceil((SEARCH.frontier[round] ?? 0) / FRONTIER_WORKGROUP_SIZE);
}

/** Rounds whose queue was not empty. */
export const WORKING_ROUNDS = SEARCH.frontier.filter(size => size > 0).length;
/** First round with an empty queue. */
export const FIRST_EMPTY_ROUND = SEARCH.frontier.findIndex(size => size === 0);
export const PEAK_SIZE = Math.max(...SEARCH.frontier);
export const PEAK_ROUND = SEARCH.frontier.indexOf(PEAK_SIZE);
export const DISPATCHED_WORKGROUPS = SEARCH.frontier.reduce(
  (total, _size, round) => total + getRoundWorkgroups(round),
  0
);

/** Nodes whose last queueing was round `round`, by round. */
export const SETTLED_BY_ROUND: number[] = SEARCH.frontier.map(
  (_size, round) => SEARCH.lastImprovedRound.filter(stamp => stamp === round).length
);

/**
 * Status `GPUNetworkReachability` publishes with a round limit of `budget`: a push into a round at
 * or past the limit sets the truncated word instead (network-frontier.ts:226-231), `converged` is
 * 1 only when nothing was truncated, and `iterationCount` counts rounds below the limit with a
 * non-empty queue (network-frontier.ts:296-309, gpu-network-reachability.ts:179-186).
 */
export function getBudgetStatus(budget: number) {
  const queuedPastLimit = (SEARCH.frontier[budget] ?? 0) > 0;
  const iterationCount = SEARCH.frontier.slice(0, budget).filter(size => size > 0).length;
  // A node joined a queue at round s after its cost improved in round s - 1, so its final
  // cost was written only when round s - 1 ran, that is when s <= budget.
  const notFinal = SEARCH.lastImprovedRound.filter(stamp => stamp > budget).length;
  return {converged: queuedPastLimit ? 0 : 1, iterationCount, notFinal};
}
