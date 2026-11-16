// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUAnalysisOperatorCatalog} from '../all-entry-points/analysis-operator-catalog';
import {GPU_SPATIAL_ANALYSIS_DATA_KINDS} from '../spatial-analysis-ontology';
import type {GPUAnalysisOperator, GPUSpatialAnalysisDataKind} from '../spatial-analysis-ontology';
import {getGPUAnalysisOperatorQuestions} from './analysis-operator-questions';
import {
  GPU_ANALYSIS_OPERATOR_KIND_RECORDS,
  type GPUAnalysisOperatorKinds
} from './analysis-operator-kind-records';

export * from './analysis-operator-kind-records';

/** Most operators a path may chain. Longer chains are rarely meaningful and the count grows fast. */
export const GPU_ANALYSIS_MAXIMUM_PATH_STEPS = 3;

/** Paths {@link findGPUAnalysisOperatorPaths} returns when `maximumPaths` is not given. */
export const GPU_ANALYSIS_MAXIMUM_PATH_COUNT = 50;

/** Largest `maximumPaths` accepted, so one answer stays small enough to render. */
export const GPU_ANALYSIS_MAXIMUM_PATH_COUNT_LIMIT = 500;

/**
 * Most distinct chains one search counts, in final ranking order. This bounds work and retained
 * candidates at hub kinds while exceeding the largest requested answer, so truncation keeps the best paths.
 */
export const GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES = 10000;

/** Where one side (inputs or outputs) of an operator's kinds came from. */
export type GPUAnalysisOperatorKindSideSource = 'catalogue' | 'supplement' | 'unknown';

/** Where {@link resolveGPUAnalysisOperatorKinds} got its answer: one source, `mixed`, or `unknown`. */
export type GPUAnalysisOperatorKindSource = GPUAnalysisOperatorKindSideSource | 'mixed';

/** Resolved kinds of one operator and where each side came from. */
export type GPUAnalysisResolvedOperatorKinds = GPUAnalysisOperatorKinds & {
  /**
   * `catalogue` or `supplement` when every known side came from there, `mixed` when one side came
   * from each, `unknown` when neither side is known.
   */
  source: GPUAnalysisOperatorKindSource;
  inputsSource: GPUAnalysisOperatorKindSideSource;
  outputsSource: GPUAnalysisOperatorKindSideSource;
};

/** The part of a catalogue record that kind resolution reads. */
export type GPUAnalysisOperatorKindRecord = Pick<
  GPUAnalysisOperator,
  'exportName' | 'inputs' | 'outputs'
>;

function validateDataKind(functionName: string, argumentName: string, value: unknown): void {
  if (!GPU_SPATIAL_ANALYSIS_DATA_KINDS.includes(value as GPUSpatialAnalysisDataKind)) {
    throw new Error(
      `${functionName} ${argumentName} ${JSON.stringify(value)} is not a data kind; accepted: ${GPU_SPATIAL_ANALYSIS_DATA_KINDS.join(', ')}`
    );
  }
}

/**
 * The kinds of one operator, resolved one side at a time: the catalogue record's array when it is
 * not empty, otherwise the supplement row's array, otherwise empty. The result says where each
 * side came from, and `source` is `mixed` when the two sides differ (for example a record that
 * lists outputs only, with the inputs taken from the row). A supplement row may declare empty
 * outputs for the operators in {@link GPU_ANALYSIS_HANDLE_OPERATORS}; that counts as a known answer
 * (`supplement`). An operator the catalogue describes on neither side and no row covers is
 * `unknown` with empty arrays for external records. The completeness spec rejects unknown sides
 * in the published catalogue.
 */
export function resolveGPUAnalysisOperatorKinds(
  operator: GPUAnalysisOperatorKindRecord
): GPUAnalysisResolvedOperatorKinds {
  const row = Object.hasOwn(GPU_ANALYSIS_OPERATOR_KIND_RECORDS, operator.exportName)
    ? GPU_ANALYSIS_OPERATOR_KIND_RECORDS[operator.exportName]
    : undefined;
  const inputs = resolveSide(operator.inputs, row?.inputs);
  const outputs = resolveSide(operator.outputs, row?.outputs);
  const known = new Set([inputs.source, outputs.source].filter(source => source !== 'unknown'));
  const source: GPUAnalysisOperatorKindSource =
    known.size === 0 ? 'unknown' : known.size === 1 ? [...known][0] : 'mixed';
  return {
    inputs: inputs.kinds,
    outputs: outputs.kinds,
    source,
    inputsSource: inputs.source,
    outputsSource: outputs.source
  };
}

function resolveSide(
  catalogueKinds: readonly GPUSpatialAnalysisDataKind[],
  supplementKinds: readonly GPUSpatialAnalysisDataKind[] | undefined
): {kinds: readonly GPUSpatialAnalysisDataKind[]; source: GPUAnalysisOperatorKindSideSource} {
  if (catalogueKinds.length > 0) return {kinds: catalogueKinds, source: 'catalogue'};
  if (supplementKinds) return {kinds: supplementKinds, source: 'supplement'};
  return {kinds: [], source: 'unknown'};
}

type IndexedOperator = {
  operator: GPUAnalysisOperator;
  kinds: GPUAnalysisResolvedOperatorKinds;
  /** Primary question is `prepare-data`: a step that gets data ready rather than answers anything. */
  isPlumbing: boolean;
  /** A workflow recipe (`add...Recipe`), which chains several operators already. */
  isRecipe: boolean;
};

type OperatorIndex = {
  /** The catalogue array this index was built from; a different array means a rebuild. */
  catalogue: readonly GPUAnalysisOperator[];
  /** Non-helper operators in catalogue order. */
  inCatalogueOrder: readonly IndexedOperator[];
  /** Non-helper operators sorted by name. */
  byName: readonly IndexedOperator[];
  indexedByName: ReadonlyMap<string, IndexedOperator>;
  /** For each input kind and then output kind, the operators making that step, sorted by name. */
  transitions: ReadonlyMap<
    GPUSpatialAnalysisDataKind,
    ReadonlyMap<GPUSpatialAnalysisDataKind, readonly IndexedOperator[]>
  >;
};

let cachedIndex: OperatorIndex | undefined;

function compareNames(left: string, right: string): number {
  return left < right ? -1 : left > right ? 1 : 0;
}

/**
 * The operator index, rebuilt only when the catalogue array is a different one (the catalogue array
 * is replaced when the catalogue changes), so repeated queries do not resolve and sort again.
 */
function getOperatorIndex(): OperatorIndex {
  const catalogue = getGPUAnalysisOperatorCatalog();
  if (cachedIndex?.catalogue === catalogue) return cachedIndex;
  const inCatalogueOrder = catalogue
    .filter(operator => operator.role !== 'helper')
    .map(operator => ({
      operator,
      kinds: resolveGPUAnalysisOperatorKinds(operator),
      isPlumbing: getGPUAnalysisOperatorQuestions(operator)[0] === 'prepare-data',
      isRecipe: /^add[A-Z]/.test(operator.exportName)
    }));
  const byName = [...inCatalogueOrder].sort((left, right) =>
    compareNames(left.operator.exportName, right.operator.exportName)
  );
  const transitions = new Map<
    GPUSpatialAnalysisDataKind,
    Map<GPUSpatialAnalysisDataKind, IndexedOperator[]>
  >();
  for (const indexed of byName) {
    for (const inputKind of indexed.kinds.inputs) {
      const byOutput = transitions.get(inputKind) ?? new Map();
      transitions.set(inputKind, byOutput);
      for (const outputKind of indexed.kinds.outputs) {
        const operators = byOutput.get(outputKind) ?? [];
        byOutput.set(outputKind, operators);
        operators.push(indexed);
      }
    }
  }
  cachedIndex = {
    catalogue,
    inCatalogueOrder,
    byName,
    indexedByName: new Map(byName.map(indexed => [indexed.operator.exportName, indexed])),
    transitions
  };
  return cachedIndex;
}

/**
 * Operators that take `input` and give `output`, by resolved kinds (an operator matches when its
 * input list contains `input` and its output list contains `output`). With neither, every operator
 * is returned. Helpers are left out. Catalogue order. Throws for a kind that is not an ontology kind.
 *
 * Limit of the answer: kinds are only as consistent as the catalogue. Label-like results are
 * `attributes` for cluster labels, masks and colours (as OPTICS and HDBSCAN are in the catalogue)
 * and `regions` only for contiguous regionalization and segmentation, so a query by output
 * `regions` is deliberately narrow and misses cluster labels; ask for `attributes` as well.
 */
export function queryGPUAnalysisOperatorsByKind(query: {
  input?: GPUSpatialAnalysisDataKind;
  output?: GPUSpatialAnalysisDataKind;
}): readonly GPUAnalysisOperator[] {
  if (query.input !== undefined)
    validateDataKind('queryGPUAnalysisOperatorsByKind', 'input', query.input);
  if (query.output !== undefined)
    validateDataKind('queryGPUAnalysisOperatorsByKind', 'output', query.output);
  return getOperatorIndex()
    .inCatalogueOrder.filter(
      ({kinds}) =>
        (query.input === undefined || kinds.inputs.includes(query.input)) &&
        (query.output === undefined || kinds.outputs.includes(query.output))
    )
    .map(({operator}) => operator);
}

/**
 * Names of the operators that can take data of `outputKind` as an input, sorted by name: "what can
 * I do next with this result". Uses resolved kinds, so it includes supplemented operators. Throws
 * for a kind that is not an ontology kind.
 */
export function getGPUAnalysisNextOperators(
  outputKind: GPUSpatialAnalysisDataKind
): readonly string[] {
  validateDataKind('getGPUAnalysisNextOperators', 'outputKind', outputKind);
  return getOperatorIndex()
    .byName.filter(({kinds}) => kinds.inputs.includes(outputKind))
    .map(({operator}) => operator.exportName);
}

/** Result of {@link findGPUAnalysisOperatorPaths}. */
export type GPUAnalysisOperatorPathResult = {
  /** Chains of operator names, best first, at most `maximumPaths`. */
  paths: readonly (readonly string[])[];
  /** True when more chains were found (or may exist) than `paths` holds. */
  truncated: boolean;
  /**
   * Number of distinct chains (of 1 to `maximumSteps` operators) the search counted. The search
   * stops counting at {@link GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES}; a `totalFound` equal to that
   * bound is a lower bound, and then `truncated` is true even when `paths` is full.
   */
  totalFound: number;
};

/**
 * Chains of operators that turn data of `fromKind` into data of `toKind`, as lists of operator
 * names. A chain has between 1 and `maximumSteps` operators (capped at
 * {@link GPU_ANALYSIS_MAXIMUM_PATH_STEPS}), never visits a kind twice and never repeats an
 * operator. Shorter chains come first (breadth-first: all chains of one length are found before the
 * next length is searched). Within a length, chains with fewer plumbing operators (primary question
 * `prepare-data`) come first, then chains with more recipes (`add...Recipe`, which already bundle
 * several steps), then operator names, so a capped answer keeps the useful chains. `maximumPaths`
 * (default {@link GPU_ANALYSIS_MAXIMUM_PATH_COUNT}, at most
 * {@link GPU_ANALYSIS_MAXIMUM_PATH_COUNT_LIMIT}) bounds `paths`; `truncated` and `totalFound` say
 * what the cap left out. `fromKind` equal to `toKind`, and an unreachable pair, give no paths. A
 * kind that is not an ontology kind, or a `maximumSteps` or `maximumPaths` that is not an integer
 * in range, throws.
 *
 * A path is a discovery hint, not a guarantee that the chain executes. Kinds do not carry the
 * companions an operator needs (spatial weights next to a statistic, a time column, a mask), nor
 * layouts, capacities or coordinate models, so a chain can still need extra inputs or an adapter
 * before it runs. `attributes` is a hub kind: most operators take or give a column, so many
 * kind-valid chains through it (a statistic fed by an unrelated column) are meaningless; prefer
 * the chains that stay on spatial kinds.
 */
export function findGPUAnalysisOperatorPaths(
  fromKind: GPUSpatialAnalysisDataKind,
  toKind: GPUSpatialAnalysisDataKind,
  maximumSteps: number,
  maximumPaths: number = GPU_ANALYSIS_MAXIMUM_PATH_COUNT
): GPUAnalysisOperatorPathResult {
  validateDataKind('findGPUAnalysisOperatorPaths', 'fromKind', fromKind);
  validateDataKind('findGPUAnalysisOperatorPaths', 'toKind', toKind);
  if (!Number.isInteger(maximumSteps) || maximumSteps < 1) {
    throw new Error(
      `findGPUAnalysisOperatorPaths maximumSteps ${String(maximumSteps)} must be an integer from 1 to ${GPU_ANALYSIS_MAXIMUM_PATH_STEPS} (larger values are capped)`
    );
  }
  if (
    !Number.isInteger(maximumPaths) ||
    maximumPaths < 1 ||
    maximumPaths > GPU_ANALYSIS_MAXIMUM_PATH_COUNT_LIMIT
  ) {
    throw new Error(
      `findGPUAnalysisOperatorPaths maximumPaths ${String(maximumPaths)} must be an integer from 1 to ${GPU_ANALYSIS_MAXIMUM_PATH_COUNT_LIMIT}`
    );
  }
  const empty: GPUAnalysisOperatorPathResult = {paths: [], truncated: false, totalFound: 0};
  if (fromKind === toKind) return empty;
  const stepLimit = Math.min(maximumSteps, GPU_ANALYSIS_MAXIMUM_PATH_STEPS);
  const index = getOperatorIndex();
  const distances = getDistancesToKind(index, toKind);
  const fromDistance = distances.get(fromKind);
  if (fromDistance === undefined || fromDistance > stepLimit) return empty;

  // Enumerate rank buckets before names: the cutoff can only discard worse-ranked chains.
  const found = new Map<string, readonly string[]>();
  for (let length = fromDistance; length <= stepLimit; length++) {
    for (let plumbing = 0; plumbing <= length; plumbing++) {
      for (let recipes = length; recipes >= 0; recipes--) {
        collectChains(
          index,
          distances,
          [{kind: fromKind, visitedKinds: new Set([fromKind])}],
          toKind,
          length,
          plumbing,
          recipes,
          [],
          found
        );
        if (found.size >= GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES) break;
      }
      if (found.size >= GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES) break;
    }
    if (found.size >= GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES) break;
  }
  const paths = [...found.values()].slice(0, maximumPaths);
  return {
    paths,
    truncated: found.size > paths.length || found.size >= GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES,
    totalFound: found.size
  };
}

/**
 * Fewest steps from each kind to `toKind` over the kind graph, ignoring the visited-kind rule (so
 * it never overstates the distance). Kinds that cannot reach `toKind` are absent. Used to cut
 * branches that cannot finish in the steps left, which bounds the search for an unreachable pair
 * to a single pass over 15 kinds.
 */
function getDistancesToKind(
  index: OperatorIndex,
  toKind: GPUSpatialAnalysisDataKind
): Map<GPUSpatialAnalysisDataKind, number> {
  const distances = new Map<GPUSpatialAnalysisDataKind, number>([[toKind, 0]]);
  for (let round = 0; round < GPU_SPATIAL_ANALYSIS_DATA_KINDS.length; round++) {
    let changed = false;
    for (const [inputKind, byOutput] of index.transitions) {
      for (const outputKind of byOutput.keys()) {
        const next = distances.get(outputKind);
        if (next !== undefined && next + 1 < (distances.get(inputKind) ?? Infinity)) {
          distances.set(inputKind, next + 1);
          changed = true;
        }
      }
    }
    if (!changed) break;
  }
  return distances;
}

/** A kind route for the current operator prefix; several routes can name the same chain. */
type ChainState = {
  kind: GPUSpatialAnalysisDataKind;
  visitedKinds: ReadonlySet<GPUSpatialAnalysisDataKind>;
};

/**
 * Adds one rank bucket in name order. All kind routes for an operator prefix travel together,
 * so a chain with multiple possible intermediate kinds is emitted once, in its proper order.
 */
function collectChains(
  index: OperatorIndex,
  distances: ReadonlyMap<GPUSpatialAnalysisDataKind, number>,
  states: readonly ChainState[],
  toKind: GPUSpatialAnalysisDataKind,
  remainingSteps: number,
  remainingPlumbing: number,
  remainingRecipes: number,
  chain: string[],
  found: Map<string, readonly string[]>
): void {
  for (const indexed of index.byName) {
    if (found.size >= GPU_ANALYSIS_MAXIMUM_PATH_CANDIDATES) return;
    const plumbing = remainingPlumbing - Number(indexed.isPlumbing);
    const recipes = remainingRecipes - Number(indexed.isRecipe);
    const nextSteps = remainingSteps - 1;
    if (plumbing < 0 || plumbing > nextSteps || recipes < 0 || recipes > nextSteps) continue;
    const name = indexed.operator.exportName;
    if (chain.includes(name)) continue;
    const nextStates: ChainState[] = [];
    for (const state of states) {
      if (!indexed.kinds.inputs.includes(state.kind)) continue;
      for (const nextKind of indexed.kinds.outputs) {
        if (state.visitedKinds.has(nextKind)) continue;
        if (nextSteps === 0 ? nextKind !== toKind : nextKind === toKind) continue;
        const nextDistance = distances.get(nextKind);
        if (nextDistance === undefined || nextDistance > nextSteps) continue;
        nextStates.push({kind: nextKind, visitedKinds: new Set([...state.visitedKinds, nextKind])});
      }
    }
    if (nextStates.length === 0) continue;
    chain.push(name);
    if (nextSteps === 0) {
      found.set(chain.join('\u0000'), [...chain]);
    } else {
      collectChains(
        index,
        distances,
        nextStates,
        toKind,
        nextSteps,
        plumbing,
        recipes,
        chain,
        found
      );
    }
    chain.pop();
  }
}
