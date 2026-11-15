// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CatalogOperationBinding,
  CatalogParameterValue
} from '../analysis/catalog-binding-types';
import {
  KEPLER_CATALOG_BINDINGS,
  normalizeCatalogParameters,
  runCatalogOperationAsync
} from '../analysis/catalog-bindings';
import type {CatalogFeatureCollection} from '../analysis/catalog-geojson';
import {KeplerAllocationError} from '../runtime/dataset-registry';
import {getWorkflowFeatures, makeWorkflowData} from './workflow-data';
import {normalizeWorkflowParameters} from './workflow-parameters';
import {getWorkflowSourceOperations} from './workflow-source-operations';
import {getWorkflowSqlOperations} from './workflow-sql-operation';
import {getWorkflowTableOperations} from './workflow-table-operations';
import {WorkflowError} from './workflow-types';
import type {
  WorkflowInputPortSpec,
  WorkflowOperation,
  WorkflowOperationCategory,
  WorkflowOperationRegistry,
  WorkflowParameterSpec,
  WorkflowPortCompatibility,
  WorkflowPortKind
} from './workflow-types';

// Bindings are immutable catalog definitions; adapters retain their identity for the UI and planner.
const CATALOG_OPERATION_CACHE = new WeakMap<CatalogOperationBinding, WorkflowOperation>();
// The full registry is a lazy cache over the fixed source, table and catalog definitions.
let workflowOperationRegistry: WorkflowOperationRegistry | undefined;

function checkCancellation(signal: AbortSignal, title: string): void {
  if (signal.aborted) throw new WorkflowError('cancelled', `${title}: Execution was cancelled.`);
}

function formatCapacity(value: number): string {
  return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

function rethrowCatalogFailure(binding: CatalogOperationBinding, cause: unknown): never {
  if (!(cause instanceof Error)) throw cause;
  const message = cause.message;
  if (cause instanceof KeplerAllocationError && cause.reason === 'capacity') {
    const limits = binding.inputs.map(
      input =>
        `${input.label} ${formatCapacity(input.maximumFeatures)} features, ${formatCapacity(input.maximumVertices)} vertices`
    );
    throw new WorkflowError(
      'capacity-exceeded',
      `${binding.title}: ${message.replace(/[.\s]+$/, '')}. Limits: ${limits.join('; ')}.`,
      {cause}
    );
  }
  if (message === 'Input geometry kind not accepted') {
    const acceptedKinds = binding.inputs.map(
      input => `"${input.label}" accepts ${input.accepts.join(', ')}`
    );
    throw new WorkflowError(
      'geometry-kind-not-accepted',
      `${binding.title}: Input geometry kind not accepted; ${acceptedKinds.join('; ')}.`,
      {cause}
    );
  }
  if (message === 'Input required') {
    throw new WorkflowError(
      'input-required',
      `${binding.title}: Input required; ${binding.inputs.map(input => `"${input.label}"`).join(', ')} must contain at least one feature.`,
      {cause}
    );
  }
  if (message === 'GPU device required') {
    throw new WorkflowError('gpu-required', `${binding.title}: GPU device required.`, {cause});
  }
  throw cause;
}

/**
 * Workflow form of a catalog binding: the same id, ports from its input and output specs
 * (geometry `none` becomes `table`), string parameters named `*Property` or `*Properties`
 * upgraded to column pickers, and a `run` that checks kinds and capacities, converts table inputs
 * to feature collections, calls `runCatalogOperationAsync` and wraps the output tables.
 */
export function makeWorkflowOperationFromCatalogBinding(
  binding: CatalogOperationBinding
): WorkflowOperation {
  const cached = CATALOG_OPERATION_CACHE.get(binding);
  if (cached) return cached;
  // A catalog parameter without a default is derived by the binding from its input data, so the
  // workflow form leaves it out and the binding keeps deriving it.
  const settableParameters = binding.parameters.filter(
    (parameter): parameter is typeof parameter & {defaultValue: CatalogParameterValue} =>
      parameter.defaultValue !== undefined
  );
  const parameters: WorkflowParameterSpec[] = settableParameters.map(parameter => {
    if (parameter.kind !== 'string') return {...parameter};
    const kind = parameter.name.endsWith('Properties')
      ? 'column-list'
      : parameter.name.endsWith('Property')
        ? 'column'
        : 'string';
    return {
      ...parameter,
      kind,
      // Workflow column lists default to 256, but catalog strings default to 64.
      ...(kind === 'column-list' ? {maximumLength: parameter.maximumLength ?? 64} : {}),
      ...(kind !== 'string' && binding.inputs.length === 1
        ? {columnsFrom: binding.inputs[0].name}
        : {})
    };
  });
  const operation: WorkflowOperation = {
    id: binding.id,
    title: binding.title,
    summary: binding.summary,
    category: binding.domain,
    origin: 'catalog',
    version: 'catalog-1',
    engine: binding.engine,
    capabilityIds: binding.capabilityIds,
    contributorNames: [...new Set([...binding.contributorNames, ...binding.gpuContributorNames])],
    declaredBehavior: binding.declaredBehavior,
    limitation: binding.limitation,
    inputs: binding.inputs.map(input => ({
      name: input.name,
      label: input.label,
      accepts: input.accepts,
      maximumRows: input.maximumFeatures,
      maximumVertices: input.maximumVertices
    })),
    outputs: binding.outputs.map(output => ({
      name: output.tableId,
      label: output.label,
      kind:
        output.geometry === 'none'
          ? 'table'
          : output.geometry === 'mixed'
            ? 'features'
            : output.geometry,
      identityColumns: output.identityColumns,
      maximumRows: output.maximumRows
    })),
    parameters,
    normalizeParameters: raw => {
      const normalized = normalizeWorkflowParameters(parameters, raw, {
        operationTitle: binding.title
      });
      try {
        return normalizeCatalogParameters(binding, normalized);
      } catch (cause) {
        throw new WorkflowError(
          'parameter-out-of-range',
          `${binding.title}: ${cause instanceof Error ? cause.message : String(cause)}`,
          {cause}
        );
      }
    },
    run: async (context, inputs, values) => {
      checkCancellation(context.signal, binding.title);
      const collections: Record<string, CatalogFeatureCollection> = Object.create(null);
      for (const input of binding.inputs) {
        if (!Object.hasOwn(inputs, input.name)) continue;
        const data = inputs[input.name];
        if (!data) continue;
        if (
          data.table.geometryColumn === undefined ||
          !Object.hasOwn(data.table.columns, data.table.geometryColumn)
        ) {
          throw new WorkflowError(
            'geometry-required',
            `${binding.title}: "${input.label}" needs geometry; the connected output is a table without a geometry column. Join it to features with Join attributes first.`,
            {port: input.name}
          );
        }
        try {
          collections[input.name] = getWorkflowFeatures(data);
        } catch (cause) {
          if (cause instanceof WorkflowError) {
            throw new WorkflowError(
              cause.code,
              `${binding.title}: "${input.label}" ${cause.message}`,
              {...cause.details, port: input.name, cause}
            );
          }
          throw cause;
        }
      }
      try {
        const result = await runCatalogOperationAsync(binding, values, collections, {
          device: context.device ?? undefined,
          engine: context.engine
        });
        const outputs = Object.fromEntries(
          result.tables.map(table => {
            const output = binding.outputs.find(spec => spec.tableId === table.id);
            const identityColumn = output?.identityColumns.find(name => {
              if (!Object.hasOwn(table.columns, name)) return false;
              const values = table.columns[name];
              return new Set(values).size === values.length;
            });
            return [table.id, makeWorkflowData(table, {identityColumn})];
          })
        );
        checkCancellation(context.signal, binding.title);
        return {
          outputs,
          engine: result.engine,
          ...(binding.engine === 'gpu-or-cpu' &&
          result.engine === 'cpu-reference' &&
          result.fallbackReason !== undefined
            ? {fallbackReason: result.fallbackReason}
            : {}),
          dispatchedContributors: result.dispatchedContributors,
          diagnostics: result.diagnostics,
          summary: result.summary
        };
      } catch (cause) {
        rethrowCatalogFailure(binding, cause);
      }
    }
  };
  CATALOG_OPERATION_CACHE.set(binding, operation);
  return operation;
}

export function createWorkflowOperationRegistry(
  operations: readonly WorkflowOperation[]
): WorkflowOperationRegistry {
  const byId = new Map<string, WorkflowOperation>();
  for (const operation of operations) {
    if (byId.has(operation.id)) {
      throw new WorkflowError(
        'duplicate-id',
        `${operation.title}: Duplicate operation id "${operation.id}".`
      );
    }
    byId.set(operation.id, operation);
  }
  const orderedOperations = Object.freeze([...operations]);
  return {list: () => orderedOperations, get: operationId => byId.get(operationId)};
}

/** Sources, table operations and every registered catalog binding. Built on first use. */
export function getWorkflowOperationRegistry(): WorkflowOperationRegistry {
  workflowOperationRegistry ??= createWorkflowOperationRegistry([
    ...getWorkflowSourceOperations(),
    ...getWorkflowTableOperations(),
    ...getWorkflowSqlOperations(),
    ...KEPLER_CATALOG_BINDINGS.map(makeWorkflowOperationFromCatalogBinding)
  ]);
  return workflowOperationRegistry;
}

/**
 * - An input that accepts `table` reads anything (geometry travels as a GeoJSON-text column).
 * - A geometry kind feeds an input that accepts it, or that accepts `features`.
 * - `features` (kind unknown until run time) may feed any geometry input.
 * - `table` may feed a geometry input only if the rows turn out to carry geometry.
 */
export function getWorkflowPortCompatibility(
  outputKind: WorkflowPortKind,
  input: Pick<WorkflowInputPortSpec, 'accepts'>
): WorkflowPortCompatibility {
  if (input.accepts.includes('table') || input.accepts.includes(outputKind)) return 'yes';
  const acceptsGeometry = input.accepts.some(kind => kind !== 'table');
  if (!acceptsGeometry) return 'no';
  if (outputKind === 'features' || outputKind === 'table') return 'maybe';
  return input.accepts.includes('features') ? 'yes' : 'no';
}

export type WorkflowOperationMatch = {
  operation: WorkflowOperation;
  /** Higher is better; 0 when there is no query. */
  score: number;
  /** First input port that can read `compatibleWith`, when that filter is set. */
  inputName?: string;
  compatibility?: WorkflowPortCompatibility;
};

/**
 * Palette search: case-insensitive match of every query word against title, id, summary,
 * category, capability ids and contributor names (title matches rank first). With
 * `compatibleWith`, only operations with an input that can read that kind are returned.
 */
export function searchWorkflowOperations(
  registry: WorkflowOperationRegistry,
  query: string,
  options: {
    category?: WorkflowOperationCategory;
    compatibleWith?: WorkflowPortKind;
    limit?: number;
  } = {}
): WorkflowOperationMatch[] {
  const words = query.toLowerCase().trim().split(/\s+/).filter(Boolean);
  const matches: WorkflowOperationMatch[] = [];
  for (const operation of registry.list()) {
    if (options.category !== undefined && operation.category !== options.category) continue;
    const match: WorkflowOperationMatch = {operation, score: 0};
    if (options.compatibleWith !== undefined) {
      const input = operation.inputs.find(
        port => getWorkflowPortCompatibility(options.compatibleWith!, port) !== 'no'
      );
      if (!input) continue;
      match.inputName = input.name;
      match.compatibility = getWorkflowPortCompatibility(options.compatibleWith, input);
    }
    const scores = words.map(word => scoreSearchWord(operation, word));
    if (scores.some(score => score === 0)) continue;
    match.score = scores.reduce((sum, score) => sum + score, 0);
    matches.push(match);
  }
  if (words.length > 0 || options.compatibleWith !== undefined) {
    matches.sort((first, second) => {
      const scoreDifference = second.score - first.score;
      if (scoreDifference !== 0) return scoreDifference;
      const compatibilityDifference =
        Number(second.compatibility === 'yes') - Number(first.compatibility === 'yes');
      if (compatibilityDifference !== 0) return compatibilityDifference;
      return words.length === 0 ? 0 : first.operation.title.localeCompare(second.operation.title);
    });
  }
  return matches.slice(0, Math.max(0, options.limit ?? 200));
}

function scoreSearchWord(operation: WorkflowOperation, word: string): number {
  const title = operation.title.toLowerCase();
  if (title.startsWith(word)) return 100;
  for (let index = title.indexOf(word); index !== -1; index = title.indexOf(word, index + 1)) {
    if (index > 0 && /[^\p{L}\p{N}]/u.test(title[index - 1])) return 80;
  }
  if (title.includes(word)) return 60;
  if (operation.id.toLowerCase().includes(word)) return 50;
  if (operation.contributorNames.some(name => name.toLowerCase().includes(word))) return 40;
  if (operation.capabilityIds.some(name => name.toLowerCase().includes(word))) return 30;
  if (operation.category.toLowerCase().includes(word)) return 25;
  return operation.summary.toLowerCase().includes(word) ? 10 : 0;
}
