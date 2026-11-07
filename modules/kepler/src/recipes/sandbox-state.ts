// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {PrepareArrowTableForGPUAnalyticsOptions} from '@luma-contributors/analysis/analysis-core';
import {
  parseKeplerAnalysisRecipe,
  type KeplerAnalysisRecipe,
  type KeplerDatasetReference
} from './analysis-recipe';

export type KeplerJSONValue =
  | null
  | boolean
  | number
  | string
  | readonly KeplerJSONValue[]
  | {
      readonly [key: string]: KeplerJSONValue;
    };
export type KeplerSandboxSource = {
  dataset: KeplerDatasetReference;
  /** A host loader resolves this reference. No network requests or source rows are embedded. */
  reference: {
    kind: 'fixture' | 'url' | 'application';
    id: string;
    version: number;
    parameters: KeplerJSONValue;
  };
  schema: readonly {name: string; type: string; nullable: boolean}[];
  normalization: PrepareArrowTableForGPUAnalyticsOptions;
};
export type KeplerSandboxSelection = {
  source: KeplerDatasetReference;
  revision: number;
  /** Reproducible native filters or explicit host selection values; never a saved GPU mask. */
  value: KeplerJSONValue;
};
export type KeplerSandboxState = {
  schemaVersion: 1;
  sources: readonly KeplerSandboxSource[];
  nativeConfig: KeplerJSONValue;
  selections: readonly KeplerSandboxSelection[];
  analyses: readonly KeplerAnalysisRecipe[];
  /** Host preferences that travel with the workspace; absent when the user never changed them. */
  settings?: KeplerSandboxSettings;
};
export type KeplerSandboxSettings = {
  /** Dataset memory budget chosen by the user, in bytes. */
  memoryBudgetBytes?: number;
  /** Named host records (catalog invocations, chunked-weights policy); opaque JSON owned by the host. */
  extensions?: Record<string, KeplerJSONValue>;
};
export type KeplerSandboxRebinding = {
  source: KeplerSandboxSource;
  /** Set only after successfully replaying the saved semantic selection against this source. */
  restoredSelectionRevision?: number;
};

/** Parse a detached, versioned document. GPU objects, typed arrays, bigint and functions are rejected. */
export function parseKeplerSandboxState(value: unknown): KeplerSandboxState {
  const state = object(value, [
    'schemaVersion',
    'sources',
    'nativeConfig',
    'selections',
    'analyses',
    'settings'
  ]);
  if (state['schemaVersion'] !== 1) throw new Error('Unsupported sandbox version');
  const sources = array(state['sources']).map(readSource);
  const selections = array(state['selections']).map(value => {
    const selection = object(value, ['source', 'revision', 'value']);
    return {
      source: reference(selection['source']),
      revision: positiveInteger(selection['revision']),
      value: cloneKeplerJSON(selection['value'])
    };
  });
  const analyses = array(state['analyses']).map(parseKeplerAnalysisRecipe);
  unique(sources.map(source => source.dataset.datasetId));
  unique(selections.map(selection => selection.source.datasetId));
  unique(analyses.map(recipe => recipe.id));
  const sourceById = new Map(sources.map(source => [source.dataset.datasetId, source]));
  for (const dependent of [
    ...selections.map(selection => selection.source),
    ...analyses.map(recipe => recipe.source)
  ]) {
    if (sourceById.get(dependent.datasetId)?.dataset.revision !== dependent.revision)
      throw new Error('Sandbox contains a stale dataset reference');
  }
  for (const recipe of analyses) {
    if (
      recipe.population === 'selected' &&
      selections.find(selection => selection.source.datasetId === recipe.source.datasetId)
        ?.revision !== recipe.selectionRevision
    )
      throw new Error('Sandbox contains a stale selection reference');
  }
  const settings =
    state['settings'] === undefined
      ? undefined
      : object(state['settings'], ['memoryBudgetBytes', 'extensions']);
  return {
    schemaVersion: 1,
    sources,
    nativeConfig: cloneKeplerJSON(state['nativeConfig']),
    selections,
    analyses,
    ...(settings
      ? {
          settings: {
            ...(settings['memoryBudgetBytes'] === undefined
              ? {}
              : {memoryBudgetBytes: positiveInteger(settings['memoryBudgetBytes'])}),
            ...(settings['extensions'] === undefined
              ? {}
              : {
                  extensions: Object.fromEntries(
                    Object.entries(object(settings['extensions'])).map(([key, value]) => [
                      key,
                      cloneKeplerJSON(value)
                    ])
                  )
                })
          }
        }
      : {})
  };
}

export function serializeKeplerSandboxState(state: KeplerSandboxState): string {
  return JSON.stringify(parseKeplerSandboxState(state));
}

/** Checks current handles before replay. A new import needs the explicit rebinding operation below. */
export function restoreKeplerSandboxState(
  serialized: string,
  resolveSource: (datasetId: string) => KeplerSandboxSource | undefined,
  resolveSelectionRevision: (datasetId: string) => number | undefined
): KeplerSandboxState {
  const state = parseKeplerSandboxState(JSON.parse(serialized));
  for (const saved of state.sources) {
    const current = resolveSource(saved.dataset.datasetId);
    if (!current || current.dataset.revision !== saved.dataset.revision)
      throw new Error('Sandbox dataset revision is unavailable or stale');
    assertKeplerSandboxSourceMatches(saved, current);
  }
  for (const selection of state.selections) {
    if (resolveSelectionRevision(selection.source.datasetId) !== selection.revision)
      throw new Error('Sandbox selection revision is unavailable or stale');
  }
  return state;
}

/**
 * Explicitly rebind a replayed selection and, optionally, a reimported source. The host attests the
 * reference/schema/normalization it actually loaded; a matching ID alone never authorizes rebinding.
 * Native dataset IDs stay unchanged. Spatial weights must still be restored by their own provider.
 */
export function rebindKeplerSandboxState(
  value: KeplerSandboxState,
  bindings: readonly KeplerSandboxRebinding[],
  options: {allowReimport: boolean}
): KeplerSandboxState {
  const state = parseKeplerSandboxState(value);
  unique(bindings.map(binding => binding.source.dataset.datasetId));
  const bindingById = new Map(bindings.map(binding => [binding.source.dataset.datasetId, binding]));
  const sources = state.sources.map(saved => {
    const binding = bindingById.get(saved.dataset.datasetId);
    if (!binding) throw new Error('Missing sandbox source binding');
    const current = readSource(binding.source);
    assertKeplerSandboxSourceMatches(saved, current);
    if (!options.allowReimport && saved.dataset.revision !== current.dataset.revision)
      throw new Error('Reimport rebinding requires explicit authorization');
    return current;
  });
  const selections = state.selections.map(selection => {
    const binding = bindingById.get(selection.source.datasetId)!;
    return {
      ...selection,
      source: binding.source.dataset,
      revision: positiveInteger(binding.restoredSelectionRevision)
    };
  });
  const analyses = state.analyses.map(recipe => ({
    ...recipe,
    source: bindingById.get(recipe.source.datasetId)!.source.dataset,
    ...(recipe.population === 'selected'
      ? {
          selectionRevision: selections.find(
            selection => selection.source.datasetId === recipe.source.datasetId
          )!.revision
        }
      : {})
  }));
  return parseKeplerSandboxState({...state, sources, selections, analyses});
}

/** JSON-only deep copy. Unlike JSON.stringify, this never silently drops unsupported values. */
export function cloneKeplerJSON(value: unknown, depth = 0): KeplerJSONValue {
  if (depth > 64) throw new Error('Sandbox JSON nesting exceeds capacity');
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (Array.isArray(value)) return value.map(item => cloneKeplerJSON(item, depth + 1));
  const record = object(value);
  return Object.fromEntries(
    Object.entries(record).map(([key, item]) => [key, cloneKeplerJSON(item, depth + 1)])
  );
}

function readSource(value: unknown): KeplerSandboxSource {
  const source = object(value, ['dataset', 'reference', 'schema', 'normalization']);
  const sourceReference = object(source['reference'], ['kind', 'id', 'version', 'parameters']);
  const kind = sourceReference['kind'];
  if (kind !== 'fixture' && kind !== 'url' && kind !== 'application')
    throw new Error('Unsupported source reference');
  const schema = array(source['schema']).map(value => {
    const field = object(value, ['name', 'type', 'nullable']);
    if (typeof field['nullable'] !== 'boolean') throw new Error('Invalid source nullability');
    return {name: name(field['name']), type: name(field['type']), nullable: field['nullable']};
  });
  unique(schema.map(field => field.name));
  return {
    dataset: reference(source['dataset']),
    reference: {
      kind,
      id: name(sourceReference['id']),
      version: positiveInteger(sourceReference['version']),
      parameters: cloneKeplerJSON(sourceReference['parameters'])
    },
    schema,
    normalization: readNormalization(source['normalization'])
  };
}

function readNormalization(value: unknown): PrepareArrowTableForGPUAnalyticsOptions {
  const options = object(value, ['columns', 'float64', 'dictionaries', 'timeColumns']);
  const result: PrepareArrowTableForGPUAnalyticsOptions = {};
  if (options['columns'] !== undefined) {
    result.columns = array(options['columns']).map(name);
    unique(result.columns);
  }
  if (options['float64'] !== undefined) {
    const policy = object(options['float64']);
    if (policy['mode'] === 'exact') {
      object(policy, ['mode']);
      result.float64 = {mode: 'exact'};
    } else {
      object(policy, ['mode', 'maximumAbsoluteError', 'maximumRelativeError']);
      if (policy['mode'] !== 'round') throw new Error('Invalid normalization precision policy');
      result.float64 = {
        mode: 'round',
        maximumAbsoluteError: nonnegative(policy['maximumAbsoluteError']),
        maximumRelativeError: nonnegative(policy['maximumRelativeError'])
      };
    }
  }
  if (options['dictionaries'] !== undefined) {
    result.dictionaries = Object.fromEntries(
      Object.entries(object(options['dictionaries'])).map(([column, value]) => {
        const dictionary = object(value, ['values', 'ordered']);
        if (typeof dictionary['ordered'] !== 'boolean')
          throw new Error('Invalid dictionary ordering');
        const values = array(dictionary['values']).map(value => {
          if (typeof value !== 'string') throw new Error('Invalid dictionary label');
          return value;
        });
        unique(values);
        return [column, {values, ordered: dictionary['ordered']}];
      })
    );
  }
  if (options['timeColumns'] !== undefined) {
    result.timeColumns = Object.fromEntries(
      Object.entries(object(options['timeColumns'])).map(([column, unit]) => {
        if (
          unit !== 'second' &&
          unit !== 'millisecond' &&
          unit !== 'microsecond' &&
          unit !== 'nanosecond'
        )
          throw new Error('Invalid normalization time unit');
        return [column, unit];
      })
    );
  }
  return result;
}

export function assertKeplerSandboxSourceMatches(
  saved: KeplerSandboxSource,
  current: KeplerSandboxSource
): void {
  saved = readSource(saved);
  current = readSource(current);
  if (
    saved.dataset.datasetId !== current.dataset.datasetId ||
    canonical(saved.reference) !== canonical(current.reference) ||
    canonical(saved.schema) !== canonical(current.schema) ||
    canonical(saved.normalization) !== canonical(current.normalization)
  )
    throw new Error('Reimport source, schema or normalization does not match saved state');
}
function canonical(value: unknown): string {
  const json = cloneKeplerJSON(value);
  function sort(value: KeplerJSONValue): KeplerJSONValue {
    if (Array.isArray(value)) return value.map(sort);
    if (value && typeof value === 'object')
      return Object.fromEntries(
        Object.entries(value)
          .sort(([left], [right]) => left.localeCompare(right))
          .map(([key, item]) => [key, sort(item)])
      );
    return value;
  }
  return JSON.stringify(sort(json));
}
function object(value: unknown, keys?: readonly string[]): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype)
    throw new Error('Expected a plain sandbox object');
  const result = value as Record<string, unknown>;
  if (keys && Object.keys(result).some(key => !keys.includes(key)))
    throw new Error('Unknown sandbox property');
  return result;
}
function array(value: unknown): unknown[] {
  if (!Array.isArray(value)) throw new Error('Expected a sandbox array');
  return value;
}
function name(value: unknown): string {
  if (typeof value !== 'string' || !value.length || value.length > 4096)
    throw new Error('Invalid sandbox identifier');
  return value;
}
function positiveInteger(value: unknown): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value) || value < 1)
    throw new Error('Invalid sandbox revision');
  return value;
}
function nonnegative(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0)
    throw new Error('Invalid normalization error budget');
  return value;
}
function reference(value: unknown): KeplerDatasetReference {
  const source = object(value, ['datasetId', 'revision']);
  return {datasetId: name(source['datasetId']), revision: positiveInteger(source['revision'])};
}
function unique(values: readonly string[]): void {
  if (new Set(values).size !== values.length) throw new Error('Duplicate sandbox identifier');
}
