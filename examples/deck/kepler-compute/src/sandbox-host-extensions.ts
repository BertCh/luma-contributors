// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerJSONValue} from '@luma-contributors/kepler';
import {
  parseCatalogInvocationRecords,
  restoreCatalogInvocations,
  serializeCatalogInvocationRecords,
  clearCatalogInvocations
} from './catalog-invocation-store';
import type {AnalysisStreamedWeightsOptions} from './analysis5-streamed-weights';
import {
  SANDBOX_EXTENSION_WORKFLOWS,
  applyWorkbenchExtension,
  collectWorkbenchExtension,
  validateWorkbenchExtension
} from './workflow/workbench-extension';

/** Workspace members saved under `settings.extensions` of the sandbox state. */
export type StreamedWeightsExtension = {
  schemaVersion: 1;
  /** Chunked point-weights policy per dataset id (the Analysis panel's debug overrides). */
  datasets: Record<string, AnalysisStreamedWeightsOptions>;
};

export const SANDBOX_EXTENSION_CATALOG = 'catalogInvocations';
export const SANDBOX_EXTENSION_STREAMED_WEIGHTS = 'streamedPointWeights';

type AnalysisPolicySource = {
  datasetId: string;
  getStreamedWeightsOptions(): AnalysisStreamedWeightsOptions;
};

/** Collects the extension members worth saving; undefined when every one is at its default. */
export function collectSandboxExtensions(
  analyses: readonly AnalysisPolicySource[]
): Record<string, KeplerJSONValue> | undefined {
  const extensions: Record<string, KeplerJSONValue> = {};
  const catalog = serializeCatalogInvocationRecords();
  if (catalog) extensions[SANDBOX_EXTENSION_CATALOG] = catalog;
  const workflows = collectWorkbenchExtension();
  if (workflows) extensions[SANDBOX_EXTENSION_WORKFLOWS] = workflows;
  const datasets: Record<string, KeplerJSONValue> = {};
  for (const source of analyses) {
    const options = source.getStreamedWeightsOptions();
    const entry: Record<string, KeplerJSONValue> = {};
    if (options.forceStreamed) entry['forceStreamed'] = true;
    if (options.maximumSlotsPerChunk !== undefined)
      entry['maximumSlotsPerChunk'] = options.maximumSlotsPerChunk;
    if (Object.keys(entry).length) datasets[source.datasetId] = entry;
  }
  if (Object.keys(datasets).length)
    extensions[SANDBOX_EXTENSION_STREAMED_WEIGHTS] = {schemaVersion: 1, datasets};
  return Object.keys(extensions).length ? extensions : undefined;
}

export function parseStreamedWeightsExtension(value: unknown): StreamedWeightsExtension {
  if (
    !value ||
    typeof value !== 'object' ||
    (value as {schemaVersion?: unknown}).schemaVersion !== 1
  )
    throw new Error('Unsupported streamed weights member');
  const rawDatasets = (value as {datasets?: unknown}).datasets;
  if (!rawDatasets || typeof rawDatasets !== 'object' || Array.isArray(rawDatasets))
    throw new Error('Malformed streamed weights member');
  const datasets: Record<string, AnalysisStreamedWeightsOptions> = {};
  for (const [datasetId, raw] of Object.entries(rawDatasets)) {
    const entry = raw as {forceStreamed?: unknown; maximumSlotsPerChunk?: unknown};
    const options: AnalysisStreamedWeightsOptions = {};
    if (entry.forceStreamed !== undefined) {
      if (entry.forceStreamed !== true) throw new Error('Malformed streamed weights member');
      options.forceStreamed = true;
    }
    if (entry.maximumSlotsPerChunk !== undefined) {
      const slots = entry.maximumSlotsPerChunk;
      if (typeof slots !== 'number' || !Number.isSafeInteger(slots) || slots < 1)
        throw new Error('Malformed streamed weights member');
      options.maximumSlotsPerChunk = slots;
    }
    datasets[datasetId] = options;
  }
  return {schemaVersion: 1, datasets};
}

/** Validates every member before replay changes anything. Unknown members are ignored (forward compatible). */
export function validateSandboxExtensions(extensions: Record<string, KeplerJSONValue> | undefined) {
  const catalog = extensions?.[SANDBOX_EXTENSION_CATALOG];
  const streamed = extensions?.[SANDBOX_EXTENSION_STREAMED_WEIGHTS];
  return {
    catalog: catalog === undefined ? [] : parseCatalogInvocationRecords(catalog),
    streamedWeights: streamed === undefined ? undefined : parseStreamedWeightsExtension(streamed),
    workflows: validateWorkbenchExtension(extensions?.[SANDBOX_EXTENSION_WORKFLOWS])
  };
}

/** Applies validated members after the workspace is rebuilt. */
export function applySandboxExtensions(
  extensions: Record<string, KeplerJSONValue> | undefined,
  setStreamedOptions: (datasetId: string, options: AnalysisStreamedWeightsOptions) => void
): void {
  const validated = validateSandboxExtensions(extensions);
  if (validated.catalog.length) restoreCatalogInvocations(validated.catalog);
  else clearCatalogInvocations();
  applyWorkbenchExtension(validated.workflows);
  for (const [datasetId, options] of Object.entries(validated.streamedWeights?.datasets ?? {}))
    setStreamedOptions(datasetId, options);
}
