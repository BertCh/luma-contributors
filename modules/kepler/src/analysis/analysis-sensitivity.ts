// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {validateRequirement} from '@luma-contributors/analysis/analysis-core';
import {AnalysisPreviewSession} from './analysis-preview-session';
import {
  normalizeCatalogParameters,
  serializeCatalogInvocation,
  parseCatalogInvocation,
  matchCatalogInvocationInputs
} from './catalog-bindings';
import type {CatalogOperationBinding, CatalogRunResult} from './catalog-binding-types';
import type {CatalogFeatureCollection} from './catalog-geojson';
import type {KeplerDatasetRegistry} from '../runtime/dataset-registry';

// Six views are the AX-42 comparison budget; larger searches need different admission.
export const ANALYSIS_SENSITIVITY_COUNT = 6;
export const ANALYSIS_SENSITIVITY_CAUTION =
  'Exploratory parameter search. Looking at more settings increases the chance of a spurious finding. Correction within one map does not control error across this search.';
export type AnalysisSensitivityPopulation = {
  scope: 'full-population' | 'sample' | 'viewport';
  rowCount: number;
  sourceRowCount: number;
  /** Stable selection/filter identity, including an explicit unfiltered population. */
  selectionKey: string;
};
/** Serializable search evidence; invocation retains exact input content hashes. */
export type AnalysisSensitivityProvenance = {
  schemaVersion: 1;
  operationId: string;
  parameter: string;
  values: readonly number[];
  chosenValue?: number;
  revisionKey: string;
  population: AnalysisSensitivityPopulation;
  invocation: string;
};
export type AnalysisSensitivityFrame = {value: number; result: CatalogRunResult};

/** Qualify one binding-owned uniform, rather than relying on descriptor labels alone. */
export function getAnalysisSensitivityUnavailableReason(
  binding: CatalogOperationBinding,
  parameter: string
): string | undefined {
  if (binding.id !== 'geometry-point-buffer' || parameter !== 'radius' || !binding.planPreview)
    return 'Sensitivity is available only for point-buffer Radius, which updates a retained GPU uniform. This parameter has no qualified retained sweep executor.';
  return undefined;
}

export function parseAnalysisSensitivityProvenance(value: unknown): AnalysisSensitivityProvenance {
  const record = value as Partial<AnalysisSensitivityProvenance> | null;
  const population = record?.population;
  validateRequirement(
    'AnalysisSensitivity',
    Boolean(
      record &&
        record.schemaVersion === 1 &&
        record.operationId === 'geometry-point-buffer' &&
        record.parameter === 'radius' &&
        typeof record.revisionKey === 'string' &&
        typeof record.invocation === 'string' &&
        Array.isArray(record.values) &&
        record.values.length === ANALYSIS_SENSITIVITY_COUNT &&
        record.values.every(
          setting => typeof setting === 'number' && Number.isFinite(setting) && setting >= 0
        ) &&
        new Set(record.values).size === ANALYSIS_SENSITIVITY_COUNT &&
        (record.chosenValue === undefined || record.values.includes(record.chosenValue)) &&
        population &&
        ['full-population', 'sample', 'viewport'].includes(population.scope) &&
        Number.isSafeInteger(population.rowCount) &&
        population.rowCount >= 0 &&
        Number.isSafeInteger(population.sourceRowCount) &&
        population.sourceRowCount >= population.rowCount &&
        (population.scope !== 'full-population' ||
          population.rowCount === population.sourceRowCount) &&
        typeof population.selectionKey === 'string' &&
        population.selectionKey.length > 0
    ),
    'provenance must contain six distinct nonnegative finite radius values, a chosen value from the search, and an explicit population'
  );
  const parsed = parseCatalogInvocation(record!.invocation!);
  validateRequirement(
    'AnalysisSensitivity',
    parsed.binding.id === record!.operationId,
    'invocation operationId must match provenance'
  );
  for (const setting of record!.values!)
    normalizeCatalogParameters(parsed.binding, {...parsed.invocation.parameters, radius: setting});
  return {
    schemaVersion: 1,
    operationId: record!.operationId!,
    parameter: record!.parameter!,
    values: [...record!.values!],
    revisionKey: record!.revisionKey!,
    invocation: record!.invocation!,
    ...(record!.chosenValue === undefined ? {} : {chosenValue: record!.chosenValue}),
    population: {
      scope: population!.scope,
      rowCount: population!.rowCount,
      sourceRowCount: population!.sourceRowCount,
      selectionKey: population!.selectionKey
    }
  };
}

/** Owns one compiled graph and six result/view slots until destroy. */
export class AnalysisSensitivitySession {
  readonly provenance: AnalysisSensitivityProvenance;
  private readonly preview: AnalysisPreviewSession;
  private readonly controller = new AbortController();
  private frames?: readonly AnalysisSensitivityFrame[];
  private execution?: Promise<readonly AnalysisSensitivityFrame[] | undefined>;
  private destroyed = false;
  private readonly deviceEpoch: number;

  constructor(
    private readonly options: {
      binding: CatalogOperationBinding;
      device: Device;
      registry: KeplerDatasetRegistry;
      inputs: Readonly<Record<string, CatalogFeatureCollection>>;
      parameters: Readonly<Record<string, unknown>>;
      parameter: string;
      values: readonly number[];
      chosenValue?: number;
      revisionKey: string;
      population: AnalysisSensitivityPopulation;
      isCurrent: () => boolean;
    }
  ) {
    validateRequirement(
      'AnalysisSensitivity',
      !getAnalysisSensitivityUnavailableReason(options.binding, options.parameter),
      getAnalysisSensitivityUnavailableReason(options.binding, options.parameter) ??
        'parameter must support retained execution'
    );
    this.deviceEpoch = options.registry.deviceEpoch;
    validateRequirement(
      'AnalysisSensitivity',
      options.population.rowCount ===
        Object.values(options.inputs).reduce((total, input) => total + input.features.length, 0),
      'population.rowCount must equal the executed input feature count'
    );
    const parameters = normalizeCatalogParameters(options.binding, options.parameters);
    this.provenance = parseAnalysisSensitivityProvenance({
      schemaVersion: 1,
      operationId: options.binding.id,
      parameter: options.parameter,
      values: options.values,
      ...(options.chosenValue === undefined ? {} : {chosenValue: options.chosenValue}),
      revisionKey: options.revisionKey,
      population: options.population,
      invocation: serializeCatalogInvocation(options.binding, parameters, options.inputs)
    });
    const plans = this.provenance.values.map(value =>
      options.binding.planPreview!(
        options.inputs,
        normalizeCatalogParameters(options.binding, {...parameters, [options.parameter]: value})
      )
    );
    const plan = plans[0];
    validateRequirement(
      'AnalysisSensitivity',
      plans.every(
        candidate =>
          candidate.key === plan.key &&
          candidate.gpuByteLength === plan.gpuByteLength &&
          candidate.cpuByteLength === plan.cpuByteLength
      ),
      'all six settings must share one compile-time plan and memory shape'
    );
    this.preview = new AnalysisPreviewSession({
      ...options,
      binding: {
        ...options.binding,
        planPreview: () => ({
          ...plan,
          // A plan covers two CPU copies. Six retained frames and SVG views plus transient
          // readback/path construction are admitted as sixteen copies. GPU outputs are reused
          // serially; execute returns independently owned CPU arrays.
          cpuByteLength: plan.cpuByteLength * 8
        })
      }
    });
  }
  run(signal?: AbortSignal): Promise<readonly AnalysisSensitivityFrame[] | undefined> {
    if (this.execution) return this.execution;
    const abort = () => this.cancel();
    signal?.addEventListener('abort', abort, {once: true});
    if (signal?.aborted) abort();
    this.execution = this.execute().finally(() => signal?.removeEventListener('abort', abort));
    return this.execution;
  }
  cancel(): void {
    this.controller.abort();
    this.preview.cancel();
  }
  choose(value: number): AnalysisSensitivityProvenance {
    validateRequirement(
      'AnalysisSensitivity',
      !this.destroyed &&
        this.options.registry.deviceEpoch === this.deviceEpoch &&
        !this.controller.signal.aborted &&
        this.options.isCurrent() &&
        Boolean(this.frames?.some(frame => frame.value === value)),
      'chosenValue must come from a completed current sweep'
    );
    return parseAnalysisSensitivityProvenance({...this.provenance, chosenValue: value});
  }
  async destroy(): Promise<void> {
    this.destroyed = true;
    this.cancel();
    await this.execution?.catch(() => undefined);
    await this.preview.destroy();
    this.frames = undefined;
  }
  private async execute(): Promise<readonly AnalysisSensitivityFrame[] | undefined> {
    const frames: AnalysisSensitivityFrame[] = [];
    const {options} = this;
    const parameters = parseCatalogInvocation(this.provenance.invocation).invocation.parameters;
    for (const value of this.provenance.values) {
      const accepted = await this.preview.request({
        parameters: {...parameters, [options.parameter]: value},
        inputs: options.inputs,
        revisionKey: options.revisionKey,
        signal: this.controller.signal,
        isCurrent: () => !this.destroyed && options.isCurrent(),
        commit: result =>
          frames.push({
            value,
            result:
              this.provenance.population.scope === 'full-population'
                ? result
                : suppressSampledInference(result)
          })
      });
      if (!accepted) return undefined;
    }
    this.frames = frames;
    return frames;
  }
}

/** A partial-population view exposes no significance columns, diagnostics or summaries. */
function suppressSampledInference(result: CatalogRunResult): CatalogRunResult {
  return {
    ...result,
    diagnostics: [],
    summary: {preview: 'Exploratory geometry only'},
    tables: result.tables.flatMap(table => {
      const geometry = table.geometryColumn;
      return geometry
        ? [
            {
              ...table,
              columns: {[geometry]: table.columns[geometry]},
              columnTypes: {[geometry]: table.columnTypes[geometry]}
            }
          ]
        : [];
    })
  };
}

/** Replay the validated search and its choice, refusing changed inputs before allocating. */
export function replayAnalysisSensitivity(
  options: Omit<
    ConstructorParameters<typeof AnalysisSensitivitySession>[0],
    'parameters' | 'parameter' | 'values' | 'population'
  > & {provenance: unknown}
): AnalysisSensitivitySession {
  const provenance = parseAnalysisSensitivityProvenance(options.provenance);
  const parsed = parseCatalogInvocation(provenance.invocation);
  validateRequirement(
    'AnalysisSensitivity',
    options.revisionKey === provenance.revisionKey &&
      matchCatalogInvocationInputs(parsed.invocation, options.inputs),
    'replay inputs and revisionKey must match the recorded population'
  );
  return new AnalysisSensitivitySession({
    ...options,
    parameters: parsed.invocation.parameters,
    parameter: provenance.parameter,
    values: provenance.values,
    chosenValue: provenance.chosenValue,
    population: provenance.population
  });
}
