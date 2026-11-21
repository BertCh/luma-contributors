// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogOperationBinding, CatalogRunResult} from '../analysis/catalog-binding-types';
import type {KeplerOperationDescriptor} from '../descriptors/operation-descriptor-types';
import {
  KEPLER_CATALOG_BINDINGS,
  normalizeCatalogParameters,
  validateCatalogInputs,
  runCatalogOperationAsync
} from '../analysis/catalog-bindings';
import type {CatalogDataInput} from '../analysis/catalog-data-input';
import {applyKeplerDatasetProfileDeclarations} from '../profile/dataset-profile';
import {resolveKeplerOperationDescriptor} from '../descriptors/operation-descriptors';
import {
  evaluateKeplerWorkbenchInputs,
  getKeplerWorkbenchCuratedPresets,
  profileKeplerWorkbenchData
} from '../descriptors/workbench-adapters';
import {getKeplerBindingResultPresentations} from '../presentation/result-presentation';
import {
  describeKeplerCatalogResult,
  type KeplerResultReportContext
} from '../presentation/result-report';
import {KeplerAllocationError} from '../runtime/dataset-registry';
import {getWorkflowCatalogInput, getWorkflowFeatures, makeWorkflowData} from './workflow-data';
import {normalizeWorkflowParameters} from './workflow-parameters';
import {getWorkflowSourceOperations} from './workflow-source-operations';
import {getWorkflowSqlOperations} from './workflow-sql-operation';
import {getWorkflowTableOperations} from './workflow-table-operations';
import {WorkflowError} from './workflow-types';
import type {
  WorkflowInputPortSpec,
  WorkflowData,
  WorkflowOperation,
  WorkflowOperationCategory,
  WorkflowOperationRegistry,
  WorkflowParameterSpec,
  WorkflowPortCompatibility,
  WorkflowPortKind
} from './workflow-types';

/** Coordinate lineage shared by catalog cards and workflow intermediates. */
export function profileKeplerCatalogOutput(
  data: WorkflowData,
  result: Pick<CatalogRunResult, 'operationId' | 'parameters'>,
  profiles: NonNullable<KeplerResultReportContext['profiles']>,
  descriptor: KeplerOperationDescriptor
) {
  const profile = profileKeplerWorkbenchData(data);
  if (data.kind === 'table' || data.kind === 'raster') return profile;
  // An explicit producer CRS describes this output, even when the input used another frame.
  if (data.table.coordinateSystem) return profile;
  const parameterModel =
    result.parameters['coordinateSystem'] === 'planar'
      ? 'planar'
      : result.parameters['coordinateSystem'] === 'EPSG:4326'
        ? 'geographic'
        : undefined;
  const inputName = descriptor.outputs.find(output => output.tableId === data.table.id)?.kindFrom;
  const sources = inputName ? [profiles[inputName]] : Object.values(profiles);
  const inputModels = sources.map(source =>
    source?.coordinates?.source === 'declared' ? source.coordinates.model : undefined
  );
  const inheritedModel =
    inputModels.length > 0 && inputModels.every(model => model === inputModels[0])
      ? inputModels[0]
      : undefined;
  // Projection changes frames; affine transforms do not establish a new unit convention.
  const coordinateModel =
    result.operationId === 'geometry-local-frame-projection'
      ? result.parameters['direction'] === 'inverse'
        ? 'geographic'
        : 'planar'
      : result.operationId === 'affine-transform'
        ? undefined
        : (parameterModel ?? inheritedModel);
  return coordinateModel === 'planar' || coordinateModel === 'geographic'
    ? applyKeplerDatasetProfileDeclarations(profile, {coordinateModel})
    : profile;
}

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
    const limits = binding.inputs.map(input =>
      input.dataKind === 'raster'
        ? `${input.label} ${formatCapacity(input.maximumCells ?? 0)} cells`
        : input.dataKind === 'table'
          ? `${input.label} ${formatCapacity(input.maximumFeatures)} rows`
          : `${input.label} ${formatCapacity(input.maximumFeatures)} features, ${formatCapacity(input.maximumVertices)} vertices`
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
      `${binding.title}: Input required; ${binding.inputs.map(input => `"${input.label}"`).join(', ')} must contain at least one ${binding.inputs.some(input => input.dataKind === 'table') ? 'row' : 'feature'}.`,
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
 * (geometry `none` becomes `table`), and descriptor field roles become column pickers. Feature ports read the geometry column; typed table and raster ports
 * preserve their payload. `runCatalogOperationAsync` checks capacities and executes the binding.
 */
export function makeWorkflowOperationFromCatalogBinding(
  binding: CatalogOperationBinding
): WorkflowOperation {
  const cached = CATALOG_OPERATION_CACHE.get(binding);
  if (cached) return cached;
  const {descriptor} = resolveKeplerOperationDescriptor(binding);
  const fields = descriptor.inputs.flatMap(input =>
    input.fields.map(field => ({input: input.name, field}))
  );
  // Keep derived catalog parameters visible and valid as explicit overrides. An omitted
  // default stays omitted until AX or the binding can derive it from the connected input.
  const parameters: WorkflowParameterSpec[] = binding.parameters.map(parameter => {
    const role = fields.find(entry => entry.field.parameter === parameter.name);
    const policy = descriptor.parameters.find(entry => entry.name === parameter.name);
    const kind = role ? (role.field.multiple ? 'column-list' : 'column') : parameter.kind;
    return {
      ...parameter,
      kind,
      ...(role ? {columnsFrom: role.input, fieldRole: role.field} : {}),
      ...(policy ? {level: policy.level, apply: policy.apply} : {}),
      ...(kind === 'column-list' ? {maximumLength: parameter.maximumLength ?? 64} : {})
    };
  });
  const operation: WorkflowOperation = {
    analysisBinding: binding,
    analysisDescriptor: descriptor,
    get analysisPresets() {
      return getKeplerWorkbenchCuratedPresets().filter(preset => preset.bindingId === binding.id);
    },
    id: binding.id,
    title: binding.title,
    summary: binding.summary,
    category: binding.domain,
    origin: 'catalog',
    version: 'catalog-ax-3',
    engine: binding.engine,
    capabilityIds: binding.capabilityIds,
    contributorNames: [...new Set([...binding.contributorNames, ...binding.gpuContributorNames])],
    declaredBehavior: binding.declaredBehavior,
    limitation: binding.limitation,
    inputs: binding.inputs.map(input => ({
      name: input.name,
      label: input.label,
      accepts:
        input.dataKind === 'table' || input.dataKind === 'raster'
          ? [input.dataKind]
          : input.acceptsTable
            ? [...input.accepts, 'table']
            : input.accepts,
      ...(input.dataKind === 'table' ? {geometryFree: true} : {}),
      ...(input.acceptsTable ? {acceptsTable: true} : {}),
      maximumRows: input.dataKind === 'raster' ? 0 : input.maximumFeatures,
      ...(input.dataKind === 'raster' ? {maximumCells: input.maximumCells} : {}),
      maximumVertices: input.maximumVertices
    })),
    outputs: binding.outputs.map(output => ({
      name: output.tableId,
      label: output.label,
      kind:
        output.dataKind === 'raster'
          ? 'raster'
          : output.geometry === 'none'
            ? 'table'
            : output.geometry === 'mixed'
              ? 'features'
              : output.geometry,
      ...(output.geometryFromInput ? {kindFrom: output.geometryFromInput} : {}),
      identityColumns: output.identityColumns,
      maximumRows: output.maximumRows
    })),
    parameters,
    normalizeParameters: raw => {
      const normalized = normalizeWorkflowParameters(parameters, raw, {
        operationTitle: binding.title,
        operationId: binding.id
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
      const collections: Record<string, CatalogDataInput> = Object.create(null);
      for (const input of binding.inputs) {
        if (!Object.hasOwn(inputs, input.name)) continue;
        const data = inputs[input.name];
        if (!data) continue;
        if (input.acceptsTable && data.kind === 'table') {
          collections[input.name] = getWorkflowCatalogInput(data, input.maximumFeatures);
          continue;
        }
        if (input.dataKind === 'table' || input.dataKind === 'raster') {
          if (data.kind !== input.dataKind)
            throw new WorkflowError(
              'geometry-kind-not-accepted',
              `${binding.title}: "${input.label}" requires ${input.dataKind}; received ${data.kind}.`,
              {port: input.name}
            );
          collections[input.name] = getWorkflowCatalogInput(data, input.maximumFeatures);
          continue;
        }
        if (data.kind === 'raster')
          throw new WorkflowError(
            'geometry-required',
            `${binding.title}: "${input.label}" needs geometry; raster cells cannot be read as features.`,
            {port: input.name}
          );
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
        validateCatalogInputs(binding, collections);
      } catch (cause) {
        rethrowCatalogFailure(binding, cause);
      }
      if (binding.engine === 'gpu' && (!context.device || context.engine === 'cpu'))
        rethrowCatalogFailure(binding, new Error('GPU device required'));
      const profiles = Object.fromEntries(
        Object.entries(inputs).map(([name, data]) => [
          name,
          data.analysisProfile ?? profileKeplerWorkbenchData(data)
        ])
      );
      const analysis = evaluateKeplerWorkbenchInputs(binding, profiles, {
        overrides: context.parameterOverrides ?? values,
        executionEngine: context.engine === 'cpu' || !context.device ? 'cpu' : 'gpu'
      });
      if (analysis.status !== 'ready') {
        throw new WorkflowError(
          'analysis-requirement',
          `${binding.title}: ${analysis.missing
            .map(missing => `${missing.what}. ${missing.remedy}`)
            .join('; ')}`
        );
      }
      try {
        const result = await runCatalogOperationAsync(binding, analysis.parameters, collections, {
          signal: context.signal,
          device: context.device ?? undefined,
          engine: context.engine
        });
        const presentations = getKeplerBindingResultPresentations(result, descriptor);
        const preset = context.curatedAnalysisId
          ? operation.analysisPresets?.find(
              candidate => candidate.curatedAnalysisId === context.curatedAnalysisId
            )
          : undefined;
        const reportContext: KeplerResultReportContext = structuredClone({
          profiles,
          presetParameters: preset
            ? Object.keys(preset.executionParameters).filter(
                name =>
                  JSON.stringify(result.parameters[name]) ===
                  JSON.stringify(preset.executionParameters[name])
              )
            : undefined,
          resolutions: analysis.resolutions,
          fieldBindings: analysis.fieldBindings,
          warnings: analysis.warnings,
          guessedCoordinates: analysis.guessedFacts.some(fact => fact.kind === 'coordinate-model')
        });
        const report = describeKeplerCatalogResult(result, binding, descriptor, reportContext);
        if (preset) report.notes = [...report.notes, `Curated analysis: ${preset.title}`];
        const outputs = Object.fromEntries(
          result.tables.map(table => {
            const output = binding.outputs.find(spec => spec.tableId === table.id);
            const identityColumn = output?.identityColumns.find(name => {
              if (!Object.hasOwn(table.columns, name)) return false;
              const values = table.columns[name];
              return new Set(values).size === values.length;
            });
            const data = makeWorkflowData(table, {identityColumn});
            if (data.kind !== 'table' && data.kind !== 'raster') {
              data.analysisProfile = profileKeplerCatalogOutput(data, result, profiles, descriptor);
            }
            return [
              table.id,
              {
                ...data,
                analysisPresentation: presentations.find(
                  presentation => presentation.tableId === table.id
                ),
                analysisReport: report,
                analysisRun: {result, context: reportContext}
              }
            ];
          })
        );
        checkCancellation(context.signal, binding.title);
        return {
          outputs,
          ...(preset ? {curatedAnalysisId: preset.curatedAnalysisId} : {}),
          resolvedParameters: result.parameters,
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
        checkCancellation(context.signal, binding.title);
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
  input: Pick<WorkflowInputPortSpec, 'accepts' | 'geometryFree' | 'acceptsTable'>
): WorkflowPortCompatibility {
  if (input.geometryFree && outputKind !== 'table')
    return outputKind === 'features' ? 'maybe' : 'no';
  if (input.accepts.includes(outputKind)) return 'yes';
  if (outputKind === 'raster') return 'no';
  // Dataset sources historically declare features before their actual representation is read.
  if (outputKind === 'features' && input.accepts.includes('raster')) return 'maybe';
  if (input.acceptsTable) return outputKind === 'features' ? 'maybe' : 'no';
  if (input.accepts.includes('table')) return 'yes';
  const acceptsGeometry = input.accepts.some(kind => kind !== 'table' && kind !== 'raster');
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
