// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {validateTabularPointPositions} from './point-position-source';
import {
  getSampleMapProvenance,
  parseSampleMapProvenance,
  type SampleMapProvenance
} from './sample-map-provenance';
import {
  assertKeplerSandboxSourceMatches,
  cloneKeplerJSON,
  materializeKeplerSelection,
  parseKeplerSandboxState,
  rebindKeplerSandboxState,
  serializeKeplerSandboxState,
  type KeplerJSONValue,
  type KeplerAnalysisRecipe,
  type KeplerSandboxSource,
  type KeplerSandboxState,
  type KeplerSessionExplorer,
  type KeplerSelectionExport
} from '@luma-contributors/kepler';
import {
  prepareArrowTableForGPUAnalytics,
  type PrepareArrowTableForGPUAnalyticsOptions
} from '@luma-contributors/analysis/analysis-core';
import {
  preparePolygonContiguityTopology,
  POLYGON_CONTIGUITY_PREPARATION_BYTES
} from './geometry-preparation';
import type {ComputeRuntime} from './compute-runtime';
import type {PolygonAnalysisSource} from './analysis-polygon-weights-cache';
import {assertSandboxSourcesCovered} from './sandbox-source-validation';
import {assertCompleteSandboxPopulations} from './sandbox-population';
import {parseNativeCellIndex, type NativeCellIndex} from './cell-layer-types';
import {
  getStreamProvenance,
  parseStreamProvenance,
  serializeStreamProvenance,
  type NativeStreamProvenance
} from './remote-stream-source';
import type {RestoredSandboxBundle, SandboxBundleSource} from './sandbox-bundle';
import {
  preparePublishedGeometry,
  preparePublishedTrips,
  preparePublishedH3,
  type CoordinateRoles
} from './source-publication';
import {readSandboxTripOptions, type SandboxTripOptions} from './trip-source-options';
import type {FlowRoles} from './flow-source';
import {
  parseResidentGroupByProvenance,
  type ResidentGroupByProvenance
} from './resident-group-by-plan';
import {parseResidentJoinLineage, type ResidentJoinLineage} from './resident-join-lineage';
import {
  parseResidentSuitabilityLineage,
  type ResidentSuitabilityLineage
} from './resident-suitability-lineage';
import {getAnalysisRecipeColumns, validateAnalysisRecipeCapacity} from './analysis-recipe-settings';
import * as arrow from 'apache-arrow';
import {assertNotResidentSource} from './resident-source-guards';
import {applySandboxHostSettings, getSandboxHostSettings} from './sandbox-host-settings';
import {
  applySandboxExtensions,
  collectSandboxExtensions,
  validateSandboxExtensions
} from './sandbox-host-extensions';
import {
  estimateResidentMaterializedBytes,
  materializeResidentRows
} from './resident-materialization';

export type SandboxCallbacks = {
  runtime: ComputeRuntime;
  getNativeConfig(): unknown;
  applyNativeConfig(config: KeplerJSONValue): void | Promise<void>;
  /** Explicitly regenerate/upload the referenced fixture, retaining its dataset ID. */
  reimportSource?(source: KeplerSandboxSource): Promise<ComputeRuntime>;
  /** Rebuild/register bundle sources; saved configuration and analysis are applied by this module. */
  restoreBundle?(bundle: RestoredSandboxBundle): Promise<ComputeRuntime>;
  /** Saved sources the host recomputes from lineage, so the bundle carries no CPU table for them. */
  derivedDatasetIds?: ReadonlySet<string>;
};

export function getSandboxSource(runtime: ComputeRuntime): KeplerSandboxSource {
  const dataset = runtime.explorer.dataset;
  return {
    dataset: {datasetId: dataset.id, revision: dataset.revision},
    reference: {
      kind: 'fixture',
      id: 'luma-kepler-observations',
      version: 1,
      parameters: {seed: 42, rowCount: dataset.rowCount}
    },
    schema: dataset.value.source.schema.fields.map(field => ({
      name: field.name,
      type: field.type.toString(),
      nullable: field.nullable
    })),
    normalization: (dataset.source?.metadata?.['normalization'] ??
      {}) as PrepareArrowTableForGPUAnalyticsOptions
  };
}

function getStreamParameter(dataset: {
  population?: Parameters<typeof getStreamProvenance>[0];
  source?: {metadata?: Parameters<typeof getStreamProvenance>[1]};
}): {stream?: KeplerJSONValue} {
  const stream = getStreamProvenance(dataset.population, dataset.source?.metadata);
  return stream ? {stream: serializeStreamProvenance(stream)} : {};
}

export function getSandboxSources(runtime: ComputeRuntime): KeplerSandboxSource[] {
  return [
    getSandboxSource(runtime),
    ...Array.from(runtime.publications?.values() ?? [], published => ({
      dataset: {datasetId: published.dataset.id, revision: published.dataset.revision},
      reference: {
        kind: 'application' as const,
        id: 'luma-kepler-published-source',
        version: 1,
        parameters: {
          label: published.label,
          ...(getSampleMapProvenance(published.dataset.source?.metadata)
            ? {
                sampleMap: cloneKeplerJSON(
                  getSampleMapProvenance(published.dataset.source?.metadata)
                )
              }
            : {}),
          coordinates: published.coordinates ? {...published.coordinates} : null,
          geometryColumn: published.geometryColumn ?? null,
          ...(published.h3Column ? {h3Column: published.h3Column} : {}),
          ...(published.groupBy ? {groupBy: cloneKeplerJSON(published.groupBy)} : {}),
          ...(published.join ? {join: cloneKeplerJSON(published.join)} : {}),
          ...(published.suitability ? {suitability: cloneKeplerJSON(published.suitability)} : {}),
          // Remote origin and explicit partial population; absent for finite complete sources.
          ...getStreamParameter(published.dataset),
          ...(published.cellIndex ? {cellIndex: {...published.cellIndex}} : {}),
          ...(published.tripOptions
            ? {tripOptions: readSandboxTripOptions(published.tripOptions)}
            : {})
        }
      },
      schema: published.dataset.value.source.schema.fields.map(field => ({
        name: field.name,
        type: field.type.toString(),
        nullable: field.nullable
      })),
      normalization: (published.dataset.source?.metadata?.['normalization'] ??
        {}) as PrepareArrowTableForGPUAnalyticsOptions
    }))
  ];
}

/**
 * Bundle sources with resident (over-cap) results streamed from GPU pages into their accounted lazy
 * Arrow view first, bounded by the bundle row and byte limits. Synchronous recovery capture keeps
 * using getSandboxBundleSources, which refuses resident sources it cannot page.
 */
export async function materializeSandboxBundleSources(
  runtime: ComputeRuntime,
  limits: {maximumRows: number; maximumSourceBytes: number}
): Promise<SandboxBundleSource[]> {
  const explorers = [
    runtime.explorer,
    ...Array.from(runtime.publications?.values() ?? [], published => published.consumer.explorer)
  ];
  for (const explorer of explorers) {
    const value = explorer.dataset.value;
    if (!value.resident) continue;
    // Bundles embed whole-source Arrow, so resident sources must fit the same row and byte budgets.
    if (
      value.source.numRows > limits.maximumRows ||
      estimateResidentMaterializedBytes(value) > limits.maximumSourceBytes
    )
      throw new Error('Resident source exceeds the portable bundle row or byte budget');
    await materializeResidentRows(value, undefined, {
      assertCurrent: () => {
        if (!explorer.isCurrent) throw new Error('Source changed during bundle export');
      }
    });
  }
  return explorers.map(explorer => ({
    datasetId: explorer.dataset.id,
    revision: explorer.dataset.revision,
    table: explorer.dataset.value.source,
    isCurrent: () => explorer.isCurrent
  }));
}

export function getSandboxBundleSources(
  runtime: ComputeRuntime,
  options: {omitResident?: boolean} = {}
): SandboxBundleSource[] {
  return [
    runtime.explorer,
    ...Array.from(runtime.publications?.values() ?? [], published => published.consumer.explorer)
  ]
    .filter(explorer => !(options.omitResident && explorer.dataset.value.resident))
    .map(explorer => {
      assertNotResidentSource(explorer.dataset.value, 'bundle export');
      return {
        datasetId: explorer.dataset.id,
        revision: explorer.dataset.revision,
        table: explorer.dataset.value.source,
        isCurrent: () => explorer.isCurrent
      };
    });
}

export function readPublishedSandboxSource(source: KeplerSandboxSource): {
  label: string;
  coordinates?: CoordinateRoles;
  geometryColumn?: string;
  tripOptions?: SandboxTripOptions;
  h3Column?: string;
  groupBy?: ResidentGroupByProvenance;
  join?: ResidentJoinLineage;
  suitability?: ResidentSuitabilityLineage;
  stream?: NativeStreamProvenance;
  cellIndex?: NativeCellIndex;
  sampleMap?: SampleMapProvenance;
} {
  const parameters = source.reference.parameters;
  if (
    source.reference.kind !== 'application' ||
    source.reference.id !== 'luma-kepler-published-source' ||
    source.reference.version !== 1 ||
    !parameters ||
    typeof parameters !== 'object' ||
    Array.isArray(parameters) ||
    !('label' in parameters) ||
    typeof parameters['label'] !== 'string'
  )
    throw new Error('Unsupported published source descriptor');
  const coordinates = 'coordinates' in parameters ? parameters['coordinates'] : null;
  const geometryColumn = 'geometryColumn' in parameters ? parameters['geometryColumn'] : null;
  if (
    geometryColumn !== null &&
    (typeof geometryColumn !== 'string' ||
      !source.schema.some(field => field.name === geometryColumn))
  )
    throw new Error('Invalid saved geometry field');
  const h3Column = 'h3Column' in parameters ? parameters['h3Column'] : null;
  if (
    h3Column !== null &&
    (typeof h3Column !== 'string' || !source.schema.some(field => field.name === h3Column))
  )
    throw new Error('Invalid saved H3 field');
  const savedTripOptions = 'tripOptions' in parameters ? parameters['tripOptions'] : null;
  const tripOptions =
    savedTripOptions === null ? undefined : readSandboxTripOptions(savedTripOptions);
  if (tripOptions && (!geometryColumn || coordinates !== null))
    throw new Error('Timed paths require a geometry field without point coordinate roles');
  if (
    tripOptions?.grouping &&
    Object.values(tripOptions.grouping).some(
      column => !source.schema.some(field => field.name === column)
    )
  )
    throw new Error('Invalid saved grouped trip fields');
  if (h3Column !== null && (coordinates !== null || geometryColumn !== null || tripOptions))
    throw new Error('H3 roles cannot be combined with point coordinates, geometry or Trips');
  let roles: CoordinateRoles | undefined;
  if (coordinates !== null) {
    if (
      !coordinates ||
      typeof coordinates !== 'object' ||
      Array.isArray(coordinates) ||
      !('longitude' in coordinates) ||
      !('latitude' in coordinates)
    )
      throw new Error('Invalid saved coordinate roles');
    const longitude = coordinates['longitude'];
    const latitude = coordinates['latitude'];
    const altitude = 'altitude' in coordinates ? coordinates['altitude'] : undefined;
    if (
      typeof longitude !== 'string' ||
      typeof latitude !== 'string' ||
      !source.schema.some(field => field.name === longitude) ||
      !source.schema.some(field => field.name === latitude) ||
      (altitude !== undefined &&
        (typeof altitude !== 'string' || !source.schema.some(field => field.name === altitude)))
    )
      throw new Error('Invalid saved coordinate roles');
    const flow =
      'flow' in coordinates ? readFlowRoles(coordinates['flow'], source.schema) : undefined;
    if (flow && (altitude !== undefined || geometryColumn !== null || tripOptions))
      throw new Error('Flow roles cannot be combined with altitude, geometry or Trips');
    roles = {
      longitude,
      latitude,
      ...(typeof altitude === 'string' ? {altitude} : {}),
      ...(flow ? {flow} : {})
    };
  }
  const stream = parseStreamProvenance('stream' in parameters ? parameters['stream'] : undefined);
  // The cell column stays in the retained source; its footprint column is the geometry role.
  const cellIndex = parseNativeCellIndex(
    'cellIndex' in parameters ? parameters['cellIndex'] : undefined,
    source.schema.map(field => field.name)
  );
  if (cellIndex && (geometryColumn === null || coordinates !== null || h3Column !== null))
    throw new Error('A saved cell index requires a footprint geometry role only');
  const groupBy =
    'groupBy' in parameters ? parseResidentGroupByProvenance(parameters['groupBy']) : undefined;
  if (
    groupBy &&
    (groupBy.resultId !== source.dataset.datasetId ||
      roles ||
      geometryColumn !== null ||
      tripOptions ||
      h3Column !== null ||
      groupBy.fields.length !== source.schema.length ||
      groupBy.fields.some((field, index) => field.name !== source.schema[index]?.name))
  )
    throw new Error('Saved Group By requires its matching table-only result schema');
  const join = 'join' in parameters ? parseResidentJoinLineage(parameters['join']) : undefined;
  if (
    join &&
    (groupBy ||
      join.resultId !== source.dataset.datasetId ||
      roles ||
      geometryColumn !== null ||
      tripOptions ||
      h3Column !== null ||
      join.fields.length !== source.schema.length ||
      join.fields.some((field, index) => field.name !== source.schema[index]?.name))
  )
    throw new Error('Saved Join requires its matching table-only result schema');
  const suitability =
    'suitability' in parameters
      ? parseResidentSuitabilityLineage(parameters['suitability'])
      : undefined;
  if (
    suitability &&
    (groupBy ||
      join ||
      suitability.resultId !== source.dataset.datasetId ||
      roles ||
      geometryColumn !== null ||
      tripOptions ||
      h3Column !== null ||
      suitability.fields.length !== source.schema.length ||
      suitability.fields.some((field, index) => field.name !== source.schema[index]?.name))
  )
    throw new Error('Saved Suitability requires its matching table-only result schema');
  return {
    label: parameters['label'],
    ...(parameters['sampleMap'] === undefined
      ? {}
      : {sampleMap: parseSampleMapProvenance(parameters['sampleMap'])}),
    ...(groupBy ? {groupBy} : {}),
    ...(join ? {join} : {}),
    ...(suitability ? {suitability} : {}),
    ...(roles ? {coordinates: roles} : {}),
    ...(geometryColumn ? {geometryColumn} : {}),
    ...(tripOptions ? {tripOptions} : {}),
    ...(h3Column ? {h3Column} : {}),
    ...(stream ? {stream} : {}),
    ...(cellIndex ? {cellIndex} : {})
  };
}

export function saveSandbox(
  callbacks: SandboxCallbacks,
  /** Native-only datasets whose originals the caller embeds in a portable bundle. */
  nativeOnlyDatasetIds: readonly string[] = []
): string {
  const runtime = callbacks.runtime;
  const explorers = [
    runtime.explorer,
    ...Array.from(runtime.publications?.values() ?? [], published => published.consumer.explorer)
  ];
  // An appended stream can replace the current registry revision before its preview rebinds.
  // Defer that checkpoint explicitly rather than clearing the last complete recovery state.
  assertCompleteSandboxPopulations(
    explorers.map(
      explorer => runtime.session.registry.getDataset(explorer.dataset.id) ?? explorer.dataset
    )
  );
  if (explorers.some(explorer => !explorer.isCurrent))
    throw new Error('Cannot save a stale resident dataset');
  const nativeConfig = cloneNativeConfig(callbacks.getNativeConfig());
  const sources = getSandboxSources(runtime);
  assertSandboxSourcesCovered(nativeConfig, [
    ...sources.map(source => source.dataset.datasetId),
    ...nativeOnlyDatasetIds
  ]);
  const hostSettings = getSandboxHostSettings();
  // Catalog invocations and chunked-weights policy travel as opaque extension members.
  const extensions = collectSandboxExtensions(
    [
      {datasetId: runtime.explorer.dataset.id, analysis: runtime.analysis},
      ...Array.from(runtime.publications?.values() ?? [], published => ({
        datasetId: published.dataset.id,
        analysis: published.analysis
      }))
    ].flatMap(({datasetId, analysis}) =>
      analysis
        ? [
            {
              datasetId,
              getStreamedWeightsOptions: () => analysis.getStreamedWeightsOptions?.() ?? {}
            }
          ]
        : []
    )
  );
  const settings = {...hostSettings, ...(extensions ? {extensions} : {})};
  return serializeKeplerSandboxState({
    schemaVersion: 1,
    ...(Object.keys(settings).length ? {settings} : {}),
    sources,
    nativeConfig,
    selections: explorers.map(explorer => ({
      source: {datasetId: explorer.dataset.id, revision: explorer.dataset.revision},
      revision: explorer.currentRevision + 1,
      value: {kind: 'native-kepler-config', filters: getNativeFilters(nativeConfig)}
    })),
    analyses: [
      runtime.analysis,
      ...Array.from(runtime.publications?.values() ?? [], published => published.analysis)
    ]
      .filter(analysis => analysis && analysis.settings.mode !== 'off')
      .map(analysis => JSON.parse(analysis!.serializeRecipe()))
  });
}

/** Validates the entire saved fixture before the host regenerates any data. */
export function readSandboxFixture(source: KeplerSandboxSource): {seed: 42; rowCount: number} {
  const parameters = source.reference.parameters;
  if (
    source.reference.kind !== 'fixture' ||
    source.reference.id !== 'luma-kepler-observations' ||
    source.reference.version !== 1 ||
    !parameters ||
    typeof parameters !== 'object' ||
    Array.isArray(parameters)
  )
    throw new Error('This sandbox loader supports the deterministic observations fixture only');
  const seed = 'seed' in parameters ? parameters['seed'] : undefined;
  const rowCount = 'rowCount' in parameters ? parameters['rowCount'] : undefined;
  if (
    seed !== 42 ||
    typeof rowCount !== 'number' ||
    !Number.isSafeInteger(rowCount) ||
    rowCount < 1 ||
    rowCount > 1_000_000
  )
    throw new Error('Unsupported fixture seed or row count');
  return {seed, rowCount};
}

export async function replaySandbox(
  serialized: string,
  callbacks: SandboxCallbacks,
  reimport = false
): Promise<ComputeRuntime> {
  const saved = parseKeplerSandboxState(JSON.parse(serialized));
  assertSandboxReplay(saved, callbacks.runtime);
  if (saved.settings) applySandboxHostSettings(saved.settings);
  let runtime = callbacks.runtime;
  if (reimport) {
    if (saved.sources.length !== 1)
      throw new Error('Uploaded sources require a portable bundle to rebuild the workspace');
    if (!callbacks.reimportSource)
      throw new Error('No explicit fixture reimport callback is installed');
    const source = saved.sources[0];
    assertKeplerSandboxSourceMatches(source, {
      ...getSandboxSource(runtime),
      reference: source.reference
    });
    if (saved.analyses[0]) {
      if (getPolygonGeometryColumn(saved.analyses[0]))
        throw new Error('Polygon analysis requires a published geometry source');
      for (const name of getAnalysisNumericColumns(saved.analyses[0])) {
        const column = runtime.explorer.dataset.value.columns[name];
        if (!column || column.format !== 'float32')
          throw new Error('Saved analysis requires prepared Float32 columns');
      }
    }
    runtime = await callbacks.reimportSource(source);
  }
  return applySandboxState(saved, {...callbacks, runtime}, reimport);
}

/** Parse and preview the complete bundle before calling this explicit workspace replacement. */
export async function replaySandboxBundle(
  bundle: RestoredSandboxBundle,
  callbacks: SandboxCallbacks
): Promise<ComputeRuntime> {
  const saved = parseKeplerSandboxState(bundle.state);
  validateSandboxBundle(bundle, callbacks.runtime, callbacks.derivedDatasetIds);
  if (!callbacks.restoreBundle) throw new Error('Portable bundle restoration is unavailable');
  const fixture = saved.sources.find(
    source => source.dataset.datasetId === callbacks.runtime.explorer.dataset.id
  )!;
  assertKeplerSandboxSourceMatches(fixture, {
    ...getSandboxSource(callbacks.runtime),
    reference: fixture.reference
  });
  const runtime = await callbacks.restoreBundle(bundle);
  return applySandboxState(saved, {...callbacks, runtime}, true);
}

/** Complete CPU preflight; never allocates a GPU resource or changes the running workspace. */
export function validateSandboxBundle(
  bundle: RestoredSandboxBundle,
  runtime: ComputeRuntime,
  derivedDatasetIds: ReadonlySet<string> = new Set()
): void {
  const saved = parseKeplerSandboxState(bundle.state);
  assertSandboxReplay(
    saved,
    runtime,
    bundle.nativeOnlySources?.map(source => source.datasetId)
  );
  if (
    bundle.sources.length !== saved.sources.length - derivedDatasetIds.size ||
    new Set(bundle.sources.map(source => source.datasetId)).size !== bundle.sources.length
  )
    throw new Error('Bundle must include each declared source exactly once');
  for (const descriptor of saved.sources) {
    // Derived sources are recomputed from lineage on the recovered inputs; no table is bundled.
    if (derivedDatasetIds.has(descriptor.dataset.datasetId)) continue;
    const source = bundle.sources.find(item => item.datasetId === descriptor.dataset.datasetId);
    if (!source || source.revision !== descriptor.dataset.revision)
      throw new Error('Bundle contains a missing or stale source');
    assertKeplerSandboxSourceMatches(descriptor, {
      ...descriptor,
      schema: source.table.schema.fields.map(field => ({
        name: field.name,
        type: field.type.toString(),
        nullable: field.nullable
      }))
    });
    const roles =
      descriptor.dataset.datasetId === runtime.explorer.dataset.id
        ? undefined
        : readPublishedSandboxSource(descriptor);
    const scalarSource =
      roles?.geometryColumn || roles?.h3Column
        ? source.table.select(
            source.table.schema.fields
              .map(field => field.name)
              .filter(name => name !== roles.geometryColumn && name !== roles.h3Column)
          )
        : source.table;
    const prepared = prepareArrowTableForGPUAnalytics(scalarSource, descriptor.normalization);
    const recipe = saved.analyses.find(
      item => item.source.datasetId === descriptor.dataset.datasetId
    );
    if (recipe) {
      validateAnalysisRecipeCapacity(recipe, source.table.numRows);
      const geometryColumn = getPolygonGeometryColumn(recipe);
      if (geometryColumn && (roles?.geometryColumn !== geometryColumn || roles.tripOptions))
        throw new Error('Saved polygon analysis requires its published polygon geometry role');
      for (const name of getAnalysisNumericColumns(recipe)) {
        const column = prepared.table.getChild(name);
        if (
          !column ||
          !column.length ||
          prepared.dictionaries[name] ||
          !arrow.DataType.isFloat(column.type) ||
          column.type.precision !== arrow.Precision.SINGLE
        )
          throw new Error('Saved analysis requires nonempty prepared Float32 columns');
      }
    }
    if (!prepared.table.schema.fields.some(field => !prepared.timeColumns[field.name]))
      throw new Error('Resident exploration requires a scalar or category field');
    if (descriptor.dataset.datasetId === runtime.explorer.dataset.id) {
      const fixture = readSandboxFixture(descriptor);
      if (fixture.rowCount !== source.table.numRows)
        throw new Error('Fixture row count does not match bundle data');
      assertKeplerSandboxSourceMatches(descriptor, {
        ...getSandboxSource(runtime),
        reference: descriptor.reference
      });
    } else if (roles) {
      // A recorded partial population must match its bundled bytes; it is never reinterpreted as finite.
      if (
        roles.stream?.partial &&
        (roles.stream.partial.rowCount !== source.table.numRows ||
          roles.stream.partial.batchCount !== source.table.batches.length)
      )
        throw new Error('Bundle partial stream counts do not match the saved manifest');
      if (roles.coordinates) validateTabularPointPositions(prepared, roles.coordinates);
      if (roles.h3Column) preparePublishedH3(source.table, roles.h3Column);
      if (roles.geometryColumn) {
        if (roles.tripOptions)
          preparePublishedTrips(source.table, roles.geometryColumn, roles.tripOptions);
        else {
          const geometry = preparePublishedGeometry(source.table, roles.geometryColumn);
          if (recipe)
            validatePolygonReplay(runtime, recipe, {
              column: roles.geometryColumn,
              source: geometry
            });
        }
      }
    }
  }
}

function assertSandboxReplay(
  saved: KeplerSandboxState,
  runtime: ComputeRuntime,
  nativeOnlyDatasetIds: readonly string[] = []
): void {
  const fixture = saved.sources.find(
    source => source.dataset.datasetId === runtime.explorer.dataset.id
  );
  if (!fixture) throw new Error('Saved source belongs to another dataset');
  if (
    saved.selections.length !== saved.sources.length ||
    new Set(saved.analyses.map(recipe => recipe.source.datasetId)).size !== saved.analyses.length
  )
    throw new Error('Every source needs one native selection and at most one analysis');
  readSandboxFixture(fixture);
  for (const source of saved.sources) if (source !== fixture) readPublishedSandboxSource(source);
  assertSandboxSourcesCovered(saved.nativeConfig, [
    ...saved.sources.map(source => source.dataset.datasetId),
    ...nativeOnlyDatasetIds
  ]);
  const selection = JSON.stringify({
    kind: 'native-kepler-config',
    filters: getNativeFilters(saved.nativeConfig)
  });
  if (saved.selections.some(savedSelection => JSON.stringify(savedSelection.value) !== selection))
    throw new Error('Saved native configuration and selection do not match');
  for (const recipe of saved.analyses) {
    const names = getAnalysisRecipeColumns(recipe);
    if (recipe.source.datasetId === fixture.dataset.datasetId)
      validateAnalysisRecipeCapacity(recipe, readSandboxFixture(fixture).rowCount);
    const source = saved.sources.find(item => item.dataset.datasetId === recipe.source.datasetId);
    if (names.some(name => !source?.schema.some(field => field.name === name)))
      throw new Error('Saved analysis field is missing from its source');
  }
}

async function applySandboxState(
  saved: KeplerSandboxState,
  callbacks: SandboxCallbacks,
  reimport: boolean
): Promise<ComputeRuntime> {
  const runtime = callbacks.runtime;
  const sources = getSandboxSources(runtime);
  const bindings = saved.sources.map(source => {
    const current = sources.find(item => item.dataset.datasetId === source.dataset.datasetId);
    if (!current) throw new Error('A saved source is unavailable; restore its portable bundle');
    assertKeplerSandboxSourceMatches(source, current);
    if (!reimport && source.dataset.revision !== current.dataset.revision)
      throw new Error('Dataset changed; use explicit reimport or bundle restoration');
    const explorer =
      source.dataset.datasetId === runtime.explorer.dataset.id
        ? runtime.explorer
        : runtime.publications.get(source.dataset.datasetId)!.consumer.explorer;
    if (!explorer.isCurrent) throw new Error('Cannot replay a stale resident dataset');
    const recipe = saved.analyses.find(item => item.source.datasetId === current.dataset.datasetId);
    if (recipe) {
      validateAnalysisRecipeCapacity(recipe, explorer.dataset.rowCount);
      const publication = runtime.publications.get(source.dataset.datasetId);
      validatePolygonReplay(
        runtime,
        recipe,
        publication?.polygonSource,
        publication?.polygonAnalysisUnavailableReason
      );
      for (const name of getAnalysisNumericColumns(recipe)) {
        const column = explorer.dataset.value.columns[name];
        if (!column || column.format !== 'float32' || !column.length)
          throw new Error('Saved analysis requires nonempty prepared Float32 columns');
      }
    }
    return {source: current, explorer};
  });
  // Malformed extension members reject the replay before native configuration changes.
  validateSandboxExtensions(saved.settings?.extensions);
  await callbacks.applyNativeConfig(saved.nativeConfig);
  if (bindings.some(binding => !binding.explorer.isCurrent))
    throw new Error('Dataset changed while replaying native configuration');
  const rebound = rebindKeplerSandboxState(
    saved,
    bindings.map(binding => ({
      source: binding.source,
      restoredSelectionRevision: binding.explorer.currentRevision + 1
    })),
    {allowReimport: reimport}
  );
  for (const datasetId of [runtime.explorer.dataset.id, ...runtime.publications.keys()]) {
    const recipe = rebound.analyses.find(item => item.source.datasetId === datasetId);
    const published = runtime.publications.get(datasetId);
    const analysis =
      datasetId === runtime.explorer.dataset.id
        ? runtime.analysis
        : recipe
          ? published!.getAnalysis()
          : published?.analysis;
    if (recipe) analysis!.restoreRecipe(JSON.stringify(recipe));
    else analysis?.setMode('off');
  }
  applySandboxExtensions(saved.settings?.extensions, (datasetId, options) =>
    (datasetId === runtime.explorer.dataset.id
      ? runtime.analysis
      : runtime.publications.get(datasetId)?.getAnalysis()
    )?.setStreamedWeightsOptions(options)
  );
  runtime.requestUpdate();
  return runtime;
}

export function getSandboxExplorer(
  runtime: ComputeRuntime,
  datasetId: string
): KeplerSessionExplorer {
  const explorer =
    datasetId === runtime.explorer.dataset.id
      ? runtime.explorer
      : runtime.publications.get(datasetId)?.consumer.explorer;
  if (!explorer || !explorer.isCurrent)
    throw new Error('Choose an available resident dataset for export');
  return explorer;
}

export async function exportSandboxSelection(
  runtime: ComputeRuntime,
  mode: 'page' | 'all-selected',
  signal: AbortSignal,
  datasetId = runtime.explorer.dataset.id
): Promise<KeplerSelectionExport> {
  const explorer = getSandboxExplorer(runtime, datasetId);
  return materializeKeplerSelection(runtime.session, explorer, {
    ...(mode === 'page' ? {mode, offset: 0, limit: 100} : {mode, maximumRows: 1_000_000}),
    columns: Object.keys(explorer.dataset.value.columns),
    maximumReadbackBytes: 64 * 1024 * 1024,
    maximumCpuBytes: 64 * 1024 * 1024,
    signal
  });
}

/** Explicit JSON representation preserves nulls and labels non-finite values without silent coercion. */
export function serializeSandboxExport(result: KeplerSelectionExport): string {
  return JSON.stringify({
    schemaVersion: 1,
    mode: result.mode,
    representation: result.representation,
    datasetId: result.datasetId,
    datasetRevision: result.datasetRevision,
    selectionRevision: result.selectionRevision,
    offset: result.offset,
    selectedCount: result.selectedCount,
    rowCount: result.rowCount,
    readbackBytes: result.readbackBytes,
    schema: result.schema,
    dictionaries: result.dictionaries,
    nonFiniteEncoding: 'NaN, Infinity and -Infinity are explicit strings',
    sourceRowIndices: Array.from(result.sourceRowIndices),
    columns: Object.fromEntries(
      Object.entries(result.columns).map(([name, values]) => [
        name,
        Array.from(values, (value, index) =>
          result.validity[name]?.[index] === 0
            ? null
            : Number.isFinite(value)
              ? value
              : String(value)
        )
      ])
    )
  });
}

function getNativeFilters(config: KeplerJSONValue): KeplerJSONValue {
  if (!config || typeof config !== 'object' || Array.isArray(config) || !('config' in config))
    throw new Error('Invalid saved Kepler config');
  const inner = config['config'];
  if (!inner || typeof inner !== 'object' || Array.isArray(inner) || !('visState' in inner))
    throw new Error('Missing Kepler visual state');
  const visual = inner['visState'];
  if (
    !visual ||
    typeof visual !== 'object' ||
    Array.isArray(visual) ||
    !('filters' in visual) ||
    !Array.isArray(visual['filters'])
  )
    throw new Error('Missing Kepler filters');
  return visual['filters'];
}

function cloneNativeConfig(value: unknown, depth = 0): KeplerJSONValue {
  if (depth > 64) throw new Error('Native configuration nesting exceeds capacity');
  // Unsynchronized split viewports carry +/-Infinity maxBounds; JSON (and Kepler's own export) store null.
  if (typeof value === 'number' && !Number.isFinite(value)) return null;
  if (Array.isArray(value)) return value.map(item => cloneNativeConfig(item, depth + 1));
  if (value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype) {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([, item]) => item !== undefined)
        .map(([key, item]) => [key, cloneNativeConfig(item, depth + 1)])
    );
  }
  return cloneKeplerJSON(value);
}

/** Geometry roles stay source data; only analysis measures need prepared Float32 columns. */
function getPolygonGeometryColumn(recipe: KeplerAnalysisRecipe): string | undefined {
  const operation = recipe.operation;
  return operation.kind === 'spatial-rate-polygons' ||
    operation.kind === 'hot-spots-polygons' ||
    operation.kind === 'local-moran-polygons' ||
    operation.kind === 'quantile-lisa-polygons'
    ? operation.weights.geometryColumn
    : undefined;
}

function getAnalysisNumericColumns(recipe: KeplerAnalysisRecipe): readonly string[] {
  const operation = recipe.operation;
  if (operation.kind === 'spatial-rate-polygons')
    return [...new Set([operation.events, operation.populations])];
  if (
    operation.kind === 'hot-spots-polygons' ||
    operation.kind === 'local-moran-polygons' ||
    operation.kind === 'quantile-lisa-polygons'
  )
    return [operation.column];
  return getAnalysisRecipeColumns(recipe);
}

/** Reject unavailable or over-capacity topology before native replay mutates the workspace. */
function validatePolygonReplay(
  runtime: ComputeRuntime,
  recipe: KeplerAnalysisRecipe,
  source?: PolygonAnalysisSource,
  unavailableReason?: string
): void {
  const geometryColumn = getPolygonGeometryColumn(recipe);
  if (!geometryColumn) return;
  if (!source || source.column !== geometryColumn)
    throw new Error(
      unavailableReason ?? 'Saved polygon analysis requires its retained original geometry'
    );
  const reservation = runtime.session.registry.reserveMemory(
    POLYGON_CONTIGUITY_PREPARATION_BYTES,
    'cpu',
    'scratch'
  );
  try {
    preparePolygonContiguityTopology(source.source);
  } finally {
    reservation.release();
  }
}

function readFlowRoles(value: unknown, schema: readonly {name: string}[]): FlowRoles {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('Invalid saved flow roles');
  const record = value as Record<string, unknown>;
  const allowed = ['targetLongitude', 'targetLatitude', 'count', 'sourceName', 'targetName'];
  if (Object.keys(record).some(key => !allowed.includes(key)))
    throw new Error('Unsupported saved flow role');
  for (const key of allowed)
    if (
      record[key] !== undefined &&
      (typeof record[key] !== 'string' || !schema.some(field => field.name === record[key]))
    )
      throw new Error('Invalid saved flow fields');
  if (typeof record['targetLongitude'] !== 'string' || typeof record['targetLatitude'] !== 'string')
    throw new Error('Flow roles require target longitude and latitude');
  return {
    targetLongitude: record['targetLongitude'],
    targetLatitude: record['targetLatitude'],
    ...(record['count'] ? {count: record['count'] as string} : {}),
    ...(record['sourceName'] ? {sourceName: record['sourceName'] as string} : {}),
    ...(record['targetName'] ? {targetName: record['targetName'] as string} : {})
  };
}
