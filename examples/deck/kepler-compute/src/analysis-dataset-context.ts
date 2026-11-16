// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createKeplerDatasetProfileCache,
  parseKeplerDatasetProfileDeclarations,
  profileKeplerColumnTable,
  serializeKeplerDatasetProfileDeclarations,
  type KeplerDatasetProfile,
  type KeplerDatasetProfileDeclarations,
  type KeplerJSONValue,
  type KeplerProfileColumnInput
} from '@luma-contributors/kepler';
import {
  guessGPUAnalysisCoordinateModel,
  matchGPUAnalysisFieldRole
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  getCatalogResidentFields,
  MAXIMUM_CATALOG_RESIDENT_READBACK_BYTES,
  readCatalogColumnsFromResidentDataset,
  type CatalogResidentDataset
} from './analysis-catalog-resident-input';
import {MAXIMUM_CATALOG_NATIVE_ROWS, readCatalogGeometry} from './analysis-catalog-input';

export const ANALYSIS_PROFILE_LIMITS = Object.freeze({
  maximumRows: MAXIMUM_CATALOG_NATIVE_ROWS,
  maximumVertices: 200_000,
  maximumColumns: 64,
  maximumSampleSize: 512,
  maximumReadbackBytes: MAXIMUM_CATALOG_RESIDENT_READBACK_BYTES
});

export type AnalysisDatasetContextEntry = {
  dataset: CatalogResidentDataset;
  profile: KeplerDatasetProfile;
  stage: 'metadata' | 'reading' | 'statistics' | 'unavailable';
  limitation?: string;
  /** Provider changes and snapshot replacement invalidate every asynchronous stage. */
  key: string;
  readRevision: string;
};

/** Publication declarations outrank AX role guesses. Every consumer can show the same mapping. */
export function getAnalysisDatasetRoles(entry: AnalysisDatasetContextEntry) {
  const {dataset, profile} = entry;
  const excluded = new Set([
    ...(profile.pointColumns
      ? [profile.pointColumns.longitude, profile.pointColumns.latitude]
      : []),
    ...profile.geometryColumns
  ]);
  const guessedValue = matchGPUAnalysisFieldRole(
    {role: 'value', label: 'Value', accepts: ['number', 'integer'], required: true},
    profile.columns.filter(column => !excluded.has(column.name))
  )[0]?.name;
  const valueColumn = dataset.valueColumn ?? guessedValue;
  const timeColumn =
    dataset.timeColumns?.[0] ?? profile.columns.find(column => column.type === 'time')?.name;
  return {
    pointColumns: profile.pointColumns,
    geometryColumn: dataset.geometryColumn,
    timeColumn,
    valueColumn,
    valueSource: dataset.valueColumn ? ('declared' as const) : ('guess' as const),
    timeSource: dataset.timeColumns?.length ? ('declared' as const) : ('guess' as const)
  };
}

/** Preserve storage declarations; strings and unknown schema types still need AX value inference. */
export function getAnalysisColumnType(type: string | undefined): KeplerProfileColumnInput['type'] {
  switch (type) {
    case 'integer':
    case 'int32':
      return 'integer';
    case 'real':
    case 'number':
    case 'float64':
      return 'number';
    case 'timestamp':
    case 'time':
    case 'date':
      return 'time';
    case 'boolean':
      return 'boolean';
    case 'string':
    case 'text':
    case 'category':
      return type;
    default:
      return undefined;
  }
}

/** The workbench's existing token-pair rule, shared by all host metadata consumers. */
export function guessAnalysisPointColumns(fields: readonly {name: string; type: string}[]) {
  const candidates = fields
    .filter(field => ['integer', 'real', 'number', 'float64', 'int32'].includes(field.type))
    .map(field => {
      const tokens = field.name
        .replace(/([a-z])([A-Z])/g, '$1_$2')
        .toLowerCase()
        .split(/[^a-z0-9]+/);
      const index = tokens.findIndex(token =>
        ['lat', 'latitude', 'lng', 'lon', 'long', 'longitude'].includes(token)
      );
      return {
        name: field.name,
        role:
          index < 0
            ? undefined
            : ['lat', 'latitude'].includes(tokens[index])
              ? 'latitude'
              : 'longitude',
        group: JSON.stringify([tokens.slice(0, index), tokens.slice(index + 1)])
      };
    });
  for (const candidate of candidates) {
    if (!candidate.role) continue;
    const partner = candidates.find(
      other => other.role && other.role !== candidate.role && other.group === candidate.group
    );
    if (partner)
      return candidate.role === 'longitude'
        ? {longitude: candidate.name, latitude: partner.name}
        : {longitude: partner.name, latitude: candidate.name};
  }
  return undefined;
}

function getAnalysisMetadataPointColumns(dataset: CatalogResidentDataset) {
  if (dataset.geometryColumn) return undefined;
  return (
    dataset.coordinates ??
    guessAnalysisPointColumns(
      getCatalogResidentFields(dataset).map(field => ({
        name: field.name,
        type: dataset.fieldTypes?.[field.name] ?? 'string'
      }))
    )
  );
}

function makeMetadataProfile(dataset: CatalogResidentDataset): KeplerDatasetProfile {
  const fields = getCatalogResidentFields(dataset);
  const pointColumns = getAnalysisMetadataPointColumns(dataset);
  // The AX profiler owns column/type semantics. Zero observed rows explicitly carry no statistics.
  const empty = profileKeplerColumnTable(
    {
      rowCount: 0,
      columns: fields.map(field => ({
        name: field.name,
        type: dataset.timeColumns?.includes(field.name)
          ? 'time'
          : getAnalysisColumnType(dataset.fieldTypes?.[field.name]),
        values: []
      }))
    },
    {
      datasetId: dataset.id,
      label: dataset.label,
      revision: dataset.revision,
      ...(dataset.geometryColumn ? {geometryColumns: [dataset.geometryColumn]} : {pointColumns})
    }
  );
  const profile: KeplerDatasetProfile = {
    datasetId: dataset.id,
    label: dataset.label,
    revision: dataset.revision,
    kind: dataset.geometryKind
      ? {point: 'points' as const, line: 'lines' as const, polygon: 'polygons' as const}[
          dataset.geometryKind
        ]
      : pointColumns
        ? 'points'
        : 'attributes',
    ...(dataset.geometryKind ? {catalogGeometry: dataset.geometryKind} : {}),
    rowCount: dataset.rowCount,
    geometryColumns: empty.geometryColumns,
    ...(pointColumns ? {pointColumns, catalogGeometry: 'point' as const} : {}),
    columns: empty.columns.map(column => ({
      name: column.name,
      type: column.type,
      rowCount: dataset.rowCount,
      nullCount: 0,
      sampleSize: 0
    }))
  };
  if (pointColumns || dataset.geometryColumn) {
    profile.coordinates = {
      model: 'geographic',
      source: 'guess',
      confidence: 'low',
      reason:
        'Coordinate roles suggest geographic degrees; confirm the coordinate system before metric analysis.'
    };
  }
  if (dataset.bounds) {
    profile.extent = dataset.bounds;
    const guess = guessGPUAnalysisCoordinateModel(dataset.bounds);
    profile.coordinates = {
      model: guess.model,
      source: 'guess',
      confidence: guess.confidence,
      reason: guess.reason
    };
  }
  return profile;
}

/** Count before the AX profiler expands geometry or runs neighbour statistics. */
function requireBoundedVertices(columns: readonly unknown[][], geometryIndex: number): void {
  if (geometryIndex < 0) return;
  let vertices = 0;
  for (const value of columns[geometryIndex]) {
    let geometry: unknown;
    try {
      geometry = typeof value === 'string' ? JSON.parse(value) : value;
    } catch {
      continue;
    }
    const pending: unknown[] = [geometry];
    while (pending.length) {
      const current = pending.pop();
      if (Array.isArray(current)) {
        if (typeof current[0] === 'number' && typeof current[1] === 'number') vertices++;
        else for (const child of current) pending.push(child);
      } else if (current && typeof current === 'object') {
        const record = current as Record<string, unknown>;
        pending.push(record['geometry'], record['coordinates'], record['geometries']);
      }
      if (vertices > ANALYSIS_PROFILE_LIMITS.maximumVertices)
        throw new Error(
          `AX profile geometry exceeds ${ANALYSIS_PROFILE_LIMITS.maximumVertices} vertices`
        );
    }
  }
}

export function createAnalysisDatasetContext() {
  const cache = createKeplerDatasetProfileCache({maximumEntries: 1000});
  const entries = new Map<string, AnalysisDatasetContextEntry>();
  const declarationIds = new Set<string>();
  const listeners = new Set<() => void>();
  const pending = new Map<AnalysisDatasetContextEntry, Promise<KeplerDatasetProfile>>();
  let version = 0;
  const changed = () => {
    version++;
    for (const listener of listeners) listener();
  };
  const requireEntry = (datasetId: string) => {
    const entry = entries.get(datasetId);
    if (
      !entry ||
      !entry.dataset.isCurrent() ||
      entry.dataset.provider.getRevision() !== entry.readRevision
    )
      throw new Error('AX profile dataset is unavailable or stale');
    return entry;
  };
  const applyDeclarations = (
    datasetId: string,
    declarations: KeplerDatasetProfileDeclarations | undefined
  ) => {
    const wasPlanar = cache.getDeclarations(datasetId)?.coordinateModel === 'planar';
    const profile = cache.setDeclarations(datasetId, declarations);
    if (declarations === undefined) declarationIds.delete(datasetId);
    else declarationIds.add(datasetId);
    const entry = entries.get(datasetId);
    if (!entry || !profile) return;
    entry.profile = profile;
    if (
      wasPlanar !== (declarations?.coordinateModel === 'planar') &&
      !entry.dataset.coordinates &&
      getAnalysisMetadataPointColumns(entry.dataset) &&
      (entry.stage === 'statistics' || entry.stage === 'unavailable')
    ) {
      // Re-evaluate the mapping with current confirmation instead of reusing a cached acceptance
      // or rejection. Publication mappings remain explicit independently of this declaration.
      entry.profile = cache.set(makeMetadataProfile(entry.dataset));
      entry.stage = 'metadata';
      delete entry.limitation;
    }
  };
  return {
    subscribe(listener: () => void) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    getVersion: () => version,
    /** Metadata only; call at a host update/commit boundary, since changes notify subscribers. */
    synchronize(datasets: readonly CatalogResidentDataset[]) {
      let lifecycleChanged = false;
      const currentIds = new Set(datasets.map(dataset => dataset.id));
      for (const datasetId of entries.keys())
        if (!currentIds.has(datasetId)) {
          entries.delete(datasetId);
          cache.invalidate(datasetId);
          lifecycleChanged = true;
        }
      for (const dataset of datasets) {
        if (!dataset.isCurrent()) {
          if (entries.delete(dataset.id)) lifecycleChanged = true;
          cache.invalidate(dataset.id);
          continue;
        }
        const readRevision = dataset.provider.getRevision();
        const key = JSON.stringify([
          dataset.revision,
          readRevision,
          dataset.rowCount,
          getCatalogResidentFields(dataset).map(field => [
            field.name,
            dataset.fieldTypes?.[field.name]
          ]),
          dataset.coordinates,
          dataset.geometryColumn,
          dataset.geometryKind,
          dataset.timeColumns,
          dataset.valueColumn,
          dataset.bounds
        ]);
        const existing = entries.get(dataset.id);
        if (
          existing?.key === key &&
          existing.dataset.isCurrent() &&
          existing.dataset.provider === dataset.provider &&
          (existing.dataset.owner ?? existing.dataset.provider) ===
            (dataset.owner ?? dataset.provider)
        )
          continue;
        cache.invalidate(dataset.id);
        const profile = cache.set(makeMetadataProfile(dataset));
        entries.set(dataset.id, {dataset, profile, key, readRevision, stage: 'metadata'});
        lifecycleChanged = true;
      }
      if (lifecycleChanged) changed();
    },
    get(datasetId: string) {
      const entry = entries.get(datasetId);
      return entry?.dataset.isCurrent() &&
        entry.dataset.provider.getRevision() === entry.readRevision
        ? entry
        : undefined;
    },
    getDeclarations: cache.getDeclarations,
    setDeclarations(datasetId: string, declarations: KeplerDatasetProfileDeclarations | undefined) {
      applyDeclarations(datasetId, declarations);
      changed();
    },
    /** Explicit bounded refinement. Never samples a population and labels it exact. */
    readStatistics(datasetId: string): Promise<KeplerDatasetProfile> {
      const entry = requireEntry(datasetId);
      if (entry.stage === 'statistics') return Promise.resolve(entry.profile);
      const running = pending.get(entry);
      if (running) return running;
      const task = (async () => {
        entry.stage = 'reading';
        delete entry.limitation;
        changed();
        try {
          const fields = getCatalogResidentFields(entry.dataset);
          if (fields.length > ANALYSIS_PROFILE_LIMITS.maximumColumns)
            throw new Error(`AX profile exceeds ${ANALYSIS_PROFILE_LIMITS.maximumColumns} columns`);
          const columns = await readCatalogColumnsFromResidentDataset(
            entry.dataset,
            ANALYSIS_PROFILE_LIMITS.maximumRows
          );
          if (
            entries.get(datasetId) !== entry ||
            !entry.dataset.isCurrent() ||
            entry.dataset.provider.getRevision() !== entry.readRevision
          )
            throw new Error('AX profile dataset changed during statistics');
          requireBoundedVertices(
            columns,
            fields.findIndex(field => field.name === entry.dataset.geometryColumn)
          );
          const inputs: KeplerProfileColumnInput[] = fields.map((field, index) => ({
            name: field.name,
            values:
              field.name === entry.dataset.geometryColumn
                ? columns[index].map(value => {
                    try {
                      const geometry = readCatalogGeometry(value);
                      return geometry ? JSON.stringify(geometry) : null;
                    } catch {
                      // Match the AX profiler's treatment of malformed geometry cells as missing.
                      return null;
                    }
                  })
                : columns[index],
            type: entry.dataset.timeColumns?.includes(field.name)
              ? 'time'
              : getAnalysisColumnType(entry.dataset.fieldTypes?.[field.name])
          }));
          // A planar declaration confirms the metadata pair in source units. Unconfirmed name
          // guesses still go through AX's degree-range validation rather than becoming mappings.
          const pointColumns =
            entry.dataset.coordinates ??
            (cache.getDeclarations(datasetId)?.coordinateModel === 'planar'
              ? (entry.profile.pointColumns ?? getAnalysisMetadataPointColumns(entry.dataset))
              : undefined);
          const profile = profileKeplerColumnTable(
            {rowCount: entry.dataset.rowCount, columns: inputs},
            {
              datasetId,
              label: entry.dataset.label,
              revision: entry.dataset.revision,
              maximumSampleSize: ANALYSIS_PROFILE_LIMITS.maximumSampleSize,
              ...(entry.dataset.geometryColumn
                ? {geometryColumns: [entry.dataset.geometryColumn]}
                : {pointColumns})
            }
          );
          // set() reapplies the latest declarations, including edits made while reading.
          entry.profile = cache.set(profile);
          entry.stage = 'statistics';
          changed();
          return entry.profile;
        } catch (error) {
          if (entries.get(datasetId) === entry) {
            entry.stage = 'unavailable';
            entry.limitation = error instanceof Error ? error.message : String(error);
            changed();
          }
          throw error;
        }
      })();
      pending.set(entry, task);
      void task.then(
        () => pending.delete(entry),
        () => pending.delete(entry)
      );
      return task;
    },
    collectDeclarations(): KeplerJSONValue | undefined {
      if (!declarationIds.size) return undefined;
      return {
        schemaVersion: 1,
        datasets: Object.fromEntries(
          [...declarationIds]
            .sort()
            .map(datasetId => [
              datasetId,
              serializeKeplerDatasetProfileDeclarations(cache.getDeclarations(datasetId)!)
            ])
        )
      };
    },
    restoreDeclarations(value: unknown) {
      const declarations = parseAnalysisProfileDeclarations(value);
      for (const datasetId of new Set([...declarationIds, ...declarations.keys()]))
        applyDeclarations(datasetId, declarations.get(datasetId));
      changed();
    }
  };
}

/** Strict extension preflight, before any host mutation. Dataset ids remain stable on replay. */
export function parseAnalysisProfileDeclarations(
  value: unknown
): Map<string, KeplerDatasetProfileDeclarations> {
  if (value === undefined) return new Map();
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error('Invalid AX profile declarations');
  const record = value as Record<string, unknown>;
  const datasets = record['datasets'];
  if (
    record['schemaVersion'] !== 1 ||
    Object.keys(record).some(key => !['schemaVersion', 'datasets'].includes(key)) ||
    !datasets ||
    typeof datasets !== 'object' ||
    Array.isArray(datasets) ||
    Object.keys(datasets).length > 1000
  )
    throw new Error('Invalid AX profile declarations schema or dataset capacity (1000)');
  return new Map(
    Object.entries(datasets).map(([datasetId, declaration]) => {
      if (!datasetId || typeof declaration !== 'string')
        throw new Error('Invalid AX profile declaration dataset');
      return [datasetId, parseKeplerDatasetProfileDeclarations(declaration)];
    })
  );
}

/** One host service shared by the catalog, dataset controls, sandbox and workbench. */
export const analysisDatasetContext = createAnalysisDatasetContext();
