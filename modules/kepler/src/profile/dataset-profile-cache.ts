// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPU_ANALYSIS_MEASUREMENT_LEVELS,
  GPU_ANALYSIS_ROW_SEMANTICS,
  type GPUAnalysisCoordinateModel,
  type GPUAnalysisMeasurementLevel,
  type GPUAnalysisRowSemantics
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import type {CatalogFeatureCollection} from '../analysis/catalog-geojson';
import {
  getKeplerDatasetNetworkTopologyKey,
  profileKeplerDatasetNetwork
} from './dataset-network-profile';
import {
  applyKeplerDatasetProfileDeclarations,
  getKeplerDatasetProfileUndeclared,
  withKeplerDatasetNetworkProfile
} from './dataset-profile';
import type {
  KeplerDatasetNetworkProfile,
  KeplerDatasetProfile,
  KeplerDatasetProfileDeclarations
} from './dataset-profile-types';

// A profile is computed once per dataset revision. The cache keeps the latest profile of each
// dataset, drops the least recently used when it is full, and keeps what a user declared about a
// dataset apart from the profile, so a correction survives a new revision (and a dropped entry).

/** Largest cache a host may ask for; a profile holds one record per column, so this is generous. */
export const KEPLER_DATASET_PROFILE_CACHE_MAXIMUM_ENTRIES_LIMIT = 10_000;

/** Version written by {@link serializeKeplerDatasetProfileDeclarations}. */
export const KEPLER_DATASET_PROFILE_DECLARATIONS_VERSION = 1;

export type KeplerDatasetProfileCache = {
  /** The profile for exactly this revision, with the dataset's declarations applied; undefined when absent or stale. Marks the entry as recently used. */
  get(datasetId: string, revision?: number): KeplerDatasetProfile | undefined;
  /** Source identity, unaffected by declarations or lazy derived facts. */
  getGeneration(datasetId: string): number | undefined;
  /** Publishes derived facts only if the cached source is still the expected generation. */
  refine(
    profile: KeplerDatasetProfile,
    expectedGeneration: number | undefined
  ): KeplerDatasetProfile | undefined;
  /** Stores the latest profile of a dataset, applies its stored declarations and returns the result. */
  set(profile: KeplerDatasetProfile): KeplerDatasetProfile;
  /**
   * Explicit lazy refinement. Loads only a new topology on the current revision; never loads stale
   * sources. Pass bindingId for binding-specific preparation such as forced splitting in snaps.
   * GPU crossing noding stays unknown. Retains at most 16 summaries per dataset.
   */
  profileNetwork(
    datasetId: string,
    revision: number | undefined,
    parameters: Readonly<Record<string, CatalogParameterValue>>,
    loadLines: () => CatalogFeatureCollection,
    executionEngine?: 'cpu' | 'gpu',
    bindingId?: string
  ): KeplerDatasetProfile | undefined;
  /** Drops the cached profile of a dataset. Its declarations are kept. */
  invalidate(datasetId: string): void;
  /** Declarations stored for a dataset. */
  getDeclarations(datasetId: string): KeplerDatasetProfileDeclarations | undefined;
  /**
   * Replaces the declarations of a dataset (undefined removes them) and re-applies them to the
   * cached profile. The declarations are validated by the strict reader of the parser (an `Error`
   * names the path) and copied, so a later change to the host's object changes nothing here. They
   * are applied to the profile without declarations, so a removed correction goes back to the
   * guess, also when the profile was made with `options.declarations`; those are replaced by the
   * stored ones while any are stored. Removing declarations that were never stored leaves the
   * cached profile as it is. A declared coordinate model recomputes the spacing statistics.
   * Returns the updated profile when one is cached.
   */
  setDeclarations(
    datasetId: string,
    declarations: KeplerDatasetProfileDeclarations | undefined
  ): KeplerDatasetProfile | undefined;
  /** Number of cached profiles. */
  readonly size: number;
};

/**
 * `base` is the profile without declarations (the one the profile functions made before applying
 * `options.declarations`), `applied` what `get` returns.
 */
type CacheEntry = {
  generation: number;
  base: KeplerDatasetProfile;
  applied: KeplerDatasetProfile;
  networks: Map<string, KeplerDatasetNetworkProfile>;
};
// Bound retained summaries when repeatedly changing the snapping slider.
const MAXIMUM_NETWORK_TOPOLOGIES_PER_DATASET = 16;

export function createKeplerDatasetProfileCache(options: {
  maximumEntries: number;
}): KeplerDatasetProfileCache {
  const maximumEntries = options?.maximumEntries;
  if (
    !Number.isInteger(maximumEntries) ||
    maximumEntries < 1 ||
    maximumEntries > KEPLER_DATASET_PROFILE_CACHE_MAXIMUM_ENTRIES_LIMIT
  ) {
    throw new RangeError(
      `createKeplerDatasetProfileCache options.maximumEntries must be an integer from 1 to ${KEPLER_DATASET_PROFILE_CACHE_MAXIMUM_ENTRIES_LIMIT}; got ${maximumEntries}`
    );
  }
  // A Map iterates in insertion order, so the first key is the least recently used one.
  const entries = new Map<string, CacheEntry>();
  let nextGeneration = 0;
  const declarationsByDataset = new Map<string, KeplerDatasetProfileDeclarations>();

  const applyStored = (datasetId: string, base: KeplerDatasetProfile): KeplerDatasetProfile => {
    const declarations = declarationsByDataset.get(datasetId);
    return declarations ? applyKeplerDatasetProfileDeclarations(base, declarations) : base;
  };
  const isUsingStoredDeclarations = (datasetId: string): boolean =>
    declarationsByDataset.has(datasetId);

  return {
    get(datasetId, revision) {
      const entry = entries.get(datasetId);
      if (!entry || entry.base.revision !== revision) return undefined;
      entries.delete(datasetId);
      entries.set(datasetId, entry);
      return entry.applied;
    },
    getGeneration(datasetId) {
      return entries.get(datasetId)?.generation;
    },
    refine(profile, expectedGeneration) {
      const entry = entries.get(profile.datasetId);
      if (
        !entry ||
        entry.generation !== expectedGeneration ||
        entry.base.revision !== profile.revision
      )
        return undefined;
      entry.base = getKeplerDatasetProfileUndeclared(profile);
      entry.applied = isUsingStoredDeclarations(profile.datasetId)
        ? applyStored(profile.datasetId, entry.base)
        : profile;
      entries.delete(profile.datasetId);
      entries.set(profile.datasetId, entry);
      return entry.applied;
    },
    set(profile) {
      // Stored declarations win over those the profile was made with; with none stored the
      // profile is kept as given.
      const base = getKeplerDatasetProfileUndeclared(profile);
      const applied = isUsingStoredDeclarations(profile.datasetId)
        ? applyStored(profile.datasetId, base)
        : profile;
      entries.delete(profile.datasetId);
      entries.set(profile.datasetId, {
        base,
        applied,
        networks: new Map(),
        generation: ++nextGeneration
      });
      while (entries.size > maximumEntries) {
        const oldest = entries.keys().next().value as string;
        entries.delete(oldest);
      }
      return applied;
    },
    profileNetwork(datasetId, revision, parameters, loadLines, executionEngine = 'cpu', bindingId) {
      const entry = entries.get(datasetId);
      if (!entry || entry.base.revision !== revision) return undefined;
      const topologyKey = getKeplerDatasetNetworkTopologyKey(
        parameters,
        executionEngine,
        bindingId
      );
      let network = topologyKey === undefined ? undefined : entry.networks.get(topologyKey);
      if (!network && topologyKey !== undefined) {
        const lines = loadLines();
        if (entries.get(datasetId) !== entry || entry.base.revision !== revision) return undefined;
        network = profileKeplerDatasetNetwork(
          entry.base,
          lines,
          parameters,
          executionEngine,
          bindingId
        );
        if (network) entry.networks.set(topologyKey, network);
        while (entry.networks.size > MAXIMUM_NETWORK_TOPOLOGIES_PER_DATASET) {
          entry.networks.delete(entry.networks.keys().next().value!);
        }
      }
      entry.applied = withKeplerDatasetNetworkProfile(entry.applied, network);
      entries.delete(datasetId);
      entries.set(datasetId, entry);
      return entry.applied;
    },
    invalidate(datasetId) {
      entries.delete(datasetId);
    },
    getDeclarations(datasetId) {
      return declarationsByDataset.get(datasetId);
    },
    setDeclarations(datasetId, declarations) {
      const entry = entries.get(datasetId);
      if (declarations === undefined) {
        if (!declarationsByDataset.delete(datasetId)) return entry?.applied;
      } else {
        const checked = readDeclarations(
          'createKeplerDatasetProfileCache setDeclarations',
          declarations,
          false
        );
        declarationsByDataset.set(datasetId, freezeDeclarations(checked));
      }
      if (!entry) return undefined;
      entry.applied = withKeplerDatasetNetworkProfile(
        applyStored(datasetId, entry.base),
        entry.applied.network
      );
      return entry.applied;
    },
    get size() {
      return entries.size;
    }
  };
}

// ---------------------------------------------------------------------------------------------
// Saving declarations with a session
// ---------------------------------------------------------------------------------------------

const DECLARATION_KEYS: readonly string[] = [
  'version',
  'measurement',
  'rowSemantics',
  'coordinateModel'
];
const COORDINATE_MODELS: readonly GPUAnalysisCoordinateModel[] = ['geographic', 'planar'];

type RawDeclarations = {
  version?: unknown;
  measurement?: unknown;
  rowSemantics?: unknown;
  coordinateModel?: unknown;
};

function isPlainObject(value: unknown): value is RawDeclarations {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function fail(functionName: string, path: string, problem: string): never {
  throw new Error(`${functionName} ${path}: ${problem}`);
}

/** Freezes a checked copy so that the host cannot change what the cache stores through `getDeclarations`. */
function freezeDeclarations(
  declarations: KeplerDatasetProfileDeclarations
): KeplerDatasetProfileDeclarations {
  if (declarations.measurement) Object.freeze(declarations.measurement);
  return Object.freeze(declarations);
}

/**
 * Checks a value as declarations, strictly: unknown keys, unknown levels and wrong types are an
 * `Error` that names the path (`$.measurement.fare`). Returns a copy built from own properties
 * only, so a `__proto__` key in saved text is read as a column name, never as a prototype.
 */
function readDeclarations(
  functionName: string,
  value: unknown,
  requireVersion: boolean
): KeplerDatasetProfileDeclarations {
  if (!isPlainObject(value)) fail(functionName, '$', 'expected an object');
  for (const key of Object.keys(value))
    if (!DECLARATION_KEYS.includes(key))
      fail(functionName, `$.${key}`, `unknown key; accepted: ${DECLARATION_KEYS.join(', ')}`);
  if (requireVersion && value.version !== KEPLER_DATASET_PROFILE_DECLARATIONS_VERSION)
    fail(
      functionName,
      '$.version',
      `expected ${KEPLER_DATASET_PROFILE_DECLARATIONS_VERSION}; got ${JSON.stringify(value.version)}`
    );
  const declarations: KeplerDatasetProfileDeclarations = {};
  if (value.measurement !== undefined) {
    if (
      typeof value.measurement !== 'object' ||
      value.measurement === null ||
      Array.isArray(value.measurement)
    )
      fail(functionName, '$.measurement', 'expected an object of column name to measurement level');
    const entries: [string, GPUAnalysisMeasurementLevel][] = [];
    for (const [name, level] of Object.entries(value.measurement as Record<string, unknown>)) {
      if (typeof level !== 'string')
        fail(
          functionName,
          `$.measurement.${name}`,
          `expected a string; got ${JSON.stringify(level)}`
        );
      if (!(GPU_ANALYSIS_MEASUREMENT_LEVELS as readonly string[]).includes(level))
        fail(
          functionName,
          `$.measurement.${name}`,
          `unknown measurement level "${level}"; accepted: ${GPU_ANALYSIS_MEASUREMENT_LEVELS.join(', ')}`
        );
      entries.push([name, level as GPUAnalysisMeasurementLevel]);
    }
    declarations.measurement = Object.fromEntries(entries);
  }
  if (value.rowSemantics !== undefined) {
    if (
      typeof value.rowSemantics !== 'string' ||
      !(GPU_ANALYSIS_ROW_SEMANTICS as readonly string[]).includes(value.rowSemantics)
    )
      fail(
        functionName,
        '$.rowSemantics',
        `expected one of ${GPU_ANALYSIS_ROW_SEMANTICS.join(', ')}; got ${JSON.stringify(value.rowSemantics)}`
      );
    declarations.rowSemantics = value.rowSemantics as GPUAnalysisRowSemantics;
  }
  if (value.coordinateModel !== undefined) {
    if (
      typeof value.coordinateModel !== 'string' ||
      !(COORDINATE_MODELS as readonly string[]).includes(value.coordinateModel)
    )
      fail(
        functionName,
        '$.coordinateModel',
        `expected one of ${COORDINATE_MODELS.join(', ')}; got ${JSON.stringify(value.coordinateModel)}`
      );
    declarations.coordinateModel = value.coordinateModel as GPUAnalysisCoordinateModel;
  }
  return declarations;
}

/**
 * Text a host can store with a session: JSON with a version, keys in a fixed order and measurement
 * names sorted, so the same corrections always give the same text. Throws for declarations that
 * the parser would reject.
 */
export function serializeKeplerDatasetProfileDeclarations(
  declarations: KeplerDatasetProfileDeclarations
): string {
  const checked = readDeclarations(
    'serializeKeplerDatasetProfileDeclarations',
    declarations,
    false
  );
  const document: RawDeclarations = {version: KEPLER_DATASET_PROFILE_DECLARATIONS_VERSION};
  if (checked.measurement)
    document.measurement = Object.fromEntries(
      Object.entries(checked.measurement).sort(([first], [second]) =>
        first < second ? -1 : first > second ? 1 : 0
      )
    );
  if (checked.rowSemantics) document.rowSemantics = checked.rowSemantics;
  if (checked.coordinateModel) document.coordinateModel = checked.coordinateModel;
  return JSON.stringify(document);
}

/**
 * Reads text written by {@link serializeKeplerDatasetProfileDeclarations}. Strict: text that is
 * not JSON, a wrong or missing version, an unknown key, an unknown measurement level, row
 * semantics or coordinate model, and a value of the wrong type each throw an `Error` that names
 * the path of the fault.
 */
export function parseKeplerDatasetProfileDeclarations(
  text: string
): KeplerDatasetProfileDeclarations {
  const functionName = 'parseKeplerDatasetProfileDeclarations';
  if (typeof text !== 'string') fail(functionName, '$', 'expected text');
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch (error) {
    return fail(functionName, '$', `not valid JSON (${(error as Error).message})`);
  }
  return readDeclarations(functionName, value, true);
}
