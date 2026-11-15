// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_SPATIAL_ANALYSIS_DATA_KINDS} from '../spatial-analysis-ontology';
import type {GPUSpatialAnalysisDataKind} from '../spatial-analysis-ontology';

// Three vocabularies say what kind of data something is: the ontology's data kinds (this package),
// the website's data types, and the geometry kinds of the Kepler catalog bindings. The analysis
// package must not import the website or Kepler, so the names of the other two are declared here as
// plain constants, and the translation lives in one place. Keep the two copies in step with
// `website/src/catalog/taxonomy.ts` (`DATA_TYPES`) and
// `modules/kepler/src/analysis/catalog-binding-types.ts` (`geometry`, `accepts`).

/** Data types of the website catalog (`DATA_TYPES` in `website/src/catalog/taxonomy.ts`). */
export const GPU_ANALYSIS_WEBSITE_DATA_TYPES = Object.freeze([
  'points',
  'lines',
  'polygons',
  'raster',
  'terrain',
  'network',
  'table',
  'time-series',
  'trajectories',
  'point-cloud',
  'cells',
  'spatial-weights'
] as const);

export type GPUAnalysisWebsiteDataType = (typeof GPU_ANALYSIS_WEBSITE_DATA_TYPES)[number];

/** Geometry kinds a Kepler catalog binding input accepts. */
export const GPU_ANALYSIS_CATALOG_INPUT_GEOMETRIES = Object.freeze([
  'point',
  'line',
  'polygon'
] as const);

export type GPUAnalysisCatalogInputGeometry =
  (typeof GPU_ANALYSIS_CATALOG_INPUT_GEOMETRIES)[number];

/** Geometry kinds of a Kepler catalog output table; `none` is a table without geometry. */
export const GPU_ANALYSIS_CATALOG_OUTPUT_GEOMETRIES = Object.freeze([
  ...GPU_ANALYSIS_CATALOG_INPUT_GEOMETRIES,
  'none'
] as const);

export type GPUAnalysisCatalogOutputGeometry =
  (typeof GPU_ANALYSIS_CATALOG_OUTPUT_GEOMETRIES)[number];

/**
 * Ontology kind to website data types. Non-obvious rows:
 * - `surface` is both `raster` and `terrain`: a sampled regular grid. The website separates them by
 *   subject (imagery against elevation); the ontology does not.
 * - `attributes`, `summary`, `pairs` and `regions` are all columns or small tables to a person, so they
 *   are all `table`.
 * - `parameters` and `render-geometry` are never something a user has or looks for, so they have no
 *   website type.
 * - `point-cloud` has no row here because it has no ontology kind of its own; see
 *   {@link GPU_ANALYSIS_WEBSITE_TYPE_KINDS}.
 */
const ONTOLOGY_KIND_WEBSITE_TYPES: Record<
  GPUSpatialAnalysisDataKind,
  readonly GPUAnalysisWebsiteDataType[]
> = {
  parameters: [],
  attributes: ['table'],
  points: ['points'],
  lines: ['lines'],
  polygons: ['polygons'],
  trajectories: ['trajectories'],
  'time-series': ['time-series'],
  cells: ['cells'],
  'spatial-weights': ['spatial-weights'],
  pairs: ['table'],
  regions: ['table'],
  surface: ['raster', 'terrain'],
  network: ['network'],
  summary: ['table'],
  'render-geometry': []
};

freezeKindLists(ONTOLOGY_KIND_WEBSITE_TYPES);

/**
 * Website data type to ontology kinds. Non-obvious rows:
 * - `raster` and `terrain` both become `surface`.
 * - `table` becomes `attributes`, `summary`, `pairs` and `regions`: a table of columns can be any of
 *   them, and which one depends on the operation. `attributes` is listed first because it is what a
 *   loaded table is.
 * - `point-cloud` becomes `points`. The ontology has no point-cloud kind: operators take positions
 *   as a point set, and a lidar cloud is a point set with a height. The return trip is lossy:
 *   `points` maps back to `points` only, so a point cloud cannot be recovered from the ontology kind.
 */
export const GPU_ANALYSIS_WEBSITE_TYPE_KINDS: Record<
  GPUAnalysisWebsiteDataType,
  readonly GPUSpatialAnalysisDataKind[]
> = {
  points: ['points'],
  lines: ['lines'],
  polygons: ['polygons'],
  raster: ['surface'],
  terrain: ['surface'],
  network: ['network'],
  table: ['attributes', 'summary', 'pairs', 'regions'],
  'time-series': ['time-series'],
  trajectories: ['trajectories'],
  'point-cloud': ['points'],
  cells: ['cells'],
  'spatial-weights': ['spatial-weights']
};

freezeKindLists(GPU_ANALYSIS_WEBSITE_TYPE_KINDS);

/**
 * Catalog geometry to ontology kind. `none` is an output table with no geometry column; it becomes
 * `attributes`, the kind of a plain table. It could equally be `summary` (statistics tables have no
 * geometry either), but a table of per-row values is the common case and `summary` stays reachable
 * through {@link getGPUAnalysisCatalogGeometryForDataKind}.
 */
const CATALOG_GEOMETRY_KINDS: Record<GPUAnalysisCatalogOutputGeometry, GPUSpatialAnalysisDataKind> =
  {
    point: 'points',
    line: 'lines',
    polygon: 'polygons',
    none: 'attributes'
  };

Object.freeze(CATALOG_GEOMETRY_KINDS);

/**
 * Kinds the catalog returns as a table without geometry (`none`). The other kinds that the catalog
 * cannot carry at all (surface, network, trajectories, cells, spatial-weights, time-series,
 * parameters, render-geometry) are absent on purpose.
 */
const OUTPUT_TABLE_KINDS: ReadonlySet<GPUSpatialAnalysisDataKind> = new Set([
  'attributes',
  'summary',
  'pairs',
  'regions'
]);

/** Freezes a table of kind lists and its lists, so callers cannot change what later calls return. */
function freezeKindLists(table: Record<string, readonly unknown[]>): void {
  for (const list of Object.values(table)) Object.freeze(list);
  Object.freeze(table);
}

/** One data kind as a user meets it. */
export type GPUAnalysisDataKindRecord = {
  id: GPUSpatialAnalysisDataKind;
  /** Short name shown in a picker. */
  label: string;
  /** One line on what the kind is. */
  description: string;
  /**
   * True when a user can load it as a dataset. False when it is only produced by other operations
   * (weights, pairs, regions, summaries) or is an internal detail (parameters, render geometry).
   */
  loadable: boolean;
};

/** One record per ontology kind, in the order of `GPU_SPATIAL_ANALYSIS_DATA_KINDS`. */
export const GPU_ANALYSIS_DATA_KIND_RECORDS: readonly GPUAnalysisDataKindRecord[] = [
  {
    id: 'parameters',
    label: 'Settings',
    description: 'Small values such as a distance or a threshold that you change while exploring.',
    loadable: false
  },
  {
    id: 'attributes',
    label: 'Table of values',
    description: 'Columns of numbers, categories or times, one row per feature.',
    loadable: true
  },
  {
    id: 'points',
    label: 'Points',
    description: 'Locations such as events, stations or samples.',
    loadable: true
  },
  {
    id: 'lines',
    label: 'Lines',
    description: 'Paths and segments such as roads, rivers or tracks.',
    loadable: true
  },
  {
    id: 'polygons',
    label: 'Areas (polygons)',
    description: 'Areas such as parcels, tracts, zones or footprints, with holes where needed.',
    loadable: true
  },
  {
    id: 'trajectories',
    label: 'Moving tracks',
    description: 'Ordered positions with times for each moving object.',
    loadable: true
  },
  {
    id: 'time-series',
    label: 'Values over time',
    description: 'Values indexed by time, per place or as a place-by-period cube.',
    loadable: true
  },
  {
    id: 'cells',
    label: 'Grid cells',
    description: 'H3 or Quadbin cells, with or without a count or value per cell.',
    loadable: true
  },
  {
    id: 'spatial-weights',
    label: 'Neighbor relationships',
    description: 'Which features count as neighbors of which, with a weight for each link.',
    // A user can supply weights built elsewhere (GAL, GWT or CSR neighbor tables): the library takes a
    // CSR uploaded as is and checks it with GPUSpatialWeightsValidation. It is also built from
    // geometry by the neighbor and contiguity operators, so it is loadable and produced.
    loadable: true
  },
  {
    id: 'pairs',
    label: 'Feature pairs',
    description: 'Matched pairs of rows, such as a point and the area that contains it.',
    loadable: false
  },
  {
    id: 'regions',
    label: 'Groups and regions',
    description: 'A label or mask that puts each row in a group, such as a cluster or a selection.',
    loadable: false
  },
  {
    id: 'surface',
    label: 'Surface (raster or terrain)',
    description: 'Values on a regular grid: imagery, elevation or a density field.',
    loadable: true
  },
  {
    id: 'network',
    label: 'Network',
    description: 'Nodes and edges placed in space, such as a street network.',
    loadable: true
  },
  {
    id: 'summary',
    label: 'Statistics and test results',
    description: 'Estimates, test statistics and diagnostics for a dataset or a group.',
    loadable: false
  },
  {
    id: 'render-geometry',
    label: 'Drawing geometry',
    description: 'Vertices and indices prepared for drawing, not for further analysis.',
    loadable: false
  }
];
for (const record of GPU_ANALYSIS_DATA_KIND_RECORDS) Object.freeze(record);
Object.freeze(GPU_ANALYSIS_DATA_KIND_RECORDS);

/**
 * Throws for a value that is not one of the accepted names. Membership is tested in the frozen
 * list of names, so keys such as `constructor` are rejected instead of reaching a lookup table.
 */
function validateName(
  functionName: string,
  argumentName: string,
  value: unknown,
  accepted: readonly string[]
): void {
  if (!accepted.includes(value as string)) {
    throw new Error(
      `${functionName} ${argumentName} ${JSON.stringify(value)} is not one of: ${accepted.join(', ')}`
    );
  }
}

/** Returns the record of one ontology kind. Throws for a name that is not an ontology kind. */
export function getGPUAnalysisDataKindRecord(
  kind: GPUSpatialAnalysisDataKind
): GPUAnalysisDataKindRecord {
  validateName('getGPUAnalysisDataKindRecord', 'kind', kind, GPU_SPATIAL_ANALYSIS_DATA_KINDS);
  return GPU_ANALYSIS_DATA_KIND_RECORDS.find(candidate => candidate.id === kind)!;
}

/**
 * Ontology kinds a user can load directly as a dataset. `spatial-weights` is one of them: weights
 * built elsewhere (a GAL or GWT neighbor list, or a CSR table of focal, neighbor and weight) enter
 * the library as uploaded CSR and are checked by `GPUSpatialWeightsValidation`. The kinds that are
 * only produced by other operations are `pairs`, `regions`, `summary`, `parameters` and
 * `render-geometry`. Returns a new array on every call.
 */
export function listGPUAnalysisLoadableDataKinds(): GPUSpatialAnalysisDataKind[] {
  return GPU_ANALYSIS_DATA_KIND_RECORDS.filter(record => record.loadable).map(record => record.id);
}

/**
 * Website data types that show an ontology kind. Empty for `parameters` and `render-geometry`.
 * `surface` returns `raster` and `terrain`. The returned array is frozen. Throws for a name that
 * is not an ontology kind.
 */
export function getGPUAnalysisWebsiteDataTypes(
  kind: GPUSpatialAnalysisDataKind
): readonly GPUAnalysisWebsiteDataType[] {
  validateName('getGPUAnalysisWebsiteDataTypes', 'kind', kind, GPU_SPATIAL_ANALYSIS_DATA_KINDS);
  return ONTOLOGY_KIND_WEBSITE_TYPES[kind];
}

/**
 * Ontology kinds a website data type stands for, with the most direct kind first. `table` returns
 * four kinds and `point-cloud` returns `points`. The returned array is frozen. Throws for a name
 * that is not a website data type.
 */
export function getGPUAnalysisDataKindsForWebsiteDataType(
  type: GPUAnalysisWebsiteDataType
): readonly GPUSpatialAnalysisDataKind[] {
  validateName(
    'getGPUAnalysisDataKindsForWebsiteDataType',
    'type',
    type,
    GPU_ANALYSIS_WEBSITE_DATA_TYPES
  );
  return GPU_ANALYSIS_WEBSITE_TYPE_KINDS[type];
}

/**
 * Ontology kind of a Kepler catalog geometry. `point`, `line` and `polygon` map to `points`,
 * `lines` and `polygons`; `none` (an output table without geometry) maps to `attributes`. Throws
 * for a name that is not a catalog geometry.
 */
export function getGPUAnalysisDataKindForCatalogGeometry(
  geometry: GPUAnalysisCatalogOutputGeometry
): GPUSpatialAnalysisDataKind {
  validateName(
    'getGPUAnalysisDataKindForCatalogGeometry',
    'geometry',
    geometry,
    GPU_ANALYSIS_CATALOG_OUTPUT_GEOMETRIES
  );
  return CATALOG_GEOMETRY_KINDS[geometry];
}

/**
 * Catalog geometry that can carry an ontology kind, or `undefined` when the catalog cannot carry it.
 * As an input only `points`, `lines` and `polygons` are accepted. As an output the same three
 * give their geometry, and `attributes`, `summary`, `pairs` and `regions` give `none`, because the
 * catalog returns them as a table without geometry. Surfaces, networks, trajectories, cells,
 * spatial weights, time series, parameters and render geometry give `undefined` in both roles.
 * Throws for a name that is not an ontology kind or a role other than `input` or `output`.
 */
export function getGPUAnalysisCatalogGeometryForDataKind(
  kind: GPUSpatialAnalysisDataKind,
  role: 'input' | 'output' = 'output'
): GPUAnalysisCatalogOutputGeometry | undefined {
  validateName(
    'getGPUAnalysisCatalogGeometryForDataKind',
    'kind',
    kind,
    GPU_SPATIAL_ANALYSIS_DATA_KINDS
  );
  validateName('getGPUAnalysisCatalogGeometryForDataKind', 'role', role, ['input', 'output']);
  if (kind === 'points') return 'point';
  if (kind === 'lines') return 'line';
  if (kind === 'polygons') return 'polygon';
  if (role === 'output' && OUTPUT_TABLE_KINDS.has(kind)) return 'none';
  return undefined;
}
