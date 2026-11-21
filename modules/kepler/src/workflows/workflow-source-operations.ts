// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export {
  MAXIMUM_WORKFLOW_TABLE_COLUMNS,
  MAXIMUM_WORKFLOW_TABLE_CELLS
} from './workflow-table-source-operation';
import {makeWorkflowTableSourceOperation} from './workflow-table-source-operation';
import type {CatalogParameterValue} from '../analysis/catalog-binding-types';
import {applyKeplerDatasetProfileDeclarations} from '../profile/dataset-profile';
import {parseKeplerDatasetProfileDeclarations} from '../profile/dataset-profile-cache';
import {profileKeplerWorkbenchData} from '../descriptors/workbench-adapters';

import {
  parseCatalogFeatureCollection,
  type CatalogFeatureCollection
} from '../analysis/catalog-geojson';
import {
  getWorkflowFeaturePropertyColumns,
  hashWorkflowValue,
  makeWorkflowDataFromFeatures
} from './workflow-data';
import type {KeplerDatasetProfileDeclarations} from '../profile/dataset-profile-types';
import {normalizeWorkflowParameters} from './workflow-parameters';
import {
  WORKFLOW_LIMITS,
  WorkflowError,
  type WorkflowData,
  type WorkflowDatasetRequest,
  type WorkflowHost,
  type WorkflowOperation,
  type WorkflowParameterSpec
} from './workflow-types';

export const WORKFLOW_SOURCE_DATASET_OPERATION_ID = 'source-dataset';
export const WORKFLOW_SOURCE_GEOJSON_OPERATION_ID = 'source-geojson';
export const WORKFLOW_SOURCE_DRAWING_OPERATION_ID = 'source-drawing';

/** Planning and execution inspect the same table schema and declaration column names. */
export function makeWorkflowSourceGeoJSONData(
  collection: CatalogFeatureCollection,
  options: {
    id: string;
    label: string;
    declarations?: KeplerDatasetProfileDeclarations;
    coordinateSystem?: string;
  }
): WorkflowData {
  const data = makeWorkflowDataFromFeatures(collection, {id: options.id, label: options.label});
  const propertyColumns = getWorkflowFeaturePropertyColumns(collection);
  data.analysisProfile = applyKeplerDatasetProfileDeclarations(profileKeplerWorkbenchData(data), {
    ...options.declarations,
    ...(options.declarations?.measurement
      ? {
          measurement: Object.fromEntries(
            Object.entries(options.declarations.measurement).map(([name, measurement]) => [
              propertyColumns.get(name) ?? name,
              measurement
            ])
          )
        }
      : {}),
    ...(options.coordinateSystem === 'planar' || options.coordinateSystem === 'EPSG:4326'
      ? {coordinateModel: options.coordinateSystem === 'planar' ? 'planar' : 'geographic'}
      : {})
  });
  return data;
}

const MAXIMUM_DRAWING_ROWS = 1000;

/** Source-local limits protect recursive JSON serialization and dense table allocation. */
export const WORKFLOW_SOURCE_GEOJSON_LIMITS = Object.freeze({
  /** Objects and arrays in the input text; the root container has depth one. */
  maximumNestingDepth: 64,
  /** Includes geometry and the optional feature identity column. */
  maximumColumns: 256,
  /** Row count multiplied by the complete output column count, including null cells. */
  maximumTableCells: 1000000
});

function freezeParameterSpecs(specs: WorkflowParameterSpec[]): readonly WorkflowParameterSpec[] {
  for (const spec of specs) {
    if (Array.isArray(spec.defaultValue)) Object.freeze(spec.defaultValue);
    if (spec.options) Object.freeze(spec.options);
    if (spec.visibleWhen) {
      for (const value of spec.visibleWhen.equals) {
        if (Array.isArray(value)) Object.freeze(value);
      }
      Object.freeze(spec.visibleWhen.equals);
      Object.freeze(spec.visibleWhen);
    }
    Object.freeze(spec);
  }
  return Object.freeze(specs);
}

function copyParameterSpecs(specs: readonly WorkflowParameterSpec[]): WorkflowParameterSpec[] {
  return specs.map(spec => ({
    ...spec,
    defaultValue: Array.isArray(spec.defaultValue) ? [...spec.defaultValue] : spec.defaultValue,
    ...(spec.options ? {options: [...spec.options]} : {}),
    ...(spec.visibleWhen
      ? {
          visibleWhen: {
            parameter: spec.visibleWhen.parameter,
            equals: spec.visibleWhen.equals.map(value =>
              Array.isArray(value) ? [...value] : value
            )
          }
        }
      : {})
  }));
}

const DATASET_PARAMETERS = freezeParameterSpecs([
  {name: 'dataset', kind: 'dataset', label: 'Dataset', defaultValue: ''},
  {
    name: 'datasetLabel',
    kind: 'string',
    label: 'Dataset label',
    maximumLength: 120,
    defaultValue: '',
    visibleWhen: {parameter: 'dataset', equals: []}
  },
  {
    name: 'geometryMode',
    kind: 'enum',
    label: 'Geometry',
    options: ['geojson-field', 'point-columns', 'none'],
    defaultValue: 'geojson-field'
  },
  {
    name: 'geometryField',
    kind: 'column',
    label: 'Geometry field',
    columnsFrom: 'dataset',
    defaultValue: '',
    visibleWhen: {parameter: 'geometryMode', equals: ['geojson-field']}
  },
  {
    name: 'longitudeField',
    kind: 'column',
    label: 'Longitude field',
    columnsFrom: 'dataset',
    defaultValue: '',
    visibleWhen: {parameter: 'geometryMode', equals: ['point-columns']}
  },
  {
    name: 'latitudeField',
    kind: 'column',
    label: 'Latitude field',
    columnsFrom: 'dataset',
    defaultValue: '',
    visibleWhen: {parameter: 'geometryMode', equals: ['point-columns']}
  },
  {
    name: 'population',
    kind: 'enum',
    label: 'Rows',
    options: ['all', 'selected'],
    defaultValue: 'all'
  },
  {
    name: 'maximumRows',
    kind: 'integer',
    label: 'Maximum rows',
    minimum: 1,
    maximum: WORKFLOW_LIMITS.maximumSourceRows,
    defaultValue: 20000
  }
]);

const GEOJSON_PARAMETERS = freezeParameterSpecs([
  {
    name: 'text',
    kind: 'text',
    label: 'GeoJSON',
    maximumLength: WORKFLOW_LIMITS.maximumTextCharacters,
    defaultValue: ''
  },
  {name: 'name', kind: 'string', label: 'Name', defaultValue: ''},
  {
    name: 'coordinateSystem',
    kind: 'enum',
    label: 'Source coordinates',
    options: ['auto', 'EPSG:4326', 'planar'],
    defaultValue: 'auto',
    description:
      'Auto keeps inferred coordinates unconfirmed. Choose a CRS to confirm the convention of this source.'
  },
  {
    name: 'profileDeclarations',
    kind: 'text',
    label: 'Dataset declarations',
    maximumLength: WORKFLOW_LIMITS.maximumTextCharacters,
    defaultValue: '',
    level: 'advanced',
    description: 'Saved coordinate, row meaning and column measurement declarations.'
  }
]);

const DRAWING_PARAMETERS = freezeParameterSpecs([
  {
    name: 'shapes',
    kind: 'enum',
    label: 'Shapes',
    options: ['all', 'chosen'],
    defaultValue: 'all'
  },
  {
    name: 'drawingIds',
    kind: 'string',
    label: 'Shape ids',
    maximumLength: 2000,
    defaultValue: '',
    visibleWhen: {parameter: 'shapes', equals: ['chosen']}
  }
]);

function parseDrawingIds(value: string): string[] {
  return [
    ...new Set(
      value
        .split(',')
        .map(identity => identity.trim())
        .filter(Boolean)
    )
  ];
}

function normalizeDatasetParameters(
  raw: Readonly<Record<string, unknown>>
): Record<string, CatalogParameterValue> {
  const parameters = normalizeWorkflowParameters(DATASET_PARAMETERS, raw, {
    operationTitle: 'Dataset'
  });
  const requiredParameters = [
    ['dataset', 'Dataset'],
    ...(parameters['geometryMode'] === 'geojson-field'
      ? [['geometryField', 'Geometry field']]
      : []),
    ...(parameters['geometryMode'] === 'point-columns'
      ? [
          ['longitudeField', 'Longitude field'],
          ['latitudeField', 'Latitude field']
        ]
      : [])
  ];
  for (const [parameter, label] of requiredParameters) {
    if (!(parameters[parameter] as string).trim()) {
      throw new WorkflowError('parameter-out-of-range', `Dataset: ${label} must be chosen.`, {
        parameter
      });
    }
  }
  if (
    parameters['geometryMode'] === 'point-columns' &&
    parameters['longitudeField'] === parameters['latitudeField']
  ) {
    throw new WorkflowError(
      'parameter-out-of-range',
      'Dataset: Latitude field must differ from Longitude field.',
      {parameter: 'latitudeField'}
    );
  }
  return parameters;
}

/** Request a `source-dataset` step makes of the host, from its normalised parameters. */
export function makeWorkflowDatasetRequest(
  parameters: Readonly<Record<string, CatalogParameterValue>>
): WorkflowDatasetRequest {
  const mode = parameters['geometryMode'];
  const datasetLabel = parameters['datasetLabel'] as string;
  return {
    datasetId: parameters['dataset'] as string,
    ...(datasetLabel ? {datasetLabel} : {}),
    geometry:
      mode === 'geojson-field'
        ? {mode, field: parameters['geometryField'] as string}
        : mode === 'point-columns'
          ? {
              mode,
              longitude: parameters['longitudeField'] as string,
              latitude: parameters['latitudeField'] as string
            }
          : {mode: 'none'},
    population: parameters['population'] as WorkflowDatasetRequest['population'],
    maximumRows: parameters['maximumRows'] as number
  };
}

function resolveDatasetRequest(
  host: WorkflowHost,
  parameters: Readonly<Record<string, CatalogParameterValue>>
): WorkflowDatasetRequest {
  const request = makeWorkflowDatasetRequest(parameters);
  const datasets = host.listDatasets();
  if (datasets.some(dataset => dataset.id === request.datasetId)) return request;
  const matches = request.datasetLabel
    ? datasets.filter(dataset => dataset.label === request.datasetLabel)
    : [];
  if (matches.length === 1) return {...request, datasetId: matches[0].id};
  const reference = request.datasetLabel || request.datasetId;
  throw new WorkflowError(
    'dataset-not-found',
    matches.length > 1
      ? `Dataset: Dataset ${JSON.stringify(reference)} must match exactly one dataset; found ${matches.length}.`
      : `Dataset: Dataset ${JSON.stringify(reference)} was not found.`,
    {parameter: 'dataset'}
  );
}

function checkCancellation(signal: AbortSignal, title: string): void {
  if (signal.aborted) throw new WorkflowError('cancelled', `${title}: Reading was cancelled.`);
}

function getErrorMessage(cause: unknown): string {
  try {
    const message = cause instanceof Error ? cause.message : String(cause);
    return typeof message === 'string' && message ? message : 'Unknown error';
  } catch {
    // A host may reject with an object that has no primitive representation.
    return 'Unknown error';
  }
}

function isWorkflowError(cause: unknown): cause is WorkflowError {
  try {
    return cause instanceof WorkflowError;
  } catch {
    // Even inspecting the prototype of a rejected proxy can fail; retain it as the cause.
    return false;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isPosition(value: unknown): value is number[] {
  return (
    Array.isArray(value) &&
    value.length >= 2 &&
    value.every(coordinate => typeof coordinate === 'number' && Number.isFinite(coordinate))
  );
}

function isLine(value: unknown): boolean {
  return Array.isArray(value) && value.length >= 2 && value.every(isPosition);
}

function isRing(value: unknown): boolean {
  if (!Array.isArray(value) || value.length < 4 || !value.every(isPosition)) return false;
  const first: number[] = value[0];
  const last: number[] = value[value.length - 1];
  return (
    first.length === last.length && first.every((coordinate, index) => coordinate === last[index])
  );
}

function isPolygon(value: unknown): boolean {
  return Array.isArray(value) && value.every(isRing);
}

function isGeometry(value: unknown): boolean {
  if (!isRecord(value)) return false;
  const coordinates = value['coordinates'];
  // The text nesting gate bounds recursion through GeometryCollections before this check.
  switch (value['type']) {
    case 'Point':
      return isPosition(coordinates);
    case 'MultiPoint':
      return Array.isArray(coordinates) && coordinates.every(isPosition);
    case 'LineString':
      return isLine(coordinates);
    case 'MultiLineString':
      return Array.isArray(coordinates) && coordinates.every(isLine);
    case 'Polygon':
      return isPolygon(coordinates);
    case 'MultiPolygon':
      return Array.isArray(coordinates) && coordinates.every(isPolygon);
    case 'GeometryCollection':
      return Array.isArray(value['geometries']) && value['geometries'].every(isGeometry);
    default:
      return false;
  }
}

function validateGeoJSONNesting(text: string): void {
  let depth = 0;
  let quoted = false;
  let escaped = false;
  for (const character of text) {
    if (quoted) {
      if (escaped) escaped = false;
      else if (character === '\\') escaped = true;
      else if (character === '"') quoted = false;
    } else if (character === '"') quoted = true;
    else if (character === '{' || character === '[') {
      depth++;
      if (depth > WORKFLOW_SOURCE_GEOJSON_LIMITS.maximumNestingDepth) {
        throw new WorkflowError(
          'parameter-out-of-range',
          `GeoJSON: GeoJSON nesting depth must be at most ${WORKFLOW_SOURCE_GEOJSON_LIMITS.maximumNestingDepth}; got ${depth}.`,
          {parameter: 'text'}
        );
      }
    } else if (character === '}' || character === ']') depth--;
  }
}

function makeInvalidGeoJSONError(cause: unknown): WorkflowError {
  const reason = getErrorMessage(cause).replace(/[.\s]+$/, '');
  return new WorkflowError('parameter-out-of-range', `GeoJSON: GeoJSON is not valid: ${reason}.`, {
    parameter: 'text',
    cause
  });
}

function parseSourceGeoJSON(text: string): CatalogFeatureCollection {
  validateGeoJSONNesting(text);
  try {
    const collection = parseCatalogFeatureCollection(text);
    // The catalog parser accepts any string root type as a geometry; sources require GeoJSON.
    for (const feature of collection.features) {
      if (
        !isRecord(feature) ||
        (feature['geometry'] !== null && !isGeometry(feature['geometry']))
      ) {
        throw new Error(
          'expected GeoJSON geometry with finite numeric coordinates and valid structure'
        );
      }
      const identity: unknown = feature.id;
      if (
        identity !== undefined &&
        identity !== null &&
        typeof identity !== 'string' &&
        !(typeof identity === 'number' && Number.isFinite(identity))
      ) {
        throw new Error('Feature id must be a string, a finite number or null');
      }
      if (
        feature.properties !== undefined &&
        feature.properties !== null &&
        !isRecord(feature.properties)
      ) {
        throw new Error('Feature properties must be an object or null');
      }
    }
    return collection;
  } catch (cause) {
    throw makeInvalidGeoJSONError(cause);
  }
}

function validateGeoJSONTableCapacity(collection: CatalogFeatureCollection): void {
  const propertyNames = new Set<string>();
  let hasIdentity = false;
  for (const feature of collection.features) {
    for (const name of Object.keys(feature.properties ?? {})) propertyNames.add(name);
    if (feature.id !== undefined && feature.id !== null) hasIdentity = true;
  }
  // The converter renames reserved property names rather than dropping them.
  const columnCount = propertyNames.size + 1 + Number(hasIdentity);
  if (columnCount > WORKFLOW_SOURCE_GEOJSON_LIMITS.maximumColumns) {
    throw new WorkflowError(
      'capacity-exceeded',
      `GeoJSON: GeoJSON table must contain at most ${WORKFLOW_SOURCE_GEOJSON_LIMITS.maximumColumns} columns; got ${columnCount}.`,
      {parameter: 'text'}
    );
  }
  const cellCount = collection.features.length * columnCount;
  if (cellCount > WORKFLOW_SOURCE_GEOJSON_LIMITS.maximumTableCells) {
    throw new WorkflowError(
      'capacity-exceeded',
      `GeoJSON: GeoJSON table must contain at most ${WORKFLOW_SOURCE_GEOJSON_LIMITS.maximumTableCells} cells; got ${cellCount}.`,
      {parameter: 'text'}
    );
  }
}

/** Host datasets, embedded GeoJSON text and shapes drawn on the host's map. */
export function getWorkflowSourceOperations(): readonly WorkflowOperation[] {
  const makeCommon = () => ({
    category: 'source' as const,
    origin: 'builtin' as const,
    engine: 'host' as const,
    version: '1',
    inputs: [],
    outputs: [
      {
        name: 'data',
        label: 'Data',
        kind: 'features' as const,
        identityColumns: [],
        maximumRows: WORKFLOW_LIMITS.maximumSourceRows
      }
    ],
    capabilityIds: [],
    contributorNames: []
  });
  return [
    {
      ...makeCommon(),
      id: WORKFLOW_SOURCE_DATASET_OPERATION_ID,
      title: 'Dataset',
      summary: 'Read all rows or the current selection from a host dataset.',
      parameters: copyParameterSpecs(DATASET_PARAMETERS),
      declaredBehavior: [
        'Reads the saved dataset id, or a unique matching label after workspace replay.',
        'Uses a GeoJSON field, longitude and latitude columns, or no geometry.',
        'Selected rows follow the current host filters and selection.'
      ],
      limitation: 'Reading is bounded by Maximum rows and requires the dataset in the host.',
      normalizeParameters: normalizeDatasetParameters,
      getExternalRevisionKey: (host, parameters) => {
        try {
          const request = resolveDatasetRequest(host, parameters);
          const revision = host.getDatasetRevisionKey(request);
          const profile = host.getAnalysisProfile?.(request.datasetId);
          return profile ? hashWorkflowValue([revision, profile]) : revision;
        } catch {
          // Planning must remain synchronous and non-throwing when a host dataset is unavailable.
          return 'missing';
        }
      },
      run: async (context, _inputs, parameters) => {
        checkCancellation(context.signal, 'Dataset');
        let request: WorkflowDatasetRequest;
        let data: WorkflowData;
        try {
          request = resolveDatasetRequest(context.host, parameters);
          data = await context.host.readDataset(request, context.signal);
        } catch (cause) {
          checkCancellation(context.signal, 'Dataset');
          if (isWorkflowError(cause)) throw cause;
          throw new WorkflowError('dataset-read-failed', `Dataset: ${getErrorMessage(cause)}`, {
            cause
          });
        }
        checkCancellation(context.signal, 'Dataset');
        if (data.table.rowCount > request.maximumRows) {
          throw new WorkflowError(
            'capacity-exceeded',
            `Dataset: Maximum rows is ${request.maximumRows}; got ${data.table.rowCount} rows.`,
            {parameter: 'maximumRows'}
          );
        }
        return {
          outputs: {data},
          engine: 'host',
          summary: {rows: data.table.rowCount, kind: data.kind},
          diagnostics:
            request.population === 'selected'
              ? [
                  {
                    code: 'selected-population',
                    severity: 'info',
                    count: data.table.rowCount,
                    message: 'Dataset: Rows follow the current host filters and selection.'
                  }
                ]
              : [],
          dispatchedContributors: []
        };
      }
    },
    {
      ...makeCommon(),
      id: WORKFLOW_SOURCE_GEOJSON_OPERATION_ID,
      title: 'GeoJSON',
      summary: 'Read embedded GeoJSON as a workflow table with geometry.',
      parameters: copyParameterSpecs(GEOJSON_PARAMETERS),
      declaredBehavior: [
        'Accepts a FeatureCollection, a single Feature or a bare GeoJSON geometry.',
        'Keeps properties and feature identity in a table with a GeoJSON geometry column.'
      ],
      limitation: `GeoJSON must contain at least one feature and at most ${WORKFLOW_LIMITS.maximumSourceRows} features within ${WORKFLOW_LIMITS.maximumTextCharacters} text characters. Input objects and arrays may nest at most ${WORKFLOW_SOURCE_GEOJSON_LIMITS.maximumNestingDepth} levels (root depth one). The output table is limited to ${WORKFLOW_SOURCE_GEOJSON_LIMITS.maximumColumns} columns and ${WORKFLOW_SOURCE_GEOJSON_LIMITS.maximumTableCells} cells, including geometry, identity and null cells.`,
      normalizeParameters: raw => {
        const parameters = normalizeWorkflowParameters(GEOJSON_PARAMETERS, raw, {
          operationTitle: 'GeoJSON'
        });
        if (!(parameters['text'] as string).trim()) {
          throw new WorkflowError('parameter-out-of-range', 'GeoJSON: GeoJSON must not be blank.', {
            parameter: 'text'
          });
        }
        if (parameters['profileDeclarations']) {
          try {
            parseKeplerDatasetProfileDeclarations(parameters['profileDeclarations'] as string);
          } catch (cause) {
            throw new WorkflowError(
              'parameter-out-of-range',
              `GeoJSON: profileDeclarations must contain valid dataset declarations: ${getErrorMessage(cause)}`,
              {parameter: 'profileDeclarations', cause}
            );
          }
        }
        return parameters;
      },
      run: async (context, _inputs, parameters) => {
        checkCancellation(context.signal, 'GeoJSON');
        const collection = parseSourceGeoJSON(parameters['text'] as string);
        if (collection.features.length === 0) {
          throw new WorkflowError(
            'parameter-out-of-range',
            'GeoJSON: GeoJSON must contain at least one feature; got 0.',
            {parameter: 'text'}
          );
        }
        if (collection.features.length > WORKFLOW_LIMITS.maximumSourceRows) {
          throw new WorkflowError(
            'capacity-exceeded',
            `GeoJSON: GeoJSON must contain at most ${WORKFLOW_LIMITS.maximumSourceRows} features; got ${collection.features.length}.`,
            {parameter: 'text'}
          );
        }
        validateGeoJSONTableCapacity(collection);
        let data: WorkflowData;
        try {
          data = makeWorkflowSourceGeoJSONData(collection, {
            id: context.stepId,
            label: (parameters['name'] as string) || 'GeoJSON',
            coordinateSystem: parameters['coordinateSystem'] as string,
            declarations: parameters['profileDeclarations']
              ? parseKeplerDatasetProfileDeclarations(parameters['profileDeclarations'] as string)
              : {}
          });
        } catch (cause) {
          throw makeInvalidGeoJSONError(cause);
        }
        checkCancellation(context.signal, 'GeoJSON');
        return {
          outputs: {data},
          engine: 'host',
          summary: {rows: data.table.rowCount, kind: data.kind},
          diagnostics: [],
          dispatchedContributors: []
        };
      }
    },
    {
      ...makeCommon(),
      id: WORKFLOW_SOURCE_DRAWING_OPERATION_ID,
      title: 'Drawn shapes',
      summary: 'Read all or chosen shapes drawn on the host map.',
      parameters: copyParameterSpecs(DRAWING_PARAMETERS),
      outputs: [
        {
          name: 'data',
          label: 'Data',
          kind: 'features',
          identityColumns: ['drawing_id'],
          maximumRows: MAXIMUM_DRAWING_ROWS
        }
      ],
      declaredBehavior: [
        'Reads polygons, rectangles, lines and points from the host map editor.',
        'Chosen shapes are identified by their comma-separated ids.',
        'Preserves drawing_id identity and the label column supplied by the host.'
      ],
      limitation: `Requires host drawing support and at least one drawn shape, with at most ${MAXIMUM_DRAWING_ROWS} rows.`,
      normalizeParameters: raw => {
        const parameters = normalizeWorkflowParameters(DRAWING_PARAMETERS, raw, {
          operationTitle: 'Drawn shapes'
        });
        if (
          parameters['shapes'] === 'chosen' &&
          parseDrawingIds(parameters['drawingIds'] as string).length === 0
        ) {
          throw new WorkflowError(
            'parameter-out-of-range',
            'Drawn shapes: Shape ids must contain at least one id when Shapes is chosen.',
            {parameter: 'drawingIds'}
          );
        }
        return parameters;
      },
      getExternalRevisionKey: host => {
        try {
          return host.getDrawingRevisionKey?.() ?? 'unsupported';
        } catch {
          // Planning must remain non-throwing when host drawing state is unavailable.
          return 'unsupported';
        }
      },
      run: async (context, _inputs, parameters) => {
        checkCancellation(context.signal, 'Drawn shapes');
        if (!context.host.readDrawings) {
          throw new WorkflowError(
            'host-capability-missing',
            'Drawn shapes: this application cannot read drawn shapes.'
          );
        }
        const drawingIds =
          parameters['shapes'] === 'chosen'
            ? parseDrawingIds(parameters['drawingIds'] as string)
            : 'all';
        let data: WorkflowData;
        try {
          if (drawingIds !== 'all') {
            if (!context.host.listDrawings) {
              throw new WorkflowError(
                'host-capability-missing',
                'Drawn shapes: this application cannot list drawn shapes.'
              );
            }
            const availableIds = new Set(context.host.listDrawings().map(drawing => drawing.id));
            for (const identity of drawingIds) {
              if (!availableIds.has(identity)) {
                throw new WorkflowError(
                  'dataset-not-found',
                  `Drawn shapes: Shape id ${JSON.stringify(identity)} was not found.`,
                  {parameter: 'drawingIds'}
                );
              }
            }
          }
          data = await context.host.readDrawings(drawingIds, context.signal);
        } catch (cause) {
          checkCancellation(context.signal, 'Drawn shapes');
          if (isWorkflowError(cause)) throw cause;
          throw new WorkflowError(
            'dataset-read-failed',
            `Drawn shapes: ${getErrorMessage(cause)}`,
            {
              cause
            }
          );
        }
        checkCancellation(context.signal, 'Drawn shapes');
        if (data.table.rowCount === 0) {
          throw new WorkflowError(
            'input-required',
            'Drawn shapes: draw a shape on the map first (polygon, rectangle or line tool).'
          );
        }
        if (data.table.rowCount > MAXIMUM_DRAWING_ROWS) {
          throw new WorkflowError(
            'capacity-exceeded',
            `Drawn shapes: Data must contain at most ${MAXIMUM_DRAWING_ROWS} rows; got ${data.table.rowCount}.`,
            {port: 'data'}
          );
        }
        return {
          outputs: {data},
          engine: 'host',
          summary: {rows: data.table.rowCount, kind: data.kind},
          diagnostics: [],
          dispatchedContributors: []
        };
      }
    },
    makeWorkflowTableSourceOperation()
  ];
}
