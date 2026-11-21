// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getCatalogValidationSampleInputs,
  getCatalogValidationSampleDeclarations
} from './analysis-catalog-validation';
import {
  findPointCloudCatalogFields,
  readPointCloudCatalogDataset,
  getPointCloudCatalogSamples
} from './analysis-catalog-pointcloud';
import {
  createKeplerCuratedAnalysisSample,
  parseCatalogFeatureCollection,
  makeCatalogTableInput,
  parseCatalogTableInput,
  type CatalogTableInput,
  type CatalogOutputTable,
  type CatalogFeature,
  type CatalogFeatureCollection,
  type KeplerDatasetProfileDeclarations
} from '@luma-contributors/kepler';
import * as arrow from 'apache-arrow';
import {
  getNetworkCategoricalSampleInputs,
  getNetworkCategoricalSampleDeclarations
} from './analysis-catalog-network-categorical-samples';

import {getSpacetimeCatalogSample} from './analysis-catalog-spacetime';
import {
  getCatalogTrajectorySampleInputs,
  normalizeCatalogTrajectoryProperties
} from './analysis-catalog-trajectory';
import {validateCapacity, validateRequirement} from '@luma-contributors/analysis/analysis-core';

export const MAXIMUM_CATALOG_NATIVE_ROWS = 20000;

type NativeDatasetLike = {
  fields: readonly {name: string}[];
  dataContainer: {numRows(): number; valueAt(rowIndex: number, columnIndex: number): unknown};
};

function readGeometry(value: unknown): CatalogFeature['geometry'] {
  const parsed =
    typeof value === 'string' && value.trim().startsWith('{') ? JSON.parse(value) : value;
  if (!parsed || typeof parsed !== 'object') return null;
  const candidate = parsed as {type?: string; geometry?: CatalogFeature['geometry']};
  if (candidate.type === 'Feature') return candidate.geometry ?? null;
  return typeof candidate.type === 'string' ? (candidate as CatalogFeature['geometry']) : null;
}

/** Shared cell normalization for catalog execution and dataset profiling. */
export {readGeometry as readCatalogGeometry};

export const CATALOG_POINT_COLUMNS_PREFIX = '@point:';

export function readCatalogPointColumns(field: string): [string, string] | null {
  if (!field.startsWith(CATALOG_POINT_COLUMNS_PREFIX)) return null;
  const fields: unknown = JSON.parse(field.slice(CATALOG_POINT_COLUMNS_PREFIX.length));
  if (
    !Array.isArray(fields) ||
    fields.length !== 2 ||
    !fields.every(value => typeof value === 'string')
  )
    throw new Error('Invalid point column mapping');
  return fields as [string, string];
}

export function makeCatalogPointColumns(horizontal: string, vertical: string): string {
  return CATALOG_POINT_COLUMNS_PREFIX + JSON.stringify([horizontal, vertical]);
}

/** Candidate geometry fields: those whose first non-null value reads as GeoJSON. */
export function findNativeGeometryFields(dataset: NativeDatasetLike): string[] {
  const rowCount = dataset.dataContainer.numRows();
  const names: string[] = [...findPointCloudCatalogFields(dataset)];
  dataset.fields.forEach((field, columnIndex) => {
    for (let rowIndex = 0; rowIndex < Math.min(rowCount, 16); rowIndex++) {
      try {
        const value = dataset.dataContainer.valueAt(rowIndex, columnIndex);
        if (value === null || value === undefined) continue;
        if (readGeometry(value)) names.push(field.name);
      } catch {
        // Unreadable or not GeoJSON; discovery must not interrupt catalog rendering.
      }
      return;
    }
  });
  return names;
}

type NativeProperty = string | number | boolean | null | NativeProperty[];

/** Iterate Arrow vectors instead of toArray(), which can erase nulls in numeric child vectors. */
function readNativeProperty(value: unknown, depth = 0): NativeProperty | undefined {
  if (
    value === null ||
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  )
    return value;
  if (typeof value === 'bigint') {
    const numericValue = Number(value);
    return Number.isSafeInteger(numericValue) ? numericValue : String(value);
  }
  if (
    Array.isArray(value) ||
    value instanceof arrow.Vector ||
    (ArrayBuffer.isView(value) && !(value instanceof DataView))
  ) {
    // Bounds nesting and also rejects cyclic JS arrays from a native data container.
    if (depth >= 8)
      throw new Error('Native catalog list properties exceed maximum nesting depth 8');
    return Array.from(value as Iterable<unknown>, item => {
      const property = readNativeProperty(item, depth + 1);
      if (property === undefined)
        throw new Error('Native catalog list properties must contain scalar or list values');
      return property;
    });
  }
  return undefined;
}

/** Geometry-free scalar rows retain their native row order and numeric precision. */
export function readCatalogTableFromNativeDataset(
  dataset: NativeDatasetLike,
  maximumRows = MAXIMUM_CATALOG_NATIVE_ROWS
): CatalogTableInput {
  const rowCount = dataset.dataContainer.numRows();
  validateCapacity(
    'catalog-native-table',
    'rowCount',
    rowCount,
    0,
    Math.min(maximumRows, MAXIMUM_CATALOG_NATIVE_ROWS)
  );
  const table: CatalogOutputTable = {
    id: 'catalog-native-table',
    label: 'Native table',
    rowCount,
    columns: {},
    columnTypes: {}
  };
  dataset.fields.forEach((field, columnIndex) => {
    const values = Array.from({length: rowCount}, (_, rowIndex) => {
      const value = readNativeProperty(dataset.dataContainer.valueAt(rowIndex, columnIndex));
      validateRequirement(
        'catalog-native-table',
        value !== undefined && !Array.isArray(value),
        `${field.name} must contain scalar values; received ${value === undefined ? 'an unsupported value' : 'a list'}; select scalar columns instead`
      );
      return value;
    });
    table.columns[field.name] = values;
    const present = values.filter(value => value !== null);
    table.columnTypes[field.name] = present.every(value => typeof value === 'number')
      ? 'float64'
      : present.every(value => typeof value === 'boolean')
        ? 'boolean'
        : 'string';
  });
  return makeCatalogTableInput(table, {maximumRows});
}

/** Explicit, capacity-bounded GeoJSON read; scalar and list columns become plain JSON properties. */
export function readCatalogFeaturesFromNativeDataset(
  dataset: NativeDatasetLike,
  geometryField: string,
  normalizeProperties?: (properties: Record<string, unknown>) => Record<string, unknown>
): CatalogFeatureCollection {
  const pointCloud = readPointCloudCatalogDataset(dataset, geometryField);
  if (pointCloud) return pointCloud;
  const rowCount = dataset.dataContainer.numRows();
  // Row capacity exceeded; filter or sample the dataset before analysis.
  if (rowCount > MAXIMUM_CATALOG_NATIVE_ROWS)
    throw new Error('Native dataset exceeds catalog capacity');
  const pointColumns = readCatalogPointColumns(geometryField);
  const horizontalIndex = pointColumns
    ? dataset.fields.findIndex(field => field.name === pointColumns[0])
    : -1;
  const verticalIndex = pointColumns
    ? dataset.fields.findIndex(field => field.name === pointColumns[1])
    : -1;
  if (
    pointColumns &&
    (!pointColumns.every(column => column.trim().length > 0) ||
      horizontalIndex < 0 ||
      verticalIndex < 0)
  )
    throw new Error('Choose both numeric point columns');
  const geometryIndex = dataset.fields.findIndex(field => field.name === geometryField);
  if (!pointColumns && geometryIndex < 0) throw new Error('Geometry field not found');
  const features: CatalogFeature[] = [];
  for (let rowIndex = 0; rowIndex < rowCount; rowIndex++) {
    const properties: Record<string, unknown> = {};
    dataset.fields.forEach((field, columnIndex) => {
      if (columnIndex === geometryIndex) return;
      const raw = dataset.dataContainer.valueAt(rowIndex, columnIndex);
      // Trajectory normalization needs the raw cell (Arrow List time arrays); others read plain JSON.
      const value = normalizeProperties ? raw : readNativeProperty(raw);
      if (value !== undefined) properties[field.name] = value;
    });
    features.push({
      id: rowIndex,
      geometry: pointColumns
        ? readPointGeometry(
            dataset.dataContainer.valueAt(rowIndex, horizontalIndex),
            dataset.dataContainer.valueAt(rowIndex, verticalIndex)
          )
        : readGeometry(dataset.dataContainer.valueAt(rowIndex, geometryIndex)),
      properties: normalizeProperties ? normalizeProperties(properties) : properties
    });
  }
  return {type: 'FeatureCollection', features};
}

function readPointGeometry(horizontal: unknown, vertical: unknown): CatalogFeature['geometry'] {
  if (
    typeof horizontal !== 'number' ||
    typeof vertical !== 'number' ||
    !Number.isFinite(horizontal) ||
    !Number.isFinite(vertical)
  )
    throw new Error('Point columns must contain finite numbers');
  return {type: 'Point', coordinates: [horizontal, vertical]};
}

/** Explicit trajectory ingestion preserves the selected time array, including Arrow List cells. */
export function readCatalogTrajectoryFeaturesFromNativeDataset(
  dataset: NativeDatasetLike,
  geometryField: string,
  parameters: Readonly<Record<string, unknown>> = {}
): CatalogFeatureCollection {
  validateCapacity(
    'catalog-trajectory-native',
    'rowCount',
    dataset.dataContainer.numRows(),
    0,
    2048
  );
  return readCatalogFeaturesFromNativeDataset(dataset, geometryField, properties =>
    normalizeCatalogTrajectoryProperties(properties, parameters)
  );
}

function lines(
  rows: {id: string; coordinates: number[][]; properties?: Record<string, unknown>}[]
): string {
  return JSON.stringify({
    type: 'FeatureCollection',
    features: rows.map(row => ({
      type: 'Feature',
      id: row.id,
      properties: row.properties ?? {},
      geometry: {type: 'LineString', coordinates: row.coordinates}
    }))
  });
}

function points(coordinates: number[][]): string {
  return JSON.stringify({
    type: 'FeatureCollection',
    features: coordinates.map(position => ({
      type: 'Feature',
      properties: {},
      geometry: {type: 'Point', coordinates: position}
    }))
  });
}

function polygons(rings: number[][][][]): string {
  return JSON.stringify({
    type: 'FeatureCollection',
    features: rings.map((polygon, index) => ({
      type: 'Feature',
      id: `polygon-${index}`,
      properties: {},
      geometry: {type: 'Polygon', coordinates: polygon}
    }))
  });
}

const SQUARE = (x: number, y: number, size: number) => [
  [x, y],
  [x + size, y],
  [x + size, y + size],
  [x, y + size],
  [x, y]
];

// City-scale samples around one longitude/latitude for the GPU operations (degrees, EPSG:4326).
const CITY = (east: number, north: number): number[] => [-87.65 + east, 41.88 + north];
const CITY_SQUARE = (east: number, north: number, size: number) => [
  CITY(east, north),
  CITY(east + size, north),
  CITY(east + size, north + size),
  CITY(east, north + size),
  CITY(east, north)
];

// Three by three street grid whose lines share vertices at every junction.
const CITY_GRID_NETWORK = lines(
  [0, 0.005, 0.01].flatMap((offset, index) => [
    {
      id: `row-${index}`,
      coordinates: [CITY(0, offset), CITY(0.005, offset), CITY(0.01, offset)]
    },
    {
      id: `column-${index}`,
      coordinates: [CITY(offset, 0), CITY(offset, 0.005), CITY(offset, 0.01)]
    }
  ])
);

// Small planar network (coordinate units): a one-way loop, a spur only enterable from its tip,
// a one-way feeder and a disconnected island, in meters-sized units.
const SAMPLE_NETWORK = lines([
  {
    id: 'ab',
    coordinates: [
      [0, 0],
      [10, 0]
    ]
  },
  {
    id: 'bc',
    coordinates: [
      [10, 0],
      [30, 0]
    ]
  },
  {
    id: 'cd',
    coordinates: [
      [30, 0],
      [30, 10]
    ],
    properties: {oneway: 'yes'}
  },
  {
    id: 'de',
    coordinates: [
      [30, 10],
      [0, 5]
    ],
    properties: {oneway: 'yes'}
  },
  {
    id: 'ea',
    coordinates: [
      [0, 5],
      [0, 0]
    ]
  },
  {
    id: 'island',
    coordinates: [
      [100, 100],
      [110, 100]
    ]
  },
  {
    id: 'spur',
    coordinates: [
      [30, 10],
      [30, 20]
    ],
    properties: {oneway: '-1'}
  },
  {
    id: 'ga',
    coordinates: [
      [-10, 0],
      [0, 0]
    ],
    properties: {oneway: 'yes'}
  }
]);

/** Author-known fixture facts also cover bindings with no coordinate-system parameter. */
export function getCatalogSampleInputDeclarations(
  operationId: string,
  inputName: string
): KeplerDatasetProfileDeclarations | undefined {
  if (operationId === 'geometry-convex-clip') return {coordinateModel: 'planar'};
  return (
    getNetworkCategoricalSampleDeclarations(operationId, inputName) ??
    getCatalogValidationSampleDeclarations(operationId)
  );
}

/** Sample GeoJSON text per operation input name, small and planar-friendly. */
export function getCatalogSampleInputs(operationId: string): Record<string, string> {
  const spacetime = getSpacetimeCatalogSample(operationId);
  if (spacetime) return {points: spacetime};
  const networkCategorical = getNetworkCategoricalSampleInputs(operationId);
  if (networkCategorical) return networkCategorical;
  const trajectory = getCatalogTrajectorySampleInputs(operationId);
  if (trajectory) return trajectory;
  const validationSample = getCatalogValidationSampleInputs(operationId);
  if (validationSample) return validationSample;
  switch (operationId) {
    case 'hotspot-gi-star':
    case 'local-moran-statistics':
    case 'global-spatial-statistics':
      return {
        features: JSON.stringify(createKeplerCuratedAnalysisSample('city-district-rates')['areas'])
      };
    case 'recipe-points-in-polygons-choropleth': {
      const sample = createKeplerCuratedAnalysisSample('city-district-incidents');
      return {points: JSON.stringify(sample['points']), polygons: JSON.stringify(sample['areas'])};
    }
    case 'geometry-validity':
      return {
        polygons: polygons([
          [SQUARE(0, 0, 10)],
          [
            [
              [20, 0],
              [30, 10],
              [30, 0],
              [20, 10],
              [20, 0]
            ]
          ],
          [SQUARE(40, 0, 10), SQUARE(60, 0, 2)]
        ])
      };
    case 'geometry-measures':
      return {features: polygons([[SQUARE(0, 0, 10), SQUARE(2, 2, 2).reverse()]])};
    case 'geometry-point-buffer':
      return {
        points: points([
          [-122.4, 37.8],
          [-74, 40.7]
        ])
      };
    case 'geometry-line-segmentize':
      return {
        lines: lines([
          {
            id: 'diagonal',
            coordinates: [
              [0, 0],
              [10, 10]
            ]
          }
        ])
      };
    case 'geometry-convex-clip':
      return {
        features: polygons([[SQUARE(0, 0, 10)], [SQUARE(50, 50, 1)]]),
        clip: polygons([[SQUARE(5, 5, 10)]])
      };
    case 'network-build-graph':
      return {lines: SAMPLE_NETWORK};
    case 'network-shortest-path':
      return {
        lines: SAMPLE_NETWORK,
        origin: points([[0, 0]]),
        destinations: points([
          [30, 10],
          [100, 100],
          [30, 20]
        ])
      };
    case 'network-service-area':
      return {lines: SAMPLE_NETWORK, origin: points([[0, 0]])};
    case 'geometry-line-buffer':
      return {
        lines: lines([
          {id: 'crossing-a', coordinates: [CITY(0, 0), CITY(0.01, 0.01)]},
          {id: 'crossing-b', coordinates: [CITY(0, 0.01), CITY(0.01, 0)]}
        ])
      };
    case 'geometry-polygon-buffer':
      return {
        polygons: polygons([
          [CITY_SQUARE(0, 0, 0.01), CITY_SQUARE(0.004, 0.004, 0.002).reverse()],
          [CITY_SQUARE(0.012, 0, 0.004)]
        ])
      };
    case 'geometry-overlay-intersection':
    case 'geometry-overlay-union':
    case 'geometry-overlay-difference':
      return {
        left: polygons([[CITY_SQUARE(0, 0, 0.01)]]),
        right: polygons([[CITY_SQUARE(0.005, 0.005, 0.01)]])
      };
    case 'geometry-topology-repair':
      return {
        polygons: polygons([
          [[CITY(0, 0), CITY(0.01, 0.01), CITY(0.01, 0), CITY(0, 0.01), CITY(0, 0)]],
          [CITY_SQUARE(0.02, 0, 0.01)]
        ])
      };
    case 'network-crossing-noding':
      return {
        lines: lines([
          {id: 'east-west', coordinates: [CITY(0, 0.005), CITY(0.01, 0.005)]},
          {id: 'north-south', coordinates: [CITY(0.005, 0), CITY(0.005, 0.01)]},
          {id: 'diagonal', coordinates: [CITY(0, 0), CITY(0.01, 0.01)]}
        ])
      };
    case 'network-multi-origin-service-area':
      return {
        lines: CITY_GRID_NETWORK,
        origins: points([CITY(0, 0), CITY(0.01, 0.01)])
      };
    case 'network-isochrone-polygons':
      return {lines: CITY_GRID_NETWORK, origins: points([CITY(0.005, 0.005)])};
    default:
      return getPointCloudCatalogSamples(operationId);
  }
}

/** Reader supplied to feature-only sensitivity controls; table/raster transport is refused explicitly. */
export async function readCatalogFeatureInputs(
  readInputs: () => Promise<Record<string, import('@luma-contributors/kepler').CatalogDataInput>>
): Promise<Record<string, CatalogFeatureCollection>> {
  const inputs = await readInputs();
  const features: Record<string, CatalogFeatureCollection> = {};
  for (const [name, input] of Object.entries(inputs)) {
    validateRequirement(
      'catalog-sensitivity',
      input.type === 'FeatureCollection',
      `${name} requires feature input; received ${input.type}`
    );
    features[name] = input;
  }
  return features;
}

/** Decode either GeoJSON or an explicitly admitted, bounded scalar-table snapshot. */
export function readCatalogTextInput(
  text: string,
  input: import('@luma-contributors/kepler').CatalogInputSpec
): import('@luma-contributors/kepler').CatalogDataInput {
  const parsed: unknown = JSON.parse(text);
  if (
    typeof parsed === 'object' &&
    parsed !== null &&
    'type' in parsed &&
    parsed.type === 'table'
  ) {
    validateRequirement(
      'catalog-text-input',
      Boolean(input.acceptsTable || input.dataKind === 'table'),
      `${input.name} does not accept table rows`
    );
    return parseCatalogTableInput(text, input.maximumFeatures);
  }
  return parseCatalogFeatureCollection(text);
}
