// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {
  CatalogOperationBinding,
  CatalogParameterSpec,
  CatalogRunOutput
} from './catalog-binding-types';
import {
  appendCatalogRow,
  makeCatalogTable,
  makeCatalogGeometryText,
  type CatalogFeatureCollection,
  type CatalogDiagnostic
} from './catalog-geojson';
import {makeLocalFrame} from './catalog-local-frame';
import {makeStringParameter, type Values} from './catalog-platform3-b-common';
import type {CatalogTableInput} from './catalog-data-input';

export type ValidationColumnInput = CatalogFeatureCollection | CatalogTableInput;

export function getValidationRowCount(input: ValidationColumnInput): number {
  return input.type === 'table' ? input.table.rowCount : input.features.length;
}

export const MAXIMUM_VALIDATION_ROWS = 2048;
const MAXIMUM_PROPERTY_NAME_LENGTH = 64;
export type ValidationInputs = Readonly<Record<string, CatalogFeatureCollection>>;

export function makeValidationProperty(
  name: string,
  label: string,
  defaultValue: string
): CatalogParameterSpec {
  return {...makeStringParameter(name, label), defaultValue};
}

/** Lists allow a full-length name per entry, plus commas and optional separator spaces. */
export function makeValidationPropertyList(
  name: string,
  label: string,
  defaultValue: string,
  maximumNames: number
): CatalogParameterSpec {
  return {
    ...makeValidationProperty(name, label, defaultValue),
    maximumLength: maximumNames * (MAXIMUM_PROPERTY_NAME_LENGTH + 2)
  };
}

/** Read once during graph setup; missing numeric values remain NaN for contributor statuses. */
export function readValidationColumn(
  collection: ValidationColumnInput,
  property: string
): Float32Array {
  if (!property.trim()) throw new Error('Catalog validation property must name a numeric column');
  return Float32Array.from({length: getValidationRowCount(collection)}, (_, index) => {
    const value =
      collection.type === 'table'
        ? collection.table.columns[property]?.[index]
        : collection.features[index].properties?.[property];
    return typeof value === 'number' ? value : NaN;
  });
}

export function readValidationNames(value: unknown, maximum = 11): string[] {
  const names = String(value)
    .split(',')
    .map(name => name.trim());
  if (
    !names.length ||
    names.length > maximum ||
    names.some(name => !name || name.length > MAXIMUM_PROPERTY_NAME_LENGTH) ||
    new Set(names).size !== names.length
  )
    throw new Error(
      `Catalog validation predictor properties require 1 to ${maximum} distinct names of at most ${MAXIMUM_PROPERTY_NAME_LENGTH} characters`
    );
  return names;
}

export function readValidationMatrix(
  collection: CatalogFeatureCollection,
  names: readonly string[]
): Float32Array {
  const columns = names.map(name => readValidationColumn(collection, name));
  return Float32Array.from(
    {length: collection.features.length * names.length},
    (_, index) => columns[index % names.length][Math.floor(index / names.length)]
  );
}

/** One shared local frame for samples and queries; never project the two independently. */
export function readValidationPositions(
  collections: readonly CatalogFeatureCollection[],
  values: Values
): Float32Array[] {
  const positions = collections.map(collection =>
    collection.features.map(feature => {
      const geometry = feature.geometry;
      if (
        geometry?.type !== 'Point' ||
        !Array.isArray(geometry.coordinates) ||
        geometry.coordinates.length < 2 ||
        !geometry.coordinates
          .slice(0, 2)
          .every(value => typeof value === 'number' && Number.isFinite(value))
      )
        throw new Error('Catalog validation positions require finite Point geometries');
      return [geometry.coordinates[0] as number, geometry.coordinates[1] as number] as const;
    })
  );
  const frame = makeLocalFrame(
    values['coordinateSystem'] as 'planar' | 'EPSG:4326',
    positions.flat(),
    {shift: 'minimum'}
  );
  return positions.map(rows => {
    const projected = Float32Array.from(rows.flatMap(position => frame.forward(position)));
    if (!projected.every(Number.isFinite))
      throw new Error('Catalog validation positions must project to finite Float32 coordinates');
    return projected;
  });
}

export function readValidationTimes(
  collection: CatalogFeatureCollection,
  values: Values
): Float32Array {
  const origin = values['timeOrigin'] as number;
  // Subtract before float32 rounding, so epoch seconds do not lose short time differences.
  const property = values['timeProperty'] as string;
  return Float32Array.from(collection.features, feature => {
    const value = feature.properties?.[property];
    return typeof value === 'number' ? value - origin : NaN;
  });
}

export function getValidationTemporalContext(values: Values) {
  return {
    unit: 'second' as const,
    epoch: BigInt(values['timeOrigin'] as number),
    encoding: 'float32-relative' as const,
    axis: {kind: 'irregular' as const}
  };
}

export function makeValidationBinding(
  binding: Omit<CatalogOperationBinding, 'engine' | 'cleanup' | 'gpuContributorNames'>
): CatalogOperationBinding {
  return {
    ...binding,
    engine: 'gpu',
    gpuContributorNames: binding.contributorNames,
    cleanup:
      'A scoped catalog GPU session releases all graphs and buffers after readback, including failed runs. Published tables are removed through native Sources.'
  };
}

/** Dual-route column bindings keep their public feature call while catalog dispatch uses typed data. */
export function makeColumnValidationBinding(
  binding: Omit<CatalogOperationBinding, 'engine' | 'cleanup' | 'gpuContributorNames'> &
    Required<Pick<CatalogOperationBinding, 'runDataGpu'>>
): CatalogOperationBinding {
  const declared = makeValidationBinding(binding);
  return {
    ...declared,
    runGpu: (context, inputs, values) => binding.runDataGpu(context, inputs, values)
  };
}

/** All nonfinite results become native nulls. Status words preserve the reason, including unbounded intervals. */
export function makeValidationResult(
  id: string,
  label: string,
  collection: ValidationColumnInput,
  columns: Record<string, ArrayLike<number>>,
  metrics: Record<string, number>,
  contributors: readonly string[],
  diagnostics: CatalogDiagnostic[] = []
): CatalogRunOutput {
  const table = makeCatalogTable(
    id,
    label,
    {
      feature_index: 'int32',
      ...Object.fromEntries(Object.keys(columns).map(name => [name, 'float64' as const])),
      ...(collection.type === 'table' ? {} : {geometry: 'string' as const})
    },
    collection.type === 'table' ? undefined : 'geometry'
  );
  for (let index = 0; index < getValidationRowCount(collection); index++) {
    const feature = collection.type === 'table' ? undefined : collection.features[index];
    appendCatalogRow(table, {
      feature_index: index,
      ...Object.fromEntries(
        Object.entries(columns).map(([name, column]) => [
          name,
          Number.isFinite(column[index]) ? column[index] : null
        ])
      ),
      ...(feature
        ? {
            geometry: feature.geometry
              ? makeCatalogGeometryText(feature.geometry.type, feature.geometry.coordinates)
              : null
          }
        : {})
    });
  }
  const diagnosticTable = makeCatalogTable(`${id}-diagnostics`, `${label} diagnostics`, {
    metric: 'string',
    value: 'float64'
  });
  for (const [metric, value] of Object.entries(metrics))
    appendCatalogRow(diagnosticTable, {metric, value: Number.isFinite(value) ? value : null});
  return {
    tables: [table, diagnosticTable],
    diagnostics,
    summary: {rowCount: table.rowCount},
    dispatchedContributors: contributors
  };
}

export const VALIDATION_BEHAVIOR = [
  'Input order and feature_index are preserved. Missing numeric properties are sent as NaN; the contributor determines exclusions and statuses. Nonfinite outputs become native nulls.',
  'Spatial operations require finite single Point geometries and refuse projection failures. Point coordinates use one shared local meter frame for EPSG:4326, or native units for planar data. All GPU inputs are Float32. Each input is limited to 2,048 rows.'
];
