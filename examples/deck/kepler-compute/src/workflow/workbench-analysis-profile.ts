// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  applyKeplerDatasetProfileDeclarations,
  profileKeplerWorkbenchData,
  type WorkflowData,
  type WorkflowDatasetRequest
} from '@luma-contributors/kepler';
import {analysisDatasetContext} from '../analysis-dataset-context';
import type {NativeWorkflowDataset} from './workbench-host-data';
import type {CatalogResidentDataset} from '../analysis-catalog-resident-input';

/** The catalog and workbench read the same revision-checked profile and saved declarations. */
export function getWorkbenchAnalysisProfile(datasetId: string) {
  return analysisDatasetContext.get(datasetId)?.profile;
}

/** A selection gets its own statistics; full-population facts cannot certify a subset. */
export function attachWorkbenchAnalysisProfile(
  data: WorkflowData & {sourceColumnAliases?: Readonly<Record<string, string>>},
  population: 'all' | 'selected',
  geometry?: WorkflowDatasetRequest['geometry']
): WorkflowData {
  const profile = getWorkbenchAnalysisProfile(data.table.id);
  const declarations = analysisDatasetContext.getDeclarations(data.table.id);
  const aliases = data.sourceColumnAliases ?? {};
  const schemaChanged = Object.entries(aliases).some(([source, converted]) => source !== converted);
  const convertedDeclarations = declarations && {
    ...declarations,
    ...(declarations.measurement
      ? {
          measurement: Object.fromEntries(
            Object.entries(declarations.measurement).map(([name, value]) => [
              Object.hasOwn(aliases, name) ? aliases[name] : name,
              value
            ])
          )
        }
      : {})
  };
  const mappingMatches =
    !geometry ||
    (geometry.mode === 'point-columns'
      ? profile?.pointColumns?.longitude === geometry.longitude &&
        profile?.pointColumns?.latitude === geometry.latitude
      : geometry.mode === 'geojson-field'
        ? profile?.geometryColumns.includes(geometry.field)
        : profile?.kind === 'attributes');
  return {
    ...data,
    analysisProfile:
      population === 'all' &&
      !schemaChanged &&
      mappingMatches &&
      profile?.rowCount === data.table.rowCount &&
      (profile.catalogGeometry === data.kind ||
        (data.kind === 'features' && profile.mixedGeometry) ||
        (data.kind === 'table' && profile.kind === 'attributes'))
        ? profile
        : applyKeplerDatasetProfileDeclarations(profileKeplerWorkbenchData(data), {
            ...(mappingMatches && profile?.coordinates?.source === 'declared'
              ? {coordinateModel: profile.coordinates.model}
              : {}),
            ...convertedDeclarations
          })
  };
}

const nativeProfileProviders = new WeakMap<
  NativeWorkflowDataset,
  CatalogResidentDataset['provider']
>();

/** Native metadata enters the same revision cache; values are read only on explicit refinement. */
export function makeWorkbenchNativeAnalysisDatasets(
  datasets: readonly NativeWorkflowDataset[],
  isCurrent: (dataset: NativeWorkflowDataset) => boolean
): CatalogResidentDataset[] {
  return datasets.map(dataset => {
    let provider = nativeProfileProviders.get(dataset);
    if (!provider) {
      provider = {
        get fields() {
          return dataset.fields.map(field => ({name: field.name, sortable: false}));
        },
        maximumPageSize: 512,
        getRevision: () => JSON.stringify([dataset.dataRevision, dataset.dataContainer.numRows()]),
        readPage: async () => null,
        readColumnValues: async ({column}) => {
          const index = dataset.fields.findIndex(field => field.name === column);
          if (index < 0 || !isCurrent(dataset)) return null;
          return {
            fieldType: dataset.fields[index].type,
            values: Array.from({length: dataset.dataContainer.numRows()}, (_, row) =>
              dataset.dataContainer.valueAt(row, index)
            )
          };
        }
      };
      nativeProfileProviders.set(dataset, provider);
    }
    return {
      id: dataset.id,
      label: dataset.label,
      rowCount: dataset.dataContainer.numRows(),
      revision: dataset.dataRevision,
      owner: dataset,
      provider,
      fieldTypes: Object.fromEntries(dataset.fields.map(field => [field.name, field.type])),
      geometryColumn: dataset.fields.find(field => field.type === 'geojson')?.name,
      isCurrent: () => isCurrent(dataset)
    };
  });
}
