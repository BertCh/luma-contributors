// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogFeatureCollection, CatalogDataInput} from '@luma-contributors/kepler';
import {
  readCatalogFeaturesFromNativeDataset,
  readCatalogTableFromNativeDataset,
  readCatalogTrajectoryFeaturesFromNativeDataset
} from './analysis-catalog-input';
import {TRAJECTORY_POINT_FIELD, readCatalogTrajectoryPoints} from './analysis-catalog-trajectory';
import {
  readCatalogFeaturesFromResidentDataset,
  readCatalogColumnsFromResidentDataset,
  getCatalogResidentFields,
  type CatalogResidentDataset
} from './analysis-catalog-resident-input';

type CatalogNativeInputOptions = {
  resident?: CatalogResidentDataset;
  source?: Parameters<typeof readCatalogFeaturesFromNativeDataset>[0];
  field: string;
  isTrajectory: boolean;
  parameters: Readonly<Record<string, unknown>>;
  maximumFeatures?: number;
  acceptsTable?: boolean;
};

export function readCatalogNativeInput(
  options: CatalogNativeInputOptions & {acceptsTable?: false}
): Promise<CatalogFeatureCollection>;
export function readCatalogNativeInput(
  options: CatalogNativeInputOptions
): Promise<CatalogDataInput>;
/** Resolve native feature mappings or the explicitly admitted geometry-free table route. */
export async function readCatalogNativeInput({
  resident,
  source,
  field,
  isTrajectory,
  parameters,
  maximumFeatures,
  acceptsTable
}: CatalogNativeInputOptions): Promise<CatalogDataInput> {
  if (acceptsTable && !field) {
    if (resident) {
      const columns = await readCatalogColumnsFromResidentDataset(resident, maximumFeatures);
      return readCatalogTableFromNativeDataset(
        {
          fields: getCatalogResidentFields(resident),
          dataContainer: {
            numRows: () => resident.rowCount,
            valueAt: (row, column) => columns[column][row]
          }
        },
        maximumFeatures
      );
    }
    if (source) return readCatalogTableFromNativeDataset(source, maximumFeatures);
    throw new Error('Choose a dataset');
  }
  if ((!resident && !source) || !field) throw new Error('Choose a dataset and geometry field');
  if (isTrajectory && field === TRAJECTORY_POINT_FIELD) {
    if (!source) throw new Error('Choose a dataset and geometry field');
    return readCatalogTrajectoryPoints(
      source as Parameters<typeof readCatalogTrajectoryPoints>[0],
      parameters['longitudeProperty'] as string,
      parameters['latitudeProperty'] as string,
      parameters
    );
  }
  return resident
    ? readCatalogFeaturesFromResidentDataset(resident, field, maximumFeatures)
    : isTrajectory
      ? readCatalogTrajectoryFeaturesFromNativeDataset(source!, field, parameters)
      : readCatalogFeaturesFromNativeDataset(source!, field);
}
