// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogFeatureCollection} from '@luma-contributors/kepler';
import {
  readCatalogFeaturesFromNativeDataset,
  readCatalogTrajectoryFeaturesFromNativeDataset
} from './analysis-catalog-input';
import {TRAJECTORY_POINT_FIELD, readCatalogTrajectoryPoints} from './analysis-catalog-trajectory';
import {
  readCatalogFeaturesFromResidentDataset,
  type CatalogResidentDataset
} from './analysis-catalog-resident-input';

/** Resolve the catalog's native input choice, including publications with a retained trajectory source. */
export async function readCatalogNativeInput({
  resident,
  source,
  field,
  isTrajectory,
  parameters,
  maximumFeatures
}: {
  resident?: CatalogResidentDataset;
  source?: Parameters<typeof readCatalogFeaturesFromNativeDataset>[0];
  field: string;
  isTrajectory: boolean;
  parameters: Readonly<Record<string, unknown>>;
  maximumFeatures?: number;
}): Promise<CatalogFeatureCollection> {
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
