// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerTable} from '@kepler.gl/table';
import {listNativeOnlyDatasets} from './bundle-native-sources';

export type NativeDatasetAccess = {
  getDatasets(): Readonly<Record<string, KeplerTable>> | undefined;
  isResidentDataset(datasetId: string): boolean;
};

let access: NativeDatasetAccess | undefined;

/** The host installs read access to native Kepler tables once; panels never import the store. */
export function registerNativeDatasetAccess(value: NativeDatasetAccess): void {
  access = value;
}

export function getNativeDatasets(): Readonly<Record<string, KeplerTable>> {
  return access?.getDatasets() ?? {};
}

/** Datasets that exist only as native tables right now (never published, failed or excluded). */
export function getNativeOnlyDatasets(): KeplerTable[] {
  return access ? listNativeOnlyDatasets(access.getDatasets(), access.isResidentDataset) : [];
}
