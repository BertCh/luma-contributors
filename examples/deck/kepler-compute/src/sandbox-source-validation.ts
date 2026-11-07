// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerJSONValue} from '@luma-contributors/kepler';

/** Pinned Kepler serializes layer/filter/chart sources as dataId, and tooltip sources as map keys. */
export function getNativeDatasetReferences(config: KeplerJSONValue): readonly string[] {
  const references = new Set<string>();
  const addReference = (value: KeplerJSONValue): void => {
    if (value === null || value === '') return;
    if (typeof value !== 'string') throw new Error('Invalid native dataset reference');
    references.add(value);
  };
  const visit = (value: KeplerJSONValue): void => {
    if (Array.isArray(value)) {
      for (const item of value) visit(item);
    } else if (value && typeof value === 'object') {
      for (const [key, item] of Object.entries(value)) {
        if (key === 'dataId' || key === 'dataIds') {
          if (Array.isArray(item)) item.forEach(addReference);
          else addReference(item);
        } else if (key === 'fieldsToShow') {
          if (item && typeof item === 'object' && !Array.isArray(item))
            Object.keys(item).forEach(addReference);
        } else visit(item);
      }
    }
  };
  visit(config);
  return [...references];
}

/** Validate before saving or making any replay/reimport mutation. */
export function assertSandboxSourcesCovered(
  config: KeplerJSONValue,
  datasetIds: readonly string[]
): void {
  const missing = getNativeDatasetReferences(config).filter(
    datasetId => !datasetIds.includes(datasetId)
  );
  if (missing.length)
    throw new Error(
      `The source manifest does not include ${missing.join(', ')}. Remove those datasets' layers, filters, charts and tooltip fields, or include their sources before saving or replaying.`
    );
}
