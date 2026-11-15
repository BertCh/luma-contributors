// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  describeSampleLicense,
  isSampleRecord,
  parseSampleMaps,
  type SampleMap
} from './sample-map-catalog';

export const SAMPLE_MAP_METADATA_KEY = 'lumaSampleMap';
export type SampleMapProvenance = {sample: SampleMap; licenseNotice: string};

export function createSampleMapProvenance(sample: SampleMap): SampleMapProvenance {
  // A detached, JSON-only value travels with the source, not with the last-opened gallery card.
  return {sample: JSON.parse(JSON.stringify(sample)), licenseNotice: describeSampleLicense(sample)};
}

export function parseSampleMapProvenance(value: unknown): SampleMapProvenance | undefined {
  if (value === undefined) return undefined;
  if (!isSampleRecord(value) || typeof value['licenseNotice'] !== 'string')
    throw new Error('Invalid sample map provenance');
  const sample = parseSampleMaps([value['sample']])[0]!;
  // Recompute notices as well: a saved workspace must not erase a known source restriction.
  return {sample: JSON.parse(JSON.stringify(sample)), licenseNotice: describeSampleLicense(sample)};
}

export function getSampleMapProvenance(metadata: unknown): SampleMapProvenance | undefined {
  return isSampleRecord(metadata)
    ? parseSampleMapProvenance(metadata[SAMPLE_MAP_METADATA_KEY])
    : undefined;
}

export function listSampleMapAttributions(datasets: readonly {id: string; metadata?: unknown}[]) {
  return datasets.flatMap(dataset => {
    const provenance = getSampleMapProvenance(dataset.metadata);
    return provenance ? [{datasetId: dataset.id, ...provenance}] : [];
  });
}
