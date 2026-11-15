// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

export const SAMPLE_MAPS_URL =
  'https://raw.githubusercontent.com/keplergl/kepler.gl-data/master/samples.json';
export const SAMPLE_REPOSITORY_LICENSE = 'Apache-2.0';
export const SAMPLE_LICENSE_URL = 'https://github.com/keplergl/kepler.gl-data/blob/master/LICENSE';
export const SAMPLE_MAP_BACKGROUND_STYLE = {
  id: 'compute-background',
  label: 'Compute background',
  style: {
    version: 8,
    sources: {},
    layers: [{id: 'background', type: 'background', paint: {'background-color': '#101d2b'}}]
  }
};

export type SampleMap = {
  id: string;
  label: string;
  description: string;
  detail: string;
  size?: number;
  visible: boolean;
  dataUrl?: string;
  configUrl?: string;
  remoteDatasetConfigUrl?: string;
  imageUrl?: string;
};

export function isSampleRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}

function readHttpUrl(value: unknown): string | undefined {
  if (typeof value !== 'string') return undefined;
  try {
    const url = new URL(value);
    return ['http:', 'https:'].includes(url.protocol) ? url.href : undefined;
  } catch {
    return undefined;
  }
}

/** Keep hidden/config-only entries discoverable, without assuming filenames or row counts. */
export function parseSampleMaps(value: unknown): SampleMap[] {
  if (!Array.isArray(value)) throw new Error('The Kepler sample catalog must be an array.');
  const identifiers = new Set<string>();
  return value.map(entry => {
    if (
      !isSampleRecord(entry) ||
      typeof entry['id'] !== 'string' ||
      !/^[a-zA-Z0-9_-]+$/.test(entry['id']) ||
      typeof entry['label'] !== 'string' ||
      identifiers.has(entry['id'])
    )
      throw new Error('The Kepler sample catalog contains an invalid or duplicate map.');
    identifiers.add(entry['id']);
    return {
      id: entry['id'],
      label: entry['label'],
      description: typeof entry['description'] === 'string' ? entry['description'] : '',
      detail:
        typeof entry['detail'] === 'string'
          ? entry['detail']
          : 'The catalog does not specify a data source or dataset licence.',
      visible: entry['visible'] !== false,
      size: typeof entry['size'] === 'number' ? entry['size'] : undefined,
      dataUrl: readHttpUrl(entry['dataUrl']),
      configUrl: readHttpUrl(entry['configUrl']),
      remoteDatasetConfigUrl: readHttpUrl(entry['remoteDatasetConfigUrl']),
      imageUrl: readHttpUrl(entry['imageUrl'])
    };
  });
}

/** Never apply the repository's code licence to third-party source data. */
export function describeSampleLicense(sample: SampleMap): string {
  if (/Noncommercial Attribution License/i.test(sample.detail))
    return 'Source data: Creative Commons Noncommercial Attribution (version unspecified).';
  return 'Source data: terms of the attributed provider; the catalog does not specify a dataset licence.';
}

export function getSampleUnsupportedReason(sample: SampleMap): string | undefined {
  if (!sample.configUrl) return 'No map config URL is provided by the public catalog.';
  if (!sample.dataUrl && !sample.remoteDatasetConfigUrl)
    return 'This config-only map has no remote dataset descriptor in the public catalog.';
  return undefined;
}

/** Render catalog Markdown links as React text/anchors, never as untrusted HTML. */
export function extractSampleAttribution(detail: string): {text: string; url?: string}[] {
  const parts: {text: string; url?: string}[] = [];
  let previousEnd = 0;
  for (const match of detail.matchAll(/\[([^\]]+)\]\(([^)]+)\)/g)) {
    if (match.index > previousEnd) parts.push({text: detail.slice(previousEnd, match.index)});
    parts.push({text: match[1]!, url: readHttpUrl(match[2])});
    previousEnd = match.index + match[0].length;
  }
  if (previousEnd < detail.length) parts.push({text: detail.slice(previousEnd)});
  return parts;
}

export function parseStartupSample(search: string): {sampleId: string; rowCount: number} {
  const parameters = new URLSearchParams(search);
  const requestedRows = Number(parameters.get('rows') ?? 100000);
  const rowCount =
    Number.isSafeInteger(requestedRows) && requestedRows > 0 && requestedRows <= 10000000
      ? requestedRows
      : 100000;
  return {
    sampleId: parameters.has('rows') ? 'synthetic' : parameters.get('sample') || 'earthquakes',
    rowCount
  };
}

/** Keep sample layers, filters, split maps and camera; use the host's token-free basemap. */
export function sanitizeSampleConfig(value: unknown): Record<string, unknown> {
  if (
    !isSampleRecord(value) ||
    value['version'] !== 'v1' ||
    !isSampleRecord(value['config']) ||
    !isSampleRecord(value['config']['visState']) ||
    !Array.isArray(value['config']['visState']['layers'])
  )
    throw new Error('The sample does not contain a supported Kepler v1 map config.');
  const config = value['config'];
  const mapStyle = isSampleRecord(config['mapStyle']) ? config['mapStyle'] : {};
  return {
    ...value,
    config: {
      ...config,
      mapStyle: {
        // Only presentation controls survive. Saved custom style URLs, tokens, style
        // sources and icons must not schedule LOAD_MAP_STYLE_TASK or basemap requests.
        ...Object.fromEntries(
          Object.entries(mapStyle).filter(([key]) =>
            [
              'topLayerGroups',
              'visibleLayerGroups',
              'threeDBuildingColor',
              'backgroundColor',
              'buildingLayer'
            ].includes(key)
          )
        ),
        styleType: SAMPLE_MAP_BACKGROUND_STYLE.id,
        mapStyles: {[SAMPLE_MAP_BACKGROUND_STYLE.id]: structuredClone(SAMPLE_MAP_BACKGROUND_STYLE)}
      }
    }
  };
}

export function describeSampleFallback(sampleId: string, error: unknown): string {
  const reason = error instanceof Error ? error.message : String(error);
  return `Could not load Kepler sample “${sampleId}”: ${reason} Showing synthetic squares instead. Open Add Data → Sample maps to retry.`;
}
