// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

const SPACETIME_OPERATION_IDS = new Set([
  'space-time-kernel-density',
  'space-time-relative-risk',
  'adaptive-bandwidth-kernel-density',
  'space-time-hawkes',
  'expectation-based-space-time-scan',
  'space-time-cross-correlation',
  'space-time-local-moran',
  'jacquez-test',
  'local-knox',
  'near-repeat-knox',
  'space-time-diggle-test',
  'space-time-k-function',
  'pettitt-test',
  'location-change-points'
]);

/** Small synthetic fixture for built-in UI samples, not bundled third-party data. */
export function getSpacetimeCatalogSample(operationId: string): string | undefined {
  if (!SPACETIME_OPERATION_IDS.has(operationId)) return undefined;
  return JSON.stringify({
    type: 'FeatureCollection',
    features: Array.from({length: 6}, (_, location) =>
      Array.from({length: 8}, (_, period) => ({
        type: 'Feature',
        id: `location-${location}-period-${period}`,
        geometry: {
          type: 'Point',
          coordinates: [-87.65 + (location % 3) * 0.002, 41.88 + Math.floor(location / 3) * 0.002]
        },
        properties: {
          location: `location-${location}`,
          time: 1700000000000 + period * 60000,
          value: (location + 1) * (period < 4 ? 1 : 3) + (period % 2),
          second: (location + 1) * (period < 3 ? 1 : 3) + ((period + 1) % 2),
          baseline: location + 1,
          case: (location + period) % 2
        }
      }))
    ).flat()
  });
}
