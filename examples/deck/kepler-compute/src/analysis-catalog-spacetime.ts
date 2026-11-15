// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUAnalysisCapabilityCatalog,
  getGPUAnalysisOperator
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {KEPLER_CATALOG_BINDINGS, type CatalogEntry} from '@luma-contributors/kepler';

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

/** The legacy discovery list covers the original spatial entry point; include bound newer operators. */
export function includeSpacetimeCatalogEntries(entries: readonly CatalogEntry[]): CatalogEntry[] {
  const result = [...entries];
  const seen = new Set(entries.map(entry => entry.contributorName));
  for (const binding of KEPLER_CATALOG_BINDINGS.filter(candidate =>
    SPACETIME_OPERATION_IDS.has(candidate.id)
  ))
    for (const contributorName of binding.contributorNames) {
      if (seen.has(contributorName)) continue;
      const operator = getGPUAnalysisOperator(contributorName);
      if (!operator) continue;
      const capability = getGPUAnalysisCapabilityCatalog().find(
        candidate => candidate.id === operator.primaryCapability
      );
      result.push({
        contributorName,
        capabilityId: operator.primaryCapability,
        capabilityTitle: capability?.title ?? operator.primaryCapability,
        domain: capability?.domain ?? 'workflow',
        runnable: 'catalog-binding',
        engine: binding.engine,
        bindingIds: KEPLER_CATALOG_BINDINGS.filter(candidate =>
          candidate.contributorNames.includes(contributorName)
        ).map(candidate => candidate.id)
      });
      seen.add(contributorName);
    }
  return result;
}

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
