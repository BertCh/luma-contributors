// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUAnalysisOperator,
  getGPUAnalysisCapabilityCatalog
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {CatalogEntry} from './catalog-bindings';
import {NETWORK_CATEGORICAL_BINDINGS} from './catalog-network-categorical-bindings';

/**
 * Adds this family's entries to a host's legacy discovery list without changing the shared
 * binding registry's original spatial/network enumeration. Existing entries keep their metadata.
 */
export function completeNetworkCategoricalCatalogEntries(
  entries: readonly CatalogEntry[]
): CatalogEntry[] {
  const completed = [...entries];
  const seen = new Set(entries.map(entry => entry.contributorName));
  const capabilities = getGPUAnalysisCapabilityCatalog();
  for (const binding of NETWORK_CATEGORICAL_BINDINGS) {
    for (const contributorName of binding.gpuContributorNames) {
      if (seen.has(contributorName)) continue;
      seen.add(contributorName);
      const operator = getGPUAnalysisOperator(contributorName);
      const capability = capabilities.find(
        candidate => candidate.id === operator?.primaryCapability
      );
      completed.push({
        contributorName,
        capabilityId: operator?.primaryCapability ?? 'gpu-network',
        capabilityTitle: capability?.title ?? binding.title,
        domain: capability?.domain ?? binding.domain,
        runnable: 'catalog-binding',
        engine: 'gpu',
        bindingIds: NETWORK_CATEGORICAL_BINDINGS.filter(candidate =>
          candidate.gpuContributorNames.includes(contributorName)
        ).map(candidate => candidate.id)
      });
    }
  }
  return completed;
}
