// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getGPUAnalysisOperatorCatalog,
  getGPUAnalysisCapabilityCatalog
} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import type {CatalogOperationBinding} from './catalog-binding-types';
import type {CatalogEntry} from './catalog-bindings';

/** The legacy discovery list scans only spatial-analysis; include this domain's ontology records. */
export function listPointCloudCatalogEntries(
  bindings: readonly CatalogOperationBinding[]
): CatalogEntry[] {
  return getGPUAnalysisOperatorCatalog()
    .filter(operator => operator.package === '@luma-contributors/analysis/pointcloud-analysis')
    .map(operator => {
      const capability = getGPUAnalysisCapabilityCatalog().find(
        candidate => candidate.id === operator.primaryCapability
      );
      const matches = bindings.filter(binding =>
        binding.gpuContributorNames.includes(operator.exportName)
      );
      return {
        contributorName: operator.exportName,
        capabilityId: operator.primaryCapability,
        capabilityTitle: capability?.title ?? operator.primaryCapability,
        domain: capability?.domain ?? 'analysis',
        runnable: matches.length ? 'catalog-binding' : 'no',
        ...(matches.length
          ? {engine: 'gpu' as const}
          : {reason: 'Point-cloud contributor has no registered binding.'}),
        bindingIds: matches.map(binding => binding.id)
      };
    });
}
