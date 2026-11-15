// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPU_SPATIAL_ANALYSIS_CAPABILITIES} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {VALIDATION_REGRESSION_BINDINGS} from './catalog-validation-regression';
import {VALIDATION_VECCHIA_BINDINGS} from './catalog-validation-vecchia';
import {VALIDATION_KRIGING_BINDING} from './catalog-validation-kriging';
import {VALIDATION_PREDICTION_BINDINGS} from './catalog-validation-prediction';
import {VALIDATION_SCORE_BINDINGS} from './catalog-validation-scores';

export const VALIDATION_BINDINGS = [
  ...VALIDATION_REGRESSION_BINDINGS,
  ...VALIDATION_VECCHIA_BINDINGS,
  VALIDATION_KRIGING_BINDING,
  ...VALIDATION_PREDICTION_BINDINGS,
  ...VALIDATION_SCORE_BINDINGS
];

/** The legacy spatial discovery list predates these newer operator descriptors. */
export function appendValidationCatalogEntries(
  entries: import('./catalog-bindings').CatalogEntry[],
  seen: Set<string>,
  describe: (
    contributor: string,
    capabilityId: string,
    capabilityTitle: string,
    domain: string
  ) => import('./catalog-bindings').CatalogEntry
): void {
  for (const binding of VALIDATION_BINDINGS)
    for (const contributor of binding.contributorNames) {
      if (seen.has(contributor)) continue;
      const capabilityId = binding.capabilityIds[0];
      const capability = GPU_SPATIAL_ANALYSIS_CAPABILITIES.find(
        candidate => candidate.id === capabilityId
      );
      entries.push(
        describe(contributor, capabilityId, capability?.title ?? binding.title, binding.domain)
      );
      seen.add(contributor);
    }
}
