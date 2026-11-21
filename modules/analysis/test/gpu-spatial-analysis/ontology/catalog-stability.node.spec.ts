// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as ontology from '../../../src/gpu-spatial-analysis/ontology/index';
import {ENTRY_POINT_CAPABILITIES} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/entry-point-capabilities';
import {COVERAGE_EXPANSION_CAPABILITIES} from '../../../src/gpu-spatial-analysis/ontology/coverage-expansion-catalog';
import {
  FITTING_KERNEL_CAPABILITY,
  FITTING_KERNEL_OPERATORS
} from '../../../src/gpu-spatial-analysis/ontology/fitting-kernel-catalog';
import {KEPLER_ANALYSIS_CAPABILITIES} from '../../../src/gpu-spatial-analysis/ontology/kepler-analysis-catalog';
import {SPATIOTEMPORAL_REVIEW_OPERATORS} from '../../../src/gpu-spatial-analysis/ontology/spatiotemporal-review-catalog';

it('publishes a stability tier for every catalog record', () => {
  const catalogs = {
    operators: ontology.getGPUAnalysisOperatorCatalog(),
    capabilities: ontology.getGPUAnalysisCapabilityCatalog(),
    spatialOperators: ontology.GPU_SPATIAL_ANALYSIS_OPERATORS,
    spatialCapabilities: ontology.GPU_SPATIAL_ANALYSIS_CAPABILITIES,
    entryPointCapabilities: ENTRY_POINT_CAPABILITIES,
    keplerCapabilities: KEPLER_ANALYSIS_CAPABILITIES,
    coverageExpansionCapabilities: COVERAGE_EXPANSION_CAPABILITIES,
    fittingKernelCapability: [FITTING_KERNEL_CAPABILITY],
    fittingKernelOperators: FITTING_KERNEL_OPERATORS,
    reusedOperators: ontology.REUSED_ANALYSIS_OPERATORS,
    spatiotemporalReviewOperators: SPATIOTEMPORAL_REVIEW_OPERATORS
  };
  for (const [catalog, records] of Object.entries(catalogs)) {
    expect(records.length, catalog).toBeGreaterThan(0);
    for (const record of records) {
      const name = 'exportName' in record ? record.exportName : record.id;
      const label = `${catalog}: ${name}`;
      expect(['stable', 'experimental'], label).toContain(record.stability);
      // A future stable record must satisfy the R19-22 admission rule in dev-docs/roadmap.md.
      expect(record.stability, label).toBe('experimental');
    }
  }
  expect(ontology.GPU_ANALYSIS_STABILITY_VALUES).toEqual(['stable', 'experimental']);
});
