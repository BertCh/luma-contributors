// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  GPU_SPATIAL_ANALYSIS_OPERATORS,
  queryGPUSpatialAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/spatial-analysis-operators';
import {
  getGPUAnalysisOperator,
  getGPUAnalysisOperatorCatalog,
  queryGPUAnalysisOperators
} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';

it('gives every original descriptor a descriptive summary in both catalog forms', () => {
  expect(
    GPU_SPATIAL_ANALYSIS_OPERATORS.filter(operator => !operator.summary?.trim()).map(
      operator => operator.exportName
    )
  ).toEqual([]);
  for (const operator of GPU_SPATIAL_ANALYSIS_OPERATORS) {
    expect(operator.summary).not.toBe(operator.exportName);
    expect(
      getGPUAnalysisOperatorCatalog().find(
        candidate =>
          candidate.exportName === operator.exportName && candidate.package === operator.package
      )?.summary
    ).toBe(operator.summary);
  }
});

it('finds point density using descriptive words rather than its export name', () => {
  expect(
    queryGPUAnalysisOperators({search: 'hexagonal'}).map(operator => operator.exportName)
  ).toContain('GPUPointDensity');
  expect(
    queryGPUSpatialAnalysisOperators({search: 'hexagonal'}).map(operator => operator.exportName)
  ).toContain('GPUPointDensity');
});

it('describes GTWR status as search convergence rather than a cutoff flag', () => {
  const limitation = getGPUAnalysisOperator(
    'GPUGeographicallyTemporallyWeightedRegression'
  )!.limitation;
  expect(limitation).toContain(
    'output.status reports profile-search convergence and requires bandwidthSearch'
  );
  expect(limitation).not.toContain('Gaussian cutoff publishes approximation');
});

it('keeps inline code balanced in the spatial join summary', () => {
  const summary = getGPUAnalysisOperator('GPUSpatialPredicateJoin')!.summary!;
  expect((summary.match(/`/g) ?? []).length % 2).toBe(0);
});

it('discovers alignment invalid counts through both public query APIs', () => {
  const name = 'addPopulationSupportAlignmentRecipe';
  expect({
    advertised: getGPUAnalysisOperator(name)!.status.includes('invalidCount'),
    spatial: queryGPUSpatialAnalysisOperators({status: 'invalidCount'}).some(
      operator => operator.exportName === name
    ),
    all: queryGPUAnalysisOperators({status: 'invalidCount'}).some(
      operator => operator.exportName === name
    )
  }).toEqual({advertised: true, spatial: true, all: true});
});
