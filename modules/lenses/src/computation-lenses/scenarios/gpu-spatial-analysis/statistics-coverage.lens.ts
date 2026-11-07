// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {registerS2LensScenarios} from './define-s2-lens-scenarios';

registerS2LensScenarios(
  {
    dataset: {
      id: 'statistics-coverage-illustrative/1',
      source:
        'Deterministic synthetic observations in arbitrary planar units. Illustrative API inputs, not measured geographic data or performance evidence.',
      licence: 'MIT (repository-authored illustrative inputs)'
    },
    loadBuilder: async () =>
      (await import('./statistics-coverage-examples')).buildStatisticsCoverageScenario
  },
  [
    {
      id: 'gpu-spatial-analysis.instrumental-variables-covariance',
      element: 'GPUInstrumentalVariablesCovariance',
      title: 'GPUInstrumentalVariablesCovariance: synthetic statistical example'
    },
    {
      id: 'gpu-spatial-analysis.restricted-regime-regression',
      element: 'GPURestrictedRegimeRegression',
      title: 'GPURestrictedRegimeRegression: synthetic statistical example'
    },
    {
      id: 'gpu-spatial-analysis.spatial-panel-error-maximum-likelihood',
      element: 'GPUSpatialPanelErrorMaximumLikelihood',
      title: 'GPUSpatialPanelErrorMaximumLikelihood: synthetic statistical example'
    },
    {
      id: 'gpu-spatial-analysis.seemingly-unrelated-regression',
      element: 'GPUSeeminglyUnrelatedRegression',
      title: 'GPUSeeminglyUnrelatedRegression: synthetic statistical example'
    },
    {
      id: 'gpu-spatial-analysis.probit-regression',
      element: 'GPUProbitRegression',
      title: 'GPUProbitRegression: synthetic statistical example'
    },
    {
      id: 'gpu-spatial-analysis.local-collinearity',
      element: 'GPULocalCollinearity',
      title: 'GPULocalCollinearity: synthetic statistical example'
    },
    {
      id: 'gpu-spatial-analysis.multivariate-local-geary-permutation-test',
      element: 'GPUMultivariateLocalGearyPermutationTest',
      title: 'GPUMultivariateLocalGearyPermutationTest: synthetic statistical example'
    },
    {
      id: 'gpu-spatial-analysis.rank-markov',
      element: 'GPURankMarkov',
      title: 'GPURankMarkov: synthetic statistical example'
    },
    {
      id: 'gpu-spatial-analysis.fuzzy-contiguity',
      element: 'GPUFuzzyContiguity',
      title: 'GPUFuzzyContiguity: synthetic statistical example'
    },
    {
      id: 'gpu-spatial-analysis.island-attachment',
      element: 'GPUIslandAttachment',
      title: 'GPUIslandAttachment: synthetic statistical example'
    },
    {
      id: 'gpu-spatial-analysis.spatial-cross-validation',
      element: 'GPUSpatialCrossValidation',
      title: 'GPUSpatialCrossValidation: synthetic statistical example'
    },
    {
      id: 'gpu-spatial-analysis.spatial-cross-validated-ols',
      element: 'GPUSpatialCrossValidatedOLS',
      title: 'GPUSpatialCrossValidatedOLS: synthetic statistical example'
    },
    {
      id: 'gpu-spatial-analysis.spatial-interaction',
      element: 'GPUSpatialInteraction',
      title: 'GPUSpatialInteraction: synthetic statistical example'
    },
    {
      id: 'gpu-spatial-analysis.spatial-panel-lag-maximum-likelihood',
      element: 'GPUSpatialPanelLagMaximumLikelihood',
      title: 'GPUSpatialPanelLagMaximumLikelihood: synthetic statistical example'
    }
  ]
);
