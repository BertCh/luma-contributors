// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getGPUAnalysisOperatorCatalog} from '@luma-contributors/analysis/gpu-spatial-analysis/ontology';
import {expect, test} from 'vitest';
import '../../lenses/src/computation-lenses/scenarios/gpu-network';
import '../../lenses/src/computation-lenses/scenarios/gpu-spatial-analysis/urban-ecology.lens';
import {listCatalogEntries} from '../src/analysis/catalog-bindings';
import {groupCatalogLensScenarios, listCatalogLensScenarios} from '../src/analysis/catalog-lens';

const NETWORK_SUPPLEMENT = [
  'GPUNetworkNoding',
  'GPUNetworkSnapping',
  'GPUNetworkReachability',
  'GPUNetworkCostMatrix',
  'GPUNetworkCostTable',
  'GPUNetworkCostWeights',
  'GPUNetworkAccessibility',
  'GPUNetworkServiceAreas',
  'GPUNetworkIsochrones',
  'GPUNetworkPathExtraction',
  'GPUNetworkNeighborhood',
  'GPUNetworkSubgraphFilter',
  'GPUNetworkLineGraph',
  'GPUNetworkStructure',
  'GPUNetworkCentrality',
  'GPUNetworkCoarsening',
  'GPUAlternativeRoutes',
  'GPUMapMatching',
  'GPUFlowAggregation',
  'GPUEdgeBundling',
  'GPUNetworkKFunction',
  'GPUNetworkStatistics'
];

test('UCO-10 discovery includes every catalog identity and qualified extension', () => {
  const entries = listCatalogEntries({lensScenarios: []});
  const operators = getGPUAnalysisOperatorCatalog();
  // Two upstream GPU operators and the raster-embedding binding's qualified host eigensolver.
  const upstream = ['GPURasterNeighborhood', 'GPURasterConnectedComponents'].map(exportName => ({
    package: '@luma.gl/graph/gpu-raster',
    exportName
  }));
  const hostStage = {
    package: '@luma-contributors/analysis/raster-analysis',
    exportName: 'getRasterEmbeddingPrincipalComponents'
  };
  expect(entries).toHaveLength(operators.length + upstream.length + 1);
  const identities = entries.map(entry => entry.identity);
  const keys = identities.map(({package: packageName, exportName}) =>
    JSON.stringify([packageName, exportName])
  );
  expect(new Set(keys).size).toBe(entries.length);
  expect(identities).toEqual(
    expect.arrayContaining([
      ...operators.map(({package: packageName, exportName}) => ({
        package: packageName,
        exportName
      })),
      ...upstream,
      hostStage
    ])
  );
  const visibility = identities.filter(
    identity => identity.exportName === 'addVisibilityGraphRecipe'
  );
  expect(visibility).toHaveLength(2);
  expect(visibility).toEqual(
    expect.arrayContaining([
      {package: '@luma-contributors/analysis/gpu-network', exportName: 'addVisibilityGraphRecipe'},
      {
        package: '@luma-contributors/analysis/gpu-spatial-analysis',
        exportName: 'addVisibilityGraphRecipe'
      }
    ])
  );
  expect(identities.map(identity => identity.exportName)).toEqual(
    expect.arrayContaining([
      'GPURasterArithmetic',
      'GPUTerrainFlowFractions',
      'GPUWideNumericScan',
      'GPUKShortestPaths'
    ])
  );
});

test('UCO-10 the shared catalog covers the retired discovery supplements', () => {
  const entries = listCatalogEntries({lensScenarios: []});
  const contributorsByPass = {
    network: NETWORK_SUPPLEMENT,
    pointCloud: [
      'GPUPointCloudNeighborhoods',
      'GPUPointCloudFeatures',
      'GPUPointCloudStatisticalFilter',
      'GPUPointCloudGround',
      'GPUPointCloudSMRF',
      'GPUPointCloudSurfaces',
      'GPUPointCloudCoverageValidity',
      'GPUPointCloudHeight',
      'GPUPointCloudICP',
      'GPUPointCloudChange',
      'GPUPointCloudNormalScale',
      'GPUPointCloudVoxelMerge'
    ],
    networkCategorical: [
      'GPUKernelFlowSmoothing',
      'GPUMapMatchedEdgeVolumes',
      'GPURadiationFlowModel',
      'GPUNetworkPotentialPathArea',
      'GPUNetworkReachability',
      'GPUNetworkTravelTimePercentiles',
      'GPUColocationQuotient',
      'GPULocalCategoryDependence',
      'GPUSpatialEntropy',
      'GPUCategoryPairCounts',
      'GPUGeographicalDetector',
      'GPURedcapSingleLinkage',
      'GPUHealpixPointToCell',
      'GPUHealpixCellAggregation',
      'GPUDatumGridTransform',
      'GPUSinkhornDivergence'
    ],
    spacetime: [
      'GPUSpaceTimeKernelDensity',
      'GPUSpaceTimeRelativeRisk',
      'GPUAdaptiveBandwidthKernelDensity',
      'GPUSpaceTimeHawkes',
      'GPUExpectationBasedSpaceTimeScan',
      'GPUSpaceTimeCrossCorrelation',
      'GPUSpaceTimeLocalMoran',
      'GPUJacquezTest',
      'GPULocalKnox',
      'GPUNearRepeatKnox',
      'GPUSpaceTimeDiggleTest',
      'GPUSpaceTimeKFunction',
      'GPUPettittTest',
      'GPULocationChangePoints'
    ],
    validation: [
      'GPUGeographicallyTemporallyWeightedRegression',
      'GPUMultiscaleGeographicallyTemporallyWeightedRegression',
      'GPUVecchiaLogLikelihood',
      'GPUVecchiaFisherScoring',
      'GPUVecchiaSpaceTimeLogLikelihood',
      'GPUSpaceTimeKriging',
      'GPUAreaOfApplicability',
      'GPUConformalPredictionIntervals',
      'GPUGeoConformalPredictionIntervals',
      'GPUGaussianPredictiveScores',
      'GPUEnsembleScores',
      'GPURasterFalseDiscoveryRateMask'
    ]
  };
  const discoveredNames = new Set(entries.map(entry => entry.identity.exportName));
  for (const [pass, names] of Object.entries(contributorsByPass)) {
    expect(
      names.filter(name => !discoveredNames.has(name)),
      pass
    ).toEqual([]);
  }
});

test('UCO-10 a network lens runs only the network visibility recipe', () => {
  const scenarios = listCatalogLensScenarios();
  const networkScenario = scenarios.find(
    scenario => scenario.id === 'gpu-network.visibility-graph-recipe'
  )!;
  const spatialScenario = scenarios.find(
    scenario => scenario.id === 'gpu-spatial-analysis.visibility-graph-recipe'
  )!;
  expect(networkScenario.entry).toBe('gpu-network');
  expect(spatialScenario.entry).toBe('gpu-spatial-analysis');

  const groups = groupCatalogLensScenarios([networkScenario, spatialScenario]);
  expect([...groups]).toEqual([
    ['["@luma-contributors/analysis/gpu-network","addVisibilityGraphRecipe"]', [networkScenario]],
    [
      '["@luma-contributors/analysis/gpu-spatial-analysis","addVisibilityGraphRecipe"]',
      [spatialScenario]
    ]
  ]);

  const entries = listCatalogEntries({lensScenarios: [networkScenario]});
  const visibility = entries.filter(
    entry => entry.identity.exportName === 'addVisibilityGraphRecipe'
  );
  expect(visibility).toHaveLength(2);
  expect(
    visibility.find(entry => entry.identity.package === '@luma-contributors/analysis/gpu-network')
  ).toMatchObject({runnable: 'lens-scenario', lensScenarioIds: [networkScenario.id]});
  const spatial = visibility.find(
    entry => entry.identity.package === '@luma-contributors/analysis/gpu-spatial-analysis'
  );
  expect(spatial?.runnable).toBe('no');
  expect(spatial?.lensScenarioIds).toBeUndefined();
});
