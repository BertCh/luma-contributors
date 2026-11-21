// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// dev-docs/contributor-contract.md section 14: a CPU twin is exported from its family entry point
// together with the operation form of its contributor and the function that does the computation.

import {GPUCPUOperation} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import * as analysisCore from '../../../src/analysis-core/index';
import * as dataframeAnalysis from '../../../src/dataframe-analysis/index';
import * as geospatialAnalysis from '../../../src/geospatial-analysis/index';
import * as networkAnalysis from '../../../src/gpu-network/index';
import * as spatialAnalysis from '../../../src/gpu-spatial-analysis/index';
import * as terrainAnalysis from '../../../src/gpu-terrain/index';
import * as pointcloudAnalysis from '../../../src/pointcloud-analysis/index';
import * as rasterAnalysis from '../../../src/raster-analysis/index';

const FAMILIES: Record<string, Record<string, unknown>> = {
  'analysis-core': analysisCore,
  'dataframe-analysis': dataframeAnalysis,
  'geospatial-analysis': geospatialAnalysis,
  'gpu-network': networkAnalysis,
  'gpu-spatial-analysis': spatialAnalysis,
  'gpu-terrain': terrainAnalysis,
  'raster-analysis': rasterAnalysis,
  'pointcloud-analysis': pointcloudAnalysis
};

// Independent expected complete-form stems from the reviewed operation matrix (FRM-14), plus
// the merged SpatialPanelErrorMaximumLikelihood family. Do not derive this list from exports:
// deleting every form of a subject must still fail. Case is part of each public name.
const COMPLETE_FORMS: Record<string, readonly string[]> = {
  'analysis-core': [],
  'geospatial-analysis': [],
  'pointcloud-analysis': [],
  'dataframe-analysis': `
    ClassBreaks ColumnQuantiles CoregionalizationFit Inequality RipleyEnvelope SpaceTimeDiggleTest
    VariogramModelFit
  `
    .trim()
    .split(/\s+/),
  'gpu-network': `
    AlternativeRoutes KShortestPaths MapMatching NetworkCentrality NetworkCostMatrix
    NetworkCostTable NetworkFlow NetworkLouvain NetworkPathExtraction NetworkPointRouting
    NetworkRationalAccess NetworkReachability NetworkServiceAreas NetworkStructure
    NetworkTimeRouting NetworkTravelTimePercentiles RadiationFlowModel
  `
    .trim()
    .split(/\s+/),
  'gpu-spatial-analysis': `
    AZPRegions BrownianBridgeUtilization CellMeasures ConditionalGaussianSimulation
    ConstrainedDelaunayTriangulation ContiguityWeights ContinuousSpatialSample
    ConvexStraightSkeleton DelaunayTessellation EnsembleScores GeodesicPairs HealpixPointToCell
    LabelPoint LineLoopRemoval LineSimplification LocalFrameProjection LocationAllocation
    LocationChangePoints MantelTest MapColoring MaxPRegions MaximumInscribedCircle MinimumBounds
    MobilityIndicators ModalSplit MultipleTestingAdjustment
    MultiscaleGeographicallyWeightedRegression NeighborhoodSummary NestedInterpolationValidation
    NormalizeGeometry NormalizePolygons OPTICS OffsetCurveComponents OrdinaryLeastSquares
    PettittTest PointPatternEnvelope PointToCell PolygonTriangulation PopulationSupportAlignment
    ProbitRegression RankMarkov RatioConcaveHull RestrictedRegimeRegression S2CapCovering
    S2PolygonCovering SeeminglyUnrelatedRegression SegmentSnap ShapeDescriptors SinkhornDivergence
    SkaterRegions SpaceTimeHawkes SpatialBalancedSample SpatialCrossValidation SpatialErrorGM
    SpatialErrorMaximumLikelihood SpatialInteraction SpatialLagImpacts SpatialLagMaximumLikelihood
    SpatialPanelErrorMaximumLikelihood SpatialPanelLagMaximumLikelihood SpatialRegressionDiagnostics
    SpatialTwoStageLeastSquares SpatialWeightsAlgebra SpatialWeightsTraces SpatialWeightsValidation
    SphericalArcOverlay TrackSimilarity TrajectoryKalman TrajectoryLocations TrajectoryRecursions
    TrajectoryRestriction TrajectorySegments TrajectorySplit TrajectoryStaypoints TrajectoryThinning
    TransitionDiagnostics VecchiaFisherScoring VecchiaLogLikelihood VecchiaSpaceTimeLogLikelihood
    VectorStraightSkeleton WardRegions
  `
    .trim()
    .split(/\s+/),
  'gpu-terrain': `
    ProfilePeaks SolarPosition TerrainBranchingFlow TerrainBreach TerrainCatchmentSummary
    TerrainDepressionHierarchy TerrainFlow TerrainFlowDistance TerrainPeakProminence
    TerrainStreamOrder TerrainWatersheds
  `
    .trim()
    .split(/\s+/),
  'raster-analysis': `
    CostDistance CostDistancePath Isolines RasterFillHoles RasterFractionalZonalDistribution
    RasterFractionalZonalStatistics RasterGeometricMedianComposite RasterMedoidComposite
    RasterReconstruction RasterRegionGraph RasterWalkingCost RasterWatershedSegmentation Streamlines
  `
    .trim()
    .split(/\s+/)
};

// These deliberately have no complete named class pair. Keep the partial and prepared-only
// routes explicit so that extra exports cannot conceal a missing complete form.
const PARTIAL_FORMS = [
  ['GPUBufferSelection', 'computeBufferSelectionOnCPU'],
  ['GPUGaussianPredictiveScores', 'computeGaussianPredictiveScoresOnCPU'],
  ['GPULineLengthPerPolygon', 'computeLineLengthPerPolygonOnCPU'],
  ['GPUNearestFeatureJoin', 'computeNearestFeatureJoinOnCPU'],
  ['GPUPointInPolygonJoin', 'computePointInPolygonJoinOnCPU', 'computePointInPolygonCountsOnCPU'],
  [
    'addPointsInPolygonsChoroplethRecipe',
    'computePointsInPolygonsChoroplethOnCPU',
    'createPointsInPolygonsChoroplethOperation',
    'createCPUPointsInPolygonsChoroplethOperation'
  ],
  ['GPUSpatialPredicateJoin', 'computeSpatialPredicateJoinOnCPU']
] as const;
const PREPARED_ONLY_FORMS = [
  'GPUContiguityWeights',
  'preparePolygonContiguityTopology',
  'computePolygonContiguityOnCPU'
] as const;

function qualifyForm(family: string, name: string): string {
  return `@luma-contributors/analysis/${family}:${name}`;
}

function getExpectedForms(): Set<string> {
  const expected = new Set<string>();
  for (const [family, stems] of Object.entries(COMPLETE_FORMS)) {
    for (const stem of stems) {
      for (const name of [
        `GPU${stem}`,
        stem === 'LocationAllocation'
          ? 'GPULocationAllocationGraphOperation'
          : `GPU${stem}Operation`,
        `CPU${stem}Operation`,
        `compute${stem}OnCPU`
      ]) {
        expected.add(qualifyForm(family, name));
      }
    }
  }
  for (const name of [...PARTIAL_FORMS.flat(), ...PREPARED_ONLY_FORMS]) {
    expected.add(qualifyForm('gpu-spatial-analysis', name));
  }
  return expected;
}

it('analysis-core exports the operation form and the CPU row helpers', () => {
  for (const name of [
    'GPUContributorOperation',
    'GPU_CONTRIBUTOR_OPERATION_LOWERING',
    'GPU_CONTRIBUTOR_OPERATION_TYPE',
    'getCPUOperationFloat32Rows',
    'getCPUOperationInt32Rows',
    'getCPUOperationUint32Rows',
    'validateOperationGraphNames'
  ]) {
    expect((analysisCore as Record<string, unknown>)[name], name).toBeDefined();
  }
});

function assertOperationForms(families = FAMILIES): void {
  const expected = getExpectedForms();
  const observed = new Set<string>();
  for (const [family, exports] of Object.entries(families)) {
    for (const [name, value] of Object.entries(exports)) {
      const qualified = qualifyForm(family, name);
      const isGPUOperation =
        typeof value === 'function' &&
        value.prototype instanceof analysisCore.GPUContributorOperation;
      if (
        expected.has(qualified) ||
        /^CPU\w+Operation$/.test(name) ||
        /^compute\w+OnCPU$/.test(name) ||
        /^create(?:CPU)?\w+Operation$/.test(name) ||
        isGPUOperation
      ) {
        observed.add(qualified);
        expect(typeof value, qualified).toBe('function');
        if (/^CPU\w+Operation$/.test(name)) {
          expect(
            (value as {prototype: unknown}).prototype instanceof GPUCPUOperation,
            qualified
          ).toBe(true);
        } else if (/^GPU\w+(?:Graph)?Operation$/.test(name)) {
          expect(isGPUOperation, qualified).toBe(true);
        } else if (/^GPU/.test(name)) {
          // Direct contributors implement the public command-node producer contract. A callable
          // export alone is insufficient: arrow functions and unrelated classes are not forms.
          expect(
            typeof (value as {prototype?: {getCommandNodes?: unknown}}).prototype?.getCommandNodes,
            `${qualified} implements getCommandNodes`
          ).toBe('function');
        }
      }
    }
  }
  expect(
    [...expected].filter(name => !observed.has(name)),
    'missing public forms'
  ).toEqual([]);
  expect(
    [...observed].filter(name => !expected.has(name)),
    'unexpected public forms'
  ).toEqual([]);
  expect([...observed].sort()).toEqual([...expected].sort());
}

it('every exported CPU twin comes with its GPU operation form and its compute function', () => {
  assertOperationForms();
});

it('detects complete CPU forms removed even when the former count floor still passes', () => {
  const families = Object.fromEntries(
    Object.entries(FAMILIES).map(([family, exports]) => [family, {...exports}])
  );
  delete families['gpu-network']['CPUKShortestPathsOperation'];
  delete families['dataframe-analysis']['CPUCoregionalizationFitOperation'];
  delete families['gpu-spatial-analysis']['CPUPopulationSupportAlignmentOperation'];
  expect(() => assertOperationForms(families)).toThrow();
});

it('compares form names case-sensitively and in both directions', () => {
  const families = Object.fromEntries(
    Object.entries(FAMILIES).map(([family, exports]) => [family, {...exports}])
  );
  const network = families['gpu-network'];
  network['computeNetworkFlowOnCpu'] = network['computeNetworkFlowOnCPU'];
  delete network['computeNetworkFlowOnCPU'];
  expect(() => assertOperationForms(families)).toThrow();
  network['computeNetworkFlowOnCPU'] = network['computeNetworkFlowOnCpu'];
  delete network['computeNetworkFlowOnCpu'];
  network['computeUnlistedOnCPU'] = () => {};
  expect(() => assertOperationForms(families)).toThrow();
});

it('requires the contributor side of partial forms and the command-node producer contract', () => {
  const families = Object.fromEntries(
    Object.entries(FAMILIES).map(([family, exports]) => [family, {...exports}])
  );
  delete families['gpu-spatial-analysis']['GPUNearestFeatureJoin'];
  expect(() => assertOperationForms(families)).toThrow();
});

it('rejects an arbitrary function in place of a complete-form GPU contributor', () => {
  const families = Object.fromEntries(
    Object.entries(FAMILIES).map(([family, exports]) => [family, {...exports}])
  );
  families['gpu-network']['GPUNetworkFlow'] = () => 0;
  expect(() => assertOperationForms(families)).toThrow();
});

it('checks public families with no expected forms for unexpected operations', () => {
  for (const family of ['analysis-core', 'geospatial-analysis', 'pointcloud-analysis']) {
    expect(Object.hasOwn(FAMILIES, family), family).toBe(true);
    const families = Object.fromEntries(
      Object.entries(FAMILIES).map(([name, exports]) => [name, {...exports}])
    );
    families[family]['CPUUnexpectedOperation'] = networkAnalysis.CPUNetworkFlowOperation;
    expect(() => assertOperationForms(families)).toThrow();
  }
});
