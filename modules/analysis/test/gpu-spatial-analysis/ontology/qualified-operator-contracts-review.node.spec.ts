// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {OPERATION_FORM_CONTRACTS} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/operation-form-contracts';
import {RECIPE_OPERATION_FORMS} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/recipe-operation-forms';
import {computeMobilityIndicatorsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-mobility/cpu-mobility-indicators-operation';
import {computeTrajectoryStaypointsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-staypoints-operation';
import {computeTrajectoryLocationsOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-staypoints/cpu-trajectory-locations-operation';
import {
  computePointPatternEnvelopeOnCPU,
  CPUPointPatternEnvelopeOperation
} from '../../../src/gpu-spatial-analysis/point-process/cpu-point-pattern-envelope-operation';
import {computeBrownianBridgeUtilizationOnCPU} from '../../../src/gpu-spatial-analysis/trajectory-home-range/cpu-brownian-bridge-utilization-operation';
import {computeNetworkPathExtractionOnCPU} from '../../../src/gpu-network/network-analysis/cpu-network-path-extraction-operation';
import {CPUMultipleTestingAdjustmentOperation} from '../../../src/gpu-spatial-analysis/multiple-testing/cpu-multiple-testing-adjustment-operation';
import {GPUMultipleTestingAdjustmentOperation} from '../../../src/gpu-spatial-analysis/multiple-testing/gpu-multiple-testing-adjustment-operation';
import {createSamplingCPUInputs} from '../point-process/validation-sampling-operation-test-utils';
import {getGPUSpatialPredicateJoinExecutionPlan} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-predicate-route-plan';
import {computeDelaunayTessellationOnCPU} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/cpu-delaunay-tessellation-operation';
import {GPUDelaunayTessellationOperation} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation-operation';
import {
  getGPUAnalysisOperator,
  getGPUAnalysisOperatorCatalog,
  describeOutputs,
  searchGPUAnalysisTasks
} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';

const SPATIAL = '@luma-contributors/analysis/gpu-spatial-analysis' as const;
const lineIdentity = {package: SPATIAL, exportName: 'GPULineChunk'};
const olsIdentity = {package: SPATIAL, exportName: 'GPUOrdinaryLeastSquares'};

function assertRadius(result: {radiusOfGyration: Float32Array}) {
  expect(result.radiusOfGyration).toHaveLength(1);
  expect(Number.isFinite(result.radiusOfGyration[0])).toBe(true);
  expect(result.radiusOfGyration[0]).toBeCloseTo(111194.9296875, 2);
}

function assertLocation(result: {
  centers: Float32Array;
  locationCount: Uint32Array;
  locationIds: Uint32Array;
}) {
  expect(result.centers).toHaveLength(6);
  expect([...result.locationCount]).toEqual([1]);
  expect([...result.locationIds]).toEqual([0, 0, 0]);
  expect([...result.centers.slice(0, 2)]).toEqual([178.5, 0]);
}

function assertMatrixSignatures(cpu: string | undefined, gpu: string | undefined) {
  expect(cpu).toBe('new CPUNetworkCostMatrixOperation(props: NetworkCostMatrixOperationProps)');
  expect(gpu).toBe('new GPUNetworkCostMatrixOperation(props: GPUNetworkCostMatrixOperationProps)');
}

function assertPathAdmission(text: string) {
  expect(text).not.toContain('targets.length * nodeCount');
  expect(text).not.toContain('targetCapacity * nodeCount');
}

it('S1 rejects a GPU props signature substituted for the CPU compatibility form', () => {
  expect(() =>
    assertMatrixSignatures(
      'new CPUNetworkCostMatrixOperation(props: GPUNetworkCostMatrixOperationProps)',
      'new GPUNetworkCostMatrixOperation(props: GPUNetworkCostMatrixOperationProps)'
    )
  ).toThrow();
});

it('S1 detects the old path rejection expressions without a comparison suffix', () => {
  for (const expression of ['targets.length * nodeCount', 'targetCapacity * nodeCount'])
    expect(() => assertPathAdmission(expression)).toThrow();
});

it('S2 rejects extra radius rows', () => {
  expect(() => assertRadius({radiusOfGyration: Float32Array.of(111194.9296875, 1)})).toThrow();
});

it('S2 rejects incorrect location counts, IDs or allocation lengths', () => {
  const location = {
    centers: Float32Array.of(178.5, 0, 0, 0, 0, 0),
    locationCount: Uint32Array.of(1),
    locationIds: Uint32Array.of(0, 0, 0)
  };
  for (const mutant of [
    {...location, locationCount: Uint32Array.of(2)},
    {...location, locationIds: Uint32Array.of(0, 1, 0)},
    {...location, centers: Float32Array.of(178.5, 0, 0, 0, 0, 0, 0, 0)}
  ])
    expect(() => assertLocation(mutant)).toThrow();
});

it('DGN-11 count-only catalog example retains mandatory count destinations', () => {
  // Three noncollinear sites have exactly one triangle, regardless of winding or vertex order.
  const result = computeDelaunayTessellationOnCPU(
    {positions: Float32Array.of(0, 0, 1, 0, 0, 1)},
    {capacity: 0}
  );
  expect([...result.triangles]).toEqual([]);
  expect([...result.count]).toEqual([0]);
  expect([...result.requiredCount]).toEqual([1]);
  expect([...result.overflow]).toEqual([1]);
  const props = {
    positions: 'positions',
    capacity: 0,
    output: {
      triangles: 'triangles',
      status: {count: 'count', requiredCount: 'required', overflow: 'overflow'}
    }
  };
  expect(() => new GPUDelaunayTessellationOperation(props)).not.toThrow();
  for (const name of ['count', 'requiredCount', 'overflow'] as const)
    expect(
      () =>
        new GPUDelaunayTessellationOperation({
          ...props,
          output: {...props.output, status: {...props.output.status, [name]: ''}}
        })
    ).toThrow(`output.status.${name} must name a graph value`);
});

it('NUM-08 catalog example preserves radius under longitude rotation and reports an indeterminate center', () => {
  const spatialContext = {
    coordinateSpace: 'longitude-latitude',
    metric: 'great-circle',
    units: 'meters',
    sphereRadius: 6371000
  } as const;
  for (const longitudes of [
    [-1, 1],
    [179, -179],
    [0, 180]
  ]) {
    const input = {
      timestamps: Float32Array.of(0, 1),
      staypoints: {
        tracks: Uint32Array.of(0, 0),
        centroids: Float32Array.of(longitudes[0], 0, longitudes[1], 0),
        startRows: Uint32Array.of(0, 1),
        durations: Float32Array.of(1, 1),
        locationIds: Uint32Array.of(0, 1),
        count: Uint32Array.of(2)
      }
    };
    const result = computeMobilityIndicatorsOnCPU(input, {trackCount: 1, spatialContext});
    expect(result.radiusOfGyration).toHaveLength(1);
    if (longitudes[1] === 180) expect(Number.isNaN(result.radiusOfGyration[0])).toBe(true);
    else {
      // Closed form: both points are one degree from the spherical mean, R*pi/180 meters.
      assertRadius(result);
    }
  }
});

it('DGN-02 catalog example gives seam aliases one vote and preserves source rows', () => {
  for (const longitude of [180, -180, 540]) {
    const positions = Float32Array.of(180, 0, longitude, 0, 177, 0);
    const spatialContext = {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const;
    const stays = computeTrajectoryStaypointsOnCPU(
      {
        positions,
        timestamps: Float32Array.of(0, 1, 2),
        trackOffsets: Uint32Array.of(0, 3),
        parameters: Float32Array.of(500000, 1, 0, 1)
      },
      {capacity: 1, spatialContext}
    );
    const locations = computeTrajectoryLocationsOnCPU(
      {
        centroids: positions,
        tracks: Uint32Array.of(0, 0, 0),
        count: Uint32Array.of(3),
        parameters: Float32Array.of(500000, 0, 0, 0)
      },
      {spatialContext}
    );
    // Two physical longitude votes, 180 and 177: 180 - 3/2 = 178.5 degrees.
    expect([...stays.centroids]).toEqual([178.5, 0]);
    assertLocation(locations);
    expect([...stays.startRows]).toEqual([0]);
    expect([...stays.endRows]).toEqual([3]);
    expect([...positions]).toEqual([180, 0, longitude, 0, 177, 0]);
  }
});

it('TWN-03 catalog example has zero exceedances despite identical displayed statistics', () => {
  const result = computePointPatternEnvelopeOnCPU(
    {
      observed: Float32Array.of(1, 2 ** -13),
      simulations: Float32Array.from({length: 38}, (_, index) => (index % 2 ? 0 : 1)),
      reference: Float32Array.of(0, 0),
      integrationWeights: Float32Array.of(1, 1)
    },
    {
      statistic: 'integrated-squared-deviation',
      nullModel: 'fixed-catalog-fixture',
      fittedParameters: 'fixed-external'
    }
  );
  // Exact squared sums: 1 + (2^-13)^2 > 1. Zero exceedances => (0+1)/(19+1)=0.05.
  expect([...result.statistics].every(Number.isFinite)).toBe(true);
  expect([...result.statistics]).toEqual([1, Math.fround(0.05), Math.fround(0.05), 19]);
  expect([...result.status]).toEqual([0]);
});

const contractText = (name: string, entry = 'gpu-spatial-analysis') =>
  JSON.stringify(OPERATION_FORM_CONTRACTS[`@luma-contributors/analysis/${entry}:${name}`]);
const arrayText = (name: string) => JSON.stringify(getGPUAnalysisOperator(name)?.forms?.arrays);

it('B1 named envelope defaults infer five simulations from ten values and two observed rows', async () => {
  const operation = new CPUPointPatternEnvelopeOperation({
    observed: 'observed',
    simulations: 'simulations',
    reference: 'reference',
    curveRowCount: 2,
    nullModel: 'fixed external curves',
    fittedParameters: 'fixed-external',
    output: {lower: 'lower', upper: 'upper', statistics: 'statistics', status: 'status'}
  });
  const result = await operation.execute({
    inputs: createSamplingCPUInputs({
      observed: {format: 'float32', values: Float32Array.of(2, 2)},
      simulations: {format: 'float32', values: new Float32Array(10)},
      reference: {format: 'float32', values: Float32Array.of(0, 0)}
    })
  });
  // Ten replicate-major values / two ordinates gives exactly five simulations.
  expect(result['statistics']).toEqual(Float32Array.of(2, 1 / 6, 1 / 6, 5));
  const defaults =
    OPERATION_FORM_CONTRACTS[`${SPATIAL}:CPUPointPatternEnvelopeOperation`]?.defaults;
  expect(defaults).toContain(
    'Omitted simulationCount is inferred as simulations.length / curveRowCount at execution; the quotient must be an integer >= 2.'
  );
  expect(defaults).toContain('statistic defaults to maximum-deviation.');
  expect(JSON.stringify(defaults)).not.toContain('Math.max(2, 2 *');
});

it('B2 envelope contracts distinguish exact comparisons from float32 display', () => {
  const maximum = computePointPatternEnvelopeOnCPU(
    {
      observed: Float32Array.of(-1),
      reference: Float32Array.of(2 ** -60),
      simulations: new Float32Array(7).fill(1)
    },
    {nullModel: 'fixed external curves', fittedParameters: 'fixed-external'}
  );
  // Exact |observed-reference|=1+2^-60 > 1-2^-60 for all seven curves: p=1/8.
  expect([...maximum.statistics].every(Number.isFinite)).toBe(true);
  expect([...maximum.statistics]).toEqual([1, 0.125, 0.125, 7]);
  for (const name of ['computePointPatternEnvelopeOnCPU', 'CPUPointPatternEnvelopeOperation']) {
    const cpu = contractText(name);
    expect(cpu).toContain(
      'CPU maximum-deviation and integrated-squared-deviation use exact dyadic arithmetic for exceedance decisions'
    );
    expect(cpu).toContain('The displayed statistic is rounded to float32.');
    expect(cpu).toContain('p=0.05');
  }
  const gpu = contractText('GPUPointPatternEnvelopeOperation');
  expect(gpu).toContain(
    'GPU maximum-deviation and integrated-squared-deviation use exact dyadic arithmetic for exceedance decisions'
  );
  expect(gpu).toContain('The displayed statistic uses float32 arithmetic.');
  expect(gpu).toContain('p=0.125');
});

it('earlier-fix CPU integrated envelope retains a residual below binary64 sum precision', () => {
  const result = computePointPatternEnvelopeOnCPU(
    {
      observed: Float32Array.of(2, 2 ** -30),
      simulations: Float32Array.from({length: 10}, (_, index) => (index % 2 ? 0 : 2)),
      reference: Float32Array.of(0, 0),
      integrationWeights: Float32Array.of(1, 1)
    },
    {
      statistic: 'integrated-squared-deviation',
      nullModel: 'fixed external curves',
      fittedParameters: 'fixed-external'
    }
  );
  // Exact 4+2^-60 > 4 for all five simulations: zero exceedances => p=1/6.
  expect(Number.isFinite(result.statistics[1])).toBe(true);
  expect(result.statistics[1]).toBe(Math.fround(1 / 6));
});

it('B3 bridge storage is an array-only option with resident-capacity as its default', () => {
  const input = {
    positions: Float32Array.of(0, 0),
    timestamps: Float32Array.of(0),
    trackOffsets: Uint32Array.of(0, 2),
    motionVariances: Float32Array.of(0),
    locationErrors: Float32Array.of(1),
    parameters: Float32Array.of(0, 0, 3, 1)
  };
  expect([...computeBrownianBridgeUtilizationOnCPU(input, {gridSize: [3, 1]}).density]).toEqual([
    0, 0, 0
  ]);
  expect(() =>
    computeBrownianBridgeUtilizationOnCPU({...input, storage: 'compact'}, {gridSize: [3, 1]})
  ).toThrow();
  const pure = OPERATION_FORM_CONTRACTS[`${SPATIAL}:computeBrownianBridgeUtilizationOnCPU`];
  expect(pure?.defaults).toContain('input.storage defaults to resident-capacity.');
  expect(JSON.stringify(pure?.defaults)).not.toContain('Defaults to compact');
  for (const name of [
    'CPUBrownianBridgeUtilizationOperation',
    'GPUBrownianBridgeUtilizationOperation'
  ])
    expect(contractText(name)).not.toContain('input.storage');
});

it('B4 path targetCount rejects an excessive numeric count and clamps the live word', () => {
  const input = {
    predecessors: Uint32Array.of(0xffffffff),
    costs: Float32Array.of(0),
    targets: Uint32Array.of(0),
    predecessorsComplete: Uint32Array.of(1),
    capacity: 1
  };
  expect(() => computeNetworkPathExtractionOnCPU({...input, targetCount: 2})).toThrow(
    /targetCount.*\[0, 1\]/
  );
  const result = computeNetworkPathExtractionOnCPU({...input, targetCount: Uint32Array.of(2)});
  expect([...result.count]).toEqual([1]);
  expect([...result.nodeRows]).toEqual([0]);
  for (const text of [
    contractText('computeNetworkPathExtractionOnCPU', 'gpu-network'),
    contractText('CPUNetworkPathExtractionOperation', 'gpu-network'),
    arrayText('GPUNetworkPathExtraction')
  ]) {
    expect(text).toContain('Numeric targetCount must lie in [0, target capacity]');
    expect(text).toContain('live uint32 targetCount word is clamped to target capacity');
  }
});

it('B5 staypoint lookup places missingFixPolicy on options in both signatures and limits', () => {
  const array = getGPUAnalysisOperator('GPUTrajectoryStaypoints')?.forms?.arrays?.[0];
  for (const signature of [array?.signature, array?.contract?.signature])
    expect(signature).toContain("missingFixPolicy?: 'break'");
  expect(JSON.stringify(array?.contract?.supportedInputs)).toContain('missingFixPolicy');
  expect(JSON.stringify(array?.contract?.limits)).toContain('options.missingFixPolicy');
  expect(JSON.stringify(array?.contract?.limits)).not.toContain('input.missingFixPolicy');
});

it('B6 geographic catalog documents equivalent float32 longitudes near the seam', () => {
  for (const name of ['GPUTrajectoryStaypoints', 'GPUTrajectoryLocations'])
    expect(arrayText(name)).toContain(
      'The float32 longitudes 179.99998474121094 and -180.00001525878906 share one vote in both forms.'
    );
});

it('S3 named multiple-testing methods require an explicit accepted value', () => {
  for (const Operation of [
    CPUMultipleTestingAdjustmentOperation,
    GPUMultipleTestingAdjustmentOperation
  ])
    expect(
      () =>
        new Operation({
          pValues: 'pValues',
          rowCount: 1,
          method: undefined as never,
          output: {adjustedPValues: 'adjusted'}
        })
    ).toThrow(/method is required/);
  for (const name of [
    'CPUMultipleTestingAdjustmentOperation',
    'GPUMultipleTestingAdjustmentOperation'
  ]) {
    const text = contractText(name);
    expect(text).toContain('method is required');
    for (const method of [
      'none',
      'bonferroni',
      'holm',
      'benjamini-hochberg',
      'benjamini-yekutieli'
    ])
      expect(text).toContain(method);
  }
});

it('S2 binds each Hawkes likelihood name to the iterate it describes', () => {
  const notes = getGPUAnalysisOperator('GPUSpaceTimeHawkes')?.forms?.arrays?.[0]?.differences;
  expect(notes).toContain(
    'logLikelihood: finite-window log likelihood at the iterate entering the final update.'
  );
  expect(notes).toContain(
    'fittedLogLikelihood: finite-window log likelihood at the published estimates.'
  );
});

it('CMP-08 separates GPU noded source completeness from CPU search convergence', () => {
  const forms = getGPUAnalysisOperator('GPUNetworkCostMatrix')?.forms;
  assertMatrixSignatures(forms?.cpu?.contract?.signature, forms?.gpu?.contract?.signature);
  for (const word of [
    'nodedTopology.nodeCount',
    'nodedTopology.pieceCount',
    'sourceComplete',
    'converged',
    'iterationLimitReached'
  ])
    expect(JSON.stringify(forms?.gpu?.contract)).toContain(word);
  expect(JSON.stringify(forms?.cpu?.contract)).toContain('GPU operation only');
});

it('W5-07 publishes active-prefix path admission rather than allocated target rejection', () => {
  for (const name of ['computeNetworkPathExtractionOnCPU', 'CPUNetworkPathExtractionOperation']) {
    const text = contractText(name, 'gpu-network');
    expect(text).toContain('activeTargetUpperBound * min(nodeCount, maximumPathLength)');
    expect(text).toContain('actual totals above 4294967295');
    assertPathAdmission(text);
  }
  expect(arrayText('GPUNetworkPathExtraction')).toContain('active target prefix');
});

it('DGN-11 distinguishes zero-capacity count-only Delaunay from missing output names', () => {
  for (const name of ['CPUDelaunayTessellationOperation', 'GPUDelaunayTessellationOperation']) {
    const text = contractText(name);
    expect(text).toContain('capacity = 0');
    expect(text).toContain('output.status.count');
    expect(text).toContain('output.status.requiredCount');
    expect(text).toContain('output.status.overflow');
  }
  expect(contractText('CPUPolygonTriangulationOperation')).toContain(
    'CPUPolygonTriangulationOperationProps'
  );
  expect(contractText('CPUConstrainedDelaunayTriangulationOperation')).toContain(
    'CPUConstrainedDelaunayTriangulationOperationProps'
  );
  for (const [name, entry] of [
    ['CPUPolygonTriangulationOperation', 'gpu-spatial-analysis'],
    ['CPUConstrainedDelaunayTriangulationOperation', 'gpu-spatial-analysis']
  ]) {
    expect(contractText(name, entry)).toContain('resident-capacity');
    expect(contractText(name, entry)).toContain('compact');
  }
});

it('TYP-16 publishes named result counts and rectangular diagonal validation in every form', () => {
  expect(arrayText('GPUMultipleTestingAdjustment')).toContain('CPUMultipleTestingAdjustmentResult');
  for (const name of [
    'computeMultipleTestingAdjustmentOnCPU',
    'CPUMultipleTestingAdjustmentOperation',
    'GPUMultipleTestingAdjustmentOperation'
  ]) {
    for (const field of ['finiteTestCount', 'effectiveFamilySize'])
      expect(contractText(name)).toContain(field);
  }
  expect(arrayText('GPUSpatialWeightsValidation')).toContain(
    'options: SpatialWeightsValidationOptions = {}'
  );
  expect(arrayText('GPUSpatialWeightsValidation')).toContain('CPUSpatialWeightsValidationResult');
  for (const name of [
    'computeSpatialWeightsValidationOnCPU',
    'CPUSpatialWeightsValidationOperation',
    'GPUSpatialWeightsValidationOperation'
  ]) {
    const text = contractText(name);
    for (const field of ['columnCount', 'forbid', 'allow', 'require-unit'])
      expect(text).toContain(field);
  }
});

it('SCL-08 retains recipe-wide estimates, CPU certificate compatibility and actual stop-cube stages', () => {
  const key = '@luma-contributors/analysis/raster-analysis:addRasterRegionGraphTiles';
  expect(contractText('addRasterRegionGraphTiles', 'raster-analysis')).toContain(
    'readonly estimatedRecipeCommandNodeCount: number'
  );
  expect(JSON.stringify(RECIPE_OPERATION_FORMS[key])).toContain(
    'readonly estimatedRecipeCommandNodeCount: number'
  );
  expect(JSON.stringify(RECIPE_OPERATION_FORMS[key])).toContain('assembly');
  expect(contractText('computeRasterRegionGraphOnCPU', 'raster-analysis')).toContain(
    'CPURasterRegionGraphOptions | RasterRegionGraphOperationProps'
  );
  const stages = RECIPE_OPERATION_FORMS[`${SPATIAL}:addStopValueCubeRecipe`]?.recipeStages;
  expect(stages?.map(stage => stage.exportName)).toEqual([
    'GPUPeriodIndex',
    'GPUPointToCell',
    'GPUKeyLookup',
    'GPUValueCube'
  ]);
  expect(stages?.map(stage => stage.package)).toEqual([
    '@luma-contributors/analysis/dataframe-analysis',
    SPATIAL,
    '@luma-contributors/analysis/dataframe-analysis',
    '@luma-contributors/analysis/dataframe-analysis'
  ]);
});

it('FRM-08 publishes bridge tails separately from midpoint integration and empty storage', () => {
  const text = arrayText('GPUBrownianBridgeUtilization');
  for (const word of [
    'tailPolicy',
    'full',
    'truncate-at-seven-sigma',
    'zero or one fix',
    'resident-capacity',
    'compact',
    'midpoint'
  ])
    expect(text).toContain(word);
  expect(text).not.toContain('minimum of two fixes');
  expect(text).not.toContain('No seven-sigma cutoff');
});

it('TWN-05 publishes inclusive stored-coordinate membership and unavailable track evidence', () => {
  for (const name of ['GPUTrackSimilarity', 'GPUTrajectoryLocations', 'GPUSpatialCrossValidation'])
    expect(arrayText(name)).toContain('inclusive Euclidean distance on stored coordinates');
  const text = arrayText('GPUTrackSimilarity');
  for (const word of [
    '8 invalidGeometry',
    '16 invalidDecisionThreshold',
    'displayed',
    'membership false'
  ])
    expect(text).toContain(word);
  expect(text).not.toContain('GPU results are undefined');
});

it('TWN-03 publishes statistic-specific inputs and residual-preserving exceedance decisions', () => {
  const text = JSON.stringify(
    getGPUAnalysisOperator('GPUPointPatternEnvelope')?.forms?.arrays?.[0]?.differences
  );
  for (const word of [
    'extreme-rank',
    'observed/simulations only',
    'ignores integrationWeights',
    'positive finite',
    'exact quotient',
    'exact dyadic arithmetic for exceedance decisions',
    'displayed statistic',
    'zero exceedances',
    'p=0.05',
    'p=0.125'
  ])
    expect(text).toContain(word);
  expect(contractText('computePointPatternEnvelopeOnCPU')).toContain(
    'The displayed statistic is rounded to float32.'
  );
});

it('DEF-13 publishes one Hawkes default and distinguishes budget from updates and convergence', () => {
  for (const name of [
    'computeSpaceTimeHawkesOnCPU',
    'CPUSpaceTimeHawkesOperation',
    'GPUSpaceTimeHawkesOperation'
  ]) {
    const text = contractText(name);
    expect(text).toContain('100 default iterations');
    expect(text).toContain('maximumIterations');
    expect(text).toContain('iterationCount');
    expect(text).toContain('converged');
  }
  expect(arrayText('GPUSpaceTimeHawkes')).not.toContain('to 1000, versus');
});

it('UPT-01 publishes the safeguarded finite-window Hawkes objective and likelihood timing', () => {
  const text = arrayText('GPUSpaceTimeHawkes');
  for (const word of [
    'compensator derivative',
    'branching ratio is refit for each candidate',
    'likelihood decrease',
    'logLikelihood',
    'fittedLogLikelihood'
  ])
    expect(text).toContain(word);
});

it('NUM-08 publishes spherical geographic centering and the indeterminate-center result', () => {
  const text = arrayText('GPUMobilityIndicators');
  for (const word of [
    'great-circle default',
    'spherical-mean',
    'coordinate-mean',
    'planar',
    'NaN',
    '111194.93'
  ])
    expect(text).toContain(word);
  expect(text).not.toContain('deliberate lack of longitude unwrapping');
});

it('DGN-02 publishes geographic identity while preserving planar equality and source rows', () => {
  for (const name of ['GPUTrajectoryStaypoints', 'GPUTrajectoryLocations']) {
    const text = arrayText(name);
    for (const word of [
      '180, -180 and 540',
      'exact poles',
      'planar equality',
      'source row IDs',
      '178.5'
    ])
      expect(text).toContain(word);
  }
});

it('B1 distinguishes transpose matrix loss from caller chunk diagnostics', () => {
  const notes = getGPUAnalysisOperator('GPUSpatialWeightsTranspose')
    ?.forms?.variants?.flatMap(form => (form.differences === 'unknown' ? [] : form.differences))
    .join(' ');
  expect(notes).toContain('packed or chunked output');
  expect(notes).toContain('outputWeights.status.candidateOverflow');
  expect(notes).toContain('chunk descriptions omit loss flags');
});

it('B2 qualifies required vertices and documents the source-index sentinel', () => {
  const outputs = describeOutputs(lineIdentity, 'chunk', {
    lineCapacity: 1,
    vertexCapacity: 8
  }).outputs;
  expect(outputs.find(output => output.path === 'output.status.requiredVertexCount')).toMatchObject(
    {
      status: ['output.status.requiredCount', 'output.status.count', 'output.status.overflow'],
      meaning: expect.stringContaining('retained pieces')
    }
  );
  expect(outputs.find(output => output.path === 'output.sourceLineIndices')?.tail).toBe(
    '0xffffffff past output.status.count'
  );
});

it('B3 checks substring uint32 product and discloses unchecked input constraints', () => {
  // 65535 * 65536 fits uint32; 65536 * 65537 does not.
  expect(() =>
    describeOutputs(lineIdentity, 'substring', {lineCount: 65535, vertexCapacity: 8})
  ).not.toThrow();
  expect(() =>
    describeOutputs(lineIdentity, 'substring', {lineCount: 65536, vertexCapacity: 8})
  ).toThrow(/lineCount.*lineCapacity/);
  expect(
    describeOutputs(lineIdentity, 'chunk', {lineCapacity: 5, vertexCapacity: 8})
  ).toMatchObject({constraints: [{checked: false}, {checked: false}]});
  expect(() =>
    describeOutputs(lineIdentity, 'chunk', {
      lineCapacity: 65536,
      lineCount: 65536,
      positionCount: 131072,
      vertexCapacity: 8
    })
  ).toThrow(/lineCount/);
});

it('B4-B6 publishes enforced KDE, natural-breaks and relate conditions', () => {
  const kde = getGPUAnalysisOperator('GPUKernelDensity')?.routes?.[0];
  expect(kde?.byteExpression).toContain('indexGridSize');
  expect(kde?.appliesWhen).toContain('all methods');
  expect(kde?.alternative).toContain('Gaussian');
  for (const route of getGPUAnalysisOperator('GPUClassBreaks')?.routes ?? []) {
    expect(route.appliesWhen).toContain('natural-breaks enabled');
  }
  const relate = getGPUAnalysisOperator('GPUSpatialPredicateJoin')?.routes?.[0]?.appliesWhen;
  expect(relate).toContain('engine auto');
  expect(relate).toContain('intersects/contains/within');
  expect(relate).not.toContain('multipart uses fast');
  // The source-owned descriptor is the auto crossover; explicit relate output forces the route.
  const shape = {
    predicate: 'intersects',
    leftKind: 'lines',
    rightKind: 'lines',
    leftCount: 1,
    rightCount: 1,
    leftVertexCount: 2,
    rightVertexCount: 2
  } as const;
  expect(getGPUSpatialPredicateJoinExecutionPlan(shape).route).toBe('fast');
  expect(getGPUSpatialPredicateJoinExecutionPlan({...shape, outputRelate: true}).route).toBe(
    'relate'
  );
});

it('B7 keeps array numerical and padding notes aligned with producers', () => {
  const notes = (name: string) =>
    getGPUAnalysisOperator(name)
      ?.forms?.arrays?.flatMap(form => (form.differences === 'unknown' ? [] : form.differences))
      .join(' ');
  expect(notes('GPURankMarkov')).toContain('pairwise');
  expect(notes('GPURankMarkov')).toContain('sorted');
  expect(notes('GPUColumnQuantiles')).toContain('above 2^24');
  expect(notes('GPUStreamlines')).toContain('repeated terminal offsets');
});

it('B8 ranks exact public names and preserves compact fillholes spelling', () => {
  for (const name of ['GPUPointDensity', 'GPUClassBreaks', 'GPURasterNDVI'])
    expect(searchGPUAnalysisTasks(name)[0]?.identity.exportName).toBe(name);
  expect(searchGPUAnalysisTasks('fillholes')[0]?.identity.exportName).toBe('GPURasterFillHoles');
});

it('B9 supports cast-free narrowed search-to-lookup and output description', () => {
  for (const result of searchGPUAnalysisTasks('GPUOrdinaryLeastSquares').slice(0, 1)) {
    if (result.owner === 'analysis') {
      expect(getGPUAnalysisOperator(result.identity)?.exportName).toBe('GPUOrdinaryLeastSquares');
      expect(
        describeOutputs(result.identity, 'default', {predictorCount: 1, rowCount: 5}).declaration
      ).toBe('declared');
    }
  }
});

it('B10 independently pins contributor/form identities and outline conditions', () => {
  for (const operator of getGPUAnalysisOperatorCatalog()) {
    const forms = operator.forms;
    if (forms?.cpu) {
      const stem = operator.exportName.slice(3);
      expect(forms.cpu).toMatchObject({
        package: operator.package,
        exportName: `CPU${stem}Operation`
      });
      expect(forms.gpu?.package).toBe(operator.package);
      expect(forms.gpu?.exportName).toBe(
        stem === 'LocationAllocation'
          ? 'GPULocationAllocationGraphOperation'
          : `GPU${stem}Operation`
      );
    }
    for (const array of forms?.arrays ?? []) {
      const expected =
        array.exportName === 'computePolygonContiguityOnCPU'
          ? 'GPUContiguityWeights'
          : array.exportName === 'computePointsInPolygonsChoroplethOnCPU'
            ? 'addPointsInPolygonsChoroplethRecipe'
            : array.exportName === 'computePointInPolygonCountsOnCPU'
              ? 'GPUPointInPolygonJoin' // Counts-only entry shares the sparse join implementation.
              : `GPU${array.exportName.slice(7, -5)}`;
      expect(operator.exportName).toBe(expected);
      expect(array.package).toBe(operator.package);
    }
  }
  const stages = getGPUAnalysisOperator('addClusterAndOutlineRecipe')?.forms?.recipeStages;
  expect(stages?.map(stage => stage.exportName)).toEqual([
    'GPUSpatialJoinPrepared',
    'GPUDBSCAN',
    'GPUGroupGeometry',
    'GPUGroupConvexHull',
    'GPUGeometryMeasures',
    'GPUGroupConcaveHull'
  ]);
  expect(stages?.find(stage => stage.exportName === 'GPUGroupConvexHull')).toMatchObject({
    appliesWhen: "props.outline?.kind !== 'alpha'"
  });
  expect(stages?.find(stage => stage.exportName === 'GPUGroupConcaveHull')).toMatchObject({
    appliesWhen: "!(props.outline?.kind !== 'alpha')"
  });
});

it('S1 separates training fitted rows from all finite-predictor predictions', () => {
  const outputs = describeOutputs(olsIdentity, 'default', {predictorCount: 1, rowCount: 5}).outputs;
  expect(outputs.find(output => output.path === 'output.predictions')?.activeLengthRule).toContain(
    'including masked rows or missing responses'
  );
  expect(outputs.find(output => output.path === 'output.fitted')?.activeLengthRule).toContain(
    'training'
  );
});

it('S2 checks complete paths, formats, capacities, optionality and status associations', () => {
  const ols = describeOutputs(olsIdentity, 'default', {predictorCount: 2, rowCount: 10}).outputs;
  expect(
    ols.map(output => [
      output.path,
      output.format,
      output.capacity,
      output.minimumCapacity,
      output.optional,
      output.status
    ])
  ).toEqual([
    ...['coefficients', 'standardErrors', 'tStatistics'].map(name => [
      `output.${name}`,
      'float32',
      16,
      3,
      false,
      ['output.status.code']
    ]),
    ['output.summary', 'float32', 22, 22, false, ['output.status.code']],
    ['output.status.code', 'uint32', 1, 1, false, []],
    ...['residuals', 'fitted', 'predictions'].map(name => [
      `output.${name}`,
      'float32',
      10,
      10,
      true,
      ['output.status.code']
    ])
  ]);
  for (const mode of ['chunk', 'substring']) {
    const outputs = describeOutputs(lineIdentity, mode, {
      lineCapacity: 5,
      lineCount: 3,
      vertexCapacity: 20
    }).outputs;
    const capacity = mode === 'chunk' ? 5 : 3;
    const loss = ['output.status.count', 'output.status.overflow'];
    expect(
      outputs.map(output => [
        output.path,
        output.format,
        output.capacity,
        output.optional,
        output.status
      ])
    ).toEqual([
      ['output.positions', 'float32x2', 20, false, loss],
      ['output.lineOffsets', 'uint32', capacity + 1, false, loss],
      ['output.sourceLineIndices', 'uint32', capacity, true, loss],
      ['output.sourceRows', 'uint32', 20, true, loss],
      ['output.measures', 'float32', 20, true, loss],
      ...['count', 'overflow', 'requiredCount', 'vertexCount', 'requiredVertexCount'].map(name => [
        `output.status.${name}`,
        'uint32',
        1,
        !['count', 'overflow'].includes(name),
        name === 'requiredVertexCount'
          ? ['output.status.requiredCount', 'output.status.count', 'output.status.overflow']
          : []
      ])
    ]);
  }
  expect(searchGPUAnalysisTasks('Pettitt', {question: 'prepare-data'})).toEqual([]);
});

it('UNFINISHED FRM-13 declares all qualified recipes and the named prepared/tiled cases', () => {
  const recipes = getGPUAnalysisOperatorCatalog().filter(operator =>
    /^add.*Recipe$/.test(operator.exportName)
  );
  expect(recipes).toHaveLength(45);
  for (const recipe of recipes) {
    expect(recipe.forms?.recipeStages, recipe.exportName).toBeDefined();
    expect(recipe.forms?.recipeImplementation, recipe.exportName).toBeTruthy();
    for (const stage of recipe.forms?.recipeStages ?? []) expect(stage.appliesWhen).toBeTruthy();
  }
  for (const [operator, variant] of [
    ['GPUPointInPolygonJoin', 'GPUSpatialJoinPrepared'],
    ['GPUSpatialJoinCandidates', 'GPUSpatialJoinPrepared'],
    ['GPUContiguityWeights', 'preparePolygonContiguityTopology'],
    ['GPUTerrainFlow', 'GPUTerrainTiledFlow'],
    ['GPUIsolines', 'GPUIsolines'],
    ['GPURasterRegionGraph', 'addRasterRegionGraphTiles']
  ])
    expect(
      getGPUAnalysisOperator(operator)?.forms?.variants?.map(form => form.exportName),
      operator
    ).toContain(variant);
});

it('UNFINISHED FRM-13 records signatures, supported inputs and option contracts for every registered form', () => {
  for (const operator of getGPUAnalysisOperatorCatalog()) {
    for (const form of [
      operator.forms?.cpu,
      operator.forms?.gpu,
      ...(operator.forms?.arrays ?? []),
      ...(operator.forms?.variants ?? [])
    ]) {
      if (!form) continue;
      expect(form.contract?.signature, form.exportName).toBeTruthy();
      expect(form.contract?.supportedInputs.length, form.exportName).toBeGreaterThan(0);
      expect(form.contract?.defaults, form.exportName).toBeDefined();
      expect(form.contract?.limits, form.exportName).toBeDefined();
      expect(form.contract?.optionHandling, form.exportName).toBeDefined();
    }
  }
});

it('B10 pins each qualified recipe stage mapping independently of the catalog tables', () => {
  // Constructor calls traced in recipe bodies and their first-party helpers; empty rows use
  // inline kernels/upstream primitives. Names are GPU stems, in first reachable source order.
  const spatial: Record<string, string> = {
    HotSpotAnalysis:
      'SpatialJoinPrepared LonLatPolygonPreparation CellCover CellAggregation PointToCell CellGeometry CellMeasures NeighborSearch CellTopology LatticeWeights HotSpotAnalysis LocalPermutationTest ClassBreaks ColorScale',
    RateClusterMap:
      'SpatialJoinPrepared EmpiricalBayesRates ContiguityWeights SpatialWeightsTransform LocalMoran LocalPermutationTest',
    SpatialRegression:
      'SpatialJoinPrepared OrdinaryLeastSquares SpatialRegressionDiagnostics LocalMoran GeographicallyWeightedRegression',
    SpaceTimeHotSpots:
      'SpatialJoinPrepared CalendarBuckets PeriodIndex GroupStatistics EmergingHotSpots ColorScale',
    PeriodComparison: 'SpatialJoinPrepared CellAggregation CellTableCompare ClassBreaks ColorScale',
    PointsInPolygonsChoropleth:
      'SpatialJoinPrepared LonLatPolygonPreparation PointInPolygonJoin GroupStatistics ZonalStatistics ClassBreaks ColorScale',
    ClusterAndOutline:
      'SpatialJoinPrepared DBSCAN GroupGeometry GroupConvexHull GeometryMeasures GroupConcaveHull',
    ChangeOfSupport:
      'SpatialJoinPrepared LonLatPolygonPreparation PolygonRasterization ArealInterpolation SpatialLag',
    StraightLineCatchments: 'SpatialJoinPrepared DistanceField RasterZonalStatistics',
    FleetDwell: 'SpatialJoinPrepared TrajectoryMetrics PointInPolygonJoin GroupStatistics',
    FleetDwellZoneEvents: 'SpatialJoinPrepared ZoneEvents GroupStatistics',
    ZonalDistribution: 'SpatialJoinPrepared PointInPolygonJoin GroupStatistics',
    InterpolationDiagnostics: 'SpatialJoinPrepared Kriging',
    RasterZonalDistribution: 'SpatialJoinPrepared PointInPolygonJoin GroupStatistics',
    ExposureByZone: 'SpatialJoinPrepared RasterZonalStatistics',
    RegionalRasterChange: 'SpatialJoinPrepared ChangeDetection RasterZonalStatistics',
    SurfacePreparation: 'SpatialJoinPrepared RasterFillNoData RasterRankFilter',
    DrainageNetwork:
      'SpatialJoinPrepared TerrainFlow TerrainStreamOrder TerrainWatersheds TerrainFlowDistance TerrainStreamNetwork GroupStatistics',
    LocalVariability: 'SpatialJoinPrepared LocalSpatialHeteroscedasticity LocalMoran',
    CellFlowTopology: 'SpatialJoinPrepared H3CellIdentities',
    DasymetricRedistribution:
      'SpatialJoinPrepared RasterFractionalCoverage ArealInterpolation SpatialLag',
    ExcessRisk: 'SpatialJoinPrepared EmpiricalBayesRates',
    SpatialRate: 'SpatialJoinPrepared SpatialEmpiricalBayesRates',
    ContiguityOrderCorrelogram:
      'SpatialJoinPrepared SpatialWeightsAlgebra SpatialWeightsTransform GlobalSpatialStatistics',
    StreetProfiles: '',
    BuildingAdjacency: 'SpatialJoinPrepared PolygonOverlay',
    Isovists: '',
    VisibilityGraph: '',
    BuildingTessellation: 'PolygonTriangulation SpatialJoinPrepared PolygonOverlay',
    BuildingTessellationReference: 'SpatialJoinPrepared PolygonOverlay',
    LocalBuildingTessellation: 'PolygonTriangulation',
    LandscapeMetrics: 'RasterPatchMetrics',
    PatchProximity: '',
    ResistanceConnectivity: '',
    SparseResistanceConnectivity: '',
    SegregationDecomposition: '',
    SegregationCounterfactual: '',
    PopulationSupportAlignment: 'PopulationSupportAlignment',
    RasterResistanceConnectivity: ''
  };
  const network: Record<string, string> = {
    DriveTimeCatchment:
      'SpatialJoinPrepared NetworkSnapping NetworkServiceAreas NetworkIsochrones PointInPolygonJoin GroupStatistics',
    NetworkDisruption: 'SpatialJoinPrepared NetworkCostMatrix NetworkAccessibility',
    FacilityScenario: 'SpatialJoinPrepared NetworkAccessibility',
    TrackQuality: 'SpatialJoinPrepared TrajectoryMetrics TrackSimilarity MapMatching',
    VisibilityGraph: 'NetworkCentrality NetworkReachability'
  };
  for (const [packageName, expected] of [
    [SPATIAL, spatial],
    ['@luma-contributors/analysis/gpu-network', network]
  ] as const) {
    for (const [stem, stages] of Object.entries(expected)) {
      const recipe = getGPUAnalysisOperator({package: packageName, exportName: `add${stem}Recipe`});
      expect(
        recipe?.forms?.recipeStages?.map(stage => stage.exportName),
        `${packageName}:${stem}`
      ).toEqual(stages ? stages.split(' ').map(name => `GPU${name}`) : []);
    }
  }
  const outline = getGPUAnalysisOperator('addClusterAndOutlineRecipe')?.forms?.recipeStages;
  expect(outline?.find(stage => stage.exportName === 'GPUGeometryMeasures')?.appliesWhen).toBe(
    "props.outline?.kind !== 'alpha'"
  );
});
