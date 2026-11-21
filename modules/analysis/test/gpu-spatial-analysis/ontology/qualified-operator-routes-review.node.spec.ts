// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';
import {GPUClassBreaks} from '../../../src/dataframe-analysis/column-classification/gpu-class-breaks';
import {getGPUClassBreaksParameterLength} from '../../../src/dataframe-analysis/column-classification/class-breaks-parameters';
import {GPURankMarkov} from '../../../src/gpu-spatial-analysis/distribution-dynamics/gpu-rank-markov';
import {GPUKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density';
import {GPUPointDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-point-density';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {GPUCellAggregation} from '../../../src/gpu-spatial-analysis/cell-aggregation/gpu-cell-aggregation';
import {GPUNeighborSearch} from '../../../src/gpu-spatial-analysis/neighbor-search/gpu-neighbor-search';
import {GPULineChunk} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-chunk';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {GPUGeographicDistribution} from '../../../src/gpu-spatial-analysis/geographic-distribution/gpu-geographic-distribution';
import {GPUSpatialInteraction} from '../../../src/gpu-spatial-analysis/spatial-interaction/gpu-spatial-interaction';
import {computeOrdinaryLeastSquaresOnCPU} from '../../../src/gpu-spatial-analysis/spatial-regression/cpu-ordinary-least-squares-operation';
import {getTerrainD8TileCommandNodeCount} from '../../../src/gpu-terrain/terrain-processing/terrain-tiled-plan';
import {getScanNearestZoneSearchPlan} from '../../../src/gpu-spatial-analysis/scan-statistics/scan-statistic-nearest-index';
import * as spatial from '../../../src/gpu-spatial-analysis/index';
import * as network from '../../../src/gpu-network/index';
import {GPUTerrainTiledFlow} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-tiled-flow';
import {GPURasterFractionalCategoricalStatistics} from '../../../src/raster-analysis/gpu-raster-fractional-categorical-statistics';
import {createScalabilityProps} from '../../gpu-raster/raster-fractional-scalability/raster-fractional-scalability-fixture';
import {
  getGPUAnalysisOperator,
  describeOutputs
} from '../../../src/gpu-spatial-analysis/ontology/all-entry-points/analysis-operator-catalog';

function fixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'float32' | 'float32x2' | 'uint32' | 'uint32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `route-review-${sequence++}`, format, length);
  return {device, graph, view};
}

const SPATIAL_PACKAGE = '@luma-contributors/analysis/gpu-spatial-analysis' as const;
const NETWORK_PACKAGE = '@luma-contributors/analysis/gpu-network' as const;

it('review B1 resolves every advertised planner by a direct public namespace lookup', () => {
  for (const packageName of [SPATIAL_PACKAGE, NETWORK_PACKAGE]) {
    for (const exportName of ['GPURankMarkov', 'GPUIslandAttachment', 'addVisibilityGraphRecipe']) {
      const record = getGPUAnalysisOperator({package: packageName, exportName});
      const namespace = packageName === SPATIAL_PACKAGE ? spatial : network;
      for (const route of record?.routes ?? []) {
        if (!route.planner) continue;
        const planner = (namespace as Record<string, unknown>)[route.planner.exportName];
        expect(planner, route.planner.exportName).toBeDefined();
        if (packageName === NETWORK_PACKAGE)
          expect(planner).toBe(network.addVisibilityGraphRecipe.executionPlanner);
      }
    }
  }
});

it('review B2 keeps the independent network lane ceiling and scopes automatic search', () => {
  const record = getGPUAnalysisOperator({
    package: NETWORK_PACKAGE,
    exportName: 'addVisibilityGraphRecipe'
  });
  expect(record?.routes?.[0].byteExpression).toContain('min(134217728,');
  expect(record?.routes?.[0].appliesWhen).toContain('pivotCount and pivotNodes are omitted');
  expect(record?.limitation).toContain('pivotCount and pivotNodes are omitted');
  const current = fixture();
  try {
    Object.assign(current.device.limits, {
      maxBufferSize: 268435456,
      maxStorageBufferBindingSize: 268435456
    });
    expect(() =>
      network.addVisibilityGraphRecipe.executionPlanner.getExecutionPlan(
        {
          mode: 'csr',
          positions: current.view('float32x2', 100000),
          segments: createTransientView(current.graph, 'review-b2-segments', 'float32x4', 1),
          edgeCapacity: 1,
          maximumDistance: 1,
          maximumIterations: 1,
          laneCount: 120,
          pivotCount: 120
        },
        current.graph
      )
    ).toThrow(/11184810/); // 120*100000*12 = 144000000 > 134217728.
  } finally {
    current.device.destroy();
  }
});

it('review B3 publishes the source-owned route names and the level-count allocation', () => {
  for (const [exportName, names] of [
    ['GPURankMarkov', ['auto-pairwise-crossover']],
    ['GPUSpatialInteraction', ['auto-dense-pair-crossover']],
    ['GPUGeographicDistribution', ['auto-chunked-crossover']],
    ['GPUSpatialPredicateJoin', ['relate-auto-crossover']],
    [
      'GPUIslandAttachment',
      ['auto-scan-crossover', 'grid-candidates', 'grid-schedule', 'forced-scan-maximum']
    ],
    ['GPUSpatialWeightsTraces', ['auto-exact-crossover', 'forced-exact']],
    ['GPUCellPyramid', ['level-count-slots']]
  ] as const) {
    const routes = getGPUAnalysisOperator(exportName)?.routes;
    for (const name of names)
      expect(
        routes?.find(route => route.name === name),
        `${exportName}:${name}`
      ).toBeDefined();
  }
  expect(
    getGPUAnalysisOperator('GPUIslandAttachment')?.routes?.find(
      route => route.name === 'forced-scan-maximum'
    )?.planner?.exportName
  ).toBe('GPUIslandAttachment');
  const current = fixture();
  try {
    const plan = spatial.GPUCellPyramid.getExecutionPlan(
      {rowCount: 1, levelCapacities: [1, 1], includeLevelCounts: true},
      {...current.device.limits, maxStorageBufferBindingSize: 256}
    );
    expect(
      plan.constraints.find(constraint => constraint.name === 'level count slots')
    ).toMatchObject({received: 512, satisfied: false});
  } finally {
    current.device.destroy();
  }
});

it.each([
  ['GPURankMarkov', 'pairwise'],
  ['GPUSpatialInteraction', 'dense'],
  ['GPUGeographicDistribution', 'chunked-reduction'],
  ['GPUSpatialPredicateJoin', 'relate'],
  ['GPUIslandAttachment', 'auto-scan'],
  ['GPUSpatialWeightsTraces', 'default-exact']
] as const)('omits the retired %s catalog route %s', (exportName, name) => {
  const routes = getGPUAnalysisOperator({package: SPATIAL_PACKAGE, exportName})?.routes;
  expect(routes).toBeDefined();
  expect(routes?.map(route => route.name)).not.toContain(name);
});

it('review B4 pins separately rounded Gaussian support products at the padding boundary', () => {
  const planner = qualifiedPlanner('GPUKernelDensity', spatial.GPUKernelDensity);
  const current = fixture();
  try {
    const plan = planner.getExecutionPlan(
      {
        pointCount: 1,
        gridSize: [2, 2],
        indexGridSize: [2, 2],
        method: 'auto',
        bandwidth: 1.32,
        cellSize: [0.1, 0.1],
        binnedPadding: 32
      },
      current.device.limits
    );
    // f32(5*f32(1.32))=6.600000381469727 > f32(f32(2*33)*f32(0.1))=6.599999904632568.
    expect(plan.route).toBe('exact');
    expect(
      getGPUAnalysisOperator('GPUKernelDensity')?.routes?.find(
        route => route.name === 'auto-method'
      )?.conditionExpression
    ).toContain('support <= f32(f32(2 * binnedMargin) * minimumCell)');
  } finally {
    current.device.destroy();
  }
});

it('review B6 pins robust fit-status precedence and construction-time refusals', () => {
  const text = getGPUAnalysisOperator('GPURestrictedRegimeRegression')?.limitation;
  expect(text).toContain('malformed declared shapes or metadata can throw at construction');
  expect(text).toContain('status 1 takes precedence');
  expect(text).toContain('once the restricted fit and residual degrees are usable');
  const input = {
    predictors: Float32Array.of(0, 1, 2, 3, 0, 1, 2, 3),
    response: Float32Array.of(1, 3, 5, 7, 1, 4, 7, 10),
    regimes: Uint32Array.of(0, 0, 0, 0, 1, 1, 1, 1)
  };
  const options = {
    predictorCount: 1,
    regimeCount: 2,
    common: [true, false],
    robust: {
      type: 'hac' as const,
      kernelWeights: {
        offsets: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7, 8),
        neighbors: Uint32Array.of(0, 1, 2, 3, 4, 5, 6, 7),
        weights: Float32Array.of(1, 1, 1, 1, 1, 1, 1, 1),
        status: {overflow: Uint32Array.of(1)}
      }
    }
  };
  // Three restricted coefficients: a shared intercept and two slopes. Eight rows give df=5;
  // the selected three independent rows give df=0, taking precedence over the same kernel loss.
  expect(spatial.computeRestrictedRegimeRegressionOnCPU(input, options).robust?.status[0]).toBe(2);
  expect(
    spatial.computeRestrictedRegimeRegressionOnCPU(
      {...input, mask: Uint32Array.of(1, 1, 0, 0, 0, 1, 0, 0)},
      options
    ).robust?.status[0]
  ).toBe(1);
});

it('review B7 isolates temporal offsets from timed query-record allocation', () => {
  const planner = qualifiedPlanner('GPUNeighborSearch', spatial.GPUNeighborSearch);
  const shape = {
    mode: 'knn' as const,
    targetRowCount: 2,
    queryRowCount: 2,
    gridSize: [2, 2] as const,
    neighborCount: 2,
    hasTimes: true
  };
  const limits = {maxBufferSize: 64, maxStorageBufferBindingSize: 48};
  expect(planner.getExecutionPlan({...shape, timeBucketCount: 2}, limits).feasible).toBe(true);
  const plan = planner.getExecutionPlan({...shape, timeBucketCount: 3}, limits);
  expect(plan.constraints.filter(constraint => !constraint.satisfied)).toMatchObject([
    {received: 52, maximum: 48}
  ]);
});

/** Resolve a declared planner by its public top-level export name. */
function qualifiedPlanner<Planner>(
  exportName: string,
  expected: Planner,
  networkRecipe = false
): Planner {
  const packageName = networkRecipe ? NETWORK_PACKAGE : SPATIAL_PACKAGE;
  const routes = getGPUAnalysisOperator({package: packageName, exportName})?.routes;
  const identity = routes?.find(route => route.planner)?.planner;
  expect(identity, exportName).toBeDefined();
  expect(identity?.package).toBe(packageName);
  const namespace = networkRecipe ? network : spatial;
  const resolved = identity
    ? (namespace as Record<string, unknown>)[identity.exportName]
    : undefined;
  expect(resolved, exportName).toBe(expected);
  return expected;
}

it('SCL-04 resolves the network visibility adapter independently of spatial adjacency', () => {
  const planner = qualifiedPlanner(
    'addVisibilityGraphRecipe',
    network.GPU_NETWORK_VISIBILITY_GRAPH_PLANNER,
    true
  );
  expect(planner).toBe(network.addVisibilityGraphRecipe.executionPlanner);
  const current = fixture();
  try {
    const props = {
      mode: 'csr' as const,
      positions: current.view('float32x2', 256),
      segments: createTransientView(current.graph, 'visibility-segments', 'float32x4', 1),
      maximumDistance: 1.1,
      edgeCapacity: 1024,
      maximumIterations: 90
    };
    const smallerBatch = planner.getExecutionPlan(
      {...props, laneCount: 1, maximumIterations: 256},
      current.graph
    );
    // 1058 fixed + 23*(5*256+64) = 31970; the next batch needs 33314 > 32768.
    expect(smallerBatch.pivotCount).toBe(23);
    expect(smallerBatch.estimatedCommandNodeCount).toBe(31970);
    // Supplying either explicit spelling must preserve all 32 pivots, even when only 23 fit.
    expect(() =>
      planner.getExecutionPlan(
        {...props, laneCount: 1, maximumIterations: 256, pivotCount: 32},
        current.graph
      )
    ).toThrow(/command node count/);
    expect(() =>
      planner.getExecutionPlan(
        {...props, laneCount: 1, maximumIterations: 256, pivotNodes: current.view('uint32', 32)},
        current.graph
      )
    ).toThrow(/command node count/);
    Object.assign(current.device.limits, {maxStorageBufferBindingSize: 24000});
    const plan = planner.getExecutionPlan(props, current.graph);
    expect(plan.feasible).toBe(true);
    expect(plan.route).toBe('sampled');
    expect(plan.pivotCount).toBe(32);
    expect(plan.laneCount).toBe(4); // floor(24000/(256*12))=7; four divides the sample of 32.
    Object.assign(current.device.limits, {maxStorageBufferBindingSize: 16000});
    expect(() => planner.getExecutionPlan(props, current.graph)).toThrow(/18440.*16000/);
    expect(network.addVisibilityGraphRecipe).not.toBe(spatial.addVisibilityGraphRecipe);
    const record = getGPUAnalysisOperator({
      package: NETWORK_PACKAGE,
      exportName: 'addVisibilityGraphRecipe'
    });
    expect(record?.limitation).toContain('lane divisibility');
    expect(record?.routes?.[0].appliesWhen).toContain('smaller feasible whole batches');
  } finally {
    current.device.destroy();
  }
});

it('SCL-14 resolves source join and neighbor planners with whole-domain and device boundaries', () => {
  const prepared = qualifiedPlanner(
    'GPUSpatialJoinPrepared',
    spatial.GPU_SPATIAL_JOIN_PREPARED_PLANNER
  );
  const predicate = qualifiedPlanner(
    'GPUSpatialPredicateJoin',
    spatial.GPU_SPATIAL_PREDICATE_JOIN_PLANNER
  );
  const neighbor = qualifiedPlanner('GPUNeighborSearch', spatial.GPUNeighborSearch);
  const limits = {maxBufferSize: 134217728, maxStorageBufferBindingSize: 134217728};
  for (const featureCount of [8388608, 8388609]) {
    const plan = prepared.getExecutionPlan(
      {kind: 'points', featureCount, vertexCount: featureCount, spatialSort: false},
      limits
    );
    // Full padded tree, independent of input partitioning: (2*L-1)*8 bytes per bounds/children binding.
    expect(plan.feasible).toBe(featureCount === 8388608);
  }
  for (const [leftVertexCount, rightVertexCount, route] of [
    [31, 33, 'fast'],
    [32, 32, 'relate']
  ] as const)
    expect(
      predicate.getExecutionPlan(
        {
          predicate: 'intersects',
          leftKind: 'lines',
          rightKind: 'lines',
          leftCount: 1,
          rightCount: 1,
          leftVertexCount,
          rightVertexCount
        },
        undefined
      ).route
    ).toBe(route);
  const shape = {
    mode: 'knn' as const,
    targetRowCount: 2,
    queryRowCount: 4,
    gridSize: [2, 2] as const,
    neighborCount: 2
  };
  expect(
    neighbor.getExecutionPlan(shape, {maxBufferSize: 64, maxStorageBufferBindingSize: 48}).feasible
  ).toBe(true);
  expect(
    neighbor.getExecutionPlan(
      {...shape, neighborCount: 4},
      {maxBufferSize: 64, maxStorageBufferBindingSize: 48}
    ).feasible
  ).toBe(false); // 4*4*4 = 64-byte list.
  expect(() => neighbor.getExecutionPlan({...shape, neighborCount: 257}, limits)).toThrow(/256/);
  expect(
    neighbor.getExecutionPlan({...shape, hasTimes: true, timeBucketCount: 65536}, limits).feasible
  ).toBe(true);
  expect(() =>
    neighbor.getExecutionPlan({...shape, hasTimes: true, timeBucketCount: 65537}, limits)
  ).toThrow(/65536/);
});

it('SCL-16 resolves five statistical planners and preserves forced, binding and schedule limits', () => {
  const island = qualifiedPlanner('GPUIslandAttachment', spatial.GPUIslandAttachment);
  const traces = qualifiedPlanner('GPUSpatialWeightsTraces', spatial.GPUSpatialWeightsTraces);
  const rank = qualifiedPlanner('GPURankMarkov', spatial.GPURankMarkov);
  const geographic = qualifiedPlanner(
    'GPUGeographicDistribution',
    spatial.GPUGeographicDistribution
  );
  const interaction = qualifiedPlanner('GPUSpatialInteraction', spatial.GPUSpatialInteraction);
  const current = fixture();
  try {
    const view = current.view;
    const weights = (rows: number) => ({
      offsets: view('uint32', rows + 1),
      neighbors: view('uint32', rows),
      weights: view('float32', rows)
    });
    const islandProps = (rows: number) => ({
      weights: weights(rows),
      positions: view('float32x2', rows),
      output: {
        offsets: view('uint32', rows + 1),
        neighbors: view('uint32', rows * 3),
        weights: view('float32', rows * 3),
        attachment: view('uint32', rows),
        added: view('uint32', rows * 3),
        status: {overflow: view('uint32', 1)}
      }
    });
    expect(island.getExecutionPlan(islandProps(4096)).route).toBe('scan');
    expect(island.getExecutionPlan(islandProps(4097)).route).toBe('grid');
    expect(island.getExecutionPlan({...islandProps(65535), nearestSearch: 'scan'}).feasible).toBe(
      true
    );
    expect(island.getExecutionPlan({...islandProps(65536), nearestSearch: 'scan'}).feasible).toBe(
      false
    );
    // Exact fallback batches jump from 1536*6 to 1536*7; 4096 + 3*batches exceeds 32768.
    expect(island.getExecutionPlan(islandProps(100663290)).feasible).toBe(true);
    expect(island.getExecutionPlan(islandProps(100663291)).feasible).toBe(false);
    const traceProps = (rows: number) => ({
      weights: weights(rows),
      maximumPower: 2,
      output: {powerTraces: view('float32', 2), powerSums: view('float32', 2)}
    });
    expect(traces.getExecutionPlan(traceProps(4096)).route).toBe('exact');
    expect(traces.getExecutionPlan(traceProps(4097)).route).toBe('stochastic');
    expect(traces.getExecutionPlan({...traceProps(16384), mode: 'exact'}).feasible).toBe(true);
    expect(traces.getExecutionPlan({...traceProps(16385), mode: 'exact'}).feasible).toBe(false);
    expect(
      traces.getExecutionPlan(traceProps(8), {maxBufferSize: 8, maxStorageBufferBindingSize: 8})
        .feasible
    ).toBe(false);
    const rankProps = (rows: number) => ({
      values: view('float32', rows * 2),
      periodCount: 2,
      output: {ranks: view('uint32', rows * 2), counts: view('uint32', rows * rows)}
    });
    expect(rank.getExecutionPlan(rankProps(64)).route).toBe('pairwise');
    expect(rank.getExecutionPlan(rankProps(65)).route).toBe('sorted');
    expect(rank.getExecutionPlan({...rankProps(65), rankingMethod: 'pairwise'}).feasible).toBe(
      false
    );
    const shifted = current.graph.createDataView(view('float32', 40).buffer, {
      format: 'float32',
      byteOffset: 128,
      length: 6
    });
    expect(
      rank.getExecutionPlan(
        {...rankProps(3), values: shifted},
        {maxBufferSize: 256, maxStorageBufferBindingSize: 64}
      ).feasible
    ).toBe(false);
    const geographicProps = (groups: number) => ({
      positions: view('float32x2', 8),
      groupIds: view('uint32', 8),
      groupCount: groups,
      parameters: view('float32', 8),
      output: {counts: view('uint32', groups)}
    });
    expect(geographic.getExecutionPlan(geographicProps(64)).route).toBe('chunked');
    expect(geographic.getExecutionPlan(geographicProps(65)).route).toBe('sorted');
    expect(
      geographic.getExecutionPlan({...geographicProps(256), reduction: 'chunked'}).feasible
    ).toBe(true);
    expect(
      geographic.getExecutionPlan({...geographicProps(257), reduction: 'chunked'}).feasible
    ).toBe(false);
    const interactionProps = (pairs: number) => ({
      origins: view('uint32', pairs),
      destinations: view('uint32', pairs),
      flows: view('float32', pairs),
      covariate: view('float32', pairs),
      originCount: 2,
      destinationCount: 2,
      constraint: 'doubly' as const,
      coefficients: [0],
      maximumIterations: 2,
      output: {
        predictions: view('float32', pairs),
        summary: view('float32', 4),
        status: view('uint32', 1),
        candidates: view('float32', 4)
      }
    });
    expect(interaction.getExecutionPlan(interactionProps(4096)).route).toBe('dense');
    expect(interaction.getExecutionPlan(interactionProps(4097)).route).toBe('sparse');
    expect(
      interaction.getExecutionPlan({...interactionProps(4097), method: 'dense'}).feasible
    ).toBe(false);
    expect(
      interaction.getExecutionPlan(interactionProps(8), {
        maxBufferSize: 8,
        maxStorageBufferBindingSize: 8
      }).feasible
    ).toBe(false);
    const boundedBase = interactionProps(16);
    const bounded = {
      ...boundedBase,
      calibration: 'bounded' as const,
      coefficients: [-1, 1],
      output: {...boundedBase.output, candidates: view('float32', 8)}
    };
    // Candidate predictions include the extra bounded-fit word: (16+1)*2*4 = 136 bytes.
    expect(
      interaction.getExecutionPlan(bounded, {maxBufferSize: 1024, maxStorageBufferBindingSize: 136})
        .feasible
    ).toBe(true);
    const refused = interaction.getExecutionPlan(bounded, {
      maxBufferSize: 1024,
      maxStorageBufferBindingSize: 135
    });
    expect(refused.feasible).toBe(false);
    expect(
      refused.constraints.find(constraint => constraint.quantity === 'candidate predictions')
        ?.received
    ).toBe(136);
  } finally {
    current.device.destroy();
  }
});

it('W5-04 resolves four family planners and distinguishes host admission from frame eligibility', () => {
  const pyramid = qualifiedPlanner('GPUCellPyramid', spatial.GPUCellPyramid);
  const point = qualifiedPlanner('GPUPointDensity', spatial.GPUPointDensity);
  const kernel = qualifiedPlanner('GPUKernelDensity', spatial.GPUKernelDensity);
  const scan = qualifiedPlanner(
    'GPUExpectationBasedSpaceTimeScan',
    spatial.GPUExpectationBasedSpaceTimeScan
  );
  const current = fixture();
  try {
    const limits = current.device.limits;
    const shape = {rowCount: 128, levelCapacities: [1]};
    expect(pyramid.getExecutionPlan(shape, limits).feasible).toBe(true);
    expect(
      pyramid.getExecutionPlan(shape, {...limits, maxStorageBufferBindingSize: 8}).feasible
    ).toBe(false);
    const pointShape = {
      pointCount: 1,
      gridSize: [2, 2] as const,
      weighted: true,
      statistic: 'sum' as const
    };
    // Four fixed-point cells require 4*12=48 bytes, independent of input position bytes.
    expect(
      point.getExecutionPlan(pointShape, {maxBufferSize: 1024, maxStorageBufferBindingSize: 48})
        .feasible
    ).toBe(true);
    expect(
      point.getExecutionPlan(pointShape, {maxBufferSize: 1024, maxStorageBufferBindingSize: 47})
        .feasible
    ).toBe(false);
    const kernelShape = {
      pointCount: 1,
      gridSize: [2, 2] as const,
      indexGridSize: [2, 2] as const,
      method: 'exact' as const
    };
    expect(kernel.getExecutionPlan(kernelShape, limits).feasible).toBe(true);
    expect(
      kernel.getExecutionPlan(kernelShape, {maxBufferSize: 16, maxStorageBufferBindingSize: 16})
        .feasible
    ).toBe(false);
    expect(kernel.getExecutionPlan({...kernelShape, method: 'auto'}, limits).route).toBe('auto');
    expect(
      kernel.getExecutionPlan(
        {...kernelShape, method: 'auto', binnedPadding: 32, bandwidth: 10, cellSize: [1, 1]},
        limits
      ).route
    ).toBe('binned');
    expect(
      kernel.getExecutionPlan(
        {...kernelShape, method: 'auto', binnedPadding: 2, bandwidth: 10, cellSize: [1, 1]},
        limits
      ).route
    ).toBe('exact');
    const scanShape = {
      zones: 2,
      timeBuckets: 2,
      maximumNeighbors: 1,
      maximumWindowLength: 1,
      maximumReplicates: 1
    };
    expect(scan.getExecutionPlan(scanShape, limits).route).toBe('indexed');
    expect(scan.getExecutionPlan({...scanShape, maximumNeighbors: 2}, limits).route).toBe(
      'full-scan'
    );
    expect(
      scan.getExecutionPlan(scanShape, {maxBufferSize: 16, maxStorageBufferBindingSize: 16})
        .feasible
    ).toBe(false);
    expect(
      getGPUAnalysisOperator('GPUKernelDensity')?.routes?.find(
        route => route.name === 'auto-method'
      )?.appliesWhen
    ).toContain('per-frame GPU gate');
  } finally {
    current.device.destroy();
  }
});

it('SCL-06 publishes saturated sticky streaming status', () => {
  const streaming = getGPUAnalysisOperator('GPURasterStreamingAccumulator');
  expect(streaming?.limitation).toContain('saturate');
  expect(streaming?.limitation).toContain('state.status');
  expect(streaming?.limitation).toContain('until reset');
  expect(streaming?.limitation).toContain('maximumStreamSliceCount defaults to 16,777,216 (2^24)');
  expect(streaming?.limitation).toContain('0 complete, 1 observation-count overflow');
  expect(streaming?.limitation).toContain('4,294,967,295 (2^32 - 1)');
  expect(streaming?.summary).toContain('saturate');
  expect(streaming?.summary).toContain('state.status until reset');
});

it('W5-05 includes the missing category in both capacity bounds', () => {
  const categorical = getGPUAnalysisOperator('GPURasterFractionalCategoricalStatistics');
  for (const mode of ['tiled', 'exact'])
    expect(
      categorical?.limits?.find(
        limit => limit.name === `zoneCount * (categoryCapacity + 1) (${mode})`
      )
    ).toBeDefined();
  expect(
    categorical?.limits?.find(limit => limit.name === 'zoneCount * (categoryCapacity + 1) (exact)')
      ?.maximum
  ).toBe(65535);
  expect(
    categorical?.limits?.find(limit => limit.name === 'zoneCount * (categoryCapacity + 1) (tiled)')
      ?.maximum
  ).toBe(4294967294);
  const current = fixture();
  try {
    for (const zoneCount of [21845, 21846]) {
      const props = createScalabilityProps(
        new GPUCommandGraph(current.device),
        zoneCount
      ).categorical;
      // categoryCapacity=2: 21845*3=65535 admits, 21846*3=65538 rejects, although 21846*2=43692.
      expect(
        GPURasterFractionalCategoricalStatistics.getExecutionPlan(
          {...props, mode: 'exact'},
          current.device.limits
        ).feasible
      ).toBe(zoneCount === 21845);
    }
  } finally {
    current.device.destroy();
  }
});

it('W5-06 publishes the conditional robust score and HAC contracts', () => {
  const restricted = getGPUAnalysisOperator('GPURestrictedRegimeRegression');
  expect(restricted?.limitation).toContain('rows * (restrictedCoefficientCount + 1) * 4');
  expect(restricted?.limitation).toContain('original-row');
  expect(restricted?.limitation).toContain('HC0/HC1');
  expect(restricted?.limitation).toContain('status 2');
  expect(restricted?.limitation).toContain(
    'symmetric finite nonnegative unstandardized kernel weights'
  );
  expect(restricted?.limitation).toContain('unit diagonal including masked rows');
  expect(restricted?.limitation).toContain('classical output.chow remains the nested-model Chow F');
  expect(restricted?.limitation).toContain('Wald chi-square');
  expect(restricted?.limitation).toContain('HC1 when smallSample is true');
  expect(restricted?.limitation).toContain('rows is response.length, including masked rows');
});

it('B5 checks explicit class-breaks storage admission below, at and above both enforced bounds', () => {
  for (const [storage, maximum] of [
    ['materialized', 1024],
    ['recompute', 8192]
  ] as const) {
    const current = fixture();
    try {
      const construct = (rows: number, natural = true) =>
        new GPUClassBreaks({
          values: current.view('float32', rows),
          parameters: current.view('float32', getGPUClassBreaksParameterLength(6)),
          maximumClassCount: 6,
          methods: natural ? ['natural-breaks'] : ['equal-interval'],
          naturalBreaksMode: 'exact',
          naturalBreaksExactStorage: storage,
          output: {breaks: current.view('float32', 7), classCount: current.view('uint32', 1)}
        });
      expect(() => construct(maximum - 1)).not.toThrow();
      expect(() => construct(maximum)).not.toThrow();
      expect(() => construct(maximum + 1)).toThrow(/values rows/);
      expect(() => construct(maximum + 1, false)).not.toThrow();
      for (const rowCount of [maximum - 1, maximum, maximum + 1]) {
        for (const natural of [false, true]) {
          const plan = GPUClassBreaks.getExecutionPlan(
            {
              rowCount,
              maximumClassCount: 6,
              methods: natural ? ['natural-breaks'] : ['equal-interval'],
              naturalBreaksMode: 'exact',
              naturalBreaksExactStorage: storage
            },
            current.device.limits
          );
          expect(plan.feasible).toBe(!natural || rowCount <= maximum);
        }
      }
      expect(
        getGPUAnalysisOperator('GPUClassBreaks')?.routes?.find(route => route.name === storage)
      ).toMatchObject({maximum, appliesWhen: expect.stringContaining('natural-breaks enabled')});
    } finally {
      current.device.destroy();
    }
  }
});

it('B3 compares LineChunk description with substring construction on both sides of the uint32 product', () => {
  const current = fixture();
  try {
    const construct = (lineCount: number) =>
      new GPULineChunk({
        positions: current.view('float32x2', 2 * lineCount),
        lineOffsets: current.view('uint32', lineCount + 1),
        parameters: current.view('float32', 4),
        mode: 'substring',
        output: {
          positions: current.view('float32x2', 8),
          lineOffsets: current.view('uint32', lineCount + 1),
          status: {count: current.view('uint32', 1), overflow: current.view('uint32', 1)}
        }
      });
    const identity = {
      package: '@luma-contributors/analysis/gpu-spatial-analysis',
      exportName: 'GPULineChunk'
    } as const;
    for (const lineCount of [65534, 65535]) {
      expect(() => construct(lineCount)).not.toThrow();
      expect(() =>
        describeOutputs(identity, 'substring', {
          lineCount,
          positionCount: 2 * lineCount,
          vertexCapacity: 8
        })
      ).not.toThrow();
    }
    expect(() => construct(65536)).toThrow(/32 bits/);
    expect(() =>
      describeOutputs(identity, 'substring', {lineCount: 65536, vertexCapacity: 8})
    ).toThrow(/lineCount/);
  } finally {
    current.device.destroy();
  }
});

it('B7 checks automatic rank routes below, at and above the 64-row crossover', () => {
  const current = fixture();
  try {
    for (const rows of [63, 64, 65, 72]) {
      const props = {
        values: current.view('float32', rows * 3),
        periodCount: 3,
        output: {
          counts: current.view('uint32', rows * rows),
          ranks: current.view('uint32', rows * 3)
        }
      };
      expect(new GPURankMarkov(props).rankingMethod).toBe(rows <= 64 ? 'pairwise' : 'sorted');
      if (rows > 64)
        expect(() => new GPURankMarkov({...props, rankingMethod: 'pairwise'})).toThrow(/rowCount/);
    }
  } finally {
    current.device.destroy();
  }
});

it('B4 checks the shared KDE index binding independently of output dimensions and method', () => {
  const current = fixture();
  try {
    for (const method of ['exact', 'binned'] as const) {
      for (const [width, shouldFit] of [
        [8191, true],
        [8192, false]
      ] as const) {
        const density = new GPUKernelDensity({
          id: `kde-${method}-${width}`,
          positions: current.view('float32x2', 1),
          parameters: current.view('float32', 9),
          grid: makeExtentGrid(3, 1, [0, 0, 3, 1]),
          indexGridSize: [width, 4096],
          indexBounds: [0, 0, 3, 1],
          method,
          output: {values: current.view('float32', 3)}
        });
        if (shouldFit) expect(() => density.getCommandNodes(current.graph)).not.toThrow();
        else expect(() => density.getCommandNodes(current.graph)).toThrow(/indexGridSize cells/);
      }
    }
  } finally {
    current.device.destroy();
  }
});

it('UNFINISHED SCL-16 describes the actual neighbor and scan automatic decisions', () => {
  const neighbor = getGPUAnalysisOperator('GPUNeighborSearch')?.routes?.find(
    route => route.name === 'auto-grid'
  );
  expect(neighbor?.conditionExpression).toContain('auto => grid');
  const scan = getGPUAnalysisOperator('GPUExpectationBasedSpaceTimeScan')?.routes?.find(
    route => route.name === 'indexed-search'
  );
  expect(scan?.appliesWhen).toContain('maximumNeighbors < zones');
  expect(scan?.maximum).toBe(65);
  // Helper is the constructor's exact shared resolver. For expectation scan listLength=maximumNeighbors.
  for (const [zones, length, expected] of [
    [65, 64, 'indexed'],
    [64, 64, 'full-scan'],
    [66, 65, 'indexed'],
    [67, 66, 'full-scan']
  ] as const)
    expect(
      getScanNearestZoneSearchPlan('catalog-route-review', zones, length).nearestZoneSearch
    ).toBe(expected);
});

it('SCL-16 compares geographic and interaction automatic crossovers with construction', () => {
  const current = fixture();
  try {
    for (const groups of [63, 64, 65]) {
      const distribution = new GPUGeographicDistribution({
        positions: current.view('float32x2', 1),
        groupIds: current.view('uint32', 1),
        groupCount: groups,
        parameters: current.view('float32', 64),
        output: {counts: current.view('uint32', groups)}
      });
      expect(distribution.reduction).toBe(groups <= 64 ? 'chunked' : 'sorted');
    }
    for (const pairs of [4095, 4096, 4097]) {
      const interaction = new GPUSpatialInteraction({
        origins: current.view('uint32', pairs),
        destinations: current.view('uint32', pairs),
        originCount: 64,
        destinationCount: 64,
        flows: current.view('float32', pairs),
        covariate: current.view('float32', pairs),
        constraint: 'doubly',
        coefficients: [0],
        maximumIterations: 1,
        output: {
          predictions: current.view('float32', pairs),
          summary: current.view('float32', 4),
          status: current.view('uint32', 1),
          candidates: current.view('float32', 4)
        }
      });
      expect(interaction.method).toBe(pairs <= 4096 ? 'dense' : 'sparse');
    }
  } finally {
    current.device.destroy();
  }
});

it('SCL-16 compares prepared BVH padded bindings at the power-of-two boundary', () => {
  for (const features of [8388607, 8388608, 8388609]) {
    const current = fixture();
    try {
      const prepared = new GPUSpatialJoinPrepared({
        geometry: {kind: 'points', positions: current.view('float32x2', features)},
        spatialSort: false
      });
      const expectedLeaves = features <= 8388608 ? 8388608 : 16777216;
      expect(prepared.leafCapacity).toBe(expectedLeaves);
      // Complete binary tree: (2L - 1) nodes, eight bytes in each bounds/children binding.
      expect((2 * expectedLeaves - 1) * 8).toBe(features <= 8388608 ? 134217720 : 268435448);
      if (features <= 8388608) expect(() => prepared.getCommandNodes(current.graph)).not.toThrow();
      else
        expect(() => prepared.getCommandNodes(current.graph)).toThrow(/BVH node bounds\/children/);
    } finally {
      current.device.destroy();
    }
  }
});

it('SCL-15 charges only the selected terrain route and retains the tiled worst case', () => {
  expect(getTerrainD8TileCommandNodeCount(6450)).toBe(32762);
  expect(getTerrainD8TileCommandNodeCount(6451)).toBe(32767);
  expect(getTerrainD8TileCommandNodeCount(6452)).toBe(32772);
  const route = getGPUAnalysisOperator('GPUTerrainTiledFlow')?.routes?.find(
    route => route.name === 'tiled-schedule'
  );
  expect(route?.maximum).toBe(32768);
  expect(route?.appliesWhen).toContain('selected tiled');
  expect(route?.appliesWhen).toContain('forced resident');
  expect(route?.accuracyImplications).toContain('whole-graph command-node');
  const current = fixture();
  try {
    const props = (count: number) => ({
      grid: {
        width: count,
        height: 1,
        affine: [1, 0, 0, 0, 1, 0] as const,
        pixelInterpretation: 'area' as const
      },
      maximumIterations: 1,
      tiles: Array.from({length: count}, (_, column) => ({
        column,
        row: 0,
        width: 1,
        height: 1,
        owned: [0, 0, 1, 1] as const,
        flowDirections: current.view('uint32', 1),
        runoff: current.view('float32', 1),
        streams: current.view('uint32', 1),
        output: {
          accumulation: current.view('float32', 1),
          status: current.view('uint32', 1),
          reachIds: current.view('uint32', 1)
        }
      }))
    });
    const input = props(6452);
    for (const [execution, estimate, worstCase] of [
      ['auto', 42, 567],
      ['resident', 42, 42],
      ['tiled', 567, 567]
    ] as const) {
      const contributor = new GPUTerrainTiledFlow({...props(11), maximumIterations: 2, execution});
      // Resident 8+2*11+2+ceil(log2(11))+6=42; tiled 512+5*11=567.
      expect(contributor.estimatedCommandNodeCount).toBe(estimate);
      expect(contributor.worstCaseCommandNodeCount).toBe(worstCase);
    }
    const branchingProps = {...props(11), maximumIterations: 2, routing: 'branching' as const};
    branchingProps.tiles = branchingProps.tiles.map(tile => ({
      ...tile,
      fractions: current.view('float32', 8)
    }));
    const branching = new GPUTerrainTiledFlow(branchingProps);
    expect(branching.selectedExecution).toBe('resident');
    expect(branching.worstCaseCommandNodeCount).toBe(branching.estimatedCommandNodeCount);
    expect(() => new GPUTerrainTiledFlow({...branchingProps, execution: 'tiled'})).toThrow(
      /requires routing 'd8'/
    );
    const resident = new GPUTerrainTiledFlow(input);
    // 8 + 2*6452 + 1 + ceil(log2(6452)) + 6 = 12932, versus 512 + 5*6452 = 32772.
    expect(resident.getExecutionPlan(current.device).selectedExecution).toBe('resident');
    expect(resident.estimatedCommandNodeCount).toBe(12932);
    expect(resident.worstCaseCommandNodeCount).toBe(32772);
    expect(() => new GPUTerrainTiledFlow({...input, execution: 'tiled'})).toThrow(/32768/);
    expect(
      new GPUTerrainTiledFlow({...props(6451), execution: 'tiled'}).estimatedCommandNodeCount
    ).toBe(32767);
    Object.assign(current.device.limits, {maxStorageBufferBindingSize: 206464});
    expect(new GPUTerrainTiledFlow(input).estimatedCommandNodeCount).toBe(12932); // 6452*32 bytes exactly.
    Object.assign(current.device.limits, {maxStorageBufferBindingSize: 206463});
    expect(() => new GPUTerrainTiledFlow(input)).toThrow(/32768/);
  } finally {
    current.device.destroy();
  }
});

it('S1 pins prediction for a finite masked predictor row with missing response', () => {
  // Exact line y = 1 + 2x, trained on x=0..3. The held-out x=4 predicts 9.
  const result = computeOrdinaryLeastSquaresOnCPU(
    {
      predictors: new Float32Array([0, 1, 2, 3, 4]),
      response: new Float32Array([1, 3, 5, 7, NaN]),
      mask: new Uint32Array([1, 1, 1, 1, 0])
    },
    1
  );
  expect(result.status[0]).toBe(0);
  expect(Number.isNaN(result.fitted[4])).toBe(true);
  expect(Number.isFinite(result.predictions[4])).toBe(true);
  expect(result.predictions[4]).toBe(9);
});

it('B6 compares the relate crossover and multipart predicate/output overrides with construction', () => {
  const current = fixture();
  try {
    const construct = (
      leftVertices: number,
      rightVertices: number,
      multipart = false,
      predicate: 'intersects' | 'touches' = 'intersects',
      relate = false
    ) =>
      new GPUSpatialPredicateJoin({
        left: {
          kind: 'lines',
          positions: current.view('float32x2', leftVertices),
          lineOffsets: current.view('uint32', multipart ? 3 : 2),
          ...(multipart ? {featureOffsets: current.view('uint32', 2)} : {})
        },
        right: {
          kind: 'lines',
          positions: current.view('float32x2', rightVertices),
          lineOffsets: current.view('uint32', multipart ? 3 : 2),
          ...(multipart ? {featureOffsets: current.view('uint32', 2)} : {})
        },
        predicate,
        output: {
          pairs: {
            leftRows: current.view('uint32', 1),
            rightRows: current.view('uint32', 1),
            status: {count: current.view('uint32', 1), overflow: current.view('uint32', 1)}
          },
          ...(relate ? {relate: current.view('uint32', 1)} : {})
        }
      });
    expect(construct(31, 33).usesRelateEngine).toBe(false); // 1023
    expect(construct(32, 32).usesRelateEngine).toBe(true); // 1024
    expect(construct(41, 25).usesRelateEngine).toBe(true); // 1025
    expect(construct(4, 4, true).usesRelateEngine).toBe(false);
    expect(construct(4, 4, true, 'intersects', true).usesRelateEngine).toBe(true);
    expect(construct(4, 4, true, 'touches').usesRelateEngine).toBe(true);
  } finally {
    current.device.destroy();
  }
});

it('SCL-16 compares first-level pyramid scratch and point-density automatic sums with construction', () => {
  const current = fixture();
  try {
    for (const rows of [127, 128, 129]) {
      // CellPyramid delegates first-level scratch selection to this exact contributor.
      const aggregation = new GPUCellAggregation({
        family: 'quadbin',
        resolution: 5,
        positions: current.view('float32x2', rows),
        scratchTileSize: 128,
        output: {
          cells: current.view('uint32x2', 1),
          counts: current.view('uint32', 1),
          status: {count: current.view('uint32', 1), overflow: current.view('uint32', 1)}
        }
      });
      expect(aggregation.usesTiledScratch).toBe(rows > 128);
    }
    for (const sumAccumulation of ['auto', 'workgroup', 'atomic'] as const) {
      const density = new GPUPointDensity({
        positions: current.view('float32x2', 1),
        weights: current.view('float32', 1),
        bounds: [0, 0, 2, 2],
        gridSize: [2, 2],
        sumAccumulation,
        output: {values: current.view('float32', 4)}
      });
      expect(density.sumAccumulation).toBe(sumAccumulation === 'atomic' ? 'atomic' : 'workgroup');
    }
  } finally {
    current.device.destroy();
  }
});

it('SCL-16 confirms neighbor auto declares the same grid route as explicit grid', () => {
  const plans: string[][] = [];
  for (const searchStructure of ['auto', 'grid'] as const) {
    const current = fixture();
    try {
      const search = new GPUNeighborSearch({
        id: 'neighbor-route',
        positions: current.view('float32x2', 2),
        parameters: current.view('float32', 64),
        gridSize: [2, 2],
        mode: 'radius',
        searchStructure,
        output: {
          offsets: current.view('uint32', 3),
          neighbors: current.view('uint32', 4),
          weights: current.view('float32', 4),
          status: {overflow: current.view('uint32', 1)}
        }
      });
      plans.push(search.getCommandNodes(current.graph).map(node => node.id));
    } finally {
      current.device.destroy();
    }
  }
  expect(plans[0].length).toBeGreaterThan(0);
  expect(plans[0]).toEqual(plans[1]);
});
