// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUExpectationBasedSpaceTimeScan,
  GPULocationChangePoints,
  GPUPettittTest,
  GPUSpaceTimeCrossCorrelation,
  GPUSpaceTimeLocalMoran,
  GPU_EXPECTATION_BASED_SCAN_CLUSTER,
  GPU_EXPECTATION_BASED_SCAN_SUMMARY,
  getGPUExpectationBasedScanParameterValues,
  getGPUPermutationParameterValues,
  type GPULocationChangePointCost,
  type GPUSpaceTimeLocalMoranVariant
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import {withCatalogGpuSession} from './catalog-gpu-session';
import type {Inputs, Values} from './catalog-platform3-b-common';
import {
  finiteSpacetimeValue,
  getSpacetimeSummary,
  makeSpacetimeNeighbors,
  makeSpacetimePointTable,
  makeSpacetimeSummaryTable,
  readSpacetimePoints,
  readSpacetimeSeries,
  uploadSpacetimeNeighbors
} from './catalog-spacetime-common';

type SeriesKind = 'scan' | 'correlation' | 'moran' | 'pettitt' | 'changes';

export async function runSpacetimeSeries(
  kind: SeriesKind,
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const points = readSpacetimePoints(inputs['points'], values);
  if (
    kind === 'scan' &&
    points.collection.features.some(feature => {
      const count = feature.properties?.[String(values['valueProperty'])];
      return typeof count !== 'number' || !Number.isInteger(count) || count < 0 || count > 16777215;
    })
  )
    throw new Error('Space-time scan requires integer counts 0 to 16777215');
  const series = readSpacetimeSeries(points, values, kind === 'correlation', kind === 'scan');
  const {cellCount, periodCount} = series;
  const permutations = Number(values['permutations'] ?? 0);
  const seed = Number(values['seed'] ?? 0);
  const lag = Number(values['lag'] ?? 1);
  const summary: CatalogRunOutput['summary'] = {
    featureCount: points.count,
    locationCount: cellCount,
    periodCount,
    periodIntervalSeconds: series.intervalSeconds,
    timeOrigin: points.timeOrigin,
    timeUnit: String(values['timeUnit']),
    permutations,
    seed,
    maximumFeatures: 512,
    maximumLocations: 64,
    maximumPeriods: 128
  };
  const expand = (entries: ArrayLike<number>) =>
    series.sourceCells.map(cell => finiteSpacetimeValue(entries[cell]));
  if (kind === 'pettitt') {
    const result = await withCatalogGpuSession(context.device, 'catalog-pettitt', async session => {
      const statistic = session.output('statistic', 'uint32', cellCount);
      const changeSlice = session.output('change', 'uint32', cellCount);
      const pValues = session.output('p-values', 'float32', cellCount);
      session.graph.add(
        new GPUPettittTest({
          values: session.input('cube', series.cube, 'float32'),
          layout: 'cell-major',
          cellCount,
          sliceCount: periodCount,
          output: {statistic: statistic.view, changeSlice: changeSlice.view, pValues: pValues.view}
        })
      );
      session.run();
      return {
        statistic: await session.readUint32(statistic),
        changeSlice: await session.readUint32(changeSlice),
        pValues: await session.readFloat32(pValues)
      };
    });
    const tested = Array.from(
      result.changeSlice,
      (slice, cell) => slice < periodCount && Number.isFinite(result.pValues[cell])
    );
    const testedPValues = Array.from(result.pValues).filter((_, cell) => tested[cell]);
    const testedStatistics = Array.from(result.statistic).filter((_, cell) => tested[cell]);
    const hasChange = (cell: number) =>
      tested[cell] && result.statistic[cell] > 0 && result.changeSlice[cell] < periodCount;
    Object.assign(summary, {
      testedLocations: testedPValues.length,
      statisticMaximum: testedStatistics.length ? Math.max(...testedStatistics) : 'unavailable',
      pValueMinimum: testedPValues.length ? Math.min(...testedPValues) : 'unavailable',
      inference: 'Pettitt asymptotic two-sided p-values; no multiple testing correction'
    });
    return {
      tables: [
        makeSpacetimePointTable(points, {
          statistic: series.sourceCells.map(cell => (tested[cell] ? result.statistic[cell] : null)),
          change_slice: series.sourceCells.map(cell =>
            hasChange(cell) ? result.changeSlice[cell] : null
          ),
          change_time: series.sourceCells.map(cell =>
            hasChange(cell) ? (series.periods[result.changeSlice[cell]] ?? null) : null
          ),
          p_value: series.sourceCells.map(cell => (tested[cell] ? result.pValues[cell] : null)),
          segment: series.sourceCells.map((cell, source) =>
            hasChange(cell)
              ? Number(series.sourcePeriods[source] >= result.changeSlice[cell])
              : null
          )
        })
      ],
      diagnostics: [],
      summary,
      dispatchedContributors: ['GPUPettittTest']
    };
  }
  if (kind === 'changes') {
    const cost = values['cost'] as GPULocationChangePointCost;
    const minimumSegmentLength = Number(values['minimumSegmentLength']);
    const smallest = cost === 'slope' ? 3 : cost === 'variance' || cost === 'mean-variance' ? 2 : 1;
    if (minimumSegmentLength < smallest || minimumSegmentLength > periodCount)
      throw new Error(
        `Location change points minimumSegmentLength must be ${smallest} to periodCount`
      );
    if (cost === 'poisson' && series.cube.some(value => value < 0))
      throw new Error('Location change points Poisson values must be nonnegative');
    const maximumChangePoints = 32;
    const result = await withCatalogGpuSession(
      context.device,
      'catalog-change-points',
      async session => {
        const count = session.output('count', 'uint32', cellCount);
        const changes = session.output('changes', 'uint32', cellCount * maximumChangePoints);
        const optimalCost = session.output('cost', 'float32', cellCount);
        const overflow = session.output('overflow', 'uint32', 1);
        session.graph.add(
          new GPULocationChangePoints({
            values: session.input('cube', series.cube, 'float32'),
            layout: 'cell-major',
            cellCount,
            sliceCount: periodCount,
            cost,
            maximumChangePoints,
            parameters: session.input(
              'parameters',
              Float32Array.from([Number(values['penalty']), minimumSegmentLength]),
              'float32'
            ),
            output: {
              changePointCount: count.view,
              changePoints: changes.view,
              optimalCost: optimalCost.view,
              status: {overflow: overflow.view}
            }
          })
        );
        session.run();
        if (await session.readScalar(overflow))
          throw new Error('Location change points exceeded 32 changes per location');
        return {
          count: await session.readUint32(count),
          changes: await session.readUint32(changes),
          optimalCost: await session.readFloat32(optimalCost)
        };
      }
    );
    const changes = Array.from({length: cellCount}, (_, cell) =>
      Array.from(
        result.changes.slice(
          cell * maximumChangePoints,
          cell * maximumChangePoints + result.count[cell]
        )
      )
    );
    Object.assign(summary, {
      cost,
      penalty: Number(values['penalty']),
      minimumSegmentLength,
      maximumChangePoints,
      changePointCount: result.count.reduce((total, value) => total + value, 0),
      optimalCostTotal: result.optimalCost.reduce((total, value) => total + value, 0)
    });
    return {
      tables: [
        makeSpacetimePointTable(points, {
          change_count: expand(result.count),
          optimal_cost: expand(result.optimalCost),
          segment: series.sourceCells.map(
            (cell, source) =>
              changes[cell].filter(period => period <= series.sourcePeriods[source]).length
          ),
          is_change_point: series.sourceCells.map((cell, source) =>
            Number(changes[cell].includes(series.sourcePeriods[source]))
          )
        })
      ],
      diagnostics: [],
      summary,
      dispatchedContributors: ['GPULocationChangePoints']
    };
  }
  if (kind === 'correlation') {
    if (periodCount - Math.abs(lag) < 4)
      throw new Error(
        'Space-time correlation requires four overlapping periods at the selected lag'
      );
    const result = await withCatalogGpuSession(
      context.device,
      'catalog-cross-correlation',
      async session => {
        const correlations = session.output('correlations', 'float32', cellCount);
        const pValues = session.output('p-values', 'float32', cellCount);
        const pairCounts = session.output('pairs', 'uint32', cellCount);
        session.graph.add(
          new GPUSpaceTimeCrossCorrelation({
            first: session.input('first', series.cube, 'float32'),
            second: session.input('second', series.second, 'float32'),
            layout: 'cell-major',
            cellCount,
            periodCount,
            lags: [lag],
            multipleTesting: 'none',
            output: {
              correlations: correlations.view,
              pValues: pValues.view,
              pairCounts: pairCounts.view
            }
          })
        );
        session.run();
        return {
          correlations: await session.readFloat32(correlations),
          pValues: await session.readFloat32(pValues),
          pairCounts: await session.readUint32(pairCounts)
        };
      }
    );
    const finitePValues = Array.from(result.pValues).filter(Number.isFinite);
    const finiteCorrelations = Array.from(result.correlations).filter(Number.isFinite);
    Object.assign(summary, {
      lag,
      correlationMinimum: finiteCorrelations.length ? Math.min(...finiteCorrelations) : 'undefined',
      correlationMaximum: finiteCorrelations.length ? Math.max(...finiteCorrelations) : 'undefined',
      testedLocations: finitePValues.length,
      pValueMinimum: finitePValues.length ? Math.min(...finitePValues) : 'undefined',
      inference: 'Fisher-z two-sided p-values; no multiple testing correction'
    });
    return {
      tables: [
        makeSpacetimePointTable(points, {
          correlation: expand(result.correlations),
          p_value: expand(result.pValues),
          pair_count: expand(result.pairCounts)
        })
      ],
      diagnostics: [],
      summary,
      dispatchedContributors: ['GPUSpaceTimeCrossCorrelation']
    };
  }
  const neighborCount = Math.min(Number(values['neighborCount']), cellCount - 1);
  if (kind === 'moran') {
    if (cellCount < 4 || lag >= periodCount)
      throw new Error('Space-time Moran requires four locations and lag below periodCount');
    const neighbors = makeSpacetimeNeighbors(series.positions, Infinity, neighborCount);
    const result = await withCatalogGpuSession(
      context.device,
      'catalog-space-time-moran',
      async session => {
        const observed = session.output('observed', 'float32', cellCount);
        const pValues = session.output('p-values', 'float32', cellCount);
        const overflow = session.output('overflow', 'uint32', 1);
        session.graph.add(
          new GPUSpaceTimeLocalMoran({
            weights: uploadSpacetimeNeighbors(session, neighbors),
            cube: session.input('cube', series.cube, 'float32'),
            layout: 'cell-major',
            periodCount,
            period: periodCount - 1,
            lag,
            variant: values['variant'] as GPUSpaceTimeLocalMoranVariant,
            direction: 'inward',
            parameters: session.input(
              'parameters',
              getGPUPermutationParameterValues(
                {seed, permutationCount: permutations, significanceLevel: 0.05},
                {maximumPermutations: permutations}
              ),
              'uint32'
            ),
            maximumPermutations: permutations,
            maximumNeighbors: 32,
            multipleTesting: 'none',
            alternative: 'two-sided',
            output: {
              observed: observed.view,
              pseudoPValues: pValues.view,
              exceedances: session.output('exceedances', 'uint32', cellCount).view,
              status: {overflow: overflow.view}
            }
          })
        );
        session.run();
        if (await session.readScalar(overflow))
          throw new Error('Space-time Moran neighbor capacity exceeded');
        return {
          observed: await session.readFloat32(observed),
          pValues: await session.readFloat32(pValues)
        };
      }
    );
    const latest = (entries: Float32Array) =>
      series.sourceCells.map((cell, source) =>
        series.sourcePeriods[source] === periodCount - 1
          ? finiteSpacetimeValue(entries[cell])
          : null
      );
    const finitePValues = Array.from(result.pValues).filter(Number.isFinite);
    const finiteObserved = Array.from(result.observed).filter(Number.isFinite);
    Object.assign(summary, {
      observedMinimum: finiteObserved.length ? Math.min(...finiteObserved) : 'undefined',
      observedMaximum: finiteObserved.length ? Math.max(...finiteObserved) : 'undefined',
      lag,
      period: periodCount - 1,
      neighborCount,
      maximumNeighbors: 32,
      maximumPermutations: 999,
      variant: String(values['variant']),
      pValueMinimum: finitePValues.length ? Math.min(...finitePValues) : 'undefined',
      inference: 'conditional permutation two-sided p-values; no multiple testing correction'
    });
    return {
      tables: [
        makeSpacetimePointTable(points, {
          observed: latest(result.observed),
          p_value: latest(result.pValues)
        })
      ],
      diagnostics: [],
      summary,
      dispatchedContributors: ['GPUSpaceTimeLocalMoran']
    };
  }
  // Scan counts must fit the exact float32 Poisson-statistic range; baseline is supplied, never fitted here.
  if (
    series.cube.some(value => !Number.isInteger(value) || value < 0 || value > 16777215) ||
    series.baseline.some(value => value <= 0)
  )
    throw new Error('Space-time scan requires integer counts 0 to 16777215 and positive baseline');
  const maximumWindowLength = Math.min(Number(values['maximumWindowLength']), periodCount);
  const maximumNeighbors = Math.min(Number(values['neighborCount']), cellCount);
  const result = await withCatalogGpuSession(
    context.device,
    'catalog-space-time-scan',
    async session => {
      const zones = session.output('zones', 'uint32', maximumNeighbors);
      const clusterStatistics = session.output('cluster-statistics', 'float32', 4);
      const outputSummary = session.output('summary', 'uint32', 5);
      session.graph.add(
        new GPUExpectationBasedSpaceTimeScan({
          positions: session.input('positions', series.positions, 'float32x2'),
          cases: session.input('cases', Uint32Array.from(series.cube), 'uint32'),
          baseline: session.input('baseline', series.baseline, 'float32'),
          timeBuckets: periodCount,
          maximumNeighbors,
          maximumWindowLength,
          maximumReplicates: permutations,
          parameters: session.input(
            'parameters',
            getGPUExpectationBasedScanParameterValues({
              seed,
              replicateCount: permutations,
              maximumWindowLength
            }),
            'uint32'
          ),
          output: {
            clusterZones: zones.view,
            clusterStatistics: clusterStatistics.view,
            statistics: session.output('statistics', 'float32', permutations + 1).view,
            summary: outputSummary.view
          }
        })
      );
      session.run();
      return {
        zones: await session.readUint32(zones),
        clusterStatistics: await session.readFloat32(clusterStatistics),
        summary: await session.readUint32(outputSummary)
      };
    }
  );
  Object.assign(
    summary,
    getSpacetimeSummary(result.clusterStatistics, GPU_EXPECTATION_BASED_SCAN_CLUSTER),
    Object.fromEntries(
      Object.entries(GPU_EXPECTATION_BASED_SCAN_SUMMARY).map(([name, index]) => [
        name,
        result.summary[index] === 0xffffffff ? 'none' : result.summary[index]
      ])
    ),
    {maximumWindowLength, maximumNeighbors, maximumReplicates: 999}
  );
  const members = new Set(Array.from(result.zones).filter(zone => zone !== 0xffffffff));
  const windowLength = result.summary[GPU_EXPECTATION_BASED_SCAN_SUMMARY.windowLength];
  return {
    tables: [
      makeSpacetimePointTable(points, {
        cluster_member: series.sourceCells.map((cell, source) =>
          Number(members.has(cell) && series.sourcePeriods[source] >= periodCount - windowLength)
        ),
        location_index: series.sourceCells
      }),
      makeSpacetimeSummaryTable(summary)
    ],
    diagnostics: [],
    summary,
    dispatchedContributors: ['GPUExpectationBasedSpaceTimeScan']
  };
}
