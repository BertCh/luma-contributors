// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPUGlobalSpatialStatistics,
  GPUHotSpotAnalysis,
  GPULocalGeary,
  GPULocalMoran,
  GPULocalPermutationTest,
  GPUSkaterRegions,
  GPUSpatialWeightsMinimumSpanningTree,
  GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT,
  GPU_GLOBAL_SPATIAL_STATISTIC_FIELD,
  GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  getGPUPermutationParameterValues,
  getGPUSkaterRegionsParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  type GPUSpatialWeights
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {CatalogGpuContext, CatalogRunOutput} from './catalog-binding-types';
import type {CatalogGpuRunner} from './catalog-gpu-bindings';
import {
  CatalogGpuUnsupportedError,
  withCatalogGpuSession,
  type CatalogGpuSession
} from './catalog-gpu-session';
import {readSkaterInput, runSkaterRegions} from './catalog-platform3-b-clusters';
import {
  guardContiguityInput,
  readWeightsSettings,
  requireValueProperty
} from './catalog-platform3-b-bindings';
import {
  buildAnalysisSet,
  getMeanAndDeviation,
  runGlobalSpatialStatistics,
  runHotSpotGiStar,
  runLocalGeary,
  runLocalMoran,
  type AnalysisSet
} from './catalog-platform3-b-statistics';
import type {Inputs, Values} from './catalog-platform3-b-common';

/** Rows the GPU contributors index with f32 counters. */
const MAXIMUM_PERMUTATION_NEIGHBORS = 32;

type UploadedWeights = {
  weights: GPUSpatialWeights;
  offsets: Uint32Array;
  neighbors: Uint32Array;
  slotWeights: Float32Array;
};

/** Packs the analysis-set neighbor lists as CSR with ascending neighbor ids, the contract of every consumer. */
export function makeAnalysisCsr(set: AnalysisSet): {
  offsets: Uint32Array;
  neighbors: Uint32Array;
  weights: Float32Array;
} {
  const rowCount = set.values.length;
  const offsets = new Uint32Array(rowCount + 1);
  const neighbors: number[] = [];
  const weights: number[] = [];
  for (let row = 0; row < rowCount; row++) {
    const order = set.weights.neighbors[row]
      .map((id, position) => ({id, weight: set.weights.weights[row][position]}))
      .sort((a, b) => a.id - b.id);
    for (const entry of order) {
      neighbors.push(entry.id);
      weights.push(entry.weight);
    }
    offsets[row + 1] = neighbors.length;
  }
  return {
    offsets,
    neighbors: Uint32Array.from(neighbors),
    weights: Float32Array.from(weights)
  };
}

function uploadAnalysisWeights(
  session: CatalogGpuSession,
  set: AnalysisSet,
  transform: 'row' | 'binary'
): UploadedWeights {
  const csr = makeAnalysisCsr(set);
  const rowCount = set.values.length;
  return {
    weights: {
      offsets: session.input('weights-offsets', csr.offsets, 'uint32'),
      neighbors: session.input('weights-neighbors', csr.neighbors, 'uint32'),
      weights: session.input('weights-values', csr.weights, 'float32'),
      information: {rowCount, transform, symmetric: 'unknown'}
    },
    offsets: csr.offsets,
    neighbors: csr.neighbors,
    slotWeights: csr.weights
  };
}

/** Same validity gates as the CPU runners, so the CPU reference raises the declared error. */
function requireLocalStatisticInput(set: AnalysisSet): void {
  const rowCount = set.values.length;
  if (rowCount < 4) throw new CatalogGpuUnsupportedError('Too few features');
  if (getMeanAndDeviation(set.values).deviation === 0)
    throw new CatalogGpuUnsupportedError('Constant values');
}

function prepare(inputs: Inputs, values: Values, rowStandardize: boolean, canonicalRows = false) {
  const settings = readWeightsSettings(values, rowStandardize);
  guardContiguityInput(inputs, settings);
  const valueProperty = requireValueProperty(values);
  const collection = inputs['features'];
  const set = buildAnalysisSet(collection, valueProperty, settings, canonicalRows);
  return {settings, valueProperty, collection, set};
}

async function runGpuGlobalStatistics(
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const {settings, valueProperty, collection, set} = prepare(inputs, values, true);
  requireLocalStatisticInput(set);
  const linkCount = set.weights.neighbors.reduce((sum, list) => sum + list.length, 0);
  if (!linkCount) throw new CatalogGpuUnsupportedError('No neighbors');
  const summary = await withCatalogGpuSession(
    context.device,
    'catalog-global-statistics',
    async session => {
      const uploaded = uploadAnalysisWeights(session, set, 'row');
      const output = session.output(
        'summary',
        'float32',
        GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.length
      );
      session.graph.add(
        new GPUGlobalSpatialStatistics({
          id: 'catalog-global-statistics',
          weights: uploaded.weights,
          values: session.input('values', Float32Array.from(set.values), 'float32'),
          statistics: ['moran', 'geary'],
          // The CPU reference keeps islands in n and the moments, as esda does.
          islands: 'keep',
          output: {summary: output.view}
        })
      );
      session.run();
      return session.readFloat32(output);
    }
  );
  const read = (offset: number) => ({
    value: summary[offset + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.statistic],
    expected: summary[offset + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.expected],
    zScore: 0,
    pValue: 0,
    variance: 0
  });
  const moran = read(GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran);
  // Both statistics use randomization variance, like the CPU.
  moran.variance =
    summary[
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran +
        GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.varianceRandomization
    ];
  moran.zScore =
    summary[
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.zRandomization
    ];
  moran.pValue =
    summary[
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.moran + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.pRandomization
    ];
  const geary = read(GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.geary);
  geary.variance =
    summary[
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.geary +
        GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.varianceRandomization
    ];
  geary.zScore =
    summary[
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.geary + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.zRandomization
    ];
  geary.pValue =
    summary[
      GPU_GLOBAL_SPATIAL_STATISTICS_LAYOUT.geary + GPU_GLOBAL_SPATIAL_STATISTIC_FIELD.pRandomization
    ];
  // The summary count must agree with the uploaded rows.
  if (summary[GPU_GLOBAL_SPATIAL_STATISTICS_SUMMARY.count] !== set.values.length)
    throw new CatalogGpuUnsupportedError('Row count mismatch');
  return {
    ...runGlobalSpatialStatistics(collection, {valueProperty, ...settings}, {set, moran, geary}),
    dispatchedContributors: ['GPUGlobalSpatialStatistics']
  };
}

function getParameterView(session: CatalogGpuSession, significanceLevel = 0.05) {
  const parameters = getGPUSpatialAutocorrelationParameterValues({significanceLevel});
  // The contributors read at least this many float32 words.
  if (parameters.length < GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH)
    throw new CatalogGpuUnsupportedError('Parameter layout');
  return session.input('parameters', parameters, 'float32');
}

/**
 * Conditional-permutation pseudo p-values of one local statistic; NaN where a row is untested.
 * Runs in its own session because a graph compiles once.
 */
async function runGpuPermutations(
  device: CatalogGpuContext['device'],
  set: AnalysisSet,
  statistic: 'localMoran' | 'localGeary',
  permutations: number,
  seed: number,
  significance: number
): Promise<{pValues: Float32Array; exceedances: Uint32Array}> {
  const rowCount = set.values.length;
  const maxNeighbors = Math.max(0, ...set.weights.neighbors.map(list => list.length));
  // Rows with more neighbors than the scratch holds are not tested.
  if (maxNeighbors > MAXIMUM_PERMUTATION_NEIGHBORS)
    throw new CatalogGpuUnsupportedError('Too many neighbors for the permutation scratch');
  return withCatalogGpuSession(device, 'catalog-local-permutation', async session => {
    const uploaded = uploadAnalysisWeights(session, set, 'row');
    const pseudo = session.output('pseudo-p-values', 'float32', rowCount);
    const exceedances = session.output('exceedances', 'uint32', rowCount);
    const overflow = session.output('permutation-overflow', 'uint32', 1);
    session.graph.add(
      new GPULocalPermutationTest({
        id: 'catalog-local-permutation',
        weights: uploaded.weights,
        values: session.input('permutation-values', Float32Array.from(set.values), 'float32'),
        statistic,
        alternative: statistic === 'localGeary' ? 'lesser' : 'two-sided',
        islands: 'keep',
        parameters: session.input(
          'permutation-parameters',
          getGPUPermutationParameterValues(
            {seed, permutationCount: permutations, significanceLevel: significance},
            {maximumPermutations: permutations}
          ),
          'uint32'
        ),
        maximumPermutations: permutations,
        output: {
          exceedances: exceedances.view,
          pseudoPValues: pseudo.view,
          status: {overflow: overflow.view}
        }
      })
    );
    session.run();
    if ((await session.readScalar(overflow)) !== 0)
      throw new CatalogGpuUnsupportedError('Permutation test incomplete');
    return {
      pValues: await session.readFloat32(pseudo),
      exceedances: await session.readUint32(exceedances)
    };
  });
}

async function runGpuLocalMoran(
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const {settings, valueProperty, collection, set} = prepare(inputs, values, true, true);
  requireLocalStatisticInput(set);
  const permutations = values['permutations'] as number;
  const significance = values['significance'] as number;
  const {deviation} = getMeanAndDeviation(set.values);
  const result = await withCatalogGpuSession(
    context.device,
    'catalog-local-moran',
    async session => {
      const uploaded = uploadAnalysisWeights(session, set, 'row');
      const valueData = Float32Array.from(set.values);
      const rowCount = valueData.length;
      const zScores = session.output('z-scores', 'float32', rowCount);
      const localI = session.output('local-i', 'float32', rowCount);
      const spatialLag = session.output('spatial-lag', 'float32', rowCount);
      session.graph.add(
        new GPULocalMoran({
          id: 'catalog-local-moran',
          weights: uploaded.weights,
          values: session.input('values', valueData, 'float32'),
          parameters: getParameterView(session, significance),
          islands: 'keep',
          output: {zScores: zScores.view, localI: localI.view, spatialLag: spatialLag.view}
        })
      );
      session.run();
      return {
        localI: await session.readFloat32(localI),
        spatialLag: await session.readFloat32(spatialLag)
      };
    }
  );
  const pValues =
    permutations > 0
      ? await runGpuPermutations(
          context.device,
          set,
          'localMoran',
          permutations,
          values['seed'] as number,
          significance
        )
      : undefined;
  const n = set.values.length;
  // The contributor scales by (n - 1) / M2 and keeps the lag in raw units; the CPU reference reports
  // the lag of z-scores and I = z * lag.
  const lag = Float64Array.from(result.spatialLag, value => value / deviation);
  const localI = Float64Array.from(result.localI, value => (value * n) / (n - 1));
  const pValue = pValues
    ? Float64Array.from(pValues.pValues, (value, row) =>
        set.weights.neighbors[row].length ? value : Number.NaN
      )
    : undefined;
  return {
    ...runLocalMoran(
      collection,
      {
        valueProperty,
        permutations,
        seed: values['seed'] as number,
        significance,
        multipleTesting: values['multipleTesting'] as 'none' | 'benjamini-hochberg' | 'bonferroni',
        ...settings
      },
      {
        set,
        statistic: localI,
        spatialLag: lag,
        pValue,
        pValueExceedances: pValues?.exceedances,
        pValueConstruction: 'doubled-minimum-tail'
      }
    ),
    dispatchedContributors:
      permutations > 0 ? ['GPULocalMoran', 'GPULocalPermutationTest'] : ['GPULocalMoran']
  };
}

async function runGpuLocalGeary(
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const {settings, valueProperty, collection, set} = prepare(inputs, values, true, true);
  requireLocalStatisticInput(set);
  const permutations = values['permutations'] as number;
  const result = await withCatalogGpuSession(
    context.device,
    'catalog-local-geary',
    async session => {
      const uploaded = uploadAnalysisWeights(session, set, 'row');
      const valueData = Float32Array.from(set.values);
      const statistic = session.output('statistic', 'float32', valueData.length);
      session.graph.add(
        new GPULocalGeary({
          id: 'catalog-local-geary',
          weights: uploaded.weights,
          values: session.input('values', valueData, 'float32'),
          parameters: getParameterView(session),
          islands: 'keep',
          output: {statistic: statistic.view}
        })
      );
      session.run();
      return {statistic: await session.readFloat32(statistic)};
    }
  );
  const pValues =
    permutations > 0
      ? await runGpuPermutations(
          context.device,
          set,
          'localGeary',
          permutations,
          values['seed'] as number,
          0.05
        )
      : undefined;
  const pValue = pValues
    ? Float64Array.from(pValues.pValues, (value, row) =>
        set.weights.neighbors[row].length ? value : Number.NaN
      )
    : new Float64Array(set.values.length).fill(Number.NaN);
  return {
    ...runLocalGeary(
      collection,
      {valueProperty, permutations, seed: values['seed'] as number, ...settings},
      {set, statistic: result.statistic, pValue}
    ),
    dispatchedContributors:
      permutations > 0 ? ['GPULocalGeary', 'GPULocalPermutationTest'] : ['GPULocalGeary']
  };
}

async function runGpuGiStar(
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const {settings, valueProperty, collection, set} = prepare(inputs, values, false);
  requireLocalStatisticInput(set);
  const zScores = await withCatalogGpuSession(context.device, 'catalog-gi-star', async session => {
    const uploaded = uploadAnalysisWeights(session, set, 'binary');
    const output = session.output('z-scores', 'float32', set.values.length);
    session.graph.add(
      new GPUHotSpotAnalysis({
        id: 'catalog-gi-star',
        weights: uploaded.weights,
        selfWeight: 1,
        values: session.input('values', Float32Array.from(set.values), 'float32'),
        parameters: getParameterView(session),
        islands: 'keep',
        output: {zScores: output.view}
      })
    );
    session.run();
    return session.readFloat32(output);
  });
  return {
    ...runHotSpotGiStar(collection, {valueProperty, ...settings}, {set, statistic: zScores}),
    dispatchedContributors: ['GPUHotSpotAnalysis']
  };
}

async function runGpuSkater(
  context: CatalogGpuContext,
  inputs: Inputs,
  values: Values
): Promise<CatalogRunOutput> {
  const settings = readWeightsSettings(values, false);
  guardContiguityInput(inputs, settings);
  const parameters = {
    properties: values['properties'] as string,
    regionCount: values['regionCount'] as number,
    minimumRegionSize: values['minimumRegionSize'] as number,
    neighborhood: settings.neighborhood,
    neighborCount: settings.neighborCount,
    distanceBand: settings.distanceBand,
    coordinateSystem: settings.coordinateSystem
  };
  const input = readSkaterInput(inputs['features'], parameters);
  const rowCount = input.rows.length;
  // The contributor needs a target of at least two regions, and the CPU raises the size errors.
  if (rowCount < parameters.regionCount || parameters.regionCount < 2)
    throw new CatalogGpuUnsupportedError('Region count outside the GPU range');
  // The CPU engine links a to b only when the lower row lists the higher row (kept rows only);
  // the same undirected edge set becomes a symmetric CSR for the forest.
  const position = new Map<number, number>();
  input.rows.forEach((row, index) => position.set(row, index));
  const adjacency: number[][] = Array.from({length: rowCount}, () => []);
  input.rows.forEach((row, a) =>
    input.set.weights.neighbors[row].forEach(other => {
      const b = position.get(other);
      if (b === undefined || b <= a) return;
      adjacency[a].push(b);
      adjacency[b].push(a);
    })
  );
  const offsets = new Uint32Array(rowCount + 1);
  const neighbors: number[] = [];
  adjacency.forEach((list, row) => {
    const unique = [...new Set(list)].sort((x, y) => x - y);
    neighbors.push(...unique);
    offsets[row + 1] = neighbors.length;
  });
  if (!neighbors.length) throw new CatalogGpuUnsupportedError('No links');
  const columnCount = input.names.length;
  const attributes = new Float32Array(rowCount * columnCount);
  input.attributes.forEach((row, index) => attributes.set(row, index * columnCount));
  const labels = await withCatalogGpuSession(context.device, 'catalog-skater', async session => {
    const slotCount = neighbors.length;
    const weights: GPUSpatialWeights = {
      offsets: session.input('weights-offsets', offsets, 'uint32'),
      neighbors: session.input('weights-neighbors', Uint32Array.from(neighbors), 'uint32'),
      weights: session.input('weights-values', new Float32Array(slotCount).fill(1), 'float32')
    };
    const treeEdgeFlags = session.output('tree-edge-flags', 'uint32', slotCount);
    const componentLabels = session.output('component-labels', 'uint32', rowCount);
    const standardized = session.output('standardized', 'float32', rowCount * columnCount);
    session.graph.add(
      new GPUSpatialWeightsMinimumSpanningTree({
        id: 'catalog-skater-tree',
        weights,
        values: session.input('attributes', attributes, 'float32'),
        columnCount,
        output: {
          treeEdgeFlags: treeEdgeFlags.view,
          componentLabels: componentLabels.view,
          standardizedValues: standardized.view
        }
      })
    );
    const regions = session.output('region-labels', 'uint32', rowCount);
    session.graph.add(
      new GPUSkaterRegions({
        id: 'catalog-skater-regions',
        weights,
        treeEdgeFlags: treeEdgeFlags.view,
        componentLabels: componentLabels.view,
        values: standardized.view,
        columnCount,
        maximumRegionCount: Math.max(2, parameters.regionCount),
        parameters: session.input(
          'skater-parameters',
          getGPUSkaterRegionsParameterValues({
            regionCount: parameters.regionCount,
            minimumRegionSize: parameters.minimumRegionSize
          }),
          'uint32'
        ),
        output: {labels: regions.view}
      })
    );
    session.run();
    return session.readUint32(regions);
  });
  return {
    ...runSkaterRegions(inputs['features'], parameters, {input, labels}),
    dispatchedContributors: ['GPUSpatialWeightsMinimumSpanningTree', 'GPUSkaterRegions']
  };
}

/**
 * PLATFORM5 GPU runners for the weights and statistics suite bindings that were CPU-reference only, keyed by
 * binding id. The neighbor lists are built by the CPU reference (exact ties, polygon contiguity) and uploaded as
 * CSR; the GPU computes the statistic, the CPU reference stays the oracle and the fallback.
 */
export const PLATFORM5_STATISTICS_GPU_RUNNERS: Readonly<Record<string, CatalogGpuRunner>> = {
  'global-spatial-statistics': {
    contributorNames: ['GPUGlobalSpatialStatistics'],
    run: runGpuGlobalStatistics
  },
  'local-moran-statistics': {
    contributorNames: ['GPULocalMoran', 'GPULocalPermutationTest'],
    run: runGpuLocalMoran
  },
  'local-geary-statistics': {
    contributorNames: ['GPULocalGeary', 'GPULocalPermutationTest'],
    run: runGpuLocalGeary
  },
  'hotspot-gi-star': {contributorNames: ['GPUHotSpotAnalysis'], run: runGpuGiStar},
  'skater-regions': {
    contributorNames: ['GPUSpatialWeightsMinimumSpanningTree', 'GPUSkaterRegions'],
    run: runGpuSkater
  }
};
