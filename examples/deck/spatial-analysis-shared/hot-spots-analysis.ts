// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Local spatial autocorrelation of a point value column: a `GPUNeighborSearch` distance band writes
 * a spatial-weights CSR, which `GPUHotSpotAnalysis` (Getis-Ord Gi*) and `GPULocalMoran` (LISA)
 * consume, optionally gated by `GPULocalPermutationTest`. Every compile-time variant (statistic,
 * false discovery rate, Moran gating, permutation alternative) is compiled up front, so parameter
 * changes are buffer writes or a choice between compiled graphs. See {@link AnalysisModeCompute}.
 */

import type {Buffer, CommandEncoder, Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  GPUElementwise,
  GPUHistogram,
  GPUReadbackRing,
  type CompiledGPUCommandGraph
} from '@luma.gl/graph/gpu-core';
import {
  getGPUNeighborSearchParameterValues,
  getGPUPermutationParameterValues,
  getGPUSpatialAutocorrelationParameterValues,
  GPUGlobalPermutationTest,
  GPUHotSpotAnalysis,
  GPULocalMoran,
  GPULocalPermutationTest,
  GPUNeighborSearch,
  GPU_GLOBAL_PERMUTATION_SUMMARY,
  GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH,
  GPU_PERMUTATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH,
  GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH,
  type GPULocalMoranQuadrantGating
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {
  type GPUInferenceAlternative,
  type GPUSpatialContext
} from '@luma-contributors/analysis/analysis-core';
import {clamp} from '@math.gl/core';
import type {
  AnalysisLegend,
  AnalysisModeCompute,
  AnalysisModeSceneHooks,
  AnalysisModeStyle
} from './analysis-mode';
import {importGraphBuffer} from './graph-buffers';
import {GI_COLORS, MORAN_COLORS, type SpatialAnalysisColor} from './layer-constants';
import {scheduleReadback} from './readback-scheduling';
import {SpatialAnalysisResources} from './resources';

/** Gi* bins -3..3 become histogram rows 0..6. */
export const HOT_SPOTS_GI_BIN_COUNT = 7;
/** Local Moran class codes 0 (not significant), 1 HH, 2 LH, 3 LL, 4 HL. */
export const HOT_SPOTS_MORAN_CLASS_COUNT = 5;
/** Histogram edges of the Gi* bins (sint32 -3..3). */
export const HOT_SPOTS_GI_HISTOGRAM_EDGES = [-3, -2, -1, 0, 1, 2, 3, 4];
/** Histogram edges of the Moran quadrant codes. */
export const HOT_SPOTS_MORAN_HISTOGRAM_EDGES = [0, 1, 2, 3, 4, 5];
/** Largest per-frame permutation count (compile-time bound of the permutation tests). */
export const HOT_SPOTS_MAXIMUM_PERMUTATIONS = 499;
/** Largest neighbor count `GPULocalPermutationTest` tests per row; denser rows are skipped. */
export const HOT_SPOTS_PERMUTATION_MAXIMUM_NEIGHBORS = 64;
/** Maximum neighbor-search lattice. Compile-time; results do not depend on it. */
const GRID_SIZE: readonly [number, number] = [256, 256];
/** Neighbor slots reserved per point in the weights CSR (capped at every other point). */
const SLOTS_PER_ROW = 1024;
/** A class value no category can take, so negative Gi* bins (large uint32) are never "no data". */
export const HOT_SPOTS_NO_DATA_VALUE = 0x7fffffff;
export const HOT_SPOTS_HIDDEN: SpatialAnalysisColor = [0, 0, 0, 0];
export const HOT_SPOTS_NEUTRAL: SpatialAnalysisColor = [150, 160, 175, 150];

/** Summary words: class counts, global statistics, then the neighbor-search overflow flag. */
const OVERFLOW_WORD = HOT_SPOTS_GI_BIN_COUNT + GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH;
const PERMUTATION_OVERFLOW_WORD = OVERFLOW_WORD + 1;
const PERMUTATION_RESULTS_WORD = PERMUTATION_OVERFLOW_WORD + 1;
const PERMUTATION_COUNTS_WORD = PERMUTATION_RESULTS_WORD + GPU_GLOBAL_PERMUTATION_SUMMARY.length;
const SUMMARY_WORDS = PERMUTATION_COUNTS_WORD + 2;

export type HotSpotsStatistic = 'gi-star' | 'local-moran';
/** `'permutation'` gates the ungated Moran quadrant by a conditional permutation test. */
export type HotSpotsGating = GPULocalMoranQuadrantGating | 'permutation';
export type HotSpotsAlternative = GPUInferenceAlternative;

export const HOT_SPOTS_ALTERNATIVES: readonly {value: HotSpotsAlternative; label: string}[] = [
  {value: 'directed', label: 'Directed (legacy esda default, tail on the observed side)'},
  {value: 'two-sided', label: 'Two-sided (default)'},
  {value: 'greater', label: 'Greater (large observed statistic)'},
  {value: 'lesser', label: 'Less (small observed statistic)'},
  {value: 'folded', label: 'Folded (|sim - mean| at least |obs - mean|)'}
];

/** Parameters of the analysis: per-frame buffer values and compile-time variant choices. */
export type HotSpotsParameters = {
  statistic: HotSpotsStatistic;
  gating: HotSpotsGating;
  alternative: HotSpotsAlternative;
  /** Per-frame: permutations per row. */
  permutations: number;
  /** Per-frame: permutation seed. */
  seed: number;
  /** Per-frame: neighborhood radius in the meters of the input positions. */
  radiusMeters: number;
  /** Per-frame: Moran significance level. */
  significanceLevel: number;
  /** Benjamini-Hochberg FDR (both settings are precompiled). */
  falseDiscoveryRate: boolean;
};

export const DEFAULT_HOT_SPOTS_PARAMETERS: Readonly<HotSpotsParameters> = {
  statistic: 'gi-star',
  gating: 'analytic',
  alternative: 'two-sided',
  permutations: 199,
  seed: 1,
  radiusMeters: 400,
  significanceLevel: 0.05,
  falseDiscoveryRate: false
};

/** Input points. `positions` are planar meters (x, y pairs), one value per point. */
export type HotSpotsInputs = {
  positions: Float32Array;
  values: Float32Array;
  spatialContext: GPUSpatialContext;
};

/** A decoded summary readback. */
export type HotSpotsSummary = {
  statistic: HotSpotsStatistic;
  /** Whether the readback came from permutation-gated Moran. */
  permutationGated: boolean;
  /** Histogram row counts: Gi* bins -3..3, or Moran codes 0 (none), 1 HH, 2 LH, 3 LL, 4 HL. */
  counts: readonly number[];
  mean: number;
  standardDeviation: number;
  /** `[significant, not significant]`; permutation-gated Moran only. */
  permutationCounts: readonly [number, number] | null;
  /** Global Moran I pseudo p-value summary; permutation-gated Moran only. */
  globalMoran: {observed: number; pseudoPValue: number; zSimulated: number} | null;
  neighborOverflow: boolean;
  permutationOverflow: boolean;
};

/** Buffers the layers of an app render. */
export type HotSpotsOutputs = {
  positions: Buffer;
  pointCount: number;
  /** Class buffer per statistic (`uint32`; Gi* bins are reinterpreted sint32). */
  classes: Record<HotSpotsStatistic, Buffer>;
};

export type HotSpotsCompute = AnalysisModeCompute<HotSpotsParameters> & {
  outputs: HotSpotsOutputs;
};

type Variant = {compiled: CompiledGPUCommandGraph<void>};

/** Builds the hot-spots compute: allocates buffers, compiles every variant, uploads parameters. */
export async function createHotSpotsCompute(
  device: Device,
  inputs: HotSpotsInputs,
  hooks: AnalysisModeSceneHooks<HotSpotsSummary>,
  initialParameters: Partial<HotSpotsParameters> = {}
): Promise<HotSpotsCompute> {
  const {positions, values, spatialContext} = inputs;
  const resources = new SpatialAnalysisResources(device, 'hot-spots');
  const pointCount = values.length;
  const parameters: HotSpotsParameters = {...DEFAULT_HOT_SPOTS_PARAMETERS, ...initialParameters};

  let minimumX = Infinity;
  let minimumY = Infinity;
  let maximumX = -Infinity;
  let maximumY = -Infinity;
  for (let index = 0; index < pointCount; index++) {
    minimumX = Math.min(minimumX, positions[index * 2]);
    maximumX = Math.max(maximumX, positions[index * 2]);
    minimumY = Math.min(minimumY, positions[index * 2 + 1]);
    maximumY = Math.max(maximumY, positions[index * 2 + 1]);
  }
  const bounds = [minimumX - 10, minimumY - 10, maximumX + 10, maximumY + 10] as const;

  let weightsDirty = true;
  let dirty = true;
  let needsReadback = true;
  let readbackPending = false;
  let destroyed = false;

  const capacity = pointCount * clamp(pointCount - 1, 1, SLOTS_PER_ROW);
  const positionsBuffer = resources.createBuffer('positions', positions);
  // The weights CSR the search writes and both statistics read.
  const offsetsBuffer = resources.createBuffer('offsets', (pointCount + 1) * 4);
  const neighborsBuffer = resources.createBuffer('neighbors', capacity * 4);
  const weightsBuffer = resources.createBuffer('weights', capacity * 4);
  const overflowBuffer = resources.createBuffer('overflow', 4);
  const searchParameters = resources.createParameterBuffer(
    'search-parameters',
    'float32',
    GPU_NEIGHBOR_SEARCH_PARAMETER_LENGTH
  );
  const valuesBuffer = resources.createBuffer('values', values);
  const parameterBuffer = resources.createParameterBuffer(
    'parameters',
    'float32',
    GPU_SPATIAL_AUTOCORRELATION_PARAMETER_LENGTH
  );
  const outputs = {
    'gi-star': {
      zScores: resources.createBuffer('gi-z-scores', pointCount * 4),
      classes: resources.createBuffer('gi-bins', pointCount * 4),
      counts: resources.createBuffer('gi-counts', HOT_SPOTS_GI_BIN_COUNT * 4),
      statistics: resources.createBuffer(
        'gi-statistics',
        GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH * 4
      )
    },
    'local-moran': {
      zScores: resources.createBuffer('moran-z-scores', pointCount * 4),
      classes: resources.createBuffer('moran-quadrants', pointCount * 4),
      counts: resources.createBuffer('moran-counts', HOT_SPOTS_MORAN_CLASS_COUNT * 4),
      statistics: resources.createBuffer(
        'moran-statistics',
        GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH * 4
      )
    }
  } as const;
  // Permutation gating: the ungated Moran quadrant, the permutation `significant` mask and their
  // product (written to the shared quadrant output) all live here.
  const permutation = {
    ungatedQuadrants: resources.createBuffer('moran-ungated-quadrants', pointCount * 4),
    exceedances: resources.createBuffer('perm-exceedances', pointCount * 4),
    pseudoPValues: resources.createBuffer('perm-pseudo-p', pointCount * 4),
    significant: resources.createBuffer('perm-significant', pointCount * 4),
    overflow: resources.createBuffer('perm-overflow', 4),
    significantCounts: resources.createBuffer('perm-significant-counts', 2 * 4),
    globalResults: resources.createBuffer(
      'perm-global-results',
      GPU_GLOBAL_PERMUTATION_SUMMARY.length * 4
    )
  };
  const permutationParameters = resources.createParameterBuffer(
    'permutation-parameters',
    'uint32',
    GPU_PERMUTATION_PARAMETER_LENGTH
  );
  const readbackRing = resources.track(
    new GPUReadbackRing(device, {id: 'hot-spots-summary', byteLength: SUMMARY_WORDS * 4})
  );

  const importWeights = (graph: GPUCommandGraph<void>) => ({
    offsets: importGraphBuffer(graph, 'offsets', offsetsBuffer, 'uint32', pointCount + 1),
    neighbors: importGraphBuffer(graph, 'neighbors', neighborsBuffer, 'uint32', capacity),
    weights: importGraphBuffer(graph, 'weights', weightsBuffer, 'float32', capacity)
  });

  // The distance-band search: radius and bounds are per-frame parameters, so changing the radius
  // is a buffer write.
  const searchGraph = new GPUCommandGraph<void>(device, {id: 'hot-spots-neighbor-search'});
  searchGraph.add(
    new GPUNeighborSearch({
      spatialContext,
      id: 'neighbor-search',
      mode: 'radius',
      gridSize: GRID_SIZE,
      positions: importGraphBuffer(
        searchGraph,
        'positions',
        positionsBuffer,
        'float32x2',
        pointCount
      ),
      parameters: searchParameters.importToGraph(searchGraph),
      output: {
        ...importWeights(searchGraph),
        status: {
          overflow: importGraphBuffer(searchGraph, 'overflow', overflowBuffer, 'uint32', 1)
        }
      }
    })
  );
  const searchCompiled = resources.track(await searchGraph.compileAsync());

  // FDR is a compile-time option of both contributors, so both settings are compiled up front and
  // the toggle selects between graphs that share one set of output buffers.
  async function compileVariant(
    kind: HotSpotsStatistic,
    fdr: boolean,
    variantGating: HotSpotsGating = 'analytic',
    variantAlternative: HotSpotsAlternative = 'two-sided'
  ): Promise<Variant> {
    const graph = new GPUCommandGraph<void>(device, {
      id: `hot-spots-${kind}${fdr ? '-fdr' : ''}-${variantGating}-${variantAlternative}`
    });
    const weights = importWeights(graph);
    const valuesView = importGraphBuffer(graph, 'values', valuesBuffer, 'float32', pointCount);
    const parameterView = parameterBuffer.importToGraph(graph);
    const output = outputs[kind];
    const zScores = importGraphBuffer(graph, 'z-scores', output.zScores, 'float32', pointCount);
    const statistics = importGraphBuffer(
      graph,
      'statistics',
      output.statistics,
      'float32',
      GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH
    );
    if (kind === 'gi-star') {
      const bins = importGraphBuffer(graph, 'bins', output.classes, 'sint32', pointCount);
      graph.add(
        new GPUHotSpotAnalysis({
          id: 'gi-star',
          weights,
          values: valuesView,
          parameters: parameterView,
          output: {zScores, bins, globalStatistics: statistics},
          multipleTesting: fdr ? 'benjamini-hochberg' : 'none'
        })
      );
      graph.add(
        new GPUHistogram({
          id: 'gi-star-counts',
          input: bins,
          output: importGraphBuffer(
            graph,
            'counts',
            output.counts,
            'uint32',
            HOT_SPOTS_GI_BIN_COUNT
          ),
          edges: HOT_SPOTS_GI_HISTOGRAM_EDGES
        })
      );
    } else {
      const classes = importGraphBuffer(graph, 'quadrants', output.classes, 'uint32', pointCount);
      const quadrants =
        variantGating === 'permutation'
          ? importGraphBuffer(
              graph,
              'ungated-quadrants',
              permutation.ungatedQuadrants,
              'uint32',
              pointCount
            )
          : classes;
      graph.add(
        new GPULocalMoran({
          id: 'local-moran',
          weights,
          values: valuesView,
          parameters: parameterView,
          output: {zScores, quadrants, globalStatistics: statistics},
          quadrantGating: variantGating === 'analytic' ? 'analytic' : 'none',
          multipleTesting: fdr ? 'benjamini-hochberg' : 'none'
        })
      );
      if (variantGating === 'permutation') {
        const permutationView = permutationParameters.importToGraph(graph);
        const significant = importGraphBuffer(
          graph,
          'perm-significant',
          permutation.significant,
          'uint32',
          pointCount
        );
        graph.add(
          new GPULocalPermutationTest({
            id: 'local-permutation',
            weights,
            values: valuesView,
            statistic: 'localMoran',
            alternative: variantAlternative,
            parameters: permutationView,
            maximumPermutations: HOT_SPOTS_MAXIMUM_PERMUTATIONS,
            maximumNeighbors: HOT_SPOTS_PERMUTATION_MAXIMUM_NEIGHBORS,
            output: {
              exceedances: importGraphBuffer(
                graph,
                'perm-exceedances',
                permutation.exceedances,
                'uint32',
                pointCount
              ),
              pseudoPValues: importGraphBuffer(
                graph,
                'perm-pseudo-p',
                permutation.pseudoPValues,
                'float32',
                pointCount
              ),
              significant,
              status: {
                overflow: importGraphBuffer(
                  graph,
                  'perm-overflow',
                  permutation.overflow,
                  'uint32',
                  1
                )
              }
            }
          })
        );
        graph.add(
          new GPUHistogram({
            id: 'perm-significant-counts',
            input: significant,
            output: importGraphBuffer(
              graph,
              'perm-significant-counts',
              permutation.significantCounts,
              'uint32',
              2
            ),
            edges: [0, 1, 2]
          })
        );
        graph.add(
          new GPUElementwise<'uint32'>({
            id: 'gate-quadrants',
            operation: 'multiply',
            input: quadrants,
            inputB: significant,
            output: classes
          })
        );
        graph.add(
          new GPUGlobalPermutationTest({
            id: 'global-permutation',
            weights,
            values: valuesView,
            statistic: 'moran',
            alternative: variantAlternative,
            parameters: permutationView,
            maximumPermutations: HOT_SPOTS_MAXIMUM_PERMUTATIONS,
            output: {
              summary: importGraphBuffer(
                graph,
                'perm-global-results',
                permutation.globalResults,
                'float32',
                GPU_GLOBAL_PERMUTATION_SUMMARY.length
              )
            }
          })
        );
      }
      graph.add(
        new GPUHistogram({
          id: 'local-moran-counts',
          input: classes,
          output: importGraphBuffer(
            graph,
            'counts',
            output.counts,
            'uint32',
            HOT_SPOTS_MORAN_CLASS_COUNT
          ),
          edges: HOT_SPOTS_MORAN_HISTOGRAM_EDGES
        })
      );
    }
    return {compiled: resources.track(await graph.compileAsync())};
  }

  const variants: Record<`${HotSpotsStatistic}:${'plain' | 'fdr'}` | 'local-moran:none', Variant> =
    {
      'gi-star:plain': await compileVariant('gi-star', false),
      'gi-star:fdr': await compileVariant('gi-star', true),
      'local-moran:plain': await compileVariant('local-moran', false),
      'local-moran:fdr': await compileVariant('local-moran', true),
      'local-moran:none': await compileVariant('local-moran', false, 'none')
    };
  const permutationVariants = Object.fromEntries(
    await Promise.all(
      HOT_SPOTS_ALTERNATIVES.map(
        async ({value}) =>
          [value, await compileVariant('local-moran', false, 'permutation', value)] as const
      )
    )
  ) as Record<HotSpotsAlternative, Variant>;

  const getActiveVariant = () => {
    const {statistic, gating, alternative, falseDiscoveryRate} = parameters;
    if (statistic === 'local-moran' && gating === 'none') return variants['local-moran:none'];
    if (statistic === 'local-moran' && gating === 'permutation') {
      return permutationVariants[alternative];
    }
    return variants[`${statistic}:${falseDiscoveryRate ? 'fdr' : 'plain'}`];
  };
  const writeParameters = () => {
    searchParameters.write(
      getGPUNeighborSearchParameterValues({
        bounds,
        radius: parameters.radiusMeters,
        weightKind: 'binary'
      })
    );
    parameterBuffer.write(
      getGPUSpatialAutocorrelationParameterValues({significanceLevel: parameters.significanceLevel})
    );
    permutationParameters.write(
      getGPUPermutationParameterValues({
        seed: parameters.seed,
        permutationCount: parameters.permutations,
        significanceLevel: parameters.significanceLevel
      })
    );
    weightsDirty = true;
    dirty = true;
    needsReadback = true;
  };
  writeParameters();

  const readSummary = async (commandEncoder: CommandEncoder) => {
    const ticket = readbackRing.tryAcquire();
    if (!ticket) return;
    const reportedStatistic = parameters.statistic;
    const output = outputs[reportedStatistic];
    const classCount =
      reportedStatistic === 'gi-star' ? HOT_SPOTS_GI_BIN_COUNT : HOT_SPOTS_MORAN_CLASS_COUNT;
    const copy = (sourceBuffer: Buffer, destinationWord: number, byteLength: number) =>
      commandEncoder.copyBufferToBuffer({
        sourceBuffer,
        destinationBuffer: ticket.buffer,
        destinationOffset: destinationWord * 4,
        size: byteLength
      });
    copy(output.counts, 0, classCount * 4);
    copy(
      output.statistics,
      HOT_SPOTS_GI_BIN_COUNT,
      GPU_SPATIAL_AUTOCORRELATION_STATISTICS_LENGTH * 4
    );
    copy(overflowBuffer, OVERFLOW_WORD, 4);
    const reportedPermutation =
      reportedStatistic === 'local-moran' && parameters.gating === 'permutation';
    if (reportedPermutation) {
      copy(permutation.overflow, PERMUTATION_OVERFLOW_WORD, 4);
      copy(
        permutation.globalResults,
        PERMUTATION_RESULTS_WORD,
        GPU_GLOBAL_PERMUTATION_SUMMARY.length * 4
      );
      copy(permutation.significantCounts, PERMUTATION_COUNTS_WORD, 8);
    }
    ticket.markEncoded({byteOffset: 0, byteLength: SUMMARY_WORDS * 4});
    readbackPending = true;
    needsReadback = false;
    try {
      const bytes = await scheduleReadback(() => ticket.read());
      if (destroyed) return;
      const words = new Uint32Array(bytes.buffer, bytes.byteOffset, SUMMARY_WORDS);
      const floats = new Float32Array(bytes.buffer, bytes.byteOffset, SUMMARY_WORDS);
      const base = PERMUTATION_RESULTS_WORD;
      hooks.publishSummary({
        statistic: reportedStatistic,
        permutationGated: reportedPermutation,
        counts: Array.from(words.subarray(0, classCount)),
        mean: floats[HOT_SPOTS_GI_BIN_COUNT + 1],
        standardDeviation: floats[HOT_SPOTS_GI_BIN_COUNT + 3],
        permutationCounts: reportedPermutation
          ? [words[PERMUTATION_COUNTS_WORD + 1], words[PERMUTATION_COUNTS_WORD]]
          : null,
        globalMoran: reportedPermutation
          ? {
              observed: floats[base + GPU_GLOBAL_PERMUTATION_SUMMARY.observed],
              pseudoPValue: floats[base + GPU_GLOBAL_PERMUTATION_SUMMARY.pseudoPValue],
              zSimulated: floats[base + GPU_GLOBAL_PERMUTATION_SUMMARY.zSimulated]
            }
          : null,
        neighborOverflow: words[OVERFLOW_WORD] !== 0,
        permutationOverflow: reportedPermutation && words[PERMUTATION_OVERFLOW_WORD] !== 0
      });
    } catch {
      // The ring or device was destroyed while the read was in flight.
      needsReadback = true;
    } finally {
      readbackPending = false;
    }
  };

  const perFrameKeys: (keyof HotSpotsParameters)[] = [
    'permutations',
    'seed',
    'radiusMeters',
    'significanceLevel'
  ];

  return {
    outputs: {
      positions: positionsBuffer,
      pointCount,
      classes: {
        'gi-star': outputs['gi-star'].classes,
        'local-moran': outputs['local-moran'].classes
      }
    },
    getCompiledGraphs: () => [
      searchCompiled,
      ...Object.values(variants).map(variant => variant.compiled),
      ...Object.values(permutationVariants).map(variant => variant.compiled)
    ],
    getParameters: () => parameters,
    setParameters(changes) {
      Object.assign(parameters, changes);
      if (perFrameKeys.some(key => key in changes)) {
        writeParameters();
      } else {
        dirty = true;
        needsReadback = true;
      }
    },
    encode(commandEncoder, frameIndex) {
      // The data is static: results only change with the radius, level, statistic or FDR.
      if (dirty || frameIndex < 2) {
        if (weightsDirty || frameIndex < 2) {
          searchCompiled.encode(commandEncoder, {parameters: undefined});
          weightsDirty = false;
        }
        getActiveVariant().compiled.encode(commandEncoder, {parameters: undefined});
        dirty = false;
      }
      if (needsReadback && !readbackPending && frameIndex >= 1) {
        void readSummary(commandEncoder);
      }
    },
    destroy() {
      destroyed = true;
      resources.destroy();
    }
  };
}

const GI_LEGEND: AnalysisLegend = {
  title: 'Gi* confidence bin (hot to cold)',
  entries: [
    {color: GI_COLORS[3], label: 'Hot 99%'},
    {color: GI_COLORS[2], label: 'Hot 95%'},
    {color: GI_COLORS[1], label: 'Hot 90%'},
    {color: HOT_SPOTS_NEUTRAL, label: 'Not significant'},
    {color: GI_COLORS[7], label: 'Cold 90%'},
    {color: GI_COLORS[6], label: 'Cold 95%'},
    {color: GI_COLORS[5], label: 'Cold 99%'}
  ]
};

const MORAN_LEGEND: AnalysisLegend = {
  title: 'Local Moran quadrant',
  entries: [
    {color: MORAN_COLORS[1], label: 'High-High'},
    {color: MORAN_COLORS[3], label: 'Low-Low'},
    {color: MORAN_COLORS[4], label: 'High-Low outlier'},
    {color: MORAN_COLORS[2], label: 'Low-High outlier'}
  ]
};

/** Legends and explanation for the hot-spots analysis. */
export const hotSpotsStyle: AnalysisModeStyle<HotSpotsParameters> = {
  // Both legends are always shown, as the explorer always has.
  getLegends: () => [GI_LEGEND, MORAN_LEGEND],
  describe: ({statistic, gating}) =>
    statistic === 'gi-star'
      ? 'Gi*: binary distance-band weights, the point itself included (selfWeight 1). Bins are 90/95/99% ' +
        'two-sided confidence, or BH-FDR corrected.'
      : gating === 'analytic'
        ? 'Moran: the point itself excluded, conditional-randomization z-score; quadrants are ' +
          'shown where p is at most the significance level (BH-FDR corrected if on).'
        : gating === 'none'
          ? 'Moran, ungated: every point with a nonzero centered value and lag gets its ' +
            'quadrant, so the quadrant counts add up to nearly all points.'
          : 'Moran gated by GPULocalPermutationTest (conditional permutation, the pseudo p-value ' +
            'tail chosen above); points with more than 64 neighbors are not tested, so reduce ' +
            'the radius if the readout reports skipped rows. FDR does not apply.'
};

/** Per-class palette of the significant points: Gi* bins or Moran quadrants. */
export function getHotSpotsSignificantPalette(
  statistic: HotSpotsStatistic
): readonly SpatialAnalysisColor[] {
  const colors = statistic === 'gi-star' ? GI_COLORS : MORAN_COLORS;
  return Array.from({length: 8}, (_, index) => colors[index] ?? HOT_SPOTS_HIDDEN);
}

/** Per-class palette of the not-significant points: only class 0 is drawn. */
export function getHotSpotsInsignificantPalette(): readonly SpatialAnalysisColor[] {
  return Array.from({length: 8}, (_, index) =>
    index === 0 ? HOT_SPOTS_NEUTRAL : HOT_SPOTS_HIDDEN
  );
}
