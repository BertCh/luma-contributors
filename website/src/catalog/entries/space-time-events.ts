// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Catalog group: space-time events. Space-time interaction tests (Jacquez, local Knox, near-repeat
// table, Diggle D), self-exciting Hawkes fits, space-time and adaptive kernel densities, relative
// risk and prospective expectation-based scans. Every example assumes `graph` is a
// GPUCommandGraph, `device` a luma Device and the named views were created on the graph.

import type {CatalogEntry} from '../schema';

const IMPORT_PATH = '@luma-contributors/analysis/gpu-spatial-analysis';
const DATAFRAME_PATH = '@luma-contributors/analysis/dataframe-analysis';
const DYNAMICS = 'api/gpu-spatial-analysis/operators-dynamics';
const DENSITY_AND_K = 'api/gpu-spatial-analysis/space-time-density-and-k-function';

const timesParameter = {
  name: 'times',
  type: 'GraphDataView<"float32">',
  required: true,
  description: 'Relative event times (getGPUSpaceTimeGuardedRelativeTimes).'
};
const spaceTimeParameters = {
  name: 'parameters',
  type: 'GraphDataView<"uint32">',
  required: true,
  description: 'getGPUSpaceTimeParameterValues: seed, permutations, time threshold.'
};
const maximumPermutationsParameter = {
  name: 'maximumPermutations',
  type: 'number',
  required: true,
  description: 'Compile-time upper bound on the permutation count, 1 to 65536.'
};

export const entries: CatalogEntry[] = [
  {
    id: 'jacquez-test',
    symbol: 'GPUJacquezTest',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Jacquez k-nearest-neighbor test',
    summary: 'Tests whether events that are nearest neighbors in space are also nearest in time.',
    explanation:
      'A space-time interaction test that needs no distance or time threshold, only `neighborCount`. It counts (event, neighbor) pairs that are among the k nearest in both space and time and compares the count with permuted times.',
    dataTypes: {input: ['points', 'spatial-weights', 'time-series'], output: ['table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics'],
    params: [
      {
        name: 'neighbors',
        type: 'GPUSpatialWeights',
        required: true,
        description: 'The k nearest events in space of every event (a knn list).'
      },
      timesParameter,
      {
        name: 'parameters',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'getGPUSpaceTimeParameterValues: seed and permutation count.'
      },
      {
        name: 'neighborCount',
        type: 'number',
        required: true,
        description: 'Nearest neighbors in time, 1 to 256.'
      },
      maximumPermutationsParameter,
      {
        name: 'stopAfterExceedances',
        type: 'number | undefined',
        description: 'Early stop after this many exceedances; needs output.permutationCount.'
      },
      {
        name: 'output',
        type: 'GPUJacquezTestOutput',
        required: true,
        description: 'Statistics and summary views.'
      }
    ],
    outputs: [
      {
        name: 'statistics',
        type: 'GraphDataView<"uint32">',
        description: 'Observed J, then J under each permutation.'
      },
      {
        name: 'permutationCount',
        type: 'GraphDataView<"uint32">',
        description: 'Optional permutations used; required with stopAfterExceedances.'
      },
      {
        name: 'summary',
        type: 'GraphDataView<"float32">',
        description: 'Means and pseudo p-values (GPU_SPACE_TIME_SUMMARY).'
      }
    ],
    example: `import {
  GPUJacquezTest,
  GPUNeighborSearch,
  getGPUSpaceTimeParameterValues
} from '${IMPORT_PATH}';

// neighbors: GPUNeighborSearch in knn mode with neighborCount = 5; times: days since the first event
graph.add(new GPUJacquezTest({
  neighbors, times, neighborCount: 5, maximumPermutations: 499,
  parameters: graph.importBuffer(parameterBuffer, 'uint32'),
  output: {statistics, summary}
}));
parameterBuffer.write(getGPUSpaceTimeParameterValues({seed: 1, permutationCount: 499}));`,
    preview: {kind: 'points', caption: 'Nearest in space and in time.', seed: 611},
    related: ['knox-test', 'local-knox', 'near-repeat-knox', 'mantel-test'],
    reference: DYNAMICS,
    maturity: 'stable'
  },
  {
    id: 'local-knox',
    symbol: 'GPULocalKnox',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Local Knox statistic',
    summary:
      'Counts, for every event, neighbors close in both space and time, with a p-value each.',
    explanation:
      'Maps where space-time interaction happens rather than only whether it does. Each event keeps its own time while the others are permuted, as pointpats `KnoxLocal`; the p-values are uncorrected per event.',
    dataTypes: {input: ['points', 'spatial-weights', 'time-series'], output: ['points', 'table']},
    useCases: ['find-patterns', 'model'],
    domains: ['spatial-statistics'],
    params: [
      {
        name: 'pairs',
        type: 'GPUSpatialWeights',
        required: true,
        description: 'Events within the spatial threshold, both directions (radius search).'
      },
      timesParameter,
      spaceTimeParameters,
      maximumPermutationsParameter,
      {
        name: 'output',
        type: 'GPULocalKnoxOutput',
        required: true,
        description: 'Per-event summary view.'
      }
    ],
    outputs: [
      {
        name: 'summary',
        type: 'GraphDataView<"float32">',
        description: 'Per event: count, permutation mean, p-values (GPU_LOCAL_KNOX_SUMMARY).'
      }
    ],
    example: `import {
  GPULocalKnox,
  GPU_LOCAL_KNOX_SUMMARY_LENGTH,
  getGPUSpaceTimeParameterValues
} from '${IMPORT_PATH}';

// pairs: radius-mode GPUNeighborSearch output (1 km); times in days
graph.add(new GPULocalKnox({
  pairs, times, maximumPermutations: 199,
  parameters: graph.importBuffer(parameterBuffer, 'uint32'),
  output: {summary} // rows * GPU_LOCAL_KNOX_SUMMARY_LENGTH floats
}));
parameterBuffer.write(
  getGPUSpaceTimeParameterValues({seed: 7, permutationCount: 199, timeThreshold: 3})
);`,
    preview: {kind: 'hotspots', caption: 'Events that interact in space and time.', seed: 612},
    related: ['knox-test', 'near-repeat-knox', 'jacquez-test'],
    reference: DYNAMICS,
    maturity: 'stable'
  },
  {
    id: 'near-repeat-knox',
    symbol: 'GPUNearRepeatKnox',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Near-repeat Knox table',
    summary: 'Knox ratio and p-value for every cell of a distance band by time band table.',
    explanation:
      'The near-repeat calculator of crime and outbreak analysis: how far and how soon a follow-up event is more likely than chance. Each cell compares its pair count with the same cell under permuted times.',
    dataTypes: {input: ['points', 'spatial-weights', 'time-series'], output: ['table']},
    useCases: ['find-patterns', 'model'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      {
        name: 'pairs',
        type: 'GPUSpatialWeights',
        required: true,
        description: 'Pairs with distances up to the last distance edge.'
      },
      timesParameter,
      {
        name: 'parameters',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'getGPUSpaceTimeParameterValues: seed and permutation count.'
      },
      {
        name: 'distanceEdges',
        type: 'readonly number[]',
        required: true,
        description: 'Ascending distance band edges, 2 to 17.'
      },
      {
        name: 'timeEdges',
        type: 'readonly number[]',
        required: true,
        description: 'Ascending time band edges in the unit of times.'
      },
      maximumPermutationsParameter,
      {
        name: 'output',
        type: 'GPUNearRepeatKnoxOutput',
        required: true,
        description: 'Counts and per-cell summary views.'
      }
    ],
    outputs: [
      {
        name: 'counts',
        type: 'GraphDataView<"uint32">',
        description: 'Pair counts per cell, observed then each permutation.'
      },
      {
        name: 'summary',
        type: 'GraphDataView<"float32">',
        description: 'Per cell: counts, Knox ratio, p-values (GPU_NEAR_REPEAT_CELL_SUMMARY).'
      }
    ],
    example: `import {GPUNearRepeatKnox, getGPUSpaceTimeParameterValues} from '${IMPORT_PATH}';

// pairs: radius-mode GPUNeighborSearch with distances, radius = the last distance edge
graph.add(new GPUNearRepeatKnox({
  pairs, times,
  distanceEdges: [0, 200, 400, 800], // metres
  timeEdges: [0, 7, 14, 28], // days
  maximumPermutations: 199,
  parameters: graph.importBuffer(parameterBuffer, 'uint32'),
  output: {counts, summary}
}));
parameterBuffer.write(getGPUSpaceTimeParameterValues({seed: 5, permutationCount: 199}));`,
    preview: {kind: 'histogram', caption: 'How near and how soon repeats cluster.', seed: 613},
    related: ['knox-test', 'local-knox', 'space-time-diggle-test'],
    reference: DYNAMICS,
    maturity: 'stable'
  },
  {
    id: 'space-time-hawkes',
    symbol: 'GPUSpaceTimeHawkes',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Space-time Hawkes process',
    summary:
      'Fits a self-exciting point process: background rate, offspring per event, decay and spread.',
    explanation:
      'For aftershocks, contagion and repeat events: separates background events from triggered ones. EM with bounded parent windows fits an exponential time kernel and a Gaussian offspring displacement, and gives each event its probability of being background (one constant background, no magnitude term, so not ETAS).',
    dataTypes: {input: ['points', 'time-series'], output: ['points', 'table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics', 'hydrology'],
    params: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Packed planar event positions.'
      },
      {
        name: 'times',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Event times in [0, windowDuration].'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'getGPUSpaceTimeHawkesParameterValues: window, bounds, start values.'
      },
      {
        name: 'maximumIterations',
        type: 'number | undefined',
        default: '100',
        description: 'EM iterations unrolled into the graph.'
      },
      {
        name: 'maximumParents',
        type: 'number | undefined',
        default: '256',
        description: 'Earlier events examined as parents of each event.'
      },
      {
        name: 'output',
        type: 'GPUSpaceTimeHawkesOutput',
        required: true,
        description: 'Estimates and optional per-event views.'
      }
    ],
    outputs: [
      {
        name: 'estimates',
        type: 'GraphDataView<"float32">',
        description: 'Background mu, branching ratio omega, decay beta, spread sigma.'
      },
      {
        name: 'backgroundProbability',
        type: 'GraphDataView<"float32">',
        description: 'Optional probability that each event is background.'
      },
      {
        name: 'logLikelihood',
        type: 'GraphDataView<"float32">',
        description:
          'Optional diagnostic log-likelihood at the parameters entering the last update.'
      },
      {
        name: 'fittedLogLikelihood',
        type: 'GraphDataView<"float32">',
        description:
          'Optional log-likelihood at the published fitted estimates, for fit comparison.'
      }
    ],
    example: `import {
  GPUSpaceTimeHawkes,
  getGPUSpaceTimeHawkesParameterValues
} from '${IMPORT_PATH}';

graph.add(new GPUSpaceTimeHawkes({
  positions, times, maximumIterations: 60,
  parameters: graph.importBuffer(parameterBuffer, 'float32'),
  output: {estimates, backgroundProbability}
}));
parameterBuffer.write(getGPUSpaceTimeHawkesParameterValues({
  windowArea: 50, windowDuration: 120, // km², days
  maximumLag: 10, maximumDistance: 3,
  initialBackground: 0.02, initialBranchingRatio: 0.5,
  initialDecayRate: 0.5, initialSpatialStandardDeviation: 0.5
}));`,
    preview: {kind: 'points', caption: 'Background and triggered events.', seed: 614},
    related: ['space-time-kernel-density', 'knox-test'],
    reference: 'api/gpu-spatial-analysis/space-time-hawkes',
    maturity: 'stable'
  },
  {
    id: 'space-time-kernel-density',
    symbol: 'GPUSpaceTimeKernelDensity',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Space-time kernel density',
    summary: 'Kernel intensity of events on a voxel cube: a density surface for every time slice.',
    explanation:
      'Shows where and when events concentrate, for example lightning moving with a storm. A product of a spatial and a temporal kernel with optional uniform edge correction, as sparr `spattemp.density`.',
    dataTypes: {input: ['points', 'time-series'], output: ['raster']},
    useCases: ['interpolate', 'summarize'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Packed point positions in grid units.'
      },
      timesParameter,
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'getGPUSpaceTimeKernelDensityParameterValues: bandwidths, slices.'
      },
      {name: 'grid', type: 'GPURasterMetadata', required: true, description: 'Output raster grid.'},
      {name: 'sliceCount', type: 'number', required: true, description: 'Number of time slices.'},
      {
        name: 'temporalKernel',
        type: 'GPUSpaceTimeTemporalKernel | undefined',
        default: "'gaussian'",
        description: 'Temporal kernel: gaussian, quartic or epanechnikov.'
      },
      {
        name: 'output',
        type: 'GPUSpaceTimeKernelDensityOutput',
        required: true,
        description: 'Voxel values view.'
      }
    ],
    moreRequired: ['indexGridSize', 'indexBounds'],
    outputs: [
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        description: 'Density per voxel, cell-major (cell * sliceCount + slice).'
      }
    ],
    example: `import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {
  GPUSpaceTimeKernelDensity,
  getGPUSpaceTimeKernelDensityParameterValues
} from '${IMPORT_PATH}';

const grid = makeGPURasterMetadata({width: 128, height: 80, cellSize: 1000, origin: [0, 80000]});
graph.add(new GPUSpaceTimeKernelDensity({
  positions, times, grid, sliceCount: 24,
  indexGridSize: [64, 40], indexBounds: [0, 0, 128000, 80000],
  parameters: graph.importBuffer(parameterBuffer, 'float32'),
  output: {values}
}));
parameterBuffer.write(getGPUSpaceTimeKernelDensityParameterValues(
  {bandwidth: 5000, temporalBandwidth: 1, firstSliceTime: 0, sliceInterval: 1},
  grid, 24
));`,
    preview: {kind: 'density-surface', caption: 'Event density at one time slice.', seed: 615},
    related: [
      'space-time-relative-risk',
      'kernel-density',
      'adaptive-bandwidth-kernel-density',
      'emerging-hot-spots'
    ],
    reference: DENSITY_AND_K,
    maturity: 'stable'
  },
  {
    id: 'space-time-relative-risk',
    symbol: 'GPUSpaceTimeRelativeRisk',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Space-time relative risk',
    summary:
      'Ratio of case to control space-time densities: where and when cases are over-represented.',
    explanation:
      'For disease surveillance with a population at risk that changes over time. Case and control densities share bandwidths, kernels and grid; the result is log(f / g) by default, as sparr `spattemp.risk`.',
    dataTypes: {input: ['points', 'time-series'], output: ['raster']},
    useCases: ['find-patterns', 'interpolate'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      {
        name: 'casePositions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Case positions.'
      },
      {
        name: 'caseTimes',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Case times.'
      },
      {
        name: 'controlPositions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Control positions.'
      },
      {
        name: 'controlTimes',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Control times, same unit and origin as the cases.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'getGPUSpaceTimeKernelDensityParameterValues for both sets.'
      },
      {name: 'grid', type: 'GPURasterMetadata', required: true, description: 'Output raster grid.'},
      {
        name: 'scale',
        type: '"log" | "ratio" | undefined',
        default: "'log'",
        description: 'Log ratio or plain ratio.'
      },
      {
        name: 'output',
        type: 'GPUSpaceTimeRelativeRiskOutput',
        required: true,
        description: 'Risk and optional density views.'
      }
    ],
    moreRequired: ['sliceCount', 'indexGridSize', 'indexBounds'],
    outputs: [
      {
        name: 'values',
        type: 'GraphDataView<"float32">',
        description: 'Risk per voxel, cell-major.'
      },
      {
        name: 'caseDensity',
        type: 'GraphDataView<"float32">',
        description: 'Optional case probability density per voxel.'
      },
      {
        name: 'controlDensity',
        type: 'GraphDataView<"float32">',
        description: 'Optional control probability density per voxel.'
      }
    ],
    example: `import {
  GPUSpaceTimeRelativeRisk,
  getGPUSpaceTimeKernelDensityParameterValues
} from '${IMPORT_PATH}';

graph.add(new GPUSpaceTimeRelativeRisk({
  casePositions, caseTimes, controlPositions, controlTimes,
  grid, sliceCount: 12, indexGridSize: [64, 40], indexBounds,
  parameters: graph.importBuffer(parameterBuffer, 'float32'),
  output: {values: logRisk}
}));
parameterBuffer.write(getGPUSpaceTimeKernelDensityParameterValues(
  {bandwidth: 2000, temporalBandwidth: 7, firstSliceTime: 0, sliceInterval: 7},
  grid, 12
));`,
    preview: {kind: 'density-surface', caption: 'Where cases outpace controls.', seed: 616},
    related: ['space-time-kernel-density', 'expectation-based-space-time-scan'],
    reference: DENSITY_AND_K,
    maturity: 'partial'
  },
  {
    id: 'adaptive-bandwidth-kernel-density',
    symbol: 'GPUAdaptiveBandwidthKernelDensity',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Adaptive-bandwidth kernel density',
    summary:
      'Kernel density whose bandwidth narrows where points are dense and widens where sparse.',
    explanation:
      'Keeps dense clusters sharp without breaking scattered points into speckles. A fixed-bandwidth pilot sets each point bandwidth by Abramson’s square-root law, clamped to `[h / L, h L]`.',
    dataTypes: {input: ['points'], output: ['raster']},
    useCases: ['interpolate', 'summarize'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Packed point positions in grid units.'
      },
      {
        name: 'weights',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Optional non-negative point weights.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'getGPUAdaptiveBandwidthKernelDensityParameterValues: h, sensitivity.'
      },
      {name: 'grid', type: 'GPURasterMetadata', required: true, description: 'Output raster grid.'},
      {
        name: 'bandwidthFactorLimit',
        type: 'number | undefined',
        default: '4',
        description: 'Clamp L of each point bandwidth; sets the search radius.'
      },
      {
        name: 'output',
        type: 'GPUAdaptiveBandwidthKernelDensityOutput',
        required: true,
        description: 'Density and optional per-point views.'
      }
    ],
    moreRequired: ['indexGridSize', 'indexBounds'],
    outputs: [
      {name: 'values', type: 'GraphDataView<"float32">', description: 'Density per output cell.'},
      {
        name: 'pointBandwidths',
        type: 'GraphDataView<"float32">',
        description: 'Optional bandwidth of each point.'
      },
      {
        name: 'pilotDensities',
        type: 'GraphDataView<"float32">',
        description: 'Optional fixed-bandwidth pilot density at each point.'
      }
    ],
    example: `import {
  GPUAdaptiveBandwidthKernelDensity,
  getGPUAdaptiveBandwidthKernelDensityParameterValues
} from '${IMPORT_PATH}';

graph.add(new GPUAdaptiveBandwidthKernelDensity({
  positions, grid, indexGridSize: [64, 40], indexBounds,
  bandwidthFactorLimit: 4,
  parameters: graph.importBuffer(parameterBuffer, 'float32'),
  output: {values: density, pointBandwidths}
}));
parameterBuffer.write(getGPUAdaptiveBandwidthKernelDensityParameterValues(
  {bandwidth: 5000, sensitivity: 0.5},
  grid
));`,
    preview: {kind: 'density-surface', caption: 'Bandwidth follows the point density.', seed: 617},
    related: ['kernel-density', 'space-time-kernel-density'],
    reference: 'api/gpu-spatial-analysis/colocation-category-dependence-adaptive-density',
    maturity: 'stable'
  },
  {
    id: 'space-time-k-function',
    symbol: 'GPUSpaceTimeKFunction',
    kind: 'class',
    importPath: DATAFRAME_PATH,
    title: 'Space-time K function',
    summary:
      'Space-time K, pair correlation and Diggle D of a point pattern over distance and lag.',
    explanation:
      'Describes space-time clustering at every distance and time lag at once. `D(u, v) = K(u, v) - K_s(u) K_t(v)` is zero without interaction; isotropic edge correction follows Gabriel and Diggle (2009).',
    dataTypes: {input: ['points', 'time-series'], output: ['table']},
    useCases: ['measure', 'find-patterns'],
    domains: ['spatial-statistics', 'ecology'],
    params: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Packed planar event positions.'
      },
      {
        name: 'times',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Relative event times.'
      },
      {
        name: 'intensities',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Optional first-order intensity per event (inhomogeneous K).'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'getGPUSpaceTimeKFunctionParameterValues: window, extents.'
      },
      {
        name: 'gridSize',
        type: 'readonly [number, number]',
        required: true,
        description: 'Pair-search lattice size; speed only.'
      },
      {
        name: 'distanceClassCount',
        type: 'number',
        required: true,
        description: 'Distance classes U.'
      },
      {name: 'timeClassCount', type: 'number', required: true, description: 'Time-lag classes V.'},
      {
        name: 'output',
        type: 'GPUSpaceTimeKFunctionOutput',
        required: true,
        description: 'K and optional D, marginals, pair correlation.'
      }
    ],
    outputs: [
      {
        name: 'k',
        type: 'GraphDataView<"float32">',
        description: 'Cumulative K per (distance, lag) class.'
      },
      {
        name: 'd',
        type: 'GraphDataView<"float32">',
        description: 'Optional Diggle D, laid out like k.'
      },
      {
        name: 'pairCorrelation',
        type: 'GraphDataView<"float32">',
        description: 'Optional space-time pair correlation g(u, v).'
      }
    ],
    example: `import {
  GPUSpaceTimeKFunction,
  getGPUSpaceTimeKFunctionParameterValues
} from '${DATAFRAME_PATH}';

graph.add(new GPUSpaceTimeKFunction({
  positions, times, gridSize: [32, 32],
  distanceClassCount: 12, timeClassCount: 10,
  parameters: graph.importBuffer(parameterBuffer, 'float32'),
  output: {k, d}
}));
parameterBuffer.write(getGPUSpaceTimeKFunctionParameterValues({
  bounds: [0, 0, 1000, 800], timeWindow: [0, 18],
  maximumDistance: 200, maximumTimeLag: 3
}));`,
    preview: {kind: 'histogram', caption: 'Clustering by distance and lag.', seed: 618},
    related: ['space-time-diggle-test', 'ripley-k', 'space-time-variogram'],
    reference: DENSITY_AND_K,
    maturity: 'stable'
  },
  {
    id: 'space-time-diggle-test',
    symbol: 'GPUSpaceTimeDiggleTest',
    kind: 'class',
    importPath: DATAFRAME_PATH,
    title: 'Space-time Diggle test',
    summary: 'Monte Carlo test of Diggle D(u, v) against permuted event times.',
    explanation:
      'Tests space-time interaction over a whole grid of distances and lags, not one threshold. The statistic U is the sum of D over every class, as splancs `stmctest` computes it, after Diggle, Chetwynd, Haggkvist and Morris (1995).',
    dataTypes: {input: ['points', 'time-series'], output: ['table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics'],
    params: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Packed planar event positions.'
      },
      {
        name: 'times',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Relative event times.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'getGPUSpaceTimeKFunctionParameterValues: window, extents.'
      },
      {
        name: 'randomParameters',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'getGPUSpaceTimeDiggleRandomParameterValues(seed).'
      },
      {
        name: 'distanceClassCount',
        type: 'number',
        required: true,
        description: 'Distance classes U.'
      },
      {name: 'timeClassCount', type: 'number', required: true, description: 'Time-lag classes V.'},
      {
        name: 'permutationCount',
        type: 'number',
        required: true,
        description: 'Time permutations, 1 to 999; compile-time.'
      },
      {
        name: 'output',
        type: 'GPUSpaceTimeDiggleTestOutput',
        required: true,
        description: 'D, statistics and summary views.'
      }
    ],
    moreRequired: ['gridSize'],
    outputs: [
      {name: 'd', type: 'GraphDataView<"float32">', description: 'Observed D per class.'},
      {
        name: 'statistics',
        type: 'GraphDataView<"float32">',
        description: 'Observed statistic U, then U under each permutation.'
      },
      {
        name: 'summary',
        type: 'GraphDataView<"float32">',
        description: 'Statistic, p-values, z-score (GPU_SPACE_TIME_DIGGLE_SUMMARY).'
      },
      {
        name: 'permutationStandardDeviation',
        type: 'GraphDataView<"float32">',
        description: 'Optional permutation spread of D per class.'
      }
    ],
    example: `import {
  GPUSpaceTimeDiggleTest,
  getGPUSpaceTimeDiggleRandomParameterValues,
  getGPUSpaceTimeKFunctionParameterValues
} from '${DATAFRAME_PATH}';

graph.add(new GPUSpaceTimeDiggleTest({
  positions, times, gridSize: [16, 16],
  distanceClassCount: 6, timeClassCount: 6, permutationCount: 99,
  parameters: graph.importBuffer(parameterBuffer, 'float32'),
  randomParameters: graph.importBuffer(seedBuffer, 'uint32'),
  output: {d, statistics, summary}
}));
seedBuffer.write(getGPUSpaceTimeDiggleRandomParameterValues(11));
parameterBuffer.write(getGPUSpaceTimeKFunctionParameterValues({
  bounds, timeWindow: [0, 121], maximumDistance: 8, maximumTimeLag: 16
}));`,
    preview: {kind: 'histogram', caption: 'D against shuffled times.', seed: 619},
    related: ['space-time-k-function', 'knox-test', 'near-repeat-knox'],
    reference: DENSITY_AND_K,
    maturity: 'stable'
  },
  {
    id: 'expectation-based-space-time-scan',
    symbol: 'GPUExpectationBasedSpaceTimeScan',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Expectation-based space-time scan',
    summary:
      'Finds the zones whose latest counts most exceed their baseline, with a Monte Carlo p-value.',
    explanation:
      'Prospective surveillance: every window ends at the latest bucket and counts are compared with a supplied baseline, so even a rise everywhere is found. Fast Subset Scan (Neill 2012) searches every subset of each neighborhood exactly.',
    dataTypes: {input: ['points', 'time-series'], output: ['table', 'polygons']},
    useCases: ['detect-change', 'find-patterns'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      {
        name: 'positions',
        type: 'GraphDataView<"float32x2">',
        required: true,
        description: 'Zone centroids.'
      },
      {
        name: 'cases',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'Observed counts, zone * timeBuckets + bucket.'
      },
      {
        name: 'baseline',
        type: 'GraphDataView<"float32">',
        required: true,
        description: 'Expected counts in the same layout.'
      },
      {name: 'timeBuckets', type: 'number', required: true, description: 'Time buckets per zone.'},
      {
        name: 'maximumNeighbors',
        type: 'number',
        required: true,
        description: 'Neighborhood size of each center, at most 64.'
      },
      {
        name: 'maximumWindowLength',
        type: 'number',
        required: true,
        description: 'Longest window in buckets, at most 32.'
      },
      {
        name: 'parameters',
        type: 'GraphDataView<"uint32">',
        required: true,
        description: 'getGPUExpectationBasedScanParameterValues: seed, replicates.'
      },
      {
        name: 'output',
        type: 'GPUExpectationBasedSpaceTimeScanOutput',
        required: true,
        description: 'Cluster, statistics and summary views.'
      }
    ],
    moreRequired: ['maximumReplicates'],
    outputs: [
      {
        name: 'clusterZones',
        type: 'GraphDataView<"uint32">',
        description: 'Zones of the most likely cluster.'
      },
      {
        name: 'clusterStatistics',
        type: 'GraphDataView<"float32">',
        description: 'Score, observed, expected, p-value.'
      },
      {
        name: 'statistics',
        type: 'GraphDataView<"float32">',
        description: 'Observed maximum score, then one per replicate.'
      },
      {
        name: 'summary',
        type: 'GraphDataView<"uint32">',
        description: 'Found, members, window length, center, replicates.'
      },
      {
        name: 'zoneStatistics',
        type: 'GraphDataView<"float32">',
        description: 'Optional best score centered at each zone.'
      }
    ],
    example: `import {
  GPUExpectationBasedSpaceTimeScan,
  getGPUExpectationBasedScanParameterValues
} from '${IMPORT_PATH}';

// cases and baseline: zones x 16 buckets; baseline = moving average of earlier buckets
graph.add(new GPUExpectationBasedSpaceTimeScan({
  positions, cases, baseline, timeBuckets: 16,
  maximumNeighbors: 9, maximumWindowLength: 8, maximumReplicates: 199,
  parameters: graph.importBuffer(parameterBuffer, 'uint32'),
  output: {clusterZones, clusterStatistics, statistics, summary, zoneStatistics}
}));
parameterBuffer.write(
  getGPUExpectationBasedScanParameterValues({seed: 2, replicateCount: 99})
);`,
    preview: {kind: 'hotspots', caption: 'Zones rising above their baseline.', seed: 620},
    related: ['spatial-scan-statistic', 'emerging-hot-spots', 'space-time-relative-risk'],
    reference: 'api/gpu-spatial-analysis/change-points-and-expectation-based-scan',
    maturity: 'stable'
  }
];
