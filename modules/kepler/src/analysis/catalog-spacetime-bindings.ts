// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {CatalogOperationBinding, CatalogParameterSpec} from './catalog-binding-types';
import {
  COORDINATE_SYSTEM_PARAMETER,
  PERMUTATION_PARAMETER_SPEC,
  SEED_PARAMETER_SPEC,
  makeInputSpec,
  makeNumberParameter,
  makeOutputSpec,
  makeStringParameter
} from './catalog-platform3-b-common';
import {SPACETIME_MAXIMUM_FEATURES, SPACETIME_MAXIMUM_VOXELS} from './catalog-spacetime-common';
import {runSpacetimeDensity} from './catalog-spacetime-density';
import {runSpacetimeEvents} from './catalog-spacetime-events';
import {runSpacetimeSeries} from './catalog-spacetime-series';

const TIME_PARAMETERS: readonly CatalogParameterSpec[] = [
  COORDINATE_SYSTEM_PARAMETER,
  makeStringParameter('timeProperty', 'Time column', 'time'),
  {
    name: 'timeUnit',
    label: 'Input epoch time unit',
    kind: 'enum',
    options: ['milliseconds', 'seconds'],
    defaultValue: 'milliseconds'
  }
];
const RANDOM_PARAMETERS = [
  {...PERMUTATION_PARAMETER_SPEC, minimum: 1, defaultValue: 19},
  SEED_PARAMETER_SPEC
];
const SPACE_PARAMETER = makeNumberParameter(
  'spatialThreshold',
  'Spatial threshold',
  1000,
  1e-6,
  1e7,
  'meters or planar units'
);
const TIME_PARAMETER = makeNumberParameter(
  'temporalThreshold',
  'Temporal threshold',
  60,
  1e-6,
  1e9,
  'seconds'
);
const NEIGHBOR_PARAMETER = makeNumberParameter(
  'neighborCount',
  'Nearest neighbors',
  3,
  1,
  32,
  'neighbors',
  'integer'
);
const SERIES_PARAMETERS = [
  makeStringParameter('locationProperty', 'Location ID column (blank uses coordinates)'),
  makeStringParameter('valueProperty', 'Value column', 'value')
];
const DENSITY_PARAMETERS = [
  makeNumberParameter(
    'spatialBandwidth',
    'Spatial bandwidth',
    250,
    1e-6,
    1e7,
    'meters or planar units'
  ),
  makeNumberParameter('gridSize', 'Grid columns and rows', 4, 1, 32, 'cells per axis', 'integer')
];
const TEMPORAL_DENSITY_PARAMETERS = [
  makeNumberParameter('temporalBandwidth', 'Temporal bandwidth', 60, 1e-6, 1e9, 'seconds'),
  makeNumberParameter('sliceCount', 'Output time slices', 4, 1, 16, 'slices', 'integer')
];
const PAIR_PARAMETERS = [
  makeNumberParameter(
    'spatialBandwidth',
    'Study window padding',
    250,
    1e-6,
    1e7,
    'meters or planar units'
  ),
  makeNumberParameter(
    'maximumDistance',
    'Maximum distance',
    500,
    1e-6,
    1e7,
    'meters or planar units'
  ),
  makeNumberParameter('maximumTimeLag', 'Maximum time lag', 120, 1e-6, 1e9, 'seconds'),
  makeNumberParameter(
    'classCount',
    'Distance and time classes',
    3,
    1,
    8,
    'classes per axis',
    'integer'
  )
];
const POINT_OUTPUT = makeOutputSpec(
  'points',
  'Spacetime point results',
  'point',
  ['source_index', 'source_id'],
  SPACETIME_MAXIMUM_FEATURES
);
const GRID_OUTPUT = makeOutputSpec(
  'grid',
  'Spacetime density grid',
  'point',
  ['voxel_index', 'cell_index', 'slice_index'],
  SPACETIME_MAXIMUM_VOXELS
);
const SUMMARY_OUTPUT = makeOutputSpec(
  'statistics',
  'Spacetime statistics',
  'none',
  ['statistic'],
  64
);
const BAND_OUTPUT = makeOutputSpec(
  'bands',
  'Spacetime distance/time bands',
  'none',
  ['distance_class', 'time_class'],
  64
);
const COMMON_BEHAVIOR = [
  'Requires a finite Point and a finite numeric time column on every input row; invalid rows fail before dispatch. Epoch milliseconds or seconds are shifted in float64 and converted to relative seconds before upload; all temporal controls use seconds.',
  'EPSG:4326 uses a local azimuthal equidistant frame in meters; planar coordinates retain their units. Results use float32. No CPU fallback. Output null means undefined or not tested.',
  'The library float32 time-resolution guard rejects spans too coarse for the selected bandwidth or time band; series use their observed period spacing. Adaptive spatial density does not need the time-resolution guard.',
  'Source-sized output tables carry source_index and source_id plus geometry for resident publication; grid and band tables carry their own cell identities.'
];
const SERIES_BEHAVIOR =
  'Series group by the Location ID column, or exact Point coordinates when blank. Requires a complete rectangle with unique location/time rows, fixed location coordinates, and equally spaced periods, sorted by time. At most 64 locations and 128 periods within the 512-row input limit.';
const CLEANUP =
  'GPU only. withCatalogGpuSession destroys all buffers and compiled graphs after readback; plain tables are staged through native Sources.';

function makeBinding(
  id: string,
  title: string,
  contributor: string,
  parameters: readonly CatalogParameterSpec[],
  outputs: CatalogOperationBinding['outputs'],
  behavior: string[],
  limitation: string,
  runGpu: NonNullable<CatalogOperationBinding['runGpu']>,
  capability = 'spatial-autocorrelation-and-inference'
): CatalogOperationBinding {
  return {
    id,
    title,
    domain: 'analysis',
    summary: behavior[0],
    capabilityIds: [capability],
    contributorNames: [contributor],
    gpuContributorNames: [contributor],
    engine: 'gpu',
    inputs: [
      makeInputSpec(
        'points',
        'Event points',
        ['point'],
        SPACETIME_MAXIMUM_FEATURES,
        SPACETIME_MAXIMUM_FEATURES
      )
    ],
    parameters: [...TIME_PARAMETERS, ...parameters],
    outputs,
    cleanup: CLEANUP,
    declaredBehavior: [...COMMON_BEHAVIOR, ...behavior],
    limitation: `GPU only; maximum 512 input rows. ${limitation}`,
    runGpu
  };
}

/** Space-time UI wrappers of existing analysis contributors; no example-local numerical implementations. */
export const SPACETIME_BINDINGS: readonly CatalogOperationBinding[] = [
  makeBinding(
    'space-time-kernel-density',
    'Space-time kernel density',
    'GPUSpaceTimeKernelDensity',
    [...DENSITY_PARAMETERS, ...TEMPORAL_DENSITY_PARAMETERS],
    [GRID_OUTPUT],
    [
      'Product Gaussian kernel density on a cell-major space-time grid with intensity normalization and no edge correction.',
      'Grid covers the event bounds padded by the spatial bandwidth. Slices span the positive observed time range, including subsecond spans; equal-time observations produce one slice.'
    ],
    '32 × 32 cells, 16 slices (16,384 voxels). Gaussian kernels truncated at five bandwidths; no edge correction.',
    (context, inputs, values) => runSpacetimeDensity('density', context, inputs, values),
    'density-and-interpolation'
  ),
  makeBinding(
    'space-time-relative-risk',
    'Space-time relative risk',
    'GPUSpaceTimeRelativeRisk',
    [
      ...DENSITY_PARAMETERS,
      ...TEMPORAL_DENSITY_PARAMETERS,
      makeStringParameter('caseProperty', 'Case indicator column (0 control, 1 case)', 'case')
    ],
    [GRID_OUTPUT],
    [
      'Case/control probability density ratio on a space-time grid.',
      'Slices span the observed time range; equal-time observations produce one slice. Both populations share the frame, origin, bandwidths and grid; a binary case indicator is required, with at least one case and one control.'
    ],
    'Same 16,384 voxel limit; ratio scale, no edge correction. Zero control density yields null.',
    (context, inputs, values) => runSpacetimeDensity('risk', context, inputs, values),
    'density-and-interpolation'
  ),
  makeBinding(
    'adaptive-bandwidth-kernel-density',
    'Adaptive bandwidth kernel density',
    'GPUAdaptiveBandwidthKernelDensity',
    [
      ...DENSITY_PARAMETERS,
      makeNumberParameter('sensitivity', 'Abramson sensitivity', 0.5, 0, 1),
      makeNumberParameter('factorLimit', 'Bandwidth factor limit', 4, 1, 16)
    ],
    [GRID_OUTPUT, POINT_OUTPUT],
    [
      'Spatial Abramson adaptive Gaussian density with pilot density and bandwidth per source point.',
      'The time column is mapped for identity metadata; the spatial contributor does not use time.'
    ],
    'Spatial only; 1,024 grid cells. Pilot bandwidths clamped by the selected factor limit.',
    (context, inputs, values) => runSpacetimeDensity('adaptive', context, inputs, values),
    'density-and-interpolation'
  ),
  makeBinding(
    'space-time-hawkes',
    'Space-time Hawkes',
    'GPUSpaceTimeHawkes',
    [
      SPACE_PARAMETER,
      TIME_PARAMETER,
      makeNumberParameter(
        'maximumIterations',
        'Maximum EM iterations',
        20,
        1,
        100,
        'iterations',
        'integer'
      ),
      makeNumberParameter(
        'maximumParents',
        'Maximum parents per event',
        64,
        1,
        512,
        'parents',
        'integer'
      )
    ],
    [POINT_OUTPUT, SUMMARY_OUTPUT],
    [
      'Fit an exponential-time Gaussian-space Hawkes process by EM; publish per-event background probabilities and fitted parameters.',
      'Window area comes from the padded point bounds, duration from the observed time span (at least one second). Initial branching ratio is 0.5, decay is inverse temporal threshold, spatial standard deviation is the spatial threshold; initial background is half the observed rate.'
    ],
    'At most 100 EM iterations and 512 parents per event. Candidate truncation and convergence status are reported; threshold bounds truncate triggering.',
    (context, inputs, values) => runSpacetimeEvents('hawkes', context, inputs, values)
  ),
  makeBinding(
    'expectation-based-space-time-scan',
    'Expectation-based space-time scan',
    'GPUExpectationBasedSpaceTimeScan',
    [
      ...SERIES_PARAMETERS,
      makeStringParameter('baselineProperty', 'Expected count column', 'baseline'),
      ...RANDOM_PARAMETERS,
      makeNumberParameter(
        'maximumWindowLength',
        'Maximum scan window',
        4,
        1,
        32,
        'periods',
        'integer'
      ),
      NEIGHBOR_PARAMETER
    ],
    [POINT_OUTPUT, SUMMARY_OUTPUT],
    [
      'Expectation-based Poisson scan against resident expected counts, with cylinders ending at the latest period.',
      SERIES_BEHAVIOR,
      'Value column must contain integer counts from 0 to 16,777,215 and baseline must be strictly positive. Monte Carlo replicates are controlled by permutations and seed.'
    ],
    'At most 32 periods per scan window; nearest-zone count is clamped to the location count. 999 Poisson replicates; p-value is for the selected cluster.',
    (context, inputs, values) => runSpacetimeSeries('scan', context, inputs, values)
  ),
  makeBinding(
    'space-time-cross-correlation',
    'Space-time cross correlation',
    'GPUSpaceTimeCrossCorrelation',
    [
      ...SERIES_PARAMETERS,
      makeStringParameter('secondValueProperty', 'Second value column', 'second'),
      makeNumberParameter('lag', 'Signed correlation lag', 1, -127, 127, 'periods', 'integer')
    ],
    [POINT_OUTPUT],
    [
      'Pearson correlation between two location series at a signed period lag; Fisher approximation p-values.',
      SERIES_BEHAVIOR
    ],
    'One selected lag; at least four overlapping periods required. Unadjusted p-values, no multiple testing correction.',
    (context, inputs, values) => runSpacetimeSeries('correlation', context, inputs, values)
  ),
  makeBinding(
    'space-time-local-moran',
    'Space-time local Moran',
    'GPUSpaceTimeLocalMoran',
    [
      ...SERIES_PARAMETERS,
      ...RANDOM_PARAMETERS,
      NEIGHBOR_PARAMETER,
      makeNumberParameter('lag', 'Moran lag', 1, 1, 127, 'periods', 'integer'),
      {
        name: 'variant',
        label: 'Moran variant',
        kind: 'enum',
        options: ['lagged', 'differential'],
        defaultValue: 'lagged'
      }
    ],
    [POINT_OUTPUT],
    [
      'Lagged inward or differential local Moran at the latest period with conditional permutation p-values.',
      SERIES_BEHAVIOR,
      'Spatial weights use the selected k nearest locations with equal positive weights; only the latest-period source rows receive values.'
    ],
    'At least four locations; at most 32 neighbors and 999 permutations. Two-sided p-values without multiple testing correction.',
    (context, inputs, values) => runSpacetimeSeries('moran', context, inputs, values)
  ),
  makeBinding(
    'jacquez-test',
    'Jacquez space-time test',
    'GPUJacquezTest',
    [NEIGHBOR_PARAMETER, ...RANDOM_PARAMETERS],
    [SUMMARY_OUTPUT],
    [
      'Directed overlap of spatial and temporal k-nearest event neighbors with time-permutation inference.',
      'Spatial ties use source row order; temporal ties use the contributor’s stable time order.'
    ],
    'Global statistic only: the contributor has no per-point output. Neighbor count must be below the event count; 999 permutations, greater-tail p-value.',
    (context, inputs, values) => runSpacetimeEvents('jacquez', context, inputs, values)
  ),
  makeBinding(
    'local-knox',
    'Local Knox',
    'GPULocalKnox',
    [SPACE_PARAMETER, TIME_PARAMETER, ...RANDOM_PARAMETERS],
    [POINT_OUTPUT],
    [
      'Per-event joint close-pair counts and permutation p-values under shuffled event times.',
      'Radius filtering compares the uploaded Float32 distances and threshold. Spatial lists include both directions; the contributor reports each event’s incident pairs.'
    ],
    'At least two events; 999 permutations. Greater-tail p-values without multiple testing correction.',
    (context, inputs, values) => runSpacetimeEvents('local-knox', context, inputs, values)
  ),
  makeBinding(
    'near-repeat-knox',
    'Near-repeat Knox',
    'GPUNearRepeatKnox',
    [
      SPACE_PARAMETER,
      TIME_PARAMETER,
      ...RANDOM_PARAMETERS,
      makeNumberParameter(
        'bandCount',
        'Distance and time bands',
        3,
        1,
        8,
        'bands per axis',
        'integer'
      )
    ],
    [BAND_OUTPUT],
    [
      'Near-repeat pair-count table over equal distance and time bands with permutation inference.',
      'Radius filtering compares uploaded Float32 distances and thresholds. Bands run from zero to the spatial and temporal thresholds; the first includes zero, subsequent bands are open on the left and closed on the right.'
    ],
    'At most 8 × 8 bands; 999 permutations. Each unordered pair counted once. Greater-tail p-values per band without correction.',
    (context, inputs, values) => runSpacetimeEvents('near-repeat', context, inputs, values)
  ),
  makeBinding(
    'space-time-diggle-test',
    'Space-time Diggle test',
    'GPUSpaceTimeDiggleTest',
    [...PAIR_PARAMETERS, ...RANDOM_PARAMETERS],
    [BAND_OUTPUT, SUMMARY_OUTPUT],
    [
      'Diggle space-time interaction surface and global time-permutation test in a padded rectangular study window.',
      'Uses the library’s splancs tie convention and no edge correction; all observations share the same relative time origin.'
    ],
    'At most 8 × 8 classes and 999 permutations; rectangular window inferred from data plus padding, observation interval padded by one second.',
    (context, inputs, values) => runSpacetimeEvents('diggle', context, inputs, values),
    'dataframe-point-patterns'
  ),
  makeBinding(
    'space-time-k-function',
    'Space-time K function',
    'GPUSpaceTimeKFunction',
    PAIR_PARAMETERS,
    [BAND_OUTPUT],
    [
      'Space-time K and interaction D over distance/time classes in a padded rectangular study window.',
      'Uses the contributor’s stpp tie convention, homogeneous intensity and no edge correction.'
    ],
    'At most 8 × 8 classes; inferred rectangular window and observation interval padded by one second. Descriptive output, no p-value.',
    (context, inputs, values) => runSpacetimeEvents('k-function', context, inputs, values),
    'dataframe-point-patterns'
  ),
  makeBinding(
    'pettitt-test',
    'Pettitt change-point test',
    'GPUPettittTest',
    SERIES_PARAMETERS,
    [POINT_OUTPUT],
    [
      'Per-location Pettitt rank change statistic, change period and approximate p-value, joined to each observation.',
      SERIES_BEHAVIOR
    ],
    '128 periods; one change per location. Approximate analytic p-values without multiple testing correction.',
    (context, inputs, values) => runSpacetimeSeries('pettitt', context, inputs, values)
  ),
  makeBinding(
    'location-change-points',
    'Location change points',
    'GPULocationChangePoints',
    [
      ...SERIES_PARAMETERS,
      makeNumberParameter('penalty', 'Segmentation penalty', 10, 1e-6, 1e9),
      makeNumberParameter(
        'minimumSegmentLength',
        'Minimum segment length',
        2,
        1,
        128,
        'periods',
        'integer'
      ),
      {
        name: 'cost',
        label: 'Segment cost',
        kind: 'enum',
        options: ['mean', 'variance', 'mean-variance', 'poisson', 'slope'],
        defaultValue: 'mean'
      }
    ],
    [POINT_OUTPUT],
    [
      'Optimal partitioning with per-observation segment labels and change-point flags.',
      SERIES_BEHAVIOR,
      'Variance costs require segments of at least two periods, slope at least three. Poisson values must be nonnegative; overflow is reported as a failure.'
    ],
    '128 periods and 32 change points per location; optimal cost and change count reported. No inference p-value.',
    (context, inputs, values) => runSpacetimeSeries('changes', context, inputs, values)
  )
];
