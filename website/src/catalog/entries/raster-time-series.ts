// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Catalog group "raster-time-series": per-pixel operations on stacks of raster time slices, on
// irregular acquisition dates or regular slots: change monitoring, smoothing, harmonic fits,
// phenology, resampling, streaming statistics and lagged correlation.

import type {CatalogEntry, OutputDoc, ParamDoc} from '../schema';

const IMPORT_PATH = '@luma-contributors/analysis/raster-analysis';

function makeParam(
  name: string,
  type: string,
  description: string,
  options: {required?: boolean; default?: string} = {}
): ParamDoc {
  return {name, type, description, ...options};
}

function makeOutput(name: string, type: string, description: string): OutputDoc {
  return {name, type, description};
}

const FLOAT = 'GraphDataView<"float32">';
const UINT = 'GraphDataView<"uint32">';
const DATE_AXIS = 'GPURasterIrregularDateAxis';
const DATE_AXIS_DESCRIPTION = 'Acquisition dates as Int64 words with a temporal context.';

const DATE_AXIS_LINE =
  "const dateAxis = {acquisitionTimes: view('dates', 'uint32x2', sliceCount), temporalContext: {unit: 'second', epoch: 0n, encoding: 'int64-words', axis: {kind: 'irregular'}} as const};";

type ExampleOptions = {
  symbols: string[];
  setup?: string[];
  /** Expression for the packed per-frame values. */
  values: string;
  constructorSymbol: string;
  props: string[];
};

/** Builds the graph, parameter buffer, compile and encode example around one contributor. */
function makeExample(options: ExampleOptions): string {
  return [
    `import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';`,
    `import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';`,
    `import {${[...options.symbols].sort().join(', ')}} from '${IMPORT_PATH}';`,
    '',
    `const graph = new GPUCommandGraph(device, {id: 'demo'});`,
    `const view = <Format extends 'float32' | 'uint32' | 'uint32x2'>(id: string, format: Format, length: number) => createTransientView(graph, id, format, length);`,
    ...(options.setup ?? []),
    `const values = ${options.values};`,
    `const parameters = new GPUParameterBuffer(device, {id: 'p', format: 'float32', length: values.length});`,
    `graph.add(new ${options.constructorSymbol}({`,
    '  parameters: parameters.importToGraph(graph),',
    ...options.props.map(line => `  ${line}`),
    '}));',
    'const compiled = await graph.compileAsync();',
    'parameters.write(values);',
    'compiled.encode(device.commandEncoder, {parameters: undefined});'
  ].join('\n');
}

export const entries: CatalogEntry[] = [
  {
    id: 'bfast-monitor',
    symbol: 'GPURasterBFASTMonitor',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'BFAST Monitor',
    summary:
      'Fits a seasonal model to a stable history per pixel and flags the first break after it.',
    explanation:
      'Use it to date a disturbance such as fire or clearing in dated scenes. Residuals after the history are summed in a moving window (MOSUM) and compared with `criticalValue * sqrt(2 * log+(t / n))`. A history that already holds the disturbance hides the break.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['detect-change', 'model'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Number of pixels.', {required: true}),
      makeParam('sliceCount', 'number', 'Observations per pixel.', {required: true}),
      makeParam('values', FLOAT, 'Series per pixel, cell-major; non-finite is missing.', {
        required: true
      }),
      makeParam('parameters', FLOAT, 'Packed period, bandwidth and criticalValue.', {
        required: true
      }),
      makeParam('dateAxis', DATE_AXIS, DATE_AXIS_DESCRIPTION, {required: true}),
      makeParam('historySliceCount', 'number', 'Leading slices that form the stable history.', {
        required: true
      }),
      makeParam('harmonicCount', 'number', 'Harmonics of the annual cycle.', {default: '3'}),
      makeParam('mask', UINT, 'Per-observation QA; zero rejects the sample.')
    ],
    outputs: [
      makeOutput('coefficients', FLOAT, 'History fit per pixel: intercept, slope, harmonics.'),
      makeOutput('scores', FLOAT, 'Per pixel median residual and largest MOSUM to boundary ratio.'),
      makeOutput('cellSummary', UINT, 'Per pixel break slice, status and history count.'),
      makeOutput('process', FLOAT, 'Optional MOSUM per monitoring slice.')
    ],
    example: makeExample({
      symbols: ['getGPURasterBFASTMonitorParameterValues', 'GPURasterBFASTMonitor'],
      setup: ['const cellCount = 256 * 256;', 'const sliceCount = 40;', DATE_AXIS_LINE],
      values: 'getGPURasterBFASTMonitorParameterValues({criticalValue: 3})',
      constructorSymbol: 'GPURasterBFASTMonitor',
      props: [
        'cellCount,',
        'sliceCount,',
        `values: view('series', 'float32', cellCount * sliceCount),`,
        'dateAxis,',
        'historySliceCount: 24,',
        'output // coefficients, scores, cellSummary, process'
      ]
    }),
    preview: {kind: 'time-series', seed: 201, caption: "A fire breaks a pixel's seasonal history."},
    related: ['cold-monitor', 'seasonal-break-detection', 'harmonic-regression'],
    reference: 'api/gpu-raster/raster-change-monitoring',
    maturity: 'partial'
  },
  {
    id: 'cold-monitor',
    symbol: 'GPURasterCOLDMonitor',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'COLD monitor',
    summary:
      'Segments every pixel series into seasonal-trend models and dates each confirmed change.',
    explanation:
      'Use it for continuous change detection over many bands and dates, as in the CCDC family. A window initializes a model, later observations are tested against it, and `consecutiveCount` anomalies in a row confirm a break and start a new segment. It is a simplified COLD: ordinary least squares and a caller-supplied minimum RMSE.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['detect-change', 'model'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Number of pixels.', {required: true}),
      makeParam('sliceCount', 'number', 'Observations per pixel.', {required: true}),
      makeParam('bandCount', 'number', 'Bands per observation.', {required: true}),
      makeParam('values', FLOAT, 'Band-interleaved series; non-finite is missing.', {
        required: true
      }),
      makeParam('parameters', FLOAT, 'Packed thresholds, RMSE floors and period.', {
        required: true
      }),
      makeParam('dateAxis', DATE_AXIS, DATE_AXIS_DESCRIPTION, {required: true}),
      makeParam('consecutiveCount', 'number', 'Anomalies in a row that confirm a change.', {
        default: '6'
      }),
      makeParam('mask', UINT, 'Per-observation QA; zero rejects the whole observation.')
    ],
    outputs: [
      makeOutput('segments', UINT, 'Start, end, break slice and coefficient count per segment.'),
      makeOutput('coefficients', FLOAT, 'Model coefficients per segment and band.'),
      makeOutput('segmentStatistics', FLOAT, 'RMSE and median change residual per segment.'),
      makeOutput('cellSummary', UINT, 'Per pixel segment count, status bits and fit count.')
    ],
    example: makeExample({
      symbols: ['getGPURasterCOLDMonitorParameterValues', 'GPURasterCOLDMonitor'],
      setup: ['const cellCount = 128 * 128;', 'const sliceCount = 96;', DATE_AXIS_LINE],
      values:
        'getGPURasterCOLDMonitorParameterValues({changeThreshold: 6.6, outlierThreshold: 24, minimumRMSE: [0.03]})',
      constructorSymbol: 'GPURasterCOLDMonitor',
      props: [
        'cellCount,',
        'sliceCount,',
        'bandCount: 1,',
        `values: view('series', 'float32', cellCount * sliceCount),`,
        'dateAxis,',
        'maximumSegmentCount: 4,',
        'output // segments, coefficients, segmentStatistics, cellSummary'
      ]
    }),
    preview: {
      kind: 'time-series',
      seed: 202,
      caption: 'Two segments, split where the series changes.'
    },
    related: ['bfast-monitor', 'seasonal-break-detection', 'change-detection'],
    reference: 'api/gpu-raster/raster-change-monitoring',
    maturity: 'partial'
  },
  {
    id: 'hants',
    symbol: 'GPURasterHANTS',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'HANTS',
    summary:
      'Harmonic analysis of time series: a seasonal fit that rejects cloud-contaminated samples.',
    explanation:
      'Use it to clean vegetation-index series. It fits harmonics, drops the worst observation beyond `fitErrorTolerance` on the chosen side, and refits until none is left or the passes run out. The fitted curve is finite in gaps.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster', 'time-series']},
    useCases: ['model', 'transform'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Number of pixels.', {required: true}),
      makeParam('sliceCount', 'number', 'Observations per pixel.', {required: true}),
      makeParam('values', FLOAT, 'Series per pixel, cell-major; non-finite is missing.', {
        required: true
      }),
      makeParam('parameters', FLOAT, 'Packed period, fitErrorTolerance and overdetermination.', {
        required: true
      }),
      makeParam('harmonicCount', 'number', 'Harmonics; coefficients are 2 * count + 1.', {
        required: true
      }),
      makeParam('rejectionSide', '"low" | "high" | "both"', 'Residual sign that is rejected.', {
        default: '"low"'
      }),
      makeParam('mask', UINT, 'Per-observation QA; zero rejects the sample.'),
      makeParam('dateAxis', DATE_AXIS, DATE_AXIS_DESCRIPTION)
    ],
    outputs: [
      makeOutput('smoothed', FLOAT, 'Fitted curve at every slice, gaps included.'),
      makeOutput('coefficients', FLOAT, 'Mean, then cosine and sine per harmonic.'),
      makeOutput('status', UINT, '0 converged, 1 too few samples, 2 rank failure, 8 passes spent.'),
      makeOutput('diagnostics', UINT, 'Optional rejected count and passes used per pixel.')
    ],
    example: makeExample({
      symbols: ['getGPURasterHANTSParameterValues', 'GPURasterHANTS'],
      setup: ['const cellCount = 256 * 256;', 'const sliceCount = 64;', DATE_AXIS_LINE],
      values: 'getGPURasterHANTSParameterValues({period: 1, fitErrorTolerance: 0.07})',
      constructorSymbol: 'GPURasterHANTS',
      props: [
        'cellCount,',
        'sliceCount,',
        `values: view('series', 'float32', cellCount * sliceCount),`,
        'harmonicCount: 2,',
        "rejectionSide: 'low',",
        'dateAxis,',
        'output // smoothed, coefficients, status, diagnostics'
      ]
    }),
    preview: {kind: 'time-series', seed: 203, caption: 'Cloud dips rejected, the season kept.'},
    related: ['whittaker-smoother', 'harmonic-regression', 'phenology-metrics'],
    reference: 'api/gpu-raster/raster-time-series',
    maturity: 'stable'
  },
  {
    id: 'whittaker-smoother',
    symbol: 'GPURasterWhittakerSmoother',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Whittaker smoother',
    summary: 'Penalised smoothing and gap filling of every pixel series on the acquisition dates.',
    explanation:
      'Use it to smooth noisy vegetation series without choosing a seasonal shape. It minimises fit error plus `smoothingParameter` times the roughness of the order-`d` differences, with zero weight on gaps. Robust passes down-weight clouds; `upper-envelope` favours the upper side.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster', 'time-series']},
    useCases: ['transform', 'model'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Number of pixels.', {required: true}),
      makeParam('sliceCount', 'number', 'Observations per pixel.', {required: true}),
      makeParam('values', FLOAT, 'Series per pixel, cell-major; non-finite has zero weight.', {
        required: true
      }),
      makeParam('parameters', FLOAT, 'Packed smoothingParameter and minimumWeight.', {
        required: true
      }),
      makeParam('mask', UINT, 'Per-observation QA; zero gives a sample zero weight.'),
      makeParam('qualityWeights', FLOAT, 'Per-observation quality weights in [0, 1].'),
      makeParam('reweightFunction', '"bisquare" | "upper-envelope"', 'Robust weight function.', {
        default: '"bisquare"'
      }),
      makeParam('dateAxis', DATE_AXIS, DATE_AXIS_DESCRIPTION)
    ],
    outputs: [
      makeOutput('smoothed', FLOAT, 'Smoothed curve at every slice, gaps included.'),
      makeOutput('status', UINT, '0 fit, 1 too few samples, 2 numerical failure.')
    ],
    example: makeExample({
      symbols: ['getGPURasterWhittakerSmootherParameterValues', 'GPURasterWhittakerSmoother'],
      setup: ['const cellCount = 256 * 256;', 'const sliceCount = 64;', DATE_AXIS_LINE],
      values: 'getGPURasterWhittakerSmootherParameterValues({smoothingParameter: 6})',
      constructorSymbol: 'GPURasterWhittakerSmoother',
      props: [
        'cellCount,',
        'sliceCount,',
        `values: view('series', 'float32', cellCount * sliceCount),`,
        "reweightFunction: 'upper-envelope',",
        'dateAxis,',
        'output // smoothed, status'
      ]
    }),
    preview: {kind: 'time-series', seed: 204, caption: 'A smooth curve through noisy dates.'},
    related: ['hants', 'phenology-metrics', 'temporal-resample'],
    reference: 'api/gpu-raster/raster-time-series',
    maturity: 'stable'
  },
  {
    id: 'harmonic-regression',
    symbol: 'GPURasterHarmonicRegression',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Harmonic regression',
    summary: 'Fits trend and harmonics of a chosen period to every pixel series by least squares.',
    explanation:
      'Use it to describe seasonality and trend per pixel, the model inside CCDC and BFAST without the monitoring. Time is centred fractional years on the real dates, so irregular acquisitions need no resampling. Residual `rmse` shows where the model fits.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster', 'time-series']},
    useCases: ['model', 'summarize'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Number of pixels.', {required: true}),
      makeParam('sliceCount', 'number', 'Observations per pixel.', {required: true}),
      makeParam('values', FLOAT, 'Series per pixel, cell-major; non-finite is missing.', {
        required: true
      }),
      makeParam('parameters', FLOAT, 'Packed period in fractional years.', {required: true}),
      makeParam('harmonicCount', 'number', 'Harmonics; coefficients are 2 + 2 * count.', {
        required: true
      }),
      makeParam('dateAxis', DATE_AXIS, DATE_AXIS_DESCRIPTION, {required: true}),
      makeParam('mask', UINT, 'Per-observation QA; zero rejects the sample.')
    ],
    outputs: [
      makeOutput('coefficients', FLOAT, 'Intercept, slope per year, then cosine and sine pairs.'),
      makeOutput('summary', UINT, 'Per pixel status and valid observation count.'),
      makeOutput('fitted', FLOAT, 'Optional fitted curve at every slice.'),
      makeOutput('rmse', FLOAT, 'Optional residual standard deviation per pixel.')
    ],
    example: makeExample({
      symbols: ['getGPURasterHarmonicRegressionParameterValues', 'GPURasterHarmonicRegression'],
      setup: ['const cellCount = 256 * 256;', 'const sliceCount = 64;', DATE_AXIS_LINE],
      values: 'getGPURasterHarmonicRegressionParameterValues({period: 1})',
      constructorSymbol: 'GPURasterHarmonicRegression',
      props: [
        'cellCount,',
        'sliceCount,',
        `values: view('series', 'float32', cellCount * sliceCount),`,
        'harmonicCount: 2,',
        'dateAxis,',
        'output // coefficients, summary, fitted, rmse'
      ]
    }),
    preview: {
      kind: 'time-series',
      seed: 205,
      caption: 'A seasonal wave fitted to scattered dates.'
    },
    related: ['hants', 'bfast-monitor', 'seasonal-break-detection'],
    reference: 'api/gpu-raster/raster-time-series',
    maturity: 'stable'
  },
  {
    id: 'phenology-metrics',
    symbol: 'GPURasterPhenologyMetrics',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Phenology metrics',
    summary:
      'Reads season start, peak, end, length and integral off a smoothed curve, up to two seasons.',
    explanation:
      'Use it after a smoother to extract growing seasons. A season starts where the curve passes `startFraction` of its amplitude and ends below `endFraction` (Copernicus MR-VPP uses 0.25 and 0.15). A flat pixel gets no season.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster', 'table']},
    useCases: ['summarize', 'classify'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Number of pixels.', {required: true}),
      makeParam('sliceCount', 'number', 'Slices per pixel, 2 to 256.', {required: true}),
      makeParam('maximumSeasonCount', 'number', 'Seasons found per pixel, 1 or 2.', {
        required: true
      }),
      makeParam('smoothed', FLOAT, 'Smoothed curve, for example from a Whittaker smoother.', {
        required: true
      }),
      makeParam('parameters', FLOAT, 'Packed start, end, amplitude and second-season levels.', {
        required: true
      }),
      makeParam('dateAxis', DATE_AXIS, 'Dates; times are then centred fractional years.')
    ],
    outputs: [
      makeOutput('metrics', FLOAT, 'Ten fields per season, such as start, peak and length.'),
      makeOutput('flags', UINT, 'Per season bits: crossing not found.'),
      makeOutput('seasonCount', UINT, 'Seasons found per pixel.'),
      makeOutput('status', UINT, '0 fit, 1 non-finite curve, 2 amplitude too small.')
    ],
    example: makeExample({
      symbols: ['getGPURasterPhenologyMetricsParameterValues', 'GPURasterPhenologyMetrics'],
      setup: ['const cellCount = 256 * 256;', 'const sliceCount = 52;', DATE_AXIS_LINE],
      values:
        'getGPURasterPhenologyMetricsParameterValues({startFraction: 0.25, endFraction: 0.15, minimumAmplitude: 0.1})',
      constructorSymbol: 'GPURasterPhenologyMetrics',
      props: [
        'cellCount,',
        'sliceCount,',
        'maximumSeasonCount: 2,',
        `smoothed: view('smoothed', 'float32', cellCount * sliceCount),`,
        'dateAxis,',
        'output // metrics, flags, seasonCount, status'
      ]
    }),
    preview: {kind: 'time-series', seed: 206, caption: 'Start, peak and end of a growing season.'},
    related: ['whittaker-smoother', 'hants', 'harmonic-regression'],
    reference: 'api/gpu-raster/raster-time-series',
    maturity: 'stable'
  },
  {
    id: 'rolling-mean',
    symbol: 'GPURasterRollingMean',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Rolling mean',
    summary:
      'Trailing rolling mean over a chunked raster stream, with window state kept between chunks.',
    explanation:
      'Use it for the aggregation that SPI and SPEI apply before fitting, such as a 3-month total. A missing sample in the window makes that output missing. `ring` and `seenCount` are caller-owned buffers carried from chunk to chunk.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster', 'time-series']},
    useCases: ['summarize', 'transform'],
    domains: ['remote-sensing', 'hydrology'],
    params: [
      makeParam('cellCount', 'number', 'Number of pixels.', {required: true}),
      makeParam('chunkSliceCount', 'number', 'Slices of each chunk, 1 to 256.', {required: true}),
      makeParam('windowLength', 'number', 'Trailing window in slices, 2 to 128.', {
        required: true
      }),
      makeParam('values', FLOAT, 'Chunk, cell-major; non-finite is missing.', {required: true}),
      makeParam('parameters', FLOAT, 'Packed reset flag and active slice count.', {
        required: true
      }),
      makeParam('ring', FLOAT, 'Caller-owned last windowLength samples per pixel.', {
        required: true
      }),
      makeParam('seenCount', UINT, 'Caller-owned samples seen so far per pixel.', {
        required: true
      }),
      makeParam('mask', UINT, 'Per-observation QA; zero makes the sample missing.')
    ],
    outputs: [
      makeOutput('output', FLOAT, 'Rolling mean per chunk slice; NaN until the window is full.')
    ],
    example: makeExample({
      symbols: ['getGPURasterStreamingAccumulatorParameterValues', 'GPURasterRollingMean'],
      setup: [
        'const cellCount = 256 * 256;',
        'const chunkSliceCount = 12;',
        'const windowLength = 3;'
      ],
      values:
        'getGPURasterStreamingAccumulatorParameterValues({reset: true, maximumValueMagnitude: 1000}, chunkSliceCount)',
      constructorSymbol: 'GPURasterRollingMean',
      props: [
        'cellCount,',
        'chunkSliceCount,',
        'windowLength,',
        `values: view('chunk', 'float32', cellCount * chunkSliceCount),`,
        `ring: view('ring', 'float32', cellCount * windowLength),`,
        `seenCount: view('seen', 'uint32', cellCount),`,
        `output: view('rolling', 'float32', cellCount * chunkSliceCount)`
      ]
    }),
    preview: {kind: 'time-series', seed: 207, caption: 'A trailing mean that skips no gaps.'},
    related: ['streaming-accumulator', 'temporal-resample', 'whittaker-smoother'],
    reference: 'api/gpu-raster/raster-time-series',
    maturity: 'stable'
  },
  {
    id: 'temporal-resample',
    symbol: 'GPURasterTemporalResample',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Temporal resample',
    summary: 'Interpolates irregular acquisitions onto a regular time grid, without extrapolating.',
    explanation:
      'Use it to feed regular-time operations from dated scenes. Grid points between valid observations are linearly interpolated; points outside the range get status 1 and points inside a gap wider than `maximumGap` status 2. A smoother output feeds it directly.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster', 'time-series']},
    useCases: ['transform'],
    domains: ['remote-sensing', 'data-engineering'],
    params: [
      makeParam('cellCount', 'number', 'Number of pixels.', {required: true}),
      makeParam('sliceCount', 'number', 'Observations per pixel.', {required: true}),
      makeParam('gridCount', 'number', 'Output grid points per pixel, 1 to 4096.', {
        required: true
      }),
      makeParam('values', FLOAT, 'Series per pixel, cell-major; non-finite is missing.', {
        required: true
      }),
      makeParam('parameters', FLOAT, 'Packed gridStart, gridStep and maximumGap.', {
        required: true
      }),
      makeParam('dateAxis', DATE_AXIS, DATE_AXIS_DESCRIPTION, {required: true}),
      makeParam('mask', UINT, 'Per-observation QA; zero rejects the sample.')
    ],
    outputs: [
      makeOutput('values', FLOAT, 'Resampled value per pixel and grid point.'),
      makeOutput('status', UINT, '0 interpolated, 1 outside the range, 2 gap too wide.')
    ],
    example: makeExample({
      symbols: ['getGPURasterTemporalResampleParameterValues', 'GPURasterTemporalResample'],
      setup: ['const cellCount = 256 * 256;', 'const sliceCount = 16;', DATE_AXIS_LINE],
      values:
        'getGPURasterTemporalResampleParameterValues({gridStart: -3.7, gridStep: 0.06, maximumGap: 0.9})',
      constructorSymbol: 'GPURasterTemporalResample',
      props: [
        'cellCount,',
        'sliceCount,',
        'gridCount: 120,',
        `values: view('series', 'float32', cellCount * sliceCount),`,
        'dateAxis,',
        'output // values, status'
      ]
    }),
    preview: {kind: 'time-series', seed: 208, caption: 'Irregular scenes on a regular grid.'},
    related: ['whittaker-smoother', 'rolling-mean', 'temporal-composite'],
    reference: 'api/gpu-raster/raster-time-series',
    maturity: 'stable'
  },
  {
    id: 'streaming-accumulator',
    symbol: 'GPURasterStreamingAccumulator',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Streaming accumulator',
    summary:
      'Accumulates counts, exact sums, squared deviations and extrema per pixel, chunk by chunk.',
    explanation:
      'Use it for stacks too long to hold at once, or for per-calendar-slot climatologies. State is per pixel and slot; set `reset` on the first chunk. Sums are 96-bit fixed point, decoded with `getGPURasterFixedPointValues`; a sample above `maximumValueMagnitude` counts as missing.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['summarize'],
    domains: ['remote-sensing', 'hydrology'],
    params: [
      makeParam('cellCount', 'number', 'Number of pixels.', {required: true}),
      makeParam('chunkSliceCount', 'number', 'Slices of each chunk, 1 to 256.', {required: true}),
      makeParam('values', FLOAT, 'Chunk, cell-major; non-finite is missing.', {required: true}),
      makeParam('parameters', FLOAT, 'Packed reset, active slices and fixed-point units.', {
        required: true
      }),
      makeParam(
        'state',
        'GPURasterStreamingState',
        'Caller-owned counts, sums, moments, extrema.',
        {
          required: true
        }
      ),
      makeParam('slotCount', 'number', 'Calendar slots per pixel.', {default: '1'}),
      makeParam('slotIndices', UINT, 'Slot of each chunk slice; required above one slot.'),
      makeParam('mask', UINT, 'Per-observation QA; zero rejects the sample.')
    ],
    outputs: [
      makeOutput('counts', UINT, 'Valid observations per pixel and slot.'),
      makeOutput('sums', UINT, '96-bit fixed-point sums, three words per slot.'),
      makeOutput('moments', UINT, 'Optional fixed-point squared deviations, three words per slot.'),
      makeOutput('extrema', FLOAT, 'Optional minimum and maximum per slot.')
    ],
    example: makeExample({
      symbols: ['getGPURasterStreamingAccumulatorParameterValues', 'GPURasterStreamingAccumulator'],
      setup: ['const cellCount = 256 * 256;', 'const chunkSliceCount = 4;'],
      values:
        'getGPURasterStreamingAccumulatorParameterValues({reset: true, maximumValueMagnitude: 1}, chunkSliceCount)',
      constructorSymbol: 'GPURasterStreamingAccumulator',
      props: [
        'cellCount,',
        'chunkSliceCount,',
        `values: view('chunk', 'float32', cellCount * chunkSliceCount),`,
        'state: {',
        `  counts: view('counts', 'uint32', cellCount),`,
        `  sums: view('sums', 'uint32', cellCount * 3)`,
        '}'
      ]
    }),
    preview: {kind: 'time-series', seed: 209, caption: 'Statistics grow as chunks arrive.'},
    related: ['rolling-mean', 'cell-statistics', 'temporal-resample'],
    reference: 'api/gpu-raster/raster-time-series',
    maturity: 'stable'
  },
  {
    id: 'lagged-correlation',
    symbol: 'GPURasterLaggedCorrelation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Lagged correlation',
    summary: 'Correlates every pixel series with a reference index at a set of slice lags.',
    explanation:
      'Use it for teleconnection maps, such as rainfall against an ocean index. At lag `L` the index at `t` is paired with the pixel at `t + L`, so a positive lag means the index leads. Each lag uses only pairs where both are valid.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['find-patterns', 'model'],
    domains: ['remote-sensing', 'spatial-statistics'],
    params: [
      makeParam('cellCount', 'number', 'Number of pixels.', {required: true}),
      makeParam('sliceCount', 'number', 'Observations per pixel.', {required: true}),
      makeParam('values', FLOAT, 'Series per pixel, cell-major; non-finite is missing.', {
        required: true
      }),
      makeParam('referenceIndex', FLOAT, 'Reference series, one value per slice.', {
        required: true
      }),
      makeParam('parameters', FLOAT, 'Packed minimumPairCount.', {required: true}),
      makeParam('lags', 'readonly number[]', 'Slice lags, within plus or minus sliceCount.', {
        required: true
      }),
      makeParam('mask', UINT, 'Per-observation QA; zero rejects the sample.')
    ],
    outputs: [
      makeOutput('correlations', FLOAT, 'Pearson correlation per pixel and lag.'),
      makeOutput('status', UINT, '0 computed, 1 too few pairs, 2 zero variance.'),
      makeOutput('pairCount', UINT, 'Optional valid pairs used per pixel and lag.')
    ],
    example: makeExample({
      symbols: ['getGPURasterLaggedCorrelationParameterValues', 'GPURasterLaggedCorrelation'],
      setup: ['const cellCount = 256 * 256;', 'const sliceCount = 240; // monthly'],
      values: 'getGPURasterLaggedCorrelationParameterValues({minimumPairCount: 12})',
      constructorSymbol: 'GPURasterLaggedCorrelation',
      props: [
        'cellCount,',
        'sliceCount,',
        `values: view('series', 'float32', cellCount * sliceCount),`,
        `referenceIndex: view('index', 'float32', sliceCount),`,
        'lags: [0, 1, 2, 3, 4, 5, 6],',
        'output // correlations, status, pairCount'
      ]
    }),
    preview: {
      kind: 'time-series',
      seed: 210,
      caption: 'Where a remote index leads, and by how much.'
    },
    related: ['harmonic-regression', 'streaming-accumulator', 'rolling-mean'],
    reference: 'api/gpu-raster/raster-time-series',
    maturity: 'stable'
  }
];
