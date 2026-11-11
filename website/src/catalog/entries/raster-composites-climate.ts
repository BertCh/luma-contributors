// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Catalog group "raster-composites-climate": compositing of cloudy image stacks (best-available
// pixel, medoid, geometric median), climate anomalies, spell and degree-day accumulators,
// standardized indices (SPI, SPEI) and EOF analysis by the method of snapshots.

import type {CatalogEntry, OutputDoc, ParamDoc} from '../schema';

const IMPORT_PATH = '@luma-contributors/analysis/raster-analysis';
const REFERENCE = 'api/gpu-raster/raster-time-series';
const EOF_REFERENCE = 'api/gpu-raster/raster-eof-analysis';

type ParameterBufferOptions = {
  format: 'float32' | 'uint32';
  /** Expression that creates the packed per-frame values, for example `getXParameterValues({})`. */
  values: string;
};

type ExampleOptions = {
  /** Symbols imported from the raster-analysis entry point. */
  symbols: string[];
  /** Lines between the graph preamble and the contributor. */
  setup?: string[];
  /** Per-frame parameter buffer; adds the buffer, the write and the `parameters` binding. */
  parameters?: ParameterBufferOptions;
  /** Contributor constructor symbol. */
  constructorSymbol: string;
  /** Property lines of the constructor argument. */
  props: string[];
  /** Replaces `graph.add(new X({...}))` for contributors that use `addToGraph`. */
  addToGraph?: boolean;
};

/** Builds the shared graph, compile and encode example around one contributor. */
function makeExample(options: ExampleOptions): string {
  const {parameters} = options;
  const lines = [
    `import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';`,
    ...(parameters
      ? [`import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';`]
      : []),
    `import {${[...options.symbols].sort().join(', ')}} from '${IMPORT_PATH}';`,
    '',
    `const graph = new GPUCommandGraph(device, {id: 'demo'});`,
    `const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(id: string, format: Format, length: number) => createTransientView(graph, id, format, length);`,
    ...(options.setup ?? []),
    ...(parameters
      ? [
          `const values = ${parameters.values};`,
          `const parameters = new GPUParameterBuffer(device, {id: 'p', format: '${parameters.format}', length: values.length});`
        ]
      : [])
  ];
  const props = [
    ...(parameters ? ['parameters: parameters.importToGraph(graph),'] : []),
    ...options.props
  ];
  if (options.addToGraph) {
    lines.push(
      `new ${options.constructorSymbol}({`,
      ...props.map(line => `  ${line}`),
      '}).addToGraph(graph);'
    );
  } else {
    lines.push(
      `graph.add(new ${options.constructorSymbol}({`,
      ...props.map(line => `  ${line}`),
      '}));'
    );
  }
  lines.push('const compiled = await graph.compileAsync();');
  if (parameters) {
    lines.push('parameters.write(values);');
  }
  lines.push('compiled.encode(device.commandEncoder, {parameters: undefined});');
  return lines.join('\n');
}

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

export const entries: CatalogEntry[] = [
  {
    id: 'best-available-pixel',
    symbol: 'GPURasterBestAvailablePixel',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Best-available-pixel composite',
    summary:
      'Scores every valid observation against a target date and keeps the best scene per cell.',
    explanation:
      'Builds a composite that is always one real acquisition (Griffiths et al. 2013). Each valid observation scores its closeness in day of year and in year to the target, and its distance from the nearest cloud; the highest score wins, the earliest on ties.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['summarize', 'transform'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Cells per scene.', {required: true}),
      makeParam('sliceCount', 'number', 'Scenes in `[1, 256]`.', {required: true}),
      makeParam('bandCount', 'number', 'Bands of the optional composite in `[0, 16]`.', {
        required: true
      }),
      makeParam('values', FLOAT, 'Observations, `(cell * sliceCount + slice) * bandCount + band`.'),
      makeParam('mask', UINT, 'Per-observation QA; zero rejects the observation.'),
      makeParam('cloudDistance', FLOAT, 'Pixels from each observation to the nearest cloud.'),
      makeParam('sliceAttributes', FLOAT, '`[dayOfYear, year]` per scene.', {required: true}),
      makeParam('parameters', FLOAT, 'Target date, scoring widths and weights, 9 values.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('bestSlice', UINT, 'Winning scene per cell; 0xffffffff when none is valid.'),
      makeOutput('bestScore', FLOAT, 'Winning score per cell; NaN when no scene is valid.'),
      makeOutput('composite', FLOAT, 'Bands of the winning observation per cell (optional).')
    ],
    example: makeExample({
      symbols: ['GPURasterBestAvailablePixel', 'getGPURasterBestAvailablePixelParameterValues'],
      setup: ['const cellCount = 256 * 256, sliceCount = 16;'],
      parameters: {
        format: 'float32',
        values: `getGPURasterBestAvailablePixelParameterValues({
  targetDayOfYear: 200, dayOfYearSigma: 20, dayOfYearWeight: 0.5, targetYear: 2021,
  maximumYearDistance: 4, yearWeight: 0.2, minimumCloudDistance: 10, maximumCloudDistance: 100, cloudWeight: 0.3
})`
      },
      constructorSymbol: 'GPURasterBestAvailablePixel',
      props: [
        'cellCount, sliceCount, bandCount: 0,',
        `cloudDistance: view('cloud', 'float32', cellCount * sliceCount),`,
        `sliceAttributes: view('dates', 'float32', 2 * sliceCount),`,
        'output: {',
        `  bestSlice: view('best', 'uint32', cellCount), bestScore: view('score', 'float32', cellCount)`,
        '}'
      ]
    }),
    preview: {kind: 'raster', seed: 201, caption: 'Best scene per cell; CPU cloud distance.'},
    related: ['medoid-composite', 'geometric-median-composite'],
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'medoid-composite',
    symbol: 'GPURasterMedoidComposite',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Medoid composite',
    summary: 'Per cell, the real observation whose bands are closest to all the others.',
    explanation:
      "Picks the valid observation with the smallest sum of Euclidean distances, over all bands, to the cell's other observations. The result is a real acquisition, so its slice is the provenance of every band, and haze or shadow outliers are rejected.",
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['summarize', 'transform'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Cells per scene.', {required: true}),
      makeParam('sliceCount', 'number', 'Scenes in `[1, 64]`.', {required: true}),
      makeParam('bandCount', 'number', 'Bands per observation in `[1, 16]`.', {required: true}),
      makeParam(
        'values',
        FLOAT,
        'Observations, `(cell * sliceCount + slice) * bandCount + band`.',
        {
          required: true
        }
      ),
      makeParam('mask', UINT, 'Per-observation QA; zero rejects the observation.')
    ],
    outputs: [
      makeOutput('composite', FLOAT, 'The medoid observation, `cell * bandCount + band`.'),
      makeOutput('sliceIndex', UINT, 'Scene the medoid came from; 0xffffffff when none.'),
      makeOutput('status', UINT, 'Per cell: 0 composite, 1 no valid observation.')
    ],
    example: makeExample({
      symbols: ['GPURasterMedoidComposite'],
      setup: ['const cellCount = 512 * 512;', 'const sliceCount = 8;', 'const bandCount = 6;'],
      constructorSymbol: 'GPURasterMedoidComposite',
      props: [
        'cellCount, sliceCount, bandCount,',
        `values: view('scenes', 'float32', cellCount * sliceCount * bandCount),`,
        `mask: view('qa', 'uint32', cellCount * sliceCount),`,
        'output: {',
        `  composite: view('composite', 'float32', cellCount * bandCount),`,
        `  sliceIndex: view('source', 'uint32', cellCount), status: view('status', 'uint32', cellCount)`,
        '}'
      ]
    }),
    preview: {kind: 'raster', seed: 202, caption: 'One real scene per cell; synthetic haze.'},
    related: ['geometric-median-composite', 'best-available-pixel', 'temporal-composite'],
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'geometric-median-composite',
    symbol: 'GPURasterGeometricMedianComposite',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Geometric median composite',
    summary: 'Per cell, the spectrum with the least total distance to all valid observations.',
    explanation:
      "Runs Weiszfeld's iteration from the mean of the valid observations, over all bands together, so band relationships survive and indices computed after compositing equal those before (Roberts et al. 2017). The output is synthetic; use the medoid when provenance matters.",
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['summarize', 'transform'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Cells per scene.', {required: true}),
      makeParam('sliceCount', 'number', 'Scenes in `[1, 256]`.', {required: true}),
      makeParam('bandCount', 'number', 'Bands per observation in `[1, 16]`.', {required: true}),
      makeParam('iterationCount', 'number', 'Weiszfeld iterations in `[1, 64]`; all run.', {
        required: true
      }),
      makeParam(
        'values',
        FLOAT,
        'Observations, `(cell * sliceCount + slice) * bandCount + band`.',
        {
          required: true
        }
      ),
      makeParam('mask', UINT, 'Per-observation QA; zero rejects the observation.'),
      makeParam('parameters', FLOAT, 'Smallest distance used as a weight denominator.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('composite', FLOAT, 'The geometric median, `cell * bandCount + band`.'),
      makeOutput('status', UINT, 'Per cell: 0 composite, 1 no valid observation.'),
      makeOutput('lastStep', FLOAT, 'Length of the last step per cell, a convergence check.')
    ],
    example: makeExample({
      symbols: [
        'GPURasterGeometricMedianComposite',
        'getGPURasterGeometricMedianCompositeParameterValues'
      ],
      setup: ['const cellCount = 512 * 512;', 'const sliceCount = 8;', 'const bandCount = 6;'],
      parameters: {
        format: 'float32',
        values: 'getGPURasterGeometricMedianCompositeParameterValues({distanceEpsilon: 1e-5})'
      },
      constructorSymbol: 'GPURasterGeometricMedianComposite',
      props: [
        'cellCount, sliceCount, bandCount, iterationCount: 8,',
        `values: view('scenes', 'float32', cellCount * sliceCount * bandCount),`,
        `mask: view('qa', 'uint32', cellCount * sliceCount),`,
        'output: {',
        `  composite: view('composite', 'float32', cellCount * bandCount),`,
        `  status: view('status', 'uint32', cellCount), lastStep: view('step', 'float32', cellCount)`,
        '}'
      ]
    }),
    preview: {kind: 'raster', seed: 203, caption: 'Least-distance point; synthetic haze.'},
    related: ['medoid-composite', 'best-available-pixel', 'temporal-composite'],
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'climate-anomaly',
    symbol: 'GPURasterClimateAnomaly',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Climate anomaly',
    summary: 'Expresses new slices as departures from a streamed per-month climatology.',
    explanation:
      'Reads the means and sample standard deviations a `GPURasterStreamingAccumulator` holds per cell and calendar slot, and returns `value - mean` or, with `standardized`, a z-score. The state is unchanged, so one climatology serves any number of chunks.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['detect-change', 'measure'],
    domains: ['hydrology', 'remote-sensing', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Cells per slice.', {required: true}),
      makeParam('slotCount', 'number', 'Calendar slots per cell, in `[1, 366]`.', {required: true}),
      makeParam(
        'state',
        'GPURasterStreamingState',
        'counts, sums and moments of the accumulator.',
        {
          required: true
        }
      ),
      makeParam(
        'accumulatorParameters',
        FLOAT,
        "The accumulator's parameters (fixed-point units).",
        {
          required: true
        }
      ),
      makeParam('chunkSliceCount', 'number', 'Slices of the chunk to express as anomalies.'),
      makeParam('values', FLOAT, 'The chunk, cell-major; with `output.anomaly`.'),
      makeParam('slotIndices', UINT, 'Slot of each chunk slice.'),
      makeParam('standardized', 'boolean', 'Divide by the slot standard deviation.', {
        default: 'false'
      })
    ],
    outputs: [
      makeOutput('anomaly', FLOAT, 'Anomaly or z-score per cell and chunk slice.'),
      makeOutput('climatologyMean', FLOAT, 'Mean per cell and slot.'),
      makeOutput('standardDeviation', FLOAT, 'Sample standard deviation per cell and slot.')
    ],
    example: makeExample({
      symbols: ['GPURasterClimateAnomaly'],
      setup: [
        'const cellCount = 256 * 256;',
        'const slotCount = 12;',
        `const state = {counts: view('counts', 'uint32', cellCount * slotCount), sums: view('sums', 'uint32', cellCount * slotCount * 3), moments: view('moments', 'uint32', cellCount * slotCount * 3)}; // from GPURasterStreamingAccumulator`
      ],
      constructorSymbol: 'GPURasterClimateAnomaly',
      props: [
        'cellCount, slotCount, state, standardized: true,',
        `accumulatorParameters: view('accumulator-parameters', 'float32', 5),`,
        'chunkSliceCount: 12,',
        `values: view('year', 'float32', cellCount * 12), slotIndices: view('slots', 'uint32', 12),`,
        `output: {anomaly: view('anomaly', 'float32', cellCount * 12)}`
      ]
    }),
    preview: {kind: 'raster', seed: 204, caption: 'Month vs its normal; synthetic record.'},
    related: ['standardized-index', 'standardized-index-calibration', 'climate-run-accumulator'],
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'climate-run-accumulator',
    symbol: 'GPURasterClimateRunAccumulator',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Climate run accumulator',
    summary: 'Streams a daily series into longest runs, spell counts and degree days per cell.',
    explanation:
      'Counts days that meet a condition against a threshold, the longest run, and the spells of at least `minimumSpellLength` days (the run and threshold indices of xclim). Run it once per chunk, with `reset` at the start of each period; degree-day totals are exact fixed-point sums.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['measure', 'summarize'],
    domains: ['hydrology', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Cells per slice.', {required: true}),
      makeParam('chunkSliceCount', 'number', 'Slices of each chunk, in `[1, 256]`.', {
        required: true
      }),
      makeParam('values', FLOAT, 'Chunk, cell-major; nonfinite samples are missing.', {
        required: true
      }),
      makeParam('mask', UINT, 'Per-observation QA; zero makes the sample missing.'),
      makeParam('parameters', FLOAT, 'reset, active slices, threshold, degree-day unit.', {
        required: true
      }),
      makeParam(
        'condition',
        '"above" | "at-or-above" | "below" | "at-or-below"',
        'Comparison against the threshold.',
        {default: '"at-or-above"'}
      ),
      makeParam('minimumSpellLength', 'number', 'Run length at which a spell counts.', {
        default: '1'
      }),
      makeParam(
        'state',
        '{runs?: GraphDataView<"uint32">; degreeDays?: GraphDataView<"uint32">}',
        '`runs` (six words per cell) and `degreeDays` (three).',
        {
          required: true
        }
      )
    ],
    outputs: [
      makeOutput(
        'runs',
        UINT,
        'Per cell: current run, longest run, matches, spells, spell days, missing.'
      ),
      makeOutput('degreeDays', UINT, 'Fixed-point degree-day sum per cell, three words.')
    ],
    example: makeExample({
      symbols: [
        'GPURasterClimateRunAccumulator',
        'getGPURasterClimateRunAccumulatorParameterValues'
      ],
      setup: ['const cellCount = 256 * 256;', 'const chunkSliceCount = 183;'],
      parameters: {
        format: 'float32',
        values: `getGPURasterClimateRunAccumulatorParameterValues(
  {reset: true, threshold: 1, maximumIncrement: 200}, chunkSliceCount
)`
      },
      constructorSymbol: 'GPURasterClimateRunAccumulator',
      props: [
        "cellCount, chunkSliceCount, condition: 'at-or-above', minimumSpellLength: 3,",
        `values: view('days', 'float32', cellCount * chunkSliceCount),`,
        `state: {runs: view('runs', 'uint32', cellCount * 6)}`
      ]
    }),
    preview: {kind: 'time-series', seed: 205, caption: 'Wet spells in a synthetic daily year.'},
    related: ['climate-anomaly', 'standardized-index', 'cell-statistics'],
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'standardized-index-calibration',
    symbol: 'GPURasterStandardizedIndexCalibration',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Standardized index calibration',
    summary: 'Fits a gamma or log-logistic distribution per cell and calendar month, streamed.',
    explanation:
      'Streams a calibration period (decades of months need more than one chunk) into per-slot statistics and writes the fit, by the Approximate Parameter method of xclim. A gamma gives SPI and a log-logistic (`fisk`) gives SPEI. Run it once per chunk, with `reset` on the first.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['model', 'summarize'],
    domains: ['hydrology', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Cells per slice.', {required: true}),
      makeParam('chunkSliceCount', 'number', 'Slices of each chunk, in `[1, 256]`.', {
        required: true
      }),
      makeParam('slotCount', 'number', 'Calendar slots per cell, usually 12 months.', {
        default: '1'
      }),
      makeParam(
        'distribution',
        '"gamma" | "fisk"',
        'gamma for SPI, fisk (log-logistic) for SPEI.',
        {
          required: true
        }
      ),
      makeParam('values', FLOAT, 'Chunk, cell-major; nonfinite samples are missing.', {
        required: true
      }),
      makeParam('mask', UINT, 'Per-observation QA; zero makes the sample missing.'),
      makeParam(
        'slotIndices',
        UINT,
        'Calendar slot of each chunk slice; required when slotCount > 1.'
      ),
      makeParam(
        'state',
        '{counts: GraphDataView<"uint32">; moments: GraphDataView<"uint32">}',
        '`counts` and `moments` accumulated across chunks.',
        {
          required: true
        }
      )
    ],
    moreRequired: ['parameters', 'fit'],
    outputs: [
      makeOutput('fit', FLOAT, '`[shape, scale, zeroProbability]` per slot; NaN shape if unusable.')
    ],
    example: makeExample({
      symbols: [
        'GPURasterStandardizedIndexCalibration',
        'getGPURasterStandardizedIndexCalibrationParameterValues'
      ],
      setup: [
        'const cellCount = 256 * 256;',
        'const chunkSliceCount = 180;',
        'const slots = cellCount * 12;'
      ],
      parameters: {
        format: 'float32',
        values: `getGPURasterStandardizedIndexCalibrationParameterValues(
  {reset: true, maximumValueMagnitude: 2000}, chunkSliceCount
)`
      },
      constructorSymbol: 'GPURasterStandardizedIndexCalibration',
      props: [
        `cellCount, chunkSliceCount, slotCount: 12, distribution: 'gamma',`,
        `values: view('months', 'float32', cellCount * chunkSliceCount),`,
        `slotIndices: view('slots', 'uint32', chunkSliceCount),`,
        `state: {counts: view('counts', 'uint32', slots * 3), moments: view('moments', 'uint32', slots * 9)},`,
        `fit: view('fit', 'float32', slots * 3)`
      ]
    }),
    preview: {
      kind: 'histogram',
      seed: 206,
      caption: 'Fits per cell and month; synthetic record.'
    },
    related: ['standardized-index', 'climate-anomaly', 'climate-run-accumulator'],
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'standardized-index',
    symbol: 'GPURasterStandardizedIndex',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Standardized index',
    summary: 'SPI or SPEI: how many standard deviations an observation is from normal, per cell.',
    explanation:
      'Evaluates the probability of each value under the distribution fitted for its cell and calendar month, then maps it to a standard normal quantile. The smaller tail is used so neither extreme loses precision; results are clipped to +-8.21 as in xclim.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['measure', 'detect-change'],
    domains: ['hydrology', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Cells per slice.', {required: true}),
      makeParam('chunkSliceCount', 'number', 'Slices of the chunk to evaluate.', {required: true}),
      makeParam('slotCount', 'number', 'Calendar slots per cell.', {default: '1'}),
      makeParam('distribution', '"gamma" | "fisk"', 'Must match the calibration.', {
        required: true
      }),
      makeParam('values', FLOAT, 'Chunk to evaluate, cell-major.', {required: true}),
      makeParam('mask', UINT, 'Per-observation QA; zero makes the sample missing.'),
      makeParam(
        'slotIndices',
        UINT,
        'Calendar slot of each chunk slice; required when slotCount > 1.'
      ),
      makeParam('parameters', FLOAT, 'Active slice count and location.', {required: true})
    ],
    moreRequired: ['fit'],
    outputs: [
      makeOutput(
        'index',
        FLOAT,
        'Standardized index in [-8.21, 8.21]; NaN where status is nonzero.'
      ),
      makeOutput('status', UINT, '0 valid, 1 missing input, 2 calibration not usable.')
    ],
    example: makeExample({
      symbols: ['GPURasterStandardizedIndex', 'getGPURasterStandardizedIndexParameterValues'],
      setup: ['const cellCount = 256 * 256;', 'const chunkSliceCount = 12;'],
      parameters: {
        format: 'float32',
        values: 'getGPURasterStandardizedIndexParameterValues({}, chunkSliceCount)'
      },
      constructorSymbol: 'GPURasterStandardizedIndex',
      props: [
        `cellCount, chunkSliceCount, slotCount: 12, distribution: 'gamma',`,
        `values: view('year', 'float32', cellCount * chunkSliceCount),`,
        `slotIndices: view('slots', 'uint32', chunkSliceCount),`,
        `fit: view('fit', 'float32', cellCount * 12 * 3), // from the calibration`,
        'output: {',
        `  index: view('spi', 'float32', cellCount * chunkSliceCount),`,
        `  status: view('status', 'uint32', cellCount * chunkSliceCount)`,
        '}'
      ]
    }),
    preview: {kind: 'raster', seed: 207, caption: 'Dry and wet on one scale; synthetic record.'},
    related: ['standardized-index-calibration', 'climate-anomaly', 'climate-run-accumulator'],
    reference: REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'snapshot-covariance',
    symbol: 'GPURasterSnapshotCovariance',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Snapshot covariance',
    summary: "The Gram matrix of a field's snapshots, the GPU half of EOF analysis.",
    explanation:
      "Removes each cell's temporal mean and sums the products of every pair of slices over cells, as exact fixed-point words, so the result does not depend on thread order. Read the words back, decode them, and hand the matrix to `getGPURasterSnapshotEOFDecomposition`.",
    dataTypes: {input: ['raster', 'time-series'], output: ['table']},
    useCases: ['summarize', 'find-patterns'],
    domains: ['hydrology', 'remote-sensing', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Cells per snapshot.', {required: true}),
      makeParam('sliceCount', 'number', 'Snapshots in `[2, 256]`.', {required: true}),
      makeParam(
        'values',
        FLOAT,
        'Field, `cell * sliceCount + slice`; a missing sample drops the cell.',
        {
          required: true
        }
      ),
      makeParam('mask', UINT, 'Per-observation QA; zero excludes the cell.'),
      makeParam('weights', FLOAT, 'Per-cell area weights in `(0, 1]`, such as cos(latitude).'),
      makeParam('parameters', FLOAT, 'Fixed-point unit exponent and value bound.', {required: true})
    ],
    outputs: [
      makeOutput(
        'covarianceWords',
        UINT,
        'Upper triangle of the Gram matrix, three words per entry.'
      ),
      makeOutput('keptCellCount', UINT, 'One word: the cells that contributed.')
    ],
    example: makeExample({
      symbols: ['GPURasterSnapshotCovariance', 'getGPURasterSnapshotCovarianceParameterValues'],
      setup: ['const cellCount = 256 * 256;', 'const sliceCount = 16;'],
      parameters: {
        format: 'float32',
        values:
          'getGPURasterSnapshotCovarianceParameterValues({maximumValueMagnitude: 1}, cellCount)'
      },
      constructorSymbol: 'GPURasterSnapshotCovariance',
      props: [
        'cellCount, sliceCount,',
        `values: view('field', 'float32', cellCount * sliceCount),`,
        'output: {',
        `  covarianceWords: view('words', 'uint32', (sliceCount * (sliceCount + 1) / 2) * 3),`,
        `  keptCellCount: view('kept', 'uint32', 1)`,
        '}'
      ]
    }),
    preview: {kind: 'regression', seed: 208, caption: 'Dates that vary together; CPU decoding.'},
    related: ['eof-projection', 'climate-anomaly', 'cell-statistics'],
    reference: EOF_REFERENCE,
    maturity: 'stable'
  },
  {
    id: 'eof-projection',
    symbol: 'GPURasterEOFProjection',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'EOF projection',
    summary: 'Projects a field onto temporal modes to give one spatial EOF map per mode.',
    explanation:
      'The last step of the method of snapshots: after the CPU eigen step on the covariance, upload `temporalModes` and `singularValues` and write one spatial pattern per mode. `componentSpace` chooses plain or area-weighted orthonormality.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['find-patterns', 'summarize'],
    domains: ['hydrology', 'remote-sensing', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Cells per snapshot.', {required: true}),
      makeParam('sliceCount', 'number', 'Snapshots in `[2, 256]`.', {required: true}),
      makeParam('modeCount', 'number', 'Modes in `[1, 32]`, at most `sliceCount`.', {
        required: true
      }),
      makeParam('values', FLOAT, 'The field given to the covariance, cell-major.', {
        required: true
      }),
      makeParam('mask', UINT, 'Per-observation QA; the same view given to the covariance.'),
      makeParam('weights', FLOAT, 'Per-cell area weights; the same view given to the covariance.'),
      makeParam('temporalModes', FLOAT, 'Mode-major `mode * sliceCount + slice`.', {
        required: true
      }),
      makeParam(
        'componentSpace',
        '"weighted" | "value"',
        'Orthonormal under the plain or weighted dot.',
        {
          default: '"weighted"'
        }
      )
    ],
    moreRequired: ['parameters', 'singularValues'],
    outputs: [
      makeOutput('eofs', FLOAT, 'Spatial EOFs, `cell * modeCount + mode`; NaN if excluded.'),
      makeOutput('status', UINT, 'Per cell: 0 projected, 1 excluded.')
    ],
    example: makeExample({
      symbols: ['GPURasterEOFProjection'],
      setup: [
        'const cellCount = 256 * 256;',
        'const sliceCount = 16;',
        'const modeCount = 4;',
        '// Modes from getGPURasterSnapshotEOFDecomposition, after reading the covariance back.'
      ],
      constructorSymbol: 'GPURasterEOFProjection',
      props: [
        'cellCount, sliceCount, modeCount,',
        `values: view('field', 'float32', cellCount * sliceCount),`,
        `parameters: view('covariance-parameters', 'float32', 2),`,
        `temporalModes: view('modes', 'float32', modeCount * sliceCount),`,
        `singularValues: view('singular', 'float32', modeCount),`,
        'output: {',
        `  eofs: view('eofs', 'float32', cellCount * modeCount), status: view('status', 'uint32', cellCount)`,
        '}'
      ]
    }),
    preview: {kind: 'raster', seed: 209, caption: 'Spatial patterns; CPU Gram matrix.'},
    related: ['snapshot-covariance', 'climate-anomaly', 'seasonal-break-detection'],
    reference: EOF_REFERENCE,
    maturity: 'stable'
  }
];
