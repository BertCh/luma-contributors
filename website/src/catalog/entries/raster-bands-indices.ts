// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Catalog group "raster-bands-indices": raster bands, spectral indices, algebra, reclassify, zonal
// statistics, patches, change detection and reprojection.

import type {CatalogEntry, OutputDoc, ParamDoc} from '../schema';

const IMPORT_PATH = '@luma-contributors/analysis/raster-analysis';
const SPATIAL_IMPORT_PATH = '@luma-contributors/analysis/gpu-spatial-analysis';

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

const GRID_LINE = 'const grid = makeGPURasterMetadata({width: 512, height: 512, cellSize: 30});';
const PIXEL_COUNT_LINE = 'const n = grid.width * grid.height;';
const BAND_LINE =
  "const band = (id: string) => makeGPURasterBand(view(id, 'float32', n), grid, {id, validity: view(`${id}-valid`, 'uint32', n)});";

/** Builds the shared graph, compile and encode example around one contributor. */
function makeExample(options: ExampleOptions): string {
  const {parameters} = options;
  const lines = [
    `import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';`,
    ...(parameters ? [`import {GPUParameterBuffer} from '${SPATIAL_IMPORT_PATH}';`] : []),
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
  // Grids and bands ---------------------------------------------------------------------------
  {
    id: 'make-raster-grid',
    symbol: 'makeGPURasterMetadata',
    kind: 'function',
    importPath: IMPORT_PATH,
    title: 'Make raster grid',
    summary: 'Describes an axis-aligned raster grid: size, cell size, origin and row direction.',
    explanation:
      'Every raster operation takes its shape from this one `grid` object, so cell size and extent are never typed twice. Use `cellSize` in meters, EPSG:3857 meters or degrees to match your `cellSizeMode`.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['transform'],
    domains: ['remote-sensing', 'data-engineering'],
    params: [
      makeParam('width', 'number', 'Grid width in pixels.', {required: true}),
      makeParam('height', 'number', 'Grid height in pixels.', {required: true}),
      makeParam('cellSize', 'number | readonly [number, number]', 'Cell size in world units.', {
        default: '1'
      }),
      makeParam(
        'origin',
        'readonly [number, number]',
        'World [x, y] of the top-left pixel corner.',
        {default: '[0, 0]'}
      ),
      makeParam('rowDirection', '"south" | "north"', 'Direction rows advance in world y.', {
        default: '"south"'
      })
    ],
    outputs: [
      makeOutput(
        'affine',
        '[number, number, number, number, number, number]',
        'Pixel-to-world transform.'
      ),
      makeOutput(
        'GPURasterMetadata',
        'GPURasterMetadata',
        'Width, height, affine and pixel interpretation.'
      )
    ],
    example: [
      `import {makeGPURasterMetadata} from '${IMPORT_PATH}';`,
      '',
      'const grid = makeGPURasterMetadata({',
      '  width: 1024,',
      '  height: 1024,',
      '  cellSize: 30,',
      '  origin: [500000, 4650000]',
      '});',
      '// grid.affine === [30, 0, 500000, 0, -30, 4650000]',
      '// Pass `grid` to any raster operation.'
    ].join('\n'),
    preview: {kind: 'cells', seed: 11, caption: 'A grid is a size plus an affine.'},
    related: ['make-raster-band', 'raster-zonal-statistics', 'raster-reprojection'],
    reference: 'api/gpu-terrain/operators',
    maturity: 'stable'
  },
  {
    id: 'make-raster-band',
    symbol: 'makeGPURasterBand',
    kind: 'function',
    importPath: IMPORT_PATH,
    title: 'Make raster band',
    summary: 'Wraps one value per pixel as a band, with optional validity, nodata and calibration.',
    explanation:
      'Feeds the output of one operation into the next without a hand-written band literal. Calibration (`scale`, `offset`) and raw `noDataValue` travel with the band, so downstream operations read reflectance, not digital numbers.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['transform'],
    domains: ['remote-sensing', 'data-engineering'],
    params: [
      makeParam('values', 'GraphDataView<Format>', 'One value per grid pixel.', {required: true}),
      makeParam('grid', 'GPURasterMetadata', 'The grid the band lives on.', {required: true}),
      makeParam(
        'options.validity',
        'GraphDataView<"uint32">',
        'Per-pixel flags; zero rejects a sample.'
      ),
      makeParam('options.noDataValue', 'number', 'Raw nodata value, compared before calibration.'),
      makeParam('options.scale', 'number', 'Calibration multiplier.', {default: '1'}),
      makeParam('options.offset', 'number', 'Calibration offset.', {default: '0'})
    ],
    outputs: [
      makeOutput(
        'band',
        'GPURasterBufferBand<Format>',
        'Buffer-backed band ready for any raster operation.'
      )
    ],
    example: [
      `import {makeGPURasterBand, makeGPURasterMetadata} from '${IMPORT_PATH}';`,
      '',
      'const grid = makeGPURasterMetadata({width: 512, height: 512, cellSize: 30});',
      '// `reflectance` and `validity` are graph views with one row per pixel.',
      'const nearInfrared = makeGPURasterBand(reflectance, grid, {',
      `  id: 'nir',`,
      '  validity,',
      '  noDataValue: 0,',
      '  scale: 0.0000275, // Landsat Collection 2 surface reflectance',
      '  offset: -0.2',
      '});'
    ].join('\n'),
    preview: {
      kind: 'pipeline',
      seed: 12,
      caption: 'Values, validity and calibration travel together.'
    },
    related: ['make-raster-grid', 'normalized-difference-water-index', 'enhanced-vegetation-index'],
    reference: 'api/gpu-terrain/operators',
    maturity: 'stable'
  },

  // Spectral indices --------------------------------------------------------------------------
  {
    id: 'normalized-difference-water-index',
    symbol: 'GPURasterNDWI',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Normalized difference water index',
    summary: 'NDWI water mask from calibrated bands, by McFeeters (green, NIR) or Gao (NIR, SWIR).',
    explanation:
      'Maps open water or canopy moisture. The `method` is required because the two published recipes use different bands.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['classify', 'detect-change'],
    domains: ['remote-sensing', 'hydrology'],
    params: [
      makeParam('method', '"gao" | "mcfeeters"', 'Band recipe; required.', {required: true}),
      makeParam('green', 'GPURasterBufferBand', 'Green band, for mcfeeters.'),
      makeParam('nearInfrared', 'GPURasterBufferBand', 'Near-infrared band.', {required: true}),
      makeParam('shortwaveInfrared', 'GPURasterBufferBand', 'SWIR band, for gao.'),
      makeParam('output', 'GraphDataView<"float32">', 'Index per pixel; NaN where invalid.', {
        required: true
      }),
      makeParam('outputValidity', 'GraphDataView<"uint32">', 'Flag per pixel; 1 where valid.', {
        required: true
      }),
      makeParam('width', 'number', 'Raster width in pixels.', {required: true}),
      makeParam('height', 'number', 'Raster height in pixels.', {required: true})
    ],
    outputs: [
      makeOutput('output', 'GraphDataView<"float32">', 'Dimensionless ratio in [-1, 1].'),
      makeOutput(
        'outputValidity',
        'GraphDataView<"uint32">',
        '1 where every source band was valid.'
      )
    ],
    example: makeExample({
      symbols: ['GPURasterNDWI', 'makeGPURasterBand', 'makeGPURasterMetadata'],
      setup: [GRID_LINE, PIXEL_COUNT_LINE, BAND_LINE],
      constructorSymbol: 'GPURasterNDWI',
      addToGraph: true,
      props: [
        `method: 'mcfeeters',`,
        'width: grid.width,',
        'height: grid.height,',
        `green: band('green'),`,
        `nearInfrared: band('nir'),`,
        `output: view('ndwi', 'float32', n),`,
        `outputValidity: view('ndwi-valid', 'uint32', n)`
      ]
    }),
    preview: {kind: 'raster', seed: 21, caption: 'Water reads high, land low.'},
    related: [
      'normalized-burn-ratio',
      'enhanced-vegetation-index',
      'soil-adjusted-vegetation-index'
    ],
    lensScenario: 'gpu-raster.raster-ndwi',
    reference: 'api/gpu-raster/spectral-indices',
    maturity: 'stable'
  },
  {
    id: 'normalized-burn-ratio',
    symbol: 'GPURasterNBR',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Normalized burn ratio',
    summary: 'NBR = (NIR - SWIR2) / (NIR + SWIR2), the standard burn-severity index.',
    explanation:
      'Best before and after a fire; subtract the two for burn severity. SWIR2 is the band near 2.2 micrometers.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['detect-change', 'classify'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('nearInfrared', 'GPURasterBufferBand', 'Near-infrared band.', {required: true}),
      makeParam('shortwaveInfrared2', 'GPURasterBufferBand', 'SWIR2 band near 2.2 um.', {
        required: true
      }),
      makeParam('width', 'number', 'Grid width in pixels.', {required: true}),
      makeParam('height', 'number', 'Grid height in pixels.', {required: true}),
      makeParam('output', 'GraphDataView<"float32">', 'Index per pixel; NaN where invalid.', {
        required: true
      }),
      makeParam('outputValidity', 'GraphDataView<"uint32">', 'Flag per pixel; 1 where valid.', {
        required: true
      }),
      makeParam('epsilon', 'number', 'Denominator magnitude must exceed this.', {default: '0'}),
      makeParam('clamp', 'readonly [number, number]', 'Clamp the index to [min, max].')
    ],
    outputs: [
      makeOutput('output', 'GraphDataView<"float32">', 'Dimensionless ratio in [-1, 1].'),
      makeOutput(
        'outputValidity',
        'GraphDataView<"uint32">',
        '1 where every source band was valid.'
      )
    ],
    example: makeExample({
      symbols: ['GPURasterNBR', 'makeGPURasterBand', 'makeGPURasterMetadata'],
      setup: [GRID_LINE, PIXEL_COUNT_LINE, BAND_LINE],
      constructorSymbol: 'GPURasterNBR',
      addToGraph: true,
      props: [
        'width: grid.width,',
        'height: grid.height,',
        `nearInfrared: band('nir'),`,
        `shortwaveInfrared2: band('swir2'),`,
        `output: view('nbr', 'float32', n),`,
        `outputValidity: view('nbr-valid', 'uint32', n)`
      ]
    }),
    preview: {kind: 'raster', seed: 22, caption: 'Burn scars drop below unburned land.'},
    related: ['normalized-difference-water-index', 'change-detection', 'enhanced-vegetation-index'],
    lensScenario: 'gpu-raster.raster-nbr',
    reference: 'api/gpu-raster/spectral-indices',
    maturity: 'stable'
  },
  {
    id: 'enhanced-vegetation-index',
    symbol: 'GPURasterEVI',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Enhanced vegetation index',
    summary:
      'EVI = 2.5 (NIR - red) / (NIR + 6 red - 7.5 blue + 1), vegetation vigor with less saturation.',
    explanation:
      'Best over dense canopy where NDVI saturates. The blue band corrects for atmosphere; supply calibrated reflectance fractions.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['measure', 'detect-change'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('nearInfrared', 'GPURasterBufferBand', 'Near-infrared band.', {required: true}),
      makeParam('red', 'GPURasterBufferBand', 'Red band.', {required: true}),
      makeParam('blue', 'GPURasterBufferBand', 'Blue band.', {required: true}),
      makeParam('width', 'number', 'Grid width in pixels.', {required: true}),
      makeParam('height', 'number', 'Grid height in pixels.', {required: true}),
      makeParam('output', 'GraphDataView<"float32">', 'Index per pixel; NaN where invalid.', {
        required: true
      }),
      makeParam('outputValidity', 'GraphDataView<"uint32">', 'Flag per pixel; 1 where valid.', {
        required: true
      }),
      makeParam('epsilon', 'number', 'Denominator magnitude must exceed this.', {default: '0'})
    ],
    outputs: [
      makeOutput('output', 'GraphDataView<"float32">', 'Index per pixel, typically in [-1, 1].'),
      makeOutput(
        'outputValidity',
        'GraphDataView<"uint32">',
        '1 where every source band was valid.'
      )
    ],
    example: makeExample({
      symbols: ['GPURasterEVI', 'makeGPURasterBand', 'makeGPURasterMetadata'],
      setup: [GRID_LINE, PIXEL_COUNT_LINE, BAND_LINE],
      constructorSymbol: 'GPURasterEVI',
      addToGraph: true,
      props: [
        'width: grid.width,',
        'height: grid.height,',
        `nearInfrared: band('nir'),`,
        `red: band('red'),`,
        `blue: band('blue'),`,
        `output: view('evi', 'float32', n),`,
        `outputValidity: view('evi-valid', 'uint32', n)`
      ]
    }),
    preview: {kind: 'raster', seed: 23, caption: 'Canopy vigor without saturation.'},
    related: [
      'soil-adjusted-vegetation-index',
      'normalized-difference-water-index',
      'temporal-composite'
    ],
    lensScenario: 'gpu-raster.raster-evi',
    reference: 'api/gpu-raster/spectral-indices',
    maturity: 'stable'
  },
  {
    id: 'soil-adjusted-vegetation-index',
    symbol: 'GPURasterSAVI',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Soil-adjusted vegetation index',
    summary:
      'SAVI = (1 + L)(NIR - red) / (NIR + red + L), vegetation with a soil-brightness correction.',
    explanation:
      'Best over sparse vegetation where bare soil biases NDVI. `soilAdjustment` is L in [0, 1]; zero gives NDVI.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['measure'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('nearInfrared', 'GPURasterBufferBand', 'Near-infrared band.', {required: true}),
      makeParam('red', 'GPURasterBufferBand', 'Red band.', {required: true}),
      makeParam('soilAdjustment', 'number', 'Soil brightness factor L in [0, 1].', {
        default: '0.5'
      }),
      makeParam('width', 'number', 'Grid width in pixels.', {required: true}),
      makeParam('height', 'number', 'Grid height in pixels.', {required: true}),
      makeParam('output', 'GraphDataView<"float32">', 'Index per pixel; NaN where invalid.', {
        required: true
      }),
      makeParam('outputValidity', 'GraphDataView<"uint32">', 'Flag per pixel; 1 where valid.', {
        required: true
      }),
      makeParam('epsilon', 'number', 'Denominator magnitude must exceed this.', {default: '0'})
    ],
    outputs: [
      makeOutput('output', 'GraphDataView<"float32">', 'Index per pixel.'),
      makeOutput(
        'outputValidity',
        'GraphDataView<"uint32">',
        '1 where every source band was valid.'
      )
    ],
    example: makeExample({
      symbols: ['GPURasterSAVI', 'makeGPURasterBand', 'makeGPURasterMetadata'],
      setup: [GRID_LINE, PIXEL_COUNT_LINE, BAND_LINE],
      constructorSymbol: 'GPURasterSAVI',
      addToGraph: true,
      props: [
        'width: grid.width,',
        'height: grid.height,',
        `nearInfrared: band('nir'),`,
        `red: band('red'),`,
        'soilAdjustment: 0.5,',
        `output: view('savi', 'float32', n),`,
        `outputValidity: view('savi-valid', 'uint32', n)`
      ]
    }),
    preview: {kind: 'raster', seed: 24, caption: 'Sparse cover, soil effect removed.'},
    related: [
      'enhanced-vegetation-index',
      'normalized-difference-water-index',
      'spectral-unmixing'
    ],
    lensScenario: 'gpu-raster.raster-savi',
    reference: 'api/gpu-raster/spectral-indices',
    maturity: 'stable'
  },

  // Raster algebra ----------------------------------------------------------------------------
  {
    id: 'raster-arithmetic',
    symbol: 'GPURasterArithmetic',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Raster arithmetic',
    summary:
      'Per-cell calculator: add, divide, power, normalized difference and more, with optional clamp.',
    explanation:
      'Best for band math and unit conversion; chain operations for longer formulas. The operation is a per-frame parameter, so changing it never recompiles.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['transform', 'measure'],
    domains: ['remote-sensing', 'data-engineering'],
    params: [
      makeParam('a', FLOAT, 'First operand raster.', {required: true}),
      makeParam('b', FLOAT, 'Second operand; omit for unary operations or a constant.'),
      makeParam('cellCount', 'number', 'Cells per raster.', {required: true}),
      makeParam('noDataValue', 'number', 'Input value treated as nodata.'),
      makeParam('parameters', FLOAT, 'Packed operation, scales, offsets, clamp.', {required: true}),
      makeParam('output', 'GPURasterArithmeticOutput', 'Result raster.', {required: true})
    ],
    outputs: [makeOutput('values', FLOAT, 'Result per cell; NaN for nodata or domain errors.')],
    example: makeExample({
      symbols: ['getGPURasterArithmeticParameterValues', 'GPURasterArithmetic'],
      parameters: {
        format: 'float32',
        values: `getGPURasterArithmeticParameterValues({operation: 'normalizedDifference'})`
      },
      setup: ['const cellCount = 512 * 512;'],
      constructorSymbol: 'GPURasterArithmetic',
      props: [
        'cellCount,',
        `a: view('nir', 'float32', cellCount),`,
        `b: view('red', 'float32', cellCount),`,
        `output: {values: view('ndvi', 'float32', cellCount)}`
      ]
    }),
    preview: {kind: 'raster', seed: 31, caption: 'Two bands in, one derived band out.'},
    related: ['raster-conditional', 'cell-statistics', 'raster-reclassify'],
    lensScenario: 'gpu-raster.raster-arithmetic',
    reference: 'api/gpu-raster/operations-analysis',
    maturity: 'stable'
  },
  {
    id: 'cell-statistics',
    symbol: 'GPURasterCellStatistics',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Cell statistics',
    summary:
      'Per-cell minimum, maximum, range, sum, mean, standard deviation or majority across a raster stack.',
    explanation:
      'Collapses layers (dates, scenarios, ensemble members) into one raster. Request only the outputs you need.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['summarize'],
    domains: ['remote-sensing', 'spatial-statistics'],
    params: [
      makeParam('stack', FLOAT, 'Layers stacked layer-major, cellCount values each.', {
        required: true
      }),
      makeParam('layerCount', 'number', 'Number of layers.', {required: true}),
      makeParam('cellCount', 'number', 'Cells per layer.', {required: true}),
      makeParam('noDataValue', 'number', 'Input value treated as nodata.'),
      makeParam('parameters', FLOAT, 'Packed nodata policy and minimum valid count.', {
        required: true
      }),
      makeParam('output', 'GPURasterCellStatisticsOutput', 'Requested statistic rasters.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('minimum', FLOAT, 'Per-cell minimum across layers.'),
      makeOutput('maximum', FLOAT, 'Per-cell maximum across layers.'),
      makeOutput('range', FLOAT, 'Per-cell maximum minus minimum.'),
      makeOutput(
        'sum / mean / standardDeviation',
        FLOAT,
        'Per-cell moments across layers (optional).'
      ),
      makeOutput('majority', FLOAT, 'Most frequent value across layers (optional).')
    ],
    example: makeExample({
      symbols: ['getGPURasterCellStatisticsParameterValues', 'GPURasterCellStatistics'],
      parameters: {
        format: 'float32',
        values: `getGPURasterCellStatisticsParameterValues({noDataPolicy: 'ignore'})`
      },
      setup: ['const cellCount = 512 * 512;', 'const layerCount = 12;'],
      constructorSymbol: 'GPURasterCellStatistics',
      props: [
        'cellCount,',
        'layerCount,',
        `stack: view('monthly', 'float32', cellCount * layerCount),`,
        `output: {`,
        `  mean: view('mean', 'float32', cellCount),`,
        `  standardDeviation: view('sd', 'float32', cellCount)`,
        '}'
      ]
    }),
    preview: {kind: 'raster', seed: 32, caption: 'Twelve layers collapse to one mean.'},
    related: ['raster-arithmetic', 'weighted-overlay', 'temporal-composite'],
    lensScenario: 'gpu-raster.raster-cell-statistics',
    reference: 'api/gpu-raster/operations-analysis',
    maturity: 'stable'
  },
  {
    id: 'raster-conditional',
    symbol: 'GPURasterConditional',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Raster conditional',
    summary: 'Per-cell where(condition, a, b), like ArcGIS Con or numpy where.',
    explanation:
      'Masks, thresholds or patches one raster with another. The condition is a comparison against a threshold, or a ready-made `mask`.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['classify', 'transform'],
    domains: ['remote-sensing', 'data-engineering'],
    params: [
      makeParam('cellCount', 'number', 'Cells per raster.', {required: true}),
      makeParam('conditionValues', FLOAT, 'Values compared with the threshold.'),
      makeParam('mask', UINT, 'Precomputed condition flags; nonzero selects a.'),
      makeParam('a', FLOAT, 'Raster used where the condition holds.'),
      makeParam('b', FLOAT, 'Raster used elsewhere.'),
      makeParam('parameters', FLOAT, 'Packed comparison, thresholds and constants.', {
        required: true
      }),
      makeParam('output', 'GPURasterConditionalOutput', 'Result raster and optional mask.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('values', FLOAT, 'Selected value per cell.'),
      makeOutput('outputMask', UINT, 'Validity flag per cell (optional).')
    ],
    example: makeExample({
      symbols: ['getGPURasterConditionalParameterValues', 'GPURasterConditional'],
      parameters: {
        format: 'float32',
        values: `getGPURasterConditionalParameterValues({comparison: '>', threshold: 0.4, constantA: 1, constantB: 0})`
      },
      setup: ['const cellCount = 512 * 512;'],
      constructorSymbol: 'GPURasterConditional',
      props: [
        'cellCount,',
        `conditionValues: view('ndvi', 'float32', cellCount),`,
        `output: {values: view('vegetated', 'float32', cellCount)}`
      ]
    }),
    preview: {
      kind: 'classification',
      seed: 33,
      caption: 'A threshold turns a surface into a mask.'
    },
    related: ['raster-arithmetic', 'raster-reclassify', 'weighted-overlay'],
    lensScenario: 'gpu-raster.raster-conditional',
    reference: 'api/gpu-raster/operations-analysis',
    maturity: 'stable'
  },
  {
    id: 'raster-reclassify',
    symbol: 'GPURasterReclassify',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Reclassify',
    summary: 'Maps raster values to classes through a per-frame break table.',
    explanation:
      'Bins continuous values into classes or to remap codes. The break table is a buffer, so editing breaks is a write, not a recompile.',
    dataTypes: {input: ['raster', 'table'], output: ['raster']},
    useCases: ['classify'],
    domains: ['remote-sensing', 'cartography'],
    params: [
      makeParam('values', FLOAT, 'Raster or column to classify.', {required: true}),
      makeParam('breaks', FLOAT, 'Ascending break values.', {required: true}),
      makeParam('classValues', FLOAT, 'Value written for each class.'),
      makeParam('validity', UINT, 'Per-cell validity flags.'),
      makeParam('noDataValue', 'number', 'Input value treated as nodata.'),
      makeParam('rowCount', 'number', 'Rows to classify; defaults to values.length.'),
      makeParam('parameters', FLOAT, 'Packed break count and interval closure.', {required: true}),
      makeParam('output', 'GPURasterReclassifyOutput', 'Requested class outputs.', {required: true})
    ],
    outputs: [
      makeOutput('classes', UINT, 'Class index per cell (optional).'),
      makeOutput('reclassified', FLOAT, 'Mapped class value per cell (optional).'),
      makeOutput('classCounts', UINT, 'Cells per class (optional).')
    ],
    example: makeExample({
      symbols: ['getGPURasterReclassifyParameterValues', 'GPURasterReclassify'],
      parameters: {
        format: 'float32',
        values: `getGPURasterReclassifyParameterValues({breakCount: 3, closed: 'right'})`
      },
      setup: [
        'const n = 512 * 512;',
        `// \`breaks\`: Float32 [0.1, 0.3, 0.6] uploaded as a graph view.`
      ],
      constructorSymbol: 'GPURasterReclassify',
      props: [
        `values: view('ndvi', 'float32', n),`,
        `breaks: view('breaks', 'float32', 3),`,
        'output: {',
        `  classes: view('classes', 'uint32', n),`,
        `  classCounts: view('counts', 'uint32', 4)`,
        '}'
      ]
    }),
    preview: {kind: 'classification', seed: 34, caption: 'Continuous values become four classes.'},
    related: ['raster-conditional', 'weighted-overlay', 'patch-metrics'],
    lensScenario: 'gpu-raster.raster-reclassify',
    reference: 'api/gpu-raster/operations-analysis',
    maturity: 'stable'
  },
  {
    id: 'weighted-overlay',
    symbol: 'GPUWeightedOverlay',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Weighted overlay',
    summary:
      'Suitability score from up to 16 rasters, each rescaled or remapped, then weighted and summed.',
    explanation:
      'Best for site selection and multi-criteria suitability. Per-layer weights, ranges and inversion are per-frame parameters, so sliders never recompile.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['model', 'classify'],
    domains: ['ecology', 'urban'],
    params: [
      makeParam('stack', FLOAT, 'Layers stacked layer-major.', {required: true}),
      makeParam('layerCount', 'number', 'Number of layers, at most 16.', {required: true}),
      makeParam('cellCount', 'number', 'Cells per layer.', {required: true}),
      makeParam('noDataValue', 'number', 'Input value treated as nodata.'),
      makeParam('parameters', FLOAT, 'Packed per-layer weights and ranges.', {required: true}),
      makeParam('remapBreaks', FLOAT, 'Break tables for table-mode layers.'),
      makeParam('remapValues', FLOAT, 'Score tables for table-mode layers.'),
      makeParam('output', 'GPUWeightedOverlayOutput', 'Score raster and optional range.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('score', FLOAT, 'Weighted suitability per cell.'),
      makeOutput('outputValidity', UINT, 'Validity flag per cell (optional).'),
      makeOutput('scoreRange', FLOAT, 'Minimum and maximum score (optional).')
    ],
    example: makeExample({
      symbols: ['getGPUWeightedOverlayParameterValues', 'GPUWeightedOverlay'],
      parameters: {
        format: 'float32',
        values: `getGPUWeightedOverlayParameterValues({layers: [{weight: 2, inputMinimum: 0, inputMaximum: 30, invert: true}, {weight: 1}], normalizeWeights: true})`
      },
      setup: ['const cellCount = 512 * 512;'],
      constructorSymbol: 'GPUWeightedOverlay',
      props: [
        'cellCount,',
        'layerCount: 2,',
        `stack: view('criteria', 'float32', cellCount * 2),`,
        `output: {score: view('score', 'float32', cellCount)}`
      ]
    }),
    preview: {kind: 'raster', seed: 35, caption: 'Weights blend criteria into one score.'},
    related: ['raster-reclassify', 'cell-statistics', 'walking-cost'],
    lensScenario: 'gpu-raster.weighted-overlay',
    reference: 'api/gpu-raster/operations-analysis',
    maturity: 'stable'
  },

  // Zonal and rasterization -------------------------------------------------------------------
  {
    id: 'polygon-rasterization',
    symbol: 'GPUPolygonRasterization',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Polygon rasterization',
    summary: 'Scan-converts polygons into a zone-ID raster, with an optional boundary-cell flag.',
    explanation:
      'Turns zones into a grid for zonal statistics or raster joins. Each cell takes the smallest feature row containing its center.',
    dataTypes: {input: ['polygons'], output: ['raster']},
    useCases: ['transform', 'relate'],
    domains: ['data-engineering', 'cartography'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Output grid.', {required: true}),
      makeParam('positions', 'GraphDataView<"float32x2">', 'Packed planar vertices.', {
        required: true
      }),
      makeParam('featureOffsets', UINT, 'Feature to polygon offsets.', {required: true}),
      makeParam('polygonOffsets', UINT, 'Polygon to ring offsets.', {required: true}),
      makeParam('ringOffsets', UINT, 'Ring to vertex offsets.', {required: true}),
      makeParam('crossingCapacity', 'number', 'Maximum edge crossings held.', {required: true}),
      makeParam('parameters', FLOAT, 'Packed raster placement.', {required: true}),
      makeParam('output', 'GPUPolygonRasterizationOutput', 'Zones, boundary flags, status.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('zones', UINT, 'Zone row per cell, or GPU_POLYGON_RASTERIZATION_NO_ZONE.'),
      makeOutput('boundary', UINT, 'Cells touched by a polygon edge (optional).'),
      makeOutput(
        'status',
        'GPUPolygonRasterizationStatus',
        'overflow flag and optional crossingCount.'
      )
    ],
    example: makeExample({
      symbols: [
        'getGPUPolygonRasterizationParameterValues',
        'GPUPolygonRasterization',
        'makeGPURasterMetadata'
      ],
      parameters: {format: 'float32', values: 'getGPUPolygonRasterizationParameterValues(grid)'},
      setup: [
        'const grid = makeGPURasterMetadata({width: 512, height: 512, cellSize: 30});',
        '// `polygons`: {positions, featureOffsets, polygonOffsets, ringOffsets} graph views.'
      ],
      constructorSymbol: 'GPUPolygonRasterization',
      props: [
        'grid,',
        '...polygons,',
        'crossingCapacity: 1 << 20,',
        'output: {',
        `  zones: view('zones', 'uint32', 512 * 512),`,
        `  status: {overflow: view('overflow', 'uint32', 1)}`,
        '}'
      ]
    }),
    preview: {kind: 'polygons', seed: 41, caption: 'Polygons become a grid of zone IDs.'},
    related: ['raster-zonal-statistics', 'raster-join', 'fractional-coverage'],
    lensScenario: 'gpu-raster.polygon-rasterization',
    reference: 'api/gpu-raster/operations-analysis',
    maturity: 'stable'
  },
  {
    id: 'raster-zonal-statistics',
    symbol: 'GPURasterZonalStatistics',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Raster zonal statistics',
    summary:
      'Per-zone count, sum, mean, minimum and maximum of a raster band over a zone-ID raster.',
    explanation:
      'Summarizes imagery or surfaces by parcel, tract or watershed. Zones come from `GPUPolygonRasterization` or any dense ID raster.',
    dataTypes: {input: ['raster'], output: ['table']},
    useCases: ['summarize'],
    domains: ['remote-sensing', 'demographics', 'ecology'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Grid shared by zones and values.', {required: true}),
      makeParam('zones', UINT, 'Dense zone ID per cell.', {required: true}),
      makeParam('values', 'GPURasterBand', 'Band to summarize.', {required: true}),
      makeParam('zoneCapacity', 'number', 'Number of zone rows in the output.', {required: true}),
      makeParam('ignoredZone', 'number', 'Zone ID to skip, such as the no-zone value.'),
      makeParam(
        'cellSizeMode',
        'GPURasterZonalStatisticsCellSizeMode',
        'Weigh cells by area for geographic grids.'
      ),
      makeParam(
        'sumOrder',
        'GPURasterZonalStatisticsSumOrder',
        'Summation order; sorted is reproducible.'
      ),
      makeParam('output', 'GPURasterZonalStatisticsOutput', 'Requested per-zone columns.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('cellCounts', UINT, 'Cells per zone.'),
      makeOutput('valueCounts', UINT, 'Valid cells per zone.'),
      makeOutput('sums', FLOAT, 'Per-zone sum.'),
      makeOutput('means', FLOAT, 'Per-zone mean.'),
      makeOutput('minimums', FLOAT, 'Per-zone minimum.'),
      makeOutput('maximums', FLOAT, 'Per-zone maximum.'),
      makeOutput('status.overflow', UINT, '1 when a zone ID exceeded zoneCapacity.')
    ],
    example: makeExample({
      symbols: ['GPURasterZonalStatistics', 'makeGPURasterBand', 'makeGPURasterMetadata'],
      setup: [GRID_LINE, PIXEL_COUNT_LINE, '// `zones` comes from GPUPolygonRasterization.'],
      constructorSymbol: 'GPURasterZonalStatistics',
      props: [
        'grid,',
        'zones,',
        `values: makeGPURasterBand(view('ndvi', 'float32', n), grid),`,
        'zoneCapacity: 64,',
        'output: {',
        `  means: view('means', 'float32', 64),`,
        `  cellCounts: view('counts', 'uint32', 64)`,
        '}'
      ]
    }),
    preview: {kind: 'polygon-overlay', seed: 42, caption: 'Each zone gets one number.'},
    related: ['polygon-rasterization', 'fractional-zonal-statistics', 'raster-join'],
    lensScenario: 'gpu-raster.raster-zonal-statistics',
    reference: 'api/gpu-raster/operations-analysis',
    maturity: 'stable'
  },
  {
    id: 'raster-join',
    symbol: 'GPURasterJoin',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Raster join',
    summary: 'Approximate point-in-polygon counts and sums by looking points up in a zone raster.',
    explanation:
      'Aggregates millions of points into zones fast. Only points in boundary cells can differ from an exact join, and the output bounds that error.',
    dataTypes: {input: ['points', 'raster'], output: ['table']},
    useCases: ['relate', 'summarize'],
    domains: ['urban', 'data-engineering'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Grid of the zone raster.', {required: true}),
      makeParam('points', 'GraphDataView<"float32x2">', 'Point positions.', {required: true}),
      makeParam('values', FLOAT, 'Per-point values to sum.'),
      makeParam('zones', UINT, 'Zone raster from GPUPolygonRasterization.', {required: true}),
      makeParam('boundary', UINT, 'Boundary-cell flags, to bound the error.'),
      makeParam('zoneCount', 'number', 'Number of zones.', {required: true}),
      makeParam('parameters', FLOAT, 'Packed raster placement.', {required: true}),
      makeParam('output', 'GPURasterJoinOutput', 'Requested counts, sums and bounds.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('counts', UINT, 'Points per zone.'),
      makeOutput('sums', FLOAT, 'Value sums per zone (optional).'),
      makeOutput('boundaryCounts', UINT, 'Points per zone that fall in boundary cells.'),
      makeOutput('pointZones', UINT, 'Joined zone per point (optional).')
    ],
    example: makeExample({
      symbols: [
        'getGPUPolygonRasterizationParameterValues',
        'GPURasterJoin',
        'makeGPURasterMetadata'
      ],
      parameters: {format: 'float32', values: 'getGPUPolygonRasterizationParameterValues(grid)'},
      setup: [
        GRID_LINE,
        '// `points`, `zones`, `boundary`: graph views; zones from GPUPolygonRasterization.'
      ],
      constructorSymbol: 'GPURasterJoin',
      props: [
        'grid,',
        'points,',
        'zones,',
        'boundary,',
        'zoneCount: 64,',
        'output: {',
        `  counts: view('counts', 'uint32', 64),`,
        `  boundaryCounts: view('boundary-counts', 'uint32', 64)`,
        '}'
      ]
    }),
    preview: {kind: 'points', seed: 43, caption: 'Points land in zone cells and are counted.'},
    related: ['polygon-rasterization', 'raster-zonal-statistics', 'raster-sampling'],
    lensScenario: 'gpu-raster.raster-join',
    reference: 'api/gpu-raster/operations-analysis',
    maturity: 'stable'
  },
  {
    id: 'fractional-coverage',
    symbol: 'GPURasterFractionalCoverage',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Fractional coverage',
    summary:
      'Exact fraction of each pixel covered by each polygon, as compact (zone, cell, fraction) tuples.',
    explanation:
      'Best when zones are small relative to pixels and center sampling would misjudge coverage. Triangles are clipped exactly against pixels; the tuples feed the fractional statistics.',
    dataTypes: {input: ['polygons', 'raster'], output: ['table']},
    useCases: ['relate', 'measure'],
    domains: ['remote-sensing', 'cartography'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Pixel grid.', {required: true}),
      makeParam(
        'zones',
        'GPUPolygonGeometryFields',
        'Polygon fields: positions and three offset views.',
        {required: true}
      ),
      makeParam('output', '{zoneIds}', 'zoneIds, cellIds, fractions, status, zoneValidity.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('zoneIds', UINT, 'Zone of each tuple.'),
      makeOutput('cellIds', UINT, 'Cell of each tuple.'),
      makeOutput('fractions', FLOAT, 'Covered fraction of the cell, in (0, 1].'),
      makeOutput('status', '{count, overflow}', 'count, requiredCount and overflow words.'),
      makeOutput('zoneValidity', UINT, '1 where the zone triangulated cleanly.')
    ],
    example: makeExample({
      symbols: ['GPURasterFractionalCoverage', 'makeGPURasterMetadata'],
      setup: [
        'const grid = makeGPURasterMetadata({width: 64, height: 64, cellSize: 10});',
        'const capacity = 1 << 16; // tuple capacity',
        '// `zones`: {positions, featureOffsets, polygonOffsets, ringOffsets} graph views.'
      ],
      constructorSymbol: 'GPURasterFractionalCoverage',
      props: [
        'grid,',
        'zones,',
        'output: {',
        `  zoneIds: view('zone-ids', 'uint32', capacity),`,
        `  cellIds: view('cell-ids', 'uint32', capacity),`,
        `  fractions: view('fractions', 'float32', capacity),`,
        `  status: {count: view('count', 'uint32', 1), requiredCount: view('required', 'uint32', 1), overflow: view('overflow', 'uint32', 1)},`,
        `  zoneValidity: view('zone-valid', 'uint32', 8)`,
        '}'
      ]
    }),
    preview: {kind: 'polygon-overlay', seed: 44, caption: 'Edge pixels carry partial coverage.'},
    related: [
      'fractional-zonal-statistics',
      'fractional-zonal-distribution',
      'fractional-category-composition'
    ],
    lensScenario: 'gpu-raster.raster-fractional-coverage',
    reference: 'api/gpu-raster/expansion',
    maturity: 'stable'
  },
  {
    id: 'fractional-zonal-statistics',
    symbol: 'GPURasterFractionalZonalStatistics',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Fractional zonal statistics',
    summary:
      'Area-weighted zone totals and means that respect partial pixel coverage and missing data.',
    explanation:
      'Best after `GPURasterFractionalCoverage` for exact zonal numbers. `quantity` says whether pixel values are totals, intensities or densities.',
    dataTypes: {input: ['raster', 'table'], output: ['table']},
    useCases: ['summarize'],
    domains: ['remote-sensing', 'demographics'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Pixel grid.', {required: true}),
      makeParam('input', 'GPURasterBufferBand', 'Band to aggregate.', {required: true}),
      makeParam('zoneCount', 'number', 'Number of zones.', {required: true}),
      makeParam(
        'coverage',
        'Record<string, GraphDataView>',
        'Output tuples of GPURasterFractionalCoverage.',
        {
          required: true
        }
      ),
      makeParam('quantity', '"extensive" | "intensive" | "density"', 'Meaning of pixel values.', {
        required: true
      }),
      makeParam('output', '{totals}', 'totals, means, coveredArea, validArea, validity.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('totals', FLOAT, 'Area-weighted total per zone.'),
      makeOutput('means', FLOAT, 'Area-weighted mean per zone.'),
      makeOutput('coveredArea', FLOAT, 'Covered area per zone.'),
      makeOutput('validArea', FLOAT, 'Area with valid data per zone.'),
      makeOutput('validity', UINT, '1 where the zone has a usable result.')
    ],
    example: makeExample({
      symbols: ['GPURasterFractionalZonalStatistics', 'makeGPURasterBand'],
      setup: [
        '// `grid`, and `coverage` from GPURasterFractionalCoverage in this graph.',
        `const input = makeGPURasterBand(view('population', 'float32', grid.width * grid.height), grid);`
      ],
      constructorSymbol: 'GPURasterFractionalZonalStatistics',
      props: [
        'grid,',
        'input,',
        'zoneCount: 8,',
        'coverage,',
        `quantity: 'extensive',`,
        'output: {',
        `  totals: view('totals', 'float32', 8), means: view('means', 'float32', 8),`,
        `  coveredArea: view('covered', 'float32', 8), validArea: view('valid-area', 'float32', 8),`,
        `  validity: view('validity', 'uint32', 8)`,
        '}'
      ]
    }),
    preview: {
      kind: 'polygon-overlay',
      seed: 45,
      caption: 'Partial pixels count by their covered area.'
    },
    related: ['fractional-coverage', 'raster-zonal-statistics', 'fractional-zonal-distribution'],
    lensScenario: 'gpu-raster.raster-fractional-zonal-statistics',
    reference: 'api/gpu-raster/expansion',
    maturity: 'stable'
  },
  {
    id: 'fractional-zonal-distribution',
    symbol: 'GPURasterFractionalZonalDistribution',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Fractional zonal distribution',
    summary: 'Coverage-weighted mean, variance, effective sample size and quantiles per zone.',
    explanation:
      'Best when you need the shape of values inside a zone, not just a mean. `probabilities` selects the quantiles; `observationWeights` adds per-pixel weights.',
    dataTypes: {input: ['raster', 'table'], output: ['table']},
    useCases: ['summarize', 'measure'],
    domains: ['remote-sensing', 'spatial-statistics'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Pixel grid.', {required: true}),
      makeParam('input', 'GPURasterBufferBand', 'Calibrated band to summarize.', {required: true}),
      makeParam('zoneCount', 'number', 'Number of zones.', {required: true}),
      makeParam('coverage', '{zoneIds}', 'Output tuples of GPURasterFractionalCoverage.', {
        required: true
      }),
      makeParam('probabilities', FLOAT, 'Quantile probabilities in [0, 1].', {required: true}),
      makeParam('observationWeights', FLOAT, 'Optional per-pixel weights.'),
      makeParam('quantileMethod', '"exactextract" | "inverse-cdf"', 'Quantile definition.'),
      makeParam('output', 'GPURasterFractionalZonalDistributionOutput', 'Moments and quantiles.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('means', FLOAT, 'Weighted mean per zone.'),
      makeOutput('populationVariances', FLOAT, 'Weighted population variance per zone.'),
      makeOutput('sampleVariances', FLOAT, 'Weighted sample variance per zone.'),
      makeOutput('effectiveSampleSizes', FLOAT, 'Kish effective sample size.'),
      makeOutput('quantiles', FLOAT, 'Requested quantiles per zone.'),
      makeOutput('quantileValidity', FLOAT, 'Flag per requested quantile.'),
      {
        name: 'weightSums',
        type: 'GraphDataView<"float32">',
        description: 'Sum of fractional weights per zone.'
      },
      {
        name: 'squaredWeightSums, coveredArea, validArea, validity, status',
        type: 'mixed views',
        description: 'Further result fields; see the reference.'
      }
    ],
    example: makeExample({
      symbols: ['GPURasterFractionalZonalDistribution'],
      setup: [
        '// `grid`, `input` (a GPURasterBufferBand), `coverage`, `probabilities` (0.1, 0.5, 0.9),',
        '// and `output` (means, variances, quantiles, ...) are created as in the other fractional ops.'
      ],
      constructorSymbol: 'GPURasterFractionalZonalDistribution',
      props: [
        'grid,',
        'input,',
        'zoneCount: 8,',
        'coverage,',
        'probabilities,',
        `quantileMethod: 'inverse-cdf',`,
        'output'
      ]
    }),
    preview: {kind: 'histogram', seed: 46, caption: 'Quantiles of values inside each zone.'},
    related: [
      'fractional-zonal-statistics',
      'fractional-category-composition',
      'fractional-coverage'
    ],
    lensScenario: 'gpu-raster.raster-fractional-zonal-distribution',
    reference: 'api/gpu-raster/fractional-zonal-distribution',
    maturity: 'stable'
  },
  {
    id: 'fractional-category-composition',
    symbol: 'GPURasterFractionalCategoricalStatistics',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Fractional category composition',
    summary:
      'Area and proportion of each land-cover class per zone, with majority and minority class.',
    explanation:
      'Answers "how much of this parcel is forest?" from a classified raster. Class areas come from exact pixel coverage and sum in a stable order.',
    dataTypes: {input: ['raster', 'polygons'], output: ['table']},
    useCases: ['summarize'],
    domains: ['ecology', 'remote-sensing', 'urban'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Pixel grid.', {required: true}),
      makeParam('input', 'GPURasterBufferBand<"uint32">', 'Classified raster of category codes.', {
        required: true
      }),
      makeParam('zoneCount', 'number', 'Number of zones.', {required: true}),
      makeParam('categoryCapacity', 'number', 'Largest category code plus one.', {required: true}),
      makeParam('coverage', '{zoneIds}', 'Output tuples of GPURasterFractionalCoverage.', {
        required: true
      }),
      makeParam('output', '{coverageSums}', 'Class areas, proportions, majority, minority.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('classAreas', FLOAT, 'Area per zone and class.'),
      makeOutput('proportions', FLOAT, 'Share of valid area per zone and class.'),
      makeOutput('majority', UINT, 'Most frequent class per zone.'),
      makeOutput('minority', UINT, 'Least frequent class per zone.'),
      makeOutput('coveredArea', FLOAT, 'Covered area per zone.'),
      makeOutput('validArea', FLOAT, 'Valid-data area per zone.'),
      makeOutput('missingArea', FLOAT, 'Missing-data area per zone.'),
      {
        name: 'coverageSums, validity',
        type: 'mixed views',
        description: 'Further result fields; see the reference.'
      }
    ],
    example: makeExample({
      symbols: ['GPURasterFractionalCategoricalStatistics'],
      setup: [
        '// `grid`, `landCover` (GPURasterBufferBand<"uint32">), `coverage` from',
        '// GPURasterFractionalCoverage, and the output views are created in this graph.'
      ],
      constructorSymbol: 'GPURasterFractionalCategoricalStatistics',
      props: [
        'grid,',
        'input: landCover,',
        'zoneCount: 8,',
        'categoryCapacity: 16,',
        'coverage,',
        'output'
      ]
    }),
    preview: {kind: 'histogram', seed: 47, caption: 'Class shares inside each zone.'},
    related: ['fractional-coverage', 'fractional-zonal-distribution', 'patch-metrics'],
    lensScenario: 'gpu-raster.raster-fractional-categorical-statistics',
    reference: 'api/gpu-raster/fractional-categorical-statistics',
    maturity: 'stable'
  },

  // Sampling and display ----------------------------------------------------------------------
  {
    id: 'raster-sampling',
    symbol: 'GPURasterSampling',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Raster sampling',
    summary: 'Extracts raster values at points with nearest, bilinear or bicubic interpolation.',
    explanation:
      'Attaches elevation, temperature or land cover to point features. Method and nodata policy are per-frame parameters.',
    dataTypes: {input: ['raster', 'points'], output: ['table']},
    useCases: ['interpolate', 'relate'],
    domains: ['remote-sensing', 'data-engineering'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Grid of the raster.', {required: true}),
      makeParam('values', FLOAT, 'Raster values, row-major.', {required: true}),
      makeParam('positions', 'GraphDataView<"float32x2">', 'Sample positions in world units.', {
        required: true
      }),
      makeParam('validity', UINT, 'Per-cell validity flags.'),
      makeParam('noDataValue', 'number', 'Raster value treated as nodata.'),
      makeParam('pointCount', UINT, 'One-row count of points to sample.'),
      makeParam('parameters', FLOAT, 'Packed method, nodata policy and origin.', {required: true}),
      makeParam('output', 'GPURasterSamplingOutput', 'Sampled values.', {required: true})
    ],
    outputs: [
      makeOutput('values', FLOAT, 'One sampled value per point.'),
      makeOutput('outputValidity', UINT, '1 where the sample was valid (optional).')
    ],
    example: makeExample({
      symbols: [
        'getGPURasterSamplingParameterValues',
        'GPURasterSampling',
        'makeGPURasterMetadata'
      ],
      parameters: {
        format: 'float32',
        values: `getGPURasterSamplingParameterValues({method: 'bilinear'}, grid)`
      },
      setup: [
        GRID_LINE,
        '// `elevation` and `positions` are graph views (one row per pixel and per point).'
      ],
      constructorSymbol: 'GPURasterSampling',
      props: [
        'grid,',
        'values: elevation,',
        'positions,',
        `output: {values: view('sampled', 'float32', positions.length)}`
      ]
    }),
    preview: {kind: 'points', seed: 51, caption: 'Points read the surface beneath them.'},
    related: ['raster-profile', 'raster-join', 'fill-nodata'],
    lensScenario: 'gpu-raster.raster-sampling',
    reference: 'api/gpu-raster/operations-analysis',
    maturity: 'stable'
  },
  {
    id: 'raster-profile',
    symbol: 'GPURasterProfile',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Raster profile',
    summary: 'Elevation profiles: samples a raster along polylines at a per-frame spacing.',
    explanation:
      'Best for cross sections, line-of-travel elevation and transect sampling. Spacing is a parameter, so zooming a profile is a buffer write.',
    dataTypes: {input: ['raster', 'lines'], output: ['table']},
    useCases: ['measure', 'interpolate'],
    domains: ['hydrology', 'transport', 'three-d'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Grid of the raster.', {required: true}),
      makeParam('values', FLOAT, 'Raster values, row-major.', {required: true}),
      makeParam('positions', 'GraphDataView<"float32x2">', 'Polyline vertices.', {required: true}),
      makeParam('lineOffsets', UINT, 'Line to vertex offsets with a terminal entry.', {
        required: true
      }),
      makeParam('sampleCapacity', 'number', 'Maximum samples written.'),
      makeParam('validity', UINT, 'Per-cell validity flags.'),
      makeParam('parameters', FLOAT, 'Packed spacing, method and origin.', {required: true}),
      makeParam('output', 'GPURasterProfileOutput', 'Samples and overflow status.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('sampleValues', FLOAT, 'Raster value at each sample.'),
      makeOutput('sampleDistances', FLOAT, 'Distance along the line (optional).'),
      makeOutput('samplePositions', 'GraphDataView<"float32x2">', 'Sample locations (optional).'),
      makeOutput('status', '{count, overflow}', 'Count and overflow words.')
    ],
    example: makeExample({
      symbols: ['getGPURasterProfileParameterValues', 'GPURasterProfile', 'makeGPURasterMetadata'],
      parameters: {
        format: 'float32',
        values: `getGPURasterProfileParameterValues({spacing: 30, method: 'bilinear'}, grid)`
      },
      setup: [GRID_LINE, '// `elevation`, `positions`, `lineOffsets` are graph views.'],
      constructorSymbol: 'GPURasterProfile',
      props: [
        'grid,',
        'values: elevation,',
        'positions,',
        'lineOffsets,',
        'sampleCapacity: 4096,',
        'output: {',
        `  sampleValues: view('profile', 'float32', 4096),`,
        `  status: {count: view('count', 'uint32', 1), overflow: view('overflow', 'uint32', 1)}`,
        '}'
      ]
    }),
    preview: {kind: 'lines', seed: 52, caption: 'Elevation along a path.'},
    related: ['raster-sampling', 'walking-cost', 'extrema-pyramid'],
    lensScenario: 'gpu-raster.raster-profile',
    reference: 'api/gpu-raster/operations-analysis',
    maturity: 'stable'
  },
  {
    id: 'contrast-stretch',
    symbol: 'GPURasterStretch',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Contrast stretch',
    summary:
      'Linear, percentile or histogram-equalized stretch with a colormap lookup and statistics.',
    explanation:
      'Makes imagery readable: pick a mode, a domain and a gamma, then get display values and colors. Statistics come from the GPU with integer atomics.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['classify'],
    domains: ['remote-sensing', 'cartography'],
    params: [
      makeParam('values', FLOAT, 'Raster to stretch.', {required: true}),
      makeParam('grid', 'GPURasterMetadata', 'Grid of the raster.', {required: true}),
      makeParam('validity', UINT, 'Per-cell validity flags.'),
      makeParam('regionMask', UINT, 'Limit statistics to masked cells.'),
      makeParam('palette', UINT, 'Packed colors for the colormap lookup.'),
      makeParam('binCount', 'number', 'Histogram bins.'),
      makeParam('parameters', FLOAT, 'Packed mode, domain, percentiles, gamma.', {required: true}),
      makeParam('output', 'GPURasterStretchOutput', 'Requested display outputs.', {required: true})
    ],
    outputs: [
      makeOutput('stretched', FLOAT, 'Stretched value in [0, 1] per cell (optional).'),
      makeOutput('colors', UINT, 'Packed color per cell from the palette (optional).'),
      makeOutput('histogram', UINT, 'Histogram counts.'),
      makeOutput('statistics', UINT, 'Summary row.')
    ],
    example: makeExample({
      symbols: ['getGPURasterStretchParameterValues', 'GPURasterStretch', 'makeGPURasterMetadata'],
      parameters: {
        format: 'float32',
        values: `getGPURasterStretchParameterValues({mode: 'percentile', percentiles: [2, 98], gamma: 1.2})`
      },
      setup: [
        GRID_LINE,
        PIXEL_COUNT_LINE,
        '// `reflectance` is a float32 graph view with one row per pixel.'
      ],
      constructorSymbol: 'GPURasterStretch',
      props: [
        'values: reflectance,',
        'grid,',
        `output: {stretched: view('stretched', 'float32', n), statistics: view('stats', 'float32', 8)}`
      ]
    }),
    preview: {kind: 'histogram', seed: 53, caption: 'Percentiles pull detail out of a dull band.'},
    related: ['raster-arithmetic', 'raster-reclassify', 'rank-filter'],
    lensScenario: 'gpu-raster.raster-stretch',
    reference: 'api/gpu-raster/operations-analysis',
    maturity: 'stable'
  },

  // Neighborhood and morphology ---------------------------------------------------------------
  {
    id: 'rank-filter',
    symbol: 'GPURasterRankFilter',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Rank filter',
    summary:
      'Focal quantile of a raster: median, minimum, maximum or any percentile in a square window.',
    explanation:
      'Use it to remove speckle (median) or grow and shrink extremes. `quantile` picks the rank; `radius` sets the window.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['transform', 'summarize'],
    domains: ['remote-sensing', 'three-d'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Grid of the raster.', {required: true}),
      makeParam('input', 'GPURasterBufferBand', 'Band to filter.', {required: true}),
      makeParam('radius', 'number', 'Window radius in pixels.', {required: true}),
      makeParam('quantile', 'number', 'Rank in [0, 1]; 0.5 is the median.'),
      makeParam('borderMode', '"clamp" | "nodata"', 'How window edges are treated.'),
      makeParam(
        'noDataPolicy',
        '"ignore" | "propagate"',
        'Skip or propagate nodata in the window.'
      ),
      makeParam('output', '{values}', 'values and validity views.', {required: true})
    ],
    outputs: [
      makeOutput('values', FLOAT, 'Filtered value per pixel.'),
      makeOutput('validity', UINT, '1 where the window had enough valid data.'),
      {
        name: 'changed',
        type: 'GraphDataView<"uint32"> | undefined',
        description: '1 where the filter changed the value.'
      }
    ],
    example: makeExample({
      symbols: ['GPURasterRankFilter', 'makeGPURasterBand', 'makeGPURasterMetadata'],
      setup: [GRID_LINE, PIXEL_COUNT_LINE, BAND_LINE],
      constructorSymbol: 'GPURasterRankFilter',
      props: [
        'grid,',
        `input: band('noisy'),`,
        'radius: 2,',
        'quantile: 0.5,',
        `output: {values: view('median', 'float32', n), validity: view('median-valid', 'uint32', n)}`
      ]
    }),
    preview: {kind: 'raster', seed: 61, caption: 'A median removes speckle and keeps edges.'},
    related: ['morphological-reconstruction', 'fill-nodata', 'glcm-texture'],
    lensScenario: 'gpu-raster.raster-rank-filter',
    reference: 'api/gpu-raster/expansion-lenses',
    maturity: 'stable'
  },
  {
    id: 'focal-correlation-regression',
    symbol: 'GPURasterFocalPairs',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Focal correlation and regression',
    summary: 'Moving-window covariance, correlation and simple regression between two rasters.',
    explanation:
      'Shows where two variables move together locally, such as vegetation versus rainfall. Slope, intercept and R-squared come from an OLS fit in each window.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics', 'remote-sensing', 'ecology'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Shared grid.', {required: true}),
      makeParam('predictor', 'GPURasterBand', 'Explanatory raster.', {required: true}),
      makeParam('response', 'GPURasterBand', 'Dependent raster.', {required: true}),
      makeParam('radius', 'number', 'Window radius in pixels.', {required: true}),
      makeParam('minimumCount', 'number', 'Valid pairs needed for a result.'),
      makeParam('borderMode', '"clamp" | "reflect" | "nodata"', 'How window edges are treated.', {
        default: "'nodata'"
      }),
      makeParam('covarianceNormalization', '"sample" | "population"', 'Covariance denominator.', {
        default: "'sample'"
      }),
      makeParam('output', 'GPURasterFocalPairsOutput', 'Requested metric rasters.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('pairedCount', UINT, 'Valid pairs per window.'),
      makeOutput('status', UINT, 'Status word per window.'),
      makeOutput('covariance', FLOAT, 'Local covariance.'),
      makeOutput('correlation', FLOAT, 'Local Pearson r.'),
      makeOutput('slope', FLOAT, 'Local OLS slope.'),
      makeOutput('intercept', FLOAT, 'Local OLS intercept.'),
      makeOutput('rSquared', FLOAT, 'Local OLS R squared.')
    ],
    example: makeExample({
      symbols: ['GPURasterFocalPairs', 'makeGPURasterBand', 'makeGPURasterMetadata'],
      setup: [GRID_LINE, PIXEL_COUNT_LINE, BAND_LINE],
      constructorSymbol: 'GPURasterFocalPairs',
      props: [
        'grid,',
        `predictor: band('rainfall'),`,
        `response: band('ndvi'),`,
        'radius: 5,',
        'output: {',
        `  pairedCount: view('pairs', 'uint32', n), status: view('status', 'uint32', n),`,
        `  correlation: view('r', 'float32', n), slope: view('slope', 'float32', n)`,
        '}'
      ]
    }),
    preview: {kind: 'regression', seed: 62, caption: 'Local slope shows where the link is strong.'},
    related: ['rank-filter', 'raster-arithmetic', 'glcm-texture'],
    lensScenario: 'gpu-raster.raster-focal-pairs',
    reference: 'api/gpu-raster/focal-pairs',
    maturity: 'stable'
  },
  {
    id: 'glcm-texture',
    symbol: 'GPURasterTexture',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'GLCM texture',
    summary:
      'Local gray-level co-occurrence texture: contrast, homogeneity, energy, correlation, entropy.',
    explanation:
      'Separates surfaces that look alike in value but differ in pattern, such as orchards and forest. Values are quantized to `levels` bins, then paired at `offset` inside a `radius` window.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['find-patterns', 'classify'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Grid of the raster.', {required: true}),
      makeParam('values', FLOAT, 'Raster values to analyze.', {required: true}),
      makeParam('levels', 'number', 'Quantization bins, 2 to 32.', {required: true}),
      makeParam('range', 'readonly [number, number]', 'Value range mapped to the bins.', {
        required: true
      }),
      makeParam('radius', 'number', 'Square window radius, 1 to 8.', {required: true}),
      makeParam('offset', 'readonly [number, number]', 'Pair direction, components in [-8, 8].', {
        required: true
      }),
      makeParam('symmetric', 'boolean', 'Count each pair in both directions.'),
      makeParam('features', FLOAT, 'Seven floats per cell.', {required: true})
    ],
    moreRequired: ['pairCount'],
    outputs: [
      makeOutput(
        'features',
        FLOAT,
        'Contrast, dissimilarity, homogeneity, ASM, energy, correlation, entropy.'
      ),
      makeOutput('pairCount', UINT, 'Pairs counted per window.')
    ],
    example: makeExample({
      symbols: ['GPURasterTexture', 'makeGPURasterMetadata'],
      setup: [
        GRID_LINE,
        PIXEL_COUNT_LINE,
        '// `panchromatic` is a float32 graph view with one row per pixel.'
      ],
      constructorSymbol: 'GPURasterTexture',
      props: [
        'grid,',
        'values: panchromatic,',
        'levels: 16,',
        'range: [0, 1],',
        'radius: 3,',
        'offset: [1, 0],',
        `features: view('texture', 'float32', n * 7),`,
        `pairCount: view('pairs', 'uint32', n)`
      ]
    }),
    preview: {kind: 'raster', seed: 63, caption: 'Pattern, not brightness, separates the fields.'},
    related: ['rank-filter', 'focal-correlation-regression', 'spectral-unmixing'],
    lensScenario: 'gpu-raster.raster-texture',
    reference: 'api/gpu-raster/remote-sensing',
    maturity: 'stable'
  },
  {
    id: 'morphological-reconstruction',
    symbol: 'GPURasterReconstruction',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Morphological reconstruction',
    summary: 'Geodesic dilation or erosion of a marker under a mask, run in bounded GPU rounds.',
    explanation:
      'Keeps only the features that touch your markers, or floods basins and fills pits. Dilation grows the marker up to the mask; erosion shrinks it down to the mask.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['transform'],
    domains: ['hydrology', 'remote-sensing'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Shared grid.', {required: true}),
      makeParam('marker', FLOAT, 'Starting values that propagate.', {required: true}),
      makeParam('mask', FLOAT, 'Values that bound the propagation.', {required: true}),
      makeParam('method', '"dilation" | "erosion"', 'Growth direction.'),
      makeParam('connectivity', '4 | 8', 'Neighbor connectivity.'),
      makeParam('maximumIterations', 'number', 'Bounded number of rounds.'),
      makeParam('output', '{values}', 'values, validity and a converged status.', {required: true})
    ],
    outputs: [
      makeOutput('values', FLOAT, 'Reconstructed raster.'),
      makeOutput('validity', UINT, '1 where the result is valid.'),
      makeOutput('status.converged', UINT, '1 when a fixpoint was reached.')
    ],
    example: makeExample({
      symbols: ['GPURasterReconstruction', 'makeGPURasterMetadata'],
      setup: [GRID_LINE, '// `marker` and `mask` are float32 graph views, one row per pixel.'],
      constructorSymbol: 'GPURasterReconstruction',
      props: [
        'grid,',
        'marker,',
        'mask,',
        `method: 'dilation',`,
        'connectivity: 8,',
        'maximumIterations: 128,',
        `output: {values: view('values', 'float32', 512 * 512), validity: view('valid', 'uint32', 512 * 512),`,
        `  status: {converged: view('converged', 'uint32', 1)}}`
      ]
    }),
    preview: {kind: 'raster', seed: 64, caption: 'Markers grow until the mask stops them.'},
    related: ['fill-holes', 'watershed-segmentation', 'rank-filter'],
    lensScenario: 'gpu-raster.raster-reconstruction',
    reference: 'api/gpu-raster/expansion-lenses',
    maturity: 'stable'
  },
  {
    id: 'fill-holes',
    symbol: 'GPURasterFillHoles',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Fill holes',
    summary: 'Fills background regions of a binary mask that do not connect to the raster edge.',
    explanation:
      'Closes lakes inside land, gaps inside footprints or voids in classified patches. Background connectivity decides what counts as enclosed.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['transform'],
    domains: ['remote-sensing', 'cartography'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Grid of the mask.', {required: true}),
      makeParam('input', UINT, 'Binary mask; nonzero is foreground.', {required: true}),
      makeParam('connectivity', '4 | 8', 'Background connectivity.'),
      makeParam('maximumIterations', 'number', 'Bounded number of rounds.'),
      makeParam('output', '{values}', 'values and a converged status.', {required: true})
    ],
    outputs: [
      makeOutput('values', UINT, 'Mask with enclosed holes filled.'),
      makeOutput('status.converged', UINT, '1 when the flood reached a fixpoint.')
    ],
    example: makeExample({
      symbols: ['GPURasterFillHoles', 'makeGPURasterMetadata'],
      setup: [GRID_LINE, '// `mask` is a uint32 graph view, one row per pixel.'],
      constructorSymbol: 'GPURasterFillHoles',
      props: [
        'grid,',
        'input: mask,',
        'connectivity: 4,',
        `output: {values: view('filled', 'uint32', 512 * 512), status: {converged: view('converged', 'uint32', 1)}}`
      ]
    }),
    preview: {kind: 'raster', seed: 65, caption: 'Enclosed holes fill; open water stays.'},
    related: ['morphological-reconstruction', 'sieve', 'skeleton'],
    lensScenario: 'gpu-raster.raster-fill-holes',
    reference: 'api/gpu-raster/expansion-lenses',
    maturity: 'stable'
  },
  {
    id: 'fill-nodata',
    symbol: 'GPURasterFillNoData',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Fill nodata',
    summary:
      'Fills nodata gaps by bounded nearest, inverse-distance or four-quadrant interpolation.',
    explanation:
      'Patches cloud gaps and sensor dropouts before analysis. `maximumDistance` bounds how far donors are searched, so unresolved cells stay honest.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['interpolate', 'transform'],
    domains: ['remote-sensing', 'hydrology'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Grid of the raster.', {required: true}),
      makeParam('input', 'GPURasterBufferBand', 'Band with gaps.', {required: true}),
      makeParam('maximumDistance', 'number', 'Search radius in pixels.', {required: true}),
      makeParam(
        'method',
        '"nearest" | "inverse-distance" | "gdal-four-quadrant"',
        'Interpolation rule.'
      ),
      makeParam('output', 'GPURasterFillNoDataOutput', 'Filled raster and flags.', {required: true})
    ],
    outputs: [
      makeOutput('values', FLOAT, 'Raster with gaps filled where possible.'),
      makeOutput('validity', UINT, '1 where a value exists after filling.'),
      makeOutput('filled', UINT, '1 where a gap was filled.'),
      makeOutput('unresolvedCount', UINT, 'Gaps no donor reached.')
    ],
    example: makeExample({
      symbols: ['GPURasterFillNoData', 'makeGPURasterBand', 'makeGPURasterMetadata'],
      setup: [GRID_LINE, PIXEL_COUNT_LINE, BAND_LINE],
      constructorSymbol: 'GPURasterFillNoData',
      props: [
        'grid,',
        `input: band('dem'),`,
        'maximumDistance: 16,',
        `method: 'inverse-distance',`,
        'output: {',
        `  values: view('filled', 'float32', n), validity: view('valid', 'uint32', n),`,
        `  filled: view('was-filled', 'uint32', n), unresolvedCount: view('unresolved', 'uint32', 1)`,
        '}'
      ]
    }),
    preview: {kind: 'interpolation', seed: 66, caption: 'Gaps close from their valid neighbors.'},
    related: ['rank-filter', 'raster-sampling', 'raster-reprojection'],
    lensScenario: 'gpu-raster.raster-fill-no-data',
    reference: 'api/gpu-raster/fill-nodata',
    maturity: 'stable'
  },

  // Patches and shape -------------------------------------------------------------------------
  {
    id: 'patch-metrics',
    symbol: 'GPURasterPatchMetrics',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Patch metrics',
    summary:
      'Per-patch area, perimeter and bounding box of a labeled raster, for landscape metrics.',
    explanation:
      'Best after connected-component labeling to measure habitat patches, burn scars or lakes. Perimeter is an exact count of exposed pixel faces.',
    dataTypes: {input: ['raster'], output: ['table']},
    useCases: ['measure'],
    domains: ['ecology', 'remote-sensing'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Grid of the label raster.', {required: true}),
      makeParam('labels', UINT, 'Patch label per pixel; 0 is background.', {required: true}),
      makeParam('validity', UINT, 'Per-pixel validity flags.'),
      makeParam('status', 'GPURasterPatchLabelsStatus', 'Status from the labeling step.'),
      makeParam('countRasterBorder', 'boolean', 'Count faces on the raster edge as perimeter.'),
      makeParam('output', 'GPURasterPatchMetricsOutput', 'Requested per-patch columns.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('pixelCounts', UINT, 'Area in pixels per patch.'),
      makeOutput('areas', FLOAT, 'Area in CRS units squared (optional).'),
      makeOutput('perimeterFaceCounts', UINT, 'Exposed pixel faces per patch.'),
      makeOutput('perimeters', FLOAT, 'Perimeter in CRS units (optional).')
    ],
    example: makeExample({
      symbols: ['GPURasterPatchMetrics', 'makeGPURasterMetadata'],
      setup: [GRID_LINE, '// `labels` is a uint32 graph view from connected-component labeling.'],
      constructorSymbol: 'GPURasterPatchMetrics',
      props: [
        'grid,',
        'labels,',
        'output: {',
        `  pixelCounts: view('pixels', 'uint32', 1024),`,
        `  areas: view('areas', 'float32', 1024)`,
        '}'
      ]
    }),
    preview: {kind: 'raster', seed: 71, caption: 'Each patch gets area and perimeter.'},
    related: ['sieve', 'region-adjacency-graph', 'fractional-category-composition'],
    lensScenario: 'gpu-raster.raster-patch-metrics',
    reference: 'api/gpu-raster/raster-patches',
    maturity: 'stable'
  },
  {
    id: 'sieve',
    symbol: 'GPURasterSieve',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Sieve',
    summary: 'Removes or merges patches smaller than a pixel threshold, like gdal_sieve.',
    explanation:
      'Cleans speckle from a classification. `remove` zeroes small patches; `merge` gives them to their largest neighbor.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['transform', 'classify'],
    domains: ['remote-sensing', 'ecology', 'cartography'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Grid of the label raster.', {required: true}),
      makeParam('labels', UINT, 'Patch label per pixel.', {required: true}),
      makeParam('patchCapacity', 'number', 'Maximum patch count.', {required: true}),
      makeParam('parameters', UINT, 'Packed minimum patch size.', {required: true}),
      makeParam('mode', 'GPURasterSieveMode', 'Remove or merge small patches.'),
      makeParam('connectivity', 'GPURasterSieveConnectivity', 'Patch connectivity, 4 or 8.'),
      makeParam('validity', UINT, 'Per-pixel validity flags.'),
      makeParam('output', 'GPURasterSieveOutput', 'Sieved labels and optional counts.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('labels', UINT, 'Sieved labels; removed patches become 0.'),
      makeOutput('patchTargets', UINT, 'Final label of each patch (optional).'),
      makeOutput('sievedCount', UINT, 'Patches below the threshold (optional).')
    ],
    example: makeExample({
      symbols: ['getGPURasterSieveParameterValues', 'GPURasterSieve', 'makeGPURasterMetadata'],
      parameters: {
        format: 'uint32',
        values: 'getGPURasterSieveParameterValues({minimumPixels: 8})'
      },
      setup: [GRID_LINE, '// `labels` is a uint32 graph view of patch labels.'],
      constructorSymbol: 'GPURasterSieve',
      props: [
        'grid,',
        'labels,',
        'patchCapacity: 1024,',
        `mode: 'merge',`,
        'connectivity: 4,',
        `output: {labels: view('sieved', 'uint32', 512 * 512)}`
      ]
    }),
    preview: {
      kind: 'raster',
      seed: 72,
      caption: 'Small patches vanish into their neighbors.'
    },
    related: ['patch-metrics', 'fill-holes', 'raster-reclassify'],
    lensScenario: 'gpu-raster.raster-sieve',
    reference: 'api/gpu-raster/raster-patches',
    maturity: 'stable'
  },
  {
    id: 'skeleton',
    symbol: 'GPURasterSkeleton',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Skeleton',
    summary: 'Thins a binary mask to a one-pixel centerline with Zhang-Suen thinning.',
    explanation:
      'Extracts road, river or crack centerlines from a mask. Iteration is bounded, and `status.converged` tells you whether it finished.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['transform', 'measure'],
    domains: ['hydrology', 'transport', 'cartography'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Grid of the mask.', {required: true}),
      makeParam('input', UINT, 'Binary mask; nonzero is foreground.', {required: true}),
      makeParam('maximumIterations', 'number', 'Bounded number of thinning rounds.'),
      makeParam('output', '{skeleton}', 'skeleton and a converged status.', {required: true})
    ],
    outputs: [
      makeOutput('skeleton', UINT, 'One-pixel-wide centerline mask.'),
      makeOutput('status.converged', UINT, '1 when thinning reached a fixpoint.')
    ],
    example: makeExample({
      symbols: ['GPURasterSkeleton', 'makeGPURasterMetadata'],
      setup: [GRID_LINE, '// `mask` is a uint32 graph view, one row per pixel.'],
      constructorSymbol: 'GPURasterSkeleton',
      props: [
        'grid,',
        'input: mask,',
        'maximumIterations: 64,',
        `output: {skeleton: view('skeleton', 'uint32', 512 * 512), status: {converged: view('converged', 'uint32', 1)}}`
      ]
    }),
    preview: {kind: 'lines', seed: 73, caption: 'A thick shape thins to its spine.'},
    related: ['medial-axis', 'fill-holes', 'sieve'],
    lensScenario: 'gpu-raster.raster-skeleton',
    reference: 'api/gpu-raster/expansion-lenses',
    maturity: 'stable'
  },
  {
    id: 'medial-axis',
    symbol: 'GPURasterMedialAxis',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Medial axis',
    summary: 'Centers and Euclidean radii of the largest disks that fit inside a binary mask.',
    explanation:
      'Measures local width: river channels, corridors, road gaps. The radius at each axis pixel is the half-width in pixels.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['measure'],
    domains: ['hydrology', 'ecology', 'urban'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Grid of the mask.', {required: true}),
      makeParam('input', UINT, 'Binary mask; nonzero is foreground.', {required: true}),
      makeParam('output', '{medialAxis}', 'Caller-owned medialAxis and radius views.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('medialAxis', UINT, '1 at disk centers.'),
      makeOutput('radius', FLOAT, 'Disk radius in pixels at each center.')
    ],
    example: makeExample({
      symbols: ['GPURasterMedialAxis', 'makeGPURasterMetadata'],
      setup: [GRID_LINE, '// `mask` is a uint32 graph view.'],
      constructorSymbol: 'GPURasterMedialAxis',
      props: [
        'grid,',
        'input: mask,',
        'output: {',
        `  medialAxis: view('axis', 'uint32', 512 * 512),`,
        `  radius: view('radius', 'float32', 512 * 512)`,
        '}'
      ]
    }),
    preview: {kind: 'lines', seed: 74, caption: 'Axis pixels carry half-width.'},
    related: ['skeleton', 'fill-holes', 'patch-metrics'],
    lensScenario: 'gpu-raster.raster-medial-axis',
    reference: 'api/gpu-raster/expansion',
    maturity: 'stable'
  },
  {
    id: 'region-adjacency-graph',
    symbol: 'GPURasterRegionGraph',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Region adjacency graph',
    summary: 'Builds four-neighbor region adjacency and merges similar regions by mean difference.',
    explanation:
      'Turns over-segmented imagery into coherent objects. Fixed or adaptive thresholds merge neighbors in bounded rounds.',
    dataTypes: {input: ['raster'], output: ['network']},
    useCases: ['find-patterns', 'classify'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Grid of the label raster.', {required: true}),
      makeParam('labels', UINT, 'Region label per pixel.', {required: true}),
      makeParam('values', FLOAT, 'Pixel values to average per region.', {required: true}),
      makeParam('regionCount', 'number', 'Number of regions.', {required: true}),
      makeParam('mergeMode', '"fixed" | "adaptive"', 'Threshold rule for merging.'),
      makeParam('mergeThreshold', 'number', 'Mean-difference threshold.', {required: true}),
      makeParam('maximumIterations', 'number', 'Bounded merge rounds.', {required: true}),
      makeParam('output', '{adjacency}', 'adjacency, weights, mapping, status.', {required: true})
    ],
    outputs: [
      makeOutput('adjacency', UINT, 'Region pairs that touch.'),
      makeOutput('weights', FLOAT, 'Mean difference per adjacent pair.'),
      makeOutput('mapping', UINT, 'Final merged region for each region.'),
      makeOutput('status', UINT, 'Merge status word.')
    ],
    example: makeExample({
      symbols: ['GPURasterRegionGraph', 'makeGPURasterMetadata'],
      setup: [
        GRID_LINE,
        '// `labels` (uint32) and `values` (float32) are graph views; 256 regions.'
      ],
      constructorSymbol: 'GPURasterRegionGraph',
      props: [
        'grid,',
        'labels,',
        'values,',
        'regionCount: 256,',
        `mergeMode: 'adaptive',`,
        'mergeThreshold: 0.05,',
        'maximumIterations: 8,',
        `output: {adjacency: view('adjacency', 'uint32', 2048), weights: view('weights', 'float32', 1024), mapping: view('mapping', 'uint32', 256), status: view('status', 'uint32', 1)}`
      ]
    }),
    preview: {kind: 'network', seed: 75, caption: 'Regions are nodes; shared borders are edges.'},
    related: ['patch-metrics', 'watershed-segmentation', 'sieve'],
    lensScenario: 'gpu-raster.raster-region-graph',
    reference: 'api/gpu-raster/remote-sensing',
    maturity: 'stable'
  },
  {
    id: 'watershed-segmentation',
    symbol: 'GPURasterWatershedSegmentation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Watershed segmentation',
    summary: 'Marker-controlled segmentation that floods a surface outward from seed labels.',
    explanation:
      'Splits touching objects such as crowns, cells or catchments. Each seed grows through the lowest-cost route; `method` picks FIFO flooding or a minimum-spanning forest.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['classify', 'find-patterns'],
    domains: ['hydrology', 'remote-sensing', 'ecology'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Shared grid.', {required: true}),
      makeParam(
        'input',
        'GPURasterBufferBand',
        'Surface to flood, such as gradient or elevation.',
        {required: true}
      ),
      makeParam('markers', UINT, 'Seed labels; 0 means unlabeled.', {required: true}),
      makeParam('connectivity', '4 | 8', 'Neighbor connectivity.'),
      makeParam('method', '"fifo" | "minimum-spanning-forest"', 'Flooding semantics.'),
      makeParam('maximumSteps', 'number', 'Bounded flood steps.'),
      makeParam('maximumIterations', 'number', 'Bounded outer rounds.'),
      makeParam('output', '{labels}', 'labels and a converged status.', {required: true})
    ],
    outputs: [
      makeOutput('labels', UINT, 'Segment label per pixel.'),
      makeOutput('status.converged', UINT, '1 when every reachable pixel was labeled.')
    ],
    example: makeExample({
      symbols: ['GPURasterWatershedSegmentation', 'makeGPURasterBand', 'makeGPURasterMetadata'],
      setup: [
        GRID_LINE,
        PIXEL_COUNT_LINE,
        BAND_LINE,
        '// `markers` is a uint32 graph view of seed labels.'
      ],
      constructorSymbol: 'GPURasterWatershedSegmentation',
      props: [
        'grid,',
        `input: band('gradient'),`,
        'markers,',
        `method: 'fifo',`,
        'connectivity: 8,',
        `output: {labels: view('labels', 'uint32', n), status: {converged: view('converged', 'uint32', 1)}}`
      ]
    }),
    preview: {kind: 'flow-accumulation', seed: 76, caption: 'Seeds flood outward until they meet.'},
    related: ['morphological-reconstruction', 'region-adjacency-graph', 'walking-cost'],
    lensScenario: 'gpu-raster.raster-watershed-segmentation',
    reference: 'api/gpu-raster/expansion-lenses',
    maturity: 'stable'
  },
  {
    id: 'extrema-pyramid',
    symbol: 'GPURasterExtremaPyramid',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Extrema pyramid',
    summary: 'Min/max mipmap of a raster band for fast range queries, peaks and culling.',
    explanation:
      'Answers "what is the highest cell in this region?" in a few lookups. Each level stores the exact minimum and maximum of a block, doubling in size per level.',
    dataTypes: {input: ['raster', 'terrain'], output: ['raster']},
    useCases: ['index', 'summarize'],
    domains: ['hydrology', 'three-d'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Grid of the band.', {required: true}),
      makeParam('input', 'GPURasterBand', 'Band to summarize.', {required: true}),
      makeParam('firstBlockSize', 'number', 'Pixel block size of level 0.'),
      makeParam('levelCount', 'number', 'Number of pyramid levels.'),
      makeParam('footprint', 'GPURasterExtremaPyramidFootprint', 'Block footprint rule.'),
      makeParam('output', 'GPURasterExtremaPyramidViews', 'maximum, minimum or combined.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('maximum', FLOAT, 'Block maxima for every level (optional).'),
      makeOutput('minimum', FLOAT, 'Block minima for every level (optional).'),
      makeOutput('combined', FLOAT, 'Interleaved min and max (optional).')
    ],
    example: makeExample({
      symbols: ['GPURasterExtremaPyramid', 'makeGPURasterBand', 'makeGPURasterMetadata'],
      setup: [
        GRID_LINE,
        PIXEL_COUNT_LINE,
        BAND_LINE,
        '// `maximum` and `minimum` are float32 views sized by the pyramid layout.'
      ],
      constructorSymbol: 'GPURasterExtremaPyramid',
      props: [
        'grid,',
        `input: band('dem'),`,
        'firstBlockSize: 4,',
        'levelCount: 5,',
        'output: {maximum, minimum}'
      ]
    }),
    preview: {kind: 'terrain', seed: 77, caption: 'Each level keeps block min and max.'},
    related: ['raster-profile', 'raster-sampling', 'rank-filter'],
    lensScenario: 'gpu-raster.raster-extrema-pyramid',
    reference: 'api/gpu-raster/operations-analysis',
    maturity: 'stable'
  },

  // Cost and movement -------------------------------------------------------------------------
  {
    id: 'walking-cost',
    symbol: 'GPURasterWalkingCost',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Walking cost',
    summary:
      'Least-cost travel over terrain with a slope-aware walking model and surface friction.',
    explanation:
      'Best for hiking time, emergency access and wildlife movement. The model is Tobler or a GRASS-style energy model; costs relax to a fixpoint from the source cells.',
    dataTypes: {input: ['terrain', 'raster'], output: ['raster']},
    useCases: ['route', 'simulate'],
    domains: ['hydrology', 'ecology', 'transport'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Shared grid.', {required: true}),
      makeParam('elevation', 'GPURasterBufferBand', 'Elevation band.', {required: true}),
      makeParam('friction', 'GPURasterBufferBand', 'Surface friction band.', {required: true}),
      makeParam('sources', UINT, 'Source cell indices.', {required: true}),
      makeParam('frictionWeight', 'number', 'Weight of friction against slope energy.'),
      makeParam(
        'model',
        '{kind: "tobler"} | {kind: "grass", coefficients, downhillSlopeThreshold}',
        'Energy model.'
      ),
      makeParam('maximumIterations', 'number', 'Bounded relaxation rounds.'),
      makeParam('output', '{costs}', 'costs, validity and a converged status.', {required: true})
    ],
    outputs: [
      makeOutput('costs', FLOAT, 'Accumulated cost from the nearest source.'),
      makeOutput('validity', UINT, '1 where a cost was found.'),
      makeOutput('status.converged', UINT, '1 when relaxation reached a fixpoint.')
    ],
    example: makeExample({
      symbols: ['GPURasterWalkingCost', 'makeGPURasterBand', 'makeGPURasterMetadata'],
      setup: [
        GRID_LINE,
        PIXEL_COUNT_LINE,
        BAND_LINE,
        '// `sources` is a uint32 view of start cell indices.'
      ],
      constructorSymbol: 'GPURasterWalkingCost',
      props: [
        'grid,',
        `elevation: band('dem'),`,
        `friction: band('land-cover-friction'),`,
        'sources,',
        `model: {kind: 'tobler'},`,
        'maximumIterations: 256,',
        `output: {costs: view('costs', 'float32', n), validity: view('valid', 'uint32', n), status: {converged: view('converged', 'uint32', 1)}}`
      ]
    }),
    preview: {kind: 'isochrones', seed: 81, caption: 'Slope bends the reach of a walker.'},
    related: ['raster-profile', 'weighted-overlay', 'watershed-segmentation'],
    lensScenario: 'gpu-raster.raster-walking-cost',
    reference: 'api/gpu-raster/expansion-lenses',
    maturity: 'stable'
  },

  // Reprojection and remote sensing -----------------------------------------------------------
  {
    id: 'raster-reprojection',
    symbol: 'GPURasterReprojection',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Raster reprojection',
    summary: 'Resamples a band onto another grid, pulling each destination pixel from the source.',
    explanation:
      'Aligns rasters on a common grid or projection before combining them. Nearest keeps categories; bilinear smooths measurements; `renormalize` handles partial nodata.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['transform'],
    domains: ['remote-sensing', 'cartography'],
    params: [
      makeParam('sourceGrid', 'GPURasterMetadata', 'Grid of the input band.', {required: true}),
      makeParam('destinationGrid', 'GPURasterMetadata', 'Grid to resample onto.', {required: true}),
      makeParam('input', 'GPURasterBufferBand', 'Band to resample.', {required: true}),
      makeParam(
        'inverseProjection',
        'CompiledProjection',
        'Destination-to-source coordinate transform.'
      ),
      makeParam('method', '"nearest" | "bilinear"', 'Resampling kernel.'),
      makeParam('noDataPolicy', '"strict" | "renormalize"', 'How nodata neighbors are handled.'),
      makeParam('output', '{values}', 'values and validity on the destination grid.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('values', FLOAT, 'Resampled value per destination pixel.'),
      makeOutput('validity', UINT, '1 where the destination pixel had source data.')
    ],
    example: makeExample({
      symbols: ['GPURasterReprojection', 'makeGPURasterBand', 'makeGPURasterMetadata'],
      setup: [
        'const sourceGrid = makeGPURasterMetadata({width: 512, height: 512, cellSize: 30});',
        'const destinationGrid = makeGPURasterMetadata({width: 600, height: 600, cellSize: 25});',
        `const input = makeGPURasterBand(view('src', 'float32', 512 * 512), sourceGrid);`
      ],
      constructorSymbol: 'GPURasterReprojection',
      props: [
        'sourceGrid,',
        'destinationGrid,',
        'input,',
        `method: 'bilinear',`,
        `noDataPolicy: 'renormalize',`,
        `output: {values: view('dst', 'float32', 600 * 600), validity: view('dst-valid', 'uint32', 600 * 600)}`
      ]
    }),
    preview: {kind: 'raster', seed: 91, caption: 'Pixels pulled onto a new grid.'},
    related: ['tile-mosaic-reprojection', 'conservative-remapping', 'raster-registration'],
    lensScenario: 'gpu-raster.raster-reprojection',
    reference: 'api/gpu-raster/expansion',
    maturity: 'stable'
  },
  {
    id: 'tile-mosaic-reprojection',
    symbol: 'GPURasterTileReprojection',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Tile mosaic reprojection',
    summary: 'Assembles source tiles into a mosaic on the GPU, then reprojects it in one pass.',
    explanation:
      'Best for tiled imagery such as COGs and map tiles. Each tile carries its pixel bounds, so halos and partial tiles need no CPU stitching.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['transform'],
    domains: ['remote-sensing', 'cartography', 'data-engineering'],
    params: [
      makeParam('sourceGrid', 'GPURasterMetadata', 'Grid of the full mosaic.', {required: true}),
      makeParam('destinationGrid', 'GPURasterMetadata', 'Grid to resample onto.', {required: true}),
      makeParam(
        'sources',
        'readonly GPURasterReprojectionSourceTile[]',
        'Tiles with pixelBounds and input band.',
        {required: true}
      ),
      makeParam(
        'inverseProjection',
        'CompiledProjection',
        'Destination-to-source coordinate transform.'
      ),
      makeParam('method', '"nearest" | "bilinear"', 'Resampling kernel.'),
      makeParam('noDataPolicy', '"strict" | "renormalize"', 'How nodata neighbors are handled.'),
      makeParam('output', '{values}', 'values and validity on the destination grid.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('values', FLOAT, 'Resampled value per destination pixel.'),
      makeOutput('validity', UINT, '1 where any tile covered the pixel.')
    ],
    example: makeExample({
      symbols: ['GPURasterTileReprojection', 'makeGPURasterBand', 'makeGPURasterMetadata'],
      setup: [
        'const sourceGrid = makeGPURasterMetadata({width: 512, height: 512, cellSize: 30});',
        'const destinationGrid = makeGPURasterMetadata({width: 512, height: 512, cellSize: 30});',
        "const tile = (pixelBounds: [number, number, number, number], id: string) => ({pixelBounds, input: makeGPURasterBand(view(id, 'float32', 256 * 256), sourceGrid)});"
      ],
      constructorSymbol: 'GPURasterTileReprojection',
      props: [
        'sourceGrid,',
        'destinationGrid,',
        `sources: [tile([0, 0, 256, 256], 'a'), tile([256, 0, 512, 256], 'b')],`,
        `method: 'nearest',`,
        `output: {values: view('mosaic', 'float32', 512 * 512), validity: view('mosaic-valid', 'uint32', 512 * 512)}`
      ]
    }),
    preview: {kind: 'cells', seed: 92, caption: 'Tiles join and reproject in one pass.'},
    related: ['raster-reprojection', 'conservative-remapping', 'fill-nodata'],
    lensScenario: 'gpu-raster.raster-tile-reprojection',
    reference: 'api/gpu-raster/tile-reprojection',
    maturity: 'stable'
  },
  {
    id: 'conservative-remapping',
    symbol: 'GPURasterConservativeRemapping',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Conservative remapping',
    summary: 'Regrids values by exact rectangle overlap, so totals are preserved.',
    explanation:
      'Moves counts, mass or area-weighted values between grids, such as population onto a coarser mesh. `normalization` chooses totals, density or covered-area mean; weights can be prepared once and reused.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['transform', 'summarize'],
    domains: ['remote-sensing', 'demographics'],
    params: [
      makeParam('source', 'GPURasterRectilinearGrid', 'Source width, height, origin, spacing.', {
        required: true
      }),
      makeParam('destination', 'GPURasterRectilinearGrid', 'Destination grid.', {required: true}),
      makeParam('values', FLOAT, 'Source values.', {required: true}),
      makeParam('validity', UINT, 'Per-cell validity flags.'),
      makeParam(
        'normalization',
        '"extensive" | "destination-area" | "covered-area"',
        'How overlap sums are scaled.',
        {required: true}
      ),
      makeParam('weightMode', '"prepare" | "reuse"', 'Build or reuse overlap weights.'),
      makeParam('preparedWeights', '{indices, areas}', 'Caller-owned weights to retain.'),
      makeParam('output', '{values}', 'values, coveredArea and status.', {required: true})
    ],
    outputs: [
      makeOutput('values', FLOAT, 'Remapped value per destination cell.'),
      makeOutput('coveredArea', FLOAT, 'Source area that overlapped each cell.'),
      makeOutput('status', UINT, '0 finite, 1 no overlap, 2 nonfinite.')
    ],
    example: makeExample({
      symbols: ['GPURasterConservativeRemapping'],
      setup: [
        'const source = {width: 256, height: 256, origin: [0, 0] as const, spacing: [1, 1] as const};',
        'const destination = {width: 64, height: 64, origin: [0, 0] as const, spacing: [4, 4] as const};'
      ],
      constructorSymbol: 'GPURasterConservativeRemapping',
      props: [
        'source,',
        'destination,',
        `values: view('population', 'float32', 256 * 256),`,
        `normalization: 'extensive',`,
        'output: {',
        `  values: view('regridded', 'float32', 64 * 64), coveredArea: view('covered', 'float32', 64 * 64),`,
        `  status: view('status', 'uint32', 64 * 64)`,
        '}'
      ]
    }),
    preview: {kind: 'cells', seed: 93, caption: 'Fine cells sum into coarse cells.'},
    related: ['raster-reprojection', 'fractional-zonal-statistics', 'tile-mosaic-reprojection'],
    lensScenario: 'gpu-raster.raster-conservative-remapping',
    reference: 'api/gpu-raster/remote-sensing',
    maturity: 'stable'
  },
  {
    id: 'raster-registration',
    symbol: 'GPURasterRegistration',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Raster registration',
    summary: 'Estimates the sub-pixel shift between two images by phase correlation.',
    explanation:
      'Use it to co-register scenes before change detection. A localized upsampled inverse DFT refines the correlation peak below one pixel.',
    dataTypes: {input: ['raster'], output: ['table']},
    useCases: ['detect-change', 'transform'],
    domains: ['remote-sensing'],
    params: [
      makeParam('grid', 'GPURasterMetadata', 'Shared grid.', {required: true}),
      makeParam('reference', FLOAT, 'Reference image.', {required: true}),
      makeParam('moving', FLOAT, 'Image to align.', {required: true}),
      makeParam('mode', '"periodic" | "masked-linear"', 'Correlation model; masked uses validity.'),
      makeParam('maximumShift', 'readonly [number, number]', 'Largest shift searched, in pixels.'),
      makeParam('upsampleFactor', 'number', 'Sub-pixel refinement factor.'),
      makeParam('minimumCorrelation', 'number', 'Reject peaks weaker than this.'),
      makeParam('output', '{shift}', 'shift, peak, status and optional overlap.', {required: true})
    ],
    outputs: [
      makeOutput(
        'shift',
        'GraphDataView<"float32x2">',
        'Shift of moving relative to reference, in pixels.'
      ),
      makeOutput('peak', FLOAT, 'Correlation strength at the peak.'),
      makeOutput('status', UINT, 'Status word; nonzero means no reliable shift.')
    ],
    example: makeExample({
      symbols: ['GPURasterRegistration', 'makeGPURasterMetadata'],
      setup: [GRID_LINE, '// `reference` and `moving` are float32 graph views, one row per pixel.'],
      constructorSymbol: 'GPURasterRegistration',
      props: [
        'grid,',
        'reference,',
        'moving,',
        'upsampleFactor: 20,',
        `output: {shift: view('shift', 'float32x2', 1), peak: view('peak', 'float32', 1), status: view('status', 'uint32', 1)}`
      ]
    }),
    preview: {kind: 'pipeline', seed: 94, caption: 'The correlation peak gives the offset.'},
    related: ['raster-reprojection', 'change-detection', 'raster-sampling'],
    lensScenario: 'gpu-raster.raster-registration',
    reference: 'api/gpu-raster/remote-sensing',
    maturity: 'stable'
  },
  {
    id: 'spectral-unmixing',
    symbol: 'GPURasterSpectralUnmixing',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spectral unmixing',
    summary:
      'Solves each pixel as a mix of endmember spectra, with optional non-negative or simplex limits.',
    explanation:
      'Estimates sub-pixel fractions of vegetation, soil and water. Up to 256 bands and 8 endmembers; the residual shows how well the mix fits.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['model', 'classify'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Pixels to unmix.', {required: true}),
      makeParam('bandCount', 'number', 'Bands per pixel.', {required: true}),
      makeParam('endmemberCount', 'number', 'Pure spectra.', {required: true}),
      makeParam('values', FLOAT, 'Pixel spectra, cell-major.', {required: true}),
      makeParam('endmembers', FLOAT, 'Endmember spectra.', {required: true}),
      makeParam(
        'constraint',
        '"unconstrained" | "nonnegative" | "simplex"',
        'Abundance constraint.'
      ),
      makeParam('validity', UINT, 'Per-pixel validity flags.'),
      makeParam('output', '{abundances}', 'abundances, residual, status, iterationCounts.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('abundances', FLOAT, 'Fraction of each endmember per pixel.'),
      makeOutput('residual', FLOAT, 'Fit error per pixel.'),
      makeOutput('status', UINT, 'Solver status per cell.'),
      makeOutput('iterationCounts', UINT, 'Iterations used per cell.')
    ],
    example: makeExample({
      symbols: ['GPURasterSpectralUnmixing'],
      setup: ['const cellCount = 512 * 512;', 'const bandCount = 6;', 'const endmemberCount = 3;'],
      constructorSymbol: 'GPURasterSpectralUnmixing',
      props: [
        'cellCount,',
        'bandCount,',
        'endmemberCount,',
        `values: view('spectra', 'float32', cellCount * bandCount),`,
        `endmembers: view('endmembers', 'float32', endmemberCount * bandCount),`,
        `constraint: 'simplex',`,
        'output: {',
        `  abundances: view('abundances', 'float32', cellCount * endmemberCount), residual: view('residual', 'float32', cellCount),`,
        `  status: view('status', 'uint32', cellCount), iterationCounts: view('iterations', 'uint32', cellCount)`,
        '}'
      ]
    }),
    preview: {kind: 'raster', seed: 95, caption: 'Each pixel splits into endmember fractions.'},
    related: ['soil-adjusted-vegetation-index', 'glcm-texture', 'temporal-composite'],
    lensScenario: 'gpu-raster.raster-spectral-unmixing',
    reference: 'api/gpu-raster/remote-sensing',
    maturity: 'stable'
  },

  // Time ----------------------------------------------------------------------------------------
  {
    id: 'change-detection',
    symbol: 'GPUChangeDetection',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Change detection',
    summary:
      'Per-cell change between dates: difference, log ratio, t-test, Theil-Sen slope and Mann-Kendall trend.',
    explanation:
      'Finds where a landscape changed and whether the change is significant. Request outputs for two-slice comparison, group tests or trends; multiband stacks give change magnitude and direction.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['detect-change'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('slices', FLOAT, 'Stack indexed (cell * sliceCount + slice) * bandCount + band.', {
        required: true
      }),
      makeParam('cellCount', 'number', 'Number of cells.', {required: true}),
      makeParam('sliceCount', 'number', 'Number of time slices.', {required: true}),
      makeParam('bandCount', 'number', 'Bands per slice; 2 to 16 gives magnitude.'),
      makeParam('mask', UINT, 'Per-cell mask; zero skips the cell.'),
      makeParam(
        'significanceSource',
        'GPUChangeDetectionSignificanceSource',
        'Test behind the significance class.'
      ),
      makeParam('parameters', FLOAT, 'Packed slices, epsilon, alpha, split.', {required: true}),
      makeParam('output', 'GPUChangeDetectionOutput', 'Requested change layers.', {required: true})
    ],
    outputs: [
      makeOutput('difference', FLOAT, 'After minus before.'),
      makeOutput('logRatio', FLOAT, 'Log ratio of after to before.'),
      makeOutput('percentChange', FLOAT, 'Percent change from before.'),
      makeOutput('tStatistic', FLOAT, 'Welch t statistic between groups.'),
      makeOutput('tPValue', FLOAT, 'Welch test p-value.'),
      makeOutput('senSlope', FLOAT, 'Sen slope per cell.'),
      makeOutput('mannKendallZ', FLOAT, 'Mann-Kendall z per cell.'),
      makeOutput('significance', UINT, '0 none, 1 significant increase, 2 decrease.')
    ],
    example: makeExample({
      symbols: ['getGPUChangeDetectionParameterValues', 'GPUChangeDetection'],
      parameters: {
        format: 'float32',
        values: 'getGPUChangeDetectionParameterValues({beforeSlice: 0, afterSlice: 5})'
      },
      setup: ['const cellCount = 512 * 512;', 'const sliceCount = 6;'],
      constructorSymbol: 'GPUChangeDetection',
      props: [
        'cellCount,',
        'sliceCount,',
        `slices: view('stack', 'float32', cellCount * sliceCount),`,
        'output: {',
        `  difference: view('difference', 'float32', cellCount),`,
        `  senSlope: view('slope', 'float32', cellCount)`,
        '}'
      ]
    }),
    preview: {kind: 'time-series', seed: 101, caption: 'Trend and significance per cell.'},
    related: ['normalized-burn-ratio', 'seasonal-break-detection', 'temporal-composite'],
    lensScenario: 'gpu-raster.change-detection',
    reference: 'api/gpu-raster/operations-analysis',
    maturity: 'stable'
  },
  {
    id: 'seasonal-break-detection',
    symbol: 'GPURasterSeasonalBreak',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Seasonal break detection',
    summary: 'Fits a trend plus seasonal harmonic per cell and finds at most one level shift.',
    explanation:
      'Use it to date disturbances such as clearing, fire or flooding in a long time series. A BIC gain test decides whether a break is real; `monitoring` can flag new observations.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['detect-change', 'model'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Number of cells.', {required: true}),
      makeParam('sliceCount', 'number', 'Observations per cell.', {required: true}),
      makeParam('values', FLOAT, 'Series per cell, cell-major.', {required: true}),
      makeParam('validity', UINT, 'Per-observation validity flags.'),
      makeParam('period', 'number', 'Seasonal period in slices.', {required: true}),
      makeParam('minimumSegmentLength', 'number', 'Fewest observations on each side.', {
        required: true
      }),
      makeParam('minimumBICGain', 'number', 'Required BIC improvement for a break.'),
      makeParam('seasonality', '"harmonic" | "dummy"', 'Seasonal model.')
    ],
    outputs: [
      makeOutput('breakIndex', UINT, 'Slice of the detected break per cell.'),
      makeOutput('improvement', FLOAT, 'BIC improvement from adding the break.'),
      makeOutput('coefficients', FLOAT, 'Fitted coefficients per cell.'),
      makeOutput('components', FLOAT, 'Model components per cell.'),
      makeOutput('status', UINT, '0 fit, 1 insufficient support, 2 rank failure.')
    ],
    example: makeExample({
      symbols: ['GPURasterSeasonalBreak'],
      setup: ['const cellCount = 256 * 256;', 'const sliceCount = 120; // monthly observations'],
      constructorSymbol: 'GPURasterSeasonalBreak',
      props: [
        'cellCount,',
        'sliceCount,',
        `values: view('series', 'float32', cellCount * sliceCount),`,
        'period: 12,',
        'minimumSegmentLength: 12,',
        'output // coefficients, components, breakIndex, improvement, status'
      ]
    }),
    preview: {kind: 'time-series', seed: 102, caption: 'A level shift against a seasonal cycle.'},
    related: ['change-detection', 'temporal-composite', 'enhanced-vegetation-index'],
    lensScenario: 'gpu-raster.raster-seasonal-break',
    reference: 'api/gpu-raster/remote-sensing',
    maturity: 'stable'
  },
  {
    id: 'temporal-composite',
    symbol: 'GPURasterTemporal',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Temporal composite',
    summary: 'Per-cell mean, median, minimum, maximum or Savitzky-Golay smoothing over time.',
    explanation:
      'Builds cloud-free composites or to smooth noisy index series. Composites write one row per cell; smoothing writes one per cell and slice.',
    dataTypes: {input: ['raster', 'time-series'], output: ['raster']},
    useCases: ['summarize', 'transform'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('cellCount', 'number', 'Number of cells.', {required: true}),
      makeParam('sliceCount', 'number', 'Observations per cell.', {required: true}),
      makeParam('values', FLOAT, 'Series per cell, cell-major.', {required: true}),
      makeParam('validity', UINT, 'Per-observation validity flags.'),
      makeParam(
        'operation',
        '"mean" | "median" | "minimum" | "maximum" | "savitzky-golay"',
        'Reduction or filter.',
        {required: true}
      ),
      makeParam('windowLength', 'number', 'Savitzky-Golay window, odd 3 to 15.'),
      makeParam('polynomialOrder', 'number', 'Savitzky-Golay polynomial order.'),
      makeParam('output', '{values}', 'values, counts, chosenSlice, status.', {required: true})
    ],
    outputs: [
      makeOutput('values', FLOAT, 'Composite or smoothed values.'),
      makeOutput('counts', UINT, 'Valid observations used.'),
      makeOutput('chosenSlice', UINT, 'Slice that supplied an extreme or median.'),
      makeOutput('status', UINT, 'Per-cell status word.')
    ],
    example: makeExample({
      symbols: ['GPURasterTemporal'],
      setup: ['const cellCount = 512 * 512;', 'const sliceCount = 8;'],
      constructorSymbol: 'GPURasterTemporal',
      props: [
        'cellCount,',
        'sliceCount,',
        `values: view('scenes', 'float32', cellCount * sliceCount),`,
        `validity: view('clear', 'uint32', cellCount * sliceCount),`,
        `operation: 'median',`,
        'output: {',
        `  values: view('composite', 'float32', cellCount), counts: view('counts', 'uint32', cellCount),`,
        `  chosenSlice: view('chosen', 'uint32', cellCount), status: view('status', 'uint32', cellCount)`,
        '}'
      ]
    }),
    preview: {kind: 'time-series', seed: 103, caption: 'Many dates collapse to one clear view.'},
    related: ['cell-statistics', 'change-detection', 'seasonal-break-detection'],
    lensScenario: 'gpu-raster.raster-temporal',
    reference: 'api/gpu-raster/remote-sensing',
    maturity: 'stable'
  }
];
