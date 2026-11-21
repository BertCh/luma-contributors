// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Catalog group "regression-weights": spatial weights, neighbor search and the regression models
// built on them (OLS, spatial lag and error, GWR, panels, regimes).

import type {CatalogEntry, OutputDoc, ParamDoc} from '../schema';

const IMPORT_PATH = '@luma-contributors/analysis/gpu-spatial-analysis';

const param = (
  name: string,
  type: string,
  description: string,
  extra: {required?: boolean; default?: string} = {}
): ParamDoc => ({name, type, description, ...extra});

const output = (name: string, type: string, description: string): OutputDoc => ({
  name,
  type,
  description
});

/**
 * Builds a runnable-shaped example. `output(id, format, length)` and `csr(rows, capacity)` in the
 * bodies stand for caller helpers that wrap storage buffers as graph views.
 */
function makeExample(symbols: string[], body: string[]): string {
  return [
    `import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';`,
    `import {${symbols.join(', ')}} from '${IMPORT_PATH}';`,
    '',
    `const graph = new GPUCommandGraph(device, {id: 'demo'});`,
    '// output(id, format, length) and csr(rows, capacity): caller-owned storage views',
    ...body,
    'const encoder = device.createCommandEncoder();',
    '(await graph.compileAsync()).encode(encoder, {parameters: undefined});',
    'device.submit(encoder.finish());'
  ].join('\n');
}

const REGRESSION_INPUTS = [
  `const predictors = graph.importGPUData<'float32'>('predictors', predictorData); // row-major, no intercept`,
  `const response = graph.importGPUData<'float32'>('response', responseData);`
];

const STATUS_OUTPUT = output(
  'status.code',
  "GraphDataView<'uint32'>",
  'Solver code; 0 on success.'
);

export const entries: CatalogEntry[] = [
  // ---------------------------------------------------------------- weights producers
  {
    id: 'contiguity-weights',
    symbol: 'GPUContiguityWeights',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Queen and rook contiguity',
    summary: 'Links polygons that touch, as sparse spatial weights.',
    explanation:
      'The starting point for most areal statistics. `queen` joins polygons sharing a vertex; `rook` needs a shared edge. Multipart features and holes count, one row per feature.',
    dataTypes: {input: ['polygons'], output: ['spatial-weights']},
    useCases: ['relate', 'index'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      param(
        'criterion',
        'GPUContiguityCriterion',
        "'queen' shares a vertex; 'rook' shares an edge.",
        {
          required: true
        }
      ),
      param('polygons', 'GPUPolygonGeometryPort', 'Polygons in the four-level layout.', {
        required: true
      }),
      param('output', 'GPUSpatialWeightsOutput', 'Caller-owned CSR with an overflow flag.', {
        required: true
      }),
      param('snapTolerance', 'number | undefined', 'Vertex snap tolerance in coordinate units.', {
        default: '0'
      }),
      param('pairCapacity', 'number | undefined', 'Capacity of the intermediate pair list.')
    ],
    outputs: [
      output('offsets', "GraphDataView<'uint32'>", 'CSR row offsets, featureCount + 1.'),
      output('neighbors', "GraphDataView<'uint32'>", 'Neighbor feature ids, ascending per row.'),
      output('weights', "GraphDataView<'float32'>", 'Binary weights, 1 per link.'),
      output('status.overflow', "GraphDataView<'uint32'>", '1 if capacity was too small.')
    ],
    example: makeExample(
      ['GPUContiguityWeights'],
      [
        'const weights = csr(featureCount, 8 * featureCount);',
        'graph.add(',
        '  new GPUContiguityWeights({',
        "    criterion: 'queen',",
        '    polygons: {',
        "      kind: 'polygons',",
        '      positions, ringOffsets, polygonOffsets, featureOffsets // graph views',
        '    },',
        '    output: weights',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'polygons', seed: 211, caption: 'Tracts that share a vertex become neighbors.'},
    related: [
      'fuzzy-contiguity',
      'lattice-weights',
      'spatial-weights-transform',
      'island-attachment'
    ],
    lensScenario: 'gpu-spatial-analysis.contiguity-weights',
    reference: 'api/gpu-spatial-analysis/operators-weights',
    maturity: 'stable'
  },
  {
    id: 'fuzzy-contiguity',
    symbol: 'GPUFuzzyContiguity',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Fuzzy contiguity',
    summary: 'Links polygons whose boundaries come within a distance, forgiving sliver gaps.',
    explanation:
      'Best when digitized boundaries do not quite touch. Two polygons are neighbors when their minimum separation is within `tolerance`. Handles multipart geometry and holes.',
    dataTypes: {input: ['polygons'], output: ['spatial-weights']},
    useCases: ['relate'],
    domains: ['spatial-statistics', 'cartography'],
    params: [
      param(
        'polygons',
        "Omit<GPUPolygonGeometryPort, 'positions'> & {positions: GraphDataView<'float32x2'>}",
        'Polygons in the four-level layout.',
        {required: true}
      ),
      param('tolerance', 'number', 'Planar minimum polygon separation.', {required: true}),
      param(
        'output',
        'GPUSpatialWeightsPort & {spatialContext?: GPUSpatialContext}',
        'Caller-owned CSR weights.',
        {required: true}
      )
    ],
    outputs: [
      output('offsets', "GraphDataView<'uint32'>", 'CSR row offsets.'),
      output('neighbors', "GraphDataView<'uint32'>", 'Neighbor polygon ids.'),
      output('weights', "GraphDataView<'float32'>", 'Binary weights.'),
      {
        name: 'status',
        type: 'GPUBoundedResultStatusPort',
        description: 'GPU status words; zero means complete.'
      },
      {
        name: 'featureStatus',
        type: 'GraphDataView<"uint32">',
        description: 'Per feature: 0 valid, 1 empty, 2 malformed.'
      },
      {
        name: 'added',
        type: 'GraphDataView<"uint32">',
        description: '1 for edges added by this operation, 0 for copied.'
      }
    ],
    example: makeExample(
      ['GPUFuzzyContiguity'],
      [
        'graph.add(',
        '  new GPUFuzzyContiguity({',
        "    polygons: {kind: 'polygons', positions, ringOffsets, polygonOffsets, featureOffsets},",
        '    tolerance: 0.5, // same units as positions',
        '    output: csr(featureCount, 8 * featureCount)',
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'spatial-weights',
      seed: 212,
      caption: 'Near-misses within tolerance still link.'
    },
    related: ['contiguity-weights', 'island-attachment'],
    lensScenario: 'gpu-spatial-analysis.fuzzy-contiguity',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  },
  {
    id: 'lattice-weights',
    symbol: 'GPULatticeWeights',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Lattice weights',
    summary: 'Connects the cells of a regular grid to their rook or queen neighbors.',
    explanation:
      'The cheap way to get weights for rasters and grid cells; the PySAL `lat2W` equivalent. `rook` is Manhattan distance, `queen` is Chebyshev distance, up to `radius` cells. A mask removes cells.',
    dataTypes: {input: ['raster'], output: ['spatial-weights']},
    useCases: ['relate', 'index'],
    domains: ['spatial-statistics', 'remote-sensing'],
    params: [
      param('width', 'number', 'Lattice columns.', {required: true}),
      param('height', 'number', 'Lattice rows.', {required: true}),
      param('criterion', 'GPULatticeCriterion', "'rook' or 'queen' neighborhood.", {
        required: true
      }),
      param('radius', 'number | undefined', 'Neighborhood radius in cells, 1 to 32.', {
        default: '1'
      }),
      param(
        'mask',
        "GraphDataView<'uint32'> | undefined",
        'Per-cell selection; zero removes the cell.'
      ),
      param(
        'cellSize',
        'readonly [number, number] | undefined',
        'Cell size; also writes distances.'
      ),
      param('output', 'GPUSpatialWeightsOutput', 'Caller-owned CSR, width * height + 1 offsets.', {
        required: true
      })
    ],
    outputs: [
      output('offsets', "GraphDataView<'uint32'>", 'CSR row offsets.'),
      output('neighbors', "GraphDataView<'uint32'>", 'Neighbor cell ids, y * width + x.'),
      output('weights', "GraphDataView<'float32'>", 'Binary weights.'),
      {
        name: 'status',
        type: 'GPUBoundedResultStatusPort',
        description: 'GPU status words; zero means complete.'
      }
    ],
    example: makeExample(
      ['GPULatticeWeights'],
      [
        'graph.add(',
        '  new GPULatticeWeights({',
        '    width: 32,',
        '    height: 32,',
        "    criterion: 'queen',",
        '    output: csr(32 * 32, 8 * 32 * 32)',
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'raster',
      seed: 213,
      caption: 'Eight queen neighbors inside, fewer at the edge.'
    },
    related: ['contiguity-weights', 'spatial-weights-transform', 'spatial-lag'],
    lensScenario: 'gpu-spatial-analysis.lattice-weights',
    reference: 'api/gpu-spatial-analysis/operators-weights',
    maturity: 'stable'
  },
  {
    id: 'neighbor-search',
    symbol: 'GPUNeighborSearch',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Nearest neighbors and distance bands',
    summary: 'Exact k-nearest or within-radius neighbors between points, as spatial weights.',
    explanation:
      'The weights builder for point data and the PySAL `KNN` and `DistanceBand` equivalent. Search runs on a GPU grid; weights can be binary, inverse distance or a kernel. Pass `queryPositions` for a cross join.',
    dataTypes: {input: ['points'], output: ['spatial-weights']},
    useCases: ['index', 'relate'],
    domains: ['spatial-statistics', 'data-engineering'],
    params: [
      param('mode', "'knn' | 'radius'", 'k nearest targets, or all within the radius.', {
        required: true
      }),
      param('positions', "GraphDataView<'float32x2'>", 'Target points.', {required: true}),
      param('neighborCount', 'number | undefined', 'Neighbors per query in knn mode, 1 to 256.'),
      param(
        'gridSize',
        'readonly [number, number]',
        'Maximum columns and rows of the search lattice.',
        {required: true}
      ),
      param(
        'parameters',
        "GraphDataView<'float32'>",
        'Per-frame bounds, radius and weight function.',
        {required: true}
      ),
      param(
        'queryPositions',
        "GraphDataView<'float32x2'> | undefined",
        'Separate query points (cross join).'
      ),
      param('rowStandardize', 'boolean | undefined', 'Row-standardize the weights while writing.'),
      param('output', 'GPUNeighborSearchOutput', 'Caller-owned CSR with an overflow flag.', {
        required: true
      })
    ],
    outputs: [
      output('offsets', "GraphDataView<'uint32'>", 'CSR row offsets, queryRows + 1.'),
      output('neighbors', "GraphDataView<'uint32'>", 'Neighbor ids, ascending per row.'),
      output('weights', "GraphDataView<'float32'>", 'Weights per link.'),
      output('distances', "GraphDataView<'float32'> | undefined", 'Optional link distances.'),
      {
        name: 'status',
        type: 'GPUBoundedResultStatusPort',
        description: 'GPU status words; zero means complete.'
      }
    ],
    example: makeExample(
      ['GPUNeighborSearch', 'GPUParameterBuffer', 'getGPUNeighborSearchParameterValues'],
      [
        "const parameters = new GPUParameterBuffer(device, {id: 'search', format: 'float32', length: 12});",
        'parameters.write(getGPUNeighborSearchParameterValues({bounds: [0, 0, 100, 100]}));',
        'graph.add(',
        '  new GPUNeighborSearch({',
        "    mode: 'knn',",
        '    neighborCount: 4,',
        "    positions: graph.importGPUData('positions', positionData),",
        '    gridSize: [64, 64],',
        '    parameters: parameters.importToGraph(graph),',
        '    output: csr(pointCount, 4 * pointCount)',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'point-clusters', seed: 214, caption: 'Each point links to its four nearest.'},
    related: ['spatial-weights-transform', 'contiguity-weights', 'neighborhood-summary'],
    lensScenario: 'gpu-spatial-analysis.neighbor-search',
    reference: 'api/gpu-spatial-analysis/operators-weights',
    maturity: 'stable'
  },
  {
    id: 'island-attachment',
    symbol: 'GPUIslandAttachment',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Island attachment',
    summary: 'Gives isolated rows a link to their nearest connected neighbor.',
    explanation:
      'Contiguity leaves islands without neighbors, which many models cannot use. Each empty row links to the nearest finite point representative, ties to the lowest id. Original rows keep their weights.',
    dataTypes: {input: ['spatial-weights', 'points'], output: ['spatial-weights']},
    useCases: ['relate', 'transform'],
    domains: ['spatial-statistics'],
    params: [
      param('weights', 'GPUSpatialWeights', 'Weights with empty rows.', {required: true}),
      param('positions', "GraphDataView<'float32x2'>", 'Planar point representative per row.', {
        required: true
      }),
      param('reciprocal', 'boolean | undefined', 'Also add the reverse edge.', {default: 'true'}),
      param(
        'output',
        'GPUSpatialWeightsPort & {spatialContext?: GPUSpatialContext}',
        'Caller-owned CSR weights.',
        {required: true}
      )
    ],
    outputs: [
      output('offsets', "GraphDataView<'uint32'>", 'CSR row offsets.'),
      output('neighbors', "GraphDataView<'uint32'>", 'Neighbor ids including attachments.'),
      output('weights', "GraphDataView<'float32'>", 'Original weights plus binary attachments.'),
      {
        name: 'status',
        type: 'GPUBoundedResultStatusPort',
        description: 'GPU status words; zero means complete.'
      },
      {
        name: 'attachment',
        type: 'GraphDataView<"uint32">',
        description: 'Selected nearest row, or 0xffffffff when none.'
      },
      {
        name: 'added',
        type: 'GraphDataView<"uint32">',
        description: '1 for edges added by this operation, 0 for copied.'
      }
    ],
    example: makeExample(
      ['GPUIslandAttachment'],
      [
        'graph.add(',
        '  new GPUIslandAttachment({',
        '    weights, // contiguity weights with islands',
        "    positions: graph.importGPUData('centroids', centroidData),",
        '    output: csr(featureCount, 10 * featureCount)',
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'spatial-weights',
      seed: 215,
      caption: 'A stray tract links to its nearest neighbor.'
    },
    related: ['contiguity-weights', 'spatial-weights-summary'],
    lensScenario: 'gpu-spatial-analysis.island-attachment',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- weights operations
  {
    id: 'spatial-weights-transform',
    symbol: 'GPUSpatialWeightsTransform',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Weights transforms',
    summary: 'Row-standardizes, binarizes, averages reciprocal weights or applies a kernel.',
    explanation:
      'Do this once after building weights and before any statistic. `row` makes each row sum to 1; `kernel` applies a distance profile at a bandwidth, fixed or adaptive. Results carry the transform state, so consumers know what they got.',
    dataTypes: {input: ['spatial-weights'], output: ['spatial-weights']},
    useCases: ['transform', 'relate'],
    domains: ['spatial-statistics'],
    params: [
      param(
        'operation',
        'GPUSpatialWeightsTransformOperation',
        'row, binary, kernel, average-reciprocal, double or variance.',
        {required: true}
      ),
      param('weights', 'GPUSpatialWeights', 'Weights to transform.', {required: true}),
      param(
        'output',
        "GraphDataView<'float32'> | undefined",
        'Destination for the new weights, one per slot.'
      ),
      param(
        'kernel',
        'GPUSpatialWeightsKernel | undefined',
        'Kernel profile for the kernel operation.'
      ),
      param('bandwidth', "number | 'adaptive' | undefined", 'Bandwidth of the kernel operation.'),
      param('doubleSum', "'one' | 'rows' | undefined", 'Total of the double transform.', {
        default: "'one'"
      }),
      param('spatialContext', 'GPUSpatialContext | undefined', 'Context of a numeric bandwidth.')
    ],
    outputs: [
      output(
        'weights',
        "GraphDataView<'float32'>",
        'Transformed weights, same pattern as the input.'
      ),
      output(
        'outputWeights',
        'GPUSpatialWeights',
        'Input CSR with the new weights and transform state.'
      )
    ],
    example: makeExample(
      ['GPUSpatialWeightsTransform'],
      [
        'graph.add(',
        '  new GPUSpatialWeightsTransform({',
        "    operation: 'row',",
        '    weights, // from GPUContiguityWeights',
        "    output: output('row-weights', 'float32', slotCount)",
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'spatial-weights', seed: 216, caption: 'Each row of weights now sums to one.'},
    related: [
      'contiguity-weights',
      'spatial-lag',
      'spatial-weights-transpose',
      'chunked-weights-transform'
    ],
    lensScenario: 'gpu-spatial-analysis.spatial-weights-transform',
    reference: 'api/gpu-spatial-analysis/operators-weights',
    maturity: 'stable'
  },
  {
    id: 'spatial-weights-algebra',
    symbol: 'GPUSpatialWeightsAlgebra',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Weights algebra',
    summary:
      'Combines spatial weights: union, intersection, higher order, self weights, subgraphs, blocks.',
    explanation:
      'Build the neighborhood you want from the ones you have, such as queen union kNN, or neighbors of neighbors. Mirrors PySAL `w_union`, `higher_order`, `w_subset` and `block_weights`. Output is a fresh CSR.',
    dataTypes: {input: ['spatial-weights'], output: ['spatial-weights']},
    useCases: ['relate', 'transform'],
    domains: ['spatial-statistics'],
    params: [
      param(
        'operation',
        "'intersection' | 'union' | 'difference' | 'symmetric-difference' | 'higherOrder' | 'selfWeight' | 'subgraph' | 'block'",
        'What to compute.',
        {required: true}
      ),
      param(
        'output',
        'GPUSpatialWeightsOutput',
        'Destination CSR with capacity and overflow flag.',
        {required: true}
      ),
      param('left', 'GPUSpatialWeights', 'Left operand of a set operation.'),
      param('right', 'GPUSpatialWeights', 'Right operand, same row count.'),
      param(
        'weightRule',
        'GPUSpatialWeightsCombineRule | undefined',
        'How to weight shared neighbors.'
      ),
      param('weights', 'GPUSpatialWeights', 'Base weights for higher order, self, subgraph.'),
      param('order', 'number', 'Step count of a higher-order operation.'),
      param('cumulative', 'boolean | undefined', 'Include all lower orders too.')
    ],
    outputs: [
      output('offsets', "GraphDataView<'uint32'>", 'CSR row offsets of the result.'),
      output('neighbors', "GraphDataView<'uint32'>", 'Neighbor ids, sorted and unique per row.'),
      output('weights', "GraphDataView<'float32'>", 'Combined weights.'),
      output('status.requiredCount', "GraphDataView<'uint32'>", 'Slots the result needs.')
    ],
    example: makeExample(
      ['GPUSpatialWeightsAlgebra'],
      [
        'graph.add(',
        '  new GPUSpatialWeightsAlgebra({',
        "    operation: 'union',",
        '    left: queen,',
        '    right: knn,',
        "    weightRule: 'binary',",
        '    output: csr(featureCount, 12 * featureCount)',
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'spatial-weights',
      seed: 217,
      caption: 'Queen links joined with nearest-neighbor links.'
    },
    related: ['spatial-weights-transform', 'chunked-weights-algebra', 'contiguity-weights'],
    lensScenario: 'gpu-spatial-analysis.spatial-weights-algebra',
    reference: 'api/gpu-spatial-analysis/operators-weights',
    maturity: 'stable'
  },
  {
    id: 'spatial-weights-transpose',
    symbol: 'GPUSpatialWeightsTranspose',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Weights transpose',
    summary: 'Flips who points at whom: the sparse transpose of a weights matrix.',
    explanation:
      'Derives incoming links from outgoing ones, or tests symmetry. Works for square and rectangular weights; output rows are sorted. An optional count tells how many slots are asymmetric.',
    dataTypes: {input: ['spatial-weights'], output: ['spatial-weights']},
    useCases: ['transform', 'relate'],
    domains: ['spatial-statistics'],
    params: [
      param('weights', 'GPUSpatialWeights', 'Weights W to transpose.', {required: true}),
      param('output', 'GPUSpatialWeightsTransposeOutput', 'Caller-owned CSR of the transpose.', {
        required: true
      }),
      param('columnCount', 'number | undefined', 'Columns of W (rows of the result).'),
      param('symmetryTolerance', 'number | undefined', 'Absolute tolerance of the symmetry check.')
    ],
    outputs: [
      output('offsets', "GraphDataView<'uint32'>", 'CSR row offsets, columnCount + 1.'),
      output('neighbors', "GraphDataView<'uint32'>", 'Source rows now acting as neighbors.'),
      output('weights', "GraphDataView<'float32'>", 'Transposed weights.'),
      output(
        'asymmetricSlots',
        "GraphDataView<'uint32'> | undefined",
        '0 when W equals its transpose.'
      )
    ],
    example: makeExample(
      ['GPUSpatialWeightsTranspose'],
      [
        'graph.add(',
        '  new GPUSpatialWeightsTranspose({',
        '    weights, // row-standardized W',
        '    output: {',
        '      ...csr(rowCount, slotCount),',
        "      asymmetricSlots: output('asymmetric', 'uint32', 1)",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'spatial-weights', seed: 218, caption: 'Outgoing links become incoming links.'},
    related: ['spatial-weights-transform', 'chunked-weights-transpose', 'spatial-weights-summary'],
    lensScenario: 'gpu-spatial-analysis.spatial-weights-transpose',
    reference: 'api/gpu-spatial-analysis/operators-weights',
    maturity: 'stable'
  },
  {
    id: 'spatial-weights-summary',
    symbol: 'GPUSpatialWeightsSummary',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Weights summary',
    summary:
      'Describes a weights matrix: S0, S1, S2, link counts, asymmetry, isolates and degrees.',
    explanation:
      'A quick health check before modeling: are there islands, is it symmetric, how uneven are the degrees. Matches libpysal `s0`, `s1`, `s2`, `cardinalities` and `islands`. Sums run in a fixed order.',
    dataTypes: {input: ['spatial-weights'], output: ['table']},
    useCases: ['measure', 'summarize'],
    domains: ['spatial-statistics', 'data-engineering'],
    params: [
      param(
        'weights',
        'GPUSpatialWeights | GPUChunkedSpatialWeights',
        'Square weights to summarize.',
        {required: true}
      ),
      param(
        'output',
        'GPUSpatialWeightsSummaryOutput',
        'Caller-owned statistics, counts and degrees.',
        {required: true}
      ),
      param(
        'symmetryTolerance',
        'number | undefined',
        'Tolerance for calling w_ij and w_ji equal.',
        {default: '0'}
      )
    ],
    outputs: [
      output('statistics', "GraphDataView<'float32'>", 'S0, S1 and S2.'),
      output(
        'counts',
        "GraphDataView<'uint32'>",
        'Slots, asymmetric slots, isolates, min and max degree, empty rows.'
      ),
      output(
        'cardinality',
        "GraphDataView<'uint32'> | undefined",
        'Optional neighbor count per row.'
      )
    ],
    example: makeExample(
      ['GPUSpatialWeightsSummary'],
      [
        'graph.add(',
        '  new GPUSpatialWeightsSummary({',
        '    weights,',
        '    output: {',
        "      statistics: output('statistics', 'float32', 3),",
        "      counts: output('counts', 'uint32', 6),",
        "      cardinality: output('cardinality', 'uint32', rowCount)",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'histogram', seed: 219, caption: 'Degree distribution of the neighbor graph.'},
    related: ['spatial-weights-validation', 'island-attachment', 'spatial-weights-transform'],
    lensScenario: 'gpu-spatial-analysis.spatial-weights-summary',
    reference: 'api/gpu-spatial-analysis/operators-weights',
    maturity: 'stable'
  },
  {
    id: 'spatial-weights-validation',
    symbol: 'GPUSpatialWeightsValidation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Weights validation',
    summary: 'Flags rows of uploaded weights that break the CSR invariants.',
    explanation:
      'Run it once on weights built elsewhere, before using them. Checks offsets, neighbor range and order, self links and non-negative finite weights. Flags are bits per row: offsets 1, range 2, order 4, self 8, weight 16.',
    dataTypes: {input: ['spatial-weights'], output: ['table']},
    useCases: ['model', 'measure'],
    domains: ['data-engineering', 'spatial-statistics'],
    params: [
      param('weights', 'Weights', 'Square weights to check.', {required: true}),
      param(
        'output',
        'GPUSpatialWeightsValidationOutput',
        'Caller-owned flags and invalid count.',
        {required: true}
      )
    ],
    outputs: [
      output('flags', "GraphDataView<'uint32'> | undefined", 'Violation bitmask per row.'),
      output('status.invalidCount', "GraphDataView<'uint32'>", 'Rows with at least one violation.')
    ],
    example: makeExample(
      ['GPUSpatialWeightsValidation'],
      [
        'graph.add(',
        '  new GPUSpatialWeightsValidation({',
        '    weights: uploaded,',
        '    output: {',
        "      flags: output('flags', 'uint32', rowCount),",
        "      status: {invalidCount: output('invalid', 'uint32', 1)}",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'spatial-weights', seed: 220, caption: 'Zero flagged rows means safe to use.'},
    related: ['spatial-weights-summary', 'chunked-weights-validation'],
    lensScenario: 'gpu-spatial-analysis.spatial-weights-validation',
    reference: 'api/gpu-spatial-analysis/operators-weights',
    maturity: 'stable'
  },
  {
    id: 'spatial-lag',
    symbol: 'GPUSpatialLag',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial lag',
    summary:
      'Replaces each value with the weighted sum of its neighbors, the building block of spatial models.',
    explanation:
      'The lag is `sum_j w_ij y_j` (PySAL `lag_spatial`). With row-standardized weights it is the neighbor average, which smooths local extremes. Lag several columns at once with `columnCount`.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['summarize', 'find-patterns'],
    domains: ['spatial-statistics'],
    params: [
      param('values', 'GPUFloat32OrUint32Values', 'Per-source values, row-major.', {
        required: true
      }),
      param(
        'weights',
        'GPUSpatialWeights | GPUChunkedSpatialWeights',
        'Weights whose neighbor ids index values.',
        {required: true}
      ),
      param('output', "GraphDataView<'float32'>", 'Lag output, rows * columnCount.', {
        required: true
      }),
      param('columnCount', 'number | undefined', 'Value columns lagged at once.', {default: '1'}),
      param('normalize', 'boolean | undefined', 'Divide by the included weight sum.'),
      param('mask', "GraphDataView<'uint32'> | undefined", 'Zero rows output 0 and are skipped.')
    ],
    outputs: [output('output', "GraphDataView<'float32'>", 'Lag per row, rows * columnCount.')],
    example: makeExample(
      ['GPUSpatialLag'],
      [
        'graph.add(',
        '  new GPUSpatialLag({',
        '    weights, // row-standardized',
        "    values: graph.importGPUData('poverty', povertyData),",
        "    output: output('lag', 'float32', rowCount)",
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'spatial-weights',
      seed: 221,
      caption: 'Values smoothed through neighbor links.'
    },
    related: [
      'spatial-weights-transform',
      'neighborhood-summary',
      'spatial-lag-maximum-likelihood'
    ],
    lensScenario: 'gpu-spatial-analysis.spatial-lag',
    reference: 'api/gpu-spatial-analysis/operators-weights',
    maturity: 'stable'
  },
  {
    id: 'neighborhood-summary',
    symbol: 'GPUNeighborhoodSummary',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Neighborhood summary',
    summary:
      'Per-row count, sum, mean, min, max, deviation, median, mode and entropy over neighbors.',
    explanation:
      'The generalized lag: describe what surrounds each feature, as momepy `describe` does. Numeric statistics go to a table; categories give the modal class and Shannon entropy (diversity). `includeFocal` adds the row itself.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['summarize', 'measure'],
    domains: ['urban', 'spatial-statistics'],
    params: [
      param('weights', 'GPUSpatialWeights', 'Square weights indexing the value rows.', {
        required: true
      }),
      param('values', 'GPUFloat32OrUint32Values | undefined', 'Numeric values, one per row.'),
      param(
        'categories',
        "GraphDataView<'uint32'> | undefined",
        'Categorical column for mode and entropy.'
      ),
      param(
        'statistics',
        'readonly GPUNeighborhoodSummaryStatistic[] | undefined',
        'Numeric statistics, in table column order.'
      ),
      param('includeFocal', 'boolean | undefined', 'Add the focal row to its neighborhood.'),
      param('maximumNeighbors', 'number | undefined', 'Median buffer capacity, 1 to 64.'),
      param('output', 'GPUNeighborhoodSummaryOutput', 'Caller-owned table, modes and entropy.', {
        required: true
      })
    ],
    outputs: [
      output(
        'table',
        "GraphDataView<'float32'> | undefined",
        'rows x statistics, one column each.'
      ),
      output('modes', "GraphDataView<'uint32'> | undefined", 'Most frequent category per row.'),
      output('entropy', "GraphDataView<'float32'> | undefined", 'Category diversity per row.')
    ],
    example: makeExample(
      ['GPUNeighborhoodSummary'],
      [
        'graph.add(',
        '  new GPUNeighborhoodSummary({',
        '    weights,',
        "    values: graph.importGPUData('height', heightData),",
        "    statistics: ['mean', 'max', 'standardDeviation'],",
        '    includeFocal: true,',
        "    output: {table: output('summary', 'float32', 3 * rowCount)}",
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'cells', seed: 222, caption: 'Each cell described by its surroundings.'},
    related: ['spatial-lag', 'neighbor-search'],
    lensScenario: 'gpu-spatial-analysis.neighborhood-summary',
    reference: 'api/gpu-spatial-analysis/operators-statistics',
    maturity: 'stable'
  },
  {
    id: 'pair-gather',
    symbol: 'GPUPairGather',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Join attributes by pairs',
    summary: 'Turns matched pairs from a spatial join into attribute rows, like GeoPandas sjoin.',
    explanation:
      'A join gives pairs of row indices or dense nearest-neighbor lists; this gathers columns for both sides into aligned outputs. `inner` keeps matches only; `left` keeps every left row, unmatched marked with a sentinel. Final capacity loss and input candidate loss are reported separately.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['relate'],
    domains: ['data-engineering'],
    params: [
      param('how', 'GPUPairGatherHow | undefined', "'inner' or 'left'.", {default: "'inner'"}),
      param('pairs', 'GPUCompactPairPort | undefined', 'Matched pairs of a predicate join.'),
      param(
        'neighbors',
        'GPUPairGatherNeighbors | undefined',
        'Dense neighbor lists of a nearest join.'
      ),
      param('unmatchedLeft', 'GPUCompactOutput | undefined', 'Unmatched left rows, for how left.'),
      param(
        'leftColumns',
        'readonly GPUPairGatherColumn[] | undefined',
        'Left columns, gathered by leftRows.'
      ),
      param(
        'rightColumns',
        'readonly GPUPairGatherColumn[] | undefined',
        'Right columns, gathered by rightRows.'
      ),
      param('output', 'GPUPairGatherOutput', 'Caller-owned bounded row outputs.', {required: true})
    ],
    outputs: [
      output('leftRows', "GraphDataView<'uint32'>", 'Left input row index per output row.'),
      output(
        'rightRows',
        "GraphDataView<'uint32'>",
        'Right input row index, sentinel when unmatched.'
      ),
      output('status.count', "GraphDataView<'uint32'>", 'Number of valid output rows.'),
      output(
        'status.overflow',
        "GraphDataView<'uint32'>",
        'Final gathered or input output-capacity loss.'
      ),
      output(
        'status.candidateOverflow',
        "GraphDataView<'uint32'>",
        'Input candidate loss, separate from final output capacity.'
      )
    ],
    example: makeExample(
      ['GPUPairGather'],
      [
        'graph.add(',
        '  new GPUPairGather({',
        "    how: 'inner',",
        '    pairs, // matched pairs from a spatial join',
        "    rightColumns: [{source: incomeData, output: output('income', 'float32', capacity)}],",
        '    output: {',
        "      leftRows: output('left', 'uint32', capacity),",
        "      rightRows: output('right', 'uint32', capacity),",
        "      status: {count: output('count', 'uint32', 1), overflow: output('overflow', 'uint32', 1), candidateOverflow: output('candidate-overflow', 'uint32', 1)}",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'polygon-overlay',
      seed: 223,
      caption: 'Points inherit attributes of their polygon.'
    },
    related: ['neighbor-search', 'contiguity-weights'],
    lensScenario: 'gpu-spatial-analysis.pair-gather',
    reference: 'api/gpu-spatial-analysis/operators-joins',
    maturity: 'stable'
  },
  {
    id: 'spatial-weights-traces',
    symbol: 'GPUSpatialWeightsTraces',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Weights traces',
    summary: "Traces tr(W^q) and sums 1'W^q1 of a weights matrix for q up to Q.",
    explanation:
      'The ingredients of spatial-lag log-determinants and impacts. `exact` runs one unit probe per row; `stochastic` uses Hutchinson probes for large matrices. Values are scaled by the infinity norm.',
    dataTypes: {input: ['spatial-weights'], output: ['table']},
    useCases: ['model', 'measure'],
    domains: ['spatial-statistics'],
    params: [
      param('weights', 'GPUSpatialWeights | GPUChunkedSpatialWeights', 'Square weights W.', {
        required: true
      }),
      param('maximumPower', 'number', 'Highest power Q, 1 to 256.', {required: true}),
      param('mode', 'GPUSpatialWeightsTracesMode | undefined', "'exact' or 'stochastic'."),
      param('probeCount', 'number | undefined', 'Probes of the stochastic mode.'),
      param('includeGramTraces', 'boolean | undefined', 'Also compute gramTraces.'),
      param('output', 'GPUSpatialWeightsTracesOutput', 'Caller-owned trace outputs.', {
        required: true
      })
    ],
    outputs: [
      output('powerTraces', "GraphDataView<'float32'>", 'tr((W/s)^q) at index q - 1.'),
      output('powerSums', "GraphDataView<'float32'>", "1'(W/s)^q 1 at index q - 1."),
      output(
        'infinityNorm',
        "GraphDataView<'float32'> | undefined",
        'Scale s, the largest absolute row sum.'
      )
    ],
    example: makeExample(
      ['GPUSpatialWeightsTraces'],
      [
        'graph.add(',
        '  new GPUSpatialWeightsTraces({',
        '    weights,',
        '    maximumPower: 30,',
        '    output: {',
        "      powerTraces: output('traces', 'float32', 30),",
        "      powerSums: output('sums', 'float32', 30),",
        "      infinityNorm: output('norm', 'float32', 1)",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'spatial-weights', seed: 224, caption: 'Traces decay with the power of W.'},
    related: [
      'spatial-lag-impacts',
      'spatial-lag-maximum-likelihood',
      'spatial-error-maximum-likelihood'
    ],
    lensScenario: 'gpu-spatial-analysis.spatial-weights-traces',
    reference: 'api/gpu-spatial-analysis/operators-regression',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- chunked weights
  {
    id: 'chunked-weights-algebra',
    symbol: 'GPUChunkedSpatialWeightsAlgebra',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Chunked weights algebra',
    summary: 'Weights algebra for matrices stored in chunks too large for one buffer.',
    explanation:
      'The row-local operations of weights algebra over chunked weights: union, intersection, difference, symmetric difference, self weights and masks. Chunks keep their row ranges; nothing is repacked.',
    dataTypes: {input: ['spatial-weights'], output: ['spatial-weights']},
    useCases: ['relate', 'transform'],
    domains: ['spatial-statistics', 'data-engineering'],
    params: [
      param(
        'operation',
        'SpatialWeightsMergeOperation',
        'union, intersection, difference or symmetric-difference.',
        {required: true}
      ),
      param(
        'output',
        'GPUChunkedSpatialWeightsOutput',
        'Chunked destination with matching row ranges.',
        {required: true}
      ),
      param('left', 'GPUChunkedSpatialWeights', 'Left operand.'),
      param('right', 'GPUChunkedSpatialWeights', 'Right operand.'),
      param(
        'weightRule',
        'GPUSpatialWeightsCombineRule | undefined',
        'Weight rule for shared neighbors.'
      ),
      param('weights', 'GPUChunkedSpatialWeights', 'Base weights for self weights or a mask.'),
      param(
        'selfWeight',
        "number | GraphDataView<'float32'>",
        'Self weight per row or one number.'
      ),
      param('mask', "GraphDataView<'uint32'>", 'Nonzero keeps the row.')
    ],
    outputs: [
      {name: 'chunks', type: 'readonly {}', description: 'Weights chunks with their row ranges.'}
    ],
    example: makeExample(
      ['GPUChunkedSpatialWeightsAlgebra'],
      [
        'graph.add(',
        '  new GPUChunkedSpatialWeightsAlgebra({',
        "    operation: 'union',",
        '    left: chunkedQueen,',
        '    right: chunkedKnn,',
        "    weightRule: 'binary',",
        '    output: chunkedOutput // same row ranges as the inputs',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'spatial-weights', seed: 225, caption: 'Union computed chunk by chunk.'},
    related: ['spatial-weights-algebra', 'chunked-weights-transform'],
    reference: 'api/gpu-spatial-analysis/operators-weights',
    maturity: 'stable'
  },
  {
    id: 'chunked-weights-transform',
    symbol: 'GPUChunkedSpatialWeightsTransform',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Chunked weights transform',
    summary: 'Row-standardizes, binarizes or kernel-weights chunked spatial weights.',
    explanation:
      'The same operations as the weights transform, applied per chunk for large matrices. Pass one destination per chunk, in order. Chunk boundaries are preserved.',
    dataTypes: {input: ['spatial-weights'], output: ['spatial-weights']},
    useCases: ['transform'],
    domains: ['spatial-statistics', 'data-engineering'],
    params: [
      param(
        'operation',
        'GPUSpatialWeightsTransformOperation',
        'row, binary, kernel, average-reciprocal, double or variance.',
        {required: true}
      ),
      param('weights', 'GPUChunkedSpatialWeights', 'Chunked weights to transform.', {
        required: true
      }),
      param(
        'output',
        "readonly (GraphDataView<'float32'> | undefined)[] | undefined",
        'Destination per chunk, same order.'
      ),
      param(
        'kernel',
        'GPUSpatialWeightsKernel | undefined',
        'Kernel profile for the kernel operation.'
      ),
      param('bandwidth', "number | 'adaptive' | undefined", 'Bandwidth of the kernel operation.'),
      param('doubleSum', "'one' | 'rows' | undefined", 'Total of the double transform.'),
      param('spatialContext', 'GPUSpatialContext | undefined', 'Context of a numeric bandwidth.')
    ],
    outputs: [
      output('output', "readonly GraphDataView<'float32'>[]", 'Transformed weights per chunk.')
    ],
    example: makeExample(
      ['GPUChunkedSpatialWeightsTransform'],
      [
        'graph.add(',
        '  new GPUChunkedSpatialWeightsTransform({',
        "    operation: 'row',",
        '    weights: chunkedWeights,',
        '    output: chunkOutputs // one view per chunk',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'spatial-weights', seed: 226, caption: 'Row sums of one, chunk by chunk.'},
    related: ['spatial-weights-transform', 'chunked-weights-algebra'],
    reference: 'api/gpu-spatial-analysis/operators-weights',
    maturity: 'stable'
  },
  {
    id: 'chunked-weights-transpose',
    symbol: 'GPUChunkedSpatialWeightsTranspose',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Chunked weights transpose',
    summary: 'Transposes chunked weights into one whole CSR.',
    explanation:
      'Best when the source is too large for a single buffer but the transpose fits. Packing is explicit: the result is one caller-owned CSR. An optional check counts asymmetric slots.',
    dataTypes: {input: ['spatial-weights'], output: ['spatial-weights']},
    useCases: ['transform'],
    domains: ['spatial-statistics', 'data-engineering'],
    params: [
      param('weights', 'GPUChunkedSpatialWeights', 'Chunked weights W.', {required: true}),
      param('output', 'GPUSpatialWeightsTransposeOutput', 'Caller-owned CSR of the transpose.', {
        required: true
      }),
      param('columnCount', 'number | undefined', 'Columns of W.'),
      param('symmetryTolerance', 'number | undefined', 'Absolute tolerance of the symmetry check.')
    ],
    outputs: [
      output('offsets', "GraphDataView<'uint32'>", 'CSR row offsets of the transpose.'),
      output('neighbors', "GraphDataView<'uint32'>", 'Transposed neighbor ids.'),
      output('weights', "GraphDataView<'float32'>", 'Transposed weights.')
    ],
    example: makeExample(
      ['GPUChunkedSpatialWeightsTranspose'],
      [
        'graph.add(',
        '  new GPUChunkedSpatialWeightsTranspose({',
        '    weights: chunkedWeights,',
        '    output: csr(rowCount, slotCount) // one whole CSR',
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'spatial-weights',
      seed: 227,
      caption: 'Chunks merge into one transposed matrix.'
    },
    related: ['spatial-weights-transpose', 'chunked-weights-transform'],
    reference: 'api/gpu-spatial-analysis/operators-weights',
    maturity: 'stable'
  },
  {
    id: 'chunked-weights-validation',
    symbol: 'GPUChunkedSpatialWeightsValidation',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Chunked weights validation',
    summary: 'Checks the CSR invariants of chunked weights, flagging bad rows globally.',
    explanation:
      'Validation for weights stored in several chunks. Flags cover all rows in global order; the invalid count totals every chunk. Run once after upload.',
    dataTypes: {input: ['spatial-weights'], output: ['table']},
    useCases: ['model', 'measure'],
    domains: ['data-engineering', 'spatial-statistics'],
    params: [
      param('weights', 'GPUChunkedSpatialWeights', 'Square chunked weights to check.', {
        required: true
      }),
      param(
        'output',
        'GPUSpatialWeightsValidationOutput',
        'Caller-owned flags and invalid count.',
        {required: true}
      )
    ],
    outputs: [
      output('flags', "GraphDataView<'uint32'> | undefined", 'Violation bitmask per global row.'),
      output('status.invalidCount', "GraphDataView<'uint32'>", 'Rows with at least one violation.')
    ],
    example: makeExample(
      ['GPUChunkedSpatialWeightsValidation'],
      [
        'graph.add(',
        '  new GPUChunkedSpatialWeightsValidation({',
        '    weights: chunkedWeights,',
        '    output: {',
        "      flags: output('flags', 'uint32', rowCount),",
        "      status: {invalidCount: output('invalid', 'uint32', 1)}",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'spatial-weights', seed: 228, caption: 'One pass over every chunk.'},
    related: ['spatial-weights-validation', 'chunked-weights-transform'],
    reference: 'api/gpu-spatial-analysis/operators-weights',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- global regression
  {
    id: 'ordinary-least-squares',
    symbol: 'GPUOrdinaryLeastSquares',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Ordinary least squares',
    summary: 'Linear regression with an intercept and a full report, fit on the GPU.',
    explanation:
      'The baseline model, and the first step before spatial diagnostics. Gives coefficients, standard errors, t statistics, R2 and heteroskedasticity tests. Rows with non-finite values or a zero mask are excluded.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      param('predictors', "GraphDataView<'float32'>", 'Row-major, row * predictorCount + column.', {
        required: true
      }),
      param('response', "GraphDataView<'float32'>", 'Response, one value per row.', {
        required: true
      }),
      param('predictorCount', 'number', 'Predictor columns, 1 to 15.', {required: true}),
      param('mask', "GraphDataView<'uint32'> | undefined", 'Zero excludes the row.'),
      param('parameters', "GraphDataView<'float32'> | undefined", 'Per-frame ridge lambda.'),
      param('tileRowCount', 'number | undefined', 'Rows reduced per invocation.'),
      param('output', 'GPUOrdinaryLeastSquaresOutput', 'Caller-owned outputs.', {required: true})
    ],
    outputs: [
      output(
        'coefficients',
        "GraphDataView<'float32'>",
        'Intercept first, then one per predictor.'
      ),
      output('standardErrors', "GraphDataView<'float32'>", 'Standard error per coefficient.'),
      output('tStatistics', "GraphDataView<'float32'>", 't statistic per coefficient.'),
      output('summary', "GraphDataView<'float32'>", 'R2, adjusted R2, sigma squared, tests.'),
      output('residuals', "GraphDataView<'float32'> | undefined", 'y - fitted per row.'),
      {
        name: 'status',
        type: 'Pick<GPUSolverStatusPort, "code">',
        description: 'GPU status words; zero means complete.'
      }
    ],
    example: makeExample(
      ['GPUOrdinaryLeastSquares'],
      [
        ...REGRESSION_INPUTS,
        'graph.add(',
        '  new GPUOrdinaryLeastSquares({',
        '    predictors,',
        '    response,',
        '    predictorCount: 2,',
        '    output: {',
        "      coefficients: output('beta', 'float32', 3),",
        "      standardErrors: output('se', 'float32', 3),",
        "      tStatistics: output('t', 'float32', 3),",
        "      summary: output('summary', 'float32', 22),",
        "      status: {code: output('status', 'uint32', 1)}",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'regression', seed: 231, caption: 'A best-fit line through the cloud.'},
    related: [
      'spatial-regression-diagnostics',
      'spatial-lag-maximum-likelihood',
      'geographically-weighted-regression'
    ],
    lensScenario: 'gpu-spatial-analysis.ordinary-least-squares',
    reference: 'api/gpu-spatial-analysis/operators-regression',
    maturity: 'stable'
  },
  {
    id: 'spatial-regression-diagnostics',
    symbol: 'GPUSpatialRegressionDiagnostics',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial regression diagnostics',
    summary: 'Tests OLS residuals for spatial lag and error dependence, including robust LM tests.',
    explanation:
      "Run after OLS to decide between lag and error models. Computes LM-lag, LM-error, their robust versions, SARMA and Moran's I of residuals, as spreg does. Pass the OLS residuals and the same mask.",
    dataTypes: {input: ['table', 'spatial-weights'], output: ['table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics'],
    params: [
      param('weights', 'GPUSpatialWeights', 'Square weights over the regression rows.', {
        required: true
      }),
      param('predictors', "GraphDataView<'float32'>", 'Predictors used in the OLS fit.', {
        required: true
      }),
      param('response', "GraphDataView<'float32'>", 'Response used in the OLS fit.', {
        required: true
      }),
      param('residuals', "GraphDataView<'float32'>", 'Residuals of the OLS fit.', {required: true}),
      param('predictorCount', 'number', 'Predictor columns, 1 to 15.', {required: true}),
      param('mask', "GraphDataView<'uint32'> | undefined", 'Same mask as the OLS fit.'),
      param('restandardizeRows', 'boolean | undefined', 'Re-standardize rows after masking.'),
      param('output', 'GPUSpatialRegressionDiagnosticsOutput', 'Caller-owned outputs.', {
        required: true
      })
    ],
    outputs: [
      output(
        'tests',
        "GraphDataView<'float32'>",
        'Six tests: statistic, degrees of freedom, p-value.'
      ),
      output('summary', "GraphDataView<'float32'>", "n, sigma squared, traces, S0, Moran's I."),
      output('status.code', "GraphDataView<'uint32'>", 'Diagnostics status code.')
    ],
    example: makeExample(
      ['GPUSpatialRegressionDiagnostics'],
      [
        'graph.add(',
        '  new GPUSpatialRegressionDiagnostics({',
        '    weights, predictors, response,',
        '    residuals, // from GPUOrdinaryLeastSquares',
        '    predictorCount: 2,',
        '    restandardizeRows: true,',
        '    output: {',
        "      tests: output('tests', 'float32', 18),",
        "      summary: output('summary', 'float32', 13),",
        "      status: {code: output('status', 'uint32', 1)}",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'regression', seed: 232, caption: 'Which dependence do the residuals show?'},
    related: [
      'ordinary-least-squares',
      'spatial-lag-maximum-likelihood',
      'spatial-error-maximum-likelihood'
    ],
    lensScenario: 'gpu-spatial-analysis.spatial-regression-diagnostics',
    reference: 'api/gpu-spatial-analysis/operators-regression',
    maturity: 'stable'
  },
  {
    id: 'probit-regression',
    symbol: 'GPUProbitRegression',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Probit regression',
    summary: 'Binary probit by Fisher scoring, with optional spatial residual diagnostics.',
    explanation:
      'Models a yes/no outcome from predictors. Iterates bounded Fisher scoring with likelihood backtracking. Optionally tests the fit for spatial dependence under the weights.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      param('predictors', "GraphDataView<'float32'>", 'Row-major predictors.', {required: true}),
      param('predictorCount', 'number', 'Predictor columns.', {required: true}),
      param('response', "GraphDataView<'float32'>", '0 or 1 per row.', {required: true}),
      param('mask', "GraphDataView<'uint32'> | undefined", 'Zero excludes the row.'),
      param('maximumIterations', 'number | undefined', 'Fisher scoring iteration cap.'),
      param('tolerance', 'number | undefined', 'Convergence tolerance.'),
      param('spatialDiagnostics', '{weights, output}}', 'Optional dependence tests.'),
      param('output', '{coefficients, standardErrors}', 'Caller-owned results.', {required: true})
    ],
    outputs: [
      output('coefficients', "GraphDataView<'float32'>", 'Intercept first, then predictors.'),
      output('standardErrors', "GraphDataView<'float32'>", 'Standard error per coefficient.'),
      {name: 'fitted', type: 'GraphDataView<"float32">', description: 'Fitted response per row.'},
      {
        name: 'marginalEffects',
        type: 'GraphDataView<"float32">',
        description: 'Average marginal effect per predictor.'
      },
      {
        name: 'summary',
        type: 'GraphDataView<"float32">',
        description: 'Fit summary: log likelihood and iteration counts.'
      },
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: 'GPU status words; zero means complete.'
      }
    ],
    example: makeExample(
      ['GPUProbitRegression'],
      [
        ...REGRESSION_INPUTS,
        'graph.add(',
        '  new GPUProbitRegression({',
        '    predictors,',
        '    predictorCount: 2,',
        '    response, // 0 or 1',
        '    output: {',
        "      coefficients: output('beta', 'float32', 3),",
        "      standardErrors: output('se', 'float32', 3),",
        "      fitted: output('fitted', 'float32', rowCount),",
        "      marginalEffects: output('effects', 'float32', 2),",
        "      summary: output('summary', 'float32', 4),",
        "      status: output('status', 'uint32', 1)",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'classification', seed: 233, caption: 'Probability rises across the boundary.'},
    related: ['ordinary-least-squares', 'spatial-regression-diagnostics'],
    lensScenario: 'gpu-spatial-analysis.probit-regression',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  },
  {
    id: 'seemingly-unrelated-regression',
    symbol: 'GPUSeeminglyUnrelatedRegression',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Seemingly unrelated regression',
    summary: 'Fits several equations jointly, using correlation between their errors.',
    explanation:
      'Best when related outcomes are modeled together. One-step feasible SUR: equation OLS, a joint residual covariance, then system GLS. An optional `spatial` block adds Kelejian-Prucha spatial errors.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      param('equationCount', 'number', 'Number of equations.', {required: true}),
      param('predictorCount', 'number', 'Predictors per equation.', {required: true}),
      param('rowCount', 'number', 'Observations per equation.', {required: true}),
      param(
        'predictors',
        "GraphDataView<'float32'>",
        'Equation-major, then row-major predictors.',
        {required: true}
      ),
      param('responses', "GraphDataView<'float32'>", 'Equation-major responses.', {required: true}),
      param('mask', "GraphDataView<'uint32'> | undefined", 'Common row mask.'),
      param('spatial', '{weights, output} | undefined', 'Spatial-error SUR option.'),
      param('output', '{coefficients, covariance}', 'Caller-owned results.', {required: true})
    ],
    outputs: [
      output('coefficients', "GraphDataView<'float32'>", 'Coefficients for every equation.'),
      output('covariance', "GraphDataView<'float32'>", 'System coefficient covariance.'),
      {
        name: 'residualCovariance',
        type: 'GraphDataView<"float32">',
        description: 'OLS residual cross-equation covariance for GLS.'
      },
      {name: 'residuals', type: 'GraphDataView<"float32">', description: 'Residual per row.'},
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: 'GPU status words; zero means complete.'
      }
    ],
    example: makeExample(
      ['GPUSeeminglyUnrelatedRegression'],
      [
        'graph.add(',
        '  new GPUSeeminglyUnrelatedRegression({',
        '    equationCount: 2,',
        '    predictorCount: 2,',
        '    rowCount,',
        "    predictors: graph.importGPUData('x', predictorData), // equation-major",
        "    responses: graph.importGPUData('y', responseData),",
        '    output // see the reference for the output fields',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'regression', seed: 234, caption: 'Two fits sharing correlated errors.'},
    related: ['ordinary-least-squares', 'spatial-error-gm'],
    lensScenario: 'gpu-spatial-analysis.seemingly-unrelated-regression',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  },
  {
    id: 'instrumental-variables-covariance',
    symbol: 'GPUInstrumentalVariablesCovariance',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'IV robust covariance',
    summary: 'Robust (HC0, HC1 or spatial HAC) covariance for instrumental-variable fits.',
    explanation:
      'Corrects standard errors of 2SLS-style models for heteroskedasticity or spatial correlation. Provide the structural design, instruments and the residuals of the same fitted rows. `HAC` needs a kernel weights matrix.',
    dataTypes: {input: ['table', 'spatial-weights'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics'],
    params: [
      param('design', "GraphDataView<'float32'>", 'Row-major structural design Z, no intercept.', {
        required: true
      }),
      param('coefficientCount', 'number', 'Columns of the design.', {required: true}),
      param('instruments', "GraphDataView<'float32'>", 'Row-major instruments H, no intercept.', {
        required: true
      }),
      param('instrumentCount', 'number', 'Columns of the instruments.', {required: true}),
      param('residuals', "GraphDataView<'float32'>", 'Structural residuals of the same rows.', {
        required: true
      }),
      param('method', "'HC0' | 'HC1' | 'HAC' | undefined", 'Sandwich estimator.'),
      param('kernel', 'GPUSpatialWeights | undefined', 'HAC kernel weights.'),
      param('output', '{covariance, status: {code}}', 'Caller-owned results.', {required: true})
    ],
    outputs: [
      output('covariance', "GraphDataView<'float32'>", 'Coefficient covariance matrix.'),
      output('status.code', "GraphDataView<'uint32'>", 'Status code.')
    ],
    example: makeExample(
      ['GPUInstrumentalVariablesCovariance'],
      [
        'graph.add(',
        '  new GPUInstrumentalVariablesCovariance({',
        "    design: graph.importGPUData('z', designData),",
        '    coefficientCount: 3,',
        "    instruments: graph.importGPUData('h', instrumentData),",
        '    instrumentCount: 5,',
        '    residuals, // structural residuals of the same rows',
        "    method: 'HAC',",
        '    kernel: bandWeights,',
        '    output // covariance and status.code views',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'regression', seed: 235, caption: 'Wider bands where errors cluster.'},
    related: ['spatial-two-stage-least-squares', 'spatial-panel-regression'],
    lensScenario: 'gpu-spatial-analysis.instrumental-variables-covariance',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- spatial regression
  {
    id: 'spatial-two-stage-least-squares',
    symbol: 'GPUSpatialTwoStageLeastSquares',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial 2SLS',
    summary: 'Spatial lag model y = rho Wy + Xb + e by two-stage least squares.',
    explanation:
      'A fast, distribution-free alternative to maximum likelihood (spreg `GM_Lag`). Spatially lagged predictors serve as instruments. Optional robust inference keeps the point estimates.',
    dataTypes: {input: ['table', 'spatial-weights'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      param('weights', 'GPUSpatialWeights', 'Square weights W over the rows.', {required: true}),
      param('predictors', "GraphDataView<'float32'>", 'Row-major predictors.', {required: true}),
      param('response', "GraphDataView<'float32'>", 'Response y.', {required: true}),
      param('predictorCount', 'number', 'Predictor columns, compile-time.', {required: true}),
      param('instrumentOrder', '2 | 1 | undefined', 'Order of the lagged instruments.', {
        default: '1'
      }),
      param(
        'robust',
        '{method: "HC0" | "HC1" | "HAC"; kernel?: GPUSpatialWeights} | undefined',
        'Robust inference.'
      ),
      param('output', 'GPUSpatialTwoStageLeastSquaresOutput', 'Caller-owned outputs.', {
        required: true
      })
    ],
    outputs: [
      output('table', "GraphDataView<'float32'>", 'Per coefficient: estimate, SE, z, p; rho last.'),
      output('summary', "GraphDataView<'float32'>", 'Fit statistics.'),
      output('residuals', "GraphDataView<'float32'> | undefined", 'Structural residuals per row.'),
      STATUS_OUTPUT
    ],
    example: makeExample(
      ['GPUSpatialTwoStageLeastSquares'],
      [
        ...REGRESSION_INPUTS,
        'graph.add(',
        '  new GPUSpatialTwoStageLeastSquares({',
        '    weights, predictors, response,',
        '    predictorCount: 2,',
        '    output: {',
        "      table: output('table', 'float32', 4 * 4), // (predictorCount + 2) rows of 4",
        "      summary: output('summary', 'float32', 16),",
        "      status: {code: output('status', 'uint32', 1)}",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'regression', seed: 236, caption: "Neighbors' outcomes enter as a predictor."},
    related: [
      'spatial-lag-maximum-likelihood',
      'spatial-lag-impacts',
      'spatial-panel-regression',
      'instrumental-variables-covariance'
    ],
    lensScenario: 'gpu-spatial-analysis.spatial-two-stage-least-squares',
    reference: 'api/gpu-spatial-analysis/operators-regression',
    maturity: 'stable'
  },
  {
    id: 'spatial-lag-maximum-likelihood',
    symbol: 'GPUSpatialLagMaximumLikelihood',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial lag model (ML)',
    summary: 'Maximum-likelihood fit of y = rho Wy + Xb + e, with a trace-series log-determinant.',
    explanation:
      'Use when the outcome itself spreads between neighbors. Searches rho on the concentrated likelihood; the log-determinant comes from `maximumPower` traces or supplied eigenvalues. Matches spreg `ML_Lag`.',
    dataTypes: {input: ['table', 'spatial-weights'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics', 'urban', 'demographics'],
    params: [
      param('weights', 'GPUSpatialWeights | GPUChunkedSpatialWeights', 'Square weights W.', {
        required: true
      }),
      param('predictors', "GraphDataView<'float32'>", 'Row-major predictors.', {required: true}),
      param('response', "GraphDataView<'float32'>", 'Response y.', {required: true}),
      param('predictorCount', 'number', 'Predictor columns, 1 to 14.', {required: true}),
      param('maximumPower', 'number | undefined', 'Series length Q of the log-determinant.'),
      param(
        'traceMode',
        'GPUSpatialWeightsTracesMode | undefined',
        "'exact' or 'stochastic' traces."
      ),
      param(
        'logDeterminant',
        'GPUSpatialLogDeterminantMode | undefined',
        'Series (default) or exact eigenvalues.',
        {default: "'series'"}
      ),
      param('output', 'GPUSpatialLagMaximumLikelihoodOutput', 'Caller-owned outputs.', {
        required: true
      })
    ],
    outputs: [
      output('table', "GraphDataView<'float32'>", 'Per coefficient: estimate, SE, z, p; rho last.'),
      output('summary', "GraphDataView<'float32'>", 'Log likelihood, sigma squared, AIC.'),
      output(
        'powerTraces',
        "GraphDataView<'float32'> | undefined",
        'Traces reusable by lag impacts.'
      ),
      STATUS_OUTPUT
    ],
    example: makeExample(
      ['GPUSpatialLagMaximumLikelihood'],
      [
        ...REGRESSION_INPUTS,
        'graph.add(',
        '  new GPUSpatialLagMaximumLikelihood({',
        '    weights, predictors, response,',
        '    predictorCount: 2,',
        '    output: {',
        "      table: output('table', 'float32', 16),",
        "      summary: output('summary', 'float32', 22),",
        "      status: {code: output('status', 'uint32', 1)}",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'regression', seed: 237, caption: 'rho captures spillover between neighbors.'},
    related: [
      'spatial-lag-impacts',
      'spatial-two-stage-least-squares',
      'spatial-error-maximum-likelihood',
      'spatial-regression-diagnostics'
    ],
    lensScenario: 'gpu-spatial-analysis.spatial-lag-maximum-likelihood',
    reference: 'api/gpu-spatial-analysis/operators-regression',
    maturity: 'stable'
  },
  {
    id: 'spatial-lag-impacts',
    symbol: 'GPUSpatialLagImpacts',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial lag impacts',
    summary: 'Direct, indirect and total effect of each predictor in a spatial lag model.',
    explanation:
      'In a lag model a change in x also spills to neighbors, so coefficients are not marginal effects. Impacts (LeSage and Pace) sum that series from the fit and weights traces. Chain it after the lag fit.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['model', 'measure'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      param('coefficients', "GraphDataView<'float32'>", 'Fit table with rho and predictors.', {
        required: true
      }),
      param('coefficientStride', 'number', 'Float32 values per coefficient row.', {required: true}),
      param('rhoIndex', 'number', 'Row of rho in the table.', {required: true}),
      param('predictorCount', 'number', 'Predictors whose impacts are written.', {required: true}),
      param('rowCount', 'number', 'Rows n of the weights.', {required: true}),
      param('maximumPower', 'number', 'Series length Q used.', {required: true}),
      param('traces', '{powerTraces, powerSums, infinityNorm}', 'Traces of the weights.', {
        required: true
      }),
      param('output', 'GPUSpatialLagImpactsOutput', 'Caller-owned impacts.', {required: true})
    ],
    outputs: [
      output('impacts', "GraphDataView<'float32'>", 'Direct, indirect, total per predictor.')
    ],
    example: makeExample(
      ['GPUSpatialLagImpacts'],
      [
        'graph.add(',
        '  new GPUSpatialLagImpacts({',
        '    coefficients: table, // from GPUSpatialTwoStageLeastSquares',
        '    coefficientStride: 4,',
        '    rhoIndex: 3,',
        '    predictorCount: 2,',
        '    rowCount,',
        '    maximumPower: 30,',
        '    traces: {powerTraces, powerSums, infinityNorm},',
        "    output: {impacts: output('impacts', 'float32', 2 * 3)}",
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'regression', seed: 238, caption: 'Direct plus indirect equals total.'},
    related: [
      'spatial-lag-maximum-likelihood',
      'spatial-two-stage-least-squares',
      'spatial-weights-traces'
    ],
    lensScenario: 'gpu-spatial-analysis.spatial-lag-impacts',
    reference: 'api/gpu-spatial-analysis/operators-regression',
    maturity: 'stable'
  },
  {
    id: 'spatial-error-maximum-likelihood',
    symbol: 'GPUSpatialErrorMaximumLikelihood',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial error model (ML)',
    summary: 'Maximum-likelihood fit of y = Xb + u with spatially autocorrelated errors.',
    explanation:
      'Use when unmeasured, spatially structured factors leave correlated residuals. Searches lambda on the concentrated likelihood with the same trace-series machinery as the lag model. Matches spreg `ML_Error`.',
    dataTypes: {input: ['table', 'spatial-weights'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      param('weights', 'GPUSpatialWeights | GPUChunkedSpatialWeights', 'Square weights W.', {
        required: true
      }),
      param('predictors', "GraphDataView<'float32'>", 'Row-major predictors.', {required: true}),
      param('response', "GraphDataView<'float32'>", 'Response y.', {required: true}),
      param('predictorCount', 'number', 'Predictor columns, 1 to 15.', {required: true}),
      param('maximumPower', 'number | undefined', 'Series length Q of the log-determinant.'),
      param(
        'traceMode',
        'GPUSpatialWeightsTracesMode | undefined',
        "'exact' or 'stochastic' traces."
      ),
      param(
        'logDeterminant',
        'GPUSpatialLogDeterminantMode | undefined',
        'Series (default) or exact eigenvalues.',
        {default: "'series'"}
      ),
      param('output', 'GPUSpatialErrorMaximumLikelihoodOutput', 'Caller-owned outputs.', {
        required: true
      })
    ],
    outputs: [
      output(
        'table',
        "GraphDataView<'float32'>",
        'Per coefficient: estimate, SE, z, p; lambda last.'
      ),
      output('summary', "GraphDataView<'float32'>", 'Log likelihood, sigma squared, AIC.'),
      output('residuals', "GraphDataView<'float32'> | undefined", 'Residuals u = y - Xb.'),
      STATUS_OUTPUT
    ],
    example: makeExample(
      ['GPUSpatialErrorMaximumLikelihood'],
      [
        ...REGRESSION_INPUTS,
        'graph.add(',
        '  new GPUSpatialErrorMaximumLikelihood({',
        '    weights, predictors, response,',
        '    predictorCount: 2,',
        '    output: {',
        "      table: output('table', 'float32', 16),",
        "      summary: output('summary', 'float32', 22),",
        "      status: {code: output('status', 'uint32', 1)}",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'regression', seed: 239, caption: 'Residual clustering absorbed by lambda.'},
    related: [
      'spatial-error-gm',
      'spatial-lag-maximum-likelihood',
      'spatial-regression-diagnostics',
      'spatial-panel-error-ml'
    ],
    lensScenario: 'gpu-spatial-analysis.spatial-error-maximum-likelihood',
    reference: 'api/gpu-spatial-analysis/operators-regression',
    maturity: 'stable'
  },
  {
    id: 'spatial-error-gm',
    symbol: 'GPUSpatialErrorGM',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial error model (GM)',
    summary: 'Generalized-moments fit of y = Xb + u with spatially autocorrelated errors.',
    explanation:
      'A likelihood-free route to the error model that scales to large weights and tolerates heteroskedasticity. `kelejian-prucha` is the robust estimator; `homoskedastic` is the default. Matches spreg `GM_Error`.',
    dataTypes: {input: ['table', 'spatial-weights'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      param('weights', 'GPUSpatialWeights | GPUChunkedSpatialWeights', 'Square weights W.', {
        required: true
      }),
      param('predictors', "GraphDataView<'float32'>", 'Row-major predictors.', {required: true}),
      param('response', "GraphDataView<'float32'>", 'Response y.', {required: true}),
      param('predictorCount', 'number', 'Predictor columns, 1 to 15.', {required: true}),
      param(
        'estimator',
        'GPUSpatialErrorGMEstimator | undefined',
        'homoskedastic or kelejian-prucha.',
        {default: "'homoskedastic'"}
      ),
      param(
        'traceMode',
        'GPUSpatialWeightsTracesMode | undefined',
        'Trace computation for the variance.'
      ),
      param('traceProbeCount', 'number | undefined', 'Hutchinson probe count.'),
      param('output', 'GPUSpatialErrorGMOutput', 'Caller-owned outputs.', {required: true})
    ],
    outputs: [
      output(
        'table',
        "GraphDataView<'float32'>",
        'Per coefficient: estimate, SE, z, p; lambda last.'
      ),
      output('summary', "GraphDataView<'float32'>", 'Fit statistics.'),
      output('residuals', "GraphDataView<'float32'> | undefined", 'Residuals u = y - Xb.'),
      STATUS_OUTPUT
    ],
    example: makeExample(
      ['GPUSpatialErrorGM'],
      [
        ...REGRESSION_INPUTS,
        'graph.add(',
        '  new GPUSpatialErrorGM({',
        '    weights, predictors, response,',
        '    predictorCount: 2,',
        "    estimator: 'kelejian-prucha',",
        '    output: {',
        "      table: output('table', 'float32', 16),",
        "      summary: output('summary', 'float32', 13),",
        "      status: {code: output('status', 'uint32', 1)}",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'regression', seed: 240, caption: 'Moment equations pin down lambda.'},
    related: [
      'spatial-error-maximum-likelihood',
      'seemingly-unrelated-regression',
      'spatial-regression-diagnostics'
    ],
    lensScenario: 'gpu-spatial-analysis.spatial-error-gm',
    reference: 'api/gpu-spatial-analysis/operators-regression',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- regimes
  {
    id: 'spatial-regime-regression',
    symbol: 'GPUSpatialRegimeRegression',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial regimes',
    summary: 'Separate OLS intercept and slopes for each region.',
    explanation:
      'Use when relationships differ between caller-defined regions, such as urban and rural. Regime ids are contiguous in `[0, regimeCount)`. Compare with a restricted model to test whether the regimes matter.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      param('predictors', "GraphDataView<'float32'>", 'Row-major predictors.', {required: true}),
      param('predictorCount', 'number', 'Predictor columns.', {required: true}),
      param('response', "GraphDataView<'float32'>", 'Response y.', {required: true}),
      param('regimes', "GraphDataView<'uint32'>", 'Regime id per row, in [0, regimeCount).', {
        required: true
      }),
      param('regimeCount', 'number', 'Number of regimes.', {required: true}),
      param('mask', "GraphDataView<'uint32'> | undefined", 'Zero excludes the row.'),
      param('output', '{coefficients, fitted}', 'Caller-owned results.', {
        required: true
      })
    ],
    outputs: [
      output('coefficients', "GraphDataView<'float32'>", 'Intercept and slopes per regime.'),
      output('fitted', "GraphDataView<'float32'>", 'Fitted value per row.'),
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: 'GPU status words; zero means complete.'
      },
      {name: 'counts', type: 'GraphDataView<"uint32">', description: 'Rows per regime.'}
    ],
    example: makeExample(
      ['GPUSpatialRegimeRegression'],
      [
        ...REGRESSION_INPUTS,
        'graph.add(',
        '  new GPUSpatialRegimeRegression({',
        '    predictors, response,',
        '    predictorCount: 2,',
        "    regimes: graph.importGPUData('regimes', regimeData), // uint32 ids",
        '    regimeCount: 3,',
        '    output // coefficients and fitted views',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'regression', seed: 241, caption: 'One line per region.'},
    related: [
      'restricted-regime-regression',
      'ordinary-least-squares',
      'geographically-weighted-regression'
    ],
    lensScenario: 'gpu-spatial-analysis.spatial-regime-regression',
    reference: 'api/gpu-spatial-analysis/expansion-lenses',
    maturity: 'stable'
  },
  {
    id: 'restricted-regime-regression',
    symbol: 'GPURestrictedRegimeRegression',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Restricted regimes and Chow test',
    summary:
      'Regime OLS where chosen terms are shared, with a joint Chow F test against full variation.',
    explanation:
      'Decide which coefficients vary by region and which stay common via `common`, intercept first. The Chow statistic tests the restriction against fully varying regimes. Use it to find out what really differs.',
    dataTypes: {input: ['table'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      param('predictors', "GraphDataView<'float32'>", 'Row-major predictors.', {required: true}),
      param('predictorCount', 'number', 'Predictor columns.', {required: true}),
      param('response', "GraphDataView<'float32'>", 'Response y.', {required: true}),
      param('regimes', "GraphDataView<'uint32'>", 'Regime id per row.', {required: true}),
      param('regimeCount', 'number', 'Number of regimes.', {required: true}),
      param('common', 'readonly boolean[]', 'True where a term is shared; intercept first.', {
        required: true
      }),
      param('mask', "GraphDataView<'uint32'> | undefined", 'Zero excludes the row.'),
      param('output', '{coefficients, covariance}', 'Caller-owned results.', {required: true})
    ],
    outputs: [
      output('coefficients', "GraphDataView<'float32'>", 'Restricted coefficients.'),
      output('covariance', "GraphDataView<'float32'>", 'Coefficient covariance.'),
      {
        name: 'stability',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Coefficient stability diagnostics.'
      },
      {name: 'fitted', type: 'GraphDataView<"float32">', description: 'Fitted response per row.'},
      {
        name: 'chow',
        type: 'GraphDataView<"float32">',
        description: 'Joint Chow F statistic and degrees of freedom.'
      },
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: 'GPU status words; zero means complete.'
      }
    ],
    example: makeExample(
      ['GPURestrictedRegimeRegression'],
      [
        ...REGRESSION_INPUTS,
        'graph.add(',
        '  new GPURestrictedRegimeRegression({',
        '    predictors, response,',
        '    predictorCount: 2,',
        "    regimes: graph.importGPUData('regimes', regimeData),",
        '    regimeCount: 3,',
        '    common: [false, true, false], // intercept varies, slope 1 shared, slope 2 varies',
        '    output // coefficients, covariance and the Chow statistic views',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'regression', seed: 242, caption: 'Shared slope, regional intercepts.'},
    related: ['spatial-regime-regression', 'ordinary-least-squares'],
    lensScenario: 'gpu-spatial-analysis.restricted-regime-regression',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- local regression
  {
    id: 'geographically-weighted-regression',
    symbol: 'GPUGeographicallyWeightedRegression',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Geographically weighted regression',
    summary: 'A separate regression at every location, with the bandwidth chosen automatically.',
    explanation:
      'Shows where a relationship is strong, weak or reversed. Each location is fit with distance-weighted neighbors; AICc picks the bandwidth by a caller ladder or golden-section search. Output maps of coefficients, t values and local R2.',
    dataTypes: {input: ['points', 'table'], output: ['points', 'table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics', 'urban', 'demographics'],
    params: [
      param('positions', "GraphDataView<'float32x2'>", 'One position per row.', {required: true}),
      param('predictors', "GraphDataView<'float32'>", 'Row-major predictors, no intercept.', {
        required: true
      }),
      param('predictorCount', 'number', 'Predictor columns, 1 to 11.', {required: true}),
      param('response', "GraphDataView<'float32'>", 'Response, one per row.', {required: true}),
      param('parameters', "GraphDataView<'float32'>", 'Per-frame bandwidth ladder and kernel.', {
        required: true
      }),
      param(
        'bandwidthSearch',
        "'ladder' | 'golden-section' | undefined",
        'Bandwidth search strategy.'
      ),
      param('spatialContext', 'GPUSpatialContext | undefined', 'Coordinate space and metric.'),
      param('output', 'GPUGeographicallyWeightedRegressionOutput', 'Caller-owned outputs.', {
        required: true
      })
    ],
    outputs: [
      output(
        'coefficients',
        "GraphDataView<'float32'>",
        'Local coefficients, row * (k + 1) + column.'
      ),
      output('tValues', "GraphDataView<'float32'> | undefined", 'Local t values.'),
      output('localR2', "GraphDataView<'float32'> | undefined", 'Local weighted R2 per row.'),
      output(
        'selectedBandwidth',
        "GraphDataView<'float32'> | undefined",
        'Chosen candidate index and bandwidth.'
      )
    ],
    example: makeExample(
      [
        'GPUGeographicallyWeightedRegression',
        'GPUParameterBuffer',
        'getGPUGeographicallyWeightedRegressionParameterValues'
      ],
      [
        ...REGRESSION_INPUTS,
        "const parameters = new GPUParameterBuffer(device, {id: 'gwr', format: 'float32', length: 64});",
        'parameters.write(getGPUGeographicallyWeightedRegressionParameterValues({bandwidths: [500, 1000, 2000]}, 16));',
        'graph.add(',
        '  new GPUGeographicallyWeightedRegression({',
        "    positions: graph.importGPUData('positions', positionData),",
        '    predictors, response, predictorCount: 2,',
        '    parameters: parameters.importToGraph(graph),',
        '    maximumBandwidthCount: 16,',
        "    output: {coefficients: output('beta', 'float32', rowCount * 3)}",
        '  })',
        ');'
      ]
    ),
    preview: {
      kind: 'regression',
      seed: 243,
      caption: 'A coefficient surface that varies across space.'
    },
    related: [
      'generalized-gwr',
      'multiscale-gwr',
      'gwr-nonstationarity-test',
      'local-collinearity',
      'ordinary-least-squares'
    ],
    lensScenario: 'gpu-spatial-analysis.geographically-weighted-regression',
    reference: 'api/gpu-spatial-analysis/operators-regression',
    maturity: 'stable'
  },
  {
    id: 'generalized-gwr',
    symbol: 'GPUGeneralizedGeographicallyWeightedRegression',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Generalized GWR',
    summary: 'Local Poisson or logistic regression with a fixed or adaptive bandwidth.',
    explanation:
      'For counts (Poisson, log link, optional exposure `offset`) and yes/no outcomes (Bernoulli, logit). Adaptive mode uses the k nearest neighbors. Optional leave-one-out bandwidth selection and model-specific standard errors.',
    dataTypes: {input: ['points', 'table'], output: ['points', 'table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics', 'demographics'],
    params: [
      param('family', "'poisson' | 'logistic'", 'Outcome distribution.', {required: true}),
      param('positions', "GraphDataView<'float32x2'>", 'One position per row.', {required: true}),
      param('predictors', "GraphDataView<'float32'>", 'Row-major predictors, no intercept.', {
        required: true
      }),
      param('predictorCount', 'number', 'Predictor columns.', {required: true}),
      param('response', "GraphDataView<'float32'>", 'Counts, or 0/1.', {required: true}),
      param('parameters', "GraphDataView<'float32'>", '[bandwidth, coefficient tolerance].', {
        required: true
      }),
      param(
        'bandwidthMode',
        "'fixed' | 'adaptive' | undefined",
        'Distance or k nearest neighbors.',
        {default: "'fixed'"}
      ),
      param('offset', "GraphDataView<'float32'> | undefined", 'Log exposure for Poisson.')
    ],
    outputs: [
      output('coefficients', "GraphDataView<'float32'>", 'Local coefficients per row.'),
      output('fitted', "GraphDataView<'float32'>", 'Fitted mean per row.'),
      {
        name: 'status',
        type: '{count, overflow}',
        description: 'GPU status words; zero means complete.'
      }
    ],
    example: makeExample(
      ['GPUGeneralizedGeographicallyWeightedRegression'],
      [
        'graph.add(',
        '  new GPUGeneralizedGeographicallyWeightedRegression({',
        "    family: 'poisson',",
        "    positions: graph.importGPUData('positions', positionData),",
        "    predictors: graph.importGPUData('x', predictorData),",
        '    predictorCount: 2,',
        "    response: graph.importGPUData('counts', countData),",
        "    offset: graph.importGPUData('log-exposure', exposureData),",
        "    bandwidthMode: 'adaptive',",
        '    parameters, // [k, tolerance]',
        '    output // coefficients and fitted views',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'regression', seed: 244, caption: 'Local rates that change across the map.'},
    related: ['geographically-weighted-regression', 'multiscale-gwr'],
    lensScenario: 'gpu-spatial-analysis.generalized-geographically-weighted-regression',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  },
  {
    id: 'multiscale-gwr',
    symbol: 'GPUMultiscaleGeographicallyWeightedRegression',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Multiscale GWR',
    summary: 'Local regression where each variable gets its own bandwidth.',
    explanation:
      'Some effects are regional, some local; one bandwidth cannot serve both. Backfitting estimates a distance bandwidth per variable, intercept included. Optional inference with a known noise variance.',
    dataTypes: {input: ['points', 'table'], output: ['points', 'table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      param('positions', "GraphDataView<'float32x2'>", 'One position per row.', {required: true}),
      param('predictors', "GraphDataView<'float32'>", 'Row-major predictors, no intercept.', {
        required: true
      }),
      param('predictorCount', 'number', 'Predictor columns.', {required: true}),
      param('response', "GraphDataView<'float32'>", 'Response, one per row.', {required: true}),
      param('parameters', "GraphDataView<'float32'>", 'Tolerance and per-variable bandwidths.', {
        required: true
      }),
      param(
        'bandwidthCandidates',
        "GraphDataView<'float32'> | undefined",
        'Candidate bandwidths per variable.'
      ),
      param('maximumIterations', 'number | undefined', 'Backfitting iteration cap.'),
      param('kernel', "'gaussian' | 'bisquare' | undefined", 'Distance kernel.')
    ],
    outputs: [
      output('coefficients', "GraphDataView<'float32'>", 'Local coefficients per row.'),
      output('fitted', "GraphDataView<'float32'>", 'Fitted value per row.'),
      {
        name: 'selectedBandwidths',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Selected bandwidth per term.'
      },
      {
        name: 'conditionalStandardErrors',
        type: 'GraphDataView<"float32"> | undefined',
        description: 'Standard errors conditional on other terms.'
      },
      {
        name: 'status',
        type: '{count, overflow}',
        description: 'GPU status words; zero means complete.'
      }
    ],
    example: makeExample(
      ['GPUMultiscaleGeographicallyWeightedRegression'],
      [
        'graph.add(',
        '  new GPUMultiscaleGeographicallyWeightedRegression({',
        "    positions: graph.importGPUData('positions', positionData),",
        "    predictors: graph.importGPUData('x', predictorData),",
        '    predictorCount: 2,',
        "    response: graph.importGPUData('y', responseData),",
        "    kernel: 'gaussian',",
        '    parameters, // [tolerance, intercept bandwidth, ...]',
        '    output // coefficients and fitted views',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'interpolation', seed: 245, caption: 'Smooth and rough effects together.'},
    related: ['geographically-weighted-regression', 'generalized-gwr'],
    lensScenario: 'gpu-spatial-analysis.multiscale-geographically-weighted-regression',
    reference: 'api/gpu-spatial-analysis/expansion-lenses',
    maturity: 'stable'
  },
  {
    id: 'gwr-nonstationarity-test',
    symbol: 'GPUGeographicallyWeightedRegressionNonstationarityTest',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'GWR nonstationarity test',
    summary: 'Monte Carlo test of whether GWR coefficients really vary across space.',
    explanation:
      'Answers: is the map of a coefficient signal or noise? Permutes locations and refits, comparing the spread of local estimates with the observed one. Pass the inputs and outputs of the GWR under test.',
    dataTypes: {input: ['points', 'table'], output: ['table']},
    useCases: ['model', 'find-patterns'],
    domains: ['spatial-statistics'],
    params: [
      param('positions', "GraphDataView<'float32x2'>", 'Positions given to the GWR.', {
        required: true
      }),
      param('predictors', "GraphDataView<'float32'>", 'Predictors given to the GWR.', {
        required: true
      }),
      param('predictorCount', 'number', 'Predictor columns.', {required: true}),
      param('response', "GraphDataView<'float32'>", 'Response given to the GWR.', {required: true}),
      param('bandwidthParameters', "GraphDataView<'float32'>", 'Per-frame parameters of the GWR.', {
        required: true
      }),
      param('selectedBandwidth', "GraphDataView<'float32'>", 'The GWR output.selectedBandwidth.', {
        required: true
      }),
      param('coefficients', "GraphDataView<'float32'>", 'The GWR output.coefficients.', {
        required: true
      }),
      param('maximumPermutations', 'number', 'Upper bound of permutations, 1 to 2^20.', {
        required: true
      })
    ],
    moreRequired: ['parameters'],
    outputs: [
      output('table', "GraphDataView<'float32'>", 'Per coefficient: spread, p-value.'),
      output('summary', "GraphDataView<'float32'>", 'Run summary.'),
      output(
        'standardDeviations',
        "GraphDataView<'float32'> | undefined",
        'Spread per run and coefficient.'
      )
    ],
    example: makeExample(
      ['GPUGeographicallyWeightedRegressionNonstationarityTest'],
      [
        'graph.add(',
        '  new GPUGeographicallyWeightedRegressionNonstationarityTest({',
        '    positions, predictors, response, predictorCount: 2, // as given to the GWR',
        '    bandwidthParameters: gwrParameters,',
        '    selectedBandwidth: gwrOutput.selectedBandwidth,',
        '    coefficients: gwrOutput.coefficients,',
        '    parameters, // uint32 permutation parameters',
        '    maximumPermutations: 99,',
        '    output // table, summary, standardDeviations',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'regression', seed: 246, caption: 'Observed spread against permuted spreads.'},
    related: ['geographically-weighted-regression', 'local-collinearity'],
    lensScenario: 'gpu-spatial-analysis.geographically-weighted-regression-nonstationarity-test',
    reference: 'api/gpu-spatial-analysis/operators-regression',
    maturity: 'stable'
  },
  {
    id: 'local-collinearity',
    symbol: 'GPULocalCollinearity',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Local collinearity',
    summary: 'Maps local variance inflation factors of predictors under explicit local weights.',
    explanation:
      'Local models fail quietly where predictors move together. Computes weighted, centered VIF per row and predictor, plus condition numbers. Use the same weights as the local fit.',
    dataTypes: {input: ['spatial-weights', 'table'], output: ['table']},
    useCases: ['model', 'measure'],
    domains: ['spatial-statistics'],
    params: [
      param(
        'weights',
        'GPUSpatialWeights | GPUChunkedSpatialWeights',
        'Local estimation weights, self included.',
        {required: true}
      ),
      param('predictors', "GraphDataView<'float32'>", 'Row-major predictors.', {required: true}),
      param('predictorCount', 'number', 'Predictor columns.', {required: true}),
      param('mask', "GraphDataView<'uint32'> | undefined", 'Zero excludes the row.'),
      param('output', '{varianceInflationFactors}', 'Caller-owned results.', {required: true})
    ],
    outputs: [
      output(
        'varianceInflationFactors',
        "GraphDataView<'float32'>",
        'Local VIF per row and predictor.'
      ),
      output('conditionNumbers', "GraphDataView<'float32'>", 'Local condition number per row.'),
      output('status', "GraphDataView<'uint32'>", 'Per-row fit status.')
    ],
    example: makeExample(
      ['GPULocalCollinearity'],
      [
        'graph.add(',
        '  new GPULocalCollinearity({',
        '    weights, // kernel weights with self slots',
        "    predictors: graph.importGPUData('x', predictorData),",
        '    predictorCount: 3,',
        '    output // varianceInflationFactors and the remaining views',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'hotspots', seed: 247, caption: 'Where predictors become redundant.'},
    related: ['geographically-weighted-regression', 'gwr-nonstationarity-test'],
    lensScenario: 'gpu-spatial-analysis.local-collinearity',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  },
  // ---------------------------------------------------------------- panels
  {
    id: 'spatial-panel-regression',
    symbol: 'GPUSpatialPanelRegression',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial panel regression',
    summary: 'Pooled spatial-lag 2SLS on a balanced panel, with optional unit effects.',
    explanation:
      'Best for places observed repeatedly over time. Weights are time-invariant across the entity axis; rows are period-major. `unit` effects absorb a separate intercept per entity.',
    dataTypes: {input: ['table', 'spatial-weights', 'time-series'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics', 'urban', 'demographics'],
    params: [
      param('weights', 'GPUSpatialWeights', 'Time-invariant weights over entities.', {
        required: true
      }),
      param('periodCount', 'number', 'Periods in the balanced panel.', {required: true}),
      param('predictors', "GraphDataView<'float32'>", 'Rows are period * entityCount + entity.', {
        required: true
      }),
      param('predictorCount', 'number', 'Predictor columns.', {required: true}),
      param('response', "GraphDataView<'float32'>", 'Response, same row order.', {required: true}),
      param('effects', "'pooled' | 'unit' | undefined", 'Unit effects absorb entity intercepts.', {
        default: "'pooled'"
      }),
      param('instrumentOrder', '2 | 1 | undefined', 'Order of the lagged instruments.'),
      param('output', 'GPUSpatialTwoStageLeastSquaresOutput', 'Caller-owned outputs.', {
        required: true
      })
    ],
    outputs: [
      output('table', "GraphDataView<'float32'>", 'Per coefficient: estimate, SE, z, p; rho last.'),
      output('summary', "GraphDataView<'float32'>", 'Fit statistics.'),
      STATUS_OUTPUT
    ],
    example: makeExample(
      ['GPUSpatialPanelRegression'],
      [
        ...REGRESSION_INPUTS,
        'graph.add(',
        '  new GPUSpatialPanelRegression({',
        '    weights, // entity weights, not panel-sized',
        '    periodCount: 10,',
        "    effects: 'unit',",
        '    predictors, response, predictorCount: 2,',
        '    output: {',
        "      table: output('table', 'float32', 16),",
        "      summary: output('summary', 'float32', 16),",
        "      status: {code: output('status', 'uint32', 1)}",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'time-series', seed: 248, caption: 'Many places, many periods, one model.'},
    related: ['spatial-panel-lag-ml', 'spatial-panel-error-ml', 'spatial-two-stage-least-squares'],
    lensScenario: 'gpu-spatial-analysis.spatial-panel-regression',
    reference: 'api/gpu-spatial-analysis/expansion-lenses',
    maturity: 'stable'
  },
  {
    id: 'spatial-panel-lag-ml',
    symbol: 'GPUSpatialPanelLagMaximumLikelihood',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial panel lag (ML)',
    summary: 'Balanced-panel spatial lag model by maximum likelihood, random or unit effects.',
    explanation:
      'When outcomes spill over between neighbors and you observe them repeatedly. Choose `random` or `unit` effects and compare them with the optional Hausman test.',
    dataTypes: {input: ['table', 'spatial-weights', 'time-series'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      param('weights', 'GPUSpatialWeights', 'Entity weights.', {required: true}),
      param('periodCount', 'number', 'Periods in the balanced panel.', {required: true}),
      param('predictors', "GraphDataView<'float32'>", 'Rows are period * entityCount + entity.', {
        required: true
      }),
      param('predictorCount', 'number', 'Predictor columns.', {required: true}),
      param('response', "GraphDataView<'float32'>", 'Response, same row order.', {required: true}),
      param('effects', "'random' | 'unit' | undefined", 'Random or unit effects.'),
      param('maximumIterations', 'number | undefined', 'Iteration cap.'),
      param('hausman', '{tests, status} | undefined', 'Random-versus-unit comparison.')
    ],
    outputs: [
      output('coefficients', "GraphDataView<'float32'>", 'Estimates including rho.'),
      output('covariance', "GraphDataView<'float32'>", 'Coefficient covariance.'),
      {
        name: 'summary',
        type: 'GraphDataView<"float32">',
        description: 'Fit summary: log likelihood and iteration counts.'
      },
      {
        name: 'status',
        type: 'GraphDataView<"uint32">',
        description: 'GPU status words; zero means complete.'
      },
      {name: 'fitted', type: 'GraphDataView<"float32">', description: 'Fitted response per row.'},
      {name: 'residuals', type: 'GraphDataView<"float32">', description: 'Residual per row.'}
    ],
    example: makeExample(
      ['GPUSpatialPanelLagMaximumLikelihood'],
      [
        'graph.add(',
        '  new GPUSpatialPanelLagMaximumLikelihood({',
        '    weights,',
        '    periodCount: 10,',
        "    effects: 'random',",
        "    predictors: graph.importGPUData('x', predictorData),",
        '    predictorCount: 2,',
        "    response: graph.importGPUData('y', responseData),",
        '    output // coefficients, covariance and the remaining views',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'time-series', seed: 249, caption: 'Spillover estimated across periods.'},
    related: [
      'spatial-panel-regression',
      'spatial-panel-error-ml',
      'spatial-lag-maximum-likelihood'
    ],
    lensScenario: 'gpu-spatial-analysis.spatial-panel-lag-maximum-likelihood',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  },
  {
    id: 'spatial-panel-error-ml',
    symbol: 'GPUSpatialPanelErrorMaximumLikelihood',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Spatial panel error (ML)',
    summary: 'Unit fixed-effects panel with spatially autocorrelated errors, by Gaussian ML.',
    explanation:
      'The panel counterpart of the spatial error model: `y_it = alpha_i + X beta + u_it`, `u_t = lambda W u_t + e_t`. Optional pooled-panel LM and robust LM diagnostics. Weights span the entity graph.',
    dataTypes: {input: ['table', 'spatial-weights', 'time-series'], output: ['table']},
    useCases: ['model'],
    domains: ['spatial-statistics', 'urban'],
    params: [
      param('weights', 'GPUSpatialWeights', 'Whole entity-graph weights.', {required: true}),
      param('periodCount', 'number', 'Periods in the balanced panel.', {required: true}),
      param('predictors', "GraphDataView<'float32'>", 'Period-major rows.', {required: true}),
      param('predictorCount', 'number', 'Predictor columns, 1 to 15.', {required: true}),
      param('response', "GraphDataView<'float32'>", 'Response, same row order.', {required: true}),
      param('maximumPower', 'number | undefined', 'Series length Q of the log-determinant.'),
      param('diagnostics', '{tests, status} | undefined', 'Pooled-panel LM tests.'),
      param('output', 'GPUSpatialErrorMaximumLikelihoodOutput', 'Caller-owned outputs.', {
        required: true
      })
    ],
    outputs: [
      output(
        'table',
        "GraphDataView<'float32'>",
        'Per coefficient: estimate, SE, z, p; lambda last.'
      ),
      output('summary', "GraphDataView<'float32'>", 'Log likelihood and fit statistics.'),
      STATUS_OUTPUT
    ],
    example: makeExample(
      ['GPUSpatialPanelErrorMaximumLikelihood'],
      [
        'graph.add(',
        '  new GPUSpatialPanelErrorMaximumLikelihood({',
        '    weights,',
        '    periodCount: 10,',
        "    predictors: graph.importGPUData('x', predictorData),",
        '    predictorCount: 2,',
        "    response: graph.importGPUData('y', responseData),",
        '    output: {',
        "      table: output('table', 'float32', 16),",
        "      summary: output('summary', 'float32', 22),",
        "      status: {code: output('status', 'uint32', 1)}",
        '    }',
        '  })',
        ');'
      ]
    ),
    preview: {kind: 'time-series', seed: 250, caption: 'Correlated errors, fixed place effects.'},
    related: [
      'spatial-panel-lag-ml',
      'spatial-panel-regression',
      'spatial-error-maximum-likelihood'
    ],
    lensScenario: 'gpu-spatial-analysis.spatial-panel-error-maximum-likelihood',
    reference: 'api/gpu-spatial-analysis/coverage-statistics',
    maturity: 'stable'
  }
];
