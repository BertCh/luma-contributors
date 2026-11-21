// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Catalog group "raster-embeddings-flow": per-pixel embedding rasters (AlphaEarth-style decoding,
// similarity search, change, clustering, classification, PCA, zone means, overviews) and the
// Lagrangian flow diagnostics (flow map, finite-time Lyapunov exponent).

import type {CatalogEntry, OutputDoc, ParamDoc} from '../schema';

const IMPORT_PATH = '@luma-contributors/analysis/raster-analysis';
const EMBEDDINGS_REFERENCE = 'api/gpu-raster/raster-embeddings';
const LAGRANGIAN_REFERENCE = 'api/gpu-raster/lagrangian-diagnostics-mob06';

const FLOAT = 'GraphDataView<"float32">';
const UINT = 'GraphDataView<"uint32">';
const POINTS = 'GraphDataView<"float32x2">';

type ExampleOptions = {
  symbols: string[];
  setup?: string[];
  /** Expression for the packed per-frame float32 parameters. */
  parameters?: string;
  constructorSymbol: string;
  props: string[];
  addToGraph?: boolean;
};

function makeExample(options: ExampleOptions): string {
  const lines = [
    `import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';`,
    ...(options.parameters
      ? [`import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';`]
      : []),
    `import {${[...options.symbols].sort().join(', ')}} from '${IMPORT_PATH}';`,
    '',
    `const graph = new GPUCommandGraph(device, {id: 'demo'});`,
    `const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(id: string, format: Format, length: number) => createTransientView(graph, id, format, length);`,
    ...(options.setup ?? []),
    ...(options.parameters
      ? [
          `const values = ${options.parameters};`,
          `const parameters = new GPUParameterBuffer(device, {id: 'p', format: 'float32', length: values.length});`
        ]
      : [])
  ];
  const props = [
    ...(options.parameters ? ['parameters: parameters.importToGraph(graph),'] : []),
    ...options.props
  ];
  const open = options.addToGraph
    ? `new ${options.constructorSymbol}({`
    : `graph.add(new ${options.constructorSymbol}({`;
  lines.push(
    open,
    ...props.map(line => `  ${line}`),
    options.addToGraph ? '}).addToGraph(graph);' : '}));'
  );
  lines.push('const compiled = await graph.compileAsync();');
  if (options.parameters) lines.push('parameters.write(values);');
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

const SHAPE_LINE = 'const pixelCount = 512 * 512; const dimensions = 64;';
const VECTORS_PROP = "vectors: view('vectors', 'float32', pixelCount * dimensions),";
const PIXEL_PARAMS = [
  makeParam('pixelCount', 'number', 'Pixels of the raster, row-major.', {required: true}),
  makeParam('dimensions', 'number', 'Vector length, 1 to 128.', {required: true}),
  makeParam('vectors', FLOAT, 'Decoded vectors, `pixel * dimensions + band`.', {required: true}),
  makeParam('mask', UINT, 'Per-pixel QA; zero excludes the pixel.')
];

export const entries: CatalogEntry[] = [
  // Embedding rasters ---------------------------------------------------------------------------
  {
    id: 'raster-embedding-dequantize',
    symbol: 'GPURasterEmbeddingDequantize',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Dequantize embedding raster',
    summary:
      'Decodes AlphaEarth int8 embedding rasters to float vectors, with NoData pixels marked invalid.',
    explanation:
      'AlphaEarth bytes decode as `sign(q) * (q / 127.5)^2` and `-128` in any band is NoData, so reading or averaging the bytes directly is wrong. Run this first; every other embedding operation takes its float vectors.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['transform'],
    domains: ['remote-sensing', 'data-engineering'],
    params: [
      makeParam('pixelCount', 'number', 'Pixels of the raster, row-major.', {required: true}),
      makeParam('dimensions', 'number', 'Bands per pixel, 1 to 128.', {required: true}),
      makeParam('quantized', UINT, 'Signed bytes packed four per word, pixel-major.', {
        required: true
      }),
      makeParam(
        'normalize',
        'boolean',
        'Divide each vector by its length so dot products are cosines.',
        {
          default: 'false'
        }
      ),
      makeParam('output', 'GPURasterEmbeddingDequantizeOutput', 'Decoded vectors and validity.', {
        required: true
      })
    ],
    outputs: [
      makeOutput('vectors', FLOAT, 'Decoded vectors; NaN in every band of an invalid pixel.'),
      makeOutput('validity', UINT, '0 where any band is NoData, else 1.')
    ],
    example: makeExample({
      symbols: ['GPURasterEmbeddingDequantize'],
      setup: [SHAPE_LINE],
      constructorSymbol: 'GPURasterEmbeddingDequantize',
      props: [
        'pixelCount, dimensions, normalize: true,',
        "quantized: view('bytes', 'uint32', (pixelCount * dimensions) / 4),",
        "output: {vectors: view('vectors', 'float32', pixelCount * dimensions), validity: view('valid', 'uint32', pixelCount)}"
      ]
    }),
    preview: {kind: 'raster', seed: 81, caption: 'Signed bytes in, unit vectors out.'},
    related: ['raster-embedding-pyramid-level', 'raster-embedding-similarity'],
    reference: EMBEDDINGS_REFERENCE,
    maturity: 'experimental'
  },
  {
    id: 'raster-embedding-similarity',
    symbol: 'GPURasterEmbeddingSimilarity',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Embedding similarity map',
    summary:
      'Cosine or dot similarity of every pixel to up to 64 query vectors, with a match mask.',
    explanation:
      'Click a pixel or average a polygon (`GPURasterEmbeddingZoneMean`) to get a query, then map how alike every other pixel is. `matchThreshold` is per-frame, so the mask follows a slider without recompiling.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['index', 'find-patterns'],
    domains: ['remote-sensing'],
    params: [
      ...PIXEL_PARAMS,
      makeParam('queryCount', 'number', 'Query vectors, 1 to 64.', {required: true}),
      makeParam('queries', FLOAT, 'Query vectors, `query * dimensions + band`.', {required: true}),
      makeParam('metric', "'cosine' | 'dot'", 'Cosine divides by both lengths.', {
        default: "'cosine'"
      }),
      makeParam('parameters', FLOAT, 'Packed `matchThreshold`; needed with `matches`.')
    ],
    outputs: [
      makeOutput('similarity', FLOAT, 'Similarity to each query, `pixel * queryCount + query`.'),
      makeOutput('maximum', FLOAT, 'Best similarity over the queries; NaN for invalid pixels.'),
      makeOutput('bestQuery', UINT, 'Index of the most similar query.'),
      makeOutput('matches', UINT, '1 where `maximum >= matchThreshold`.')
    ],
    example: makeExample({
      symbols: ['getGPURasterEmbeddingSimilarityParameterValues', 'GPURasterEmbeddingSimilarity'],
      setup: [SHAPE_LINE],
      parameters: 'getGPURasterEmbeddingSimilarityParameterValues({matchThreshold: 0.9})',
      constructorSymbol: 'GPURasterEmbeddingSimilarity',
      props: [
        'pixelCount, dimensions, queryCount: 1,',
        VECTORS_PROP,
        "queries: view('query', 'float32', dimensions),",
        "output: {maximum: view('maximum', 'float32', pixelCount), matches: view('matches', 'uint32', pixelCount)}"
      ]
    }),
    preview: {kind: 'raster', seed: 82, caption: 'Places like the clicked pixel light up.'},
    related: [
      'raster-embedding-nearest-pixels',
      'raster-embedding-zone-mean',
      'raster-embedding-change'
    ],
    reference: EMBEDDINGS_REFERENCE,
    maturity: 'experimental'
  },
  {
    id: 'raster-embedding-nearest-pixels',
    symbol: 'GPURasterEmbeddingNearestPixels',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Nearest embedding pixels',
    summary:
      'The `resultCount` pixels most similar to each query vector, ranked by cosine similarity.',
    explanation:
      'Find places like this one as a ranked list instead of a heat map. It runs the exact `GPUSimilaritySearch` of luma.gl over the raster, so equal scores order by pixel index.',
    dataTypes: {input: ['raster'], output: ['table']},
    useCases: ['index', 'find-patterns'],
    domains: ['remote-sensing'],
    params: [
      ...PIXEL_PARAMS,
      makeParam('queryCount', 'number', 'Query vectors, 1 to 64.', {required: true}),
      makeParam('queries', FLOAT, 'Query vectors, `query * dimensions + band`.', {required: true}),
      makeParam('resultCount', 'number', 'Matches kept per query.', {required: true})
    ],
    outputs: [
      makeOutput(
        'pixelRows',
        UINT,
        'Source pixel row of each match, `query * resultCount + rank`.'
      ),
      makeOutput('scores', FLOAT, 'Cosine similarity of each match, best first.'),
      makeOutput('resultCounts', UINT, 'Matches found per query.')
    ],
    example: makeExample({
      symbols: ['GPURasterEmbeddingNearestPixels'],
      setup: [SHAPE_LINE, 'const resultCount = 100;'],
      constructorSymbol: 'GPURasterEmbeddingNearestPixels',
      props: [
        'pixelCount, dimensions, queryCount: 1, resultCount,',
        VECTORS_PROP,
        "queries: view('query', 'float32', dimensions),",
        'output: {',
        "  pixelRows: view('rows', 'uint32', resultCount),",
        "  scores: view('scores', 'float32', resultCount),",
        "  resultCounts: view('found', 'uint32', 1)",
        '}'
      ]
    }),
    preview: {kind: 'points', seed: 83, caption: 'The best matches for one pixel, anywhere.'},
    related: ['raster-embedding-similarity', 'raster-embedding-zone-mean'],
    reference: EMBEDDINGS_REFERENCE,
    maturity: 'experimental'
  },
  {
    id: 'raster-embedding-change',
    symbol: 'GPURasterEmbeddingChange',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Embedding change',
    summary:
      'Year-to-year change of an embedding raster: cosine similarity, angle, change length and a mask.',
    explanation:
      'Compare two years of embeddings pixel by pixel; a low similarity means the place changed. `similarityThreshold` is per-frame; pair the similarity with a false discovery rate mask for significant change only.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['detect-change'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('pixelCount', 'number', 'Pixels of each year, row-major.', {required: true}),
      makeParam('dimensions', 'number', 'Vector length, 1 to 128.', {required: true}),
      makeParam('before', FLOAT, 'Decoded vectors of the earlier year.', {required: true}),
      makeParam('after', FLOAT, 'Decoded vectors of the later year.', {required: true}),
      makeParam('beforeMask', UINT, 'QA of the earlier year; zero makes the pixel invalid.'),
      makeParam('afterMask', UINT, 'QA of the later year.'),
      makeParam('parameters', FLOAT, 'Packed `similarityThreshold`; needed with `changed`.')
    ],
    outputs: [
      makeOutput('similarity', FLOAT, 'Cosine similarity of the two years, -1 to 1.'),
      makeOutput('angle', FLOAT, 'Angle between the vectors in radians.'),
      makeOutput('magnitude', FLOAT, 'Length of the change vector `|after - before|`.'),
      makeOutput('changed', UINT, '1 where `similarity < similarityThreshold`.')
    ],
    example: makeExample({
      symbols: ['getGPURasterEmbeddingChangeParameterValues', 'GPURasterEmbeddingChange'],
      setup: [SHAPE_LINE],
      parameters: 'getGPURasterEmbeddingChangeParameterValues({similarityThreshold: 0.7})',
      constructorSymbol: 'GPURasterEmbeddingChange',
      props: [
        'pixelCount, dimensions,',
        "before: view('2023', 'float32', pixelCount * dimensions),",
        "after: view('2024', 'float32', pixelCount * dimensions),",
        "output: {similarity: view('similarity', 'float32', pixelCount), changed: view('changed', 'uint32', pixelCount)}"
      ]
    }),
    preview: {kind: 'raster', seed: 84, caption: 'Low similarity between years marks change.'},
    related: ['change-detection', 'raster-embedding-similarity'],
    reference: EMBEDDINGS_REFERENCE,
    maturity: 'experimental'
  },
  {
    id: 'raster-embedding-kmeans',
    symbol: 'GPURasterEmbeddingKMeans',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Embedding k-means',
    summary:
      'Unsupervised k-means (Lloyd) clustering of the per-pixel vectors of an embedding raster.',
    explanation:
      'A quick land-cover map with no labels: pixels with similar vectors share a cluster. Seeding is deterministic (evenly spaced valid rows); normalize the vectors first to cluster by direction.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['classify', 'find-patterns'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      ...PIXEL_PARAMS,
      makeParam('clusterCount', 'number', 'Number of clusters.', {required: true}),
      makeParam('maximumIterations', 'number', 'Lloyd iterations encoded in the graph, 1 to 64.', {
        default: '10'
      })
    ],
    outputs: [
      makeOutput('labels', UINT, 'Cluster of each pixel; `0xffffffff` for invalid pixels.'),
      makeOutput('centroids', FLOAT, 'Cluster centres, `cluster * dimensions + band`.'),
      makeOutput('counts', UINT, 'Pixels in each cluster.'),
      makeOutput('status', UINT, '`[executedIterations, changedLabels, converged]`.')
    ],
    example: makeExample({
      symbols: ['GPURasterEmbeddingKMeans'],
      setup: [SHAPE_LINE, 'const clusterCount = 8;'],
      constructorSymbol: 'GPURasterEmbeddingKMeans',
      props: [
        'pixelCount, dimensions, clusterCount, maximumIterations: 20,',
        VECTORS_PROP,
        'output: {',
        "  labels: view('labels', 'uint32', pixelCount),",
        "  centroids: view('centroids', 'float32', clusterCount * dimensions),",
        "  counts: view('counts', 'uint32', clusterCount)",
        '}'
      ]
    }),
    preview: {kind: 'classification', seed: 85, caption: 'Clusters found from the vectors alone.'},
    related: [
      'raster-embedding-knn-classifier',
      'raster-embedding-linear-probe',
      'k-means-clustering'
    ],
    reference: EMBEDDINGS_REFERENCE,
    maturity: 'experimental'
  },
  {
    id: 'raster-embedding-knn-classifier',
    symbol: 'GPURasterEmbeddingKNearestClassifier',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Embedding k-nearest classifier',
    summary:
      'Classifies every pixel by a vote of its nearest labelled vectors, like scikit-learn kNN.',
    explanation:
      'Label a few pixels per class and map the whole raster: each pixel takes the majority class of its `neighborCount` nearest labelled vectors. Cost grows with the labelled set times the pixels.',
    dataTypes: {input: ['raster', 'points'], output: ['raster']},
    useCases: ['classify'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('pixelCount', 'number', 'Pixels of the raster, row-major.', {required: true}),
      makeParam('vectors', FLOAT, 'Decoded vectors, `pixel * dimensions + band`.', {
        required: true
      }),
      makeParam('trainingVectors', FLOAT, 'Labelled vectors, `sample * dimensions + band`.', {
        required: true
      }),
      makeParam('trainingLabels', UINT, 'Class of each labelled vector.', {required: true}),
      makeParam('trainingCount', 'number', 'Labelled vectors, 1 to 4096.', {required: true}),
      makeParam('classCount', 'number', 'Classes, 1 to 32.', {required: true}),
      makeParam('neighborCount', 'number', 'Neighbours that vote, 1 to 16.', {required: true}),
      makeParam('metric', "'euclidean' | 'cosine'", 'Distance between vectors.', {
        default: "'euclidean'"
      })
    ],
    moreRequired: ['dimensions'],
    outputs: [
      makeOutput('labels', UINT, 'Predicted class per pixel; `0xffffffff` for invalid pixels.'),
      makeOutput('confidence', FLOAT, 'Vote share of the predicted class.'),
      makeOutput('probabilities', FLOAT, 'Vote share of every class, `pixel * classCount + class`.')
    ],
    example: makeExample({
      symbols: ['GPURasterEmbeddingKNearestClassifier'],
      setup: [SHAPE_LINE, 'const trainingCount = 300;'],
      constructorSymbol: 'GPURasterEmbeddingKNearestClassifier',
      props: [
        'pixelCount, dimensions, trainingCount, classCount: 5, neighborCount: 7,',
        VECTORS_PROP,
        "trainingVectors: view('samples', 'float32', trainingCount * dimensions),",
        "trainingLabels: view('sample-labels', 'uint32', trainingCount),",
        "output: {labels: view('labels', 'uint32', pixelCount), confidence: view('confidence', 'float32', pixelCount)}"
      ]
    }),
    preview: {
      kind: 'classification',
      seed: 86,
      caption: 'Each pixel votes among labelled neighbours.'
    },
    related: ['raster-embedding-linear-probe', 'raster-embedding-kmeans'],
    reference: EMBEDDINGS_REFERENCE,
    maturity: 'experimental'
  },
  {
    id: 'raster-embedding-linear-probe',
    symbol: 'GPURasterEmbeddingLinearProbe',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Embedding linear probe',
    summary:
      'Applies a host-trained logistic regression, `softmax(W x + b)`, to every pixel vector.',
    explanation:
      'Train a multinomial logistic regression off the GPU (scikit-learn `coef_` and `intercept_`) and map class probabilities over the whole raster. Weights are view contents, so a retrained model never recompiles.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['classify', 'model'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      ...PIXEL_PARAMS,
      makeParam('classCount', 'number', 'Classes, 1 to 64.', {required: true}),
      makeParam('weights', FLOAT, 'Weights, `class * dimensions + band` (`coef_`).', {
        required: true
      }),
      makeParam('bias', FLOAT, 'One intercept per class (`intercept_`).', {required: true})
    ],
    outputs: [
      makeOutput('labels', UINT, 'Argmax class per pixel; `0xffffffff` for invalid pixels.'),
      makeOutput('confidence', FLOAT, 'Probability of the predicted class.'),
      makeOutput(
        'probabilities',
        FLOAT,
        'Probability of every class, `pixel * classCount + class`.'
      )
    ],
    example: makeExample({
      symbols: ['GPURasterEmbeddingLinearProbe'],
      setup: [SHAPE_LINE, 'const classCount = 4;'],
      constructorSymbol: 'GPURasterEmbeddingLinearProbe',
      props: [
        'pixelCount, dimensions, classCount,',
        VECTORS_PROP,
        "weights: view('coef', 'float32', classCount * dimensions),",
        "bias: view('intercept', 'float32', classCount),",
        "output: {labels: view('labels', 'uint32', pixelCount), confidence: view('confidence', 'float32', pixelCount)}"
      ]
    }),
    preview: {
      kind: 'classification',
      seed: 87,
      caption: 'Class probabilities from a trained probe.'
    },
    related: ['raster-embedding-knn-classifier', 'raster-embedding-kmeans'],
    reference: EMBEDDINGS_REFERENCE,
    maturity: 'experimental'
  },
  {
    id: 'raster-embedding-covariance',
    symbol: 'GPURasterEmbeddingCovariance',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Embedding covariance',
    summary: 'Mean vector and sample covariance of the usable pixels, the GPU half of PCA.',
    explanation:
      'Two-pass, fixed-order sums make the result reproducible and stable even with a large mean. Feed the covariance to `getRasterEmbeddingPrincipalComponents` and project with `GPURasterEmbeddingProjection`.',
    dataTypes: {input: ['raster'], output: ['table']},
    useCases: ['summarize', 'measure'],
    domains: ['remote-sensing', 'data-engineering'],
    params: [
      ...PIXEL_PARAMS,
      makeParam('rangeCount', 'number', 'Pixel ranges reduced in parallel, 1 to 256.', {
        default: 'min(256, pixelCount)'
      })
    ],
    outputs: [
      makeOutput('mean', FLOAT, 'Mean vector, `dimensions` entries.'),
      makeOutput('covariance', FLOAT, 'Sample covariance, `dimensions x dimensions`, row-major.'),
      makeOutput('usableCount', UINT, 'Number of usable pixels.')
    ],
    example: makeExample({
      symbols: ['GPURasterEmbeddingCovariance'],
      setup: [SHAPE_LINE],
      constructorSymbol: 'GPURasterEmbeddingCovariance',
      props: [
        'pixelCount, dimensions,',
        VECTORS_PROP,
        "mask: view('valid', 'uint32', pixelCount),",
        "output: {mean: view('mean', 'float32', dimensions), covariance: view('covariance', 'float32', dimensions * dimensions)}"
      ]
    }),
    preview: {kind: 'histogram', seed: 88, caption: 'How the bands vary together inside a window.'},
    related: ['raster-embedding-projection'],
    reference: EMBEDDINGS_REFERENCE,
    maturity: 'experimental'
  },
  {
    id: 'raster-embedding-projection',
    symbol: 'GPURasterEmbeddingProjection',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Embedding projection',
    summary: 'Projects every vector on principal components and stretches the first three to RGB.',
    explanation:
      'The false-color view of an embedding: subtract the mean, project on the components, stretch. `getRasterEmbeddingStretchRanges(eigenvalues)` gives a two-sigma stretch, and the ranges are per-frame.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['transform', 'classify'],
    domains: ['remote-sensing', 'cartography'],
    params: [
      ...PIXEL_PARAMS,
      makeParam('componentCount', 'number', 'Components, 1 to 16; at least 3 for `rgb`.', {
        required: true
      }),
      makeParam('mean', FLOAT, 'Mean vector subtracted first.', {required: true}),
      makeParam('components', FLOAT, 'Unit components, `component * dimensions + band`.', {
        required: true
      }),
      makeParam('parameters', FLOAT, 'Packed stretch ranges; needed with `rgb`.')
    ],
    outputs: [
      makeOutput('projected', FLOAT, 'Scores, `pixel * componentCount + component`.'),
      makeOutput('rgb', FLOAT, 'First three scores stretched to 0 to 1, `pixel * 3 + channel`.')
    ],
    example: makeExample({
      symbols: [
        'getGPURasterEmbeddingProjectionParameterValues',
        'getRasterEmbeddingStretchRanges',
        'GPURasterEmbeddingProjection'
      ],
      setup: [
        SHAPE_LINE,
        'const {eigenvalues} = principalComponents; // getRasterEmbeddingPrincipalComponents'
      ],
      parameters:
        'getGPURasterEmbeddingProjectionParameterValues({stretchRanges: getRasterEmbeddingStretchRanges(eigenvalues)})',
      constructorSymbol: 'GPURasterEmbeddingProjection',
      props: [
        'pixelCount, dimensions, componentCount: 3,',
        VECTORS_PROP,
        "mean: view('mean', 'float32', dimensions),",
        "components: view('components', 'float32', 3 * dimensions),",
        "output: {rgb: view('rgb', 'float32', pixelCount * 3)}"
      ]
    }),
    preview: {kind: 'raster', seed: 89, caption: 'Principal components as a false-color image.'},
    related: ['raster-embedding-covariance', 'raster-embedding-kmeans'],
    reference: EMBEDDINGS_REFERENCE,
    maturity: 'experimental'
  },
  {
    id: 'raster-embedding-zone-mean',
    symbol: 'GPURasterEmbeddingZoneMean',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Embedding zone mean',
    summary: 'Mean embedding vector of every zone, from a zone raster or weighted coverage tuples.',
    explanation:
      'Summarize a polygon as one vector: a feature for a model or a query for similarity search. Fixed-point sums make it order-independent; `meanLengths` is low where a zone mixes unlike pixels.',
    dataTypes: {input: ['raster', 'polygons'], output: ['table']},
    useCases: ['summarize'],
    domains: ['remote-sensing', 'ecology'],
    params: [
      makeParam('pixelCount', 'number', 'Pixels of the raster, row-major.', {required: true}),
      makeParam('dimensions', 'number', 'Vector length, 1 to 128.', {required: true}),
      makeParam('zoneCount', 'number', 'Zones, 1 to 65536.', {required: true}),
      makeParam('vectors', FLOAT, 'Decoded vectors, `pixel * dimensions + band`.', {
        required: true
      }),
      makeParam('zoneRaster', UINT, 'Zone id per pixel; give this or `tuples`.'),
      makeParam('tuples', 'GPURasterEmbeddingZoneTuples', 'Weighted pixel-to-zone tuples.'),
      makeParam('normalize', 'boolean', 'Divide each mean by its length.', {default: 'true'}),
      makeParam('maximumComponentMagnitude', 'number', 'Largest absolute component.', {
        default: '1'
      })
    ],
    outputs: [
      makeOutput('means', FLOAT, 'Mean vector per zone, `zone * dimensions + band`.'),
      makeOutput('weights', FLOAT, 'Total coverage weight of the usable pixels per zone.'),
      makeOutput('meanLengths', FLOAT, 'Length of each mean before renormalization.')
    ],
    example: makeExample({
      symbols: ['GPURasterEmbeddingZoneMean'],
      setup: [SHAPE_LINE, 'const zoneCount = 120;'],
      constructorSymbol: 'GPURasterEmbeddingZoneMean',
      props: [
        'pixelCount, dimensions, zoneCount,',
        VECTORS_PROP,
        "zoneRaster: view('zones', 'uint32', pixelCount),",
        "output: {means: view('means', 'float32', zoneCount * dimensions), meanLengths: view('lengths', 'float32', zoneCount)}"
      ]
    }),
    preview: {kind: 'polygons', seed: 90, caption: 'One vector per zone; mixed zones are short.'},
    related: ['raster-embedding-similarity', 'fractional-coverage'],
    reference: EMBEDDINGS_REFERENCE,
    maturity: 'experimental'
  },
  {
    id: 'raster-embedding-pyramid-level',
    symbol: 'GPURasterEmbeddingPyramidLevel',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Embedding pyramid level',
    summary: 'One overview level of an embedding raster: renormalized mean of each footprint.',
    explanation:
      'Decode and sum original vectors, then normalize the displayed direction; meanLength measures coherence. Chain levels using aggregation sums and counts independently of vector validity: [east, west] retains count 2 and sum [0, 0] with direction validity 0. Each output supports at most 2^32 - 1 original pixels.',
    dataTypes: {input: ['raster'], output: ['raster']},
    useCases: ['transform', 'summarize'],
    domains: ['remote-sensing', 'data-engineering'],
    params: [
      makeParam('sourceWidth', 'number', 'Source width in pixels.', {required: true}),
      makeParam('sourceHeight', 'number', 'Source height in pixels.', {required: true}),
      makeParam('dimensions', 'number', 'Vector length, 1 to 128.', {required: true}),
      makeParam(
        'scale',
        'number | readonly [number, number]',
        'Source pixels per overview pixel, 1 to 16.',
        {
          required: true
        }
      ),
      makeParam(
        'vectors',
        FLOAT,
        'Decoded vectors; give exactly one of vectors, quantized, aggregation.'
      ),
      makeParam('quantized', UINT, 'Packed int8 AlphaEarth bytes, decoded in the kernel.'),
      makeParam(
        'aggregation',
        `{vectorSums: ${FLOAT}; validCount: ${UINT}}`,
        'Original sums/counts retain cancelling-child support; ignore direction validity.'
      ),
      makeParam('mask', UINT, 'Source QA; zero excludes the pixel.')
    ],
    outputs: [
      makeOutput(
        'vectors',
        FLOAT,
        'Unit-length overview vectors; NaN for empty or cancelling footprints.'
      ),
      makeOutput(
        'vectorSums',
        FLOAT,
        'Unnormalized vector sums; pair with validCount to chain levels.'
      ),
      makeOutput(
        'validity',
        UINT,
        '1 for nonempty support with a nonzero sum and usable direction; 0 if cancelled.'
      ),
      makeOutput(
        'validCount',
        UINT,
        'Valid original pixels, including support whose vectors cancel.'
      ),
      makeOutput('meanLength', FLOAT, 'Length of the mean before renormalization, 0 to 1.')
    ],
    example: makeExample({
      symbols: ['GPURasterEmbeddingPyramidLevel'],
      setup: ['const dimensions = 64; const overviewCount = 256 * 256;'],
      constructorSymbol: 'GPURasterEmbeddingPyramidLevel',
      props: [
        'sourceWidth: 512, sourceHeight: 512, dimensions, scale: 2,',
        "quantized: view('bytes', 'uint32', (512 * 512 * dimensions) / 4),",
        "output: {vectors: view('overview', 'float32', overviewCount * dimensions), meanLength: view('agreement', 'float32', overviewCount)}"
      ]
    }),
    preview: {kind: 'raster', seed: 91, caption: 'Coarser pixels average the vectors under them.'},
    related: ['raster-embedding-dequantize', 'extrema-pyramid'],
    reference: EMBEDDINGS_REFERENCE,
    maturity: 'experimental'
  },

  // Lagrangian diagnostics ----------------------------------------------------------------------
  {
    id: 'lagrangian-flow-map',
    symbol: 'GPULagrangianFlowMap',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Lagrangian flow map',
    summary: 'Carries every seed through a time-varying vector field and writes where it ends up.',
    explanation:
      'The input of FTLE, connectivity matrices and residence times. RK2 steps with bilinear sampling and linear time interpolation between two slices; a negative `timeStep` integrates backward.',
    dataTypes: {input: ['raster', 'points'], output: ['points']},
    useCases: ['simulate'],
    domains: ['hydrology', 'ecology'],
    params: [
      makeParam('seeds', POINTS, 'Seed positions in the field frame.', {required: true}),
      makeParam('velocitiesStart', POINTS, 'Field at `sliceStartTime`, row-major.', {
        required: true
      }),
      makeParam('velocitiesEnd', POINTS, 'Field at `sliceEndTime`, same layout.', {required: true}),
      makeParam('fieldWidth', 'number', 'Field width in cells.', {required: true}),
      makeParam('fieldHeight', 'number', 'Field height in cells.', {required: true}),
      makeParam('stepCount', 'number', 'RK2 steps per particle, 1 to 8192.', {required: true}),
      makeParam('parameters', FLOAT, 'Packed extent, start time, `timeStep`, slice times.', {
        required: true
      }),
      makeParam('regions', UINT, 'Region per field cell, for region classes and residence.')
    ],
    outputs: [
      makeOutput('positions', POINTS, 'Final position per seed; NaN once a particle is lost.'),
      makeOutput('regionClasses', UINT, 'Start then end region per seed (needs `regions`).'),
      makeOutput('residenceTimes', FLOAT, 'Time in the start region before the first exit.')
    ],
    example: makeExample({
      symbols: ['getGPULagrangianFlowMapParameterValues', 'GPULagrangianFlowMap'],
      setup: ['const seedCount = 256 * 256; const cellCount = 360 * 180;'],
      parameters:
        'getGPULagrangianFlowMapParameterValues({fieldPlacement: {origin: [0, 0], cellSize: [1, 1]}, startTime: 0, timeStep: -0.1, sliceStartTime: 0, sliceEndTime: 6})',
      constructorSymbol: 'GPULagrangianFlowMap',
      props: [
        'fieldWidth: 360, fieldHeight: 180, stepCount: 60,',
        "seeds: view('seeds', 'float32x2', seedCount),",
        "velocitiesStart: view('field-0', 'float32x2', cellCount),",
        "velocitiesEnd: view('field-6', 'float32x2', cellCount),",
        "output: {positions: view('ends', 'float32x2', seedCount)}"
      ]
    }),
    preview: {kind: 'trajectories', seed: 92, caption: 'Where each seed ends up after time T.'},
    related: ['lagrangian-finite-time-lyapunov-exponent', 'particle-advection', 'streamlines'],
    reference: LAGRANGIAN_REFERENCE,
    maturity: 'experimental'
  },
  {
    id: 'lagrangian-finite-time-lyapunov-exponent',
    symbol: 'GPUFiniteTimeLyapunovExponent',
    kind: 'class',
    importPath: IMPORT_PATH,
    title: 'Finite-time Lyapunov exponent',
    summary:
      'FTLE of a flow map on a seed grid: how fast neighbouring particles separate over time T.',
    explanation:
      'Ridges of the forward FTLE are repelling Lagrangian coherent structures; a backward flow map gives attracting ones. Feed it the final positions of a regular seed grid from `GPULagrangianFlowMap`.',
    dataTypes: {input: ['points'], output: ['raster']},
    useCases: ['find-patterns', 'simulate'],
    domains: ['hydrology', 'ecology'],
    params: [
      makeParam('positions', POINTS, 'Final positions of a row-major seed grid.', {required: true}),
      makeParam('seedWidth', 'number', 'Seed grid width, 2 to 65535.', {required: true}),
      makeParam('seedHeight', 'number', 'Seed grid height, 2 to 65535.', {required: true}),
      makeParam('parameters', FLOAT, 'Packed seed spacing and `integrationTime`.', {required: true})
    ],
    outputs: [makeOutput('ftle', FLOAT, 'FTLE per seed; NaN where a neighbour was lost.')],
    example: makeExample({
      symbols: ['getGPUFiniteTimeLyapunovExponentParameterValues', 'GPUFiniteTimeLyapunovExponent'],
      setup: [
        'const seedWidth = 256; const seedHeight = 256; const seedCount = seedWidth * seedHeight;'
      ],
      parameters:
        'getGPUFiniteTimeLyapunovExponentParameterValues({seedSpacingX: 0.25, seedSpacingY: 0.25, integrationTime: -6})',
      constructorSymbol: 'GPUFiniteTimeLyapunovExponent',
      props: [
        'seedWidth, seedHeight,',
        "positions: view('flow-map', 'float32x2', seedCount),",
        "output: {ftle: view('ftle', 'float32', seedCount)}"
      ]
    }),
    preview: {kind: 'raster', seed: 93, caption: 'Bright ridges split or gather the flow.'},
    related: ['lagrangian-flow-map', 'line-integral-convolution', 'streamlines'],
    reference: LAGRANGIAN_REFERENCE,
    maturity: 'experimental'
  }
];
