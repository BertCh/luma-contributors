// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterEmbeddingOperation} from '../analysis/raster-embeddings';
import type {
  KeplerOperationDescriptor,
  KeplerOperationOutputDescriptor
} from './operation-descriptor-types';

// These catalog bindings do not consume loaded rasters, queries, training data or zones.
// Keep their empty input/requirement/parameter contracts: the separate raster tools own those
// capabilities. Notes travel with result reports as well as appearing before execution.
const EMBEDDING_FIXTURE_NOTE =
  'Generated fixture only: an 8 × 8 four-band unit-vector raster with the last pixel masked. This catalog demonstration accepts no loaded datasets; use Raster embedding tools for loaded multiband raster tiles.';
const TABLE_NOTE =
  'The result is a native table without geometry or georeferencing. Its index identifies an output row, not a geographic location. Nonfinite values and invalid class labels become null; validity flags retain zero and packed words retain every uint32 value.';
const PROTOTYPE_NOTE =
  'The two illustrative prototypes are + and - the first-band unit vector, labelled 0 and 1. They are not learned land-cover classes or a trained earth model.';

type EmbeddingDemonstration = {
  operation: RasterEmbeddingOperation;
  question: KeplerOperationDescriptor['questions'][number];
  action: string;
  presentation?: KeplerOperationOutputDescriptor['presentation'];
  reading: string;
  next: NonNullable<KeplerOperationOutputDescriptor['next']>;
};

// Columns and layouts match runKeplerRasterEmbedding and createKeplerEmbeddingDemoOptions.
// Fixed settings mentioned here describe that fixture; they are not invented form parameters.
const EMBEDDING_DEMONSTRATIONS: readonly EmbeddingDemonstration[] = [
  {
    operation: 'kmeans',
    question: 'which-groups-or-regions',
    action: 'Group fixture embedding pixels',
    presentation: {column: 'label', scale: 'categorical'},
    reading:
      'Each row is a pixel; label is one of two unsupervised embedding clusters after at most 32 iterations. Cluster numbers are arbitrary and do not identify land-cover classes.',
    next: ['how-related', 'what-changed']
  },
  {
    operation: 'nearest-prototype',
    question: 'which-groups-or-regions',
    action: 'Match fixture pixels to prototypes',
    presentation: {column: 'prototype', scale: 'categorical'},
    reading:
      'Each pixel reports the best prototype, its maximum cosine similarity and a matches flag at the fixed 0.9 threshold. A best prototype alone does not imply that the threshold was met.',
    next: ['how-related', 'what-changed']
  },
  {
    operation: 'nearest-pixels',
    question: 'how-related',
    action: 'Rank fixture pixels by similarity',
    presentation: {column: 'score', scale: 'diverging', center: 0},
    reading:
      'The 16 rows are eight ranked matches for each of two query vectors. query, rank and pixel_id identify the query, zero-based rank and source pixel; score is cosine similarity, not geographic proximity.',
    next: ['which-groups-or-regions', 'what-changed']
  },
  {
    operation: 'knn',
    question: 'which-groups-or-regions',
    action: 'Classify fixture embedding pixels',
    presentation: {column: 'label', scale: 'categorical'},
    reading:
      'Each pixel reports the class of its single nearest training vector by cosine similarity and a confidence value. With only one neighbour, confidence is not a calibrated probability or an accuracy estimate.',
    next: ['how-related', 'what-changed']
  },
  {
    operation: 'linear-probe',
    question: 'which-groups-or-regions',
    action: 'Score fixture prototype classes',
    presentation: {column: 'label', scale: 'categorical'},
    reading:
      'Each pixel reports a label and softmax confidence from fixed first-band weights with zero bias. This demonstration applies illustrative weights; it does not fit or validate a model.',
    next: ['how-related', 'what-changed']
  },
  {
    operation: 'change',
    question: 'what-changed',
    action: 'Compare fixture embedding vectors',
    presentation: {column: 'changed', scale: 'categorical'},
    reading:
      'The comparison raster flips the first band on the right half of the fixture. Each pixel reports cosine similarity, angle, vector-difference magnitude and a changed flag using a fixed 0.9 similarity threshold. This is a constructed change, not an observed time series.',
    next: ['how-related', 'which-groups-or-regions']
  },
  {
    operation: 'covariance',
    question: 'how-related',
    action: 'Measure fixture band covariance',
    presentation: {column: 'covariance', scale: 'diverging', center: 0},
    reading:
      'The 16 rows flatten the 4 × 4 sample covariance matrix in row-major band order: index = firstBand * 4 + secondBand, with zero-based bands. The matrix uses the 63 usable pixels; rows are matrix entries, not pixels.',
    next: ['prepare-data', 'which-groups-or-regions']
  },
  {
    operation: 'pca',
    question: 'prepare-data',
    action: 'Project fixture principal components',
    presentation: {column: 'component_1', scale: 'diverging', center: 0},
    reading:
      'Each pixel reports three centered principal-component scores and three stretched RGB channels. GPU covariance, the library CPU eigensolver and GPU projection form this demonstration; component signs are arbitrary and RGB values are a display stretch.',
    next: ['which-groups-or-regions', 'how-related']
  },
  {
    operation: 'projection',
    question: 'prepare-data',
    action: 'Display fixture embedding components',
    presentation: {column: 'component_1', scale: 'diverging', center: 0},
    reading:
      'This binding runs the same three-component PCA path as the PCA demonstration, including centered component scores and stretched RGB channels. It does not accept a user-supplied projection matrix.',
    next: ['which-groups-or-regions', 'how-related']
  },
  {
    operation: 'dequantize',
    question: 'prepare-data',
    action: 'Decode fixture embedding roundtrip',
    presentation: {column: 'band_1', scale: 'diverging', center: 0},
    reading:
      'The fixture is first quantized to packed AlphaEarth signed bytes, then decoded and normalized. Rows are pixels with band_1 through band_4 and usable; this is a lossy codec roundtrip, not implicit decoding of ordinary float bands.',
    next: ['how-related', 'which-groups-or-regions']
  },
  {
    operation: 'quantize',
    question: 'prepare-data',
    action: 'Encode fixture embedding vectors',
    reading:
      'Each row contains one packed_word holding four little-endian signed bytes. The four-band fixture produces 64 words; these unsigned integers are codec storage, not scalar measurements or class labels. The value 4294967295 is preserved as a valid packed word.',
    next: ['prepare-data', 'how-related']
  },
  {
    operation: 'pyramid',
    question: 'prepare-data',
    action: 'Aggregate fixture embedding blocks',
    presentation: {column: 'mean_length', scale: 'sequential'},
    reading:
      'A fixed scale of 2 produces 16 overview rows on a 4 × 4 grid. Each contains a normalized four-band mean, usable, valid_count and mean_length before normalization. The last block has three usable source pixels; this reduces resolution rather than creating new observations.',
    next: ['which-groups-or-regions', 'how-related']
  },
  {
    operation: 'similarity',
    question: 'how-related',
    action: 'Measure fixture prototype similarity',
    presentation: {column: 'similarity', scale: 'diverging', center: 0},
    reading:
      'Each pixel reports maximum cosine similarity, the best prototype and a matches flag at a fixed 0.9 threshold. This is the same contributor path as nearest prototype; similarity is not a calibrated class probability.',
    next: ['which-groups-or-regions', 'what-changed']
  },
  {
    operation: 'zone-mean',
    question: 'how-much-per-area',
    action: 'Summarize fixture embedding halves',
    presentation: {column: 'mean_length', scale: 'sequential'},
    reading:
      'The two rows summarize the left and right halves of the raster, not loaded polygons. index is the zero-based zone ID, band_1 through band_4 are normalized means, weight counts usable pixels (32 and 31), and mean_length measures directional agreement before normalization; it is not a population total or a rate per area.',
    next: ['how-related', 'what-changed']
  },
  {
    operation: 'usable-mask',
    question: 'prepare-data',
    action: 'Inspect fixture vector validity',
    presentation: {column: 'usable', scale: 'categorical'},
    reading:
      'Each pixel reports the public quantizer validity output: usable is 1 for an admitted vector and 0 for the masked last pixel. Zero is retained as a validity flag, not converted to a missing result; this is not a cloud or land-cover classifier.',
    next: ['prepare-data', 'which-groups-or-regions']
  }
];

/** Specialist catalog demonstrations, explicitly separate from analyses of loaded raster tiles. */
export const RASTER_EMBEDDING_OPERATION_DESCRIPTORS: readonly KeplerOperationDescriptor[] = [
  ...EMBEDDING_DEMONSTRATIONS.map(
    ({operation, question, action, presentation, reading, next}): KeplerOperationDescriptor => ({
      bindingId: `raster-embedding-${operation}`,
      questions: [question],
      action,
      prominence: 3,
      inputs: [],
      parameters: [],
      requirements: [],
      outputs: [
        {
          tableId: 'embedding-result',
          kind: 'summary',
          ...(presentation ? {presentation} : {}),
          interpretation: {
            summary:
              presentation && presentation.scale !== 'categorical' ? 'value-range' : 'row-count',
            cautions: []
          },
          next
        }
      ],
      notes: [
        EMBEDDING_FIXTURE_NOTE,
        TABLE_NOTE,
        reading,
        ...(['nearest-prototype', 'nearest-pixels', 'knn', 'linear-probe', 'similarity'].includes(
          operation
        )
          ? [PROTOTYPE_NOTE]
          : [])
      ]
    })
  ),
  {
    bindingId: 'raster-lagrangian-ftle',
    questions: ['how-it-moves'],
    action: 'Trace generated strain flow',
    prominence: 3,
    inputs: [],
    parameters: [
      // Both integration and FTLE duration use parameter buffers; the seed grid and ten-step
      // graph remain fixed. Hosts still own scheduling and rerunning those passes.
      {name: 'timeStep', defaultPolicy: {from: 'constant'}, level: 'basic', apply: 'live'}
    ],
    requirements: [],
    outputs: [
      {
        tableId: 'lagrangian-result',
        kind: 'summary',
        presentation: {column: 'ftle', scale: 'sequential'},
        interpretation: {summary: 'value-range', cautions: []},
        next: ['shape-of-the-surface', 'what-changed']
      }
    ],
    notes: [
      'Generated fixture only: an 8 × 8 cell-centered planar strain field with u = 0.02 x and v = -0.02 y metres/second. This catalog demonstration accepts no loaded datasets; use Raster embedding and flow tools for native Flow Field grids.',
      'Ten RK2 steps integrate 36 seeds on a 6 × 6 inset lattice. The constant default timeStep is 0.1 seconds (one second total); a negative value integrates backward and zero duration is refused before GPU allocation.',
      'Each row identifies a seed_index and reports final_x and final_y in local metres and ftle in inverse seconds. FTLE measures finite-time stretching, not velocity or statistical significance. The analytic strain rate is 0.02 per second; lost seeds or invalid derivative stencils become null.',
      'The result is a native table without geometry or georeferencing. Coordinates belong to the generated local frame; they are not longitude and latitude.'
    ]
  }
];
