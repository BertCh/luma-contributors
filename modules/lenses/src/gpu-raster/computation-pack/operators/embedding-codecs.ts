// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {RasterOperatorTable} from '../types';

/** AlphaEarth encoding of caller-supplied embedding components. */
export const RASTER_EMBEDDING_CODEC_OPERATORS: RasterOperatorTable = {
  GPURasterEmbeddingQuantize: {
    family: 'embedding-codecs',
    descriptor: {scope: 'local', reduction: 'none'},
    variables: {vectors: 'score', quantized: 'category', validity: 'category'},
    stages: ['representation', 'operator'],
    datumKinds: [{kind: 'pixel', label: 'embedding pixel', indexSpace: '1d'}],
    neighbourhood: 'none',
    guards: [],
    choices: [{stage: 'representation', name: 'dimensions', sweepable: false}],
    formula: {
      id: 'gpu-raster.embedding-quantize',
      derivation: 'curated',
      tex: String.raw`\term{code}{q}=\operatorname{clip}(\operatorname{roundEven}(127.5\operatorname{sign}(\term{value}{v})\sqrt{|\term{value}{v}|}),-127,127)`,
      ascii:
        'Nearest-even signed square-root encoding, saturated to [-127,127], four bytes per uint32 low byte first. Masked or nonfinite pixels write -128 in every component; unused final lanes are zero.',
      symbols: [
        {id: 'code', role: 'intermediate', binding: 'wgsl:code'},
        {id: 'value', role: 'input', binding: 'wgsl:value'}
      ]
    },
    sources: [
      'modules/analysis/src/raster-analysis/raster-embeddings/gpu-raster-embedding-quantize.ts'
    ]
  }
};
