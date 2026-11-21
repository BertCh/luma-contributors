// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  planRasterEmbeddingChunks,
  type RasterEmbeddingScalarView,
  type RasterEmbeddingView
} from '../../../src/raster-analysis/raster-embeddings/raster-embedding-chunks';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {createVectorView} from '../../utils/gpu-contributor-test-utils';
import {GPURasterEmbeddingQuantize} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-quantize';

it('S1 rejects packed output topology through the public quantize constructor before alias checks', () => {
  const {device, view} = createKernelAuditContext('quantize-topology');
  try {
    expect(
      () =>
        new GPURasterEmbeddingQuantize({
          id: 'study-quantize',
          pixelCount: 5,
          dimensions: 4,
          vectors: createVectorView('float-input', 'float32', [view('float32', 20)]),
          output: {
            quantized: createVectorView('packed-output', 'uint32', [
              view('uint32', 3),
              view('uint32', 2)
            ])
          }
        })
    ).toThrowError(
      expect.objectContaining({
        id: 'study-quantize',
        path: 'output.quantized',
        message: expect.stringMatching(
          /^study-quantize output.quantized must match vectors chunk topology/
        )
      })
    );
  } finally {
    device.destroy();
  }
});

it.each([
  'parent',
  'chunk'
] as const)('CCK-02 rejects a wrong %s format for float vectors and packed words', level => {
  const {device, view} = createKernelAuditContext('embedding-format');
  try {
    for (const name of ['vectors', 'output.quantized'] as const) {
      const expectedFormat = name === 'vectors' ? 'float32' : 'uint32';
      const wrongFormat = expectedFormat === 'float32' ? 'uint32' : 'float32';
      const vectors = view('float32', 8);
      const quantized = view('uint32', 2);
      // Deliberately inconsistent metadata: neither the parent nor a child is authoritative.
      const chunk = view(expectedFormat, name === 'vectors' ? 8 : 2);
      const malformed = createVectorView(`malformed-${name}`, expectedFormat, [
        chunk
      ]) as RasterEmbeddingScalarView;
      Object.defineProperty(level === 'parent' ? malformed : chunk, 'format', {value: wrongFormat});
      expect(
        () =>
          planRasterEmbeddingChunks(
            'embedding-format',
            2,
            {
              vectors: {
                view: name === 'vectors' ? malformed : vectors,
                format: 'float32',
                components: 4
              },
              'output.quantized': {
                view: name === 'output.quantized' ? malformed : quantized,
                format: 'uint32',
                components: 4,
                componentsPerWord: 4
              }
            },
            'vectors',
            2
          ),
        name
      ).toThrow(new RegExp(expectedFormat));
    }
  } finally {
    device.destroy();
  }
});

it('CCK-02 preserves complete records, matching topology and native word alignment', () => {
  const {device, view} = createKernelAuditContext('embedding-records');
  try {
    const rows = <Format extends 'float32' | 'uint32'>(format: Format, lengths: number[]) =>
      createVectorView(
        `rows-${format}`,
        format,
        lengths.map(length => view(format, length))
      );
    const plan = (
      vectors: RasterEmbeddingView<'float32'>,
      quantized: RasterEmbeddingView<'uint32'>,
      dimensions = 4
    ) =>
      planRasterEmbeddingChunks(
        'embedding-records',
        2,
        {
          vectors: {view: vectors, format: 'float32', components: dimensions},
          'output.quantized': {
            view: quantized,
            format: 'uint32',
            components: dimensions,
            componentsPerWord: 4
          }
        },
        'vectors',
        2
      );
    const chunks = plan(rows('float32', [4, 4]), rows('uint32', [1, 1]));
    expect(chunks.map(chunk => chunk.pixelCount)).toEqual([1, 1]);
    expect(chunks.map(chunk => chunk.views['output.quantized'].format)).toEqual([
      'uint32',
      'uint32'
    ]);
    expect(() => plan(rows('float32', [4, 4]), rows('uint32', [2]))).toThrow(/chunk topology/);
    expect(() => plan(rows('float32', [3, 3]), rows('uint32', [1, 1]), 3)).toThrow(
      /packed-word alignment/
    );
    expect(() => plan(rows('float32', [3, 5]), rows('uint32', [1, 1]))).toThrow(/pixelCount/);
  } finally {
    device.destroy();
  }
});
