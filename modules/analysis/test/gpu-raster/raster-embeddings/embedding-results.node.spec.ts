// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {getGPURasterEmbeddingSimilarityParameterValues} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-similarity';
import {getGPURasterEmbeddingChangeParameterValues} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-change';
import {getGPURasterEmbeddingProjectionParameterValues} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-projection';
import {
  extractSnippets,
  getSnippetDiagnostics
} from '../../analysis-core/doc-snippet-diagnostics.test-utils';

it('PCK-16 embedding packers require full destinations and preserve padded tails', () => {
  const writers = [
    {
      length: 1,
      expected: [0.5],
      write: (target: Float32Array) =>
        getGPURasterEmbeddingSimilarityParameterValues({matchThreshold: 0.5}, target)
    },
    {
      length: 1,
      expected: [0.5],
      write: (target: Float32Array) =>
        getGPURasterEmbeddingChangeParameterValues({similarityThreshold: 0.5}, target)
    },
    {
      length: 6,
      expected: [-1, 2, -3, 4, -5, 6],
      write: (target: Float32Array) =>
        getGPURasterEmbeddingProjectionParameterValues(
          {
            stretchRanges: [
              [-1, 2],
              [-3, 4],
              [-5, 6]
            ]
          },
          target
        )
    }
  ];
  for (const writer of writers) {
    for (const length of new Set([0, writer.length - 1]))
      expect(() => writer.write(new Float32Array(length))).toThrow(/target/);
    expect(Array.from(writer.write(new Float32Array(writer.length)))).toEqual(writer.expected);
    const padded = new Float32Array(writer.length + 2).fill(42);
    expect(writer.write(padded)).toBe(padded);
    expect(Array.from(padded)).toEqual([...writer.expected, 42, 42]);
  }
});
it('PCK-16 rejected third stretch preserves every destination byte', () => {
  const target = new Float32Array([11, 12, 13, 14, 15, 16, 17]);
  const before = new Uint8Array(target.buffer).slice();
  expect(() =>
    getGPURasterEmbeddingProjectionParameterValues(
      {
        stretchRanges: [
          [-1, 1],
          [-2, 2],
          [1, 0]
        ]
      },
      target
    )
  ).toThrow();
  expect(new Uint8Array(target.buffer)).toEqual(before);
});
it('W3-07 embedding documentation snippets have strict public bindings', () => {
  expect(getSnippetDiagnostics(extractSnippets('gpu-raster/raster-embeddings.md'))).toEqual([]);
});

it('STA-11 status decoder accepts offset byte reads and rejects missing words', async () => {
  const {decodeGPURasterEmbeddingKMeansStatus: decode} = await import(
    '@luma-contributors/analysis/raster-analysis'
  );
  const words = new Uint32Array([77, 1, 5, 0, 88]);
  expect(decode(words.subarray(1, 4))).toEqual({
    executedIterations: 1,
    changedLabels: 5,
    converged: false
  });
  expect(decode(new Uint8Array(words.buffer, 4, 12))).toEqual({
    executedIterations: 1,
    changedLabels: 5,
    converged: false
  });
  expect(decode(new Uint32Array([2, 0, 1]).buffer).converged).toBe(true);
  for (const length of [0, 8, 13]) expect(() => decode(new Uint8Array(length))).toThrow(/status/);
});

it('embedding numerical fixes generate auditable WGSL and expose cached KMeans stopping storage', async () => {
  const {createKernelAuditContext} = await import(
    '../../analysis-core/kernel-audit/kernel-audit-context'
  );
  const {getGPUCommandNodeKernelSources} = await import('../../utils/gpu-contributor-test-utils');
  const {auditWGSLModule} = await import('../../analysis-core/kernel-audit/kernel-audit-wgsl');
  const {GPURasterEmbeddingSimilarity} = await import(
    '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-similarity'
  );
  const {GPURasterEmbeddingCovariance} = await import(
    '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-covariance'
  );
  const {GPURasterEmbeddingProjection} = await import(
    '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-projection'
  );
  const {GPURasterEmbeddingKMeans} = await import(
    '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-kmeans'
  );
  const context = createKernelAuditContext('embedding-results');
  const {graph, view} = context;
  try {
    const contributors = [
      new GPURasterEmbeddingSimilarity({
        pixelCount: 1,
        dimensions: 1,
        queryCount: 2,
        vectors: view('float32', 1),
        queries: view('float32', 2),
        output: {maximum: view('float32', 1)}
      }),
      new GPURasterEmbeddingCovariance({
        pixelCount: 256,
        dimensions: 1,
        vectors: view('float32', 256),
        output: {mean: view('float32', 1), covariance: view('float32', 1)}
      }),
      new GPURasterEmbeddingProjection({
        pixelCount: 1,
        dimensions: 3,
        componentCount: 3,
        vectors: view('float32', 3),
        mean: view('float32', 3),
        components: view('float32', 9),
        parameters: view('float32', 6),
        output: {rgb: view('float32', 3)}
      })
    ];
    for (const contributor of contributors) {
      const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      expect(sources.length).toBeGreaterThan(0);
      for (const source of sources)
        expect(auditWGSLModule(source.source, source.constants)).toEqual([]);
      if (contributor instanceof GPURasterEmbeddingCovariance) {
        const anchorSource = sources.find(source => source.id.endsWith('-anchor'))!.source;
        expect(anchorSource).not.toContain('pixel < 256u');
        const meanSource = sources.find(source => source.id.endsWith('-mean'))!.source;
        expect(meanSource).toContain('addCompensatedValue');
      }
    }
    const clustering = new GPURasterEmbeddingKMeans({
      pixelCount: 5,
      dimensions: 1,
      clusterCount: 2,
      maximumIterations: 1,
      vectors: view('float32', 5),
      output: {labels: view('uint32', 5), centroids: view('float32', 2), counts: view('uint32', 2)}
    });
    const result = clustering.getResult(graph);
    expect(result.status.length).toBe(3);
    expect(clustering.getResult(graph)).toBe(result);
    expect(clustering.getCommandNodes(graph)).toBe(clustering.getCommandNodes(graph));
  } finally {
    context.device.destroy();
  }
});
