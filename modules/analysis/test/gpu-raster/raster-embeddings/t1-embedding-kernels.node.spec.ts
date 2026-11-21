// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {GPURasterEmbeddingSimilarity} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-similarity';
import {GPURasterEmbeddingCovariance} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-covariance';
import {createKernelAuditContext} from '../../analysis-core/kernel-audit/kernel-audit-context';
import {getGPUCommandNodeKernelSources} from '../../utils/gpu-contributor-test-utils';

it.each([
  'cosine',
  'dot'
] as const)('t1 similarity %s parses with all four output bindings', metric => {
  const {device, graph, view} = createKernelAuditContext(`similarity-${metric}`);
  try {
    const contributor = new GPURasterEmbeddingSimilarity({
      pixelCount: 1,
      dimensions: 1,
      queryCount: 2,
      metric,
      vectors: view('float32', 1),
      queries: view('float32', 2),
      parameters: view('float32', 1),
      output: {
        similarity: view('float32', 2),
        maximum: view('float32', 1),
        bestQuery: view('uint32', 1),
        matches: view('uint32', 1)
      }
    });
    for (const {source} of getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))) {
      expect(source.includes('${')).toBe(false);
      expect(() => new WgslReflect(source)).not.toThrow();
    }
  } finally {
    device.destroy();
  }
});

it('t1 covariance centers anchor-relative values through integer significands before subtracting the mean', () => {
  const {device, graph, view} = createKernelAuditContext('covariance-centering');
  try {
    const contributor = new GPURasterEmbeddingCovariance({
      pixelCount: 256,
      dimensions: 1,
      vectors: view('float32', 256),
      output: {mean: view('float32', 1), covariance: view('float32', 1)}
    });
    const source = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(kernel =>
      kernel.id.endsWith('-partial-covariance')
    )!.source;
    expect(source.includes('fn getEmbeddingAnchorDifference(')).toBe(true);
    expect(source.match(/getEmbeddingAnchorDifference\(vectors\[/g)).toHaveLength(2);
    expect(() => new WgslReflect(source)).not.toThrow();
  } finally {
    device.destroy();
  }
});
