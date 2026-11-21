// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPURasterEmbeddingKNearestClassifier} from '../../../src/raster-analysis/raster-embeddings/gpu-raster-embedding-knn-classifier';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';

for (const metric of ['euclidean', 'cosine'] as const) {
  it(`RT-141 ${metric} shares arithmetic across one brute and two tile dispatches`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const floats = (length: number) =>
      createTransientView(graph, `float-${serial++}`, 'float32', length);
    const integers = (length: number) =>
      createTransientView(graph, `integer-${serial++}`, 'uint32', length);
    const vectors = floats(2);
    const trainingVectors = floats(516);
    const trainingLabels = integers(258);
    const fragments: string[] = [];
    try {
      for (const searchStructure of ['brute-force', 'tiled'] as const) {
        const contributor = new GPURasterEmbeddingKNearestClassifier({
          id: searchStructure,
          pixelCount: 1,
          dimensions: 2,
          vectors,
          trainingVectors,
          trainingLabels,
          trainingCount: 258,
          classCount: 3,
          neighborCount: 4,
          metric,
          searchStructure,
          output: {labels: integers(1), confidence: floats(1), probabilities: floats(3)}
        });
        const sources = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).map(
          kernel => kernel.source
        );
        const scans = sources.filter(source => source.includes('var dotProduct = 0.0;'));
        expect(scans).toHaveLength(searchStructure === 'tiled' ? 2 : 1);
        for (const source of sources) expect(auditWGSLModule(source)).toEqual([]);
        for (const source of scans) {
          const start = source.indexOf('var dotProduct = 0.0;');
          const end = source.indexOf(
            'filledNeighborCount = min(filledNeighborCount + 1u, 4u);',
            start
          );
          expect(end).toBeGreaterThan(start);
          fragments.push(source.slice(start, end));
        }
      }
      expect(new Set(fragments).size).toBe(1);
    } finally {
      device.destroy();
    }
  });
}
