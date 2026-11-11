// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPURasterEmbeddingChange,
  GPURasterEmbeddingCovariance,
  GPURasterEmbeddingDequantize,
  GPURasterEmbeddingKMeans,
  GPURasterEmbeddingKNearestClassifier,
  GPURasterEmbeddingLinearProbe,
  GPURasterEmbeddingNearestPixels,
  GPURasterEmbeddingProjection,
  GPURasterEmbeddingPyramidLevel,
  GPURasterEmbeddingSimilarity,
  GPURasterEmbeddingZoneMean
} from '@luma-contributors/analysis/raster-analysis';
import {GPURasterFalseDiscoveryRateMask} from '@luma-contributors/analysis/gpu-spatial-analysis';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const DIMENSIONS = 128;
/** Pixels whose 128-band float32 vectors fill one 128 MiB storage binding exactly. */
const pixelsAtBindingLimit = ({maximumRows}: KernelAuditContext) => maximumRows(DIMENSIONS * 4);

function dequantize(context: KernelAuditContext, withOptional: boolean): void {
  const {graph, view} = context;
  const pixelCount = pixelsAtBindingLimit(context);
  graph.add(
    new GPURasterEmbeddingDequantize({
      pixelCount,
      dimensions: DIMENSIONS,
      normalize: withOptional,
      quantized: view('uint32', (pixelCount * DIMENSIONS) / 4),
      output: {
        vectors: view('float32', pixelCount * DIMENSIONS),
        validity: withOptional ? view('uint32', pixelCount) : undefined
      }
    })
  );
}

function similarity(context: KernelAuditContext, metric: 'cosine' | 'dot', queryCount = 64): void {
  const {graph, view} = context;
  const pixelCount = pixelsAtBindingLimit(context);
  graph.add(
    new GPURasterEmbeddingSimilarity({
      pixelCount,
      dimensions: DIMENSIONS,
      queryCount,
      metric,
      vectors: view('float32', pixelCount * DIMENSIONS),
      mask: view('uint32', pixelCount),
      queries: view('float32', queryCount * DIMENSIONS),
      parameters: view('float32', 1),
      output: {
        similarity: view('float32', pixelCount * queryCount),
        maximum: view('float32', pixelCount),
        bestQuery: view('uint32', pixelCount),
        matches: view('uint32', pixelCount)
      }
    })
  );
}

function classifier(
  context: KernelAuditContext,
  trainingCount = 4096,
  metric: 'euclidean' | 'cosine' = 'euclidean'
): void {
  const {graph, view} = context;
  const pixelCount = pixelsAtBindingLimit(context);
  graph.add(
    new GPURasterEmbeddingKNearestClassifier({
      pixelCount,
      dimensions: DIMENSIONS,
      metric,
      vectors: view('float32', pixelCount * DIMENSIONS),
      mask: view('uint32', pixelCount),
      trainingVectors: view('float32', trainingCount * DIMENSIONS),
      trainingLabels: view('uint32', trainingCount),
      trainingCount,
      classCount: 32,
      neighborCount: 16,
      output: {
        labels: view('uint32', pixelCount),
        confidence: view('float32', pixelCount),
        probabilities: view('float32', pixelCount * 32)
      }
    })
  );
}

function covariance(context: KernelAuditContext, rangeCount = 256): void {
  const {graph, view} = context;
  const pixelCount = pixelsAtBindingLimit(context);
  graph.add(
    new GPURasterEmbeddingCovariance({
      pixelCount,
      dimensions: DIMENSIONS,
      rangeCount,
      vectors: view('float32', pixelCount * DIMENSIONS),
      mask: view('uint32', pixelCount),
      output: {
        mean: view('float32', DIMENSIONS),
        covariance: view('float32', DIMENSIONS * DIMENSIONS),
        usableCount: view('uint32', 1)
      }
    })
  );
}

function projection(context: KernelAuditContext, componentCount = 16): void {
  const {graph, view} = context;
  const pixelCount = pixelsAtBindingLimit(context);
  graph.add(
    new GPURasterEmbeddingProjection({
      pixelCount,
      dimensions: DIMENSIONS,
      componentCount,
      vectors: view('float32', pixelCount * DIMENSIONS),
      mask: view('uint32', pixelCount),
      mean: view('float32', DIMENSIONS),
      components: view('float32', componentCount * DIMENSIONS),
      parameters: view('float32', 6),
      output: {
        projected: view('float32', pixelCount * componentCount),
        rgb: view('float32', pixelCount * 3)
      }
    })
  );
}

function zoneMean(
  context: KernelAuditContext,
  source: 'tuples' | 'zone-raster',
  zoneCount = 65536
): void {
  const {graph, view} = context;
  const pixelCount = pixelsAtBindingLimit(context);
  const tupleCapacity = 1 << 20;
  graph.add(
    new GPURasterEmbeddingZoneMean({
      pixelCount,
      dimensions: DIMENSIONS,
      zoneCount,
      vectors: view('float32', pixelCount * DIMENSIONS),
      mask: view('uint32', pixelCount),
      ...(source === 'tuples'
        ? {
            tuples: {
              zoneIds: view('uint32', tupleCapacity),
              cellIds: view('uint32', tupleCapacity),
              fractions: view('float32', tupleCapacity),
              status: {count: view('uint32', 1)}
            }
          }
        : {zoneRaster: view('uint32', pixelCount)}),
      output: {
        means: view('float32', zoneCount * DIMENSIONS),
        weights: view('float32', zoneCount),
        meanLengths: view('float32', zoneCount)
      }
    })
  );
}

function pyramid(context: KernelAuditContext, source: 'vectors' | 'quantized', scale = 16): void {
  const {graph, view} = context;
  const sourceWidth = 512;
  const sourceHeight = 512;
  const sourcePixels = sourceWidth * sourceHeight;
  const outputPixels = Math.ceil(sourceWidth / scale) * Math.ceil(sourceHeight / scale);
  graph.add(
    new GPURasterEmbeddingPyramidLevel({
      sourceWidth,
      sourceHeight,
      dimensions: DIMENSIONS,
      scale,
      ...(source === 'vectors'
        ? {vectors: view('float32', sourcePixels * DIMENSIONS)}
        : {quantized: view('uint32', (sourcePixels * DIMENSIONS) / 4)}),
      mask: view('uint32', sourcePixels),
      output: {
        vectors: view('float32', outputPixels * DIMENSIONS),
        validity: view('uint32', outputPixels),
        validCount: view('uint32', outputPixels),
        meanLength: view('float32', outputPixels)
      }
    })
  );
}

function falseDiscoveryRate(
  {graph, view}: KernelAuditContext,
  method: 'benjamini-hochberg' | 'holm',
  withOptional: boolean,
  cellCount = 1 << 20
): void {
  graph.add(
    new GPURasterFalseDiscoveryRateMask({
      cellCount,
      method,
      pValues: view('float32', cellCount),
      validity: withOptional ? view('uint32', cellCount) : undefined,
      parameters: view('float32', 1),
      output: {
        significant: view('uint32', cellCount),
        adjustedPValues: withOptional ? view('float32', cellCount) : undefined,
        testedCount: withOptional ? view('uint32', 1) : undefined,
        significantCount: withOptional ? view('uint32', 1) : undefined,
        criticalPValue: withOptional ? view('float32', 1) : undefined,
        histogram: withOptional ? view('uint32', 65536) : undefined
      }
    })
  );
}

/** Kernel-audit cases of RTS-01 (field significance) and RTS-03 (embedding rasters). */
export const RTS_01_03_EMBEDDING_FDR_AUDIT_CASES: readonly KernelAuditCase[] = [
  {
    exportName: 'GPURasterFalseDiscoveryRateMask',
    name: 'rts01-benjamini-hochberg-all-outputs',
    build: context => falseDiscoveryRate(context, 'benjamini-hochberg', true)
  },
  {
    exportName: 'GPURasterFalseDiscoveryRateMask',
    name: 'rts01-holm-mask-only',
    build: context => falseDiscoveryRate(context, 'holm', false)
  },
  {
    exportName: 'GPURasterFalseDiscoveryRateMask',
    name: 'rts01-histogram-above-maximum',
    build: ({graph, view}) =>
      graph.add(
        new GPURasterFalseDiscoveryRateMask({
          cellCount: 1024,
          pValues: view('float32', 1024),
          parameters: view('float32', 1),
          output: {significant: view('uint32', 1024), histogram: view('uint32', 65537)}
        })
      )
  },
  {
    exportName: 'GPURasterEmbeddingDequantize',
    name: 'rts03-int8-128-bands-normalized',
    build: context => dequantize(context, true)
  },
  {
    exportName: 'GPURasterEmbeddingDequantize',
    name: 'rts03-int8-128-bands-raw',
    build: context => dequantize(context, false)
  },
  {
    exportName: 'GPURasterEmbeddingSimilarity',
    name: 'rts03-cosine-64-queries',
    build: context => similarity(context, 'cosine')
  },
  {
    exportName: 'GPURasterEmbeddingSimilarity',
    name: 'rts03-dot-64-queries',
    build: context => similarity(context, 'dot')
  },
  {
    exportName: 'GPURasterEmbeddingSimilarity',
    name: 'rts03-queries-above-maximum',
    build: context => similarity(context, 'cosine', 65)
  },
  {
    exportName: 'GPURasterEmbeddingNearestPixels',
    name: 'rts03-top-16-for-64-queries',
    build: context => {
      const {graph, view} = context;
      const pixelCount = pixelsAtBindingLimit(context);
      new GPURasterEmbeddingNearestPixels({
        pixelCount,
        dimensions: DIMENSIONS,
        vectors: view('float32', pixelCount * DIMENSIONS),
        mask: view('uint32', pixelCount),
        queryCount: 64,
        queries: view('float32', 64 * DIMENSIONS),
        resultCount: 16,
        output: {
          pixelIds: view('uint32', 64 * 16),
          scores: view('float32', 64 * 16),
          resultCounts: view('uint32', 64)
        }
      }).addToGraph(graph);
    }
  },
  {
    exportName: 'GPURasterEmbeddingChange',
    name: 'rts03-all-outputs',
    build: context => {
      const {graph, view} = context;
      const pixelCount = pixelsAtBindingLimit(context);
      graph.add(
        new GPURasterEmbeddingChange({
          pixelCount,
          dimensions: DIMENSIONS,
          before: view('float32', pixelCount * DIMENSIONS),
          after: view('float32', pixelCount * DIMENSIONS),
          beforeMask: view('uint32', pixelCount),
          afterMask: view('uint32', pixelCount),
          parameters: view('float32', 1),
          output: {
            similarity: view('float32', pixelCount),
            angle: view('float32', pixelCount),
            magnitude: view('float32', pixelCount),
            changed: view('uint32', pixelCount)
          }
        })
      );
    }
  },
  {
    exportName: 'GPURasterEmbeddingKMeans',
    name: 'rts03-64-clusters-64-iterations',
    build: context => {
      const {graph, view} = context;
      const pixelCount = pixelsAtBindingLimit(context);
      new GPURasterEmbeddingKMeans({
        pixelCount,
        dimensions: DIMENSIONS,
        vectors: view('float32', pixelCount * DIMENSIONS),
        mask: view('uint32', pixelCount),
        clusterCount: 64,
        maximumIterations: 64,
        output: {
          labels: view('uint32', pixelCount),
          centroids: view('float32', 64 * DIMENSIONS),
          counts: view('uint32', 64),
          status: view('uint32', 3)
        }
      }).addToGraph(graph);
    }
  },
  {
    exportName: 'GPURasterEmbeddingKNearestClassifier',
    name: 'rts03-4096-training-euclidean',
    build: context => classifier(context)
  },
  {
    exportName: 'GPURasterEmbeddingKNearestClassifier',
    name: 'rts03-4096-training-cosine',
    build: context => classifier(context, 4096, 'cosine')
  },
  {
    exportName: 'GPURasterEmbeddingKNearestClassifier',
    name: 'rts03-training-above-maximum',
    build: context => classifier(context, 4097)
  },
  {
    exportName: 'GPURasterEmbeddingLinearProbe',
    name: 'rts03-64-classes',
    build: context => {
      const {graph, view} = context;
      const pixelCount = pixelsAtBindingLimit(context);
      graph.add(
        new GPURasterEmbeddingLinearProbe({
          pixelCount,
          dimensions: DIMENSIONS,
          classCount: 64,
          vectors: view('float32', pixelCount * DIMENSIONS),
          mask: view('uint32', pixelCount),
          weights: view('float32', 64 * DIMENSIONS),
          bias: view('float32', 64),
          output: {
            labels: view('uint32', pixelCount),
            confidence: view('float32', pixelCount),
            probabilities: view('float32', pixelCount * 64)
          }
        })
      );
    }
  },
  {
    exportName: 'GPURasterEmbeddingCovariance',
    name: 'rts03-128-bands-256-ranges',
    build: context => covariance(context)
  },
  {
    exportName: 'GPURasterEmbeddingCovariance',
    name: 'rts03-ranges-above-maximum',
    build: context => covariance(context, 257)
  },
  {
    exportName: 'GPURasterEmbeddingProjection',
    name: 'rts03-16-components-rgb',
    build: context => projection(context)
  },
  {
    exportName: 'GPURasterEmbeddingProjection',
    name: 'rts03-components-above-maximum',
    build: context => projection(context, 17)
  },
  {
    exportName: 'GPURasterEmbeddingZoneMean',
    name: 'rts03-65536-zones-tuples',
    build: context => zoneMean(context, 'tuples')
  },
  {
    exportName: 'GPURasterEmbeddingZoneMean',
    name: 'rts03-65536-zones-raster',
    build: context => zoneMean(context, 'zone-raster')
  },
  {
    exportName: 'GPURasterEmbeddingZoneMean',
    name: 'rts03-zones-above-maximum',
    build: context => zoneMean(context, 'zone-raster', 65537)
  },
  {
    exportName: 'GPURasterEmbeddingPyramidLevel',
    name: 'rts03-unit-vector-mean-vectors',
    build: context => pyramid(context, 'vectors')
  },
  {
    exportName: 'GPURasterEmbeddingPyramidLevel',
    name: 'rts03-unit-vector-mean-int8',
    build: context => pyramid(context, 'quantized')
  },
  {
    exportName: 'GPURasterEmbeddingPyramidLevel',
    name: 'rts03-scale-above-maximum',
    build: context => pyramid(context, 'vectors', 17)
  }
];
