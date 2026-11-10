// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the RTS-01 and RTS-03 cases. */
export const RTS_01_03_EMBEDDING_FDR_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPURasterFalseDiscoveryRateMask',
    name: 'rts01-histogram-above-maximum',
    kind: 'binding',
    limit: 65536,
    rejects:
      /^Error: raster-false-discovery-rate-mask output\.histogram bin count must be an integer in \[1, 65536\]$/
  },
  {
    exportName: 'GPURasterEmbeddingSimilarity',
    name: 'rts03-queries-above-maximum',
    kind: 'binding',
    limit: 64,
    rejects: /^Error: raster-embedding-similarity queryCount must be an integer in \[1, 64\]$/
  },
  {
    exportName: 'GPURasterEmbeddingKNearestClassifier',
    name: 'rts03-training-above-maximum',
    kind: 'binding',
    limit: 4096,
    rejects:
      /^Error: raster-embedding-knn-classifier trainingCount must be an integer in \[1, 4096\]$/
  },
  {
    exportName: 'GPURasterEmbeddingCovariance',
    name: 'rts03-ranges-above-maximum',
    kind: 'binding',
    limit: 256,
    rejects: /^Error: raster-embedding-covariance rangeCount must be an integer in \[1, 256\]$/
  },
  {
    exportName: 'GPURasterEmbeddingProjection',
    name: 'rts03-components-above-maximum',
    kind: 'binding',
    limit: 16,
    rejects: /^Error: raster-embedding-projection componentCount must be an integer in \[3, 16\]$/
  },
  {
    exportName: 'GPURasterEmbeddingZoneMean',
    name: 'rts03-zones-above-maximum',
    kind: 'binding',
    limit: 65536,
    rejects: /^Error: raster-embedding-zone-mean zoneCount must be an integer in \[1, 65536\]$/
  },
  {
    exportName: 'GPURasterEmbeddingPyramidLevel',
    name: 'rts03-scale-above-maximum',
    kind: 'binding',
    limit: 16,
    rejects:
      /^Error: raster-embedding-pyramid-level horizontal scale must be an integer in \[1, 16\]$/
  }
];
