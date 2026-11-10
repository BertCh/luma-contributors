// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the RTS-05 compositing and phenology cases. */
export const RTS05_COMPOSITING_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPURasterMedoidComposite',
    name: 'rts05-slices-above-maximum',
    kind: 'binding',
    limit: 64,
    rejects: /^Error: raster-medoid-composite sliceCount must be an integer in \[1, 64\]$/
  },
  {
    exportName: 'GPURasterGeometricMedianComposite',
    name: 'rts05-iterations-above-maximum',
    kind: 'binding',
    limit: 64,
    rejects:
      /^Error: raster-geometric-median-composite iterationCount must be an integer in \[1, 64\]$/
  },
  {
    exportName: 'GPURasterBestAvailablePixel',
    name: 'rts05-slices-above-maximum',
    kind: 'binding',
    limit: 256,
    rejects: /^Error: raster-best-available-pixel sliceCount must be an integer in \[1, 256\]$/
  },
  {
    exportName: 'GPURasterPhenologyMetrics',
    name: 'rts05-slices-above-maximum',
    kind: 'binding',
    limit: 256,
    rejects: /^Error: raster-phenology-metrics sliceCount must be an integer in \[2, 256\]$/
  }
];
