// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the RTS-04 and RTS-07 cases. */
export const RTS04_RTS07_MONITORING_EOF_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPURasterHarmonicRegression',
    name: 'rts04-memory-above-maximum',
    kind: 'binding',
    limit: 2048,
    rejects:
      /^Error: raster-harmonic-regression per-thread QR working memory .*must fit 2048 floats$/
  },
  {
    exportName: 'GPURasterHarmonicRegression',
    name: 'rts04-storage-slices-above-maximum',
    kind: 'binding',
    limit: 4096,
    rejects: /^Error: raster-harmonic-regression sliceCount must be an integer in \[1, 4096\]$/
  },
  {
    exportName: 'GPURasterCOLDMonitor',
    name: 'rts04-cold-memory-above-maximum',
    kind: 'binding',
    limit: 2048,
    rejects: /^Error: raster-cold-monitor per-thread working memory .*must fit 2048 floats$/
  },
  {
    exportName: 'GPURasterCOLDMonitor',
    name: 'rts04-cold-fit-budget-above-maximum',
    kind: 'binding',
    limit: 163,
    rejects: /^Error: raster-cold-monitor maximumFitCount must be an integer in \[1, 163\]$/
  },
  {
    exportName: 'GPURasterBFASTMonitor',
    name: 'rts04-bfast-memory-above-maximum',
    kind: 'binding',
    limit: 2048,
    rejects: /^Error: raster-bfast-monitor per-thread QR working memory .*must fit 2048 floats$/
  },
  {
    exportName: 'GPURasterSnapshotCovariance',
    name: 'rts07-invocations-above-maximum',
    kind: 'binding',
    limit: 16776960,
    rejects:
      /^Error: raster-snapshot-covariance accumulation invocations .*must be at most 16776960$/
  },
  {
    exportName: 'GPURasterEOFProjection',
    name: 'rts07-modes-above-maximum',
    kind: 'binding',
    limit: 32,
    rejects: /^Error: raster-eof-projection modeCount must be an integer in \[1, 32\]$/
  },
  {
    exportName: 'GPURasterLaggedCorrelation',
    name: 'rts07-lags-above-maximum',
    kind: 'binding',
    limit: 64,
    rejects: /^Error: raster-lagged-correlation lags length must be an integer in \[1, 64\]$/
  }
];
