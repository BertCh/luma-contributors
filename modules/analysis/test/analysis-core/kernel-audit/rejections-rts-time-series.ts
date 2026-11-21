// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KernelAuditRejection} from './kernel-audit-rejection-type';

/** Expected declaration refusals of the TC-07, RTS-02 and RTS-06 cases. */
export const RTS_TIME_SERIES_REJECTIONS: readonly KernelAuditRejection[] = [
  {
    exportName: 'GPURasterStreamingAccumulator',
    name: 'tc07-chunk-above-maximum',
    kind: 'binding',
    limit: 256,
    rejects:
      /^Error: raster-streaming-accumulator chunkSliceCount must be an integer in \[1, 256\]$/
  },
  {
    exportName: 'GPURasterRollingMean',
    name: 'rts06-window-above-maximum',
    kind: 'binding',
    limit: 128,
    rejects: /^Error: raster-rolling-mean windowLength must be an integer in \[1, 128\]$/
  },
  {
    exportName: 'GPURasterWhittakerSmoother',
    name: 'rts02-slices-above-maximum',
    kind: 'binding',
    limit: 256,
    rejects: /^Error: raster-whittaker-smoother sliceCount must be an integer in \[2, 256\]$/
  },
  {
    exportName: 'GPURasterHANTS',
    name: 'rts02-memory-above-maximum',
    kind: 'binding',
    limit: 2048,
    rejects: /^Error: raster-hants per-thread QR working memory .*must fit 2048 floats$/
  },
  {
    exportName: 'GPURasterTemporalResample',
    name: 'rts02-grid-above-maximum',
    kind: 'binding',
    limit: 4096,
    rejects: /^Error: raster-temporal-resample gridCount must be an integer in \[1, 4096\]$/
  },
  {
    exportName: 'GPURasterStandardizedIndexCalibration',
    name: 'rts06-calibration-chunk-above-maximum',
    kind: 'binding',
    limit: 256,
    rejects:
      /^Error: raster-standardized-index-calibration chunkSliceCount must be an integer in \[1, 256\]$/
  },
  {
    exportName: 'GPURasterStandardizedIndex',
    name: 'rts06-index-chunk-above-maximum',
    kind: 'binding',
    limit: 16777216,
    rejects:
      /^Error: raster-standardized-index chunkSliceCount must be an integer in \[1, 16777216\]$/
  },
  {
    exportName: 'GPURasterClimateRunAccumulator',
    name: 'rts06-run-chunk-above-maximum',
    kind: 'binding',
    limit: 256,
    rejects:
      /^Error: raster-climate-run-accumulator chunkSliceCount must be an integer in \[1, 256\]$/
  }
];
