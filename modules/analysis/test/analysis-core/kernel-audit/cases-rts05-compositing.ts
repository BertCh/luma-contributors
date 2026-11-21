// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPURasterBestAvailablePixel,
  GPURasterGeometricMedianComposite,
  GPURasterMedoidComposite,
  GPURasterPhenologyMetrics
} from '@luma-contributors/analysis/raster-analysis';
import type {GPUTemporalContext} from '../../../src/analysis-core/contracts/index';
import type {KernelAuditCase, KernelAuditContext} from './kernel-audit-context';

const CONTEXT: GPUTemporalContext = {
  unit: 'nanosecond',
  epoch: 'unix',
  encoding: 'int64-words',
  axis: {kind: 'irregular'}
};

const CELLS = 4096;

function medoid(
  {graph, view, construct}: KernelAuditContext,
  sliceCount: number,
  bandCount: number,
  withValidity: boolean
): void {
  graph.add(
    construct(GPURasterMedoidComposite, {
      cellCount: CELLS,
      sliceCount,
      bandCount,
      values: view('float32', CELLS * sliceCount * bandCount),
      mask: withValidity ? view('uint32', CELLS * sliceCount) : undefined,
      output: {
        composite: view('float32', CELLS * bandCount),
        sliceIndex: view('uint32', CELLS),
        status: view('uint32', CELLS)
      }
    })
  );
}

function geometricMedian(
  {graph, view, construct}: KernelAuditContext,
  sliceCount: number,
  bandCount: number,
  iterationCount: number,
  withOptional: boolean
): void {
  graph.add(
    construct(GPURasterGeometricMedianComposite, {
      cellCount: CELLS,
      sliceCount,
      bandCount,
      iterationCount,
      values: view('float32', CELLS * sliceCount * bandCount),
      mask: withOptional ? view('uint32', CELLS * sliceCount) : undefined,
      parameters: view('float32', 1),
      output: {
        composite: view('float32', CELLS * bandCount),
        status: view('uint32', CELLS),
        lastStep: withOptional ? view('float32', CELLS) : undefined
      }
    })
  );
}

function bestAvailablePixel(
  {graph, view, construct}: KernelAuditContext,
  sliceCount: number,
  bandCount: number,
  withOptional: boolean
): void {
  graph.add(
    construct(GPURasterBestAvailablePixel, {
      cellCount: CELLS,
      sliceCount,
      bandCount,
      values: bandCount > 0 ? view('float32', CELLS * sliceCount * bandCount) : undefined,
      mask: withOptional ? view('uint32', CELLS * sliceCount) : undefined,
      cloudDistance: withOptional ? view('float32', CELLS * sliceCount) : undefined,
      sliceAttributes: view('float32', 2 * sliceCount),
      parameters: view('float32', 9),
      output: {
        bestSlice: view('uint32', CELLS),
        bestScore: view('float32', CELLS),
        composite: bandCount > 0 ? view('float32', CELLS * bandCount) : undefined
      }
    })
  );
}

function phenology(
  {graph, view, construct}: KernelAuditContext,
  sliceCount: number,
  maximumSeasonCount: number,
  withAxis: boolean
): void {
  graph.add(
    construct(GPURasterPhenologyMetrics, {
      cellCount: CELLS,
      sliceCount,
      maximumSeasonCount,
      smoothed: view('float32', CELLS * sliceCount),
      parameters: view('float32', 4),
      dateAxis: withAxis
        ? {acquisitionTimes: view('uint32x2', sliceCount), temporalContext: CONTEXT}
        : undefined,
      output: {
        metrics: view('float32', CELLS * maximumSeasonCount * 10),
        flags: view('uint32', CELLS * maximumSeasonCount),
        seasonCount: view('uint32', CELLS),
        status: view('uint32', CELLS)
      }
    })
  );
}

/** Kernel-audit cases of the RTS-05 compositing and phenology contributors. */
export const RTS05_COMPOSITING_AUDIT_CASES: readonly KernelAuditCase[] = [
  {
    exportName: 'GPURasterMedoidComposite',
    name: 'rts05-maximum-slices-and-bands',
    build: context => medoid(context, 64, 16, true)
  },
  {
    exportName: 'GPURasterMedoidComposite',
    name: 'rts05-without-validity',
    build: context => medoid(context, 8, 4, false)
  },
  {
    exportName: 'GPURasterMedoidComposite',
    name: 'rts05-slices-above-maximum',
    build: context => medoid(context, 65, 4, true)
  },
  {
    exportName: 'GPURasterGeometricMedianComposite',
    name: 'rts05-maximum-iterations',
    build: context => geometricMedian(context, 256, 16, 64, true)
  },
  {
    exportName: 'GPURasterGeometricMedianComposite',
    name: 'rts05-minimal',
    build: context => geometricMedian(context, 8, 4, 1, false)
  },
  {
    exportName: 'GPURasterGeometricMedianComposite',
    name: 'rts05-iterations-above-maximum',
    build: context => geometricMedian(context, 32, 6, 65, true)
  },
  {
    exportName: 'GPURasterBestAvailablePixel',
    name: 'rts05-maximum-with-composite',
    build: context => bestAvailablePixel(context, 256, 16, true)
  },
  {
    exportName: 'GPURasterBestAvailablePixel',
    name: 'rts05-selection-only',
    build: context => bestAvailablePixel(context, 32, 0, false)
  },
  {
    exportName: 'GPURasterBestAvailablePixel',
    name: 'rts05-slices-above-maximum',
    build: context => bestAvailablePixel(context, 257, 4, true)
  },
  {
    exportName: 'GPURasterPhenologyMetrics',
    name: 'rts05-two-seasons-irregular-dates',
    build: context => phenology(context, 256, 2, true)
  },
  {
    exportName: 'GPURasterPhenologyMetrics',
    name: 'rts05-one-season-slot-index',
    build: context => phenology(context, 64, 1, false)
  },
  {
    exportName: 'GPURasterPhenologyMetrics',
    name: 'rts05-slices-above-maximum',
    build: context => phenology(context, 257, 2, true)
  }
];
