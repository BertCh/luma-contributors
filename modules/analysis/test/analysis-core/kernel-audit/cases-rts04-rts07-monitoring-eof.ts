// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  GPURasterBFASTMonitor,
  GPURasterCOLDMonitor,
  GPURasterEOFProjection,
  GPURasterHarmonicRegression,
  GPURasterLaggedCorrelation,
  GPURasterSnapshotCovariance,
  type GPURasterIrregularDateAxis
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

function makeAxis({view}: KernelAuditContext, slices: number): GPURasterIrregularDateAxis {
  return {acquisitionTimes: view('uint32x2', slices), temporalContext: CONTEXT};
}

function harmonicRegression(
  context: KernelAuditContext,
  slices: number,
  harmonicCount: number
): void {
  const {graph, view} = context;
  const columns = 2 + 2 * harmonicCount;
  graph.add(
    new GPURasterHarmonicRegression({
      cellCount: CELLS,
      sliceCount: slices,
      harmonicCount,
      values: view('float32', CELLS * slices),
      mask: view('uint32', CELLS * slices),
      parameters: view('float32', 1),
      dateAxis: makeAxis(context, slices),
      output: {
        coefficients: view('float32', CELLS * columns),
        summary: view('uint32', CELLS * 2),
        fitted: view('float32', CELLS * slices),
        rmse: view('float32', CELLS)
      }
    })
  );
}

function coldMonitor(
  context: KernelAuditContext,
  slices: number,
  bands: number,
  columns: 4 | 6 | 8,
  options: {consecutiveCount?: number; maximumFitCount?: number} = {}
): void {
  const {graph, view} = context;
  const segments = 8;
  graph.add(
    new GPURasterCOLDMonitor({
      cellCount: CELLS,
      sliceCount: slices,
      bandCount: bands,
      values: view('float32', CELLS * slices * bands),
      mask: view('uint32', CELLS * slices),
      parameters: view('float32', 6 + bands),
      dateAxis: makeAxis(context, slices),
      maximumCoefficientCount: columns,
      maximumSegmentCount: segments,
      ...options,
      output: {
        segments: view('uint32', CELLS * segments * 4),
        coefficients: view('float32', CELLS * segments * bands * 8),
        segmentStatistics: view('float32', CELLS * segments * bands * 2),
        cellSummary: view('uint32', CELLS * 3)
      }
    })
  );
}

function bfastMonitor(
  context: KernelAuditContext,
  slices: number,
  history: number,
  harmonics: number
): void {
  const {graph, view} = context;
  const columns = 2 + 2 * harmonics;
  graph.add(
    new GPURasterBFASTMonitor({
      cellCount: CELLS,
      sliceCount: slices,
      historySliceCount: history,
      harmonicCount: harmonics,
      values: view('float32', CELLS * slices),
      mask: view('uint32', CELLS * slices),
      parameters: view('float32', 3),
      dateAxis: makeAxis(context, slices),
      output: {
        coefficients: view('float32', CELLS * columns),
        scores: view('float32', CELLS * 2),
        cellSummary: view('uint32', CELLS * 3),
        process: view('float32', CELLS * (slices - history))
      }
    })
  );
}

function snapshotCovariance(
  context: KernelAuditContext,
  cells: number,
  slices: number,
  // The invocation limit is checked first; short views keep a refused case within the binding limit.
  sizedViews = true
): void {
  const {graph, view} = context;
  graph.add(
    new GPURasterSnapshotCovariance({
      cellCount: cells,
      sliceCount: slices,
      values: view('float32', sizedViews ? cells * slices : 4),
      mask: view('uint32', sizedViews ? cells * slices : 4),
      weights: view('float32', sizedViews ? cells : 4),
      parameters: view('float32', 2),
      output: {
        covarianceWords: view('uint32', ((slices * (slices + 1)) / 2) * 3),
        keptCellCount: view('uint32', 1)
      }
    })
  );
}

function eofProjection(context: KernelAuditContext, slices: number, modes: number): void {
  const {graph, view} = context;
  graph.add(
    new GPURasterEOFProjection({
      cellCount: CELLS,
      sliceCount: slices,
      modeCount: modes,
      values: view('float32', CELLS * slices),
      mask: view('uint32', CELLS * slices),
      weights: view('float32', CELLS),
      parameters: view('float32', 2),
      temporalModes: view('float32', modes * slices),
      singularValues: view('float32', modes),
      output: {eofs: view('float32', CELLS * modes), status: view('uint32', CELLS)}
    })
  );
}

function laggedCorrelation(context: KernelAuditContext, slices: number, lagCount: number): void {
  const {graph, view} = context;
  graph.add(
    new GPURasterLaggedCorrelation({
      cellCount: CELLS,
      sliceCount: slices,
      values: view('float32', CELLS * slices),
      mask: view('uint32', CELLS * slices),
      referenceIndex: view('float32', slices),
      parameters: view('float32', 1),
      lags: Array.from({length: lagCount}, (_, lag) => lag),
      output: {
        correlations: view('float32', CELLS * lagCount),
        status: view('uint32', CELLS * lagCount),
        pairCount: view('uint32', CELLS * lagCount)
      }
    })
  );
}

export const RTS04_RTS07_MONITORING_EOF_AUDIT_CASES: readonly KernelAuditCase[] = [
  {
    exportName: 'GPURasterHarmonicRegression',
    name: 'rts04-maximum-memory-one-harmonic',
    build: context => harmonicRegression(context, 256, 1)
  },
  {
    exportName: 'GPURasterHarmonicRegression',
    name: 'rts04-memory-above-maximum',
    build: context => harmonicRegression(context, 256, 2)
  },
  {
    exportName: 'GPURasterCOLDMonitor',
    name: 'rts04-cold-maximum-slices-four-coefficients',
    build: context => coldMonitor(context, 256, 5, 4)
  },
  {
    exportName: 'GPURasterCOLDMonitor',
    name: 'rts04-cold-eight-coefficients-five-bands',
    build: context => coldMonitor(context, 160, 5, 8)
  },
  {
    exportName: 'GPURasterCOLDMonitor',
    name: 'rts04-cold-memory-above-maximum',
    build: context => coldMonitor(context, 256, 5, 8)
  },
  {
    exportName: 'GPURasterCOLDMonitor',
    name: 'rts04-cold-fit-budget-above-maximum',
    build: context => coldMonitor(context, 160, 5, 8, {maximumFitCount: 1024})
  },
  {
    exportName: 'GPURasterBFASTMonitor',
    name: 'rts04-bfast-maximum-memory',
    build: context => bfastMonitor(context, 256, 150, 3)
  },
  {
    exportName: 'GPURasterBFASTMonitor',
    name: 'rts04-bfast-memory-above-maximum',
    build: context => bfastMonitor(context, 256, 200, 3)
  },
  {
    exportName: 'GPURasterSnapshotCovariance',
    name: 'rts07-maximum-slices',
    build: context => snapshotCovariance(context, CELLS, 256)
  },
  {
    exportName: 'GPURasterSnapshotCovariance',
    name: 'rts07-invocations-above-maximum',
    build: context => snapshotCovariance(context, 256 * 2048, 256, false)
  },
  {
    exportName: 'GPURasterEOFProjection',
    name: 'rts07-maximum-modes',
    build: context => eofProjection(context, 256, 32)
  },
  {
    exportName: 'GPURasterEOFProjection',
    name: 'rts07-modes-above-maximum',
    build: context => eofProjection(context, 256, 33)
  },
  {
    exportName: 'GPURasterLaggedCorrelation',
    name: 'rts07-maximum-lags',
    build: context => laggedCorrelation(context, 256, 64)
  },
  {
    exportName: 'GPURasterLaggedCorrelation',
    name: 'rts07-lags-above-maximum',
    build: context => laggedCorrelation(context, 256, 65)
  }
];
