// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import type {KeplerComputeSession, KeplerSessionExplorer} from '@luma-contributors/kepler';
import type {AnalysisRuntime} from './analysis-runtime';
import {exportSelectedAnalysis} from './analysis-export';
import {
  runStreamingExport,
  STREAMING_EXPORT_DEFAULTS,
  type StreamingExportFormat,
  type StreamingExportProgress,
  type StreamingPageSource,
  type StreamingSink
} from './streaming-export';

/** Bounds applied to every page; memory is limited by one page, not by the selection size. */
export type AnalysisStreamLimits = {
  maximumPageReadbackBytes: number;
  maximumPageOutputBytes: number;
};

export const ANALYSIS_STREAM_DEFAULT_LIMITS: AnalysisStreamLimits = Object.freeze({
  maximumPageReadbackBytes: 16 * 1024 * 1024,
  maximumPageOutputBytes: 64 * 1024 * 1024
});

/**
 * Pages selected analysis results through `exportSelectedAnalysis` in page mode. Each page reserves
 * its own bounded readback/output memory (tagged `output`) and releases it before the next page, and
 * the selection, dataset and analysis result are re-validated on every page.
 */
export function makeAnalysisPageSource(
  session: KeplerComputeSession,
  explorer: KeplerSessionExplorer,
  analysis: AnalysisRuntime,
  options: {limits?: AnalysisStreamLimits; includeOriginalSource?: boolean} = {}
): StreamingPageSource {
  const limits = options.limits ?? ANALYSIS_STREAM_DEFAULT_LIMITS;
  let totalRows = 0;
  const readOne = async (offset: number, limit: number, signal: AbortSignal | undefined) => {
    const result = await exportSelectedAnalysis(session, explorer, analysis, {
      mode: 'page',
      offset,
      limit,
      format: 'arrow',
      maximumReadbackBytes: limits.maximumPageReadbackBytes,
      maximumCpuBytes: limits.maximumPageOutputBytes,
      maximumOutputBytes: limits.maximumPageOutputBytes,
      ...(options.includeOriginalSource ? {includeOriginalSource: true} : {}),
      ...(signal ? {signal} : {})
    });
    try {
      const table = arrow.tableFromIPC(result.bytes);
      const metadata = table.schema.metadata.get('luma:analysis-export');
      if (metadata) totalRows = JSON.parse(metadata).selectedCount ?? totalRows;
      else totalRows = Math.max(totalRows, offset + table.numRows);
      return table;
    } finally {
      result.destroy();
    }
  };
  return {
    async getTotalRows() {
      // The selected count is part of every page's metadata.
      await readOne(0, 1, undefined);
      return totalRows;
    },
    readPage: readOne
  };
}

/** Streams a selected analysis result to CSV or Arrow chunks; cancel with `signal`. */
export function streamSelectedAnalysis(props: {
  session: KeplerComputeSession;
  explorer: KeplerSessionExplorer;
  analysis: AnalysisRuntime;
  format: StreamingExportFormat;
  sink: StreamingSink;
  pageRows?: number;
  limits?: AnalysisStreamLimits;
  includeOriginalSource?: boolean;
  signal?: AbortSignal;
  onProgress?(progress: StreamingExportProgress): void;
}): Promise<StreamingExportProgress> {
  return runStreamingExport({
    source: makeAnalysisPageSource(props.session, props.explorer, props.analysis, props),
    format: props.format,
    sink: props.sink,
    pageRows: props.pageRows ?? STREAMING_EXPORT_DEFAULTS.pageRows,
    ...(props.signal ? {signal: props.signal} : {}),
    ...(props.onProgress ? {onProgress: props.onProgress} : {})
  });
}
