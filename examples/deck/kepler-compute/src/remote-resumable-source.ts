// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  DEFAULT_BACKOFF_OPTIONS,
  isTransientFetchError,
  type BackoffOptions,
  type Timers
} from './automatic-recovery-policy';
import {CsvChunkParser} from './csv-resume-stream';
import {
  REMOTE_SOURCE_MAXIMUM_BYTES,
  getLabelFromUrl,
  makeArrowTableFromCsvRows,
  makeSourceCandidateFromBytes,
  makeSourceCandidateFromTable,
  sniffRemoteSourceFormat,
  type RemoteLoadOptions
} from './remote-source-loader';
import {
  ContentChangedError,
  openResumableRemoteStream,
  type ResumeEvent
} from './resumable-remote-stream';
import {ResumeBatchDedupe} from './resume-batch-dedupe';
import type {SourceCandidate} from './source-workspace-panel';

/**
 * Remote "Open from URL" with mid-body resume (`?automatic=retry`). The body is read through
 * {@link openResumableRemoteStream} (Range/If-Range resume, or restart-and-discard with tail
 * verification), so a cut never duplicates or drops bytes. CSV is parsed incrementally into fixed
 * row chunks (header once, partial trailing records and quoted multi-line fields carried across
 * cuts). If resume itself gives up on a transient failure, the whole stream is restarted up to
 * `maximumRestarts` times and the re-delivered chunks are dropped by {@link ResumeBatchDedupe},
 * which fails explicitly when the re-delivered prefix differs. Other formats buffer the exact bytes
 * and use the existing decoders.
 */
export type ResumableRemoteLoadOptions = RemoteLoadOptions & {
  retry?: {options: BackoffOptions; timers?: Timers};
  maximumResumes?: number;
  /** Whole-stream restarts after resume is exhausted; defaults to 1. */
  maximumRestarts?: number;
  csvChunkRowCount?: number;
  onResumeEvent?(event: ResumeEvent): void;
};

export async function loadResumableRemoteSource(
  url: string,
  options: ResumableRemoteLoadOptions = {}
): Promise<SourceCandidate> {
  const maximumBytes = options.maximumBytes ?? REMOTE_SOURCE_MAXIMUM_BYTES;
  const signal = options.signal ?? new AbortController().signal;
  const label = getLabelFromUrl(url);
  const dedupe = new ResumeBatchDedupe();
  const csvRows: string[][] = [];
  let header: string[] | undefined;
  let format: ReturnType<typeof sniffRemoteSourceFormat> | undefined;
  const byteChunks: Uint8Array[] = [];
  let attemptsLeft = (options.maximumRestarts ?? 1) + 1;

  while (attemptsLeft-- > 0) {
    const parser = new CsvChunkParser({chunkRowCount: options.csvChunkRowCount});
    let loadedBytes = 0;
    let isFirstPart = true;
    const admit = (chunks: ReturnType<CsvChunkParser['pushBytes']>) => {
      for (const chunk of chunks)
        if (
          dedupe.consider({
            ordinal: chunk.ordinal,
            rowCount: chunk.rows.length,
            byteOffset: chunk.byteEnd,
            fingerprint: chunk.fingerprint
          }) === 'admit'
        )
          for (const row of chunk.rows) csvRows.push(row);
    };
    try {
      const body = await openResumableRemoteStream({
        url: new URL(url, globalThis.location?.href),
        signal,
        maximumInputBytes: maximumBytes,
        retry: options.retry ?? {options: DEFAULT_BACKOFF_OPTIONS},
        maximumResumes: options.maximumResumes,
        fetchImplementation: options.fetchFunction,
        onEvent: options.onResumeEvent
      });
      const reader = body.getReader();
      for (;;) {
        const {done, value} = await reader.read();
        if (done) break;
        loadedBytes += value.byteLength;
        options.onProgress?.({loadedBytes, totalBytes: null});
        if (isFirstPart) {
          isFirstPart = false;
          format = sniffRemoteSourceFormat(label, null, value);
        }
        if (format === 'csv') admit(parser.pushBytes(value));
        else byteChunks.push(value);
      }
      if (format === 'csv') {
        admit(parser.finish());
        header = parser.header;
      }
      break;
    } catch (failure) {
      // Restart only when resume gave up on a transient failure; content changes still fail.
      if (
        !attemptsLeft ||
        signal.aborted ||
        failure instanceof ContentChangedError ||
        !isTransientFetchError(failure)
      )
        throw failure;
      byteChunks.length = 0;
    }
  }

  if (format === 'csv') {
    if (!header) throw new Error('CSV has no header row');
    return makeSourceCandidateFromTable(
      makeArrowTableFromCsvRows([header, ...csvRows]),
      label,
      'csv'
    );
  }
  const bytes = new Uint8Array(byteChunks.reduce((sum, chunk) => sum + chunk.byteLength, 0));
  let offset = 0;
  for (const chunk of byteChunks) {
    bytes.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return makeSourceCandidateFromBytes(bytes, label, null);
}
