// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Batch-count dedupe for streams that can re-deliver already published chunks (a restarted Arrow
 * IPC reader, or a restarted CSV row-chunk reader). Chunks are identified by their delivery
 * ordinal; a redelivered ordinal is skipped (never counted twice) after its row count, byte offset
 * and fingerprint are compared with what was admitted, so a changed source fails explicitly
 * instead of being spliced into the published rows.
 */
export type DeliveredChunk = Readonly<{
  /** Zero-based position of the chunk within one full delivery of the stream. */
  ordinal: number;
  rowCount: number;
  /** Byte offset just past the chunk (CSV); omitted when the reader cannot report one (Arrow). */
  byteOffset?: number;
  /** Optional content fingerprint compared on redelivery. */
  fingerprint?: string;
}>;

export type ResumeDedupeDecision = 'admit' | 'skip';

export class ResumeBatchDedupe {
  private readonly admittedChunks: DeliveredChunk[] = [];
  private admittedRowCount = 0;
  private skippedChunkCount = 0;

  /** Chunks published exactly once. */
  get admittedBatchCount(): number {
    return this.admittedChunks.length;
  }
  /** Rows published exactly once (redelivered rows are not included). */
  get rowCount(): number {
    return this.admittedRowCount;
  }
  /** Redelivered chunks dropped so far. */
  get skippedBatchCount(): number {
    return this.skippedChunkCount;
  }

  consider(chunk: DeliveredChunk): ResumeDedupeDecision {
    const admittedCount = this.admittedChunks.length;
    if (chunk.ordinal > admittedCount) {
      // A gap means a chunk was lost between deliveries.
      throw new Error('Stream delivery skipped a chunk');
    }
    if (chunk.ordinal === admittedCount) {
      this.admittedChunks.push(chunk);
      this.admittedRowCount += chunk.rowCount;
      return 'admit';
    }
    const previous = this.admittedChunks[chunk.ordinal];
    if (
      previous.rowCount !== chunk.rowCount ||
      (previous.byteOffset !== undefined &&
        chunk.byteOffset !== undefined &&
        previous.byteOffset !== chunk.byteOffset) ||
      (previous.fingerprint !== undefined &&
        chunk.fingerprint !== undefined &&
        previous.fingerprint !== chunk.fingerprint)
    ) {
      // Restarted prefix differs from what was already published (changed or reordered source).
      throw new Error('Restarted stream prefix differs from the admitted rows');
    }
    this.skippedChunkCount++;
    return 'skip';
  }
}
