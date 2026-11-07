// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Incremental CSV record parser for resumable remote streams. Bytes may be split anywhere,
 * including inside a multi-byte character, a line or a quoted (multi-line) field; the incomplete
 * trailing record is carried to the next push. The header row is consumed exactly once. Rows are
 * emitted in fixed-size chunks so the chunk ordinal, row offset and byte offset are identical for
 * every delivery of the same bytes, which lets {@link ResumeBatchDedupe} drop re-delivered chunks.
 * Record splitting follows d3-dsv `parseRows` (quotes only at field start, CRLF/LF/CR endings).
 */
export type CsvRowChunk = Readonly<{
  ordinal: number;
  /** Data rows (excluding the header) before this chunk. */
  rowOffset: number;
  /** UTF-8 byte offset just past the last record in this chunk. */
  byteEnd: number;
  rows: string[][];
  fingerprint: string;
}>;

export const CSV_DELIMITERS = [',', ';', '\t', '|'] as const;

function detectDelimiterFromLine(firstLine: string): string {
  let best = ',';
  let bestCount = 0;
  for (const delimiter of CSV_DELIMITERS) {
    const count = firstLine.split(delimiter).length - 1;
    if (count > bestCount) {
      best = delimiter;
      bestCount = count;
    }
  }
  return best;
}

function getUtf8Length(codePoint: number): number {
  return codePoint < 0x80 ? 1 : codePoint < 0x800 ? 2 : codePoint < 0x10000 ? 3 : 4;
}

function fingerprintRows(rows: readonly string[][]): string {
  let hash = 0x811c9dc5;
  for (const row of rows) {
    for (const value of row) {
      for (let index = 0; index < value.length; index++)
        hash = Math.imul(hash ^ value.charCodeAt(index), 0x01000193) >>> 0;
      hash = Math.imul(hash ^ 0x1f, 0x01000193) >>> 0;
    }
    hash = Math.imul(hash ^ 0x1e, 0x01000193) >>> 0;
  }
  return hash.toString(16);
}

export class CsvChunkParser {
  /** Header cells, set once the first record is complete. */
  header: string[] | undefined;
  /** Data rows emitted or buffered so far. */
  rowCount = 0;

  private readonly chunkRowCount: number;
  private readonly decoder = new TextDecoder('utf-8');
  private delimiter: string | undefined;
  private waiting = '';
  private isFirstText = true;
  private field = '';
  private record: string[] = [];
  private fieldStart = true;
  private inQuotes = false;
  private quotePending = false;
  private carriageReturnPending = false;
  private consumedBytes = 0;
  private pendingRows: string[][] = [];
  private chunkOrdinal = 0;
  private chunkRowOffset = 0;

  constructor(options: {chunkRowCount?: number; delimiter?: string} = {}) {
    this.chunkRowCount = options.chunkRowCount ?? 1000;
    this.delimiter = options.delimiter;
  }

  pushBytes(bytes: Uint8Array): CsvRowChunk[] {
    return this.pushText(this.decoder.decode(bytes, {stream: true}), false);
  }

  /** Flushes the final record (no trailing newline required) and the last partial chunk. */
  finish(): CsvRowChunk[] {
    return this.pushText(this.decoder.decode(), true);
  }

  private pushText(text: string, isFinal: boolean): CsvRowChunk[] {
    const chunks: CsvRowChunk[] = [];
    let incoming = text;
    if (this.delimiter === undefined) {
      this.waiting += incoming;
      const newline = this.waiting.indexOf('\n');
      // The delimiter comes from the first line (as in the whole-buffer decoder), so wait for it.
      if (newline < 0 && !isFinal) return chunks;
      const first = this.waiting.replace(/^﻿/, '');
      this.delimiter = detectDelimiterFromLine(
        first.slice(0, first.indexOf('\n') >= 0 ? first.indexOf('\n') : 4096)
      );
      incoming = this.waiting;
      this.waiting = '';
    }
    if (this.isFirstText && incoming.length) {
      this.isFirstText = false;
      if (incoming.charCodeAt(0) === 0xfeff) {
        this.consumedBytes += 3;
        incoming = incoming.slice(1);
      }
    }
    for (const character of incoming) this.consumeCharacter(character, chunks);
    if (isFinal) {
      this.quotePending = false;
      this.inQuotes = false;
      if (!(this.fieldStart && this.record.length === 0)) this.endRecord(chunks);
      if (this.pendingRows.length) this.flushChunk(chunks);
    }
    return chunks;
  }

  private consumeCharacter(character: string, chunks: CsvRowChunk[]): void {
    this.consumedBytes += getUtf8Length(character.codePointAt(0)!);
    if (this.quotePending) {
      this.quotePending = false;
      if (character === '"') {
        this.field += '"';
        return;
      }
      this.inQuotes = false;
    }
    if (this.inQuotes) {
      if (character === '"') this.quotePending = true;
      else this.field += character;
      return;
    }
    if (this.carriageReturnPending) {
      this.carriageReturnPending = false;
      // CRLF: the record already ended at the carriage return.
      if (character === '\n') return;
    }
    if (character === '"' && this.fieldStart) {
      this.inQuotes = true;
      this.fieldStart = false;
    } else if (character === this.delimiter) {
      this.record.push(this.field);
      this.field = '';
      this.fieldStart = true;
    } else if (character === '\n' || character === '\r') {
      this.endRecord(chunks);
      this.carriageReturnPending = character === '\r';
    } else {
      this.field += character;
      this.fieldStart = false;
    }
  }

  private endRecord(chunks: CsvRowChunk[]): void {
    this.record.push(this.field);
    const completed = this.record;
    this.record = [];
    this.field = '';
    this.fieldStart = true;
    if (!this.header) {
      this.header = completed;
      return;
    }
    this.pendingRows.push(completed);
    this.rowCount++;
    if (this.pendingRows.length >= this.chunkRowCount) this.flushChunk(chunks);
  }

  private flushChunk(chunks: CsvRowChunk[]): void {
    const rows = this.pendingRows;
    this.pendingRows = [];
    chunks.push({
      ordinal: this.chunkOrdinal++,
      rowOffset: this.chunkRowOffset,
      byteEnd: this.consumedBytes,
      rows,
      fingerprint: fingerprintRows(rows)
    });
    this.chunkRowOffset += rows.length;
  }
}
