// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {dsvFormat} from 'd3-dsv';
import {CsvChunkParser} from '../../../examples/deck/kepler-compute/src/csv-resume-stream';
import {ResumeBatchDedupe} from '../../../examples/deck/kepler-compute/src/resume-batch-dedupe';
import {loadResumableRemoteSource} from '../../../examples/deck/kepler-compute/src/remote-resumable-source';
import type {Timers} from '../../../examples/deck/kepler-compute/src/automatic-recovery-policy';

const csvText =
  'id,name,note\r\n1,Alpha,"line one\nline two, with ""quotes"""\r\n2,Béta 🌍,plain\n3,Gamma,\n\n4,"Delta",x\n5,Eps,tail';
const csvBytes = new TextEncoder().encode(csvText);
const expectedRows = dsvFormat(',').parseRows(csvText);

function parseSplit(cuts: number[], chunkRowCount = 2) {
  const parser = new CsvChunkParser({chunkRowCount});
  const chunks = [];
  let previous = 0;
  for (const cut of [...cuts, csvBytes.length]) {
    chunks.push(...parser.pushBytes(csvBytes.subarray(previous, cut)));
    previous = cut;
  }
  chunks.push(...parser.finish());
  return {header: parser.header, rows: chunks.flatMap(chunk => chunk.rows), chunks};
}

describe('CsvChunkParser', () => {
  it('matches d3-dsv for every single split point (header once, no lost or duplicated rows)', () => {
    for (let cut = 0; cut <= csvBytes.length; cut++) {
      const {header, rows} = parseSplit([cut]);
      expect([header, ...rows], `cut ${cut}`).toEqual(expectedRows);
    }
  });

  it('handles byte-at-a-time delivery and produces split-independent chunk boundaries', () => {
    const whole = parseSplit([]);
    const bytewise = parseSplit(Array.from({length: csvBytes.length - 1}, (_, index) => index + 1));
    expect(bytewise.chunks).toEqual(whole.chunks);
    expect(whole.chunks.map(chunk => chunk.ordinal)).toEqual(whole.chunks.map((_, index) => index));
    expect(whole.chunks.at(-1)!.rowOffset + whole.chunks.at(-1)!.rows.length).toBe(
      expectedRows.length - 1
    );
  });

  it('strips a BOM and detects a semicolon delimiter from the first line', () => {
    const parser = new CsvChunkParser();
    const chunks = [
      ...parser.pushBytes(new TextEncoder().encode('﻿a;b\n1;2\n')),
      ...parser.finish()
    ];
    expect(parser.header).toEqual(['a', 'b']);
    expect(chunks.flatMap(chunk => chunk.rows)).toEqual([['1', '2']]);
  });
});

describe('ResumeBatchDedupe', () => {
  it('admits new chunks, skips verified redelivery and never double counts rows', () => {
    const dedupe = new ResumeBatchDedupe();
    expect(dedupe.consider({ordinal: 0, rowCount: 2, byteOffset: 10})).toBe('admit');
    expect(dedupe.consider({ordinal: 1, rowCount: 3, byteOffset: 25})).toBe('admit');
    expect(dedupe.consider({ordinal: 0, rowCount: 2, byteOffset: 10})).toBe('skip');
    expect(dedupe.consider({ordinal: 1, rowCount: 3, byteOffset: 25})).toBe('skip');
    expect(dedupe.consider({ordinal: 2, rowCount: 1})).toBe('admit');
    expect([dedupe.admittedBatchCount, dedupe.rowCount, dedupe.skippedBatchCount]).toEqual([
      3, 6, 2
    ]);
  });

  it('fails on a changed prefix and on a skipped chunk', () => {
    const dedupe = new ResumeBatchDedupe();
    dedupe.consider({ordinal: 0, rowCount: 2, byteOffset: 10, fingerprint: 'a'});
    expect(() => dedupe.consider({ordinal: 0, rowCount: 3, byteOffset: 10})).toThrow();
    expect(() => dedupe.consider({ordinal: 0, rowCount: 2, byteOffset: 11})).toThrow();
    expect(() => dedupe.consider({ordinal: 0, rowCount: 2, fingerprint: 'b'})).toThrow();
    expect(() => dedupe.consider({ordinal: 2, rowCount: 1})).toThrow();
  });
});

const timers: Timers = {
  setTimeout: callback => {
    queueMicrotask(callback);
    return 0;
  },
  clearTimeout: () => undefined
};
const retry = {
  options: {
    maximumAttempts: 3,
    initialDelayMilliseconds: 1,
    maximumDelayMilliseconds: 2,
    multiplier: 2
  },
  timers
};

function createServer(mode: 'range' | 'norange' | 'changed', failAfter: number) {
  let requests = 0;
  const seen: (string | null)[] = [];
  const fetchImplementation = (async (_url: unknown, init?: RequestInit) => {
    const headers = new Headers(init?.headers);
    seen.push(headers.get('Range'));
    const first = requests++ === 0;
    const rangeStart = /bytes=(\d+)-/.exec(headers.get('Range') ?? '')?.[1];
    const body =
      mode === 'changed' && !first ? csvBytes.map(value => (value === 65 ? 66 : value)) : csvBytes;
    const start = mode === 'range' && rangeStart ? Number(rangeStart) : 0;
    const slice = body.slice(start);
    const responseHeaders: Record<string, string> = {
      'content-length': String(slice.length),
      etag: mode === 'changed' && !first ? '"b"' : '"a"',
      ...(mode === 'norange' ? {} : {'accept-ranges': 'bytes'})
    };
    if (start)
      responseHeaders['content-range'] = `bytes ${start}-${csvBytes.length - 1}/${csvBytes.length}`;
    let pulled = false;
    const stream = new ReadableStream<Uint8Array>({
      pull(controller) {
        if (!first) {
          controller.enqueue(slice);
          controller.close();
        } else if (!pulled) {
          pulled = true;
          controller.enqueue(slice.slice(0, failAfter));
        } else controller.error(new TypeError('network error'));
      }
    });
    return new Response(stream, {status: start ? 206 : 200, headers: responseHeaders});
  }) as typeof fetch;
  return {fetchImplementation, seen};
}

const quotedFieldCut = csvText.indexOf('line two') - 3;
const lineCut = csvText.indexOf('Béta') + 2;
const boundaryCut = new TextEncoder().encode(csvText.slice(0, csvText.indexOf('2,B'))).length;

async function load(
  server: ReturnType<typeof createServer>,
  extra: Partial<Parameters<typeof loadResumableRemoteSource>[1]> = {}
) {
  const source = await loadResumableRemoteSource('https://example.test/data.csv', {
    retry,
    fetchFunction: server.fetchImplementation,
    csvChunkRowCount: 2,
    ...extra
  });
  return source.table;
}

describe('loadResumableRemoteSource CSV', () => {
  for (const [name, cut] of [
    ['inside a quoted multi-line field', quotedFieldCut],
    ['inside a line', lineCut],
    ['exactly on a line boundary', boundaryCut]
  ] as const) {
    it(`resumes by Range when cut ${name}`, async () => {
      const server = createServer('range', cut);
      const table = await load(server);
      expect(table.numRows).toBe(expectedRows.length - 1);
      expect(server.seen[1]).toMatch(/^bytes=\d+-$/);
    });
  }

  it('matches the uncut whole-buffer table', async () => {
    const cut = await load(createServer('range', quotedFieldCut));
    const baseline = await load(createServer('range', csvBytes.length));
    expect(cut.toArray().map(row => row.toJSON())).toEqual(
      baseline.toArray().map(row => row.toJSON())
    );
    expect(cut.getChild('note')!.get(0)).toBe('line one\nline two, with "quotes"');
  });

  it('restarts and discards the delivered prefix when the server ignores Range', async () => {
    const server = createServer('norange', quotedFieldCut);
    const table = await load(server);
    expect(table.numRows).toBe(expectedRows.length - 1);
    expect(server.seen).toEqual([null, null]);
  });

  it('fails explicitly when the content changed under resume', async () => {
    await expect(load(createServer('changed', lineCut))).rejects.toThrow();
  });

  it('restarts the whole stream when resume is exhausted and dedupes redelivered chunks', async () => {
    const server = createServer('range', lineCut);
    const table = await load(server, {maximumResumes: 0});
    expect(server.seen).toEqual([null, null]);
    expect(table.numRows).toBe(expectedRows.length - 1);
  });
});
