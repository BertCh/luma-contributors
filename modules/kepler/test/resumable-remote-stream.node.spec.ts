// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  createBatchDeduper,
  openResumableRemoteStream,
  type ResumeEvent
} from '../../../examples/deck/kepler-compute/src/resumable-remote-stream';
import {
  planRemoteRefetch,
  runRemoteRefetch
} from '../../../examples/deck/kepler-compute/src/remote-refetch-recovery';
import {collectVisibleDatasetIds} from '../../../examples/deck/kepler-compute/src/resident-eviction-host';
import type {Timers} from '../../../examples/deck/kepler-compute/src/automatic-recovery-policy';

const payload = Uint8Array.from({length: 1000}, (_, index) => index % 251);
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

/** Server that fails the first body after `failAfter` bytes, then serves per `mode`. */
function createServer(mode: 'range' | 'norange' | 'changed', failAfter: number) {
  let requests = 0;
  const seen: Record<string, string | null>[] = [];
  const fetchImplementation = (async (_url: unknown, init?: RequestInit) => {
    const headers = new Headers(init?.headers);
    seen.push({range: headers.get('Range'), ifRange: headers.get('If-Range')});
    const first = requests++ === 0;
    const rangeStart = /bytes=(\d+)-/.exec(headers.get('Range') ?? '')?.[1];
    const body = mode === 'changed' && !first ? payload.map(value => value ^ 1) : payload;
    const start = mode === 'range' && rangeStart ? Number(rangeStart) : 0;
    const slice = body.slice(start);
    const responseHeaders: Record<string, string> = {
      'content-length': String(slice.length),
      etag: mode === 'changed' && !first ? '"b"' : '"a"',
      ...(mode === 'norange' ? {} : {'accept-ranges': 'bytes'})
    };
    if (start)
      responseHeaders['content-range'] = `bytes ${start}-${payload.length - 1}/${payload.length}`;
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

async function readAll(stream: ReadableStream<Uint8Array>): Promise<Uint8Array> {
  const chunks: Uint8Array[] = [];
  const reader = stream.getReader();
  for (let result = await reader.read(); !result.done; result = await reader.read())
    chunks.push(result.value);
  const output = new Uint8Array(chunks.reduce((total, chunk) => total + chunk.length, 0));
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.length;
  }
  return output;
}

describe('openResumableRemoteStream', () => {
  const open = (server: ReturnType<typeof createServer>, events: ResumeEvent[]) =>
    openResumableRemoteStream({
      url: new URL('https://example.test/data.arrows'),
      signal: new AbortController().signal,
      maximumInputBytes: 1 << 20,
      retry,
      fetchImplementation: server.fetchImplementation,
      onEvent: event => events.push(event)
    });

  it('resumes by Range with If-Range and delivers each byte exactly once', async () => {
    const server = createServer('range', 400);
    const events: ResumeEvent[] = [];
    expect(await readAll(await open(server, events))).toEqual(payload);
    expect(server.seen[1]).toEqual({range: 'bytes=400-', ifRange: '"a"'});
    expect(events.at(-1)?.phase).toBe('resumed');
  });

  it('restarts and discards the delivered prefix when ranges are unsupported', async () => {
    const server = createServer('norange', 400);
    const events: ResumeEvent[] = [];
    expect(await readAll(await open(server, events))).toEqual(payload);
    expect(server.seen[1]?.['range']).toBeNull();
    expect(events.at(-1)?.phase).toBe('restarted');
  });

  it('fails explicitly when the representation changed', async () => {
    const server = createServer('changed', 400);
    const events: ResumeEvent[] = [];
    await expect(readAll(await open(server, events))).rejects.toThrow(/changed/);
    expect(events.at(-1)?.phase).toBe('failed');
  });
});

describe('createBatchDeduper', () => {
  it('skips admitted batches and validates the row boundary', () => {
    const dedupe = createBatchDeduper({batchCount: 2, rowCount: 10});
    expect([dedupe(4), dedupe(6), dedupe(3)]).toEqual(['skip', 'skip', 'admit']);
    const mismatch = createBatchDeduper({batchCount: 1, rowCount: 10});
    expect(() => mismatch(9)).toThrow(/differs/);
  });
});

describe('remote refetch recovery', () => {
  it('plans new dataset ids and records failures per source', async () => {
    const plan = planRemoteRefetch(
      [
        {datasetId: 'a', label: 'A', stream: {origin: {kind: 'remote', url: 'https://x.test/a'}}},
        {datasetId: 'b', label: 'B'}
      ],
      new Set(['a', 'a-refetch-1'])
    );
    expect(plan.map(item => item.newDatasetId)).toEqual(['a-refetch-2']);
    const statuses = await runRemoteRefetch({
      plan,
      signal: new AbortController().signal,
      refetch: async () => {
        throw new Error('HTTP 500');
      }
    });
    expect(statuses[0]).toMatchObject({phase: 'failed', error: 'HTTP 500'});
  });
});

describe('collectVisibleDatasetIds lineage', () => {
  it('keeps inputs of visible results resident', () => {
    const visible = collectVisibleDatasetIds({
      layers: [{isVisible: true, dataId: 'result'}],
      filters: [],
      pinnedIds: ['base'],
      lineage: [
        {resultId: 'result', inputIds: ['mid']},
        {resultId: 'mid', inputIds: ['source']}
      ]
    });
    expect([...visible].sort()).toEqual(['base', 'mid', 'result', 'source']);
  });
});
