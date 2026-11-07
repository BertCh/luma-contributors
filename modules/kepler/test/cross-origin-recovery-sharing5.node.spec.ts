// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, it} from 'vitest';
import {
  openResumableRemoteStream,
  type ResumeEvent
} from '../../../examples/deck/kepler-compute/src/resumable-remote-stream';
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

/**
 * Cross-origin server without Access-Control-Expose-Headers: only CORS-safelisted response
 * headers (Content-Length, Last-Modified) are readable; ETag, Accept-Ranges and Content-Range are hidden.
 */
function createCrossOriginServer(
  mode: 'range' | 'preflight-blocked' | 'wrong-offset',
  failAfter: number
) {
  let requests = 0;
  const seen: Record<string, string | null>[] = [];
  const fetchImplementation = (async (_url: unknown, init?: RequestInit) => {
    const headers = new Headers(init?.headers);
    seen.push({range: headers.get('Range'), ifRange: headers.get('If-Range')});
    if (mode === 'preflight-blocked' && headers.get('Range'))
      throw new TypeError('Failed to fetch (preflight rejected)');
    const first = requests++ === 0;
    const rangeStart = Number(/bytes=(\d+)-/.exec(headers.get('Range') ?? '')?.[1] ?? 0);
    const start =
      mode === 'wrong-offset' && rangeStart
        ? rangeStart - 10
        : mode === 'range' || mode === 'wrong-offset'
          ? rangeStart
          : 0;
    const slice = payload.slice(start);
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
    return new Response(stream, {
      status: start ? 206 : 200,
      headers: {
        'content-length': String(slice.length),
        'last-modified': 'Tue, 01 Jan 2030 00:00:00 GMT'
      }
    });
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

describe('cross-origin resumable streams', () => {
  const open = (server: ReturnType<typeof createCrossOriginServer>, events: ResumeEvent[]) =>
    openResumableRemoteStream({
      url: new URL('https://other-origin.test/data.arrows'),
      signal: new AbortController().signal,
      maximumInputBytes: 1 << 20,
      optimisticRanges: true,
      retry,
      fetchImplementation: server.fetchImplementation,
      onEvent: event => events.push(event)
    });

  it('resumes by Range with If-Range from Last-Modified when Accept-Ranges and Content-Range are hidden', async () => {
    const server = createCrossOriginServer('range', 400);
    const events: ResumeEvent[] = [];
    expect(await readAll(await open(server, events))).toEqual(payload);
    expect(server.seen[1]).toEqual({range: 'bytes=400-', ifRange: 'Tue, 01 Jan 2030 00:00:00 GMT'});
    expect(events.at(-1)?.phase).toBe('resumed');
  });

  it('falls back to restart-and-discard when the Range preflight is rejected', async () => {
    const server = createCrossOriginServer('preflight-blocked', 400);
    const events: ResumeEvent[] = [];
    expect(await readAll(await open(server, events))).toEqual(payload);
    expect(events.at(-1)?.phase).toBe('restarted');
  });

  it('rejects a hidden-range 206 whose length does not match the missing bytes', async () => {
    const server = createCrossOriginServer('wrong-offset', 400);
    const events: ResumeEvent[] = [];
    await expect(readAll(await open(server, events))).rejects.toThrow(/unexpected byte range/);
    expect(events.at(-1)?.phase).toBe('failed');
  });
});
