// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  isTransientFetchError,
  retryWithBackoff,
  type BackoffOptions,
  type Timers
} from './automatic-recovery-policy';

/**
 * Byte-exact resumable remote body for Arrow IPC and CSV streams. A mid-body failure reopens the
 * request and continues from the number of bytes already delivered, so the consumer (which admits
 * whole batches/rows from those bytes) can never see a duplicated or skipped byte:
 *
 * - Range mode: `Range: bytes=N-` plus `If-Range: <strong ETag or Last-Modified>`; only a 206 whose
 *   Content-Range starts exactly at N is accepted.
 * - Restart mode (no validator, no Accept-Ranges, content-encoded body, or the server answered 200):
 *   restart from byte 0 and discard the first N bytes. The response validator must still match and
 *   the last bytes before N are compared with the retained tail, so changed content fails explicitly
 *   instead of being spliced into the stream.
 *
 * Retries are bounded by the backoff policy per failure and by `maximumResumes` per stream.
 */
export type ResumePhase = 'retrying' | 'resumed' | 'restarted' | 'failed';
export type ResumeEvent = Readonly<{
  phase: ResumePhase;
  /** Failure episodes handled so far. */
  resumes: number;
  bytesDelivered: number;
  error?: string;
}>;
export type ResumableRemoteStreamProps = {
  url: URL;
  signal: AbortSignal;
  maximumInputBytes: number;
  retry: {options: BackoffOptions; timers?: Timers};
  /** Total resume episodes allowed for one stream; defaults to 5. */
  maximumResumes?: number;
  fetchImplementation?: typeof fetch;
  /**
   * Treat a hidden Accept-Ranges as "try it" (cross-origin responses hide it without CORS
   * exposure). Defaults to true only when the URL origin differs from the page origin.
   */
  optimisticRanges?: boolean;
  onEvent?(event: ResumeEvent): void;
};

const TAIL_BYTES = 64;

export class ContentChangedError extends Error {}

type Validator = {etag?: string; lastModified?: string};

export async function openResumableRemoteStream(
  props: ResumableRemoteStreamProps
): Promise<ReadableStream<Uint8Array>> {
  const fetchImplementation = props.fetchImplementation ?? fetch;
  const maximumResumes = props.maximumResumes ?? 5;
  const optimisticRanges =
    props.optimisticRanges ??
    (typeof location !== 'undefined' && props.url.origin !== location.origin);
  let delivered = 0;
  let resumes = 0;
  let tail = new Uint8Array(0);
  let validator: Validator = {};
  let canUseRange = false;
  let totalBytes: number | undefined;
  let reader: ReadableStreamDefaultReader<Uint8Array> | undefined;
  let lastMode: 'range' | 'restart' = 'restart';
  let rangeBlocked = false;

  const emit = (phase: ResumePhase, error?: unknown) =>
    props.onEvent?.({
      phase,
      resumes,
      bytesDelivered: delivered,
      ...(error === undefined
        ? {}
        : {error: error instanceof Error ? error.message : String(error)})
    });

  async function openAt(offset: number): Promise<ReadableStreamDefaultReader<Uint8Array>> {
    const headers = new Headers();
    const useRange = offset > 0 && canUseRange;
    if (useRange) {
      headers.set('Range', `bytes=${offset}-`);
      headers.set('If-Range', validator.etag ?? validator.lastModified!);
    }
    let response: Response;
    try {
      response = await fetchImplementation(props.url, {
        signal: props.signal,
        credentials: 'omit',
        cache: 'no-store',
        headers
      });
    } catch (error) {
      // A cross-origin server that does not allow Range/If-Range fails the preflight with a
      // TypeError. Give up on ranges for this stream and restart from byte 0 instead.
      if (!useRange || props.signal.aborted || !(error instanceof TypeError)) throw error;
      rangeBlocked = true;
      canUseRange = false;
      return openAt(offset);
    }
    if (!response.ok)
      throw Object.assign(new Error(`Remote stream request failed with HTTP ${response.status}`), {
        status: response.status
      });
    if (!response.body) throw new Error('Remote stream response has no readable body');
    const declared = Number(response.headers.get('content-length'));
    const responseValidator: Validator = {
      etag: getStrongEtag(response.headers),
      lastModified: response.headers.get('last-modified') ?? undefined
    };
    const encoded = isContentEncoded(response.headers);
    if (response.status === 206) {
      // Content-Range must begin exactly where delivery stopped.
      const contentRange = response.headers.get('content-range');
      const start = /^bytes (\d+)-/.exec(contentRange ?? '')?.[1];
      // A hidden Content-Range (cross-origin without exposure) is accepted only when the readable
      // Content-Length equals exactly the bytes still missing, which a wrong offset cannot satisfy.
      const hiddenRangeVerified =
        contentRange === null &&
        totalBytes !== undefined &&
        Number.isFinite(declared) &&
        declared === totalBytes - offset;
      if (!useRange || (hiddenRangeVerified ? false : Number(start) !== offset)) {
        void response.body.cancel().catch(() => undefined);
        throw new ContentChangedError('Remote stream returned an unexpected byte range');
      }
      lastMode = 'range';
      return response.body.getReader();
    }
    lastMode = 'restart';
    if (offset === 0) {
      if (Number.isFinite(declared) && declared > props.maximumInputBytes) {
        void response.body.cancel().catch(() => undefined);
        throw new ContentChangedError('Remote IPC stream exceeds the configured input limit');
      }
      validator = responseValidator;
      const acceptRanges = response.headers.get('accept-ranges');
      // Cross-origin responses hide Accept-Ranges unless the server exposes it. A hidden header
      // (null) is tried optimistically with a validator when optimisticRanges is set; an explicit `none` is respected. A 200
      // answer or a failed preflight falls back to restart-and-discard below.
      canUseRange =
        !encoded &&
        (acceptRanges === null ? optimisticRanges : /\bbytes\b/i.test(acceptRanges)) &&
        !rangeBlocked &&
        Boolean(validator.etag ?? validator.lastModified);
      totalBytes = !encoded && Number.isFinite(declared) ? declared : undefined;
      return response.body.getReader();
    }
    // Restart: the server returned the full body. It must be the same representation.
    if (
      (validator.etag && responseValidator.etag !== validator.etag) ||
      (!validator.etag &&
        validator.lastModified &&
        responseValidator.lastModified !== validator.lastModified)
    ) {
      void response.body.cancel().catch(() => undefined);
      throw new ContentChangedError('Remote stream changed while it was being read');
    }
    return skipDeliveredPrefix(response.body.getReader(), offset);
  }

  /** Discards `offset` bytes, requiring the bytes just before it to equal the delivered tail. */
  async function skipDeliveredPrefix(
    source: ReadableStreamDefaultReader<Uint8Array>,
    offset: number
  ): Promise<ReadableStreamDefaultReader<Uint8Array>> {
    const compareFrom = offset - tail.length;
    const observed = new Uint8Array(tail.length);
    let position = 0;
    let remainder: Uint8Array | undefined;
    while (position < offset) {
      const result = await source.read();
      if (result.done) throw new Error('Remote stream ended before the resume offset');
      const chunk = result.value;
      const end = position + chunk.length;
      if (end > compareFrom)
        observed.set(
          chunk.subarray(
            Math.max(0, compareFrom - position),
            Math.min(chunk.length, offset - position)
          ),
          Math.max(0, position - compareFrom)
        );
      if (end > offset) remainder = chunk.subarray(offset - position);
      position = end;
    }
    for (let index = 0; index < tail.length; index++)
      if (observed[index] !== tail[index]) {
        void source.cancel().catch(() => undefined);
        throw new ContentChangedError(
          'Remote stream bytes differ from the already delivered prefix'
        );
      }
    if (!remainder?.length) return source;
    let pending: Uint8Array | undefined = remainder;
    return {
      read: async () => {
        if (pending) {
          const value = pending;
          pending = undefined;
          return {done: false, value};
        }
        return source.read();
      },
      cancel: reason => source.cancel(reason),
      releaseLock: () => source.releaseLock(),
      closed: source.closed
    } as ReadableStreamDefaultReader<Uint8Array>;
  }

  async function recoverFrom(error: unknown): Promise<void> {
    void reader?.cancel().catch(() => undefined);
    reader = undefined;
    if (
      error instanceof ContentChangedError ||
      props.signal.aborted ||
      resumes >= maximumResumes ||
      !isTransientFetchError(error)
    ) {
      emit('failed', error);
      throw error;
    }
    resumes++;
    const offset = delivered;
    try {
      reader = await retryWithBackoff(() => openAt(offset), {
        options: props.retry.options,
        signal: props.signal,
        shouldRetry: failure =>
          !(failure instanceof ContentChangedError) && isTransientFetchError(failure),
        onRetry: failure => emit('retrying', failure),
        ...(props.retry.timers ? {timers: props.retry.timers} : {})
      });
    } catch (failure) {
      emit('failed', failure);
      throw failure;
    }
    emit(lastMode === 'range' ? 'resumed' : 'restarted');
  }

  reader = await openAt(0);
  return new ReadableStream<Uint8Array>({
    async pull(controller) {
      while (true) {
        try {
          const result = await reader!.read();
          if (result.done) {
            // A premature end (known length not reached) is a mid-body failure, not completion.
            if (totalBytes !== undefined && delivered < totalBytes)
              throw new TypeError('Remote stream ended before its declared length');
            controller.close();
            return;
          }
          delivered += result.value.length;
          if (delivered > props.maximumInputBytes)
            throw new ContentChangedError('Remote IPC stream exceeds the configured input limit');
          const keep = new Uint8Array(Math.min(TAIL_BYTES, tail.length + result.value.length));
          const combined = new Uint8Array(tail.length + result.value.length);
          combined.set(tail);
          combined.set(result.value, tail.length);
          keep.set(combined.subarray(combined.length - keep.length));
          tail = keep;
          controller.enqueue(result.value);
          return;
        } catch (error) {
          await recoverFrom(error);
        }
      }
    },
    cancel(reason) {
      return reader?.cancel(reason);
    }
  });
}

function getStrongEtag(headers: Headers): string | undefined {
  const etag = headers.get('etag');
  return etag && !etag.startsWith('W/') ? etag : undefined;
}

function isContentEncoded(headers: Headers): boolean {
  const encoding = headers.get('content-encoding');
  return Boolean(encoding && encoding.toLowerCase() !== 'identity');
}

/**
 * Restart-and-dedupe by admitted batch count, for consumers that cannot rely on byte identity
 * (for example a re-encoded export). Returns a filter that drops the first `admittedBatches`
 * batches of the restarted stream after checking their cumulative row count equals what was
 * admitted, and throws when the restarted prefix does not match.
 */
export function createBatchDeduper(admitted: {
  batchCount: number;
  rowCount: number;
}): (rowsInBatch: number) => 'skip' | 'admit' {
  let seenBatches = 0;
  let seenRows = 0;
  return rowsInBatch => {
    if (seenBatches >= admitted.batchCount) return 'admit';
    seenBatches++;
    seenRows += rowsInBatch;
    // Row count equality at the boundary guards against a changed or reordered source.
    if (seenBatches === admitted.batchCount && seenRows !== admitted.rowCount)
      throw new Error('Restarted stream prefix differs from the admitted rows');
    return 'skip';
  };
}
