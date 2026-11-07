// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerDatasetPopulation, KeplerJSONValue} from '@luma-contributors/kepler';

/**
 * Recorded stream provenance for a published native source. It is the only stream information
 * saved in the source manifest: a remote origin URL (no credentials, no response headers) and an
 * explicit partial marker. Source rows always travel as bundle Arrow bytes or recovery snapshots.
 *
 * Recovery policy: device recovery and bundle restore rehydrate from the retained or bundled,
 * checksummed Arrow bytes and never fetch silently. The recorded URL is for explicit user
 * refetch only; a refetch is a new stream that is added as a new dataset (never an in-place
 * replacement of a referenced dataset id) and its failure is shown as an explicit failed phase.
 * Optional bounded exponential-backoff retry covers only the request, before any body byte is
 * admitted, so a retry can never duplicate rows; exhausted retries fall back to that explicit phase.
 */
export type NativeStreamProvenance = {
  origin?: {kind: 'remote'; url: string};
  partial?: {rowCount: number; batchCount: number};
};

export const STREAM_ORIGIN_METADATA_KEY = 'nativeStreamOrigin';

export function parseRemoteStreamUrl(value: string, base = globalThis.location?.href): URL {
  const url = new URL(value, base);
  // Credentials in the URL would be written into shareable manifests.
  if ((url.protocol !== 'https:' && url.protocol !== 'http:') || url.username || url.password)
    throw new Error('Remote streams require an http(s) URL without credentials');
  return url;
}

/** Fetch a bounded, abortable byte stream. The caller still enforces byte and chunk capacities. */
export async function openRemoteStream(
  url: URL,
  signal: AbortSignal,
  maximumInputBytes: number,
  fetchImplementation: typeof fetch = fetch
): Promise<ReadableStream<Uint8Array>> {
  const response = await fetchImplementation(url, {signal, credentials: 'omit', cache: 'no-store'});
  if (!response.ok)
    throw Object.assign(new Error(`Remote stream request failed with HTTP ${response.status}`), {
      status: response.status
    });
  if (!response.body) throw new Error('Remote stream response has no readable body');
  const declared = Number(response.headers.get('content-length'));
  if (Number.isFinite(declared) && declared > maximumInputBytes) {
    void response.body.cancel().catch(() => undefined);
    throw new Error('Remote IPC stream exceeds the configured input limit');
  }
  return response.body;
}

export function getStreamProvenance(
  population: KeplerDatasetPopulation | undefined,
  metadata: Readonly<Record<string, unknown>> | undefined
): NativeStreamProvenance | undefined {
  const origin = metadata?.[STREAM_ORIGIN_METADATA_KEY] as NativeStreamProvenance['origin'];
  const partial = population?.partial;
  if (!origin && !partial) return undefined;
  return {...(origin ? {origin: {...origin}} : {}), ...(partial ? {partial: {...partial}} : {})};
}

export function serializeStreamProvenance(provenance: NativeStreamProvenance): KeplerJSONValue {
  return {
    ...(provenance.origin ? {origin: {kind: 'remote', url: provenance.origin.url}} : {}),
    ...(provenance.partial
      ? {
          partial: {
            rowCount: provenance.partial.rowCount,
            batchCount: provenance.partial.batchCount
          }
        }
      : {})
  };
}

/** Strict reader; unknown shapes fail restore rather than being treated as finite sources. */
export function parseStreamProvenance(value: unknown): NativeStreamProvenance | undefined {
  if (value === undefined || value === null) return undefined;
  if (typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid stream source');
  const record = value as Record<string, unknown>;
  const result: NativeStreamProvenance = {};
  if (record['origin'] !== undefined) {
    const origin = record['origin'] as Record<string, unknown>;
    if (!origin || origin['kind'] !== 'remote' || typeof origin['url'] !== 'string')
      throw new Error('Invalid stream origin');
    result.origin = {kind: 'remote', url: parseRemoteStreamUrl(origin['url'], undefined).href};
  }
  if (record['partial'] !== undefined) {
    const partial = record['partial'] as Record<string, unknown>;
    if (
      !partial ||
      !Number.isSafeInteger(partial['rowCount']) ||
      !Number.isSafeInteger(partial['batchCount']) ||
      (partial['rowCount'] as number) < 0 ||
      (partial['batchCount'] as number) < 1
    )
      throw new Error('Invalid partial population');
    result.partial = {
      rowCount: partial['rowCount'] as number,
      batchCount: partial['batchCount'] as number
    };
  }
  return result;
}

export function getStreamImportOptions(provenance: NativeStreamProvenance | undefined): {
  population?: KeplerDatasetPopulation;
  metadata?: Record<string, unknown>;
} {
  if (!provenance) return {};
  return {
    ...(provenance.partial
      ? {
          population: {
            kind: 'stream' as const,
            streamId: 'restored-partial-stream',
            state: 'complete' as const,
            partial: provenance.partial
          }
        }
      : {}),
    ...(provenance.origin ? {metadata: {[STREAM_ORIGIN_METADATA_KEY]: provenance.origin}} : {})
  };
}

/** Idempotent visible label. Partial populations stay usable but never read as complete. */
export function getPopulationLabel(
  label: string,
  population: KeplerDatasetPopulation | undefined
): string {
  const partial = population?.partial;
  if (!partial) return label;
  const suffix = ` (partial: ${partial.rowCount.toLocaleString('en-US')} of ? rows, ${partial.batchCount} batches)`;
  return label.endsWith(suffix) ? label : `${label}${suffix}`;
}
