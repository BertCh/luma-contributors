// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// @ts-check

/** @param {unknown} value @returns {value is Record<string, unknown>} */
function isRecord(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** @param {unknown} descriptor @returns {string} */
export function getSampleRemoteServiceUrl(descriptor) {
  const metadata =
    isRecord(descriptor) && isRecord(descriptor['metadata']) ? descriptor['metadata'] : {};
  const info = isRecord(descriptor) && isRecord(descriptor['info']) ? descriptor['info'] : {};
  const address = metadata['tilesetMetadataUrl'] || metadata['tilesetDataUrl'];
  if (typeof address !== 'string') throw new Error('Remote sample has no service URL.');
  const url = new URL(address);
  if (!['https:', 'http:'].includes(url.protocol))
    throw new Error('Remote sample requires an HTTP service URL.');
  if (info['type'] === 'wms-tile') {
    // Probe capabilities, not a WMS endpoint without the required operation parameters.
    for (const key of [...url.searchParams.keys()]) {
      if (['service', 'request'].includes(key.toLowerCase())) url.searchParams.delete(key);
    }
    url.searchParams.set('service', 'WMS');
    url.searchParams.set('request', 'GetCapabilities');
  }
  return url.href;
}

/**
 * A bounded range request also checks GET access (HEAD is not supported by every tile service).
 * Cancel the body: a server may ignore Range, and PMTiles archives must never be downloaded here.
 * @param {unknown} descriptor
 * @param {typeof fetch} fetchRemote
 * @param {AbortSignal} [signal]
 */
export async function probeSampleRemoteService(descriptor, fetchRemote = fetch, signal) {
  const url = getSampleRemoteServiceUrl(descriptor);
  const response = await fetchRemote(url, {
    headers: {Range: 'bytes=0-0'},
    signal: AbortSignal.any([AbortSignal.timeout(20000), ...(signal ? [signal] : [])])
  });
  await response.body?.cancel();
  return {url, status: response.status, available: response.ok};
}

export class SampleRemoteServiceError extends Error {
  code = 'remote-service-unavailable';
  /** @param {{url: string, status: number}} result */
  constructor(result) {
    super(
      `Remote service unavailable (HTTP ${result.status}): ${result.url}. The sample's upstream service is unavailable; retry when the provider restores it.`
    );
    this.name = 'SampleRemoteServiceError';
    this.url = result.url;
    this.status = result.status;
  }
}
