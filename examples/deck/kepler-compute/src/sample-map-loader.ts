// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {createSampleMapProvenance, SAMPLE_MAP_METADATA_KEY} from './sample-map-provenance';
import {
  processArrowTable,
  processCsvData,
  processGeojson,
  processRowObject
} from '@kepler.gl/processors';
import type {addDataToMap} from '@kepler.gl/actions';
import {
  adaptSampleMapConfig,
  alignSampleRemoteDataset,
  type SampleMapLimitation
} from './sample-map-support';
import {probeSampleRemoteService, SampleRemoteServiceError} from './sample-map-remote.mjs';
import {
  SAMPLE_MAPS_URL,
  parseSampleMaps,
  sanitizeSampleConfig,
  getSampleUnsupportedReason,
  isSampleRecord,
  type SampleMap
} from './sample-map-catalog';

type SamplePayload = Parameters<typeof addDataToMap>[0];
type FetchSample = typeof fetch;

/** One cancellation scope owns headers, response bodies and every sibling request. */
function createSampleFetchSession(fetchRemote: FetchSample, signal?: AbortSignal) {
  const controller = new AbortController();
  const combined = AbortSignal.any([
    controller.signal,
    AbortSignal.timeout(60000),
    ...(signal ? [signal] : [])
  ]);
  const responses = new Set<Response>();
  async function read<Value>(url: string, operation: () => Promise<Value>): Promise<Value> {
    try {
      return await operation();
    } catch (error) {
      const reason = combined.aborted
        ? combined.reason?.name === 'TimeoutError'
          ? '60,000 ms download deadline exceeded'
          : `cancelled: ${String(combined.reason)}`
        : 'network or response decoding failure';
      throw new Error(
        `Sample request ${url} failed (${reason}): ${error instanceof Error ? error.message : String(error)}`,
        {cause: error}
      );
    }
  }
  return {
    signal: combined,
    read,
    fetch: (async (input, options) => {
      const response = await read(String(input), () =>
        fetchRemote(input, {
          ...options,
          signal: options?.signal ? AbortSignal.any([combined, options.signal]) : combined
        })
      );
      responses.add(response);
      return response;
    }) as FetchSample,
    async response(url: string): Promise<Response> {
      return read(url, async () => {
        const response = await fetchRemote(url, {signal: combined});
        responses.add(response);
        if (combined.aborted) {
          await response.body?.cancel().catch(() => {});
          combined.throwIfAborted();
        }
        if (!response.ok) throw new Error(`HTTP ${response.status} fetching ${url}`);
        return response;
      });
    },
    async close() {
      controller.abort();
      await Promise.allSettled([...responses].map(response => response.body?.cancel()));
    }
  };
}

export async function fetchSampleMaps(
  fetchRemote: FetchSample = fetch,
  signal?: AbortSignal
): Promise<SampleMap[]> {
  const request = createSampleFetchSession(fetchRemote, signal);
  try {
    const response = await request.response(SAMPLE_MAPS_URL);
    const result = parseSampleMaps(await request.read(SAMPLE_MAPS_URL, () => response.json()));
    await request.read(SAMPLE_MAPS_URL, async () => request.signal.throwIfAborted());
    return result;
  } finally {
    await request.close();
  }
}

export async function fetchSampleMap(
  sample: SampleMap,
  fetchRemote: FetchSample = fetch,
  signal?: AbortSignal
): Promise<SamplePayload & {limitations: SampleMapLimitation[]}> {
  const unsupported = getSampleUnsupportedReason(sample);
  if (unsupported) throw new Error(unsupported);
  const request = createSampleFetchSession(fetchRemote, signal);
  try {
    const dataUrl = sample.dataUrl ?? sample.remoteDatasetConfigUrl!;
    const dataRequest = request.response(dataUrl);
    // Observe rejection even if config parsing fails before this sibling's headers arrive.
    void dataRequest.catch(() => {});
    const configResponse = await request.response(sample.configUrl!);
    const {config, limitations} = adaptSampleMapConfig(
      sample.id,
      sanitizeSampleConfig(await request.read(sample.configUrl!, () => configResponse.json()))
    );
    const dataResponse = await dataRequest;
    let datasets: SamplePayload['datasets'];
    if (!sample.dataUrl) {
      // The demo's descriptor already has the normal addDataToMap {info,data,metadata} shape.
      const descriptor: unknown = await request.read(dataUrl, () => dataResponse.json());
      if (
        !isSampleRecord(descriptor) ||
        !isSampleRecord(descriptor['info']) ||
        typeof descriptor['info']['id'] !== 'string' ||
        !isSampleRecord(descriptor['data']) ||
        !isSampleRecord(descriptor['metadata']) ||
        descriptor['metadata']['type'] !== 'remote'
      )
        throw new Error('This config-only map has an unsupported remote dataset descriptor.');
      // Fail before Kepler's asynchronous metadata task on a definite HTTP service failure.
      // In particular, a moved PMTiles archive must not become an uncaught PMTiles page error.
      const service = await probeSampleRemoteService(descriptor, request.fetch, request.signal);
      if (!service.available) throw new SampleRemoteServiceError(service);
      datasets = alignSampleRemoteDataset(
        descriptor as Record<string, unknown> & {info: Record<string, unknown>},
        config
      ) as SamplePayload['datasets'];
    } else {
      const pathname = new URL(sample.dataUrl).pathname.toLowerCase();
      let data;
      if (pathname.endsWith('.arrow')) {
        const table = arrow.tableFromIPC(
          await request.read(dataUrl, () => dataResponse.arrayBuffer())
        );
        // Vite resolves Arrow to the host's single copy; Kepler's declaration references its nested copy.
        data = processArrowTable({shape: 'arrow-table', data: table} as unknown as Parameters<
          typeof processArrowTable
        >[0]);
      } else if (pathname.endsWith('.csv')) {
        data = processCsvData(await request.read(dataUrl, () => dataResponse.text()));
      } else {
        const rawData: unknown = await request.read(dataUrl, () => dataResponse.json());
        data = Array.isArray(rawData) ? processRowObject(rawData) : processGeojson(rawData);
      }
      if (!data || !data.fields.length)
        throw new Error('Kepler could not process the sample data.');
      datasets = {info: {id: sample.id, label: sample.label}, data};
    }
    await request.read(dataUrl, async () => request.signal.throwIfAborted());
    const provenance = createSampleMapProvenance(sample);
    const withProvenance = (Array.isArray(datasets) ? datasets : [datasets]).map(dataset => ({
      ...dataset,
      metadata: {...dataset.metadata, [SAMPLE_MAP_METADATA_KEY]: provenance}
    }));
    return {
      limitations,
      datasets: Array.isArray(datasets) ? withProvenance : withProvenance[0]!,
      // The compatibility adapter preserves opaque saved JSON; Kepler parses its v1 schema on add.
      config: config as SamplePayload['config'],
      options: {centerMap: false, keepExistingConfig: false, autoCreateLayers: false}
    };
  } finally {
    await request.close();
  }
}
