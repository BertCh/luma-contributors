// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, test, vi} from 'vitest';
import * as arrow from 'apache-arrow';
import {
  adaptSampleMapConfig,
  alignSampleRemoteDataset
} from '../../../examples/deck/kepler-compute/src/sample-map-support';
import {
  getSampleRemoteServiceUrl,
  probeSampleRemoteService,
  SampleRemoteServiceError
} from '../../../examples/deck/kepler-compute/src/sample-map-remote.mjs';
import {withSyntheticSample} from '../../../examples/deck/kepler-compute/scripts/native-navigation.mjs';
import {
  describeSampleFallback,
  describeSampleLicense,
  extractSampleAttribution,
  getSampleUnsupportedReason,
  parseSampleMaps,
  parseStartupSample,
  sanitizeSampleConfig,
  SAMPLE_MAP_BACKGROUND_STYLE,
  type SampleMap
} from '../../../examples/deck/kepler-compute/src/sample-map-catalog';
import {
  fetchSampleMap,
  fetchSampleMaps
} from '../../../examples/deck/kepler-compute/src/sample-map-loader';

// The native processors import Kepler's Deck 9 math shim, installed by Vite only in the browser.
// Test the I/O/action boundary here; scripts/kepler-demo-data.mjs exercises the real processors.
const processors = vi.hoisted(() => ({
  processArrowTable: vi.fn(() => ({fields: [{name: 'arrow'}], rows: []})),
  processCsvData: vi.fn(() => ({fields: [{name: 'csv'}], rows: []})),
  processGeojson: vi.fn(() => ({fields: [{name: 'geojson'}], rows: []})),
  processRowObject: vi.fn(() => ({fields: [{name: 'rows'}], rows: []}))
}));
vi.mock('@kepler.gl/processors', () => processors);

const sample: SampleMap = {
  id: 'observations',
  label: 'Observations',
  description: 'Generated test observations',
  detail: 'Data source: [Provider](https://example.com/data)',
  visible: true,
  dataUrl: 'https://example.com/observations.csv',
  configUrl: 'https://example.com/config.json'
};
const config = {
  version: 'v1',
  config: {
    visState: {
      layers: [{id: 'points', type: 'point', config: {dataId: sample.id}}],
      filters: [{id: 'magnitude', value: [2, 8]}],
      splitMaps: [{layers: {points: true}}, {layers: {points: false}}]
    },
    mapState: {latitude: 34, longitude: -118, zoom: 7},
    mapStyle: {styleType: 'dark'}
  }
};

test('catalog retains config-only, hidden and irregularly named samples without inventing URLs', () => {
  const maps = parseSampleMaps([
    {...sample, id: 'world_flights', dataUrl: 'https://example.com/world_flights_soei4h.json'},
    {
      ...sample,
      id: 'remote',
      dataUrl: false,
      remoteDatasetConfigUrl: 'https://example.com/remote.json'
    },
    {id: 'hidden', label: 'Hidden', visible: false}
  ]);
  expect(maps).toHaveLength(3);
  expect(maps[0]?.dataUrl).toContain('world_flights_soei4h.json');
  expect(maps[1]?.dataUrl).toBeUndefined();
  expect(getSampleUnsupportedReason(maps[1]!)).toBeUndefined();
  expect(maps[2]?.visible).toBe(false);
  expect(getSampleUnsupportedReason(maps[2]!)).toContain('No map config');
  expect(() => parseSampleMaps([sample, sample])).toThrow('duplicate');
  expect(() => parseSampleMaps({})).toThrow('array');
});

test('attribution preserves provider links as safe text and keeps data terms distinct from code licence', () => {
  expect(extractSampleAttribution(sample.detail)).toEqual([
    {text: 'Data source: '},
    {text: 'Provider', url: 'https://example.com/data'}
  ]);
  expect(extractSampleAttribution('[unsafe](javascript:alert)')).toEqual([
    {text: 'unsafe', url: undefined}
  ]);
  expect(describeSampleLicense(sample)).toContain('does not specify a dataset licence');
  expect(
    describeSampleLicense({...sample, detail: 'Creative Commons Noncommercial Attribution License'})
  ).toContain('Noncommercial');
});

test('startup defaults to earthquakes and explicit row fixtures take precedence', () => {
  expect(parseStartupSample('')).toEqual({sampleId: 'earthquakes', rowCount: 100000});
  expect(parseStartupSample('?sample=nyc_earnings').sampleId).toBe('nyc_earnings');
  expect(parseStartupSample('?sample=synthetic').sampleId).toBe('synthetic');
  expect(parseStartupSample('?rows=256&sample=earthquakes')).toEqual({
    sampleId: 'synthetic',
    rowCount: 256
  });
  for (const rows of ['0', '-1', 'NaN', '1.5', '10000001'])
    expect(parseStartupSample(`?rows=${rows}`)).toEqual({sampleId: 'synthetic', rowCount: 100000});
  expect(describeSampleFallback('earthquakes', new Error('offline'))).toContain('offline');
  expect(describeSampleFallback('earthquakes', 'offline')).toContain('Showing synthetic squares');
});

test('legacy browser flows request synthetic data while preserving row and policy parameters', () => {
  expect(
    parseStartupSample(new URL(withSyntheticSample('http://localhost:5320/')).search).sampleId
  ).toBe('synthetic');
  const url = new URL(withSyntheticSample('http://localhost:5320/?rows=128&autoPrepare=0'));
  expect(parseStartupSample(url.search)).toEqual({sampleId: 'synthetic', rowCount: 128});
  expect(url.searchParams.get('autoPrepare')).toBe('0');
});

test('config sanitation preserves layers, filters, split views and camera without mutating the source', () => {
  const sanitized = sanitizeSampleConfig(config);
  expect(sanitized).toEqual({
    ...config,
    config: {
      ...config.config,
      mapStyle: {
        styleType: 'compute-background',
        mapStyles: {'compute-background': SAMPLE_MAP_BACKGROUND_STYLE}
      }
    }
  });
  expect(config.config.mapStyle.styleType).toBe('dark');
  expect(() => sanitizeSampleConfig({})).toThrow('Kepler v1');
});

test('sample basemaps retain presentation controls but remove remote styles and credentials', () => {
  const input = {
    ...structuredClone(config),
    config: {
      ...structuredClone(config.config),
      mapStyle: {
        styleType: 'muted_night',
        visibleLayerGroups: {water: true, label: false},
        topLayerGroups: {label: true},
        backgroundColor: [1, 2, 3],
        url: 'https://example.com/style.json',
        accessToken: 'sample-token',
        mapStyles: {remote: {url: 'https://example.com/style.json', accessToken: 'sample-token'}}
      }
    }
  };
  const sanitized = sanitizeSampleConfig(input) as {config: {mapStyle: Record<string, unknown>}};
  expect(sanitized.config.mapStyle).toEqual({
    styleType: 'compute-background',
    visibleLayerGroups: {water: true, label: false},
    topLayerGroups: {label: true},
    backgroundColor: [1, 2, 3],
    mapStyles: {'compute-background': SAMPLE_MAP_BACKGROUND_STYLE}
  });
  expect(JSON.stringify(sanitized.config.mapStyle)).not.toContain('https://');
  expect(JSON.stringify(sanitized.config.mapStyle)).not.toContain('sample-token');
});

test('named sample limitations preserve original settings and filters with explicit location fallbacks', () => {
  for (const [sampleId, type, code] of [
    ['sftrees', 'hexagon', 'dense-aggregation-capacity'],
    ['la_assessorparcels', 'point', 'wide-integer-style']
  ] as const) {
    const originalLayer = {
      id: 'original',
      type,
      config: {
        dataId: sampleId,
        isVisible: true,
        columns: {lat: 'latitude', lng: 'longitude'},
        color: [1, 2, 3],
        visConfig: {worldUnitSize: 0.05}
      },
      visualChannels: {sizeField: {name: 'roll_totalvalue', type: 'integer'}}
    };
    const saved = {
      version: 'v1',
      config: {
        visState: {
          layers: [originalLayer],
          filters: [{id: 'filter', value: [2000000, 40000000]}],
          splitMaps: [{layers: {original: true}}, {layers: {original: false}}],
          layerOrder: ['original']
        }
      }
    };
    const before = JSON.stringify(saved);
    const result = adaptSampleMapConfig(sampleId, saved);
    expect(result.limitations).toMatchObject([
      {code, layerId: 'original', fallbackLayerId: 'original-sample-locations'}
    ]);
    const adapted = result.config.config.visState;
    expect(adapted.layers).toHaveLength(2);
    expect(adapted.layers[0]).toEqual({
      ...originalLayer,
      config: {...originalLayer.config, isVisible: false}
    });
    expect(adapted.layers[1]).toMatchObject({
      type: 'point',
      visualChannels: {colorField: null, sizeField: null}
    });
    expect((adapted as Record<string, unknown>)['filters']).toEqual(saved.config.visState.filters);
    expect(adapted.splitMaps).toEqual([
      {layers: {original: true, 'original-sample-locations': true}},
      {layers: {original: false, 'original-sample-locations': false}}
    ]);
    expect(JSON.stringify(saved)).toBe(before);
    expect(adaptSampleMapConfig('other-sample', saved).limitations).toEqual([]);
  }
});

function createFetch(data: BodyInit, requests: string[]): typeof fetch {
  return async input => {
    const url = String(input);
    requests.push(url);
    return new Response(url.endsWith('config.json') ? JSON.stringify(config) : data);
  };
}

test('runtime loading delegates CSV/GeoJSON/row/Arrow to public processors and preserves map config', async () => {
  const cases: {extension: string; content: BodyInit}[] = [
    {extension: 'csv', content: 'longitude,latitude,value\n-118,34,4\n-117,35,6'},
    {
      extension: 'geojson',
      content: JSON.stringify({
        type: 'FeatureCollection',
        features: [
          {
            type: 'Feature',
            properties: {value: 4},
            geometry: {type: 'Point', coordinates: [-118, 34]}
          }
        ]
      })
    },
    {extension: 'json', content: JSON.stringify([{longitude: -118, latitude: 34, value: 4}])},
    {
      extension: 'arrow',
      content: arrow
        .tableToIPC(arrow.tableFromArrays({longitude: [-118], latitude: [34], value: [4]}))
        .slice().buffer as ArrayBuffer
    }
  ];
  for (const entry of cases) {
    Object.values(processors).forEach(processor => processor.mockClear());
    const requests: string[] = [];
    const currentSample = {
      ...sample,
      dataUrl: `https://example.com/custom-name.${entry.extension}`
    };
    const payload = await fetchSampleMap(currentSample, createFetch(entry.content, requests));
    expect(requests.sort()).toEqual([sample.configUrl!, currentSample.dataUrl].sort());
    expect(payload.config).toEqual(sanitizeSampleConfig(config));
    expect(payload.options).toMatchObject({centerMap: false, autoCreateLayers: false});
    expect(payload.datasets).toMatchObject({info: {id: sample.id, label: sample.label}});
    const dataset = payload.datasets as {data: {fields: unknown[]}};
    expect(dataset.data.fields.length).toBeGreaterThan(0);
    const processor =
      entry.extension === 'arrow'
        ? processors.processArrowTable
        : entry.extension === 'csv'
          ? processors.processCsvData
          : entry.extension === 'geojson'
            ? processors.processGeojson
            : processors.processRowObject;
    expect(processor).toHaveBeenCalledOnce();
  }
});

test('config-only remote descriptors retain native IDs, metadata and map config', async () => {
  const descriptor = {
    info: {id: 'remote-id', type: 'vector-tile'},
    data: {fields: [], rows: []},
    metadata: {
      type: 'remote',
      remoteTileFormat: 'pmtiles',
      tilesetDataUrl: 'https://example.com/tiles.pmtiles'
    }
  };
  const remote = {
    ...sample,
    dataUrl: undefined,
    remoteDatasetConfigUrl: 'https://example.com/remote.json'
  };
  const remoteConfig = {
    version: 'v1',
    config: {visState: {layers: [{id: 'tiles', type: 'vectorTile', config: {dataId: 'remote-id'}}]}}
  };
  const payload = await fetchSampleMap(
    remote,
    async input =>
      new Response(JSON.stringify(String(input) === sample.configUrl ? remoteConfig : descriptor))
  );
  expect(payload.datasets).toMatchObject(descriptor);
  expect(payload.datasets).toHaveProperty('metadata.lumaSampleMap.sample.id', sample.id);
  await expect(fetchSampleMap(remote, createFetch('{}', []))).rejects.toThrow(
    'unsupported remote dataset'
  );
  await expect(fetchSampleMap({...remote, remoteDatasetConfigUrl: undefined})).rejects.toThrow(
    'no remote dataset'
  );
});

test('remote sample descriptor binds to the single saved dataset ID without changing config', () => {
  for (const [type, layerType] of [
    ['vector-tile', 'vectorTile'],
    ['raster-tile', 'rasterTile'],
    ['wms-tile', 'wms']
  ]) {
    const descriptor = {
      info: {id: 'public-name', type},
      data: {fields: [], rows: []},
      metadata: {type: 'remote'}
    };
    const saved = {
      config: {
        visState: {
          layers: [{id: 'remote-layer', type: layerType, config: {dataId: 'generated-export-id'}}],
          interactionConfig: {tooltip: {fieldsToShow: {'generated-export-id': []}}}
        }
      }
    };
    const original = JSON.stringify(saved);
    expect(alignSampleRemoteDataset(descriptor, saved)).toEqual({
      ...descriptor,
      info: {...descriptor.info, id: 'generated-export-id'}
    });
    expect(descriptor.info.id).toBe('public-name');
    expect(JSON.stringify(saved)).toBe(original);
    saved.config.visState.layers.push({id: 'other', type: layerType, config: {dataId: 'other-id'}});
    expect(() => alignSampleRemoteDataset(descriptor, saved)).toThrow('exactly one dataset ID');
    saved.config.visState.layers[0]!.type = 'point';
    expect(() => alignSampleRemoteDataset(descriptor, saved)).toThrow('do not match');
  }
});

test('remote service preflight reports HTTP failures and cancels successful range bodies', async () => {
  const descriptor = {
    info: {type: 'vector-tile'},
    metadata: {tilesetDataUrl: 'https://example.com/data.pmtiles'}
  };
  const cancel = vi.fn();
  const fetchRemote = vi.fn(async () => new Response(new ReadableStream({cancel})));
  expect(await probeSampleRemoteService(descriptor, fetchRemote)).toEqual({
    url: descriptor.metadata.tilesetDataUrl,
    status: 200,
    available: true
  });
  expect(fetchRemote).toHaveBeenCalledWith(
    descriptor.metadata.tilesetDataUrl,
    expect.objectContaining({headers: {Range: 'bytes=0-0'}})
  );
  expect(cancel).toHaveBeenCalledOnce();
  const result = await probeSampleRemoteService(
    descriptor,
    async () => new Response(null, {status: 404})
  );
  expect(result.available).toBe(false);
  expect(new SampleRemoteServiceError(result)).toMatchObject({
    code: 'remote-service-unavailable',
    status: 404,
    message: expect.stringContaining('Remote service unavailable (HTTP 404)')
  });
  expect(
    getSampleRemoteServiceUrl({
      info: {type: 'wms-tile'},
      metadata: {tilesetDataUrl: 'https://example.com/wms?REQUEST=GetMap&token=public'}
    })
  ).toBe('https://example.com/wms?token=public&service=WMS&request=GetCapabilities');
  const remote = {
    ...sample,
    dataUrl: undefined,
    remoteDatasetConfigUrl: 'https://example.com/remote.json'
  };
  const fetchUnavailable: typeof fetch = async input => {
    if (String(input) === descriptor.metadata.tilesetDataUrl)
      return new Response(null, {status: 404});
    return new Response(
      JSON.stringify(
        String(input) === sample.configUrl
          ? config
          : {
              ...descriptor,
              info: {...descriptor.info, id: 'remote-id'},
              data: {fields: [], rows: []},
              metadata: {...descriptor.metadata, type: 'remote'}
            }
      )
    );
  };
  await expect(fetchSampleMap(remote, fetchUnavailable)).rejects.toMatchObject({
    code: 'remote-service-unavailable',
    status: 404
  });
  await expect(
    probeSampleRemoteService(descriptor, async () => {
      throw new Error('CORS or offline');
    })
  ).rejects.toThrow('CORS or offline');
});

test('HTTP, parsing and offline failures reject before any native transaction', async () => {
  await expect(
    fetchSampleMaps(async () => new Response('unavailable', {status: 503}))
  ).rejects.toThrow('503');
  await expect(
    fetchSampleMap(sample, async () => {
      throw new Error('offline');
    })
  ).rejects.toThrow('offline');
  await expect(fetchSampleMap(sample, async () => new Response('{}'))).rejects.toThrow('Kepler v1');
  await expect(fetchSampleMaps(async () => new Response('{}'))).rejects.toThrow('array');
});

test('malformed config aborts sibling downloads and cancels unused response bodies', async () => {
  const cancel = vi.fn();
  let dataSignal: AbortSignal | undefined;
  const fetchRemote: typeof fetch = async (input, options) => {
    if (String(input) === sample.configUrl) return new Response('{}');
    dataSignal = options?.signal ?? undefined;
    return new Response(new ReadableStream({cancel}));
  };
  await expect(fetchSampleMap(sample, fetchRemote)).rejects.toThrow('Kepler v1');
  expect(dataSignal?.aborted).toBe(true);
  expect(cancel).toHaveBeenCalledOnce();
});

test('external cancellation aborts catalog and config/data siblings, including late headers', async () => {
  const controller = new AbortController();
  const signals: AbortSignal[] = [];
  const cancelled = vi.fn();
  let resolveData: (response: Response) => void = () => {};
  const data = new Promise<Response>(resolve => {
    resolveData = resolve;
  });
  const fetchRemote: typeof fetch = async (input, options) => {
    const signal = options?.signal!;
    signals.push(signal);
    if (String(input) === sample.dataUrl) return data;
    return new Promise((_resolve, reject) =>
      signal.addEventListener('abort', () => reject(signal.reason), {once: true})
    );
  };
  const loading = fetchSampleMap(sample, fetchRemote, controller.signal);
  const catalog = fetchSampleMaps(fetchRemote, controller.signal);
  const expectations = Promise.all([
    expect(loading).rejects.toThrow('superseded'),
    expect(catalog).rejects.toThrow('superseded')
  ]);
  controller.abort(new Error('superseded'));
  await expectations;
  expect(signals.every(signal => signal.aborted)).toBe(true);
  resolveData(new Response(new ReadableStream({cancel: cancelled})));
  await vi.waitFor(() => expect(cancelled).toHaveBeenCalledOnce());
});

test('fetch and response-body failures name the failed sample URL and preserve the cause', async () => {
  const failure = new TypeError('Failed to fetch');
  const fetchRemote: typeof fetch = async input => {
    if (String(input) === sample.configUrl) return new Response(JSON.stringify(config));
    throw failure;
  };
  const headerError = await fetchSampleMap(sample, fetchRemote).catch(error => error);
  expect(headerError.message).toContain(sample.dataUrl);
  expect(headerError.message).toContain('network or response decoding failure');
  expect(headerError.cause).toBe(failure);
  const bodyError = await fetchSampleMap(sample, async input =>
    String(input) === sample.configUrl
      ? new Response(JSON.stringify(config))
      : new Response(
          new ReadableStream({
            start(controller) {
              controller.error(failure);
            }
          })
        )
  ).catch(error => error);
  expect(bodyError.message).toContain(sample.dataUrl);
  expect(bodyError.cause).toBe(failure);
});

test('completed sample bodies are consumed before the shared scope is closed', async () => {
  let signal: AbortSignal | undefined;
  let finish: () => void = () => {};
  const loading = fetchSampleMap(sample, async (input, options) => {
    if (String(input) === sample.configUrl) return new Response(JSON.stringify(config));
    signal = options?.signal ?? undefined;
    return new Response(
      new ReadableStream({
        start(controller) {
          finish = () => {
            controller.enqueue(new TextEncoder().encode('latitude,longitude\n1,2'));
            controller.close();
          };
        }
      })
    );
  });
  await vi.waitFor(() => expect(signal).toBeDefined());
  expect(signal?.aborted).toBe(false);
  finish();
  await loading;
  expect(signal?.aborted).toBe(true);
});

test('the unchanged download deadline reports its URL and aborts outstanding sibling I/O', async () => {
  const deadline = new AbortController();
  const timeout = vi.spyOn(AbortSignal, 'timeout').mockReturnValue(deadline.signal);
  const signals: AbortSignal[] = [];
  try {
    const loading = fetchSampleMap(sample, async (input, options) => {
      const signal = options?.signal!;
      signals.push(signal);
      if (String(input) === sample.configUrl) return new Response(JSON.stringify(config));
      return new Promise((_resolve, reject) =>
        signal.addEventListener('abort', () => reject(signal.reason), {once: true})
      );
    });
    const assertion = expect(loading).rejects.toThrow(
      `Sample request ${sample.dataUrl} failed (60,000 ms download deadline exceeded)`
    );
    await vi.waitFor(() => expect(signals).toHaveLength(2));
    // Drain successful config parsing before timing out the still-pending data headers.
    await new Promise(resolve => setTimeout(resolve, 0));
    deadline.abort(new DOMException('download timed out', 'TimeoutError'));
    await assertion;
    expect(timeout).toHaveBeenCalledWith(60000);
    expect(signals.every(signal => signal.aborted)).toBe(true);
  } finally {
    timeout.mockRestore();
  }
});
