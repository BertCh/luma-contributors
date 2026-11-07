// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {describe, expect, test} from 'vitest';
import {
  LOCAL_CLOUD_FILE_CONFLICT,
  LocalCloudService,
  createMemoryCloudStore,
  type LocalCloudHost
} from '../../../examples/deck/kepler-compute/src/local-cloud-service';

function createHost(bundle = '{"bundle":1}') {
  const restored: string[] = [];
  const host: LocalCloudHost = {
    captureBundle: async () => bundle,
    restoreBundle: async serialized => {
      restored.push(serialized);
      return {nativeConfig: {version: 'v1'}};
    }
  };
  return {host, restored};
}

const LIMITS = {maximumMaps: 2, maximumBundleBytes: 64, maximumTotalBytes: 100};

describe('LocalCloudService', () => {
  test('saves, lists and replays through the host bundle path', async () => {
    const {host, restored} = createHost();
    const service = new LocalCloudService(createMemoryCloudStore(), () => host, LIMITS);
    const record = await service.saveMap({title: 'a'});
    expect((await service.listMaps()).map(item => item.id)).toEqual([record.id]);
    const loaded = await service.loadMap(record.id);
    expect(restored).toEqual(['{"bundle":1}']);
    expect(loaded.nativeConfig).toEqual({version: 'v1'});
  });

  test('requires overwrite for an existing title and keeps its id', async () => {
    const service = new LocalCloudService(
      createMemoryCloudStore(),
      () => createHost().host,
      LIMITS
    );
    const first = await service.saveMap({title: 'a'});
    await expect(service.saveMap({title: 'a'})).rejects.toThrow(LOCAL_CLOUD_FILE_CONFLICT);
    expect((await service.saveMap({title: 'a', overwrite: true})).id).toBe(first.id);
  });

  test('overwrites the saved map id after a rename and rejects colliding renames', async () => {
    const service = new LocalCloudService(
      createMemoryCloudStore(),
      () => createHost().host,
      LIMITS
    );
    const first = await service.saveMap({title: 'a'});
    const second = await service.saveMap({title: 'b'});
    const renamed = await service.saveMap({
      title: 'a2',
      overwrite: true,
      mapIdToOverwrite: first.id
    });
    expect(renamed.id).toBe(first.id);
    expect((await service.listMaps()).map(item => item.title).sort()).toEqual(['a2', 'b']);
    await expect(
      service.saveMap({title: 'b', overwrite: true, mapIdToOverwrite: first.id})
    ).rejects.toThrow(LOCAL_CLOUD_FILE_CONFLICT);
    expect(second.id).not.toBe(first.id);
  });

  test('rejects public publishing, missing hosts and budget overruns', async () => {
    const store = createMemoryCloudStore();
    await expect(
      new LocalCloudService(store, () => null, LIMITS).saveMap({title: 'x'})
    ).rejects.toThrow('No workspace');
    const service = new LocalCloudService(store, () => createHost().host, LIMITS);
    await expect(service.saveMap({title: 'x', isPublic: true})).rejects.toThrow(
      'Public publishing'
    );
    const large = new LocalCloudService(store, () => createHost('x'.repeat(65)).host, LIMITS);
    await expect(large.saveMap({title: 'big'})).rejects.toThrow('size budget');
    await service.saveMap({title: 'one'});
    await service.saveMap({title: 'two'});
    await expect(service.saveMap({title: 'three'})).rejects.toThrow('limit reached');
  });

  test('propagates host capture failures without storing', async () => {
    const store = createMemoryCloudStore();
    const host: LocalCloudHost = {
      captureBundle: async () => {
        throw new Error('Unfinished streams cannot be saved');
      },
      restoreBundle: async () => ({nativeConfig: {}})
    };
    const service = new LocalCloudService(store, () => host, LIMITS);
    await expect(service.saveMap({title: 'p'})).rejects.toThrow('Unfinished streams');
    expect(await service.listMaps()).toEqual([]);
  });

  test('reports store failures cleanly', async () => {
    const store = createMemoryCloudStore();
    store.putMap = async () => {
      throw new Error('QuotaExceededError');
    };
    const service = new LocalCloudService(store, () => createHost().host, LIMITS);
    await expect(service.saveMap({title: 'q'})).rejects.toThrow(
      'Local map storage is unavailable: QuotaExceededError'
    );
  });

  test('deleted maps cannot be loaded', async () => {
    const service = new LocalCloudService(
      createMemoryCloudStore(),
      () => createHost().host,
      LIMITS
    );
    const record = await service.saveMap({title: 'd'});
    await service.deleteMap(record.id);
    await expect(service.loadMap(record.id)).rejects.toThrow('no longer available');
  });
});
