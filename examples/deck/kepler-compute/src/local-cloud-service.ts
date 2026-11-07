// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Browser-local map storage behind Kepler's cloud-provider contract. It stores the host's portable
 * sandbox bundle (retained Arrow sources plus native configuration), never Kepler's CPU datasets,
 * so saved maps keep the same precision, memory, population and export contracts as manual export.
 * It is private to this browser profile and is not public hosting.
 */

export const LOCAL_CLOUD_FILE_CONFLICT = 'file_conflict';

export type LocalCloudMapRecord = {
  id: string;
  title: string;
  description: string;
  updatedAt: number;
  byteLength: number;
  /** Data URL preview stored from the Kepler save payload; absent when none was supplied. */
  thumbnail?: string;
};

/** Thumbnails above this many characters are dropped rather than bloating the metadata store. */
export const LOCAL_CLOUD_MAXIMUM_THUMBNAIL_LENGTH = 512 * 1024;

/** Local-only URL scheme; these links open nothing outside this browser profile. */
export const LOCAL_CLOUD_URL_PREFIX = 'local://map/';

export function formatLocalCloudUrl(id: string): string {
  return `${LOCAL_CLOUD_URL_PREFIX}${id}`;
}

export function parseLocalCloudUrl(url: string): string | null {
  return url.startsWith(LOCAL_CLOUD_URL_PREFIX) && url.length > LOCAL_CLOUD_URL_PREFIX.length
    ? url.slice(LOCAL_CLOUD_URL_PREFIX.length)
    : null;
}

export type LocalCloudStore = {
  /** False for the in-memory fallback, whose maps vanish with the page. */
  readonly persistent: boolean;
  listRecords(): Promise<LocalCloudMapRecord[]>;
  getBundle(id: string): Promise<string | undefined>;
  putMap(record: LocalCloudMapRecord, bundle: string): Promise<void>;
  deleteMap(id: string): Promise<void>;
};

/** Host adapter: the same bundle export/replay entry points the Portable workspace panel uses. */
export type LocalCloudHost = {
  /** Must apply the population, precision, row and byte limits of the manual bundle export. */
  captureBundle(): Promise<string>;
  /** Validates then replays a bundle; returns the restored native Kepler configuration. */
  restoreBundle(serialized: string): Promise<{
    nativeConfig: unknown /** Kepler dataset payloads the bundle supplies, if any. */;
    datasets?: unknown[];
  }>;
};

export type LocalCloudLimits = {
  maximumMaps: number;
  maximumBundleBytes: number;
  maximumTotalBytes: number;
};

export const LOCAL_CLOUD_LIMITS: LocalCloudLimits = {
  maximumMaps: 32,
  maximumBundleBytes: 96 * 1024 * 1024,
  maximumTotalBytes: 256 * 1024 * 1024
};

export class LocalCloudService {
  constructor(
    private readonly store: LocalCloudStore,
    private readonly getHost: () => LocalCloudHost | null,
    private readonly limits: LocalCloudLimits = LOCAL_CLOUD_LIMITS
  ) {}

  get isPersistent(): boolean {
    return this.store.persistent;
  }

  async listMaps(): Promise<LocalCloudMapRecord[]> {
    const records = await this.guardStore(() => this.store.listRecords());
    return records.sort((left, right) => right.updatedAt - left.updatedAt);
  }

  async saveMap(request: {
    title: string;
    description?: string;
    overwrite?: boolean;
    /** Kepler's savedMapId: an explicit overwrite targets this map even if its title changed. */
    mapIdToOverwrite?: string | null;
    isPublic?: boolean;
    thumbnail?: string;
  }): Promise<LocalCloudMapRecord> {
    // Local storage is not public hosting; refuse rather than pretend a link is shareable.
    if (request.isPublic)
      throw new Error('Public publishing is unavailable; local maps are private to this browser');
    const host = this.getHost();
    if (!host) throw new Error('No workspace is ready to save');
    const records = await this.listMaps();
    const sameTitle = records.find(record => record.title === request.title);
    const target = request.overwrite
      ? records.find(record => record.id === request.mapIdToOverwrite)
      : undefined;
    // Renaming onto another saved map's title is still a conflict, never a silent replacement.
    if (target && sameTitle && sameTitle.id !== target.id)
      throw new Error(LOCAL_CLOUD_FILE_CONFLICT);
    const existing = target ?? sameTitle;
    if (existing && !request.overwrite) throw new Error(LOCAL_CLOUD_FILE_CONFLICT);
    if (!existing && records.length >= this.limits.maximumMaps)
      throw new Error('Local map limit reached; delete a saved map first');
    const bundle = await host.captureBundle();
    if (bundle.length > this.limits.maximumBundleBytes)
      throw new Error('Workspace exceeds the local map size budget');
    const otherBytes = records.reduce(
      (total, record) => total + (record.id === existing?.id ? 0 : record.byteLength),
      0
    );
    if (otherBytes + bundle.length > this.limits.maximumTotalBytes)
      throw new Error('Local map storage budget exceeded; delete a saved map first');
    const record: LocalCloudMapRecord = {
      id: existing?.id ?? createMapIdentifier(),
      title: request.title,
      description: request.description ?? '',
      updatedAt: Date.now(),
      byteLength: bundle.length,
      ...(request.thumbnail && request.thumbnail.length <= LOCAL_CLOUD_MAXIMUM_THUMBNAIL_LENGTH
        ? {thumbnail: request.thumbnail}
        : {})
    };
    await this.guardStore(() => this.store.putMap(record, bundle));
    return record;
  }

  async loadMap(
    id: string
  ): Promise<{record: LocalCloudMapRecord; nativeConfig: unknown; datasets: unknown[]}> {
    const host = this.getHost();
    if (!host) throw new Error('No workspace is ready to load into');
    const records = await this.listMaps();
    const record = records.find(item => item.id === id);
    const bundle = record ? await this.guardStore(() => this.store.getBundle(id)) : undefined;
    if (!record || bundle === undefined) throw new Error('Saved map is no longer available');
    // Replay validates every source before replacing resident data (same path as bundle import).
    const restored = await host.restoreBundle(bundle);
    return {record, nativeConfig: restored.nativeConfig, datasets: restored.datasets ?? []};
  }

  async deleteMap(id: string): Promise<void> {
    await this.guardStore(() => this.store.deleteMap(id));
  }

  private async guardStore<T>(action: () => Promise<T>): Promise<T> {
    try {
      return await action();
    } catch (error) {
      throw new Error(
        `Local map storage is unavailable: ${error instanceof Error ? error.message : String(error)}`
      );
    }
  }
}

function createMapIdentifier(): string {
  return (
    globalThis.crypto?.randomUUID?.() ??
    `map-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
  );
}

export function createMemoryCloudStore(): LocalCloudStore {
  const records = new Map<string, {record: LocalCloudMapRecord; bundle: string}>();
  return {
    persistent: false,
    async listRecords() {
      return Array.from(records.values(), entry => ({...entry.record}));
    },
    async getBundle(id) {
      return records.get(id)?.bundle;
    },
    async putMap(record, bundle) {
      records.set(record.id, {record: {...record}, bundle});
    },
    async deleteMap(id) {
      records.delete(id);
    }
  };
}

const DATABASE_NAME = 'luma-kepler-local-cloud';

/** IndexedDB store with separate metadata and bundle stores so listing never reads bundles. */
export async function createIndexedDbCloudStore(
  factory: IDBFactory = globalThis.indexedDB
): Promise<LocalCloudStore> {
  const database = await new Promise<IDBDatabase>((resolve, reject) => {
    const request = factory.open(DATABASE_NAME, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore('maps', {keyPath: 'id'});
      request.result.createObjectStore('bundles');
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB open failed'));
    request.onblocked = () => reject(new Error('IndexedDB open blocked'));
  });
  const run = <T>(
    names: string[],
    mode: IDBTransactionMode,
    action: (transaction: IDBTransaction) => IDBRequest<T> | void
  ): Promise<T | undefined> =>
    new Promise((resolve, reject) => {
      const transaction = database.transaction(names, mode);
      const request = action(transaction);
      transaction.oncomplete = () => resolve(request ? request.result : undefined);
      transaction.onerror = () =>
        reject(transaction.error ?? new Error('IndexedDB transaction failed'));
      transaction.onabort = () =>
        reject(transaction.error ?? new Error('IndexedDB transaction aborted'));
    });
  return {
    persistent: true,
    async listRecords() {
      return ((await run<LocalCloudMapRecord[]>(['maps'], 'readonly', tx =>
        tx.objectStore('maps').getAll()
      )) ?? []) as LocalCloudMapRecord[];
    },
    async getBundle(id) {
      return (
        (await run<string>(['bundles'], 'readonly', tx => tx.objectStore('bundles').get(id))) ??
        undefined
      );
    },
    async putMap(record, bundle) {
      // One transaction: a quota failure aborts both writes, so metadata never outlives its bundle.
      await run(['maps', 'bundles'], 'readwrite', tx => {
        tx.objectStore('maps').put(record);
        tx.objectStore('bundles').put(bundle, record.id);
      });
    },
    async deleteMap(id) {
      await run(['maps', 'bundles'], 'readwrite', tx => {
        tx.objectStore('maps').delete(id);
        tx.objectStore('bundles').delete(id);
      });
    }
  };
}

/** Prefer IndexedDB; fall back to memory (reported through `persistent`) when it cannot open. */
export async function createLocalCloudStore(): Promise<LocalCloudStore> {
  try {
    if (!globalThis.indexedDB) throw new Error('IndexedDB is unavailable');
    return await createIndexedDbCloudStore();
  } catch {
    return createMemoryCloudStore();
  }
}
