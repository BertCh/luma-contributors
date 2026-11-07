// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Provider, KEPLER_FORMAT, type MapListItem} from '@kepler.gl/cloud-providers';
import {
  LocalCloudService,
  createLocalCloudStore,
  formatLocalCloudUrl,
  parseLocalCloudUrl,
  type LocalCloudHost,
  type LocalCloudMapRecord
} from './local-cloud-service';

import {blobToDataUrl} from './local-cloud-thumbnail';

export type LocalCloudUser = {name: string; email: string};

/** Mock account: there are no credentials, tokens or network calls. */
export const LOCAL_CLOUD_USER: LocalCloudUser = {name: 'Local user', email: 'local@localhost'};
export const LOCAL_CLOUD_ACCESS_TOKEN = 'local-token';

let activeHost: LocalCloudHost | null = null;
const storePromise = createLocalCloudStore();
const defaultService = new LocalCloudService(
  {
    persistent: false,
    listRecords: () => storePromise.then(store => store.listRecords()),
    getBundle: id => storePromise.then(store => store.getBundle(id)),
    putMap: (record, bundle) => storePromise.then(store => store.putMap(record, bundle)),
    deleteMap: id => storePromise.then(store => store.deleteMap(id))
  },
  () => activeHost
);

/** Install (or clear with null) the workspace the provider saves from and loads into. */
export function bindLocalCloudHost(host: LocalCloudHost | null): void {
  activeHost = host;
}

function toListItem(record: LocalCloudMapRecord): MapListItem {
  return {
    id: record.id,
    title: record.title,
    description: record.description,
    updatedAt: record.updatedAt,
    loadParams: {id: record.id},
    privateMap: true,
    ...(record.thumbnail ? {thumbnail: record.thumbnail, imageUrl: record.thumbnail} : {})
  };
}

export type LocalCloudProviderOptions = {
  /** Injection point for tests: a service over a memory store and a stub host. */
  service?: LocalCloudService;
  user?: LocalCloudUser;
};

/** Kepler cloud provider backed by browser-local storage of portable sandbox bundles. */
export class LocalCloudProvider extends Provider {
  private signedIn = true;
  private lastMapId: string | null = null;
  private readonly changeListeners = new Set<() => void>();
  constructor(
    private readonly service: LocalCloudService = defaultService,
    private readonly user: LocalCloudUser = LOCAL_CLOUD_USER
  ) {
    super({
      name: 'luma-local',
      displayName: 'This browser',
      storageMessage:
        'Saved in this browser as a portable bundle with its Arrow data. Not shared or public.'
    });
  }
  override hasPrivateStorage(): boolean {
    return true;
  }
  /** Public links cannot exist for browser-local storage; Kepler hides its share-by-URL entry. */
  override hasSharingUrl(): boolean {
    return false;
  }
  override hasLazyThumbnails(): boolean {
    return false;
  }
  override async getAccessToken(): Promise<string | null> {
    return this.signedIn ? LOCAL_CLOUD_ACCESS_TOKEN : null;
  }
  override getUserName(): string {
    return this.signedIn ? this.user.name : '';
  }
  override async getUser() {
    return this.signedIn ? {...this.user} : null;
  }
  override async login(): Promise<never> {
    // No credentials exist; login only restores the mock local session.
    this.signedIn = true;
    return undefined as never;
  }
  override async logout(): Promise<void> {
    // Saved maps stay in browser storage; logout only ends the mock session.
    this.signedIn = false;
  }
  /** Local-only URL (local://map/<id>); nothing outside this browser can resolve it. */
  override getMapUrl(loadParams: {id: string}): string {
    return formatLocalCloudUrl(loadParams.id);
  }
  override getShareUrl(): string {
    return this.lastMapId ? formatLocalCloudUrl(this.lastMapId) : '';
  }
  /** Resolves a local:// URL to Kepler load parameters, or null for any other URL. */
  resolveMapUrl(url: string): {id: string} | null {
    const id = parseLocalCloudUrl(url);
    return id ? {id} : null;
  }
  override async uploadMap({mapData, options}: Parameters<Provider['uploadMap']>[0]) {
    // Public publishing is refused in the service; keep the refusal explicit at this boundary too.
    if (options?.isPublic)
      throw new Error('Public publishing is unavailable; local maps are private to this browser');
    const info = (mapData.map as {info?: {title?: string; description?: string}} | undefined)?.info;
    const thumbnail = mapData.thumbnail ? await blobToDataUrl(mapData.thumbnail) : undefined;
    // The Kepler dataset payload is ignored on purpose: resident rows live in Arrow, so the host bundle is authoritative.
    const record = await this.service.saveMap({
      title: info?.title || (mapData as {title?: string}).title || 'Untitled map',
      description: info?.description,
      overwrite: options?.overwrite,
      mapIdToOverwrite: (options as {mapIdToOverwrite?: string | null} | undefined)
        ?.mapIdToOverwrite,
      isPublic: options?.isPublic,
      thumbnail
    });
    this.lastMapId = record.id;
    this.notifyChanged();
    return {...toListItem(record), info: {id: record.id, title: record.title}};
  }
  override async listMaps(): Promise<MapListItem[]> {
    return (await this.service.listMaps()).map(toListItem);
  }
  override async getMapThumbnail(map: MapListItem): Promise<string | undefined> {
    return map.thumbnail;
  }
  override async downloadMap(loadParams: {id: string}) {
    const {record, nativeConfig, datasets} = await this.service.loadMap(loadParams.id);
    this.lastMapId = record.id;
    // Resident data was restored by the bundle replay; only datasets the bundle explicitly supplies
    // are returned, so Kepler never adds rows outside the host guards.
    return {
      map: {
        datasets,
        config: (nativeConfig ?? {}) as never,
        info: {
          app: 'kepler.gl',
          title: record.title,
          description: record.description,
          created_at: new Date(record.updatedAt).toISOString()
        }
      } as never,
      format: KEPLER_FORMAT
    };
  }
  /** Kepler's provider contract has no delete; the host UI calls this directly. */
  async deleteMap(id: string): Promise<void> {
    await this.service.deleteMap(id);
    if (this.lastMapId === id) this.lastMapId = null;
    this.notifyChanged();
  }
  /** Lets host UI refresh a map list after a save or delete. Returns an unsubscribe function. */
  subscribeToChanges(listener: () => void): () => void {
    this.changeListeners.add(listener);
    return () => this.changeListeners.delete(listener);
  }
  private notifyChanged(): void {
    for (const listener of this.changeListeners) listener();
  }
}

export function createLocalCloudProvider(
  options: LocalCloudProviderOptions = {}
): LocalCloudProvider {
  const provider = new LocalCloudProvider(options.service, options.user);
  if (options.service) return provider;
  // Test hook for browser scripts; carries no credentials.
  (globalThis as {__lumaLocalCloudProvider?: LocalCloudProvider}).__lumaLocalCloudProvider =
    provider;
  // Test hook: raw stored bundle JSON, so scripts can assert what a save embedded.
  (
    globalThis as {__lumaLocalCloudStore?: {getBundle(id: string): Promise<string | undefined>}}
  ).__lumaLocalCloudStore = {getBundle: id => storePromise.then(store => store.getBundle(id))};
  return provider;
}
