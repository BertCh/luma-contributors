// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/**
 * Source-reference share links. Unlike `#sandbox=` links (state plus the built-in fixture), a
 * `#sources=` link names remote data: https URLs with a format, optional SHA-256 and column roles,
 * and an optional saved Kepler map JSON URL. Links never contain rows or credentials.
 */

const LINK_PREFIX = '#sources=v1.';
const MAXIMUM_SOURCES = 8;
const MAXIMUM_URL_LENGTH = 2048;
const MAXIMUM_LINK_LENGTH = 16 * 1024;
const MAXIMUM_DOWNLOAD_BYTES = 256 * 1024 * 1024;
const ROLE_NAME = /^[A-Za-z][A-Za-z0-9]{0,31}$/;

export type LinkedSourceFormat = 'csv' | 'arrow' | 'geojson' | 'json';

export type LinkedSource = {
  /** Display name; also the downloaded file name stem. */
  label: string;
  url: string;
  format: LinkedSourceFormat;
  /** Lowercase hex SHA-256 of the exact bytes; a mismatch asks for reattachment. */
  sha256?: string;
  /** Column roles such as longitude, latitude, geometry, time. Applied after loading. */
  roles?: Record<string, string>;
};

export type SourceLinkDescriptor = {
  sources: LinkedSource[];
  /** Saved Kepler map JSON (layers, filters) applied together with the sources. */
  savedMapUrl?: string;
};

export type LinkedSourceOutcome =
  | {status: 'loaded'; source: LinkedSource; file: File}
  | {status: 'needs-reattachment'; source: LinkedSource; message: string};

export type SourceLinkLoadResult = {
  outcomes: LinkedSourceOutcome[];
  savedMap?:
    | {status: 'loaded'; file: File}
    | {status: 'needs-reattachment'; url: string; message: string};
};

/** Accepts only credential-free HTTPS addresses (HTTP only for loopback development). */
export function assertLinkedUrl(value: string): URL {
  if (value.length > MAXIMUM_URL_LENGTH) throw new Error('Linked source address is too long');
  const url = new URL(value);
  const isLoopback = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
  if (
    !(url.protocol === 'https:' || (url.protocol === 'http:' && isLoopback)) ||
    url.username ||
    url.password
  )
    throw new Error('Linked sources require credential-free HTTPS addresses');
  return url;
}

export function parseSourceLinkDescriptor(value: unknown): SourceLinkDescriptor {
  if (!value || typeof value !== 'object') throw new Error('Malformed source link');
  const record = value as Record<string, unknown>;
  const rawSources = record['sources'];
  if (!Array.isArray(rawSources) || rawSources.length > MAXIMUM_SOURCES)
    throw new Error(`Source links reference at most ${MAXIMUM_SOURCES} sources`);
  const sources = rawSources.map((entry): LinkedSource => {
    const item = entry as Record<string, unknown>;
    if (
      !item ||
      typeof item['label'] !== 'string' ||
      !item['label'] ||
      item['label'].length > 80 ||
      typeof item['url'] !== 'string' ||
      !['csv', 'arrow', 'geojson', 'json'].includes(item['format'] as string)
    )
      throw new Error('Malformed linked source');
    const sha256 = item['sha256'];
    if (sha256 !== undefined && (typeof sha256 !== 'string' || !/^[a-f0-9]{64}$/.test(sha256)))
      throw new Error('Linked source checksum must be a lowercase SHA-256');
    const roles = item['roles'];
    if (roles !== undefined) {
      if (!roles || typeof roles !== 'object' || Array.isArray(roles))
        throw new Error('Malformed linked source roles');
      for (const [role, column] of Object.entries(roles))
        if (!ROLE_NAME.test(role) || typeof column !== 'string' || column.length > 128)
          throw new Error('Malformed linked source roles');
    }
    assertLinkedUrl(item['url']);
    return {
      label: item['label'],
      url: item['url'],
      format: item['format'] as LinkedSourceFormat,
      ...(sha256 ? {sha256} : {}),
      ...(roles ? {roles: roles as Record<string, string>} : {})
    };
  });
  const savedMapUrl = record['savedMapUrl'];
  if (savedMapUrl !== undefined) {
    if (typeof savedMapUrl !== 'string') throw new Error('Malformed saved map address');
    assertLinkedUrl(savedMapUrl);
  }
  if (!sources.length && !savedMapUrl)
    throw new Error('A source link needs a source or a saved map');
  return {sources, ...(savedMapUrl ? {savedMapUrl} : {})};
}

export function createSourceShareLink(descriptor: SourceLinkDescriptor, location: string): string {
  const parsed = parseSourceLinkDescriptor(descriptor);
  const address = new URL(location);
  if (!['http:', 'https:'].includes(address.protocol) || address.username || address.password)
    throw new Error('Share links require an HTTP or HTTPS sandbox address');
  address.search = '';
  const bytes = new TextEncoder().encode(JSON.stringify(parsed));
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  address.hash = `${LINK_PREFIX}${btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')}`;
  if (address.href.length > MAXIMUM_LINK_LENGTH) throw new Error('Source link is too long');
  return address.href;
}

/** Returns null for unrelated fragments, including `#sandbox=` state links. */
export function readSourceShareLink(location: string): SourceLinkDescriptor | null {
  const address = new URL(location);
  if (!address.hash.startsWith('#sources=')) return null;
  if (address.href.length > MAXIMUM_LINK_LENGTH) throw new Error('Source link is too long');
  if (!address.hash.startsWith(LINK_PREFIX)) throw new Error('Unsupported source link version');
  const encoded = address.hash.slice(LINK_PREFIX.length);
  if (!encoded || !/^[A-Za-z0-9_-]+$/.test(encoded) || encoded.length % 4 === 1)
    throw new Error('Malformed source link');
  const binary = atob(encoded.replace(/-/g, '+').replace(/_/g, '/'));
  const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
  return parseSourceLinkDescriptor(
    JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(bytes))
  );
}

export async function computeSha256Hex(bytes: Uint8Array): Promise<string> {
  const hash = await crypto.subtle.digest('SHA-256', Uint8Array.from(bytes));
  return Array.from(new Uint8Array(hash), value => value.toString(16).padStart(2, '0')).join('');
}

/**
 * Fetches every referenced file. Private, missing, oversized, blocked or tampered data does not
 * abort the link: it yields `needs-reattachment` with a message the user can read, so they can
 * attach their own copy (checked against the same checksum).
 */
export async function loadSourceLink(
  descriptor: SourceLinkDescriptor,
  options: {fetch?: typeof fetch; signal?: AbortSignal; maximumBytes?: number} = {}
): Promise<SourceLinkLoadResult> {
  const fetchImplementation = options.fetch ?? globalThis.fetch.bind(globalThis);
  const maximumBytes = options.maximumBytes ?? MAXIMUM_DOWNLOAD_BYTES;
  const download = async (
    url: string,
    name: string,
    mimeType: string,
    sha256?: string
  ): Promise<{file: File} | {message: string}> => {
    try {
      assertLinkedUrl(url);
      const response = await fetchImplementation(url, {
        signal: options.signal,
        credentials: 'omit',
        redirect: 'follow'
      });
      if (!response.ok)
        return {
          message: `${name} could not be loaded (HTTP ${response.status}${response.status === 401 || response.status === 403 ? ', private data' : ''}). Attach your own copy to continue.`
        };
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (bytes.byteLength > maximumBytes)
        return {
          message: `${name} is larger than the ${Math.round(maximumBytes / (1024 * 1024))} MiB link limit. Attach the file instead.`
        };
      if (sha256 && (await computeSha256Hex(bytes)) !== sha256)
        return {
          message: `${name} does not match the checksum recorded in the link (the data changed or was replaced). Attach the original file to continue.`
        };
      return {file: new File([bytes], name, {type: mimeType})};
    } catch (error) {
      if (options.signal?.aborted) throw error;
      return {
        message: `${name} could not be loaded (${error instanceof Error ? error.message : String(error)}); it may be missing, private or blocked by CORS. Attach your own copy to continue.`
      };
    }
  };
  const outcomes: LinkedSourceOutcome[] = [];
  for (const source of descriptor.sources) {
    const result = await download(
      source.url,
      `${source.label}.${source.format === 'arrow' ? 'arrow' : source.format}`,
      FORMAT_MIME_TYPES[source.format],
      source.sha256
    );
    outcomes.push(
      'file' in result
        ? {status: 'loaded', source, file: result.file}
        : {status: 'needs-reattachment', source, message: result.message}
    );
  }
  let savedMap: SourceLinkLoadResult['savedMap'];
  if (descriptor.savedMapUrl) {
    const result = await download(descriptor.savedMapUrl, 'linked-map.json', 'application/json');
    savedMap =
      'file' in result
        ? {status: 'loaded', file: result.file}
        : {status: 'needs-reattachment', url: descriptor.savedMapUrl, message: result.message};
  }
  return {outcomes, ...(savedMap ? {savedMap} : {})};
}

/** Checks a user-attached replacement against the checksum recorded in the link. */
export async function verifyReattachedFile(source: LinkedSource, file: File): Promise<void> {
  if (
    source.sha256 &&
    (await computeSha256Hex(new Uint8Array(await file.arrayBuffer()))) !== source.sha256
  )
    throw new Error(`${file.name} does not match the checksum recorded for ${source.label}`);
}

const FORMAT_MIME_TYPES: Record<LinkedSourceFormat, string> = {
  csv: 'text/csv',
  arrow: 'application/vnd.apache.arrow.stream',
  geojson: 'application/geo+json',
  json: 'application/json'
};

/** Host seam: main installs how files and maps enter native Kepler. */
export type SourceLinkHost = {loadFiles(files: File[]): void};
let host: SourceLinkHost | undefined;
export function registerSourceLinkHost(value: SourceLinkHost): void {
  host = value;
}
export function getSourceLinkHost(): SourceLinkHost | undefined {
  return host;
}
