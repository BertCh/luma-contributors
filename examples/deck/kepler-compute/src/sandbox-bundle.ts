// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import {
  assertKeplerSandboxSourceMatches,
  parseKeplerSandboxState,
  type KeplerSandboxState,
  type KeplerSandboxSource
} from '@luma-contributors/kepler';
import {assertSandboxSourcesCovered} from './sandbox-source-validation';
import {parseIconCatalogDocument, type IconCatalogDocument} from './icon-catalog-assets';
import type {NativeIconDefinition} from './icon-source';
import {
  readEmbeddedNativeSources,
  type EmbeddedNativeSource,
  type RestoredNativeSource
} from './bundle-native-sources';

export type SandboxBundleLimits = {
  maximumDocumentBytes: number;
  maximumSourceBytes: number;
  maximumRows: number;
  maximumSources: number;
};

export type SandboxBundleSource = {
  datasetId: string;
  revision: number;
  /** Retained CPU source Arrow, before GPU normalization and selection. */
  table: arrow.Table;
  isCurrent(): boolean;
};

export type RestoredSandboxBundle = {
  state: KeplerSandboxState;
  sources: {datasetId: string; revision: number; table: arrow.Table}[];
  /** Embedded custom icon meshes; absent when the workspace used only the default catalog. */
  iconCatalog?: NativeIconDefinition[];
  /** Original values of datasets that exist only as native Kepler tables; restored as native datasets. */
  nativeOnlySources?: RestoredNativeSource[];
};

type EncodedSource = {
  datasetId: string;
  revision: number;
  encoding: 'arrow-ipc-stream-base64';
  byteLength: number;
  sha256: string;
  data: string;
};

/** Explicit source export. Limits bound payload sizes, not peak JSON/Arrow parser memory. */
export async function serializeSandboxBundle(
  serializedState: string,
  sources: readonly SandboxBundleSource[],
  limits: SandboxBundleLimits,
  iconCatalog?: IconCatalogDocument,
  nativeOnlySources: readonly EmbeddedNativeSource[] = []
): Promise<string> {
  assertLimits(limits);
  assertDocumentSize(serializedState, limits);
  const state = parseKeplerSandboxState(JSON.parse(serializedState));
  assertSources(
    state,
    sources,
    limits,
    nativeOnlySources.map(source => source.datasetId)
  );
  const encoded: EncodedSource[] = [];
  let sourceBytes = 0;
  for (const descriptor of state.sources) {
    const source = sources.find(item => item.datasetId === descriptor.dataset.datasetId)!;
    if (!source.isCurrent()) throw new Error('Source changed before bundle export');
    assertTable(descriptor, source.table, limits);
    const bytes = arrow.tableToIPC(source.table, 'stream');
    sourceBytes += bytes.byteLength;
    if (sourceBytes > limits.maximumSourceBytes)
      throw new Error('Bundle sources exceed the Arrow payload budget');
    encoded.push({
      datasetId: source.datasetId,
      revision: source.revision,
      encoding: 'arrow-ipc-stream-base64',
      byteLength: bytes.byteLength,
      sha256: await computeDigest(bytes),
      data: encodeBytes(bytes)
    });
  }
  if (sources.some(source => !source.isCurrent()))
    throw new Error('Source changed during bundle export');
  const serialized = JSON.stringify({
    format: 'luma-kepler-sandbox',
    version: 1,
    state,
    sources: encoded,
    ...(iconCatalog ? {iconCatalog} : {}),
    ...(nativeOnlySources.length ? {nativeOnlySources} : {})
  });
  assertDocumentSize(serialized, limits);
  return serialized;
}

/** Validates all sources before returning a document; it performs no GPU uploads or native edits. */
export async function parseSandboxBundle(
  serialized: string,
  limits: SandboxBundleLimits
): Promise<RestoredSandboxBundle> {
  assertLimits(limits);
  assertDocumentSize(serialized, limits);
  const document = JSON.parse(serialized);
  if (
    !document ||
    document.format !== 'luma-kepler-sandbox' ||
    document.version !== 1 ||
    !Array.isArray(document.sources)
  )
    throw new Error('Unsupported sandbox bundle');
  const state = parseKeplerSandboxState(document.state);
  const encoded: EncodedSource[] = document.sources.map((value: unknown) =>
    readEncodedSource(value)
  );
  // Verified before any restore step so a corrupt native-only member rejects the whole bundle.
  const nativeOnlySources = await readEmbeddedNativeSources(document.nativeOnlySources);
  assertSources(
    state,
    encoded,
    limits,
    nativeOnlySources.map(source => source.datasetId)
  );
  if (encoded.reduce((total, source) => total + source.byteLength, 0) > limits.maximumSourceBytes)
    throw new Error('Bundle sources exceed the Arrow payload budget');
  const sources: RestoredSandboxBundle['sources'] = [];
  for (const descriptor of state.sources) {
    const source = encoded.find(item => item.datasetId === descriptor.dataset.datasetId)!;
    const bytes = decodeBytes(source.data);
    if (bytes.byteLength !== source.byteLength || (await computeDigest(bytes)) !== source.sha256)
      throw new Error('Bundle source checksum or byte length does not match');
    const table = arrow.tableFromIPC(bytes);
    assertTable(descriptor, table, limits);
    sources.push({datasetId: source.datasetId, revision: source.revision, table});
  }
  // A present but malformed catalog rejects the whole bundle so failures stay visible.
  const iconCatalog =
    document.iconCatalog === undefined ? undefined : parseIconCatalogDocument(document.iconCatalog);
  return {
    state,
    sources,
    ...(iconCatalog ? {iconCatalog} : {}),
    ...(nativeOnlySources.length ? {nativeOnlySources} : {})
  };
}

function assertSources(
  state: KeplerSandboxState,
  sources: readonly {datasetId: string; revision: number}[],
  limits: SandboxBundleLimits,
  nativeOnlyDatasetIds: readonly string[] = []
): void {
  if (
    sources.length !== state.sources.length ||
    sources.length > limits.maximumSources ||
    new Set(sources.map(source => source.datasetId)).size !== sources.length
  )
    throw new Error('Bundle must include each declared source exactly once');
  for (const descriptor of state.sources) {
    if (
      !sources.some(
        source =>
          source.datasetId === descriptor.dataset.datasetId &&
          source.revision === descriptor.dataset.revision
      )
    )
      throw new Error('Bundle contains a missing or stale source');
  }
  assertSandboxSourcesCovered(state.nativeConfig, [
    ...state.sources.map(source => source.dataset.datasetId),
    ...nativeOnlyDatasetIds
  ]);
}

function assertTable(
  source: KeplerSandboxSource,
  table: arrow.Table,
  limits: SandboxBundleLimits
): void {
  if (table.numRows > limits.maximumRows) throw new Error('Bundle source exceeds the row budget');
  assertKeplerSandboxSourceMatches(source, {
    ...source,
    schema: table.schema.fields.map(field => ({
      name: field.name,
      type: field.type.toString(),
      nullable: field.nullable
    }))
  });
}

function assertLimits(limits: SandboxBundleLimits): void {
  for (const value of [
    limits.maximumDocumentBytes,
    limits.maximumSourceBytes,
    limits.maximumRows,
    limits.maximumSources
  ])
    if (!Number.isSafeInteger(value) || value < 1) throw new Error('Invalid bundle capacity');
}

function assertDocumentSize(serialized: string, limits: SandboxBundleLimits): void {
  if (new TextEncoder().encode(serialized).byteLength > limits.maximumDocumentBytes)
    throw new Error('Sandbox bundle exceeds the document budget');
}

function readEncodedSource(value: unknown): EncodedSource {
  if (
    !value ||
    typeof value !== 'object' ||
    !('datasetId' in value) ||
    typeof value.datasetId !== 'string' ||
    !('revision' in value) ||
    typeof value.revision !== 'number' ||
    !Number.isSafeInteger(value.revision) ||
    value.revision < 1 ||
    !('encoding' in value) ||
    value.encoding !== 'arrow-ipc-stream-base64' ||
    !('byteLength' in value) ||
    typeof value.byteLength !== 'number' ||
    !Number.isSafeInteger(value.byteLength) ||
    value.byteLength < 0 ||
    !('sha256' in value) ||
    typeof value.sha256 !== 'string' ||
    !/^[a-f0-9]{64}$/.test(value.sha256) ||
    !('data' in value) ||
    typeof value.data !== 'string' ||
    value.data.length !== Math.ceil(value.byteLength / 3) * 4 ||
    !/^[A-Za-z0-9+/]*={0,2}$/.test(value.data)
  )
    throw new Error('Malformed bundle source');
  return {
    datasetId: value.datasetId,
    revision: value.revision,
    encoding: value.encoding,
    byteLength: value.byteLength,
    sha256: value.sha256,
    data: value.data
  };
}

async function computeDigest(bytes: Uint8Array): Promise<string> {
  const hash = await crypto.subtle.digest('SHA-256', Uint8Array.from(bytes));
  return Array.from(new Uint8Array(hash), value => value.toString(16).padStart(2, '0')).join('');
}

function encodeBytes(bytes: Uint8Array): string {
  const chunks: string[] = [];
  for (let offset = 0; offset < bytes.length; offset += 8192)
    chunks.push(String.fromCharCode(...bytes.subarray(offset, offset + 8192)));
  return btoa(chunks.join(''));
}

function decodeBytes(encoded: string): Uint8Array {
  return Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
}
