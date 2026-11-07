// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import * as arrow from 'apache-arrow';
import type {KeplerTable} from '@kepler.gl/table';
import {makeArrowSourceFromNativeDataset} from './native-dataset-source';

/** Explicit, visible capacity for native-only originals embedded in a portable bundle. */
export const NATIVE_ONLY_BUNDLE_LIMITS = Object.freeze({
  maximumTotalBytes: 32 * 1024 * 1024,
  maximumRows: 1_000_000,
  maximumCells: 16_000_000,
  maximumSources: 8
});

export type NativeOnlyBundleLimits = {
  maximumTotalBytes: number;
  maximumRows: number;
  maximumCells: number;
  maximumSources: number;
};

export type NativeFieldMetadata = {name: string; type: string; format: string};

/** JSON form stored in the bundle under `nativeOnlySources`. */
export type EmbeddedNativeSource = {
  datasetId: string;
  label: string;
  rowCount: number;
  encoding: 'arrow-ipc-stream-base64';
  byteLength: number;
  sha256: string;
  data: string;
  /** Original Kepler fields in native order, used to rebuild the native table. */
  fields: NativeFieldMetadata[];
};

export type RestoredNativeSource = {
  datasetId: string;
  label: string;
  table: arrow.Table;
  fields: NativeFieldMetadata[];
};

type RawEmbeddedField = {name?: unknown; type?: unknown; format?: unknown};
type RawEmbeddedSource = {
  datasetId?: unknown;
  label?: unknown;
  encoding?: unknown;
  byteLength?: unknown;
  sha256?: unknown;
  data?: unknown;
  fields?: unknown;
};

export type SkippedNativeSource = {datasetId: string; label: string; reason: string};

export type NativeOnlyCollection = {
  embedded: EmbeddedNativeSource[];
  skipped: SkippedNativeSource[];
};

/** Datasets present only as native Kepler tables (never resident, not derived, not table-only views). */
export function listNativeOnlyDatasets(
  datasets: Readonly<Record<string, KeplerTable>> | undefined,
  isResidentDataset: (datasetId: string) => boolean
): KeplerTable[] {
  return Object.values(datasets ?? {}).filter(dataset => {
    const metadata = dataset.metadata as Record<string, unknown> | undefined;
    return (
      !isResidentDataset(dataset.id) &&
      !(dataset as {type?: string}).type &&
      !metadata?.['derivedDataset'] &&
      Boolean(dataset.dataContainer) &&
      dataset.dataContainer.numRows() > 0
    );
  });
}

/** Encodes every native-only dataset that fits the explicit budget; the rest are reported, never dropped silently. */
export async function collectNativeOnlySources(
  datasets: readonly KeplerTable[],
  limits: NativeOnlyBundleLimits = NATIVE_ONLY_BUNDLE_LIMITS
): Promise<NativeOnlyCollection> {
  const embedded: EmbeddedNativeSource[] = [];
  const skipped: SkippedNativeSource[] = [];
  let totalBytes = 0;
  for (const dataset of datasets) {
    const reason = await tryEmbed(dataset, limits, embedded.length, totalBytes, embedded);
    if (typeof reason === 'string')
      skipped.push({datasetId: dataset.id, label: dataset.label, reason});
    else totalBytes += reason;
  }
  return {embedded, skipped};
}

/** Returns the embedded byte length on success or a user-visible reason when the dataset does not fit. */
async function tryEmbed(
  dataset: KeplerTable,
  limits: NativeOnlyBundleLimits,
  embeddedCount: number,
  totalBytes: number,
  output: EmbeddedNativeSource[]
): Promise<number | string> {
  if (embeddedCount >= limits.maximumSources)
    return `more than ${limits.maximumSources} native-only datasets; the limit is ${limits.maximumSources}`;
  let table: arrow.Table;
  try {
    table = makeArrowSourceFromNativeDataset(dataset, limits).table;
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
  const bytes = arrow.tableToIPC(table, 'stream');
  if (totalBytes + bytes.byteLength > limits.maximumTotalBytes)
    return `needs ${formatMebibytes(bytes.byteLength)} but only ${formatMebibytes(limits.maximumTotalBytes - totalBytes)} of the ${formatMebibytes(limits.maximumTotalBytes)} native-only budget remain`;
  output.push({
    datasetId: dataset.id,
    label: dataset.label,
    rowCount: table.numRows,
    encoding: 'arrow-ipc-stream-base64',
    byteLength: bytes.byteLength,
    sha256: await computeDigest(bytes),
    data: encodeBytes(bytes),
    fields: dataset.fields.map(field => ({
      name: field.name,
      type: field.type,
      format: field.format ?? ''
    }))
  });
  return bytes.byteLength;
}

export function describeSkippedNativeSources(skipped: readonly SkippedNativeSource[]): string {
  return skipped.length
    ? `Native-only datasets not embedded: ${skipped.map(item => `${item.label} (${item.reason})`).join('; ')}.`
    : '';
}

/** Parses, size-checks and verifies checksums for the optional `nativeOnlySources` document member. */
export async function readEmbeddedNativeSources(
  value: unknown,
  limits: NativeOnlyBundleLimits = NATIVE_ONLY_BUNDLE_LIMITS
): Promise<RestoredNativeSource[]> {
  if (value === undefined) return [];
  if (!Array.isArray(value) || value.length > limits.maximumSources)
    throw new Error('Malformed native-only bundle sources');
  let totalBytes = 0;
  const restored: RestoredNativeSource[] = [];
  for (const entry of value as RawEmbeddedSource[]) {
    if (
      !entry ||
      typeof entry.datasetId !== 'string' ||
      typeof entry.label !== 'string' ||
      entry.encoding !== 'arrow-ipc-stream-base64' ||
      typeof entry.byteLength !== 'number' ||
      !Number.isSafeInteger(entry.byteLength) ||
      typeof entry.sha256 !== 'string' ||
      !/^[a-f0-9]{64}$/.test(entry.sha256) ||
      typeof entry.data !== 'string' ||
      !/^[A-Za-z0-9+/]*={0,2}$/.test(entry.data) ||
      !Array.isArray(entry.fields)
    )
      throw new Error('Malformed native-only bundle source');
    totalBytes += entry.byteLength;
    if (totalBytes > limits.maximumTotalBytes)
      throw new Error('Native-only bundle sources exceed their byte budget');
    const bytes = decodeBytes(entry.data);
    if (bytes.byteLength !== entry.byteLength || (await computeDigest(bytes)) !== entry.sha256)
      throw new Error('Native-only bundle source checksum or byte length does not match');
    const table = arrow.tableFromIPC(bytes);
    if (table.numRows > limits.maximumRows)
      throw new Error('Native-only bundle source exceeds the row budget');
    const fields = (entry.fields as RawEmbeddedField[]).map(field => ({
      name: String(field.name),
      type: String(field.type),
      format: String(field.format ?? '')
    }));
    if (
      fields.length !== table.schema.fields.length ||
      fields.some((field, index) => field.name !== table.schema.fields[index].name)
    )
      throw new Error('Native-only bundle fields do not match their Arrow schema');
    restored.push({datasetId: entry.datasetId, label: entry.label, table, fields});
  }
  if (new Set(restored.map(item => item.datasetId)).size !== restored.length)
    throw new Error('Native-only bundle sources must be unique');
  return restored;
}

/** Rebuilds `{fields, rows}` for Kepler `addDataToMap`; null Arrow values stay null. */
export function makeNativeDataFromSource(source: RestoredNativeSource): {
  fields: NativeFieldMetadata[];
  rows: unknown[][];
} {
  const columns = source.table.schema.fields.map(
    (_field, index) => source.table.getChildAt(index)!
  );
  const rows: unknown[][] = [];
  for (let rowIndex = 0; rowIndex < source.table.numRows; rowIndex++)
    rows.push(
      columns.map((column, columnIndex) => {
        if (!column.isValid(rowIndex)) return null;
        const value: unknown = column.get(rowIndex);
        if (typeof value === 'bigint')
          return source.fields[columnIndex].type === 'integer' ? Number(value) : value.toString();
        return value;
      })
    );
  return {fields: source.fields, rows};
}

/**
 * Datasets staged by a bundle restore; the next native configuration replay adds them to the map
 * together with the resident datasets so saved layers and filters bind to them.
 */
const stagedNativeSources: RestoredNativeSource[] = [];

export function stageNativeSources(sources: readonly RestoredNativeSource[]): void {
  stagedNativeSources.length = 0;
  stagedNativeSources.push(...sources);
}

export function getStagedNativeDatasets(): {
  info: {id: string; label: string};
  data: {fields: NativeFieldMetadata[]; rows: unknown[][]};
}[] {
  return stagedNativeSources.map(source => ({
    info: {id: source.datasetId, label: source.label},
    data: makeNativeDataFromSource(source)
  }));
}

export function clearStagedNativeSources(): void {
  stagedNativeSources.length = 0;
}

function formatMebibytes(byteLength: number): string {
  return `${(byteLength / (1024 * 1024)).toFixed(1)} MiB`;
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
