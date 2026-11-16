// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  parseAnalysisSensitivityProvenance,
  type AnalysisSensitivityProvenance
} from '@luma-contributors/kepler';
import type {KeplerJSONValue} from '@luma-contributors/kepler';

/** Where one catalog input came from; sample and text inputs are re-read, native inputs are re-resolved. */
export type CatalogInputReference =
  | {mode: 'sample'}
  | {mode: 'text'; text: string}
  | {mode: 'native'; datasetId: string; field: string};

/** One catalog run saved with the workspace: the checksummed invocation plus input provenance. */
export type CatalogInvocationRecord = {
  /** Stable key, `${operationId}#${n}`. */
  id: string;
  operationId: string;
  /** Output of serializeCatalogInvocation: normalized parameters and input content hashes. */
  invocation: string;
  inputs: Record<string, CatalogInputReference>;
  engine: 'auto' | 'cpu';
  sensitivity?: AnalysisSensitivityProvenance;
};

export const MAXIMUM_CATALOG_INVOCATIONS = 32;
export const MAXIMUM_CATALOG_TEXT_INPUT_CHARACTERS = 65536;

let records: CatalogInvocationRecord[] = [];
const listeners = new Set<() => void>();

export function listCatalogInvocations(): readonly CatalogInvocationRecord[] {
  return records;
}

export function subscribeCatalogInvocations(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function notifyListeners(): void {
  for (const listener of listeners) listener();
}

/** Replaces an identical earlier run (same operation, invocation text and inputs) and bounds the list. */
export function recordCatalogInvocation(
  record: Omit<CatalogInvocationRecord, 'id'>
): CatalogInvocationRecord {
  // Large pasted inputs are not embedded; the content hash in the invocation still detects a mismatch.
  const inputs = Object.fromEntries(
    Object.entries(record.inputs).map(([name, input]) => [
      name,
      input.mode === 'text' && input.text.length > MAXIMUM_CATALOG_TEXT_INPUT_CHARACTERS
        ? {mode: 'text' as const, text: ''}
        : input
    ])
  );
  const key = JSON.stringify([
    record.operationId,
    record.invocation,
    inputs,
    record.engine,
    record.sensitivity
  ]);
  const existing = records.find(
    item =>
      JSON.stringify([
        item.operationId,
        item.invocation,
        item.inputs,
        item.engine,
        item.sensitivity
      ]) === key
  );
  if (existing) return existing;
  const sequence = records.filter(item => item.operationId === record.operationId).length + 1;
  const created = {...record, inputs, id: `${record.operationId}#${sequence}`};
  records = [...records, created].slice(-MAXIMUM_CATALOG_INVOCATIONS);
  notifyListeners();
  return created;
}

export function clearCatalogInvocations(): void {
  records = [];
  notifyListeners();
}

export function serializeCatalogInvocationRecords(): KeplerJSONValue | undefined {
  return records.length
    ? (JSON.parse(JSON.stringify({schemaVersion: 1, records})) as KeplerJSONValue)
    : undefined;
}

/** Strict reader for a saved extension member; a malformed member is rejected as a whole. */
export function parseCatalogInvocationRecords(value: unknown): CatalogInvocationRecord[] {
  if (!isRecord(value) || value['schemaVersion'] !== 1 || !Array.isArray(value['records']))
    throw new Error('Unsupported catalog invocation member');
  if (value['records'].length > MAXIMUM_CATALOG_INVOCATIONS)
    throw new Error('Too many saved catalog invocations');
  const seen = new Set<string>();
  return value['records'].map((raw: unknown) => {
    if (
      !isRecord(raw) ||
      typeof raw['id'] !== 'string' ||
      typeof raw['operationId'] !== 'string' ||
      typeof raw['invocation'] !== 'string' ||
      (raw['engine'] !== 'auto' && raw['engine'] !== 'cpu') ||
      !isRecord(raw['inputs']) ||
      seen.has(raw['id'])
    )
      throw new Error('Malformed catalog invocation record');
    seen.add(raw['id']);
    const inputs: Record<string, CatalogInputReference> = {};
    for (const [name, input] of Object.entries(raw['inputs'])) {
      if (!isRecord(input)) throw new Error('Malformed catalog input reference');
      if (input['mode'] === 'sample') inputs[name] = {mode: 'sample'};
      else if (input['mode'] === 'text' && typeof input['text'] === 'string')
        inputs[name] = {mode: 'text', text: input['text']};
      else if (
        input['mode'] === 'native' &&
        typeof input['datasetId'] === 'string' &&
        typeof input['field'] === 'string'
      )
        inputs[name] = {mode: 'native', datasetId: input['datasetId'], field: input['field']};
      else throw new Error('Malformed catalog input reference');
    }
    return {
      id: raw['id'],
      operationId: raw['operationId'],
      invocation: raw['invocation'],
      engine: raw['engine'],
      inputs,
      ...(raw['sensitivity'] === undefined
        ? {}
        : {sensitivity: parseAnalysisSensitivityProvenance(raw['sensitivity'])})
    };
  });
}

/** Replaces the list with a restored one (restore replaces the workspace; it never merges). */
export function restoreCatalogInvocations(restored: readonly CatalogInvocationRecord[]): void {
  records = [...restored];
  notifyListeners();
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}
