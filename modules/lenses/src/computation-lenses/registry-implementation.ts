// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  serializeGPUComputationCaptures,
  deserializeGPUComputationCaptures,
  type GPUComputationCaptures
} from '@luma-contributors/analysis/computation-record';
import {LENS_RUNG_IDS} from '../computation-views/types';
import {compactLensRecordEnvelope, expandLensRecordEnvelope} from './record-encoding';
import type {
  DecodedLensFile,
  LensRecordingMetadata,
  LensScenarioDescriptor,
  LensScenarioPresentation,
  LensScenarioResult,
  RecordedLensFile
} from './types';
import {
  assertLensSidecarIdentity,
  assertValidLensRecord,
  assertValidLensSidecars
} from './validate-lens-record';

const SCENARIO_ID =
  /^[a-z0-9]+(?:-[a-z0-9]+)*\.[a-z0-9]+(?:-[a-z0-9]+)*(?:\.[a-z0-9]+(?:-[a-z0-9]+)*)?$/;
const scenarios = new Map<string, LensScenarioDescriptor>();

/** Checks the public scenario-id syntax without loading GPU code. */
export function isLensScenarioId(id: string): boolean {
  return SCENARIO_ID.test(id);
}

/**
 * Registers a lazy descriptor. Scenario modules call this at module scope, then hosts import
 * those registration modules explicitly; the registry never imports or executes GPU scenarios.
 * Returns an idempotent disposer for temporary registrations; it removes only this descriptor.
 */
export function registerLensScenario(descriptor: LensScenarioDescriptor): () => void {
  if (!isLensScenarioId(descriptor.id)) throw new Error(`Invalid scenario id: ${descriptor.id}`);
  const [entry] = descriptor.id.split('.');
  if (entry !== descriptor.entry) throw new Error('Scenario entry must match the first id segment');
  // The element is the documented export (class name, or function name for recipes), so the
  // coverage contract can join scenarios to pack operators and formulas by that name.
  if (!/^[A-Za-z_$][\w$]*$/.test(descriptor.element))
    throw new Error(`Scenario element must be an exported identifier: ${descriptor.element}`);
  if (
    !descriptor.rungs.includes(descriptor.defaultRung) ||
    descriptor.rungs.some(rung => !LENS_RUNG_IDS.includes(rung))
  )
    throw new Error('Scenario defaultRung must belong to valid rungs');
  if (scenarios.has(descriptor.id)) throw new Error(`Duplicate scenario id: ${descriptor.id}`);
  scenarios.set(descriptor.id, descriptor);
  const id = descriptor.id;
  let registered = true;
  return () => {
    if (!registered) return;
    registered = false;
    if (scenarios.get(id) === descriptor) scenarios.delete(id);
  };
}

/** Looks up a scenario descriptor without loading it. */
export function getLensScenario(id: string): LensScenarioDescriptor | undefined {
  return scenarios.get(id);
}

/** Lists scenarios deterministically by id. */
export function listLensScenarios(): LensScenarioDescriptor[] {
  return [...scenarios.values()].sort((first, second) =>
    first.id < second.id ? -1 : first.id > second.id ? 1 : 0
  );
}

/**
 * Copies a result to a JSON-safe, compacted envelope: non-finite values become explicit tokens,
 * typed captures base64, and the record is losslessly compacted (`record-encoding.ts`).
 * Metadata must satisfy the same timestamp, scenario ID and adapter requirements as the reader.
 * Decode the opaque payload with fromRecordedLensFile before accessing graph objects or values.
 */
export function toRecordedLensFile(
  result: LensScenarioResult,
  options: {
    scenarioId: string;
    recordedAt: string;
    adapterLabel: string;
    variant?: string;
    /** Persisted so isolated readers select the same headline; see LensRecordingMetadata. */
    presentation?: LensScenarioPresentation;
  }
): RecordedLensFile {
  const {presentation, ...rest} = options;
  const metadata = {
    schema: 'luma.lens-record/1' as const,
    ...rest,
    ...(presentation !== undefined ? {presentation: {...presentation}} : {})
  };
  validateLensRecordingMetadata(metadata);
  assertLensSidecarIdentity(result);
  // The record schema is numeric in memory; only this serialization boundary uses string tokens.
  // Lossless compaction (see record-encoding.ts) keeps docs records within their size budget.
  return compactLensRecordEnvelope({
    ...metadata,
    record: transformRecordValues(result, false)
  }) as RecordedLensFile;
}

/** Validates an envelope and restores capture arrays, including NaN and infinities. */
export function fromRecordedLensFile(json: unknown): DecodedLensFile {
  validateLensRecordingMetadata(json);
  if (
    !isObject(json['record']) ||
    json['record']['schema'] !== 'luma.computation/1' ||
    typeof json['record']['runId'] !== 'string' ||
    !json['record']['runId'] ||
    !Array.isArray(json['record']['graphs'])
  )
    throw new Error('Invalid lens record record');
  // Validation and recursive conversion are localized to the external JSON boundary.
  const expanded = expandLensRecordEnvelope(json);
  const expandedRecord = expanded['record'];
  if (
    !isObject(expandedRecord) ||
    !Array.isArray(expandedRecord['graphs']) ||
    !expandedRecord['graphs'].every(
      (graph: unknown) =>
        isObject(graph) &&
        typeof graph['id'] === 'string' &&
        graph['id'].length > 0 &&
        Array.isArray(graph['nodes']) &&
        graph['nodes'].every(
          (node: unknown) =>
            isObject(node) && typeof node['id'] === 'string' && node['id'].length > 0
        )
    )
  ) {
    throw new Error('Invalid lens record graph: expected decoded graph and node objects with ids');
  }
  const {record: _record, ...metadata} = expanded;
  const envelope = {
    ...(transformValues(metadata, true) as object),
    record: transformRecordValues(expandedRecord, true)
  } as DecodedLensFile;
  assertValidLensRecord(envelope.record);
  assertLensSidecarIdentity(envelope.record);
  assertValidLensSidecars(envelope.record, envelope.record);
  return envelope;
}

function validateLensRecordingMetadata(
  json: unknown
): asserts json is Record<string, unknown> & LensRecordingMetadata {
  if (!isObject(json) || json['schema'] !== 'luma.lens-record/1')
    throw new Error('Invalid lens record schema');
  if (typeof json['scenarioId'] !== 'string' || !isLensScenarioId(json['scenarioId']))
    throw new Error('Invalid lens record scenarioId');
  if (typeof json['recordedAt'] !== 'string' || !Number.isFinite(Date.parse(json['recordedAt'])))
    throw new Error('Invalid lens record recordedAt');
  if (typeof json['adapterLabel'] !== 'string' || !json['adapterLabel'].trim())
    throw new Error('Invalid lens record adapterLabel');
  if (json['variant'] !== undefined && typeof json['variant'] !== 'string')
    throw new Error('Invalid lens record variant');
  const presentation = json['presentation'];
  if (
    presentation !== undefined &&
    (!isObject(presentation) ||
      Object.keys(presentation).some(key => key !== 'primaryOutputCaptureId') ||
      (presentation['primaryOutputCaptureId'] !== undefined &&
        (typeof presentation['primaryOutputCaptureId'] !== 'string' ||
          !presentation['primaryOutputCaptureId'])))
  )
    throw new Error(
      'Invalid lens record presentation: expected {primaryOutputCaptureId?: nonempty string}'
    );
}

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Canonicalize legacy numeric arrays before JSON can erase signed zero or nonfinite lanes. */
function prepareLegacyCaptureArrays(value: Record<string, unknown>): Record<string, unknown> {
  const wire = {...value};
  for (const category of ['snapshots', 'stamps', 'replays']) {
    const captures = value[category];
    if (!Array.isArray(captures)) continue; // The public decoder reports malformed envelopes.
    wire[category] = captures.map((capture: unknown) => {
      if (
        !isObject(capture) ||
        !(Array.isArray(capture['values']) || ArrayBuffer.isView(capture['values']))
      )
        return capture;
      const encoded = JSON.parse(
        serializeGPUComputationCaptures({
          schema: 'luma.computation-captures/1',
          runId: value['runId'],
          snapshots: [capture],
          stamps: [],
          replays: []
        } as unknown as GPUComputationCaptures)
      ) as {snapshots: {values: string}[]};
      // Preserve any recorded valueCount so malformed legacy payload lengths still reject.
      return {...capture, values: encoded.snapshots[0].values};
    });
  }
  return wire;
}

/** The shared capture codec belongs only to the record's actual sidecar property. */
function transformRecordValues(value: unknown, restore: boolean): unknown {
  if (!isObject(value)) return value;
  const {captures, ...record} = value;
  return {
    ...(transformValues(record, restore) as object),
    ...(captures === undefined
      ? {}
      : {
          captures: restore
            ? deserializeGPUComputationCaptures(
                JSON.stringify(isObject(captures) ? prepareLegacyCaptureArrays(captures) : captures)
              )
            : JSON.parse(serializeGPUComputationCaptures(captures as GPUComputationCaptures))
        })
  };
}

function transformValues(value: unknown, restore: boolean): unknown {
  if (Array.isArray(value)) return value.map(item => transformValues(item, restore));
  if (!isObject(value)) return value;
  const result: Record<string, unknown> = {};
  for (const [key, item] of Object.entries(value)) {
    const isArrayLike =
      item !== null &&
      typeof item === 'object' &&
      'length' in item &&
      typeof item.length === 'number';
    // Free-form attributes may also be named "values" without representing a capture.
    if (key !== 'values' || (!isArrayLike && value['elementType'] === undefined)) {
      result[key] = transformValues(item, restore);
      continue;
    }
    if (
      item === null ||
      typeof item !== 'object' ||
      !('length' in item) ||
      typeof item.length !== 'number'
    )
      throw new Error('Invalid record values: expected an array');
    const values = Array.from(item as ArrayLike<unknown>, entry => {
      if (restore) {
        if (entry === 'NaN') return NaN;
        if (entry === 'Infinity') return Infinity;
        if (entry === '-Infinity') return -Infinity;
        if (typeof entry !== 'number')
          throw new Error('Invalid record values: expected numbers or non-finite tokens');
        return entry;
      }
      if (typeof entry !== 'number') throw new Error('Invalid record values: expected numbers');
      return Number.isFinite(entry) ? entry : String(entry);
    });
    result[key] = values;
  }
  return result;
}
