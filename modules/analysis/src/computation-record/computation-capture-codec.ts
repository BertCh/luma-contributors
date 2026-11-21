// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getCaptureMetadataIssues, isMatchingCaptureArray} from './capture-value-validation';
import {GPU_COMPUTATION_CAPTURES_SCHEMA} from './computation-record-types';
import type {
  GPUComputationCaptures,
  GPUComputationCaptureElementType
} from './computation-record-types';

const LITTLE_ENDIAN = new Uint8Array(Uint32Array.of(1).buffer)[0] === 1;
const CATEGORIES = ['snapshots', 'stamps', 'replays'] as const;

/**
 * Lossless JSON, compatible with lenses' `b64:` little-endian 32-bit capture values.
 * Element type stays explicit on each capture; valueCount checks the decoded payload length.
 * Typed subarrays serialize only their own bytes, preserving NaN payloads and signed zero.
 */
export function serializeGPUComputationCaptures(captures: GPUComputationCaptures): string {
  validateCapturesEnvelope(captures);
  const wire = {...captures} as Record<string, unknown>;
  for (const category of CATEGORIES) {
    wire[category] = captures[category].map(capture => {
      validateCaptureMetadata(capture);
      const components = capture.components === undefined ? 1 : capture.components;
      if (capture.values.length % components !== 0)
        throw new Error('computation-captures values: expected whole components');
      return {
        ...capture,
        valueCount: capture.values.length,
        values: encodeValues(capture.values, capture.elementType)
      };
    });
  }
  return JSON.stringify(wire);
}

/** Loads binary captures or legacy numeric arrays; null lanes and malformed payloads throw. */
export function deserializeGPUComputationCaptures(text: string): GPUComputationCaptures {
  const wire: unknown = JSON.parse(text);
  validateCapturesEnvelope(wire);
  for (const category of CATEGORIES) {
    const captures = wire[category];
    if (!Array.isArray(captures))
      throw new Error(`computation-captures ${category}: expected array`);
    wire[category] = captures.map((capture: unknown) => {
      if (!isObject(capture)) throw new Error('computation-captures: expected capture object');
      validateCaptureMetadata(capture);
      const elementType = capture['elementType'];
      if (elementType !== 'f32' && elementType !== 'u32' && elementType !== 'i32')
        throw new Error('computation-captures elementType: expected f32, u32 or i32');
      const values = decodeValues(capture['values'], elementType);
      const count = capture['valueCount'];
      if (count !== undefined && (!Number.isSafeInteger(count) || count !== values.length))
        throw new Error('computation-captures valueCount: payload length does not match');
      const components = capture['components'] === undefined ? 1 : capture['components'];
      if (
        typeof components !== 'number' ||
        !Number.isSafeInteger(components) ||
        components <= 0 ||
        values.length % components !== 0
      )
        throw new Error('computation-captures values: expected whole components');
      return {...capture, values};
    });
  }
  return wire as unknown as GPUComputationCaptures;
}

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function decodeValues(
  values: unknown,
  elementType: GPUComputationCaptureElementType
): Float32Array | Uint32Array | Int32Array {
  if (typeof values === 'string') {
    if (!isBase64Values(values)) throw new Error('computation-captures values: expected b64 text');
    const binary = atob(values.slice(4));
    if (binary.length % 4 !== 0)
      throw new Error('computation-captures values: expected whole 32-bit elements');
    const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
    if (!LITTLE_ENDIAN) reverseWords(bytes);
    return elementType === 'f32'
      ? new Float32Array(bytes.buffer)
      : elementType === 'u32'
        ? new Uint32Array(bytes.buffer)
        : new Int32Array(bytes.buffer);
  }
  if (!Array.isArray(values)) throw new Error('computation-captures values: expected legacy array');
  values = Array.from(values, value => {
    if (elementType === 'f32') {
      if (value === 'NaN') return NaN;
      if (value === 'Infinity') return Infinity;
      if (value === '-Infinity') return -Infinity;
    }
    if (typeof value !== 'number')
      throw new Error('computation-captures values: expected numeric legacy lane');
    return value;
  });
  const numbers = values as number[];
  if (
    elementType !== 'f32' &&
    numbers.some(
      value =>
        !Number.isInteger(value) ||
        value < (elementType === 'u32' ? 0 : -2147483648) ||
        value > (elementType === 'u32' ? 4294967295 : 2147483647)
    )
  )
    throw new Error(`computation-captures values: invalid ${elementType} lane`);
  return elementType === 'f32'
    ? Float32Array.from(numbers)
    : elementType === 'u32'
      ? Uint32Array.from(numbers)
      : Int32Array.from(numbers);
}

function encodeValues(
  values: ArrayLike<number>,
  elementType: GPUComputationCaptureElementType
): string {
  const matches = isMatchingCaptureArray(values, elementType);
  const typed = matches
    ? (values as Float32Array | Uint32Array | Int32Array)
    : decodeValues(Array.from(values), elementType);
  const bytes = new Uint8Array(typed.buffer, typed.byteOffset, typed.byteLength).slice();
  if (!LITTLE_ENDIAN) reverseWords(bytes);
  let binary = '';
  for (let offset = 0; offset < bytes.length; offset += 0x8000)
    binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
  return `b64:${btoa(binary)}`;
}

function reverseWords(bytes: Uint8Array): void {
  for (let offset = 0; offset < bytes.length; offset += 4)
    bytes.subarray(offset, offset + 4).reverse();
}

function validateCaptureMetadata(capture: Record<string, unknown>): void {
  const issue = getCaptureMetadataIssues(capture)[0];
  if (issue) throw new Error(`computation-captures ${issue.field}: ${issue.message}`);
}

// Linear character scan: no repeated regexp groups or input-dependent regexp stack.
function isBase64Values(values: string): boolean {
  if (!values.startsWith('b64:')) return false;
  const payload = values.slice(4);
  if (payload.length % 4 !== 0) return false;
  const padding = payload.endsWith('==') ? 2 : payload.endsWith('=') ? 1 : 0;
  for (let index = 0; index < payload.length - padding; index++) {
    const code = payload.charCodeAt(index);
    if (
      !(
        (code >= 65 && code <= 90) ||
        (code >= 97 && code <= 122) ||
        (code >= 48 && code <= 57) ||
        code === 43 ||
        code === 47
      )
    )
      return false;
  }
  return true;
}

function validateCapturesEnvelope(value: unknown): asserts value is Record<string, unknown> {
  if (
    !isObject(value) ||
    value['schema'] !== GPU_COMPUTATION_CAPTURES_SCHEMA ||
    typeof value['runId'] !== 'string' ||
    !value['runId']
  )
    throw new Error('computation-captures: expected captures schema and runId');
  if (
    value['executionId'] !== undefined &&
    (typeof value['executionId'] !== 'string' || !value['executionId'])
  )
    throw new Error('computation-captures executionId: expected a nonempty string');
  for (const category of CATEGORIES)
    if (!Array.isArray(value[category]))
      throw new Error(`computation-captures ${category}: expected array`);
}
