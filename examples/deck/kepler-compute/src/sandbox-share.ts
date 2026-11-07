// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {parseKeplerSandboxState, serializeKeplerSandboxState} from '@luma-contributors/kepler';
import {readSandboxFixture} from './sandbox-state';
import {DATASET_ID} from './sample-data';
import {assertSandboxSourcesCovered} from './sandbox-source-validation';

const SHARE_PREFIX = '#sandbox=v1.';
const MAXIMUM_SHARE_BYTES = 48 * 1024;
const MAXIMUM_LINK_LENGTH = 64 * 1024;

/** A share link contains configuration and a reproducible fixture reference, never uploaded rows. */
export function createSandboxShareLink(serialized: string, location: string): string {
  const shared = validateSharedState(serialized);
  const bytes = new TextEncoder().encode(shared.serialized);
  if (bytes.byteLength > MAXIMUM_SHARE_BYTES)
    throw new Error('State is too large for a share link; use Save state instead');
  const address = new URL(location);
  if (!['http:', 'https:'].includes(address.protocol) || address.username || address.password)
    throw new Error('Share links require an HTTP or HTTPS sandbox address');
  // Preserve the sandbox route, but do not carry unrelated query parameters or credentials.
  address.search = '';
  address.searchParams.set('rows', String(shared.rowCount));
  let binary = '';
  for (const byte of bytes) binary += String.fromCharCode(byte);
  address.hash = `${SHARE_PREFIX}${btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '')}`;
  if (address.href.length > MAXIMUM_LINK_LENGTH)
    throw new Error('State is too large for a share link; use Save state instead');
  return address.href;
}

/** Returns validated state for explicit replay. Unrelated URL fragments are left alone. */
export function readSandboxShareLink(location: string): string | null {
  const address = new URL(location);
  if (!address.hash.startsWith('#sandbox=')) return null;
  if (address.href.length > MAXIMUM_LINK_LENGTH)
    throw new Error('Share link exceeds the supported size');
  if (!address.hash.startsWith(SHARE_PREFIX)) throw new Error('Unsupported share link version');
  const encoded = address.hash.slice(SHARE_PREFIX.length);
  if (!encoded || !/^[A-Za-z0-9_-]+$/.test(encoded) || encoded.length % 4 === 1)
    throw new Error('Malformed share link');
  const binary = atob(encoded.replace(/-/g, '+').replace(/_/g, '/'));
  if (binary.length > MAXIMUM_SHARE_BYTES)
    throw new Error('Shared state exceeds the supported size');
  const bytes = Uint8Array.from(binary, character => character.charCodeAt(0));
  return validateSharedState(new TextDecoder('utf-8', {fatal: true}).decode(bytes)).serialized;
}

function validateSharedState(serialized: string): {serialized: string; rowCount: number} {
  if (new TextEncoder().encode(serialized).byteLength > MAXIMUM_SHARE_BYTES)
    throw new Error('State is too large for a share link; use Save state instead');
  const saved = parseKeplerSandboxState(JSON.parse(serialized));
  if (saved.sources.length !== 1 || saved.selections.length !== 1 || saved.analyses.length > 1)
    throw new Error('Share links currently support one observations fixture and one analysis');
  if (saved.sources[0].dataset.datasetId !== DATASET_ID)
    throw new Error('Shared fixture belongs to an unsupported native dataset');
  const {rowCount} = readSandboxFixture(saved.sources[0]);
  assertSandboxSourcesCovered(saved.nativeConfig, [saved.sources[0].dataset.datasetId]);
  return {serialized: serializeKeplerSandboxState(saved), rowCount};
}

export {
  createSourceShareLink,
  readSourceShareLink,
  type SourceLinkDescriptor
} from './share-link-sources';
