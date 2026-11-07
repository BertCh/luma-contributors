// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  collectNativeOnlySources,
  describeSkippedNativeSources,
  type EmbeddedNativeSource,
  type SkippedNativeSource
} from './bundle-native-sources';
import {getNativeOnlyDatasets} from './native-source-access';
import {serializeSandboxBundle, type SandboxBundleLimits} from './sandbox-bundle';
import {materializeSandboxBundleSources, saveSandbox, type SandboxCallbacks} from './sandbox-state';

export type SerializedBundleWithNativeSources = {
  serialized: string;
  embedded: EmbeddedNativeSource[];
  skipped: SkippedNativeSource[];
  /** Empty when nothing was skipped; otherwise a user-visible sentence listing every skipped table. */
  skippedMessage: string;
  /** Empty when nothing was embedded. */
  embeddedMessage: string;
};

/**
 * Shared by the local-cloud save and the standalone-HTML export: serializes the sandbox bundle
 * together with the originals of native-only (non-resident) Kepler tables, exactly like the manual
 * Portable bundle panel. Tables over the native-only limits are returned in `skipped`, never dropped silently.
 */
export async function serializeBundleWithNativeSources(
  callbacks: SandboxCallbacks,
  limits: SandboxBundleLimits
): Promise<SerializedBundleWithNativeSources> {
  const nativeOnly = await collectNativeOnlySources(getNativeOnlyDatasets());
  const skippedMessage = describeSkippedNativeSources(nativeOnly.skipped);
  let state: string;
  try {
    // saveSandbox rejects unfinished stream populations and stale datasets.
    state = saveSandbox(
      callbacks,
      nativeOnly.embedded.map(source => source.datasetId)
    );
  } catch (error) {
    throw new Error(
      `${error instanceof Error ? error.message : String(error)} ${skippedMessage}`.trim()
    );
  }
  const serialized = await serializeSandboxBundle(
    state,
    await materializeSandboxBundleSources(callbacks.runtime, limits),
    limits,
    undefined,
    nativeOnly.embedded
  );
  return {
    serialized,
    embedded: nativeOnly.embedded,
    skipped: nativeOnly.skipped,
    skippedMessage,
    embeddedMessage: nativeOnly.embedded.length
      ? `Native-only datasets embedded: ${nativeOnly.embedded.map(source => source.label).join(', ')}.`
      : ''
  };
}
