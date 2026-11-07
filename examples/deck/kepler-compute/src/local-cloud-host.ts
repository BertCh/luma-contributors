// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {parseSandboxBundle, type SandboxBundleLimits} from './sandbox-bundle';
import {replaySandboxBundle, type SandboxCallbacks} from './sandbox-state';
import {serializeBundleWithNativeSources} from './bundle-native-embed';
import type {LocalCloudHost} from './local-cloud-service';

/** Same limits as the Portable workspace panel; saving to local storage must not widen them. */
export const LOCAL_CLOUD_BUNDLE_LIMITS: SandboxBundleLimits = {
  maximumDocumentBytes: 96 * 1024 * 1024,
  maximumSourceBytes: 64 * 1024 * 1024,
  maximumRows: 1_000_000,
  maximumSources: 16
};

/** Routes cloud save/load through the manual bundle export and replay entry points. */
export function createLocalCloudHost(
  callbacks: SandboxCallbacks,
  /** Surfaces save-time warnings (native-only tables that did not fit) in the Kepler UI. */
  reportNotice?: (message: string) => void
): LocalCloudHost {
  return {
    async captureBundle() {
      // Native-only Kepler tables embed their originals exactly like the Portable workspace panel.
      const result = await serializeBundleWithNativeSources(callbacks, LOCAL_CLOUD_BUNDLE_LIMITS);
      if (result.skippedMessage) reportNotice?.(`Map saved. ${result.skippedMessage}`);
      return result.serialized;
    },
    async restoreBundle(serialized) {
      const bundle = await parseSandboxBundle(serialized, LOCAL_CLOUD_BUNDLE_LIMITS);
      await replaySandboxBundle(bundle, callbacks);
      return {nativeConfig: bundle.state.nativeConfig};
    }
  };
}
