// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

/** Capture a run before queueing work; invalidation is permanent for that execution. */
export function createAnalysisRunIdentity<Runtime>(options: {
  signal: AbortSignal;
  getRuntime: () => Runtime;
  getRevisionKey: () => string;
  isCurrent?: () => boolean;
}): {isCurrent: () => boolean} {
  const runtime = options.getRuntime();
  const revisionKey = options.getRevisionKey();
  let obsolete = false;
  return {
    isCurrent: () => {
      obsolete ||= options.signal.aborted;
      if (!obsolete) {
        try {
          obsolete =
            options.getRuntime() !== runtime ||
            options.getRevisionKey() !== revisionKey ||
            options.isCurrent?.() === false;
        } catch {
          obsolete = true;
        }
      }
      return !obsolete;
    }
  };
}
