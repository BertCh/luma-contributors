// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationSourceRef} from './computation-record-types';

/** Merge matching refs without discarding available text or changing preferred pipeline metadata. */
export function mergeGPUComputationSourceRefs(
  preferred: GPUComputationSourceRef,
  fallback: GPUComputationSourceRef
): GPUComputationSourceRef {
  if (preferred.id !== fallback.id) return preferred;
  if (
    preferred.text !== undefined &&
    fallback.text !== undefined &&
    preferred.text !== fallback.text
  )
    throw new Error(`Conflicting source text: ${preferred.id}`);
  return {
    ...fallback,
    ...Object.fromEntries(Object.entries(preferred).filter(([, value]) => value !== undefined))
  } as GPUComputationSourceRef;
}
