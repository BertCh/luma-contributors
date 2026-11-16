// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {luma} from '@luma.gl/core';

/** Active memory-owning resources in luma's global GPU Resource Counts bucket. */
export type GPUResourceCounts = Readonly<{buffers: number; textures: number}>;

/**
 * Take the baseline after creating the shared test device and warming any intentional caches.
 * These counters are global: use in sequential specs and destroy compiled graphs, contributors
 * and caller-owned buffers before comparing. luma 10 counts textures as `Textures Active`.
 */
export function getGPUResourceCounts(): GPUResourceCounts {
  const statistics = luma.stats.get('GPU Resource Counts');
  return {
    buffers: statistics.get('Buffers Active').count,
    textures: statistics.get('Textures Active').count
  };
}

/** Positive deltas only; destroying a pre-existing resource cannot hide a leak of another type. */
export function getGPUResourceLeaks(
  baseline: GPUResourceCounts,
  current: GPUResourceCounts = getGPUResourceCounts()
): GPUResourceCounts {
  return {
    buffers: Math.max(0, current.buffers - baseline.buffers),
    textures: Math.max(0, current.textures - baseline.textures)
  };
}

/** Throws with the individual resource deltas, suitable for an afterEach or a scoped spec. */
export function validateNoGPUResourceLeaks(
  id: string,
  baseline: GPUResourceCounts,
  current: GPUResourceCounts = getGPUResourceCounts()
): void {
  const leaks = getGPUResourceLeaks(baseline, current);
  if (leaks.buffers || leaks.textures) {
    throw new Error(
      `${id} leaked GPU resources: ${leaks.buffers} buffers, ${leaks.textures} textures`
    );
  }
}
