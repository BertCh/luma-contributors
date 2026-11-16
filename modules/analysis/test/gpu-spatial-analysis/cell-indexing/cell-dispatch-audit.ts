// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GraphDataView, type GPUCommandNode} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import {getCoreDefaultBuffer} from '../../../src/analysis-core/network-topology';

/** Checks Metal's physical-buffer usage scope, independently of logical/aligned byte overlap. */
export function expectCellStorageScopes<Parameters>(
  nodes: readonly GPUCommandNode<Parameters>[]
): void {
  for (const node of nodes) {
    const usages = new Map<unknown, Set<string>>();
    for (const resource of node.resources ?? []) {
      if (!('buffer' in resource) || !resource.usage.startsWith('storage-')) continue;
      const handle =
        resource.buffer instanceof GraphDataView ? resource.buffer.buffer : resource.buffer;
      const physical = handle.defaultBuffer ? getCoreDefaultBuffer(handle.defaultBuffer) : handle;
      const modes = usages.get(physical) ?? new Set<string>();
      modes.add(resource.usage);
      usages.set(physical, modes);
    }
    for (const modes of usages.values())
      expect(
        modes.has('storage-read') &&
          (modes.has('storage-read-write') || modes.has('storage-write')),
        `${node.id} binds one physical buffer as both read-only and writable storage`
      ).toBe(false);
  }
}

/** Cooperative kernels must not use an invocation-dependent early return before any barrier. */
export function expectCellBarrierEntry(source: string): void {
  const main = source.slice(source.indexOf('fn main('));
  const lastBarrier = Math.max(
    main.lastIndexOf('workgroupBarrier('),
    main.lastIndexOf('storageBarrier(')
  );
  if (lastBarrier >= 0) {
    // A workgroup-id guard is uniform across all lanes. Published scan kernels and the shared
    // node builder use it to bound padded groups; an invocation-index guard is not uniform.
    const guarded = main
      .slice(0, lastBarrier)
      .replace(
        /if\s*\(workgroupIndex\s*>=\s*(?:[A-Z][A-Z_0-9]*|[0-9]+u)\)\s*\{\s*return\s*;\s*\}/g,
        ''
      );
    expect(guarded).not.toMatch(/\breturn\s*;/);
  }
}
