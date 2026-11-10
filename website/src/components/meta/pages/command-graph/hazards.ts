// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Per-buffer hazards of an uncollapsed view model, derived from each kernel's read and write sets
// in schedule order. The rule is the one the records use
// (modules/analysis/src/computation-record/describe-command-graph.ts:118-121): a read after the
// last writer is read-after-write; a write after readers since the last write is write-after-read;
// a write (without a read) after the last writer is write-after-write.

import type {LensViewModel} from '../../../../lens/types';

export type HazardKind = 'raw' | 'war' | 'waw';

export type HazardInstance = {
  kind: HazardKind;
  /** Index into `model.nodes` of the kernel that must run first. */
  from: number;
  /** Index into `model.nodes` of the kernel that waits. */
  to: number;
  buffer: string;
};

/** Every per-buffer hazard of the model, in schedule order of the waiting kernel. */
export function getHazardInstances(model: LensViewModel): HazardInstance[] {
  const instances: HazardInstance[] = [];
  const lastWriter = new Map<string, number>();
  const readersSinceWrite = new Map<string, number[]>();
  model.nodes.forEach((node, index) => {
    const reads = new Set(node.reads ?? []);
    const writes = new Set(node.writes ?? []);
    for (const buffer of new Set([...reads, ...writes])) {
      const writer = lastWriter.get(buffer);
      const isRead = reads.has(buffer);
      if (isRead && writer !== undefined)
        instances.push({kind: 'raw', from: writer, to: index, buffer});
      if (writes.has(buffer)) {
        for (const reader of readersSinceWrite.get(buffer) ?? [])
          instances.push({kind: 'war', from: reader, to: index, buffer});
        if (writer !== undefined && !isRead)
          instances.push({kind: 'waw', from: writer, to: index, buffer});
      }
    }
    for (const buffer of reads) {
      if (!writes.has(buffer))
        readersSinceWrite.set(buffer, [...(readersSinceWrite.get(buffer) ?? []), index]);
    }
    for (const buffer of writes) {
      lastWriter.set(buffer, index);
      readersSinceWrite.set(buffer, []);
    }
  });
  return instances;
}

/** The first hazard of a kind, or undefined. */
export function findFirstHazard(
  model: LensViewModel,
  kind: HazardKind
): HazardInstance | undefined {
  return getHazardInstances(model).find(instance => instance.kind === kind);
}

/** Buffers of `node` that `reads` and the earlier kernels that last wrote each one. */
export function getReadSources(
  model: LensViewModel,
  nodeIndex: number
): {buffer: string; writer: number | undefined}[] {
  const node = model.nodes[nodeIndex];
  return (node?.reads ?? []).map(buffer => {
    let writer: number | undefined;
    for (let index = nodeIndex - 1; index >= 0; index--) {
      if (model.nodes[index].writes?.includes(buffer)) {
        writer = index;
        break;
      }
    }
    return {buffer, writer};
  });
}
