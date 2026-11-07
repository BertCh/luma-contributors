// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {KeplerArrowDataset} from '@luma-contributors/kepler';

/** Pages requested per bounded step; each step reads at most this many pages before yielding. */
const PAGES_PER_STEP = 64;

export type ResidentMaterializationOptions = {
  /** Throws when the dataset or selection changed; called between bounded steps. */
  assertCurrent?: () => void;
};

/**
 * Stream GPU pages into the dataset's accounted lazy CPU view. Rows are read through
 * `resident.ensureRows` in bounded steps (never one full-column readback) and may be a sorted
 * subset or, when omitted, every row. Non-resident datasets already hold CPU cells and return.
 */
export async function materializeResidentRows(
  value: Pick<KeplerArrowDataset, 'resident' | 'source'>,
  rows?: ArrayLike<number>,
  options: ResidentMaterializationOptions = {}
): Promise<void> {
  const resident = value.resident;
  if (!resident) return;
  const stepRows = resident.pageRows * PAGES_PER_STEP;
  if (rows) {
    for (let start = 0; start < rows.length; start += stepRows) {
      options.assertCurrent?.();
      await resident.ensureRows(
        Array.from(
          {length: Math.min(stepRows, rows.length - start)},
          (_, index) => rows[start + index]
        )
      );
      await yieldMaterialization();
    }
    options.assertCurrent?.();
    return;
  }
  // Pages never cross source batch boundaries, so enumerate the first row of every page per batch.
  let batchOffset = 0;
  let pending: number[] = [];
  for (const batch of value.source.batches) {
    for (let row = 0; row < batch.numRows; row += resident.pageRows) {
      pending.push(batchOffset + row);
      if (pending.length >= PAGES_PER_STEP) {
        options.assertCurrent?.();
        await resident.ensureRows(pending);
        pending = [];
        await yieldMaterialization();
      }
    }
    batchOffset += batch.numRows;
  }
  options.assertCurrent?.();
  if (pending.length) await resident.ensureRows(pending);
  options.assertCurrent?.();
}

/** Estimated CPU bytes of a fully materialized resident dataset: four bytes per cell plus validity. */
export function estimateResidentMaterializedBytes(
  value: Pick<KeplerArrowDataset, 'source'>
): number {
  return (
    value.source.schema.fields.length *
    (value.source.numRows * 4 + Math.ceil(value.source.numRows / 8))
  );
}

function yieldMaterialization(): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, 0));
}
