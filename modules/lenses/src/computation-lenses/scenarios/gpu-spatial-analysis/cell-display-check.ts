// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationCapture} from '@luma-contributors/analysis/computation-record';

/** Refuse a complete-result claim when any pyramid level lost output or source cells. */
export function checkCellPyramidCompleteness(
  snapshots: readonly GPUComputationCapture[],
  levelCount: number
): void {
  for (let level = 0; level < levelCount; level++) {
    for (const flag of ['Overflow', 'CandidateOverflow']) {
      const name = `level${level}${flag}`;
      const snapshot = snapshots.find(capture => capture.name === name);
      if (!snapshot || Number(snapshot.values[0]) !== 0)
        throw new Error(`${name}: cell pyramid is incomplete or its status is unavailable`);
    }
  }
}

/** A count the GPU wrote and the number of table rows the card figure draws for it. */
export type DrawnPrefixCheck = {
  /** Capture name holding the GPU-written row count. */
  capture: string;
  /** Rows the figure's display offsets draw. */
  drawn: number;
  /** Element of the capture to compare, for per-level count buffers. */
  index?: number;
};

/**
 * Fails the scenario run when a cell table's GPU-written count differs from the occupied prefix
 * the figure draws (sized before the run from the input). The figure then always shows exactly
 * the rows the GPU produced; a mismatch would draw empty-key rows or hide real ones.
 */
export function checkDrawnPrefix(
  snapshots: readonly GPUComputationCapture[],
  checks: readonly DrawnPrefixCheck[]
): void {
  for (const {capture, drawn, index = 0} of checks) {
    const snapshot = snapshots.find(candidate => candidate.name === capture);
    const written = snapshot ? Number(snapshot.values[index]) : Number.NaN;
    if (written !== drawn) {
      throw new Error(
        `${capture}[${index}] is ${written} on the GPU but the figure draws ${drawn}`
      );
    }
  }
}
