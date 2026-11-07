// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  MAXIMUM_KEPLER_GROUP_BY_GROUPS,
  MAXIMUM_KEPLER_GROUP_BY_MEASURES,
  MAXIMUM_KEPLER_GROUP_BY_SOURCE_ROWS
} from './kepler-group-by';

/** One graph execution: a window of dictionary slots crossed with a batch of measures. */
export type KeplerGroupByPass = Readonly<{
  /** First dictionary slot of the window; the null group is slot dictionaryCount. */
  keyStart: number;
  keyCapacity: number;
  /** [measureStart, measureEnd) into the settings aggregations. */
  measureStart: number;
  measureEnd: number;
}>;

export type KeplerGroupByPassPlan = Readonly<{
  passes: readonly KeplerGroupByPass[];
  windowCount: number;
  batchCount: number;
}>;

export const KEPLER_GROUP_BY_PASS_WORKING_BYTES = 256 * 1024 * 1024;

/**
 * Splits a Group By into passes that each fit the per-graph limits and the working-memory budget.
 * keySlotCount includes the null slot. Windows partition the dictionary slots (a pass scans all
 * rows but only aggregates its window); batches partition the measures. Pass order is
 * window-major so one window's batches are adjacent.
 */
export function planKeplerGroupByPasses(props: {
  keySlotCount: number;
  measureCount: number;
  rowCount: number;
  workingBytes?: number;
  maximumSlotsPerPass?: number;
  maximumMeasuresPerPass?: number;
}): KeplerGroupByPassPlan {
  const {keySlotCount, measureCount, rowCount} = props;
  const maximumMeasures = props.maximumMeasuresPerPass ?? MAXIMUM_KEPLER_GROUP_BY_MEASURES;
  const maximumSlots = props.maximumSlotsPerPass ?? MAXIMUM_KEPLER_GROUP_BY_GROUPS;
  if (
    !Number.isSafeInteger(keySlotCount) ||
    keySlotCount < 1 ||
    !Number.isSafeInteger(rowCount) ||
    rowCount < 0
  )
    throw new Error('Group By pass planning requires nonnegative integer sizes');
  const batchSize = Math.max(1, Math.min(maximumMeasures, measureCount));
  const batchRanges: [number, number][] = [];
  for (let start = 0; start < Math.max(measureCount, 1); start += batchSize)
    batchRanges.push([start, Math.min(measureCount, start + batchSize)]);
  const widest = Math.min(batchSize, measureCount);
  // Mirrors getKeplerGroupByMemoryEstimate: row scratch is fixed, slot scratch scales with window.
  const rowBytes = Math.min(rowCount, MAXIMUM_KEPLER_GROUP_BY_SOURCE_ROWS) * (4 + widest * 4);
  const slotBytes = 64 + widest * 16 + 12 + widest * 8;
  const budget = props.workingBytes ?? KEPLER_GROUP_BY_PASS_WORKING_BYTES;
  const affordable = Math.floor((budget - rowBytes - 65536) / slotBytes);
  // A window never shrinks below 4096 slots; the memory reservation then decides admission.
  const windowSize = Math.max(1, Math.min(maximumSlots, Math.max(4096, affordable)));
  const passes: KeplerGroupByPass[] = [];
  let windowCount = 0;
  for (let start = 0; start < keySlotCount; start += windowSize) {
    windowCount++;
    for (const [measureStart, measureEnd] of batchRanges)
      passes.push({
        keyStart: start,
        keyCapacity: Math.min(windowSize, keySlotCount - start),
        measureStart,
        measureEnd
      });
  }
  return {passes, windowCount, batchCount: batchRanges.length};
}

export type KeplerGroupByMeasureColumn = Uint32Array | Float32Array;

/** Readback of one pass, truncated to its groupCount prefix. Keys are window-local. */
export type KeplerGroupByPassResult = {
  pass: KeplerGroupByPass;
  keys: Uint32Array;
  firstRows: Uint32Array;
  values: readonly KeplerGroupByMeasureColumn[];
  validity: readonly Uint32Array[];
};

export type KeplerGroupByMergedResult = {
  groupCount: number;
  /** Dictionary slot per group (global); dictionaryCount marks the null group. */
  slots: Uint32Array;
  firstRows: Uint32Array;
  values: KeplerGroupByMeasureColumn[];
  validity: Uint32Array[];
};

/**
 * Joins pass readbacks into one table in first-source-occurrence order. Batches of one window must
 * report identical groups; windows are disjoint and ordered by their first source row.
 */
export function mergeKeplerGroupByPasses(
  results: readonly KeplerGroupByPassResult[],
  measureCount: number,
  measureIsCount: readonly boolean[]
): KeplerGroupByMergedResult {
  const slotParts: Uint32Array[] = [];
  const rowParts: Uint32Array[] = [];
  const valueParts: KeplerGroupByMeasureColumn[][] = Array.from({length: measureCount}, () => []);
  const validityParts: Uint32Array[][] = Array.from({length: measureCount}, () => []);
  let groupCount = 0;
  let reference: KeplerGroupByPassResult | undefined;
  for (const result of results) {
    if (!reference || result.pass.keyStart !== reference.pass.keyStart) {
      reference = result;
      const slots = new Uint32Array(result.keys.length);
      for (let index = 0; index < slots.length; index++)
        slots[index] = result.keys[index] + result.pass.keyStart;
      slotParts.push(slots);
      rowParts.push(result.firstRows);
      groupCount += slots.length;
    } else if (
      result.keys.length !== reference.keys.length ||
      result.keys.some((key, index) => key !== reference!.keys[index]) ||
      result.firstRows.some((row, index) => row !== reference!.firstRows[index])
    )
      // Batches of one window disagree on the groups; the source changed or a kernel misbehaved.
      throw new Error('Group By passes disagree on group identity');
    for (let local = 0; local < result.values.length; local++) {
      const measure = result.pass.measureStart + local;
      valueParts[measure][slotParts.length - 1] = result.values[local];
      validityParts[measure][slotParts.length - 1] = result.validity[local];
    }
  }
  // k-way merge by first source row; every row has one key so first rows are unique.
  const order = new Uint32Array(groupCount);
  const partOf = new Uint32Array(groupCount);
  const indexOf = new Uint32Array(groupCount);
  let cursor = 0;
  for (const [part, rows] of rowParts.entries())
    for (let index = 0; index < rows.length; index++) {
      partOf[cursor] = part;
      indexOf[cursor] = index;
      order[cursor] = cursor++;
    }
  const sortedOrder = Array.from(order).sort(
    (first, second) =>
      rowParts[partOf[first]][indexOf[first]] - rowParts[partOf[second]][indexOf[second]]
  );
  const slots = new Uint32Array(groupCount);
  const firstRows = new Uint32Array(groupCount);
  for (const [target, source] of sortedOrder.entries()) {
    slots[target] = slotParts[partOf[source]][indexOf[source]];
    firstRows[target] = rowParts[partOf[source]][indexOf[source]];
  }
  const values: KeplerGroupByMeasureColumn[] = [];
  const validity: Uint32Array[] = [];
  for (let measure = 0; measure < measureCount; measure++) {
    const column = measureIsCount[measure]
      ? new Uint32Array(groupCount)
      : new Float32Array(groupCount);
    const valid = new Uint32Array(groupCount);
    for (const [target, source] of sortedOrder.entries()) {
      column[target] = valueParts[measure][partOf[source]][indexOf[source]];
      valid[target] = validityParts[measure][partOf[source]][indexOf[source]];
    }
    values.push(column);
    validity.push(valid);
  }
  return {groupCount, slots, firstRows, values, validity};
}
