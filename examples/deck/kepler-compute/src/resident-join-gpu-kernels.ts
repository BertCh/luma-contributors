// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {Kernel} from '@luma-contributors/graph';
import {
  getBoundedDispatchLayout,
  getBoundedInvocationIndexSource,
  type GPUCommandGraph,
  type GraphDataView
} from '@luma-contributors/graph/gpu-core';
import {RESIDENT_JOIN_NO_KEY, RESIDENT_JOIN_NO_ROW} from './resident-join-plan';
import type {ResidentJoinKeyMode} from './resident-join-gpu-reference';

/**
 * Small WGSL passes for the GPU Join. Sorting, scans and gathers reuse @luma.gl/graph primitives
 * (GPUSort, GPUScan, GPUScanUint64, GPUGather); only key canonicalization, the duplicate-aware
 * range probe and the output-parallel pair fill are new, because GPUHashJoin is a unique-key,
 * capacity-truncating inner join. Every pass is data-parallel over rows; no pass loops over
 * duplicate keys, so many-to-many skew does not unbalance threads.
 */
const WORKGROUP_SIZE = 256;
/** `summary` words: left output total, unmatched right total, status bits, valid right row count. */
export const RESIDENT_JOIN_SUMMARY_WORDS = 4;

export type ResidentJoinBindingAccess = 'read' | 'write' | 'read-write' | 'atomic';
export type ResidentJoinPassBindings = Record<
  string,
  {view: GraphDataView<'uint32'>; access: ResidentJoinBindingAccess}
>;

const literal = (value: number) => `${value}u`;

/** Adds one compute pass over `count` invocations; `body` sees `index` and the named bindings. */
export function addResidentJoinPass(
  graph: GPUCommandGraph<undefined>,
  props: {
    id: string;
    count: number;
    constants?: string;
    bindings: ResidentJoinPassBindings;
    body: string;
  }
): void {
  const names = Object.keys(props.bindings);
  const declarations = names
    .map((name, location) => {
      const {access} = props.bindings[name];
      return `@group(0) @binding(${location}) var<storage, ${access === 'read' ? 'read' : 'read_write'}> ${name}: array<${access === 'atomic' ? 'atomic<u32>' : 'u32'}>;`;
    })
    .join('\n');
  const dispatch = (device: Device) =>
    getBoundedDispatchLayout(
      props.id,
      props.count,
      WORKGROUP_SIZE,
      device.limits.maxComputeWorkgroupsPerDimension
    );
  graph.addComputePass({
    id: props.id,
    resources: names.map(name => ({
      buffer: props.bindings[name].view,
      usage:
        props.bindings[name].access === 'read'
          ? ('storage-read' as const)
          : props.bindings[name].access === 'write'
            ? ('storage-write' as const)
            : ('storage-read-write' as const)
    })),
    compile: ({device}) => {
      const layout = dispatch(device);
      const kernel = new Kernel(device, {
        id: props.id,
        source: `
const NO_ROW: u32 = ${literal(RESIDENT_JOIN_NO_ROW)};
const NO_KEY: u32 = ${literal(RESIDENT_JOIN_NO_KEY)};
const COUNT: u32 = ${literal(props.count)};
${props.constants ?? ''}
${declarations}
@compute @workgroup_size(${WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexSource(layout, WORKGROUP_SIZE)}
  if (index >= COUNT) { return; }
${props.body}
}`,
        shaderLayout: {
          bindings: names.map((name, location) => ({
            name,
            group: 0,
            location,
            type: 'storage' as const
          }))
        }
      });
      return {
        encode: ({computePass, getBuffer}) =>
          kernel.dispatch(computePass, {
            bindings: Object.fromEntries(
              names.map(name => [name, getBuffer(props.bindings[name].view.buffer)])
            ),
            ...layout
          }),
        destroy: () => kernel.destroy()
      };
    }
  });
}

/**
 * Canonical key per row. String keys map dictionary codes through a dictionary-sized table;
 * numeric keys use the int32 bit pattern (Uint32 high values are dropped when they would alias
 * negative Int32 patterns). Writes `ok` (row participates in matching) and, for the right side,
 * `notOk` and an identity row ID vector that feed the two stable sorts. Status bits go to
 * `summary[2]`: 1 non-integral Float32 key, 2 invalid dictionary code.
 */
export function addResidentJoinKeyPass(
  graph: GPUCommandGraph<undefined>,
  props: {
    id: string;
    count: number;
    mode: ResidentJoinKeyMode;
    dropHighKeys: boolean;
    dictionarySize: number;
    words: GraphDataView<'uint32'>;
    validity?: GraphDataView<'uint32'>;
    table?: GraphDataView<'uint32'>;
    key: GraphDataView<'uint32'>;
    ok: GraphDataView<'uint32'>;
    notOk?: GraphDataView<'uint32'>;
    rowIds?: GraphDataView<'uint32'>;
    summary: GraphDataView<'uint32'>;
  }
): void {
  let decode: string;
  if (props.mode === 'string')
    decode = `
    if (word >= ${literal(props.dictionarySize)}) { atomicOr(&summary[2], 2u); }
    else { keyValue = table[word]; okValue = select(0u, 1u, keyValue != NO_KEY); }`;
  else if (props.mode === 'float32')
    decode = `
    let value = bitcast<f32>(word);
    if (abs(value) <= 16777216.0 && value == floor(value)) { keyValue = bitcast<u32>(i32(value)); okValue = 1u; }
    else { atomicOr(&summary[2], 1u); }`;
  else
    decode = `
    keyValue = word;
    okValue = ${props.mode === 'uint32' && props.dropHighKeys ? 'select(1u, 0u, word >= 0x80000000u)' : '1u'};`;
  addResidentJoinPass(graph, {
    id: props.id,
    count: props.count,
    bindings: {
      words: {view: props.words, access: 'read'},
      ...(props.validity ? {validity: {view: props.validity, access: 'read'}} : {}),
      ...(props.table ? {table: {view: props.table, access: 'read'}} : {}),
      key: {view: props.key, access: 'write'},
      ok: {view: props.ok, access: 'write'},
      ...(props.notOk ? {notOk: {view: props.notOk, access: 'write'}} : {}),
      ...(props.rowIds ? {rowIds: {view: props.rowIds, access: 'write'}} : {}),
      summary: {view: props.summary, access: 'atomic'}
    },
    body: `
  // Static use keeps summary in the pipeline layout for integer keys, which never write it.
  let summaryWord = atomicLoad(&summary[3]);
  let word = words[index] + (summaryWord & 0u);
  var keyValue = 0u;
  var okValue = 0u;
  if (${props.validity ? 'validity[index] != 0u' : 'true'}) {${decode}
  }
  key[index] = keyValue;
  ok[index] = okValue;${props.notOk ? '\n  notOk[index] = 1u - okValue;' : ''}${props.rowIds ? '\n  rowIds[index] = index;' : ''}`
  });
}

/** Run flags and sorted keys over right rows ordered valid-first; also clears the match marks. */
export function addResidentJoinRunPass(
  graph: GPUCommandGraph<undefined>,
  props: {
    count: number;
    sortedRows: GraphDataView<'uint32'>;
    rightKey: GraphDataView<'uint32'>;
    rightOk: GraphDataView<'uint32'>;
    sortedKeys: GraphDataView<'uint32'>;
    validAt: GraphDataView<'uint32'>;
    runStart: GraphDataView<'uint32'>;
    matchedRun: GraphDataView<'uint32'>;
  }
): void {
  addResidentJoinPass(graph, {
    id: 'join-right-runs',
    count: props.count,
    bindings: {
      sortedRows: {view: props.sortedRows, access: 'read'},
      rightKey: {view: props.rightKey, access: 'read'},
      rightOk: {view: props.rightOk, access: 'read'},
      sortedKeys: {view: props.sortedKeys, access: 'write'},
      validAt: {view: props.validAt, access: 'write'},
      runStart: {view: props.runStart, access: 'write'},
      matchedRun: {view: props.matchedRun, access: 'write'}
    },
    body: `
  let row = sortedRows[index];
  let isValid = rightOk[row] != 0u;
  let key = rightKey[row];
  var start = 0u;
  if (isValid) {
    start = 1u;
    // Valid rows are a prefix of the order, so the previous position is valid too.
    if (index > 0u && rightKey[sortedRows[index - 1u]] == key) { start = 0u; }
  }
  sortedKeys[index] = key;
  validAt[index] = select(0u, 1u, isValid);
  runStart[index] = start;
  matchedRun[index] = 0u;`
  });
}

/**
 * Duplicate-aware probe: lower and upper bound of each left key among the valid sorted right
 * keys. Writes first sorted position, match count and emitted row count (unmatched left rows
 * emit one row unless INNER), and marks matched right key runs.
 */
export function addResidentJoinProbePass(
  graph: GPUCommandGraph<undefined>,
  props: {
    count: number;
    rightCount: number;
    keepUnmatchedLeft: boolean;
    /** Which duplicate right rows a left row joins: all, only the first or only the last. */
    duplicateRightMatches?: 'all' | 'first' | 'last';
    leftKey: GraphDataView<'uint32'>;
    leftOk: GraphDataView<'uint32'>;
    sortedKeys: GraphDataView<'uint32'>;
    validIncluded: GraphDataView<'uint32'>;
    runIncluded: GraphDataView<'uint32'>;
    lowerBound: GraphDataView<'uint32'>;
    matches: GraphDataView<'uint32'>;
    emit: GraphDataView<'uint32'>;
    matchedRun: GraphDataView<'uint32'>;
  }
): void {
  addResidentJoinPass(graph, {
    id: 'join-probe',
    count: props.count,
    bindings: {
      leftKey: {view: props.leftKey, access: 'read'},
      leftOk: {view: props.leftOk, access: 'read'},
      sortedKeys: {view: props.sortedKeys, access: 'read'},
      validIncluded: {view: props.validIncluded, access: 'read'},
      lowerBound: {view: props.lowerBound, access: 'write'},
      matches: {view: props.matches, access: 'write'},
      emit: {view: props.emit, access: 'write'}
    },
    body: `
  let validCount = validIncluded[${literal(props.rightCount - 1)}];
  var lower = 0u;
  var found = 0u;
  if (leftOk[index] != 0u && validCount > 0u) {
    let key = leftKey[index];
    var high = validCount;
    while (lower < high) {
      let middle = lower + (high - lower) / 2u;
      if (sortedKeys[middle] < key) { lower = middle + 1u; } else { high = middle; }
    }
    var upper = lower;
    high = validCount;
    while (upper < high) {
      let middle = upper + (high - upper) / 2u;
      if (sortedKeys[middle] <= key) { upper = middle + 1u; } else { high = middle; }
    }
    found = upper - lower;
    ${
      props.duplicateRightMatches === 'first'
        ? 'found = min(found, 1u);'
        : props.duplicateRightMatches === 'last'
          ? 'lower = upper - 1u; found = 1u;'
          : ''
    }
  }
  lowerBound[index] = lower;
  matches[index] = found;
  emit[index] = select(${props.keepUnmatchedLeft ? '1u' : '0u'}, found, found > 0u);`
  });
  // Separate pass: the default per-stage limit is 8 storage buffers and the probe already uses 7.
  addResidentJoinPass(graph, {
    id: 'join-mark-matched-runs',
    count: props.count,
    bindings: {
      lowerBound: {view: props.lowerBound, access: 'read'},
      matches: {view: props.matches, access: 'read'},
      runIncluded: {view: props.runIncluded, access: 'read'},
      matchedRun: {view: props.matchedRun, access: 'atomic'}
    },
    body: `
  if (matches[index] > 0u) { atomicStore(&matchedRun[runIncluded[lowerBound[index]] - 1u], 1u); }`
  });
}

/** Marks right rows (in source order) that FULL must append because no left row matched them. */
export function addResidentJoinUnmatchedPass(
  graph: GPUCommandGraph<undefined>,
  props: {
    count: number;
    emitUnmatchedRight: boolean;
    sortedRows: GraphDataView<'uint32'>;
    validIncluded: GraphDataView<'uint32'>;
    runIncluded: GraphDataView<'uint32'>;
    matchedRun: GraphDataView<'uint32'>;
    unmatched: GraphDataView<'uint32'>;
  }
): void {
  addResidentJoinPass(graph, {
    id: 'join-unmatched-right',
    count: props.count,
    bindings: {
      sortedRows: {view: props.sortedRows, access: 'read'},
      validIncluded: {view: props.validIncluded, access: 'read'},
      runIncluded: {view: props.runIncluded, access: 'read'},
      matchedRun: {view: props.matchedRun, access: 'read'},
      unmatched: {view: props.unmatched, access: 'write'}
    },
    body: `
  var matched = 0u;
  if (index < validIncluded[${literal(props.count - 1)}]) { matched = matchedRun[runIncluded[index] - 1u]; }
  unmatched[sortedRows[index]] = ${props.emitUnmatchedRight ? 'select(1u, 0u, matched != 0u)' : '0u'};`
  });
}

/**
 * One-invocation summary read back by the CPU: totals from the inclusive scans, valid right row
 * count, and a 64-bit overflow check (the high word of the left total must be zero).
 */
export function addResidentJoinSummaryPass(
  graph: GPUCommandGraph<undefined>,
  props: {
    leftCount: number;
    rightCount: number;
    leftIncludedLow: GraphDataView<'uint32'>;
    leftIncludedHigh: GraphDataView<'uint32'>;
    unmatchedIncluded: GraphDataView<'uint32'>;
    validIncluded: GraphDataView<'uint32'>;
    summary: GraphDataView<'uint32'>;
  }
): void {
  addResidentJoinPass(graph, {
    id: 'join-summary',
    count: 1,
    bindings: {
      leftIncludedLow: {view: props.leftIncludedLow, access: 'read'},
      leftIncludedHigh: {view: props.leftIncludedHigh, access: 'read'},
      unmatchedIncluded: {view: props.unmatchedIncluded, access: 'read'},
      validIncluded: {view: props.validIncluded, access: 'read'},
      summary: {view: props.summary, access: 'atomic'}
    },
    body: `
  atomicStore(&summary[0], leftIncludedLow[${literal(props.leftCount - 1)}]);
  atomicStore(&summary[1], unmatchedIncluded[${literal(props.rightCount - 1)}]);
  atomicStore(&summary[3], validIncluded[${literal(props.rightCount - 1)}]);
  if (leftIncludedHigh[${literal(props.leftCount - 1)}] != 0u) { atomicOr(&summary[2], 4u); }`
  });
}

/**
 * Output-parallel pair fill: output row `index` finds its owning left row (or unmatched right row)
 * by binary search over the inclusive offsets, so skewed duplicate keys cost the same per row.
 * Also writes per-side validity (row present) used for payload columns without own validity.
 */
export function addResidentJoinFillPass(
  graph: GPUCommandGraph<undefined>,
  props: {
    count: number;
    leftCount: number;
    rightCount: number;
    leftTotal: number;
    leftIncludedLow: GraphDataView<'uint32'>;
    emit: GraphDataView<'uint32'>;
    matches: GraphDataView<'uint32'>;
    lowerBound: GraphDataView<'uint32'>;
    sortedRows: GraphDataView<'uint32'>;
    unmatchedIncluded: GraphDataView<'uint32'>;
    leftRows: GraphDataView<'uint32'>;
    rightRows: GraphDataView<'uint32'>;
    leftValid: GraphDataView<'uint32'>;
    rightValid: GraphDataView<'uint32'>;
  }
): void {
  addResidentJoinPass(graph, {
    id: 'join-fill',
    count: props.count,
    bindings: {
      leftIncludedLow: {view: props.leftIncludedLow, access: 'read'},
      emit: {view: props.emit, access: 'read'},
      matches: {view: props.matches, access: 'read'},
      lowerBound: {view: props.lowerBound, access: 'read'},
      sortedRows: {view: props.sortedRows, access: 'read'},
      unmatchedIncluded: {view: props.unmatchedIncluded, access: 'read'},
      leftRows: {view: props.leftRows, access: 'write'},
      rightRows: {view: props.rightRows, access: 'write'}
    },
    body: `
  var leftRow = NO_ROW;
  var rightRow = NO_ROW;
  if (index < ${literal(props.leftTotal)}) {
    var low = 0u;
    var high = ${literal(props.leftCount)};
    while (low < high) {
      let middle = low + (high - low) / 2u;
      if (leftIncludedLow[middle] > index) { high = middle; } else { low = middle + 1u; }
    }
    leftRow = low;
    if (matches[low] > 0u) {
      rightRow = sortedRows[lowerBound[low] + (index - (leftIncludedLow[low] - emit[low]))];
    }
  } else {
    let unmatchedOrdinal = index - ${literal(props.leftTotal)};
    var low = 0u;
    var high = ${literal(props.rightCount)};
    while (low < high) {
      let middle = low + (high - low) / 2u;
      if (unmatchedIncluded[middle] > unmatchedOrdinal) { high = middle; } else { low = middle + 1u; }
    }
    rightRow = low;
  }
  leftRows[index] = leftRow;
  rightRows[index] = rightRow;`
  });
  // Separate pass keeps every kernel within the default 8 storage buffers per shader stage.
  addResidentJoinPass(graph, {
    id: 'join-fill-validity',
    count: props.count,
    bindings: {
      leftRows: {view: props.leftRows, access: 'read'},
      rightRows: {view: props.rightRows, access: 'read'},
      leftValid: {view: props.leftValid, access: 'write'},
      rightValid: {view: props.rightValid, access: 'write'}
    },
    body: `
  leftValid[index] = select(1u, 0u, leftRows[index] == NO_ROW);
  rightValid[index] = select(1u, 0u, rightRows[index] == NO_ROW);`
  });
}
