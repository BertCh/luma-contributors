// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type GPUCommandGraph, type GraphDataView} from './gpu-command-graph';
import type {GPUCommandNode} from './gpu-command-node';
import {
  getBoundedDispatchLayout,
  getBoundedDispatchOverrides,
  getBoundedInvocationIndexOverrideSource
} from './gpu-dispatch-utils';
import {
  createGPUKernelNode,
  getGPUKernelBindingDeclarations,
  getGPUKernelOverrideDeclarations,
  type GPUKernelBinding
} from './gpu-kernel-node';
import {createTransientView, getViewElementOffset} from './graph-data-view-utils';

/** Threads per workgroup of the chained scan. */
const LOOKBACK_WORKGROUP_SIZE = 256;
/** Rows each thread owns, so one workgroup scans a tile of 1024 rows. */
const LOOKBACK_ITEMS_PER_THREAD = 4;
/** Rows scanned by one workgroup. */
export const LOOKBACK_TILE_SIZE = LOOKBACK_WORKGROUP_SIZE * LOOKBACK_ITEMS_PER_THREAD;
/** Published-state polls a workgroup makes on one predecessor before recomputing it. */
const DEFAULT_LOOKBACK_MAXIMUM_SPINS = 1024;
/** Words of published state per tile: aggregate, aggregate check, inclusive prefix, inclusive check. */
const STATE_WORDS_PER_TILE = 4;

/**
 * Adds a single-pass chained scan with decoupled look-back (Merrill and Garland) and Levien's
 * decoupled fallback.
 *
 * Workgroups claim tiles in dispatch order through an atomic ticket, publish their tile aggregate,
 * and walk backwards over predecessors until they find an inclusive prefix. A predecessor that has
 * not published within `maximumSpins` polls is reduced from its input directly and its aggregate
 * published on its behalf, so progress never depends on another workgroup being scheduled. Every
 * published value carries an XOR check word, so a reader accepts a value only when both words are
 * visible and no cross-location memory ordering is assumed.
 *
 * One clear dispatch zeroes the published state, then one scan dispatch covers any input size.
 * Arithmetic wraps modulo 2^32.
 */
export function addDecoupledLookbackScan<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    input: GraphDataView<'uint32'>;
    output: GraphDataView<'uint32'>;
    mode: 'exclusive' | 'inclusive';
    countNonzeroInput?: boolean;
    /**
     * Nonzero rows start new segments (row zero always does). The scan then runs over
     * (head flag, value) pairs with the segmented monoid, and a tile's published state carries its
     * head flag inside the check word. A single chunk only: vector carries use the multi-pass scan.
     */
    segmentFlags?: GraphDataView<'uint32'>;
    finalSum?: GraphDataView<'uint32'>;
    /** Polls before the fallback recomputes a predecessor; zero always recomputes. @internal */
    maximumSpins?: number;
    maxComputeWorkgroupsPerDimension: number;
  }
): readonly GPUCommandNode<Parameters>[] {
  const length = props.input.length;
  const tileCount = Math.ceil(length / LOOKBACK_TILE_SIZE);
  const stateLength = tileCount * STATE_WORDS_PER_TILE + 1;
  const state = createTransientView(graph, `${props.id}-lookback-state`, 'uint32', stateLength);

  const clearBindings: GPUKernelBinding[] = [{name: 'stateValues', view: state, access: 'write'}];
  const clearLayout = getBoundedDispatchLayout(
    'GPUScan lookback clear',
    stateLength,
    LOOKBACK_WORKGROUP_SIZE,
    props.maxComputeWorkgroupsPerDimension
  );
  const clearConstants = {
    STATE_LENGTH: stateLength,
    STATE_OFFSET: getViewElementOffset(state),
    ...getBoundedDispatchOverrides(clearLayout)
  };
  const clearNode = createGPUKernelNode(graph, {
    id: `${props.id}-clear`,
    operation: 'GPUScan',
    variant: 'lookback-clear',
    moduleId: 'GPUScan/lookback-clear',
    dedupName: 'lookback-clear',
    formula: 'gpu-core.scan',
    source: /* wgsl */ `
${getGPUKernelOverrideDeclarations(clearConstants)}
${getGPUKernelBindingDeclarations(clearBindings)}
@compute @workgroup_size(${LOOKBACK_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexOverrideSource(LOOKBACK_WORKGROUP_SIZE).body}
  if (index < STATE_LENGTH) { stateValues[STATE_OFFSET + index] = 0u; }
}`,
    constants: clearConstants,
    bindings: clearBindings,
    workgroupSize: LOOKBACK_WORKGROUP_SIZE,
    dispatchLayout: clearLayout,
    invocationCount: stateLength
  });

  const layout = getBoundedDispatchLayout(
    'GPUScan lookback',
    length,
    LOOKBACK_TILE_SIZE,
    props.maxComputeWorkgroupsPerDimension
  );
  const bindings: (GPUKernelBinding | undefined)[] = [
    {name: 'inputValues', view: props.input, access: 'read'},
    props.segmentFlags && {name: 'segmentFlags', view: props.segmentFlags, access: 'read'},
    {name: 'outputValues', view: props.output, access: 'write'},
    {name: 'state', view: state, access: 'read_write', atomic: true}
  ];
  if (props.finalSum) bindings.push({name: 'sumValues', view: props.finalSum, access: 'write'});
  const segmented = Boolean(props.segmentFlags);
  const constants: Record<string, number> = {
    ELEMENT_COUNT: length,
    TILE_COUNT: tileCount,
    INPUT_OFFSET: getViewElementOffset(props.input),
    OUTPUT_OFFSET: getViewElementOffset(props.output),
    ...(props.segmentFlags && {FLAGS_OFFSET: getViewElementOffset(props.segmentFlags)}),
    STATE_OFFSET: getViewElementOffset(state),
    MAXIMUM_SPINS: props.maximumSpins ?? DEFAULT_LOOKBACK_MAXIMUM_SPINS,
    ...(props.finalSum && {SUM_OFFSET: getViewElementOffset(props.finalSum)}),
    ...getBoundedDispatchOverrides(layout)
  };
  const readElement = props.countNonzeroInput
    ? 'min(inputValues[INPUT_OFFSET + index], 1u)'
    : 'inputValues[INPUT_OFFSET + index]';
  const source = segmented
    ? getSegmentedLookbackSource(constants, bindings, readElement, props.mode)
    : /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}

const THREADS: u32 = ${LOOKBACK_WORKGROUP_SIZE}u;
const ITEMS: u32 = ${LOOKBACK_ITEMS_PER_THREAD}u;
const TILE: u32 = ${LOOKBACK_TILE_SIZE}u;
const AGGREGATE_CHECK: u32 = 0x9e3779b9u;
const INCLUSIVE_CHECK: u32 = 0x85ebca6bu;

var<workgroup> tileData: array<u32, ${LOOKBACK_TILE_SIZE}>;
var<workgroup> threadSums: array<u32, ${LOOKBACK_WORKGROUP_SIZE}>;
var<workgroup> sharedTile: u32;
var<workgroup> sharedPrefix: u32;

fn readElement(index: u32) -> u32 {
  if (index >= ELEMENT_COUNT) { return 0u; }
  return ${readElement};
}

// Recomputes one predecessor's aggregate straight from the input (the decoupled fallback).
fn reduceTile(tile: u32) -> u32 {
  var total = 0u;
  let tileBase = tile * TILE;
  for (var offset = 0u; offset < TILE; offset++) {
    total += readElement(tileBase + offset);
  }
  return total;
}

fn publishAggregate(tile: u32, aggregate: u32) {
  let base = STATE_OFFSET + tile * ${STATE_WORDS_PER_TILE}u;
  atomicStore(&state[base], aggregate);
  atomicStore(&state[base + 1u], aggregate ^ AGGREGATE_CHECK);
}

fn publishInclusive(tile: u32, inclusive: u32) {
  let base = STATE_OFFSET + tile * ${STATE_WORDS_PER_TILE}u;
  atomicStore(&state[base + 2u], inclusive);
  atomicStore(&state[base + 3u], inclusive ^ INCLUSIVE_CHECK);
}

// Runs on one thread: publishes this tile and returns the sum of every earlier tile.
fn getTilePrefix(tile: u32, tileTotal: u32) -> u32 {
  if (tile == 0u) {
    publishInclusive(0u, tileTotal);
    return 0u;
  }
  publishAggregate(tile, tileTotal);
  var prefix = 0u;
  var look = tile;
  loop {
    look -= 1u;
    let lookBase = STATE_OFFSET + look * ${STATE_WORDS_PER_TILE}u;
    var found = false;
    var foundInclusive = false;
    var value = 0u;
    for (var spin = 0u; spin < MAXIMUM_SPINS; spin++) {
      let inclusive = atomicLoad(&state[lookBase + 2u]);
      if (atomicLoad(&state[lookBase + 3u]) == (inclusive ^ INCLUSIVE_CHECK)) {
        value = inclusive;
        found = true;
        foundInclusive = true;
        break;
      }
      let aggregate = atomicLoad(&state[lookBase]);
      if (atomicLoad(&state[lookBase + 1u]) == (aggregate ^ AGGREGATE_CHECK)) {
        value = aggregate;
        found = true;
        break;
      }
    }
    if (!found) {
      value = reduceTile(look);
      publishAggregate(look, value);
    }
    prefix += value;
    if (foundInclusive || look == 0u) { break; }
  }
  publishInclusive(tile, prefix + tileTotal);
  return prefix;
}

@compute @workgroup_size(${LOOKBACK_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let linearWorkgroup = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  if (linearWorkgroup >= TILE_COUNT) { return; }
  // Tiles are claimed in the order workgroups start, so every earlier tile is already running.
  if (lane == 0u) { sharedTile = atomicAdd(&state[STATE_OFFSET + TILE_COUNT * ${STATE_WORDS_PER_TILE}u], 1u); }
  workgroupBarrier();
  let tile = sharedTile;
  let tileBase = tile * TILE;

  for (var item = 0u; item < ITEMS; item++) {
    let local = item * THREADS + lane;
    tileData[local] = readElement(tileBase + local);
  }
  workgroupBarrier();

  var values: array<u32, ${LOOKBACK_ITEMS_PER_THREAD}>;
  var threadTotal = 0u;
  for (var item = 0u; item < ITEMS; item++) {
    values[item] = tileData[lane * ITEMS + item];
    threadTotal += values[item];
  }
  threadSums[lane] = threadTotal;
  workgroupBarrier();
  for (var stride = 1u; stride < THREADS; stride = stride * 2u) {
    var addend = 0u;
    if (lane >= stride) { addend = threadSums[lane - stride]; }
    workgroupBarrier();
    if (lane >= stride) { threadSums[lane] = threadSums[lane] + addend; }
    workgroupBarrier();
  }
  let threadInclusive = threadSums[lane];
  let tileTotal = threadSums[THREADS - 1u];

  if (lane == 0u) { sharedPrefix = getTilePrefix(tile, tileTotal); }
  workgroupBarrier();
  let tilePrefix = sharedPrefix;

  var running = tilePrefix + threadInclusive - threadTotal;
  for (var item = 0u; item < ITEMS; item++) {
    let inclusiveValue = running + values[item];
    tileData[lane * ITEMS + item] = ${props.mode === 'inclusive' ? 'inclusiveValue' : 'running'};
    running = inclusiveValue;
  }
  workgroupBarrier();

  for (var item = 0u; item < ITEMS; item++) {
    let local = item * THREADS + lane;
    if (tileBase + local < ELEMENT_COUNT) {
      outputValues[OUTPUT_OFFSET + tileBase + local] = tileData[local];
    }
  }
  ${props.finalSum ? `if (tile == TILE_COUNT - 1u && lane == THREADS - 1u) { sumValues[SUM_OFFSET] = tilePrefix + tileTotal; }` : ''}
}`;
  const scanNode = createGPUKernelNode(graph, {
    id: `${props.id}-scan`,
    operation: 'GPUScan',
    variant: 'decoupled-lookback',
    moduleId: 'GPUScan/decoupled-lookback',
    dedupName: 'decoupled-lookback',
    formula: 'gpu-core.scan',
    source,
    constants,
    bindings,
    workgroupSize: LOOKBACK_WORKGROUP_SIZE,
    dispatchLayout: layout,
    invocationCount: length,
    workload: {
      readByteLength: length * Uint32Array.BYTES_PER_ELEMENT,
      writeByteLength: (length + 1) * Uint32Array.BYTES_PER_ELEMENT
    }
  });
  return [clearNode, scanNode];
}

/**
 * Segmented variant of the look-back kernel. Values pair with a segment-head flag and combine with
 * `(lf, lv) + (rf, rv) = (lf | rf, rf ? rv : lv + rv)`, which is associative, so tiles, threads, and
 * look-back steps compose exactly like the unsegmented sums.
 *
 * A tile publishes its (flag, aggregate) pair and later its inclusive value. The flag travels in
 * the check word: a reader accepts `value ^ CHECK` as a flag-free aggregate and
 * `value ^ CHECK ^ HEAD_CHECK` as an aggregate holding a head; cleared state matches neither.
 * Row zero is always a head, so every inclusive prefix is flagged and a look-back stops at the first
 * published inclusive value or at the first tile whose aggregate holds a head.
 */
function getSegmentedLookbackSource(
  constants: Readonly<Record<string, number>>,
  bindings: readonly (GPUKernelBinding | false | undefined)[],
  readElement: string,
  mode: 'exclusive' | 'inclusive'
): string {
  return /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}

const THREADS: u32 = ${LOOKBACK_WORKGROUP_SIZE}u;
const ITEMS: u32 = ${LOOKBACK_ITEMS_PER_THREAD}u;
const TILE: u32 = ${LOOKBACK_TILE_SIZE}u;
const AGGREGATE_CHECK: u32 = 0x9e3779b9u;
const INCLUSIVE_CHECK: u32 = 0x85ebca6bu;
const HEAD_CHECK: u32 = 0x27d4eb2fu;

var<workgroup> tileValues: array<u32, ${LOOKBACK_TILE_SIZE}>;
var<workgroup> tileHeads: array<u32, ${LOOKBACK_TILE_SIZE}>;
var<workgroup> threadValues: array<u32, ${LOOKBACK_WORKGROUP_SIZE}>;
var<workgroup> threadHeads: array<u32, ${LOOKBACK_WORKGROUP_SIZE}>;
var<workgroup> sharedTile: u32;
var<workgroup> sharedPrefix: u32;

fn readElement(index: u32) -> u32 {
  if (index >= ELEMENT_COUNT) { return 0u; }
  return ${readElement};
}

fn readHead(index: u32) -> u32 {
  if (index >= ELEMENT_COUNT) { return 0u; }
  return select(min(segmentFlags[FLAGS_OFFSET + index], 1u), 1u, index == 0u);
}

// Recomputes one predecessor's (head flag, aggregate) straight from the input (the fallback).
fn reduceTile(tile: u32) -> vec2<u32> {
  var head = 0u;
  var total = 0u;
  let tileBase = tile * TILE;
  for (var offset = 0u; offset < TILE; offset++) {
    let index = tileBase + offset;
    if (readHead(index) != 0u) {
      head = 1u;
      total = readElement(index);
    } else {
      total += readElement(index);
    }
  }
  return vec2<u32>(head, total);
}

fn publishAggregate(tile: u32, head: u32, aggregate: u32) {
  let base = STATE_OFFSET + tile * ${STATE_WORDS_PER_TILE}u;
  atomicStore(&state[base], aggregate);
  atomicStore(&state[base + 1u], aggregate ^ AGGREGATE_CHECK ^ select(0u, HEAD_CHECK, head != 0u));
}

fn publishInclusive(tile: u32, inclusive: u32) {
  let base = STATE_OFFSET + tile * ${STATE_WORDS_PER_TILE}u;
  atomicStore(&state[base + 2u], inclusive);
  atomicStore(&state[base + 3u], inclusive ^ INCLUSIVE_CHECK);
}

// Runs on one thread: publishes this tile and returns the segmented carry into it.
fn getTilePrefix(tile: u32, tileHead: u32, tileTotal: u32) -> u32 {
  if (tile == 0u) {
    publishInclusive(0u, tileTotal);
    return 0u;
  }
  publishAggregate(tile, tileHead, tileTotal);
  var prefix = 0u;
  var look = tile;
  loop {
    look -= 1u;
    let lookBase = STATE_OFFSET + look * ${STATE_WORDS_PER_TILE}u;
    var found = false;
    var foundInclusive = false;
    var foundHead = 0u;
    var value = 0u;
    for (var spin = 0u; spin < MAXIMUM_SPINS; spin++) {
      let inclusive = atomicLoad(&state[lookBase + 2u]);
      if (atomicLoad(&state[lookBase + 3u]) == (inclusive ^ INCLUSIVE_CHECK)) {
        value = inclusive;
        found = true;
        foundInclusive = true;
        break;
      }
      let aggregate = atomicLoad(&state[lookBase]);
      let check = atomicLoad(&state[lookBase + 1u]);
      if (check == (aggregate ^ AGGREGATE_CHECK)) {
        value = aggregate;
        found = true;
        break;
      }
      if (check == (aggregate ^ AGGREGATE_CHECK ^ HEAD_CHECK)) {
        value = aggregate;
        foundHead = 1u;
        found = true;
        break;
      }
    }
    if (!found) {
      let recomputed = reduceTile(look);
      foundHead = recomputed.x;
      value = recomputed.y;
      publishAggregate(look, foundHead, value);
    }
    prefix += value;
    if (foundInclusive || foundHead != 0u) { break; }
  }
  publishInclusive(tile, select(prefix + tileTotal, tileTotal, tileHead != 0u));
  return prefix;
}

@compute @workgroup_size(${LOOKBACK_WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let linearWorkgroup = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  if (linearWorkgroup >= TILE_COUNT) { return; }
  // Tiles are claimed in the order workgroups start, so every earlier tile is already running.
  if (lane == 0u) { sharedTile = atomicAdd(&state[STATE_OFFSET + TILE_COUNT * ${STATE_WORDS_PER_TILE}u], 1u); }
  workgroupBarrier();
  let tile = sharedTile;
  let tileBase = tile * TILE;

  for (var item = 0u; item < ITEMS; item++) {
    let local = item * THREADS + lane;
    tileValues[local] = readElement(tileBase + local);
    tileHeads[local] = readHead(tileBase + local);
  }
  workgroupBarrier();

  var values: array<u32, ${LOOKBACK_ITEMS_PER_THREAD}>;
  var heads: array<u32, ${LOOKBACK_ITEMS_PER_THREAD}>;
  var threadHead = 0u;
  var threadTotal = 0u;
  for (var item = 0u; item < ITEMS; item++) {
    values[item] = tileValues[lane * ITEMS + item];
    heads[item] = tileHeads[lane * ITEMS + item];
    if (heads[item] != 0u) {
      threadHead = 1u;
      threadTotal = values[item];
    } else {
      threadTotal += values[item];
    }
  }
  threadValues[lane] = threadTotal;
  threadHeads[lane] = threadHead;
  workgroupBarrier();
  for (var stride = 1u; stride < THREADS; stride = stride * 2u) {
    var leftValue = 0u;
    var leftHead = 0u;
    if (lane >= stride) {
      leftValue = threadValues[lane - stride];
      leftHead = threadHeads[lane - stride];
    }
    workgroupBarrier();
    if (lane >= stride) {
      threadValues[lane] = select(leftValue + threadValues[lane], threadValues[lane], threadHeads[lane] != 0u);
      threadHeads[lane] = leftHead | threadHeads[lane];
    }
    workgroupBarrier();
  }
  let tileHead = threadHeads[THREADS - 1u];
  let tileTotal = threadValues[THREADS - 1u];

  if (lane == 0u) { sharedPrefix = getTilePrefix(tile, tileHead, tileTotal); }
  workgroupBarrier();
  let tilePrefix = sharedPrefix;

  // The carry entering this thread: earlier threads' segmented total on top of the tile carry.
  var carry = tilePrefix;
  if (lane > 0u) {
    carry = select(tilePrefix + threadValues[lane - 1u], threadValues[lane - 1u], threadHeads[lane - 1u] != 0u);
  }
  for (var item = 0u; item < ITEMS; item++) {
    let previous = carry;
    carry = select(carry + values[item], values[item], heads[item] != 0u);
    tileValues[lane * ITEMS + item] = ${mode === 'inclusive' ? 'carry' : 'select(previous, 0u, heads[item] != 0u)'};
  }
  workgroupBarrier();

  for (var item = 0u; item < ITEMS; item++) {
    let local = item * THREADS + lane;
    if (tileBase + local < ELEMENT_COUNT) {
      outputValues[OUTPUT_OFFSET + tileBase + local] = tileValues[local];
    }
  }
}`;
}
