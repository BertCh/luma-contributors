// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUComputationLoopCoordinate} from '../computation-record/computation-record-types';
import type {GPUCommandGraph, GraphDataView} from './gpu-command-graph';
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

const WORKGROUP_SIZE = 256;
const MASK_WORD_COUNT = WORKGROUP_SIZE / 32;
/** Published-state polls per predecessor digit before the workgroup recomputes that tile. */
const DEFAULT_MAXIMUM_SPINS = 1024;
const STATE_WORDS_PER_DIGIT = 4;

/** Inputs shared by every Onesweep kernel of one sort. @internal */
export type OnesweepSortProps = {
  id: string;
  keys: GraphDataView<'uint32'>;
  values: GraphDataView<'uint32'>;
  outputKeys: GraphDataView<'uint32'>;
  outputValues: GraphDataView<'uint32'>;
  direction: 'ascending' | 'descending';
  keyBits: number;
  digitBits: number;
  elementsPerThread: number;
  /** Polling budget per predecessor digit; zero always recomputes (tests only). */
  onesweepMaximumSpins?: number;
};

/**
 * Onesweep radix sort (Adinets and Merrill) with decoupled look-back and Levien's fallback.
 *
 * One clear dispatch and one pre-pass that histograms every digit from a single read of the keys
 * replace the per-pass histogram and scan dispatches. Each digit pass is then a single dispatch in
 * which a workgroup claims the next tile by atomic ticket, counts the tile's digits, learns how many
 * keys with each digit precede the tile through a per-digit chained scan, and scatters stably.
 *
 * Progress never depends on another workgroup being scheduled: a workgroup that cannot read a
 * predecessor's published count within the polling budget recounts that predecessor's keys itself.
 * Published words carry XOR check words, so readers accept a count only when both words are visible
 * and no cross-location memory ordering is assumed.
 */
export function addOnesweepSort<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  sort: OnesweepSortProps,
  maxComputeWorkgroupsPerDimension: number
): readonly GPUCommandNode<Parameters>[] {
  const nodes: GPUCommandNode<Parameters>[] = [];
  const length = sort.keys.length;
  const tileSize = WORKGROUP_SIZE * sort.elementsPerThread;
  const tileCount = Math.max(1, Math.ceil(length / tileSize));
  const passCount = Math.ceil(sort.keyBits / sort.digitBits);
  const maximumSpins = sort.onesweepMaximumSpins ?? DEFAULT_MAXIMUM_SPINS;
  const sweepLayout = getBoundedDispatchLayout(
    'GPUSort onesweep',
    length,
    tileSize,
    maxComputeWorkgroupsPerDimension
  );

  // State layout in words: one ticket per pass, one 256-bin histogram per pass, then per pass
  // tileCount * bucketCount digits of four published words each.
  const histogramBase = passCount;
  let partitionBase = histogramBase + passCount * WORKGROUP_SIZE;
  const passes = Array.from({length: passCount}, (_, pass) => {
    const digitBits = Math.min(sort.digitBits, sort.keyBits - pass * sort.digitBits);
    const bucketCount = 2 ** digitBits;
    const layout = {pass, digitBits, bucketCount, partitionBase};
    partitionBase += tileCount * bucketCount * STATE_WORDS_PER_DIGIT;
    return layout;
  });
  const stateLength = partitionBase;
  // Onesweep state must fit one storage binding; sort larger inputs with the radix algorithm.
  if (
    stateLength * Uint32Array.BYTES_PER_ELEMENT >
    graph.device.limits.maxStorageBufferBindingSize
  ) {
    throw new Error(`${sort.id} onesweep state exceeds maxStorageBufferBindingSize`);
  }
  const state = createTransientView(graph, `${sort.id}-onesweep-state`, 'uint32', stateLength);

  nodes.push(addClearPass(graph, sort.id, state, stateLength, maxComputeWorkgroupsPerDimension));
  nodes.push(
    addHistogramPass(graph, sort, {
      state,
      tileCount,
      tileSize,
      passCount,
      histogramBase,
      dispatchLayout: sweepLayout
    })
  );

  const scratchKeys =
    passCount > 1
      ? createTransientView(graph, `${sort.id}-onesweep-keys`, 'uint32', length)
      : undefined;
  const scratchValues =
    passCount > 1
      ? createTransientView(graph, `${sort.id}-onesweep-values`, 'uint32', length)
      : undefined;
  let currentKeys = sort.keys;
  let currentValues = sort.values;
  for (const pass of passes) {
    const writesFinalOutput = (passCount - pass.pass) % 2 === 1;
    const nextKeys = writesFinalOutput ? sort.outputKeys : scratchKeys;
    const nextValues = writesFinalOutput ? sort.outputValues : scratchValues;
    if (!nextKeys || !nextValues) throw new Error(`${sort.id} onesweep scratch is missing`);
    nodes.push(
      addSweepPass(graph, sort, {
        ...pass,
        state,
        tileCount,
        tileSize,
        histogramIndex: histogramBase + pass.pass * WORKGROUP_SIZE,
        maximumSpins,
        keys: currentKeys,
        values: currentValues,
        outputKeys: nextKeys,
        outputValues: nextValues,
        dispatchLayout: sweepLayout,
        loop: graph.recording
          ? {
              id: `${sort.id}/passes`,
              iteration: pass.pass,
              phase: 'unconditional',
              maximumIterations: passCount
            }
          : undefined
      })
    );
    currentKeys = nextKeys;
    currentValues = nextValues;
  }
  return nodes;
}

function addClearPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  state: GraphDataView<'uint32'>,
  stateLength: number,
  maxComputeWorkgroupsPerDimension: number
): GPUCommandNode<Parameters> {
  // A grid-stride loop lets a bounded dispatch clear state larger than one launch covers.
  const dispatchLayout = getBoundedDispatchLayout(
    'GPUSort onesweep clear',
    Math.min(stateLength, WORKGROUP_SIZE * maxComputeWorkgroupsPerDimension ** 3),
    WORKGROUP_SIZE,
    maxComputeWorkgroupsPerDimension
  );
  const bindings: GPUKernelBinding[] = [{name: 'state', view: state, access: 'write'}];
  const constants = {
    STATE_LENGTH: stateLength,
    STATE_OFFSET: getViewElementOffset(state),
    GRID_STRIDE: dispatchLayout.x * dispatchLayout.y * dispatchLayout.z * WORKGROUP_SIZE,
    ...getBoundedDispatchOverrides(dispatchLayout)
  };
  return createGPUKernelNode(graph, {
    id: `${id}-onesweep-clear`,
    operation: 'GPUSort',
    variant: 'onesweep-clear',
    moduleId: 'GPUSort/onesweep-clear',
    dedupName: 'onesweep-clear',
    source: /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
@compute @workgroup_size(${WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) localInvocationIndex: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  ${getBoundedInvocationIndexOverrideSource(WORKGROUP_SIZE).body}
  for (var entry = index; entry < STATE_LENGTH; entry += GRID_STRIDE) {
    state[STATE_OFFSET + entry] = 0u;
  }
}`,
    constants,
    bindings,
    workgroupSize: WORKGROUP_SIZE,
    dispatchLayout,
    invocationCount: stateLength,
    workloadScope: 'recording'
  });
}

/** Counts every digit of every pass in one read of the keys. */
function addHistogramPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  sort: OnesweepSortProps,
  props: {
    state: GraphDataView<'uint32'>;
    tileCount: number;
    tileSize: number;
    passCount: number;
    histogramBase: number;
    dispatchLayout: {x: number; y: number; z: number};
  }
): GPUCommandNode<Parameters> {
  const bindings: GPUKernelBinding[] = [
    {name: 'keys', view: sort.keys, access: 'read'},
    {name: 'state', view: props.state, access: 'read_write', atomic: true}
  ];
  const constants = {
    ELEMENT_COUNT: sort.keys.length,
    TILE_COUNT: props.tileCount,
    KEYS_OFFSET: getViewElementOffset(sort.keys),
    STATE_OFFSET: getViewElementOffset(props.state),
    HISTOGRAM_BASE: props.histogramBase,
    ...getBoundedDispatchOverrides(props.dispatchLayout)
  };
  const descending = sort.direction === 'descending';
  return createGPUKernelNode(graph, {
    id: `${sort.id}-onesweep-histogram`,
    operation: 'GPUSort',
    variant: 'onesweep-histogram',
    moduleId: 'GPUSort/onesweep-histogram',
    dedupName: 'onesweep-histogram',
    source: /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
const PASS_COUNT: u32 = ${props.passCount}u;
const DIGIT_BITS: u32 = ${sort.digitBits}u;
const KEY_BITS: u32 = ${sort.keyBits}u;
const ELEMENTS_PER_THREAD: u32 = ${sort.elementsPerThread}u;
const TILE_SIZE: u32 = ${props.tileSize}u;
var<workgroup> localHistogram: array<atomic<u32>, ${props.passCount * WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let tile = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  if (tile >= TILE_COUNT) { return; }
  for (var entry = lane; entry < PASS_COUNT * ${WORKGROUP_SIZE}u; entry += ${WORKGROUP_SIZE}u) {
    atomicStore(&localHistogram[entry], 0u);
  }
  workgroupBarrier();

  let tileBase = tile * TILE_SIZE;
  for (var slot = 0u; slot < ELEMENTS_PER_THREAD; slot++) {
    let index = tileBase + slot * ${WORKGROUP_SIZE}u + lane;
    if (index < ELEMENT_COUNT) {
      let key = keys[KEYS_OFFSET + index];
      for (var passIndex = 0u; passIndex < PASS_COUNT; passIndex++) {
        let bits = min(DIGIT_BITS, KEY_BITS - passIndex * DIGIT_BITS);
        let mask = (1u << bits) - 1u;
        let digit = (key >> (passIndex * DIGIT_BITS)) & mask;
        let bucket = ${descending ? 'mask - digit' : 'digit'};
        atomicAdd(&localHistogram[passIndex * ${WORKGROUP_SIZE}u + bucket], 1u);
      }
    }
  }
  workgroupBarrier();

  for (var entry = lane; entry < PASS_COUNT * ${WORKGROUP_SIZE}u; entry += ${WORKGROUP_SIZE}u) {
    let count = atomicLoad(&localHistogram[entry]);
    if (count != 0u) { atomicAdd(&state[STATE_OFFSET + HISTOGRAM_BASE + entry], count); }
  }
}`,
    constants,
    bindings,
    workgroupSize: WORKGROUP_SIZE,
    dispatchLayout: props.dispatchLayout,
    invocationCount: sort.keys.length,
    workloadScope: 'recording'
  });
}

/** One digit pass: ticketed tile, per-digit decoupled look-back, stable ballot scatter. */
function addSweepPass<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  sort: OnesweepSortProps,
  props: {
    pass: number;
    digitBits: number;
    bucketCount: number;
    partitionBase: number;
    state: GraphDataView<'uint32'>;
    tileCount: number;
    tileSize: number;
    histogramIndex: number;
    maximumSpins: number;
    keys: GraphDataView<'uint32'>;
    values: GraphDataView<'uint32'>;
    outputKeys: GraphDataView<'uint32'>;
    outputValues: GraphDataView<'uint32'>;
    dispatchLayout: {x: number; y: number; z: number};
    loop?: GPUComputationLoopCoordinate;
  }
): GPUCommandNode<Parameters> {
  const bindings: GPUKernelBinding[] = [
    {name: 'keys', view: props.keys, access: 'read'},
    {name: 'values', view: props.values, access: 'read'},
    {name: 'outputKeys', view: props.outputKeys, access: 'write'},
    {name: 'outputValues', view: props.outputValues, access: 'write'},
    {name: 'state', view: props.state, access: 'read_write', atomic: true}
  ];
  const constants = {
    ELEMENT_COUNT: sort.keys.length,
    TILE_COUNT: props.tileCount,
    BIT_OFFSET: props.pass * sort.digitBits,
    KEYS_OFFSET: getViewElementOffset(props.keys),
    VALUES_OFFSET: getViewElementOffset(props.values),
    OUTPUT_KEYS_OFFSET: getViewElementOffset(props.outputKeys),
    OUTPUT_VALUES_OFFSET: getViewElementOffset(props.outputValues),
    STATE_OFFSET: getViewElementOffset(props.state),
    TICKET_INDEX: props.pass,
    HISTOGRAM_INDEX: props.histogramIndex,
    PARTITION_INDEX: props.partitionBase,
    MAXIMUM_SPINS: props.maximumSpins,
    ...getBoundedDispatchOverrides(props.dispatchLayout)
  };
  const descending = sort.direction === 'descending';
  const bucketCount = props.bucketCount;
  return createGPUKernelNode(graph, {
    id: `${sort.id}-onesweep-digit-${props.pass * sort.digitBits}-sweep`,
    operation: 'GPUSort',
    variant: 'onesweep-sweep',
    moduleId: 'GPUSort/onesweep-sweep',
    dedupName: 'onesweep-sweep',
    loop: props.loop,
    source: /* wgsl */ `
${getGPUKernelOverrideDeclarations(constants)}
${getGPUKernelBindingDeclarations(bindings)}
const BUCKET_COUNT: u32 = ${bucketCount}u;
const DIGIT_MASK: u32 = ${bucketCount - 1}u;
const ELEMENTS_PER_THREAD: u32 = ${sort.elementsPerThread}u;
const TILE_SIZE: u32 = ${props.tileSize}u;
const MASK_WORD_COUNT: u32 = ${MASK_WORD_COUNT}u;
const MASK_COUNT: u32 = ${bucketCount * MASK_WORD_COUNT}u;
const AGGREGATE_CHECK: u32 = 0x9e3779b9u;
const INCLUSIVE_CHECK: u32 = 0x85ebca6bu;

var<workgroup> digitCounts: array<atomic<u32>, ${bucketCount}>;
var<workgroup> bucketBase: array<u32, ${bucketCount}>;
var<workgroup> bucketCursors: array<u32, ${bucketCount}>;
var<workgroup> digitMasks: array<atomic<u32>, ${bucketCount * MASK_WORD_COUNT}>;
var<workgroup> sharedTile: u32;
var<workgroup> levelFlags: atomic<u32>;
var<workgroup> uniformFlags: u32;

fn getBucket(key: u32) -> u32 {
  let digit = (key >> BIT_OFFSET) & DIGIT_MASK;
  return ${descending ? 'DIGIT_MASK - digit' : 'digit'};
}

fn getDigitState(tile: u32, digit: u32) -> u32 {
  return STATE_OFFSET + PARTITION_INDEX + (tile * BUCKET_COUNT + digit) * ${STATE_WORDS_PER_DIGIT}u;
}

fn publishAggregate(tile: u32, digit: u32, count: u32) {
  let base = getDigitState(tile, digit);
  atomicStore(&state[base], count);
  atomicStore(&state[base + 1u], count ^ AGGREGATE_CHECK);
}

fn publishInclusive(tile: u32, digit: u32, count: u32) {
  let base = getDigitState(tile, digit);
  atomicStore(&state[base + 2u], count);
  atomicStore(&state[base + 3u], count ^ INCLUSIVE_CHECK);
}

@compute @workgroup_size(${WORKGROUP_SIZE}) fn main(
  @builtin(local_invocation_index) lane: u32,
  @builtin(workgroup_id) workgroupId: vec3<u32>
) {
  let linearWorkgroup = (workgroupId.z * DISPATCH_Y + workgroupId.y) * DISPATCH_X + workgroupId.x;
  if (linearWorkgroup >= TILE_COUNT) { return; }
  // Tiles are claimed in the order workgroups start, so every earlier tile is already running.
  if (lane == 0u) { sharedTile = atomicAdd(&state[STATE_OFFSET + TICKET_INDEX], 1u); }
  var ownGlobalCount = 0u;
  for (var bucket = lane; bucket < BUCKET_COUNT; bucket += ${WORKGROUP_SIZE}u) {
    atomicStore(&digitCounts[bucket], 0u);
    bucketCursors[bucket] = 0u;
    ownGlobalCount = atomicLoad(&state[STATE_OFFSET + HISTOGRAM_INDEX + bucket]);
    bucketBase[bucket] = ownGlobalCount;
  }
  let tile = workgroupUniformLoad(&sharedTile);
  let tileBase = tile * TILE_SIZE;

  // Exclusive scan of the global digit counts gives each digit's first output position.
  for (var stride = 1u; stride < BUCKET_COUNT; stride = stride * 2u) {
    var addend = 0u;
    if (lane < BUCKET_COUNT && lane >= stride) { addend = bucketBase[lane - stride]; }
    workgroupBarrier();
    if (lane < BUCKET_COUNT && lane >= stride) { bucketBase[lane] = bucketBase[lane] + addend; }
    workgroupBarrier();
  }
  if (lane < BUCKET_COUNT) { bucketBase[lane] = bucketBase[lane] - ownGlobalCount; }

  // Count this tile's digits, keeping the keys in registers for the scatter.
  var tileKeys: array<u32, ${sort.elementsPerThread}>;
  for (var slot = 0u; slot < ELEMENTS_PER_THREAD; slot++) {
    let index = tileBase + slot * ${WORKGROUP_SIZE}u + lane;
    if (index < ELEMENT_COUNT) {
      let key = keys[KEYS_OFFSET + index];
      tileKeys[slot] = key;
      atomicAdd(&digitCounts[getBucket(key)], 1u);
    }
  }
  workgroupBarrier();

  // Each digit owns a chain of per-tile counts; thread d walks digit d backwards.
  let isDigitThread = lane < BUCKET_COUNT;
  var tileCount = 0u;
  if (isDigitThread) {
    tileCount = atomicLoad(&digitCounts[lane]);
    publishAggregate(tile, lane, tileCount);
    if (tile == 0u) { publishInclusive(0u, lane, tileCount); }
  }
  var prefix = 0u;
  var done = tile == 0u || !isDigitThread;
  var look = tile;
  loop {
    if (look == 0u) { break; }
    look -= 1u;
    if (lane == 0u) { atomicStore(&levelFlags, 0u); }
    workgroupBarrier();
    var needsFallback = false;
    if (!done) {
      let base = getDigitState(look, lane);
      var found = false;
      for (var spin = 0u; spin < MAXIMUM_SPINS; spin++) {
        let inclusive = atomicLoad(&state[base + 2u]);
        if (atomicLoad(&state[base + 3u]) == (inclusive ^ INCLUSIVE_CHECK)) {
          prefix += inclusive;
          done = true;
          found = true;
          break;
        }
        let aggregate = atomicLoad(&state[base]);
        if (atomicLoad(&state[base + 1u]) == (aggregate ^ AGGREGATE_CHECK)) {
          prefix += aggregate;
          found = true;
          break;
        }
      }
      needsFallback = !found;
      atomicOr(&levelFlags, select(2u, 3u, needsFallback));
    }
    workgroupBarrier();
    if (lane == 0u) { uniformFlags = atomicLoad(&levelFlags); }
    let flags = workgroupUniformLoad(&uniformFlags);
    if ((flags & 1u) != 0u) {
      // Decoupled fallback: recount tile 'look' here instead of waiting for its workgroup.
      for (var bucket = lane; bucket < BUCKET_COUNT; bucket += ${WORKGROUP_SIZE}u) {
        atomicStore(&digitCounts[bucket], 0u);
      }
      workgroupBarrier();
      let lookBase = look * TILE_SIZE;
      for (var slot = 0u; slot < ELEMENTS_PER_THREAD; slot++) {
        let index = lookBase + slot * ${WORKGROUP_SIZE}u + lane;
        if (index < ELEMENT_COUNT) {
          atomicAdd(&digitCounts[getBucket(keys[KEYS_OFFSET + index])], 1u);
        }
      }
      workgroupBarrier();
      if (needsFallback) {
        let recounted = atomicLoad(&digitCounts[lane]);
        prefix += recounted;
        publishAggregate(look, lane, recounted);
      }
    }
    if ((flags & 2u) == 0u) { break; }
  }
  if (isDigitThread) {
    publishInclusive(tile, lane, prefix + tileCount);
    bucketBase[lane] = bucketBase[lane] + prefix;
  }
  workgroupBarrier();

  // Stable scatter: rank each key among equal digits in the tile, in ascending key order.
  let maskWord = lane >> 5u;
  let precedingBits = (1u << (lane & 31u)) - 1u;
  for (var slot = 0u; slot < ELEMENTS_PER_THREAD; slot++) {
    for (var mask = lane; mask < MASK_COUNT; mask += ${WORKGROUP_SIZE}u) {
      atomicStore(&digitMasks[mask], 0u);
    }
    workgroupBarrier();

    let index = tileBase + slot * ${WORKGROUP_SIZE}u + lane;
    let valid = index < ELEMENT_COUNT;
    var bucket = 0u;
    if (valid) {
      bucket = getBucket(tileKeys[slot]);
      atomicOr(&digitMasks[bucket * MASK_WORD_COUNT + maskWord], 1u << (lane & 31u));
    }
    workgroupBarrier();

    if (valid) {
      let maskBase = bucket * MASK_WORD_COUNT;
      var localRank = 0u;
      for (var word = 0u; word < maskWord; word++) {
        localRank += countOneBits(atomicLoad(&digitMasks[maskBase + word]));
      }
      localRank += countOneBits(atomicLoad(&digitMasks[maskBase + maskWord]) & precedingBits);
      let outputIndex = bucketBase[bucket] + bucketCursors[bucket] + localRank;
      outputKeys[OUTPUT_KEYS_OFFSET + outputIndex] = tileKeys[slot];
      outputValues[OUTPUT_VALUES_OFFSET + outputIndex] = values[VALUES_OFFSET + index];
    }
    workgroupBarrier();

    for (var digit = lane; digit < BUCKET_COUNT; digit += ${WORKGROUP_SIZE}u) {
      var slotCount = 0u;
      for (var word = 0u; word < MASK_WORD_COUNT; word++) {
        slotCount += countOneBits(atomicLoad(&digitMasks[digit * MASK_WORD_COUNT + word]));
      }
      bucketCursors[digit] += slotCount;
    }
    workgroupBarrier();
  }
}`,
    constants,
    bindings,
    workgroupSize: WORKGROUP_SIZE,
    dispatchLayout: props.dispatchLayout,
    invocationCount: sort.keys.length,
    workloadScope: 'recording'
  });
}
