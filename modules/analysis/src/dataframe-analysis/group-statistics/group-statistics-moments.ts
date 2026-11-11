// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLActiveCountDispatch,
  createWGSLKernelNode
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getFixedPointWGSL, ORDERED_KEY_WGSL} from '../../analysis-core/cell-key-wgsl';
import {
  FIXED_POINT_WGSL,
  FIXED_POINT_WORD_COUNT,
  getFixedPointAtomicAddWGSL,
  getFixedPointHeadroomBits
} from '../../analysis-core/kernels/fixed-point-accumulator';
import {
  atomicBinding,
  GROUP_NONE,
  GROUP_VALUE_KEY_WGSL,
  readBinding,
  writeBinding
} from './group-statistics-common';
import type {GroupStructure} from './group-statistics-sort';

const MOMENT_WORKGROUP_SIZE = 256;

/** Output views of one column that the reductions in this file write. @internal */
export type GroupColumnReductionOutput = {
  counts?: GraphDataView<'uint32'>;
  sums?: GraphDataView<'uint32x2'>;
  sumValues?: GraphDataView<'float32'>;
  means?: GraphDataView<'float32'>;
  minimums?: GraphDataView<'float32'>;
  maximums?: GraphDataView<'float32'>;
  variances?: GraphDataView<'float32'>;
  standardDeviations?: GraphDataView<'float32'>;
  skewness?: GraphDataView<'float32'>;
  kurtosis?: GraphDataView<'float32'>;
  zScores?: GraphDataView<'float32'>;
};

/**
 * WGSL helpers that turn the shifted power sums `(S1, S2, S3, S4)` of `d = v - mean0` and the
 * finite count `n` into central moments. Writing `delta = S1 / n`, the central sums about the true
 * mean are `m2 = S2 - n delta^2`, `m3 = S3 - 3 delta S2 + 2 n delta^3`, and
 * `m4 = S4 - 4 delta S3 + 6 delta^2 S2 - 3 n delta^4`. A group whose `m2` is at most 2^-19 of `S2`
 * is constant up to the float32 rounding of the shift mean, and `m2` is snapped to 0.
 */
function getMomentWGSL(variance: 'sample' | 'population'): string {
  return /* wgsl */ `
const SAMPLE_VARIANCE: bool = ${variance === 'sample'};

fn getMomentM2(count: u32, sums: vec4f) -> f32 {
  let n = f32(count);
  let delta = sums.x / n;
  var m2 = sums.y - sums.x * delta;
  if (m2 <= 1.9073486e-6 * sums.y) {
    m2 = 0.0;
  }
  return max(m2, 0.0);
}

fn getMomentVariance(count: u32, m2: f32) -> f32 {
  if (SAMPLE_VARIANCE) {
    return select(getNaN(), m2 / f32(count - 1u), count >= 2u);
  }
  return m2 / f32(count);
}

fn getMomentSkewness(count: u32, sums: vec4f, m2: f32) -> f32 {
  if (m2 == 0.0) {
    return getNaN();
  }
  let n = f32(count);
  let delta = sums.x / n;
  let m3 = sums.z - 3.0 * delta * sums.y + 2.0 * n * delta * delta * delta;
  let scale = m2 / n;
  return (m3 / n) / (scale * sqrt(scale));
}

fn getMomentKurtosis(count: u32, sums: vec4f, m2: f32) -> f32 {
  if (m2 == 0.0) {
    return getNaN();
  }
  let n = f32(count);
  let delta = sums.x / n;
  let delta2 = delta * delta;
  let m4 = sums.w - 4.0 * delta * sums.z + 6.0 * delta2 * sums.y - 3.0 * n * delta2 * delta2;
  let scale = m2 / n;
  return (m4 / n) / (scale * scale) - 3.0;
}
`;
}

/** Which reductions of one column to schedule. @internal */
export type GroupColumnReductionNeeds = {
  sums: boolean;
  minimum: boolean;
  maximum: boolean;
  moments: boolean;
  zScore: boolean;
};

/**
 * Largest group capacity accumulated in workgroup-private tables (at most 8 words per group: a
 * count, three fixed-point words, two caller-scale sum words and two extreme keys; 8 KiB of the
 * 16 KiB default workgroup storage).
 */
const PRIVATIZED_GROUP_LIMIT = 256;
const ACCUMULATE_WORKGROUP_SIZE = 256;

/**
 * Per-column reductions over the sorted group layout: finite counts, exact fixed-point sums and
 * means (a 96-bit accumulator with a data-derived unit for the float outputs, and the 64-bit
 * `1 / sumScale` words for `output.sums`), ordered-key extremes (integer atomics, order
 * independent), fixed-order moment sums (one
 * 256-thread workgroup per group: strided partial sums then a fixed binary tree, so bitwise
 * reproducible), derived moments, and per-row z-scores.
 *
 * Returns the graph-owned finite-count view, which order statistics reuse.
 *
 * @internal
 */
export function getColumnReductionNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    structure: GroupStructure;
    valueBits: GraphDataView<'uint32'>;
    needs: GroupColumnReductionNeeds;
    variance: 'sample' | 'population';
    sumScale: number;
    output: GroupColumnReductionOutput;
  }
): {
  nodes: GPUCommandNode<Parameters>[];
  finiteCounts: GraphDataView<'uint32'>;
} {
  const {id, operation, structure, valueBits, needs, output} = props;
  const {capacity, rowCount} = structure;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const u32 = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'uint32', length);
  const f32 = (name: string, length: number) =>
    createTransientView(graph, `${id}-${name}`, 'float32', length);

  const finiteCounts = u32('finite-counts', capacity);
  const needsSums = needs.sums || needs.moments || needs.zScore;
  // Float sums and means come from the shared 96-bit fixed-point accumulator, whose unit follows
  // the largest finite value of the column that lies in a kept group. Three words per group, then
  // one word that holds the largest biased exponent (the scale of the unit).
  const needsFloatSums =
    Boolean(output.sumValues) || Boolean(output.means) || needs.moments || needs.zScore;
  const sumWords =
    needsSums && needsFloatSums
      ? u32('sum-words', FIXED_POINT_WORD_COUNT * capacity + 1)
      : undefined;
  // The 64-bit words in units of `1 / sumScale` are accumulated only for `output.sums`, in place.
  const sums = output.sums;
  const fixedPointDeclarations = /* wgsl */ `
const SUM_SCALE_INDEX: u32 = ${FIXED_POINT_WORD_COUNT * capacity}u;
const FIXED_POINT_HEADROOM_BITS: u32 = ${getFixedPointHeadroomBits(rowCount)}u;
${FIXED_POINT_WGSL}`;
  const minimumKeys = needs.minimum ? u32('minimum-keys', capacity) : undefined;
  const maximumKeys = needs.maximum ? u32('maximum-keys', capacity) : undefined;
  const needsMoments = needs.moments || needs.zScore;
  const needsMeans = Boolean(output.means) || needsMoments;
  const meanScratch = needsMoments ? f32('means', capacity) : undefined;

  // 1. Clear accumulators.
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-clear`,
      operation,
      variant: 'clear',
      bindings: [
        writeBinding('finiteCounts', finiteCounts),
        ...(sumWords ? [writeBinding('sumWords', sumWords)] : []),
        ...(sums ? [writeBinding('sums', sums)] : []),
        ...(minimumKeys ? [writeBinding('minimumKeys', minimumKeys)] : []),
        ...(maximumKeys ? [writeBinding('maximumKeys', maximumKeys)] : [])
      ],
      invocationCount: capacity,
      body: `finiteCounts[finiteCountsOffset + index] = 0u;
  ${
    sumWords
      ? `for (var word = 0u; word < ${FIXED_POINT_WORD_COUNT}u; word++) {
    sumWords[sumWordsOffset + ${FIXED_POINT_WORD_COUNT}u * index + word] = 0u;
  }
  if (index == 0u) {
    sumWords[sumWordsOffset + ${FIXED_POINT_WORD_COUNT * capacity}u] = 0u;
  }`
      : ''
  }
  ${sums ? 'sums[sumsOffset + 2u * index] = 0u;\n  sums[sumsOffset + 2u * index + 1u] = 0u;' : ''}
  ${minimumKeys ? 'minimumKeys[minimumKeysOffset + index] = 0xffffffffu;' : ''}
  ${maximumKeys ? 'maximumKeys[maximumKeysOffset + index] = 0u;' : ''}`
    })
  );

  // 2. Scale of the fixed-point unit: the largest biased exponent of the values that step 3 adds
  // (the same predicate), one `atomicMax` per workgroup. Excluded rows (a masked sentinel such as
  // 1e30) do not raise the unit.
  if (sumWords) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-sum-scale`,
        operation,
        variant: 'sum-scale',
        bindings: [
          readBinding('rowGroups', structure.rowGroups),
          readBinding('valueBits', valueBits),
          atomicBinding('sumWords', sumWords)
        ],
        invocationCount: rowCount,
        guardIndex: false,
        workgroupSize: ACCUMULATE_WORKGROUP_SIZE,
        declarations: `${fixedPointDeclarations}
var<workgroup> workgroupExponentBits: atomic<u32>;`,
        // No early return: every invocation of a workgroup must reach the barriers.
        body: `if (localInvocationIndex == 0u) {
    atomicStore(&workgroupExponentBits, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT && rowGroups[rowGroupsOffset + index] != ${GROUP_NONE}u) {
    // 0 for zero and for non-finite values, which step 3 skips.
    let exponentBits = getFixedPointExponentBits(bitcast<f32>(valueBits[valueBitsOffset + index]));
    if (exponentBits != 0u) {
      atomicMax(&workgroupExponentBits, exponentBits);
    }
  }
  workgroupBarrier();
  if (localInvocationIndex == 0u) {
    let workgroupMaximum = atomicLoad(&workgroupExponentBits);
    if (workgroupMaximum != 0u) {
      atomicMax(&sumWords[sumWordsOffset + SUM_SCALE_INDEX], workgroupMaximum);
    }
  }`
      })
    );
  }

  // 3. Accumulate with integer atomics, one invocation per source row.
  const isPrivatized = capacity <= PRIVATIZED_GROUP_LIMIT;
  const sumWordsDeclarations = sumWords
    ? `${fixedPointDeclarations}
${getFixedPointAtomicAddWGSL('sumWords')}
fn getSumUnitExponent() -> i32 {
  return getFixedPointUnitExponent(
    atomicLoad(&sumWords[sumWordsOffset + SUM_SCALE_INDEX]),
    FIXED_POINT_HEADROOM_BITS
  );
}`
    : '';
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-accumulate`,
      operation,
      variant: 'accumulate',
      bindings: [
        readBinding('rowGroups', structure.rowGroups),
        readBinding('valueBits', valueBits),
        atomicBinding('finiteCounts', finiteCounts),
        ...(sumWords ? [atomicBinding('sumWords', sumWords)] : []),
        ...(sums ? [atomicBinding('sums', sums)] : []),
        ...(minimumKeys ? [atomicBinding('minimumKeys', minimumKeys)] : []),
        ...(maximumKeys ? [atomicBinding('maximumKeys', maximumKeys)] : [])
      ],
      invocationCount: rowCount,
      // Few groups make every row contend on the same handful of global counters, so each
      // workgroup first accumulates into workgroup-private tables and flushes once per group.
      // Integer adds, mins and maxes are associative, so the result is unchanged and exact.
      ...(isPrivatized
        ? {
            guardIndex: false,
            workgroupSize: ACCUMULATE_WORKGROUP_SIZE,
            declarations: `${getFixedPointWGSL(props.sumScale)}
${GROUP_VALUE_KEY_WGSL}
${sumWordsDeclarations}
const GROUP_CAPACITY: u32 = ${capacity}u;
var<workgroup> localCounts: array<atomic<u32>, ${capacity}>;
${
  sumWords
    ? `var<workgroup> localWords: array<atomic<u32>, ${FIXED_POINT_WORD_COUNT * capacity}>;
const localWordsOffset: u32 = 0u;
${getFixedPointAtomicAddWGSL('localWords')}`
    : ''
}
${sums ? `var<workgroup> localSums: array<atomic<u32>, ${2 * capacity}>;` : ''}
${minimumKeys ? `var<workgroup> localMinimums: array<atomic<u32>, ${capacity}>;` : ''}
${maximumKeys ? `var<workgroup> localMaximums: array<atomic<u32>, ${capacity}>;` : ''}`,
            // No early return: every invocation of a workgroup must reach the barriers.
            body: `for (var slot = localInvocationIndex; slot < GROUP_CAPACITY; slot += ${ACCUMULATE_WORKGROUP_SIZE}u) {
    atomicStore(&localCounts[slot], 0u);
    ${
      sumWords
        ? `for (var word = 0u; word < ${FIXED_POINT_WORD_COUNT}u; word++) {
      atomicStore(&localWords[${FIXED_POINT_WORD_COUNT}u * slot + word], 0u);
    }`
        : ''
    }
    ${sums ? 'atomicStore(&localSums[2u * slot], 0u);\n    atomicStore(&localSums[2u * slot + 1u], 0u);' : ''}
    ${minimumKeys ? 'atomicStore(&localMinimums[slot], 0xffffffffu);' : ''}
    ${maximumKeys ? 'atomicStore(&localMaximums[slot], 0u);' : ''}
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT) {
    let group = rowGroups[rowGroupsOffset + index];
    let bits = valueBits[valueBitsOffset + index];
    if (group != ${GROUP_NONE}u && isFiniteBits(bits)) {
      atomicAdd(&localCounts[group], 1u);
      ${
        sumWords
          ? 'addFixedPointToLocalWords(group, quantizeFixedPoint(bitcast<f32>(bits), getSumUnitExponent()));'
          : ''
      }
      ${
        sums
          ? `let value = cellScaleValue(bitcast<f32>(bits));
      let previous = atomicAdd(&localSums[2u * group], value.y);
      let high = value.x + select(0u, 1u, previous + value.y < previous);
      if (high != 0u) {
        atomicAdd(&localSums[2u * group + 1u], high);
      }`
          : ''
      }
      ${minimumKeys ? 'atomicMin(&localMinimums[group], getValueKey(bits));' : ''}
      ${maximumKeys ? 'atomicMax(&localMaximums[group], getValueKey(bits));' : ''}
    }
  }
  workgroupBarrier();
  for (var slot = localInvocationIndex; slot < GROUP_CAPACITY; slot += ${ACCUMULATE_WORKGROUP_SIZE}u) {
    let count = atomicLoad(&localCounts[slot]);
    if (count != 0u) {
      atomicAdd(&finiteCounts[finiteCountsOffset + slot], count);
      ${
        sumWords
          ? `addFixedPointToSumWords(slot, vec3<u32>(
        atomicLoad(&localWords[${FIXED_POINT_WORD_COUNT}u * slot]),
        atomicLoad(&localWords[${FIXED_POINT_WORD_COUNT}u * slot + 1u]),
        atomicLoad(&localWords[${FIXED_POINT_WORD_COUNT}u * slot + 2u])
      ));`
          : ''
      }
      ${
        sums
          ? `let localLow = atomicLoad(&localSums[2u * slot]);
      let previous = atomicAdd(&sums[sumsOffset + 2u * slot], localLow);
      let high = atomicLoad(&localSums[2u * slot + 1u]) + select(0u, 1u, previous + localLow < previous);
      if (high != 0u) {
        atomicAdd(&sums[sumsOffset + 2u * slot + 1u], high);
      }`
          : ''
      }
      ${minimumKeys ? 'atomicMin(&minimumKeys[minimumKeysOffset + slot], atomicLoad(&localMinimums[slot]));' : ''}
      ${maximumKeys ? 'atomicMax(&maximumKeys[maximumKeysOffset + slot], atomicLoad(&localMaximums[slot]));' : ''}
    }
  }`
          }
        : {
            declarations: `${getFixedPointWGSL(props.sumScale)}
${GROUP_VALUE_KEY_WGSL}
${sumWordsDeclarations}`,
            body: `let group = rowGroups[rowGroupsOffset + index];
  if (group == ${GROUP_NONE}u) {
    return;
  }
  let bits = valueBits[valueBitsOffset + index];
  if (!isFiniteBits(bits)) {
    return;
  }
  atomicAdd(&finiteCounts[finiteCountsOffset + group], 1u);
  ${
    sumWords
      ? 'addFixedPointToSumWords(group, quantizeFixedPoint(bitcast<f32>(bits), getSumUnitExponent()));'
      : ''
  }
  ${
    sums
      ? `let value = cellScaleValue(bitcast<f32>(bits));
  let previous = atomicAdd(&sums[sumsOffset + 2u * group], value.y);
  let high = value.x + select(0u, 1u, previous + value.y < previous);
  if (high != 0u) {
    atomicAdd(&sums[sumsOffset + 2u * group + 1u], high);
  }`
      : ''
  }
  ${minimumKeys ? 'atomicMin(&minimumKeys[minimumKeysOffset + group], getValueKey(bits));' : ''}
  ${maximumKeys ? 'atomicMax(&maximumKeys[maximumKeysOffset + group], getValueKey(bits));' : ''}`
          })
    })
  );

  // 4. Counts, sums and means per group. Groups past the count have no finite values.
  if (output.counts || sumWords) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish-sums`,
        operation,
        variant: 'finish-sums',
        bindings: [
          readBinding('finiteCounts', finiteCounts),
          ...(sumWords ? [readBinding('sumWords', sumWords)] : []),
          ...(output.counts ? [writeBinding('countsOut', output.counts)] : []),
          ...(output.sumValues ? [writeBinding('sumValuesOut', output.sumValues, 'f32')] : []),
          ...(output.means ? [writeBinding('meansOut', output.means, 'f32')] : []),
          ...(meanScratch ? [writeBinding('meanScratch', meanScratch, 'f32')] : [])
        ],
        invocationCount: capacity,
        declarations: `${ORDERED_KEY_WGSL}
${sumWords ? fixedPointDeclarations : ''}`,
        body: `let count = finiteCounts[finiteCountsOffset + index];
  ${output.counts ? 'countsOut[countsOutOffset + index] = count;' : ''}
  ${
    sumWords
      ? `let base = sumWordsOffset + ${FIXED_POINT_WORD_COUNT}u * index;
  // The exact integer sum, rounded once to the nearest float32.
  let total = getFixedPointFloat(
    vec3<u32>(sumWords[base], sumWords[base + 1u], sumWords[base + 2u]),
    getFixedPointUnitExponent(sumWords[sumWordsOffset + SUM_SCALE_INDEX], FIXED_POINT_HEADROOM_BITS)
  );
  ${output.sumValues ? 'sumValuesOut[sumValuesOutOffset + index] = total;' : ''}
  ${needsMeans ? `let mean = select(getNaN(), total / f32(count), count > 0u);` : ''}
  ${output.means ? 'meansOut[meansOutOffset + index] = mean;' : ''}
  ${meanScratch ? 'meanScratch[meanScratchOffset + index] = mean;' : ''}`
      : ''
  }`
      })
    );
  }

  // 5. Extremes (NaN for groups without finite values).
  if (minimumKeys || maximumKeys) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish-extremes`,
        operation,
        variant: 'finish-extremes',
        bindings: [
          readBinding('finiteCounts', finiteCounts),
          ...(minimumKeys ? [readBinding('minimumKeys', minimumKeys)] : []),
          ...(maximumKeys ? [readBinding('maximumKeys', maximumKeys)] : []),
          ...(output.minimums ? [writeBinding('minimumsOut', output.minimums, 'f32')] : []),
          ...(output.maximums ? [writeBinding('maximumsOut', output.maximums, 'f32')] : [])
        ],
        invocationCount: capacity,
        declarations: ORDERED_KEY_WGSL,
        body: `let isEmpty = finiteCounts[finiteCountsOffset + index] == 0u;
  ${minimumKeys && output.minimums ? 'minimumsOut[minimumsOutOffset + index] = select(decodeOrderedKey(minimumKeys[minimumKeysOffset + index]), getNaN(), isEmpty);' : ''}
  ${maximumKeys && output.maximums ? 'maximumsOut[maximumsOutOffset + index] = select(decodeOrderedKey(maximumKeys[maximumKeysOffset + index]), getNaN(), isEmpty);' : ''}`
      })
    );
  }

  if (!needsMoments || !meanScratch) {
    return {nodes, finiteCounts};
  }

  // 6. Shifted power sums in fixed order, one workgroup per group. The dispatch is sized from the
  // group total the sort stage wrote on the GPU, so unoccupied capacity launches no workgroups.
  // Groups past the total keep unwritten `momentSums`; every reader skips groups whose finite
  // count is zero, and those groups have a zero count.
  //
  // Scaling limit: one 256-lane workgroup reduces a whole group, strided over its sorted range, so
  // a group of g rows costs about g / 256 sequential loop trips per lane while the rest of the
  // device idles. Past roughly 2^20 rows in one group (a few thousand trips per lane, about 100
  // times the work of an average group when thousands of groups are occupied) that group is the
  // bottleneck. The fix is to split such groups into fixed-size row chunks with
  // `load-balanced-for-each` and merge the chunk partials in fixed order (or reduce the sorted
  // values with `sorted-segment-sums`' segment tree), which keeps the result deterministic.
  const momentSums = f32('moment-sums', 4 * capacity);
  const momentDispatch = createWGSLActiveCountDispatch<Parameters>(graph, {
    id: `${id}-moments-dispatch`,
    operation,
    count: structure.groupTotal,
    maximumItemCount: capacity,
    workgroupSize: MOMENT_WORKGROUP_SIZE,
    invocationsPerItem: MOMENT_WORKGROUP_SIZE
  });
  nodes.push(momentDispatch.updateNode);
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments`,
      operation,
      variant: 'moments',
      bindings: [
        readBinding('offsets', structure.offsets),
        readBinding('permutation', structure.permutation),
        readBinding('valueBits', valueBits),
        readBinding('means', meanScratch, 'f32'),
        readBinding('finiteCounts', finiteCounts),
        writeBinding('momentSums', momentSums, 'f32')
      ],
      workgroupSize: MOMENT_WORKGROUP_SIZE,
      invocationCount: capacity * MOMENT_WORKGROUP_SIZE,
      guardIndex: false,
      condition: momentDispatch.condition,
      declarations: `var<workgroup> partials: array<f32, ${4 * MOMENT_WORKGROUP_SIZE}>;
fn isFiniteBits(bits: u32) -> bool {
  return (bits & 0x7f800000u) != 0x7f800000u;
}`,
      // No early return: every invocation of a workgroup must reach the barriers.
      body: `let group = index / ${MOMENT_WORKGROUP_SIZE}u;
  let isInRange = index < INVOCATION_COUNT;
  var s1 = 0.0;
  var s2 = 0.0;
  var s3 = 0.0;
  var s4 = 0.0;
  if (isInRange && finiteCounts[finiteCountsOffset + group] > 0u) {
    let begin = offsets[offsetsOffset + group];
    let end = offsets[offsetsOffset + group + 1u];
    let mean = means[meansOffset + group];
    for (var position = begin + localInvocationIndex; position < end; position += ${MOMENT_WORKGROUP_SIZE}u) {
      let bits = valueBits[valueBitsOffset + permutation[permutationOffset + position]];
      if (isFiniteBits(bits)) {
        let d = bitcast<f32>(bits) - mean;
        let d2 = d * d;
        s1 += d;
        s2 += d2;
        s3 += d2 * d;
        s4 += d2 * d2;
      }
    }
  }
  partials[4u * localInvocationIndex] = s1;
  partials[4u * localInvocationIndex + 1u] = s2;
  partials[4u * localInvocationIndex + 2u] = s3;
  partials[4u * localInvocationIndex + 3u] = s4;
  workgroupBarrier();
  for (var stride = ${MOMENT_WORKGROUP_SIZE / 2}u; stride > 0u; stride = stride / 2u) {
    if (localInvocationIndex < stride) {
      for (var component = 0u; component < 4u; component++) {
        partials[4u * localInvocationIndex + component] += partials[4u * (localInvocationIndex + stride) + component];
      }
    }
    workgroupBarrier();
  }
  if (isInRange && localInvocationIndex == 0u) {
    for (var component = 0u; component < 4u; component++) {
      momentSums[momentSumsOffset + 4u * group + component] = partials[component];
    }
  }`
    })
  );

  // 7. Variance, standard deviation, skewness and kurtosis.
  if (output.variances || output.standardDeviations || output.skewness || output.kurtosis) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-finish-moments`,
        operation,
        variant: 'finish-moments',
        bindings: [
          readBinding('finiteCounts', finiteCounts),
          readBinding('momentSums', momentSums, 'f32'),
          ...(output.variances ? [writeBinding('variancesOut', output.variances, 'f32')] : []),
          ...(output.standardDeviations
            ? [writeBinding('deviationsOut', output.standardDeviations, 'f32')]
            : []),
          ...(output.skewness ? [writeBinding('skewnessOut', output.skewness, 'f32')] : []),
          ...(output.kurtosis ? [writeBinding('kurtosisOut', output.kurtosis, 'f32')] : [])
        ],
        invocationCount: capacity,
        declarations: `${ORDERED_KEY_WGSL}
${getMomentWGSL(props.variance)}`,
        body: `let count = finiteCounts[finiteCountsOffset + index];
  let nan = getNaN();
  var variance = nan;
  var skewness = nan;
  var kurtosis = nan;
  if (count > 0u) {
    let sums = vec4f(
      momentSums[momentSumsOffset + 4u * index],
      momentSums[momentSumsOffset + 4u * index + 1u],
      momentSums[momentSumsOffset + 4u * index + 2u],
      momentSums[momentSumsOffset + 4u * index + 3u]
    );
    let m2 = getMomentM2(count, sums);
    variance = getMomentVariance(count, m2);
    skewness = getMomentSkewness(count, sums, m2);
    kurtosis = getMomentKurtosis(count, sums, m2);
  }
  ${output.variances ? 'variancesOut[variancesOutOffset + index] = variance;' : ''}
  ${output.standardDeviations ? 'deviationsOut[deviationsOutOffset + index] = select(nan, sqrt(variance), count > 0u && (!SAMPLE_VARIANCE || count >= 2u));' : ''}
  ${output.skewness ? 'skewnessOut[skewnessOutOffset + index] = skewness;' : ''}
  ${output.kurtosis ? 'kurtosisOut[kurtosisOutOffset + index] = kurtosis;' : ''}`
      })
    );
  }

  // 8. Per-row z-scores in source row order.
  if (output.zScores) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-z-scores`,
        operation,
        variant: 'z-scores',
        bindings: [
          readBinding('rowGroups', structure.rowGroups),
          readBinding('valueBits', valueBits),
          readBinding('means', meanScratch, 'f32'),
          readBinding('finiteCounts', finiteCounts),
          readBinding('momentSums', momentSums, 'f32'),
          writeBinding('zScoresOut', output.zScores, 'f32')
        ],
        invocationCount: rowCount,
        declarations: `${ORDERED_KEY_WGSL}
${getMomentWGSL(props.variance)}
fn isFiniteBits(bits: u32) -> bool {
  return (bits & 0x7f800000u) != 0x7f800000u;
}`,
        body: `let group = rowGroups[rowGroupsOffset + index];
  let bits = valueBits[valueBitsOffset + index];
  var z = getNaN();
  if (group != ${GROUP_NONE}u && isFiniteBits(bits)) {
    let count = finiteCounts[finiteCountsOffset + group];
    let sums = vec4f(
      momentSums[momentSumsOffset + 4u * group],
      momentSums[momentSumsOffset + 4u * group + 1u],
      momentSums[momentSumsOffset + 4u * group + 2u],
      momentSums[momentSumsOffset + 4u * group + 3u]
    );
    let m2 = getMomentM2(count, sums);
    let variance = select(0.0, getMomentVariance(count, m2), count >= 2u || !SAMPLE_VARIANCE);
    // Centre on the mean the variance uses: the shift mean plus delta = S1 / n. delta is at most
    // half a float32 step of the mean, so it survives only when it is subtracted from the
    // deviation, never when it is added to the mean. A shader compiler may rewrite
    // (v - mean) - delta as v - (mean + delta) and lose it (measured on the test GPU: values
    // 1e7 .. 1e7 + 3 came out 0.39 off). The deviation therefore passes through a bit operation
    // with a runtime zero the compiler cannot see (a group index is below 2^31), which fixes
    // its rounding before delta is subtracted.
    let delta = sums.x / f32(count);
    let runtimeZero = group >> 31u;
    let deviation = bitcast<f32>(
      bitcast<u32>(bitcast<f32>(bits) - means[meansOffset + group]) | runtimeZero
    );
    let centered = deviation - delta;
    z = select(centered / sqrt(variance), 0.0, m2 == 0.0 || !(variance > 0.0));
  }
  zScoresOut[zScoresOutOffset + index] = z;`
      })
    );
  }
  return {nodes, finiteCounts};
}
