// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createGroupStatisticsKernelNode} from './group-statistics-kernel';
import {getFixedPointWGSL, ORDERED_KEY_WGSL} from '../../analysis-core/kernels/cell-key-wgsl';
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
  writeBinding,
  type GroupValueChunk
} from './group-statistics-common';
import type {GroupStructure} from './group-statistics-sort';

import {getSumOverflowNodes} from '../key-join/sum-overflow-nodes';
import {FINITE_MEAN_WGSL} from '../key-join/finite-mean-wgsl';
import {getGroupMomentTreeNodes} from './group-statistics-moment-tree';

/**
 * With at most 2^31 rows (the published sort limit), bound each scaled value below 2^31
 * and each scaled deviation below 2^32. Cubic absolute sums then stay below 2^127, even
 * before opposite signs cancel. Values below this exponent need no scaling.
 */
const MAXIMUM_SCALED_VALUE_EXPONENT = 30;

/** Output views of one column that the reductions in this file write. @internal */
export type GroupColumnReductionOutput = {
  counts?: GraphDataView<'uint32'>;
  sums?: GraphDataView<'uint32x2'>;
  status?: {sumOverflow?: GraphDataView<'uint32'>};
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
 * independent), fixed-order moment sums (a row-parallel binary tree with logarithmic depth, so bitwise
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
    valueChunks: readonly GroupValueChunk[];
    needs: GroupColumnReductionNeeds;
    variance: 'sample' | 'population';
    sumScale: number;
    scaleMode?: 'shared' | 'per-group';
    output: GroupColumnReductionOutput;
    /** Tiled moments retain the global shift mean but never allocate a global row tree. */
    getMomentNodes?: (
      means: GraphDataView<'float32'>,
      output: GraphDataView<'float32'>
    ) => GPUCommandNode<Parameters>[];
  }
): {
  nodes: GPUCommandNode<Parameters>[];
  finiteCounts: GraphDataView<'uint32'>;
} {
  const {id, operation, structure, valueChunks, needs, output} = props;
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
      ? u32(
          'sum-words',
          FIXED_POINT_WORD_COUNT * capacity + (props.scaleMode === 'per-group' ? capacity : 1)
        )
      : undefined;
  // The 64-bit words in units of `1 / sumScale` are accumulated only for `output.sums`, in place.
  const sums = output.sums;
  const fixedPointDeclarations = /* wgsl */ `
const SUM_SCALE_INDEX: u32 = ${FIXED_POINT_WORD_COUNT * capacity}u;
const FIXED_POINT_HEADROOM_BITS: u32 = ${getFixedPointHeadroomBits(rowCount)}u;
${FIXED_POINT_WGSL}`;
  const needsMoments = needs.moments || needs.zScore;
  // Moments also use the extremes to select a separate power-of-two scale for each group.
  const minimumKeys = needs.minimum || needsMoments ? u32('minimum-keys', capacity) : undefined;
  const maximumKeys = needs.maximum || needsMoments ? u32('maximum-keys', capacity) : undefined;
  const needsMeans = Boolean(output.means) || needsMoments;
  // Interleaved [mean, scaleExponent]; one binding keeps the terms kernel within eight bindings.
  const meanScratch = needsMoments ? f32('means', 2 * capacity) : undefined;

  // 1. Clear accumulators.
  nodes.push(
    createGroupStatisticsKernelNode<Parameters>(graph, {
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

  if (sumWords && props.scaleMode === 'per-group') {
    nodes.push(
      createGroupStatisticsKernelNode(graph, {
        id: `${id}-clear-group-scales`,
        operation,
        variant: 'clear-group-scales',
        bindings: [writeBinding('sumWords', sumWords)],
        invocationCount: capacity,
        body: `sumWords[sumWordsOffset + ${FIXED_POINT_WORD_COUNT * capacity}u + index] = 0u;`
      })
    );
  }
  // 2. Scale of the fixed-point unit: the largest biased exponent of the values that step 3 adds
  // (the same predicate), one `atomicMax` per workgroup. Excluded rows (a masked sentinel such as
  // 1e30) do not raise the unit.
  if (sumWords) {
    for (const [chunkIndex, {valueBits, rowOffset, rowGroups}] of valueChunks.entries()) {
      nodes.push(
        createGroupStatisticsKernelNode<Parameters>(graph, {
          id: valueChunks.length === 1 ? `${id}-sum-scale` : `${id}-sum-scale-${chunkIndex}`,
          operation,
          variant: 'sum-scale',
          bindings: [
            readBinding('rowGroups', rowGroups ?? structure.rowGroups),
            readBinding('valueBits', valueBits),
            atomicBinding('sumWords', sumWords),
            ...(props.scaleMode === 'per-group'
              ? [atomicBinding('finiteCounts', finiteCounts)]
              : [])
          ],
          invocationCount: valueBits.length,
          rowOffset,
          guardIndex: false,
          declarations: `${fixedPointDeclarations}
var<workgroup> workgroupExponentBits: atomic<u32>;`,
          // All shared-scale invocations reach both barriers.
          body:
            props.scaleMode === 'per-group'
              ? `if (index < INVOCATION_COUNT) {
    let group = rowGroups[rowGroupsOffset + index + GROUP_ROW_OFFSET];
    let bits = valueBits[valueBitsOffset + index];
    if (group != ${GROUP_NONE}u && (bits & 0x7fffffffu) < 0x7f800000u) {
      atomicAdd(&finiteCounts[finiteCountsOffset + group], 1u);
      atomicMax(&sumWords[sumWordsOffset + SUM_SCALE_INDEX + group], getFixedPointExponentBits(bitcast<f32>(valueBits[valueBitsOffset + index])));
    }
  }`
              : `if (localInvocationIndex == 0u) {
    atomicStore(&workgroupExponentBits, 0u);
  }
  workgroupBarrier();
  if (index < INVOCATION_COUNT && rowGroups[rowGroupsOffset + index + GROUP_ROW_OFFSET] != ${GROUP_NONE}u) {
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
  }

  // 3. Accumulate with integer atomics, one invocation per source row.
  const isPrivatized = capacity <= PRIVATIZED_GROUP_LIMIT;
  const sumWordsDeclarations = sumWords
    ? `${fixedPointDeclarations}
${getFixedPointAtomicAddWGSL('sumWords')}
fn getSumUnitExponent(group: u32) -> i32 {
  return getFixedPointUnitExponent(
    atomicLoad(&sumWords[sumWordsOffset + SUM_SCALE_INDEX${props.scaleMode === 'per-group' ? ' + group' : ''}]),
    ${props.scaleMode === 'per-group' ? 'firstLeadingBit(max(atomicLoad(&finiteCounts[finiteCountsOffset + group]), 1u)) + 1u' : 'FIXED_POINT_HEADROOM_BITS'}
  );
}`
    : '';
  for (const [chunkIndex, {valueBits, rowOffset, rowGroups}] of valueChunks.entries()) {
    nodes.push(
      createGroupStatisticsKernelNode<Parameters>(graph, {
        id: valueChunks.length === 1 ? `${id}-accumulate` : `${id}-accumulate-${chunkIndex}`,
        operation,
        variant: 'accumulate',
        bindings: [
          readBinding('rowGroups', rowGroups ?? structure.rowGroups),
          readBinding('valueBits', valueBits),
          atomicBinding('finiteCounts', finiteCounts),
          ...(sumWords ? [atomicBinding('sumWords', sumWords)] : []),
          ...(sums ? [atomicBinding('sums', sums)] : []),
          ...(minimumKeys ? [atomicBinding('minimumKeys', minimumKeys)] : []),
          ...(maximumKeys ? [atomicBinding('maximumKeys', maximumKeys)] : [])
        ],
        invocationCount: valueBits.length,
        rowOffset,
        // Few groups make every row contend on the same handful of global counters, so each
        // workgroup first accumulates into workgroup-private tables and flushes once per group.
        // Integer adds, mins and maxes are associative, so the result is unchanged and exact.
        ...(isPrivatized
          ? {
              guardIndex: false,
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
    let group = rowGroups[rowGroupsOffset + index + GROUP_ROW_OFFSET];
    let bits = valueBits[valueBitsOffset + index];
    if (group != ${GROUP_NONE}u && isFiniteBits(bits)) {
      atomicAdd(&localCounts[group], 1u);
      ${
        sumWords
          ? 'addFixedPointToLocalWords(group, quantizeFixedPoint(bitcast<f32>(bits), getSumUnitExponent(group)));'
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
      ${sumWords && props.scaleMode === 'per-group' ? '' : 'atomicAdd(&finiteCounts[finiteCountsOffset + slot], count);'}
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
              body: `let group = rowGroups[rowGroupsOffset + index + GROUP_ROW_OFFSET];
  if (group == ${GROUP_NONE}u) {
    return;
  }
  let bits = valueBits[valueBitsOffset + index];
  if (!isFiniteBits(bits)) {
    return;
  }
  ${sumWords && props.scaleMode === 'per-group' ? '' : 'atomicAdd(&finiteCounts[finiteCountsOffset + group], 1u);'}
  ${
    sumWords
      ? 'addFixedPointToSumWords(group, quantizeFixedPoint(bitcast<f32>(bits), getSumUnitExponent(group)));'
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
  }

  if (output.status?.sumOverflow) {
    nodes.push(
      ...getSumOverflowNodes(graph, {
        id: `${id}-sum-overflow`,
        capacity,
        sumScale: props.sumScale,
        useGroupDispatchUniforms: true,
        chunks: valueChunks.map(chunk => ({
          values: chunk.valueBits,
          groups: chunk.rowGroups ?? structure.rowGroups,
          rowOffset: chunk.rowOffset
        })),
        output: output.status.sumOverflow
      })
    );
  }

  // 4. Counts, sums and means per group. Groups past the count have no finite values.
  if (output.counts || sumWords) {
    nodes.push(
      createGroupStatisticsKernelNode<Parameters>(graph, {
        id: `${id}-finish-sums`,
        operation,
        variant: 'finish-sums',
        bindings: [
          readBinding('finiteCounts', finiteCounts),
          ...(sumWords ? [readBinding('sumWords', sumWords)] : []),
          ...(output.counts ? [writeBinding('countsOut', output.counts)] : []),
          ...(output.sumValues ? [writeBinding('sumValuesOut', output.sumValues, 'f32')] : []),
          ...(output.means ? [writeBinding('meansOut', output.means, 'f32')] : []),
          ...(meanScratch && minimumKeys && maximumKeys
            ? [
                writeBinding('meanScratch', meanScratch, 'f32'),
                readBinding('minimumKeys', minimumKeys),
                readBinding('maximumKeys', maximumKeys)
              ]
            : [])
        ],
        invocationCount: capacity,
        declarations: `${ORDERED_KEY_WGSL}
${sumWords ? `${fixedPointDeclarations}\n${FINITE_MEAN_WGSL}` : ''}`,
        body: `let count = finiteCounts[finiteCountsOffset + index];
  ${output.counts ? 'countsOut[countsOutOffset + index] = count;' : ''}
  ${
    sumWords
      ? `let base = sumWordsOffset + ${FIXED_POINT_WORD_COUNT}u * index;
  // The exact integer sum, rounded once to the nearest float32.
  let words = vec3<u32>(sumWords[base], sumWords[base + 1u], sumWords[base + 2u]);
  let unitExponent = getFixedPointUnitExponent(sumWords[sumWordsOffset + SUM_SCALE_INDEX${props.scaleMode === 'per-group' ? ' + index' : ''}], ${props.scaleMode === 'per-group' ? 'firstLeadingBit(max(count, 1u)) + 1u' : 'FIXED_POINT_HEADROOM_BITS'});
  let total = getFixedPointFloat(words, unitExponent);
  ${output.sumValues ? 'sumValuesOut[sumValuesOutOffset + index] = total;' : ''}
  ${
    needsMeans
      ? `// Divide the exact integer total by the exact count before rounding to float32.
  let mean = getFiniteFixedPointMean(words, unitExponent, count);`
      : ''
  }
  ${output.means ? 'meansOut[meansOutOffset + index] = mean;' : ''}
  ${
    meanScratch
      ? `meanScratch[meanScratchOffset + 2u * index] = mean;
  let magnitudeBits = max(
    bitcast<u32>(decodeOrderedKey(minimumKeys[minimumKeysOffset + index])) & 0x7fffffffu,
    bitcast<u32>(decodeOrderedKey(maximumKeys[maximumKeysOffset + index])) & 0x7fffffffu
  );
  let scaleExponent = max(i32(magnitudeBits >> 23u) - 127 - ${MAXIMUM_SCALED_VALUE_EXPONENT}, 0);
  meanScratch[meanScratchOffset + 2u * index + 1u] = f32(select(0, scaleExponent, count > 0u));`
      : ''
  }`
      : ''
  }`
      })
    );
  }

  // 5. Extremes (NaN for groups without finite values).
  if (output.minimums || output.maximums) {
    nodes.push(
      createGroupStatisticsKernelNode<Parameters>(graph, {
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

  // 6. Shifted power sums in a fixed binary tree over sorted row blocks. Empty groups never
  // read momentSums: the finishing and z-score kernels test their finite count first.
  const momentSums = f32('moment-sums', 4 * capacity);
  nodes.push(
    ...(props.getMomentNodes
      ? props.getMomentNodes(meanScratch, momentSums)
      : getGroupMomentTreeNodes(graph, {
          id,
          operation,
          structure,
          valueChunks,
          means: meanScratch,
          output: momentSums
        }))
  );

  // 7. Variance, standard deviation, skewness and kurtosis.
  if (output.variances || output.standardDeviations || output.skewness || output.kurtosis) {
    nodes.push(
      createGroupStatisticsKernelNode<Parameters>(graph, {
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
    for (const [chunkIndex, {valueBits, rowOffset, rowGroups}] of valueChunks.entries()) {
      nodes.push(
        createGroupStatisticsKernelNode<Parameters>(graph, {
          id: valueChunks.length === 1 ? `${id}-z-scores` : `${id}-z-scores-${chunkIndex}`,
          operation,
          variant: 'z-scores',
          bindings: [
            readBinding('rowGroups', rowGroups ?? structure.rowGroups),
            readBinding('valueBits', valueBits),
            readBinding('means', meanScratch, 'f32'),
            readBinding('finiteCounts', finiteCounts),
            readBinding('momentSums', momentSums, 'f32'),
            writeBinding('zScoresOut', output.zScores, 'f32')
          ],
          invocationCount: valueBits.length,
          rowOffset,
          declarations: `${ORDERED_KEY_WGSL}
${getMomentWGSL(props.variance)}
fn isFiniteBits(bits: u32) -> bool {
  return (bits & 0x7f800000u) != 0x7f800000u;
}`,
          body: `let group = rowGroups[rowGroupsOffset + index + GROUP_ROW_OFFSET];
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
      bitcast<u32>(bitcast<f32>(bits) - means[meansOffset + 2u * group]) | runtimeZero
    );
    let centered = deviation - delta;
    z = select(centered / sqrt(variance), 0.0, m2 == 0.0 || !(variance > 0.0));
  }
  zScoresOut[zScoresOutOffset + index + GROUP_ROW_OFFSET] = z;`
        })
      );
    }
  }
  return {nodes, finiteCounts};
}
