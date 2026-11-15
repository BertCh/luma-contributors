// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {
  GROUP_NONE,
  GROUP_VALUE_KEY_WGSL,
  readBinding,
  writeBinding
} from './group-statistics-common';
import type {GroupStructure} from './group-statistics-sort';

/**
 * Fixed binary tree of the four shifted group power sums. The published segmented reduction
 * assigns a whole segment to one workgroup; this analysis-specific tree also distributes a
 * dominant group over the device. At each level a block only merges the group crossing its
 * midpoint. Other groups already have their complete sum at their first position. The roots
 * of the two fragments are `max(groupBegin, blockBegin)` and `midpoint`, respectively.
 *
 * Blocks are disjoint within a level, so in-place fragment updates have no inter-invocation
 * read/write overlap. There are fewer than 2 * rowCount merge invocations altogether and
 * ceil(log2(rowCount)) levels, regardless of group skew or unused output capacity. Separate
 * scalar bindings keep the scratch binding limit at four bytes per input row.
 * Signed first and third powers are scaled per group before reduction and restored afterwards,
 * so a same-sign branch cannot overflow before cancelling an opposite-sign branch.
 * @internal
 */
export function getGroupMomentTreeNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  props: {
    id: string;
    operation: string;
    structure: GroupStructure;
    valueBits: GraphDataView<'uint32'>;
    /** Interleaved [shift mean, nonnegative power-of-two scale exponent] per group. */
    means: GraphDataView<'float32'>;
    output: GraphDataView<'float32'>;
  }
): GPUCommandNode<Parameters>[] {
  const {id, operation, structure, valueBits, means, output} = props;
  const {rowCount, positionGroups, permutation, offsets} = structure;
  const powers = Array.from({length: 4}, (_, component) =>
    createTransientView(graph, `${id}-moment-power-${component + 1}`, 'float32', rowCount)
  );
  const nodes: GPUCommandNode<Parameters>[] = [
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moment-terms`,
      operation,
      variant: 'moment-terms',
      bindings: [
        readBinding('positionGroups', positionGroups),
        readBinding('permutation', permutation),
        readBinding('valueBits', valueBits),
        readBinding('means', means, 'f32'),
        ...powers.map((view, component) => writeBinding(`power${component + 1}`, view, 'f32'))
      ],
      invocationCount: rowCount,
      declarations: GROUP_VALUE_KEY_WGSL,
      body: `let group = positionGroups[positionGroupsOffset + index];
  var deviation = 0.0;
  var scaledDeviation = 0.0;
  if (group != ${GROUP_NONE}u) {
    let bits = valueBits[valueBitsOffset + permutation[permutationOffset + index]];
    if (isFiniteBits(bits)) {
      let value = bitcast<f32>(bits);
      let mean = means[meansOffset + 2u * group];
      let scaleExponent = i32(means[meansOffset + 2u * group + 1u]);
      deviation = value - mean;
      scaledDeviation = deviation;
      if (scaleExponent > 0) {
        // Scale before subtracting or cubing: either unscaled operation could overflow.
        scaledDeviation = ldexp(value, -scaleExponent) - ldexp(mean, -scaleExponent);
      }
    }
  }
  let squared = deviation * deviation;
  power1[power1Offset + index] = scaledDeviation;
  power2[power2Offset + index] = squared;
  power3[power3Offset + index] = scaledDeviation * scaledDeviation * scaledDeviation;
  power4[power4Offset + index] = squared * squared;`
    })
  ];
  for (let halfWidth = 1; halfWidth < rowCount; halfWidth *= 2) {
    nodes.push(
      createWGSLKernelNode<Parameters>(graph, {
        id: `${id}-moment-merge-${halfWidth}`,
        operation,
        variant: 'moment-merge',
        bindings: [
          readBinding('positionGroups', positionGroups),
          readBinding('offsets', offsets),
          ...powers.map((view, component) => writeBinding(`power${component + 1}`, view, 'f32'))
        ],
        invocationCount: Math.ceil(rowCount / (2 * halfWidth)),
        body: `let blockBegin = index * ${halfWidth}u * 2u;
  let midpoint = blockBegin + ${halfWidth}u;
  if (midpoint >= ${rowCount}u) {
    return;
  }
  let group = positionGroups[positionGroupsOffset + midpoint];
  if (group == ${GROUP_NONE}u) {
    return;
  }
  let groupBegin = offsets[offsetsOffset + group];
  if (groupBegin >= midpoint) {
    return;
  }
  let root = max(groupBegin, blockBegin);
  ${powers
    .map((_, component) => {
      const name = `power${component + 1}`;
      return `${name}[${name}Offset + root] += ${name}[${name}Offset + midpoint];`;
    })
    .join('\n  ')}`
      })
    );
  }
  nodes.push(
    createWGSLKernelNode<Parameters>(graph, {
      id: `${id}-moments`,
      operation,
      variant: 'moments',
      bindings: [
        readBinding('positionGroups', positionGroups),
        readBinding('means', means, 'f32'),
        ...powers.map((view, component) => readBinding(`power${component + 1}`, view, 'f32')),
        writeBinding('momentSums', output, 'f32')
      ],
      invocationCount: rowCount,
      body: `let group = positionGroups[positionGroupsOffset + index];
  if (group == ${GROUP_NONE}u) {
    return;
  }
  if (index > 0u && positionGroups[positionGroupsOffset + index - 1u] == group) {
    return;
  }
  let scaleExponent = i32(means[meansOffset + 2u * group + 1u]);
  momentSums[momentSumsOffset + 4u * group] = ldexp(power1[power1Offset + index], scaleExponent);
  momentSums[momentSumsOffset + 4u * group + 1u] = power2[power2Offset + index];
  // Three bounded exponent steps also handle a zero sum when 3 * scaleExponent exceeds 127.
  momentSums[momentSumsOffset + 4u * group + 2u] = ldexp(ldexp(ldexp(
    power3[power3Offset + index], scaleExponent), scaleExponent), scaleExponent);
  momentSums[momentSumsOffset + 4u * group + 3u] = power4[power4Offset + index];`
    })
  );
  return nodes;
}
