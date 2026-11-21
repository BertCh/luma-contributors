// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUSort,
  type GPUCommandGraph,
  type GPUCommandNode,
  type GraphDataView
} from '@luma.gl/graph/gpu-core';
import {
  createWGSLKernelNode,
  type WGSLKernelBinding
} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {getSortedSegmentOffsetNodes} from '../../analysis-core/kernels/sorted-segment-offsets';
import {getSortKeyBits} from '../../analysis-core/kernels/sorted-segment-sums';
import {
  INTERACTION_EXACT_WGSL,
  INTERACTION_EXACT_LIMB_COUNT
} from './spatial-interaction-identification-exact';
import type {GPUSpatialInteractionProps} from './gpu-spatial-interaction';

/**
 * Reads coefficient evidence independently of prediction success. Code 0 is undetermined,
 * 1 proves absorption by the intercept or declared margins, 2 is a flat sampled profile,
 * and 3 is a varying sampled profile. Neither sampled outcome proves identification.
 * Sampled deviances are compared after float32 publication with relative tolerance 8*2^-23,
 * scaled by max(1, abs(first), abs(second)). GPU grid distinctness uses float32 coefficients;
 * CPU grid distinctness uses evaluated float64 coefficients. A single distinct point is
 * undetermined in the absence of a structural absorption proof; code 1 takes precedence.
 * Structural checks use exact integer dyadics, with an independent root per support component.
 * One-margin comparisons are linear after grouping. Joint margins require V grouped propagation
 * rounds (V = originCount + destinationCount); their 256 + 2*V command allowance and scratch
 * must fit the contributor's execution plan. Neither sign-zero equality nor subnormal handling
 * depends on floating-point comparisons.
 */
export function decodeGPUSpatialInteractionCoefficientStatus(
  values: Uint32Array | Uint8Array | ArrayBuffer
) {
  const bytes =
    values instanceof ArrayBuffer
      ? new Uint8Array(values)
      : new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  if (bytes.byteLength < 4 || bytes.byteLength % 4 !== 0)
    throw new Error(
      'decodeGPUSpatialInteractionCoefficientStatus requires at least one whole uint32 word'
    );
  const code = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(0, true);
  if (code > 3)
    throw new Error(`decodeGPUSpatialInteractionCoefficientStatus unknown code ${code}`);
  return {
    code,
    evidence: (
      ['undetermined', 'structurally-absorbed', 'sampled-flat', 'sampled-varying'] as const
    )[code as 0 | 1 | 2 | 3],
    structurallyAbsorbed: code === 1,
    sampledFlatProfile: code === 2,
    sampledVaryingProfile: code === 3
  };
}

/** Publication-scale tolerance; sampled variation still does not prove identification. */
export const SPATIAL_INTERACTION_PROFILE_RELATIVE_TOLERANCE = 8 * 2 ** -23;
/** Conservative allowance for two published radix sorts, offsets, evidence and publication. */
export const SPATIAL_INTERACTION_IDENTIFICATION_BASE_COMMAND_ALLOWANCE = 256;
export function getSpatialInteractionIdentificationCommandNodeCount(
  props: GPUSpatialInteractionProps
): number {
  return props.output.coefficientStatus
    ? SPATIAL_INTERACTION_IDENTIFICATION_BASE_COMMAND_ALLOWANCE +
        (props.constraint === 'doubly' ? 2 * (props.originCount + props.destinationCount) : 0)
    : 0;
}

/** Grouped exact dyadic incidence constraints; disconnected components have independent roots. */
export function createSpatialInteractionIdentificationNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUSpatialInteractionProps
): GPUCommandNode<Parameters>[] {
  const output = props.output.coefficientStatus;
  if (!output) return [];
  const pairs = props.flows.length;
  const nodes: GPUCommandNode<Parameters>[] = [];
  const evidence = createTransientView(graph, `${id}-coefficient-evidence`, 'uint32', 2);
  const transient = (name: string, length: number) =>
    createTransientView(graph, `${id}-identification-${name}`, 'uint32', length);
  const binding = (
    name: string,
    view: GraphDataView,
    type: WGSLKernelBinding['type'] = 'u32',
    access: WGSLKernelBinding['access'] = 'read'
  ): WGSLKernelBinding => ({name, view, type, access});
  const add = (
    name: string,
    invocationCount: number,
    bindings: WGSLKernelBinding[],
    body: string,
    declarations = ''
  ) =>
    nodes.push(
      createWGSLKernelNode(graph, {
        id: `${id}-identification-${name}`,
        operation: 'GPUSpatialInteraction',
        variant: `identification-${name.replace(/-\d+$/, '')}`,
        invocationCount,
        bindings,
        body,
        declarations
      })
    );
  const state = (pair: string) =>
    props.pairState ? `pairState[pairStateOffset + ${pair}] == 1u` : 'true';
  const stateBinding = props.pairState ? [binding('pairState', props.pairState)] : [];
  add(
    'init',
    2,
    [binding('evidence', evidence, 'u32', 'read_write')],
    'evidence[evidenceOffset + index] = 0u;'
  );
  const joint = props.constraint === 'doubly';
  const axes = joint
    ? (['origin', 'destination'] as const)
    : ([props.constraint === 'destination' ? 'destination' : 'origin'] as const);
  const grouped = axes.map(axis => {
    const ids = axis === 'origin' ? props.origins : props.destinations;
    const bound = axis === 'origin' ? props.originCount : props.destinationCount;
    const count = props.constraint === 'unconstrained' ? 1 : bound;
    const keys = transient(`${axis}-keys`, pairs);
    const indices = transient(`${axis}-indices`, pairs);
    const sortedKeys = transient(`${axis}-sorted-keys`, pairs);
    const sortedIndices = transient(`${axis}-sorted-indices`, pairs);
    const offsets = transient(`${axis}-offsets`, count + 1);
    add(
      `${axis}-keys`,
      pairs,
      [
        binding('ids', ids),
        ...stateBinding,
        binding('keys', keys, 'u32', 'read_write'),
        binding('indices', indices, 'u32', 'read_write')
      ],
      `
      keys[keysOffset + index] = select(${count}u,${props.constraint === 'unconstrained' ? '0u' : 'ids[idsOffset + index]'},${state('index')} && ids[idsOffset + index] < ${bound}u);
      indices[indicesOffset + index] = index;`
    );
    nodes.push(
      ...new GPUSort({
        id: `${id}-identification-${axis}-sort`,
        keys,
        values: indices,
        outputKeys: sortedKeys,
        outputValues: sortedIndices,
        keyBits: getSortKeyBits(count),
        algorithm: 'radix'
      }).getCommandNodes(graph)
    );
    nodes.push(
      ...getSortedSegmentOffsetNodes(graph, {
        id: `${id}-identification-${axis}`,
        operation: 'GPUSpatialInteraction',
        groupCount: count,
        sortedGroupKeys: sortedKeys,
        segmentOffsets: offsets
      })
    );
    return {axis, ids, count, sortedIndices, offsets};
  });
  if (joint) {
    const vertices = props.originCount + props.destinationCount;
    const stride = INTERACTION_EXACT_LIMB_COUNT + 1;
    let current = transient('potentials-a', vertices * stride);
    let next = transient('potentials-b', vertices * stride);
    add(
      'potentials-init',
      vertices,
      [binding('records', current, 'u32', 'read_write')],
      `let base=index*11u; records[recordsOffset + base] = index; for (var word=1u;word<11u;word++) {records[recordsOffset + base + word]=0u;}`
    );
    for (let round = 0; round < vertices; round++) {
      for (const group of grouped) {
        const origin = group.axis === 'origin';
        const otherIds = origin ? props.destinations : props.origins;
        add(
          `${group.axis}-propagate-${round}`,
          group.count,
          [
            binding('indices', group.sortedIndices),
            binding('offsets', group.offsets),
            binding('otherIds', otherIds),
            binding('covariateBits', props.covariate),
            binding('records', current),
            binding('result', next, 'u32', 'read_write')
          ],
          `
          let vertex=index + ${origin ? 0 : props.originCount}u; let base=vertex*11u;
          var label=records[recordsOffset + base]; var potential:ExactCovariate;
          for (var word=0u;word<10u;word++) {potential[word]=records[recordsOffset + base + 1u + word];}
          for (var slot=offsets[offsetsOffset + index];slot<offsets[offsetsOffset + index + 1u];slot++) {
            let pair=indices[indicesOffset + slot]; let other=otherIds[otherIdsOffset + pair];
            if (other >= ${origin ? props.destinationCount : props.originCount}u) {continue;}
            let otherBase=(other + ${origin ? props.originCount : 0}u)*11u;
            let otherLabel=records[recordsOffset + otherBase];
            if (otherLabel < label) {
              var otherPotential:ExactCovariate;
              for (var word=0u;word<10u;word++) {otherPotential[word]=records[recordsOffset + otherBase + 1u + word];}
              potential=addExact(exactFromBits(covariateBits[covariateBitsOffset + pair]),negateExact(otherPotential)); label=otherLabel;
            }
          }
          result[resultOffset + base]=label;
          for (var word=0u;word<10u;word++) {result[resultOffset + base + 1u + word]=potential[word];}`,
          INTERACTION_EXACT_WGSL
        );
      }
      [current, next] = [next, current];
    }
    add(
      'structural',
      pairs,
      [
        binding('origins', props.origins),
        binding('destinations', props.destinations),
        binding('covariateBits', props.covariate),
        ...stateBinding,
        binding('records', current),
        binding('evidence', evidence, 'atomic<u32>', 'read_write')
      ],
      `
      if (${state('index')}) {
        let origin=origins[originsOffset + index]; let destination=destinations[destinationsOffset + index];
        if (origin < ${props.originCount}u && destination < ${props.destinationCount}u) {
          var first:ExactCovariate;var second:ExactCovariate;
          for (var word=0u;word<10u;word++) {first[word]=records[recordsOffset + origin*11u+1u+word];second[word]=records[recordsOffset + (${props.originCount}u+destination)*11u+1u+word];}
          let bits=covariateBits[covariateBitsOffset + index];
          if ((bits & 0x7fffffffu) >= 0x7f800000u || !equalExact(addExact(first,second),exactFromBits(bits))) {atomicOr(&evidence[evidenceOffset],1u);}
        }
      }`,
      INTERACTION_EXACT_WGSL
    );
  } else {
    const group = grouped[0];
    add(
      'structural',
      pairs,
      [
        binding('ids', group.ids),
        binding('indices', group.sortedIndices),
        binding('offsets', group.offsets),
        binding('covariateBits', props.covariate),
        ...stateBinding,
        binding('evidence', evidence, 'atomic<u32>', 'read_write')
      ],
      `
      if (${state('index')} && ids[idsOffset + index] < ${group.axis === 'origin' ? props.originCount : props.destinationCount}u) {
        let group=${props.constraint === 'unconstrained' ? '0u' : 'ids[idsOffset + index]'};
        let first=indices[indicesOffset + offsets[offsetsOffset + group]];
        let bits=canonicalBits(covariateBits[covariateBitsOffset + index]);
        if (bits != canonicalBits(covariateBits[covariateBitsOffset + first]) || (bits & 0x7fffffffu) >= 0x7f800000u) {atomicOr(&evidence[evidenceOffset],1u);}
      }`,
      INTERACTION_EXACT_WGSL
    );
  }
  add(
    'profile',
    props.coefficients.length,
    [
      binding('candidates', props.output.candidates, 'f32'),
      binding('evidence', evidence, 'atomic<u32>', 'read_write')
    ],
    `
    let valid=candidates[candidatesOffset + index*4u+3u] == 0.0;
    let first=candidates[candidatesOffset];let value=candidates[candidatesOffset + index*4u];
    var flags=select(4u,0u,valid);
    if (valid && abs(value-first) > ${SPATIAL_INTERACTION_PROFILE_RELATIVE_TOLERANCE} * max(1.0,max(abs(first),abs(value)))) {flags |= 1u;}
    atomicOr(&evidence[evidenceOffset + 1u],flags);`
  );
  add(
    'publish',
    1,
    [
      binding('evidence', evidence),
      binding('status', props.output.status),
      binding('output', output, 'u32', 'read_write')
    ],
    `
    let sampled=evidence[evidenceOffset + 1u];var code=0u;
    if (${new Set(props.coefficients.map(Math.fround)).size > 1 && props.calibration !== 'bounded'} && (sampled & 4u) == 0u) {code=select(2u,3u,(sampled & 1u) != 0u);}
    if (evidence[evidenceOffset] == 0u) {code=1u;}
    if (status[statusOffset] != 0u && status[statusOffset] != 4u) {code=0u;}
    output[outputOffset]=code;`
  );
  return nodes;
}
