// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import type {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {GPUSegmentRingAssemblyProps} from './segment-ring-assembly-types';
import {validateGraphChunkBindingRanges} from '../kernels/graph-chunk-dispatch';
import {
  validateGPUStorageBindingByteLength,
  validateCapacity
} from '../kernels/gpu-contributor-utils';
import {getOutputViews} from './ring-output-validation';
import {RING_SHELL_RECORD_WORDS} from './ring-shell-index';

/** Validates every assembler scratch binding before graph mutation. @internal */
export function getRingAssemblyPlan<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  props: GPUSegmentRingAssemblyProps
) {
  const {output} = props;
  validateGraphChunkBindingRanges(graph, id, {
    endpoints: props.endpoints,
    count: props.count,
    segmentOverflow: props.segmentStatus?.overflow,
    segmentCandidateOverflow: props.segmentStatus?.candidateOverflow,
    groups: props.groups,
    ...Object.fromEntries(getOutputViews(output).map((view, index) => [`output ${index}`, view]))
  });
  const segments = props.endpoints.length;
  const ringCapacity = output.ringOffsets.length - 1;
  const vertexCapacity = output.positions.length;
  const validateScratch = (name: string, byteLength: number) => {
    validateGPUStorageBindingByteLength(
      graph.device,
      id,
      name,
      byteLength,
      'reduce the segment or ring capacity'
    );
    validateCapacity(id, `${name} byteLength`, byteLength, 0, graph.device.limits.maxBufferSize);
  };
  // Endpoint storage bounds per-segment sort/scan words. Ring words can be wider when callers
  // reserve more output rings than input segments; include every fixed/derived allocation.
  validateScratch('ring counters scratch', 6 * Uint32Array.BYTES_PER_ELEMENT);
  validateScratch('segment word scratch', segments * Uint32Array.BYTES_PER_ELEMENT);
  validateScratch('ring word scratch', ringCapacity * Uint32Array.BYTES_PER_ELEMENT);
  if (output.ringShells || output.polygons) {
    const leafCapacity = 2 ** Math.ceil(Math.log2(ringCapacity));
    validateScratch(
      'shell records',
      ringCapacity * RING_SHELL_RECORD_WORDS * Uint32Array.BYTES_PER_ELEMENT
    );
    validateScratch('shell bounds', ringCapacity * 4 * Float32Array.BYTES_PER_ELEMENT);
    validateScratch('shell BVH nodes', (2 * leafCapacity - 1) * 2 * Float32Array.BYTES_PER_ELEMENT);
  }
  return {segments, ringCapacity, vertexCapacity};
}
