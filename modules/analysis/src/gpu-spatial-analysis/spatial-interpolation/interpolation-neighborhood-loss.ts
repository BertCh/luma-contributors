// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {GPUCommandGraph, GPUCommandNode, GraphDataView} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../analysis-core/kernels/wgsl-kernel-nodes';
import {GET_QUIET_NAN_SEEDED_WGSL} from '../../analysis-core/kernels/wgsl-common';
import {getPackedUint32Values} from '../../analysis-core/kernels/packed-readback';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';
import {validateGraphChunkBindingRanges} from '../../analysis-core/kernels/graph-chunk-dispatch';
import {validateWeightsLossSources} from '../spatial-weights/spatial-weights-loss';

/** Resident candidate CSR; producer loss words remain live across frames. */
export type InterpolationNeighborhoodWeights = {
  offsets: GraphDataView<'uint32'>;
  neighbors: GraphDataView<'uint32'>;
  status?: {
    overflow?: GraphDataView<'uint32'>;
    candidateOverflow?: GraphDataView<'uint32'>;
    complete?: GraphDataView<'uint32'>;
  };
};

/** @internal Reuse the spatial-weights loss vocabulary and validation. */
export function getInterpolationNeighborhoodLossViews(
  id: string,
  props: {
    neighborhoodWeights?: InterpolationNeighborhoodWeights;
    neighborhoodStatus?: InterpolationNeighborhoodWeights['status'];
  }
) {
  return validateWeightsLossSources(id, [
    {status: props.neighborhoodWeights?.status},
    {status: props.neighborhoodStatus}
  ]);
}

/** @internal Separate parallel publication pass preserves the numerical solve kernels. */
export function createInterpolationNeighborhoodLossNodes<Parameters>(
  graph: GPUCommandGraph<Parameters>,
  id: string,
  losses: readonly GraphDataView<'uint32'>[],
  output: {
    values: GraphDataView<'float32x4'>;
    status: GraphDataView<'uint32'>;
    weights?: GraphDataView<'float32'>;
  },
  count: number
): GPUCommandNode<Parameters>[] {
  validateGraphChunkBindingRanges(
    graph,
    id,
    Object.fromEntries(losses.map((view, index) => [`loss${index}`, view]))
  );
  return losses.map((loss, lossIndex) =>
    createWGSLKernelNode(graph, {
      id: `${id}-input-incomplete-${lossIndex}`,
      operation: 'InterpolationNeighborhoodLoss',
      invocationCount: Math.max(count * 4, output.weights?.length ?? 0),
      bindings: [
        {name: 'loss', view: loss, type: 'u32', access: 'read'},
        {name: 'values', view: output.values, type: 'f32', access: 'read_write'},
        {name: 'statuses', view: output.status, type: 'u32', access: 'read_write'},
        ...(output.weights
          ? [
              {
                name: 'weights',
                view: output.weights,
                type: 'f32' as const,
                access: 'read_write' as const
              }
            ]
          : [])
      ],
      declarations: GET_QUIET_NAN_SEEDED_WGSL,
      body: `if (loss[lossOffset] != 0u) {
  if (index < ${count * 4}u) { values[valuesOffset + index] = getQuietNaN(index); }
  if (index < ${count}u) { statuses[statusesOffset + index] = 3u; }
  ${output.weights ? `if (index < ${output.weights.length}u) { weights[weightsOffset + index] = 0.0; }` : ''}
}`
    })
  );
}

/** @internal Decode the shared prediction codes without accepting missing status words. */
export function decodeInterpolationPredictionStatus(
  id: string,
  input: Uint32Array | Uint8Array | ArrayBuffer,
  row: number = 0
) {
  const typedArrayName = ArrayBuffer.isView(input)
    ? Object.getOwnPropertyDescriptor(
        Object.getPrototypeOf(Uint8Array.prototype),
        Symbol.toStringTag
      )!.get!.call(input)
    : undefined;
  let isBuffer = false;
  try {
    // The intrinsic getter recognizes foreign buffers without trusting a spoofable tag.
    Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, 'byteLength')!.get!.call(input);
    isBuffer = true;
  } catch {
    // Typed-array readbacks are admitted separately by their intrinsic element tag above.
  }
  validateRequirement(
    id,
    isBuffer || typedArrayName === 'Uint32Array' || typedArrayName === 'Uint8Array',
    'status must be Uint32Array, Uint8Array or ArrayBuffer'
  );
  const byteLength = input.byteLength;
  validateRequirement(
    id,
    byteLength % 4 === 0,
    `status byte length must contain whole uint32 words; received ${byteLength}`
  );
  validateCapacity(id, 'row', row, 0);
  validateRequirement(
    id,
    byteLength >= (row + 1) * 4,
    `status must contain row ${row}; received ${byteLength / 4} words`
  );
  // Read only the requested word, including unaligned byte subviews and foreign realms.
  const bytes = isBuffer
    ? new Uint8Array(input as ArrayBuffer, row * 4, 4)
    : new Uint8Array(
        (input as Uint32Array | Uint8Array).buffer,
        (input as Uint32Array | Uint8Array).byteOffset + row * 4,
        4
      );
  const code = getPackedUint32Values(bytes, 1)[0];
  validateCapacity(id, 'status code', code, 0, 3);
  return {
    code,
    complete: Number(code === 0),
    invalidInput: Number(code === 1),
    singular: Number(code === 2),
    candidateOverflow: Number(code === 3)
  };
}
