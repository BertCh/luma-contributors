// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  getTypedArrayName,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Concrete CPU weights certificates. */
export type CPUWeightsStatus = {
  complete?: Uint32Array;
  overflow?: Uint32Array;
  candidateOverflow?: Uint32Array;
  count?: Uint32Array;
  requiredCount?: Uint32Array;
};
export type CPUWeightsLossInput = {
  status?: CPUWeightsStatus;
  offsets?: Uint32Array;
  neighbors?: Uint32Array;
  chunks?: readonly {weights: CPUWeightsLossInput}[];
};

const STATUS_FIELDS = [
  'complete',
  'overflow',
  'candidateOverflow',
  'count',
  'requiredCount'
] as const;
const RETIRED_FIELDS = [
  ...STATUS_FIELDS.map(name => [name, name] as const),
  ['weightsOverflow', 'overflow'],
  ['weightsCandidateOverflow', 'candidateOverflow']
] as const;

/** @internal Check supplied certificates and retained extent, including individual chunks. */
export function hasCPUWeightsLoss(id: string, input: CPUWeightsLossInput): boolean {
  for (const [retired, replacement] of RETIRED_FIELDS) {
    validateRequirement(
      id,
      Reflect.get(input, retired) === undefined,
      `weights.${retired} was removed; use weights.status.${replacement}`
    );
  }
  const {status, offsets, neighbors, chunks} = input;
  for (const name of STATUS_FIELDS) {
    const view = status?.[name];
    if (view === undefined) continue;
    validateRequirement(
      id,
      getTypedArrayName(view) === 'Uint32Array' && view.length >= 1,
      `weights.status.${name} must hold at least one uint32; received ${getTypedArrayName(view)}`
    );
  }
  let countLoss = false;
  if (status?.requiredCount) {
    const hasTerminalChunks = chunks?.every(chunk => Boolean(chunk.weights.offsets?.length));
    const terminal =
      offsets?.[offsets.length - 1] ??
      (hasTerminalChunks
        ? chunks!.reduce((total, chunk) => {
            const chunkOffsets = chunk.weights.offsets!;
            return total + chunkOffsets[chunkOffsets.length - 1];
          }, 0)
        : undefined);
    const retained = status.count?.[0];
    validateRequirement(
      id,
      terminal !== undefined || retained !== undefined,
      'weights.status.requiredCount needs concrete terminal offsets or a retained count'
    );
    const required = status.requiredCount[0];
    countLoss =
      (terminal !== undefined && required > terminal) ||
      (retained !== undefined && required > retained) ||
      (neighbors !== undefined && required > neighbors.length);
  }
  return (
    countLoss ||
    status?.complete?.[0] === 0 ||
    Boolean(status?.overflow?.[0] || status?.candidateOverflow?.[0]) ||
    Boolean(chunks?.some(chunk => hasCPUWeightsLoss(id, chunk.weights)))
  );
}
