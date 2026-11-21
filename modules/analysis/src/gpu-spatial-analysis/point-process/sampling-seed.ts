// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** A scalar seed is mandatory only when there is no resident seed source. @internal */
export function validateSamplingSeed(
  id: string,
  seed: number | undefined,
  resident: boolean
): void {
  validateRequirement(
    id,
    resident || seed !== undefined,
    `seed is required when parameters is absent; received ${seed}`
  );
  if (seed !== undefined) validateCapacity(id, 'seed', seed, 0, 0xffffffff);
}

/** Snapshot the effective stream words for recording; never substitute inactive scalar hints. @internal */
export function getSamplingReplayParameters(
  id: string,
  seed: number | undefined,
  replicate: number | undefined,
  resident: boolean,
  words?: Uint32Array
): Uint32Array {
  validateSamplingSeed(id, seed, resident);
  if (resident) {
    validateRequirement(
      id,
      words instanceof Uint32Array && words.length === 2,
      `parameters replay words must be Uint32Array [seed, replicate] of length 2; received ${words?.length}`
    );
    return words!.slice();
  }
  return Uint32Array.of(seed!, replicate ?? 0);
}
