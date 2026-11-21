// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Shared compile-time options; speed is length units per timestamp unit. */
export type ModalSpeedOptions = {
  modeCount: number;
  /**
   * Exclusive speed upper bounds in length units per timestamp unit, encoded to float32.
   * The final class is unbounded. Trackintel's simple-coarse classes use
   * [15 / 3.6, 100 / 3.6, Infinity] in metres per second.
   */
  modeSpeedUpperBoundsFloat32?: readonly number[];
  /** Unknown supplied IDs enter the last bin by default; the missing key is always excluded. */
  unknownModePolicy?: 'last-bin' | 'exclude';
};

/** @internal Reject the retired key on the caller's original options. */
export function validateModalSpeedOptionName(id: string, options: object): void {
  validateRequirement(
    id,
    !('modeSpeedBreaks' in options),
    'modeSpeedBreaks was removed; use modeSpeedUpperBoundsFloat32'
  );
}

/** @internal Normalize the same encoded table in every backend and operation. */
export function resolveModalSpeedOptions(
  id: string,
  options: ModalSpeedOptions
): readonly number[] | undefined {
  validateCapacity(id, 'modeCount', options.modeCount);
  validateRequirement(
    id,
    options.unknownModePolicy === undefined ||
      options.unknownModePolicy === 'last-bin' ||
      options.unknownModePolicy === 'exclude',
    `unknownModePolicy must be last-bin or exclude; received ${options.unknownModePolicy}`
  );
  const values = options.modeSpeedUpperBoundsFloat32;
  if (!values) return undefined;
  validateRequirement(
    id,
    values.length === options.modeCount,
    `modeSpeedUpperBoundsFloat32 must hold modeCount = ${options.modeCount} values; received ${values.length}`
  );
  const encoded = values.map(Math.fround);
  validateRequirement(
    id,
    values.every(
      (value, index) =>
        Number.isFinite(encoded[index]) || (value === Infinity && index === values.length - 1)
    ),
    `modeSpeedUpperBoundsFloat32 must contain finite float32 values, with positive Infinity only at the end; received ${values}`
  );
  validateRequirement(
    id,
    encoded.every((value, index) => index === 0 || value > encoded[index - 1]),
    `modeSpeedUpperBoundsFloat32 must be strictly ascending after float32 encoding; received ${values}`
  );
  return Object.freeze(encoded);
}
