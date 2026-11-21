// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getPackedUint32Values} from '../../analysis-core/kernels/packed-readback';
import {validateRequirement} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Explicit byte readback, bounded to the logical record by the caller. */
export type NetworkReadbackBytes = Uint8Array | ArrayBuffer | {bytes: Uint8Array | ArrayBuffer};

/** Check numeric uint32 words before decoding a network record. */
export function getNetworkReadbackWords(
  name: string,
  values: Uint32Array | readonly number[],
  minimumLength: number
): Uint32Array {
  validateRequirement(
    name,
    ![
      '[object Uint8Array]',
      '[object Int8Array]',
      '[object Uint8ClampedArray]',
      '[object ArrayBuffer]'
    ].includes(Object.prototype.toString.call(values)),
    'expected numeric words; use the FromBytes reader for byte readback'
  );
  const words = getPackedUint32Values(values);
  validateRequirement(
    name,
    words.length >= minimumLength,
    `${name.includes('Counts') ? 'counts' : 'summary'} requires at least ${minimumLength} words; received ${words.length}`
  );
  return words;
}

/** Use core's bounded little-endian conversion, including unaligned byte subviews. */
export function getNetworkReadbackBytes(
  name: string,
  values: NetworkReadbackBytes,
  minimumLength: number
): Uint32Array {
  const bytes = values && typeof values === 'object' && 'bytes' in values ? values.bytes : values;
  const tag = Object.prototype.toString.call(bytes);
  validateRequirement(
    name,
    tag === '[object ArrayBuffer]' || (ArrayBuffer.isView(bytes) && tag === '[object Uint8Array]'),
    'expected Uint8Array or ArrayBuffer; use the Words reader for numeric words'
  );
  const words = getPackedUint32Values(bytes);
  validateRequirement(
    name,
    words.length >= minimumLength,
    `${name.includes('Counts') ? 'counts' : 'summary'} requires at least ${minimumLength} words; received ${words.length}`
  );
  return words;
}
