// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import type {GPUVector} from '@luma-contributors/graph/gpu-data';
import {runWideNumericKernel} from './wide-numeric-gpu-runner';

export const WIDE_NUMERIC_PROBE_SLOT_WORDS = 32;

export type WideNumericProbeCase = {
  addition: [number, number];
  subtraction: [number, number];
  negation: [number, number];
  lessUnsigned: boolean;
  lessSigned: boolean;
  multiplyLow: [number, number];
  multiply128: [number, number, number, number];
  divideUnsigned: {quotient: [number, number]; remainder: number};
  divideSigned: {
    quotient: [number, number];
    remainderMagnitude: number;
    remainderNegative: boolean;
  };
  unsignedFloat32: {bits: number; exact: boolean};
  signedFloat32: {bits: number; exact: boolean};
  leadingZeros: number;
  shiftRight: [number, number];
  shiftLeft: [number, number];
};

/**
 * Diagnostic probe: evaluates every WGSL helper on pairs (row 2i = operand a, row 2i+1 = operand b;
 * the divisor is b.x, the shift is b.x & 63). Used by the GPU spec to compare the WGSL helpers
 * word for word with the TypeScript emulation.
 */
export async function probeWideNumericHelpersOnGpu(props: {
  device: Device;
  words: GPUVector<'uint32x2'>;
  caseCount: number;
}): Promise<WideNumericProbeCase[]> {
  const slot = WIDE_NUMERIC_PROBE_SLOT_WORDS;
  const result = await runWideNumericKernel({
    device: props.device,
    id: 'wide-numeric-helper-probe',
    words: props.words,
    resultWordCount: props.caseCount * slot,
    phaseBodies: [
      `if ((index & 1u) != 0u) {return;}
  let a = pair;
  let b = vec2<u32>(words[wordOffset + 2u], words[wordOffset + 3u]);
  let base = (index / 2u) * ${slot}u;
  let addition = word64Add(a, b); atomicStore(&result[base], addition.x); atomicStore(&result[base + 1u], addition.y);
  let subtraction = word64Subtract(a, b); atomicStore(&result[base + 2u], subtraction.x); atomicStore(&result[base + 3u], subtraction.y);
  let negation = word64Negate(a); atomicStore(&result[base + 4u], negation.x); atomicStore(&result[base + 5u], negation.y);
  atomicStore(&result[base + 6u], select(0u, 1u, word64LessUnsigned(a, b)));
  atomicStore(&result[base + 7u], select(0u, 1u, word64LessSigned(a, b)));
  let low = word64MultiplyLow(a, b); atomicStore(&result[base + 8u], low.x); atomicStore(&result[base + 9u], low.y);
  let wide = word64Multiply128(a, b);
  atomicStore(&result[base + 10u], wide.x); atomicStore(&result[base + 11u], wide.y); atomicStore(&result[base + 12u], wide.z); atomicStore(&result[base + 13u], wide.w);
  let divided = word64DivideSmall(a, b.x);
  atomicStore(&result[base + 14u], divided.x); atomicStore(&result[base + 15u], divided.y); atomicStore(&result[base + 16u], divided.z);
  let dividedSigned = word64DivideSmallSigned(a, b.x);
  atomicStore(&result[base + 17u], dividedSigned.x); atomicStore(&result[base + 18u], dividedSigned.y); atomicStore(&result[base + 19u], dividedSigned.z); atomicStore(&result[base + 20u], dividedSigned.w);
  let unsignedFloat = word64ToFloat32Unsigned(a);
  atomicStore(&result[base + 21u], bitcast<u32>(unsignedFloat.value)); atomicStore(&result[base + 22u], unsignedFloat.exact);
  let signedFloat = word64ToFloat32Signed(a);
  atomicStore(&result[base + 23u], bitcast<u32>(signedFloat.value)); atomicStore(&result[base + 24u], signedFloat.exact);
  atomicStore(&result[base + 25u], word64LeadingZeros(a));
  let shiftedRight = word64ShiftRight(a, b.x & 63u); atomicStore(&result[base + 26u], shiftedRight.x); atomicStore(&result[base + 27u], shiftedRight.y);
  let shiftedLeft = word64ShiftLeft(a, b.x & 63u); atomicStore(&result[base + 28u], shiftedLeft.x); atomicStore(&result[base + 29u], shiftedLeft.y);`
    ]
  });
  return Array.from({length: props.caseCount}, (_, caseIndex) => {
    const slotWords = result.subarray(caseIndex * slot, (caseIndex + 1) * slot);
    return {
      addition: [slotWords[0], slotWords[1]],
      subtraction: [slotWords[2], slotWords[3]],
      negation: [slotWords[4], slotWords[5]],
      lessUnsigned: slotWords[6] === 1,
      lessSigned: slotWords[7] === 1,
      multiplyLow: [slotWords[8], slotWords[9]],
      multiply128: [slotWords[10], slotWords[11], slotWords[12], slotWords[13]],
      divideUnsigned: {quotient: [slotWords[14], slotWords[15]], remainder: slotWords[16]},
      divideSigned: {
        quotient: [slotWords[17], slotWords[18]],
        remainderMagnitude: slotWords[19],
        remainderNegative: slotWords[20] === 1
      },
      unsignedFloat32: {bits: slotWords[21], exact: slotWords[22] === 1},
      signedFloat32: {bits: slotWords[23], exact: slotWords[24] === 1},
      leadingZeros: slotWords[25],
      shiftRight: [slotWords[26], slotWords[27]],
      shiftLeft: [slotWords[28], slotWords[29]]
    };
  });
}
