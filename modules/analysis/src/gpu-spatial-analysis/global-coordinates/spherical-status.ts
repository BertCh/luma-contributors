// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  getPackedUint32Values,
  type GPUPackedReadbackValues
} from '../../analysis-core/kernels/packed-readback';
import {AnalysisValidationError} from '../../analysis-core/kernels/analysis-validation-error';
import {
  validateCapacity,
  validateRequirement
} from '../../analysis-core/kernels/gpu-contributor-utils';

/** Packed uint32 values or the bytes returned by one buffer read. */
export type SphericalReadback = GPUPackedReadbackValues | {bytes: Uint8Array | ArrayBuffer};

/** Count and status ports for one paired convex-intersection row. */
export type GPUSphericalConvexIntersectionReadbackInput = {
  counts: SphericalReadback;
  status: SphericalReadback;
};

/** Count and status ports for one whole intrinsic arc-overlay result. */
export type GPUSphericalArcOverlayReadbackInput = {
  counts: SphericalReadback;
  status: SphericalReadback;
};

function getSphericalWords(
  name: string,
  values: SphericalReadback,
  minimumLength: number,
  exactLength = false
): Uint32Array {
  const input = values && typeof values === 'object' && 'bytes' in values ? values.bytes : values;
  try {
    let numericWords: number[] | undefined;
    const isNumeric =
      input &&
      typeof input === 'object' &&
      'length' in input &&
      Object.prototype.toString.call(input) !== '[object Uint8Array]';
    // Capture only consumed words before conversion, preserving malformed values and getters.
    if (isNumeric && Number.isSafeInteger(input.length) && input.length >= minimumLength) {
      if (exactLength && input.length !== minimumLength) {
        throw new AnalysisValidationError(
          name,
          `expected exactly ${minimumLength} uint32 words; received ${input.length}`,
          {
            code: 'VIEW_LENGTH',
            path: 'values',
            received: input.length,
            expected: {words: minimumLength}
          }
        );
      }
      numericWords = Array.from({length: minimumLength}, (_, index) => input[index]);
      for (const [index, word] of numericWords.entries()) {
        validateCapacity(name, `words[${index}]`, word, 0, 0xffffffff);
      }
    }
    const words = getPackedUint32Values(numericWords ?? input, minimumLength);
    if (exactLength && words.length !== minimumLength) {
      throw new AnalysisValidationError(
        name,
        `expected exactly ${minimumLength} uint32 words; received ${words.length}`,
        {
          code: 'VIEW_LENGTH',
          path: 'values',
          received: words.length,
          expected: {words: minimumLength}
        }
      );
    }
    return words;
  } catch (error) {
    if (error instanceof AnalysisValidationError) {
      if (error.id === name) throw error;
      throw new AnalysisValidationError(
        name,
        `expected at least ${minimumLength} uint32 words; ${error.message}`,
        {code: error.code, path: error.path, received: error.received, expected: error.expected}
      );
    }
    throw error;
  }
}

function getContainmentStatus(name: string, input: number | SphericalReadback): number {
  const status =
    typeof input === 'object' && input !== null
      ? getSphericalWords(name, input, 1, true)[0]
      : input;
  // Containment producers publish exactly 0, 1 or 2, rather than arbitrary bit sets.
  validateRequirement(
    name,
    typeof status === 'number' && Number.isInteger(status) && status >= 0 && status <= 2,
    `status must be 0 (valid), 1 (invalidInput), or 2 (boundary); received ${String(status)}`
  );
  return status as number;
}

/** Named interpretation of `GPUSphericalConvexContains.output.status` (0/1/2). */
export function decodeGPUSphericalConvexContainsStatus(input: number | SphericalReadback) {
  const status = getContainmentStatus('decodeGPUSphericalConvexContainsStatus', input);
  return {
    valid: (status & 1) === 0,
    invalidInput: (status & 1) !== 0,
    boundary: (status & 2) !== 0
  };
}

/** Named interpretation of `GPUSphericalPolygonContains.output.status` (0/1/2). */
export function decodeGPUSphericalPolygonContainsStatus(input: number | SphericalReadback) {
  const status = getContainmentStatus('decodeGPUSphericalPolygonContainsStatus', input);
  return {
    valid: (status & 1) === 0,
    invalidInput: (status & 1) !== 0,
    boundary: (status & 2) !== 0
  };
}

/** Named bits of `GPUSphericalConvexIntersection.output.status`; zero means a complete result. */
export function decodeGPUSphericalConvexIntersectionStatus(status: number) {
  return {
    complete: status === 0,
    invalidInput: (status & 1) !== 0,
    outputVertexCapacityExceeded: (status & 2) !== 0,
    numericallyAmbiguous: (status & 4) !== 0
  };
}

/**
 * Named bits of `GPUSphericalArcOverlay.output.status`; capacity covers vertices and rings.
 * Bit 4 covers short split pieces, insufficient side separation, ambiguous membership or ring
 * junction/assembly failures. It does not identify which numerical cause occurred.
 */
export function decodeGPUSphericalArcOverlayStatus(status: number) {
  return {
    complete: status === 0,
    invalidInput: (status & 1) !== 0,
    outputCapacityExceeded: (status & 2) !== 0,
    numericallyUnresolved: (status & 4) !== 0
  };
}

/** Decode one convex-intersection row's `(written, required)` vertex counts. */
export function decodeGPUSphericalConvexIntersectionCounts(input: SphericalReadback) {
  const counts = getSphericalWords('decodeGPUSphericalConvexIntersectionCounts', input, 2);
  return {writtenVertexCount: counts[0], requiredVertexCount: counts[1]};
}

/** Decode the intrinsic overlay's four packed counts; required segments are not a ring count. */
export function decodeGPUSphericalArcOverlayCounts(input: SphericalReadback) {
  const counts = getSphericalWords('decodeGPUSphericalArcOverlayCounts', input, 4);
  return {
    writtenVertexCount: counts[0],
    writtenRingCount: counts[1],
    requiredSegmentCount: counts[2],
    referenceInside: counts[3] !== 0
  };
}

/** Decode one row's readback ports together; incomplete counts never imply success. */
export function decodeGPUSphericalConvexIntersectionReadback(
  input: GPUSphericalConvexIntersectionReadbackInput
) {
  return {
    counts: decodeGPUSphericalConvexIntersectionCounts(input.counts),
    status: decodeGPUSphericalConvexIntersectionStatus(
      getSphericalWords('decodeGPUSphericalConvexIntersectionReadback', input.status, 1, true)[0]
    )
  };
}

/** Decode the whole overlay's readback ports together, preserving all loss bits. */
export function decodeGPUSphericalArcOverlayReadback(input: GPUSphericalArcOverlayReadbackInput) {
  return {
    counts: decodeGPUSphericalArcOverlayCounts(input.counts),
    status: decodeGPUSphericalArcOverlayStatus(
      getSphericalWords('decodeGPUSphericalArcOverlayReadback', input.status, 1, true)[0]
    )
  };
}
