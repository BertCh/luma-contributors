// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {
  validateUnitVector,
  GPU_SPHERICAL_UNIT_VECTOR_LENGTH_TOLERANCE,
  GPU_SPHERICAL_UNIT_VECTOR_SQUARED_LENGTH_TOLERANCE
} from '../../../src/gpu-spatial-analysis/global-coordinates/assert-unit-vector';
import {
  decodeGPUSphericalConvexContainsStatus,
  decodeGPUSphericalPolygonContainsStatus,
  decodeGPUSphericalConvexIntersectionStatus,
  decodeGPUSphericalArcOverlayStatus,
  decodeGPUSphericalConvexIntersectionCounts,
  decodeGPUSphericalArcOverlayCounts
} from '../../../src/gpu-spatial-analysis/global-coordinates/spherical-status';

it('keeps the strict host length threshold distinct from the inclusive kernel squared-length threshold', () => {
  expect(GPU_SPHERICAL_UNIT_VECTOR_LENGTH_TOLERANCE).toBe(1e-6);
  expect(GPU_SPHERICAL_UNIT_VECTOR_SQUARED_LENGTH_TOLERANCE).toBe(1e-4);
  expect(() => validateUnitVector([1, 0, 0])).not.toThrow();
  // Length error 7.5e-7 passes; squared-length error 1.50000056e-6 would fail a mistaken check.
  expect(() => validateUnitVector([1.00000075, 0, 0])).not.toThrow();
  for (const vector of [
    [NaN, 0, 0],
    [Infinity, 0, 0],
    [0, 0, 0]
  ])
    expect(() => validateUnitVector(vector)).toThrow(/finite unit vector/);
  // Hand calculation: length error = 2e-5, squared-length error = 4.00004e-5.
  // These vectors pass the kernel's looser check and fail the unchanged host check.
  for (const length of [1.00002, 0.99998]) {
    expect(Math.abs(length * length - 1)).toBeLessThanOrEqual(
      GPU_SPHERICAL_UNIT_VECTOR_SQUARED_LENGTH_TOLERANCE
    );
    expect(() => validateUnitVector([length, 0, 0])).toThrow(/finite unit vector/);
  }
});

it('names every spherical status bit without changing packed values', () => {
  for (const decode of [
    decodeGPUSphericalConvexContainsStatus,
    decodeGPUSphericalPolygonContainsStatus
  ]) {
    expect(decode(0)).toEqual({valid: true, invalidInput: false, boundary: false});
    expect(decode(1)).toEqual({valid: false, invalidInput: true, boundary: false});
    expect(decode(2)).toEqual({valid: true, invalidInput: false, boundary: true});
    expect(() => decode(3)).toThrow(/status must be/);
  }
  for (const bit of [0, 1, 2, 4, 7]) {
    expect(decodeGPUSphericalConvexIntersectionStatus(bit)).toEqual({
      complete: bit === 0,
      invalidInput: Boolean(bit & 1),
      outputVertexCapacityExceeded: Boolean(bit & 2),
      numericallyAmbiguous: Boolean(bit & 4)
    });
    expect(decodeGPUSphericalArcOverlayStatus(bit)).toEqual({
      complete: bit === 0,
      invalidInput: Boolean(bit & 1),
      outputCapacityExceeded: Boolean(bit & 2),
      numericallyUnresolved: Boolean(bit & 4)
    });
  }
  expect(decodeGPUSphericalConvexIntersectionCounts([3, 7])).toEqual({
    writtenVertexCount: 3,
    requiredVertexCount: 7
  });
  expect(decodeGPUSphericalArcOverlayCounts([9, 2, 12, 1])).toEqual({
    writtenVertexCount: 9,
    writtenRingCount: 2,
    requiredSegmentCount: 12,
    referenceInside: true
  });
  expect(decodeGPUSphericalArcOverlayCounts([0, 0, 0, 0]).referenceInside).toBe(false);
});

it('OUT-02 decodes bounded raw-byte subviews and explicit bytes as words', () => {
  // Synthetic endianness probe, not a possible convex producer result: written 257 > required 2.
  // Little-endian 257 is [1, 1, 0, 0]; the prefix/suffix are outside the readback.
  const storage = new Uint8Array(23);
  const view = new DataView(storage.buffer);
  [257, 2, 300, 1].forEach((word, index) => view.setUint32(3 + index * 4, word, true));
  const bytes = storage.subarray(3, 19);
  for (const input of [bytes, {bytes}, bytes.slice().buffer]) {
    expect(decodeGPUSphericalConvexIntersectionCounts(input)).toEqual({
      writtenVertexCount: 257,
      requiredVertexCount: 2
    });
    expect(decodeGPUSphericalArcOverlayCounts(input)).toEqual({
      writtenVertexCount: 257,
      writtenRingCount: 2,
      requiredSegmentCount: 300,
      referenceInside: true
    });
  }
  for (const decode of [
    decodeGPUSphericalConvexIntersectionCounts,
    decodeGPUSphericalArcOverlayCounts
  ]) {
    expect(() => decode(bytes.subarray(0, 15))).toThrow(/whole.*4-byte|4-byte.*whole/);
  }
});

it('OUT-06 rejects incomplete counts and malformed containment status', () => {
  expect(() => decodeGPUSphericalConvexIntersectionCounts([3])).toThrow(
    /decodeGPUSphericalConvexIntersectionCounts.*2.*received 1/
  );
  expect(() => decodeGPUSphericalArcOverlayCounts([3, 1, 4])).toThrow(
    /decodeGPUSphericalArcOverlayCounts.*4.*received 3/
  );
  for (const decode of [
    decodeGPUSphericalConvexContainsStatus,
    decodeGPUSphericalPolygonContainsStatus
  ]) {
    for (const value of [undefined, NaN, 3, 4, '0', -1, 0.5]) {
      expect(() => decode(value as number)).toThrow(
        /decodeGPUSpherical.*ContainsStatus.*0.*1.*2.*received/
      );
    }
    expect(decode(Uint8Array.of(2, 0, 0, 0))).toEqual({
      valid: true,
      invalidInput: false,
      boundary: true
    });
    for (const value of [NaN, 0.5, 4])
      expect(() => decode([value])).toThrow(/status must be|words\[0\] must be/);
    expect(() => decode(new Uint32Array(0))).toThrow(/received 0/);
  }
});

it('B4 rejects malformed consumed numeric count words before uint32 coercion', () => {
  for (const value of [NaN, Infinity, -1, 0.5, 0x100000000]) {
    for (let index = 0; index < 2; index++) {
      const words = [0, 0];
      words[index] = value;
      expect(() => decodeGPUSphericalConvexIntersectionCounts(words)).toThrow(/words\[/);
    }
    for (let index = 0; index < 4; index++) {
      const words = [0, 0, 0, 0];
      words[index] = value;
      expect(() => decodeGPUSphericalArcOverlayCounts(words)).toThrow(/words\[/);
    }
  }
});

it('S2 requires exactly one containment status word in typed and raw readbacks', () => {
  for (const decode of [
    decodeGPUSphericalConvexContainsStatus,
    decodeGPUSphericalPolygonContainsStatus
  ]) {
    for (const input of [Uint32Array.of(0, 1), Uint8Array.of(0, 0, 0, 0, 1, 0, 0, 0), [0, 1]]) {
      expect(() => decode(input)).toThrow(/exactly 1.*received 2/);
    }
  }
});

it('S3 preserves structured layout details while naming the spherical decoder', () => {
  for (const decode of [
    decodeGPUSphericalConvexIntersectionCounts,
    decodeGPUSphericalArcOverlayCounts
  ]) {
    let error: unknown;
    try {
      decode([3]);
    } catch (caught) {
      error = caught;
    }
    expect(error).toMatchObject({code: 'VIEW_LENGTH', received: 1, path: 'values'});
    expect((error as Error).message).toContain(decode.name);
  }
});

it('S4 decodes separate valid producer count layouts from words and raw bytes', () => {
  // A complete triangular convex result has written = required = 3. A complete arc triangle
  // has one ring, three required segments and a reference point outside its region.
  const convex = Uint32Array.of(3, 3);
  const arc = Uint32Array.of(3, 1, 3, 0);
  for (const words of [convex, new Uint8Array(convex.buffer)])
    expect(decodeGPUSphericalConvexIntersectionCounts(words)).toEqual({
      writtenVertexCount: 3,
      requiredVertexCount: 3
    });
  for (const words of [arc, new Uint8Array(arc.buffer)])
    expect(decodeGPUSphericalArcOverlayCounts(words)).toEqual({
      writtenVertexCount: 3,
      writtenRingCount: 1,
      requiredSegmentCount: 3,
      referenceInside: false
    });
});
