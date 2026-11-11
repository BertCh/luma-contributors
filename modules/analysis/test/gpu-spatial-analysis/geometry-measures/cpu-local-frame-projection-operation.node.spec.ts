// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createGPULocalFrame} from '../../../src/analysis-core/contracts/local-frame';
import {
  computeLocalFrameProjectionOnCPU,
  CPULocalFrameProjectionOperation
} from '../../../src/gpu-spatial-analysis/geometry-measures/cpu-local-frame-projection-operation';
import {
  GPULocalFrameProjectionOperation,
  type LocalFrameProjectionOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-measures/gpu-local-frame-projection-operation';

const SPHERE_RADIUS = 6371000;
const FRAME = createGPULocalFrame({origin: [0, 0], sphereRadius: SPHERE_RADIUS});
const PROPS: LocalFrameProjectionOperationProps = {
  id: 'projection-test',
  positions: 'positions',
  rowCount: 3,
  frame: FRAME,
  bounds: [-2, -2, 2, 2],
  output: 'projected',
  validity: 'validity'
};

it('matches closed-form equatorial and meridional spherical distances', () => {
  const input = new Float32Array([0, 0, 1, 0, 0, 1]);
  const result = computeLocalFrameProjectionOnCPU(input, FRAME, PROPS.bounds);
  const oneDegree = (SPHERE_RADIUS * Math.PI) / 180;
  expect(Array.from(result.validity)).toEqual([1, 1, 1]);
  const expected = [0, 0, oneDegree, 0, 0, oneDegree];
  expected.forEach((value, index) =>
    expect(Math.abs(result.output[index] - value)).toBeLessThan(0.005)
  );
});

it('matches the ellipsoidal Lambert equatorial chord closed form', () => {
  const frame = createGPULocalFrame({origin: [0, 0], kind: 'lambert-equal-area'});
  const result = computeLocalFrameProjectionOnCPU(
    new Float32Array([-1, 0, 0, 0, 1, 0]),
    frame,
    PROPS.bounds
  );
  const chord = 2 * 6378137 * Math.sin(Math.PI / 360);
  expect(Math.abs(result.output[0] + chord)).toBeLessThan(0.005);
  expect(Math.abs(result.output[4] - chord)).toBeLessThan(0.005);
  expect(Array.from(result.output.filter((_, index) => index % 2 === 1))).toEqual([0, 0, 0]);
});

it('decodes little-endian binary64 word rows, including a nonzero byte offset', () => {
  const storage = new Float64Array([99, 99, 0.000000123456789, 0, 1, 0]);
  const words = new Uint32Array(storage.buffer, 2 * Float64Array.BYTES_PER_ELEMENT);
  const result = computeLocalFrameProjectionOnCPU(words, FRAME, PROPS.bounds);
  expect(result.output[0]).toBeCloseTo((SPHERE_RADIUS * storage[2] * Math.PI) / 180, 7);
  expect(result.output[2]).toBe(Math.fround((SPHERE_RADIUS * Math.PI) / 180));
  expect(Array.from(result.validity)).toEqual([1, 1]);
});

it('uses inclusive bounds and writes zero for outside/non-finite rows', () => {
  const result = computeLocalFrameProjectionOnCPU(
    new Float32Array([-2, 0, 2, 0, 3, 0, NaN, 0, 0, Infinity]),
    FRAME,
    PROPS.bounds
  );
  expect(Array.from(result.validity)).toEqual([1, 1, 0, 0, 0]);
  expect(Array.from(result.output.slice(4))).toEqual([0, 0, 0, 0, 0, 0]);
  expect(() => computeLocalFrameProjectionOnCPU(new Float32Array(3), FRAME, PROPS.bounds)).toThrow(
    /positions/
  );
  expect(() => computeLocalFrameProjectionOnCPU(new Uint32Array(3), FRAME, PROPS.bounds)).toThrow(
    /positions/
  );
  expect(() => computeLocalFrameProjectionOnCPU(new Float32Array(2), FRAME, [1, 0, -1, 1])).toThrow(
    /bounds/
  );
});

it('validates both operations at construction, with CPU numerical settings checked immediately', () => {
  for (const Operation of [GPULocalFrameProjectionOperation, CPULocalFrameProjectionOperation]) {
    expect(() => new Operation(PROPS)).not.toThrow();
    expect(() => new Operation({...PROPS, rowCount: 0})).toThrow(/^projection-test/);
    expect(() => new Operation({...PROPS, output: 'positions'})).toThrow(/^projection-test/);
    expect(() => new Operation({...PROPS, validity: 'projected'})).toThrow(/^projection-test/);
  }
  expect(() => new CPULocalFrameProjectionOperation({...PROPS, tolerance: -1})).toThrow(
    /^projection-test.*tolerance/
  );
  expect(() => new CPULocalFrameProjectionOperation({...PROPS, bounds: [1, 0, -1, 1]})).toThrow(
    /^projection-test.*bounds/
  );
});

it('rejects singular projected rows using the GPU zero/validity convention', () => {
  const result = computeLocalFrameProjectionOnCPU(
    new Float32Array([180, 0]),
    FRAME,
    [-180, -1, 180, 1]
  );
  expect(Array.from(result.output)).toEqual([0, 0]);
  expect(Array.from(result.validity)).toEqual([0]);
});

it('keeps the ordinary four-row headless fixture valid and maps its origin to zero', () => {
  for (const kind of ['azimuthal-equidistant', 'lambert-equal-area'] as const) {
    const frame = createGPULocalFrame({origin: [0, 0], kind});
    const coordinates = [0, 0, 0.5, 0.25, 0.123456789, 0.234567891, 1, 1];
    for (const positions of [
      new Float32Array(coordinates),
      new Uint32Array(new Float64Array(coordinates).buffer)
    ]) {
      const result = computeLocalFrameProjectionOnCPU(positions, frame, [-1, -1, 1, 1]);
      // All four rows are finite and inside inclusive bounds; the first equals the frame origin.
      expect(Array.from(result.validity)).toEqual([1, 1, 1, 1]);
      expect(Array.from(result.output.slice(0, 2))).toEqual([0, 0]);
    }
  }
});
