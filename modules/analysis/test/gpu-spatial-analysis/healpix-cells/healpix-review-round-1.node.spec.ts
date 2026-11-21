// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {WgslReflect} from 'wgsl_reflect';
import {expect, it} from 'vitest';
import {
  createHealpixDGGSJSON,
  createHealpixDGGSJSONFromReadback
} from '../../../src/gpu-spatial-analysis/index';
import {GPUHealpixPointToCellOperation} from '../../../src/gpu-spatial-analysis/healpix-cells/gpu-healpix-point-to-cell-operation';
import {
  CPUHealpixPointToCellOperation,
  computeHealpixPointToCellOnCPU
} from '../../../src/gpu-spatial-analysis/healpix-cells/cpu-healpix-point-to-cell-operation';
import {HEALPIX_INDEX_WGSL} from '../../../src/gpu-spatial-analysis/healpix-cells/healpix-index-wgsl';

it('B3 refuses ambiguous property byte reads while preserving sparse numeric bytes', () => {
  const values = new Uint8Array(Uint32Array.of(513, 257).buffer);
  const props = {
    order: 1,
    properties: {count: {values}},
    output: {cells: Uint32Array.of(1, 0, 2, 0), status: {count: 2, requiredCount: 2, overflow: 0}}
  };
  expect(() => createHealpixDGGSJSONFromReadback(props)).toThrow(
    /property count.*Uint8Array.*decoded/
  );
  expect(
    createHealpixDGGSJSONFromReadback({
      ...props,
      properties: {count: {values: Uint32Array.of(513, 257)}}
    }).values['count'][0].data
  ).toEqual([null, 513, 257, null]);
  // Explicit sparse values keep numeric array semantics: these are values 1 and 2, not uint32 bytes.
  expect(
    createHealpixDGGSJSON({order: 1, cells: [1n, 2n], properties: {count: {values}}}).values[
      'count'
    ][0].data
  ).toEqual([null, 1, 2, null]);
});

it('earlier fix: HEALPix named forms reject supplied empty mask and validity names', () => {
  const props = {
    order: 1,
    positions: 'positions',
    rowCount: 1,
    output: {cells: 'cells', validity: 'validity'}
  };
  for (const Operation of [GPUHealpixPointToCellOperation, CPUHealpixPointToCellOperation]) {
    expect(() => new Operation({...props, mask: ''})).toThrow(/mask.*non-empty/);
    expect(() => new Operation({...props, output: {cells: 'cells', validity: ''}})).toThrow(
      /validity.*non-empty/
    );
  }
});

it('earlier fix: HEALPix keeps the west side when reciprocal turn division rounds upward', () => {
  // Both angles are exactly 360*4 apart. At order 11: face 4, x=1023, y=1024,
  // interleaved path 2446677; 4*4^11 + path = 19223893.
  expect(
    computeHealpixPointToCellOnCPU(Float32Array.of(1799.9998779296875, 0, 359.9998779296875, 0), {
      order: 11
    }).cells
  ).toEqual(Uint32Array.of(19223893, 0, 19223893, 0));
  const wrapped = Math.fround(
    1799.9998779296875 -
      Math.fround(360 * Math.floor(Math.fround(1799.9998779296875 * Math.fround(1 / 360))))
  );
  expect(wrapped).toBe(-0.0001220703125);
  // Check the shader repairs a negative residual by its sign bits before turn division.
  const body = HEALPIX_INDEX_WGSL.slice(
    HEALPIX_INDEX_WGSL.indexOf('let reduced = cellIndexHealpixReduceDegrees')
  );
  expect(body.slice(0, body.indexOf('var tt'))).toMatch(/bitcast<u32>\(wrapped\)/);
  expect(() => new WgslReflect(HEALPIX_INDEX_WGSL)).not.toThrow();
});
