// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getGPULineChunkParameterValues} from '../../../src/gpu-spatial-analysis/line-segmentize/gpu-line-chunk';
import {getGPULineLocateParameterValues} from '../../../src/gpu-spatial-analysis/linear-referencing/gpu-line-locate';

it('packs default line chunk settings byte-for-byte with no argument', () => {
  const omitted = getGPULineChunkParameterValues();
  const explicit = getGPULineChunkParameterValues({});
  expect(new Uint8Array(omitted.buffer)).toEqual(new Uint8Array(explicit.buffer));
  // Float32 encoding of the documented end-measure sentinel 3.4e38.
  expect(Array.from(new Uint32Array(omitted.buffer))).toEqual([0, 0, 0x7f7fc99e, 0]);
});

it('packs default line locate settings byte-for-byte with no argument', () => {
  const omitted = getGPULineLocateParameterValues();
  const explicit = getGPULineLocateParameterValues({});
  expect(new Uint8Array(omitted.buffer)).toEqual(new Uint8Array(explicit.buffer));
  expect(Array.from(omitted)).toEqual([1, 0, 0, 0]);
});
