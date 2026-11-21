// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import * as core from '../../src/analysis-core/kernels/spatial-wgsl-point-helpers';
import * as compatibility from '../../src/gpu-spatial-analysis/shared/spatial-wgsl-point-helpers';

it('STD-09 keeps every old point-helper import bound to the core implementation', () => {
  expect(Object.keys(compatibility).sort()).toEqual(Object.keys(core).sort());
  for (const name of Object.keys(core) as (keyof typeof core)[])
    expect(compatibility[name]).toBe(core[name]);
});

it('STD-09 retains packed point readers, offsets and integer finite checks', () => {
  expect(core.getReadPointWGSL('readOrigin', 'origins')).toBe(
    'fn readOrigin(row: u32) -> vec2f { return vec2f(origins[originsOffset + row * 2u], origins[originsOffset + row * 2u + 1u]); }'
  );
  expect(core.getReadPoint3WGSL('readSite', 'sites')).toBe(
    'fn readSite(row: u32) -> vec3f { let base = sitesOffset + row * 3u; return vec3f(sites[base], sites[base + 1u], sites[base + 2u]); }'
  );
  expect(core.IS_SPATIAL_FINITE_WGSL).toContain(
    '(bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u'
  );
  expect(core.GET_SPATIAL_NAN_WGSL).toContain('var bits = 0x7fc00000u; return bitcast<f32>(bits)');
});
