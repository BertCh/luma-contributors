// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

import {dggs} from '@luma.gl/shaders';
import {describe, expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {H3_INDEX_WGSL} from '../../../src/gpu-spatial-analysis/cell-indexing/h3-index-wgsl';
import {H3_INDEX_EXTENDED_WGSL} from '../../../src/gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl';
import {H3_FACE_BASES} from './h3-index-oracle';

describe('H3_INDEX_EXTENDED_WGSL source', () => {
  it('defines the contract function and parses together with the float32 shader', () => {
    expect(H3_INDEX_EXTENDED_WGSL).toContain(
      'fn cellIndexH3ExtendedFromLngLat(lngLatDegrees: vec2f, resolution: u32) -> vec2u'
    );
    expect(
      () => new WgslReflect(`${dggs.source}\n${H3_INDEX_WGSL}\n${H3_INDEX_EXTENDED_WGSL}`)
    ).not.toThrow();
  });

  it('uses only prefixed identifiers that do not collide with dggs or the float32 shader', () => {
    const names = [...H3_INDEX_EXTENDED_WGSL.matchAll(/^(?:fn|const|var)\s+(\w+)/gm)].map(
      match => match[1]
    );
    expect(names.length).toBeGreaterThan(15);
    for (const name of names) {
      expect(name.startsWith('cellIndexH3Ext') || name.startsWith('CELL_INDEX_H3_EXT_'), name).toBe(
        true
      );
    }
  });

  it('embeds double-float face bases that reconstruct the float64 bases', () => {
    const match = H3_INDEX_EXTENDED_WGSL.match(
      /CELL_INDEX_H3_EXT_FACE_BASES = array<vec2f, 180>\(([^;]*)\);/
    );
    const pairs = [...match![1].matchAll(/vec2f\(([^,]+), ([^)]+)\)/g)].map(
      pair => Number(pair[1]) + Number(pair[2])
    );
    expect(pairs).toHaveLength(180);
    for (let face = 0; face < 20; face++) {
      for (let axis = 0; axis < 3; axis++) {
        for (let component = 0; component < 3; component++) {
          expect(
            Math.abs(
              pairs[(face * 3 + axis) * 3 + component] - H3_FACE_BASES[face][axis][component]
            )
          ).toBeLessThan(1e-9);
        }
      }
    }
  });
});
