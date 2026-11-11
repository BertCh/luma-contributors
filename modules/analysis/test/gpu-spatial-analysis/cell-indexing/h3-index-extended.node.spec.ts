// luma.gl
// SPDX-License-Identifier: MIT AND Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
// SPDX-FileCopyrightText: Copyright 2016-2024 Uber Technologies, Inc.

import {dggs} from '@luma.gl/shaders';
import {describe, expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {H3_INDEX_WGSL} from '../../../src/gpu-spatial-analysis/cell-indexing/h3-index-wgsl';
import {
  H3_INDEX_EXTENDED_WGSL,
  parseDggsFaceTable
} from '../../../src/gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl';
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

  // Math review luma-official-06: the face tables are read out of the text of the published
  // `dggs` shader, which is not a public contract. A layout change must fail loudly at import.
  it('parses 20 rows from each published dggs face table', () => {
    const centers = parseDggsFaceTable('dggs_h3_get_face_center_geo', 2);
    const azimuths = parseDggsFaceTable('dggs_h3_get_face_axis_azimuths', 3);
    expect(centers).toHaveLength(20);
    expect(azimuths).toHaveLength(20);
    // Published H3 face 0 centre (latitude, longitude in radians) and its first axis azimuth.
    expect(centers[0][0]).toBeCloseTo(0.8035826497189899, 12);
    expect(centers[0][1]).toBeCloseTo(1.248397419617396, 12);
    expect(azimuths[0][0]).toBeCloseTo(5.619958268523939, 12);
  });

  it('throws a named error when a dggs face table is missing or has the wrong shape', () => {
    // A function that the shader does not define: the old parser returned 0 rows without an error.
    expect(() => parseDggsFaceTable('dggs_h3_get_no_such_table', 2)).toThrow(
      /h3-index-extended: dggs shader table dggs_h3_get_no_such_table .*expected 20 rows/
    );
    const row = (values: string) => `    vec2f(${values}),\n`;
    const table = (rows: string) =>
      `fn dggs_test_table(face: u32) -> vec2f {\n  let values = array<vec2f, 20>(\n${rows}  );\n  return values[face];\n}`;
    // 19 rows.
    expect(() =>
      parseDggsFaceTable('dggs_test_table', 2, table(row('0.5, 1.5').repeat(19)))
    ).toThrow(/dggs_test_table has 19 rows.*expected 20 rows/);
    // 20 rows of the wrong width.
    expect(() =>
      parseDggsFaceTable('dggs_test_table', 3, table(row('0.5, 1.5').repeat(20)))
    ).toThrow(/dggs_test_table .*3 finite numbers/);
    // A row that does not parse as numbers.
    expect(() =>
      parseDggsFaceTable('dggs_test_table', 2, table(row('0.5, 1.5').repeat(19) + row('0.5, PI')))
    ).toThrow(/dggs_test_table .*2 finite numbers/);
    // The well-formed table parses.
    expect(
      parseDggsFaceTable('dggs_test_table', 2, table(row('0.5, 1.5').repeat(20)))
    ).toHaveLength(20);
  });
});
