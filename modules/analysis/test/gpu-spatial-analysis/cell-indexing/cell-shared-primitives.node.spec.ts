// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getDegreeSinCosWGSL} from '../../../src/gpu-spatial-analysis/cell-indexing/cell-index-wgsl';
import {cellToLatLng} from 'h3-js';
import {computePointToCellOnCPU} from '../../../src/gpu-spatial-analysis/cell-indexing/cpu-point-to-cell-operation';
import {
  getFaceBasisVectors,
  H3_INDEX_EXTENDED_WGSL
} from '../../../src/gpu-spatial-analysis/cell-indexing/h3-index-extended-wgsl';
import fixture from '../cell-topology/face-edge-identities-fixture.json';

it('retains literal H3 keys when CPU indexing shares face bases (DGGS-115)', () => {
  // h3-js fixtures identify pentagon centers and face-crossing cells independently.
  // Coarse/middle resolutions keep f32-rounded centers well inside the original cells.
  for (const row of fixture.h3.filter(row => ((BigInt(`0x${row.cell}`) >> 52n) & 15n) <= 5n)) {
    const [latitude, longitude] = cellToLatLng(row.cell);
    const resolution = Number((BigInt(`0x${row.cell}`) >> 52n) & 15n);
    const result = computePointToCellOnCPU(Float32Array.of(longitude, latitude), {
      family: 'h3',
      resolution
    });
    expect(BigInt(result.cells[0]) | (BigInt(result.cells[1]) << 32n)).toBe(
      BigInt(`0x${row.cell}`)
    );
    expect(result.validity[0]).toBe(1);
  }
});

it('keeps the cached face basis private and orthonormal', () => {
  const bases = getFaceBasisVectors();
  const original = bases[0][0];
  bases[0][0] = 123;
  const next = getFaceBasisVectors();
  expect(next[0][0]).toBe(original);
  // Each face has center, i-axis, j-axis: closed-form unit lengths and zero pairwise dots.
  for (let face = 0; face < 20; face++) {
    const axes = next.slice(3 * face, 3 * face + 3);
    for (let first = 0; first < 3; first++)
      for (let second = first; second < 3; second++) {
        const dot = axes[first].reduce((sum, value, index) => sum + value * axes[second][index], 0);
        expect(Number.isFinite(dot)).toBe(true);
        expect(dot).toBeCloseTo(first === second ? 1 : 0, 14);
      }
  }
});

it('extended H3 reduces large integer degrees before selecting a quadrant (earlier-fix note)', () => {
  expect(H3_INDEX_EXTENDED_WGSL).toContain(
    'let reducedInput = cellIndexH3ExtReduceDegrees(degrees);'
  );
  expect(H3_INDEX_EXTENDED_WGSL).toContain('round(reducedInput * (1.0 / 90.0))');
});

it('degree trig shares one quadrant mapping while preserving both polynomial orders (UNFINISHED)', () => {
  expect(getDegreeSinCosWGSL.toString().match(/let turn =/g)).toHaveLength(1);
  const s2 = getDegreeSinCosWGSL('s2Trig', 's2Reduce', 's2');
  const h3 = getDegreeSinCosWGSL('h3Trig', 'h3Reduce', 'h3');
  expect(s2).toContain('degrees / 90.0');
  expect(h3).toContain('degrees * (1.0 / 90.0)');
  expect(s2).not.toContain('39916800.0');
  expect(h3).toContain('39916800.0');
  expect(h3).toContain('479001600.0');
});
