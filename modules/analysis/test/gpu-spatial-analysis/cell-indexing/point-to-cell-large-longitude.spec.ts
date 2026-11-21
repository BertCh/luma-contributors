// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// DGGS-106: S2 and H3 point-to-cell document "longitude may be any finite value", but the f32
// quadrant reduction `degrees - round(degrees / 90) * 90` rounds `quadrant * 90` once the
// longitude passes 2^24: at 1e8 the effective longitude was 278 instead of 280, at 2^30 90
// instead of 64, at 1e30 0 instead of 120 (f32 transcription, Math.fround). Each large longitude
// below is paired with its exact residue modulo 360, signed like the input, from
//   /usr/bin/python3 -c "import numpy as np; from fractions import Fraction;
//     f = float(np.float32(x)); r = int(Fraction(f)) % 360; print(r if f > 0 else r - 360)"
// The large row must give the same key as its residue row, and the residue row the reference
// key (h3-js, and the binary64 S2 reference of the cell-indexing oracle).

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {latLngToCell} from 'h3-js';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUPointToCell,
  type GPUCellIndexFamily
} from '../../../src/gpu-spatial-analysis/cell-indexing/index';
import {h3ToBigInt} from '../cell-aggregation/cell-aggregation-oracle';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {joinKey, s2PointToCell} from './cell-indexing-oracle';
import {CLIPPED_LONGITUDES, CLIPPED_LONGITUDE_FIXTURES} from './clipped-longitude-fixtures';
import {requireTestDevice} from '../../utils/require-test-device';

/** [large longitude, its signed residue modulo 360, latitude]; every value is exact in float32. */
const LARGE_LONGITUDES: [number, number, number][] = [
  [100000000, 280, 10],
  [1073741824, 64, 60],
  [-3000000000, -120, -30],
  [1.0000000150474662e30, 120, 5],
  [-2.9000000188361036e38, -224, -50],
  [16777216, 136, 45]
];

async function getKeys(
  family: GPUCellIndexFamily,
  resolution: number,
  positions: Float32Array,
  precision?: 'float32' | 'double-single'
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rows = positions.length / 2;
  const positionBuffer = createInputBuffer(device, positions);
  const cellBuffer = createOutputBuffer(device, 2 * rows);
  const validityBuffer = createOutputBuffer(device, rows);
  const graph = new GPUCommandGraph(device, {id: `large-longitude-${family}-${resolution}`});
  graph.add(
    new GPUPointToCell({
      family,
      resolution,
      precision,
      positions: importGraphBuffer(graph, 'positions', positionBuffer, 'float32x2', rows),
      output: {
        cells: importGraphBuffer(graph, 'cells', cellBuffer, 'uint32x2', rows),
        validity: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', rows)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const words = await readUint32(cellBuffer, 2 * rows);
    const validity = await readUint32(validityBuffer, rows);
    return {
      keys: Array.from({length: rows}, (_, row) => joinKey(words[2 * row], words[2 * row + 1])),
      validity
    };
  } finally {
    compiled.destroy();
    for (const buffer of [positionBuffer, cellBuffer, validityBuffer]) buffer.destroy();
  }
}

for (const [family, resolution, getReference] of [
  ['s2', 10, (longitude: number, latitude: number) => s2PointToCell(longitude, latitude, 10)],
  [
    'h3',
    5,
    (longitude: number, latitude: number) => h3ToBigInt(latLngToCell(latitude, longitude, 5))
  ]
] as const) {
  it(`GPUPointToCell ${family} reduces longitudes beyond 2^24 exactly modulo 360`, async () => {
    // Rows 2k: the large longitude; rows 2k + 1: its residue.
    const positions = Float32Array.from(
      LARGE_LONGITUDES.flatMap(([large, residue, latitude]) => [large, latitude, residue, latitude])
    );
    for (let index = 0; index < positions.length; index++) {
      expect(positions[index], 'test longitudes must be exact float32 values').toBe(
        LARGE_LONGITUDES[Math.floor(index / 4)][[0, 2, 1, 2][index % 4]]
      );
    }
    const {keys, validity} = await getKeys(family, resolution, positions);
    expect(Array.from(validity)).toEqual(new Array(keys.length).fill(1));
    LARGE_LONGITUDES.forEach(([large, residue, latitude], pair) => {
      const reference = getReference(residue, latitude);
      expect(keys[2 * pair + 1].toString(16), `${family} residue ${residue}`).toBe(
        reference.toString(16)
      );
      expect(keys[2 * pair].toString(16), `${family} longitude ${large}`).toBe(
        reference.toString(16)
      );
    });
  });
}

// Clipping families intentionally differ from the modulo-360 families above.
for (const fixture of CLIPPED_LONGITUDE_FIXTURES) {
  it(`GPUPointToCell ${fixture.family} clips both signs beyond 2^24 to literal endpoint bins`, async () => {
    const {keys, validity} = await getKeys(fixture.family, fixture.resolution, CLIPPED_LONGITUDES);
    expect(keys).toEqual([fixture.negative, fixture.positive, fixture.negative, fixture.positive]);
    expect(Array.from(validity)).toEqual([1, 1, 1, 1]);
  });
}

for (const precision of ['float32', 'double-single'] as const) {
  it(`GPUPointToCell H3 ${precision} reduces 2^31 to its exact 128-degree residue`, async () => {
    // Exact integer modulo: 2147483648 = 360*5965232 + 128. h3-js 4.x at lat=7,
    // lng=128, resolution=5 gives 8573a403fffffff; the old extended quadrant gave
    // an effective 180 degrees and the unrelated key 857194c3fffffff.
    const {keys, validity} = await getKeys('h3', 5, Float32Array.of(2 ** 31, 7, 128, 7), precision);
    expect(keys).toEqual([0x8573a403fffffffn, 0x8573a403fffffffn]);
    expect(Array.from(validity)).toEqual([1, 1]);
  });
}
