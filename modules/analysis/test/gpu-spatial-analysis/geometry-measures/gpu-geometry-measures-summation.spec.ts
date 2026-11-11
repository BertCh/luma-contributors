// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUGeometryMeasures} from '../../../src/gpu-spatial-analysis/geometry-measures/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// Math correctness review 2026-11-10, follow-up on compensated sums. The shader compiler of the
// test GPU (Metal, fast math) folds the float Neumaier correction `(sum - total) + value` to
// zero, so a "compensated" f32 sum is the plain f32 sum. This spec shows it on the two kinds of
// sum the kernel keeps: lengths (non-negative terms) and shoelace cross products (signed terms).
//
// The ring walks the triangle A, B, C many times. In coordinates relative to its first vertex A
// the triangle is (0, 0), (4001, 0), (0, 3989), so every loop adds the same three edge lengths
// and one cross product, 4001 * 3989 = 15,959,989, which f32 holds exactly. The binary64 sums are
// therefore known in closed form: the area is loops * 15,959,989 / 2 and the perimeter is
// loops * (4001 + 3989 + hypot(4001, 3989)). A plain f32 running sum of 100,000 loops is 1.2e-3
// off for the area and 1.1e-3 off for the perimeter (scratch script sum-probe.mjs); a sum that
// carries the rounding error of every addition is within 1e-6.
const ANCHOR = [100000, 200000];
const FIRST_LEG = 4001;
const SECOND_LEG = 3989;

function createWoundTriangle(loops: number): Float32Array {
  const positions = new Float32Array(6 * loops);
  for (let loop = 0; loop < loops; loop++) {
    positions.set(
      [ANCHOR[0], ANCHOR[1], ANCHOR[0] + FIRST_LEG, ANCHOR[1], ANCHOR[0], ANCHOR[1] + SECOND_LEG],
      6 * loop
    );
  }
  return positions;
}

async function measureWoundTriangle(loops: number, cooperativeRingRows: number) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const positions = createWoundTriangle(loops);
  const rowCount = positions.length / 2;
  const buffers: Buffer[] = [];
  const track = (buffer: Buffer) => {
    buffers.push(buffer);
    return buffer;
  };
  const graph = new GPUCommandGraph(device, {id: 'geometry-measures-summation'});
  const importOffsets = (name: string, values: number[]) =>
    importGraphBuffer(
      graph,
      name,
      track(createInputBuffer(device, new Uint32Array(values))),
      'uint32',
      values.length
    );
  const lengths = track(createOutputBuffer(device, 1));
  const areas = track(createOutputBuffer(device, 1));
  graph.add(
    new GPUGeometryMeasures({
      geometryType: 'polygons',
      positions: importGraphBuffer(
        graph,
        'positions',
        track(createInputBuffer(device, positions)),
        'float32x2',
        rowCount
      ),
      ringOffsets: importOffsets('ring-offsets', [0, rowCount]),
      polygonOffsets: importOffsets('polygon-offsets', [0, 1]),
      featureOffsets: importOffsets('feature-offsets', [0, 1]),
      cooperativeRingRows,
      output: {
        lengths: importGraphBuffer(graph, 'lengths', lengths, 'float32', 1),
        areas: importGraphBuffer(graph, 'areas', areas, 'float32', 1)
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    return {
      length: (await readFloat32(lengths, 1))[0],
      area: (await readFloat32(areas, 1))[0]
    };
  } finally {
    compiled.destroy();
    for (const buffer of buffers) {
      buffer.destroy();
    }
  }
}

function getExpected(loops: number) {
  return {
    length: loops * (FIRST_LEG + SECOND_LEG + Math.hypot(FIRST_LEG, SECOND_LEG)),
    area: (loops * FIRST_LEG * SECOND_LEG) / 2
  };
}

it('GPUGeometryMeasures serial sums carry the rounding error of every addition', async () => {
  // One invocation walks all 300,000 vertices (the ring is below cooperativeRingRows).
  const loops = 100000;
  const result = await measureWoundTriangle(loops, 4000000);
  const expected = getExpected(loops);
  expect(Math.abs(result.length / expected.length - 1), 'perimeter').toBeLessThan(2e-6);
  expect(Math.abs(result.area / expected.area - 1), 'area').toBeLessThan(2e-6);
}, 120000);

it('GPUGeometryMeasures cooperative lane sums carry the rounding error of every addition', async () => {
  // 64 lanes of 60,000 vertices each: a plain f32 lane sum of 20,000 loops is 1.1e-4 off for the
  // area and 2.8e-4 off for the perimeter.
  const loops = 1280000;
  const result = await measureWoundTriangle(loops, 512);
  const expected = getExpected(loops);
  expect(Math.abs(result.length / expected.length - 1), 'perimeter').toBeLessThan(2e-6);
  expect(Math.abs(result.area / expected.area - 1), 'area').toBeLessThan(2e-6);
}, 120000);
