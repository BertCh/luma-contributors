// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getMaskBindings,
  getMaskTestWGSL,
  getMaskView,
  getRowAccessorWGSL
} from '../../src/analysis-core/kernels/row-access-wgsl';
import {SPHERE_DISTANCE_WGSL} from '../../src/analysis-core/kernels/sphere-distance-wgsl';
import {createNullWebGPUDevice, createVectorView} from './gpu-contributor-test-utils';
import {requireTestDevice} from './require-test-device';
import {createSeededRandom, getFloatWords, runWGSLModuleKernel} from './wgsl-module-harness';

// SPHERE_DISTANCE_WGSL against float64 great-circle angles, including the antipode guard of FX-17,
// and the row accessors and mask test of row-access-wgsl.ts.

const DEGREES = Math.PI / 180;

it('sphere haversine and chord forms match float64 and return pi at the antipode', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createSeededRandom(51);
  // Per case: delta latitude, delta longitude, cosine product (radians), chord.
  const cases: number[][] = [];
  const expectedAngles: number[] = [];
  for (let index = 0; index < 400; index++) {
    const latitude1 = (180 * random() - 90) * DEGREES;
    // Mix nearby, far and nearly antipodal pairs.
    const kind = index % 4;
    const latitude2 =
      kind === 3 ? -latitude1 + 1e-4 * (random() - 0.5) : (180 * random() - 90) * DEGREES;
    const deltaLongitude =
      kind === 0
        ? 1e-3 * (random() - 0.5)
        : kind === 3
          ? Math.PI - 1e-4 * random()
          : 2 * Math.PI * (random() - 0.5);
    const deltaLatitude = latitude2 - latitude1;
    const cosineProduct = Math.cos(latitude1) * Math.cos(latitude2);
    const term =
      Math.sin(deltaLatitude / 2) ** 2 + cosineProduct * Math.sin(deltaLongitude / 2) ** 2;
    const angle = 2 * Math.atan2(Math.sqrt(term), Math.sqrt(1 - term));
    cases.push([deltaLatitude, deltaLongitude, cosineProduct, 2 * Math.sin(angle / 2)]);
    expectedAngles.push(angle);
  }
  // Exact antipodes and coincident points: h = 1 and chord = 2 must give pi, h = 0 gives 0.
  cases.push([Math.PI, 0, 1, 2], [0, Math.PI, 1, 2], [0, 0, 1, 0]);
  expectedAngles.push(Math.PI, Math.PI, 0);
  const words = await runWGSLModuleKernel(device, {
    id: 'sphere-distance',
    inputs: {cases: Float32Array.from(cases.flat())},
    outputLength: cases.length * 5,
    invocationCount: cases.length,
    declarations: SPHERE_DISTANCE_WGSL,
    body: `let base = casesOffset + index * 4u;
  let term = getSphereHaversineTerm(cases[base], cases[base + 1u], cases[base + 2u]);
  let angle = getSphereAngleFromHaversineTerm(term);
  let outputBase = outputOffset + index * 5u;
  output[outputBase] = bitcast<u32>(angle);
  output[outputBase + 1u] = bitcast<u32>(getSphereAngleFromChord(cases[base + 3u]));
  output[outputBase + 2u] = bitcast<u32>(getSphereHaversineTermFromAngle(angle) - term);
  output[outputBase + 3u] = bitcast<u32>(getSphereChordFromAngle(angle));
  output[outputBase + 4u] = bitcast<u32>(getSphereDistanceFromHaversineTerm(term, 6371008.8));`
  });
  const floats = getFloatWords(words);
  const mismatches: string[] = [];
  expectedAngles.forEach((expected, index) => {
    const [angle, chordAngle, termDifference, chord, distance] = floats.slice(
      index * 5,
      index * 5 + 5
    );
    // Float32 inputs carry 6e-8 relative error, which near the antipode moves the angle by about
    // sqrt of that: allow 1e-3 there and 2e-6 elsewhere.
    const tolerance = expected > 3.1 ? 1e-3 : 2e-6 + 2e-7 * expected;
    if (
      !(Math.abs(angle - expected) <= tolerance) ||
      !(Math.abs(chordAngle - expected) <= tolerance)
    ) {
      mismatches.push(
        `case ${index}: angle ${angle} chord angle ${chordAngle} expected ${expected}`
      );
    }
    if (
      !(Math.abs(termDifference) <= 1e-6) ||
      !(Math.abs(chord - 2 * Math.sin(angle / 2)) <= 1e-6)
    ) {
      mismatches.push(`case ${index}: round trip ${termDifference} chord ${chord}`);
    }
    if (!(Math.abs(distance - angle * 6371008.8) <= 1e-6 * 6371008.8 * Math.max(angle, 1e-3))) {
      mismatches.push(`case ${index}: distance ${distance}`);
    }
  });
  expect(mismatches.slice(0, 10)).toEqual([]);
  // FX-17: the antipode guard returns +pi exactly (as float32), never -pi/2 or NaN.
  const antipodes = cases.length - 3;
  expect(Array.from(floats.slice(antipodes * 5, antipodes * 5 + 2))).toEqual([
    Math.fround(Math.PI),
    Math.fround(Math.PI)
  ]);
  expect(floats[(antipodes + 1) * 5]).toBe(Math.fround(Math.PI));
  expect(floats[(antipodes + 2) * 5]).toBe(0);
});

it('row accessors read and write interleaved rows and the mask test selects rows', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = 8;
  // Three components with a stride of four, as a vec3 table padded to 16 bytes.
  const positions = Float32Array.from({length: rowCount * 4}, (_, index) => index + 0.5);
  const labels = Uint32Array.from({length: rowCount}, (_, index) => 100 + index);
  const maskWords = Uint32Array.from({length: rowCount}, (_, index) => (index % 3 === 0 ? 0 : 7));
  const accessors = `${getRowAccessorWGSL('positions', 3, {stride: 4})}
${getRowAccessorWGSL('labels', 1, {scalarType: 'u32'})}
${getRowAccessorWGSL('output', 2, {scalarType: 'u32', writable: true, functionStem: 'Result'})}`;
  const body = `let position = readPositions(index);
  writeResult(index, vec2<u32>(bitcast<u32>(position.x + position.y + position.z), select(0u, readLabels(index), isRowSelected(index))));`;
  const masked = await runWGSLModuleKernel(device, {
    id: 'rows-masked',
    inputs: {positions, labels, mask: maskWords},
    outputLength: rowCount * 2,
    invocationCount: rowCount,
    declarations: `${accessors}${getMaskTestWGSL(createTransientView(new GPUCommandGraph(createNullWebGPUDevice(), {id: 'presence'}), 'mask', 'uint32', 1))}`,
    body
  });
  const unmasked = await runWGSLModuleKernel(device, {
    id: 'rows-unmasked',
    inputs: {positions, labels},
    outputLength: rowCount * 2,
    invocationCount: rowCount,
    declarations: `${accessors}${getMaskTestWGSL(undefined)}`,
    body
  });
  const floats = getFloatWords(masked);
  for (let row = 0; row < rowCount; row++) {
    expect(floats[2 * row]).toBe(
      positions[4 * row] + positions[4 * row + 1] + positions[4 * row + 2]
    );
    expect(masked[2 * row + 1]).toBe(row % 3 === 0 ? 0 : 100 + row);
    expect(unmasked[2 * row + 1]).toBe(100 + row);
  }
});

it('mask helpers bind only present masks and reject chunked masks', () => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice(), {id: 'mask-helpers'});
  const first = createTransientView(graph, 'first', 'uint32', 4);
  const second = createTransientView(graph, 'second', 'uint32', 4);
  expect(getMaskView('GPUDemo', undefined)).toBeUndefined();
  expect(getMaskView('GPUDemo', first)).toBe(first);
  expect(getMaskView('GPUDemo', createVectorView('one-chunk', 'uint32', [first]))).toBe(first);
  expect(() =>
    getMaskView('GPUDemo', createVectorView('two-chunks', 'uint32', [first, second]))
  ).toThrow(/^GPUDemo mask must be one packed view, got 2 chunks/);
  expect(getMaskBindings(undefined)).toEqual([]);
  expect(getMaskBindings(first, 'selection')).toEqual([
    {name: 'selection', view: first, type: 'u32', access: 'read'}
  ]);
  expect(() => getRowAccessorWGSL('positions', 5)).toThrow(
    /componentCount must be an integer in \[1, 4\]/
  );
  expect(() => getRowAccessorWGSL('positions', 2, {stride: 1})).toThrow(
    /stride must be an integer/
  );
});
