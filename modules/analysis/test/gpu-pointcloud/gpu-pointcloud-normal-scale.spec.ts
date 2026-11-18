// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUPointCloudNormalScale,
  getGPUPointCloudNormalScaleParameterValues,
  GPUPointCloudChange
} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import oracle from '../oracles/fixtures/pointcloud-normal-scale.json';
import {GPUParameterBuffer} from '@luma-contributors/analysis/analysis-core';

for (const testCase of oracle.cases) {
  for (const indexed of [false, true]) {
    it(`selected normals feed M3C2 against NumPy: ${testCase.name}, indexed=${indexed}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = makeSpatialFixture(device);
      try {
        const reference = fixture.view('float32x3', Float32Array.from(testCase.reference.flat()));
        const normals = fixture.view('float32x3', new Float32Array(3), 12);
        const normalScales = fixture.view('float32', new Float32Array(1), 4);
        const normalVariances = fixture.view('float32', new Float32Array(1));
        const normalStatus = fixture.view('uint32', new Uint32Array(1));
        fixture.graph.add(
          new GPUPointCloudNormalScale({
            parameters: fixture.view('float32', new Float32Array([3, 0, 0, 0, 0])).view,
            positions: reference.view,
            queryRows: fixture.view('uint32', new Uint32Array([testCase.coreRow])).view,
            radii: testCase.radii,
            capacity: 64,
            ...(indexed
              ? {grid: {origin: [-2, -2, -2] as const, dimensions: [4, 4, 4] as const, cellSize: 1}}
              : {}),
            output: {
              normals: normals.view,
              normalScales: normalScales.view,
              normalVariances: normalVariances.view,
              status: normalStatus.view
            }
          })
        );
        const changes = fixture.view('float32x3', new Float32Array(3));
        const roughness = fixture.view('float32x3', new Float32Array(3));
        const counts = fixture.view('uint32', new Uint32Array(2));
        const status = fixture.view('uint32', new Uint32Array(1));
        fixture.graph.add(
          new GPUPointCloudChange({
            reference: reference.view,
            comparison: fixture.view('float32x3', Float32Array.from(testCase.comparison.flat()))
              .view,
            cores: fixture.view(
              'float32x3',
              Float32Array.from(testCase.reference[testCase.coreRow])
            ).view,
            normals: normals.view,
            alignment: 'already-aligned',
            radius: 1.1,
            halfDepth: 0.5,
            output: {
              changes: changes.view,
              roughness: roughness.view,
              counts: counts.view,
              status: status.view
            }
          })
        );
        await fixture.run();
        expect(await readIntegers(normalStatus.buffer)).toEqual([0]);
        expect((await readFloats(normalScales.buffer, 4))[0]).toBeCloseTo(
          testCase.selected.radius,
          6
        );
        const actualNormals = await readFloats(normals.buffer, 12);
        testCase.selected.normal.forEach((expected, axis) =>
          expect(actualNormals[axis]).toBeCloseTo(expected, 5)
        );
        expect((await readFloats(normalVariances.buffer))[0]).toBeCloseTo(
          testCase.selected.variance,
          6
        );
        expect(await readIntegers(counts.buffer)).toEqual(testCase.counts);
        expect(await readIntegers(status.buffer)).toEqual([0]);
        const actualChanges = await readFloats(changes.buffer);
        testCase.changes.forEach((expected, component) =>
          expect(actualChanges[component]).toBeCloseTo(expected, 5)
        );
        expect(actualChanges[0]).toBeCloseTo(testCase.knownOffset, 5);
        const actualRoughness = await readFloats(roughness.buffer);
        testCase.roughness.forEach((expected, component) =>
          expect(actualRoughness[component]).toBeCloseTo(expected, 5)
        );
      } finally {
        fixture.destroy();
      }
    });
  }
}

it('rejects overflowing scales, preserves complete smaller scales, and clears invalid rows and padding slots', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view(
      'float32x3',
      new Float32Array([
        0, 0, 0, 0.25, 0, 0, -0.25, 0, 0, 0, 0.25, 0, 0, -0.25, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0,
        -1, 0, 10, 10, 10
      ])
    );
    const normals = fixture.view('float32x3', new Float32Array(12).fill(99));
    const scales = fixture.view('float32', new Float32Array(4).fill(99));
    const variances = fixture.view('float32', new Float32Array(4).fill(99));
    const status = fixture.view('uint32', new Uint32Array(4).fill(99));
    fixture.graph.add(
      new GPUPointCloudNormalScale({
        parameters: fixture.view('float32', new Float32Array([3, 0, 0, 0, 0])).view,
        positions: positions.view,
        queryRows: fixture.view('uint32', new Uint32Array([0, 9, 0xffffffff, 1])).view,
        radii: [0.3, 2],
        capacity: 5,
        output: {
          normals: normals.view,
          normalScales: scales.view,
          normalVariances: variances.view,
          status: status.view
        }
      })
    );
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([2, 1, 1, 3]);
    expect(await readFloats(scales.buffer)).toEqual([Math.fround(0.3), 0, 0, 0]);
    expect(await readFloats(normals.buffer)).toEqual([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    expect(await readFloats(variances.buffer)).toEqual([0, 0, 0, 0]);
  } finally {
    fixture.destroy();
  }
});

it('rejects omitted ties on the radius boundary rather than selecting a truncated plane', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const normals = fixture.view('float32x3', new Float32Array(3).fill(99));
    const scales = fixture.view('float32', new Float32Array([99]));
    const variances = fixture.view('float32', new Float32Array([99]));
    const status = fixture.view('uint32', new Uint32Array([99]));
    fixture.graph.add(
      new GPUPointCloudNormalScale({
        parameters: fixture.view('float32', new Float32Array([3, 0, 0, 0, 0])).view,
        positions: fixture.view(
          'float32x3',
          new Float32Array([0, 0, 0, 0.25, 0, 0, -0.25, 0, 0, 0, 0.25, 0, 0, -0.25, 0])
        ).view,
        queryRows: fixture.view('uint32', new Uint32Array([0])).view,
        radii: [0.25],
        capacity: 3,
        output: {
          normals: normals.view,
          normalScales: scales.view,
          normalVariances: variances.view,
          status: status.view
        }
      })
    );
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([3]);
    expect(await readFloats(normals.buffer)).toEqual([0, 0, 0]);
    expect(await readFloats(scales.buffer)).toEqual([0]);
    expect(await readFloats(variances.buffer)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it('updates support eligibility and orientation on the same compiled graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  const parameters = new GPUParameterBuffer(device, {
    id: 'normal-scale-parameters',
    format: 'float32',
    length: 5,
    values: getGPUPointCloudNormalScaleParameterValues({viewpoint: [0, 0, -10]})
  });
  try {
    const normals = fixture.view('float32x3', new Float32Array(3));
    const scales = fixture.view('float32', new Float32Array(1));
    const variances = fixture.view('float32', new Float32Array(1));
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudNormalScale({
        positions: fixture.view(
          'float32x3',
          new Float32Array([0, 0, 0, 0.25, 0, 0, -0.25, 0, 0, 0, 0.25, 0, 0, -0.25, 0])
        ).view,
        queryRows: fixture.view('uint32', new Uint32Array([0])).view,
        parameters: parameters.importToGraph(fixture.graph),
        radii: [0.3],
        capacity: 8,
        output: {
          normals: normals.view,
          normalScales: scales.view,
          normalVariances: variances.view,
          status: status.view
        }
      })
    );
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([0]);
    expect((await readFloats(normals.buffer))[2]).toBe(-1);
    parameters.write(getGPUPointCloudNormalScaleParameterValues({minimumSupport: 5}));
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([1]);
    expect(await readFloats(normals.buffer)).toEqual([0, 0, 0]);
    expect(await readFloats(scales.buffer)).toEqual([0]);
    parameters.write(getGPUPointCloudNormalScaleParameterValues());
    await fixture.run();
    expect(await readIntegers(status.buffer)).toEqual([0]);
    expect((await readFloats(normals.buffer))[2]).toBe(1);
  } finally {
    fixture.destroy();
    parameters.destroy();
  }
});
