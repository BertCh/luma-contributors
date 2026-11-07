// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUPointCloudSMRF} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import oracle from './fixtures/smrf.json';

for (const scenario of oracle.fixtures) {
  it(`SMRF ${scenario.name} matches independent morphology/inpainting and classification`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const positions = fixture.view('float32x3', new Float32Array(scenario.positions));
      const classifications = fixture.view('uint32', new Uint32Array(scenario.classifications));
      const selection = fixture.view('uint32', new Uint32Array(scenario.selection));
      const groundSurface = fixture.view(
        'float32',
        new Float32Array(scenario.width * scenario.height)
      );
      const groundMask = fixture.view('uint32', new Uint32Array(scenario.classifications.length));
      const outputClassifications = fixture.view(
        'uint32',
        new Uint32Array(scenario.classifications.length)
      );
      const status = fixture.view('uint32', new Uint32Array(scenario.classifications.length));
      fixture.graph.add(
        new GPUPointCloudSMRF({
          positions: positions.view,
          classifications: classifications.view,
          selection: selection.view,
          grid: {origin: [0, 0], cellSize: 1, width: scenario.width, height: scenario.height},
          slope: scenario.slope,
          scalar: scenario.scalar,
          threshold: scenario.threshold,
          window: scenario.window,
          cut: scenario.cut,
          groundSurface: groundSurface.view,
          groundMask: groundMask.view,
          outputClassifications: outputClassifications.view,
          status: status.view
        })
      );
      await fixture.run();
      const actual = await readIntegers(outputClassifications.buffer);
      expect(actual).toEqual(scenario.expectedClasses);
      if ('pdalClasses' in scenario) expect(actual).toEqual(scenario.pdalClasses);
      expect(await readIntegers(groundMask.buffer)).toEqual(
        scenario.expectedClasses.map(value => Number(value === 2))
      );
      const surface = await readFloats(groundSurface.buffer);
      scenario.groundSurface.forEach((value, index) =>
        expect(surface[index]).toBeCloseTo(value, 4)
      );
      expect((await readIntegers(status.buffer)).at(-1)).toBe(4);
      selection.buffer.write(new Uint32Array(scenario.selection.length));
      await fixture.run();
      expect(await readIntegers(outputClassifications.buffer)).toEqual(scenario.classifications);
      expect(await readIntegers(status.buffer)).toEqual(Array(scenario.selection.length).fill(4));
      expect(await readFloats(groundSurface.buffer)).toEqual(
        Array(scenario.width * scenario.height).fill(0)
      );
    } finally {
      fixture.destroy();
    }
  });
}

it('SMRF preserves original classifications with onlyGround and reports missing terrain', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view(
      'float32x3',
      new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 5])
    );
    const classifications = fixture.view('uint32', new Uint32Array([6, 6, 6, 6, 6]));
    const groundSurface = fixture.view('float32', new Float32Array(4));
    const groundMask = fixture.view('uint32', new Uint32Array(5));
    const outputClassifications = fixture.view('uint32', new Uint32Array(5));
    const status = fixture.view('uint32', new Uint32Array(5));
    fixture.graph.add(
      new GPUPointCloudSMRF({
        positions: positions.view,
        classifications: classifications.view,
        grid: {origin: [0, 0], cellSize: 1, width: 2, height: 2},
        window: 1,
        groundClass: 9,
        onlyGround: true,
        groundSurface: groundSurface.view,
        groundMask: groundMask.view,
        outputClassifications: outputClassifications.view,
        status: status.view
      })
    );
    await fixture.run();
    expect(await readIntegers(outputClassifications.buffer)).toEqual([9, 9, 9, 9, 6]);
    positions.buffer.write(
      new Float32Array([100, 0, 0, 100, 0, 0, 100, 0, 0, 100, 0, 0, 100, 0, 0])
    );
    await fixture.run();
    expect(await readIntegers(outputClassifications.buffer)).toEqual([6, 6, 6, 6, 6]);
    expect(await readIntegers(status.buffer)).toEqual([1, 1, 1, 1, 1]);
    expect(await readFloats(groundSurface.buffer)).toEqual([0, 0, 0, 0]);
  } finally {
    fixture.destroy();
  }
});
