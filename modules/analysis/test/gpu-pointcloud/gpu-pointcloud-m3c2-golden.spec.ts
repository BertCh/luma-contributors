// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUPointCloudChange,
  GPU_POINT_CLOUD_CHANGE_STATUS_INSUFFICIENT_SUPPORT
} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import reference from '../oracles/fixtures/pdal-m3c2-upstream-golden.json';
import {expectPointCloudChangeBounds} from './pointcloud-review-assertions';

for (const omitFirstReferencePoint of [false, true])
  it(`M3C2 agrees with pinned PDAL golden outputs, upstream first-point exclusion=${omitFirstReferencePoint}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const rows = reference.cores.filter(
        row => row.omitFirstReferencePoint === omitFirstReferencePoint
      );
      expect(rows.length).toBe(omitFirstReferencePoint ? 1 : 90);
      for (const row of rows) {
        expect(row.counts).toHaveLength(2);
        expect(row.counts.every(count => count >= 2)).toBe(true);
        expect(row.populationRoughness).toHaveLength(2);
      }
      const changes = fixture.view('float32x3', new Float32Array(rows.length * 3));
      const roughness = fixture.view('float32x3', new Float32Array(rows.length * 3));
      const counts = fixture.view('uint32', new Uint32Array(rows.length * 2));
      const status = fixture.view('uint32', new Uint32Array(rows.length));
      fixture.graph.add(
        new GPUPointCloudChange({
          reference: fixture.view(
            'float32x3',
            Float32Array.from(reference.reference.slice(omitFirstReferencePoint ? 1 : 0).flat())
          ).view,
          comparison: fixture.view('float32x3', Float32Array.from(reference.comparison.flat()))
            .view,
          cores: fixture.view('float32x3', Float32Array.from(rows.flatMap(row => row.position)))
            .view,
          normals: fixture.view('float32x3', Float32Array.from(rows.flatMap(row => row.normal)))
            .view,
          alignment: 'already-aligned',
          radius: reference.radius,
          halfDepth: reference.halfDepth,
          minimumSupport: 2,
          output: {
            changes: changes.view,
            roughness: roughness.view,
            counts: counts.view,
            status: status.view
          }
        })
      );
      await fixture.run();
      expect(await readIntegers(status.buffer)).toEqual(new Array(rows.length).fill(0));
      const actualCounts = await readIntegers(counts.buffer);
      const actualChanges = await readFloats(changes.buffer);
      const actualRoughness = await readFloats(roughness.buffer);
      expect(actualCounts).toHaveLength(rows.length * 2);
      expect(actualChanges).toHaveLength(rows.length * 3);
      expect(actualRoughness).toHaveLength(rows.length * 3);
      expect(actualChanges.every(Number.isFinite)).toBe(true);
      expect(actualRoughness.every(Number.isFinite)).toBe(true);
      for (let row = 0; row < rows.length; row++) {
        const expected = rows[row];
        expect(actualCounts.slice(row * 2, row * 2 + 2), `golden row ${expected.row}`).toEqual(
          expected.counts
        );
        expect(
          Math.abs(actualChanges[row * 3] - expected.distance),
          `golden row ${expected.row}`
        ).toBeLessThan(0.01);
        for (let epoch = 0; epoch < 2; epoch++) {
          const count = expected.counts[epoch];
          const population = actualRoughness[row * 3 + epoch] * Math.sqrt((count - 1) / count);
          expect(
            Math.abs(population - expected.populationRoughness[epoch]),
            `golden roughness row ${expected.row}`
          ).toBeLessThan(0.002);
        }
        // PDAL pins population standard deviations p. GPU uses sample variance p²*n/(n-1),
        // so variance of the mean is p²/(n-1). No GPU roughness enters the bound oracle.
        const samplingVariance = expected.populationRoughness.reduce(
          (sum, population, epoch) => sum + population ** 2 / (expected.counts[epoch] - 1),
          0
        );
        const sampleLimit = 1.96 * Math.sqrt(samplingVariance);
        expectPointCloudChangeBounds(
          actualChanges.slice(row * 3, row * 3 + 3),
          actualRoughness[row * 3 + 2],
          expected.distance,
          sampleLimit
        );
        const populationLimit =
          1.96 *
          Math.sqrt(
            expected.populationRoughness.reduce(
              (sum, population, epoch) => sum + population ** 2 / expected.counts[epoch],
              0
            )
          );
        expect(
          Math.abs(populationLimit - expected.uncertainty),
          `golden uncertainty row ${expected.row}`
        ).toBeLessThan(0.002);
        expect(Number(Math.abs(actualChanges[row * 3]) > populationLimit)).toBe(
          expected.significant
        );
      }
    } finally {
      fixture.destroy();
    }
  });

it('M3C2 overwrites prefilled bounds when either epoch has insufficient support', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    // Vertical cylinders of radius 1 at x=0 and x=10, half-depth 5. The first core sees
    // reference z={0}, comparison z={1,3}; the second sees reference z={0,2}, comparison z={1}.
    const cores = [
      [0, 0, 0],
      [10, 0, 0]
    ];
    const referencePositions = [
      [0, 0, 0],
      [10, 0, 0],
      [10, 0, 2]
    ];
    const comparisonPositions = [
      [0, 0, 1],
      [0, 0, 3],
      [10, 0, 1]
    ];
    expect(cores).toHaveLength(2);
    expect(referencePositions).toHaveLength(3);
    expect(comparisonPositions).toHaveLength(3);
    const changes = fixture.view('float32x3', new Float32Array(6).fill(123));
    const roughness = fixture.view('float32x3', new Float32Array(6).fill(456));
    const counts = fixture.view('uint32', new Uint32Array(4).fill(99));
    const status = fixture.view('uint32', new Uint32Array(2).fill(99));
    fixture.graph.add(
      new GPUPointCloudChange({
        reference: fixture.view('float32x3', Float32Array.from(referencePositions.flat())).view,
        comparison: fixture.view('float32x3', Float32Array.from(comparisonPositions.flat())).view,
        cores: fixture.view('float32x3', Float32Array.from(cores.flat())).view,
        normals: fixture.view('float32x3', new Float32Array([0, 0, 1, 0, 0, 1])).view,
        alignment: 'already-aligned',
        radius: 1,
        halfDepth: 5,
        minimumSupport: 2,
        output: {
          changes: changes.view,
          roughness: roughness.view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    await fixture.run();
    expect(await readIntegers(counts.buffer)).toEqual([1, 2, 2, 1]);
    expect(await readIntegers(status.buffer)).toEqual([
      GPU_POINT_CLOUD_CHANGE_STATUS_INSUFFICIENT_SUPPORT,
      GPU_POINT_CLOUD_CHANGE_STATUS_INSUFFICIENT_SUPPORT
    ]);
    expect(await readFloats(changes.buffer)).toEqual([0, 0, 0, 0, 0, 0]);
    expect(await readFloats(roughness.buffer)).toEqual([0, 0, 0, 0, 0, 0]);
  } finally {
    fixture.destroy();
  }
});
