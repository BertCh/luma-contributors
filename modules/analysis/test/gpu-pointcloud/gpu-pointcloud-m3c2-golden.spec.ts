// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUPointCloudChange} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import reference from '../oracles/fixtures/pdal-m3c2-upstream-golden.json';

for (const omitFirstReferencePoint of [false, true])
  it(`M3C2 agrees with pinned PDAL golden outputs, upstream first-point exclusion=${omitFirstReferencePoint}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const rows = reference.cores.filter(
        row => row.omitFirstReferencePoint === omitFirstReferencePoint
      );
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
        const populationLimit =
          1.96 *
          Math.sqrt(
            (actualRoughness[row * 3] ** 2 * (expected.counts[0] - 1)) / expected.counts[0] ** 2 +
              (actualRoughness[row * 3 + 1] ** 2 * (expected.counts[1] - 1)) /
                expected.counts[1] ** 2
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
