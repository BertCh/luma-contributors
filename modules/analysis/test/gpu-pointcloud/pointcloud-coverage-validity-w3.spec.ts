// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUPointCloudCoverageValidity,
  type GPUPointCloudCoverageValidityOutput,
  type GPUPointCloudCoverageValidityProps,
  GPUPointCloudSurfaces,
  getRasterMetadataFromPointCloudGrid
} from '@luma-contributors/analysis/pointcloud-analysis';
import {getGPUPointCloudGridParameterValues} from '../../src/pointcloud-analysis/operations/pointcloud-grid-parameters';
import {GPUPointCloudChange} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-change';
import {makeGPURasterBand} from '../../src/raster-analysis/raster-grid/gpu-raster-grid';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';

it.each([
  false,
  true
])('UNI-01 and UNI-04: aligns coverage validity and preserves zero-elevation ground in both directions (runtime grid %s)', async runtimeGrid => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    // South footprint: [100,120] x [200,220]. Non-ground return at southwest;
    // zero-elevation ground at northwest. The interior-edge point at Y=210 goes north.
    const positions = fixture.view(
      'float32x3',
      new Float32Array([105, 205, 7, 105, 215, 0, 115, 210, 5])
    );
    const classifications = fixture.view('uint32', new Uint32Array([1, 2, 2]));
    const runs = (['south', 'north'] as const).map(rowDirection => {
      const grid = {origin: [100, 200] as const, width: 2, height: 2, cellSize: 10, rowDirection};
      const terrain = fixture.view('float32', new Float32Array(4));
      const surface = fixture.view('float32', new Float32Array(4));
      const canopy = fixture.view('float32', new Float32Array(4));
      const coverage = fixture.view('uint32', new Uint32Array(4));
      const surfaceValidity = fixture.view('uint32', new Uint32Array(4).fill(0xdeadbeef));
      const terrainValidity = fixture.view('uint32', new Uint32Array(4).fill(0xdeadbeef));
      const canopyValidity = fixture.view('uint32', new Uint32Array(4).fill(0xdeadbeef));
      fixture.graph.add(
        new GPUPointCloudSurfaces({
          id: `surfaces-validity-${rowDirection}`,
          positions: positions.view,
          classifications: classifications.view,
          gridParameters: runtimeGrid
            ? fixture.view('float32', getGPUPointCloudGridParameterValues(grid)).view
            : undefined,
          grid,
          output: {
            terrain: terrain.view,
            surface: surface.view,
            canopy: canopy.view,
            coverage: coverage.view
          }
        })
      );
      const validityOutput: GPUPointCloudCoverageValidityOutput = {
        surfaceValidity: surfaceValidity.view,
        terrainValidity: terrainValidity.view,
        canopyValidity: canopyValidity.view
      };
      const validityProps: GPUPointCloudCoverageValidityProps = {
        id: `validity-${rowDirection}`,
        coverage: coverage.view,
        output: validityOutput
      };
      fixture.graph.add(new GPUPointCloudCoverageValidity(validityProps));
      // Runnable cross-family handoff: borrow the caller-owned terrain view and the converted
      // ground validity on metadata aligned to the producer's actual cell footprint.
      const terrainBand = makeGPURasterBand(
        terrain.view,
        getRasterMetadataFromPointCloudGrid(grid),
        {validity: terrainValidity.view}
      );
      return {
        terrainBand,
        rowDirection,
        grid,
        terrain,
        coverage,
        surfaceValidity,
        terrainValidity,
        canopyValidity
      };
    });
    await fixture.run();
    for (const run of runs) {
      const south = run.rowDirection === 'south';
      expect(run.terrainBand.validity).toBe(run.terrainValidity.view);
      expect(getRasterMetadataFromPointCloudGrid(run.grid).affine).toEqual([
        10,
        0,
        100,
        0,
        south ? -10 : 10,
        south ? 220 : 200
      ]);
      expect(await readIntegers(run.coverage.buffer)).toEqual(south ? [3, 3, 1, 0] : [1, 0, 3, 3]);
      expect(await readIntegers(run.surfaceValidity.buffer)).toEqual(
        south ? [1, 1, 1, 0] : [1, 0, 1, 1]
      );
      expect(await readIntegers(run.terrainValidity.buffer)).toEqual(
        south ? [1, 1, 0, 0] : [0, 0, 1, 1]
      );
      expect(await readIntegers(run.canopyValidity.buffer)).toEqual(
        south ? [1, 1, 0, 0] : [0, 0, 1, 1]
      );
      const terrain = await readFloats(run.terrain.buffer);
      expect(terrain.every(Number.isFinite)).toBe(true);
      expect(terrain).toEqual(south ? [0, 5, 0, 0] : [0, 0, 0, 5]);
    }
  } finally {
    fixture.destroy();
  }
});

it('B1: fractional large origins publish the same column described by raster metadata', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const grid = {origin: [1000000.03, 200.03] as const, cellSize: 0.1, width: 2, height: 2};
    const positions = fixture.view('float32x3', Float32Array.of(1000000.125, 200.125, 8));
    const classifications = fixture.view('uint32', Uint32Array.of(2));
    const terrain = fixture.view('float32', new Float32Array(4));
    const surface = fixture.view('float32', new Float32Array(4));
    const canopy = fixture.view('float32', new Float32Array(4));
    const coverage = fixture.view('uint32', new Uint32Array(4));
    fixture.graph.add(
      new GPUPointCloudSurfaces({
        positions: positions.view,
        classifications: classifications.view,
        grid,
        output: {
          terrain: terrain.view,
          surface: surface.view,
          canopy: canopy.view,
          coverage: coverage.view
        }
      })
    );
    await fixture.run();
    expect(await readIntegers(coverage.buffer)).toEqual([0, 0, 0, 3]);
    const values = await readFloats(terrain.buffer);
    expect(values.every(Number.isFinite)).toBe(true);
    expect(values).toEqual([0, 0, 0, 8]);
    const affine = getRasterMetadataFromPointCloudGrid(grid).affine;
    expect(Math.floor((1000000.125 - affine[2]) / affine[0])).toBe(1);
  } finally {
    fixture.destroy();
  }
});

it('N3: surfaces and indexed change compile large integer-valued float32 origins on Metal', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const reference = fixture.view(
      'float32x3',
      Float32Array.of(1e19, -1e19, 0, 1e19, -1e19, 0, 1e19, -1e19, 0)
    );
    const comparison = fixture.view(
      'float32x3',
      Float32Array.of(1e19, -1e19, 0.5, 1e19, -1e19, 0.5, 1e19, -1e19, 0.5)
    );
    const cores = fixture.view('float32x3', Float32Array.of(1e19, -1e19, 0.25));
    const normals = fixture.view('float32x3', Float32Array.of(0, 0, 1));
    const changes = fixture.view('float32x3', new Float32Array(3));
    const roughness = fixture.view('float32x3', new Float32Array(3));
    const counts = fixture.view('uint32', new Uint32Array(2));
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudChange({
        reference: reference.view,
        comparison: comparison.view,
        cores: cores.view,
        normals: normals.view,
        alignment: 'already-aligned',
        radius: 1,
        halfDepth: 1,
        grid: {origin: [1e19, -1e19, -1], dimensions: [2, 2, 4], cellSize: 1},
        output: {
          changes: changes.view,
          roughness: roughness.view,
          counts: counts.view,
          status: status.view
        }
      })
    );
    const classifications = fixture.view('uint32', Uint32Array.of(2, 2, 2));
    const terrain = fixture.view('float32', new Float32Array(4));
    const surface = fixture.view('float32', new Float32Array(4));
    const canopy = fixture.view('float32', new Float32Array(4));
    const coverage = fixture.view('uint32', new Uint32Array(4));
    fixture.graph.add(
      new GPUPointCloudSurfaces({
        positions: reference.view,
        classifications: classifications.view,
        grid: {origin: [1e19, -1e19], width: 2, height: 2, cellSize: 1},
        output: {
          terrain: terrain.view,
          surface: surface.view,
          canopy: canopy.view,
          coverage: coverage.view
        }
      })
    );
    await fixture.run();
    // Three identical points per epoch: mean difference=0.5, both sample variances=0.
    // Cylinder crosses the X/Y grid minimum, independently setting gridIncomplete=256.
    const values = await readFloats(changes.buffer);
    expect(values.every(Number.isFinite)).toBe(true);
    expect(values).toEqual([0.5, 0.5, 0.5]);
    expect(await readIntegers(counts.buffer)).toEqual([3, 3]);
    expect(await readIntegers(status.buffer)).toEqual([256]);
    expect(await readIntegers(coverage.buffer)).toEqual([0, 0, 3, 0]);
  } finally {
    fixture.destroy();
  }
});
