// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUPointCloudNeighborhoods,
  GPUPointCloudFeatures,
  GPUPointCloudStatisticalFilter,
  GPUPointCloudCoverageValidity,
  GPUPointCloudSurfaces,
  getRasterMetadataFromPointCloudGrid,
  decodeGPUPointCloudFeaturesStatus
} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeGPURasterBand} from '../../../analysis/src/raster-analysis/raster-grid/gpu-raster-grid';
import {requireTestDevice} from '../../../analysis/src/testing/require-test-device';
import {
  makeSpatialFixture,
  readFloats,
  readIntegers
} from '../../../analysis/test/spatial-test-utils';
import {
  createGroundSeedCoverageNode,
  createFeatureDisplayClassesNode,
  addLidarFeatureSummary
} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/terrain/point-cloud-ground-support';
import {addKernelPass} from '../../../../examples/deck/spatial-analysis-showcase/src/engine/mode-kernels';
import {
  QUAKE_DISTANCE_CELL_WGSL,
  QUAKE_EVENT_DISTANCE_WGSL,
  QUAKE_LINKS_WGSL
} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/time/quake-distance-cell';

it('STA-04 certified bounds distinguish complete and usable incomplete fits from truncation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view(
      'float32x3',
      Float32Array.of(0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0)
    );
    const queryRows = fixture.view('uint32', Uint32Array.of(0));
    const results = [0, 1, 2].map(route => {
      const capacity = route === 2 ? 3 : 4;
      const indices = fixture.view('uint32', new Uint32Array(capacity));
      const squaredDistances = fixture.view('float32', new Float32Array(capacity));
      const counts = fixture.view('uint32', new Uint32Array(1));
      const status = fixture.view('uint32', new Uint32Array(1));
      const normals = fixture.view('float32x3', new Float32Array(3));
      const eigenvalues = fixture.view('float32x3', new Float32Array(3));
      const shapes = fixture.view('float32x3', new Float32Array(3));
      const featureStatus = fixture.view('uint32', new Uint32Array(1));
      const display = fixture.view('uint32', new Uint32Array(1));
      const meanDistances = fixture.view('float32', new Float32Array(1));
      const validity = fixture.view('uint32', new Uint32Array(1));
      const thresholdComplete = fixture.view('uint32', new Uint32Array(1));
      fixture.graph.add(
        new GPUPointCloudNeighborhoods({
          id: `support-${route}`,
          positions: positions.view,
          queryRows: queryRows.view,
          radius: 1.1,
          capacity,
          supportBounds: {minimum: [route === 1 ? -1.05 : -2, -2, -2], maximum: [2, 2, 2]},
          output: {
            indices: indices.view,
            squaredDistances: squaredDistances.view,
            counts: counts.view,
            status: status.view
          }
        })
      );
      fixture.graph.add(
        new GPUPointCloudFeatures({
          id: `features-${route}`,
          positions: positions.view,
          indices: indices.view,
          counts: counts.view,
          neighborhoodStatus: status.view,
          capacity,
          output: {
            normals: normals.view,
            eigenvalues: eigenvalues.view,
            shapes: shapes.view,
            status: featureStatus.view
          }
        })
      );
      fixture.graph.add(
        new GPUPointCloudStatisticalFilter({
          id: `filter-${route}`,
          squaredDistances: squaredDistances.view,
          counts: counts.view,
          neighborhoodStatus: status.view,
          capacity,
          output: {
            meanDistances: meanDistances.view,
            validity: validity.view,
            thresholdComplete: thresholdComplete.view
          }
        })
      );
      fixture.graph.add(
        createFeatureDisplayClassesNode(
          fixture.graph,
          featureStatus.view,
          display.view,
          `lidar-feature-display-${route}`
        )
      );
      return {status, normals, eigenvalues, featureStatus, display, thresholdComplete, validity};
    });
    await fixture.run();
    for (const [route, result] of results.entries()) {
      // Four unit-distance neighbours around the origin: covariance diag(.5,.5,0),
      // normal +Z, no failed eigensolve. Only route 1's radius exceeds certified min-X.
      expect(await readIntegers(result.status.buffer)).toEqual([[0], [4], [2]][route]);
      expect(await readIntegers(result.featureStatus.buffer)).toEqual([[0], [8], [2]][route]);
      expect(await readIntegers(result.display.buffer)).toEqual([[0], [1], [2]][route]);
      const flags = Uint32Array.from(await readIntegers(result.featureStatus.buffer));
      expect(decodeGPUPointCloudFeaturesStatus(flags).converged).toBe(true);
      expect(await readIntegers(result.thresholdComplete.buffer)).toEqual(route === 1 ? [0] : [1]);
      expect(await readIntegers(result.validity.buffer)).toEqual(route === 1 ? [0] : [1]);
      const eigenvalues = await readFloats(result.eigenvalues.buffer);
      expect(eigenvalues.every(Number.isFinite)).toBe(true);
      if (route !== 2) {
        expect(eigenvalues).toEqual([0.5, 0.5, 0]);
        const normals = await readFloats(result.normals.buffer);
        expect(normals.every(Number.isFinite)).toBe(true);
        expect(normals).toEqual([0, 0, 1]);
      }
    }
  } finally {
    fixture.destroy();
  }
});

it('UNI-01 public coverage conversion preserves valid zero terrain on the southward 2x2 grid', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const grid = {
      origin: [100, 200] as const,
      cellSize: 10,
      width: 2,
      height: 2,
      rowDirection: 'south' as const
    };
    const positions = fixture.view('float32x3', Float32Array.of(105, 215, 7, 115, 215, 0));
    const classifications = fixture.view('uint32', Uint32Array.of(1, 2));
    const terrain = fixture.view('float32', new Float32Array(4));
    const surface = fixture.view('float32', new Float32Array(4));
    const canopy = fixture.view('float32', new Float32Array(4));
    const coverage = fixture.view('uint32', new Uint32Array(4));
    const seed = fixture.view('uint32', new Uint32Array(4));
    const surfaceValidity = fixture.view('uint32', new Uint32Array(4));
    const terrainValidity = fixture.view('uint32', new Uint32Array(4));
    const canopyValidity = fixture.view('uint32', new Uint32Array(4));
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
    fixture.graph.add(createGroundSeedCoverageNode(fixture.graph, coverage.view, seed.view));
    fixture.graph.add(
      new GPUPointCloudCoverageValidity({
        coverage: coverage.view,
        output: {
          surfaceValidity: surfaceValidity.view,
          terrainValidity: terrainValidity.view,
          canopyValidity: canopyValidity.view
        }
      })
    );
    const band = makeGPURasterBand(terrain.view, getRasterMetadataFromPointCloudGrid(grid), {
      validity: terrainValidity.view
    });
    await fixture.run();
    // Northern cells contain non-ground z=7, ground z=0; southern cells are empty.
    expect(await readIntegers(coverage.buffer)).toEqual([1, 3, 0, 0]);
    expect(await readIntegers(seed.buffer)).toEqual([0, 1, 0, 0]);
    expect(await readIntegers(terrainValidity.buffer)).toEqual([0, 1, 0, 0]);
    expect(await readIntegers(surfaceValidity.buffer)).toEqual([1, 1, 0, 0]);
    expect(await readIntegers(canopyValidity.buffer)).toEqual([0, 1, 0, 0]);
    const heights = await readFloats(terrain.buffer);
    expect(heights.every(Number.isFinite)).toBe(true);
    expect(heights).toEqual([0, 0, 0, 0]);
    expect(band.validity).toBe(terrainValidity.view);
    expect(getRasterMetadataFromPointCloudGrid(grid).affine).toEqual([10, 0, 100, 0, -10, 220]);
  } finally {
    fixture.destroy();
  }
});

it.each([
  {name: 'positive 1312 m step', direction: 1, origin: 0, cellSize: 1312},
  {name: 'negative 1312 m step', direction: -1, origin: 0, cellSize: 1312},
  {name: 'translated original-frame edge', direction: 1, origin: 99.875, cellSize: 0.3}
])('W5-28 both quake readers select the original-frame edge cell: $name', async ({
  direction,
  origin,
  cellSize
}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const step = direction * Math.fround(cellSize);
    const translated = origin !== 0;
    const width = translated ? 3 : 2;
    const positions = fixture.view(
      'float32',
      Float32Array.of(
        translated ? 100.4749984741211 : step,
        origin,
        origin,
        origin,
        origin + width * step,
        origin,
        origin - step,
        origin
      )
    );
    const settings = new Float32Array(16);
    settings[6] = 1; // links enabled
    settings[7] = 5; // minimum depth
    settings[8] = origin;
    settings[9] = origin;
    settings[10] = 1 / step;
    settings[11] = step;
    settings[12] = width;
    settings[13] = 1;
    const parameters = fixture.view('float32', settings);
    const distances = fixture.view('float32', Float32Array.of(1000, 2000, 4000).slice(0, width));
    const nearestCells = fixture.view('uint32', Uint32Array.of(0, 1, 2).slice(0, width));
    const distanceKm = fixture.view('float32', new Float32Array(4));
    const depth = fixture.view('float32', new Float32Array(4).fill(10));
    const linkFade = fixture.view('float32', new Float32Array(4).fill(99));
    const linkSegments = fixture.view('float32', new Float32Array(16).fill(99));
    const read = (name: string, view: typeof parameters.view, type: 'f32' | 'u32' = 'f32') => ({
      name,
      view,
      type,
      access: 'read' as const
    });
    addKernelPass(fixture.graph, {
      id: 'quake-event-distance',
      invocationCount: 4,
      declarations: QUAKE_DISTANCE_CELL_WGSL,
      body: QUAKE_EVENT_DISTANCE_WGSL,
      bindings: [
        read('positions', positions.view),
        read('parameters', parameters.view),
        read('distances', distances.view),
        {name: 'distanceKm', view: distanceKm.view, type: 'f32', access: 'read_write'}
      ]
    });
    addKernelPass(fixture.graph, {
      id: 'quake-links',
      invocationCount: 4,
      declarations: QUAKE_DISTANCE_CELL_WGSL,
      body: QUAKE_LINKS_WGSL,
      bindings: [
        read('positions', positions.view),
        read('parameters', parameters.view),
        read('distanceKm', distanceKm.view),
        read('depth', depth.view),
        {name: 'nearestCells', view: nearestCells.view, type: 'u32', access: 'read'},
        {name: 'linkFade', view: linkFade.view, type: 'f32', access: 'read_write'},
        {name: 'linkSegments', view: linkSegments.view, type: 'f32', access: 'read_write'}
      ]
    });
    await fixture.run();
    // Interior edge belongs to cell 1; origin to cell 0; the upper edge and before-origin are outside.
    const values = await readFloats(distanceKm.buffer);
    expect(values.every(Number.isFinite)).toBe(true);
    expect(values).toEqual(translated ? [4, 1, -1, -1] : [2, 1, -1, -1]);
    expect(await readFloats(linkFade.buffer)).toEqual([1, 1, 0, 0]);
    const segments = await readFloats(linkSegments.buffer);
    expect(segments.every(Number.isFinite)).toBe(true);
    // Translated edge is exactly f32(origin + 2*step), although the relative offset
    // is below f32(2*step). The cell-2 centre rounds to 100.625 and row centre to
    // 100.0250015258789. Literals are independent coordinate geometry expectations.
    expect(segments).toEqual(
      translated
        ? [
            100.4749984741211, 99.875, 100.625, 100.0250015258789, 99.875, 99.875,
            100.0250015258789, 100.0250015258789, 0, 0, 0, 0, 0, 0, 0, 0
          ]
        : [step, 0, 1.5 * step, 0.5 * step, 0, 0, 0.5 * step, 0.5 * step, 0, 0, 0, 0, 0, 0, 0, 0]
    );
  } finally {
    fixture.destroy();
  }
});

it('S1 LiDAR feature summary reduces five independent status counts and overwrites poisoned totals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const status = fixture.view('uint32', Uint32Array.of(0, 8, 2, 1, 4, 10, 12, 9));
    const counts = fixture.view('uint32', new Uint32Array(5).fill(0xffffffff));
    addLidarFeatureSummary(fixture.graph, status.view, counts.view);
    await fixture.run();
    // Bit inspection by hand: four support losses, two usable incomplete fits,
    // two truncations, two failed solves, two degeneracies. Causes overlap.
    expect(await readIntegers(counts.buffer)).toEqual([4, 2, 2, 2, 2]);
    status.buffer.write(new Uint32Array(8));
    await fixture.run();
    expect(await readIntegers(counts.buffer)).toEqual([0, 0, 0, 0, 0]);
  } finally {
    fixture.destroy();
  }
});
