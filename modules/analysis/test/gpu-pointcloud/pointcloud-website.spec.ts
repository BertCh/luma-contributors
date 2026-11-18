// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUPointCloudVoxelMerge,
  GPUPointCloudFeatures,
  GPUPointCloudNeighborhoods,
  GPUPointCloudChange,
  GPUPointCloudSurfaces,
  GPUPointCloudGround,
  GPUPointCloudHeight,
  GPUPointCloudICP,
  GPUPointCloudSMRF
} from '@luma-contributors/analysis/pointcloud-analysis';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';
import {constructPointCloudShowcaseSnippet} from './pointcloud-showcase-snippets';
import {addPointCloudSupportVoxelPart} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/terrain/point-cloud-voxel-parts';
import {
  createGroundSeedCoverageNode,
  createSupportedGroundClassesNode,
  createGroundDisplayClassesNode,
  LIDAR_GROUND_GRID,
  LIDAR_GROUND_RADII,
  LIDAR_GROUND_DISPLAY_CATEGORIES
} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/terrain/point-cloud-ground-support';
import {projectLidarGroundTile} from './pointcloud-lidar-fixture';
import {
  createPointCloudDisplayCopyNode,
  getPointCloudFeatureClass,
  getPointCloudChangeSupportText,
  getPointCloudHeightSupportText,
  getPointCloudICPSupportText,
  summarizePointCloudGround
} from '../../../../website/src/gpu/runs/entries/dataframe-pointcloud-records/_lib/pointcloud-display';

it('merges two points into the arithmetic centroid with an empty second tile', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = new Float32Array([0.03125, 0.05, 0.05, 0.15625, 0.05, 0.05]);
    const selected = fixture.view('float32x3', new Float32Array(3));
    const counts = fixture.view('uint32', new Uint32Array(1));
    const representatives = fixture.view('uint32', new Uint32Array(1));
    const parts = [positions, new Float32Array()].map((sourcePositions, tile) =>
      addPointCloudSupportVoxelPart(fixture.graph, {
        id: `showcase-tile-${tile}`,
        positions: fixture.view('float32x3', sourcePositions).view,
        sourcePositions,
        origin: [0, 0, 0],
        gridSize: [1, 1, 1],
        voxelSize: 0.2,
        counts: tile === 0 ? counts.view : fixture.view('uint32', new Uint32Array(1)).view,
        representatives:
          tile === 0 ? representatives.view : fixture.view('uint32', new Uint32Array(1)).view,
        selectedPositions:
          tile === 0 ? selected.view : fixture.view('float32x3', new Float32Array(3)).view,
        uploadMeans: means => fixture.view('float32x3', means).view,
        sourceRowOffset: tile === 0 ? 0 : 2
      })
    );
    const merged = fixture.view('float32x3', new Float32Array(3));
    const mergedCounts = fixture.view('uint32', new Uint32Array(1));
    const mergedRows = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudVoxelMerge({
        partitionPolicy: 'disjoint-owned-rows',
        parts,
        output: {
          counts: mergedCounts.view,
          representatives: mergedRows.view,
          centroids: merged.view,
          status: fixture.view('uint32', new Uint32Array(1)).view
        }
      })
    );
    await fixture.run();
    expect(await readIntegers(mergedCounts.buffer)).toEqual([2]);
    expect(await readIntegers(mergedRows.buffer)).toEqual([0]);
    const [selectedX] = await readFloats(selected.buffer);
    const [centroidX] = await readFloats(merged.buffer);
    expect(Number.isFinite(selectedX)).toBe(true);
    expect(selectedX).toBe(0.03125);
    expect(Number.isFinite(centroidX)).toBe(true);
    expect(centroidX).toBe(0.09375);
  } finally {
    fixture.destroy();
  }
});

it('suppresses an isolated shape and a change core with insufficient support', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const position = fixture.view('float32x3', new Float32Array([0, 0, 0]));
    const indices = fixture.view('uint32', new Uint32Array(3));
    const counts = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudNeighborhoods({
        positions: position.view,
        radius: 1,
        capacity: 3,
        output: {
          indices: indices.view,
          counts: counts.view,
          squaredDistances: fixture.view('float32', new Float32Array(3)).view
        }
      })
    );
    const shapes = fixture.view('float32x3', new Float32Array(3));
    const featureStatus = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudFeatures({
        positions: position.view,
        indices: indices.view,
        counts: counts.view,
        capacity: 3,
        output: {
          normals: fixture.view('float32x3', new Float32Array(3)).view,
          eigenvalues: fixture.view('float32x3', new Float32Array(3)).view,
          shapes: shapes.view,
          status: featureStatus.view
        }
      })
    );
    const featureDisplay = fixture.view('float32', new Float32Array(1));
    fixture.graph.add(
      createPointCloudDisplayCopyNode(fixture.graph, {
        id: 'shape-display',
        source: shapes.view,
        status: featureStatus.view,
        output: featureDisplay.view,
        invalidBits: 5
      })
    );
    const changes = fixture.view('float32x3', new Float32Array(3));
    const changeStatus = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudChange({
        reference: fixture.view('float32x3', new Float32Array([-0.1, 0, 0, 0, 0, 0, 0.1, 0, 0]))
          .view,
        comparison: position.view,
        cores: position.view,
        normals: fixture.view('float32x3', new Float32Array([0, 0, 1])).view,
        alignment: 'already-aligned',
        radius: 1,
        halfDepth: 1,
        minimumSupport: 3,
        output: {
          changes: changes.view,
          roughness: fixture.view('float32x3', new Float32Array(3)).view,
          counts: fixture.view('uint32', new Uint32Array(2)).view,
          status: changeStatus.view
        }
      })
    );
    const changeDisplay = fixture.view('float32', new Float32Array(1));
    fixture.graph.add(
      createPointCloudDisplayCopyNode(fixture.graph, {
        id: 'change-display',
        source: changes.view,
        status: changeStatus.view,
        output: changeDisplay.view
      })
    );
    await fixture.run();
    const [featureCode] = await readIntegers(featureStatus.buffer);
    expect(featureCode & 1).toBe(1);
    expect(Number.isNaN(getPointCloudFeatureClass(0, 0, 0, featureCode))).toBe(true);
    expect(Number.isNaN((await readFloats(featureDisplay.buffer))[0])).toBe(true);
    const [changeCode] = await readIntegers(changeStatus.buffer);
    expect(changeCode).toBe(2);
    expect(getPointCloudChangeSupportText(changeCode)).toBe('insufficiently supported');
    expect(Number.isNaN((await readFloats(changeDisplay.buffer))[0])).toBe(true);
  } finally {
    fixture.destroy();
  }
});

it('F2 Surfaces → Ground → Surfaces never recovers terrain from a class-1-only cell', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view('float32x3', new Float32Array([0.5, 0.5, 0.1])).view;
    const classifications = fixture.view('uint32', new Uint32Array([1])).view;
    const grid = {origin: [0, 0] as const, cellSize: 1, width: 1, height: 1};
    const firstTerrain = fixture.view('float32', new Float32Array(1));
    const firstCoverage = fixture.view('uint32', new Uint32Array(1));
    const seedCoverage = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudSurfaces({
        id: 'website-seed-surfaces',
        positions,
        classifications,
        grid,
        output: {
          terrain: firstTerrain.view,
          surface: fixture.view('float32', new Float32Array(1)).view,
          canopy: fixture.view('float32', new Float32Array(1)).view,
          coverage: firstCoverage.view
        }
      })
    );
    fixture.graph.add(
      createGroundSeedCoverageNode(fixture.graph, firstCoverage.view, seedCoverage.view)
    );
    const groundStatus = fixture.view('uint32', new Uint32Array(1));
    const classified = fixture.view('uint32', new Uint32Array(1));
    const supported = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudGround({
        positions,
        classifications,
        grid,
        radii: [1],
        initialThreshold: 0.3,
        slope: 0.1,
        maximumThreshold: 2,
        minimumSurface: firstTerrain.view,
        coverage: seedCoverage.view,
        output: {
          groundMask: fixture.view('uint32', new Uint32Array(1)).view,
          outputClassifications: classified.view,
          status: groundStatus.view
        }
      })
    );
    fixture.graph.add(
      createSupportedGroundClassesNode(
        fixture.graph,
        classified.view,
        groundStatus.view,
        supported.view
      )
    );
    const recoveredCoverage = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudSurfaces({
        id: 'website-recovered-surfaces',
        positions,
        classifications: supported.view,
        grid,
        output: {
          terrain: fixture.view('float32', new Float32Array(1)).view,
          surface: fixture.view('float32', new Float32Array(1)).view,
          canopy: fixture.view('float32', new Float32Array(1)).view,
          coverage: recoveredCoverage.view
        }
      })
    );
    await fixture.run();
    expect(await readIntegers(firstCoverage.buffer)).toEqual([1]);
    expect(await readIntegers(seedCoverage.buffer)).toEqual([0]);
    expect(await readIntegers(groundStatus.buffer)).toEqual([1]);
    expect(await readIntegers(supported.buffer)).toEqual([1]);
    expect(await readIntegers(recoveredCoverage.buffer)).toEqual([1]);
  } finally {
    fixture.destroy();
  }
});

it('F4 / final2 G4 displayed lidar height distinguishes genuine zero from unsupported zero', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const heights = fixture.view('float32', new Float32Array(2));
    const status = fixture.view('uint32', new Uint32Array(2));
    fixture.graph.add(
      constructPointCloudShowcaseSnippet<GPUPointCloudHeight>('lidar', 'GPUPointCloudHeight', {
        GPUPointCloudHeight,
        positions: fixture.view('float32x3', new Float32Array([0.25, 0.5, 0, 1.25, 0.5, 0])).view,
        terrain: fixture.view('float32', new Float32Array([0, 0])).view,
        coverage: fixture.view('uint32', new Uint32Array([2, 1])).view,
        grid: {origin: [0, 0], cellSize: 1, width: 2, height: 1},
        heights: heights.view,
        heightStatus: status.view
      })
    );
    const display = fixture.view('float32', new Float32Array(2));
    fixture.graph.add(
      createPointCloudDisplayCopyNode(fixture.graph, {
        id: 'height-display',
        source: heights.view,
        status: status.view,
        output: display.view
      })
    );
    await fixture.run();
    expect(await readFloats(heights.buffer)).toEqual([0, 0]);
    const codes = await readIntegers(status.buffer);
    expect(codes).toEqual([0, 2]);
    expect(codes.map(getPointCloudHeightSupportText)).toEqual([
      'valid height',
      'unsupported terrain'
    ]);
    const values = await readFloats(display.buffer);
    expect(values[0]).toBe(0);
    expect(Number.isNaN(values[1])).toBe(true);
  } finally {
    fixture.destroy();
  }
});

it('F4 hides alignment with fewer than three matches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view('float32x3', new Float32Array([0, 0, 0, 1, 0, 0])).view;
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudICP({
        source: positions,
        target: positions,
        maximumDistance: 0.1,
        iterations: 1,
        output: {
          transform: fixture.view('float32', new Float32Array(16)).view,
          history: fixture.view('float32x3', new Float32Array(3)).view,
          inlierMask: fixture.view('uint32', new Uint32Array(2)).view,
          status: status.view
        }
      })
    );
    const display = fixture.view('float32', new Float32Array(2));
    fixture.graph.add(
      createPointCloudDisplayCopyNode(fixture.graph, {
        id: 'alignment-display',
        source: positions,
        status: status.view,
        output: display.view,
        statusIndex: 0
      })
    );
    await fixture.run();
    const [code] = await readIntegers(status.buffer);
    expect(code).toBe(2);
    expect(getPointCloudICPSupportText(code)).toBe('insufficient matches');
    expect((await readFloats(display.buffer)).every(Number.isNaN)).toBe(true);
  } finally {
    fixture.destroy();
  }
});

it('F4 suppresses provisional Ground and SMRF classifications at a grid edge', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view('float32x3', new Float32Array([0.5, 0.5, 0.1])).view;
    const classifications = fixture.view('uint32', new Uint32Array([1])).view;
    // SMRF's one-sided slope differences require two rows and columns. The point remains
    // in the lower-left cell, so the radius-one morphology window crosses the grid edge.
    const grid = {origin: [0, 0] as const, cellSize: 1, width: 2, height: 2};
    const cases = [];
    const groundCategory = fixture.view('uint32', new Uint32Array(1));
    for (const kind of ['ground', 'smrf'] as const) {
      const mask = fixture.view('uint32', new Uint32Array(1));
      const status = fixture.view('uint32', new Uint32Array(1));
      const output = {
        groundMask: mask.view,
        status: status.view,
        outputClassifications: fixture.view('uint32', new Uint32Array(1)).view
      };
      fixture.graph.add(
        kind === 'ground'
          ? new GPUPointCloudGround({
              positions,
              classifications,
              grid,
              radii: [1],
              initialThreshold: 0.3,
              slope: 0.1,
              maximumThreshold: 2,
              minimumSurface: fixture.view('float32', new Float32Array(4).fill(0.1)).view,
              coverage: fixture.view('uint32', new Uint32Array(4).fill(1)).view,
              output
            })
          : new GPUPointCloudSMRF({
              positions,
              classifications,
              grid,
              window: 1,
              cut: 0,
              threshold: 0.5,
              output: {...output, groundSurface: fixture.view('float32', new Float32Array(4)).view}
            })
      );
      const display = fixture.view('float32', new Float32Array(1));
      if (kind === 'ground')
        fixture.graph.add(
          createGroundDisplayClassesNode(fixture.graph, mask.view, status.view, groundCategory.view)
        );
      fixture.graph.add(
        createPointCloudDisplayCopyNode(fixture.graph, {
          id: `${kind}-display`,
          source: mask.view,
          status: status.view,
          output: display.view
        })
      );
      cases.push({kind, mask, status, display});
    }
    await fixture.run();
    for (const {kind, mask, status, display} of cases) {
      const codes = await readIntegers(status.buffer);
      expect(codes).toEqual([kind === 'ground' ? 2 : 8]);
      expect(await readIntegers(mask.buffer)).toEqual([1]);
      expect(summarizePointCloudGround(await readIntegers(mask.buffer), codes)).toEqual({
        ground: 0,
        supported: 0,
        incomplete: 1
      });
      expect(Number.isNaN((await readFloats(display.buffer))[0])).toBe(true);
    }
    expect(await readIntegers(groundCategory.buffer)).toEqual([2]);
    expect(LIDAR_GROUND_DISPLAY_CATEGORIES[2].label).toContain('Provisional');
  } finally {
    fixture.destroy();
  }
});

it('G5 mutated caller settings retain the control mask, status writer and height', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const results = [];
  for (const mutate of [false, true]) {
    const fixture = makeSpatialFixture(device);
    try {
      const positions = fixture.view('float32x3', new Float32Array([0.5, 0.5, 0.2])).view;
      const origin: [number, number] = [0, 0];
      const grid = {origin, cellSize: 1, width: 1, height: 1};
      const radii = [1];
      const terrain = fixture.view('float32', new Float32Array([0.1])).view;
      const coverage = fixture.view('uint32', new Uint32Array([1])).view;
      const mask = fixture.view('uint32', new Uint32Array(1));
      const status = fixture.view('uint32', new Uint32Array([37]));
      const height = fixture.view('float32', new Float32Array(1));
      const heightStatus = fixture.view('uint32', new Uint32Array([37]));
      const ground = new GPUPointCloudGround({
        positions,
        classifications: fixture.view('uint32', new Uint32Array([1])).view,
        grid,
        radii,
        minimumSurface: terrain,
        coverage,
        initialThreshold: 0.3,
        slope: 0.1,
        maximumThreshold: 2,
        output: {
          groundMask: mask.view,
          status: status.view,
          outputClassifications: fixture.view('uint32', new Uint32Array(1)).view
        }
      });
      const heightContributor = new GPUPointCloudHeight({
        positions,
        grid,
        terrain,
        coverage,
        output: {heights: height.view, status: heightStatus.view}
      });
      if (mutate) {
        radii.length = 0;
        origin[0] = 100;
      }
      fixture.graph.add(ground);
      fixture.graph.add(heightContributor);
      await fixture.run();
      const values = {
        mask: await readIntegers(mask.buffer),
        status: await readIntegers(status.buffer),
        height: await readFloats(height.buffer),
        heightStatus: await readIntegers(heightStatus.buffer)
      };
      expect(values.mask).toEqual([1]);
      expect(values.status).toEqual([2]);
      expect(values.heightStatus).toEqual([0]);
      expect(values.height[0]).toBeCloseTo(0.1, 6);
      results.push(values);
    } finally {
      fixture.destroy();
    }
  }
  expect(results[1]).toEqual(results[0]);
});

it('H3 Surfaces and Height retain a shared one-cell frame after caller origin mutation', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const results = [];
  for (const mutate of [false, true]) {
    const fixture = makeSpatialFixture(device);
    try {
      const origin: [number, number] = [0, 0];
      const grid = {origin, cellSize: 1, width: 1, height: 1};
      const positions = fixture.view(
        'float32x3',
        new Float32Array([0.5, 0.5, 4, 0.5, 0.5, 5])
      ).view;
      const terrain = fixture.view('float32', new Float32Array([NaN]));
      const coverage = fixture.view('uint32', new Uint32Array([0xffffffff]));
      const heights = fixture.view('float32', new Float32Array(2).fill(NaN));
      const status = fixture.view('uint32', new Uint32Array(2).fill(0xffffffff));
      const surfaces = new GPUPointCloudSurfaces({
        positions,
        grid,
        classifications: fixture.view('uint32', new Uint32Array([2, 1])).view,
        output: {
          terrain: terrain.view,
          coverage: coverage.view,
          surface: fixture.view('float32', new Float32Array(1)).view,
          canopy: fixture.view('float32', new Float32Array(1)).view
        }
      });
      const height = new GPUPointCloudHeight({
        positions,
        grid,
        terrain: terrain.view,
        coverage: coverage.view,
        coverageBit: 2,
        output: {heights: heights.view, status: status.view}
      });
      if (mutate) origin[0] = 100;
      fixture.graph.add(surfaces);
      fixture.graph.add(height);
      await fixture.run();
      const values = {
        terrain: await readFloats(terrain.buffer),
        coverage: await readIntegers(coverage.buffer),
        heights: await readFloats(heights.buffer),
        status: await readIntegers(status.buffer)
      };
      expect(values).toEqual({terrain: [4], coverage: [3], heights: [0, 1], status: [0, 0]});
      results.push(values);
    } finally {
      fixture.destroy();
    }
  }
  expect(results[1]).toEqual(results[0]);
});

it('G2 bundled lidar tile supplies at least one supported height with the showcase settings', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const urls = [
      new URL(
        '../../../../examples/deck/spatial-analysis-showcase/public/data/lidar-ground/position.bin',
        import.meta.url
      ),
      new URL(
        '../../../../examples/deck/spatial-analysis-showcase/public/data/lidar-ground/elevation.bin',
        import.meta.url
      ),
      new URL(
        '../../../../examples/deck/spatial-analysis-showcase/public/data/lidar-ground/classification.bin',
        import.meta.url
      ),
      new URL(
        '../../../../examples/deck/spatial-analysis-showcase/public/data/lidar-ground/manifest.json',
        import.meta.url
      )
    ];
    const responses = await Promise.all(urls.map(url => fetch(url)));
    for (const response of responses) expect(response.ok).toBe(true);
    const [coordinateBytes, elevationBytes, classBytes] = await Promise.all(
      responses.slice(0, 3).map(response => response.arrayBuffer())
    );
    const manifest = await responses[3].json();
    const data = projectLidarGroundTile(
      new Float64Array(coordinateBytes),
      new Float32Array(elevationBytes),
      manifest.bbox
    );
    expect(data.length / 3).toBe(116213);
    const positions = fixture.view('float32x3', data).view;
    const classifications = fixture.view('uint32', new Uint32Array(classBytes)).view;
    const grid = LIDAR_GROUND_GRID;
    const cells = grid.width * grid.height;
    const raster = () => fixture.view('float32', new Float32Array(cells)).view;
    const firstTerrain = raster();
    const firstCoverage = fixture.view('uint32', new Uint32Array(cells)).view;
    const seedCoverage = fixture.view('uint32', new Uint32Array(cells)).view;
    fixture.graph.add(
      new GPUPointCloudSurfaces({
        id: 'tile-seed-surfaces',
        positions,
        classifications,
        grid,
        output: {
          terrain: firstTerrain,
          surface: raster(),
          canopy: raster(),
          coverage: firstCoverage
        }
      })
    );
    fixture.graph.add(createGroundSeedCoverageNode(fixture.graph, firstCoverage, seedCoverage));
    const classified = fixture.view('uint32', new Uint32Array(116213)).view;
    const groundStatus = fixture.view('uint32', new Uint32Array(116213)).view;
    fixture.graph.add(
      new GPUPointCloudGround({
        positions,
        classifications,
        grid,
        radii: LIDAR_GROUND_RADII,
        initialThreshold: 0.3,
        slope: 0.1,
        maximumThreshold: 2,
        minimumSurface: firstTerrain,
        coverage: seedCoverage,
        output: {
          groundMask: fixture.view('uint32', new Uint32Array(116213)).view,
          outputClassifications: classified,
          status: groundStatus
        }
      })
    );
    const supportedClasses = fixture.view('uint32', new Uint32Array(116213)).view;
    fixture.graph.add(
      createSupportedGroundClassesNode(fixture.graph, classified, groundStatus, supportedClasses)
    );
    const terrain = raster();
    const coverage = fixture.view('uint32', new Uint32Array(cells)).view;
    fixture.graph.add(
      new GPUPointCloudSurfaces({
        id: 'tile-recovered-surfaces',
        positions,
        classifications: supportedClasses,
        grid,
        output: {terrain, coverage, surface: raster(), canopy: raster()}
      })
    );
    const heights = fixture.view('float32', new Float32Array(116213).fill(NaN));
    const heightStatus = fixture.view('uint32', new Uint32Array(116213).fill(0xffffffff));
    fixture.graph.add(
      new GPUPointCloudHeight({
        positions,
        grid,
        terrain,
        coverage,
        coverageBit: 2,
        output: {heights: heights.view, status: heightStatus.view}
      })
    );
    await fixture.run();
    const statuses = await readIntegers(heightStatus.buffer);
    const values = await readFloats(heights.buffer);
    expect(statuses.every(code => code !== 0xffffffff)).toBe(true);
    expect(statuses.filter(code => code === 0)).toHaveLength(1155);
    for (let row = 0; row < statuses.length; row++)
      if (statuses[row] === 0) expect(Number.isFinite(values[row])).toBe(true);
  } finally {
    fixture.destroy();
  }
});
