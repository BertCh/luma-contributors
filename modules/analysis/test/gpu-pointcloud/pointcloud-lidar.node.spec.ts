// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../utils/gpu-contributor-test-utils';
import {auditWGSLModule} from '../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  LIDAR_GROUND_GRID,
  LIDAR_GROUND_RADII,
  LIDAR_GROUND_DISPLAY_CATEGORIES,
  getLidarGroundDisplayClass,
  createGroundDisplayClassesNode
} from '../../../../examples/deck/spatial-analysis-showcase/src/scenes/terrain/point-cloud-ground-support';
import {projectLidarGroundTile, replayLidarGroundSupport} from './pointcloud-lidar-fixture';

it('H4 actual bundled-tile readback checks reject untouched and partially written outputs', () => {
  const source = readFileSync(
    'modules/analysis/test/gpu-pointcloud/pointcloud-website.spec.ts',
    'utf8'
  );
  const bundled = source.slice(source.indexOf("it('G2 bundled lidar tile"));
  const assertions = bundled.match(
    /const values = await readFloats\(heights.buffer\);\n([\s\S]*?)\n  \} finally/
  )?.[1];
  expect(assertions).toBeDefined();
  const check = (statuses: number[], values: number[]) =>
    runInNewContext(assertions!, {expect, statuses, values});
  expect(() => check(new Array(116213).fill(0), new Array(116213).fill(0))).toThrow();
  const partial = new Array(116213).fill(0xffffffff);
  partial.fill(0, 0, 1155);
  expect(() => check(partial, new Array(116213).fill(0))).toThrow();
  const complete = new Array(116213).fill(2);
  complete.fill(0, 0, 1155);
  expect(() => check(complete, new Array(116213).fill(0))).not.toThrow();
  expect(() => check(complete, new Array(116213).fill(NaN))).toThrow();
});

it('G2 ground display binds support-qualified categories instead of the raw acceptance mask', () => {
  const source = readFileSync(
    'examples/deck/spatial-analysis-showcase/src/scenes/terrain/lidar-ground.compute.ts',
    'utf8'
  );
  const binding = source.match(/id: 'lidar-ground-pmf-classification',[\s\S]*?values: (\w+)/)?.[1];
  expect(binding).toBeDefined();
  expect(binding).not.toBe('groundMaskBuffer');
});

it('G2 bundled lidar tile retains fully supported terrain and heights with the showcase settings', () => {
  const directory = 'examples/deck/spatial-analysis-showcase/public/data/lidar-ground/';
  const manifest = JSON.parse(readFileSync(`${directory}manifest.json`, 'utf8'));
  const bytes = (name: string) => {
    const data = readFileSync(`${directory}${name}.bin`);
    return data.buffer.slice(data.byteOffset, data.byteOffset + data.byteLength);
  };
  const positions = projectLidarGroundTile(
    new Float64Array(bytes('position')),
    new Float32Array(bytes('elevation')),
    manifest.bbox
  );
  const support = replayLidarGroundSupport(
    positions,
    new Uint32Array(bytes('classification')),
    LIDAR_GROUND_GRID,
    LIDAR_GROUND_RADII
  );
  expect(positions.length / 3).toBe(116213);
  expect(support.supportedAccepted, JSON.stringify(support)).toBeGreaterThan(0);
  expect(support.supportedHeights).toBeGreaterThan(0);
  expect(support.supportedHeights, JSON.stringify(support)).toBe(1155);
  // Fixed expected count from the independent replay of this bundled tile.
  expect(support.supportedAccepted).toBe(1155);
});

it('G2 provisional and unsupported ground never receive the accepted category', () => {
  expect(getLidarGroundDisplayClass(1, 0)).toBe(1);
  expect(getLidarGroundDisplayClass(0, 0)).toBe(0);
  expect(getLidarGroundDisplayClass(1, 2)).toBe(2);
  expect(LIDAR_GROUND_DISPLAY_CATEGORIES[2].label).toContain('Provisional');
  expect(getLidarGroundDisplayClass(1, 1)).toBe(3);
  expect(getLidarGroundDisplayClass(0xffffffff, 0)).toBe(3);
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  const mask = createTransientView(graph, 'mask', 'uint32', 1);
  const status = createTransientView(graph, 'status', 'uint32', 1);
  const output = createTransientView(graph, 'display', 'uint32', 1);
  const [shader] = getGPUCommandNodeKernelSources([
    createGroundDisplayClassesNode(graph, mask, status, output)
  ]);
  expect(shader.source).toContain('select(3u, 2u, code == 2u)');
  expect(auditWGSLModule(shader.source)).toEqual([]);
});
