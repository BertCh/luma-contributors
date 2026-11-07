// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {requireTestDevice} from '@luma-contributors/analysis/testing';
import {listLensScenarios} from '../../src/computation-lenses';
import '../../src/computation-lenses/scenarios/gpu-spatial-analysis/global-coordinates.lens';

const elements = [
  'GPUS2CellHierarchy',
  'GPUS2CellMeasures',
  'GPUS2CapCovering',
  'GPUSphericalConvexContains',
  'GPUSphericalConvexIntersection',
  'GPUS2PolygonCovering',
  'GPUSphericalPolygonContains',
  'GPUSphericalPolygonOverlay',
  'GPUSphericalArcOverlay'
];
for (const element of elements)
  it(`${element} lens captures numerically checkable GPU results`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scenario = listLensScenarios().find(candidate => candidate.element === element)!;
    const loaded = await scenario.load();
    const result = await loaded.run(device);
    expect(result.graphs.length).toBe(1);
    expect(result.graphs[0].nodes.length).toBeGreaterThan(0);
    expect(result.captures?.runId).toBe(result.runId);
    const read = (name: string) =>
      Array.from(result.captures!.snapshots.find(snapshot => snapshot.name === name)!.values);
    if (element === 'GPUS2CellHierarchy') {
      expect(read('status')).toEqual([0, 0, 0, 0, 0, 0]);
      const children = read('children');
      for (let face = 0; face < 6; face++)
        for (let child = 0; child < 4; child++) {
          const key = (BigInt(face) << 61n) + (BigInt(2 * child + 1) << 58n);
          expect(children.slice((face * 4 + child) * 2, (face * 4 + child + 1) * 2)).toEqual([
            Number(key & 0xffffffffn),
            Number(key >> 32n)
          ]);
        }
    } else if (element === 'GPUS2CellMeasures') {
      expect(read('status')).toEqual([0, 0, 0, 0, 0, 0]);
      const measures = read('areaPerimeter');
      for (let face = 0; face < 6; face++) {
        expect(measures[face * 2]).toBeCloseTo((2 * Math.PI) / 3, 5);
        expect(measures[face * 2 + 1]).toBeCloseTo(4 * Math.acos(1 / 3), 5);
      }
    } else if (element === 'GPUS2CapCovering') {
      expect(read('status')).toEqual([0]);
      expect(read('counts')).toEqual([1, 1]);
      expect(read('coverCells')).toEqual([0, 0x50000000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    } else if (
      element === 'GPUSphericalConvexContains' ||
      element === 'GPUSphericalPolygonContains'
    ) {
      expect(read('contains')).toEqual([1, 0, 1]);
      expect(read('status')).toEqual([0, 0, 2]);
    } else if (element === 'GPUSphericalPolygonOverlay') {
      expect(read('resultOverflow')).toEqual([0]);
      expect(read('resultCandidateOverflow')).toEqual([0]);
      expect(read('resultCount')).toEqual([1]);
      expect(read('complement')).toEqual([0]);
    } else if (element === 'GPUSphericalArcOverlay') {
      expect(read('status')).toEqual([0]);
      expect(read('counts')).toEqual([3, 1, 3, 0]);
    } else if (element === 'GPUS2PolygonCovering') {
      expect(read('status')).toEqual([0]);
      expect(read('counts')[0]).toBeGreaterThan(0);
      expect(read('counts')[0]).toBe(read('counts')[1]);
      expect(read('visited')[0]).toBeLessThanOrEqual(126);
    } else {
      expect(read('status')).toEqual([0]);
      expect(read('counts')).toEqual([3, 3]);
      const measures = read('areaFractionsUnion');
      expect(measures[0]).toBeCloseTo(Math.PI / 2, 5);
      expect(measures[1]).toBeCloseTo(1, 6);
      expect(measures[2]).toBeCloseTo(1, 6);
      expect(measures[3]).toBeCloseTo(Math.PI / 2, 5);
    }
  });
