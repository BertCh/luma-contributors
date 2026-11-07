// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {runTerrainProcessing} from '../../../src/computation-lenses/scenarios/gpu-terrain/terrain-processing-runner';
import {TERRAIN_PROCESSING_SCENARIOS} from '../../../src/computation-lenses/scenarios/gpu-terrain/terrain-processing.scenarios';
import {requireTestDevice} from '@luma-contributors/analysis/testing';

it.each(
  TERRAIN_PROCESSING_SCENARIOS
)('$element records real GPU outputs and validated sidecars', async ({element, slug}) => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const record = await runTerrainProcessing(device, element, `gpu-terrain.${slug}`);
  const snapshots = record.captures?.snapshots ?? [];
  expect(snapshots.length).toBeGreaterThan(1);
  const read = (name: string) => {
    const capture = snapshots.find(candidate => candidate.name === name);
    expect(capture, name).toBeDefined();
    return Array.from(capture!.values);
  };
  switch (element) {
    case 'GPUTerrainBreach':
      expect(read('editedElevation').slice(0, 3)).toEqual([2, 1.5, 0]);
      expect(read('pitStatus')).toEqual([1]);
      break;
    case 'GPUTerrainBranchingFlow':
      expect(read('transport').slice(0, 5)).toEqual([8, 2, 0, 6, 8]);
      break;
    case 'GPUTerrainFlowFractions': {
      const fractions = read('fractions');
      for (let cell = 0; cell < 8; cell++)
        expect(
          fractions.slice(cell * 8, cell * 8 + 8).reduce((sum, value) => sum + value, 0)
        ).toBeCloseTo(1);
      expect(fractions.slice(64)).toEqual(Array(8).fill(0));
      break;
    }
    case 'GPUTerrainDepressionHierarchy':
      expect(read('status')).toEqual([3, 3, 0]);
      expect(read('curves').slice(8)).toEqual([2, 1, 3, 6]);
      break;
    case 'GPUTerrainCatchmentSummary':
      expect(read('statistics').slice(0, 5)).toEqual([3, 2, 1, 3, 2]);
      break;
    case 'GPUTerrainDEMCoregistration':
      expect(read('fit')).toEqual([2, 0, 4, 0]);
      expect(read('volume')[0]).toBe(101);
      break;
    case 'GPUTerrainTiledFlow':
      expect(read('tile0Accumulation')[0]).toBe(8);
      expect(read('tile0Accumulation')[3]).toBe(6);
      expect(read('tile1Accumulation')[4]).toBe(8);
      break;
  }
});
