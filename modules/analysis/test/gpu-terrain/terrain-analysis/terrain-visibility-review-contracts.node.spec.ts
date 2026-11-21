// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {expect, it} from 'vitest';
import {GPUParameterBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {getGPUSolarIrradianceSunTable} from '../../../src/gpu-terrain/terrain-illumination/gpu-solar-irradiance';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {getGPUTerrainHorizonParameterValues} from '../../../src/gpu-terrain/terrain-illumination/terrain-horizon-parameters';
import {computeProfilePeaksOnCPU} from '../../../src/gpu-terrain/terrain-features/cpu-profile-peaks-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const documentation = readFileSync('docs/api/gpu-terrain/operators.md', 'utf8');
const prose = documentation.replace(/\s+/g, ' ');

it('B3 scopes stream sentinels to their consumer domains', () => {
  expect(
    prose.includes('FlowDistance, StreamNetwork, CatchmentSummary and TiledFlow reserve both')
  ).toBe(true);
  expect(prose.includes('HAND and StreamOrder accept `0xfffffffe` as membership')).toBe(true);
});

it('B4 uploads only complete solar-table rows from a target with trailing padding', () => {
  const device = createNullWebGPUDevice();
  const target = new Float32Array(11).fill(17);
  const table = getGPUSolarIrradianceSunTable({
    longitude: 0,
    latitude: 0,
    start: 0,
    end: 600000,
    stepMinutes: 5,
    target
  });
  // Ten minutes / five-minute intervals = two live and allocated four-float rows.
  expect(table.counts).toEqual({sampleCount: 2, sampleCapacity: 2});
  expect(Array.from(table.values.subarray(8))).toEqual([17, 17, 17]);
  const allocation = documentation.match(
    /const sunTableBuffer = new GPUParameterBuffer\(graph.device, \{[\s\S]*?\n\}\);/
  )?.[0];
  expect(allocation).toBeDefined();
  let buffer: GPUParameterBuffer<'float32'> | undefined;
  try {
    // Execute the actual documented allocation with real returned metadata, rather than a copy.
    buffer = runInNewContext(`${allocation}\nsunTableBuffer`, {
      GPUParameterBuffer,
      graph: {device},
      ...table
    }) as GPUParameterBuffer<'float32'>;
    expect(buffer.length).toBe(8);
  } finally {
    buffer?.destroy();
    device.destroy();
  }
});

it('B5 limits profile precision agreement to thresholding and candidate priority', () => {
  const result = computeProfilePeaksOnCPU(
    Float32Array.of(2 ** -26, 1, 0, 0, 1, 0),
    Uint32Array.of(0, 6),
    Float32Array.of(1),
    {
      window: 1,
      minimumSide: 1,
      suppressionRadius: 3,
      capacity: 6
    }
  );
  // Peak 1 shifts left by 0.5*2^-26/(2^-26-2); peak 4 stays at 4.
  // Their binary64 distance exceeds 3, while its float32 value is exactly 3.
  const refinedDistance = 4 - (1 + (0.5 * 2 ** -26) / (2 ** -26 - 2));
  expect(refinedDistance).toBeGreaterThan(3);
  expect(Math.fround(refinedDistance)).toBe(3);
  expect(Array.from(result.ids.subarray(0, result.count[0]))).toEqual([1, 4]);
  expect([result.prominence[1], result.prominence[4]]).toEqual([1, 1]);
  expect(prose.includes('CPU binary64 intermediates do not change acceptance or the winner')).toBe(
    false
  );
  expect(prose.includes('refined-distance rounding can change which candidates compete')).toBe(
    true
  );
});

it('S1 qualifies missing corners by positive interpolation weight in the detailed model', () => {
  expect(prose.includes('samples with an invalid positive-weight corner')).toBe(true);
});

it('Earlier fix: horizon parameter allocations follow the selected 8/12/24-element layout', () => {
  const grid = makeGPURasterMetadata({width: 2, height: 2, cellSize: 1});
  expect([
    getGPUTerrainHorizonParameterValues({}, grid).length,
    getGPUTerrainHorizonParameterValues({rvtDirectionDegrees: 90}, grid).length,
    getGPUTerrainHorizonParameterValues(
      {sweepReferenceCellSize: [1, 1], sweepSourceGrid: grid},
      grid
    ).length
  ]).toEqual([8, 12, 24]);
  expect(prose.includes('8 base, 12 anisotropic, or 24 sweep float32 elements')).toBe(true);
});
