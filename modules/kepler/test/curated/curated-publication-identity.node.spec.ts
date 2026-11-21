// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {createNullWebGPUDevice} from '../../../analysis/test/utils/gpu-contributor-test-utils';
import {CatalogGpuSession} from '../../src/analysis/catalog-gpu-session';
import {
  CURATED_FLEET_DWELL_BINDING,
  prepareFleetDwell,
  makeFleetDwellOutput
} from '../../src/curated/bindings/curated-fleet-dwell-binding';
import {
  CURATED_TRACK_QUALITY_BINDING,
  prepareTrackQuality,
  publishTrackQuality
} from '../../src/curated/bindings/curated-track-quality-binding';
import {createFleetDwellSample} from '../../src/curated/samples/fleet-dwell';
import {createTrackQualitySample} from '../../src/curated/samples/track-quality';

it.each([
  'fleet-dwell',
  'track-quality'
] as const)('rejects ambiguous %s identities before allocation or submission and retains distinct source IDs', async name => {
  const inputs =
    name === 'fleet-dwell' ? createFleetDwellSample() : {tracks: createTrackQualitySample()};
  const first = name === 'fleet-dwell' ? 'vehicle-0' : 'outlier';
  const second = name === 'fleet-dwell' ? 'vehicle-1' : 'gap';
  // Numeric 3 and text "3" are distinct source identities but have the same published spelling.
  for (const feature of inputs['tracks'].features) {
    const properties = feature.properties!;
    if (properties['track_id'] === first) properties['track_id'] = 3;
    else if (properties['track_id'] === second) properties['track_id'] = '3';
  }
  const binding =
    name === 'fleet-dwell' ? CURATED_FLEET_DWELL_BINDING : CURATED_TRACK_QUALITY_BINDING;
  const device = createNullWebGPUDevice();
  const allocated = vi.spyOn(device, 'createBuffer');
  const submitted = vi.spyOn(CatalogGpuSession.prototype, 'run').mockImplementation(() => {
    throw new Error('must not reach submission');
  });
  try {
    await expect(binding.runGpu!({device}, inputs, {coordinateSystem: 'planar'})).rejects.toThrow(
      /distinct string.*track IDs/
    );
    expect(allocated).not.toHaveBeenCalled();
    expect(submitted).not.toHaveBeenCalled();
  } finally {
    allocated.mockRestore();
    submitted.mockRestore();
    device.destroy();
  }
  for (const feature of inputs['tracks'].features)
    if (feature.properties!['track_id'] === '3') feature.properties!['track_id'] = 'text-3';
  const trajectory =
    name === 'fleet-dwell'
      ? prepareFleetDwell(inputs, {coordinateSystem: 'planar'}).trajectory
      : prepareTrackQuality(inputs['tracks'], {coordinateSystem: 'planar'}).trajectory;
  expect(trajectory.trackIds).toContain('3');
  expect(trajectory.trackIds).toContain('text-3');
  expect(trajectory.fixes.map(fix => fix.sourceIndex).sort((left, right) => left - right)).toEqual(
    inputs['tracks'].features.map((_, index) => index)
  );
  for (const fix of trajectory.fixes)
    expect(fix.properties['track_id']).toBe(
      inputs['tracks'].features[fix.sourceIndex].properties!['track_id']
    );
  if (name === 'fleet-dwell') {
    const prepared = prepareFleetDwell(inputs, {coordinateSystem: 'planar'});
    // First observations of the first two 12-fix tracks retain source rows 0 and 12.
    const output = makeFleetDwellOutput(prepared, {
      count: 2,
      requiredCount: 2,
      stopOverflow: 0,
      joinOverflow: 0,
      joinUncertainCount: 0,
      tableOverflow: 0,
      tableCount: 3,
      tracks: Uint32Array.of(0, 1),
      startRows: Uint32Array.of(0, 12),
      endRows: Uint32Array.of(0, 12),
      centroids: new Float32Array(4),
      durations: new Float32Array(2),
      stopZones: Uint32Array.of(0xffffffff, 0xffffffff),
      counts: new Uint32Array(3),
      totals: new Float32Array(3),
      means: new Float32Array(3),
      maximums: new Float32Array(3)
    });
    expect(output.tables[0].columns['track_id']).toEqual(['3', 'text-3']);
    expect(output.tables[0].columns['source_index']).toEqual([0, 12]);
  } else {
    const prepared = prepareTrackQuality(inputs['tracks'], {coordinateSystem: 'planar'});
    const output = publishTrackQuality(prepared.trajectory, prepared.settings, {
      stepSpeeds: new Float32Array(8),
      trackLengths: new Float32Array(3),
      trackDurations: new Float32Array(3),
      maximumSpeeds: new Float32Array(3),
      flags: new Uint32Array(8),
      flaggedCounts: new Uint32Array(3)
    });
    // Source fixture order is [3,6,0,7,2,5,4,1]; publication sorts within each track.
    expect(output.tables[0].columns['source_index']).toEqual([2, 7, 4, 0, 6, 5, 1, 3]);
    expect(output.tables[1].columns['track_id']).toEqual(['3', 'text-3', 'singleton']);
  }
});
