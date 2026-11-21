// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../utils/require-test-device';
import fixture from '../../oracles/fixtures/space-time-hdbscan.json';
import {runHDBSCAN} from './hierarchical-density-clustering-harness';
import {canonicalizeLabels, getAdjustedRandIndex} from './hierarchical-density-clustering-oracle';

it('ST-HDBSCAN matches sklearn on scaled coordinates including ties, padding and maximum samples', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    for (const scene of fixture.scenes) {
      const actual = await runHDBSCAN(device, Float32Array.from(scene.positions.flat()), {
        times: Float32Array.from(scene.times),
        timeScale: scene.timeScale,
        minimumSamples: scene.minimumSamples,
        minimumClusterSize: scene.minimumClusterSize
      });
      expect(
        getAdjustedRandIndex(canonicalizeLabels(actual.labels), scene.labels),
        scene.name
      ).toBeGreaterThanOrEqual(0.97);
      expect(actual.clusterCount, scene.name).toBe(scene.clusterCount);
      actual.coreDistances.forEach((distance, row) => {
        expect(
          Math.abs(distance - scene.coreDistances[row]),
          `${scene.name} row ${row}`
        ).toBeLessThanOrEqual(2e-5 * Math.max(1, scene.coreDistances[row]));
      });
    }
  } finally {
    device.destroy();
  }
}, 300000);

it('ST-HDBSCAN excludes invalid times, including when minimumSamples is one', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    const scene = fixture.scenes[0];
    for (const minimumSamples of [1, scene.minimumSamples]) {
      const baseline = await runHDBSCAN(device, Float32Array.from(scene.positions.flat()), {
        times: Float32Array.from(scene.times),
        timeScale: 1,
        minimumSamples,
        minimumClusterSize: scene.minimumClusterSize
      });
      const actual = await runHDBSCAN(
        device,
        Float32Array.from([...scene.positions.flat(), 0, 0, 1, 1, NaN, 0]),
        {
          times: Float32Array.from([...scene.times, NaN, Infinity, 0]),
          timeScale: 1,
          minimumSamples,
          minimumClusterSize: scene.minimumClusterSize
        }
      );
      expect(actual.labels.slice(-3)).toEqual([0xffffffff, 0xffffffff, 0xffffffff]);
      expect(actual.coreDistances.slice(-3)).toEqual([Infinity, Infinity, Infinity]);
      expect(canonicalizeLabels(actual.labels.slice(0, -3))).toEqual(
        canonicalizeLabels(baseline.labels)
      );
      expect(actual.clusterCount).toBe(baseline.clusterCount);
    }
  } finally {
    device.destroy();
  }
}, 180000);

it('ST-HDBSCAN handles empty valid sets, one row and too few finite samples', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    for (const times of [[NaN, Infinity, NaN], [0], [0, 1, 2]]) {
      const minimumSamples = times.length === 1 ? 1 : 4;
      const actual = await runHDBSCAN(device, new Float32Array(times.length * 2), {
        times: Float32Array.from(times),
        timeScale: 1,
        minimumSamples,
        minimumClusterSize: 2
      });
      expect(actual.labels).toEqual(times.map(() => 0xffffffff));
      expect(actual.coreDistances).toEqual(times.map(() => (minimumSamples === 1 ? 0 : Infinity)));
      expect(actual.clusterCount).toBe(0);
    }
  } finally {
    device.destroy();
  }
}, 120000);

it('ST-HDBSCAN re-encodes the same graph with changed time scales', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  try {
    const scene = fixture.scenes[0];
    const updates = [fixture.scenes[2], fixture.scenes[1], scene];
    const actual = await runHDBSCAN(device, Float32Array.from(scene.positions.flat()), {
      times: Float32Array.from(scene.times),
      timeScale: scene.timeScale,
      timeScaleUpdates: updates.map(entry => entry.timeScale),
      minimumSamples: scene.minimumSamples,
      minimumClusterSize: scene.minimumClusterSize
    });
    actual.updated.forEach((result, index) => {
      const reference = updates[index];
      expect(
        getAdjustedRandIndex(canonicalizeLabels(result.labels), reference.labels),
        reference.name
      ).toBeGreaterThanOrEqual(0.97);
      expect(result.clusterCount).toBe(reference.clusterCount);
      result.coreDistances.forEach((distance, row) => {
        expect(Math.abs(distance - reference.coreDistances[row])).toBeLessThanOrEqual(
          2e-5 * Math.max(1, reference.coreDistances[row])
        );
      });
    });
  } finally {
    device.destroy();
  }
}, 180000);
