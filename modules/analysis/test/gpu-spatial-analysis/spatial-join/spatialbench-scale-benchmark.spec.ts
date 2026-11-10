// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {getBenchmarkDistribution, getBenchmarkSettings} from '../../utils/benchmark-settings';
import {requireTestDevice} from '../../utils/require-test-device';
import {getSpatialBenchRowCounts} from '../../utils/spatialbench-scale-factors';
import {createRandom} from './spatial-join-oracle';
import {createPolygonJoinRun, createSquareFeature} from './spatial-join-sort-utils';

const environment = (import.meta as unknown as {env?: Record<string, string | undefined>}).env;

/**
 * PLAT-07. Point-in-polygon join with per-polygon counts ("trips per zone", the SpatialBench
 * join and aggregation shape) sized by SpatialBench scale factors: trips = 6M x SF points against
 * the SpatialBench zone count. Zones are a synthetic square grid with the published count, not
 * the Overture zone geometry. Scale factors come from `VITE_SPATIALBENCH_SF` (comma separated,
 * default `0.1,1`). SF10 (60M trips) crashed the headless page on 2026-11-09 because the runner builds the points as
 * JavaScript arrays; SF0.1, 1, 2 and 5 ran.
 */
it('point-in-zone join and per-zone count at SpatialBench scale factors', {
  timeout: 900000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const settings = getBenchmarkSettings(environment);
  const scaleFactors = (environment?.['VITE_SPATIALBENCH_SF'] ?? '0.1,1').split(',').map(Number);
  for (const scaleFactor of scaleFactors) {
    const counts = getSpatialBenchRowCounts(scaleFactor);
    const columns = Math.ceil(Math.sqrt(counts.zone));
    const pointBytes = counts.trip * 8;
    if (pointBytes > device.limits.maxStorageBufferBindingSize) {
      console.log(
        `SPATIALBENCH_SKIPPED sf=${scaleFactor}: ${pointBytes} point bytes exceed the binding limit`
      );
      continue;
    }
    const zones = [];
    for (let zone = 0; zone < counts.zone; zone++) {
      zones.push(createSquareFeature(zone % columns, Math.floor(zone / columns), 0.9));
    }
    const random = createRandom(7001);
    const points: [number, number][] = new Array(counts.trip);
    for (let row = 0; row < counts.trip; row++) {
      points[row] = [Math.fround(random() * columns), Math.fround(random() * columns)];
    }
    const run = createPolygonJoinRun(device, zones, points, {
      spatialSort: true,
      candidateCapacity: counts.trip
    });
    try {
      const result = await run.readResult();
      expect(result.overflow).toBe(0);
      const matched = result.counts.reduce((sum, count) => sum + count, 0);
      // Zones cover 81% of each unit cell, so about 81% of trips land in a zone.
      expect(matched / counts.trip).toBeGreaterThan(0.78);
      expect(matched / counts.trip).toBeLessThan(0.84);
      const samples: number[] = [];
      for (let index = 0; index < settings.warmupCount + settings.sampleCount; index++) {
        const start = performance.now();
        await run.encodeAndWait();
        if (index >= settings.warmupCount) samples.push(performance.now() - start);
      }
      console.log(
        `SPATIALBENCH_JOIN ${JSON.stringify({
          scaleFactor,
          trips: counts.trip,
          zones: counts.zone,
          device: device.info.vendor,
          matched,
          ...getBenchmarkDistribution(samples)
        })}`
      );
    } finally {
      run.destroy();
    }
  }
});
