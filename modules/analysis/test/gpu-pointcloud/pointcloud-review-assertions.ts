// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect} from 'vitest';
import {GPU_POINT_CLOUD_ICP_STATUS_CONVERGED} from '@luma-contributors/analysis/pointcloud-analysis';

export function expectPointCloudChangeBounds(
  changes: readonly number[],
  detectionLimit: number,
  distance: number,
  sampleLimit: number
) {
  expect(changes).toHaveLength(3);
  expect(changes.every(Number.isFinite)).toBe(true);
  expect(Number.isFinite(detectionLimit)).toBe(true);
  // Rounding each population deviation to 0.001 perturbs the norm by at most
  // 1.96*0.0005*sqrt(1/(n0-1)+1/(n1-1)) <= 0.001386 for counts >= 2.
  // Add 0.0005 for rounded distance and a small float32 allowance to bound endpoints.
  expect(Math.abs(changes[1] - (distance - sampleLimit))).toBeLessThan(0.002);
  expect(Math.abs(changes[2] - (distance + sampleLimit))).toBeLessThan(0.002);
  expect(Math.abs(detectionLimit - sampleLimit)).toBeLessThan(0.0015);
}

export function expectPointCloudICPConverged(status: readonly number[]) {
  expect(status).toEqual([GPU_POINT_CLOUD_ICP_STATUS_CONVERGED]);
}
