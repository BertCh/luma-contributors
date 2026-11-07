// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {getLensScenario} from '../../../src/computation-lenses/registry';
import '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/point-process.lens';
import '../../../src/computation-lenses/scenarios/gpu-network/point-process.lens';
import {requireTestDevice} from '@luma-contributors/graph/testing';

for (const [id, output, expected] of [
  ['gpu-spatial-analysis.inhomogeneous-point-pattern', 'k', [5 / 3.5, 2]],
  ['gpu-spatial-analysis.point-pattern-envelope', 'statistics', [13, 0.5, 0.25, 3]],
  ['gpu-spatial-analysis.spatial-balanced-sample', 'inclusionProbabilities', [0.5, 0.5]],
  ['gpu-spatial-analysis.optics', 'labels', [0, 0, 1, 1]],
  ['gpu-network.network-kernel-density', 'intensity', [1, 1, 1]],
  ['gpu-network.network-inhomogeneous-k', 'k', [0, 1]]
] as const) {
  it(`executes and captures ${id}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const descriptor = getLensScenario(id)!;
    const loaded = await descriptor.load();
    const result = await loaded.run(device);
    expect(result.graphs[0].nodes.length).toBeGreaterThan(0);
    const snapshot = result.captures?.snapshots.find(capture => capture.name === output);
    expect(snapshot).toBeDefined();
    expect(snapshot!.values.length).toBe(expected.length);
    expected.forEach((value, index) => expect(snapshot!.values[index]).toBeCloseTo(value, 5));
    const status = result.captures?.snapshots.find(capture => capture.name === 'status');
    expect(status).toBeDefined();
    const expectedStatus = id.endsWith('.optics')
      ? [4, 1, 0]
      : id.endsWith('.spatial-balanced-sample')
        ? [2, 0]
        : new Array(status!.values.length).fill(0);
    expect(Array.from(status!.values)).toEqual(expectedStatus);
  });
}
