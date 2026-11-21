// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {getLensScenario} from '../../../src/computation-lenses/registry-implementation';
import '../../../src/computation-lenses/scenarios/gpu-spatial-analysis/point-process.lens';
import '../../../src/computation-lenses/scenarios/gpu-network/point-process.lens';
import {requireTestDevice} from '@luma-contributors/analysis/testing';

for (const [id, output, expected, expectedStatus] of [
  ['gpu-spatial-analysis.inhomogeneous-point-pattern', 'k', [5 / 3.5, 2], [0, 0]],
  ['gpu-spatial-analysis.point-pattern-envelope', 'statistics', [13, 0.5, 0.25, 3], [0]],
  ['gpu-spatial-analysis.spatial-balanced-sample', 'inclusionProbabilities', [0.5, 0.5], [2, 0]],
  ['gpu-spatial-analysis.optics', 'labels', [0, 0, 1, 1], null],
  ['gpu-network.network-kernel-density', 'intensity', [1, 1, 1], null],
  ['gpu-network.network-inhomogeneous-k', 'k', [0, 1], [0, 0]]
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
    if (id === 'gpu-network.network-kernel-density') {
      const validity = result.captures?.snapshots.find(capture => capture.name === 'validity');
      expect(validity).toBeDefined();
      expect(Array.from(validity!.values)).toEqual([1, 1, 1]);
      return;
    }
    if (id === 'gpu-spatial-analysis.optics') {
      // All four fixture points are visited, and each has finite coordinates.
      for (const [name, value] of [
        ['iterationCount', 4],
        ['converged', 1],
        ['invalidCount', 0]
      ] as const) {
        const diagnostic = result.captures?.snapshots.find(capture => capture.name === name);
        expect(diagnostic, name).toBeDefined();
        expect(Array.from(diagnostic!.values), name).toEqual([value]);
      }
      return;
    }
    const status = result.captures?.snapshots.find(capture => capture.name === 'status');
    expect(status).toBeDefined();
    expect(Array.from(status!.values)).toEqual(expectedStatus);
  });
}
