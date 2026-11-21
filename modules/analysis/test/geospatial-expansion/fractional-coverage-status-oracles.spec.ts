// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPURasterFractionalCoverage,
  getGPURasterFractionalCoverageParameterValues
} from '../../src/raster-analysis/gpu-raster-fractional-coverage';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createParityGraph} from '../oracles/parity-wp2-geodesic-harness';
import {readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';

it('fractional coverage derives output overflow from two occupied cells and clears it on replay', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const rig = createParityGraph(device);
  const grid = {
    width: 2,
    height: 1,
    affine: [1, 0, 0, 0, 1, 0] as const,
    pixelInterpretation: 'area' as const
  };
  const positions = rig.input('positions', [0, 0, 2, 0, 2, 1, 0, 1], 'float32x2');
  const count = rig.output('count', 1, 'uint32');
  const requiredCount = rig.output('required', 1, 'uint32');
  const overflow = rig.output('overflow', 1, 'uint32');
  const candidateOverflow = rig.output('candidate-overflow', 1, 'uint32');
  const zoneValidity = rig.output('zone-validity', 1, 'uint32');
  rig.graph.add(
    new GPURasterFractionalCoverage({
      grid,
      candidateCapacity: 2,
      zones: {
        positions: positions.view,
        ringOffsets: rig.input('rings', [0, 4], 'uint32').view,
        polygonOffsets: rig.input('polygons', [0, 1], 'uint32').view,
        featureOffsets: rig.input('features', [0, 1], 'uint32').view
      },
      parameters: rig.input(
        'parameters',
        Array.from(getGPURasterFractionalCoverageParameterValues(grid)),
        'float32'
      ).view,
      output: {
        zoneIds: rig.output('zones', 1, 'uint32').view,
        cellIds: rig.output('cells', 1, 'uint32').view,
        fractions: rig.output('fractions', 1, 'float32').view,
        status: {
          count: count.view,
          requiredCount: requiredCount.view,
          overflow: overflow.view,
          candidateOverflow: candidateOverflow.view
        },
        zoneValidity: zoneValidity.view
      }
    })
  );
  const compiled = rig.graph.compile();
  try {
    // Closed form: [0,2] x [0,1] occupies exactly two unit cells. Capacity one stores one
    // tuple, requires two and overflows. Shrinking to the first cell removes the overflow.
    // Both candidate rectangles fit the budget of two; no unresolved candidate remains.
    for (const frame of [
      {right: 2, expected: [1, 2, 1, 0, 1]},
      {right: 1, expected: [1, 1, 0, 0, 1]},
      {right: 2, expected: [1, 2, 1, 0, 1]}
    ]) {
      positions.buffer.write(Float32Array.of(0, 0, frame.right, 0, frame.right, 1, 0, 1));
      // Poison complete words, including the zero-only candidate diagnostic, before each run.
      for (const port of [count, requiredCount, overflow, candidateOverflow, zoneValidity]) {
        port.buffer.write(Uint32Array.of(0xdeadbeef));
      }
      submitGraph(device, compiled, undefined);
      const actual = [];
      for (const port of [count, requiredCount, overflow, candidateOverflow, zoneValidity]) {
        actual.push((await readUint32(port.buffer, 1))[0]);
      }
      expect(actual).toEqual(frame.expected);
    }
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});
