// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {GUARDED_ATAN2_WGSL} from '../../src/analysis-core/kernels/wgsl-common';
import {createWGSLKernelNode} from '../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  submitGraph
} from './gpu-contributor-test-utils';
import {requireTestDevice} from './require-test-device';

// Smallest float32 subnormal is about 1.4e-45; 1e-40 is a subnormal that float32 represents.
const SUBNORMAL = Math.fround(1e-40);
const ANGLE_TOLERANCE = 1e-6;

/** [y, x] rows: both signs of zero in each argument, subnormals, and all four quadrants. */
const INPUT_ROWS: [number, number][] = [
  [0, 0],
  [-0, 0],
  [0, -0],
  [-0, -0],
  [1, 0],
  [1, -0],
  [-1, 0],
  [-1, -0],
  [2.5, -0],
  [-0.001, -0],
  [1e-30, -0],
  [0, 1],
  [-0, 1],
  [0, -1],
  [-0, -1],
  [1, SUBNORMAL],
  [1, -SUBNORMAL],
  [-1, -SUBNORMAL],
  [SUBNORMAL, -1],
  [-SUBNORMAL, -1],
  [SUBNORMAL, 1],
  [SUBNORMAL, SUBNORMAL],
  [SUBNORMAL, -SUBNORMAL],
  [-SUBNORMAL, -0],
  [1, 2],
  [1, -2],
  [-1, -2],
  [-1, 2],
  [3, 4],
  [-4, -3],
  [1e-30, -1],
  [-1e-30, -1],
  [100, 1e-3],
  [100, -1e-3]
];

function flushSubnormal(value: number): number {
  return value !== 0 && Math.abs(value) < 1.1754943508222875e-38 ? (value < 0 ? -0 : 0) : value;
}

/**
 * The documented contract against `Math.atan2`: a zero `x` of either sign is +0; a zero `y` with a
 * negative `x` may give +pi or -pi; both zero gives zero of either sign; subnormals may be flushed.
 */
function getAcceptedAngles(y: number, x: number): number[] {
  const accepted: number[] = [];
  for (const candidateY of [y, flushSubnormal(y)]) {
    for (const candidateX of [x, flushSubnormal(x)]) {
      const guardedX = candidateX === 0 ? 0 : candidateX;
      if (candidateY === 0 && guardedX < 0) {
        accepted.push(Math.PI, -Math.PI);
      } else if (candidateY === 0 && guardedX === 0) {
        accepted.push(0);
      } else {
        accepted.push(Math.atan2(candidateY, guardedX));
      }
    }
  }
  return accepted;
}

// FX-17: `atan2(y, -0.0)` returns `-sign(y) * pi / 2` on the test GPU. `guardedAtan2` must match
// `Math.atan2` (binary64) on every row except for the documented exceptions above.
it('guardedAtan2 matches Math.atan2 with a zero second argument treated as +0', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rowCount = INPUT_ROWS.length;
  const graph = new GPUCommandGraph(device, {id: 'wgsl-guarded-atan2'});
  const inputBuffer = createInputBuffer(device, Float32Array.from(INPUT_ROWS.flat()));
  const outputBuffer = createOutputBuffer(device, rowCount * 2);
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'guarded-atan2',
      operation: 'WGSLGuardedAtan2Test',
      bindings: [
        {
          name: 'inputs',
          view: importGraphBuffer(graph, 'inputs', inputBuffer, 'float32', rowCount * 2),
          type: 'f32',
          access: 'read'
        },
        {
          name: 'angles',
          view: importGraphBuffer(graph, 'angles', outputBuffer, 'float32', rowCount * 2),
          type: 'f32',
          access: 'read_write'
        }
      ],
      invocationCount: rowCount,
      declarations: GUARDED_ATAN2_WGSL,
      body: `let y = inputs[inputsOffset + index * 2u];
  let x = inputs[inputsOffset + index * 2u + 1u];
  angles[anglesOffset + index * 2u] = guardedAtan2(y, x);
  angles[anglesOffset + index * 2u + 1u] = atan2(y, x);`
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const angles = await readFloat32(outputBuffer, rowCount * 2);
  compiled.destroy();
  inputBuffer.destroy();
  outputBuffer.destroy();

  // The builtin result is written next to each row only for the failure message. Observed on the
  // test GPU on 2026-11-10: atan2(y, -0) = -sign(y) * pi / 2, atan2(+-0, +-0) = NaN, subnormals are
  // flushed to zero (atan2(1, -1e-40) = -pi / 2), and atan2(-0, -1) = +pi.
  const failures: string[] = [];
  INPUT_ROWS.forEach(([y, x], row) => {
    const guarded = angles[row * 2];
    const builtin = angles[row * 2 + 1];
    const label = `atan2(${Object.is(y, -0) ? '-0' : y}, ${Object.is(x, -0) ? '-0' : x})`;
    const accepted = getAcceptedAngles(y, x);
    if (!accepted.some(angle => Math.abs(guarded - angle) <= ANGLE_TOLERANCE)) {
      failures.push(`${label}: guarded ${guarded}, builtin ${builtin}, accepted ${accepted}`);
    }
  });
  expect(failures).toEqual([]);
});
