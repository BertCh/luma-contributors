// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  GPUSpatialKMeans,
  getGPUSpatialKMeansParameterValues
} from '../../../src/gpu-spatial-analysis/spatial-clustering/gpu-spatial-kmeans';
import {GraphHarness} from '../trajectory-staypoints/movement-spec-utils';
import {readFloat32, readUint32} from '../../utils/gpu-contributor-test-utils';

// k=1, initial first-valid center (0,0), points (0,0),(2,0). The first
// mean is (1,0), a shift of 1 > tolerance 0: work remains at budget 1.
// The next mean is unchanged, exactly a fixed point; iterationCount=2,
// converged=1 even with budget 3. These values require no iterative solver.
for (const scenario of [
  {budget: 1, status: [1, 0]},
  {budget: 2, status: [2, 1]},
  {budget: 3, status: [2, 1]}
]) {
  it(`kmeans exact mean-shift stopping oracle at budget ${scenario.budget}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = new GraphHarness(device, 'kmeans-status');
    const iterations = harness.output('iterations', 'uint32', 1);
    const converged = harness.output('converged', 'uint32', 1);
    const centers = harness.output('centers', 'float32x2', 1);
    for (const output of [iterations, converged])
      output.buffer.write(new Uint32Array([0xdeadbeef]));
    harness.graph.add(
      new GPUSpatialKMeans({
        positions: harness.input('positions', new Float32Array([0, 0, 2, 0]), 'float32x2', 2),
        clusterCount: 1,
        initialization: 'first-valid',
        maximumIterations: scenario.budget,
        parameters: harness.input(
          'parameters',
          getGPUSpatialKMeansParameterValues({seed: 0, tolerance: 0}),
          'uint32',
          2
        ),
        output: {
          labels: harness.output('labels', 'uint32', 2).view,
          centers: centers.view,
          status: {iterationCount: iterations.view, converged: converged.view}
        }
      })
    );
    await harness.run(async () => {
      expect([
        ...(await readUint32(iterations.buffer, 1)),
        ...(await readUint32(converged.buffer, 1))
      ]).toEqual(scenario.status);
      expect(await readFloat32(centers.buffer, 2)).toEqual([1, 0]);
    });
  });
}
