// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPURatioConcaveHullOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-ratio-concave-hull-operation';
import {
  GPURatioConcaveHullOperation,
  type RatioConcaveHullOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-ratio-concave-hull-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {createLatticeFixture} from '../delaunay-tessellation/delaunay-reference';

for (const scenario of ['rounds', 'holes', 'collinear'] as const) {
  it(`ratio hull operations agree on every output for ${scenario}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const positions =
      scenario === 'holes'
        ? createLatticeFixture(25, 5)
        : scenario === 'collinear'
          ? Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0)
          : Float32Array.of(
              11.75,
              5.25,
              4.5,
              8.5,
              14,
              10,
              8,
              5.75,
              12.5,
              1.75,
              12.5,
              3.25,
              6.75,
              13.25,
              9.25,
              2.25,
              10.75,
              5.75
            );
    const props: RatioConcaveHullOperationProps = {
      positions: 'positions',
      pointCount: positions.length / 2,
      ratio: 0.2,
      allowHoles: scenario === 'holes',
      output: {
        triangles: 'triangles',
        triangleValidity: 'triangleValidity',
        boundaryEdges: 'boundaryEdges',
        edgeValidity: 'edgeValidity',
        status: {converged: 'converged'}
      }
    };
    const outputs = ['triangles', 'triangleValidity', 'boundaryEdges', 'edgeValidity', 'converged'];
    const results = [];
    for (const Operation of [GPURatioConcaveHullOperation, CPURatioConcaveHullOperation]) {
      results.push(
        await runOperationGraph(device, {
          inputs: {positions: {format: 'float32x2', values: positions}},
          operations: [new Operation(props)],
          outputs
        })
      );
    }
    // Zero tolerance: integer topology/masks. The chosen edges are well away from the threshold;
    // the lattice also exercises exact length ties and lowest-row selection. Spare Delaunay
    // triangle rows are unwritten; their triangle/edge words are masked and unspecified.
    const realTriangleCount = scenario === 'rounds' ? 11 : scenario === 'holes' ? 32 : 0;
    for (const output of outputs) {
      expect(results[0][output].length).toBe(results[1][output].length);
      const length =
        output === 'triangles'
          ? 3 * realTriangleCount
          : output === 'boundaryEdges'
            ? 6 * realTriangleCount
            : results[0][output].length;
      expect(Array.from(results[0][output].subarray(0, length))).toEqual(
        Array.from(results[1][output].subarray(0, length))
      );
    }
    expect(results[0]['converged'][0]).toBe(1);
    if (scenario === 'rounds')
      expect([...results[0]['triangleValidity']].reduce((sum, valid) => sum + valid, 0)).toBe(7);
    if (scenario === 'collinear')
      expect([...results[0]['edgeValidity']].every(valid => valid === 0)).toBe(true);
  });
}
