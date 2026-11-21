// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUDelaunayTessellationOperation} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/cpu-delaunay-tessellation-operation';
import {
  GPUDelaunayTessellationOperation,
  type DelaunayTessellationOperationProps
} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation-operation';
import {getDelaunayStepCount} from '../../../src/gpu-spatial-analysis/delaunay-tessellation/gpu-delaunay-tessellation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

for (const scenario of ['groups', 'overflow', 'collinear', 'duplicates'] as const) {
  it(`Delaunay operations agree on topology and report their iteration counts on ${scenario}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const positions =
      scenario === 'duplicates'
        ? Float32Array.of(2, 2, 2, 2, 2, 2, 2, 2)
        : scenario === 'collinear'
          ? Float32Array.of(0, 0, 1, 0, 2, 0, 2, 0)
          : Float32Array.of(0, 0, 4, 0, 4, 4, 0, 4, 2, 1, 10, 0, 13, 0, 13, 3, 10, 3, 10, -0);
    const grouped = scenario === 'groups';
    const props: DelaunayTessellationOperationProps = {
      id: 'delaunay-twin',
      positions: 'positions',
      capacity: scenario === 'overflow' ? 2 : 20,
      groups: grouped ? {ids: 'groupIds', count: 2} : undefined,
      output: {
        triangles: 'triangles',
        status: {
          count: 'count',
          requiredCount: 'requiredCount',
          overflow: 'overflow',
          invalidCount: 'invalidCount',
          duplicateCount: 'duplicateCount',
          converged: 'converged',
          iterationCount: 'iterationCount',
          iterationLimitReached: 'iterationLimitReached'
        },
        groups: grouped
          ? {
              triangleCounts: 'groupTriangleCounts',
              triangleOffsets: 'groupTriangleOffsets',
              status: {converged: 'groupConverged'}
            }
          : undefined
      }
    };
    const outputs = [
      'triangles',
      ...Object.values(props.output.status),
      ...(grouped ? ['groupTriangleCounts', 'groupTriangleOffsets', 'groupConverged'] : [])
    ];
    const results = [];
    for (const Operation of [GPUDelaunayTessellationOperation, CPUDelaunayTessellationOperation]) {
      results.push(
        await runOperationGraph(device, {
          inputs: {
            positions: {format: 'float32x2', values: positions},
            ...(grouped
              ? {
                  groupIds: {
                    format: 'uint32' as const,
                    values: Uint32Array.of(0, 0, 0, 0, 0, 1, 1, 1, 1, 1)
                  }
                }
              : {})
          },
          operations: [new Operation(props)],
          outputs
        })
      );
    }
    // Zero tolerance: canonical indices and status/group words are integer exact. The two
    // iteration meanings differ, so both iterationCount results are checked separately below.
    // Delaunay leaves spare triangle rows unwritten, so their words have no specified value.
    for (const output of outputs) {
      expect(results[0][output].length).toBe(results[1][output].length);
      if (output === 'iterationCount') continue;
      const length =
        output === 'triangles' ? 3 * results[0]['count'][0] : results[0][output].length;
      expect(Array.from(results[0][output].subarray(0, length))).toEqual(
        Array.from(results[1][output].subarray(0, length))
      );
    }
    // CPU counts distinct sites: nine (the last row duplicates row 5), three collinear sites,
    // or one repeated site. GPU counts active insertion/flip rounds, not submitted gate nodes.
    expect(Array.from(results[1]['iterationCount'])).toEqual([
      scenario === 'duplicates' ? 1 : scenario === 'collinear' ? 3 : 9
    ]);
    const gpuIterationCount = results[0]['iterationCount'][0];
    if (scenario === 'duplicates') {
      // One site is inserted in the first round; no remaining site or illegal edge keeps it active.
      expect(gpuIterationCount).toBe(1);
    } else {
      // A noncollinear simplex needs the first three seed rounds. For the collinear input,
      // round 3 has no noncollinear seed, so the middle site cannot compete until round 4.
      expect(gpuIterationCount).toBeGreaterThanOrEqual(scenario === 'collinear' ? 4 : 3);
      expect(gpuIterationCount).toBeLessThanOrEqual(getDelaunayStepCount(positions.length / 2));
    }
    expect(results[0]['overflow'][0]).toBe(scenario === 'overflow' ? 1 : 0);
    expect(results[0]['count'][0]).toBe(
      scenario === 'collinear' || scenario === 'duplicates' ? 0 : scenario === 'overflow' ? 2 : 6
    );
  });
}

for (const capacity of [0, 1]) {
  it(`Delaunay named forms report one triangle at capacity ${capacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    // One noncollinear simplex: demand is one even when no payload fits.
    for (const Operation of [GPUDelaunayTessellationOperation, CPUDelaunayTessellationOperation]) {
      const result = await runOperationGraph(device, {
        inputs: {positions: {format: 'float32x2', values: Float32Array.of(0, 0, 2, 0, 0, 2)}},
        operations: [
          new Operation({
            positions: 'positions',
            capacity,
            output: {
              triangles: 'triangles',
              status: {count: 'count', requiredCount: 'requiredCount', overflow: 'overflow'}
            }
          })
        ],
        outputs: ['count', 'requiredCount', 'overflow']
      });
      expect(Array.from(result['requiredCount'])).toEqual([1]);
      expect(Array.from(result['count'])).toEqual([capacity]);
      expect(Array.from(result['overflow'])).toEqual([1 - capacity]);
    }
  });
}
