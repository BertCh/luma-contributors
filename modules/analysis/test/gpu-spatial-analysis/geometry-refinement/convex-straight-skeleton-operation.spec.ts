// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUConvexStraightSkeletonOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-convex-straight-skeleton-operation';
import {
  GPUConvexStraightSkeletonOperation,
  type ConvexStraightSkeletonOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-convex-straight-skeleton-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

for (const scene of ['ordinary', 'overflow', 'extinct', 'unsupported'] as const) {
  it(`convex CPU/GPU operations compare every output: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: ConvexStraightSkeletonOperationProps = {
      id: `convex-${scene}`,
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonCount: 2,
      maximumVertices: 5,
      maximumIterations: 8,
      segmentCapacity: scene === 'overflow' ? 2 : 10,
      offsetDistance: scene === 'extinct' ? 3 : 1.8,
      output: {
        segments: 'segments',
        sourceEdges: 'sourceEdges',
        offsetPositions: 'offsetPositions',
        status: 'status'
      }
    };
    // The clipped rectangle has two events: corner collapse, then rectangle ridge.
    // The other ring has reversed winding and explicit closure. Neither input is ignored.
    const positions =
      scene === 'unsupported'
        ? Float32Array.of(0, 0, 3, 0, 1, 1, 3, 3, 0, 3, 10, 0, 10, 4, 16, 4, 16, 0, 10, 0)
        : Float32Array.of(1, 0, 6, 0, 6, 4, 0, 4, 0, 1, 10, 0, 10, 4, 16, 4, 16, 0, 10, 0);
    const input = {
      positions: {format: 'float32x2' as const, values: positions},
      ringOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 5, 10)}
    };
    const outputs = Object.values(props.output);
    const gpu = await runOperationGraph(device, {
      inputs: input,
      operations: [new GPUConvexStraightSkeletonOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs: input,
      operations: [new CPUConvexStraightSkeletonOperation(props)],
      outputs
    });
    expect(Array.from(gpu['status'])).toEqual(Array.from(cpu['status']));
    expect(Array.from(gpu['sourceEdges'])).toEqual(Array.from(cpu['sourceEdges']));
    // Float64 vs shader float32: absolute tolerance 2e-5 planar units, including padded rows.
    for (const port of ['segments', 'offsetPositions']) {
      expect(gpu[port].length).toBe(cpu[port].length);
      cpu[port].forEach((value, index) =>
        expect(Math.abs(gpu[port][index] - value)).toBeLessThanOrEqual(2e-5)
      );
    }
    if (scene === 'ordinary') {
      expect(Array.from(cpu['status'])).toEqual([7, 7, 4, 0, 5, 5, 4, 0]);
    } else if (scene === 'overflow') {
      expect(Array.from(cpu['status'])).toEqual([2, 7, 4, 2, 2, 5, 4, 2]);
    } else if (scene === 'extinct') {
      expect(cpu['status'][2]).toBe(0);
      expect(cpu['status'][6]).toBe(0);
    } else {
      expect(Array.from(cpu['status'].subarray(0, 4))).toEqual([0, 0, 0, 1]);
    }
  });
}

for (const ringOffsets of [Uint32Array.of(3, 1), Uint32Array.of(3, 1, 4)]) {
  it(`compares unsupported decreasing ring offsets ${Array.from(ringOffsets).join(',')}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: ConvexStraightSkeletonOperationProps = {
      id: 'convex-decreasing-offsets',
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonCount: ringOffsets.length - 1,
      maximumVertices: 4,
      offsetDistance: 0,
      output: {
        segments: 'segments',
        sourceEdges: 'sourceEdges',
        offsetPositions: 'offsetPositions',
        status: 'status'
      }
    };
    const inputs = {
      positions: {format: 'float32x2' as const, values: Float32Array.of(0, 0, 4, 0, 4, 2, 0, 2)},
      ringOffsets: {format: 'uint32' as const, values: ringOffsets}
    };
    const outputs = Object.values(props.output);
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUConvexStraightSkeletonOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUConvexStraightSkeletonOperation(props)],
      outputs
    });
    const expectedStatus = ringOffsets.length === 2 ? [0, 0, 0, 1] : [0, 0, 0, 1, 3, 3, 3, 0];
    for (const result of [gpu, cpu]) expect(Array.from(result['status'])).toEqual(expectedStatus);
    expect(Array.from(gpu['sourceEdges'])).toEqual(Array.from(cpu['sourceEdges']));
    for (const port of ['segments', 'offsetPositions']) {
      expect(gpu[port].length).toBe(cpu[port].length);
      // Any subsequent valid triangle uses shader f32 arithmetic: 2e-5 planar-unit tolerance.
      cpu[port].forEach((value, index) =>
        expect(Math.abs(gpu[port][index] - value)).toBeLessThanOrEqual(2e-5)
      );
    }
  });
}
