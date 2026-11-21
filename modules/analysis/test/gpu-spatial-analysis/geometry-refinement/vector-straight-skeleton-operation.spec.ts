// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {PENTAGON_POSITIONS, expectPentagonSupports} from './pentagon-support-oracle';
import {expect, it} from 'vitest';
import {CPUVectorStraightSkeletonOperation} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-vector-straight-skeleton-operation';
import {
  GPUVectorStraightSkeletonOperation,
  type VectorStraightSkeletonOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-vector-straight-skeleton-operation';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  SQUARE_HOLE_INPUT,
  SQUARE_HOLE_SEGMENTS,
  SQUARE_HOLE_SOURCE_EDGES
} from './vector-square-hole-fixture';

it('GPU square-with-hole capture pins every live arc and final source-edge pairs', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = {
    ...createReviewProps(1),
    maximumVertices: 8,
    maximumEvents: 16,
    segmentCapacity: 32,
    offsetDistance: 0.25,
    eventTolerance: 1e-5
  };
  const result = await runOperationGraph(device, {
    inputs: {
      positions: {format: 'float32x2', values: SQUARE_HOLE_INPUT.positions},
      ringOffsets: {format: 'uint32', values: SQUARE_HOLE_INPUT.ringOffsets},
      polygonOffsets: {format: 'uint32', values: SQUARE_HOLE_INPUT.polygonOffsets},
      featureOffsets: {format: 'uint32', values: SQUARE_HOLE_INPUT.featureOffsets}
    },
    operations: [new GPUVectorStraightSkeletonOperation(props)],
    outputs: ['segments', 'sourceEdges', 'status']
  });
  expect(Array.from(result['status'])).toEqual([12, 12, 8, 0]);
  expect(Array.from(result['segments'].slice(0, 48))).toEqual(SQUARE_HOLE_SEGMENTS);
  expect(Array.from(result['sourceEdges'].slice(0, 24))).toEqual(SQUARE_HOLE_SOURCE_EDGES);
  // Read rows 8–11 explicitly before the geometry-guide owner refreshes the lens capture.
  expect(Array.from(result['sourceEdges'].slice(16, 24))).toEqual([7, 0, 1, 6, 2, 5, 3, 4]);
});

function createReviewProps(polygonCount: number): VectorStraightSkeletonOperationProps {
  return {
    id: 'vector-review',
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    polygonOffsets: 'polygonOffsets',
    featureOffsets: 'featureOffsets',
    polygonCount,
    maximumVertices: 32,
    maximumEvents: 0,
    segmentCapacity: 128,
    offsetDistance: 0,
    eventTolerance: 0.1,
    output: {
      segments: 'segments',
      sourceEdges: 'sourceEdges',
      offsetPositions: 'offsetPositions',
      offsetLinks: 'offsetLinks',
      status: 'status'
    }
  };
}

for (const scene of ['ordinary', 'overflow', 'extinct', 'unsupported', 'empty'] as const) {
  it(`vector skeleton CPU/GPU compare every output: ${scene}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const props: VectorStraightSkeletonOperationProps = {
      id: `vector-${scene}`,
      positions: 'positions',
      ringOffsets: 'ringOffsets',
      polygonOffsets: 'polygonOffsets',
      featureOffsets: 'featureOffsets',
      polygonCount: 2,
      maximumVertices: scene === 'unsupported' ? 7 : 8,
      maximumEvents: 32,
      segmentCapacity: scene === 'overflow' ? 2 : 32,
      offsetDistance: scene === 'extinct' ? 6 : 1.25,
      output: {
        segments: 'segments',
        sourceEdges: 'sourceEdges',
        offsetPositions: 'offsetPositions',
        offsetLinks: 'offsetLinks',
        status: 'status'
      }
    };
    // An asymmetric hole first merges at t=1. The second polygon is a concave C whose
    // simultaneous ridges finish at t=1. Axis-aligned planes keep discrete event ties exact.
    const inputs = {
      positions: {
        format: 'float32x2' as const,
        values: Float32Array.of(
          0,
          0,
          12,
          0,
          12,
          10,
          0,
          10,
          3,
          2,
          3,
          5,
          6,
          5,
          6,
          2,
          20,
          0,
          28,
          0,
          28,
          2,
          22,
          2,
          22,
          6,
          28,
          6,
          28,
          8,
          20,
          8
        )
      },
      ringOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 4, 8, 16)},
      polygonOffsets: {
        format: 'uint32' as const,
        values: scene === 'empty' ? Uint32Array.of(0, 0, 0) : Uint32Array.of(0, 2, 3)
      },
      featureOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 2)}
    };
    // Two declared empty polygons own no rings or rows: compact terminal coverage (UGE-10)
    // rejects uncovered children, so the empty scene declares none.
    if (scene === 'empty') {
      inputs.positions.values = new Float32Array();
      inputs.ringOffsets.values = Uint32Array.of(0);
    }
    const outputs = Object.values(props.output);
    const gpu = await runOperationGraph(device, {
      inputs,
      operations: [new GPUVectorStraightSkeletonOperation(props)],
      outputs
    });
    const cpu = await runOperationGraph(device, {
      inputs,
      operations: [new CPUVectorStraightSkeletonOperation(props)],
      outputs
    });
    for (const port of ['status', 'sourceEdges', 'offsetLinks']) {
      expect(gpu[port].length).toBe(cpu[port].length);
      // Numeric equality deliberately treats the two signs of zero alike.
      cpu[port].forEach((value, index) => expect(gpu[port][index] === value).toBe(true));
    }
    for (const port of ['segments', 'offsetPositions']) {
      expect(gpu[port].length).toBe(cpu[port].length);
      // Absolute 3e-5 planar-unit tolerance, including unused rows on fresh zeroed buffers.
      cpu[port].forEach((value, index) =>
        expect(Math.abs(gpu[port][index] - value)).toBeLessThanOrEqual(3e-5)
      );
    }
    const expectedStatus =
      scene === 'ordinary'
        ? [16, 16, 8, 0, 11, 11, 0, 0]
        : scene === 'overflow'
          ? [2, 16, 8, 2, 2, 11, 0, 2]
          : scene === 'extinct'
            ? [16, 16, 0, 0, 11, 11, 0, 0]
            : scene === 'unsupported'
              ? [0, 0, 0, 1, 0, 0, 0, 1]
              : [0, 0, 0, 0, 0, 0, 0, 0];
    expect(Array.from(cpu['status'])).toEqual(expectedStatus);
  });
}

it('both execution forms honor zero event work after capturing the initial snapshot', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = createReviewProps(1);
  const inputs = {
    positions: {
      format: 'float32x2' as const,
      values: Float32Array.of(5, 3.5, 1.5, 2.5, 1, 8, -6.5, -3.7, -1.6, -2.2, -1, -5, 3, -2)
    },
    ringOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 7)},
    polygonOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 1)},
    featureOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 1)}
  };
  const outputs = Object.values(props.output);
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUVectorStraightSkeletonOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUVectorStraightSkeletonOperation(props)],
    outputs
  });
  // Capture precedes the GPU budget check; emission follows it. This explicitly tests the
  // reviewer's device hypothesis, rather than assuming interpreted WGSL proves device behaviour.
  expect(Array.from(gpu['status'])).toEqual([0, 0, 7, 4]);
  expect(gpu['segments'].every(value => value === 0)).toBe(true);
  expect(gpu['sourceEdges'].every(value => value === 0)).toBe(true);
  expect(Array.from(cpu['status'])).toEqual([0, 0, 7, 4]);
  expect(cpu['status'][0]).toBe(cpu['status'][1]);
  expect(cpu['status'][0]).toBe(0);
  expect(cpu['status'][0]).toBeLessThan(128);
  expect(cpu['segments'].every(Number.isFinite)).toBe(true);
  expect(cpu['segments'].length).toBe(gpu['segments'].length);
  expect(cpu['sourceEdges'].length).toBe(gpu['sourceEdges'].length);
  expect(Array.from(cpu['offsetLinks'])).toEqual(Array.from(gpu['offsetLinks']));
  expect(cpu['offsetPositions'].length).toBe(gpu['offsetPositions'].length);
  cpu['offsetPositions'].forEach((value, index) =>
    expect(Math.abs(gpu['offsetPositions'][index] - value)).toBeLessThanOrEqual(3e-5)
  );
});

it('compares all five zero-length outputs for a zero-polygon batch', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const props = createReviewProps(0);
  const inputs = {
    positions: {format: 'float32x2' as const, values: new Float32Array()},
    ringOffsets: {format: 'uint32' as const, values: Uint32Array.of(0)},
    polygonOffsets: {format: 'uint32' as const, values: Uint32Array.of(0)},
    featureOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 0)}
  };
  const outputs = Object.values(props.output);
  const gpu = await runOperationGraph(device, {
    inputs,
    operations: [new GPUVectorStraightSkeletonOperation(props)],
    outputs
  });
  const cpu = await runOperationGraph(device, {
    inputs,
    operations: [new CPUVectorStraightSkeletonOperation(props)],
    outputs
  });
  for (const output of outputs) {
    expect(gpu[output]).toHaveLength(0);
    expect(cpu[output]).toHaveLength(0);
    expect(Array.from(gpu[output])).toEqual(Array.from(cpu[output]));
  }
});

it('terminal oblique pentagon arcs retain both independent supporting edges on every form', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs = {
    positions: {format: 'float32x2' as const, values: Float32Array.from(PENTAGON_POSITIONS)},
    ringOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 5)},
    polygonOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 1)},
    featureOffsets: {format: 'uint32' as const, values: Uint32Array.of(0, 1)}
  };
  const props = {
    ...createReviewProps(1),
    maximumVertices: 32,
    maximumEvents: 64,
    segmentCapacity: 128,
    offsetDistance: 0.25,
    eventTolerance: 1e-5
  };
  for (const operation of [
    new GPUVectorStraightSkeletonOperation(props),
    new CPUVectorStraightSkeletonOperation(props)
  ]) {
    const rows = await runOperationGraph(device, {
      inputs,
      operations: [operation],
      outputs: ['segments', 'sourceEdges', 'status']
    });
    expectPentagonSupports({
      segments: rows['segments'],
      sourceEdges: rows['sourceEdges'],
      status: rows['status']
    });
  }
});
