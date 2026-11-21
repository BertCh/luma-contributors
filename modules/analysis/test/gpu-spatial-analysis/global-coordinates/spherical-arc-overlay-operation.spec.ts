// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {CPUSphericalArcOverlayOperation} from '../../../src/gpu-spatial-analysis/global-coordinates/cpu-spherical-arc-overlay-operation';
import {
  GPUSphericalArcOverlayOperation,
  type SphericalArcOverlayOperationProps
} from '../../../src/gpu-spatial-analysis/global-coordinates/gpu-spherical-arc-overlay-operation';
import {runOperationGraph, type OperationGraphInput} from '../../utils/mixed-execution-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import reference from './fixtures/spherical-arc-overlay.json';

function projectRings(rings: number[][][]): {positions: Float32Array; offsets: Uint32Array} {
  const offsets = [0];
  for (const ring of rings) offsets.push(offsets[offsets.length - 1] + ring.length);
  return {
    positions: Float32Array.from(
      rings.flat().flatMap(([horizontal, vertical]) => {
        const length = Math.hypot(horizontal, vertical, 1);
        return [horizontal / length, vertical / length, 1 / length];
      })
    ),
    offsets: Uint32Array.from(offsets)
  };
}
function rectangle(left: number, bottom: number, right: number, top: number): number[][] {
  return [
    [left, bottom],
    [right, bottom],
    [right, top],
    [left, top]
  ];
}
function getProps(): SphericalArcOverlayOperationProps {
  return {
    id: 'arc',
    operation: 'intersection',
    referencePoint: [0, 0, -1],
    capacity: 64,
    ringCapacity: 16,
    left: {
      positions: 'leftPositions',
      ringOffsets: 'leftOffsets',
      referenceInside: false
    },
    right: {
      positions: 'rightPositions',
      ringOffsets: 'rightOffsets',
      referenceInside: false
    },
    output: {positions: 'positions', ringOffsets: 'ringOffsets', counts: 'counts', status: 'status'}
  };
}
async function compareOperations(
  device: NonNullable<Awaited<ReturnType<typeof getWebGPUTestDevice>>>,
  props: SphericalArcOverlayOperationProps,
  inputs: Record<string, OperationGraphInput>,
  expectedStatus = 0
) {
  const results: Awaited<ReturnType<typeof runOperationGraph>>[] = [];
  for (const Operation of [GPUSphericalArcOverlayOperation, CPUSphericalArcOverlayOperation])
    results.push(
      await runOperationGraph(device, {
        inputs,
        operations: [new Operation(props)],
        outputs: Object.values(props.output)
      })
    );
  for (const port of ['counts', 'status', 'ringOffsets'])
    expect(Array.from(results[0][port]), port).toEqual(Array.from(results[1][port]));
  expect(Array.from(results[1]['status'])).toEqual([expectedStatus]);
  // Inputs are f32 and the GPU evaluates trigonometry in f32. Absolute 3e-6 is appropriate
  // for unit-vector components; zeros are compared numerically, never by their sign.
  results[1]['positions'].forEach((expected, index) => {
    const actual = results[0]['positions'][index];
    if (index >= results[1]['counts'][0] * 3) expect(actual).toBe(0);
    else expect(Math.abs(actual - expected), `positions[${index}]`).toBeLessThan(3e-6);
  });
  return results[1];
}
function getInputs(
  left: ReturnType<typeof projectRings>,
  right: ReturnType<typeof projectRings>
): Record<string, OperationGraphInput> {
  return {
    leftPositions: {format: 'float32x3', values: left.positions},
    leftOffsets: {format: 'uint32', values: left.offsets},
    rightPositions: {format: 'float32x3', values: right.positions},
    rightOffsets: {format: 'uint32', values: right.offsets}
  };
}

it('spherical twins agree on every output for noding, parity holes and coincident boundaries', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const scenes = [
    {
      left: projectRings([rectangle(-0.6, -0.4, 0.4, 0.5)]),
      right: projectRings([rectangle(-0.2, -0.6, 0.7, 0.2)])
    },
    {
      left: projectRings([rectangle(-0.8, -0.8, 0.8, 0.8), rectangle(-0.2, -0.2, 0.2, 0.2)]),
      right: projectRings([rectangle(0, -0.5, 0.6, 0.5)])
    },
    {
      left: projectRings([rectangle(-0.6, -0.4, 0.4, 0.5)]),
      right: projectRings([rectangle(-0.6, -0.4, 0.4, 0.5).reverse()])
    }
  ];
  for (const scene of scenes)
    for (const operation of [
      'intersection',
      'union',
      'difference',
      'symmetric-difference'
    ] as const)
      await compareOperations(
        device,
        {
          ...getProps(),
          operation
        },
        getInputs(scene.left, scene.right)
      );
});

it('spherical twins agree on non-hemisphere arrangements and complemented operands', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const inputs = getInputs(
    {positions: Float32Array.from(reference.left.flat()), offsets: Uint32Array.of(0, 4)},
    {positions: Float32Array.from(reference.right.flat()), offsets: Uint32Array.of(0, 4)}
  );
  for (const scene of reference.cases) {
    const props = getProps();
    const result = await compareOperations(
      device,
      {
        ...props,
        referencePoint: [0, 0, 1],
        operation: scene.operation as SphericalArcOverlayOperationProps['operation'],
        left: {...props.left, referenceInside: scene.leftInside},
        right: {...props.right, referenceInside: scene.rightInside}
      },
      inputs
    );
    expect(result['counts'][1]).toBe(scene.rings);
    expect(result['counts'][3]).toBe(Number(scene.referenceInside));
  }
});

it('spherical twins agree on vertex/ring overflow, invalid topology, unresolved pieces and empty results', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const left = projectRings([rectangle(-0.6, -0.4, 0.4, 0.5)]);
  const right = projectRings([rectangle(-0.2, -0.6, 0.7, 0.2)]);
  const inputs = getInputs(left, right);
  const vertexFailure = await compareOperations(
    device,
    {...getProps(), operation: 'union', capacity: 3},
    inputs,
    2
  );
  expect(Array.from(vertexFailure['counts'])).toEqual([0, 0, 8, 0]);
  const ringFailure = await compareOperations(
    device,
    {...getProps(), operation: 'symmetric-difference', ringCapacity: 1},
    inputs,
    2
  );
  expect(Array.from(ringFailure['counts'])).toEqual([0, 0, 12, 0]);
  await compareOperations(
    device,
    getProps(),
    {...inputs, leftOffsets: {format: 'uint32', values: Uint32Array.of(1, 4)}},
    1
  );
  const tiny = projectRings([
    [
      [0, 0],
      [3e-6, 0],
      [0, 0.5]
    ]
  ]);
  await compareOperations(device, getProps(), getInputs(tiny, right), 4);
  const disjoint = projectRings([rectangle(0.5, -0.6, 0.8, -0.2)]);
  const empty = await compareOperations(device, getProps(), getInputs(left, disjoint));
  expect(Array.from(empty['counts'])).toEqual([0, 0, 0, 0]);
});

it('derives four-vertex operand lengths from vectors despite stale count metadata', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const square = projectRings([rectangle(-0.6, -0.4, 0.4, 0.5)]);
  for (const vertexCount of [1, 4]) {
    const base = getProps();
    const props = {...base, left: {...base.left, vertexCount, ringOffsetCount: 1}};
    const result = await compareOperations(device, props, getInputs(square, square));
    expect(Array.from(result['counts'])).toEqual([4, 1, 4, 0]);
  }
});

it('coincident squares require four unique edges in both forms at capacities three and four', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const square = projectRings([rectangle(-0.6, -0.4, 0.4, 0.5)]);
  for (const capacity of [3, 4])
    for (const Operation of [GPUSphericalArcOverlayOperation, CPUSphericalArcOverlayOperation]) {
      const props = {...getProps(), operation: 'union' as const, capacity};
      const result = await runOperationGraph(device, {
        inputs: getInputs(square, square),
        operations: [new Operation(props)],
        outputs: Object.values(props.output)
      });
      // Set union has four unique oriented edges even when only three output slots exist.
      expect(Array.from(result['counts'])).toEqual(capacity === 3 ? [0, 0, 4, 0] : [4, 1, 4, 0]);
      expect(Array.from(result['status'])).toEqual([capacity === 3 ? 2 : 0]);
      if (capacity === 3) {
        expect(result['positions'].every(value => value === 0)).toBe(true);
        expect(result['ringOffsets'].every(value => value === 0)).toBe(true);
      }
    }
});
