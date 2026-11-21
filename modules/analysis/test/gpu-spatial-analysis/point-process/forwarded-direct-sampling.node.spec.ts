// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUContinuousSpatialSample} from '../../../src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample';
import {GPUContinuousSpatialSampleOperation} from '../../../src/gpu-spatial-analysis/point-process/gpu-continuous-spatial-sample-operation';
import {computeContinuousSpatialSampleOnCPU} from '../../../src/gpu-spatial-analysis/point-process/cpu-continuous-spatial-sample-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function guardLineBounds<Frame extends {type: 'line'}>(frame: Frame): Frame {
  return new Proxy(frame, {
    get(target, key, receiver) {
      if (key === 'bounds') throw new Error('line bounds must not be read');
      return Reflect.get(target, key, receiver);
    }
  });
}

it('N1 direct GPU sampling captures a changing frame discriminator once', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    let reads = 0;
    const positions = createTransientView(graph, 'frame', 'float32x2', 2);
    const sample = new GPUContinuousSpatialSample({
      frame: guardLineBounds({
        get type(): 'line' {
          return (++reads === 1 ? 'line' : 'area') as 'line';
        },
        positions
      }),
      spacing: 1,
      seed: 7,
      capacity: 8,
      output: {
        positions: createTransientView(graph, 'sample', 'float32x2', 8),
        designWeights: createTransientView(graph, 'weights', 'float32', 8),
        provenance: createTransientView(graph, 'provenance', 'uint32x4', 8),
        status: createTransientView(graph, 'status', 'uint32x4', 1)
      }
    });
    expect(reads).toBe(1);
    expect(sample.props.frame.type).toBe('line');
    expect(sample.props.frame.positions).toBe(positions);
  } finally {
    device.destroy();
  }
});

it('N1 pure CPU sampling captures one line variant and ignores unused bounds', () => {
  let reads = 0;
  const result = computeContinuousSpatialSampleOnCPU(
    {positions: Float32Array.of(0, 0, 2, 0)},
    {
      frame: guardLineBounds({
        get type(): 'line' {
          return (++reads === 1 ? 'line' : 'area') as 'line';
        }
      }),
      spacing: 1,
      seed: 7,
      capacity: 8
    }
  );
  expect(reads).toBe(1);
  // A randomly shifted unit lattice on the length-two segment has exactly two sites.
  expect(Array.from(result.status)).toEqual([2, 2, 2, 0]);
  expect(Array.from(result.designWeights.subarray(0, 2))).toEqual([1, 1]);
});

it('rejects bounds on line frames in the direct and named sampling forms', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const commonOptions = {spacing: 1, seed: 7, capacity: 8};
  const message =
    /^continuous-spatial-sample frame.bounds is only supported for area frames; omit bounds for a line frame$/;
  try {
    expect(
      () =>
        new GPUContinuousSpatialSample({
          ...commonOptions,
          frame: {
            type: 'line',
            positions: createTransientView(graph, 'frame', 'float32x2', 2),
            // @ts-expect-error Line frames do not accept area bounds.
            get bounds(): never {
              throw new Error('line bounds must not be read');
            }
          },
          output: {
            positions: createTransientView(graph, 'sample', 'float32x2', 8),
            designWeights: createTransientView(graph, 'weights', 'float32', 8),
            provenance: createTransientView(graph, 'provenance', 'uint32x4', 8),
            status: createTransientView(graph, 'status', 'uint32x4', 1)
          }
        })
    ).toThrow(message);
    expect(() =>
      computeContinuousSpatialSampleOnCPU(
        {positions: Float32Array.of(0, 0, 2, 0)},
        {
          ...commonOptions,
          frame: {
            type: 'line',
            // @ts-expect-error Line frames do not accept area bounds.
            bounds: [0, 0, 2, 1]
          }
        }
      )
    ).toThrow(message);
    expect(
      () =>
        new GPUContinuousSpatialSampleOperation({
          ...commonOptions,
          frameRowCount: 2,
          frame: {
            type: 'line',
            positions: 'positions',
            // @ts-expect-error Line frames do not accept area bounds.
            bounds: [0, 0, 2, 1]
          },
          output: {
            positions: 'sample',
            designWeights: 'weights',
            provenance: 'provenance',
            status: 'status'
          }
        })
    ).toThrow(message);
  } finally {
    device.destroy();
  }
});
