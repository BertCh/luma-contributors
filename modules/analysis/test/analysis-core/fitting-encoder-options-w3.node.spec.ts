// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {createTransientView, type GraphVectorView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUFittingCoordinateEncoder,
  type GPUFittingCoordinateEncoderProps
} from '../../src/analysis-core/kernels/gpu-fitting-coordinate-encoder';
import {createNullWebGPUDevice} from '../../src/testing/gpu-contributor-test-utils';

it('TYP-04 rejects hidden unknown options and accepts both declared branches', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const coordinates = createTransientView(graph, 'coordinates', 'float32', 2);
    const output = {
      features: createTransientView(graph, 'features', 'float32', 2),
      validity: createTransientView(graph, 'validity', 'uint32', 1)
    };
    const fourier = {
      encoding: 'fourier' as const,
      rowCount: 1,
      coordinates,
      dimensions: 2,
      frequencyCount: 1,
      frequencies: createTransientView(graph, 'frequencies', 'float32', 2),
      output
    };
    expect(() => new GPUFittingCoordinateEncoder(fourier)).not.toThrow();
    const hidden = Object.defineProperty({...fourier}, 'frequencyCout', {value: 1});
    expect(
      () => new GPUFittingCoordinateEncoder(hidden as GPUFittingCoordinateEncoderProps)
    ).toThrow(/fitting-coordinate-encoder.*frequencyCout/);
    // Type fixtures are checked by the lead's test typecheck, never executed.
    if (false) {
      // @ts-expect-error Reject misspelled fresh literal keys.
      new GPUFittingCoordinateEncoder({...fourier, frequencyCout: 1});
      new GPUFittingCoordinateEncoder({
        encoding: 'spherical-harmonics',
        rowCount: 1,
        coordinates,
        maximumDegree: 0,
        output
      });
      const chunked = new GPUFittingCoordinateEncoder({
        ...fourier,
        coordinates: null as unknown as GraphVectorView<'float32'>,
        output: {
          features: null as unknown as GraphVectorView<'float32'>,
          validity: null as unknown as GraphVectorView<'uint32'>
        }
      });
      chunked.props.coordinates.data;
    }
  } finally {
    device.destroy();
  }
});
