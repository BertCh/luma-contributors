// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPURasterHANTS} from '../../src/raster-analysis/raster-time-series/gpu-raster-hants';
import {GPURasterTemporalResample} from '../../src/raster-analysis/raster-time-series/gpu-raster-temporal-resample';
import {createNullWebGPUDevice, createVectorView} from '../utils/gpu-contributor-test-utils';

it('typechecks and lowers packed sources with one-chunk output vectors in both pilots', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const field = <Format extends 'float32' | 'uint32' | 'uint32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `mixed-${sequence++}`, format, length);
  try {
    const values = field('float32', 8);
    const parameters = field('float32', 3);
    const hants = new GPURasterHANTS({
      cellCount: 1,
      sliceCount: 8,
      harmonicCount: 1,
      values,
      parameters,
      output: {
        smoothed: createVectorView('smoothed', 'float32', [field('float32', 8)]),
        coefficients: createVectorView('coefficients', 'float32', [field('float32', 3)]),
        status: createVectorView('status', 'uint32', [field('uint32', 1)])
      }
    });
    const resample = new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 8,
      gridCount: 3,
      values,
      parameters,
      dateAxis: {
        acquisitionTimes: field('uint32x2', 8),
        temporalContext: {
          encoding: 'int64-words',
          unit: 'second',
          epoch: 'unix',
          axis: {kind: 'irregular'}
        }
      },
      output: {
        values: createVectorView('resampled', 'float32', [field('float32', 3)]),
        status: createVectorView('resampled-status', 'uint32', [field('uint32', 3)])
      }
    });
    expect(hants.getCommandNodes(graph)).toHaveLength(1);
    expect(resample.getCommandNodes(graph)).toHaveLength(1);
    const chunkedValues = createVectorView('chunked-source', 'float32', [field('float32', 8)]);
    const packedHants = new GPURasterHANTS({
      ...hants.props,
      values: chunkedValues,
      output: {
        smoothed: field('float32', 8),
        coefficients: field('float32', 3),
        status: field('uint32', 1)
      }
    });
    const packedResample = new GPURasterTemporalResample({
      ...resample.props,
      values: chunkedValues,
      output: {
        values: field('float32', 3),
        status: field('uint32', 3)
      }
    });
    expect(packedHants.getCommandNodes(graph)).toHaveLength(1);
    expect(packedResample.getCommandNodes(graph)).toHaveLength(1);
    expect(
      () =>
        new GPURasterHANTS({
          ...hants.props,
          cellCount: 2,
          values: field('float32', 16),
          output: {
            smoothed: createVectorView('split-fit', 'float32', [
              field('float32', 8),
              field('float32', 8)
            ]),
            coefficients: field('float32', 6),
            status: field('uint32', 2)
          }
        })
    ).toThrow(/cell chunk topology/);
    expect(
      () =>
        new GPURasterTemporalResample({
          ...resample.props,
          cellCount: 2,
          values: field('float32', 16),
          output: {
            values: createVectorView('split-resample', 'float32', [
              field('float32', 3),
              field('float32', 3)
            ]),
            status: field('uint32', 6)
          }
        })
    ).toThrow(/cell chunk topology/);
  } finally {
    device.destroy();
  }
});
