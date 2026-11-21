// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../../src/testing/gpu-contributor-test-utils';
import {
  GPURasterTemporalResample,
  getGPURasterTemporalResampleParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-temporal-resample';
import {
  GPURasterSnapshotCovariance,
  getGPURasterSnapshotCovarianceParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-snapshot-covariance';
import {GPURasterEOFProjection} from '../../../src/raster-analysis/raster-time-series/gpu-raster-eof-projection';
import {getGPURasterFixedPointValues} from '../../../src/raster-analysis/raster-time-series/gpu-raster-streaming-accumulator';
import {executeKernel} from './streaming-moments-w5.test-utils';

// Model only the floating reassociations admitted by Metal; leave integer/bit operations intact.
// This is a host regression for the optimization hazard, not a substitute for GPU qualification.
function reassociateCentering(source: string): string {
  return source.replace(
    /\(values\[([^\]]+)\] - anchor\) - meanOffset/g,
    'values[$1] - f32(anchor + meanOffset)'
  );
}

it('t1 RT-130 midpoint survives a fused year-conversion subtraction', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const floats = (name: string, length: number) =>
      createTransientView(graph, name, 'float32', length);
    const contributor = new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 3,
      gridCount: 1,
      values: floats('values', 3),
      parameters: floats('parameters', 3),
      dateAxis: {
        acquisitionTimes: createTransientView(graph, 'dates', 'uint32x2', 3),
        temporalContext: {
          encoding: 'int64-words',
          unit: 'second',
          epoch: 0n,
          axis: {kind: 'irregular'}
        }
      },
      output: {
        values: floats('resampled', 1),
        status: createTransientView(graph, 'codes', 'uint32', 1)
      }
    });
    const kernel = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))[0];
    let source = kernel.source.replace(
      /rasterDispatch\.words\[(\d+)u\]\[(\d+)u\]/g,
      (_match, row: string, column: string) => (row === '0' && column === '0' ? '1u' : '0u')
    );
    source = source.replace(
      /fn rasterSliceYears\(slice: u32\) -> f32 \{[^}]*\}/,
      'fn rasterSliceYears(slice: u32) -> f32 { if (slice == 0u) { return -10.000000953674316; } if (slice == 1u) { return -9.99999713897705; } return 10.000000953674316; }'
    );
    source = source.replace(
      /fn rasterYearsBetween\(later: u32, earlier: u32\) -> f32 \{[^}]*\}/,
      'fn rasterYearsBetween(later: u32, earlier: u32) -> f32 { return 0.0000020597680321365086; }'
    );
    // Python struct f32 replay: r=f32(1/31556952), left=f32(f32(-315569520)*r),
    // rightTicks=f32(-315569455); f32(rightTicks*r-left)=3.4138761293434072e-6.
    // A fused multiply-subtract can use this denominator instead of the rounded-coordinate span.
    source = source.replace(
      'let localSpan = rasterSliceYears(right) - rasterSliceYears(left);',
      'let localSpan = 0.0000034138761293434072;'
    );
    const fields = {
      values: Float32Array.of(0, 100, 200),
      parameters: getGPURasterTemporalResampleParameterValues({
        gridStart: -9.999999046325684,
        gridStep: 1
      }),
      acquisitionTimes: new Uint32Array(6),
      resampled: new Float32Array(1),
      resampleStatus: new Uint32Array(1)
    };
    executeKernel(source, kernel.constants, fields);
    // The target is the exact midpoint of represented coordinates, so the fraction is .5.
    expect([...fields.resampleStatus]).toEqual([0]);
    expect([...fields.resampled]).toEqual([50]);
  } finally {
    device.destroy();
  }
});

it('t1 FLT-06 Gram entries survive reassociation of anchor and mean displacement', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const contributor = new GPURasterSnapshotCovariance({
      cellCount: 1,
      sliceCount: 2,
      values: createTransientView(graph, 'values', 'float32', 2),
      parameters: createTransientView(graph, 'parameters', 'float32', 2),
      output: {
        covarianceWords: createTransientView(graph, 'gram', 'uint32', 9),
        keptCellCount: createTransientView(graph, 'count', 'uint32', 1)
      }
    });
    const kernel = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph)).find(entry =>
      entry.id.endsWith('-accumulate')
    )!;
    const parameters = getGPURasterSnapshotCovarianceParameterValues(
      {maximumValueMagnitude: 10000001},
      1
    );
    const fields = {
      values: Float32Array.of(10000000, 10000001),
      parameters,
      cellMeans: Float32Array.of(0.5),
      cellAnchors: Float32Array.of(10000000),
      cellKept: Uint32Array.of(1),
      covarianceWords: new Uint32Array(9)
    };
    executeKernel(reassociateCentering(kernel.source), kernel.constants, fields);
    // X=[-.5,.5], upper triangle of X'X is [.25,-.25,.25].
    expect([...getGPURasterFixedPointValues(fields.covarianceWords, parameters[0])]).toEqual([
      0.25, -0.25, 0.25
    ]);
  } finally {
    device.destroy();
  }
});

it('t1 EOF projection shares reassociation-safe centering', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const floats = (name: string, length: number) =>
      createTransientView(graph, name, 'float32', length);
    const contributor = new GPURasterEOFProjection({
      cellCount: 1,
      sliceCount: 2,
      modeCount: 1,
      values: floats('values', 2),
      parameters: floats('parameters', 2),
      temporalModes: floats('modes', 2),
      singularValues: floats('singular', 1),
      output: {eofs: floats('eofs', 1), status: createTransientView(graph, 'status', 'uint32', 1)}
    });
    const kernel = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph))[0];
    const source = reassociateCentering(kernel.source).replace(
      /rasterDispatch\.words\[(\d+)u\]\[(\d+)u\]/g,
      (_match, row: string, column: string) => (row === '0' && column === '0' ? '1u' : '0u')
    );
    const fields = {
      values: Float32Array.of(10000000, 10000001),
      parameters: getGPURasterSnapshotCovarianceParameterValues(
        {maximumValueMagnitude: 10000001},
        1
      ),
      temporalModes: Float32Array.of(Math.SQRT1_2, -Math.SQRT1_2),
      singularValues: Float32Array.of(Math.SQRT1_2),
      eofs: new Float32Array(1),
      status: new Uint32Array(1)
    };
    executeKernel(source, kernel.constants, fields);
    // [-.5,.5] dot [1/sqrt(2),-1/sqrt(2)] / (1/sqrt(2)) = -1.
    expect([...fields.status]).toEqual([0]);
    expect(Number.isFinite(fields.eofs[0])).toBe(true);
    expect(fields.eofs[0]).toBeCloseTo(-1, 6);
  } finally {
    device.destroy();
  }
});
