// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {WgslReflect} from 'wgsl_reflect';
import {executeKernel} from './streaming-moments-w5.test-utils';
import {
  GPURasterTemporalResample,
  getGPURasterTemporalResampleParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-temporal-resample';
import {
  GPURasterSnapshotCovariance,
  getGPURasterSnapshotCovarianceParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-snapshot-covariance';
import {GPURasterMedoidComposite} from '../../../src/raster-analysis/raster-time-series/gpu-raster-medoid-composite';
import {
  GPURasterGeometricMedianComposite,
  getGPURasterGeometricMedianCompositeParameterValues
} from '../../../src/raster-analysis/raster-time-series/gpu-raster-geometric-median-composite';
import {getGPURasterCentredFractionalYears} from '../../../src/raster-analysis/raster-time-series/raster-irregular-date-axis';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';

/** Execute generated per-cell WGSL, including float32 arithmetic, without claiming device qualification. */
function execute(
  make: (graph: GPUCommandGraph) => GPUCommandNodeProducer,
  buffers: Record<string, Float32Array | Uint32Array>,
  kernelIndex = 0,
  transformSource: (source: string) => string = source => source
) {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const kernels = getGPUCommandNodeKernelSources(make(graph).getCommandNodes(graph));
    const originalKernel = kernels[kernelIndex];
    // Interpreter cannot index the dispatch uniform's nested vectors; specialize this one-cell,
    // zero-offset fixture's immutable dispatch words without changing the numerical body.
    let kernel = {
      ...originalKernel,
      source: originalKernel.source.replace(
        /rasterDispatch\.words\[(\d+)u\]\[(\d+)u\]/g,
        (_match, row, column) => (row === '0' && column === '0' ? '1u' : '0u')
      )
    };
    if (buffers['acquisitionTimes']) {
      // The interpreter does not reproduce unsigned word subtraction reliably. Specialize only
      // the time-conversion helpers using the public host equivalent; keep interpolation intact.
      const times = new BigInt64Array(buffers['acquisitionTimes'].buffer).subarray(0, 3);
      const years = getGPURasterCentredFractionalYears(times, 'second');
      const yearsBody = Array.from(
        years,
        (value, index) => `if (slice == ${index}u) { return ${value}; }`
      ).join(' ');
      kernel = {
        ...kernel,
        source: kernel.source
          .replace(
            /fn rasterSliceYears\(slice: u32\) -> f32 \{[^}]*\}/,
            `fn rasterSliceYears(slice: u32) -> f32 { ${yearsBody} return 0.0; }`
          )
          .replace(
            /fn rasterYearsBetween\(later: u32, earlier: u32\) -> f32 \{[^}]*\}/,
            `fn rasterYearsBetween(later: u32, earlier: u32) -> f32 { return ${Number(times[1] - times[0]) / 31556952}; }`
          )
      };
    }
    kernel = {...kernel, source: transformSource(kernel.source)};
    const reflection = new WgslReflect(kernel.source);
    const interpreterFields: Record<string, Float32Array | Uint32Array> = Object.fromEntries(
      reflection.storage.map(declaration => [
        declaration.name,
        buffers[declaration.name] ?? new Uint32Array(4)
      ])
    );
    // Use the shared instance-local fixes for ldexp, uint32 vector storage and unsigned shifts;
    // temporal centering and rounded year differences now both execute integer word arithmetic.
    executeKernel(kernel.source, kernel.constants, interpreterFields);
    return buffers;
  } finally {
    device.destroy();
  }
}
const view = <Format extends 'float32' | 'uint32' | 'uint32x2'>(
  graph: GPUCommandGraph,
  name: string,
  format: Format,
  length: number
) => createTransientView(graph, name, format, length);
// Extra words accommodate speculative RHS evaluation of && in the interpreter.
const timeWords = (times: bigint[]) => {
  const words = new Uint32Array(128);
  words.set(new Uint32Array(new BigInt64Array(times).buffer));
  return words;
};

it('FLT-13 fractional interpolation uses the same rounded coordinates in numerator and denominator', () => {
  // Rounded axis coordinates bracket -9.999999046325684 halfway between the first two dates.
  const result = execute(
    graph =>
      new GPURasterTemporalResample({
        cellCount: 1,
        sliceCount: 3,
        gridCount: 1,
        values: view(graph, 'values', 'float32', 3),
        parameters: view(graph, 'parameters', 'float32', 3),
        dateAxis: {
          acquisitionTimes: view(graph, 'acquisitionTimes', 'uint32x2', 3),
          temporalContext: {
            encoding: 'int64-words',
            unit: 'second',
            epoch: 0n,
            axis: {kind: 'irregular'}
          }
        },
        output: {
          values: view(graph, 'resampled', 'float32', 1),
          status: view(graph, 'resampleStatus', 'uint32', 1)
        }
      }),
    {
      values: Float32Array.of(0, 100, 200),
      parameters: getGPURasterTemporalResampleParameterValues({
        gridStart: -9.999999046325684,
        gridStep: 1
      }),
      acquisitionTimes: timeWords([0n, 65n, 631139040n]),
      resampled: new Float32Array(1),
      resampleStatus: new Uint32Array(1)
    }
  );
  expect(result['resampled'][0]).toBe(50);
  expect(result['resampleStatus'][0]).toBe(0);
});

it('FLT-06 stores the mean displacement rather than a rounded absolute mean', () => {
  // Mean offset from 10,000,000 is exactly .5; anomalies are [-.5,.5], Gram [.25,-.25;-.25,.25].
  const result = execute(
    graph =>
      new GPURasterSnapshotCovariance({
        cellCount: 1,
        sliceCount: 2,
        values: view(graph, 'values', 'float32', 2),
        parameters: view(graph, 'parameters', 'float32', 2),
        output: {
          covarianceWords: view(graph, 'covarianceWords', 'uint32', 9),
          keptCellCount: view(graph, 'keptCellCount', 'uint32', 1)
        }
      }),
    {
      values: Float32Array.of(10000000, 10000001),
      parameters: getGPURasterSnapshotCovarianceParameterValues(
        {maximumValueMagnitude: 10000001},
        1
      ),
      cellMeans: new Float32Array(1),
      cellAnchors: new Float32Array(1),
      cellKept: new Uint32Array(1)
    }
  );
  expect([...result['cellMeans']]).toEqual([0.5]);
  expect([...result['cellAnchors']]).toEqual([10000000]);
  expect(result['cellKept'][0]).toBe(1);
});

for (const scale of [1, 1e-25, 1e20]) {
  it(`URT-14 multiband medoid preserves the unique middle acquisition at scale ${scale}`, () => {
    // Three collinear spectra have distance totals [3s,2s,3s]; middle slice is uniquely optimal.
    const result = execute(
      graph =>
        new GPURasterMedoidComposite({
          cellCount: 1,
          sliceCount: 3,
          bandCount: 2,
          values: view(graph, 'values', 'float32', 6),
          output: {
            composite: view(graph, 'composite', 'float32', 2),
            sliceIndex: view(graph, 'sliceIndex', 'uint32', 1),
            status: view(graph, 'status', 'uint32', 1)
          }
        }),
      {
        values: Float32Array.of(0, 0, scale, 0, 2 * scale, 0),
        composite: new Float32Array(2),
        sliceIndex: new Uint32Array(1),
        status: new Uint32Array(1)
      }
    );
    expect(result['sliceIndex'][0]).toBe(1);
    expect(result['status'][0]).toBe(0);
    expect(result['composite'][0]).toBe(Math.fround(scale));
  });
}
for (const values of [
  [0.1, 0.2, 0.3, 0.4],
  [0.1, 0.3, 0.2, 0.4]
]) {
  it(`TWN-41 scalar medoid selects earliest middle observation ${values}`, () => {
    const result = execute(
      graph =>
        new GPURasterMedoidComposite({
          cellCount: 1,
          sliceCount: 4,
          bandCount: 1,
          values: view(graph, 'values', 'float32', 4),
          output: {
            composite: view(graph, 'composite', 'float32', 1),
            sliceIndex: view(graph, 'sliceIndex', 'uint32', 1),
            status: view(graph, 'status', 'uint32', 1)
          }
        }),
      {
        values: Float32Array.from(values),
        composite: new Float32Array(1),
        sliceIndex: new Uint32Array(1),
        status: new Uint32Array(1)
      }
    );
    expect(result['sliceIndex'][0]).toBe(1);
    expect(result['composite'][0]).toBe(Math.fround(values[1]));
    expect(result['status'][0]).toBe(0);
  });
}
for (const values of [[456787.5], [456787.5, 456787.5, 456787.5]]) {
  it(`TWN-41 geometric median preserves identical observations (${values.length})`, () => {
    const result = execute(
      graph =>
        new GPURasterGeometricMedianComposite({
          cellCount: 1,
          sliceCount: values.length,
          bandCount: 1,
          iterationCount: 1,
          values: view(graph, 'values', 'float32', values.length),
          parameters: view(graph, 'parameters', 'float32', 1),
          output: {
            composite: view(graph, 'composite', 'float32', 1),
            lastStep: view(graph, 'lastStep', 'float32', 1),
            status: view(graph, 'status', 'uint32', 1)
          }
        }),
      {
        values: Float32Array.from(values),
        parameters: getGPURasterGeometricMedianCompositeParameterValues(),
        composite: new Float32Array(1),
        lastStep: new Float32Array(1),
        status: new Uint32Array(1)
      }
    );
    expect(result['composite'][0]).toBe(456787.5);
    expect(result['lastStep'][0]).toBe(0);
    expect(result['status'][0]).toBe(0);
  });
}

it('review B1 audits the exact-word resampler variant for reserved WGSL identifiers', async () => {
  const {findReservedIdentifiers, stripWGSLComments} = await import(
    '../../analysis-core/kernel-audit/kernel-audit-wgsl'
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const nodes = new GPURasterTemporalResample({
      cellCount: 1,
      sliceCount: 2,
      gridCount: 1,
      values: view(graph, 'values', 'float32', 2),
      parameters: view(graph, 'parameters', 'float32', 3),
      targetTimes: view(graph, 'targetTimes', 'uint32x2', 1),
      dateAxis: {
        acquisitionTimes: view(graph, 'times', 'uint32x2', 2),
        temporalContext: {
          encoding: 'int64-words',
          unit: 'millisecond',
          epoch: 0n,
          axis: {kind: 'irregular'}
        }
      },
      output: {
        values: view(graph, 'result', 'float32', 1),
        status: view(graph, 'code', 'uint32', 1)
      }
    }).getCommandNodes(graph);
    for (const kernel of getGPUCommandNodeKernelSources(nodes))
      expect(findReservedIdentifiers(stripWGSLComments(kernel.source))).toEqual([]);
  } finally {
    device.destroy();
  }
});

for (const values of [
  [0.1, 0.875, 0.2, 0.875, 0.3, 0.875, 0.4, 0.875],
  [1e24, 0, 1e24, 1, 1e24, 2]
]) {
  it(`review B5 S1 constant bands preserve acquisition provenance ${values}`, () => {
    const result = execute(
      graph =>
        new GPURasterMedoidComposite({
          cellCount: 1,
          sliceCount: values.length / 2,
          bandCount: 2,
          values: view(graph, 'values', 'float32', values.length),
          output: {
            composite: view(graph, 'composite', 'float32', 2),
            sliceIndex: view(graph, 'sliceIndex', 'uint32', 1),
            status: view(graph, 'status', 'uint32', 1)
          }
        }),
      {
        values: Float32Array.from(values),
        composite: new Float32Array(2),
        sliceIndex: new Uint32Array(1),
        status: new Uint32Array(1)
      }
    );
    // Constant coordinates add exactly zero to every distance. The earliest scalar median is row 1.
    expect(Array.from(result['sliceIndex'])).toEqual([1]);
    expect(Array.from(result['status'])).toEqual([0]);
    expect(Array.from(result['composite'])).toEqual(
      Array.from(Float32Array.from(values.slice(2, 4)))
    );
  });
}

it('review B3 preserves the signed-field third anomaly in projection', async () => {
  const {GPURasterEOFProjection} = await import(
    '../../../src/raster-analysis/raster-time-series/gpu-raster-eof-projection'
  );
  const result = execute(
    graph =>
      new GPURasterEOFProjection({
        cellCount: 1,
        sliceCount: 3,
        modeCount: 1,
        values: view(graph, 'values', 'float32', 3),
        parameters: view(graph, 'parameters', 'float32', 2),
        temporalModes: view(graph, 'temporalModes', 'float32', 3),
        singularValues: view(graph, 'singularValues', 'float32', 1),
        output: {
          eofs: view(graph, 'eofs', 'float32', 1),
          status: view(graph, 'status', 'uint32', 1)
        }
      }),
    {
      values: Float32Array.of(32768, -32768, 1 / 1024),
      parameters: getGPURasterSnapshotCovarianceParameterValues({maximumValueMagnitude: 32768}, 1),
      temporalModes: Float32Array.of(0, 0, 1),
      singularValues: Float32Array.of(1),
      eofs: new Float32Array(1),
      status: new Uint32Array(1)
    }
  );
  // Exact mean is 1/3072; projection onto unit e_3 is (1/1024 - 1/3072)=1/1536.
  expect(Number.isFinite(result['eofs'][0])).toBe(true);
  expect(Math.abs(result['eofs'][0] - 1 / 1536)).toBeLessThan(1e-10);
  expect(Array.from(result['status'])).toEqual([0]);
});

it('review earlier-date note uses a pinned reciprocal for the 419-day endpoints', () => {
  // f32(f32(419 days / 2) * f32(1/f32(seconds-per-year * 1000))).
  const years = getGPURasterCentredFractionalYears(
    BigInt64Array.of(0n, 36201600000n),
    'millisecond'
  );
  expect(Array.from(years)).toEqual([-0.5735914707183838, 0.5735914707183838]);
});

it('review B7 projection discriminator rejects the earlier absolute-mean arithmetic', async () => {
  const {GPURasterEOFProjection} = await import(
    '../../../src/raster-analysis/raster-time-series/gpu-raster-eof-projection'
  );
  const project = (mutate: boolean) =>
    execute(
      graph =>
        new GPURasterEOFProjection({
          cellCount: 1,
          sliceCount: 6,
          modeCount: 1,
          values: view(graph, 'values', 'float32', 6),
          parameters: view(graph, 'parameters', 'float32', 2),
          temporalModes: view(graph, 'temporalModes', 'float32', 6),
          singularValues: view(graph, 'singularValues', 'float32', 1),
          output: {
            eofs: view(graph, 'eofs', 'float32', 1),
            status: view(graph, 'status', 'uint32', 1)
          }
        }),
      {
        values: new Float32Array(6).fill(10000002),
        parameters: getGPURasterSnapshotCovarianceParameterValues(
          {maximumValueMagnitude: 10000002},
          1
        ),
        temporalModes: Float32Array.of(
          1 / Math.sqrt(14),
          2 / Math.sqrt(14),
          -3 / Math.sqrt(14),
          0,
          0,
          0
        ),
        singularValues: Float32Array.of(Math.sqrt(14) / 1024),
        eofs: new Float32Array(1),
        status: new Uint32Array(1)
      },
      0,
      source =>
        mutate
          ? source
              .replace(
                'for (var mode =',
                'var absoluteTotal = 0.0; for (var acquisition = 0u; acquisition < 6u; acquisition++) { absoluteTotal += values[valuesOffset + index * 6u + acquisition]; } let absoluteMean = absoluteTotal / 6.0; for (var mode ='
              )
              .replace(
                /getRasterTemporalCenteredValue\(values\[([^\]]+)\], anchor, meanOffset, meanUnitExponent\)/g,
                '(values[$1] - absoluteMean)'
              )
          : source
    );
  // A constant cell has exactly zero anomalies, regardless of the orthonormal temporal mode.
  expect(Array.from(project(false)['eofs'])).toEqual([0]);
  const earlier = project(true)['eofs'][0];
  expect(Number.isFinite(earlier)).toBe(true);
  expect(Math.abs(earlier)).toBeGreaterThan(1e-6);
});

for (const scale of [1e-25, 1, 1e20]) {
  it(`review S1 difference scaling works with two varying bands at ${scale}`, () => {
    const result = execute(
      graph =>
        new GPURasterMedoidComposite({
          cellCount: 1,
          sliceCount: 3,
          bandCount: 3,
          values: view(graph, 'values', 'float32', 9),
          output: {
            composite: view(graph, 'composite', 'float32', 3),
            sliceIndex: view(graph, 'sliceIndex', 'uint32', 1),
            status: view(graph, 'status', 'uint32', 1)
          }
        }),
      {
        values: Float32Array.of(1e24, 0, 0, 1e24, scale, scale, 1e24, 2 * scale, 2 * scale),
        composite: new Float32Array(3),
        sliceIndex: new Uint32Array(1),
        status: new Uint32Array(1)
      }
    );
    // Collinear varying coordinates have the unique middle medoid; a constant coordinate is irrelevant.
    expect(Array.from(result['sliceIndex'])).toEqual([1]);
    expect(Array.from(result['composite'])).toEqual([
      Math.fround(1e24),
      Math.fround(scale),
      Math.fround(scale)
    ]);
    expect(Array.from(result['status'])).toEqual([0]);
  });
}

it('review B3 decodes the signed-field Gram entry from generated covariance WGSL', async () => {
  const {getGPURasterSnapshotCovarianceMatrix} = await import(
    '../../../src/raster-analysis/raster-time-series/raster-snapshot-eof-decomposition'
  );
  const parameters = getGPURasterSnapshotCovarianceParameterValues(
    {maximumValueMagnitude: 32768},
    1
  );
  const buffers = {
    values: Float32Array.of(32768, -32768, 1 / 1024),
    parameters,
    cellMeans: new Float32Array(1),
    cellAnchors: new Float32Array(1),
    cellKept: new Uint32Array(1),
    covarianceWords: new Uint32Array(18)
  };
  const make = (graph: GPUCommandGraph) =>
    new GPURasterSnapshotCovariance({
      cellCount: 1,
      sliceCount: 3,
      values: view(graph, 'values', 'float32', 3),
      parameters: view(graph, 'parameters', 'float32', 2),
      output: {
        covarianceWords: view(graph, 'covarianceWords', 'uint32', 18),
        keptCellCount: view(graph, 'kept', 'uint32', 1)
      }
    });
  execute(make, buffers);
  execute(make, buffers, 2);
  const matrix = getGPURasterSnapshotCovarianceMatrix(buffers.covarianceWords, 3, parameters[0]);
  expect(Number.isFinite(matrix[2])).toBe(true);
  expect(Math.abs(matrix[2] - 21.333333121405707)).toBeLessThan(2e-6);
});
