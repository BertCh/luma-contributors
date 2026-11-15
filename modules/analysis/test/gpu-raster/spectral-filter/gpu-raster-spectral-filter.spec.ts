// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPURasterSpectralFilter,
  getGPURasterSpectralFilterParameterValues,
  type GPURasterSpectralFilterProps
} from '../../../src/raster-analysis/spectral-filter/index';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {requireTestDevice} from '../../utils/require-test-device';
import {createRemoteSensingHarness} from '../remote-sensing/remote-sensing-test-utils';
import oracle from '../../oracles/fixtures/raster-spectral-filter.json';

for (const fixture of oracle.filters) {
  it(`NumPy FFT spectral filtering: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const harness = createRemoteSensingHarness(device);
    const count = oracle.width * oracle.height;
    // Invalid prefix/tail rows are outside the raster domain. Offsets must survive the
    // grid-sized views passed to the published raster canonicalizer.
    const sourceValues = harness.input([NaN, ...oracle.input, NaN, 999]);
    const sourceValidity = harness.mask([0, 0, ...new Array(count).fill(1), 0, 0]);
    const result = await harness.run(
      new GPURasterSpectralFilter({
        grid: makeGPURasterMetadata({width: oracle.width, height: oracle.height}),
        input: {
          id: 'source',
          format: 'float32',
          storage: {
            kind: 'buffer',
            values: harness.graph.createDataView(sourceValues.buffer, {
              format: 'float32',
              byteOffset: 4,
              length: count + 2
            })
          },
          validity: harness.graph.createDataView(sourceValidity.buffer, {
            format: 'uint32',
            byteOffset: 8,
            length: count + 2
          }),
          scale: 1.5,
          offset: -2
        },
        filter: fixture.filter as GPURasterSpectralFilterProps['filter'],
        parameters: harness.input(
          Array.from(
            getGPURasterSpectralFilterParameterValues({
              ...fixture.settings,
              notchCenter: fixture.settings.notchCenter as [number, number]
            })
          )
        ),
        output: {
          values: harness.output('values', 'float32', count + 2),
          validity: harness.output('validity', 'uint32', count + 2),
          status: {invalidCount: harness.output('invalidCount', 'uint32', 1)}
        }
      })
    );
    expect(result['invalidCount']).toEqual([0]);
    expect(result['validity'].slice(0, count)).toEqual(new Array(count).fill(1));
    for (let index = 0; index < count; index++)
      expect(result['values'][index]).toBeCloseTo(fixture.expected[index], 4);
    expect(result['values'].slice(count)).toEqual([0, 0]);
    expect(result['validity'].slice(count)).toEqual([0, 0]);
  });
}

it('global no-data propagation honors mask, sentinel and nonfinite input, including all-missing rasters', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const missing of ['mask', 'sentinel', 'nan', 'infinity', 'all'] as const) {
    const harness = createRemoteSensingHarness(device);
    const values = [1, 2, 3, 4];
    if (missing === 'sentinel') values[1] = -999;
    if (missing === 'nan') values[1] = NaN;
    if (missing === 'infinity') values[1] = Infinity;
    const result = await harness.run(
      new GPURasterSpectralFilter({
        grid: makeGPURasterMetadata({width: 2, height: 2}),
        input: {
          id: 'source',
          format: 'float32',
          storage: {kind: 'buffer', values: harness.input(values)},
          validity: harness.mask(
            missing === 'all' ? [0, 0, 0, 0] : missing === 'mask' ? [1, 0, 1, 1] : [1, 1, 1, 1]
          ),
          noDataValue: -999
        },
        filter: 'low-pass',
        parameters: harness.input(Array.from(getGPURasterSpectralFilterParameterValues())),
        output: {
          values: harness.output('values', 'float32', 4),
          validity: harness.output('validity', 'uint32', 4),
          status: {invalidCount: harness.output('invalidCount', 'uint32', 1)}
        }
      })
    );
    expect(result['invalidCount']).toEqual([missing === 'all' ? 4 : 1]);
    expect(result['validity']).toEqual([0, 0, 0, 0]);
    for (const value of result['values']) expect(value).toBeNaN();
  }
});

it('invalid per-frame parameters invalidate the result on the GPU', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createRemoteSensingHarness(device);
  const parameters = getGPURasterSpectralFilterParameterValues();
  parameters[3] = 0;
  const result = await harness.run(
    new GPURasterSpectralFilter({
      grid: makeGPURasterMetadata({width: 2, height: 2}),
      input: {
        id: 'source',
        format: 'float32',
        storage: {kind: 'buffer', values: harness.input([1, 2, 3, 4])}
      },
      filter: 'gaussian',
      parameters: harness.input(Array.from(parameters)),
      output: {
        values: harness.output('values', 'float32', 4),
        validity: harness.output('validity', 'uint32', 4)
      }
    })
  );
  expect(result['validity']).toEqual([0, 0, 0, 0]);
  for (const value of result['values']) expect(value).toBeNaN();
});

it('updates settings and validity in one compiled graph and preserves output subview padding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const source = createInputBuffer(device, new Float32Array([1, 2, 3, 4]));
  const mask = createInputBuffer(device, new Uint32Array([1, 1, 1, 1]));
  const values = createOutputBuffer(device, 6);
  const validity = createOutputBuffer(device, 6);
  const invalidCount = createOutputBuffer(device, 1);
  values.write(new Float32Array([91, 92, 93, 94, 95, 96]));
  validity.write(new Uint32Array([11, 12, 13, 14, 15, 16]));
  const valueView = importGraphBuffer(graph, 'values', values, 'float32');
  const validityView = importGraphBuffer(graph, 'validity', validity, 'uint32');
  const parameters = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: 8,
    values: getGPURasterSpectralFilterParameterValues({cutoff: 0})
  });
  graph.add(
    new GPURasterSpectralFilter({
      grid: makeGPURasterMetadata({width: 2, height: 2}),
      input: {
        id: 'source',
        format: 'float32',
        storage: {kind: 'buffer', values: importGraphBuffer(graph, 'source', source, 'float32')},
        validity: importGraphBuffer(graph, 'mask', mask, 'uint32')
      },
      filter: 'low-pass',
      parameters: parameters.importToGraph(graph),
      output: {
        values: graph.createDataView(valueView.buffer, {
          format: 'float32',
          byteOffset: 4,
          length: 4
        }),
        validity: graph.createDataView(validityView.buffer, {
          format: 'uint32',
          byteOffset: 4,
          length: 4
        }),
        status: {invalidCount: importGraphBuffer(graph, 'invalid-count', invalidCount, 'uint32')}
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled);
    expect(await readFloat32(values)).toEqual([91, 2.5, 2.5, 2.5, 2.5, 96]);
    parameters.write(getGPURasterSpectralFilterParameterValues({cutoff: Math.SQRT1_2}));
    submitGraph(device, compiled);
    const allPass = await readFloat32(values);
    [91, 1, 2, 3, 4, 96].forEach((expected, index) =>
      expect(allPass[index]).toBeCloseTo(expected, 5)
    );
    mask.write(new Uint32Array([0, 0, 0, 0]));
    submitGraph(device, compiled);
    expect(await readUint32(invalidCount)).toEqual([4]);
    expect(await readUint32(validity)).toEqual([11, 0, 0, 0, 0, 16]);
    mask.write(new Uint32Array([1, 1, 1, 1]));
    submitGraph(device, compiled);
    expect(await readUint32(invalidCount)).toEqual([0]);
    expect(await readUint32(validity)).toEqual([11, 1, 1, 1, 1, 16]);
  } finally {
    compiled.destroy();
    parameters.destroy();
    [source, mask, values, validity, invalidCount].forEach(buffer => buffer.destroy());
  }
});

for (const transpose of [false, true]) {
  for (const centerSign of [1, -1]) {
    it(`preserves both Nyquist conjugate bins at a near-boundary notch (transpose=${transpose}, centerSign=${centerSign})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const harness = createRemoteSensingHarness(device);
      // Two Fourier bins: (-1/2, +/-1/8), or their transpose. The exact shortest
      // Nyquist-axis distance is 1/2 - float32(0.1), strictly greater than this radius.
      const values = Array.from({length: 64}, (_, index) => {
        const column = index % 8;
        const row = Math.floor(index / 8);
        const alternatingCoordinate = transpose ? row : column;
        const cosineCoordinate = transpose ? column : row;
        return Math.fround(
          (-1) ** alternatingCoordinate * Math.cos((2 * Math.PI * cosineCoordinate) / 8)
        );
      });
      const notchCenter: [number, number] = transpose
        ? [centerSign * 0.125, centerSign * 0.1]
        : [centerSign * 0.1, centerSign * 0.125];
      const result = await harness.run(
        new GPURasterSpectralFilter({
          grid: makeGPURasterMetadata({width: 8, height: 8}),
          input: {
            id: 'nyquist-signal',
            format: 'float32',
            storage: {kind: 'buffer', values: harness.input(values)}
          },
          filter: 'notch',
          parameters: harness.input(
            Array.from(
              getGPURasterSpectralFilterParameterValues({
                notchCenter,
                notchRadius: 0.3999999761581421
              })
            )
          ),
          output: {
            values: harness.output('values', 'float32', 64),
            validity: harness.output('validity', 'uint32', 64)
          }
        })
      );
      expect(result['validity']).toEqual(new Array(64).fill(1));
      // Preserving the analytic signal detects the old asymmetric mask's half amplitude.
      for (let index = 0; index < 64; index++)
        expect(result['values'][index]).toBeCloseTo(values[index], 4);
    });
  }
}
