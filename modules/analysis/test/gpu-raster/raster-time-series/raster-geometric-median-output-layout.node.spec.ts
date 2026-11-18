// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {readFileSync} from 'node:fs';
import * as rasterTimeSeries from '../../../src/raster-analysis/raster-time-series/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {constructRasterOperationContributor} from '../../utils/raster-operation-test-utils';

it('describes unequal cell and band counts and omits unrequested lastStep storage', () => {
  const options = {cellCount: 2, bandCount: 3};
  const layout = rasterTimeSeries.getGPURasterGeometricMedianCompositeOutputLayout(options);
  expect(layout).toEqual({
    composite: {format: 'float32', length: 6},
    status: {format: 'uint32', length: 2}
  });
  expect(Object.hasOwn(layout, 'lastStep')).toBe(false);
  expect(
    rasterTimeSeries.getGPURasterGeometricMedianCompositeOutputLayout({...options, lastStep: true})
  ).toEqual({...layout, lastStep: {format: 'float32', length: 2}});
  expect(
    rasterTimeSeries.getGPURasterGeometricMedianCompositeOutputLayout({...options, lastStep: false})
  ).toEqual(layout);
});

it.each([
  'composite',
  'status',
  'lastStep'
] as const)('rejects a wrong-format %s even when its row count matches', name => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const output = {
      composite: createTransientView(graph, 'composite', 'float32', 6),
      status: createTransientView(graph, 'status', 'uint32', 2),
      lastStep: createTransientView(graph, 'lastStep', 'float32', 2)
    };
    const wrong = createTransientView(
      graph,
      'wrong',
      name === 'status' ? 'float32' : 'uint32',
      name === 'composite' ? 6 : 2
    );
    expect(
      () =>
        new rasterTimeSeries.GPURasterGeometricMedianComposite({
          id: 'layout-format',
          cellCount: 2,
          sliceCount: 1,
          bandCount: 3,
          iterationCount: 1,
          values: createTransientView(graph, 'values', 'float32', 6),
          parameters: createTransientView(graph, 'parameters', 'float32', 1),
          output: {...output, [name]: wrong} as typeof output
        })
    ).toThrow(
      new RegExp(`layout-format output.${name}.*${name === 'status' ? 'uint32' : 'float32'}`)
    );
  } finally {
    device.destroy();
  }
});

it.each([
  false,
  true
])('shares requested output lengths and formats across every form: lastStep=%s', async lastStep => {
  const props: rasterTimeSeries.RasterGeometricMedianCompositeOperationProps = {
    id: 'layout-forms',
    cellCount: 2,
    bandCount: 3,
    sliceCount: 1,
    iterationCount: 1,
    values: 'observations',
    parameters: 'settings',
    output: {
      composite: 'median-values',
      status: 'cell-codes',
      ...(lastStep ? {lastStep: 'final-step'} : {})
    }
  };
  const layout = rasterTimeSeries.getGPURasterGeometricMedianCompositeOutputLayout({
    ...props,
    lastStep
  });
  const gpuOperation = new rasterTimeSeries.GPURasterGeometricMedianCompositeOperation(props);
  const cpuOperation = new rasterTimeSeries.CPURasterGeometricMedianCompositeOperation(props);
  expect(Object.keys(gpuOperation.outputs).sort()).toEqual(Object.keys(layout).sort());
  expect(Object.keys(cpuOperation.outputs).sort()).toEqual(Object.values(props.output).sort());
  for (const [name, descriptor] of Object.entries(layout)) {
    const value = props.output[name as keyof typeof props.output]!;
    expect(gpuOperation.outputs[name]).toEqual({...descriptor, value});
    expect(cpuOperation.outputs[value]).toEqual({type: 'vector', ...descriptor});
  }
  constructRasterOperationContributor(gpuOperation, {
    values: {format: 'float32', length: 6},
    parameters: {format: 'float32', length: 1}
  });
  const values = Float32Array.of(10, 11, 12, 20, 21, 22);
  const parameters = Float32Array.of(1e-5);
  const result = rasterTimeSeries.computeRasterGeometricMedianCompositeOnCPU(
    values,
    parameters,
    props
  );
  expect(Array.from(result.composite)).toEqual([10, 11, 12, 20, 21, 22]);
  expect(Array.from(result.status)).toEqual([0, 0]);
  expect(Array.from(result.lastStep)).toEqual([0, 0]);
  const cpuVector = (rows: Float32Array) => ({
    format: 'float32' as const,
    length: rows.length,
    chunks: [{values: rows, length: rows.length}],
    pack: () => rows
  });
  const published = await cpuOperation.execute({
    inputs: {values: cpuVector(values), parameters: cpuVector(parameters)}
  });
  expect(Object.keys(published).sort()).toEqual(Object.values(props.output).sort());
  for (const [name, descriptor] of Object.entries(layout)) {
    const output = published[props.output[name as keyof typeof props.output]!] as
      | Float32Array
      | Uint32Array;
    expect(output.length).toBe(descriptor.length);
    expect(output).toEqual(result[name as keyof typeof result]);
  }
});

it('validates every requested output against its used prefix while allowing pooled spare capacity', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const options = {
      id: 'layout-capacity',
      cellCount: 2,
      bandCount: 3,
      sliceCount: 1,
      iterationCount: 1
    };
    const layout = rasterTimeSeries.getGPURasterGeometricMedianCompositeOutputLayout({
      ...options,
      lastStep: true
    });
    const output = Object.fromEntries(
      Object.entries(layout).map(([name, descriptor]) => {
        const backing = createTransientView(graph, name, descriptor.format, descriptor.length + 3);
        return [
          name,
          graph.createDataView(backing.buffer, {
            ...descriptor,
            byteOffset: 4,
            length: descriptor.length + 2
          })
        ];
      })
    ) as {
      composite: GraphDataView<'float32'>;
      status: GraphDataView<'uint32'>;
      lastStep: GraphDataView<'float32'>;
    };
    const props = {
      ...options,
      values: createTransientView(graph, 'values', 'float32', 6),
      parameters: createTransientView(graph, 'parameters', 'float32', 1),
      output
    };
    expect(() => new rasterTimeSeries.GPURasterGeometricMedianComposite(props)).not.toThrow();
    for (const [name, descriptor] of Object.entries(layout)) {
      const short = graph.createDataView(output[name as keyof typeof output].buffer, {
        ...descriptor,
        byteOffset: 4,
        length: descriptor.length - 1
      });
      expect(
        () =>
          new rasterTimeSeries.GPURasterGeometricMedianComposite({
            ...props,
            output: {...output, [name]: short}
          })
      ).toThrow(
        new RegExp(`layout-capacity output.${name} must hold at least ${descriptor.length} entries`)
      );
    }
  } finally {
    device.destroy();
  }
});

it('rejects invalid layout dimensions before allocation and retains CPU band counts above the GPU cap', () => {
  for (const [cellCount, bandCount] of [
    [0, 3],
    [2, 0],
    [1.5, 3],
    [2, NaN],
    [Infinity, 3],
    [65536, 65536]
  ]) {
    expect(() =>
      rasterTimeSeries.getGPURasterGeometricMedianCompositeOutputLayout({
        id: 'layout-limits',
        cellCount,
        bandCount
      })
    ).toThrow(/^layout-limits (cellCount|bandCount)/);
  }
  expect(
    rasterTimeSeries.getGPURasterGeometricMedianCompositeOutputLayout({cellCount: 1, bandCount: 17})
      .composite.length
  ).toBe(17);
  expect(
    rasterTimeSeries.getGPURasterGeometricMedianCompositeOutputLayout({
      cellCount: 65535,
      bandCount: 65537
    }).composite.length
  ).toBe(0xffffffff);
});

it('keeps the documented output table equal to the public layout for unequal dimensions', () => {
  const documentation = readFileSync(
    new URL('../../../../../docs/api/gpu-raster/raster-time-series.md', import.meta.url),
    'utf8'
  );
  const table = documentation
    .split('{/* geometric-median-output-layout:start */}')[1]
    ?.split('{/* geometric-median-output-layout:end */}')[0];
  expect(table, 'the geometric median allocation table must be present').toBeDefined();
  for (const options of [
    {cellCount: 2, bandCount: 3},
    {cellCount: 3, bandCount: 2}
  ]) {
    const documented = Object.fromEntries(
      table!.split('\n').flatMap(line => {
        const match = line.match(/^\| `(\w+)` \| `(float32|uint32)` \| `([^`]+)` \|/);
        if (!match) return [];
        const [, name, format, expression] = match;
        const length = expression
          .split(' * ')
          .reduce((product, dimension) => product * options[dimension as keyof typeof options], 1);
        return [[name, {format, length}]];
      })
    );
    expect(documented).toEqual(
      rasterTimeSeries.getGPURasterGeometricMedianCompositeOutputLayout({
        ...options,
        lastStep: true
      })
    );
  }
});
