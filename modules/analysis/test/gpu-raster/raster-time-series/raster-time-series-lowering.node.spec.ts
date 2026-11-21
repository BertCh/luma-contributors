// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {createTransientView, GPUCommandGraph, GraphVectorView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  planRasterTimeSeriesChunks as planProperties,
  validateRasterTimeSeriesViews as validateProperties
} from '../../../src/raster-analysis/raster-time-series/raster-time-series-chunks';
import {
  planRasterTimeSeriesChunks as planFields,
  validateRasterTimeSeriesViews as validateFields
} from '../../../src/raster-analysis/raster-time-series/raster-time-series-chunks-b';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

for (const counts of [[3], [1, 2], [0, 1, 2]]) {
  for (const optional of [false, true]) {
    it(`RT-112 both planners preserve ports and offsets for ${counts}, optional=${optional}`, () => {
      const device = createNullWebGPUDevice();
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const packed = (length: number) => {
        const backing = createTransientView(graph, `backing-${serial++}`, 'float32', length + 64);
        return graph.createDataView(backing.buffer, {format: 'float32', length, byteOffset: 256});
      };
      const vector = (width: number) => {
        const data = counts.map(count => packed(count * width));
        return new GraphVectorView({
          id: `vector-${serial++}`,
          name: 'cells',
          format: 'float32',
          length: 3 * width,
          valueLength: 3 * width,
          stride: 1,
          byteStride: 4,
          rowByteLength: 4,
          data
        });
      };
      const values = vector(2);
      const means = vector(1);
      const diagnostics = optional ? vector(3) : undefined;
      const parameters = packed(4);
      const properties = {cellCount: 3, values, parameters, output: {means, diagnostics}};
      const fields = {
        values: {view: values, entriesPerCell: 2},
        means: {view: means, entriesPerCell: 1},
        diagnostics: {view: diagnostics, entriesPerCell: 3}
      };
      try {
        const first = planProperties('lowering', properties, {
          values: 2,
          'output.means': 1,
          'output.diagnostics': 3
        });
        const second = planFields('lowering', 3, fields);
        expect(first.map(chunk => chunk.cellCount)).toEqual(counts.filter(Boolean));
        expect(second.chunks.map(chunk => chunk.cellCount)).toEqual(counts);
        let dispatch = 0;
        for (const [index, count] of counts.entries()) {
          const mapped = second.chunks[index].views;
          expect(mapped.get(values)).toBe(values.data[index]);
          if (!count) continue;
          const chunk = first[dispatch++];
          expect(chunk.values).toBe(mapped.get(values));
          expect(chunk.output.means).toBe(means.data[index]);
          expect(mapped.get(means)).toBe(means.data[index]);
          expect(chunk.parameters).toBe(parameters);
          expect(chunk.output.diagnostics).toBe(diagnostics?.data[index]);
          if (diagnostics) expect(mapped.get(diagnostics)).toBe(diagnostics.data[index]);
          const lowered = mapped.get(values)!;
          expect([lowered.byteOffset, lowered.byteStride, lowered.length]).toEqual([
            256,
            4,
            count * 2
          ]);
        }
        const outputs = [means, ...(diagnostics ? [diagnostics] : [])];
        expect(() =>
          validateProperties('lowering', [values, parameters], outputs, [
            {inputs: [values, parameters], outputs}
          ])
        ).not.toThrow();
        expect(() =>
          validateFields('lowering', [values, parameters], outputs, second, [
            {inputs: [values, parameters], outputs}
          ])
        ).not.toThrow();
        expect(() => validateProperties('lowering', [values], [values])).toThrow(/overlap|share/);
        expect(() => validateFields('lowering', [values], [values], second)).toThrow(
          /overlap|share/
        );
      } finally {
        device.destroy();
      }
    });
  }
}

it('RT-112 packed over-allocation stays packed, and one-chunk vectors accept matching packed outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const values = createTransientView(graph, 'values', 'float32', 8);
  const output = createTransientView(graph, 'output', 'float32', 4);
  try {
    const properties = {cellCount: 3, values, output: {means: output}};
    expect(planProperties('packed', properties, {values: 2, 'output.means': 1})).toEqual([
      properties
    ]);
    expect(
      planFields('packed', 3, {
        values: {view: values, entriesPerCell: 2},
        means: {view: output, entriesPerCell: 1}
      }).chunked
    ).toBe(false);
    const exact = graph.createDataView(values.buffer, {format: 'float32', length: 6});
    const exactOutput = graph.createDataView(output.buffer, {format: 'float32', length: 3});
    const vector = new GraphVectorView({
      id: 'one',
      name: 'one',
      format: 'float32',
      length: 6,
      valueLength: 6,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data: [exact]
    });
    const first = planProperties(
      'one',
      {cellCount: 3, values: vector, output: {means: exactOutput}},
      {values: 2, 'output.means': 1}
    );
    const second = planFields('one', 3, {
      values: {view: vector, entriesPerCell: 2},
      means: {view: exactOutput, entriesPerCell: 1}
    });
    expect(first[0].output.means).toBe(exactOutput);
    expect(second.chunks[0].views.get(exactOutput)).toBe(exactOutput);
  } finally {
    device.destroy();
  }
});

it('RT-112 both planners reject strided storage and split records before lowering', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'bad-backing', 'float32', 32);
  const strided = graph.createDataView(backing.buffer, {
    format: 'float32',
    length: 6,
    byteStride: 8
  });
  const chunks = [3, 3].map((length, index) =>
    graph.createDataView(backing.buffer, {format: 'float32', length, byteOffset: index * 12})
  );
  const split = new GraphVectorView({
    id: 'split',
    name: 'split',
    format: 'float32',
    length: 6,
    valueLength: 6,
    stride: 1,
    byteStride: 4,
    rowByteLength: 4,
    data: chunks
  });
  try {
    for (const values of [strided, split]) {
      expect(() => planProperties('invalid', {cellCount: 3, values}, {values: 2})).toThrow(
        /packed|stride|split cell record/
      );
      expect(() => planFields('invalid', 3, {values: {view: values, entriesPerCell: 2}})).toThrow(
        /packed|stride|split cell record/
      );
    }
  } finally {
    device.destroy();
  }
});

it('RT-112 both validators use actual per-dispatch accesses for disjoint pooled fields', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const backing = createTransientView(graph, 'pool', 'float32', 129);
  const input = graph.createDataView(backing.buffer, {format: 'float32', length: 1, byteOffset: 0});
  const output = graph.createDataView(backing.buffer, {
    format: 'float32',
    length: 1,
    byteOffset: 256
  });
  const plan = planFields('pooled', 1, {
    input: {view: input, entriesPerCell: 1},
    output: {view: output, entriesPerCell: 1}
  });
  try {
    const separate = [
      {inputs: [input], outputs: []},
      {inputs: [], outputs: [output]}
    ];
    expect(() => validateProperties('pooled', [input], [output], separate)).not.toThrow();
    expect(() => validateFields('pooled', [input], [output], plan, separate)).not.toThrow();
    const together = [{inputs: [input], outputs: [output]}];
    expect(() => validateProperties('pooled', [input], [output], together)).toThrow(
      /same physical buffer|share buffers/
    );
    expect(() => validateFields('pooled', [input], [output], plan, together)).toThrow(
      /same physical buffer|share buffers/
    );
  } finally {
    device.destroy();
  }
});

for (const optional of [false, true]) {
  it(`RT-112 harmonic caller lowers optional fitted and RMSE writers: ${optional}`, async () => {
    const {GPURasterHarmonicRegression} = await import(
      '../../../src/raster-analysis/raster-time-series/gpu-raster-harmonic-regression'
    );
    const {getGPUCommandNodeKernelSources} = await import('../../utils/gpu-contributor-test-utils');
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const floats = (length: number) =>
      createTransientView(graph, `floats-${serial++}`, 'float32', length);
    const integers = (length: number) =>
      createTransientView(graph, `integers-${serial++}`, 'uint32', length);
    const vectors = (width: number) =>
      new GraphVectorView({
        id: `vector-${serial++}`,
        name: 'two-cells',
        format: 'float32',
        length: width * 2,
        valueLength: width * 2,
        stride: 1,
        byteStride: 4,
        rowByteLength: 4,
        data: [floats(width), floats(width)]
      });
    try {
      const contributor = new GPURasterHarmonicRegression({
        cellCount: 2,
        sliceCount: 5,
        harmonicCount: 1,
        mode: 'reference',
        values: vectors(5),
        parameters: floats(4),
        dateAxis: {
          acquisitionTimes: createTransientView(graph, 'dates', 'uint32x2', 5),
          temporalContext: {
            encoding: 'int64-words',
            unit: 'second',
            epoch: 0n,
            axis: {kind: 'irregular'}
          }
        },
        output: {
          coefficients: vectors(4),
          summary: new GraphVectorView({
            id: 'summary',
            name: 'summary',
            format: 'uint32',
            length: 4,
            valueLength: 4,
            stride: 1,
            byteStride: 4,
            rowByteLength: 4,
            data: [integers(2), integers(2)]
          }),
          ...(optional ? {fitted: vectors(5), rmse: vectors(1)} : {})
        }
      });
      const kernels = getGPUCommandNodeKernelSources(contributor.getCommandNodes(graph));
      expect(kernels).toHaveLength(2);
      for (const kernel of kernels) {
        expect(kernel.source.includes('fitted[fittedOffset')).toBe(optional);
        expect(kernel.source.includes('rmse[rmseOffset')).toBe(optional);
        expect(kernel.source).toContain('summary[summaryOffset + index * 2u]');
      }
    } finally {
      device.destroy();
    }
  });
}
