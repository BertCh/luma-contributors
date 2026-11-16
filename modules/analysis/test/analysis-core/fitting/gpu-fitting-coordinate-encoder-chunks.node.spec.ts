// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph, type GraphVectorView} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUFittingCoordinateEncoder,
  type GPUFittingCoordinateEncoderProps
} from '../../../src/analysis-core/kernels/gpu-fitting-coordinate-encoder';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getGPUCommandNodePipelineKeys,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {expectValidShaderIdentifiers} from '../../gpu-dataframe/calendar-buckets/chunk-shader-validation.test-utils';

function createFixture(lengths: number[], encoding: 'fourier' | 'spherical-harmonics' = 'fourier') {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const dimensions = encoding === 'fourier' ? 3 : 2;
  const column = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    width: number
  ) =>
    createVectorView(
      name,
      format,
      lengths.map((rows, index) => {
        const backing = createTransientView(graph, `${name}-${index}`, format, rows * width + 128);
        return graph.createDataView(backing.buffer, {
          format,
          byteOffset: (index % 64) * 4,
          length: rows * width
        });
      })
    );
  const common = {
    id: 'encoder',
    rowCount: lengths.reduce((sum, length) => sum + length, 0),
    coordinates: column('coordinates', 'float32', dimensions),
    mask: column('mask', 'uint32', 1),
    output: {features: column('features', 'float32', 4), validity: column('validity', 'uint32', 1)}
  };
  const props: GPUFittingCoordinateEncoderProps<GraphVectorView> =
    encoding === 'fourier'
      ? {
          ...common,
          encoding,
          dimensions,
          frequencyCount: 2,
          frequencies: createTransientView(graph, 'frequencies', 'float32', 6)
        }
      : {
          ...common,
          encoding,
          maximumDegree: 1,
          spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
        };
  return {device, graph, props};
}

it('validates whole source rows, topology and writable aliases across flat scalar chunks', () => {
  const {device, graph, props} = createFixture([1, 3, 2]);
  try {
    const coordinates = props.coordinates as GraphVectorView<'float32'>;
    const features = props.output.features as GraphVectorView<'float32'>;
    expect(
      () =>
        new GPUFittingCoordinateEncoder({
          ...props,
          coordinates: createVectorView('reversed', 'float32', [...coordinates.data].reverse())
        })
    ).toThrow(/features chunk 0.*topology|mask chunk 0.*topology/);
    const first = graph.createDataView(coordinates.data[0].buffer, {format: 'float32', length: 4});
    const second = graph.createDataView(coordinates.data[1].buffer, {format: 'float32', length: 8});
    expect(
      () =>
        new GPUFittingCoordinateEncoder({
          ...props,
          coordinates: createVectorView('partial-rows', 'float32', [
            first,
            second,
            coordinates.data[2]
          ])
        })
    ).toThrow(/coordinates chunks must contain whole rows of 3 elements/);
    expect(
      () =>
        new GPUFittingCoordinateEncoder({
          ...props,
          output: {
            ...props.output,
            features: createVectorView('repeated', 'float32', [features.data[1], features.data[1]])
          }
        })
    ).toThrow(/topology|must not overlap/);
    const validity = graph.createDataView(features.data[1].buffer, {
      format: 'uint32',
      length: 3,
      byteOffset: 4
    });
    const originalValidity = props.output.validity as GraphVectorView<'uint32'>;
    expect(
      () =>
        new GPUFittingCoordinateEncoder({
          ...props,
          output: {
            ...props.output,
            validity: createVectorView('alias', 'uint32', [
              originalValidity.data[0],
              validity,
              originalValidity.data[2]
            ])
          }
        })
    ).toThrow(/must not overlap/);
  } finally {
    device.destroy();
  }
});

it('prechecks full flattened bindings and side frequencies before creating graph resources', () => {
  for (const field of ['coordinates', 'features', 'frequencies'] as const) {
    const {device, graph, props} = createFixture([1, 3, 2]);
    if (props.encoding !== 'fourier') throw new Error('expected Fourier fixture');
    Object.defineProperty(device.limits, 'maxStorageBufferBindingSize', {value: 256});
    const backing = createTransientView(graph, 'oversized', 'float32', 128);
    const length = field === 'frequencies' ? 6 : field === 'coordinates' ? 9 : 12;
    const oversized = graph.createDataView(backing.buffer, {
      format: 'float32',
      byteOffset: 252,
      length
    });
    const coordinates = props.coordinates as GraphVectorView<'float32'>;
    const features = props.output.features as GraphVectorView<'float32'>;
    const changed =
      field === 'frequencies'
        ? {...props, frequencies: oversized}
        : field === 'coordinates'
          ? {
              ...props,
              coordinates: createVectorView('late', 'float32', [
                coordinates.data[0],
                oversized,
                coordinates.data[2]
              ])
            }
          : {
              ...props,
              output: {
                ...props.output,
                features: createVectorView('late', 'float32', [
                  features.data[0],
                  oversized,
                  features.data[2]
                ])
              }
            };
    const contributor = new GPUFittingCoordinateEncoder(changed);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    const createView = vi.spyOn(graph, 'createDataView');
    const add = vi.spyOn(graph, 'add');
    expect(() => contributor.getCommandNodes(graph)).toThrow(/needs .*bytes/);
    expect(allocate).not.toHaveBeenCalled();
    expect(createView).not.toHaveBeenCalled();
    expect(add).not.toHaveBeenCalled();
    expect(() => contributor.getCommandNodes(new GPUCommandGraph(device))).toThrow(
      /views must belong/
    );
    device.destroy();
  }
});

for (const encoding of ['fourier', 'spherical-harmonics'] as const) {
  it(`keeps ${encoding} shader/pipeline counts constant and splits large flat feature outputs`, () => {
    const keys = new Set<string>();
    const sources = new Set<string>();
    for (const lengths of [
      [513],
      [513, 1, 257],
      Array.from({length: 30}, (_, index) => [513, 1, 257][index % 3]),
      [0, 4_000_000, 3_000_000, 5_000_000]
    ]) {
      const {device, graph, props} = createFixture(lengths, encoding);
      const contributor = new GPUFittingCoordinateEncoder(props);
      const nodes = contributor.getCommandNodes(graph);
      expect(nodes).toHaveLength(contributor.estimatedCommandNodeCount);
      expect(getParsedKernelSources(nodes).size).toBe(lengths.filter(length => length > 0).length);
      expect(getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}).size).toBe(1);
      if (lengths[0] === 513)
        for (const key of getGPUCommandNodePipelineKeys(nodes, {includeShaderLayout: true}))
          keys.add(key);
      for (const {source} of getGPUCommandNodeKernelSources(nodes)) sources.add(source);
      device.destroy();
    }
    expect(keys.size).toBe(1);
    expect(sources.size).toBe(1);
    expectValidShaderIdentifiers(sources);
  });
}

it('handles a zero-chunk vector without declaring nodes or scratch', () => {
  const {device, graph, props} = createFixture([]);
  try {
    const contributor = new GPUFittingCoordinateEncoder(props);
    expect(contributor.estimatedCommandNodeCount).toBe(0);
    const allocate = vi.spyOn(graph, 'createTransientBuffer');
    expect(contributor.getCommandNodes(graph)).toEqual([]);
    expect(allocate).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});
