// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {Buffer} from '@luma.gl/core';
import {
  createTransientView,
  GPUCommandGraph,
  GraphDataView,
  GraphVectorView
} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  getNetworkViews,
  validateNetworkWritableViews
} from '../../../src/gpu-network/network-view-utils';
import {
  GPUNetworkIsochrones,
  type GPUNetworkIsochronesProps
} from '../../../src/gpu-network/network-isochrones/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const other = createRequire(import.meta.url)(
  '@luma.gl/graph/gpu-core'
) as typeof import('@luma.gl/graph/gpu-core');

function foreignView<Format extends 'float32' | 'uint32'>(view: GraphDataView<Format>) {
  return new other.GraphDataView(view.buffer, view);
}

it('collects mixed constructors in nested input and output maps and checks aliases (ENT-02)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    expect(other.GraphDataView === GraphDataView).toBe(false);
    expect(other.GraphVectorView === GraphVectorView).toBe(false);
    const local = createTransientView(graph, 'local', 'float32', 1);
    const remote = foreignView(createTransientView(graph, 'other', 'float32', 1));
    const vector = new other.GraphVectorView({
      id: 'vector',
      name: 'vector',
      format: 'float32',
      length: 2,
      valueLength: 2,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data: [local, remote]
    });
    expect(getNetworkViews({input: {rows: [local, remote]}, output: {nested: vector}}).length).toBe(
      3
    );
    expect(() => validateNetworkWritableViews('mixed', {nested: {local, remote}})).not.toThrow();
    expect(() => validateNetworkWritableViews('mixed', {nested: {vector, alias: remote}})).toThrow(
      /mixed.*overlap/
    );
    expect(() =>
      validateNetworkWritableViews('mixed', {nested: {result: remote}}, [
        {outputs: {nested: {result: remote}}, inputs: {nested: {source: remote}}}
      ])
    ).toThrow(/mixed.*share buffers with inputs/);
  } finally {
    device.destroy();
  }
});

function createProps(graph: GPUCommandGraph): GPUNetworkIsochronesProps {
  return {
    id: 'mixed-isochrones',
    topology: {
      offsets: createTransientView(graph, 'offsets', 'uint32', 5),
      neighbors: createTransientView(graph, 'neighbors', 'uint32', 1),
      weights: createTransientView(graph, 'weights', 'float32', 1),
      nodePositions: createTransientView(graph, 'positions', 'float32x2', 4)
    },
    costs: foreignView(createTransientView(graph, 'costs', 'float32', 4)),
    costBreaks: foreignView(createTransientView(graph, 'breaks', 'float32', 1)),
    parameters: foreignView(createTransientView(graph, 'parameters', 'float32', 12)),
    raster: {
      width: 2,
      height: 2,
      output: {bandClasses: foreignView(createTransientView(graph, 'bands', 'uint32', 4))}
    }
  };
}

it('checks mixed-constructor isochrone aliases, formats, chunking and binding usage (ENT-02)', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const props = createProps(graph);
    expect(() => new GPUNetworkIsochrones(props)).not.toThrow();
    expect(
      () =>
        new GPUNetworkIsochrones({
          ...props,
          raster: {
            ...props.raster!,
            output: {
              ...props.raster!.output,
              values: new other.GraphDataView(props.costs.buffer, {...props.costs, length: 4})
            }
          }
        })
    ).toThrow(/mixed-isochrones.*values.*costs/);
    expect(
      () =>
        new GPUNetworkIsochrones({
          ...props,
          costs: foreignView(
            createTransientView(graph, 'wrong-costs', 'uint32', 4)
          ) as unknown as GraphDataView<'float32'>
        })
    ).toThrow(/mixed-isochrones.*costs.*uint32/);
    const vector = new other.GraphVectorView({
      id: 'chunked-costs',
      name: 'costs',
      format: 'float32',
      length: 4,
      valueLength: 4,
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data: [props.costs]
    });
    expect(
      () =>
        new GPUNetworkIsochrones({...props, costs: vector as unknown as GraphDataView<'float32'>})
    ).toThrow(/mixed-isochrones.*costs.*packed/);
    const contributor = new GPUNetworkIsochrones(props);
    Object.defineProperty(props.costBreaks.buffer, 'usage', {value: Buffer.COPY_DST});
    const scratch = vi.spyOn(graph, 'createTransientBuffer');
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /mixed-isochrones.*costBreaks.*STORAGE/
    );
    expect(scratch).not.toHaveBeenCalled();
  } finally {
    device.destroy();
  }
});
