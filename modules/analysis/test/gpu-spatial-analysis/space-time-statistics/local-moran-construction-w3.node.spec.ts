// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUSpaceTimeLocalMoran,
  type GPUSpaceTimeLocalMoranProps
} from '../../../src/gpu-spatial-analysis/space-time-statistics/gpu-space-time-local-moran';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function fixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32'>(format: Format, length: number) =>
    createTransientView(graph, `input-${serial++}`, format, length);
  const props: GPUSpaceTimeLocalMoranProps = {
    id: 'retry-moran',
    weights: {
      offsets: view('uint32', 4),
      neighbors: view('uint32', 3),
      weights: view('float32', 3)
    },
    cube: view('float32', 6),
    layout: 'period-major',
    periodCount: 2,
    period: 1,
    variant: 'lagged',
    parameters: view('uint32', 4),
    maximumPermutations: 3,
    output: {
      exceedances: view('uint32', 3),
      pseudoPValues: view('float32', 3),
      status: {overflow: view('uint32', 1)}
    }
  };
  return {device, graph, props};
}

it('VAL-08 rejects delegated options before scratch and permits the same graph and ID', () => {
  const {device, graph, props} = fixture();
  try {
    const state = graph as unknown as {buffers: Map<string, unknown>; nodes: unknown[]};
    const resourceIds = [...state.buffers.keys()];
    expect(() => new GPUSpaceTimeLocalMoran({...props, maximumNeighbors: NaN})).toThrow(
      /retry-moran.*maximumNeighbors/
    );
    expect([...state.buffers.keys()]).toEqual(resourceIds);
    expect(state.nodes).toHaveLength(0);
    const nodes = new GPUSpaceTimeLocalMoran({...props, maximumNeighbors: 2}).getCommandNodes(
      graph
    );
    expect(nodes.length).toBeGreaterThan(1);
    expect([...state.buffers.keys()].filter(id => id === 'retry-moran-current')).toHaveLength(1);
  } finally {
    device.destroy();
  }
});

it('VAL-10 identifies required cube and parameter formats before dereferencing', () => {
  const {device, graph, props} = fixture();
  try {
    expect(() => new GPUSpaceTimeLocalMoran({...props, cube: undefined as never})).toThrow(
      /retry-moran.*cube.*data graph view.*float32.*undefined/
    );
    expect(
      () =>
        new GPUSpaceTimeLocalMoran({
          ...props,
          parameters: createTransientView(graph, 'bad-parameters', 'float32', 4) as never
        })
    ).toThrow(/retry-moran.*parameters.*uint32.*float32/);
    for (const name of ['layout', 'variant', 'direction'] as const)
      expect(() => new GPUSpaceTimeLocalMoran({...props, [name]: 'typo'} as never)).toThrow(
        new RegExp(`retry-moran.*${name}.*typo`)
      );
  } finally {
    device.destroy();
  }
});

it('B8 checks the child eight-binding requirement before reserving parent scratch', () => {
  const {device, graph, props} = fixture();
  const originalLimits = device.limits;
  try {
    const contributor = new GPUSpaceTimeLocalMoran(props);
    const state = graph as unknown as {buffers: Map<string, unknown>; nodes: unknown[]};
    const resourceIds = [...state.buffers.keys()];
    Object.defineProperty(device, 'limits', {
      configurable: true,
      value: {...originalLimits, maxStorageBuffersPerShaderStage: 7}
    });
    expect(() => contributor.getCommandNodes(graph)).toThrow(/retry-moran.*(limit|binding)/);
    expect([...state.buffers.keys()]).toEqual(resourceIds);
    expect(state.nodes).toHaveLength(0);
    Object.defineProperty(device, 'limits', {configurable: true, value: originalLimits});
    expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(1);
  } finally {
    device.destroy();
  }
});
