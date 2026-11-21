// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {GPUTerrainWatersheds} from '../../../src/gpu-terrain/hydrology/gpu-terrain-watersheds';
import {GPUTerrainFlow} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow';
import {GPUTerrainFlowDistance} from '../../../src/gpu-terrain/hydrology/gpu-terrain-flow-distance';
import {GPUTerrainStreamNetwork} from '../../../src/gpu-terrain/hydrology/gpu-terrain-stream-network';

function fixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const view = <Format extends 'uint32' | 'float32' | 'float32x2'>(format: Format, length = 4) =>
    createTransientView(graph, `forward-${serial++}`, format, length);
  const grid = {
    width: 2,
    height: 2,
    affine: [1, 0, 0, 0, -1, 2] as const,
    pixelInterpretation: 'area' as const
  };
  return {device, graph, view, grid};
}

it('captures inherited direct watershed budget and diagnostics once', () => {
  const {device, graph, grid, view} = fixture();
  try {
    let reads = 0;
    const iterationCount = view('uint32', 1);
    const status = Object.create({iterationCount});
    const output = Object.create({labels: view('uint32'), status});
    const props = Object.create({
      grid,
      flowDirections: view('uint32'),
      maximumIterations: 1,
      get output() {
        reads++;
        return output;
      }
    });
    const contributor = new GPUTerrainWatersheds(props);
    expect(reads).toBe(1);
    expect(contributor.props.maximumIterations).toBe(1);
    expect(contributor.props.output.status?.iterationCount).toBe(iterationCount);
    const nodes = contributor.getCommandNodes(graph);
    expect(nodes.filter(node => node.id.endsWith('-snapshot'))).toHaveLength(1);
    expect(
      () =>
        new GPUTerrainWatersheds(
          Object.create({
            ...props,
            grid,
            flowDirections: view('uint32'),
            output,
            maximumIterations: 1025
          })
        )
    ).toThrow(/maximumIterations/);
  } finally {
    device.destroy();
  }
});

it('captures inherited direct flow settings and an enumerable output getter once', () => {
  const {device, graph, grid, view} = fixture();
  try {
    let reads = 0;
    const output = {filledElevation: view('float32')};
    const inherited = {
      grid,
      elevation: {
        id: 'z',
        format: 'float32' as const,
        storage: {kind: 'buffer' as const, values: view('float32')}
      },
      parameters: view('float32', 16),
      fillDepressions: true,
      maximumFillIterations: 1
    };
    const props = Object.create(inherited);
    Object.defineProperty(props, 'output', {
      enumerable: true,
      get() {
        reads++;
        return output;
      }
    });
    const contributor = new GPUTerrainFlow(props);
    expect(reads).toBe(1);
    inherited.fillDepressions = false;
    inherited.maximumFillIterations = 1025;
    expect(contributor.props.fillDepressions).toBe(true);
    expect(contributor.props.maximumFillIterations).toBe(1);
    expect(() => contributor.getCommandNodes(graph)).not.toThrow();
    expect(
      () => new GPUTerrainFlow(Object.create({...inherited, fillDepressions: true, output}))
    ).toThrow(/maximumFillIterations/);
  } finally {
    device.destroy();
  }
});

it('rejects float D8 codes before tracing and scalar positions before reach publication', () => {
  const {device, grid, view} = fixture();
  try {
    const distance = {
      grid,
      flowDirections: view('uint32'),
      output: {distance: view('float32'), validity: view('uint32'), terminalCells: view('uint32')}
    };
    expect(
      () => new GPUTerrainFlowDistance({...distance, flowDirections: view('float32') as never})
    ).toThrow(/flowDirections.*uint32/);
    expect(
      () => new GPUTerrainFlowDistance({...distance, transitionCosts: view('float32x2') as never})
    ).toThrow(/transitionCosts.*float32/);
    const network = {
      grid,
      flowDirections: view('uint32'),
      streams: view('uint32'),
      output: {
        sourceCells: view('uint32'),
        targetCells: view('uint32'),
        vertexOffsets: view('uint32', 5),
        positions: view('float32x2', 8),
        lengths: view('float32'),
        reachCount: view('uint32', 1),
        vertexCount: view('uint32', 1),
        unresolvedCount: view('uint32', 1),
        status: {overflow: view('uint32', 1)}
      }
    };
    expect(
      () =>
        new GPUTerrainStreamNetwork({
          ...network,
          output: {...network.output, positions: view('float32', 8) as never}
        })
    ).toThrow(/output.positions.*float32x2/);
    expect(
      () =>
        new GPUTerrainStreamNetwork({
          ...network,
          output: {...network.output, status: {overflow: view('float32', 1) as never}}
        })
    ).toThrow(/status.overflow.*uint32/);
    expect(() => new GPUTerrainStreamNetwork(network)).not.toThrow();
  } finally {
    device.destroy();
  }
});
