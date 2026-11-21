// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {WgslReflect} from 'wgsl_reflect';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {expect, it} from 'vitest';
import {GPUKernelDensity} from '../../../src/gpu-spatial-analysis/point-density/gpu-kernel-density';
import {GPULineDensity} from '../../../src/gpu-spatial-analysis/line-density/gpu-line-density';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {makeExtentGrid} from '../spatial-interpolation/extent-grid';

it('builds a per-frame binned bandwidth diagnostic for explicit and automatic bandwidth', () => {
  for (const bandwidthRule of [undefined, 'scott'] as const) {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const status = {
      bandwidthBelowThreeCells: createTransientView(graph, 'small-bandwidth', 'uint32', 1)
    };
    const density = new GPUKernelDensity({
      positions: createTransientView(graph, 'positions', 'float32x2', 4),
      parameters: createTransientView(graph, 'parameters', 'float32', 9),
      grid: makeExtentGrid(4, 3, [0, 0, 4, 6]),
      indexGridSize: [2, 2],
      indexBounds: [0, 0, 4, 6],
      method: 'binned',
      bandwidthRule,
      output: {
        values: createTransientView(graph, 'values', 'float32', 12),
        status,
        bandwidth: bandwidthRule ? createTransientView(graph, 'bandwidth', 'float32', 1) : undefined
      }
    });
    const nodes = density.getCommandNodes(graph);
    const diagnostic = nodes.find(node => node.id.endsWith('-bandwidth-resolution'));
    expect(diagnostic).toBeDefined();
    const source = getGPUCommandNodeKernelSources([diagnostic!])
      .map(kernel => kernel.source)
      .join('\n');
    expect(source).toContain('isBandwidthBelowThreeCells(bandwidth, abs(cellSize.x))');
    expect(source).toContain('isBandwidthBelowThreeCells(bandwidth, abs(cellSize.y))');
    expect(source).toContain(
      bandwidthRule ? 'bandwidthState[bandwidthStateOffset]' : 'params[paramsOffset + 4u]'
    );
    expect(() => new WgslReflect(source)).not.toThrow();
    const memory = createCPUGraphMemory();
    for (const [bandwidth, expected] of [
      [2.5, 1],
      [3, 1],
      [5.5, 1],
      [6, 0],
      [6.5, 0],
      [2.5, 1]
    ]) {
      memory.write(
        density.props.parameters,
        new Float32Array([0, 0, 4, 6, bandwidthRule ? 99 : bandwidth, 0, 0, 4, 6])
      );
      if (density.props.output.bandwidth)
        memory.write(density.props.output.bandwidth, new Float32Array([bandwidth]));
      executeCommandNodesOnCPU([diagnostic!], memory);
      expect(Array.from(memory.readUint32(status.bandwidthBelowThreeCells))).toEqual([expected]);
    }

    device.destroy();
  }
});

it('builds an optional geographic seam count independently of the length walk', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const status = {
    overflow: createTransientView(graph, 'overflow', 'uint32', 1),
    antimeridianUncertainCount: createTransientView(graph, 'seams', 'uint32', 1)
  };
  const density = new GPULineDensity({
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    lineOffsets: createTransientView(graph, 'offsets', 'uint32', 3),
    parameters: createTransientView(graph, 'parameters', 'float32', 4),
    columns: 360,
    rows: 1,
    spatialContext: {
      coordinateSpace: 'longitude-latitude',
      metric: 'great-circle',
      units: 'meters'
    },
    output: {lengths: createTransientView(graph, 'lengths', 'float32', 360), status}
  });
  const nodes = density.getCommandNodes(graph);
  expect(nodes.some(node => node.id.endsWith('-seam-flags'))).toBe(true);
  for (const kernel of getGPUCommandNodeKernelSources(
    nodes.filter(node => node.id.includes('-seam-'))
  )) {
    expect(() => new WgslReflect(kernel.source), kernel.id).not.toThrow();
  }

  device.destroy();
});

it('reports the small-axis condition only when auto actually selects the binned route', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const parameters = createTransientView(graph, 'parameters', 'float32', 9);
  const resolution = createTransientView(graph, 'resolution', 'uint32', 1);
  const density = new GPUKernelDensity({
    positions: createTransientView(graph, 'positions', 'float32x2', 4),
    parameters,
    grid: makeExtentGrid(4, 3, [0, 0, 1, 12]),
    indexBounds: [0, 0, 1, 12],
    indexGridSize: [2, 2],
    method: 'auto',
    binnedPadding: 32,
    output: {
      values: createTransientView(graph, 'values', 'float32', 12),
      status: {bandwidthBelowThreeCells: resolution}
    }
  });
  const diagnostic = density
    .getCommandNodes(graph)
    .find(node => node.id.endsWith('-bandwidth-resolution'))!;
  const memory = createCPUGraphMemory();
  // Cells .25 x 4: auto support 5h/.25 exceeds 32 at h>1.6; y needs h>=12 for three cells.
  // At h=4 the fine-axis required margin 2.5*16 exceeds 33 while y spans only one
  // cell; zero therefore distinguishes the exact fallback from a binned warning.
  for (const [bandwidth, expected] of [
    [0.5, 0],
    [2, 1],
    [4, 0],
    [2, 1]
  ]) {
    memory.write(parameters, new Float32Array([0, 0, 1, 12, bandwidth, 0, 0, 1, 12]));
    executeCommandNodesOnCPU([diagnostic], memory);
    expect(Array.from(memory.readUint32(resolution))).toEqual([expected]);
  }
  device.destroy();
});
