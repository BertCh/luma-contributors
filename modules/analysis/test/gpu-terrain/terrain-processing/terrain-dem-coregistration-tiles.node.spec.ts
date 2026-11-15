// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUTerrainDEMCoregistration} from '../../../src/gpu-terrain/terrain-processing/gpu-terrain-dem-coregistration';
import {auditWGSLModule} from '../../analysis-core/kernel-audit/kernel-audit-wgsl';
import {
  createNullWebGPUDevice,
  getParsedKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

function setup(cellCount: number, correlationLength: number) {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  let identifier = 0;
  const view = <Format extends 'float32' | 'uint32'>(format: Format, length: number) =>
    createTransientView(graph, `view-${identifier++}`, format, length);
  return {
    graph,
    props: {
      grid: {
        width: cellCount,
        height: 1,
        affine: [1, 0, 0, 0, -1, 0] as const,
        pixelInterpretation: 'area' as const
      },
      reference: view('float32', cellCount),
      moving: view('float32', cellCount),
      stableMask: view('uint32', cellCount),
      differenceVariance: view('float32', cellCount),
      correlationLength,
      output: {
        fit: view('float32', 4),
        differences: view('float32', cellCount * 2),
        validity: view('uint32', cellCount),
        volume: view('float32', 3)
      }
    }
  };
}

it('bounds dense covariance terms per invocation and joins every partial in fixed order', () => {
  const {graph, props} = setup(1025, 16);
  const nodes = new GPUTerrainDEMCoregistration(props).getCommandNodes(graph);
  expect(nodes.map(node => node.id)).toContain('terrain-dem-coregistration-stable-covariance-rows');
  expect(nodes.map(node => node.id)).toContain(
    'terrain-dem-coregistration-measurement-covariance-rows'
  );
  const sources = getParsedKernelSources(nodes);
  const covariance = [...sources].find(([name]) =>
    name.includes('terrain-dem-coregistration-covariance-tiles')
  )?.[1];
  expect(covariance).toContain('local < 256u');
  expect(covariance).toContain('let first = index / 5u');
  expect(covariance).not.toContain('second < 1025u; second++');
  for (const [name, source] of sources) {
    expect(getStorageBindingCount(source), name).toBeLessThanOrEqual(8);
    expect(auditWGSLModule(source), name).toEqual([]);
  }
});

it('keeps independent covariance linear above the correlated-grid cap and rejects oversized correlated grids', () => {
  const independent = setup(100000, 0);
  const nodes = new GPUTerrainDEMCoregistration(independent.props).getCommandNodes(
    independent.graph
  );
  expect(nodes.map(node => node.id)).not.toContain('terrain-dem-coregistration-covariance-offsets');
  const oversized = setup(100000, 1);
  expect(() => new GPUTerrainDEMCoregistration(oversized.props)).toThrow(
    'terrain-dem-coregistration grid cell count with correlationLength > 0 must be at most 92672'
  );
  const overflow = setup(2000000, 1);
  expect(() => new GPUTerrainDEMCoregistration(overflow.props)).toThrow(
    'terrain-dem-coregistration grid cell count with correlationLength > 0 must be at most 92672'
  );
});

it('refuses the first correlated grid above the portable binding maximum at construction', () => {
  // 92,672 cells use 362 tiles; the next cell adds a whole tile to every covariance row.
  const maximumCellCount = 92672;
  expect(maximumCellCount * Math.ceil(maximumCellCount / 256) * 4).toBeLessThanOrEqual(134217728);
  expect((maximumCellCount + 1) * Math.ceil((maximumCellCount + 1) / 256) * 4).toBeGreaterThan(
    134217728
  );
  const oversized = setup(maximumCellCount + 1, 2);
  expect(
    () => new GPUTerrainDEMCoregistration({...oversized.props, id: 'bounded-coregistration'})
  ).toThrow(
    'bounded-coregistration grid cell count with correlationLength > 0 must be at most 92672'
  );
  const boundary = setup(maximumCellCount, 2);
  expect(() =>
    new GPUTerrainDEMCoregistration(boundary.props).getCommandNodes(boundary.graph)
  ).not.toThrow();
});
