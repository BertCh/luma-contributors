// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it, vi} from 'vitest';
import {
  GPUCrossVariogram,
  type GPUCrossVariogramProps
} from '../../../src/dataframe-analysis/pair-statistics/gpu-cross-variogram';
import {
  GPUMarkedPointStatistics,
  type GPUMarkedPointStatisticsProps
} from '../../../src/dataframe-analysis/pair-statistics/gpu-marked-point-statistics';
import {
  createNullWebGPUDevice,
  getParsedKernelSources
} from '../../utils/gpu-contributor-test-utils';

function createFixture(rows: number, bins: number) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'positions', 'float32x2', rows);
  const counts = createTransientView(graph, 'counts', 'uint32', bins);
  const status = createTransientView(graph, 'status', 'uint32', bins);
  const marked: GPUMarkedPointStatisticsProps = {
    id: 'marked',
    positions,
    parameters: createTransientView(graph, 'marked-parameters', 'float32', 6),
    radiusCount: bins,
    output: {
      statistics: createTransientView(graph, 'statistics', 'float32', 4 * bins),
      pairCounts: counts,
      status
    }
  };
  const cross: GPUCrossVariogramProps = {
    id: 'cross',
    positions,
    values: createTransientView(graph, 'values', 'float32', rows),
    parameters: createTransientView(graph, 'cross-parameters', 'float32', 1),
    lagCount: bins,
    output: {
      semivariances: createTransientView(graph, 'semivariances', 'float32', bins),
      pairCounts: counts,
      status
    }
  };
  return {device, graph, marked, cross};
}

it('both pair-statistics defaults accept more than 16,384 rows and build a cell index', () => {
  const {device, graph, marked, cross} = createFixture(65536, 4);
  try {
    const contributors = [new GPUMarkedPointStatistics(marked), new GPUCrossVariogram(cross)];
    for (const contributor of contributors) {
      const sources = getParsedKernelSources(contributor.getCommandNodes(graph));
      expect([...sources.keys()].some(name => name.includes('cell-keys'))).toBe(true);
      // Moment and finishing stages must not keep the former default's whole-row serial walks.
      for (const source of sources.values()) {
        expect(source).not.toMatch(/for\s*\([^;]*;[^;]*<\s*65536u\s*;/);
      }
    }
  } finally {
    device.destroy();
  }
});

it('the marked dense reference is explicit, bounded and incompatible with a gridSize', () => {
  const {device, graph, marked} = createFixture(8, 2);
  try {
    const reference = new GPUMarkedPointStatistics({...marked, searchStructure: 'brute-force'});
    const sources = getParsedKernelSources(reference.getCommandNodes(graph));
    expect([...sources.keys()].some(name => name.includes('cell-keys'))).toBe(false);
    expect(
      () => new GPUMarkedPointStatistics({...marked, searchStructure: 'unknown' as never})
    ).toThrow(/^marked .*searchStructure.*grid.*brute-force/);
    expect(
      () =>
        new GPUMarkedPointStatistics({...marked, searchStructure: 'brute-force', gridSize: [2, 2]})
    ).toThrow(/^marked .*gridSize.*grid/);
    const largePositions = createTransientView(graph, 'large-positions', 'float32x2', 16385);
    expect(
      () =>
        new GPUMarkedPointStatistics({
          ...marked,
          positions: largePositions,
          searchStructure: 'brute-force'
        })
    ).toThrow(/^marked positions length must be an integer in \[1, 16384\]/);
  } finally {
    device.destroy();
  }
});

it('the default marked index validates its partial-table budget at construction', () => {
  const {device, marked} = createFixture(2 ** 20, 128);
  try {
    expect(() => new GPUMarkedPointStatistics(marked)).toThrow(
      /^marked positions length x radiusCount x 3 must be at most 16777216/
    );
  } finally {
    device.destroy();
  }
});

it('the default cross index validates its partial-table budget at construction', () => {
  const {device, cross} = createFixture(2 ** 20, 128);
  try {
    expect(() => new GPUCrossVariogram(cross)).toThrow(
      /^cross positions length x bins must be at most 16777216/
    );
  } finally {
    device.destroy();
  }
});

it('marked partials are checked against the device before allocating intermediates', () => {
  const {device, graph, marked} = createFixture(2048, 4);
  const contributor = new GPUMarkedPointStatistics(marked);
  Object.assign(device.limits, {maxStorageBufferBindingSize: 65536});
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /^marked pair partials needs 98304 bytes in one binding \(limit 65536\)/
    );
    expect(allocations).not.toHaveBeenCalled();
  } finally {
    allocations.mockRestore();
    device.destroy();
  }
});

it('marked cell offsets are checked against the device before allocating intermediates', () => {
  const {device, graph, marked} = createFixture(8, 2);
  const contributor = new GPUMarkedPointStatistics({...marked, gridSize: [512, 512]});
  Object.assign(device.limits, {maxStorageBufferBindingSize: 2 ** 20});
  const allocations = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() => contributor.getCommandNodes(graph)).toThrow(
      /^marked cell offsets needs 1048580 bytes in one binding \(limit 1048576\)/
    );
    expect(allocations).not.toHaveBeenCalled();
  } finally {
    allocations.mockRestore();
    device.destroy();
  }
});
