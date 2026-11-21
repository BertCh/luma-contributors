// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  GPUCellAggregation,
  GPUCellPyramid
} from '../../../src/gpu-spatial-analysis/cell-aggregation';
import {
  createNullWebGPUDevice,
  createVectorView,
  getGPUCommandNodeKernelSources,
  getStorageBindingCount
} from '../../utils/gpu-contributor-test-utils';

it('exposes the same auto crossover as construction without allocating the workload', () => {
  const device = createNullWebGPUDevice();
  Object.assign(device.limits, {maxStorageBufferBindingSize: 4096, maxBufferSize: 65536});
  try {
    expect(GPUCellPyramid.limits.length).toBeGreaterThan(0);
    // 4 bytes of scalar sorting scratch per row: 1024 rows exactly fill 4096
    // bytes; 1025 require tiled scratch. Default tile width = floorPowerOfTwo(4096/8)=512.
    for (const rowCount of [1024, 1025]) {
      const shape = {
        rowCount,
        inputChunkRowCounts: rowCount === 1024 ? [512, 512] : [512, 512, 1],
        levelCapacities: [8, 4],
        includeLevelCounts: true
      };
      const plan = GPUCellPyramid.getExecutionPlan(shape, device.limits);
      expect(plan.route).toBe(rowCount === 1024 ? 'packed' : 'tiled');
      expect(plan.scratchTileSize).toBe(512);
      expect(plan.feasible).toBe(true);
      expect(plan.constraints.every(constraint => constraint.satisfied)).toBe(true);
      expect(
        plan.bindingRequirements.find(binding => binding.name === 'aggregation row scratch')
          ?.byteLength
      ).toBe(rowCount === 1024 ? 4096 : 2048);
      expect(plan.commands.aggregationUpperBound).toBe(rowCount === 1024 ? 515 : 2478);
      // Transient declarations are shape metadata on the null device, not GPU allocations.
      const graph = new GPUCommandGraph(device);
      let serial = 0;
      const scalar = () => createTransientView(graph, `scalar-${serial++}`, 'uint32', 1);
      const cells = createVectorView(
        'input-cells',
        'uint32x2',
        shape.inputChunkRowCounts.map((length, index) =>
          createTransientView(graph, `input-${index}`, 'uint32x2', length)
        )
      );
      const pyramid = new GPUCellPyramid({
        family: 'quadbin',
        cells,
        resolutions: [2, 1],
        output: {
          levels: [8, 4].map((capacity, index) => ({
            cells: createTransientView(graph, `level-${index}-cells`, 'uint32x2', capacity),
            counts: createTransientView(graph, `level-${index}-counts`, 'uint32', capacity),
            status: {count: scalar(), overflow: scalar(), candidateOverflow: scalar()}
          }))
        }
      });
      const aggregation = new GPUCellAggregation({
        family: 'quadbin',
        resolution: 2,
        cells,
        output: pyramid.props.output.levels[0]
      });
      expect(aggregation.usesTiledScratch).toBe(plan.route === 'tiled');
      expect(aggregation.scratchTileSize).toBe(plan.scratchTileSize);
      expect(aggregation.estimatedCommandNodeCount).toBe(plan.commands.aggregationUpperBound);
      expect(pyramid.executionPlan.route).toBe(plan.route);
      expect(pyramid.executionPlan.scratchTileSize).toBe(plan.scratchTileSize);
    }
  } finally {
    device.destroy();
  }
});

it('reports numeric failure and a valid exact alternative for forced packed scratch', () => {
  const device = createNullWebGPUDevice();
  const limits = {...device.limits, maxStorageBufferBindingSize: 4096, maxBufferSize: 65536};
  try {
    const plan = GPUCellPyramid.getExecutionPlan(
      {
        rowCount: 1025,
        inputChunkRowCounts: [512, 512, 1],
        levelCapacities: [8],
        scratchMode: 'packed'
      },
      limits
    );
    expect(plan.feasible).toBe(false);
    expect(
      plan.constraints.find(constraint => constraint.name === 'aggregation row scratch')?.received
    ).toBe(4100);
    expect(plan.validAlternative).toMatchObject({
      route: 'tiled',
      scratchMode: 'auto',
      scratchTileSize: 512
    });
    expect(plan.accuracyImplications).toEqual([
      'Exact integer cell keys and fixed-point sums on both scratch layouts.'
    ]);
    expect(() =>
      GPUCellPyramid.getExecutionPlan(
        {rowCount: 4, levelCapacities: [4], scratchMode: 'bogus' as 'auto'},
        limits
      )
    ).toThrow(/scratchMode.*auto.*packed/);
  } finally {
    device.destroy();
  }
});

it('uses explicit tile thresholds and reports a packed alternative to a command-limited tile plan', () => {
  const device = createNullWebGPUDevice();
  try {
    for (const rowCount of [32, 33]) {
      const plan = GPUCellPyramid.getExecutionPlan(
        {rowCount, levelCapacities: [8], scratchTileSize: 32},
        device.limits
      );
      expect(plan.route).toBe(rowCount === 32 ? 'packed' : 'tiled');
      expect(plan.scratchTileSize).toBe(32);
    }
    const plan = GPUCellPyramid.getExecutionPlan(
      {rowCount: 8000, levelCapacities: [8], scratchTileSize: 32},
      device.limits
    );
    // 250 tiles: 800*250 + 6*250^2 + 8*1 = 575008 > 32768 commands.
    expect(plan.commands.aggregationUpperBound).toBe(575008);
    expect(plan.feasible).toBe(false);
    expect(plan.validAlternative).toMatchObject({route: 'packed', scratchMode: 'packed'});
  } finally {
    device.destroy();
  }
});

it.each([
  false,
  true
])('B4 checks the binding and workgroup peaks of lowered packed stages (all columns %s)', allColumns => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  const scalar = () => createTransientView(graph, `peak-${serial++}`, 'uint32', 1);
  const pyramid = new GPUCellPyramid({
    family: 'quadbin',
    resolutions: [2],
    cells: createTransientView(graph, 'peak-cells', 'uint32x2', 257),
    values: createTransientView(graph, 'peak-values', 'float32', 257),
    output: {
      levels: [
        {
          cells: createTransientView(graph, 'peak-output', 'uint32x2', 8),
          counts: createTransientView(graph, 'peak-counts', 'uint32', 8),
          sums: createTransientView(graph, 'peak-sums', 'uint32x2', 8),
          ...(allColumns
            ? {
                sumValues: createTransientView(graph, 'peak-sum-values', 'float32', 8),
                minimums: createTransientView(graph, 'peak-minimums', 'float32', 8),
                maximums: createTransientView(graph, 'peak-maximums', 'float32', 8)
              }
            : {}),
          status: {count: scalar(), overflow: scalar()}
        }
      ]
    }
  });
  try {
    const kernels = getGPUCommandNodeKernelSources(pyramid.getCommandNodes(graph));
    const peakBindings = Math.max(...kernels.map(kernel => getStorageBindingCount(kernel.source)));
    // The offsets stage binds six columns plus the cell parameter buffer; cells
    // and sum accumulation reach seven as well. Cell workgroups have 256 lanes.
    expect(peakBindings).toBe(allColumns ? 8 : 7);
    expect(kernels.some(kernel => /@workgroup_size\(256\)/.test(kernel.source))).toBe(true);
    const shape = {
      rowCount: 257,
      levelCapacities: [8],
      hasValues: true,
      levelColumns: [
        {sums: true, sumValues: allColumns, minimums: allColumns, maximums: allColumns}
      ]
    };
    const matchingPlan = GPUCellPyramid.getExecutionPlan(shape, {
      ...device.limits,
      maxStorageBuffersPerShaderStage: peakBindings
    });
    expect(matchingPlan.feasible).toBe(true);
    expect(
      Math.max(...matchingPlan.stageRequirements.map(stage => stage.storageBindingCount))
    ).toBe(peakBindings);
    expect(
      GPUCellPyramid.getExecutionPlan(shape, {...device.limits, maxStorageBuffersPerShaderStage: 5})
        .feasible
    ).toBe(false);
    expect(
      GPUCellPyramid.getExecutionPlan(shape, {
        ...device.limits,
        maxStorageBuffersPerShaderStage: peakBindings - 1
      }).feasible
    ).toBe(false);
    expect(
      GPUCellPyramid.getExecutionPlan(shape, {
        ...device.limits,
        maxComputeInvocationsPerWorkgroup: 128,
        maxComputeWorkgroupSizeX: 128
      }).feasible
    ).toBe(false);
  } finally {
    device.destroy();
  }
});

it('B5 finds a device-derived tile width independently of the rejected width', () => {
  const device = createNullWebGPUDevice();
  const limits = {...device.limits, maxStorageBufferBindingSize: 4096, maxBufferSize: 65536};
  const shape = {
    rowCount: 1200,
    inputChunkRowCounts: [400, 400, 400],
    levelCapacities: [8],
    scratchMode: 'packed' as const,
    scratchTileSize: 1
  };
  try {
    const plan = GPUCellPyramid.getExecutionPlan(shape, limits);
    expect(plan.feasible).toBe(false);
    expect(plan.validAlternative).toBeDefined();
    const alternative = plan.validAlternative!;
    expect(alternative.route).toBe('tiled');
    expect(alternative.scratchTileSize).toBeGreaterThan(1);
    expect(
      GPUCellPyramid.getExecutionPlan(
        {
          ...shape,
          scratchMode: alternative.scratchMode,
          scratchTileSize: alternative.scratchTileSize
        },
        limits
      ).feasible
    ).toBe(true);
    Object.assign(device.limits, limits);
    const graph = new GPUCommandGraph(device);
    const cells = createVectorView(
      'alternative-input',
      'uint32x2',
      shape.inputChunkRowCounts.map((length, index) =>
        createTransientView(graph, `alternative-chunk-${index}`, 'uint32x2', length)
      )
    );
    const pyramid = new GPUCellPyramid({
      family: 'quadbin',
      cells,
      resolutions: [2],
      scratchMode: alternative.scratchMode,
      scratchTileSize: alternative.scratchTileSize,
      output: {
        levels: [
          {
            cells: createTransientView(graph, 'alternative-cells', 'uint32x2', 8),
            counts: createTransientView(graph, 'alternative-counts', 'uint32', 8),
            status: {
              count: createTransientView(graph, 'alternative-count', 'uint32', 1),
              overflow: createTransientView(graph, 'alternative-overflow', 'uint32', 1)
            }
          }
        ]
      }
    });
    expect(() => pyramid.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});
