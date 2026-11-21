// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {loadGuideFunction, readGuide} from './family-guide-execution.test-utils';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import type {Device} from '@luma.gl/core';
import type {
  GPUCellPyramidExecutionPlan,
  GPUCellCompactionInput,
  GPUCellSetOutlineInput,
  GPUCountedCellSource,
  GPUCellCompaction,
  GPUCellSetOutline
} from '../../../src/gpu-spatial-analysis';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';

it('W5-06 explains column-dependent feasibility, limits and retained-input sizing', () => {
  const guide = readGuide('operators-cells');
  for (const text of [
    'levelColumns',
    'stageRequirements',
    '256',
    '2,147,483,648',
    '1,073,741,824',
    '32,768',
    '4096 input edges'
  ]) {
    expect(guide).toContain(text);
  }
  const planAllocation = loadGuideFunction<
    (limits: Device['limits']) => GPUCellPyramidExecutionPlan
  >('operators-cells', 'planCellAllocationExample');
  const device = createNullWebGPUDevice();
  try {
    const plan = planAllocation(device.limits);
    expect(plan.feasible).toBe(true);
    expect(plan.route).toBe('tiled');
    expect(plan.stageRequirements.every(stage => stage.workgroupSize === 256)).toBe(true);
    expect(planAllocation({...device.limits, maxComputeWorkgroupSizeX: 128}).feasible).toBe(false);
    expect(planAllocation({...device.limits, maxStorageBuffersPerShaderStage: 6}).feasible).toBe(
      false
    );
  } finally {
    device.destroy();
  }
});

it('B4 executes the printed counted sort/dedup-to-outline declaration with loss evidence intact', () => {
  const declare = loadGuideFunction<
    (
      graph: GPUCommandGraph,
      inputs: {
        family: GPUCellCompactionInput['family'];
        resolution: number;
        cover: GPUCountedCellSource;
        sorted: GPUCellCompactionInput['output'];
        outline: GPUCellSetOutlineInput['output'];
      }
    ) => {compaction: GPUCellCompaction; boundary: GPUCellSetOutline}
  >('operators-cells', 'addCountedCoverOutline');
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const scalar = () => createTransientView(graph, `scalar-${serial++}`, 'uint32', 1);
    const cover = {
      cells: createTransientView(graph, 'cover', 'uint32x2', 4),
      status: {count: scalar(), overflow: scalar(), candidateOverflow: scalar(), complete: scalar()}
    };
    const sorted = {
      cells: createTransientView(graph, 'sorted', 'uint32x2', 4),
      status: {count: scalar(), overflow: scalar(), candidateOverflow: scalar()}
    };
    const outline = {
      cells: createTransientView(graph, 'outline', 'uint32x2', 16),
      rows: createTransientView(graph, 'rows', 'uint32', 16),
      edgeIndices: createTransientView(graph, 'edges', 'uint32', 16),
      endpoints: createTransientView(graph, 'endpoints', 'float32x4', 16),
      status: {count: scalar(), overflow: scalar(), candidateOverflow: scalar()}
    };
    const result = declare(graph, {family: 'quadbin', resolution: 2, cover, sorted, outline});
    expect(result.compaction.props.operation).toMatchObject({
      type: 'compact',
      resolution: 2,
      minimumResolution: 2
    });
    expect(result.compaction.props.source?.status.complete).toBe(cover.status.complete);
    expect(result.boundary.props.cells).toBe(sorted.cells);
    expect(result.boundary.props.count).toBe(sorted.status.count);
    expect(result.boundary.props.source?.status.candidateOverflow).toBe(
      sorted.status.candidateOverflow
    );
  } finally {
    device.destroy();
  }
});

it('EGR-12 runs the actual public signed-sum readback example independently of completeness', () => {
  const read = loadGuideFunction<
    (status?: {
      count: Uint32Array;
      overflow: Uint32Array;
      candidateOverflow: Uint32Array;
      sumOverflow: Uint32Array;
    }) => {
      scaledInteger: bigint;
      value: number;
      cellsComplete: boolean;
      sumSaturated: boolean;
      fixedPointSumUsable: boolean;
    }
  >('operators-cells', 'readCellSumExample');
  // Row one has signed two's complement -80 = 0xffffffffffffffb0; -80 / 16 = -5.
  expect(read()).toEqual({
    scaledInteger: -80n,
    value: -5,
    cellsComplete: true,
    sumSaturated: true,
    fixedPointSumUsable: false
  });
  for (const saturated of [false, true]) {
    expect(
      read({
        count: Uint32Array.of(2),
        overflow: Uint32Array.of(1),
        candidateOverflow: Uint32Array.of(0),
        sumOverflow: Uint32Array.of(Number(saturated))
      })
    ).toEqual({
      scaledInteger: -80n,
      value: -5,
      cellsComplete: false,
      sumSaturated: saturated,
      fixedPointSumUsable: false
    });
  }
});

it('B4/B5 names the composition types, distinct-key preparation and the omitted outline tile', () => {
  const guide = readGuide('operators-cells');
  expect(guide).toContain('GPUCellSetOutlineInput');
  expect(guide).toContain('GPUCellCompactionInput');
  expect(guide).toContain('minimumResolution: resolution');
  expect(guide).toContain('omit tile `(0,0)`');
  expect(guide).toContain('omit tile `(0,1)`');
  // Independent four-neighbor enumeration on the wrapped 4x4 tile grid.
  const exposedEdges = (missing?: readonly [number, number]) => {
    let edges = 0;
    const present = (column: number, row: number) =>
      row >= 0 && row < 4 && (!missing || (column + 4) % 4 !== missing[0] || row !== missing[1]);
    for (let row = 0; row < 4; row++)
      for (let column = 0; column < 4; column++)
        if (present(column, row))
          for (const [deltaColumn, deltaRow] of [
            [-1, 0],
            [1, 0],
            [0, -1],
            [0, 1]
          ])
            if (!present(column + deltaColumn, row + deltaRow)) edges++;
    return edges;
  };
  expect([exposedEdges(), exposedEdges([0, 0]), exposedEdges([0, 1])]).toEqual([8, 10, 12]);
});
