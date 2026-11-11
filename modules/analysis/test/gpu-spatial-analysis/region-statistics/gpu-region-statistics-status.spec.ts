// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, GPUGridIndex} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  decodeGPURegionStatistics,
  getGPURegionStatisticsSummaryLength,
  GPURegionStatistics,
  type GPURegionStatisticsResult
} from '../../../src/gpu-spatial-analysis/region-statistics/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {LASSO, POSITIONS, RECTANGLE, VALUES} from './region-statistics-fixtures';
import {requireTestDevice} from '../../utils/require-test-device';

const ROW_COUNT = POSITIONS.length / 2;
const SUMMARY_LENGTH = getGPURegionStatisticsSummaryLength(0);

type Fixture = {
  graph: GPUCommandGraph;
  positions: ReturnType<typeof importGraphBuffer<'float32x2', void>>;
  values: ReturnType<typeof importGraphBuffer<'float32', void>>;
  output: (name: string, length: number) => {buffer: Buffer; view: ReturnType<typeof view>};
  destroy: () => void;
};

function view(graph: GPUCommandGraph, name: string, buffer: Buffer, length: number) {
  return importGraphBuffer(graph, name, buffer, 'uint32', length);
}

function createFixture(device: Device, id: string): Fixture {
  const graph = new GPUCommandGraph(device, {id});
  const positionsBuffer = createInputBuffer(device, POSITIONS);
  const valuesBuffer = createInputBuffer(device, VALUES);
  const buffers: Buffer[] = [positionsBuffer, valuesBuffer];
  return {
    graph,
    positions: importGraphBuffer(graph, 'positions', positionsBuffer, 'float32x2', ROW_COUNT),
    values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', ROW_COUNT),
    output: (name, length) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: view(graph, name, buffer, length)};
    },
    destroy: () => {
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

type StatusReadback = {
  summary: GPURegionStatisticsResult;
  overflow: number;
  candidateOverflow: number;
};

async function readStatus(
  summary: Buffer,
  overflow: Buffer,
  candidateOverflow: Buffer
): Promise<StatusReadback> {
  const result = {
    summary: decodeGPURegionStatistics(await summary.readAsync(0, SUMMARY_LENGTH * 4)),
    overflow: (await readUint32(overflow, 1))[0],
    candidateOverflow: (await readUint32(candidateOverflow, 1))[0]
  };
  // The named ports always agree with the header flags, which keep their meaning.
  expect(result.overflow).toBe(
    Number(result.summary.selectionTruncated || result.summary.regionTruncated)
  );
  expect(result.candidateOverflow).toBe(Number(result.summary.candidatesTruncated));
  return result;
}

it('GPURegionStatistics publishes region and selection truncation as status.overflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const selectionCapacity of [ROW_COUNT, 1]) {
    const fixture = createFixture(device, `region-status-${selectionCapacity}`);
    const lasso = createInputBuffer(device, LASSO);
    const vertexCount = new GPUParameterBuffer(device, {
      id: 'vertex-count',
      format: 'uint32',
      length: 1,
      values: Uint32Array.of(5)
    });
    const summary = fixture.output('summary', SUMMARY_LENGTH);
    const overflow = fixture.output('overflow', 1);
    const candidateOverflow = fixture.output('candidate-overflow', 1);
    fixture.graph.add(
      new GPURegionStatistics({
        region: {
          kind: 'polygon',
          vertices: importGraphBuffer(fixture.graph, 'lasso', lasso, 'float32x2', 5),
          vertexCount: vertexCount.importToGraph(fixture.graph)
        },
        positions: fixture.positions,
        values: fixture.values,
        output: {
          summary: summary.view,
          selectedRows: {
            ids: fixture.output('ids', selectionCapacity).view,
            status: {
              count: fixture.output('count', 1).view,
              overflow: fixture.output('selection-overflow', 1).view
            }
          },
          status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
        }
      })
    );
    const compiled = fixture.graph.compile();
    const selectionTruncated = selectionCapacity < ROW_COUNT;

    submitGraph(device, compiled, undefined);
    let status = await readStatus(summary.buffer, overflow.buffer, candidateOverflow.buffer);
    expect(status.summary.regionTruncated).toBe(false);
    expect(status.summary.selectionTruncated).toBe(selectionTruncated);
    expect(status.overflow).toBe(selectionTruncated ? 1 : 0);
    expect(status.candidateOverflow).toBe(0);

    // A vertex count above the lasso capacity truncates the region.
    vertexCount.write(Uint32Array.of(99));
    submitGraph(device, compiled, undefined);
    status = await readStatus(summary.buffer, overflow.buffer, candidateOverflow.buffer);
    expect(status.summary.regionTruncated).toBe(true);
    expect(status.overflow).toBe(1);
    expect(status.candidateOverflow).toBe(0);

    compiled.destroy();
    vertexCount.destroy();
    lasso.destroy();
    fixture.destroy();
  }
});

it('GPURegionStatistics publishes grid candidate truncation as status.candidateOverflow', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const gridSize = [4, 4] as const;
  for (const candidateCapacity of [ROW_COUNT, 2]) {
    const fixture = createFixture(device, `region-grid-status-${candidateCapacity}`);
    const bounds = new GPUParameterBuffer(device, {
      id: 'bounds',
      format: 'float32',
      length: 4,
      values: RECTANGLE
    });
    const index = new GPUGridIndex({
      id: 'index',
      positions: fixture.positions,
      gridSize,
      bounds: [0, 0, 10, 10],
      cellOffsets: fixture.output('cell-offsets', gridSize[0] * gridSize[1] + 1).view,
      objectIds: fixture.output('object-ids', ROW_COUNT).view,
      count: fixture.output('index-count', 1).view,
      overflow: fixture.output('index-overflow', 1).view
    });
    fixture.graph.add(index);
    const summary = fixture.output('summary', SUMMARY_LENGTH);
    const overflow = fixture.output('overflow', 1);
    const candidateOverflow = fixture.output('candidate-overflow', 1);
    fixture.graph.add(
      new GPURegionStatistics({
        region: {kind: 'rectangle', bounds: bounds.importToGraph(fixture.graph)},
        positions: fixture.positions,
        values: fixture.values,
        spatialIndex: {kind: 'grid', index, candidateCapacity},
        output: {
          summary: summary.view,
          status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
        }
      })
    );
    const compiled = fixture.graph.compile();
    submitGraph(device, compiled, undefined);
    const status = await readStatus(summary.buffer, overflow.buffer, candidateOverflow.buffer);
    const truncated = candidateCapacity < ROW_COUNT;
    expect(status.summary.candidatesTruncated).toBe(truncated);
    expect(status.candidateOverflow).toBe(truncated ? 1 : 0);
    // A candidate overflow is not folded into overflow.
    expect(status.overflow).toBe(0);
    compiled.destroy();
    bounds.destroy();
    fixture.destroy();
  }
});
