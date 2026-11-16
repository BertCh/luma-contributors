// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {describe, expect, it, vi} from 'vitest';
import {GPUCalendarBuckets} from '../../src/dataframe-analysis/calendar-buckets/index';
import {GPUTemporalReduction} from '../../src/dataframe-analysis/temporal-reduction/index';
import {GPUCellAggregation} from '../../src/gpu-spatial-analysis/cell-aggregation/index';
import {GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH} from '../../src/gpu-spatial-analysis/emerging-hot-spots/index';
import {addSpaceTimeHotSpotsRecipe} from '../../src/gpu-spatial-analysis/recipes/space-time-hot-spots-recipe';
import {GPUParameterBuffer} from '../../src/analysis-core/kernels/gpu-contributor-utils';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';

// PLAT-02: a row-proportional binding larger than `maxStorageBufferBindingSize` fails while the
// graph is built, naming the contributor and the binding, instead of at dispatch. The default
// device holds 128 MiB, which is 16,777,216 rows of 8-byte Int64 words or cell keys.
// GPUEmergingHotSpots already refuses an oversized cube with a named limit and is covered by
// emerging-hot-spots.node.spec.ts ('refuses a cube larger than one storage binding').

const LIMIT_BYTES = 128 * 1024 * 1024;
const MAXIMUM_EIGHT_BYTE_ROWS = LIMIT_BYTES / 8;

let serial = 0;

function createContext() {
  const device = createNullWebGPUDevice();
  expect(device.limits.maxStorageBufferBindingSize).toBe(LIMIT_BYTES);
  const graph = new GPUCommandGraph(device);
  const view = <Format extends 'uint32' | 'sint32' | 'float32' | 'float32x2' | 'uint32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `plat02-${serial++}`, format, length);
  return {device, graph, view};
}

type Context = ReturnType<typeof createContext>;

function createCalendarBuckets({graph, view}: Context, rows: number) {
  return new GPUCalendarBuckets({
    timestamps: view('uint32x2', rows),
    parameters: view('sint32', 2),
    output: {hour: view('uint32', rows)}
  });
}

function createTemporalReduction({graph, view}: Context, rows: number) {
  return new GPUTemporalReduction({
    cellIds: view('uint32', rows),
    timestamps: view('uint32x2', rows),
    values: view('float32', rows),
    parameters: view('uint32', 4),
    cellCount: 3,
    bucketCount: 4,
    output: {
      counts: view('uint32', 12),
      minimums: view('float32', 12),
      maximums: view('float32', 12),
      first: view('float32', 12),
      last: view('float32', 12),
      occupiedSlots: {
        ids: view('uint32', 12),
        status: {count: view('uint32', 1), overflow: view('uint32', 1)}
      }
    }
  });
}

function createCellAggregation({view}: Context, rows: number) {
  const capacity = 16;
  return new GPUCellAggregation({
    family: 'quadbin',
    resolution: 8,
    positions: view('float32x2', rows),
    output: {
      cells: view('uint32x2', capacity),
      counts: view('uint32', capacity),
      status: {count: view('uint32', 1), overflow: view('uint32', 1)}
    }
  });
}

function addRecipe(context: Context, eventCount: number, width: number, height: number) {
  const {device, graph, view} = context;
  const key = `${serial++}`;
  return addSpaceTimeHotSpotsRecipe(graph, {
    id: `plat02-recipe-${key}`,
    timestamps: view('uint32x2', eventCount),
    calendarParameters: view('sint32', 2),
    slices: {step: 'month', origin: {year: 2023, month: 1}, count: 9},
    cells: {
      kind: 'lattice',
      positions: view('float32x2', eventCount),
      width,
      height,
      bounds: [0, 0, width, height]
    },
    settings: {temporalWindow: 1},
    parameters: new GPUParameterBuffer(device, {
      id: `plat02-parameters-${key}`,
      format: 'float32',
      length: GPU_EMERGING_HOT_SPOTS_PARAMETER_LENGTH
    })
  } as never);
}

describe('single-view temporal and cell contributors check the storage binding size', () => {
  it('GPUCalendarBuckets rejects Int64 timestamps beyond one binding and accepts the limit', () => {
    const context = createContext();
    const accepted = createCalendarBuckets(context, MAXIMUM_EIGHT_BYTE_ROWS);
    expect(() => accepted.getCommandNodes(context.graph)).not.toThrow();
    const oversized = createCalendarBuckets(context, MAXIMUM_EIGHT_BYTE_ROWS + 1);
    expect(() => oversized.getCommandNodes(context.graph)).toThrow(
      /^calendar-buckets timestamps chunk 0 needs 134217736 bytes in one binding \(limit 134217728\)/
    );
    context.device.destroy();
  });

  it('GPUTemporalReduction rejects word timestamps beyond one binding', () => {
    const context = createContext();
    const oversized = createTemporalReduction(context, MAXIMUM_EIGHT_BYTE_ROWS + 1);
    const allocate = vi.spyOn(context.graph, 'createTransientBuffer');
    const declareView = vi.spyOn(context.graph, 'createDataView');
    expect(() => oversized.getCommandNodes(context.graph)).toThrow(
      /^temporal-reduction timestamps chunk 0 needs 134217736 bytes in one binding \(limit 134217728\)/
    );
    expect(allocate).not.toHaveBeenCalled();
    expect(declareView).not.toHaveBeenCalled();
    allocate.mockRestore();
    declareView.mockRestore();
    expect(() =>
      createTemporalReduction(context, MAXIMUM_EIGHT_BYTE_ROWS).getCommandNodes(context.graph)
    ).not.toThrow();
    context.device.destroy();
  });

  it('GPUCellAggregation rejects positions beyond one binding', () => {
    const context = createContext();
    const oversized = createCellAggregation(context, MAXIMUM_EIGHT_BYTE_ROWS + 1);
    const allocate = vi.spyOn(context.graph, 'createTransientBuffer');
    expect(() => oversized.getCommandNodes(context.graph)).toThrow(
      /^cell-aggregation input\/output field 0 chunk 0 needs 134217736 bytes in one binding \(limit 134217728\)/
    );
    expect(allocate).not.toHaveBeenCalled();
    allocate.mockRestore();
    expect(() =>
      createCellAggregation(context, MAXIMUM_EIGHT_BYTE_ROWS).getCommandNodes(context.graph)
    ).not.toThrow();
    context.device.destroy();
  });

  it('addSpaceTimeHotSpotsRecipe rejects Int64 event times and a cube beyond one binding', () => {
    const context = createContext();
    expect(() => addRecipe(context, MAXIMUM_EIGHT_BYTE_ROWS + 1, 2, 2)).toThrow(
      /^plat02-recipe-\d+ timestamps needs 134217736 bytes in one binding \(limit 134217728\)/
    );
    expect(() => addRecipe(context, 4, 2048, 2048)).toThrow(
      /^plat02-recipe-\d+ cube needs 150994944 bytes in one binding \(limit 134217728\)/
    );
    // Rejection happens before the recipe adds anything to the graph.
    context.device.destroy();
  });
});
