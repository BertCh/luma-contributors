// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createRequire} from 'node:module';
import {expect, it} from 'vitest';
import {GraphVectorView, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../utils/gpu-contributor-test-utils';
import {getTerrainBindingViews} from '../../src/gpu-terrain/terrain-processing/terrain-binding-preflight';
import {getNamedRasterOutputViews} from '../../src/raster-analysis/raster-algebra/raster-algebra-utils';
import {validateRasterFractionalWritableViews} from '../../src/raster-analysis/fractional-coverage/raster-fractional-tile-kernel';
import {validateRasterWritableDispatches} from '../../src/raster-analysis/raster-grid/raster-view-validation';
import {
  planRasterTimeSeriesChunks,
  validateRasterTimeSeriesViews
} from '../../src/raster-analysis/raster-time-series/raster-time-series-chunks-b';
import {planRasterTimeSeriesChunks as planFirst} from '../../src/raster-analysis/raster-time-series/raster-time-series-chunks';

const foreign = createRequire(import.meta.url)(
  '@luma.gl/graph/gpu-core'
) as typeof import('@luma.gl/graph/gpu-core');

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new foreign.GPUCommandGraph(device);
  const storage = foreign.createTransientView(graph, 'foreign', 'float32', 128);
  const view = graph.createDataView(storage.buffer, {format: 'float32', length: 8});
  const vector = (chunks: GraphDataView<'float32'>[]) =>
    new foreign.GraphVectorView({
      id: 'foreign-vector',
      name: 'foreign-vector',
      format: 'float32',
      length: chunks.reduce((sum, chunk) => sum + chunk.length, 0),
      valueLength: chunks.reduce((sum, chunk) => sum + chunk.length, 0),
      stride: 1,
      byteStride: 4,
      rowByteLength: 4,
      data: chunks
    });
  return {device, graph, view, vector};
}

it('discovers foreign data and vectors without descending into their resources', () => {
  const {device, view, vector} = createFixture();
  try {
    expect(foreign.GraphVectorView).not.toBe(GraphVectorView);
    const chunks = vector([view]);
    expect(getTerrainBindingViews({nested: {chunks}})).toEqual({'nested.chunks': chunks});
    expect(getNamedRasterOutputViews({nested: {view}})).toEqual({'output.nested.view': view});
    const empty = vector([]);
    expect(getTerrainBindingViews({empty})).toEqual({empty});
  } finally {
    device.destroy();
  }
});

it('detects writable foreign aliases, including fractional status destinations', () => {
  const {device, view} = createFixture();
  try {
    expect(() =>
      validateRasterFractionalWritableViews(
        'foreign-alias',
        {values: view, status: {count: view}},
        []
      )
    ).toThrow(/overlap|distinct|share/);
  } finally {
    device.destroy();
  }
});

it('plans foreign complete records, rejects split records and packed-only shared tables', () => {
  const {device, graph, view, vector} = createFixture();
  try {
    const chunks = vector([
      graph.createDataView(view.buffer, {format: 'float32', length: 4}),
      graph.createDataView(view.buffer, {format: 'float32', byteOffset: 16, length: 4})
    ]);
    const plan = planRasterTimeSeriesChunks('foreign-plan', 2, {
      values: {view: chunks, entriesPerCell: 4}
    });
    expect(plan.chunked).toBe(true);
    expect(plan.chunks.map(chunk => chunk.cellCount)).toEqual([1, 1]);
    const split = vector([
      graph.createDataView(view.buffer, {format: 'float32', length: 3}),
      graph.createDataView(view.buffer, {format: 'float32', byteOffset: 12, length: 5})
    ]);
    expect(() =>
      planRasterTimeSeriesChunks('foreign-plan', 2, {values: {view: split, entriesPerCell: 4}})
    ).toThrow(/split cell record/);
    expect(() =>
      planFirst('foreign-plan', {cellCount: 2, values: chunks, parameters: chunks}, {values: 4})
    ).toThrow(/one packed|single packed|GraphDataView|vector/);
    const packedPlan = planRasterTimeSeriesChunks('packed', 2, {values: {view, entriesPerCell: 4}});
    expect(() =>
      validateRasterTimeSeriesViews('foreign-shared', [view, chunks], [], packedPlan)
    ).toThrow(/shared tables must be one packed view/);
    const empty = vector([]);
    const emptyPlan = planRasterTimeSeriesChunks('foreign-empty', 0, {
      values: {view: empty, entriesPerCell: 4}
    });
    expect(emptyPlan.chunks).toEqual([]);
    expect(() =>
      validateRasterTimeSeriesViews('foreign-empty', [empty], [], emptyPlan)
    ).not.toThrow();
    // An unrelated empty vector is still a shared table, which must remain packed.
    expect(() =>
      validateRasterTimeSeriesViews('foreign-shared-empty', [empty, vector([])], [], emptyPlan)
    ).toThrow(/shared tables/);
  } finally {
    device.destroy();
  }
});

it('compares only matching foreign chunk dispatches and ignores empty chunks', () => {
  const {device, graph, view, vector} = createFixture();
  try {
    const other = foreign.createTransientView(graph, 'other', 'float32', 128);
    const first = graph.createDataView(view.buffer, {format: 'float32', length: 1});
    const second = graph.createDataView(other.buffer, {format: 'float32', length: 1});
    const firstOutput = graph.createDataView(other.buffer, {
      format: 'float32',
      byteOffset: 256,
      length: 1
    });
    const secondOutput = graph.createDataView(view.buffer, {
      format: 'float32',
      byteOffset: 256,
      length: 1
    });
    const empty = graph.createDataView(view.buffer, {format: 'float32', length: 0});
    // Buffer identities swap across dispatches, but coexisting read/write buffers differ.
    // All logical ranges remain disjoint, including across the two dispatches.
    const inputs = vector([first, second]);
    const outputs = vector([firstOutput, secondOutput]);
    expect(() =>
      validateRasterWritableDispatches(
        'foreign-matching',
        [inputs],
        [outputs],
        [{inputs: [inputs], outputs: [outputs]}]
      )
    ).not.toThrow();
    expect(() =>
      validateRasterWritableDispatches(
        'foreign-empty',
        [vector([empty])],
        [vector([empty])],
        [{inputs: [vector([empty])], outputs: [vector([empty])]}]
      )
    ).not.toThrow();
    expect(() => validateRasterWritableDispatches('foreign-alias', [inputs], [inputs])).toThrow(
      /overlap/
    );
  } finally {
    device.destroy();
  }
});

it('S2 reports all missing required ports before optional formats in either descriptor order', () => {
  const {device, graph} = createFixture();
  try {
    const mask = foreign.createTransientView(graph, 'wrong-mask', 'float32', 1);
    const required = {required: true, viewType: 'either', formats: ['float32']} as const;
    const optional = {required: false, viewType: 'either', formats: ['uint32']} as const;
    for (const ports of [
      {mask: optional, values: required},
      {values: required, mask: optional}
    ]) {
      expect(() =>
        planFirst('required-order', {cellCount: 1, mask}, {values: 1, mask: 1}, ports)
      ).toThrow(/required-order.*values.*received undefined/);
      expect(() =>
        planRasterTimeSeriesChunks(
          'required-order',
          1,
          {mask: {view: mask, entriesPerCell: 1}, values: {view: undefined, entriesPerCell: 1}},
          ports
        )
      ).toThrow(/required-order.*values.*received undefined/);
    }
  } finally {
    device.destroy();
  }
});
