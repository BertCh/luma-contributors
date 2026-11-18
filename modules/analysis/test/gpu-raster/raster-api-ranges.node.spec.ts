// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer} from '@luma.gl/core';
import {createTransientView, GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {validateRasterBuffersDistinct} from '../../src/raster-analysis/raster-band-utils';
import {validateRasterWritableDispatches} from '../../src/raster-analysis/raster-grid/raster-view-validation';
import {planRasterTimeSeriesChunks as planFirstChunks} from '../../src/raster-analysis/raster-time-series/raster-time-series-chunks';
import {planRasterTimeSeriesChunks as planSecondChunks} from '../../src/raster-analysis/raster-time-series/raster-time-series-chunks-b';
import {GPURasterHANTS} from '../../src/raster-analysis/raster-time-series/gpu-raster-hants';
import {GPURasterTemporalResample} from '../../src/raster-analysis/raster-time-series/gpu-raster-temporal-resample';
import {GPURasterReprojection} from '../../src/raster-analysis/gpu-raster-reprojection';
import {GPUTerrainStreamNetwork} from '../../src/gpu-terrain/hydrology/gpu-terrain-stream-network';
import {createNullWebGPUDevice, createVectorView} from '../utils/gpu-contributor-test-utils';

it('allows storage-aligned disjoint raster outputs and rejects physical read/write aliases', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const physical = device.createBuffer({byteLength: 1024, usage: Buffer.STORAGE});
  try {
    const backing = createTransientView(graph, 'backing', 'float32', 256);
    const view = (byteOffset: number) =>
      graph.createDataView(backing.buffer, {format: 'float32', byteOffset, length: 1});
    expect(() => validateRasterBuffersDistinct('ranges', [view(0), view(256)], [])).toThrow(
      /without explicit per-dispatch accesses/
    );
    const outputs = [view(0), view(256)];
    expect(() =>
      validateRasterWritableDispatches('ranges', [], outputs, [{inputs: [], outputs}])
    ).not.toThrow();
    expect(() => validateRasterBuffersDistinct('ranges', [view(0), view(4)], [])).toThrow();
    expect(() => validateRasterBuffersDistinct('ranges', [view(0), view(0)], [])).toThrow();
    expect(() => validateRasterBuffersDistinct('ranges', [view(256)], [view(0)])).toThrow(
      /physical buffer/
    );
    const importView = (id: string, byteOffset = 0) =>
      graph.createDataView(
        graph.importBuffer({id, byteLength: physical.byteLength, usage: physical.usage}, physical),
        {format: 'float32', length: 1, byteOffset}
      );
    expect(() =>
      validateRasterBuffersDistinct('aliases', [importView('write')], [importView('read')])
    ).toThrow(/^aliases .*logical byte ranges must not overlap/);
    expect(() =>
      validateRasterBuffersDistinct(
        'write-alias',
        [importView('disjoint-write', 256)],
        [importView('disjoint-read')]
      )
    ).toThrow(/^write-alias .*physical buffer.*write.*one imported handle/);
  } finally {
    physical.destroy();
    device.destroy();
  }
});

it('rejects pooled stream endpoints before a later dispatch can read one and write the other', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const view = <Format extends 'float32' | 'uint32' | 'float32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `stream-${sequence++}`, format, length);
  try {
    const pool = view('uint32', 128);
    expect(
      () =>
        new GPUTerrainStreamNetwork({
          id: 'pooled-stream',
          grid: {width: 2, height: 1, affine: [1, 0, 0, 0, -1, 0], pixelInterpretation: 'area'},
          flowDirections: view('uint32', 2),
          streams: view('uint32', 2),
          output: {
            sourceCells: graph.createDataView(pool.buffer, {format: 'uint32', length: 1}),
            targetCells: graph.createDataView(pool.buffer, {
              format: 'uint32',
              byteOffset: 256,
              length: 1
            }),
            vertexOffsets: view('uint32', 2),
            positions: view('float32x2', 2),
            lengths: view('float32', 1),
            reachCount: view('uint32', 1),
            vertexCount: view('uint32', 1),
            status: {overflow: view('uint32', 1)},
            unresolvedCount: view('uint32', 1)
          }
        })
    ).toThrow(/pooled-stream.*output.*share.*buffer/);
  } finally {
    device.destroy();
  }
});

it('allows separately imported read-only raster values and validity from one physical buffer', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const physical = device.createBuffer({byteLength: 512, usage: Buffer.STORAGE});
  try {
    const importView = <Format extends 'float32' | 'uint32'>(
      id: string,
      format: Format,
      byteOffset: number
    ) =>
      graph.createDataView(
        graph.importBuffer({id, byteLength: physical.byteLength, usage: physical.usage}, physical),
        {format, byteOffset, length: 1}
      );
    const grid = {
      width: 1,
      height: 1,
      affine: [1, 0, 0, 0, -1, 0] as const,
      pixelInterpretation: 'area' as const
    };
    const contributor = new GPURasterReprojection({
      id: 'read-alias',
      sourceGrid: grid,
      destinationGrid: grid,
      parameters: createTransientView(graph, 'parameters', 'float32', 20),
      input: {
        id: 'input',
        format: 'float32',
        storage: {kind: 'buffer', values: importView('values-import', 'float32', 0)},
        validity: importView('validity-import', 'uint32', 256)
      },
      output: {
        values: createTransientView(graph, 'output', 'float32', 1),
        validity: createTransientView(graph, 'validity', 'uint32', 1)
      }
    });
    try {
      expect(contributor.getCommandNodes(graph)).toHaveLength(2);
    } finally {
      contributor.destroy();
    }
  } finally {
    physical.destroy();
    device.destroy();
  }
});

it('both time-series planners accept packed fields beside a matching single chunk', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const values = createTransientView(graph, 'values', 'float32', 8);
    const output = createTransientView(graph, 'output', 'float32', 2);
    const vector = createVectorView('values-vector', 'float32', [values]);
    const props = {cellCount: 2, values: vector, output};
    const first = planFirstChunks('first', props, {values: 4, output: 1});
    expect(first).toHaveLength(1);
    expect(first[0].values).toBe(values);
    expect(first[0].output).toBe(output);
    const second = planSecondChunks('second', 2, {
      values: {view: vector, entriesPerCell: 4},
      output: {view: output, entriesPerCell: 1}
    });
    expect(second.chunks).toHaveLength(1);
    expect(second.chunks[0].views.get(vector)).toBe(values);
    expect(second.chunks[0].views.get(output)).toBe(output);
    const split = createVectorView('split', 'float32', [
      graph.createDataView(values.buffer, {format: 'float32', length: 4}),
      graph.createDataView(values.buffer, {format: 'float32', byteOffset: 16, length: 4})
    ]);
    expect(() =>
      planFirstChunks('first', {...props, values: split}, {values: 4, output: 1})
    ).toThrow(/topology/);
    expect(() =>
      planSecondChunks('second', 2, {
        values: {view: split, entriesPerCell: 4},
        output: {view: output, entriesPerCell: 1}
      })
    ).toThrow(/topology/);
  } finally {
    device.destroy();
  }
});

it('HANTS and temporal resampling accept the same pooled outputs with a one-chunk source', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let sequence = 0;
  const field = <Format extends 'float32' | 'uint32' | 'uint32x2'>(
    format: Format,
    length: number
  ) => createTransientView(graph, `field-${sequence++}`, format, length);
  const pool = field('float32', 256);
  const pooled = <Format extends 'float32' | 'uint32'>(
    format: Format,
    byteOffset: number,
    length: number
  ): GraphDataView<Format> => graph.createDataView(pool.buffer, {format, byteOffset, length});
  try {
    const values = createVectorView('values', 'float32', [field('float32', 8)]);
    const common = {cellCount: 1, sliceCount: 8, values, parameters: field('float32', 3)};
    const hants = new GPURasterHANTS({
      ...common,
      harmonicCount: 1,
      output: {
        smoothed: pooled('float32', 0, 8),
        coefficients: pooled('float32', 256, 3),
        status: pooled('uint32', 512, 1)
      }
    });
    const resample = new GPURasterTemporalResample({
      ...common,
      gridCount: 3,
      dateAxis: {
        acquisitionTimes: field('uint32x2', 8),
        temporalContext: {
          encoding: 'int64-words',
          unit: 'second',
          epoch: 'unix',
          axis: {kind: 'irregular'}
        }
      },
      output: {values: pooled('float32', 0, 3), status: pooled('uint32', 256, 3)}
    });
    expect(hants.getCommandNodes(graph)).toHaveLength(1);
    expect(resample.getCommandNodes(graph)).toHaveLength(1);
  } finally {
    device.destroy();
  }
});
