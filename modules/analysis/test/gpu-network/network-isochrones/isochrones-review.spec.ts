// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {type Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUNetworkIsochrones,
  getGPUNetworkIsochronesParameterValues
} from '../../../src/gpu-network/network-isochrones/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

it.each([
  false,
  true
])('writes the caller raster through an inherited output getter: %s', async inherited => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    values: Uint32Array | Float32Array,
    format: Format,
    length: number
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const rasterBuffer = createOutputBuffer(device, 4);
  rasterBuffer.write(Float32Array.of(-99, -99, -99, -99));
  buffers.push(rasterBuffer);
  const values = importGraphBuffer(graph, 'raster-values', rasterBuffer, 'float32', 4);
  const bands = createOutputBuffer(device, 4);
  buffers.push(bands);
  const output = inherited
    ? Object.create({
        get values() {
          return values;
        }
      })
    : {values};
  output.bandClasses = importGraphBuffer(graph, 'bands', bands, 'uint32', 4);
  try {
    graph.add(
      new GPUNetworkIsochrones({
        topology: {
          offsets: input('offsets', Uint32Array.of(0, 0, 0), 'uint32', 3),
          neighbors: input('neighbors', Uint32Array.of(0), 'uint32', 1),
          weights: input('weights', Float32Array.of(0), 'float32', 1),
          nodePositions: input('positions', Float32Array.of(0, 0, 1, 0), 'float32x2', 2)
        },
        costs: input('costs', Float32Array.of(Infinity, Infinity), 'float32', 2),
        costBreaks: input('breaks', Float32Array.of(10), 'float32', 1),
        parameters: input(
          'parameters',
          getGPUNetworkIsochronesParameterValues({
            extent: [0, 0, 2, 2],
            breakCount: 1
          }),
          'float32',
          12
        ),
        raster: {width: 2, height: 2, unreachedCost: 100, output}
      })
    );
    const compiled = graph.compile();
    try {
      submitGraph(device, compiled, undefined);
      // No reached edges: every pixel must decode the explicitly chosen unreached value.
      expect(await readFloat32(rasterBuffer, 4)).toEqual([100, 100, 100, 100]);
    } finally {
      compiled.destroy();
    }
  } finally {
    for (const buffer of buffers) buffer.destroy();
  }
});

for (const tableCapacity of [1, 2]) {
  it(`handles two nonadjacent reached cells with transient table capacity ${tableCapacity}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      values: Uint32Array | Float32Array,
      format: Format,
      length: number
    ) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, format, length);
    };
    const output = <Format extends 'uint32' | 'uint32x2' | 'float32x4'>(
      name: string,
      format: Format,
      length: number,
      components = 1
    ) => {
      const buffer = createOutputBuffer(device, length * components);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
    };
    const count = output('count', 'uint32', 1);
    const overflow = output('overflow', 'uint32', 1);
    try {
      const create = () =>
        new GPUNetworkIsochrones({
          topology: {
            offsets: input('offsets', Uint32Array.of(0, 0, 0), 'uint32', 3),
            neighbors: input('neighbors', Uint32Array.of(0), 'uint32', 1),
            weights: input('weights', Float32Array.of(0), 'float32', 1),
            nodePositions: input('positions', Float32Array.of(-40, 0, 40, 0), 'float32x2', 2)
          },
          costs: input('costs', Float32Array.of(0, 0), 'float32', 2),
          costBreaks: input('breaks', Float32Array.of(10), 'float32', 1),
          parameters: input(
            'parameters',
            getGPUNetworkIsochronesParameterValues({
              breakCount: 1,
              extent: [-180, -80, 180, 80],
              cellCostLimit: 10
            }),
            'float32',
            12
          ),
          cellOutline: {
            family: 'quadbin',
            resolution: 4,
            tableCapacity,
            output: {
              rows: output('rows', 'uint32', 8).view,
              cells: output('cells', 'uint32x2', 8, 2).view,
              edgeIndices: output('edges', 'uint32', 8).view,
              endpoints: output('endpoints', 'float32x4', 8, 4).view,
              status: {count: count.view, overflow: overflow.view}
            }
          }
        });
      if (tableCapacity === 1) {
        // Refusal is the selected incompleteness policy: the unexposed cell status cannot be lost.
        expect(create).toThrow(/tableCapacity.*node count.*2/);
      } else {
        graph.add(create());
        const compiled = graph.compile();
        try {
          submitGraph(device, compiled, undefined);
          expect(await readUint32(count.buffer, 1)).toEqual([8]);
          expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
        } finally {
          compiled.destroy();
        }
      }
    } finally {
      for (const buffer of buffers) buffer.destroy();
    }
  });
}

for (const suppliedFacilities of [false, true]) {
  it(`labels a supplied two-row cell table from four nodes (supplied facilities: ${suppliedFacilities})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
      name: string,
      values: Uint32Array | Float32Array,
      format: Format,
      length: number
    ) => {
      const buffer = createInputBuffer(device, values);
      buffers.push(buffer);
      return importGraphBuffer(graph, name, buffer, format, length);
    };
    const output = <Format extends 'uint32' | 'uint32x2' | 'float32x4'>(
      name: string,
      format: Format,
      length: number,
      components = 1
    ) => {
      const buffer = createOutputBuffer(device, length * components);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
    };
    const cells = output('cells', 'uint32x2', 2, 2);
    const counts = output('counts', 'uint32', 2);
    const tableCount = output('table-count', 'uint32', 1);
    const tableOverflow = output('table-overflow', 'uint32', 1);
    const cellFacilities = output('facilities', 'uint32', 2);
    const outlineCount = output('outline-count', 'uint32', 1);
    const outlineOverflow = output('outline-overflow', 'uint32', 1);
    const groups = output('groups', 'uint32', 8);
    try {
      graph.add(
        new GPUNetworkIsochrones({
          topology: {
            offsets: input('offsets', Uint32Array.of(0, 0, 0, 0, 0), 'uint32', 5),
            neighbors: input('neighbors', Uint32Array.of(0), 'uint32', 1),
            weights: input('weights', Float32Array.of(0), 'float32', 1),
            nodePositions: input(
              'positions',
              Float32Array.of(8.4, 47.1, 8.4, 47.1, 8.4, 47.1, 8.4, 47.1),
              'float32x2',
              4
            )
          },
          // All nodes lie in one cell. Costs 1 and 1 tie, so facility row 3 beats 7.
          costs: input('costs', Float32Array.of(4, 1, 1, 2), 'float32', 4),
          assignments: input('assignments', Uint32Array.of(9, 7, 3, 5), 'uint32', 4),
          costBreaks: input('breaks', Float32Array.of(10), 'float32', 1),
          parameters: input(
            'parameters',
            getGPUNetworkIsochronesParameterValues({
              breakCount: 1,
              extent: [8, 47, 9, 48],
              cellCostLimit: 10
            }),
            'float32',
            12
          ),
          cellOutline: {
            family: 'quadbin',
            resolution: 10,
            byFacility: true,
            tableCapacity: 3,
            table: {
              cells: cells.view,
              counts: counts.view,
              status: {count: tableCount.view, overflow: tableOverflow.view}
            },
            cellFacilities: suppliedFacilities ? cellFacilities.view : undefined,
            output: {
              rows: output('rows', 'uint32', 8).view,
              cells: output('outline-cells', 'uint32x2', 8, 2).view,
              edgeIndices: output('edges', 'uint32', 8).view,
              endpoints: output('endpoints', 'float32x4', 8, 4).view,
              groups: groups.view,
              status: {count: outlineCount.view, overflow: outlineOverflow.view}
            }
          }
        })
      );
      const compiled = graph.compile();
      try {
        submitGraph(device, compiled, undefined);
        expect(await readUint32(tableCount.buffer, 1)).toEqual([1]);
        expect(await readUint32(tableOverflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(counts.buffer, 2)).toEqual([4, 0]);
        if (suppliedFacilities) {
          expect(await readUint32(cellFacilities.buffer, 2)).toEqual([3, 0xffffffff]);
        }
        expect(await readUint32(outlineCount.buffer, 1)).toEqual([4]);
        expect(await readUint32(outlineOverflow.buffer, 1)).toEqual([0]);
        expect((await readUint32(groups.buffer, 8)).slice(0, 4)).toEqual([3, 3, 3, 3]);
      } finally {
        compiled.destroy();
      }
    } finally {
      for (const buffer of buffers) buffer.destroy();
    }
  });
}
