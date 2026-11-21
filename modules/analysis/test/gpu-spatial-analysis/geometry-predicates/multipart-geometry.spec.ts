// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGeometryPredicates} from '../../../src/gpu-spatial-analysis/geometry-predicates/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  createMultipartGeometry,
  expectMultipartPredicates,
  FIXTURE,
  MULTIPART_COLUMNS,
  expectSignedZeroUpload
} from './multipart-geometry-test-utils';

for (const kind of ['points', 'lines'] as const) {
  it(`multipart ${kind} writes every predicate port against 15 Shapely aligned pairs`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const graph = new GPUCommandGraph(device);
    const buffers: Buffer[] = [];
    const records = FIXTURE[kind];
    const upload = (prefix: string, operand: 'left' | 'right') =>
      createMultipartGeometry(
        kind,
        records.map(record => record[operand]),
        (name, values, format) => {
          if (format === 'float32x2') expectSignedZeroUpload(records, operand, values);
          const buffer = createInputBuffer(device, values);
          buffers.push(buffer);
          return importGraphBuffer(
            graph,
            `${prefix}-${name}`,
            buffer,
            format,
            format === 'float32x2' ? values.length / 2 : values.length
          );
        }
      );
    const geometry = upload('left', 'left');
    const other = upload('right', 'right');
    const output = (name: string, length = records.length) => {
      const buffer = createOutputBuffer(device, length);
      buffers.push(buffer);
      return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
    };
    const columns = Object.fromEntries(MULTIPART_COLUMNS.map(name => [name, output(name)]));
    const uniqueBuffer = createOutputBuffer(device, geometry.positions.length * 2);
    buffers.push(uniqueBuffer);
    const uniquePositions = importGraphBuffer(
      graph,
      'uniquePositions',
      uniqueBuffer,
      'float32x2',
      geometry.positions.length
    );
    const uniqueOffsets = output('uniqueOffsets', records.length + 1);
    const equalsExact = output('equalsExact');
    const equalsIdentical = output('equalsIdentical');
    const overflow = output('overflow', 1);
    const intersectionCount = output('intersectionCount', 1);
    const parameterBuffer = createInputBuffer(device, new Float32Array([0, 0, 0, 0]));
    buffers.push(parameterBuffer);
    const parameters = importGraphBuffer(graph, 'parameters', parameterBuffer, 'float32', 4);
    graph.add(
      new GPUGeometryPredicates({
        geometry,
        other,
        parameters,
        intersectionCapacity: 64,
        output: {
          ...Object.fromEntries(
            Object.entries(columns).map(([name, column]) => [name, column.view])
          ),
          uniquePositions,
          uniqueOffsets: uniqueOffsets.view,
          equalsExact: equalsExact.view,
          equalsIdentical: equalsIdentical.view,
          status: {overflow: overflow.view, intersectionCount: intersectionCount.view}
        }
      })
    );
    const compiled = graph.compile();
    try {
      for (const [toleranceIndex, tolerance] of [0, 0.25].entries()) {
        parameterBuffer.write(new Float32Array([tolerance, 0, 0, 0]));
        for (const column of Object.values(columns))
          column.buffer.write(new Uint32Array(records.length).fill(99));
        uniqueOffsets.buffer.write(new Uint32Array(records.length + 1).fill(99));
        uniqueBuffer.write(new Float32Array(geometry.positions.length * 2).fill(Number.NaN));
        overflow.buffer.write(new Uint32Array([99]));
        intersectionCount.buffer.write(new Uint32Array([99]));
        submitGraph(device, compiled, undefined);
        const values = Object.fromEntries(
          await Promise.all(
            Object.entries(columns).map(async ([name, column]) => [
              name,
              await readUint32(column.buffer, records.length)
            ])
          )
        );
        expectMultipartPredicates(
          records,
          values,
          await readUint32(uniqueOffsets.buffer, records.length + 1),
          await readFloat32(uniqueBuffer, geometry.positions.length * 2),
          await readUint32(equalsExact.buffer, records.length),
          await readUint32(equalsIdentical.buffer, records.length),
          toleranceIndex
        );
        expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
        expect(await readUint32(intersectionCount.buffer, 1)).toEqual([
          records.reduce((count, record) => count + record.intersectionCount, 0)
        ]);
      }
    } finally {
      compiled.destroy();
      for (const buffer of buffers) buffer.destroy();
    }
  });
}

it('intersection overflow suppresses simplicity of both bowties', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const upload = <Format extends 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format,
    length: number
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const positions = upload(
    'positions',
    Float32Array.of(0, 0, 2, 2, 0, 2, 2, 0, 10, 0, 12, 2, 10, 2, 12, 0),
    'float32x2',
    8
  );
  const lineOffsets = upload('lines', Uint32Array.of(0, 4, 8), 'uint32', 3);
  const isSimple = output('simple', 2);
  const overflow = output('overflow', 1);
  const intersectionCount = output('intersectionCount', 1);
  // Hand oracle: one nonadjacent crossing in each of two disjoint bowties; one list slot.
  graph.add(
    new GPUGeometryPredicates({
      geometry: {kind: 'lines', positions, lineOffsets},
      intersectionCapacity: 1,
      output: {
        isSimple: isSimple.view,
        status: {overflow: overflow.view, intersectionCount: intersectionCount.view}
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(isSimple.buffer, 2)).toEqual([0, 0]);
    expect(await readUint32(overflow.buffer, 1)).toEqual([1]);
    expect(await readUint32(intersectionCount.buffer, 1)).toEqual([2]);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
