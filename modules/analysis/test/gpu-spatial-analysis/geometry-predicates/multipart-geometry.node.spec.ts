// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUGeometryPredicates} from '../../../src/gpu-spatial-analysis/geometry-predicates/index';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {
  createMultipartGeometry,
  expectMultipartPredicates,
  FIXTURE,
  MULTIPART_COLUMNS,
  expectSignedZeroUpload
} from './multipart-geometry-test-utils';

useCPUExecutorTestHooks();

it('multipart predicates accept one logical point feature with two parts', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const contributor = new GPUGeometryPredicates({
      geometry: {
        kind: 'points',
        positions: createTransientView(graph, 'positions', 'float32x2', 2),
        featureOffsets: createTransientView(graph, 'features', 'uint32', 2)
      },
      output: {numCoordinates: createTransientView(graph, 'count', 'uint32', 1)}
    });
    expect(contributor.featureCount).toBe(1);
  } finally {
    device.destroy();
  }
});

for (const kind of ['points', 'lines'] as const) {
  it(`multipart ${kind} executes every predicate port against 15 Shapely aligned pairs`, () => {
    const device = createNullWebGPUDevice();
    const graph = new GPUCommandGraph(device);
    const memory = createCPUGraphMemory();
    const records = FIXTURE[kind];
    const upload = (prefix: string, operand: 'left' | 'right') =>
      createMultipartGeometry(
        kind,
        records.map(record => record[operand]),
        (name, values, format) => {
          if (format === 'float32x2') expectSignedZeroUpload(records, operand, values);
          const view = createTransientView(
            graph,
            `${prefix}-${name}`,
            format,
            format === 'float32x2' ? values.length / 2 : values.length
          );
          memory.write(view, values);
          return view;
        }
      );
    const geometry = upload('left', 'left');
    const other = upload('right', 'right');
    const column = (name: string, length = records.length) =>
      createTransientView(graph, name, 'uint32', length);
    const columns = Object.fromEntries(MULTIPART_COLUMNS.map(name => [name, column(name)]));
    const uniquePositions = createTransientView(
      graph,
      'uniquePositions',
      'float32x2',
      geometry.positions.length
    );
    const uniqueOffsets = column('uniqueOffsets', records.length + 1);
    const equalsExact = column('equalsExact');
    const equalsIdentical = column('equalsIdentical');
    const overflow = column('overflow', 1);
    const intersectionCount = column('intersectionCount', 1);
    const parameters = createTransientView(graph, 'parameters', 'float32', 4);
    const contributor = new GPUGeometryPredicates({
      geometry,
      other,
      parameters,
      intersectionCapacity: 64,
      output: {
        ...columns,
        uniquePositions,
        uniqueOffsets,
        equalsExact,
        equalsIdentical,
        status: {overflow, intersectionCount}
      }
    });
    try {
      const nodes = contributor.getCommandNodes(graph);
      for (const [toleranceIndex, tolerance] of [0, 0.25].entries()) {
        memory.write(parameters, new Float32Array([tolerance, 0, 0, 0]));
        for (const view of Object.values(columns))
          memory.write(view, new Uint32Array(records.length).fill(99));
        memory.write(uniqueOffsets, new Uint32Array(records.length + 1).fill(99));
        memory.write(
          uniquePositions,
          new Float32Array(geometry.positions.length * 2).fill(Number.NaN)
        );
        // Poison statuses to prove each encoding writes them, including multipoints.
        memory.write(overflow, new Uint32Array([99]));
        memory.write(intersectionCount, new Uint32Array([99]));
        executeCommandNodesOnCPU(nodes, memory);
        expectMultipartPredicates(
          records,
          Object.fromEntries(
            Object.entries(columns).map(([name, view]) => [
              name,
              Array.from(memory.readUint32(view))
            ])
          ),
          Array.from(memory.readUint32(uniqueOffsets)),
          Array.from(memory.readFloat32(uniquePositions)),
          Array.from(memory.readUint32(equalsExact)),
          Array.from(memory.readUint32(equalsIdentical)),
          toleranceIndex
        );
        expect(Array.from(memory.readUint32(overflow))).toEqual([0]);
        expect(Array.from(memory.readUint32(intersectionCount))).toEqual([
          records.reduce((count, record) => count + record.intersectionCount, 0)
        ]);
      }
    } finally {
      device.destroy();
    }
  });
}

it('incomplete segment intersections suppress every dependent simplicity result', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const positions = createTransientView(graph, 'positions', 'float32x2', 8);
  const lineOffsets = createTransientView(graph, 'lines', 'uint32', 3);
  memory.write(positions, Float32Array.of(0, 0, 2, 2, 0, 2, 2, 0, 10, 0, 12, 2, 10, 2, 12, 0));
  memory.write(lineOffsets, Uint32Array.of(0, 4, 8));
  const isSimple = createTransientView(graph, 'isSimple', 'uint32', 2);
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
  const intersectionCount = createTransientView(graph, 'intersectionCount', 'uint32', 1);
  try {
    // Two disjoint bowties each have one nonadjacent proper segment crossing (hand oracle).
    const contributor = new GPUGeometryPredicates({
      geometry: {kind: 'lines', positions, lineOffsets},
      intersectionCapacity: 1,
      output: {isSimple, status: {overflow, intersectionCount}}
    });
    executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
    expect(Array.from(memory.readUint32(overflow))).toEqual([1]);
    expect(Array.from(memory.readUint32(intersectionCount))).toEqual([2]);
    expect(Array.from(memory.readUint32(isSimple))).toEqual([0, 0]);
  } finally {
    device.destroy();
  }
});

it('multipoint simplicity works without requesting unique-point outputs', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const positions = createTransientView(graph, 'positions', 'float32x2', 6);
  const featureOffsets = createTransientView(graph, 'features', 'uint32', 3);
  memory.write(positions, Float32Array.of(0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0));
  memory.write(featureOffsets, Uint32Array.of(0, 3, 6));
  const isSimple = createTransientView(graph, 'simple', 'uint32', 2);
  const overflow = createTransientView(graph, 'overflow', 'uint32', 1);
  try {
    executeCommandNodesOnCPU(
      new GPUGeometryPredicates({
        geometry: {kind: 'points', positions, featureOffsets},
        intersectionCapacity: 1,
        output: {isSimple, status: {overflow}}
      }).getCommandNodes(graph),
      memory
    );
    // Hand oracle: one duplicate in the first feature, all three distinct in the second.
    expect(Array.from(memory.readUint32(isSimple))).toEqual([0, 1]);
    expect(Array.from(memory.readUint32(overflow))).toEqual([0]);
  } finally {
    device.destroy();
  }
});
