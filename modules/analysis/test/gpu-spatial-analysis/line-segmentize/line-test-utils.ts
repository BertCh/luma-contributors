// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {expect, vi} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import type {GPUCommandNodeProducer} from '@luma.gl/graph/gpu-core';
import type {GPULineGeometryOutput} from '../../../src/gpu-spatial-analysis/line-segmentize/line-segmentize-types';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {getHaversineDistance} from '../geometry-measures/geodesic-oracle';
import type {FlatLineResult} from './line-segmentize-oracle';

/**
 * Read-back of a `GPULineGeometryOutput`. `count` and `requiredCount` count lines,
 * `vertexCount` and `requiredVertexCount` count vertices (contributor contract rule 8). With
 * `minimalStatus` the optional scalars are not wired and read as `NaN`.
 */
export type LineLineReadback = {
  positions: number[][];
  lineOffsets: number[];
  count: number;
  overflow: number;
  requiredCount: number;
  /** Present only when the fixture was built with `withApproximation`. */
  approximation: number;
  sourceLineIndices: number[];
  sourceRows: number[];
  measures: number[];
  vertexCount: number;
  requiredVertexCount: number;
};

/** A compiled single-contributor graph with typed inputs, a parameter buffer and a line output. */
export type LineLineFixture = {
  /** Writes parameters, encodes once, and reads the output. */
  run(parameters: Float32Array): Promise<LineLineReadback>;
  /** Number of `compile()` calls after setup. */
  getCompileCount(): number;
  destroy(): void;
};

/**
 * Builds a fixture around one contributor. `inputs` are imported by name; `createContributor` receives the
 * imported views, the parameter view and the output views.
 */
export function createLineLineFixture(
  device: Device,
  options: {
    inputs: Record<
      string,
      {
        values: Float32Array | Uint32Array;
        format: 'float32x2' | 'uint32' | 'float32';
      }
    >;
    capacity: number;
    /** Whether to request the per-vertex `sourceRows` and `measures` outputs. Default true. */
    withVertexColumns?: boolean;
    /** Whether to request `status.approximation` (only segmentize and arcs write it). Default false. */
    withApproximation?: boolean;
    /**
     * Wire only the required `status.count` and `status.overflow`, leaving out `requiredCount`,
     * `vertexCount` and `requiredVertexCount`. Default false.
     */
    minimalStatus?: boolean;
    lineCapacity: number;
    createContributor: (
      inputs: Record<string, GraphDataView>,
      parameters: GraphDataView<'float32'>,
      output: GPULineGeometryOutput
    ) => GPUCommandNodeProducer;
  }
): LineLineFixture {
  const buffers: Buffer[] = [];
  const graph = new GPUCommandGraph(device, {id: 'line-line-fixture'});
  const views: Record<string, GraphDataView> = {};
  for (const [name, input] of Object.entries(options.inputs)) {
    const buffer = createInputBuffer(device, input.values);
    buffers.push(buffer);
    const components = input.format === 'float32x2' ? 2 : 1;
    views[name] = importGraphBuffer(
      graph,
      name,
      buffer,
      input.format,
      input.values.length / components
    );
  }
  const {capacity, lineCapacity} = options;
  const output = (name: string, length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return buffer;
  };
  const outputs = {
    positions: output('positions', 2 * capacity),
    lineOffsets: output('line-offsets', lineCapacity + 1),
    count: output('count', 1),
    overflow: output('overflow', 1),
    requiredCount: output('total', 1),
    approximation: output('approximation', 1),
    sourceLineIndices: output('source-line-indexs', lineCapacity),
    sourceRows: output('source-rows', capacity),
    measures: output('measures', capacity),
    vertexCount: output('vertex-count', 1),
    requiredVertexCount: output('required-vertex-count', 1)
  };
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'parameters',
    format: 'float32',
    length: 4
  });
  graph.add(
    options.createContributor(views, parameterBuffer.importToGraph(graph), {
      positions: importGraphBuffer(graph, 'o-positions', outputs.positions, 'float32x2', capacity),
      lineOffsets: importGraphBuffer(
        graph,
        'o-line-offsets',
        outputs.lineOffsets,
        'uint32',
        lineCapacity + 1
      ),
      status: {
        count: importGraphBuffer(graph, 'o-count', outputs.count, 'uint32', 1),
        overflow: importGraphBuffer(graph, 'o-overflow', outputs.overflow, 'uint32', 1),
        ...(options.minimalStatus
          ? {}
          : {
              requiredCount: importGraphBuffer(
                graph,
                'o-total',
                outputs.requiredCount,
                'uint32',
                1
              ),
              vertexCount: importGraphBuffer(
                graph,
                'o-vertex-count',
                outputs.vertexCount,
                'uint32',
                1
              ),
              requiredVertexCount: importGraphBuffer(
                graph,
                'o-required-vertex-count',
                outputs.requiredVertexCount,
                'uint32',
                1
              )
            }),
        ...(options.withApproximation
          ? {
              approximation: importGraphBuffer(
                graph,
                'o-approximation',
                outputs.approximation,
                'uint32',
                1
              )
            }
          : {})
      },
      sourceLineIndices: importGraphBuffer(
        graph,
        'o-source-line-indexs',
        outputs.sourceLineIndices,
        'uint32',
        lineCapacity
      ),
      ...(options.withVertexColumns === false
        ? {}
        : {
            sourceRows: importGraphBuffer(
              graph,
              'o-source-rows',
              outputs.sourceRows,
              'uint32',
              capacity
            ),
            measures: importGraphBuffer(graph, 'o-measures', outputs.measures, 'float32', capacity)
          })
    })
  );
  const compiled = graph.compile();
  const compileSpy = vi.spyOn(graph, 'compile');
  return {
    async run(parameters) {
      parameterBuffer.write(parameters);
      submitGraph(device, compiled, undefined);
      const lineOffsets = await readUint32(outputs.lineOffsets, lineCapacity + 1);
      const readOptional = async (buffer: Buffer) =>
        options.minimalStatus ? Number.NaN : (await readUint32(buffer, 1))[0];
      const vertexCount = await readOptional(outputs.vertexCount);
      // The last line offset is the clamped vertex total; it sizes the read when vertexCount is not wired.
      const writtenVertexCount = options.minimalStatus ? lineOffsets[lineCapacity] : vertexCount;
      const flatPositions = await readFloat32(outputs.positions, 2 * capacity);
      return {
        positions: Array.from({length: writtenVertexCount}, (_, row) => [
          flatPositions[2 * row],
          flatPositions[2 * row + 1]
        ]),
        lineOffsets,
        count: (await readUint32(outputs.count, 1))[0],
        overflow: (await readUint32(outputs.overflow, 1))[0],
        requiredCount: await readOptional(outputs.requiredCount),
        approximation: (await readUint32(outputs.approximation, 1))[0],
        sourceLineIndices: await readUint32(outputs.sourceLineIndices, lineCapacity),
        sourceRows: (await readUint32(outputs.sourceRows, capacity)).slice(0, writtenVertexCount),
        measures: (await readFloat32(outputs.measures, capacity)).slice(0, writtenVertexCount),
        vertexCount,
        requiredVertexCount: await readOptional(outputs.requiredVertexCount)
      };
    },
    getCompileCount() {
      return compileSpy.mock.calls.length;
    },
    destroy() {
      compiled.destroy();
      parameterBuffer.destroy();
      for (const buffer of buffers) {
        buffer.destroy();
      }
    }
  };
}

/**
 * Asserts that a read-back line output equals an f64 oracle result truncated to `capacity`.
 *
 * @param positionTolerance Absolute tolerance per coordinate.
 * @param measureTolerance Relative tolerance on measures (scaled by `max(1, |expected|)`).
 * @param getPositionError Error metric, by default the largest absolute coordinate difference.
 * @param expectedOverflow Expected overflow flag when another capacity (lines) can overflow.
 * @returns The largest position error.
 */
export function expectLineLineParity(
  actual: LineLineReadback,
  expected: FlatLineResult,
  capacity: number,
  positionTolerance: number,
  measureTolerance: number,
  getPositionError: (actual: number[], expected: number[]) => number = getMaximumCoordinateError,
  expectedOverflow?: number
): number {
  const total = expected.positions.length;
  expect(actual.requiredVertexCount).toBe(total);
  expect(actual.vertexCount).toBe(Math.min(total, capacity));
  expect(actual.overflow).toBe(expectedOverflow ?? (total > capacity ? 1 : 0));
  expect(actual.lineOffsets).toEqual(
    expected.lineOffsets.map(offset => Math.min(offset, capacity))
  );
  if (expected.sourceRows.length > 0) {
    expect(actual.sourceRows).toEqual(expected.sourceRows.slice(0, capacity));
  }
  let maximumError = 0;
  actual.positions.forEach((vertex, row) => {
    const reference = expected.positions[row];
    const error = getPositionError(vertex, reference);
    maximumError = Math.max(maximumError, error);
    expect(error, `vertex ${row}: ${vertex} vs ${reference}`).toBeLessThanOrEqual(
      positionTolerance
    );
  });
  (expected.measures.length > 0 ? actual.measures : []).forEach((measure, row) => {
    const reference = expected.measures[row];
    expect(Math.abs(measure - reference), `measure ${row}`).toBeLessThanOrEqual(
      measureTolerance * Math.max(1, Math.abs(reference))
    );
  });
  return maximumError;
}

/** Largest absolute coordinate difference. */
export function getMaximumCoordinateError(actual: number[], expected: number[]): number {
  return Math.max(Math.abs(actual[0] - expected[0]), Math.abs(actual[1] - expected[1]));
}

/** Great-circle distance in meters between two longitude/latitude positions. */
export function getSphericalErrorMeters(actual: number[], expected: number[]): number {
  return getHaversineDistance(actual, expected, 6371008.8);
}
