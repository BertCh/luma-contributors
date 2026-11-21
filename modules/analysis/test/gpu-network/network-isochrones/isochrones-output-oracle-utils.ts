// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {Buffer, type Device} from '@luma.gl/core';
import {
  GPUCommandGraph,
  type GraphDataView,
  type GPUCommandNodeProducer
} from '@luma.gl/graph/gpu-core';
import {expect} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import type {GPUSegmentRingAssemblyOutput} from '../../../src/analysis-core/ring-assembly/index';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';
import {rotateOracleRing} from './isochrones-output-oracle-fixtures';

type Format = 'float32' | 'float32x2' | 'float32x4' | 'uint32' | 'uint32x2';
export const POISON = 12345;
export const NONE = 0xffffffff;

/** Own each buffer separately so read/write aliasing cannot obscure an output assertion. */
export function createIsochroneOracleHarness(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const outputs = new Map<string, {buffer: Buffer; words: number; integer: boolean}>();
  const inputBuffers = new Map<GraphDataView, Buffer>();
  let serial = 0;
  function input<SelectedFormat extends Format>(
    format: SelectedFormat,
    values: number[]
  ): GraphDataView<SelectedFormat> {
    const integer = format.startsWith('uint32');
    const components = format.endsWith('x2') ? 2 : format.endsWith('x4') ? 4 : 1;
    const buffer = device.createBuffer({
      data: integer ? Uint32Array.from(values) : Float32Array.from(values),
      usage: Buffer.STORAGE | Buffer.COPY_SRC | Buffer.COPY_DST
    });
    buffers.push(buffer);
    const view = importGraphBuffer(
      graph,
      `isochrone-oracle-${serial++}`,
      buffer,
      format,
      values.length / components
    );
    inputBuffers.set(view, buffer);
    return view;
  }
  return {
    graph,
    input,
    write(view: GraphDataView, values: number[]) {
      inputBuffers
        .get(view)!
        .write(
          view.format.startsWith('uint32') ? Uint32Array.from(values) : Float32Array.from(values)
        );
    },
    output<SelectedFormat extends Format>(name: string, format: SelectedFormat, length: number) {
      const components = format.endsWith('x2') ? 2 : format.endsWith('x4') ? 4 : 1;
      const words = components * length;
      const view = input(format, new Array(words).fill(POISON));
      outputs.set(name, {
        buffer: buffers[buffers.length - 1],
        words,
        integer: format.startsWith('uint32')
      });
      return view;
    },
    destroy() {
      buffers.forEach(buffer => buffer.destroy());
    },
    async run(
      contributor?: GPUCommandNodeProducer,
      secondFrame?: (first: Record<string, number[]>) => void
    ) {
      let compiled: ReturnType<typeof graph.compile> | undefined;
      try {
        if (contributor) graph.add(contributor);
        compiled = graph.compile();
        submitGraph(device, compiled, undefined);
        const readOutputs = async () => {
          const result: Record<string, number[]> = {};
          for (const [name, output] of outputs) {
            result[name] = await (output.integer ? readUint32 : readFloat32)(
              output.buffer,
              output.words
            );
          }
          return result;
        };
        const first = await readOutputs();
        if (!secondFrame) return first;
        secondFrame(first);
        submitGraph(device, compiled, undefined);
        return await readOutputs();
      } finally {
        compiled?.destroy();
        buffers.forEach(buffer => buffer.destroy());
      }
    }
  };
}
export type IsochroneOracleHarness = ReturnType<typeof createIsochroneOracleHarness>;

/** Optional ports are exercised individually, with only their required dependency group. */
export const RING_PORTS = [
  'ringAreas',
  'ringIsHole',
  'ringShells',
  'ringGroups',
  'segmentRings',
  'segmentVertices',
  'segmentFlags',
  'ringRequiredCount',
  'openSegmentCount',
  'touchingSegmentCount',
  'polygons',
  'sourceIds'
] as const;
export type RingSelection = 'all' | 'minimal' | (typeof RING_PORTS)[number];

export const RASTER_ORACLE_SELECTIONS = [
  'all',
  'minimal',
  ...RING_PORTS,
  'no-rings',
  'edges',
  'triangles',
  'triangleFacilities',
  'triangleRequiredCount',
  'vertexCount',
  'edgeRequiredCount',
  'edgeCount'
] as const;

export function createOracleRingOutput(
  harness: IsochroneOracleHarness,
  selection: RingSelection,
  segments: number,
  vertexCapacity = 5
): GPUSegmentRingAssemblyOutput {
  const enabled = (name: RingSelection) => selection === 'all' || selection === name;
  const polygons = enabled('polygons') || enabled('sourceIds');
  return {
    ringOffsets: harness.output('ringOffsets', 'uint32', 3),
    positions: harness.output('ringPositions', 'float32x2', vertexCapacity),
    ...(enabled('ringAreas') ? {ringAreas: harness.output('ringAreas', 'float32', 2)} : {}),
    ...(enabled('ringIsHole') ? {ringIsHole: harness.output('ringIsHole', 'uint32', 2)} : {}),
    ...(enabled('ringShells') ? {ringShells: harness.output('ringShells', 'uint32', 2)} : {}),
    ...(enabled('ringGroups') ? {ringGroups: harness.output('ringGroups', 'uint32', 2)} : {}),
    ...(enabled('segmentRings')
      ? {segmentRings: harness.output('segmentRings', 'uint32', segments)}
      : {}),
    ...(enabled('segmentVertices')
      ? {segmentVertices: harness.output('segmentVertices', 'uint32', segments)}
      : {}),
    ...(enabled('segmentFlags')
      ? {segmentFlags: harness.output('segmentFlags', 'uint32', segments)}
      : {}),
    ...(polygons
      ? {
          polygons: {
            kind: 'polygons' as const,
            positions: harness.output('polygonPositions', 'float32x2', vertexCapacity),
            ringOffsets: harness.output('polygonRingOffsets', 'uint32', 3),
            polygonOffsets: harness.output('polygonOffsets', 'uint32', 3),
            featureOffsets: harness.output('featureOffsets', 'uint32', 3),
            ...(enabled('sourceIds') ? {sourceIds: harness.output('sourceIds', 'uint32', 2)} : {})
          }
        }
      : {}),
    status: {
      count: harness.output('ringCount', 'uint32', 1),
      overflow: harness.output('ringOverflow', 'uint32', 1),
      ...(enabled('ringRequiredCount')
        ? {requiredCount: harness.output('ringRequiredCount', 'uint32', 1)}
        : {}),
      ...(enabled('openSegmentCount')
        ? {openSegmentCount: harness.output('openSegmentCount', 'uint32', 1)}
        : {}),
      ...(enabled('touchingSegmentCount')
        ? {touchingSegmentCount: harness.output('touchingSegmentCount', 'uint32', 1)}
        : {})
    }
  };
}

/** Exact integer comparison and NaN-safe float comparison. */
export function expectOracleValues(actual: number[], expected: number[], tolerance = 0) {
  expect(actual.length).toBe(expected.length);
  expected.forEach((value, index) => {
    expect(Number.isFinite(actual[index]), `element ${index}`).toBe(true);
    expect(Math.abs(actual[index] - value), `element ${index}`).toBeLessThanOrEqual(tolerance);
  });
}

/** Correlate unrotated positions and mapping against independently derived segment starts. */
export function expectOracleSegmentStarts(
  positions: number[],
  vertices: number[],
  starts: number[],
  tolerance = 0
) {
  expect(vertices.length * 2).toBe(starts.length);
  vertices.forEach((vertex, segment) => {
    expect(Number.isSafeInteger(vertex)).toBe(true);
    expect(vertex).toBeGreaterThanOrEqual(0);
    expect(vertex * 2 + 1).toBeLessThan(positions.length);
    expectOracleValues(
      positions.slice(2 * vertex, 2 * vertex + 2),
      starts.slice(2 * segment, 2 * segment + 2),
      tolerance
    );
  });
}

/** One shell, padded to two ring rows; test every requested optional port against literals. */
export function expectOracleSquareRings(
  result: Record<string, number[]>,
  corners: number[],
  area: number,
  group: number,
  segmentVertices: number[],
  segmentEndpoints: number[],
  tolerance = 0
) {
  expectOracleValues(result['ringOffsets'], [0, 5, 5]);
  expectOracleValues(rotateOracleRing(result['ringPositions'].slice(0, 8)), corners, tolerance);
  expectOracleValues(result['ringPositions'].slice(8, 10), result['ringPositions'].slice(0, 2));
  expectOracleValues(result['ringCount'], [1]);
  expectOracleValues(result['ringOverflow'], [0]);
  if (result['ringAreas']) expectOracleValues(result['ringAreas'], [area, 0], tolerance * 180);
  if (result['ringIsHole']) expectOracleValues(result['ringIsHole'], [0, 0]);
  if (result['ringShells']) expectOracleValues(result['ringShells'], [0, NONE]);
  if (result['ringGroups']) expectOracleValues(result['ringGroups'], [group, 0]);
  if (result['segmentRings']) expectOracleValues(result['segmentRings'], [0, 0, 0, 0]);
  if (result['segmentVertices']) {
    expectOracleValues(result['segmentVertices'], segmentVertices);
    const starts = segmentEndpoints.flatMap((_, index) =>
      index % 4 === 0 ? segmentEndpoints.slice(index, index + 2) : []
    );
    expectOracleSegmentStarts(
      result['ringPositions'],
      result['segmentVertices'],
      starts,
      tolerance
    );
  }
  if (result['segmentFlags']) expectOracleValues(result['segmentFlags'], [0, 0, 0, 0]);
  for (const [name, value] of [
    ['ringRequiredCount', 1],
    ['openSegmentCount', 0],
    ['touchingSegmentCount', 0]
  ] as const) {
    if (result[name]) expectOracleValues(result[name], [value]);
  }
  if (result['polygonPositions']) {
    expectOracleValues(
      rotateOracleRing(result['polygonPositions'].slice(0, 8)),
      corners,
      tolerance
    );
    // Polygon rings omit the closing duplicate; unused geometry stays untouched.
    expectOracleValues(result['polygonPositions'].slice(8, 10), [POISON, POISON]);
    expectOracleValues(result['polygonRingOffsets'], [0, 4, 4]);
    expectOracleValues(result['polygonOffsets'], [0, 1, 1]);
    expectOracleValues(result['featureOffsets'], [0, 1, 1]);
    if (result['sourceIds']) expectOracleValues(result['sourceIds'], [group, NONE]);
  }
}
