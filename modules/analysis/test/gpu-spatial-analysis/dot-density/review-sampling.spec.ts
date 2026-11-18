// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph, type GraphDataView} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUDotDensity,
  GPURandomPointsInPolygon,
  getGPUDotDensityParameterValues
} from '../../../src/gpu-spatial-analysis/dot-density';
import {DOT_DENSITY_CANDIDATE_PURPOSE} from '../../../src/gpu-spatial-analysis/dot-density/dot-density-constants';
import {computePolygonTriangulationOnCPU} from '../../../src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation';
import {getPolygonTriangulationIndexCount} from '../../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation';
import {getPhilox4x32, getPhiloxUnitFloat} from '../../../src/analysis-core/kernels/philox-random';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createPolygonColumns} from './dot-density-scene';
import type {DotDensityCPUPolygons} from './dot-density-cpu';
import {requireTestDevice} from '../../utils/require-test-device';

async function runSampling(
  polygons: DotDensityCPUPolygons,
  requests: Uint32Array,
  options: {
    route?: 'values' | 'counts';
    capacity?: number;
    normalize?: boolean;
    packed?: boolean;
    nextRequests?: Uint32Array;
  } = {}
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers = new Set<Buffer>();
  const track = (buffer: Buffer) => {
    buffers.add(buffer);
    return buffer;
  };
  const input = <Format extends 'uint32' | 'float32' | 'float32x2'>(
    name: string,
    values: Uint32Array | Float32Array,
    format: Format,
    rows = values.length
  ) => importGraphBuffer(graph, name, track(createInputBuffer(device, values)), format, rows);
  const output = (name: string, length: number) => {
    const buffer = track(createOutputBuffer(device, length));
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', length)};
  };
  const featureCount = requests.length;
  const capacity = options.capacity ?? requests.reduce((sum, value) => sum + value, 0);
  const coordinates = track(
    createOutputBuffer(device, options.packed ? 3 * capacity : 2 * capacity)
  );
  const positionsBacking = importGraphBuffer(
    graph,
    'dot-positions',
    coordinates,
    'float32x2',
    capacity
  );
  const ids = options.packed
    ? {
        buffer: coordinates,
        view: graph.createDataView(positionsBacking.buffer, {
          format: 'uint32' as const,
          byteOffset: 8 * capacity,
          length: capacity
        })
      }
    : output('ids', capacity);
  const count = output('count', 1);
  const overflow = output('overflow', 1);
  const candidateOverflow = output('candidate-overflow', 1);
  const required = output('required', 1);
  const failed = output('failed', 1);
  const slotCounts = output('slot-counts', featureCount);
  const slotOffsets = output('slot-offsets', featureCount);
  const props = {
    positions: input('positions', polygons.positions, 'float32x2', polygons.positions.length / 2),
    featureOffsets: input('features', polygons.featureOffsets, 'uint32'),
    polygonOffsets: input('polygons', polygons.polygonOffsets, 'uint32'),
    ringOffsets: input('rings', polygons.ringOffsets, 'uint32'),
    parameters: input(
      'parameters',
      getGPUDotDensityParameterValues({seed: 123, maskExtent: [0, 0, 64, 16]}),
      'uint32'
    ),
    mask: options.normalize
      ? {
          weights: input('mask', new Float32Array([0.25]), 'float32'),
          width: 1,
          height: 1,
          normalizeFeatures: true
        }
      : undefined,
    output: {
      positions: positionsBacking,
      dots: {
        ids: ids.view,
        status: {
          count: count.view,
          overflow: overflow.view,
          candidateOverflow: candidateOverflow.view,
          requiredCount: required.view
        }
      },
      status: {failedCount: failed.view},
      slotCounts: slotCounts.view,
      slotOffsets: slotOffsets.view
    }
  };
  const requestBuffer = track(
    createInputBuffer(device, options.route === 'counts' ? requests : Float32Array.from(requests))
  );
  const requestView = importGraphBuffer(
    graph,
    'requests',
    requestBuffer,
    options.route === 'counts' ? 'uint32' : 'float32',
    requests.length
  );
  graph.add(
    options.route === 'counts'
      ? new GPURandomPointsInPolygon({
          ...props,
          counts: requestView as GraphDataView<'uint32'>
        })
      : new GPUDotDensity({
          ...props,
          values: requestView as GraphDataView<'float32'>
        })
  );
  const compiled = graph.compile();
  try {
    const read = async () => {
      const identifiers = await readUint32(ids.buffer, options.packed ? 3 * capacity : capacity);
      return {
        positions: await readFloat32(coordinates, 2 * capacity),
        ids: options.packed ? identifiers.slice(2 * capacity) : identifiers,
        count: (await readUint32(count.buffer, 1))[0],
        overflow: (await readUint32(overflow.buffer, 1))[0],
        candidateOverflow: (await readUint32(candidateOverflow.buffer, 1))[0],
        required: (await readUint32(required.buffer, 1))[0],
        failed: (await readUint32(failed.buffer, 1))[0],
        slotCounts: await readUint32(slotCounts.buffer, featureCount),
        slotOffsets: await readUint32(slotOffsets.buffer, featureCount)
      };
    };
    submitGraph(device, compiled, undefined);
    const first = await read();
    if (!options.nextRequests) return {...first, next: undefined};
    requestBuffer.write(
      options.route === 'counts' ? options.nextRequests : Float32Array.from(options.nextRequests)
    );
    submitGraph(device, compiled, undefined);
    return {...first, next: await read()};
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

it('samples multipolygons, holes and empty polygons against a CPU quantized-CDF oracle with packed outputs', async () => {
  const polygons = createPolygonColumns([
    [
      [
        [
          [0, 0],
          [8, 0],
          [8, 8],
          [0, 8]
        ],
        [
          [2, 2],
          [4, 2],
          [4, 4],
          [2, 4]
        ]
      ],
      [],
      [
        [
          [20, 0],
          [24, 0],
          [20, 4]
        ]
      ]
    ],
    [],
    [
      [
        [
          [30, 0],
          [32, 0],
          [30, 2]
        ]
      ]
    ]
  ]);
  const requests = new Uint32Array([32, 0, 16]);
  const result = await runSampling(polygons, requests, {packed: true});
  expect([result.count, result.required, result.overflow, result.failed]).toEqual([48, 48, 0, 0]);
  expect(result.slotCounts).toEqual([32, 0, 16]);
  const indexCapacity = getPolygonTriangulationIndexCount(
    polygons.positions.length / 2,
    polygons.ringOffsets.length - 1,
    polygons.polygonOffsets.length - 1
  );
  const mesh = computePolygonTriangulationOnCPU(polygons, {indexCapacity});
  const point = (row: number) => [polygons.positions[2 * row], polygons.positions[2 * row + 1]];
  const expectedAreas = [68, 0, 2]; // 8x8 shell - 2x2 hole + right triangle, empty, right triangle.
  let outputRow = 0;
  for (let feature = 0; feature < requests.length; feature++) {
    const first = mesh.triangleOffsets[polygons.featureOffsets[feature]];
    const end = mesh.triangleOffsets[polygons.featureOffsets[feature + 1]];
    const cumulative: number[] = [];
    let running = 0;
    for (let triangle = first; triangle < end; triangle++) {
      const firstPoint = point(mesh.indices[3 * triangle]);
      const second = point(mesh.indices[3 * triangle + 1]);
      const third = point(mesh.indices[3 * triangle + 2]);
      const area =
        Math.abs(
          (second[0] - firstPoint[0]) * (third[1] - firstPoint[1]) -
            (second[1] - firstPoint[1]) * (third[0] - firstPoint[0])
        ) / 2;
      running += Math.trunc(
        Math.fround(Math.fround(area / expectedAreas[feature]) * 4_000_000_000)
      );
      cumulative.push(running);
    }
    for (let rank = 0; rank < requests[feature]; rank++, outputRow++) {
      const random = getPhilox4x32([feature, rank, 0, 0], [123, DOT_DENSITY_CANDIDATE_PURPOSE]);
      const target = Math.min(
        Math.trunc(Math.fround(getPhiloxUnitFloat(random[0]) * Math.fround(running))),
        running - 1
      );
      const triangle = first + cumulative.findIndex(value => value > target);
      const firstPoint = point(mesh.indices[3 * triangle]);
      const second = point(mesh.indices[3 * triangle + 1]);
      const third = point(mesh.indices[3 * triangle + 2]);
      const root = Math.sqrt(getPhiloxUnitFloat(random[1]));
      const fraction = getPhiloxUnitFloat(random[2]);
      const expected = [0, 1].map(
        axis =>
          firstPoint[axis] +
          root *
            ((1 - fraction) * (second[axis] - firstPoint[axis]) +
              fraction * (third[axis] - firstPoint[axis]))
      );
      expect(result.ids[outputRow]).toBe(feature);
      expect(result.positions[2 * outputRow]).toBeCloseTo(expected[0], 4);
      expect(result.positions[2 * outputRow + 1]).toBeCloseTo(expected[1], 4);
    }
  }
});

for (const route of ['values', 'counts'] as const) {
  it(`slot-prefix clamp with spare final capacity reports candidate loss and clears on reuse (${route})`, async () => {
    const polygons = createPolygonColumns([
      [
        [
          [
            [0, 0],
            [1, 0],
            [0, 1]
          ]
        ]
      ]
    ]);
    polygons.featureOffsets = new Uint32Array(65537).fill(1);
    polygons.featureOffsets[0] = 0;
    const requests = new Uint32Array(65536);
    requests[0] = 65536;
    const nextRequests = new Uint32Array(65536);
    nextRequests[0] = 1;
    const result = await runSampling(polygons, requests, {route, capacity: 65536, nextRequests});
    expect([
      result.count,
      result.required,
      result.overflow,
      result.candidateOverflow,
      result.failed
    ]).toEqual([65535, 65535, 0, 1, 0]);
    expect(Number.isFinite(result.positions[0]) && Number.isFinite(result.positions[1])).toBe(true);
    expect(result.ids[0]).toBe(0);
    expect([
      result.next!.count,
      result.next!.required,
      result.next!.overflow,
      result.next!.candidateOverflow,
      result.next!.failed
    ]).toEqual([1, 1, 0, 0, 0]);
  });
}

for (const route of ['values', 'counts'] as const) {
  it(`concurrent ${route} slots clamp with exact independent counts and overflow`, async () => {
    const polygons = createPolygonColumns(
      Array.from({length: 256}, () => [
        [
          [
            [0, 0],
            [1, 0],
            [0, 1]
          ]
        ]
      ])
    );
    const result = await runSampling(polygons, new Uint32Array(256).fill(100), {
      route,
      capacity: 1
    });
    expect(result.slotCounts).toEqual(Array(256).fill(2));
    expect(result.slotOffsets).toEqual(Array.from({length: 256}, (_, row) => 2 * row));
    expect([result.count, result.required, result.overflow, result.failed]).toEqual([1, 512, 1, 0]);
    expect(result.ids).toEqual([0]);
  });
}

for (const route of ['values', 'counts'] as const) {
  for (const featureCount of [1, 65535, 65536]) {
    it(`normalizes ${route} for ${featureCount} features and preserves the final feature across padded workgroups`, async () => {
      const polygons = createPolygonColumns([
        [
          [
            [
              [0, 0],
              [1, 0],
              [0, 1]
            ]
          ]
        ]
      ]);
      polygons.featureOffsets = new Uint32Array(featureCount + 1);
      polygons.featureOffsets[featureCount] = 1;
      const requests = new Uint32Array(featureCount);
      requests[featureCount - 1] = 1;
      const result = await runSampling(polygons, requests, {
        route,
        normalize: true,
        capacity: 1,
        packed: true
      });
      expect([result.count, result.required, result.overflow, result.failed]).toEqual([1, 1, 0, 0]);
      expect(result.ids).toEqual([featureCount - 1]);
      expect(result.slotCounts.slice(0, -1).every(value => value === 0)).toBe(true);
      expect(result.slotCounts[featureCount - 1]).toBe(1);
      expect(result.positions.every(Number.isFinite)).toBe(true);
      expect(result.positions[0]).toBeGreaterThanOrEqual(0);
      expect(result.positions[1]).toBeGreaterThanOrEqual(0);
      expect(result.positions[0] + result.positions[1]).toBeLessThanOrEqual(1.00001);
      // A constant nonzero mask normalizes to acceptance probability one. The first
      // candidate in this unit right triangle is therefore the exact Philox draw.
      const random = getPhilox4x32(
        [featureCount - 1, 0, 0, 0],
        [123, DOT_DENSITY_CANDIDATE_PURPOSE]
      );
      const root = Math.sqrt(getPhiloxUnitFloat(random[1]));
      const fraction = getPhiloxUnitFloat(random[2]);
      const mesh = computePolygonTriangulationOnCPU(polygons, {indexCapacity: 3});
      expect(Array.from(mesh.indices).sort()).toEqual([0, 1, 2]);
      for (const axis of [0, 1]) {
        const first = polygons.positions[2 * mesh.indices[0] + axis];
        const second = polygons.positions[2 * mesh.indices[1] + axis];
        const third = polygons.positions[2 * mesh.indices[2] + axis];
        expect(result.positions[axis]).toBeCloseTo(
          first + root * ((1 - fraction) * (second - first) + fraction * (third - first)),
          5
        );
      }
    }, 120000);
  }
}
