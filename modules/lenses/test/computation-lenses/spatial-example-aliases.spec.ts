// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, test} from 'vitest';
import {
  GPUNearestFeatureJoin,
  GPURandomPointsInPolygon,
  getGPUDotDensityParameterValues
} from '@luma-contributors/analysis/gpu-spatial-analysis';
import {importGraphBuffer} from '../../../analysis/src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../analysis/src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  readFloat32,
  submitGraph
} from '../../../analysis/test/utils/gpu-contributor-test-utils';

type Format = 'uint32' | 'float32' | 'float32x2';
function createFixture(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const output = <DataFormat extends Format>(format: DataFormat, length: number) => {
    const buffer = createOutputBuffer(device, length * (format === 'float32x2' ? 2 : 1));
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `output-${buffers.length}`, buffer, format, length)
    };
  };
  const input = <DataFormat extends Format>(format: DataFormat, values: number[] | Uint32Array) => {
    const buffer = createInputBuffer(
      device,
      format === 'uint32' ? Uint32Array.from(values) : Float32Array.from(values)
    );
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(
        graph,
        `input-${buffers.length}`,
        buffer,
        format,
        values.length / (format === 'float32x2' ? 2 : 1)
      )
    };
  };
  return {graph, input, output, destroy: () => buffers.forEach(buffer => buffer.destroy())};
}

test.each([
  'lowest-row',
  'all'
] as const)('NAM-03 GPU equidistant IDs [99,1] retain feature rows (%s)', {
  timeout: 300000
}, async ties => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  const capacity = ties === 'all' ? 2 : 1;
  const rows = fixture.output('uint32', capacity);
  const ids = fixture.output('uint32', capacity);
  const counts = fixture.output('uint32', 1);
  const distances = fixture.output('float32', capacity);
  const overflow = fixture.output('uint32', 1);
  const candidateOverflow = fixture.output('uint32', 1);
  const props = {
    queries: {kind: 'points', positions: fixture.input('float32x2', [0, 0]).view},
    features: {
      kind: 'points',
      positions: fixture.input('float32x2', [-1, 0, 1, 0]).view,
      sourceIds: fixture.input('uint32', [99, 1]).view
    },
    neighborCount: 1,
    neighborCapacity: capacity,
    ties,
    output: {
      neighborRows: rows.view,
      neighborIds: ids.view,
      neighborCounts: counts.view,
      neighborDistances: distances.view,
      status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
    }
  } as const;
  expect(() => new GPUNearestFeatureJoin({...props, ties: 'lowest-id' as never})).toThrow(
    "ties 'lowest-id' was renamed; use 'lowest-row'"
  );
  fixture.graph.add(new GPUNearestFeatureJoin(props));
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    // Both distances equal 1 exactly; source ID order must never reverse feature-row order.
    expect(await readUint32(counts.buffer, 1)).toEqual([capacity]);
    expect(await readUint32(rows.buffer, capacity)).toEqual(ties === 'all' ? [0, 1] : [0]);
    expect(await readUint32(ids.buffer, capacity)).toEqual(ties === 'all' ? [99, 1] : [99]);
    const actualDistances = await readFloat32(distances.buffer, capacity);
    expect(actualDistances.every(Number.isFinite)).toBe(true);
    expect(actualDistances).toEqual(new Array(capacity).fill(1));
    expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
    expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

test('NAM-04 GPU signed mask placement selects source row five on replay', {
  timeout: 300000
}, async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createFixture(device);
  const maskPlacement = {origin: [100, 230], cellSize: [20, -20]} as const;
  const parameters = fixture.input(
    'uint32',
    getGPUDotDensityParameterValues({seed: 7, maskPlacement})
  );
  const positions = fixture.output('float32x2', 8);
  const ids = fixture.output('uint32', 8);
  const count = fixture.output('uint32', 1);
  const overflow = fixture.output('uint32', 1);
  const failed = fixture.output('uint32', 1);
  const maskWeights = fixture.input('float32', [0, 0, 0, 0, 0, 1]);
  fixture.graph.add(
    new GPURandomPointsInPolygon({
      // This rectangle lies entirely in row 2, column 1 of the asymmetric 2x3 mask.
      positions: fixture.input('float32x2', [120, 170, 140, 170, 140, 190, 120, 190]).view,
      featureOffsets: fixture.input('uint32', [0, 1]).view,
      polygonOffsets: fixture.input('uint32', [0, 1]).view,
      ringOffsets: fixture.input('uint32', [0, 4]).view,
      counts: fixture.input('uint32', [8]).view,
      parameters: parameters.view,
      mask: {weights: maskWeights.view, width: 2, height: 3},
      output: {
        positions: positions.view,
        dots: {ids: ids.view, status: {count: count.view, overflow: overflow.view}},
        status: {failedCount: failed.view}
      }
    })
  );
  const compiled = fixture.graph.compile();
  try {
    for (const seed of [7, 8]) {
      maskWeights.buffer.write(Float32Array.of(0, 0, 0, 0, 0, 1));
      parameters.buffer.write(getGPUDotDensityParameterValues({seed, maskPlacement}));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(count.buffer, 1)).toEqual([8]);
      expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
      expect(await readUint32(failed.buffer, 1)).toEqual([0]);
      const coordinates = await readFloat32(positions.buffer, 16);
      expect(coordinates.every(Number.isFinite)).toBe(true);
      for (let row = 0; row < 8; row++) {
        const column = Math.floor((coordinates[2 * row] - 100) / 20);
        const maskRow = Math.floor((coordinates[2 * row + 1] - 230) / -20);
        expect(maskRow * 2 + column).toBe(5);
      }
      // Ignoring the mask must fail this replay: every requested placement is now rejected.
      maskWeights.buffer.write(new Float32Array(6));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(failed.buffer, 1)).toEqual([8]);
      expect((await readFloat32(positions.buffer, 16)).every(Number.isNaN)).toBe(true);
    }
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
