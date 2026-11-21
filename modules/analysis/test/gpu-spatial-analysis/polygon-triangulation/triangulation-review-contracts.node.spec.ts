// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {expect, it} from 'vitest';
import {
  CPUPolygonTriangulationOperation,
  computePolygonTriangulationOnCPU
} from '../../../src/analysis-core/polygon-triangulation/cpu-polygon-triangulation-operation';
import {CPUConstrainedDelaunayTriangulationOperation} from '../../../src/gpu-spatial-analysis/polygon-triangulation/cpu-constrained-delaunay-triangulation-operation';
import {GPUPolygonTriangulationOperation} from '../../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation-operation';
import {
  GPUPolygonTriangulation,
  validatePolygonTriangulationWorkLimits
} from '../../../src/analysis-core/polygon-triangulation/gpu-polygon-triangulation';

const input = {
  positions: Float32Array.of(0, 0, 2, 0, 2, 2, 0, 2),
  ringOffsets: Uint32Array.of(0, 3),
  polygonOffsets: Uint32Array.of(0, 1),
  featureOffsets: Uint32Array.of(0, 1)
};

for (const Operation of [
  CPUPolygonTriangulationOperation,
  CPUConstrainedDelaunayTriangulationOperation
]) {
  it(`${Operation.name} retains and enforces compact host storage`, async () => {
    // 4 positions with terminal ring offset 3 leaves one orphan row: compact must reject.
    const props = {
      positions: 'positions',
      featureOffsets: 'featureOffsets',
      polygonOffsets: 'polygonOffsets',
      ringOffsets: 'ringOffsets',
      polygonCount: 1,
      indexCapacity: 6,
      storage: 'compact' as const,
      output: {indices: 'indices', valid: 'valid', status: {converged: 'converged'}}
    };
    const operation = new Operation(props);
    // Mutation after construction must not switch validation back to resident behavior.
    Object.assign(props, {storage: 'resident-capacity'});
    const inputs = Object.fromEntries(
      Object.entries(input).map(([name, values]) => {
        const format = name === 'positions' ? ('float32x2' as const) : ('uint32' as const);
        const length = values.length / (name === 'positions' ? 2 : 1);
        return [name, {format, length, chunks: [{values, length}], pack: () => values}];
      })
    );
    await expect(Promise.resolve().then(() => operation.execute({inputs}))).rejects.toThrow(
      /ringOffsets.*terminal.*4/
    );
    const resident = await new Operation({...props, storage: 'resident-capacity'}).execute({
      inputs
    });
    expect(resident['valid']).toEqual(Uint32Array.of(1));
  });
}

it('rejects null routing rather than selecting the monotone default', () => {
  expect(() =>
    validatePolygonTriangulationWorkLimits('work-check', {largePolygonMethod: null as never})
  ).toThrow(/work-check largePolygonMethod.*monotone.*none.*null/);
});

it.each([
  ['largePolygonMethod', 'unknown'],
  ['largePolygonMethod', null],
  ['monotoneMinimumVertices', NaN],
  ['monotoneMaximumVertices', 0]
] as const)('validates named polygon operation option %s=%s before lowering', (name, value) => {
  expect(
    () =>
      new GPUPolygonTriangulationOperation({
        id: 'named-work',
        positions: 'positions',
        ringOffsets: 'rings',
        polygonOffsets: 'polygons',
        featureOffsets: 'features',
        polygonCount: 1,
        indexCapacity: 6,
        output: {indices: 'indices', valid: 'valid'},
        [name]: value
      } as never)
  ).toThrow(new RegExp(`named-work ${name}`));
});

it('validates resident offset levels once each without losing raw monotonicity', () => {
  const reads: Record<string, number> = {ringOffsets: 0, polygonOffsets: 0, featureOffsets: 0};
  const track = (name: string, values: number[]): Uint32Array => {
    const target = Uint32Array.from(values);
    const proxy: Uint32Array = new Proxy(target, {
      get(array, property) {
        if (property === Symbol.iterator)
          return function* () {
            for (let index = 0; index < array.length; index++) yield proxy[index];
          };
        if (typeof property === 'string' && /^\d+$/.test(property)) reads[name]++;
        return Reflect.get(array, property, array);
      }
    });
    return proxy;
  };
  const result = computePolygonTriangulationOnCPU(
    {
      positions: new Float32Array(),
      ringOffsets: track('ringOffsets', [0]),
      polygonOffsets: track('polygonOffsets', [0]),
      featureOffsets: track('featureOffsets', [0, 0])
    },
    {indexCapacity: 0}
  );
  expect(Array.from(result.triangleOffsets)).toEqual([0]);
  // Zero polygons means no geometry walk; each offset word is read only by validation.
  expect(reads).toEqual({ringOffsets: 1, polygonOffsets: 1, featureOffsets: 2});
  expect(() =>
    computePolygonTriangulationOnCPU(
      {...input, ringOffsets: Uint32Array.of(0, 10, 9)},
      {indexCapacity: 12}
    )
  ).toThrow(/ringOffsets/);
});

it('allows zero polygons with unused resident position capacity and zero index capacity', () => {
  const result = computePolygonTriangulationOnCPU(
    {
      positions: Float32Array.of(0, 0, 1, 1),
      ringOffsets: Uint32Array.of(0),
      polygonOffsets: Uint32Array.of(0),
      featureOffsets: Uint32Array.of(0)
    },
    {indexCapacity: 0}
  );
  expect(Array.from(result.indices)).toEqual([]);
  expect(Array.from(result.triangleCount)).toEqual([]);
  expect(Array.from(result.triangleOffsets)).toEqual([0]);
});

it('preserves resident ring-end clamping and the original padded slice capacity', () => {
  const result = computePolygonTriangulationOnCPU(
    {...input, ringOffsets: Uint32Array.of(0, 7)},
    {indexCapacity: 15}
  );
  // Four resident vertices tile the square with two triangles; the raw ring end reserves five.
  expect(Array.from(result.valid)).toEqual([1]);
  expect(Array.from(result.triangleCount)).toEqual([2]);
  expect(Array.from(result.triangleOffsets)).toEqual([0, 5]);
  expect(Array.from(result.indices.subarray(6))).toEqual(new Array(9).fill(0));
});

it('accepts the capacity helper zero in the direct GPU polygon constructor', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  try {
    const contributor = new GPUPolygonTriangulation({
      positions: createTransientView(graph, 'positions', 'float32x2', 2),
      ringOffsets: createTransientView(graph, 'rings', 'uint32', 1),
      polygonOffsets: createTransientView(graph, 'polygons', 'uint32', 1),
      featureOffsets: createTransientView(graph, 'features', 'uint32', 1),
      output: {
        indices: createTransientView(graph, 'indices', 'uint32', 0),
        valid: createTransientView(graph, 'valid', 'uint32', 0)
      }
    });
    expect(contributor.indexCount).toBe(0);
  } finally {
    device.destroy();
  }
});
