// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUPointCloudNeighborhoods} from '../../src/gpu-pointcloud/gpu-pointcloud-neighborhoods';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

function getExpected(
  positions: Float32Array,
  queries: Uint32Array,
  capacity: number,
  indexed: boolean
) {
  const pointCount = positions.length / 3;
  const indices: number[] = [];
  const squaredDistances: number[] = [];
  const counts: number[] = [];
  const status: number[] = [];
  const point = (index: number) => Array.from(positions.subarray(index * 3, index * 3 + 3));
  const inside = (position: number[]) =>
    position.every(value => Number.isFinite(value) && value >= -5 && value < 5);
  for (const queryRow of queries) {
    const position = point(Math.min(queryRow, Math.max(pointCount - 1, 0)));
    const valid =
      queryRow < pointCount && position.every(Number.isFinite) && (!indexed || inside(position));
    const support: {index: number; distance: number}[] = [];
    if (valid) {
      for (let candidate = 0; candidate < pointCount; candidate++) {
        if (candidate === queryRow) continue;
        const candidatePosition = point(candidate);
        if (indexed && !inside(candidatePosition)) continue;
        const distance = candidatePosition.reduce(
          (total, value, axis) => total + (value - position[axis]) ** 2,
          0
        );
        if (distance <= 4) support.push({index: candidate, distance});
      }
    }
    support.sort((first, second) => first.distance - second.distance || first.index - second.index);
    counts.push(support.length);
    const complete = position.every(value => value - 2 >= -5 && value + 2 <= 5);
    status.push((valid ? 0 : 1) | (support.length > capacity ? 2 : 0) | (complete ? 0 : 4));
    for (let slot = 0; slot < capacity; slot++) {
      indices.push(support[slot]?.index ?? 0xffffffff);
      squaredDistances.push(support[slot]?.distance ?? Math.fround(3.402823e38));
    }
  }
  return {indices, squaredDistances, counts, status};
}

async function checkCooperative(
  capacity: number,
  indexed: boolean,
  boundedDispatch = false,
  featureLevel: 'core' | 'max' = 'max'
) {
  const device = await getWebGPUTestDevice(featureLevel);
  requireTestDevice(device);
  const descriptor = Object.getOwnPropertyDescriptor(device, 'limits');
  if (boundedDispatch) {
    const limits = device.limits;
    Object.defineProperty(device, 'limits', {
      configurable: true,
      value: new Proxy(limits, {
        get: (target, property) =>
          property === 'maxComputeWorkgroupsPerDimension'
            ? 2
            : Reflect.get(target, property, target)
      })
    });
  }
  const fixture = makeSpatialFixture(device);
  try {
    const values = new Float32Array(257 * 3);
    for (let index = 0; index < 257; index++)
      values.set(
        [((index * 17) % 9) - 4, ((index * 31) % 9) - 4, ((index * 7) % 5) - 2],
        index * 3
      );
    values.set([Number.POSITIVE_INFINITY, 0, 0], 254 * 3);
    values.set([Number.NaN, 0, 0], 255 * 3);
    values.set([50, 0, 0], 256 * 3);
    const queryValues = new Uint32Array([0, 254, 7, 128, 255, 256, 257]);
    const positions = fixture.view('float32x3', values, 16);
    const queryRows = fixture.view('uint32', queryValues, 16);
    const outputs = (['serial', 'cooperative'] as const).map(strategy => {
      const indices = fixture.view('uint32', new Uint32Array(queryValues.length * capacity), 16);
      const squaredDistances = fixture.view(
        'float32',
        new Float32Array(queryValues.length * capacity),
        16
      );
      const counts = fixture.view('uint32', new Uint32Array(queryValues.length), 16);
      const status = fixture.view('uint32', new Uint32Array(queryValues.length), 16);
      fixture.graph.add(
        new GPUPointCloudNeighborhoods({
          id: strategy,
          positions: positions.view,
          queryRows: queryRows.view,
          capacity,
          radius: 2,
          strategy,
          ...(indexed
            ? {
                grid: {
                  origin: [-5, -5, -5] as const,
                  dimensions: [10, 10, 10] as const,
                  cellSize: 1
                }
              }
            : {}),
          supportBounds: {minimum: [-5, -5, -5], maximum: [5, 5, 5]},
          indices: indices.view,
          squaredDistances: squaredDistances.view,
          counts: counts.view,
          status: status.view
        })
      );
      return {indices, squaredDistances, counts, status};
    });
    for (const mode of ['original', 'coincident', 'outside', 'restored'] as const) {
      const nextPositions = values.slice();
      const nextQueries = queryValues.slice();
      if (mode === 'coincident') nextPositions.fill(0);
      if (mode === 'outside')
        for (let index = 0; index < 257; index++) nextPositions.set([100, 0, 0], index * 3);
      if (mode === 'restored') nextQueries.reverse();
      positions.buffer.write(nextPositions, 16);
      queryRows.buffer.write(nextQueries, 16);
      await fixture.run();
      const expected = getExpected(nextPositions, nextQueries, capacity, indexed);
      for (const output of outputs) {
        expect(await readIntegers(output.indices.buffer, 16)).toEqual(expected.indices);
        expect(await readFloats(output.squaredDistances.buffer, 16)).toEqual(
          expected.squaredDistances
        );
        expect(await readIntegers(output.counts.buffer, 16)).toEqual(expected.counts);
        expect(await readIntegers(output.status.buffer, 16)).toEqual(expected.status);
      }
    }
  } finally {
    if (boundedDispatch) {
      if (descriptor) Object.defineProperty(device, 'limits', descriptor);
      else Reflect.deleteProperty(device, 'limits');
    }
    fixture.destroy();
  }
}

for (let capacity = 1; capacity <= 64; capacity++) {
  for (const indexed of [false, true]) {
    it(`cooperative neighborhoods preserve exact support and ordered outputs at capacity ${capacity}, grid ${indexed}`, async () => {
      await checkCooperative(capacity, indexed);
    });
  }
}

for (const indexed of [false, true]) {
  it(`cooperative neighborhoods handle a padded 3D dispatch, grid ${indexed}`, async () => {
    await checkCooperative(7, indexed, true);
  });
}

for (const pointCount of [0, 1, 2, 129]) {
  for (const indexed of [false, true]) {
    it(`cooperative neighborhoods initialize optional outputs with ${pointCount} source rows, grid ${indexed}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = makeSpatialFixture(device);
      try {
        const positions = fixture.view('float32x3', new Float32Array(pointCount * 3));
        const indices = fixture.view('uint32', new Uint32Array(pointCount * 64));
        const distances = fixture.view('float32', new Float32Array(pointCount * 64));
        const counts = fixture.view('uint32', new Uint32Array(pointCount));
        fixture.graph.add(
          new GPUPointCloudNeighborhoods({
            positions: positions.view,
            capacity: 64,
            radius: 2,
            strategy: 'cooperative',
            ...(indexed
              ? {grid: {origin: [-1, -1, -1] as const, dimensions: [2, 2, 2] as const, cellSize: 1}}
              : {}),
            indices: indices.view,
            squaredDistances: distances.view,
            counts: counts.view
          })
        );
        for (const invalid of [false, true]) {
          if (invalid && pointCount)
            positions.buffer.write(new Float32Array(pointCount * 3).fill(Number.NaN));
          await fixture.run();
          const expectedIndices: number[] = [];
          const expectedDistances: number[] = [];
          for (let query = 0; query < pointCount; query++) {
            const neighbors = invalid
              ? []
              : Array.from({length: pointCount}, (_, index) => index).filter(
                  index => index !== query
                );
            for (let slot = 0; slot < 64; slot++) {
              expectedIndices.push(neighbors[slot] ?? 0xffffffff);
              expectedDistances.push(slot < neighbors.length ? 0 : Math.fround(3.402823e38));
            }
          }
          expect((await readIntegers(indices.buffer)).slice(0, pointCount * 64)).toEqual(
            expectedIndices
          );
          expect((await readFloats(distances.buffer)).slice(0, pointCount * 64)).toEqual(
            expectedDistances
          );
          expect((await readIntegers(counts.buffer)).slice(0, pointCount)).toEqual(
            Array(pointCount).fill(invalid ? 0 : Math.max(pointCount - 1, 0))
          );
        }
      } finally {
        fixture.destroy();
      }
    });
  }
}

for (const indexed of [false, true]) {
  it(`cooperative neighborhoods preserve inclusive float32 radius boundaries, grid ${indexed}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const outside = new Float32Array(new Uint32Array([0x3f800001]).buffer)[0];
      const inside = new Float32Array(new Uint32Array([0x3f7fffff]).buffer)[0];
      const positions = fixture.view(
        'float32x3',
        new Float32Array([0, 0, 0, 1, 0, 0, outside, 0, 0, inside, 0, 0, -0, 0, 0])
      );
      const queryRows = fixture.view('uint32', new Uint32Array([0]));
      const indices = fixture.view('uint32', new Uint32Array(4));
      const distances = fixture.view('float32', new Float32Array(4));
      const counts = fixture.view('uint32', new Uint32Array(1));
      const status = fixture.view('uint32', new Uint32Array(1));
      fixture.graph.add(
        new GPUPointCloudNeighborhoods({
          positions: positions.view,
          queryRows: queryRows.view,
          capacity: 4,
          radius: 1,
          strategy: 'cooperative',
          ...(indexed
            ? {grid: {origin: [-1, -1, -1] as const, dimensions: [5, 4, 4] as const, cellSize: 0.5}}
            : {}),
          indices: indices.view,
          squaredDistances: distances.view,
          counts: counts.view,
          status: status.view
        })
      );
      await fixture.run();
      expect(await readIntegers(indices.buffer)).toEqual([4, 3, 1, 0xffffffff]);
      expect(await readFloats(distances.buffer)).toEqual([
        0,
        Math.fround(inside * inside),
        1,
        Math.fround(3.402823e38)
      ]);
      expect(await readIntegers(counts.buffer)).toEqual([3]);
      expect(await readIntegers(status.buffer)).toEqual([0]);
    } finally {
      fixture.destroy();
    }
  });
}

for (const capacity of [8, 64]) {
  it(`cooperative neighborhoods preserve exact outputs at core device limits, capacity ${capacity}`, async () => {
    await checkCooperative(capacity, true, false, 'core');
  });
}

for (const capacity of [1, 4, 16, 64]) {
  for (const indexed of [false, true]) {
    it(`cooperative neighborhoods preserve noninteger 3D distances, capacity ${capacity}, grid ${indexed}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = makeSpatialFixture(device);
      try {
        let randomState = 20261024;
        const positions = Float32Array.from({length: 1000 * 3}, () => {
          randomState = (Math.imul(randomState, 1664525) + 1013904223) >>> 0;
          return (randomState / 0x100000000) * 8 - 4;
        });
        const queries = new Uint32Array([0, 17, 131, 255, 510, 765, 998, 999]);
        const source = fixture.view('float32x3', positions, 16);
        const queryRows = fixture.view('uint32', queries, 16);
        const outputs = (['serial', 'cooperative'] as const).map(strategy => {
          const indices = fixture.view('uint32', new Uint32Array(queries.length * capacity), 16);
          const squaredDistances = fixture.view(
            'float32',
            new Float32Array(queries.length * capacity),
            16
          );
          const counts = fixture.view('uint32', new Uint32Array(queries.length), 16);
          const status = fixture.view('uint32', new Uint32Array(queries.length), 16);
          fixture.graph.add(
            new GPUPointCloudNeighborhoods({
              id: strategy,
              positions: source.view,
              queryRows: queryRows.view,
              capacity,
              radius: 2,
              strategy,
              ...(indexed
                ? {
                    grid: {
                      origin: [-5, -5, -5] as const,
                      dimensions: [10, 10, 10] as const,
                      cellSize: 1
                    }
                  }
                : {}),
              supportBounds: {minimum: [-5, -5, -5], maximum: [5, 5, 5]},
              indices: indices.view,
              squaredDistances: squaredDistances.view,
              counts: counts.view,
              status: status.view
            })
          );
          return {indices, squaredDistances, counts, status};
        });
        await fixture.run();
        const expected = getExpected(positions, queries, capacity, indexed);
        const distanceOutputs = [];
        for (const output of outputs) {
          expect(await readIntegers(output.indices.buffer, 16)).toEqual(expected.indices);
          expect(await readIntegers(output.counts.buffer, 16)).toEqual(expected.counts);
          expect(await readIntegers(output.status.buffer, 16)).toEqual(expected.status);
          const actualDistances = await readFloats(output.squaredDistances.buffer, 16);
          distanceOutputs.push(actualDistances);
          for (const [slot, distance] of actualDistances.entries()) {
            if (expected.indices[slot] === 0xffffffff) {
              expect(distance).toBe(Math.fround(3.402823e38));
            } else {
              // Independent f64 distances from the same quantized inputs allow f32 arithmetic error.
              expect(Math.abs(distance - expected.squaredDistances[slot])).toBeLessThanOrEqual(
                Math.max(1e-8, expected.squaredDistances[slot] * 1e-5)
              );
            }
          }
        }
        expect(distanceOutputs[1]).toEqual(distanceOutputs[0]);
      } finally {
        fixture.destroy();
      }
    });
  }
}
