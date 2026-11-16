// luma.gl contributors
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getPointCloudDiamondMorphologyNodes} from '../../src/pointcloud-analysis/pointcloud-diamond-morphology';
import {getPointCloudKernelNodes} from '../../src/pointcloud-analysis/pointcloud-kernel-nodes';
import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUPointCloudNormalScale,
  getGPUPointCloudNormalScaleParameterValues,
  GPUPointCloudFeatures,
  GPUPointCloudGround,
  GPUPointCloudICP,
  GPUPointCloudVoxelMerge,
  GPUPointCloudHeight,
  GPUPointCloudNeighborhoods,
  GPUPointCloudSurfaces,
  getGPUPointCloudGridParameterValues,
  getGPUPointCloudHeightParameterValues,
  getGPUPointCloudNeighborhoodsParameterValues,
  getGPUPointCloudSurfacesParameterValues
} from '../../src/pointcloud-analysis';
import {createVectorView} from '../utils/gpu-contributor-test-utils';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {requireTestDevice} from '../utils/require-test-device';

const GRID = {
  origin: [0, 0] as const,
  cellSize: 1,
  width: 3,
  height: 1,
  rowDirection: 'north' as const
};

it('normalizes finite vectors at large and small scales against a unit-vector oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const scales = [1e-30, 1, 3e38];
    const positions = fixture.view(
      'float32x3',
      Float32Array.from(scales.flatMap(scale => [scale, scale / 2, -scale / 4]))
    );
    const normals = fixture.view('float32x3', new Float32Array(9));
    const nodes = getPointCloudKernelNodes(fixture.graph, {
      id: 'extreme-normalization',
      length: 3,
      inputs: {positions: positions.view},
      outputs: {normals: normals.view},
      body: 'writeNormals(index, safeNormal(readPositions(index)));'
    });
    fixture.graph.add({getCommandNodes: () => nodes});
    await fixture.run();
    const expected = [1, 0.5, -0.25].map(value => value / Math.hypot(1, 0.5, -0.25));
    const actual = await readFloats(normals.buffer);
    for (let row = 0; row < 3; row++)
      for (let axis = 0; axis < 3; axis++)
        expect(actual[row * 3 + axis]).toBeCloseTo(expected[axis], 6);
  } finally {
    fixture.destroy();
  }
});

it('samples extreme grid scales with one imported buffer for disjoint writable ranges', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const cellSize of [2e-30, 2e38]) {
    for (const runtime of [false, true]) {
      const fixture = makeSpatialFixture(device);
      try {
        // Import the physical output once. All writable subviews address that handle by offset.
        const shared = fixture.view('uint32', new Uint32Array(66));
        const heights = fixture.graph.createDataView(shared.view.buffer, {
          format: 'float32',
          length: 2
        });
        const status = fixture.graph.createDataView(shared.view.buffer, {
          format: 'uint32',
          length: 2,
          byteOffset: 256
        });
        fixture.graph.add(
          new GPUPointCloudHeight({
            positions: fixture.view(
              'float32x3',
              new Float32Array([
                cellSize / 4,
                cellSize / 4,
                10,
                cellSize * 0.75,
                cellSize * 0.75,
                20
              ])
            ).view,
            grid: {...GRID, width: 1, cellSize},
            ...(runtime
              ? {
                  gridParameters: fixture.view(
                    'float32',
                    getGPUPointCloudGridParameterValues({origin: [0, 0], cellSize})
                  ).view
                }
              : {}),
            terrain: fixture.view('float32', new Float32Array([2])).view,
            coverage: fixture.view('uint32', new Uint32Array([1])).view,
            heights,
            status
          })
        );
        await fixture.run();
        expect((await readFloats(shared.buffer)).slice(0, 2)).toEqual([8, 18]);
        expect(await readIntegers(shared.buffer, 256)).toEqual([0, 0]);
      } finally {
        fixture.destroy();
      }
    }
  }
});

it('updates runtime height radius, power, minimum support and exact coverage words on one graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const pack = (
      radius: number,
      power: number,
      minimumSupport: number,
      coverageBit = 0x80000000
    ) =>
      getGPUPointCloudHeightParameterValues(
        {coverageBit, interpolation: {method: 'idw', radius, power, minimumSupport}},
        GRID
      );
    const parameters = fixture.view('uint32', pack(2, 2, 1), 4);
    const chunks = [
      fixture.view('float32x3', new Float32Array([1, 0.5, 30]), 4),
      fixture.view('float32x3', new Float32Array([3, 0.5, 30, 0.5, 0.5, 30]), 12)
    ];
    const heights = [
      fixture.view('float32', new Float32Array(1), 8),
      fixture.view('float32', new Float32Array(2), 4)
    ];
    const status = [
      fixture.view('uint32', new Uint32Array(1), 12),
      fixture.view('uint32', new Uint32Array(2), 8)
    ];
    fixture.graph.add(
      new GPUPointCloudHeight({
        grid: GRID,
        positions: createVectorView(
          'positions',
          'float32x3',
          chunks.map(chunk => chunk.view)
        ),
        heights: createVectorView(
          'heights',
          'float32',
          heights.map(chunk => chunk.view)
        ),
        status: createVectorView(
          'status',
          'uint32',
          status.map(chunk => chunk.view)
        ),
        terrain: fixture.view('float32', new Float32Array([0, 10, 20]), 4).view,
        coverage: fixture.view('uint32', new Uint32Array(3).fill(0x80000000)).view,
        parameters: parameters.view,
        interpolation: {method: 'idw', radius: 0.1},
        coverageBit: 1
      })
    );
    // At x=1 the three centers have distances 0.5, 0.5, 1.5 and elevations 0, 10, 20.
    for (const {radius, power, minimumSupport, coverageBit, expected, code} of [
      {radius: 0.6, power: 2, minimumSupport: 2, expected: 25, code: 0},
      {radius: 2, power: 2, minimumSupport: 3, expected: 30 - (10 + 20 / 9) / (2 + 1 / 9), code: 0},
      {
        radius: 2,
        power: 4,
        minimumSupport: 3,
        expected: 30 - (10 + 20 / 81) / (2 + 1 / 81),
        code: 0
      },
      {radius: 0.6, power: 2, minimumSupport: 3, expected: 0, code: 2},
      {radius: 2, power: 2, minimumSupport: 1, coverageBit: 1, expected: 0, code: 2}
    ]) {
      parameters.buffer.write(pack(radius, power, minimumSupport, coverageBit), 4);
      await fixture.run();
      expect((await readFloats(heights[0].buffer, 8))[0]).toBeCloseTo(expected, 5);
      expect(await readIntegers(status[0].buffer, 12)).toEqual([code]);
      expect(await readIntegers(status[1].buffer, 8)).toEqual([1, coverageBit === 1 ? 2 : 0]);
      // An exact center wins even if minimumSupport is larger than the available support.
      expect((await readFloats(heights[1].buffer, 4))[1]).toBe(coverageBit === 1 ? 0 : 30);
    }
  } finally {
    fixture.destroy();
  }
});

for (const strategy of ['serial', 'cooperative'] as const) {
  for (const indexed of [false, true]) {
    it(`updates neighborhood radius/support on one graph against CPU distances, ${strategy}, indexed=${indexed}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = makeSpatialFixture(device);
      try {
        const points = [
          [0, 0, 0],
          [1, 0, 0],
          [-1, 0, 0],
          [0, 2, 0]
        ];
        const queryRows = [0, 3, 9];
        const positions = fixture.view('float32x3', new Float32Array(points.flat()));
        const indices = fixture.view('uint32', new Uint32Array(6), 4);
        const squaredDistances = fixture.view('float32', new Float32Array(6));
        const counts = fixture.view('uint32', new Uint32Array(3));
        const status = fixture.view('uint32', new Uint32Array(3));
        const pack = (radius: number, supported: boolean) =>
          getGPUPointCloudNeighborhoodsParameterValues({
            radius,
            ...(supported
              ? {supportBounds: {minimum: [-10, -10, -10] as const, maximum: [10, 10, 10] as const}}
              : {supportBounds: {minimum: [-1, -1, -1] as const, maximum: [1, 1, 1] as const}})
          });
        const parameters = fixture.view('float32', pack(1, true), 4);
        fixture.graph.add(
          new GPUPointCloudNeighborhoods({
            positions: positions.view,
            queryRows: fixture.view('uint32', new Uint32Array(queryRows)).view,
            capacity: 2,
            radius: 0.01,
            parameters: parameters.view,
            strategy,
            ...(indexed
              ? {grid: {origin: [-3, -3, -3] as const, dimensions: [6, 6, 6] as const, cellSize: 1}}
              : {}),
            indices: indices.view,
            squaredDistances: squaredDistances.view,
            counts: counts.view,
            status: status.view
          })
        );
        for (const {radius, supported} of [
          {radius: 1, supported: true},
          {radius: 3, supported: false},
          {radius: 0.5, supported: true}
        ]) {
          parameters.buffer.write(pack(radius, supported), 4);
          await fixture.run();
          const expectedIds: number[] = [];
          const expectedDistances: number[] = [];
          const expectedCounts: number[] = [];
          const expectedStatus: number[] = [];
          for (const queryRow of queryRows) {
            const query = points[queryRow];
            const matches = query
              ? points
                  .map((point, row) => ({
                    row,
                    distance: point.reduce(
                      (sum, value, axis) => sum + (value - query[axis]) ** 2,
                      0
                    )
                  }))
                  .filter(match => match.row !== queryRow && match.distance <= radius ** 2)
                  .sort(
                    (first, second) => first.distance - second.distance || first.row - second.row
                  )
              : [];
            expectedCounts.push(matches.length);
            expectedStatus.push(
              (query ? 0 : 1) | (matches.length > 2 ? 2 : 0) | (supported ? 0 : 4)
            );
            for (let slot = 0; slot < 2; slot++) {
              expectedIds.push(matches[slot]?.row ?? 0xffffffff);
              expectedDistances.push(matches[slot]?.distance ?? Math.fround(3.402823e38));
            }
          }
          expect(await readIntegers(indices.buffer, 4)).toEqual(expectedIds);
          expect(await readFloats(squaredDistances.buffer)).toEqual(expectedDistances);
          expect(await readIntegers(counts.buffer)).toEqual(expectedCounts);
          expect(await readIntegers(status.buffer)).toEqual(expectedStatus);
        }
      } finally {
        fixture.destroy();
      }
    });
  }
}

it('changes the surface ground class at uint32 precision and preserves the reference route', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view(
      'float32x3',
      new Float32Array([0.5, 0.5, 3, 0.5, 0.5, 7, 1.5, 0.5, 11])
    );
    const classifications = fixture.view('uint32', new Uint32Array([0xffffffff, 2, 2]));
    const parameters = fixture.view(
      'uint32',
      getGPUPointCloudSurfacesParameterValues({groundClass: 2}),
      4
    );
    const output = () => ({
      terrain: fixture.view('float32', new Float32Array(3)),
      surface: fixture.view('float32', new Float32Array(3)),
      canopy: fixture.view('float32', new Float32Array(3)),
      coverage: fixture.view('uint32', new Uint32Array(3))
    });
    const runtime = output();
    const reference = output();
    for (const [id, outputs, runtimeParameters] of [
      ['runtime', runtime, parameters.view],
      ['reference', reference, undefined]
    ] as const)
      fixture.graph.add(
        new GPUPointCloudSurfaces({
          id,
          grid: GRID,
          positions: positions.view,
          classifications: classifications.view,
          parameters: runtimeParameters,
          ...(Object.fromEntries(
            Object.entries(outputs).map(([name, field]) => [name, field.view])
          ) as {
            terrain: typeof runtime.terrain.view;
            surface: typeof runtime.surface.view;
            canopy: typeof runtime.canopy.view;
            coverage: typeof runtime.coverage.view;
          })
        })
      );
    for (const groundClass of [2, 0xffffffff, 2]) {
      parameters.buffer.write(getGPUPointCloudSurfacesParameterValues({groundClass}), 4);
      await fixture.run();
      expect(await readFloats(runtime.terrain.buffer)).toEqual(
        groundClass === 2 ? [7, 11, 0] : [3, 0, 0]
      );
      expect(await readFloats(runtime.surface.buffer)).toEqual([7, 11, 0]);
      expect(await readFloats(runtime.canopy.buffer)).toEqual(
        groundClass === 2 ? [0, 0, 0] : [4, 0, 0]
      );
      expect(await readIntegers(runtime.coverage.buffer)).toEqual(
        groundClass === 2 ? [3, 3, 0] : [3, 1, 0]
      );
      if (groundClass === 2) {
        expect(await readFloats(runtime.terrain.buffer)).toEqual(
          await readFloats(reference.terrain.buffer)
        );
        expect(await readFloats(runtime.canopy.buffer)).toEqual(
          await readFloats(reference.canopy.buffer)
        );
      }
    }
  } finally {
    fixture.destroy();
  }
});

it('samples more rows than a core 128 MiB packed positions binding can hold', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    // 11.2 million positions exceed the previous single-binding route (11,184,810 float32x3 rows).
    const chunkLength = 2_800_000;
    const positions = [];
    const heights = [];
    const status = [];
    for (let chunkIndex = 0; chunkIndex < 4; chunkIndex++) {
      const points = new Float32Array(chunkLength * 3);
      for (let row = 0; row < chunkLength; row++) points.set([0.5, 0.5, 7], row * 3);
      positions.push(fixture.view('float32x3', points));
      heights.push(fixture.view('float32', new Float32Array(chunkLength)));
      status.push(fixture.view('uint32', new Uint32Array(chunkLength)));
    }
    fixture.graph.add(
      new GPUPointCloudHeight({
        positions: createVectorView(
          'large-positions',
          'float32x3',
          positions.map(chunk => chunk.view)
        ),
        heights: createVectorView(
          'large-heights',
          'float32',
          heights.map(chunk => chunk.view)
        ),
        status: createVectorView(
          'large-status',
          'uint32',
          status.map(chunk => chunk.view)
        ),
        grid: {...GRID, width: 1},
        terrain: fixture.view('float32', new Float32Array([2])).view,
        coverage: fixture.view('uint32', new Uint32Array([1])).view
      })
    );
    await fixture.run();
    for (let chunkIndex = 0; chunkIndex < 4; chunkIndex++) {
      for (const row of [0, chunkLength - 1]) {
        const heightBytes = await heights[chunkIndex].buffer.readAsync(row * 4, 4);
        const statusBytes = await status[chunkIndex].buffer.readAsync(row * 4, 4);
        expect(new Float32Array(heightBytes.buffer, heightBytes.byteOffset, 1)[0]).toBe(5);
        expect(new Uint32Array(statusBytes.buffer, statusBytes.byteOffset, 1)[0]).toBe(0);
      }
    }
  } finally {
    fixture.destroy();
  }
});

it('merges voxel counts and centroids sharing one alignment window across separate dispatches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const backing = fixture.view('uint32', new Uint32Array(4));
    const counts = fixture.graph.createDataView(backing.view.buffer, {format: 'uint32', length: 1});
    const centroids = fixture.graph.createDataView(backing.view.buffer, {
      format: 'float32x3',
      length: 1,
      byteOffset: 4
    });
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudVoxelMerge({
        partitionPolicy: 'disjoint-owned-rows',
        parts: [
          [2, 1],
          [3, 3]
        ].map(([count, center]) => ({
          counts: fixture.view('uint32', new Uint32Array([count])).view,
          centroids: fixture.view('float32x3', new Float32Array([center, 0, 0])).view,
          representatives: fixture.view('uint32', new Uint32Array([0])).view,
          sourceRowOffset: 0
        })),
        counts,
        centroids,
        representatives: fixture.view('uint32', new Uint32Array(1)).view,
        status: status.view
      })
    );
    await fixture.run();
    expect((await readIntegers(backing.buffer))[0]).toBe(5);
    const centroid = await readFloats(backing.buffer, 4);
    expect(centroid[0]).toBeCloseTo(2.2, 6);
    expect(centroid.slice(1)).toEqual([0, 0]);
    expect(await readIntegers(status.buffer)).toEqual([0]);
  } finally {
    fixture.destroy();
  }
});

it('checks every packed and chunked height at workgroup boundaries in both row directions', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const lengths = [0, 1, 255, 256, 257, 513, 0];
  const totalLength = lengths.reduce((sum, length) => sum + length, 0);
  for (const rowDirection of ['north', 'south'] as const) {
    for (const method of ['cell', 'bilinear', 'idw'] as const) {
      for (const chunked of [false, true]) {
        const fixture = makeSpatialFixture(device);
        try {
          const grid = {...GRID, width: 5, height: 5, rowDirection};
          const terrain = Float32Array.from({length: 25}, (_, cell) => {
            const column = cell % 5;
            const row = Math.floor(cell / 5);
            const northRow = rowDirection === 'north' ? row : 4 - row;
            return 2 * (column + 0.5) + 3 * (northRow + 0.5);
          });
          let rowBase = 0;
          const chunks = (chunked ? lengths : [totalLength]).map((length, chunkIndex) => {
            const data = new Float32Array(length * 3);
            const expected = new Float32Array(length);
            for (let row = 0; row < length; row++) {
              const globalRow = rowBase + row;
              const horizontal = 1 + (globalRow % 3);
              const vertical = 1 + (Math.floor(globalRow / 3) % 3);
              data.set(
                [horizontal, vertical, 2 * horizontal + 3 * vertical + globalRow / 4],
                row * 3
              );
              // Bilinear and symmetric IDW reproduce this plane at integer-coordinate midpoints.
              // Cell sampling uses the center half a cell north and east: its elevation is 2.5 higher.
              expected[row] = globalRow / 4 - (method === 'cell' ? 2.5 : 0);
            }
            rowBase += length;
            const positionOffset = ((chunkIndex * 13) % 63) * 4;
            const heightOffset = ((chunkIndex * 7) % 63) * 4;
            const statusOffset = ((chunkIndex * 11) % 63) * 4;
            return {
              expected,
              heightOffset,
              statusOffset,
              positions: fixture.view('float32x3', data, positionOffset),
              heights: fixture.view('float32', new Float32Array(length), heightOffset),
              status: fixture.view('uint32', new Uint32Array(length), statusOffset)
            };
          });
          fixture.graph.add(
            new GPUPointCloudHeight({
              positions: chunked
                ? createVectorView(
                    'boundary-positions',
                    'float32x3',
                    chunks.map(chunk => chunk.positions.view)
                  )
                : chunks[0].positions.view,
              heights: chunked
                ? createVectorView(
                    'boundary-heights',
                    'float32',
                    chunks.map(chunk => chunk.heights.view)
                  )
                : chunks[0].heights.view,
              status: chunked
                ? createVectorView(
                    'boundary-status',
                    'uint32',
                    chunks.map(chunk => chunk.status.view)
                  )
                : chunks[0].status.view,
              grid,
              terrain: fixture.view('float32', terrain).view,
              coverage: fixture.view('uint32', new Uint32Array(25).fill(1)).view,
              ...(method === 'cell'
                ? {}
                : {
                    interpolation:
                      method === 'bilinear'
                        ? {method}
                        : {method, radius: 0.8, power: 2, minimumSupport: 4}
                  })
            })
          );
          await fixture.run();
          for (const chunk of chunks) {
            if (!chunk.expected.length) continue;
            const actual = await readFloats(chunk.heights.buffer, chunk.heightOffset);
            const status = await readIntegers(chunk.status.buffer, chunk.statusOffset);
            chunk.expected.forEach((expected, row) => {
              expect(actual[row], `${rowDirection}/${method}/${chunked}/${row}`).toBeCloseTo(
                expected,
                5
              );
              expect(status[row]).toBe(0);
            });
          }
        } finally {
          fixture.destroy();
        }
      }
    }
  }
});

for (const metric of ['point-to-point', 'point-to-plane'] as const) {
  it(`reduces more than 256 ICP blocks against a closed-form rigid translation, ${metric}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const points = [
        [0, 1, 1],
        [0, -1, 2],
        [1, 0, -1],
        [2, 0, 1],
        [-1, 1, 0],
        [1, -2, 0]
      ];
      const normals = [
        [1, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 1]
      ];
      const translation = [1 / 32, 1 / 16, -1 / 32];
      const rowCount = 72_000;
      const source = new Float32Array(rowCount * 3);
      for (let row = 0; row < rowCount; row++) source.set(points[row % points.length], row * 3);
      const target = points.map(point => point.map((value, axis) => value + translation[axis]));
      const transform = fixture.view('float32', new Float32Array(16));
      const history = fixture.view('float32x3', new Float32Array(3));
      const status = fixture.view('uint32', new Uint32Array(1));
      const inlierMask = fixture.view('uint32', new Uint32Array(rowCount));
      fixture.graph.add(
        new GPUPointCloudICP({
          source: fixture.view('float32x3', source).view,
          target: fixture.view('float32x3', new Float32Array(target.flat())).view,
          targetNormals: fixture.view('float32x3', new Float32Array(normals.flat())).view,
          metric,
          maximumDistance: 0.5,
          iterations: 1,
          translationTolerance: 1,
          rotationTolerance: 1,
          transform: transform.view,
          history: history.view,
          status: status.view,
          inlierMask: inlierMask.view
        })
      );
      await fixture.run();
      const expected = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...translation, 1];
      const actual = await readFloats(transform.buffer);
      expected.forEach((value, entry) => expect(actual[entry]).toBeCloseTo(value, 4));
      expect(await readIntegers(status.buffer)).toEqual([0]);
      expect((await readIntegers(inlierMask.buffer)).every(value => value === 1)).toBe(true);
      const step = Math.hypot(...translation);
      const residual = metric === 'point-to-point' ? step : step / Math.sqrt(3);
      const actualHistory = await readFloats(history.buffer);
      expect(actualHistory[0]).toBe(rowCount);
      expect(actualHistory[1]).toBeCloseTo(residual, 5);
      expect(actualHistory[2]).toBeCloseTo(step, 4);
    } finally {
      fixture.destroy();
    }
  });
}

it('computes chunked neighborhood covariance, eigenvalues, normals and shapes against a planar oracle', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const lengths = [0, 1, 255, 257, 0];
    const rows = lengths.map((length, index) => ({
      counts: fixture.view('uint32', new Uint32Array(length).fill(4), index * 4),
      indices: fixture.view(
        'uint32',
        Uint32Array.from({length: length * 4}, (_, slot) => slot % 4),
        index * 8
      ),
      normals: fixture.view('float32x3', new Float32Array(length * 3), index * 4),
      eigenvalues: fixture.view('float32x3', new Float32Array(length * 3), index * 8),
      covariance: fixture.view('float32', new Float32Array(length * 6), index * 12),
      shapes: fixture.view('float32x3', new Float32Array(length * 3), index * 16),
      status: fixture.view('uint32', new Uint32Array(length), index * 20)
    }));
    fixture.graph.add(
      new GPUPointCloudFeatures({
        positions: fixture.view(
          'float32x3',
          new Float32Array([1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0])
        ).view,
        capacity: 4,
        counts: createVectorView(
          'feature-counts',
          'uint32',
          rows.map(row => row.counts.view)
        ),
        indices: createVectorView(
          'feature-indices',
          'uint32',
          rows.map(row => row.indices.view)
        ),
        normals: createVectorView(
          'feature-normals',
          'float32x3',
          rows.map(row => row.normals.view)
        ),
        eigenvalues: createVectorView(
          'feature-values',
          'float32x3',
          rows.map(row => row.eigenvalues.view)
        ),
        covariance: createVectorView(
          'feature-covariance',
          'float32',
          rows.map(row => row.covariance.view)
        ),
        shapes: createVectorView(
          'feature-shapes',
          'float32x3',
          rows.map(row => row.shapes.view)
        ),
        status: createVectorView(
          'feature-status',
          'uint32',
          rows.map(row => row.status.view)
        )
      })
    );
    await fixture.run();
    for (const [chunkIndex, length] of lengths.entries()) {
      if (!length) continue;
      const chunk = rows[chunkIndex];
      const normals = await readFloats(chunk.normals.buffer, chunkIndex * 4);
      const eigenvalues = await readFloats(chunk.eigenvalues.buffer, chunkIndex * 8);
      const covariance = await readFloats(chunk.covariance.buffer, chunkIndex * 12);
      const shapes = await readFloats(chunk.shapes.buffer, chunkIndex * 16);
      expect(await readIntegers(chunk.status.buffer, chunkIndex * 20)).toEqual(
        new Array(length).fill(0)
      );
      for (let row = 0; row < length; row++) {
        [0, 0, 1].forEach((value, axis) => expect(normals[row * 3 + axis]).toBeCloseTo(value, 6));
        [0.5, 0.5, 0].forEach((value, axis) =>
          expect(eigenvalues[row * 3 + axis]).toBeCloseTo(value, 6)
        );
        [0.5, 0, 0, 0.5, 0, 0].forEach((value, axis) =>
          expect(covariance[row * 6 + axis]).toBeCloseTo(value, 6)
        );
        [0, 1, 0].forEach((value, axis) => expect(shapes[row * 3 + axis]).toBeCloseTo(value, 6));
      }
    }
  } finally {
    fixture.destroy();
  }
});

it('classifies chunked ground rows with inclusive thresholds and independent class/status expectations', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = [
      new Float32Array(0),
      new Float32Array([4.5, 4.5, 0, 4.5, 4.5, 0.5]),
      new Float32Array([4.5, 4.5, 0.75, 4.5, 4.5, -0.25, 9, 4.5, 0])
    ];
    const rows = positions.map((values, index) => ({
      positions: fixture.view('float32x3', values, index * 4),
      classifications: fixture.view(
        'uint32',
        new Uint32Array(values.length / 3).fill(2),
        index * 8
      ),
      groundMask: fixture.view('uint32', new Uint32Array(values.length / 3), index * 12),
      outputClassifications: fixture.view('uint32', new Uint32Array(values.length / 3), index * 16),
      status: fixture.view('uint32', new Uint32Array(values.length / 3), index * 20)
    }));
    fixture.graph.add(
      new GPUPointCloudGround({
        positions: createVectorView(
          'ground-positions',
          'float32x3',
          rows.map(row => row.positions.view)
        ),
        classifications: createVectorView(
          'ground-input-classes',
          'uint32',
          rows.map(row => row.classifications.view)
        ),
        groundMask: createVectorView(
          'ground-mask',
          'uint32',
          rows.map(row => row.groundMask.view)
        ),
        outputClassifications: createVectorView(
          'ground-output-classes',
          'uint32',
          rows.map(row => row.outputClassifications.view)
        ),
        status: createVectorView(
          'ground-status',
          'uint32',
          rows.map(row => row.status.view)
        ),
        grid: {...GRID, width: 9, height: 9},
        radii: [1, 2],
        initialThreshold: 0.5,
        slope: 0,
        maximumThreshold: 0.5,
        minimumSurface: fixture.view('float32', new Float32Array(81)).view,
        coverage: fixture.view('uint32', new Uint32Array(81).fill(1)).view
      })
    );
    await fixture.run();
    const masks = [],
      classifications = [],
      status = [];
    for (const [index, chunk] of rows.entries()) {
      if (!positions[index].length) continue;
      masks.push(...(await readIntegers(chunk.groundMask.buffer, index * 12)));
      classifications.push(...(await readIntegers(chunk.outputClassifications.buffer, index * 16)));
      status.push(...(await readIntegers(chunk.status.buffer, index * 20)));
    }
    expect(masks).toEqual([1, 1, 0, 1, 0]);
    expect(classifications).toEqual([2, 2, 1, 2, 1]);
    expect(status).toEqual([0, 0, 0, 0, 1]);
  } finally {
    fixture.destroy();
  }
});

it('updates surface and height grid origin/cellSize without rebuilding the graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view('float32x3', new Float32Array([0.5, 0.5, 3, 1.5, 0.5, 5]));
    const gridParameters = fixture.view('float32', getGPUPointCloudGridParameterValues(GRID), 4);
    const terrain = fixture.view('float32', new Float32Array(3));
    const surface = fixture.view('float32', new Float32Array(3));
    const canopy = fixture.view('float32', new Float32Array(3));
    const coverage = fixture.view('uint32', new Uint32Array(3));
    const heights = fixture.view('float32', new Float32Array(2));
    const status = fixture.view('uint32', new Uint32Array(2));
    const grid = {...GRID, origin: [-100, -100] as const, cellSize: 0.25};
    fixture.graph.add(
      new GPUPointCloudSurfaces({
        grid,
        gridParameters: gridParameters.view,
        positions: positions.view,
        classifications: fixture.view('uint32', new Uint32Array([2, 2])).view,
        parameters: fixture.view('uint32', getGPUPointCloudSurfacesParameterValues()).view,
        terrain: terrain.view,
        surface: surface.view,
        canopy: canopy.view,
        coverage: coverage.view
      })
    );
    fixture.graph.add(
      new GPUPointCloudHeight({
        grid,
        gridParameters: gridParameters.view,
        positions: positions.view,
        terrain: fixture.view('float32', new Float32Array([10, 20, 30])).view,
        coverage: fixture.view('uint32', new Uint32Array([1, 1, 1])).view,
        heights: heights.view,
        status: status.view
      })
    );
    for (const entry of [
      {
        origin: [0, 0] as const,
        cellSize: 1,
        terrain: [3, 5, 0],
        surface: [3, 5, 0],
        canopy: [0, 0, 0],
        coverage: [3, 3, 0],
        heights: [-7, -15],
        status: [0, 0]
      },
      {
        origin: [1, 0] as const,
        cellSize: 1,
        terrain: [5, 0, 0],
        surface: [5, 0, 0],
        canopy: [0, 0, 0],
        coverage: [3, 0, 0],
        heights: [0, -5],
        status: [1, 0]
      },
      {
        origin: [0, 0] as const,
        cellSize: 2,
        terrain: [3, 0, 0],
        surface: [5, 0, 0],
        canopy: [2, 0, 0],
        coverage: [3, 0, 0],
        heights: [-7, -5],
        status: [0, 0]
      }
    ]) {
      gridParameters.buffer.write(getGPUPointCloudGridParameterValues(entry), 4);
      await fixture.run();
      expect(await readFloats(terrain.buffer)).toEqual(entry.terrain);
      expect(await readFloats(surface.buffer)).toEqual(entry.surface);
      expect(await readFloats(canopy.buffer)).toEqual(entry.canopy);
      expect(await readIntegers(coverage.buffer)).toEqual(entry.coverage);
      expect(await readFloats(heights.buffer)).toEqual(entry.heights);
      expect(await readIntegers(status.buffer)).toEqual(entry.status);
    }
  } finally {
    fixture.destroy();
  }
});

for (const strategy of ['serial', 'cooperative'] as const) {
  it(`updates indexed neighborhood grid origin/cellSize with radius parameters, ${strategy}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const gridParameters = fixture.view(
        'float32',
        getGPUPointCloudGridParameterValues({origin: [0, 0, 0], cellSize: 1}),
        4
      );
      const counts = fixture.view('uint32', new Uint32Array(3));
      const indices = fixture.view('uint32', new Uint32Array(6));
      const status = fixture.view('uint32', new Uint32Array(3));
      fixture.graph.add(
        new GPUPointCloudNeighborhoods({
          positions: fixture.view(
            'float32x3',
            new Float32Array([0.5, 0.5, 0, 1.5, 0.5, 0, 3.5, 0.5, 0])
          ).view,
          queryRows: fixture.view('uint32', new Uint32Array([0, 1, 2])).view,
          grid: {origin: [100, 100, 100], dimensions: [3, 1, 1], cellSize: 0.25},
          gridParameters: gridParameters.view,
          radius: 0.01,
          parameters: fixture.view(
            'float32',
            getGPUPointCloudNeighborhoodsParameterValues({radius: 3})
          ).view,
          strategy,
          capacity: 2,
          counts: counts.view,
          indices: indices.view,
          status: status.view,
          squaredDistances: fixture.view('float32', new Float32Array(6)).view
        })
      );
      const missing = 0xffffffff;
      for (const entry of [
        {
          origin: [0, 0, 0] as const,
          cellSize: 1,
          counts: [1, 1, 0],
          status: [0, 0, 1],
          indices: [1, missing, 0, missing, missing, missing]
        },
        {
          origin: [1, 0, 0] as const,
          cellSize: 1,
          counts: [0, 1, 1],
          status: [1, 0, 0],
          indices: [missing, missing, 2, missing, 1, missing]
        },
        {
          origin: [0, 0, 0] as const,
          cellSize: 2,
          counts: [2, 2, 2],
          status: [0, 0, 0],
          indices: [1, 2, 0, 2, 1, 0]
        }
      ]) {
        gridParameters.buffer.write(getGPUPointCloudGridParameterValues(entry), 4);
        await fixture.run();
        expect(await readIntegers(counts.buffer)).toEqual(entry.counts);
        expect(await readIntegers(status.buffer)).toEqual(entry.status);
        expect(await readIntegers(indices.buffer)).toEqual(entry.indices);
      }
    } finally {
      fixture.destroy();
    }
  });
}

it('merges padded voxel chunks against count-weighted centroid and representative oracles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const lengths = [0, 1, 255, 257, 0];
    let rowBase = 0;
    const chunks = lengths.map((length, index) => {
      const means = Float32Array.from({length: length * 3}, (_, component) =>
        component % 3 === 0 ? rowBase + Math.floor(component / 3) : 0
      );
      rowBase += length;
      const part = (count: number, shift: number) => ({
        counts: fixture.view('uint32', new Uint32Array(length).fill(count), index * 4),
        centroids: fixture.view(
          'float32x3',
          Float32Array.from(means, (value, component) => (component % 3 === 0 ? value + shift : 0)),
          index * 8
        ),
        representatives: fixture.view('uint32', new Uint32Array(length).fill(1), index * 12)
      });
      return {
        first: part(2, 1),
        second: part(3, 3),
        counts: fixture.view('uint32', new Uint32Array(length), index * 16),
        centroids: fixture.view('float32x3', new Float32Array(length * 3), index * 20),
        representatives: fixture.view('uint32', new Uint32Array(length), index * 24),
        status: fixture.view('uint32', new Uint32Array(length), index * 28)
      };
    });
    fixture.graph.add(
      new GPUPointCloudVoxelMerge({
        partitionPolicy: 'disjoint-owned-rows',
        parts: (['first', 'second'] as const).map((name, index) => ({
          counts: createVectorView(
            `${name}-counts`,
            'uint32',
            chunks.map(chunk => chunk[name].counts.view)
          ),
          centroids: createVectorView(
            `${name}-centroids`,
            'float32x3',
            chunks.map(chunk => chunk[name].centroids.view)
          ),
          representatives: createVectorView(
            `${name}-representatives`,
            'uint32',
            chunks.map(chunk => chunk[name].representatives.view)
          ),
          sourceRowOffset: index === 0 ? 10 : 20
        })),
        counts: createVectorView(
          'voxel-counts',
          'uint32',
          chunks.map(chunk => chunk.counts.view)
        ),
        centroids: createVectorView(
          'voxel-centroids',
          'float32x3',
          chunks.map(chunk => chunk.centroids.view)
        ),
        representatives: createVectorView(
          'voxel-representatives',
          'uint32',
          chunks.map(chunk => chunk.representatives.view)
        ),
        status: createVectorView(
          'voxel-status',
          'uint32',
          chunks.map(chunk => chunk.status.view)
        )
      })
    );
    await fixture.run();
    rowBase = 0;
    for (const [index, length] of lengths.entries()) {
      if (!length) continue;
      const chunk = chunks[index];
      expect(await readIntegers(chunk.counts.buffer, index * 16)).toEqual(
        new Array(length).fill(5)
      );
      expect(await readIntegers(chunk.representatives.buffer, index * 24)).toEqual(
        new Array(length).fill(11)
      );
      expect(await readIntegers(chunk.status.buffer, index * 28)).toEqual(
        new Array(length).fill(0)
      );
      const means = await readFloats(chunk.centroids.buffer, index * 20);
      for (let row = 0; row < length; row++) {
        expect(means[row * 3]).toBeCloseTo(rowBase + row + 2.2, 4);
        expect(means[row * 3 + 1]).toBe(0);
        expect(means[row * 3 + 2]).toBe(0);
      }
      rowBase += length;
    }
  } finally {
    fixture.destroy();
  }
});

it('composes fixed-radius public diamond passes against independent clipped min/max oracles', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const grid = {width: 7, height: 5};
    const values = Float32Array.from({length: 35}, (_, row) => ((row * 17) % 37) - 18);
    const input = fixture.view('float32', values);
    const runs = [];
    for (const radius of [1, 3, 9])
      for (const operation of ['min', 'max'] as const) {
        const identifier = `diamond-${radius}-${operation}`;
        const output = fixture.view('float32', new Float32Array(35));
        const built = getPointCloudDiamondMorphologyNodes(
          fixture.graph,
          identifier,
          input.view,
          grid,
          radius,
          operation
        );
        const nodes = [
          ...built.nodes,
          ...getPointCloudKernelNodes(fixture.graph, {
            id: `${identifier}-publish`,
            length: 35,
            inputs: {valuesIn: built.output},
            outputs: {valuesOut: output.view},
            body: 'writeValuesOut(index, readValuesIn(index));'
          })
        ];
        fixture.graph.add({getCommandNodes: () => nodes});
        const expected = Array.from({length: 35}, (_, cell) => {
          const column = cell % 7,
            row = Math.floor(cell / 7);
          const neighbors = [...values].filter(
            (_, candidate) =>
              Math.abs((candidate % 7) - column) + Math.abs(Math.floor(candidate / 7) - row) <=
              radius
          );
          return operation === 'min' ? Math.min(...neighbors) : Math.max(...neighbors);
        });
        runs.push({output, expected});
      }
    await fixture.run();
    for (const {output, expected} of runs)
      expect(await readFloats(output.buffer)).toEqual(expected);
  } finally {
    fixture.destroy();
  }
});

it('uses runtime grid cell size when choosing bounded IDW support', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const gridParameters = fixture.view('float32', getGPUPointCloudGridParameterValues(GRID));
    const heights = fixture.view('float32', new Float32Array(1));
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudHeight({
        positions: fixture.view('float32x3', new Float32Array([1, 0.5, 30])).view,
        grid: GRID,
        gridParameters: gridParameters.view,
        terrain: fixture.view('float32', new Float32Array([0, 10, 20])).view,
        coverage: fixture.view('uint32', new Uint32Array([1, 1, 1])).view,
        interpolation: {method: 'idw', radius: 1.6, power: 2},
        heights: heights.view,
        status: status.view
      })
    );
    for (const grid of [
      {origin: [0, 0] as const, cellSize: 1},
      {origin: [1, 0] as const, cellSize: 1},
      {origin: [0, 0] as const, cellSize: 2}
    ]) {
      gridParameters.buffer.write(getGPUPointCloudGridParameterValues(grid));
      await fixture.run();
      let weightSum = 0,
        elevationSum = 0;
      for (let column = 0; column < 3; column++) {
        const horizontal = grid.origin[0] + (column + 0.5) * grid.cellSize - 1;
        const vertical = grid.origin[1] + 0.5 * grid.cellSize - 0.5;
        const squaredDistance = horizontal ** 2 + vertical ** 2;
        if (squaredDistance > 1.6 ** 2) continue;
        const weight = 1 / squaredDistance;
        weightSum += weight;
        elevationSum += column * 10 * weight;
      }
      expect(await readIntegers(status.buffer)).toEqual([0]);
      expect((await readFloats(heights.buffer))[0]).toBeCloseTo(30 - elevationSum / weightSum, 5);
    }
  } finally {
    fixture.destroy();
  }
});

it('updates the normal-scale search-grid affine on one graph', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const pack = (origin: readonly [number, number, number]) =>
      getGPUPointCloudGridParameterValues({origin, cellSize: 1});
    const gridParameters = fixture.view('float32', pack([-2, -2, -2]));
    const normals = fixture.view('float32x3', new Float32Array(3));
    const normalScales = fixture.view('float32', new Float32Array(1));
    const normalVariances = fixture.view('float32', new Float32Array(1));
    const status = fixture.view('uint32', new Uint32Array(1));
    fixture.graph.add(
      new GPUPointCloudNormalScale({
        positions: fixture.view(
          'float32x3',
          new Float32Array([0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 0, 0, -1, 0])
        ).view,
        queryRows: fixture.view('uint32', new Uint32Array([0])).view,
        radii: [1.1],
        capacity: 4,
        grid: {origin: [-2, -2, -2], dimensions: [4, 4, 4], cellSize: 1},
        gridParameters: gridParameters.view,
        parameters: fixture.view('float32', getGPUPointCloudNormalScaleParameterValues()).view,
        normals: normals.view,
        normalScales: normalScales.view,
        normalVariances: normalVariances.view,
        status: status.view
      })
    );
    for (const supported of [true, false, true]) {
      gridParameters.buffer.write(pack(supported ? [-2, -2, -2] : [10, 10, 10]));
      await fixture.run();
      const normal = await readFloats(normals.buffer);
      [0, 0, supported ? 1 : 0].forEach((value, axis) =>
        expect(normal[axis]).toBeCloseTo(value, 6)
      );
      expect((await readFloats(normalScales.buffer))[0]).toBeCloseTo(supported ? 1.1 : 0, 6);
      expect(await readFloats(normalVariances.buffer)).toEqual([0]);
      expect(await readIntegers(status.buffer)).toEqual([supported ? 0 : 1]);
    }
  } finally {
    fixture.destroy();
  }
});
