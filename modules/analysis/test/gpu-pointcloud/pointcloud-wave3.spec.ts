// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {
  GPUPointCloudNeighborhoods,
  getGPUPointCloudNeighborhoodsParameterValues
} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-neighborhoods';
import {GPUPointCloudFeatures} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-features';
import {GPUPointCloudStatisticalFilter} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-statistical-filter';
import {GPUPointCloudSMRF} from '../../src/pointcloud-analysis/operations/gpu-pointcloud-smrf';
import {GPUTransferMask} from '../../src/computation-record/gpu-transfer-mask';
import {createVectorView} from '../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {makeSpatialFixture, readFloats, readIntegers} from '../spatial-test-utils';
import {expectPointCloudReadback} from './pointcloud-readback-assertions';

for (const strategy of ['serial', 'cooperative'] as const) {
  it(`review B1 rejects overflowing coordinate differences ${strategy}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const indices = fixture.view('uint32', new Uint32Array(1));
      const squaredDistances = fixture.view('float32', new Float32Array(1));
      const counts = fixture.view('uint32', new Uint32Array(1));
      const status = fixture.view('uint32', new Uint32Array(1));
      fixture.graph.add(
        new GPUPointCloudNeighborhoods({
          positions: fixture.view('float32x3', new Float32Array([-3e38, 0, 0, 3e38, 0, 0])).view,
          queryRows: fixture.view('uint32', new Uint32Array([0])).view,
          radius: new Float32Array(new Uint32Array([0x5f7fffff]).buffer)[0],
          capacity: 1,
          strategy,
          output: {
            indices: indices.view,
            squaredDistances: squaredDistances.view,
            counts: counts.view,
            status: status.view
          }
        })
      );
      await fixture.run();
      // Exact separation exceeds 5e38, far beyond the largest legal radius (below 2e19).
      expect(await readIntegers(counts.buffer)).toEqual([0]);
      expect(await readIntegers(indices.buffer)).toEqual([0xffffffff]);
      expect(await readIntegers(squaredDistances.buffer)).toEqual([0x7f7fffff]);
      expect(await readIntegers(status.buffer)).toEqual([0]);
    } finally {
      fixture.destroy();
    }
  });
  it(`OUT-07 exact empty word and extreme capacity-one boundary ${strategy}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const radius = new Float32Array(new Uint32Array([0x5f7fffff]).buffer)[0];
      const positions = fixture.view('float32x3', new Float32Array([0, 0, 0, radius, 0, 0]));
      const indices = fixture.view('uint32', new Uint32Array(1));
      const squaredDistances = fixture.view('float32', new Float32Array(1));
      const counts = fixture.view('uint32', new Uint32Array(1));
      const status = fixture.view('uint32', new Uint32Array(1));
      const queryRows = fixture.view('uint32', new Uint32Array([0]));
      fixture.graph.add(
        new GPUPointCloudNeighborhoods({
          positions: positions.view,
          queryRows: queryRows.view,
          radius,
          capacity: 1,
          strategy,
          output: {
            indices: indices.view,
            squaredDistances: squaredDistances.view,
            counts: counts.view,
            status: status.view
          }
        })
      );
      await fixture.run();
      // Squaring float32 bits 0x5f7fffff rounds to 0x7f7ffffe, just below maximum finite.
      expect(await readIntegers(counts.buffer)).toEqual([1]);
      expect(await readIntegers(indices.buffer)).toEqual([1]);
      expect(await readIntegers(squaredDistances.buffer)).toEqual([0x7f7ffffe]);
      expect(await readIntegers(status.buffer)).toEqual([0]);
      queryRows.buffer.write(new Uint32Array([2]));
      await fixture.run();
      expect(await readIntegers(counts.buffer)).toEqual([0]);
      expect(await readIntegers(indices.buffer)).toEqual([0xffffffff]);
      expect(await readIntegers(squaredDistances.buffer)).toEqual([0x7f7fffff]);
      expect(await readIntegers(status.buffer)).toEqual([1]);
    } finally {
      fixture.destroy();
    }
  });

  for (const radius of [10000, 20000]) {
    it(`FLT-14 exact norm tie at radius ${radius} ${strategy}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = makeSpatialFixture(device);
      try {
        const indices = fixture.view('uint32', new Uint32Array(1));
        const squaredDistances = fixture.view('float32', new Float32Array(1));
        const counts = fixture.view('uint32', new Uint32Array(1));
        const status = fixture.view('uint32', new Uint32Array(1));
        fixture.graph.add(
          new GPUPointCloudNeighborhoods({
            positions: fixture.view(
              'float32x3',
              new Float32Array([
                10000000, 10000000, 0, 10010000, 10000001, 0, 10010000, 10000000, 0
              ])
            ).view,
            queryRows: fixture.view('uint32', new Uint32Array([0])).view,
            radius,
            capacity: 1,
            strategy,
            output: {
              indices: indices.view,
              squaredDistances: squaredDistances.view,
              counts: counts.view,
              status: status.view
            }
          })
        );
        await fixture.run();
        // Exact squared norms are 100000001 and 100000000; their float32 keys coincide.
        expect(await readIntegers(counts.buffer)).toEqual([radius === 10000 ? 1 : 2]);
        expect(await readIntegers(indices.buffer)).toEqual([2]);
        expect(await readFloats(squaredDistances.buffer)).toEqual([100000000]);
        expect(await readIntegers(status.buffer)).toEqual([radius === 10000 ? 0 : 2]);
      } finally {
        fixture.destroy();
      }
    });
  }
}

// Independent float32 expectations from review1-reproduce.py (NumPy sqrt/nextafter), not a CPU twin.
const MISSING_SQUARED_DISTANCE = new Float32Array(new Uint32Array([0x7f7fffff]).buffer)[0];
const DISTANCE_CASES: readonly {
  name: string;
  squared: readonly number[];
  counts: readonly number[];
  capacity: number;
  means: readonly number[];
  validity: readonly number[];
  standardDeviations?: number;
  tolerance?: number;
}[] = [
  {
    name: 'three sqrt(11)',
    squared: [11, 11, 11],
    counts: [1, 1, 1],
    capacity: 1,
    means: [3.316624879837036, 3.316624879837036, 3.316624879837036],
    validity: [1, 1, 1]
  },
  {
    name: 'five 31',
    squared: [961, 961, 961, 961, 961],
    counts: [1, 1, 1, 1, 1],
    capacity: 1,
    means: [31, 31, 31, 31, 31],
    validity: [1, 1, 1, 1, 1]
  },
  {
    name: 'distance 32 control',
    squared: [961, 961, 961, 961, 1024],
    counts: [1, 1, 1, 1, 1],
    capacity: 1,
    means: [31, 31, 31, 31, 32],
    validity: [1, 1, 1, 1, 0]
  },
  {
    name: 'large constant',
    squared: [1e30, 1e30, 1e30],
    counts: [1, 1, 1],
    capacity: 1,
    means: [999999986991104, 999999986991104, 999999986991104],
    validity: [1, 1, 1],
    tolerance: 67108864
  },
  // The exact mean is 31.00000063578288, below the final row by less than one float32 step.
  // Its float32 threshold is 31: an epsilon large enough to hide UDF-10 must fail this control.
  {
    name: 'one float32 step control',
    squared: [961, 961, 961.0001220703125],
    counts: [1, 1, 1],
    capacity: 1,
    means: [31, 31, 31.000001907348633],
    validity: [1, 1, 0]
  },
  {
    name: 'mixed support sqrt(10)',
    // Neighborhoods fills unused slots with the maximum-finite float32 sentinel.
    squared: [
      10,
      MISSING_SQUARED_DISTANCE,
      MISSING_SQUARED_DISTANCE,
      10,
      MISSING_SQUARED_DISTANCE,
      MISSING_SQUARED_DISTANCE,
      10,
      10,
      10
    ],
    counts: [1, 1, 3],
    capacity: 3,
    means: [3.1622776985168457, 3.1622776985168457, 3.1622776985168457],
    validity: [1, 1, 1]
  },
  {
    name: 'large static multiplier',
    squared: [1, 4],
    counts: [1, 1],
    capacity: 1,
    // The population mean is 1.5 and the standard deviation is 0.5.
    standardDeviations: 1e20,
    means: [1, 2],
    validity: [1, 1]
  }
];
for (const expected of DISTANCE_CASES) {
  it(`DGN-04 constant threshold and unequal control ${expected.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const meanDistances = fixture.view('float32', new Float32Array(expected.counts.length));
      const validity = fixture.view('uint32', new Uint32Array(expected.counts.length));
      fixture.graph.add(
        new GPUPointCloudStatisticalFilter({
          squaredDistances: fixture.view('float32', new Float32Array(expected.squared)).view,
          counts: fixture.view('uint32', new Uint32Array(expected.counts)).view,
          capacity: expected.capacity,
          standardDeviations: expected.standardDeviations ?? 0,
          output: {meanDistances: meanDistances.view, validity: validity.view}
        })
      );
      await fixture.run();
      const means = await readFloats(meanDistances.buffer);
      expect(means).toHaveLength(expected.means.length);
      expectPointCloudReadback(means, expected.means, () => expected.tolerance ?? 0);
      expect(await readIntegers(validity.buffer)).toEqual(expected.validity);
    } finally {
      fixture.destroy();
    }
  });
}

for (const chunked of [false, true]) {
  it(`STA-04 carries support completeness through features and filter, chunks ${chunked}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    for (const supportCase of ['generous', 'tight', 'mixed'] as const) {
      const supported = supportCase === 'generous';
      const fixture = makeSpatialFixture(device);
      try {
        const positions = fixture.view(
          'float32x3',
          new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0])
        );
        const indices = fixture.view('uint32', new Uint32Array(16));
        const squaredDistances = fixture.view('float32', new Float32Array(16));
        const counts = fixture.view('uint32', new Uint32Array(4));
        const neighborhoodStatus = fixture.view('uint32', new Uint32Array(4));
        fixture.graph.add(
          new GPUPointCloudNeighborhoods({
            positions: positions.view,
            radius: 2,
            capacity: 4,
            supportBounds: supported
              ? {minimum: [-3, -3, -3], maximum: [4, 4, 4]}
              : supportCase === 'mixed'
                ? {minimum: [-2, -2, -2], maximum: [2.5, 2.5, 3]}
                : {minimum: [-0.1, -0.1, -0.1], maximum: [1.1, 1.1, 0.1]},
            output: {
              indices: indices.view,
              squaredDistances: squaredDistances.view,
              counts: counts.view,
              status: neighborhoodStatus.view
            }
          })
        );
        const normals = fixture.view('float32x3', new Float32Array(12));
        const eigenvalues = fixture.view('float32x3', new Float32Array(12));
        const shapes = fixture.view('float32x3', new Float32Array(12));
        const featureStatus = fixture.view('uint32', new Uint32Array(4));
        const split = <Format extends 'float32x3' | 'uint32'>(
          view: import('@luma.gl/graph/gpu-core').GraphDataView<Format>,
          stride = 1
        ) => {
          if (!chunked) return view;
          const rowBytes = view.format === 'float32x3' ? 12 : 4;
          return createVectorView(
            'split',
            view.format,
            [0, 1].map(chunk =>
              fixture.graph.createDataView(view.buffer, {
                format: view.format,
                length: 2 * stride,
                byteOffset: chunk * 2 * stride * rowBytes
              })
            )
          );
        };
        fixture.graph.add(
          new GPUPointCloudFeatures({
            positions: positions.view,
            indices: split(indices.view, 4),
            counts: split(counts.view),
            neighborhoodStatus: split(neighborhoodStatus.view),
            capacity: 4,
            output: {
              normals: split(normals.view),
              eigenvalues: split(eigenvalues.view),
              shapes: split(shapes.view),
              status: split(featureStatus.view)
            }
          })
        );
        const validity = fixture.view('uint32', new Uint32Array(4));
        const thresholdComplete = fixture.view('uint32', new Uint32Array(1));
        fixture.graph.add(
          new GPUPointCloudStatisticalFilter({
            squaredDistances: squaredDistances.view,
            counts: counts.view,
            neighborhoodStatus: neighborhoodStatus.view,
            capacity: 4,
            output: {
              meanDistances: fixture.view('float32', new Float32Array(4)).view,
              validity: validity.view,
              thresholdComplete: thresholdComplete.view
            }
          })
        );
        await fixture.run();
        expect(await readIntegers(counts.buffer)).toEqual([3, 3, 3, 3]);
        expect(await readIntegers(neighborhoodStatus.buffer)).toEqual(
          supported ? [0, 0, 0, 0] : supportCase === 'mixed' ? [0, 4, 4, 4] : [4, 4, 4, 4]
        );
        expect(await readIntegers(featureStatus.buffer)).toEqual(
          supported ? [0, 0, 0, 0] : supportCase === 'mixed' ? [0, 8, 8, 8] : [8, 8, 8, 8]
        );
        expect(await readIntegers(thresholdComplete.buffer)).toEqual([supported ? 1 : 0]);
        expect(await readIntegers(validity.buffer)).toEqual(Array(4).fill(supported ? 1 : 0));
      } finally {
        fixture.destroy();
      }
    }
  });
}

for (const upstreamFlags of [
  [0, 0, 4],
  [0, 0, 1],
  [2, 2, 2]
]) {
  it(`STA-04 one unsupported query suppresses the global threshold, nearest-K flags ${upstreamFlags}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = makeSpatialFixture(device);
    try {
      const meanDistances = fixture.view('float32', new Float32Array(3));
      const validity = fixture.view('uint32', new Uint32Array(3));
      const thresholdComplete = fixture.view('uint32', new Uint32Array(1));
      fixture.graph.add(
        new GPUPointCloudStatisticalFilter({
          squaredDistances: fixture.view('float32', new Float32Array([1, 4, 10000])).view,
          counts: fixture.view('uint32', new Uint32Array([1, 1, 1])).view,
          neighborhoodStatus: fixture.view('uint32', new Uint32Array(upstreamFlags)).view,
          capacity: 1,
          standardDeviations: 0,
          output: {
            meanDistances: meanDistances.view,
            validity: validity.view,
            thresholdComplete: thresholdComplete.view
          }
        })
      );
      await fixture.run();
      expect(await readFloats(meanDistances.buffer)).toEqual([1, 2, 100]);
      const supported = upstreamFlags[2] === 2;
      expect(await readIntegers(thresholdComplete.buffer)).toEqual([supported ? 1 : 0]);
      // Nearest-K is intentional: without support loss, threshold 103/3 retains 1 and 2.
      expect(await readIntegers(validity.buffer)).toEqual(supported ? [1, 1, 0] : [0, 0, 0]);
    } finally {
      fixture.destroy();
    }
  });
}

it('UDF-03 scatters query validity to source rows before SMRF', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = makeSpatialFixture(device);
  try {
    const positions = fixture.view('float32x3', new Float32Array([0, 0, 0, 1, 0, 0, 10, 0, 0]));
    const queryRows = fixture.view('uint32', new Uint32Array([2, 0, 1]));
    const indices = fixture.view('uint32', new Uint32Array(3));
    const squaredDistances = fixture.view('float32', new Float32Array(3));
    const counts = fixture.view('uint32', new Uint32Array(3));
    const queryMask = fixture.view('uint32', new Uint32Array(3));
    const sourceMask = fixture.view('uint32', new Uint32Array(3));
    const status = fixture.view('uint32', new Uint32Array(3));
    fixture.graph.add(
      new GPUPointCloudNeighborhoods({
        positions: positions.view,
        queryRows: queryRows.view,
        radius: 20,
        capacity: 1,
        output: {
          indices: indices.view,
          squaredDistances: squaredDistances.view,
          counts: counts.view
        }
      })
    );
    fixture.graph.add(
      new GPUPointCloudStatisticalFilter({
        squaredDistances: squaredDistances.view,
        counts: counts.view,
        capacity: 1,
        standardDeviations: 0,
        output: {
          meanDistances: fixture.view('float32', new Float32Array(3)).view,
          validity: queryMask.view
        }
      })
    );
    fixture.graph.add(
      new GPUTransferMask({
        id: 'query-to-source',
        source: queryMask.view,
        relation: {layout: 'index', targets: queryRows.view},
        target: sourceMask.view,
        mode: 'any'
      })
    );
    fixture.graph.add(
      new GPUPointCloudSMRF({
        positions: positions.view,
        classifications: fixture.view('uint32', new Uint32Array([1, 1, 1])).view,
        mask: sourceMask.view,
        grid: {origin: [-1, -1], cellSize: 1, width: 12, height: 3},
        window: 1,
        output: {
          groundSurface: fixture.view('float32', new Float32Array(36)).view,
          groundMask: fixture.view('uint32', new Uint32Array(3)).view,
          outputClassifications: fixture.view('uint32', new Uint32Array(3)).view,
          status: status.view
        }
      })
    );
    await fixture.run();
    // Hand worked nearest distances in query order: [9,1,1], threshold 11/3.
    expect(await readIntegers(queryMask.buffer)).toEqual([0, 1, 1]);
    expect(await readIntegers(sourceMask.buffer)).toEqual([1, 1, 0]);
    const flags = await readIntegers(status.buffer);
    expect(flags.map(value => value & 4)).toEqual([0, 0, 4]);
  } finally {
    fixture.destroy();
  }
});

for (const strategy of ['serial', 'cooperative'] as const) {
  for (const competing of [false, true]) {
    it(`review B1 inclusive norm and smaller-ID retention ${strategy}, competing ${competing}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = makeSpatialFixture(device);
      try {
        const indices = fixture.view('uint32', new Uint32Array(1));
        const squaredDistances = fixture.view('float32', new Float32Array(1));
        const counts = fixture.view('uint32', new Uint32Array(1));
        const status = fixture.view('uint32', new Uint32Array(1));
        fixture.graph.add(
          new GPUPointCloudNeighborhoods({
            positions: fixture.view(
              'float32x3',
              new Float32Array(
                competing ? [0, 0, 0, 15827, 50964, 0, 53365, 0, 0] : [0, 0, 0, 15827, 50964, 0]
              )
            ).view,
            queryRows: fixture.view('uint32', new Uint32Array([0])).view,
            radius: 53365,
            capacity: 1,
            strategy,
            output: {
              indices: indices.view,
              squaredDistances: squaredDistances.view,
              counts: counts.view,
              status: status.view
            }
          })
        );
        await fixture.run();
        // 15827^2 + 50964^2 = 53365^2 = 2847823225 exactly; lower source row wins.
        // Dot and cutoff round in opposite directions: 2847823360 versus 2847823104.
        expect(await readIntegers(counts.buffer)).toEqual([competing ? 2 : 1]);
        expect(await readIntegers(indices.buffer)).toEqual([1]);
        expect(await readFloats(squaredDistances.buffer)).toEqual([2847823360]);
        expect(await readIntegers(status.buffer)).toEqual([competing ? 2 : 0]);
      } finally {
        fixture.destroy();
      }
    });
  }
  for (const runtime of [false, true]) {
    it(`review N1 certifies halo using coordinate margins ${strategy}, runtime ${runtime}`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = makeSpatialFixture(device);
      try {
        const positions = fixture.view('float32x3', new Float32Array([10000000, 0, 0]));
        const status = fixture.view('uint32', new Uint32Array(1));
        const supportBounds = {
          minimum: [9999998, -1, -1] as const,
          maximum: [10000000, 1, 1] as const
        };
        fixture.graph.add(
          new GPUPointCloudNeighborhoods({
            positions: positions.view,
            radius: 0.25,
            capacity: 1,
            strategy,
            supportBounds,
            ...(runtime
              ? {
                  parameters: fixture.view(
                    'float32',
                    getGPUPointCloudNeighborhoodsParameterValues({radius: 0.25, supportBounds})
                  ).view
                }
              : {}),
            output: {
              indices: fixture.view('uint32', new Uint32Array(1)).view,
              squaredDistances: fixture.view('float32', new Float32Array(1)).view,
              counts: fixture.view('uint32', new Uint32Array(1)).view,
              status: status.view
            }
          })
        );
        for (const [coordinate, expected] of [
          [10000000, 4],
          [9999998, 4],
          [9999999, 0]
        ]) {
          positions.buffer.write(new Float32Array([coordinate, 0, 0]));
          await fixture.run();
          // At either bound, the exact margin is zero, which cannot contain a radius of 0.25.
          expect(await readIntegers(status.buffer)).toEqual([expected]);
        }
      } finally {
        fixture.destroy();
      }
    });
  }
}
