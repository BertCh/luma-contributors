// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUNetworkIsochrones,
  getGPUNetworkIsochronesParameterValues,
  type GPUNetworkIsochronesSettings
} from '../../../src/gpu-network/network-isochrones/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {buildCSR, type NetworkEdge} from '../network-reachability/network-reachability-oracle';

/** Pixel size of every scene below, in the units of the node positions. */
const PIXEL_SIZE = 10;
/** The default `unreachedCost` as the float32 the raster holds. */
const UNREACHED_COST = Math.fround(1e30);

type RasterRun = {values: Float32Array; bandClasses: Uint32Array};

/** Splats caller-given node costs (no search) and reads the cost raster and the band classes. */
async function runCostRaster(
  device: Device,
  scene: {
    positions: readonly [number, number][];
    edges: readonly NetworkEdge[];
    costs: readonly number[];
    width: number;
    height: number;
    mode: 'min' | 'max';
    costBreaks: readonly number[];
    settings: Omit<GPUNetworkIsochronesSettings, 'breakCount' | 'extent'>;
  }
): Promise<RasterRun> {
  const nodeCount = scene.positions.length;
  const csr = buildCSR(nodeCount, scene.edges);
  const buffers: Buffer[] = [];
  const input = (values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return buffer;
  };
  const pixelCount = scene.width * scene.height;
  const valuesBuffer = createOutputBuffer(device, pixelCount);
  const bandClassesBuffer = createOutputBuffer(device, pixelCount);
  buffers.push(valuesBuffer, bandClassesBuffer);
  const graph = new GPUCommandGraph(device, {id: 'isochrones-max-mode'});
  const parameters = new GPUParameterBuffer(device, {
    id: 'isochrone-parameters',
    format: 'float32',
    length: 12,
    values: getGPUNetworkIsochronesParameterValues({
      ...scene.settings,
      breakCount: scene.costBreaks.length,
      extent: [0, 0, scene.width * PIXEL_SIZE, scene.height * PIXEL_SIZE]
    })
  });
  const breaks = new GPUParameterBuffer(device, {
    id: 'isochrone-breaks',
    format: 'float32',
    length: scene.costBreaks.length,
    values: Float32Array.from(scene.costBreaks)
  });
  graph.add(
    new GPUNetworkIsochrones({
      topology: {
        offsets: importGraphBuffer(graph, 'offsets', input(csr.offsets), 'uint32', nodeCount + 1),
        neighbors: importGraphBuffer(
          graph,
          'neighbors',
          input(csr.neighbors),
          'uint32',
          csr.neighbors.length
        ),
        weights: importGraphBuffer(
          graph,
          'weights',
          input(csr.weights),
          'float32',
          csr.weights.length
        ),
        nodePositions: importGraphBuffer(
          graph,
          'positions',
          input(Float32Array.from(scene.positions.flat())),
          'float32x2',
          nodeCount
        )
      },
      costs: importGraphBuffer(
        graph,
        'costs',
        input(Float32Array.from(scene.costs)),
        'float32',
        nodeCount
      ),
      costBreaks: breaks.importToGraph(graph),
      parameters: parameters.importToGraph(graph),
      raster: {
        width: scene.width,
        height: scene.height,
        mode: scene.mode,
        output: {
          values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', pixelCount),
          bandClasses: importGraphBuffer(
            graph,
            'band-classes',
            bandClassesBuffer,
            'uint32',
            pixelCount
          )
        }
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    values: Float32Array.from(await readFloat32(valuesBuffer, pixelCount)),
    bandClasses: Uint32Array.from(await readUint32(bandClassesBuffer, pixelCount))
  };
  compiled.destroy?.();
  parameters.destroy();
  breaks.destroy();
  for (const buffer of buffers) {
    buffer.destroy();
  }
  return result;
}

/** A road of 100 units along pixel row 0, nodes on the centers of pixels 0 and 10. */
const ROAD_POSITIONS: [number, number][] = [
  [5, 5],
  [105, 5]
];
const ROAD_WEIGHT = 10;
/** Cost per unit of length along the road. */
const ROAD_COST_RATE = ROAD_WEIGHT / 100;

it('GPUNetworkIsochrones max mode gives a two-way road its network cost, low at the start and 10 at the far end (network-paths-01)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Review network-paths-01. One two-way road of weight 10 from the start node (cost 0) to the far
  // node (cost 10). The cost of the road point at distance x from the start is x / 10 whichever
  // direction reaches it. The kernel used to combine the two directed edges with a maximum, so the
  // start pixel read 20 - 0.69 and the far pixel 10.69.
  const scene = {
    positions: ROAD_POSITIONS,
    edges: [
      [0, 1, ROAD_WEIGHT],
      [1, 0, ROAD_WEIGHT]
    ] as NetworkEdge[],
    costs: [0, ROAD_WEIGHT],
    width: 11,
    height: 2,
    costBreaks: [5],
    settings: {}
  };
  const max = await runCostRaster(device, {...scene, mode: 'max'});
  const min = await runCostRaster(device, {...scene, mode: 'min'});
  // With no buffer the splat radius is half a pixel diagonal, 7.0711 units. The road is sampled
  // every 100 / 29 units (30 samples). Max mode keeps the costliest road sample within the radius
  // of the pixel center, min mode the cheapest, so both stay within radius * rate of the true cost
  // of the pixel center, max above it and min below it.
  const radius = 0.5 * Math.hypot(PIXEL_SIZE, PIXEL_SIZE);
  const sampleStep = 100 / 29;
  for (let column = 0; column < 11; column++) {
    const trueCost = column;
    const lastSample = Math.min(29, Math.floor((PIXEL_SIZE * column + radius) / sampleStep));
    const firstSample = Math.max(0, Math.ceil((PIXEL_SIZE * column - radius) / sampleStep));
    expect(max.values[column], `max pixel ${column}`).toBeCloseTo(
      (ROAD_WEIGHT * lastSample) / 29,
      4
    );
    expect(min.values[column], `min pixel ${column}`).toBeCloseTo(
      (ROAD_WEIGHT * firstSample) / 29,
      4
    );
    expect(max.values[column]).toBeGreaterThanOrEqual(trueCost - 1e-4);
    expect(max.values[column]).toBeLessThanOrEqual(trueCost + radius * ROAD_COST_RATE + 1e-4);
    expect(min.values[column]).toBeLessThanOrEqual(max.values[column]);
  }
  // Hand values: the start pixel holds the sample 2 steps along the road, the far pixel the far node.
  expect(max.values[0]).toBeCloseTo(20 / 29, 4);
  expect(max.values[10]).toBeCloseTo(10, 4);
  // A band limit of 5 covers the near half of the road, not the far half.
  expect(Array.from(max.bandClasses.subarray(0, 11))).toEqual([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]);
  // Row 1 is 10 units from the road, outside the radius: unreached, so in the last band.
  expect(Array.from(max.values.subarray(11))).toEqual(new Array(11).fill(UNREACHED_COST));
  expect(Array.from(max.bandClasses.subarray(11))).toEqual(new Array(11).fill(1));
});

it('GPUNetworkIsochrones max mode keeps the one-way cost of a one-way road', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // The far node has a cheaper cost through some other route (4), but the road is one-way from the
  // start, so a road point is only reached from the start: its cost is x / 10, never 4 + ...
  const max = await runCostRaster(device, {
    positions: ROAD_POSITIONS,
    edges: [[0, 1, ROAD_WEIGHT]],
    costs: [0, 4],
    width: 11,
    height: 2,
    mode: 'max',
    costBreaks: [5],
    settings: {}
  });
  expect(max.values[0]).toBeCloseTo(20 / 29, 4);
  expect(max.values[5]).toBeCloseTo((ROAD_WEIGHT * 16) / 29, 4);
  expect(max.values[10]).toBeCloseTo(10, 4);
});

it('GPUNetworkIsochrones max mode takes the maximum over the roads that cover a pixel', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Two-way roads: A from (5, 5) to (105, 5), weight 10; a connector from (105, 5) to (105, 15),
  // weight 1; B from (105, 15) back to (5, 15), weight 10. Start at (5, 5): node costs 0, 10, 11, 21.
  // Road A at x has cost (x - 5) / 10, road B at x has cost 11 + (105 - x) / 10.
  // With a buffer of 12 units and a walk cost of 0.1 per unit, a pixel center is within the buffer
  // of the road in its own row and of the road in the neighboring row (10 units away), and each
  // road is sampled every 6 units.
  const bufferRadius = 12;
  const walkCostPerUnit = 0.1;
  const scene = {
    positions: [
      [5, 5],
      [105, 5],
      [105, 15],
      [5, 15]
    ] as [number, number][],
    edges: [
      [0, 1, 10],
      [1, 0, 10],
      [1, 2, 1],
      [2, 1, 1],
      [2, 3, 10],
      [3, 2, 10]
    ] as NetworkEdge[],
    costs: [0, 10, 11, 21],
    width: 11,
    height: 2,
    costBreaks: [5],
    settings: {bufferRadius, walkCostPerUnit}
  };
  const max = await runCostRaster(device, {...scene, mode: 'max'});
  const min = await runCostRaster(device, {...scene, mode: 'min'});
  // Hand computation for the pixel (column, row). A sample at distance x along a road of 100 units
  // sits at x = 100 * i / 17 (18 samples: ceil(100 / 6) + 1), 5.88 units apart. The costliest
  // sample of the costlier road within 12 units decides max mode; the bound below is the true cost
  // of the nearest road point plus the walk, widened by the buffer on both terms.
  for (let column = 0; column < 11; column++) {
    // Row 0 lies on road A (cost column) and 10 units from road B (cost 21 - column, plus walk).
    const roadBAtColumn = 21 - column;
    const lowerRow0 = roadBAtColumn + walkCostPerUnit * 10;
    expect(max.values[column], `max row 0 column ${column}`).toBeGreaterThanOrEqual(
      lowerRow0 - 1e-3
    );
    expect(max.values[column], `max row 0 column ${column}`).toBeLessThanOrEqual(
      roadBAtColumn + bufferRadius * ROAD_COST_RATE + walkCostPerUnit * bufferRadius + 1e-3
    );
    // Row 1 lies on road B itself.
    expect(max.values[11 + column], `max row 1 column ${column}`).toBeGreaterThanOrEqual(
      roadBAtColumn - 1e-3
    );
    expect(max.values[11 + column], `max row 1 column ${column}`).toBeLessThanOrEqual(
      roadBAtColumn + bufferRadius * ROAD_COST_RATE + walkCostPerUnit * bufferRadius + 1e-3
    );
    // Min mode keeps the cheaper road A for both rows.
    expect(min.values[column]).toBeLessThanOrEqual(column + 1e-3);
    expect(min.values[11 + column]).toBeLessThanOrEqual(column + walkCostPerUnit * 10 + 1e-3);
    expect(min.values[column]).toBeLessThanOrEqual(max.values[column]);
    expect(min.values[11 + column]).toBeLessThanOrEqual(max.values[11 + column]);
  }
  // Exact hand values at the start pixel (column 0, row 0), center (5, 5). Road B samples sit at
  // x = 105 - 100 i / 17 on y = 15 with cost 11 + 10 i / 17. Sample 17 (x = 5) is 10 units away:
  // 21 + 1 = 22. Sample 16 (x = 10.882) is hypot(5.882, 10) = 11.602 away: 20.412 + 1.160 = 21.572.
  // Sample 15 is 15.4 units away, outside the buffer. Road A reaches at most 2 * 10 / 17 + walk.
  expect(max.values[0]).toBeCloseTo(22, 3);
  // Far pixel of row 1 (column 10, center (105, 15)), on node 2 (cost 11). Road B sample 2
  // (x = 93.235, cost 12.176) is 11.765 away: 12.176 + 1.176 = 13.353, the costliest in reach.
  expect(max.values[11 + 10]).toBeCloseTo(11 + 20 / 17 + walkCostPerUnit * (200 / 17), 3);
});

it('GPUNetworkIsochrones raster bands are upper-inclusive, (a, b] (network-paths-07)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  // Roadmap 12.3: cost bands are (a, b], as in GPUNetworkReachability. Two nodes one pixel apart
  // joined by a free two-way road; both have a cost exactly on a break, so every pixel that the
  // road covers holds exactly that cost.
  for (const [cost, expectedBand] of [
    [300, 0],
    [600, 1],
    [601, 2],
    [299, 0]
  ] as const) {
    const run = await runCostRaster(device, {
      positions: [
        [5, 5],
        [15, 5]
      ],
      edges: [
        [0, 1, 0],
        [1, 0, 0]
      ],
      costs: [cost, cost],
      width: 2,
      height: 2,
      mode: 'min',
      costBreaks: [300, 600],
      settings: {}
    });
    // Row 1 is unreached (1e30), which is above every break.
    expect(Array.from(run.values), `cost ${cost}`).toEqual([
      cost,
      cost,
      UNREACHED_COST,
      UNREACHED_COST
    ]);
    expect(Array.from(run.bandClasses), `cost ${cost}`).toEqual([expectedBand, expectedBand, 2, 2]);
  }
});
