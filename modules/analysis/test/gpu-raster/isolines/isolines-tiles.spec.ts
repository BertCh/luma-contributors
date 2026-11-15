// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer, Device} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {GPUIsolines} from '../../../src/raster-analysis/isolines/gpu-isolines';
import {
  GPUIsolinesOperation,
  type IsolinesOperationProps
} from '../../../src/raster-analysis/isolines/gpu-isolines-operation';
import {CPUIsolinesOperation} from '../../../src/raster-analysis/isolines/cpu-isolines-operation';
import {GPUSegmentIntersection} from '../../../src/gpu-spatial-analysis/segment-intersection/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  expectSameSegments,
  expectSkimageSegments,
  getSegmentRows,
  getTileScenes,
  getWholeScene,
  getLargeOffsetScene,
  tileFixtures,
  type IsolinesTileFixture,
  type IsolinesTileScene
} from './isolines-tiles-test-utils';

const operationOutputs = {
  segments: 'segments',
  segmentLevels: 'segmentLevels',
  segmentEdges: 'segmentEdges',
  status: {
    count: 'count',
    requiredCount: 'requiredCount',
    overflow: 'overflow',
    polylineOverflow: 'polylineOverflow'
  },
  polylines: {
    vertexCapacity: 512,
    vertices: 'vertices',
    polylineOffsets: 'polylineOffsets',
    polylineLevels: 'polylineLevels',
    polylineClosed: 'polylineClosed',
    polylineCount: 'polylineCount',
    vertexCount: 'vertexCount'
  }
};

const outputNames = [
  'segments',
  'segmentLevels',
  'segmentEdges',
  'count',
  'requiredCount',
  'overflow',
  'polylineOverflow',
  'vertices',
  'polylineOffsets',
  'polylineLevels',
  'polylineClosed',
  'polylineCount',
  'vertexCount'
];

async function runTileOperation(
  device: Device,
  scene: IsolinesTileScene,
  capacity = 256,
  cpu = false
) {
  const props: IsolinesOperationProps = {
    grid: scene.grid,
    tile: scene.tile,
    values: 'values',
    mask: 'mask',
    levels: 'levels',
    parameters: 'parameters',
    segmentCapacity: capacity,
    output: operationOutputs
  };
  const Operation = cpu ? CPUIsolinesOperation : GPUIsolinesOperation;
  const result = await runOperationGraph(device, {
    inputs: {
      values: {format: 'float32', values: scene.values},
      mask: {format: 'uint32', values: scene.mask},
      levels: {format: 'float32', values: scene.levels},
      parameters: {format: 'float32', values: scene.parameters}
    },
    operations: [new Operation(props)],
    outputs: outputNames
  });
  return Object.assign(result, {
    count: result['count'],
    segments: result['segments'],
    segmentLevels: result['segmentLevels'],
    segmentEdges: result['segmentEdges']
  });
}

for (const fixture of tileFixtures) {
  it(`GPU tile ownership matches whole-domain and skimage: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const whole = await runTileOperation(device, getWholeScene(fixture));
    const wholeRows = getSegmentRows(whole);
    const tiles = [];
    for (const scene of getTileScenes(fixture)) {
      const result = await runTileOperation(device, scene);
      expect(result['overflow'][0]).toBe(0);
      expect(result['polylineOverflow'][0]).toBe(0);
      tiles.push(...getSegmentRows(result));
    }
    expectSameSegments(tiles, wholeRows);
    expectSkimageSegments(tiles, fixture);
    // A canonical domain edge/level crossing has bit-identical endpoint copies across tiles.
    const crossings = new Map<string, number[]>();
    for (const row of tiles)
      for (const endpoint of [0, 1]) {
        const key = `${row[0]}:${row[1 + endpoint]}`;
        const point = row.slice(3 + endpoint * 2, 5 + endpoint * 2);
        const previous = crossings.get(key);
        if (previous) expect(point).toEqual(previous);
        crossings.set(key, point);
      }
  });
}

for (const mode of ['ordinary', 'empty', 'overflow', 'equality'] as const) {
  it(`tiled GPU and CPU operations agree on every defined port: ${mode}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const scene = getTileScenes({
      ...tileFixtures[0],
      levels: mode === 'equality' ? [9] : tileFixtures[0].levels
    })[mode === 'equality' ? 0 : 3];
    if (mode === 'empty') scene.tile = {...scene.tile!, ownedCells: [1, 1, 1, 2]};
    const capacity = mode === 'overflow' ? 1 : 256;
    const gpu = await runTileOperation(device, scene, capacity);
    const cpu = await runTileOperation(device, scene, capacity, true);
    for (const port of outputNames) {
      const length =
        port === 'segments'
          ? cpu['count'][0] * 4
          : port === 'segmentLevels'
            ? cpu['count'][0]
            : port === 'segmentEdges'
              ? cpu['count'][0] * 2
              : port === 'vertices'
                ? cpu['vertexCount'][0] * 2
                : port === 'polylineOffsets'
                  ? cpu['polylineCount'][0] + 1
                  : port === 'polylineLevels' || port === 'polylineClosed'
                    ? cpu['polylineCount'][0]
                    : 1;
      for (let index = 0; index < length; index++) {
        if (port === 'segments' || port === 'vertices')
          expect(Math.abs(gpu[port][index] - cpu[port][index])).toBeLessThan(3e-5);
        else expect(gpu[port][index], `${port}[${index}]`).toBe(cpu[port][index]);
      }
    }
    expect(gpu['overflow'][0]).toBe(Number(mode === 'overflow'));
    if (mode === 'empty') expect(gpu['requiredCount'][0]).toBe(0);
  });
}

it('preserves the equality bridge at a four-tile seam vertex', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = {...tileFixtures[0], levels: [9]};
  const whole = await runTileOperation(device, getWholeScene(fixture));
  const tiled: number[][] = [];
  for (const scene of getTileScenes(fixture)) {
    const result = await runTileOperation(device, scene);
    tiled.push(...getSegmentRows(result));
  }
  expectSameSegments(tiled, getSegmentRows(whole));
  expect(tiled).toContainEqual([0, 23, 68, 7, 10.5, 7, 10.5]);
});

/** Feed resident contour segments into existing vector overlay predicates without readback. */
async function intersectExternalPolygon(
  device: Device,
  scene: IsolinesTileScene,
  fixture: IsolinesTileFixture
) {
  const capacity = 256;
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const upload = <Format extends 'float32' | 'float32x2' | 'uint32'>(
    name: string,
    values: Float32Array | Uint32Array,
    format: Format,
    length = values.length
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, length);
  };
  const output = <Format extends 'float32x4' | 'float32x2' | 'uint32x2' | 'uint32'>(
    name: string,
    format: Format,
    length: number,
    components = 1
  ) => {
    const buffer = createOutputBuffer(device, length * components);
    // Deliberately dirty output padding: only count/offset-delimited rows may reach the overlay.
    buffer.write(new Uint32Array(length * components).fill(0x3f800000));
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const segments = output('segments', 'float32x4', capacity, 4);
  const levels = output('segmentLevels', 'uint32', capacity);
  const count = output('count', 'uint32', 1);
  const overflow = output('overflow', 'uint32', 1);
  const lineOffsets = output('lineOffsets', 'uint32', capacity + 1);
  const pairCount = output('pairCount', 'uint32', 1);
  const pairOverflow = output('pairOverflow', 'uint32', 1);
  const leftIds = output('leftIds', 'uint32', capacity);
  const rightIds = output('rightIds', 'uint32', capacity);
  const points = output('points', 'float32x2', capacity, 2);
  graph.add(
    new GPUIsolines({
      grid: scene.grid,
      tile: scene.tile,
      values: upload('values', scene.values, 'float32'),
      mask: upload('mask', scene.mask, 'uint32'),
      levels: upload('levels', scene.levels, 'float32'),
      parameters: upload('parameters', scene.parameters, 'float32'),
      output: {
        segments: segments.view,
        segmentLevels: levels.view,
        status: {count: count.view, overflow: overflow.view}
      }
    })
  );
  graph.add(
    createWGSLKernelNode(graph, {
      id: 'contour-line-offsets',
      operation: 'ContourOverlayAcceptance',
      variant: 'line-offsets',
      invocationCount: capacity + 1,
      bindings: [
        {name: 'count', view: count.view, type: 'u32', access: 'read'},
        {name: 'offsets', view: lineOffsets.view, type: 'u32', access: 'read_write'}
      ],
      body: 'offsets[offsetsOffset + index] = 2u * min(index, count[countOffset]);'
    })
  );
  graph.add(
    new GPUSegmentIntersection({
      left: {
        kind: 'lines',
        positions: graph.createDataView(segments.view.buffer, {
          format: 'float32x2',
          length: capacity * 2
        }),
        lineOffsets: lineOffsets.view
      },
      right: {
        kind: 'polygons',
        positions: upload('polygon', Float32Array.from(fixture.polygon.flat()), 'float32x2', 4),
        featureOffsets: upload('featureOffsets', Uint32Array.of(0, 1), 'uint32'),
        polygonOffsets: upload('polygonOffsets', Uint32Array.of(0, 1), 'uint32'),
        ringOffsets: upload('ringOffsets', Uint32Array.of(0, 4), 'uint32')
      },
      output: {
        pairs: {
          leftIds: leftIds.view,
          rightIds: rightIds.view,
          status: {count: pairCount.view, overflow: pairOverflow.view}
        },
        points: points.view
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect((await readUint32(overflow.buffer, 1))[0]).toBe(0);
    expect((await readUint32(pairOverflow.buffer, 1))[0]).toBe(0);
    const length = (await readUint32(pairCount.buffer, 1))[0];
    const [leftRows, rightRows, coordinates, segmentLevels] = await Promise.all([
      readUint32(leftIds.buffer, length),
      readUint32(rightIds.buffer, length),
      readFloat32(points.buffer, length * 2),
      readUint32(levels.buffer, capacity)
    ]);
    return Array.from({length}, (_, index) => [
      segmentLevels[leftRows[index] / 2],
      rightRows[index],
      coordinates[2 * index],
      coordinates[2 * index + 1]
    ]);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
}

for (const fixture of tileFixtures) {
  it(`whole and tiled contours overlay an external polygon like Shapely: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const tiled: number[][] = [];
    for (const scene of getTileScenes(fixture))
      tiled.push(...(await intersectExternalPolygon(device, scene, fixture)));
    const whole = await intersectExternalPolygon(device, getWholeScene(fixture), fixture);
    for (const actual of [whole, tiled]) {
      const remaining = fixture.overlay.map(row => [...row]);
      expect(actual.length).toBe(remaining.length);
      for (const row of actual) {
        const found = remaining.findIndex(
          expected =>
            expected[0] === row[0] &&
            expected[1] === row[1] &&
            Math.abs(row[2] - expected[2]) < 3e-5 &&
            Math.abs(row[3] - expected[3]) < 3e-5
        );
        expect(found, `unmatched overlay ${JSON.stringify(row)}`).toBeGreaterThanOrEqual(0);
        remaining.splice(found, 1);
      }
    }
  });
}

it('does not expose dirty contour slots to an external overlay for an empty owned tile', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = tileFixtures[0];
  const scene = getTileScenes(fixture)[3];
  scene.tile = {...scene.tile!, ownedCells: [1, 1, 1, 2]};
  expect(await intersectExternalPolygon(device, scene, fixture)).toEqual([]);
});

for (const axis of ['column', 'row'] as const) {
  for (const originSample of [16777218, 16777217]) {
    it(`keeps the large-${axis} equality bridge at its origin-relative position (${originSample})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const scene = getLargeOffsetScene(axis, originSample);
      const expectedCoordinate = 16777218.5 - originSample;
      const expectedPoint =
        axis === 'column' ? [expectedCoordinate, 0.5] : [0.5, expectedCoordinate];
      const results = [
        await runTileOperation(device, scene, 1),
        await runTileOperation(device, scene, 1, true)
      ];
      for (const result of results) {
        // Hand-derived sample position: offset + one sample + half a cell - shared origin.
        expect(Array.from(result['segments'])).toEqual([...expectedPoint, ...expectedPoint]);
        expect(Array.from(result['vertices'].slice(0, 2))).toEqual(expectedPoint);
        expect(Array.from(result['polylineOffsets'].slice(0, 2))).toEqual([0, 1]);
        for (const port of ['count', 'requiredCount', 'polylineCount', 'vertexCount'])
          expect(result[port][0]).toBe(1);
        for (const port of [
          'overflow',
          'polylineOverflow',
          'segmentLevels',
          'polylineLevels',
          'polylineClosed'
        ])
          expect(result[port][0]).toBe(0);
      }
      expect(Array.from(results[0]['segmentEdges'])).toEqual(
        Array.from(results[1]['segmentEdges'])
      );
    });
  }
}
