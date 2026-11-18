// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

// Vertex rows before the first ring or line offset and from the last offset on belong to no
// feature: unwritten capacity or a transient the graph reuses, holding anything. The feature
// bounds kernel, the BVH built from them, the polygon edge layout and the exact refinement must
// read only rows that belong to a ring or line. Each case runs once without padding and again with
// 2 leading and 3 trailing vertex rows poisoned by NaN, by far finite values and by a value that
// puts phantom edges across the scene, on either side of the join, and expects the same result.

import type {Buffer, Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPUPointInPolygonJoin,
  GPUSpatialPredicateJoin,
  GPU_SPATIAL_JOIN_NO_FEATURE,
  type GPUSpatialJoinGeometry
} from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';
import {
  buildOracleArrays,
  line,
  point,
  polygonWithRings,
  rectangle,
  type OracleFeature
} from './spatial-predicate-oracle';

const LEADING = 2;
const TRAILING = 3;
/** NaN, far values, and 1 and 5, which put phantom edges and degenerate bounds inside the scene. */
const POISONS = [Number.NaN, 3e6, -3e6, 1, 5];

const POINTS = [point(1, 1), point(5, 5), point(11, 1), point(30, 30), point(3, 3), point(14, 14)];
const POLYGONS = [
  rectangle(0, 0, 2, 2),
  polygonWithRings(
    [
      [10, 0],
      [16, 0],
      [16, 6],
      [10, 6],
      [10, 0]
    ],
    [
      [12, 2],
      [14, 2],
      [14, 4],
      [12, 4],
      [12, 2]
    ]
  ),
  rectangle(20, 20, 2, 2)
];
const LINES = [line([-1, 1], [3, 1]), line([13, 3], [13, 8]), line([40, 40], [41, 41])];

type Padding = {leading: number; trailing: number; value: number};
const NO_PADDING: Padding = {leading: 0, trailing: 0, value: 0};
const PADDING = (value: number): Padding => ({leading: LEADING, trailing: TRAILING, value});

function createGeometry(
  device: Device,
  graph: GPUCommandGraph,
  buffers: Buffer[],
  name: string,
  features: OracleFeature[],
  padding: Padding
): GPUSpatialJoinGeometry {
  const arrays = buildOracleArrays(features[0].kind, features);
  const upload = (data: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, data);
    buffers.push(buffer);
    return buffer;
  };
  const shift = (offsets: Uint32Array) => offsets.map(offset => offset + padding.leading);
  const view = (suffix: string, data: Uint32Array) =>
    importGraphBuffer(graph, `${name}-${suffix}`, upload(data), 'uint32', data.length);
  const padded = Float32Array.from([
    ...new Array<number>(2 * padding.leading).fill(padding.value),
    ...arrays.positions,
    ...new Array<number>(2 * padding.trailing).fill(padding.value)
  ]);
  const positions = importGraphBuffer(
    graph,
    `${name}-positions`,
    upload(padded),
    'float32x2',
    padded.length / 2
  );
  if (arrays.kind === 'points') {
    return {kind: 'points', positions};
  }
  if (arrays.kind === 'lines') {
    return {kind: 'lines', positions, lineOffsets: view('line-offsets', shift(arrays.lineOffsets))};
  }
  return {
    kind: 'polygons',
    positions,
    featureOffsets: view('feature-offsets', arrays.featureOffsets),
    polygonOffsets: view('polygon-offsets', arrays.polygonOffsets),
    ringOffsets: view('ring-offsets', shift(arrays.ringOffsets))
  };
}

async function runPredicate(
  left: OracleFeature[],
  right: OracleFeature[],
  leftPadding: Padding,
  rightPadding: Padding
) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'spatial-join-padding-rows'});
  const buffers: Buffer[] = [];
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `out-${buffers.length}`, buffer, 'uint32', length)
    };
  };
  const leftRows = output(64);
  const rightRows = output(64);
  const count = output(1);
  const pairOverflow = output(1);
  const overflow = output(1);
  const candidateOverflow = output(1);
  graph.add(
    new GPUSpatialPredicateJoin({
      left: createGeometry(device, graph, buffers, 'left', left, leftPadding),
      right: createGeometry(device, graph, buffers, 'right', right, rightPadding),
      predicate: 'intersects',
      how: 'inner',
      candidateCapacity: 64,
      output: {
        pairs: {
          leftRows: leftRows.view,
          rightRows: rightRows.view,
          status: {count: count.view, overflow: pairOverflow.view}
        },
        status: {overflow: overflow.view, candidateOverflow: candidateOverflow.view}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const [pairCount] = await readUint32(count.buffer, 1);
  const lefts = (await readUint32(leftRows.buffer, 64)).slice(0, pairCount);
  const rights = (await readUint32(rightRows.buffer, 64)).slice(0, pairCount);
  const result = {
    pairs: lefts.map((row, slot) => [row, rights[slot]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]),
    pairOverflow: await readUint32(pairOverflow.buffer, 1),
    overflow: await readUint32(overflow.buffer, 1),
    candidateOverflow: await readUint32(candidateOverflow.buffer, 1)
  };
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
  return result;
}

async function runContainment(padding: Padding) {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device, {id: 'point-in-polygon-padding-rows'});
  const buffers: Buffer[] = [];
  const output = (length: number) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `out-${buffers.length}`, buffer, 'uint32', length)
    };
  };
  const featureIds = output(POINTS.length);
  const overflow = output(1);
  const uncertainCount = output(1);
  graph.add(
    new GPUPointInPolygonJoin({
      points: createGeometry(device, graph, buffers, 'points', POINTS, NO_PADDING) as never,
      polygons: createGeometry(device, graph, buffers, 'polygons', POLYGONS, padding) as never,
      candidateCapacity: 64,
      output: {
        pointFeatureIds: featureIds.view,
        status: {overflow: overflow.view, uncertainCount: uncertainCount.view}
      }
    })
  );
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    featureIds: await readUint32(featureIds.buffer, POINTS.length),
    overflow: await readUint32(overflow.buffer, 1),
    uncertainCount: await readUint32(uncertainCount.buffer, 1)
  };
  compiled.destroy();
  buffers.forEach(buffer => buffer.destroy());
  return result;
}

it('GPUPointInPolygonJoin ignores polygon vertex rows outside every ring', async () => {
  const reference = await runContainment(NO_PADDING);
  expect(reference.overflow).toEqual([0]);
  expect(reference.featureIds.slice(0, 3)).toEqual([0, GPU_SPATIAL_JOIN_NO_FEATURE, 1]);
  for (const value of POISONS) {
    expect(await runContainment(PADDING(value)), `padding ${value}`).toEqual(reference);
  }
});

const CASES: [string, OracleFeature[], OracleFeature[], 'left' | 'right' | 'both'][] = [
  ['points x polygons', POINTS, POLYGONS, 'right'],
  ['lines x polygons', LINES, POLYGONS, 'both'],
  ['polygons x lines', POLYGONS, LINES, 'both'],
  ['points x lines', POINTS, LINES, 'right']
];

for (const [name, left, right, sides] of CASES) {
  it(`GPUSpatialPredicateJoin (${name}) ignores vertex rows outside every ring and line`, async () => {
    const reference = await runPredicate(left, right, NO_PADDING, NO_PADDING);
    expect(reference.overflow).toEqual([0]);
    expect(reference.pairs.length).toBeGreaterThan(0);
    for (const value of POISONS) {
      const padding = PADDING(value);
      expect(
        await runPredicate(left, right, sides === 'right' ? NO_PADDING : padding, padding),
        `padding ${value}`
      ).toEqual(reference);
      if (sides === 'both') {
        expect(
          await runPredicate(left, right, padding, NO_PADDING),
          `left only padding ${value}`
        ).toEqual(reference);
      }
    }
  });
}
