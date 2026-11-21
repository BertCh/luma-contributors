// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createReviewPolygonizeProps} from './polygonize-review-output-oracles-fixtures';
import {createWaveHarness} from '../geometry-validity/w5-test-harness';
it.each([
  false,
  true
])('GPU exterior tolerance contact preserves area 3/8, exteriorFirst=%s', async exteriorFirst => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createWaveHarness(device);
  const triangle = [0, 0, 3 / 16, 0, 2, 4, 0, 0];
  const exterior = [3 / 32, -1 / 16, 3 / 32, -2];
  const props = createReviewPolygonizeProps(harness.allocate);
  props.lines = {
    kind: 'lines',
    positions: harness.allocate(
      'triangle',
      'float32x2',
      6,
      exteriorFirst ? [...exterior, ...triangle] : [...triangle, ...exterior]
    ),
    lineOffsets: harness.allocate('parts', 'uint32', 3, exteriorFirst ? [0, 2, 6] : [0, 4, 6])
  };
  props.precision = {...props.precision, vertexTolerance: 1 / 8};
  const nodes = new GPUPolygonize(props).getCommandNodes(harness.graph);
  try {
    await harness.run(nodes);
    expect(await harness.read(props.output.status.count)).toEqual([1]);
    expect((await harness.read(props.output.polygons.ringOffsets)).slice(0, 2)).toEqual([0, 3]);
    const coordinates = (await harness.read(props.output.polygons.positions)).slice(0, 6);
    let twiceArea = 0;
    for (let row = 0; row < 3; row++) {
      const next = (row + 1) % 3;
      twiceArea +=
        coordinates[2 * row] * coordinates[2 * next + 1] -
        coordinates[2 * next] * coordinates[2 * row + 1];
    }
    // Shoelace on the requested vertices gives exactly (3/16)*4 = 3/4.
    expect(Number.isFinite(twiceArea)).toBe(true);
    expect(Math.abs(twiceArea) / 2).toBe(3 / 8);
    expect(await harness.read(props.output.diagnostics.dangleCount)).toEqual([1]);
  } finally {
    harness.destroy();
  }
});

function createLineProps(
  harness: ReturnType<typeof createWaveHarness>,
  lines: readonly (readonly number[])[]
) {
  const coordinates = lines.flat();
  const offsets = [0];
  for (const line of lines) offsets.push(offsets[offsets.length - 1] + line.length / 2);
  const props = createReviewPolygonizeProps(harness.allocate);
  props.lines = {
    kind: 'lines',
    positions: harness.allocate('lines', 'float32x2', coordinates.length / 2, coordinates),
    lineOffsets: harness.allocate('parts', 'uint32', offsets.length, offsets)
  };
  props.precision = {...props.precision, vertexTolerance: 1 / 8};
  return props;
}

// Review B1: the bottom endpoints of a gapped triangle both lie within 1/8 of the spur's top
// endpoint (3/32,-1/16); the corners themselves are 3/16 apart. The bounded face survives and
// only the spur's two directed rows are dangles, in either source order.
it.each([
  false,
  true
])('GPU gapped triangle keeps its face beside a two-corner spur, spurFirst=%s', async spurFirst => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createWaveHarness(device);
  const paths = [
    [0, 0, 3 / 16, 0],
    [5 / 16, 0, 2, 4, -1 / 8, 0]
  ];
  const spur = [3 / 32, -1 / 16, 3 / 32, -2];
  const props = createLineProps(harness, spurFirst ? [spur, ...paths] : [...paths, spur]);
  try {
    await harness.run(new GPUPolygonize(props).getCommandNodes(harness.graph));
    expect(await harness.read(props.output.directedEdges.status.count)).toEqual([8]);
    expect((await harness.read(props.output.directedEdges.ringIds)).slice(0, 8)).toEqual(
      spurFirst
        ? [0xffffffff, 0xffffffff, 0, 1, 0, 1, 0, 1]
        : [0, 1, 0, 1, 0, 1, 0xffffffff, 0xffffffff]
    );
    expect(await harness.read(props.output.status.count)).toEqual([1]);
  } finally {
    harness.destroy();
  }
});

// Review B1: a square whose closure gap is 3/32 survives alone and with one outward spur
// attached exactly at each end of the gap; the four spur rows are the only dangles.
it.each([false, true])('GPU square closure gap survives exact outward spurs=%s', async spurs => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const harness = createWaveHarness(device);
  const props = createLineProps(harness, [
    [0, 0, 4, 0, 4, 4, 0, 4, 0, 3 / 32],
    ...(spurs
      ? [
          [0, 0, -1, -1],
          [0, 3 / 32, -1, 1]
        ]
      : [])
  ]);
  try {
    await harness.run(new GPUPolygonize(props).getCommandNodes(harness.graph));
    const rows = spurs ? 12 : 8;
    expect(await harness.read(props.output.directedEdges.status.count)).toEqual([rows]);
    expect((await harness.read(props.output.directedEdges.ringIds)).slice(0, rows)).toEqual([
      0,
      1,
      0,
      1,
      0,
      1,
      0,
      1,
      ...(spurs ? [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff] : [])
    ]);
    expect(await harness.read(props.output.status.count)).toEqual([1]);
  } finally {
    harness.destroy();
  }
});
