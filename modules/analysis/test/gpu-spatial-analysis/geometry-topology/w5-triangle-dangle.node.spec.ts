// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUPolygonize} from '../../../src/gpu-spatial-analysis/geometry-topology/gpu-polygonize';
import {createReviewPolygonizeProps} from './polygonize-review-output-oracles-fixtures';
import {createWaveHarness} from '../geometry-validity/w5-test-harness';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
useCPUExecutorTestHooks();
it.each([
  false,
  true
])('exterior tolerance contact preserves triangle ring owners, exteriorFirst=%s', async exteriorFirst => {
  const harness = createWaveHarness();
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
  const placement = nodes.findIndex(
    node => node.id === 'polygonize-faces-propagate-duplicate-owners'
  );
  expect(placement).toBeGreaterThan(0);
  try {
    await harness.run(nodes.slice(0, placement + 1));
    expect(await harness.read(props.output.directedEdges.status.count)).toEqual([8]);
    // Shoelace: (3/16 * 4)/2 = 3/8. Three edges bound a face; only the exterior edge is a dangle.
    expect((await harness.read(props.output.directedEdges.ringIds)).slice(0, 8)).toEqual(
      exteriorFirst
        ? [0xffffffff, 0xffffffff, 0, 1, 0, 1, 0, 1]
        : [0, 1, 0, 1, 0, 1, 0xffffffff, 0xffffffff]
    );
  } finally {
    harness.destroy();
  }
});

it.each([
  false,
  true
])('review B1 gapped triangle retains direct continuations with spur first=%s', async exteriorFirst => {
  const harness = createWaveHarness();
  const paths = [
    [0, 0, 3 / 16, 0],
    [5 / 16, 0, 2, 4, -1 / 8, 0]
  ];
  const spur = [3 / 32, -1 / 16, 3 / 32, -2];
  const lines = exteriorFirst ? [spur, ...paths] : [...paths, spur];
  const coordinates = lines.flat();
  const offsets = [0];
  for (const line of lines) offsets.push(offsets[offsets.length - 1] + line.length / 2);
  const props = createReviewPolygonizeProps(harness.allocate);
  props.lines = {
    kind: 'lines',
    positions: harness.allocate('gapped', 'float32x2', coordinates.length / 2, coordinates),
    lineOffsets: harness.allocate('parts', 'uint32', offsets.length, offsets)
  };
  props.precision = {...props.precision, vertexTolerance: 1 / 8};
  const nodes = new GPUPolygonize(props).getCommandNodes(harness.graph);
  const placement = nodes.findIndex(
    node => node.id === 'polygonize-faces-propagate-duplicate-owners'
  );
  try {
    await harness.run(nodes.slice(0, placement + 1));
    expect(await harness.read(props.output.directedEdges.status.count)).toEqual([8]);
    expect((await harness.read(props.output.directedEdges.ringIds)).slice(0, 8)).toEqual(
      exteriorFirst
        ? [0xffffffff, 0xffffffff, 0, 1, 0, 1, 0, 1]
        : [0, 1, 0, 1, 0, 1, 0xffffffff, 0xffffffff]
    );
  } finally {
    harness.destroy();
  }
});
it.each([
  false,
  true
])('review B1 square closure gap survives exact outward spurs=%s', async spurs => {
  const harness = createWaveHarness();
  const lines = [
    [0, 0, 4, 0, 4, 4, 0, 4, 0, 3 / 32],
    ...(spurs
      ? [
          [0, 0, -1, -1],
          [0, 3 / 32, -1, 1]
        ]
      : [])
  ];
  const coordinates = lines.flat();
  const offsets = [0];
  for (const line of lines) offsets.push(offsets[offsets.length - 1] + line.length / 2);
  const props = createReviewPolygonizeProps(harness.allocate);
  props.lines = {
    kind: 'lines',
    positions: harness.allocate('square', 'float32x2', coordinates.length / 2, coordinates),
    lineOffsets: harness.allocate('parts', 'uint32', offsets.length, offsets)
  };
  props.precision = {...props.precision, vertexTolerance: 1 / 8};
  const nodes = new GPUPolygonize(props).getCommandNodes(harness.graph);
  const placement = nodes.findIndex(
    node => node.id === 'polygonize-faces-propagate-duplicate-owners'
  );
  try {
    await harness.run(nodes.slice(0, placement + 1));
    expect(await harness.read(props.output.directedEdges.status.count)).toEqual([spurs ? 12 : 8]);
    expect(
      (await harness.read(props.output.directedEdges.ringIds)).slice(0, spurs ? 12 : 8)
    ).toEqual([
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
  } finally {
    harness.destroy();
  }
});
