// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {createVertexClearanceNodes} from '../../../src/gpu-spatial-analysis/minimum-clearance/vertex-clearance-nodes';
import {createWaveHarness} from '../geometry-validity/w5-test-harness';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
useCPUExecutorTestHooks();
it.each([
  false,
  true
])('diagonal clearance uses the lowest partner row with sorting=%s', async spatialSort => {
  const harness = createWaveHarness();
  const positions = harness.allocate('positions', 'float32x2', 3, [0, 0, 1, 1, -1, -1]);
  const features = harness.allocate('features', 'uint32', 3, [0, 0, 0]);
  const nearest = harness.allocate('nearest', 'float32x4', 3, [
    Infinity,
    0,
    0,
    0,
    Infinity,
    0,
    0,
    0,
    Infinity,
    0,
    0,
    0
  ]);
  const nodes = createVertexClearanceNodes(harness.graph, {
    id: 'tie',
    positions,
    features,
    nearest,
    leafCapacity: 4,
    spatialSort
  });
  try {
    await harness.run(nodes);
    // Exact squared separation is 1²+1²=2; partners at rows 1 and 2 tie, so row 1 (1,1) wins.
    const result = await harness.read(nearest);
    expect(result.slice(1, 3)).toEqual([1, 1]);
    const distance = result[0];
    expect(Number.isFinite(distance)).toBe(true);
    expect(distance).toBeCloseTo(1.4142135623730951, 6);
  } finally {
    harness.destroy();
  }
});

import {vi} from 'vitest';
import type {GraphDataView} from '@luma.gl/graph/gpu-core';
import {createCollapsedClearanceReviewCase} from '../geometry-predicates/multipart-review-fixtures';
it.each([
  false,
  true
])('review B2 collapsed endpoint wins a mixed segment tie with sorting=%s', async spatialSort => {
  const harness = createWaveHarness();
  const views: GraphDataView[] = [];
  const createDataView = harness.graph.createDataView.bind(harness.graph);
  const observe = vi.spyOn(harness.graph, 'createDataView').mockImplementation((...parameters) => {
    const view = createDataView(...parameters);
    views.push(view);
    return view;
  });
  const fixture = createCollapsedClearanceReviewCase(
    harness.graph,
    (name, format, length, values) =>
      harness.allocate(
        name,
        format,
        length,
        name === 'collapsed-clearance-input' ? [0, 0, 0, 0, 1, 1, 1, 1, -1, -1, -3, -3] : values
      ),
    false,
    true,
    spatialSort
  );
  observe.mockRestore();
  const nearest = views.find(view => view.buffer.id === 'minimum-clearance-vertex-nearest');
  expect(nearest).toBeDefined();
  const stop = fixture.nodes.findIndex(node => node.id === 'minimum-clearance-feature-minimum');
  expect(stop).toBeGreaterThan(0);
  try {
    await harness.run(fixture.nodes.slice(0, stop));
    const result = await harness.read(nearest!);
    expect(result.slice(1, 3)).toEqual([1, 1]);
    expect(Number.isFinite(result[0])).toBe(true);
    expect(result[0]).toBeCloseTo(1.4142135623730951, 6);
  } finally {
    harness.destroy();
  }
});
