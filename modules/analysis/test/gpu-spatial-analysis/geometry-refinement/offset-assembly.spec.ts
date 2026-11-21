// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createOffsetComponentAssemblyNodes} from '../../../src/gpu-spatial-analysis/geometry-refinement/offset-component-assembly';
import {createOracleFixture} from '../global-coordinates/output-oracle-fixture';
import {submitGraph} from '../../utils/gpu-contributor-test-utils';
import {OFFSET_ASSEMBLY_SCENES, packOffsetAssemblyFixture} from './offset-assembly-fixtures';
import {createOffsetSequenceBudgetFixture} from './offset-sequence-budget-fixture';

it('forward F9 GPU keeps every repetition and command order across sequence batches', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const result = fixture.output('uint32', 1);
  const sequence = createOffsetSequenceBudgetFixture(fixture.graph, result.view);
  expect(sequence.batches.map(node => node.workload?.maximumInvocationCount)).toEqual([
    4000000000, 2000000000
  ]);
  fixture.graph.add(sequence.nodes);
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await result.read()).toEqual([121212]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it.each(
  OFFSET_ASSEMBLY_SCENES
)('GPU deterministic parallel offset assembly: $name', async scene => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  const packed = packOffsetAssemblyFixture(scene);
  const positions = fixture.output('float32x2', 16);
  const components = fixture.output('uint32x4', 8);
  const sources = fixture.output('uint32', 16);
  const status = fixture.output('uint32x4', 2);
  const nodes = createOffsetComponentAssemblyNodes(fixture.graph, {
    id: 'assembly',
    capacity: packed.capacity,
    fragments: fixture.input('float32x4', Array.from(packed.fragments)),
    metadata: fixture.input('uint32', Array.from(packed.metadata)),
    output: {
      positions: positions.view,
      components: components.view,
      sourceEdges: sources.view,
      status: status.view
    }
  });
  fixture.graph.add(nodes);
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const count = scene.components.length / 4;
    const vertices = scene.expected.length / 2;
    expect(await status.read()).toEqual([count, vertices, count, vertices, 0, 0, 0, 0]);
    expect((await positions.read()).slice(0, scene.expected.length)).toEqual(scene.expected);
    expect((await sources.read()).slice(0, vertices)).toEqual(scene.sources);
    expect(await components.read()).toEqual([
      ...scene.components,
      ...Array(32 - scene.components.length).fill(0)
    ]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});

it('GPU batches a cycle and cancels a reversed spur across workgroup rounds', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const fixture = createOracleFixture(device);
  // Equivalent to an 11-vertex line with segmentCapacity 512, above the previous batching limit.
  const capacity = 512 * (11 * 6 + 1);
  // A simple rectangular cycle with 64 unit edges spans two traversal dispatches.
  const cycle = [
    ...Array.from({length: 20}, (_, column) => [column, 0]),
    ...Array.from({length: 12}, (_, row) => [20, row]),
    ...Array.from({length: 20}, (_, column) => [20 - column, 12]),
    ...Array.from({length: 12}, (_, row) => [0, 12 - row])
  ];
  const spurRow = 33;
  const sourceRow = (edge: number) => (edge < spurRow ? edge : edge + 2);
  const fragments = new Float32Array(capacity * 4);
  const metadata = new Uint32Array(capacity * 4);
  for (let edge = 0; edge < cycle.length; edge++) {
    const row = sourceRow(edge);
    fragments.set([...cycle[edge], ...cycle[(edge + 1) % cycle.length]], row * 4);
    metadata.set([0, row, 1, 0], row * 4);
  }
  // The first cancelling candidate is in the second dispatch.
  fragments.set([100, 0, 101, 0, 101, 0, 100, 0], spurRow * 4);
  metadata.set([0, spurRow, 1, 0, 0, spurRow + 1, 1, 0], spurRow * 4);
  const vertexCount = cycle.length + 1;
  const positions = fixture.output('float32x2', vertexCount);
  const components = fixture.output('uint32x4', 1);
  const sources = fixture.output('uint32', vertexCount);
  const status = fixture.output('uint32x4', 2);
  fixture.graph.add(
    createOffsetComponentAssemblyNodes(fixture.graph, {
      id: 'batched-offset',
      capacity,
      fragments: fixture.input('float32x4', Array.from(fragments)),
      metadata: fixture.input('uint32', Array.from(metadata)),
      output: {
        positions: positions.view,
        components: components.view,
        sourceEdges: sources.view,
        status: status.view
      }
    })
  );
  const compiled = fixture.graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await status.read()).toEqual([1, vertexCount, 1, vertexCount, 0, 0, 0, 0]);
    expect(await components.read()).toEqual([0, vertexCount, 0, 0]);
    expect(await positions.read()).toEqual([...cycle, cycle[0]].flat());
    expect(await sources.read()).toEqual([0, ...cycle.map((_, edge) => sourceRow(edge))]);
  } finally {
    compiled.destroy();
    fixture.destroy();
  }
});
