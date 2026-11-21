// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUSharedPaths} from '../../../src/gpu-spatial-analysis/line-clip/gpu-shared-paths';
import {createSharedTruncationFixture} from './line-clip-output-oracle-fixture';
import {createWaveHarness} from '../geometry-validity/w5-test-harness';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
useCPUExecutorTestHooks();
it('rejects adjacent count and overflow bindings before lowering', () => {
  const harness = createWaveHarness();
  try {
    const fixture = createSharedTruncationFixture(harness.graph, (format, length, values) =>
      harness.allocate('shared', format, length, values)
    );
    const count = harness.graph.createDataView(fixture.packed.buffer, {
      format: 'uint32',
      length: 1
    });
    const overflow = harness.graph.createDataView(fixture.packed.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    expect(
      () =>
        new GPUSharedPaths({
          ...fixture.contributor.props,
          output: {...fixture.contributor.props.output, status: {count, overflow}}
        })
    ).toThrow(/count.*overflow|overflow.*count/);
  } finally {
    harness.destroy();
  }
});
it('preserves separately dispatched adjacent overflow and uncertainty on replay', async () => {
  const harness = createWaveHarness();
  try {
    const fixture = createSharedTruncationFixture(harness.graph, (format, length, values) =>
      harness.allocate('shared', format, length, values)
    );
    const nodes = fixture.contributor.getCommandNodes(harness.graph);
    for (let replay = 0; replay < 2; replay++) {
      if (replay)
        harness.write(
          fixture.contributor.props.right.positions,
          Float32Array.of(0, 0, 1, 0, 0, 4, 1, 4)
        );
      await harness.run(nodes);
      expect(await harness.read(fixture.packed)).toEqual([replay === 0 ? 1 : 0, 0]);
      expect(await harness.read(fixture.count)).toEqual([1]);
    }
  } finally {
    harness.destroy();
  }
});

it.each([
  'run-writers',
  'provenance-writers',
  'provenance-count'
] as const)('review earlier fix rejects incompatible %s bindings', route => {
  const harness = createWaveHarness();
  try {
    const fixture = createSharedTruncationFixture(harness.graph, (format, length, values) =>
      harness.allocate('base', format, length, values)
    );
    const props = fixture.contributor.props;
    const packed = harness.allocate('packed', 'uint32', route === 'provenance-count' ? 65 : 4);
    const first = harness.graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
    const second = harness.graph.createDataView(packed.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: route === 'run-writers' ? 12 : route === 'provenance-count' ? 256 : 4
    });
    const runs = {
      ...props.output.runs,
      leftSourceIds: first,
      ...(route === 'run-writers'
        ? {forward: second}
        : {
            leftPartRows:
              route === 'provenance-count' ? harness.allocate('part', 'uint32', 1) : second
          })
    };
    const status = {
      ...props.output.status,
      ...(route === 'provenance-count' ? {count: second} : {})
    };
    expect(() => new GPUSharedPaths({...props, output: {runs, status}})).toThrow(
      route === 'run-writers'
        ? /leftSourceIds.*forward|forward.*leftSourceIds/
        : route === 'provenance-writers'
          ? /leftSourceIds.*leftPartRows|leftPartRows.*leftSourceIds/
          : /leftSourceIds.*count|count.*leftSourceIds/
    );
  } finally {
    harness.destroy();
  }
});
