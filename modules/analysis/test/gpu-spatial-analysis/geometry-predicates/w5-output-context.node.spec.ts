// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUGeometryPredicates} from '../../../src/gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates';
import {createWaveHarness} from '../geometry-validity/w5-test-harness';
it.each([
  false,
  true
])('separate topological passes allow adjacent words and reject logical overlap=%s', async overlap => {
  const harness = createWaveHarness();
  try {
    const geometry = {
      kind: 'points' as const,
      positions: harness.allocate('points', 'float32x2', 1, [1, 2])
    };
    const packed = harness.allocate('packed', 'uint32', 2);
    const equals = harness.graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
    const touches = harness.graph.createDataView(packed.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: overlap ? 0 : 4
    });
    const construct = () =>
      new GPUGeometryPredicates({
        geometry,
        other: geometry,
        output: {topological: {equals, touches}}
      });
    if (overlap) expect(construct).toThrow(/equals.*touches|touches.*equals/);
    else {
      await harness.run(construct().getCommandNodes(harness.graph));
      expect(await harness.read(packed)).toEqual([1, 0]);
    }
  } finally {
    harness.destroy();
  }
});
it.each([false, true])('captures nested spatial contexts once, aliased=%s', aliased => {
  const harness = createWaveHarness();
  try {
    const geometry = {
      kind: 'points' as const,
      positions: harness.allocate('points', 'float32x2', 1, [1, 2])
    };
    let reads = 0;
    const changing = {
      coordinateSpace: 'planar' as const,
      get metric() {
        reads++;
        return reads === 1 ? ('native' as const) : ('great-circle' as const);
      }
    };
    const primary = aliased
      ? changing
      : {coordinateSpace: 'planar' as const, metric: 'native' as const};
    const contributor = new GPUGeometryPredicates({
      geometry,
      other: geometry,
      spatialContext: primary,
      otherSpatialContext: changing,
      output: {topological: {equals: harness.allocate('result', 'uint32', 1)}}
    });
    expect(reads).toBe(1);
    expect(contributor.props.otherSpatialContext?.metric).toBe('native');
    if (aliased)
      expect(contributor.props.otherSpatialContext).toBe(contributor.props.spatialContext);
  } finally {
    harness.destroy();
  }
});

it('rejects adjacent exact and identical words co-bound by the comparison dispatch', () => {
  const harness = createWaveHarness();
  try {
    const geometry = {
      kind: 'points' as const,
      positions: harness.allocate('points', 'float32x2', 1, [1, 2])
    };
    const packed = harness.allocate('packed', 'uint32', 2);
    const equalsExact = harness.graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
    const equalsIdentical = harness.graph.createDataView(packed.buffer, {
      format: 'uint32',
      length: 1,
      byteOffset: 4
    });
    expect(
      () =>
        new GPUGeometryPredicates({
          geometry,
          other: geometry,
          parameters: harness.allocate('parameters', 'float32', 4, [0, 0, 0, 0]),
          output: {equalsExact, equalsIdentical}
        })
    ).toThrow(/equalsExact.*equalsIdentical|equalsIdentical.*equalsExact/);
  } finally {
    harness.destroy();
  }
});

it.each([
  'point-simple-status',
  'point-simple-ring',
  'empty-line-status'
] as const)('review B4 permits separately dispatched adjacent outputs: %s', async route => {
  const harness = createWaveHarness();
  const packed = harness.allocate('packed', 'uint32', 2);
  const first = harness.graph.createDataView(packed.buffer, {format: 'uint32', length: 1});
  const second = harness.graph.createDataView(packed.buffer, {
    format: 'uint32',
    length: 1,
    byteOffset: 4
  });
  const geometry =
    route === 'empty-line-status'
      ? {
          kind: 'lines' as const,
          positions: harness.allocate('positions', 'float32x2', 0, []),
          lineOffsets: harness.allocate('parts', 'uint32', 2, [0, 0])
        }
      : {kind: 'points' as const, positions: harness.allocate('positions', 'float32x2', 1, [1, 2])};
  const output =
    route === 'point-simple-ring'
      ? {
          isSimple: first,
          isRing: second,
          status: {overflow: harness.allocate('overflow', 'uint32', 1)}
        }
      : route === 'point-simple-status'
        ? {isSimple: first, status: {overflow: second}}
        : {status: {overflow: first, intersectionCount: second}};
  try {
    const contributor = new GPUGeometryPredicates({geometry, intersectionCapacity: 1, output});
    await harness.run(contributor.getCommandNodes(harness.graph));
    expect(await harness.read(packed)).toEqual(route === 'empty-line-status' ? [0, 0] : [1, 0]);
  } finally {
    harness.destroy();
  }
});
it.each([
  'spatialContext',
  'otherSpatialContext'
] as const)('review S1 rejects malformed %s with contributor and prop', field => {
  const harness = createWaveHarness();
  const geometry = {
    kind: 'points' as const,
    positions: harness.allocate('position', 'float32x2', 1, [0, 0])
  };
  try {
    expect(
      () =>
        new GPUGeometryPredicates({
          id: 'context-review',
          geometry,
          other: geometry,
          [field]: 'planar' as never,
          output: {}
        })
    ).toThrow(new RegExp(`context-review.*${field}.*object`));
  } finally {
    harness.destroy();
  }
});

it('review B5 point status requests require capacity and simplicity requests require overflow', () => {
  const harness = createWaveHarness();
  const geometry = {
    kind: 'points' as const,
    positions: harness.allocate('point', 'float32x2', 1, [0, 0])
  };
  const result = harness.allocate('result', 'uint32', 1);
  try {
    expect(
      () => new GPUGeometryPredicates({geometry, output: {status: {intersectionCount: result}}})
    ).toThrow(/intersectionCapacity.*integer/);
    for (const field of ['isSimple', 'isRing'] as const)
      expect(
        () =>
          new GPUGeometryPredicates({geometry, intersectionCapacity: 1, output: {[field]: result}})
      ).toThrow(/isSimple and isRing require status.overflow/);
  } finally {
    harness.destroy();
  }
});
