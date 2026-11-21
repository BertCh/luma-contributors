// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it, vi} from 'vitest';
import {
  computePointsInPolygonsChoroplethOnCPU,
  createCPUPointsInPolygonsChoroplethOperation
} from '../../../src/gpu-spatial-analysis/index';
import {createPointsInPolygonsChoroplethOperation} from '../../../src/gpu-spatial-analysis/index';

it.each([
  'zonal',
  'group'
] as const)('B2 %s CPU count respects get-only multipart offsets', backend => {
  const featureOffsets = Uint32Array.of(0, 2);
  let reads = 0;
  const points = new Proxy(
    {kind: 'points' as const, positions: Float32Array.of(0.25, 0.25, 0.75, 0.75)},
    {
      get(target, key, receiver) {
        if (key === 'featureOffsets') {
          reads++;
          return featureOffsets;
        }
        return Reflect.get(target, key, receiver);
      }
    }
  );
  const polygons = {
    kind: 'polygons' as const,
    positions: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1),
    featureOffsets: Uint32Array.of(0, 1),
    polygonOffsets: Uint32Array.of(0, 1),
    ringOffsets: Uint32Array.of(0, 4)
  };
  // Both parts belong to one feature, counted once in its polygon.
  expect(
    Array.from(
      computePointsInPolygonsChoroplethOnCPU({points, polygons}, {backend, candidateCapacity: 8})
        .counts
    )
  ).toEqual([1]);
  expect(reads).toBe(1);
});

it.each([
  {name: 'GPU', createOperation: createPointsInPolygonsChoroplethOperation},
  {name: 'CPU', createOperation: createCPUPointsInPolygonsChoroplethOperation}
])('B3 published $name factory never reads or declares unused ID descriptors', ({
  createOperation
}) => {
  const props: Parameters<typeof createOperation>[0] = {
    candidateCapacity: 8,
    points: {
      kind: 'points',
      positions: {value: 'points', length: 2},
      featureOffsets: {value: 'pointOffsets', length: 2},
      get sourceIds(): never {
        throw new Error('unused point IDs');
      }
    },
    polygons: {
      kind: 'polygons',
      positions: {value: 'polygons', length: 4},
      featureOffsets: {value: 'features', length: 2},
      polygonOffsets: {value: 'polygonOffsets', length: 2},
      ringOffsets: {value: 'rings', length: 2},
      get sourceIds(): never {
        throw new Error('unused polygon IDs');
      }
    },
    output: {counts: 'counts', overflow: 'overflow', uncertainCount: 'uncertainCount'}
  };
  const operation = createOperation(props);
  expect(Object.keys(operation.inputs).some(name => name.includes('sourceIds'))).toBe(false);
  const supplied = createOperation({
    ...props,
    points: {
      kind: 'points',
      positions: props.points.positions,
      sourceIds: {value: 'unneededPointIds', length: 2}
    },
    polygons: {
      kind: 'polygons',
      positions: props.polygons.positions,
      featureOffsets: props.polygons.featureOffsets,
      polygonOffsets: props.polygons.polygonOffsets,
      ringOffsets: props.polygons.ringOffsets,
      sourceIds: {value: 'unneededPolygonIds', length: 1}
    }
  });
  expect(Object.values(supplied.inputs)).not.toContain('unneededPointIds');
  expect(Object.values(supplied.inputs)).not.toContain('unneededPolygonIds');
});

it.each([
  'zonal',
  'group'
] as const)('U1 %s ordinary sum avoids the dense feature/polygon allocation', async backend => {
  const points = {
    kind: 'points' as const,
    positions: Float32Array.of(0.25, 0.25, 1.25, 0.25, 1.75, 0.75)
  };
  const polygons = {
    kind: 'polygons' as const,
    positions: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 1, 0, 2, 0, 2, 1, 1, 1),
    featureOffsets: Uint32Array.of(0, 1, 2),
    polygonOffsets: Uint32Array.of(0, 1, 2),
    ringOffsets: Uint32Array.of(0, 4, 8)
  };
  const allocations: number[] = [];
  const Native = Uint32Array;
  vi.stubGlobal(
    'Uint32Array',
    new Proxy(Native, {
      construct(target, arguments_) {
        if (typeof arguments_[0] === 'number') allocations.push(arguments_[0]);
        return Reflect.construct(target, arguments_);
      }
    })
  );
  try {
    const result = computePointsInPolygonsChoroplethOnCPU(
      {points, polygons, values: Float32Array.of(7, 11, 13)},
      {backend, statistic: 'sum', candidateCapacity: 8}
    );
    expect(Array.from(result.counts)).toEqual([1, 2]);
    expect(Array.from(result.featureValues)).toEqual([7, 24]);
    expect(allocations).not.toContain(6);
    allocations.length = 0;
    const operation = createCPUPointsInPolygonsChoroplethOperation({
      backend,
      statistic: 'sum',
      candidateCapacity: 8,
      points: {kind: 'points', positions: {value: 'points', length: 3}},
      polygons: {
        kind: 'polygons',
        positions: {value: 'polygons', length: 8},
        featureOffsets: {value: 'features', length: 3},
        polygonOffsets: {value: 'polygonOffsets', length: 3},
        ringOffsets: {value: 'rings', length: 3}
      },
      values: {value: 'values', length: 3},
      output: {
        counts: 'counts',
        featureValues: 'featureValues',
        overflow: 'overflow',
        uncertainCount: 'uncertainCount'
      }
    });
    const rows = {
      points: points.positions,
      polygons: polygons.positions,
      features: polygons.featureOffsets,
      polygonOffsets: polygons.polygonOffsets,
      rings: polygons.ringOffsets,
      values: Float32Array.of(7, 11, 13)
    };
    const output = (await operation.execute({
      inputs: Object.fromEntries(
        Object.entries(operation.inputs).map(([port, name]) => {
          const data = rows[name as keyof typeof rows];
          const format =
            name === 'points' || name === 'polygons'
              ? ('float32x2' as const)
              : name === 'values'
                ? ('float32' as const)
                : ('uint32' as const);
          const length = data.length / (format === 'float32x2' ? 2 : 1);
          return [port, {format, length, chunks: [{values: data, length}], pack: () => data}];
        })
      )
    })) as Record<string, Float32Array | Uint32Array>;
    expect(Array.from(output['counts'])).toEqual([1, 2]);
    expect(Array.from(output['featureValues'])).toEqual([7, 24]);
    expect(allocations).not.toContain(6);
  } finally {
    vi.unstubAllGlobals();
  }
});

it.each([
  'zonal',
  'group'
] as const)('W5-14 %s captures get-only sum options once and preserves inherited closed boundaries', async backend => {
  const options: Parameters<typeof computePointsInPolygonsChoroplethOnCPU>[1] = Object.assign(
    Object.create({boundaryRule: 'closed'}),
    {backend, candidateCapacity: 8}
  );
  const reads = new Map<PropertyKey, number>();
  const proxy = new Proxy(options, {
    get(target, key, receiver) {
      reads.set(key, (reads.get(key) ?? 0) + 1);
      return key === 'statistic' ? 'sum' : Reflect.get(target, key, receiver);
    }
  });
  const points = {kind: 'points' as const, positions: Float32Array.of(1, 0.5)};
  const polygons = {
    kind: 'polygons' as const,
    positions: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1),
    featureOffsets: Uint32Array.of(0, 1),
    polygonOffsets: Uint32Array.of(0, 1),
    ringOffsets: Uint32Array.of(0, 4)
  };
  const result = computePointsInPolygonsChoroplethOnCPU(
    {points, polygons, values: Float32Array.of(7)},
    proxy
  );
  // Closed boundary includes the right-edge observation; sum of its one value is exactly seven.
  expect(Array.from(result.counts)).toEqual([1]);
  expect(Array.from(result.featureValues)).toEqual([7]);
  for (const name of ['statistic', 'backend', 'boundaryRule', 'candidateCapacity', 'leafCapacity'])
    expect(reads.get(name)).toBe(1);
  const props: Parameters<typeof createCPUPointsInPolygonsChoroplethOperation>[0] = {
    backend,
    candidateCapacity: 8,
    points: {kind: 'points', positions: {value: 'points', length: 1}},
    polygons: {
      kind: 'polygons',
      positions: {value: 'polygons', length: 4},
      featureOffsets: {value: 'features', length: 2},
      polygonOffsets: {value: 'polygonOffsets', length: 2},
      ringOffsets: {value: 'rings', length: 2}
    },
    values: {value: 'values', length: 1},
    output: {
      counts: 'counts',
      featureValues: 'featureValues',
      overflow: 'overflow',
      uncertainCount: 'uncertainCount'
    }
  };
  Object.setPrototypeOf(props, {boundaryRule: 'closed'});
  const operation = createCPUPointsInPolygonsChoroplethOperation(
    new Proxy(props, {
      get(target, key, receiver) {
        return key === 'statistic' ? 'sum' : Reflect.get(target, key, receiver);
      }
    })
  );
  const rows = {
    points: points.positions,
    polygons: polygons.positions,
    features: polygons.featureOffsets,
    polygonOffsets: polygons.polygonOffsets,
    rings: polygons.ringOffsets,
    values: Float32Array.of(7)
  };
  const output = (await operation.execute({
    inputs: Object.fromEntries(
      Object.entries(operation.inputs).map(([port, name]) => {
        const data = rows[name as keyof typeof rows];
        const format =
          name === 'points' || name === 'polygons'
            ? ('float32x2' as const)
            : name === 'values'
              ? ('float32' as const)
              : ('uint32' as const);
        const length = data.length / (format === 'float32x2' ? 2 : 1);
        return [port, {format, length, chunks: [{values: data, length}], pack: () => data}];
      })
    )
  })) as Record<string, Uint32Array | Float32Array>;
  expect(Array.from(output['counts'])).toEqual([1]);
  expect(Array.from(output['featureValues'])).toEqual([7]);
});

for (const backend of ['zonal', 'group'] as const)
  for (const statistic of ['count', 'sum'] as const)
    it(`TSC-02 UGE-01 ${backend} ${statistic} preserves multipart rows through the published CPU operation`, async () => {
      const points = {
        kind: 'points' as const,
        positions: Float32Array.of(0.25, 0.25, 0.75, 0.75),
        featureOffsets: Uint32Array.of(0, 2),
        sourceIds: Uint32Array.of(701)
      };
      const polygons = {
        kind: 'polygons' as const,
        positions: Float32Array.of(0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1),
        featureOffsets: Uint32Array.of(0, 1, 2),
        polygonOffsets: Uint32Array.of(0, 1, 2),
        ringOffsets: Uint32Array.of(0, 4, 8),
        sourceIds: Uint32Array.of(99, 5)
      };
      const values = Float32Array.of(7);
      const expected = statistic === 'count' ? [1, 1] : [7, 7];
      // Each identical polygon contains both parts of one feature: count/value appears once per row.
      const result = computePointsInPolygonsChoroplethOnCPU(
        {points, polygons, values},
        {backend, statistic, candidateCapacity: 8}
      );
      expect(Array.from(result.counts)).toEqual([1, 1]);
      expect(Array.from(result.featureValues)).toEqual(expected);
      const operation = createCPUPointsInPolygonsChoroplethOperation({
        backend,
        statistic,
        candidateCapacity: 8,
        points: {
          kind: 'points',
          positions: {value: 'points', length: 2},
          featureOffsets: {value: 'pointOffsets', length: 2}
        },
        polygons: {
          kind: 'polygons',
          positions: {value: 'polygons', length: 8},
          featureOffsets: {value: 'features', length: 3},
          polygonOffsets: {value: 'polygonOffsets', length: 3},
          ringOffsets: {value: 'rings', length: 3}
        },
        values: {value: 'values', length: 1},
        output: {
          counts: 'counts',
          ...(statistic === 'count' ? {} : {featureValues: 'featureValues'}),
          overflow: 'overflow',
          uncertainCount: 'uncertainCount'
        }
      });
      const rows = {
        points: points.positions,
        pointOffsets: points.featureOffsets,
        polygons: polygons.positions,
        features: polygons.featureOffsets,
        polygonOffsets: polygons.polygonOffsets,
        rings: polygons.ringOffsets,
        values
      };
      const output = (await operation.execute({
        inputs: Object.fromEntries(
          Object.entries(operation.inputs).map(([port, name]) => {
            const data = rows[name as keyof typeof rows];
            const format =
              name === 'points' || name === 'polygons'
                ? ('float32x2' as const)
                : name === 'values'
                  ? ('float32' as const)
                  : ('uint32' as const);
            const length = data.length / (format === 'float32x2' ? 2 : 1);
            return [port, {format, length, chunks: [{values: data, length}], pack: () => data}];
          })
        )
      })) as Record<string, Uint32Array | Float32Array>;
      expect(Array.from(output['counts'])).toEqual([1, 1]);
      if (statistic !== 'count') expect(Array.from(output['featureValues'])).toEqual([7, 7]);
      expect(Array.from(output['overflow'])).toEqual([0]);
    });
