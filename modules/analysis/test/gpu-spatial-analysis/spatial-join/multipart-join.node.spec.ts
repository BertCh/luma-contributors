// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUCPUVector
} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it, vi} from 'vitest';
import {readFileSync} from 'node:fs';
import {entries as geometryProcessingEntries} from '../../../../../website/src/catalog/entries/geometry-processing';
import {GPUGeometryValidity} from '../../../src/gpu-spatial-analysis/geometry-validity/gpu-geometry-validity';
import {addPointsInPolygonsChoroplethRecipe} from '../../../src/gpu-spatial-analysis/recipes/points-in-polygons-choropleth-recipe';
import * as publicSpatialJoin from '../../../src/gpu-spatial-analysis/spatial-join/index';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {
  GPUGeometryPredicates,
  type GPUGeometryPredicatesGeometry
} from '../../../src/gpu-spatial-analysis/geometry-predicates/gpu-geometry-predicates';
import type {
  GPUPackedGeometryPort,
  GPUPointGeometryPort,
  GPULineGeometryPort,
  GPUMultipartPointGeometryPort,
  GPUMultipartLineGeometryPort
} from '../../../src/analysis-core/contracts/index';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {GPUSpatialJoinCandidates} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-candidates';
import {
  computeSpatialPredicateJoinOnCPU,
  computeNearestFeatureJoinOnCPU,
  type CPUSpatialJoinGeometry
} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';
import {
  createSpatialPredicateJoinOperation,
  createCPUSpatialPredicateJoinOperation,
  createNearestFeatureJoinOperation,
  createCPUNearestFeatureJoinOperation,
  type FeatureJoinGeometryValues
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import type {GPUSpatialJoinGeometry} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-types';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {getGPUCommandNodeKernelSources} from '../../../src/testing/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';

useCPUExecutorTestHooks();

expectTypeOf<GPUMultipartPointGeometryPort>().not.toExtend<GPUPointGeometryPort>();
expectTypeOf<GPUMultipartLineGeometryPort>().not.toExtend<GPULineGeometryPort>();

for (const backend of ['zonal', 'group'] as const)
  it(`x1 X3 ${backend} choropleth rejects grouped points before graph mutation`, () => {
    const {device, graph, positions, integers} = createFixture();
    const points = {
      kind: 'points' as const,
      positions: positions([1, 1, 2, 2]),
      featureOffsets: integers([0, 2])
    };
    const polygons = {
      kind: 'polygons' as const,
      positions: positions([0, 0, 3, 0, 3, 3, 0, 3]),
      featureOffsets: integers([0, 1]),
      polygonOffsets: integers([0, 1]),
      ringOffsets: integers([0, 4]),
      candidateCapacity: 8
    };
    const add = vi.spyOn(graph, 'add');
    try {
      expect(() =>
        addPointsInPolygonsChoroplethRecipe(graph, {
          backend,
          points: {...points},
          polygons
        } as never)
      ).toThrow(/points.featureOffsets.*multipart/);
      expect(add).not.toHaveBeenCalled();
    } finally {
      add.mockRestore();
      device.destroy();
    }
  });

for (const kind of ['points', 'lines'] as const)
  for (const outputCount of [1, 2])
    it(`x1 X3 validity rejects multipart ${kind} with ${outputCount} output rows and spread input`, () => {
      const {device, positions, integers} = createFixture();
      const geometry = {
        kind,
        positions: positions([1, 1, 2, 2]),
        featureOffsets: integers([0, kind === 'points' ? 2 : 1]),
        ...(kind === 'lines' ? {lineOffsets: integers([0, 2])} : {})
      };
      try {
        expect(
          () =>
            new GPUGeometryValidity({
              [kind]: {...geometry},
              output: {
                issueCode: integers(Array(outputCount).fill(0)),
                validity: integers(Array(outputCount).fill(0))
              }
            } as never)
        ).toThrow(/featureOffsets.*multipart/);
      } finally {
        device.destroy();
      }
    });

it('x1 X9 website geometry predicate parameters match the multipart refusal', () => {
  const entry = geometryProcessingEntries.find(entry => entry.symbol === 'GPUGeometryPredicates')!;
  expect(entry.params!.find(parameter => parameter.name === 'geometry')!.type).toBe(
    'GPUGeometryPredicatesGeometry'
  );
  expect(entry.params!.find(parameter => parameter.name === 'other')!.type).toBe(
    'GPUGeometryPredicatesGeometry | undefined'
  );
  for (const name of ['geometry', 'other'])
    expect(entry.params!.find(parameter => parameter.name === name)!.description).toMatch(
      /multipart.*reject/i
    );
});

it('x1 X10 prepared sort docs follow the 255/256 logical-feature threshold and consumers', () => {
  const {device, positions, integers} = createFixture();
  try {
    for (const count of [255, 256]) {
      const prepared = new GPUSpatialJoinPrepared({
        geometry: {
          kind: 'points',
          positions: positions(Array(count * 4).fill(0)),
          featureOffsets: integers(Array.from({length: count + 1}, (_, row) => row * 2))
        }
      });
      expect(prepared.featureCount).toBe(count);
      expect(prepared.spatialSort).toBe(count >= 256);
      prepared.destroy();
    }
    const guide = readFileSync(
      new URL('../../../../../docs/api/gpu-spatial-analysis/operators-joins.md', import.meta.url),
      'utf8'
    );
    expect(guide.includes('`spatialSort` is allowed only for')).toBe(false);
    expect(/GPUSpatialJoinPrepared` still\s+defaults to `false`/.test(guide)).toBe(false);
    expect(/256 logical feature rows/.test(guide)).toBe(true);
    expect(
      /sorted prepared handle[\s\S]{0,250}GPUSpatialPredicateJoin[\s\S]{0,150}GPUSpatialJoinCandidates/.test(
        guide
      )
    ).toBe(true);
  } finally {
    device.destroy();
  }
});

for (const predicate of ['within', 'coveredBy'] as const)
  for (const interior of [true, false])
    it(`x1 X1 CPU ${predicate} preserves oblique boundary intervals (interior=${interior})`, async () => {
      const left: CPUSpatialJoinGeometry = {
        kind: 'lines',
        positions: Float32Array.from(interior ? [0, 0, 3, 9, 0, 5, 1, 6] : [0, 0, 3, 9]),
        lineOffsets: Uint32Array.from(interior ? [0, 2, 4] : [0, 2]),
        featureOffsets: Uint32Array.from(interior ? [0, 2] : [0, 1])
      };
      const right: CPUSpatialJoinGeometry = {
        kind: 'polygons',
        positions: Float32Array.of(0, 0, 2, 6, 3, 9, 0, 9),
        featureOffsets: Uint32Array.of(0, 1),
        polygonOffsets: Uint32Array.of(0, 1),
        ringOffsets: Uint32Array.of(0, 4)
      };
      const result = computeSpatialPredicateJoinOnCPU(left, right, {
        predicate,
        candidateCapacity: 8,
        pairCapacity: 8
      });
      expect(result.status.count).toBe(Number(interior || predicate === 'coveredBy'));
      expect(result.status.overflow).toBe(0);
      expect(result.status.candidateOverflow).toBe(0);
      const inputs: Record<string, GPUCPUVector> = {};
      const operation = createCPUSpatialPredicateJoinOperation({
        left: describeX1CPUValues('left', left, inputs),
        right: describeX1CPUValues('right', right, inputs),
        predicate,
        candidateCapacity: 8,
        pairCapacity: 8,
        output: {
          leftRows: 'leftRows',
          rightRows: 'rightRows',
          count: 'count',
          requiredCount: 'requiredCount',
          overflow: 'overflow',
          candidateOverflow: 'candidateOverflow'
        }
      });
      const actual = (await operation.execute({inputs})) as Record<string, Uint32Array>;
      expect(Array.from(actual['count'])).toEqual([result.status.count]);
      expect(Array.from(actual['overflow'])).toEqual([0]);
      expect(Array.from(actual['candidateOverflow'])).toEqual([0]);
    });

function describeX1CPUValues(
  prefix: string,
  geometry: CPUSpatialJoinGeometry,
  inputs: Record<string, GPUCPUVector>
): FeatureJoinGeometryValues {
  return {
    kind: geometry.kind,
    ...Object.fromEntries(
      Object.entries(geometry)
        .filter(([field, values]) => field !== 'kind' && values !== undefined)
        .map(([field, data]) => {
          const values = data as Float32Array | Uint32Array;
          const name = prefix + field[0].toUpperCase() + field.slice(1),
            format = field === 'positions' ? ('float32x2' as const) : ('uint32' as const);
          const length = values.length / (field === 'positions' ? 2 : 1);
          inputs[name] = {format, length, chunks: [{values, length}], pack: () => values};
          return [field, {value: name, length}];
        })
    )
  } as FeatureJoinGeometryValues;
}

async function runX1CPUOperation(
  queries: CPUSpatialJoinGeometry,
  features: CPUSpatialJoinGeometry
) {
  const inputs: Record<string, GPUCPUVector> = {};
  const operation = createCPUNearestFeatureJoinOperation({
    queries: describeX1CPUValues('queries', queries, inputs),
    features: describeX1CPUValues('features', features, inputs),
    output: {
      neighborRows: 'rows',
      neighborIds: 'ids',
      neighborCounts: 'counts',
      neighborDistances: 'distances',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow'
    }
  });
  return (await operation.execute({inputs})) as Record<string, Uint32Array | Float32Array>;
}

function runX1Nearest(
  query: number[],
  targets: number[],
  lineOffsets?: number[],
  featureOffsets?: number[],
  radii?: number[],
  greatCircle = false
) {
  const {device, graph, memory, positions, integers, read} = createFixture();
  const rows = integers([0]),
    counts = integers([0]);
  const distances = createTransientView(graph, 'x1-distances', 'float32', 1);
  const radius = radii ? createTransientView(graph, 'x1-radius', 'float32', 1) : undefined;
  try {
    const geometry = {
      positions: positions(targets),
      ...(featureOffsets ? {featureOffsets: integers(featureOffsets)} : {})
    };
    const join = new GPUNearestFeatureJoin({
      queries: {kind: 'points', positions: positions(query)},
      features: lineOffsets
        ? {...geometry, kind: 'lines', lineOffsets: integers(lineOffsets)}
        : {...geometry, kind: 'points'},
      spatialContext: greatCircle
        ? {coordinateSpace: 'longitude-latitude', metric: 'great-circle'}
        : undefined,
      radius,
      spatialSort: false,
      output: {
        neighborIds: integers([0]),
        neighborRows: rows,
        neighborCounts: counts,
        neighborDistances: distances,
        status: {overflow: integers([0])}
      }
    });
    const nodes = join.getCommandNodes(graph);
    return (radii ?? [undefined]).map(value => {
      if (radius) memory.write(radius, Float32Array.of(value!));
      executeCommandNodesOnCPU(adaptUniforms(nodes), memory);
      return {
        rows: read(rows),
        counts: read(counts),
        distances: Array.from(memory.readFloat32(distances))
      };
    });
  } finally {
    device.destroy();
  }
}

for (const reverse of [false, true]) {
  const road = reverse ? [1, 1, -8191, -8190] : [-8191, -8190, 1, 1];
  it(`x1 X2 cancellation does not change the nearest winner (reverse=${reverse})`, async () => {
    expect(
      runX1Nearest([0, 0], [...road, 0, -2, 1, -2, 0, 0, 1, 0], [0, 2, 4, 6], [0, 2, 3])[0].rows
    ).toEqual([1]);
    const cpu = await runX1CPUOperation(
      {kind: 'points', positions: Float32Array.of(0, 0)},
      {
        kind: 'lines',
        positions: Float32Array.from([...road, 0, -2, 1, -2, 0, 0, 1, 0]),
        lineOffsets: Uint32Array.of(0, 2, 4, 6),
        featureOffsets: Uint32Array.of(0, 2, 3)
      }
    );
    expect(Array.from(cpu['rows'])).toEqual([1]);
  });
  it(`x1 X2 zero radius rejects the nearby road (reverse=${reverse})`, () => {
    expect(runX1Nearest([0, 0], [...road, 0, -2, 1, -2], [0, 2, 4], [0, 2], [0])[0].counts).toEqual(
      [0]
    );
  });
  it(`x1 X6 CPU endpoint distance survives a distant start (reverse=${reverse})`, async () => {
    const result = computeNearestFeatureJoinOnCPU(
      {kind: 'points', positions: Float32Array.of(1, 0)},
      {
        kind: 'lines',
        positions: Float32Array.from(reverse ? [0, 0, -1e20, 0] : [-1e20, 0, 0, 0]),
        lineOffsets: Uint32Array.of(0, 2)
      }
    );
    expect(Array.from(result.neighborDistances)).toEqual([1]);
    const cpu = await runX1CPUOperation(
      {kind: 'points', positions: Float32Array.of(1, 0)},
      {
        kind: 'lines',
        positions: Float32Array.from(reverse ? [0, 0, -1e20, 0] : [-1e20, 0, 0, 0]),
        lineOffsets: Uint32Array.of(0, 2)
      }
    );
    expect(Array.from(cpu['distances'])).toEqual([1]);
  });
}

it('x1 X4 great-circle negative-zero radius remains zero over three submissions', () => {
  expect(
    runX1Nearest([0, 0], [1, 0], undefined, undefined, [0, -0, 0], true).map(
      result => result.counts
    )
  ).toEqual([[0], [0], [0]]);
});

for (const distance of [1e-20, 1e-24]) {
  it(`x1 X5 small nonzero distance ${distance} ranks after a hit in both forms`, async () => {
    const cpu = computeNearestFeatureJoinOnCPU(
      {kind: 'points', positions: Float32Array.of(0, 0)},
      {kind: 'points', positions: Float32Array.of(distance, 0, 0, 0)}
    );
    expect(Array.from(cpu.neighborRows)).toEqual([1]);
    expect(runX1Nearest([0, 0], [distance, 0, 0, 0])[0].rows).toEqual([1]);
    const operation = await runX1CPUOperation(
      {kind: 'points', positions: Float32Array.of(0, 0)},
      {kind: 'points', positions: Float32Array.of(distance, 0, 0, 0)}
    );
    expect(Array.from(operation['rows'])).toEqual([1]);
    expect(runX1Nearest([0, 0], [distance, 0])[0].distances).toEqual([Math.fround(distance)]);
  });
  it(`x1 X5 zero radius rejects small nonzero distance ${distance}`, () => {
    expect(runX1Nearest([0, 0], [distance, 0], undefined, undefined, [0])[0].counts).toEqual([0]);
  });
}

it('x1 X5 small-range encoding retains squared precision below one unit', () => {
  const targets = [0.5, 0.00015, 0.5, 0];
  const result = computeNearestFeatureJoinOnCPU(
    {kind: 'points', positions: Float32Array.of(0, 0)},
    {kind: 'points', positions: Float32Array.from(targets)}
  );
  expect(Array.from(result.neighborRows)).toEqual([1]);
  expect(runX1Nearest([0, 0], targets)[0].rows).toEqual([1]);
});

it('x1 X5 rank encoding skips nonfinite bit patterns at the exponent boundary', () => {
  const targets = [2 + 2 ** -22, 0, 2 - 2 ** -23, 0];
  expect(runX1Nearest([0, 0], targets)[0].rows).toEqual([1]);
  for (const distance of [2 ** -126, 1e-24, 1e20, Math.fround(3.402823466e38)]) {
    const actual = runX1Nearest([0, 0], [distance, 0])[0];
    expect(actual.counts).toEqual([1]);
    expect(actual.rows).toEqual([0]);
    expect(Number.isFinite(actual.distances[0])).toBe(true);
    expect(actual.distances[0] / Math.fround(distance)).toBeCloseTo(1, 6);
  }
});

for (const grouped of [false, true])
  it(`x1 X7 identity right grouping preserves uncertainCount (grouped=${grouped})`, () => {
    const {device, graph, memory, positions, integers, pairs, read} = createFixture();
    const large = 2 ** 120,
      delta = 2 ** 98,
      small = 2 ** -100;
    const uncertainCount = integers([0]);
    try {
      const join = new GPUSpatialPredicateJoin({
        left: {
          kind: 'lines',
          positions: positions([large / 4, large / 4 + delta, large / 2, large / 2 + delta]),
          lineOffsets: integers([0, 2]),
          featureOffsets: integers([0, 1])
        },
        right: {
          kind: 'lines',
          positions: positions([small, small, large, large]),
          lineOffsets: integers([0, 2]),
          ...(grouped ? {featureOffsets: integers([0, 1])} : {})
        },
        predicate: 'intersects',
        candidateCapacity: 8,
        output: {pairs, status: {uncertainCount}}
      });
      executeCommandNodesOnCPU(adaptUniforms(join.getCommandNodes(graph)), memory);
      expect(read(pairs.status.count)).toEqual([1]);
      expect(read(uncertainCount)).toEqual([1]);
    } finally {
      device.destroy();
    }
  });

it('x1 X8 one-query sorted preflight does not pack metadata', () => {
  const {device, graph, positions, integers} = createFixture();
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 128},
    configurable: true
  });
  try {
    expect(() =>
      new GPUNearestFeatureJoin({
        queries: {
          kind: 'lines',
          positions: positions([0, 0, 1, 0]),
          lineOffsets: integers([...Array(31).fill(0), 2]),
          featureOffsets: integers([0, 31])
        },
        features: {kind: 'points', positions: positions([0, 0])},
        querySort: true,
        spatialSort: false,
        output: {
          neighborIds: integers([0]),
          neighborCounts: integers([0]),
          status: {overflow: integers([0])}
        }
      }).getCommandNodes(graph)
    ).not.toThrow();
  } finally {
    device.destroy();
  }
});
expectTypeOf<
  GPUPackedGeometryPort<GPUMultipartPointGeometryPort>
>().not.toExtend<GPUGeometryPredicatesGeometry>();
expectTypeOf<
  GPUPackedGeometryPort<GPUMultipartLineGeometryPort>
>().not.toExtend<GPUGeometryPredicatesGeometry>();

it('internal feature-join operation factories are absent from the family barrel', () => {
  for (const name of [
    'createSpatialPredicateJoinOperation',
    'createCPUSpatialPredicateJoinOperation',
    'createNearestFeatureJoinOperation',
    'createCPUNearestFeatureJoinOperation'
  ])
    expect(publicSpatialJoin).not.toHaveProperty(name);
});

for (const metric of ['planar', 'great-circle'] as const)
  it(`Metal nearest ${metric} uses finite shader constants and rejects invalid radii on reuse`, () => {
    const {device, graph, memory, positions, integers, read} = createFixture();
    const radius = createTransientView(graph, 'metal-radius', 'float32', 1);
    const neighborRows = integers([0]),
      neighborCounts = integers([0]),
      neighborDistances = createTransientView(graph, 'metal-distances', 'float32', 1);
    try {
      const join = new GPUNearestFeatureJoin({
        queries: {kind: 'points', positions: positions([0, 0])},
        features: {kind: 'points', positions: positions([0, 0])},
        spatialContext:
          metric === 'planar'
            ? {coordinateSpace: 'planar', metric: 'native'}
            : {coordinateSpace: 'longitude-latitude', metric: 'great-circle'},
        radius,
        spatialSort: false,
        output: {
          neighborRows,
          neighborCounts,
          neighborIds: integers([0]),
          neighborDistances,
          status: {overflow: integers([0])}
        }
      });
      const nodes = join.getCommandNodes(graph);
      const clear = getGPUCommandNodeKernelSources(nodes).find(kernel =>
        kernel.source.includes('let limit = maxDistance[')
      );
      expect(clear).toBeDefined();
      // WGSL constant evaluation rejects infinity even when constructed with a bitcast.
      const constants = [...clear!.source.matchAll(/bitcast<f32>\((0x[\da-f]+)u\)/gi)];
      expect(constants.length).toBeGreaterThan(0);
      for (const [, literal] of constants) {
        const value = new Float32Array(Uint32Array.of(Number(literal)).buffer)[0];
        expect(Number.isFinite(value), `nonfinite WGSL constant ${literal}`).toBe(true);
      }
      for (const limit of [0, -1, NaN, Infinity, -Infinity, Math.fround(3.402823466e38), 0]) {
        memory.write(radius, Float32Array.of(limit));
        executeCommandNodesOnCPU(adaptUniforms(nodes), memory);
        const valid = Number.isFinite(limit) && limit >= 0;
        expect(read(neighborCounts)).toEqual([Number(valid)]);
        expect(read(neighborRows)).toEqual([valid ? 0 : 0xffffffff]);
        expect(Array.from(memory.readFloat32(neighborDistances))).toEqual([valid ? 0 : -1]);
      }
    } finally {
      device.destroy();
    }
  });

for (const predicate of ['within', 'coveredBy'] as const) {
  it(`R1 CPU ${predicate} covers an oblique line split at a nonbinary parameter`, () => {
    const result = computeSpatialPredicateJoinOnCPU(
      {kind: 'lines', positions: Float32Array.of(0, 0, 3, 9), lineOffsets: Uint32Array.of(0, 2)},
      {
        kind: 'lines',
        positions: Float32Array.of(0, 0, 2, 6, 2, 6, 3, 9),
        lineOffsets: Uint32Array.of(0, 2, 4),
        featureOffsets: Uint32Array.of(0, 2)
      },
      {predicate, candidateCapacity: 8, pairCapacity: 8}
    );
    expect(result.status.count).toBe(1);
    expect([result.leftRows[0], result.rightRows[0]]).toEqual([0, 0]);
  });
  it(`R2 shader ${predicate} preserves a one-unit gap on a long line and recovers`, () => {
    const {device, memory, positions, integers, run} = createFixture();
    try {
      const left = {
        kind: 'lines' as const,
        positions: positions([-1e8, 0, 1e8, 0]),
        lineOffsets: integers([0, 2])
      };
      const right = {
        kind: 'lines' as const,
        positions: positions([-1e8, 0, 0, 0, 1, 0, 1e8, 0]),
        lineOffsets: integers([0, 2, 4]),
        featureOffsets: integers([0, 2])
      };
      expect(run(left, right, predicate).rows).toEqual([]);
      memory.write(right.positions, Float32Array.of(-1e8, 0, 0, 0, 0, 0, 1e8, 0));
      expect(run(left, right, predicate).rows).toEqual([[0, 0]]);
    } finally {
      device.destroy();
    }
  });
}

for (const targets of [
  [1, 0.0003, 1, 0],
  [1e20, 0]
])
  it(`R${targets.length === 4 ? 4 : 5} nearest ranks before publication and supports finite large gaps`, () => {
    const {device, graph, memory, positions, integers, read} = createFixture();
    const queries = {
      kind: 'points' as const,
      positions: positions([0, 0, -10, 0]),
      featureOffsets: integers([0, 2])
    };
    const features = {kind: 'points' as const, positions: positions(targets)};
    const neighborRows = integers([0]),
      neighborCounts = integers([0]),
      neighborDistances = createTransientView(graph, 'review-distances', 'float32', 1);
    const expectedRow = targets.length === 4 ? 1 : 0;
    try {
      const cpu = computeNearestFeatureJoinOnCPU(
        {
          kind: 'points',
          positions: Float32Array.of(0, 0, -10, 0),
          featureOffsets: Uint32Array.of(0, 2)
        },
        {kind: 'points', positions: Float32Array.from(targets)}
      );
      expect(Array.from(cpu.neighborRows)).toEqual([expectedRow]);
      const join = new GPUNearestFeatureJoin({
        queries,
        features,
        spatialSort: false,
        output: {
          neighborIds: integers([0]),
          neighborRows,
          neighborCounts,
          neighborDistances,
          status: {overflow: integers([0])}
        }
      });
      executeCommandNodesOnCPU(adaptUniforms(join.getCommandNodes(graph)), memory);
      expect(read(neighborRows)).toEqual([expectedRow]);
      expect(read(neighborCounts)).toEqual([1]);
      expect(Array.from(memory.readFloat32(neighborDistances))).toEqual([
        Math.fround(targets.length === 4 ? 1 : 1e20)
      ]);
    } finally {
      device.destroy();
    }
  });

for (const targets of [
  [1e20, 0, 1, 0],
  [2e20, 0, 1e20, 0],
  [1.9e19, 0, 1.8e19, 0]
])
  it(`R5 rank ranges remain ordered for targets ${targets}`, () => {
    const {device, graph, memory, positions, integers, read} = createFixture();
    const queries = {
      kind: 'points' as const,
      positions: positions([0, 0, -10, 0]),
      featureOffsets: integers([0, 2])
    };
    const features = {kind: 'points' as const, positions: positions(targets)};
    const neighborRows = integers([0, 0]),
      neighborCounts = integers([0]);
    const neighborDistances = createTransientView(graph, 'rank-range-distances', 'float32', 2);
    try {
      const join = new GPUNearestFeatureJoin({
        queries,
        features,
        k: 2,
        spatialSort: false,
        output: {
          neighborRows,
          neighborIds: integers([0, 0]),
          neighborCounts,
          neighborDistances,
          status: {overflow: integers([0])}
        }
      });
      executeCommandNodesOnCPU(adaptUniforms(join.getCommandNodes(graph)), memory);
      expect(read(neighborRows)).toEqual([1, 0]);
      expect(read(neighborCounts)).toEqual([2]);
      expect(Array.from(memory.readFloat32(neighborDistances))).toEqual([
        Math.fround(targets[2]),
        Math.fround(targets[0])
      ]);
      const cpu = computeNearestFeatureJoinOnCPU(
        {
          kind: 'points',
          positions: Float32Array.of(0, 0, -10, 0),
          featureOffsets: Uint32Array.of(0, 2)
        },
        {kind: 'points', positions: Float32Array.from(targets)}
      );
      expect(Array.from(cpu.neighborRows)).toEqual([1]);
    } finally {
      device.destroy();
    }
  });

for (const kind of ['points', 'lines'] as const)
  for (const consumer of ['predicate', 'candidates', 'nearest'] as const)
    for (const direction of [
      'ordinary-multipart',
      'multipart-ordinary',
      'different-offsets'
    ] as const)
      it(`R3 ${consumer} refuses mismatched prepared ${kind} feature topology (${direction})`, () => {
        const {device, positions, integers, pairs} = createFixture();
        const ordinary = {
          kind,
          positions: positions(kind === 'points' ? [0, 0, 2, 0] : [0, 0, 1, 0, 2, 0, 3, 0]),
          ...(kind === 'lines' ? {lineOffsets: integers([0, 2, 4])} : {})
        } as GPUSpatialJoinGeometry;
        const first = {...ordinary, featureOffsets: integers([0, 2])};
        const second = {...ordinary, featureOffsets: integers([0, 2])};
        const indexed = direction === 'ordinary-multipart' ? ordinary : first;
        const supplied =
          direction === 'multipart-ordinary'
            ? ordinary
            : direction === 'different-offsets'
              ? second
              : first;
        const prepared = new GPUSpatialJoinPrepared({geometry: indexed});
        try {
          expect(() =>
            consumer === 'predicate'
              ? new GPUSpatialPredicateJoin({
                  left: first,
                  right: supplied,
                  prepared,
                  predicate: 'intersects',
                  candidateCapacity: 8,
                  output: {pairs}
                })
              : consumer === 'candidates'
                ? new GPUSpatialJoinCandidates({
                    left: first,
                    right: supplied,
                    prepared,
                    output: pairs
                  })
                : new GPUNearestFeatureJoin({
                    queries: first,
                    features: supplied,
                    prepared,
                    output: {
                      neighborIds: integers([0]),
                      neighborCounts: integers([0]),
                      status: {overflow: integers([0])}
                    }
                  })
          ).toThrow(/prepared.*same|prepared.*match/);
        } finally {
          prepared.destroy();
          device.destroy();
        }
      });

for (const kind of ['points', 'lines'] as const)
  for (const operand of ['geometry', 'other'] as const)
    it(`R6 geometry predicates refuses spread multipart ${kind} ${operand}`, () => {
      const {device, positions, integers} = createFixture();
      const ordinary = {
        kind,
        positions: positions(kind === 'points' ? [0, 0, 2, 0] : [0, 0, 1, 0, 2, 0, 3, 0]),
        lineOffsets: integers([0, 2, 4])
      };
      const multipart = {...ordinary, featureOffsets: integers([0, 2])};
      try {
        expect(
          () =>
            new GPUGeometryPredicates({
              geometry: ordinary,
              other: ordinary,
              [operand]: {...multipart},
              output: {numGeometries: integers([0, 0])}
            })
        ).toThrow(/featureOffsets.*multipart.*unsupported/);
      } finally {
        device.destroy();
      }
    });

for (const form of ['predicate', 'nearest'] as const)
  for (const oversized of ['positions', 'featureOffsets'] as const)
    it(`R7 ${form} GPU lowering rejects oversized ${oversized} like its CPU twin`, async () => {
      const {device, positions, integers} = createFixture();
      const geometry: FeatureJoinGeometryValues = {
        kind: 'points',
        positions: {value: 'positions', length: 1},
        featureOffsets: {value: 'features', length: 2}
      };
      const predicateProps = {
        left: geometry,
        right: geometry,
        predicate: 'intersects' as const,
        candidateCapacity: 8,
        pairCapacity: 8,
        output: {
          leftRows: 'leftRows',
          rightRows: 'rightRows',
          count: 'count',
          requiredCount: 'requiredCount',
          overflow: 'overflow',
          candidateOverflow: 'candidateOverflow'
        }
      };
      const nearestProps = {
        queries: geometry,
        features: geometry,
        output: {
          neighborRows: 'neighborRows',
          neighborIds: 'neighborIds',
          neighborCounts: 'neighborCounts',
          neighborDistances: 'neighborDistances',
          overflow: 'overflow',
          candidateOverflow: 'candidateOverflow'
        }
      };
      const operation =
        form === 'predicate'
          ? createSpatialPredicateJoinOperation(predicateProps)
          : createNearestFeatureJoinOperation(nearestProps);
      try {
        expect(() =>
          operation.createContributor(
            port =>
              (port.endsWith('Positions')
                ? positions(oversized === 'positions' ? [0, 0, 2, 0] : [0, 0])
                : port.endsWith('FeatureOffsets')
                  ? integers(oversized === 'featureOffsets' ? [0, 1, 1] : [0, 1])
                  : integers(Array(port.endsWith('Rows') ? 8 : 1).fill(0))) as never
          )
        ).toThrow(/must match its declared length/);
        const cpu =
          form === 'predicate'
            ? createCPUSpatialPredicateJoinOperation(predicateProps)
            : createCPUNearestFeatureJoinOperation(nearestProps);
        const inputs: Record<string, GPUCPUVector> = {};
        for (const port of Object.keys(operation.inputs)) {
          const values = port.endsWith('Positions')
            ? Float32Array.from(oversized === 'positions' ? [0, 0, 2, 0] : [0, 0])
            : Uint32Array.from(oversized === 'featureOffsets' ? [0, 1, 1] : [0, 1]);
          const format = port.endsWith('Positions') ? 'float32x2' : 'uint32',
            length = values.length / (format === 'float32x2' ? 2 : 1);
          inputs[port] = {format, length, chunks: [{values, length}], pack: () => values};
        }
        expect(() => cpu.execute({inputs})).toThrow(/must match its declared length/);
      } finally {
        operation.destroy();
        device.destroy();
      }
    });

it('R8 combined sorted metadata preflight leaves graph untouched and permits unsorted retry', () => {
  const {device, graph, positions, integers} = createFixture();
  Object.defineProperty(device, 'limits', {
    value: {...device.limits, maxStorageBufferBindingSize: 128},
    configurable: true
  });
  const queries = {
    kind: 'lines' as const,
    positions: positions([0, 0, 1, 0]),
    lineOffsets: integers([...Array(31).fill(0), 2]),
    featureOffsets: integers([0, 1, 31])
  };
  const features = {kind: 'points' as const, positions: positions([0, 0])};
  const output = {
    neighborIds: integers([0, 0]),
    neighborCounts: integers([0, 0]),
    status: {overflow: integers([0])}
  };
  const allocate = vi.spyOn(graph, 'createTransientBuffer');
  try {
    expect(() =>
      new GPUNearestFeatureJoin({
        id: 'preflight',
        queries,
        features,
        querySort: true,
        spatialSort: false,
        output
      }).getCommandNodes(graph)
    ).toThrow(/metadata.*140 bytes/);
    expect(allocate).not.toHaveBeenCalled();
    expect(
      new GPUNearestFeatureJoin({
        id: 'preflight',
        queries,
        features,
        querySort: false,
        spatialSort: false,
        output
      }).getCommandNodes(graph).length
    ).toBeGreaterThan(0);
  } finally {
    allocate.mockRestore();
    device.destroy();
  }
});

for (const empty of ['queries', 'features'] as const)
  it(`R9 nearest operation factories allow zero ${empty} rows`, async () => {
    const normal: FeatureJoinGeometryValues = {
      kind: 'points',
      positions: {value: 'positions', length: 1}
    };
    const missing: FeatureJoinGeometryValues = {
      kind: 'points',
      positions: {value: 'emptyPositions', length: 0},
      featureOffsets: {value: 'emptyFeatures', length: 1}
    };
    const props = {
      queries: empty === 'queries' ? missing : normal,
      features: empty === 'features' ? missing : normal,
      output: {
        neighborRows: 'neighborRows',
        neighborIds: 'neighborIds',
        neighborCounts: 'neighborCounts',
        neighborDistances: 'neighborDistances',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow'
      }
    };
    expect(() => createNearestFeatureJoinOperation(props)).not.toThrow();
    expect(() => createCPUNearestFeatureJoinOperation(props)).not.toThrow();
    const {device, graph, memory, integers, positions, read} = createFixture();
    const operation = createNearestFeatureJoinOperation(props),
      cpu = createCPUNearestFeatureJoinOperation(props);
    const views: Record<string, GraphDataView> = {},
      inputs: Record<string, GPUCPUVector> = {};
    for (const port of Object.keys(operation.inputs)) {
      const isPositions = port.endsWith('Positions'),
        isEmpty = port.startsWith(empty),
        values = isPositions
          ? isEmpty
            ? new Float32Array(0)
            : Float32Array.of(0, 0)
          : Uint32Array.of(0);
      const format = isPositions ? 'float32x2' : 'uint32',
        length = values.length / (isPositions ? 2 : 1);
      inputs[port] = {format, length, chunks: [{values, length}], pack: () => values};
      views[port] = isPositions ? positions(Array.from(values)) : integers(Array.from(values));
    }
    for (const [port, definition] of Object.entries(operation.outputs))
      views[port] = createTransientView(
        graph,
        `empty-${port}`,
        definition.format,
        definition.length
      );
    try {
      const contributor = operation.createContributor(
        port => views[port] as never
      ) as GPUNearestFeatureJoin;
      executeCommandNodesOnCPU(adaptUniforms(contributor.getCommandNodes(graph)), memory);
      const result = (await cpu.execute({inputs})) as Record<string, Uint32Array | Float32Array>;
      for (const port of ['neighborRows', 'neighborCounts', 'neighborIds', 'neighborDistances']) {
        const actual =
          port === 'neighborDistances'
            ? Array.from(memory.readFloat32(views[port]))
            : read(views[port] as GraphDataView<'uint32'>);
        expect(actual).toEqual(Array.from(result[port]));
      }
      expect(Array.from(result['neighborRows'])).toEqual(empty === 'queries' ? [] : [0xffffffff]);
      expect(Array.from(result['neighborCounts'])).toEqual(empty === 'queries' ? [] : [0]);
      expect(Array.from(result['neighborDistances'])).toEqual(empty === 'queries' ? [] : [-1]);
    } finally {
      operation.destroy();
      device.destroy();
    }
  });

it('R10 disjoint x ranges prune indexed bounds rather than copying every target prefix', () => {
  const positions = (start: number) =>
    Float32Array.from(Array.from({length: 1024}, (_, row) => [start + row, 0]).flat());
  const original = Array.prototype.slice;
  let copied = 0;
  const spy = vi.spyOn(Array.prototype, 'slice').mockImplementation(function (
    this: unknown[],
    start?: number,
    end?: number
  ) {
    const result = original.call(this, start, end);
    if (this.length === 1024) copied += result.length;
    return result;
  });
  let result: ReturnType<typeof computeSpatialPredicateJoinOnCPU>;
  try {
    result = computeSpatialPredicateJoinOnCPU(
      {kind: 'points', positions: positions(2000)},
      {kind: 'points', positions: positions(0)},
      {predicate: 'intersects', candidateCapacity: 8, pairCapacity: 8}
    );
  } finally {
    spy.mockRestore();
  }
  expect(result.status.count).toBe(0);
  expect(copied).toBeLessThanOrEqual(2048);
  expect(result.candidateBoundsTestCount).toBe(1024);
});

it('standalone candidate bounds preserve multipart feature rows', () => {
  const {device, graph, memory, positions, integers, pairs, read} = createFixture();
  try {
    const left = {
      kind: 'points' as const,
      positions: positions([-100, 0, 100, 0]),
      featureOffsets: integers([0, 2])
    };
    const right = {kind: 'points' as const, positions: positions([-100, 0, 100, 0])};
    const candidates = new GPUSpatialJoinCandidates({left, right, output: pairs});
    executeCommandNodesOnCPU(adaptUniforms(candidates.getCommandNodes(graph)), memory);
    expect(read(pairs.status.count)).toEqual([2]);
    expect(read(pairs.leftRows).slice(0, 2)).toEqual([0, 0]);
    expect(read(pairs.rightRows).slice(0, 2)).toEqual([0, 1]);
  } finally {
    device.destroy();
  }
});

// The interpreter only inventories storage declarations. Preserve the uniform words and layout,
// changing their declaration for interpretation alone (production WGSL is unchanged).
function adaptUniforms(nodes: readonly unknown[]): unknown[] {
  type Node = {
    compile(context: unknown): {encode(context: Record<string, unknown>): void; destroy?(): void};
  };
  type Pipeline = {source: string; constants: Record<string, number>};
  type Pass = {setPipeline(pipeline: Pipeline): void};
  return nodes.map(value => {
    const node = value as Node;
    return {
      ...node,
      compile(context: unknown) {
        // The shared executor's NullBuffer shim retains later writes, but not createBuffer(data).
        // Preserve initial uniform uploads here, without changing any computed shader values.
        type BufferDevice = {
          createBuffer(props: {data?: ArrayBufferView}): {write(data: ArrayBufferView): void};
        };
        const device = (context as {device: BufferDevice}).device;
        const createBuffer = device.createBuffer;
        device.createBuffer = function (props) {
          const buffer = createBuffer.call(this, props);
          if (props.data) buffer.write(props.data);
          return buffer;
        };
        let compiled: ReturnType<Node['compile']>;
        try {
          compiled = node.compile(context);
        } finally {
          device.createBuffer = createBuffer;
        }
        return {
          ...compiled,
          encode(encoding: Record<string, unknown>) {
            const pass = encoding['computePass'] as Pass;
            const proxy = new Proxy(pass, {
              get(target, key) {
                if (key === 'setPipeline')
                  return (pipeline: Pipeline) =>
                    target.setPipeline({
                      ...pipeline,
                      source: pipeline.source
                        .replace(
                          'var<uniform> graphChunkGroupParameters: GraphChunkGroupParameters',
                          'var<storage, read> graphChunkGroupParameters: array<u32>'
                        )
                        .replace(
                          'graphChunkGroupParameters.words[word / 4u][word % 4u]',
                          'graphChunkGroupParameters[word]'
                        )
                        .replace(
                          /var<uniform> selectionSettings: array<vec4<u32>, \d+>/,
                          'var<storage, read> selectionSettings: array<u32>'
                        )
                        .replace(
                          /selectionSettings\[(\d+)\]\.([xyzw])/g,
                          (_, row, component) =>
                            `selectionSettings[${Number(row) * 4 + 'xyzw'.indexOf(component)}]`
                        )
                        .replace(
                          /var<uniform> chunkSettings: array<vec4<u32>, \d+>/,
                          'var<storage, read> chunkSettings: array<u32>'
                        )
                        .replace(
                          /chunkSettings\[(\d+)\]\[(\d+)\]/g,
                          (_, row, component) =>
                            `chunkSettings[${Number(row) * 4 + Number(component)}]`
                        )
                    });
                if (key === 'setBindings')
                  return (bindings: Record<string, unknown>) => {
                    bindings = Object.fromEntries(
                      Object.entries(bindings).map(([name, value]) => {
                        const buffer = value as {backing?: ArrayBuffer; byteLength: number};
                        return [
                          name,
                          buffer.backing ? {buffer, offset: 0, size: buffer.byteLength} : value
                        ];
                      })
                    );
                    return (Reflect.get(target, key) as (value: unknown) => void)(bindings);
                  };
                return Reflect.get(target, key);
              }
            });
            compiled.encode({...encoding, computePass: proxy});
          }
        };
      }
    };
  });
}

it('multipart nearest minimizes over both query and target parts with feature-row outputs', () => {
  const {device, graph, memory, positions, integers, read} = createFixture();
  const queries = {
    kind: 'points' as const,
    positions: positions([-100, 0, 2, 0]),
    featureOffsets: integers([0, 2])
  };
  const features = {
    kind: 'lines' as const,
    positions: positions([99, 0, 99, 1, 3, -1, 3, 1, 20, 0, 20, 1]),
    lineOffsets: integers([0, 2, 4, 6]),
    featureOffsets: integers([0, 2, 3]),
    sourceIds: integers([42, 42])
  };
  const neighborIds = integers([0]);
  const neighborRows = integers([0]);
  const neighborCounts = integers([0]);
  const neighborDistances = createTransientView(graph, 'distances', 'float32', 1);
  try {
    const join = new GPUNearestFeatureJoin({
      queries,
      features,
      spatialSort: false,
      output: {
        neighborIds,
        neighborRows,
        neighborCounts,
        neighborDistances,
        status: {overflow: integers([0]), candidateOverflow: integers([0])}
      }
    });
    executeCommandNodesOnCPU(adaptUniforms(join.getCommandNodes(graph)), memory);
    expect(read(neighborRows)).toEqual([0]);
    expect(read(neighborIds)).toEqual([42]);
    expect(read(neighborCounts)).toEqual([1]);
    expect(Array.from(memory.readFloat32(neighborDistances))).toEqual([1]);
  } finally {
    device.destroy();
  }
});

function createFixture() {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let serial = 0;
  const integers = (values: number[]) => {
    const view = createTransientView(graph, `integers-${serial++}`, 'uint32', values.length);
    memory.write(view, Uint32Array.from(values));
    return view;
  };
  const positions = (values: number[]) => {
    const view = createTransientView(
      graph,
      `positions-${serial++}`,
      'float32x2',
      values.length / 2
    );
    memory.write(view, Float32Array.from(values));
    return view;
  };
  const pairs = {
    leftRows: integers(Array(8).fill(0)),
    rightRows: integers(Array(8).fill(0)),
    status: {
      count: integers([0]),
      requiredCount: integers([0]),
      overflow: integers([0]),
      candidateOverflow: integers([0])
    }
  };
  const read = (view: GraphDataView<'uint32'>) => Array.from(memory.readUint32(view));
  const run = (
    left: GPUSpatialJoinGeometry,
    right: GPUSpatialJoinGeometry,
    predicate: 'intersects' | 'within' | 'coveredBy',
    candidateCapacity = 8,
    leafCapacity?: number
  ) => {
    const candidateTruncated = integers(
      Array(
        left.featureOffsets !== undefined
          ? left.featureOffsets.length - 1
          : left.kind === 'points'
            ? left.positions.length
            : left.kind === 'lines'
              ? left.lineOffsets.length - 1
              : 0
      ).fill(0)
    );
    const join = new GPUSpatialPredicateJoin({
      id: `join-${serial++}`,
      left,
      right,
      predicate,
      candidateCapacity,
      leafCapacity,
      output: {pairs, candidateTruncated}
    });
    const nodes = join.getCommandNodes(graph);
    executeCommandNodesOnCPU(adaptUniforms(nodes), memory);
    const count = read(pairs.status.count)[0];
    return {
      rows: read(pairs.leftRows)
        .slice(0, count)
        .map((row, index) => [row, read(pairs.rightRows)[index]]),
      loss: read(pairs.status.candidateOverflow)[0],
      truncated: read(candidateTruncated),
      overflow: read(pairs.status.overflow)[0]
    };
  };
  return {device, graph, memory, integers, positions, pairs, read, run};
}

it('candidate loss on a later part marks the owning feature without final overflow', () => {
  const {device, positions, integers, run} = createFixture();
  try {
    const left = {
      kind: 'points' as const,
      positions: positions([-100, 0, 100, 0]),
      featureOffsets: integers([0, 2])
    };
    const right = {kind: 'points' as const, positions: positions([-100, 0, 100, 0])};
    expect(run(left, right, 'intersects', 1)).toEqual({
      rows: [[0, 0]],
      loss: 1,
      truncated: [1],
      overflow: 0
    });
  } finally {
    device.destroy();
  }
});

it('within and coveredBy reduce all parts and distinguish boundary-only features', () => {
  const {device, positions, integers, run} = createFixture();
  try {
    const left = {
      kind: 'points' as const,
      positions: positions([0, 0, 4, 0, 0, 0, 2, 0]),
      featureOffsets: integers([0, 2, 4])
    };
    const right = {
      kind: 'lines' as const,
      positions: positions([0, 0, 2, 0, 2, 0, 4, 0]),
      lineOffsets: integers([0, 2, 4]),
      featureOffsets: integers([0, 2])
    };
    expect(run(left, right, 'within').rows).toEqual([[1, 0]]);
    expect(run(left, right, 'coveredBy').rows).toEqual([
      [0, 0],
      [1, 0]
    ]);
  } finally {
    device.destroy();
  }
});

it('a target feature whose parts straddle separate leaves keeps its feature row', () => {
  const {device, positions, integers, run} = createFixture();
  try {
    const left = {kind: 'points' as const, positions: positions([100, 0])};
    const right = {
      kind: 'points' as const,
      positions: positions([-100, 0, 100, 0, 0, 0]),
      featureOffsets: integers([0, 2, 3])
    };
    expect(run(left, right, 'intersects').rows).toEqual([[0, 0]]);
    expect(run(left, right, 'intersects', 8, 1).loss).toBe(1);
  } finally {
    device.destroy();
  }
});

for (const predicate of ['intersects', 'within', 'coveredBy'] as const) {
  it(`GPU operation and CPU twin agree on multipart ${predicate}, candidate loss and final capacity`, async () => {
    const {device, graph, memory, positions, integers, read} = createFixture();
    const left = {
      kind: 'points' as const,
      positions: positions([-100, 0, 100, 0, 1, 0, 2, 0]),
      featureOffsets: integers([0, 2, 4])
    };
    const right = {
      kind: 'lines' as const,
      positions: positions([-100, 0, 0, 0, 0, 0, 100, 0]),
      lineOffsets: integers([0, 2, 4]),
      featureOffsets: integers([0, 1, 2])
    };
    const views: Record<string, GraphDataView> = {};
    const cpuInputs: Record<string, GPUCPUVector> = {};
    const describe = (
      prefix: string,
      geometry: typeof left | typeof right
    ): FeatureJoinGeometryValues =>
      ({
        kind: geometry.kind,
        ...Object.fromEntries(
          Object.entries(geometry)
            .filter(([field]) => field !== 'kind')
            .map(([field, value]) => {
              const view = value as GraphDataView;
              const port = prefix + field[0].toUpperCase() + field.slice(1);
              views[port] = view;
              const values =
                view.format === 'float32x2' ? memory.readFloat32(view) : memory.readUint32(view);
              cpuInputs[port] = {
                format: view.format as 'float32x2' | 'uint32',
                length: view.length,
                chunks: [{values, length: view.length}],
                pack: () => values
              };
              return [field, {value: port, length: view.length}];
            })
        )
      }) as FeatureJoinGeometryValues;
    const descriptorLeft = describe('left', left),
      descriptorRight = describe('right', right);
    const output = {
      leftRows: 'leftRows',
      rightRows: 'rightRows',
      count: 'count',
      requiredCount: 'requiredCount',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow',
      candidateTruncated: 'candidateTruncated'
    };
    try {
      for (const candidateCapacity of [1, 8])
        for (const pairCapacity of [1, 8]) {
          const props = {
            id: `operation-${candidateCapacity}-${pairCapacity}`,
            left: descriptorLeft,
            right: descriptorRight,
            predicate,
            candidateCapacity,
            pairCapacity,
            output
          };
          const gpu = createSpatialPredicateJoinOperation(props),
            cpu = createCPUSpatialPredicateJoinOperation(props);
          for (const [port, definition] of Object.entries(gpu.outputs))
            views[port] = integers(Array(definition.length).fill(0));
          const contributor = gpu.createContributor(
            port => views[port] as never
          ) as GPUSpatialPredicateJoin;
          executeCommandNodesOnCPU(adaptUniforms(contributor.getCommandNodes(graph)), memory);
          const result = (await cpu.execute({inputs: cpuInputs})) as Record<string, Uint32Array>;
          for (const name of [
            'count',
            'requiredCount',
            'overflow',
            'candidateOverflow',
            'candidateTruncated'
          ])
            expect(read(views[name] as GraphDataView<'uint32'>), name).toEqual(
              Array.from(result[name])
            );
          const count = result['count'][0];
          for (const name of ['leftRows', 'rightRows'])
            expect(read(views[name] as GraphDataView<'uint32'>).slice(0, count), name).toEqual(
              Array.from(result[name]).slice(0, count)
            );
          gpu.destroy();
        }
    } finally {
      device.destroy();
    }
  });
}

it('CPU nearest uses every part and operation factories retain the same feature layout', async () => {
  const queries: CPUSpatialJoinGeometry = {
    kind: 'points',
    positions: Float32Array.of(-100, 0, 2, 0),
    featureOffsets: Uint32Array.of(0, 2)
  };
  const features: CPUSpatialJoinGeometry = {
    kind: 'lines',
    positions: Float32Array.of(99, 0, 99, 1, 3, -1, 3, 1, 20, 0, 20, 1),
    featureOffsets: Uint32Array.of(0, 2, 3),
    lineOffsets: Uint32Array.of(0, 2, 4, 6),
    sourceIds: Uint32Array.of(42, 42)
  };
  const result = computeNearestFeatureJoinOnCPU(queries, features);
  expect(Array.from(result.neighborRows)).toEqual([0]);
  expect(Array.from(result.neighborIds)).toEqual([42]);
  expect(Array.from(result.neighborDistances)).toEqual([1]);
  const inputs: Record<string, GPUCPUVector> = {};
  const describe = (prefix: string, geometry: CPUSpatialJoinGeometry): FeatureJoinGeometryValues =>
    ({
      kind: geometry.kind,
      ...Object.fromEntries(
        Object.entries(geometry)
          .filter(([field]) => field !== 'kind')
          .map(([field, data]) => {
            const values = data as Float32Array | Uint32Array;
            const name = prefix + field[0].toUpperCase() + field.slice(1),
              length = values.length / (field === 'positions' ? 2 : 1);
            inputs[name] = {
              format: field === 'positions' ? 'float32x2' : 'uint32',
              length,
              chunks: [{values, length}],
              pack: () => values
            };
            return [field, {value: name, length}];
          })
      )
    }) as FeatureJoinGeometryValues;
  const props = {
    queries: describe('queries', queries),
    features: describe('features', features),
    output: {
      neighborRows: 'neighborRows',
      neighborIds: 'neighborIds',
      neighborCounts: 'neighborCounts',
      neighborDistances: 'neighborDistances',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow'
    }
  };
  const gpu = createNearestFeatureJoinOperation(props),
    cpu = createCPUNearestFeatureJoinOperation(props);
  expect(gpu.outputs['neighborRows'].length).toBe(1);
  const actual = (await cpu.execute({inputs})) as Record<string, Uint32Array | Float32Array>;
  expect(Array.from(actual['neighborRows'])).toEqual([0]);
  expect(Array.from(actual['neighborIds'])).toEqual([42]);
  expect(Array.from(actual['neighborDistances'])).toEqual([1]);
  const containment = computeSpatialPredicateJoinOnCPU(queries, features, {
    predicate: 'coveredBy',
    candidateCapacity: 8,
    pairCapacity: 8
  });
  expect(containment.status.count).toBe(0);
});

for (const reverse of [false, true])
  it(`nearest detects containment of a later multipart point (${reverse ? 'target' : 'query'})`, () => {
    const {device, graph, memory, positions, integers} = createFixture();
    const multipart = {
      kind: 'points' as const,
      positions: positions([-100, 0, 2, 0]),
      featureOffsets: integers([0, 2])
    };
    const polygon = {
      kind: 'polygons' as const,
      positions: positions([1, -1, 3, -1, 3, 1, 1, 1]),
      featureOffsets: integers([0, 1]),
      polygonOffsets: integers([0, 1]),
      ringOffsets: integers([0, 4])
    };
    const neighborDistances = createTransientView(graph, 'distances', 'float32', 1);
    try {
      const join = new GPUNearestFeatureJoin({
        queries: reverse ? polygon : multipart,
        features: reverse ? multipart : polygon,
        spatialSort: false,
        output: {
          neighborIds: integers([0]),
          neighborCounts: integers([0]),
          neighborDistances,
          status: {overflow: integers([0])}
        }
      });
      executeCommandNodesOnCPU(adaptUniforms(join.getCommandNodes(graph)), memory);
      expect(Array.from(memory.readFloat32(neighborDistances))).toEqual([0]);
    } finally {
      device.destroy();
    }
  });

for (const querySort of [false, true])
  it(`multipart line nearest preserves owners with querySort=${querySort}`, () => {
    const {device, graph, memory, positions, integers, read} = createFixture();
    const queries = {
      kind: 'lines' as const,
      positions: positions([-100, 0, -99, 0, 2, 0, 2.5, 0, 19, 0, 19.5, 0]),
      lineOffsets: integers([0, 2, 4, 6]),
      featureOffsets: integers([0, 2, 3])
    };
    const features = {
      kind: 'lines' as const,
      positions: positions([99, 0, 99, 1, 3, -1, 3, 1, 20, 0, 20, 1]),
      lineOffsets: integers([0, 2, 4, 6]),
      featureOffsets: integers([0, 2, 3])
    };
    const neighborRows = integers([0, 0]),
      neighborCounts = integers([0, 0]);
    const neighborDistances = createTransientView(graph, 'distances', 'float32', 2);
    try {
      const join = new GPUNearestFeatureJoin({
        queries,
        features,
        querySort,
        spatialSort: false,
        output: {
          neighborRows,
          neighborIds: integers([0, 0]),
          neighborCounts,
          neighborDistances,
          status: {overflow: integers([0])}
        }
      });
      executeCommandNodesOnCPU(adaptUniforms(join.getCommandNodes(graph)), memory);
      expect(read(neighborRows)).toEqual([0, 1]);
      expect(read(neighborCounts)).toEqual([1, 1]);
      expect(Array.from(memory.readFloat32(neighborDistances))).toEqual([0.5, 0.5]);
    } finally {
      device.destroy();
    }
  });

it('empty multipart features and empty line parts remain distinct from missing nearest rows', () => {
  const {device, graph, memory, positions, integers, read} = createFixture();
  const queries = {
    kind: 'points' as const,
    positions: positions([0, 0, 2, 0]),
    featureOffsets: integers([0, 0, 2, 2])
  };
  const features = {
    kind: 'lines' as const,
    positions: positions([1, -1, 1, 1]),
    lineOffsets: integers([0, 0, 2]),
    featureOffsets: integers([0, 0, 2, 2])
  };
  const neighborRows = integers([0, 0, 0]),
    neighborCounts = integers([0, 0, 0]);
  const neighborDistances = createTransientView(graph, 'empty-distances', 'float32', 3);
  try {
    const join = new GPUNearestFeatureJoin({
      queries,
      features,
      spatialSort: false,
      output: {
        neighborRows,
        neighborIds: integers([0, 0, 0]),
        neighborCounts,
        neighborDistances,
        status: {overflow: integers([0])}
      }
    });
    executeCommandNodesOnCPU(adaptUniforms(join.getCommandNodes(graph)), memory);
    expect(read(neighborRows)).toEqual([0xffffffff, 1, 0xffffffff]);
    expect(read(neighborCounts)).toEqual([0, 1, 0]);
    expect(Array.from(memory.readFloat32(neighborDistances))).toEqual([-1, 1, -1]);
    const cpu = computeNearestFeatureJoinOnCPU(
      {
        kind: 'points',
        positions: Float32Array.of(0, 0, 2, 0),
        featureOffsets: Uint32Array.of(0, 0, 2, 2)
      },
      {
        kind: 'lines',
        positions: Float32Array.of(1, -1, 1, 1),
        lineOffsets: Uint32Array.of(0, 0, 2),
        featureOffsets: Uint32Array.of(0, 0, 2, 2)
      }
    );
    expect(Array.from(cpu.neighborRows)).toEqual([0xffffffff, 1, 0xffffffff]);
    expect(Array.from(cpu.neighborCounts)).toEqual([0, 1, 0]);
    expect(Array.from(cpu.neighborDistances)).toEqual([-1, 1, -1]);
  } finally {
    device.destroy();
  }
});

for (const kind of ['points', 'lines'] as const) {
  for (const predicate of ['intersects', 'within', 'coveredBy'] as const) {
    it(`multipart ${kind} ${predicate} reduces parts to feature rows`, () => {
      const fixture = createFixture();
      const {positions, integers, run, device} = fixture;
      const left = {
        kind,
        positions: positions(
          kind === 'points' ? [-10, 0, 2, 0, 30, 0] : [-10, 0, -9, 0, 2, 0, 3, 0, 30, 0, 31, 0]
        ),
        lineOffsets: integers([0, 2, 4, 6]),
        featureOffsets: integers([0, 2, 3])
      };
      const right = {
        kind: 'polygons' as const,
        positions: positions([0, -1, 5, -1, 5, 1, 0, 1, 20, -1, 40, -1, 40, 1, 20, 1]),
        featureOffsets: integers([0, 1, 2]),
        polygonOffsets: integers([0, 1, 2]),
        ringOffsets: integers([0, 4, 8])
      };
      try {
        expect(run(left, right, predicate).rows).toEqual(
          predicate === 'intersects'
            ? [
                [0, 0],
                [1, 1]
              ]
            : [[1, 1]]
        );
      } finally {
        device.destroy();
      }
    });
  }
  it(`multipart ${kind} against multipart targets emits one feature pair`, () => {
    const {positions, integers, run, device} = createFixture();
    const left = {
      kind,
      positions: positions(kind === 'points' ? [-10, 0, 2, 0] : [-10, 0, -9, 0, 2, 0, 3, 0]),
      lineOffsets: integers([0, 2, 4]),
      featureOffsets: integers([0, 2])
    };
    const right = {
      kind,
      positions: positions(
        kind === 'points' ? [99, 0, 2, 0, 100, 0] : [99, 0, 99, 1, 2, -1, 2, 1, 100, 0, 100, 1]
      ),
      lineOffsets: integers([0, 2, 4, 6]),
      featureOffsets: integers([0, 2, 3])
    };
    try {
      expect(run(left, right, 'intersects').rows).toEqual([[0, 0]]);
    } finally {
      device.destroy();
    }
  });
}
