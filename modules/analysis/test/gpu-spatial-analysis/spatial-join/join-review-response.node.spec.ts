// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView, type GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {adaptMultipartNearestUniforms as adaptUniforms} from './multipart-nearest-executor';
import {createJoinReviewCases} from './join-review-response-fixture';
import {
  computeNearestFeatureJoinOnCPU,
  computeSpatialPredicateJoinOnCPU
} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';
import {
  createCPUNearestFeatureJoinOperation,
  createSpatialPredicateJoinOperation,
  createCPUSpatialPredicateJoinOperation
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import {snapshotConstructionSpatialContext} from '../../../src/analysis-core/kernels/construction-snapshot';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';

useCPUExecutorTestHooks();
for (const entry of createJoinReviewCases())
  it(entry.name, () => {
    const device = createNullWebGPUDevice(),
      graph = new GPUCommandGraph(device),
      memory = createCPUGraphMemory();
    const input = Object.fromEntries(
      Object.entries(entry.inputs).map(([name, definition]) => {
        const view = createTransientView(
          graph,
          `input-${name}`,
          definition.format,
          definition.values.length / (definition.format === 'float32x2' ? 2 : 1)
        );
        memory.write(view, definition.values);
        return [name, view];
      })
    );
    const output = Object.fromEntries(
      Object.entries(entry.outputs).map(([name, definition]) => {
        const view = createTransientView(
          graph,
          `output-${name}`,
          definition.format,
          definition.length
        );
        memory.write(
          view,
          definition.format === 'uint32'
            ? new Uint32Array(definition.length).fill(99)
            : new Float32Array(
                definition.length * (definition.format === 'float32x2' ? 2 : 1)
              ).fill(NaN)
        );
        return [name, view];
      })
    );
    try {
      const nodes = adaptUniforms(entry.declare(graph, input, output));
      for (let submission = 0; submission < 2; submission++) {
        executeCommandNodesOnCPU(nodes, memory);
        for (const [name, expected] of Object.entries(entry.expected)) {
          const actual =
            output[name].format === 'uint32'
              ? memory.readUint32(output[name])
              : memory.readFloat32(output[name]);
          expected.forEach((value, row) => {
            expect(Number.isFinite(actual[row]), name).toBe(true);
            expect(Math.abs(actual[row] - value), name).toBeLessThanOrEqual(entry.tolerance ?? 0);
          });
        }
      }
    } finally {
      device.destroy();
    }
  });

const vector = (values: Float32Array | Uint32Array, positions = false): GPUCPUVector => {
  const length = values.length / (positions ? 2 : 1),
    format = positions ? ('float32x2' as const) : ('uint32' as const);
  return {format, length, chunks: [{values, length}], pack: () => values};
};
for (const reverse of [false, true])
  it(`R2 CPU duplicate-road exact tie reverse=${reverse}`, async () => {
    const road = reverse ? [-65535, -65534, 1, 1] : [1, 1, -65535, -65534];
    const values = Float32Array.from([...road, ...[road[2], road[3], road[0], road[1]]]);
    const queries = {kind: 'points' as const, positions: Float32Array.of(0, 0)},
      targets = {
        kind: 'lines' as const,
        positions: values,
        lineOffsets: Uint32Array.of(0, 2, 4),
        sourceIds: Uint32Array.from(reverse ? [42, 33] : [33, 42])
      };
    const actual = computeNearestFeatureJoinOnCPU(queries, targets);
    expect(Array.from(actual.neighborRows)).toEqual([0]);
    expect(Array.from(actual.neighborIds)).toEqual([reverse ? 42 : 33]);
    expect(Array.from(actual.neighborDistances)).toEqual([Math.fround(0.00001078967579815071)]);
    const operation = createCPUNearestFeatureJoinOperation({
      queries: {kind: 'points', positions: {value: 'q', length: 1}},
      features: {
        kind: 'lines',
        positions: {value: 'p', length: 4},
        lineOffsets: {value: 'l', length: 3},
        sourceIds: {value: 'i', length: 2}
      },
      output: {
        neighborRows: 'rows',
        neighborIds: 'ids',
        neighborCounts: 'counts',
        neighborDistances: 'distances',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow'
      }
    });
    const result = (await operation.execute({
      inputs: {
        queriesPositions: vector(queries.positions, true),
        featuresPositions: vector(values, true),
        featuresLineOffsets: vector(targets.lineOffsets),
        featuresSourceIds: vector(targets.sourceIds)
      }
    })) as Record<string, Uint32Array>;
    expect(
      Object.fromEntries(Object.entries(result).map(([name, values]) => [name, Array.from(values)]))
    ).toEqual({
      rows: [0],
      ids: [reverse ? 42 : 33],
      counts: [1],
      distances: [Math.fround(0.00001078967579815071)],
      overflow: [0],
      candidateOverflow: [0]
    });
  });
for (const create of [createSpatialPredicateJoinOperation, createCPUSpatialPredicateJoinOperation])
  for (const field of ['relate', 'unmatchedRows', 'candidateTruncated', 'uncertainCount'])
    it(`R6 ${create.name} omits unset ${field}`, () => {
      let reads = 0;
      const output = Object.assign(
        Object.create({
          get [field]() {
            reads++;
            return undefined;
          }
        }),
        {
          leftRows: 'leftRows',
          rightRows: 'rightRows',
          count: 'count',
          requiredCount: 'requiredCount',
          overflow: 'overflow',
          candidateOverflow: 'candidateOverflow'
        }
      );
      const operation = create({
        left: {kind: 'points', positions: {value: 'left', length: 1}},
        right: {kind: 'points', positions: {value: 'right', length: 1}},
        predicate: 'intersects',
        candidateCapacity: 1,
        pairCapacity: 1,
        output
      });
      expect(Object.keys(operation.contract.outputs ?? {}).sort()).toEqual([
        'candidateOverflow',
        'count',
        'leftRows',
        'overflow',
        'requiredCount',
        'rightRows'
      ]);
      expect(reads).toBe(1);
    });
for (const create of [createSpatialPredicateJoinOperation, createCPUSpatialPredicateJoinOperation])
  it(`R6 ${create.name} still validates an unset required destination`, () => {
    const output = {
      leftRows: 'leftRows',
      rightRows: 'rightRows',
      count: undefined as unknown as string,
      requiredCount: 'requiredCount',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow'
    };
    expect(() =>
      create({
        left: {kind: 'points', positions: {value: 'left', length: 1}},
        right: {kind: 'points', positions: {value: 'right', length: 1}},
        predicate: 'intersects',
        candidateCapacity: 1,
        pairCapacity: 1,
        output
      })
    ).toThrow(/outputs.count.value.*non-empty graph value/);
  });
for (const reverse of [false, true])
  it(`R7 exact dwithin threshold reverse=${reverse}`, async () => {
    const end = Math.fround(0.01),
      road = reverse ? [end, end, -8191, -8190] : [-8191, -8190, end, end];
    const result = computeSpatialPredicateJoinOnCPU(
      {kind: 'points', positions: Float32Array.of(0, 0)},
      {kind: 'lines', positions: Float32Array.from(road), lineOffsets: Uint32Array.of(0, 2)},
      {predicate: 'dwithin', distance: 8.633243737676821e-7, candidateCapacity: 1, pairCapacity: 1}
    );
    expect(result.status).toEqual({count: 0, requiredCount: 0, overflow: 0, candidateOverflow: 0});
    const operation = createCPUSpatialPredicateJoinOperation({
      left: {kind: 'points', positions: {value: 'left', length: 1}},
      right: {
        kind: 'lines',
        positions: {value: 'right', length: 2},
        lineOffsets: {value: 'offsets', length: 2}
      },
      predicate: 'dwithin',
      distance: 8.633243737676821e-7,
      candidateCapacity: 1,
      pairCapacity: 1,
      output: {
        leftRows: 'leftRows',
        rightRows: 'rightRows',
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow'
      }
    });
    const actual = (await operation.execute({
      inputs: {
        leftPositions: vector(Float32Array.of(0, 0), true),
        rightPositions: vector(Float32Array.from(road), true),
        rightLineOffsets: vector(Uint32Array.of(0, 2))
      }
    })) as Record<string, Uint32Array>;
    expect(Array.from(actual['count'])).toEqual([0]);
  });
for (const field of ['origin', 'areaOfUse', 'unit'] as const)
  it(`R8 spatial frame ${field} getter captured once including undefined`, () => {
    let reads = 0;
    const frame = Object.assign(
      Object.create({
        get [field]() {
          return ++reads === 1
            ? undefined
            : field === 'unit'
              ? {name: 'meters', metersPerUnit: 1, degreesPerUnit: 0}
              : [7, 9];
        }
      }),
      {id: 'test', kind: 'crs'}
    );
    const result = snapshotConstructionSpatialContext({
      coordinateSpace: 'planar',
      metric: 'native',
      frame
    });
    expect(reads).toBe(1);
    expect(result.frame![field]).toBeUndefined();
    reads = 0;
    const device = createNullWebGPUDevice(),
      graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x2', 1);
    const integers = (name: string) => createTransientView(graph, name, 'uint32', 1);
    try {
      const join = new GPUNearestFeatureJoin({
        queries: {kind: 'points', positions},
        features: {kind: 'points', positions},
        spatialContext: {coordinateSpace: 'planar', metric: 'native', frame},
        output: {
          neighborIds: integers('ids'),
          neighborCounts: integers('counts'),
          status: {overflow: integers('overflow')}
        }
      });
      expect(reads).toBe(1);
      expect(join.props.spatialContext!.frame![field]).toBeUndefined();
    } finally {
      device.destroy();
    }
  });

for (const grouped of [false, true])
  it(`R5 exact CPU operation writes uncertainty zero grouped=${grouped}`, async () => {
    const entry = createJoinReviewCases().find(
      entry => entry.name === `R5 operation uncertainty grouped=${grouped}`
    )!;
    const output = {
      leftRows: 'leftRows',
      rightRows: 'rightRows',
      count: 'count',
      requiredCount: 'requiredCount',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow',
      uncertainCount: 'uncertainCount'
    };
    const operation = createCPUSpatialPredicateJoinOperation({
      left: {
        kind: 'lines',
        positions: {value: 'leftPositions', length: 2},
        lineOffsets: {value: 'leftLineOffsets', length: 2},
        featureOffsets: {value: 'leftFeatureOffsets', length: 2}
      },
      right: {
        kind: 'lines',
        positions: {value: 'rightPositions', length: 2},
        lineOffsets: {value: 'rightLineOffsets', length: 2},
        ...(grouped ? {featureOffsets: {value: 'rightFeatureOffsets', length: 2}} : {})
      },
      predicate: 'intersects',
      candidateCapacity: 8,
      pairCapacity: 1,
      output
    });
    const inputs = Object.fromEntries(
      Object.entries(entry.inputs).map(([name, input]) => [
        name,
        vector(input.values, input.format === 'float32x2')
      ])
    );
    const result = (await operation.execute({inputs})) as Record<string, Uint32Array>;
    expect(Array.from(result['count'])).toEqual([0]);
    expect(Array.from(result['uncertainCount'])).toEqual([0]);
  });
