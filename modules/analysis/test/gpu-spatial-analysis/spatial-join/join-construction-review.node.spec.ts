// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph, createTransientView, type GPUCPUVector} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createSpatialPredicateJoinOperation,
  createCPUSpatialPredicateJoinOperation,
  createNearestFeatureJoinOperation,
  createCPUNearestFeatureJoinOperation
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import {GPUNearestFeatureJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-nearest-feature-join';
import {formatGPUSpatialRelate} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-relate-types';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

function createVector(values: Float32Array | Uint32Array): GPUCPUVector {
  const format = values instanceof Float32Array ? 'float32x2' : 'uint32';
  const length = values.length / (format === 'float32x2' ? 2 : 1);
  return {format, length, chunks: [{values, length}], pack: () => values};
}

function createInheritedOutput<Output extends Record<string, string>>(destinations: Output) {
  const reads: Record<string, number> = {};
  const prototype = Object.fromEntries(
    Object.keys(destinations).map(field => [
      field,
      {
        get() {
          reads[field] = (reads[field] ?? 0) + 1;
          return destinations[field];
        }
      }
    ])
  );
  return {output: Object.create(Object.defineProperties({}, prototype)) as Output, reads};
}

for (const cpu of [false, true]) {
  it(`predicate ${cpu ? 'CPU' : 'GPU'} adapter captures inherited output destinations once`, async () => {
    const destinations = {
      leftRows: 'leftRows',
      rightRows: 'rightRows',
      relate: 'relate',
      count: 'count',
      requiredCount: 'requiredCount',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow',
      candidateTruncated: 'candidateTruncated'
    };
    const {output, reads} = createInheritedOutput(destinations);
    const create = cpu
      ? createCPUSpatialPredicateJoinOperation
      : createSpatialPredicateJoinOperation;
    const operation = create({
      left: {kind: 'points', positions: {value: 'left', length: 1}},
      right: {kind: 'points', positions: {value: 'right', length: 1}},
      predicate: 'intersects',
      candidateCapacity: 1,
      pairCapacity: 1,
      output
    });
    expect(Object.keys(operation.contract.outputs ?? {}).sort()).toEqual(
      Object.keys(destinations).sort()
    );
    expect(reads).toEqual(Object.fromEntries(Object.keys(destinations).map(field => [field, 1])));
    destinations.count = 'replacement';
    if (cpu) {
      const actual = (await (
        operation as ReturnType<typeof createCPUSpatialPredicateJoinOperation>
      ).execute({
        inputs: {
          leftPositions: createVector(Float32Array.of(1, 2)),
          rightPositions: createVector(Float32Array.of(1, 2))
        }
      })) as Record<string, Uint32Array>;
      // Two equal singleton points have one (0,0) pair, no loss, and DE-9IM 0FFFFFFF2:
      // their interiors intersect in dimension 0; only exterior/exterior has dimension 2.
      for (const name of [
        'leftRows',
        'rightRows',
        'overflow',
        'candidateOverflow',
        'candidateTruncated'
      ])
        expect(Array.from(actual[name]), name).toEqual([0]);
      for (const name of ['count', 'requiredCount'])
        expect(Array.from(actual[name]), name).toEqual([1]);
      expect(formatGPUSpatialRelate(actual['relate'][0])).toBe('0FFFFFFF2');
      expect(actual).not.toHaveProperty('replacement');
    }
  });

  it(`nearest ${cpu ? 'CPU' : 'GPU'} adapter captures inherited output destinations once`, async () => {
    const destinations = {
      neighborRows: 'rows',
      neighborIds: 'ids',
      neighborCounts: 'counts',
      neighborDistances: 'distances',
      overflow: 'overflow',
      candidateOverflow: 'candidateOverflow'
    };
    const {output, reads} = createInheritedOutput(destinations);
    const create = cpu ? createCPUNearestFeatureJoinOperation : createNearestFeatureJoinOperation;
    const operation = create({
      queries: {kind: 'points', positions: {value: 'queries', length: 1}},
      features: {
        kind: 'points',
        positions: {value: 'features', length: 2},
        sourceIds: {value: 'sourceIds', length: 2}
      },
      output
    });
    expect(Object.keys(operation.contract.outputs ?? {}).sort()).toEqual(
      Object.values(destinations).sort()
    );
    expect(reads).toEqual(Object.fromEntries(Object.keys(destinations).map(field => [field, 1])));
    destinations.neighborRows = 'replacement';
    if (cpu) {
      const actual = (await (
        operation as ReturnType<typeof createCPUNearestFeatureJoinOperation>
      ).execute({
        inputs: {
          queriesPositions: createVector(Float32Array.of(0, 0)),
          featuresPositions: createVector(Float32Array.of(6, 8, 3, 4)),
          featuresSourceIds: createVector(Uint32Array.of(77, 42))
        }
      })) as Record<string, Uint32Array | Float32Array>;
      // Pythagoras: row 0 is distance 10; row 1 is distance 5 and has caller ID 42.
      expect(
        Object.fromEntries(
          Object.entries(actual).map(([name, values]) => [name, Array.from(values)])
        )
      ).toEqual({
        rows: [1],
        ids: [42],
        counts: [1],
        distances: [5],
        overflow: [0],
        candidateOverflow: [0]
      });
    }
  });
}

it('nearest captures the matches status getter once before validating its destinations', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const positions = createTransientView(graph, 'positions', 'float32x2', 1);
  const integers = (name: string) => createTransientView(graph, name, 'uint32', 1);
  const firstStatus = {count: integers('first-count'), overflow: integers('first-overflow')};
  const secondStatus = {count: integers('second-count'), overflow: integers('second-overflow')};
  let statusReads = 0;
  const matches = {
    ids: integers('matches'),
    get status() {
      return ++statusReads === 1 ? firstStatus : secondStatus;
    }
  };
  try {
    const contributor = new GPUNearestFeatureJoin({
      queries: {kind: 'points', positions},
      features: {kind: 'points', positions},
      radius: createTransientView(graph, 'radius', 'float32', 1),
      candidateCapacity: 1,
      output: {
        nearestFeatureIds: integers('nearest'),
        matches,
        status: {overflow: integers('overflow')}
      }
    });
    expect(statusReads).toBe(1);
    expect(contributor.props.output.matches!.status.count).toBe(firstStatus.count);
    expect(contributor.props.output.matches!.status.overflow).toBe(firstStatus.overflow);
  } finally {
    device.destroy();
  }
});

for (const create of [createSpatialPredicateJoinOperation, createCPUSpatialPredicateJoinOperation])
  it(`${create.name} captures the relate pattern getter once`, async () => {
    let patternReads = 0;
    const operation = create({
      left: {kind: 'points', positions: {value: 'left', length: 1}},
      right: {kind: 'points', positions: {value: 'right', length: 1}},
      predicate: 'relate',
      get pattern() {
        return ++patternReads === 1 ? ['T********'] : ['F********'];
      },
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
    expect(patternReads).toBe(1);
    if ('execute' in operation) {
      const actual = (await operation.execute({
        inputs: {
          leftPositions: createVector(Float32Array.of(1, 2)),
          rightPositions: createVector(Float32Array.of(1, 2))
        }
      })) as Record<string, Uint32Array>;
      // Equal singleton points have an interior/interior intersection, satisfying T********.
      expect(
        Object.fromEntries(
          Object.entries(actual).map(([name, values]) => [name, Array.from(values)])
        )
      ).toEqual({
        leftRows: [0],
        rightRows: [0],
        count: [1],
        requiredCount: [1],
        overflow: [0],
        candidateOverflow: [0]
      });
    }
  });
