// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, expectTypeOf, it} from 'vitest';
import {
  GPUSpatialPredicateJoin,
  type GPUSpatialPredicateJoinOutput
} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {computeSpatialPredicateJoinOnCPU} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';
import {
  createCPUSpatialPredicateJoinOperation,
  createSpatialPredicateJoinOperation
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import {
  createPredicateOracleGeometries,
  getPredicateOracleOperationProps,
  getPredicateOracleOperationInputs
} from './predicate-output-oracles-fixture';

const UNSUPPORTED_STATUS_FIELDS = [
  'invalidCount',
  'iterationCount',
  'converged',
  'iterationLimitReached',
  'approximation',
  'candidateCount',
  'candidateTruncatedCount',
  'antimeridianUncertainCount',
  'poleUncertainCount'
] as const;

it('GEO-103 anti construction requires a compact candidate-loss destination when loss is possible', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
    const unmatched = {
      ids: createTransientView(graph, 'unmatched', 'uint32', 2),
      status: {count: scalar('count'), overflow: scalar('overflow')}
    };
    const props = {
      id: 'anti-bounds',
      left: {kind: 'points', positions: createTransientView(graph, 'left', 'float32x2', 2)},
      right: {kind: 'points', positions: createTransientView(graph, 'right', 'float32x2', 2)},
      predicate: 'intersects',
      how: 'anti',
      candidateCapacity: 4,
      output: {unmatched}
    } as const;
    const missing = /^anti-bounds.*Add output\.unmatched\.status\.candidateOverflow/;
    expect(() => new GPUSpatialPredicateJoin({...props, candidateCapacity: 3})).toThrow(missing);
    expect(() => new GPUSpatialPredicateJoin({...props, leafCapacity: 1})).toThrow(missing);
    expect(() => new GPUSpatialPredicateJoin({...props, leftFeatureCount: scalar('live')})).toThrow(
      missing
    );
    expect(() => new GPUSpatialPredicateJoin(props)).not.toThrow();
  } finally {
    device.destroy();
  }
});

it('GEO-103 both named anti operations require a graph value for candidate loss', () => {
  const props = {
    ...getPredicateOracleOperationProps(),
    id: 'anti-operation',
    how: 'anti' as const,
    candidateCapacity: 1,
    output: {
      unmatchedRows: 'unmatchedRows',
      count: 'count',
      requiredCount: 'requiredCount',
      overflow: 'overflow',
      candidateOverflow: ''
    }
  };
  for (const create of [
    createSpatialPredicateJoinOperation,
    createCPUSpatialPredicateJoinOperation
  ])
    expect(() => create(props)).toThrow(/^anti-operation.*output\.candidateOverflow.*graph value/);
});

for (const field of UNSUPPORTED_STATUS_FIELDS) {
  for (const declaration of ['own', 'non-enumerable', 'inherited'] as const) {
    for (const engine of ['auto', 'fast', 'relate'] as const) {
      it(`predicate ${engine} refuses unwritten ${field} from ${declaration} weights status`, () => {
        const device = createNullWebGPUDevice();
        try {
          const graph = new GPUCommandGraph(device);
          const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
          const status = {overflow: scalar('weightsOverflow')};
          const descriptor = {value: scalar('unsupported'), enumerable: declaration === 'own'};
          if (declaration === 'inherited')
            Object.setPrototypeOf(status, Object.create(null, {[field]: descriptor}));
          else Object.defineProperty(status, field, descriptor);
          // Independent contract expectation: topology/capacity/uncertainty diagnostics exist;
          // this finite noniterative join has no invalid-row or convergence output destinations.
          expect(
            () =>
              new GPUSpatialPredicateJoin({
                id: 'predicate-status-oracle',
                engine,
                predicate: 'intersects',
                candidateCapacity: 1,
                left: {
                  kind: 'points',
                  positions: createTransientView(graph, 'left', 'float32x2', 1)
                },
                right: {
                  kind: 'points',
                  positions: createTransientView(graph, 'right', 'float32x2', 1)
                },
                output: {
                  weights: {
                    offsets: createTransientView(graph, 'offsets', 'uint32', 2),
                    neighbors: scalar('neighbors'),
                    weights: createTransientView(graph, 'weights', 'float32', 1),
                    status
                  },
                  status: {overflow: scalar('overallOverflow')}
                }
              })
          ).toThrow(
            `predicate-status-oracle output.weights.status.${field} is unsupported because this join publishes only count, requiredCount, overflow, candidateOverflow and uncertainCount; omit ${field} and use those result diagnostics`
          );
        } finally {
          device.destroy();
        }
      });
    }
  }
}

it('predicate weights expose only diagnostics with actual writers', () => {
  type WeightsStatus = NonNullable<NonNullable<GPUSpatialPredicateJoinOutput['weights']>['status']>;
  expectTypeOf<keyof WeightsStatus>().toEqualTypeOf<
    'count' | 'requiredCount' | 'overflow' | 'candidateOverflow' | 'uncertainCount'
  >();
  expectTypeOf<
    keyof NonNullable<GPUSpatialPredicateJoin['outputWeights']>['status']
  >().toEqualTypeOf<
    'count' | 'requiredCount' | 'overflow' | 'candidateOverflow' | 'uncertainCount'
  >();
});

it('predicate weights omit undeclared geometry metadata from their captured status', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const scalar = (name: string) => createTransientView(graph, name, 'uint32', 1);
    const status = {overflow: scalar('overflow'), vertexCount: scalar('vertexCount')};
    const contributor = new GPUSpatialPredicateJoin({
      left: {kind: 'points', positions: createTransientView(graph, 'left', 'float32x2', 1)},
      right: {kind: 'points', positions: createTransientView(graph, 'right', 'float32x2', 1)},
      predicate: 'intersects',
      candidateCapacity: 1,
      output: {
        weights: {
          offsets: createTransientView(graph, 'offsets', 'uint32', 2),
          neighbors: scalar('neighbors'),
          weights: createTransientView(graph, 'weights', 'float32', 1),
          status
        },
        status: {overflow: scalar('overallOverflow')}
      }
    });
    expect(contributor.outputWeights?.status).not.toHaveProperty('vertexCount');
    expect(contributor.outputWeights?.status.overflow).toBe(status.overflow);
  } finally {
    device.destroy();
  }
});

it('predicate CPU function and operation retain Shapely pair and completeness literals', async () => {
  const {left, right} = createPredicateOracleGeometries();
  const props = getPredicateOracleOperationProps();
  const direct = computeSpatialPredicateJoinOnCPU(left, right, {
    predicate: props.predicate,
    pattern: props.pattern,
    distance: props.distance,
    engine: props.engine,
    how: props.how,
    candidateCapacity: props.candidateCapacity,
    pairCapacity: props.pairCapacity
  });
  const operation = createCPUSpatialPredicateJoinOperation(props);
  const inputs = Object.fromEntries(
    Object.entries(getPredicateOracleOperationInputs()).map(([name, input]) => [
      name,
      {
        format: input.format,
        length: input.length,
        chunks: [{values: input.values, length: input.length}],
        pack: () => input.values
      }
    ])
  );
  const result = await operation.execute({inputs});
  // Shapely enumerates (0,0),(0,1),(1,0),(1,1); candidate and pair capacity both
  // equal four. No candidate truncation, and the unbounded requirement is exactly four.
  expect(Array.from(direct.leftRows)).toEqual([0, 0, 1, 1]);
  expect(Array.from(direct.rightRows)).toEqual([0, 1, 0, 1]);
  expect(direct.status).toEqual({count: 4, requiredCount: 4, overflow: 0, candidateOverflow: 0});
  expect(Array.from(result['leftRows'] as Uint32Array)).toEqual([0, 0, 1, 1]);
  expect(Array.from(result['rightRows'] as Uint32Array)).toEqual([0, 1, 0, 1]);
  expect(Array.from(result['count'] as Uint32Array)).toEqual([4]);
  expect(Array.from(result['requiredCount'] as Uint32Array)).toEqual([4]);
  expect(Array.from(result['overflow'] as Uint32Array)).toEqual([0]);
  expect(Array.from(result['candidateOverflow'] as Uint32Array)).toEqual([0]);
  expect(Array.from(result['candidateTruncated'] as Uint32Array)).toEqual([0, 0, 0, 0]);
});

for (const candidateCapacity of [1, 4]) {
  it(`bounded anti CPU function and operation suppress truncated rows at candidate capacity ${candidateCapacity}`, async () => {
    const {left, right} = createPredicateOracleGeometries();
    const props = {
      ...getPredicateOracleOperationProps(),
      how: 'anti' as const,
      pairCapacity: 1,
      candidateCapacity,
      output: {
        unmatchedRows: 'unmatchedRows',
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow'
      }
    };
    const direct = computeSpatialPredicateJoinOnCPU(left, right, {
      predicate: props.predicate,
      pattern: props.pattern,
      distance: props.distance,
      engine: props.engine,
      how: props.how,
      candidateCapacity: props.candidateCapacity,
      pairCapacity: props.pairCapacity
    });
    const inputs = Object.fromEntries(
      Object.entries(getPredicateOracleOperationInputs()).map(([name, input]) => [
        name,
        {
          format: input.format,
          length: input.length,
          chunks: [{values: input.values, length: input.length}],
          pack: () => input.values
        }
      ])
    );
    const result = await createCPUSpatialPredicateJoinOperation(props).execute({inputs});
    // Shapely unmatched rows are [2,3]. Rows0,1 are never certain unmatched,
    // including when row1 loses all candidates. One slot retains row2 only.
    expect(Array.from(direct.leftRows)).toEqual([2]);
    expect(Array.from(direct.unmatchedRows)).toEqual([2]);
    expect(direct.unmatchedStatus).toEqual(direct.status);
    expect(direct.status).toEqual({
      count: 1,
      requiredCount: 2,
      overflow: 1,
      candidateOverflow: candidateCapacity === 1 ? 1 : 0
    });
    expect(Array.from(result['unmatchedRows'] as Uint32Array)).toEqual([2]);
    expect(Array.from(result['count'] as Uint32Array)).toEqual([1]);
    expect(Array.from(result['requiredCount'] as Uint32Array)).toEqual([2]);
    expect(Array.from(result['overflow'] as Uint32Array)).toEqual([1]);
    expect(Array.from(result['candidateOverflow'] as Uint32Array)).toEqual([
      candidateCapacity === 1 ? 1 : 0
    ]);
  });
}
