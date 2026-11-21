// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSpatialPredicateJoinOnCPU,
  computeSpatialRelateOnCPU,
  type CPUSpatialJoinGeometry
} from '../../../src/gpu-spatial-analysis/spatial-join/cpu-feature-join';
import {formatGPUSpatialRelate} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-relate-types';
import {MULTIPART_PREDICATES, MULTIPART_PREDICATE_FIXTURES} from './multipart-predicates-fixtures';
import {
  createMultipartCPUGeometry,
  createMultipartOperationInputs,
  MULTIPART_ROW_FIXTURE,
  MULTIPART_BUDGET_CASES
} from './multipart-predicates-test-utils';
import {
  createCPUSpatialPredicateJoinOperation,
  createSpatialPredicateJoinOperation
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

for (const fixture of MULTIPART_PREDICATE_FIXTURES) {
  it(`Shapely feature matrix and every predicate: ${fixture.name}`, () => {
    const left = createMultipartCPUGeometry(fixture.left),
      right = createMultipartCPUGeometry(fixture.right);
    expect(formatGPUSpatialRelate(computeSpatialRelateOnCPU(left, right))).toBe(fixture.matrix);
    for (const [index, predicate] of MULTIPART_PREDICATES.entries()) {
      const result = computeSpatialPredicateJoinOnCPU(left, right, {
        predicate,
        relate: true,
        candidateCapacity: 1,
        pairCapacity: 2
      });
      expect(result.status, predicate).toEqual({
        count: Number(fixture.predicates[index]),
        requiredCount: Number(fixture.predicates[index]),
        overflow: 0,
        candidateOverflow: 0
      });
      expect(Array.from(result.candidateTruncated), predicate).toEqual([0]);
      expect(result.leftRows[0], predicate).toBe(fixture.predicates[index] ? 0 : 0xffffffff);
      expect(result.rightRows[0], predicate).toBe(fixture.predicates[index] ? 0 : 0xffffffff);
      if (fixture.predicates[index])
        expect(formatGPUSpatialRelate(result.relate[0]), predicate).toBe(fixture.matrix);
      expect(result.leftRows[1]).toBe(0xffffffff);
    }
    for (const [predicate, expected] of [
      ['relate', fixture.pattern],
      ['dwithin', fixture.dwithin]
    ] as const) {
      const result = computeSpatialPredicateJoinOnCPU(left, right, {
        predicate,
        ...(predicate === 'relate' ? {pattern: 'T********'} : {distance: 0.5}),
        candidateCapacity: 1,
        pairCapacity: 1
      });
      expect(result.status.count, predicate).toBe(Number(expected));
    }
    const anti = computeSpatialPredicateJoinOnCPU(left, right, {
      predicate: 'intersects',
      how: 'anti',
      candidateCapacity: 1,
      pairCapacity: 1
    });
    expect(anti.status).toEqual({
      count: Number(!fixture.predicates[0]),
      requiredCount: Number(!fixture.predicates[0]),
      overflow: 0,
      candidateOverflow: 0
    });
    expect(anti.leftRows[0]).toBe(fixture.predicates[0] ? 0xffffffff : 0);
  });
}

it('feature budgets keep row identities and suppress incomplete anti results', () => {
  for (const expected of MULTIPART_BUDGET_CASES) {
    const result = computeSpatialPredicateJoinOnCPU(
      createMultipartCPUGeometry(MULTIPART_ROW_FIXTURE.left),
      createMultipartCPUGeometry(MULTIPART_ROW_FIXTURE.right),
      {predicate: 'within', ...expected}
    );
    expect(result.status).toEqual({
      count: expected.count,
      requiredCount: expected.requiredCount,
      overflow: expected.overflow,
      candidateOverflow: expected.candidateOverflow
    });
    expect(Array.from(result.leftRows).slice(0, expected.count)).toEqual(expected.leftRows);
    if (expected.how === 'inner')
      expect(Array.from(result.rightRows).slice(0, expected.count)).toEqual(expected.rightRows);
    expect(Array.from(result.candidateTruncated)).toEqual(expected.candidateTruncated);
  }
});

it('operation adapters accept multipart relate, distance and anti options and snapshot patterns', async () => {
  const fixture = MULTIPART_PREDICATE_FIXTURES.find(
    fixture => fixture.name === 'union line coverage'
  )!;
  const {left, right, inputs} = createMultipartOperationInputs(fixture.left, fixture.right);
  const device = createNullWebGPUDevice();
  try {
    for (const how of ['inner', 'anti'] as const) {
      const pattern = ['T*F**FFF*'];
      const output = {
        ...(how === 'anti'
          ? {unmatchedRows: 'unmatchedRows'}
          : {leftRows: 'leftRows', rightRows: 'rightRows', relate: 'relate'}),
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow',
        candidateTruncated: 'candidateTruncated'
      };
      const props = {
        left,
        right,
        predicate: 'relate' as const,
        pattern,
        engine: 'relate' as const,
        how,
        candidateCapacity: 1,
        pairCapacity: 1,
        output
      };
      const cpu = createCPUSpatialPredicateJoinOperation(props),
        gpu = createSpatialPredicateJoinOperation(props);
      pattern[0] = 'F********';
      const vectors = Object.fromEntries(
        Object.entries(cpu.inputs).map(([port, value]) => {
          const {format, values} = inputs[value];
          const length = values.length / (format === 'float32x2' ? 2 : 1);
          return [port, {format, length, chunks: [{values, length}], pack: () => values}];
        })
      );
      const result = (await cpu.execute({inputs: vectors})) as Record<string, Uint32Array>;
      expect(Array.from(result['count'])).toEqual([how === 'anti' ? 0 : 1]);
      expect(Array.from(result['requiredCount'])).toEqual([how === 'anti' ? 0 : 1]);
      for (const name of ['overflow', 'candidateOverflow', 'candidateTruncated'])
        expect(Array.from(result[name])).toEqual([0]);
      const graph = new GPUCommandGraph(device);
      const views = Object.fromEntries([
        ...Object.entries(gpu.inputs).map(([port, value]) => [
          port,
          graph.createDataView(
            graph.createTransientBuffer({
              id: value,
              byteLength: inputs[value].values.byteLength,
              usage: 128
            }),
            {
              format: inputs[value].format,
              length: inputs[value].values.length / (inputs[value].format === 'float32x2' ? 2 : 1)
            }
          )
        ]),
        ...Object.entries(gpu.outputs).map(([port, definition]) => [
          port,
          graph.createDataView(
            graph.createTransientBuffer({
              id: definition.value,
              byteLength: definition.length * 4,
              usage: 128
            }),
            {format: definition.format, length: definition.length}
          )
        ])
      ]);
      const contributor = gpu.createContributor(port => views[port] as never);
      expect(contributor.getCommandNodes(graph).length).toBeGreaterThan(0);
      gpu.destroy();
    }
  } finally {
    device.destroy();
  }
});

it('implements polygon containment in the CPU feature join', () => {
  const polygon = (positions: number[]): CPUSpatialJoinGeometry => ({
    kind: 'polygons',
    positions: Float32Array.from(positions),
    featureOffsets: Uint32Array.of(0, 1),
    polygonOffsets: Uint32Array.of(0, 1),
    ringOffsets: Uint32Array.of(0, 4)
  });
  // By hand: the unit square lies strictly inside the 4-by-4 square.
  const result = computeSpatialPredicateJoinOnCPU(
    polygon([1, 1, 2, 1, 2, 2, 1, 2]),
    polygon([0, 0, 4, 0, 4, 4, 0, 4]),
    {predicate: 'within', candidateCapacity: 1, pairCapacity: 1}
  );
  expect(result.status).toEqual({count: 1, requiredCount: 1, overflow: 0, candidateOverflow: 0});
  expect(Array.from(result.leftRows)).toEqual([0]);
  expect(Array.from(result.rightRows)).toEqual([0]);
});
