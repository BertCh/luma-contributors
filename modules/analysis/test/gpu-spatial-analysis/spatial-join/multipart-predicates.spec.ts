// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import type {Buffer} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {GPUSpatialJoinPrepared} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-prepared';
import {GPUSpatialJoinCandidates} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-join-candidates';
import {GPUSpatialPredicateJoin} from '../../../src/gpu-spatial-analysis/spatial-join/gpu-spatial-predicate-join';
import type {GPUSpatialJoinGeometry} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-join-types';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createSpatialPredicateJoinOperation,
  createCPUSpatialPredicateJoinOperation,
  type SpatialPredicateJoinOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-join/feature-join-operation';
import {formatGPUSpatialRelate} from '../../../src/gpu-spatial-analysis/spatial-join/spatial-relate-types';
import {runOperationGraph} from '../../utils/mixed-execution-test-utils';
import {MULTIPART_PREDICATES, MULTIPART_PREDICATE_FIXTURES} from './multipart-predicates-fixtures';
import {
  createMultipartOperationInputs,
  MULTIPART_ROW_FIXTURE,
  MULTIPART_BUDGET_CASES
} from './multipart-predicates-test-utils';

it('prepared multipart candidates count features and preserve union coverage in relate refinement', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const input = (name: string, values: Float32Array | Uint32Array) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      name,
      buffer,
      values instanceof Float32Array ? 'float32x2' : 'uint32',
      values.length / (values instanceof Float32Array ? 2 : 1)
    );
  };
  const output = (name: string) => {
    const buffer = createOutputBuffer(device, 1);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, 'uint32', 1)};
  };
  const left: GPUSpatialJoinGeometry = {
    kind: 'lines',
    positions: input('left', Float32Array.of(0, 0, 2, 0)) as never,
    lineOffsets: input('left-offsets', Uint32Array.of(0, 2)) as never
  };
  const right: GPUSpatialJoinGeometry = {
    kind: 'lines',
    positions: input('right', Float32Array.of(0, 0, 1, 0, 1, 0, 2, 0)) as never,
    lineOffsets: input('right-lines', Uint32Array.of(0, 2, 4)) as never,
    featureOffsets: input('right-features', Uint32Array.of(0, 2)) as never
  };
  const prepared = new GPUSpatialJoinPrepared({geometry: right, leafCapacity: 1});
  graph.add(prepared);
  const outputs = [
    'candidate-left',
    'candidate-right',
    'candidate-count',
    'candidate-required',
    'candidate-overflow',
    'candidate-loss',
    'left',
    'right',
    'count',
    'required',
    'overflow',
    'loss',
    'relate',
    'truncated'
  ].map(name => [name, output(`output-${name}`)] as const);
  const views = Object.fromEntries(outputs);
  graph.add(
    new GPUSpatialJoinCandidates({
      left,
      right,
      prepared,
      output: {
        leftRows: views['candidate-left'].view,
        rightRows: views['candidate-right'].view,
        status: {
          count: views['candidate-count'].view,
          requiredCount: views['candidate-required'].view,
          overflow: views['candidate-overflow'].view,
          candidateOverflow: views['candidate-loss'].view
        }
      }
    })
  );
  graph.add(
    new GPUSpatialPredicateJoin({
      left,
      right,
      prepared,
      predicate: 'equals',
      engine: 'relate',
      candidateCapacity: 1,
      output: {
        pairs: {
          leftRows: views['left'].view,
          rightRows: views['right'].view,
          status: {
            count: views['count'].view,
            requiredCount: views['required'].view,
            overflow: views['overflow'].view,
            candidateOverflow: views['loss'].view
          }
        },
        relate: views['relate'].view,
        candidateTruncated: views['truncated'].view
      }
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    for (const name of ['candidate-count', 'candidate-required', 'count', 'required'])
      expect(await readUint32(views[name].buffer, 1), name).toEqual([1]);
    for (const name of [
      'candidate-left',
      'candidate-right',
      'candidate-overflow',
      'candidate-loss',
      'left',
      'right',
      'overflow',
      'loss',
      'truncated'
    ])
      expect(await readUint32(views[name].buffer, 1), name).toEqual([0]);
    // Shapely line / split multiline: identical interiors and mod-2 boundary endpoints.
    expect(formatGPUSpatialRelate((await readUint32(views['relate'].buffer, 1))[0])).toBe(
      '1FFF0FFF2'
    );
  } finally {
    compiled.destroy();
    prepared.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

for (const fixture of MULTIPART_PREDICATE_FIXTURES) {
  it(`feature predicates and matrices match Shapely through both engines: ${fixture.name}`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const {left, right, inputs} = createMultipartOperationInputs(fixture.left, fixture.right);
    const cases: {props: SpatialPredicateJoinOperationProps; expected: boolean; matrix: boolean}[] =
      [];
    const append = (
      name: string,
      options: Pick<
        SpatialPredicateJoinOperationProps,
        'predicate' | 'engine' | 'how' | 'pattern' | 'distance'
      >,
      expected: boolean,
      matrix: boolean
    ) => {
      const output = {
        count: `${name}-count`,
        requiredCount: `${name}-requiredCount`,
        overflow: `${name}-overflow`,
        candidateOverflow: `${name}-candidateOverflow`,
        candidateTruncated: `${name}-candidateTruncated`,
        ...(options.how === 'anti'
          ? {unmatchedRows: `${name}-unmatchedRows`}
          : {
              leftRows: `${name}-leftRows`,
              rightRows: `${name}-rightRows`,
              ...(matrix ? {relate: `${name}-relate`} : {})
            })
      };
      cases.push({
        props: {id: name, left, right, ...options, candidateCapacity: 1, pairCapacity: 2, output},
        expected,
        matrix
      });
    };
    for (const [index, predicate] of MULTIPART_PREDICATES.entries()) {
      const fast = ['intersects', 'contains', 'within'].includes(predicate);
      append(
        `${predicate}-default`,
        {predicate, engine: fast ? 'fast' : 'auto'},
        fixture.predicates[index],
        false
      );
      append(`${predicate}-relate`, {predicate, engine: 'relate'}, fixture.predicates[index], true);
    }
    for (const engine of ['auto', 'relate'] as const) {
      append(
        `pattern-${engine}`,
        {predicate: 'relate', pattern: 'T********', engine},
        fixture.pattern,
        true
      );
      append(
        `distance-${engine}`,
        {predicate: 'dwithin', distance: 0.5, engine},
        fixture.dwithin,
        false
      );
      append(
        `anti-${engine}`,
        {predicate: 'intersects', engine, how: 'anti'},
        !fixture.predicates[0],
        false
      );
    }
    const result = await runOperationGraph(device, {
      inputs,
      operations: cases.map(testCase => createSpatialPredicateJoinOperation(testCase.props)),
      outputs: cases.flatMap(testCase => Object.values(testCase.props.output))
    });
    for (const {props, expected, matrix} of cases) {
      const {output} = props;
      // Exact integer and DE-9IM expectations need no numeric tolerance. Distance is only used
      // as a threshold; all fixtures are separated from 0.5 by much more than float32 error.
      expect(Array.from(result[output.count]), props.id).toEqual([Number(expected)]);
      expect(Array.from(result[output.requiredCount]), props.id).toEqual([Number(expected)]);
      for (const value of [output.overflow, output.candidateOverflow, output.candidateTruncated!])
        expect(Array.from(result[value]), props.id).toEqual([0]);
      if (expected) {
        expect(result[output.leftRows ?? output.unmatchedRows!][0], props.id).toBe(0);
        if (output.rightRows) expect(result[output.rightRows][0], props.id).toBe(0);
        if (matrix)
          expect(formatGPUSpatialRelate(result[output.relate!][0]), props.id).toBe(fixture.matrix);
      }
    }
  }, 120_000);
}

it('both engines keep feature rows and separate candidate loss, final overflow and anti unknowns', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const {left, right, inputs} = createMultipartOperationInputs(
    MULTIPART_ROW_FIXTURE.left,
    MULTIPART_ROW_FIXTURE.right
  );
  for (const engine of ['fast', 'relate'] as const)
    for (const expected of MULTIPART_BUDGET_CASES) {
      const output = {
        ...(expected.how === 'anti'
          ? {unmatchedRows: 'unmatchedRows'}
          : {leftRows: 'leftRows', rightRows: 'rightRows'}),
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow',
        candidateTruncated: 'candidateTruncated'
      };
      const operation = createSpatialPredicateJoinOperation({
        left,
        right,
        predicate: 'within',
        engine,
        how: expected.how,
        candidateCapacity: expected.candidateCapacity,
        pairCapacity: expected.pairCapacity,
        output
      });
      const result = await runOperationGraph(device, {
        inputs,
        operations: [operation],
        outputs: Object.values(output)
      });
      for (const name of ['count', 'requiredCount', 'overflow', 'candidateOverflow'] as const)
        expect(Array.from(result[name]), `${engine} ${name}`).toEqual([expected[name]]);
      expect(
        Array.from(result[expected.how === 'anti' ? 'unmatchedRows' : 'leftRows']).slice(
          0,
          expected.count
        )
      ).toEqual(expected.leftRows);
      if (expected.how === 'inner')
        expect(Array.from(result['rightRows']).slice(0, expected.count)).toEqual(
          expected.rightRows
        );
      expect(Array.from(result['candidateTruncated'])).toEqual(expected.candidateTruncated);
    }
});

it('GPU and CPU operation forms independently match polygon containment and line union literals', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  for (const name of ['union line coverage', 'polygon contained']) {
    const fixture = MULTIPART_PREDICATE_FIXTURES.find(fixture => fixture.name === name)!;
    const {left, right, inputs} = createMultipartOperationInputs(fixture.left, fixture.right);
    for (const factory of [
      createSpatialPredicateJoinOperation,
      createCPUSpatialPredicateJoinOperation
    ]) {
      const output = {
        leftRows: 'leftRows',
        rightRows: 'rightRows',
        relate: 'relate',
        count: 'count',
        requiredCount: 'requiredCount',
        overflow: 'overflow',
        candidateOverflow: 'candidateOverflow',
        candidateTruncated: 'candidateTruncated'
      };
      const operation = factory({
        left,
        right,
        predicate: 'within',
        engine: 'relate',
        candidateCapacity: 1,
        pairCapacity: 1,
        output
      });
      const result = await runOperationGraph(device, {
        inputs,
        operations: [operation],
        outputs: Object.values(output)
      });
      for (const key of [
        'leftRows',
        'rightRows',
        'overflow',
        'candidateOverflow',
        'candidateTruncated'
      ])
        expect(Array.from(result[key]), key).toEqual([0]);
      for (const key of ['count', 'requiredCount'])
        expect(Array.from(result[key]), key).toEqual([1]);
      expect(formatGPUSpatialRelate(result['relate'][0])).toBe(fixture.matrix);
    }
  }
});
