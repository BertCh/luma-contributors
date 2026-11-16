// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeSegmentSnapOnCPU,
  CPUSegmentSnapOperation
} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-segment-snap-operation';
import {
  GPUSegmentSnapOperation,
  type SegmentSnapOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap-operation';

it('matches hand-ordered insertions, duplicate suppression, source edges and per-line overflow', () => {
  const result = computeSegmentSnapOnCPU(
    {
      positions: new Float32Array([0.2, 0, 6.2, 0, 10.1, 0, 20, 0, 20, 0, 21, 0]),
      lineOffsets: new Uint32Array([0, 3, 6]),
      referencePositions: new Float32Array([0, 0, 2, 0.5, 2, -0.5, 6, 0, 8, 0, 10, 0, 2, 0.5])
    },
    {tolerance: 0.6, maximumVerticesPerLine: 5}
  );
  expect(Array.from(result.counts)).toEqual([5, 6, 1, 0, 2, 2, 0, 0]);
  expect(Array.from(result.positions.slice(0, 10))).toEqual([0, 0, 2, 0.5, 2, -0.5, 6, 0, 8, 0]);
  expect(Array.from(result.sourceEdges.slice(0, 5))).toEqual([0, 0, 0, 0, 1]);
  expect(Array.from(result.positions.slice(10, 14))).toEqual([20, 0, 21, 0]);
  expect(Array.from(result.sourceEdges.slice(5, 7))).toEqual([3, 4]);
  expect(Array.from(result.validity)).toEqual([4, 0]);
});

it('uses inclusive tolerance, reference-row ties and excludes endpoint projections', () => {
  const result = computeSegmentSnapOnCPU(
    {
      positions: new Float32Array([0, 1, 4, 0]),
      lineOffsets: new Uint32Array([0, 2]),
      referencePositions: new Float32Array([0, 0, 0, 2, 2, 1, 4, 0])
    },
    {tolerance: 1, maximumVerticesPerLine: 5}
  );
  expect(Array.from(result.positions.slice(0, 6))).toEqual([0, 0, 2, 1, 4, 0]);
  expect(Array.from(result.counts)).toEqual([3, 3, 0, 0]);
});

it('matches clamp, empty, collapsed and nonfinite status conventions', () => {
  const result = computeSegmentSnapOnCPU(
    {
      positions: new Float32Array([1, 2, 1, 2, NaN, 0]),
      lineOffsets: new Uint32Array([0, 2, 2, 3, 99, 1]),
      referencePositions: new Float32Array(0)
    },
    {tolerance: 0, maximumVerticesPerLine: 2}
  );
  expect(Array.from(result.counts)).toEqual([
    1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0
  ]);
  expect(Array.from(result.validity)).toEqual([1, 0, 2, 0, 0]);
  const invalidReference = computeSegmentSnapOnCPU(
    {
      positions: new Float32Array([1, 2]),
      lineOffsets: new Uint32Array([0, 0, 1]),
      referencePositions: new Float32Array([Infinity, 0])
    },
    {tolerance: 0, maximumVerticesPerLine: 1}
  );
  expect(Array.from(invalidReference.counts)).toEqual([0, 0, 0, 1, 0, 0, 0, 1]);
  expect(Array.from(invalidReference.validity)).toEqual([2, 2]);
});

it('rejects malformed arrays and checks both operation contracts at construction', () => {
  const input = {
    positions: new Float32Array(2),
    lineOffsets: new Uint32Array([0, 1]),
    referencePositions: new Float32Array(2)
  };
  const options = {tolerance: 0, maximumVerticesPerLine: 1};
  expect(() =>
    computeSegmentSnapOnCPU({...input, positions: new Float32Array(1)}, options)
  ).toThrow(/positions/);
  expect(() =>
    computeSegmentSnapOnCPU({...input, referencePositions: new Float32Array(1)}, options)
  ).toThrow(/referencePositions/);
  expect(() =>
    computeSegmentSnapOnCPU({...input, lineOffsets: new Uint32Array(1)}, options)
  ).toThrow(/lineOffsets/);
  const props: SegmentSnapOperationProps = {
    id: 'snap-check',
    positions: 'positions',
    lineOffsets: 'offsets',
    referencePositions: 'references',
    rowCount: 1,
    lineCount: 1,
    referenceCount: 1,
    ...options,
    output: {positions: 'snapped', sourceEdges: 'edges', counts: 'counts'}
  };
  for (const Operation of [CPUSegmentSnapOperation, GPUSegmentSnapOperation]) {
    expect(() => new Operation(props)).not.toThrow();
    expect(() => new Operation({...props, maximumVerticesPerLine: 0})).toThrow(/^snap-check/);
    expect(
      () => new Operation({...props, output: {...props.output, positions: 'positions'}})
    ).toThrow(/^snap-check/);
  }
  expect(() => new CPUSegmentSnapOperation({...props, tolerance: -1})).toThrow(
    /^snap-check.*tolerance/
  );
  expect(
    () =>
      new CPUSegmentSnapOperation({...props, searchStructure: 'brute-force', rowCount: 0xffffffff})
  ).toThrow(/^snap-check.*referenceCount/);
});

it.each([
  'canonical',
  'legacy',
  'both'
] as const)('forwards the %s diagnostic through both operation contracts', async spelling => {
  const props: SegmentSnapOperationProps = {
    id: 'status-operation',
    positions: 'positions',
    lineOffsets: 'offsets',
    referencePositions: 'references',
    rowCount: 2,
    referenceCount: 3,
    lineCount: 1,
    tolerance: 0,
    maximumVerticesPerLine: 3,
    candidateCapacity: 1,
    output: {
      positions: 'snapped',
      sourceEdges: 'edges',
      counts: 'counts',
      ...(spelling === 'legacy' ? {} : {status: {candidateOverflow: 'diagnostic'}}),
      ...(spelling === 'canonical' ? {} : {candidateOverflow: 'diagnostic'})
    }
  };
  const gpu = new GPUSegmentSnapOperation(props);
  expect(gpu.outputs['candidateOverflow']).toEqual({
    value: 'diagnostic',
    format: 'uint32',
    length: 1
  });
  expect(gpu.contract.outputs).toHaveProperty('diagnostic');
  expect(gpu.contract.values?.filter(value => value.value === 'diagnostic')).toHaveLength(1);
  const cpu = new CPUSegmentSnapOperation(props);
  expect(cpu.outputs['diagnostic']).toMatchObject({format: 'uint32', length: 1});
  const positions = new Float32Array([-100, 0, -94, 0]);
  const offsets = new Uint32Array([0, 2]);
  const references = new Float32Array([-99, 0, -99, 0, -99, 0]);
  const result = await cpu.execute({
    inputs: {
      positions: {
        format: 'float32x2',
        length: 2,
        chunks: [{values: positions, length: 2}],
        pack: () => positions
      },
      lineOffsets: {
        format: 'uint32',
        length: 2,
        chunks: [{values: offsets, length: 2}],
        pack: () => offsets
      },
      referencePositions: {
        format: 'float32x2',
        length: 3,
        chunks: [{values: references, length: 3}],
        pack: () => references
      }
    }
  });
  expect(result).toEqual({
    snapped: new Float32Array([-100, 0, -99, 0, -94, 0]),
    edges: new Uint32Array(3),
    counts: new Uint32Array([3, 3, 0, 0]),
    diagnostic: new Uint32Array([1])
  });
  for (const Operation of [CPUSegmentSnapOperation, GPUSegmentSnapOperation])
    expect(
      () =>
        new Operation({
          ...props,
          output: {
            ...props.output,
            status: {candidateOverflow: 'diagnostic'},
            candidateOverflow: 'different'
          }
        })
    ).toThrow(/^status-operation output.status.candidateOverflow.*deprecated alias/);
});
