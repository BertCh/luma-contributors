// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeLineLoopRemovalOnCPU,
  CPULineLoopRemovalOperation
} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-line-loop-removal-operation';
import {
  GPULineLoopRemovalOperation,
  type LineLoopRemovalOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-line-loop-removal-operation';

function getProps(): LineLoopRemovalOperationProps {
  return {
    id: 'loop-test',
    lines: {positions: 'inputPositions', lineOffsets: 'inputOffsets'},
    lineCount: 2,
    vertexCapacity: 12,
    intersectionCapacity: 32,
    output: {
      positions: 'positions',
      lineOffsets: 'lineOffsets',
      sourceEdges: 'sourceEdges',
      status: 'status'
    }
  };
}

it('matches hand-noded crossing and overlapping-backtrack traversals on independent lines', () => {
  // Crossing at (2,2) removes the bowtie's upper triangle. On the other line, 14->12
  // erases the suffix up to x=12; the final 12->16 traversal restores x=14 as a node.
  const result = computeLineLoopRemovalOnCPU(
    {
      positions: Float32Array.of(0, 0, 4, 4, 0, 4, 4, 0, 10, 0, 14, 0, 12, 0, 16, 0),
      lineOffsets: Uint32Array.of(0, 4, 8)
    },
    getProps()
  );
  expect(Array.from(result.status)).toEqual([7, 7, 0, 0]);
  expect(Array.from(result.lineOffsets)).toEqual([0, 3, 7]);
  expect(Array.from(result.positions.subarray(0, 14))).toEqual([
    0, 0, 2, 2, 4, 0, 10, 0, 12, 0, 14, 0, 16, 0
  ]);
  expect(Array.from(result.sourceEdges.subarray(0, 7))).toEqual([0, 2, 2, 4, 6, 6, 6]);
  expect(result.positions.subarray(14).every(value => value === 0)).toBe(true);
});

it('uses a closed-form rational crossing, with only one output rounding', () => {
  // y=x meets y=1-2x at (1/3,1/3); the closed middle excursion is erased.
  const result = computeLineLoopRemovalOnCPU(
    {positions: Float32Array.of(0, 0, 1, 1, 0, 1, 1, -1), lineOffsets: Uint32Array.of(0, 4)},
    {vertexCapacity: 3}
  );
  expect(Array.from(result.status)).toEqual([3, 3, 0, 0]);
  expect(Array.from(result.positions)).toEqual([
    0,
    0,
    Math.fround(1 / 3),
    Math.fround(1 / 3),
    1,
    -1
  ]);
  expect(Array.from(result.sourceEdges)).toEqual([0, 2, 2]);
});

it('inserts mutual crossing nodes into both lines without erasing either line', () => {
  const result = computeLineLoopRemovalOnCPU(
    {positions: Float32Array.of(0, 0, 4, 0, 2, -1, 2, 1), lineOffsets: Uint32Array.of(0, 2, 4)},
    {vertexCapacity: 6}
  );
  expect(Array.from(result.status)).toEqual([6, 6, 0, 0]);
  expect(Array.from(result.positions)).toEqual([0, 0, 2, 0, 4, 0, 2, -1, 2, 0, 2, 1]);
  expect(Array.from(result.lineOffsets)).toEqual([0, 3, 6]);
  expect(Array.from(result.sourceEdges)).toEqual([0, 0, 0, 2, 2, 2]);
});

it('keeps a closed-form crossing finite across very different axis magnitudes', () => {
  const scaleX = 2 ** 100;
  const scaleY = 2 ** -100;
  const result = computeLineLoopRemovalOnCPU(
    {
      positions: Float32Array.of(0, 0, 4 * scaleX, 4 * scaleY, 0, 4 * scaleY, 4 * scaleX, 0),
      lineOffsets: Uint32Array.of(0, 4)
    },
    {vertexCapacity: 3}
  );
  expect(Array.from(result.status)).toEqual([3, 3, 0, 0]);
  expect(Array.from(result.positions)).toEqual([0, 0, 2 * scaleX, 2 * scaleY, 4 * scaleX, 0]);
  expect(result.positions.every(Number.isFinite)).toBe(true);
});

it('drops an overflowing whole line and still writes a later short line', () => {
  const result = computeLineLoopRemovalOnCPU(
    {
      positions: Float32Array.of(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 10, 0, 11, 0),
      lineOffsets: Uint32Array.of(0, 5, 7)
    },
    {vertexCapacity: 3}
  );
  expect(Array.from(result.status)).toEqual([2, 7, 1, 0]);
  expect(Array.from(result.lineOffsets)).toEqual([0, 0, 2]);
  expect(Array.from(result.positions)).toEqual([10, 0, 11, 0, 0, 0]);
  expect(Array.from(result.sourceEdges)).toEqual([5, 5, 0]);
});

it('reports the two required vertices of an unsplit line with zero output capacity', () => {
  // No intersection or erasure: both endpoints are required, and the whole line is dropped.
  const result = computeLineLoopRemovalOnCPU(
    {positions: Float32Array.of(0, 0, 1, 0), lineOffsets: Uint32Array.of(0, 2)},
    {vertexCapacity: 0}
  );
  expect(Array.from(result.status)).toEqual([0, 2, 1, 0]);
  expect(Array.from(result.lineOffsets)).toEqual([0, 0]);
  expect(result.positions).toHaveLength(0);
  expect(result.sourceEdges).toHaveLength(0);
  const props = {...getProps(), lineCount: 1, vertexCapacity: 0};
  const gpu = new GPULineLoopRemovalOperation(props);
  const cpu = new CPULineLoopRemovalOperation(props);
  expect(gpu.outputs['positions'].length).toBe(0);
  expect(gpu.outputs['sourceEdges'].length).toBe(0);
  expect(cpu.outputs['positions'].length).toBe(0);
  expect(cpu.outputs['sourceEdges'].length).toBe(0);
});

it('accepts a truly empty coordinate buffer and returns zero-filled outputs', () => {
  // There is one empty line and no vertices to retain or drop. GPU line splitting refuses this
  // input, so this documented CPU extension is deliberately excluded from GPU comparison cases.
  const result = computeLineLoopRemovalOnCPU(
    {positions: new Float32Array(0), lineOffsets: Uint32Array.of(0, 0)},
    {vertexCapacity: 2}
  );
  expect(Array.from(result.status)).toEqual([0, 0, 0, 0]);
  expect(Array.from(result.lineOffsets)).toEqual([0, 0]);
  expect(Array.from(result.positions)).toEqual([0, 0, 0, 0]);
  expect(Array.from(result.sourceEdges)).toEqual([0, 0]);
});

it('handles empty, singleton, repeated, closed and padded input lines', () => {
  const result = computeLineLoopRemovalOnCPU(
    {
      positions: Float32Array.of(
        99,
        99,
        0,
        0,
        1,
        1,
        1,
        1,
        2,
        0,
        10,
        0,
        11,
        0,
        11,
        1,
        10,
        0,
        22,
        22
      ),
      lineOffsets: Uint32Array.of(1, 1, 2, 5, 9)
    },
    {vertexCapacity: 8}
  );
  expect(Array.from(result.status)).toEqual([2, 2, 0, 0]);
  expect(Array.from(result.lineOffsets)).toEqual([0, 0, 0, 2, 2]);
  expect(Array.from(result.positions.subarray(0, 4))).toEqual([1, 1, 2, 0]);
  expect(Array.from(result.sourceEdges.subarray(0, 2))).toEqual([3, 3]);
});

it('erasure is idempotent and the result has no repeated visit for an integer walk', () => {
  const first = computeLineLoopRemovalOnCPU(
    {
      positions: Float32Array.of(0, 0, 2, 0, 2, 2, 0, 2, 0, 0, 3, 0, 3, 3),
      lineOffsets: Uint32Array.of(0, 7)
    },
    {vertexCapacity: 20}
  );
  const written = first.status[0];
  const second = computeLineLoopRemovalOnCPU(
    {positions: first.positions.subarray(0, written * 2), lineOffsets: Uint32Array.of(0, written)},
    {vertexCapacity: 20}
  );
  expect(Array.from(second.positions)).toEqual(Array.from(first.positions));
  expect(second.status[0]).toBe(written);
  const visits = new Set<string>();
  for (let vertex = 0; vertex < written; vertex++)
    visits.add(`${first.positions[vertex * 2]},${first.positions[vertex * 2 + 1]}`);
  expect(visits.size).toBe(written);
});

it('reports nonfinite input and rejects malformed arrays by argument name', () => {
  const result = computeLineLoopRemovalOnCPU(
    {positions: Float32Array.of(0, 0, NaN, 1, 2, 0), lineOffsets: Uint32Array.of(0, 3)},
    {vertexCapacity: 4}
  );
  expect(result.status[3]).toBe(1);
  expect(result.status[0]).toBe(3);
  expect(Number.isNaN(result.positions[2])).toBe(true);
  expect(() =>
    computeLineLoopRemovalOnCPU(
      {positions: Float32Array.of(0), lineOffsets: Uint32Array.of(0, 1)},
      {vertexCapacity: 1}
    )
  ).toThrow(/lines.positions/);
  for (const lineOffsets of [Uint32Array.of(0, 3), Uint32Array.of(0, 2, 1), new Uint32Array(0)])
    expect(() =>
      computeLineLoopRemovalOnCPU(
        {positions: new Float32Array(4), lineOffsets},
        {vertexCapacity: 1}
      )
    ).toThrow(/lines.lineOffsets/);
});

it('validates CPU props and spatial context at construction', () => {
  for (const [name, value] of [
    ['lineCount', 0],
    ['vertexCapacity', -1],
    ['intersectionCapacity', 0]
  ] as const)
    expect(() => new CPULineLoopRemovalOperation({...getProps(), [name]: value})).toThrow(
      new RegExp(`^loop-test .*${name}`)
    );
  expect(
    () =>
      new CPULineLoopRemovalOperation({
        ...getProps(),
        spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'none'}
      })
  ).toThrow(/^loop-test/);
});

it.each([
  GPULineLoopRemovalOperation,
  CPULineLoopRemovalOperation
])('%s validates graph shapes and disjoint names at construction', Operation => {
  expect(() => new Operation(getProps())).not.toThrow();
  expect(() => new Operation({...getProps(), vertexCapacity: -1})).toThrow(/^loop-test/);
  expect(() => new Operation({...getProps(), vertexCapacity: 0.5})).toThrow(/^loop-test/);
  expect(
    () =>
      new Operation({...getProps(), output: {...getProps().output, positions: 'inputPositions'}})
  ).toThrow(/^loop-test/);
  expect(
    () => new Operation({...getProps(), output: {...getProps().output, sourceEdges: 'positions'}})
  ).toThrow(/^loop-test/);
});

it('executes the CPU graph ports with the declared shapes and validates the offset length', async () => {
  // Two independent crossings exceed the explicit one-pair budget: suppress incomplete CPU geometry.
  const operation = new CPULineLoopRemovalOperation({...getProps(), intersectionCapacity: 1});
  const positions = Float32Array.of(0, 0, 4, 4, 0, 4, 4, 0, 10, 0, 14, 4, 10, 4, 14, 0);
  const lineOffsets = Uint32Array.of(0, 4, 8);
  const inputs = {
    inputPositions: {
      format: 'float32x2' as const,
      length: 8,
      chunks: [{values: positions, length: 8}],
      pack: () => positions
    },
    inputOffsets: {
      format: 'uint32' as const,
      length: 3,
      chunks: [{values: lineOffsets, length: 3}],
      pack: () => lineOffsets
    }
  };
  const output = await operation.execute({inputs});
  expect(Array.from(output['status'] as Uint32Array)).toEqual([0, 0, 0, 1]);
  expect(Array.from(output['lineOffsets'] as Uint32Array)).toEqual([0, 0, 0]);
  expect((output['positions'] as Float32Array).length).toBe(24);
  expect(operation.outputs['positions']).toEqual({
    type: 'vector',
    format: 'float32x2',
    length: 12
  });
  expect(() =>
    new CPULineLoopRemovalOperation({...getProps(), lineCount: 1}).execute({inputs})
  ).toThrow(/lineOffsets/);
});

it('multipart loop removal direct and CPU operation forms preserve every output', async () => {
  const positions = Float32Array.of(0, 0, 2, 2, 0, 2, 2, 0, 5, 0, 7, 0, 10, 0, 12, 0);
  const lineOffsets = Uint32Array.of(0, 4, 6, 8);
  const featureOffsets = Uint32Array.of(0, 0, 2, 3);
  const expected = {
    positions: [0, 0, 1, 1, 2, 0, 5, 0, 7, 0, 10, 0, 12, 0],
    lineOffsets: [0, 3, 5, 7],
    featureOffsets: [0, 0, 2, 3],
    sourceEdges: [0, 2, 2, 4, 4, 6, 6],
    status: [7, 7, 0, 0]
  };
  // Independent derivation: revisiting the exact crossing (1,1) erases the bowtie's upper
  // triangle; the two unrelated two-vertex parts retain their endpoints and original edge rows.
  const direct = computeLineLoopRemovalOnCPU(
    {positions, lineOffsets, featureOffsets},
    {vertexCapacity: 7, intersectionCapacity: 8}
  );
  for (const [name, values] of Object.entries(expected))
    expect(Array.from(direct[name as keyof typeof direct]!), name).toEqual(values);
  const props: LineLoopRemovalOperationProps = {
    ...getProps(),
    lineCount: 3,
    featureCount: 3,
    vertexCapacity: 7,
    lines: {
      positions: 'inputPositions',
      lineOffsets: 'inputOffsets',
      featureOffsets: 'inputFeatures'
    },
    output: {...getProps().output, featureOffsets: 'featureOffsets'}
  };
  const value = (format: 'float32x2' | 'uint32', values: Float32Array | Uint32Array) => ({
    format,
    length: values.length / (format === 'float32x2' ? 2 : 1),
    chunks: [{values, length: values.length / (format === 'float32x2' ? 2 : 1)}],
    pack: () => values
  });
  const operation = new CPULineLoopRemovalOperation(props);
  props.lines.featureOffsets = 'changed';
  const result = await operation.execute({
    inputs: {
      inputPositions: value('float32x2', positions),
      inputOffsets: value('uint32', lineOffsets),
      inputFeatures: value('uint32', featureOffsets)
    }
  });
  for (const [name, values] of Object.entries(expected))
    expect(Array.from(result[name] as Float32Array | Uint32Array), name).toEqual(values);
});
