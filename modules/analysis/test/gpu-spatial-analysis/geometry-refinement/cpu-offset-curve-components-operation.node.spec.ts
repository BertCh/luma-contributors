// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeOffsetCurveComponentsOnCPU,
  CPUOffsetCurveComponentsOperation
} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-offset-curve-components-operation';
import {
  GPUOffsetCurveComponentsOperation,
  type OffsetCurveComponentsOperationProps
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-offset-curve-components-operation';

function getInput() {
  return {
    original: {
      positions: Float32Array.of(0, 0, 6, 0, 10, 0, 16, 0),
      lineOffsets: Uint32Array.of(0, 2, 4)
    },
    raw: {
      positions: Float32Array.of(
        0,
        1,
        3,
        1,
        3,
        2,
        3,
        1,
        6,
        1,
        10,
        1,
        12,
        1,
        12,
        0,
        14,
        0,
        14,
        1,
        16,
        1
      ),
      lineOffsets: Uint32Array.of(0, 5, 11)
    }
  };
}

function getProps(): OffsetCurveComponentsOperationProps {
  return {
    id: 'components-test',
    original: {positions: 'originalPositions', lineOffsets: 'originalOffsets'},
    raw: {positions: 'rawPositions', lineOffsets: 'rawOffsets'},
    lineCount: 2,
    distance: 1,
    tolerance: 0,
    intersectionCapacity: 16,
    segmentCapacity: 32,
    vertexCapacity: 16,
    componentCapacity: 4,
    output: {
      positions: 'positions',
      components: 'components',
      sourceEdges: 'sourceEdges',
      status: 'status'
    }
  };
}

it('matches hand-computed distance bands, spur cancellation and three ordered components on two lines', () => {
  // The first curve has a doubled vertical spur; the two opposite fragments cancel.
  // The second dips inside the unit band, leaving separate x=10..12 and x=14..16 runs.
  const result = computeOffsetCurveComponentsOnCPU(getInput(), getProps());
  expect(Array.from(result.status)).toEqual([3, 7, 3, 7, 0, 0, 0, 0]);
  expect(Array.from(result.positions.subarray(0, 14))).toEqual([
    0, 1, 3, 1, 6, 1, 10, 1, 12, 1, 14, 1, 16, 1
  ]);
  expect(Array.from(result.components)).toEqual([0, 3, 0, 0, 3, 2, 1, 0, 5, 2, 1, 0, 0, 0, 0, 0]);
  expect(Array.from(result.sourceEdges.subarray(0, 7))).toEqual([0, 0, 3, 5, 5, 9, 9]);
  const provenance = computeOffsetCurveComponentsOnCPU(getInput(), {
    ...getProps(),
    rowsPerVertex: 2
  });
  expect(Array.from(provenance.sourceEdges.subarray(0, 7))).toEqual([0, 0, 1, 2, 2, 4, 4]);
});

it('publishes only complete component records while preserving partial vertex writes', () => {
  const result = computeOffsetCurveComponentsOnCPU(getInput(), {
    ...getProps(),
    vertexCapacity: 4,
    componentCapacity: 2
  });
  expect(Array.from(result.status)).toEqual([1, 3, 3, 7, 1, 0, 0, 0]);
  expect(Array.from(result.components)).toEqual([0, 3, 0, 0, 0, 0, 0, 0]);
  expect(Array.from(result.positions)).toEqual([0, 1, 3, 1, 6, 1, 10, 1]);
  const metadataOverflow = computeOffsetCurveComponentsOnCPU(getInput(), {
    ...getProps(),
    componentCapacity: 1
  });
  expect(Array.from(metadataOverflow.status)).toEqual([1, 3, 3, 7, 1, 0, 0, 0]);
  expect(Array.from(metadataOverflow.positions.subarray(0, 14))).toEqual([
    0, 1, 3, 1, 6, 1, 10, 1, 12, 1, 14, 1, 16, 1
  ]);
});

it.each([
  [0, 0],
  [0, 1],
  [2, 0]
])('supports vertexCapacity=%i and componentCapacity=%i without padding outputs', (vertexCapacity, componentCapacity) => {
  // A single unit-distance segment forms one component with two required vertices. Metadata is
  // withheld if either capacity is zero; coordinates are still written when vertex space exists.
  const props = {...getProps(), lineCount: 1, vertexCapacity, componentCapacity};
  const result = computeOffsetCurveComponentsOnCPU(
    {
      original: {positions: Float32Array.of(0, 0, 1, 0), lineOffsets: Uint32Array.of(0, 2)},
      raw: {positions: Float32Array.of(0, 1, 1, 1), lineOffsets: Uint32Array.of(0, 2)}
    },
    props
  );
  expect(Array.from(result.status)).toEqual([0, 0, 1, 2, 1, 0, 0, 0]);
  expect(Array.from(result.positions)).toEqual(vertexCapacity === 0 ? [] : [0, 1, 1, 1]);
  expect(Array.from(result.sourceEdges)).toEqual(new Array(vertexCapacity).fill(0));
  expect(Array.from(result.components)).toEqual(new Array(componentCapacity * 4).fill(0));
  const gpu = new GPUOffsetCurveComponentsOperation(props);
  const cpu = new CPUOffsetCurveComponentsOperation(props);
  expect(gpu.outputs['positions'].length).toBe(vertexCapacity);
  expect(gpu.outputs['components'].length).toBe(componentCapacity);
  expect(cpu.outputs['positions'].length).toBe(vertexCapacity);
  expect(cpu.outputs['components'].length).toBe(componentCapacity);
});

it('retains the GPU separation sentinel when a singleton original has no usable edge', () => {
  // Separation stays at the finite 3e38 initializer, below the requested 3.2e38 distance.
  // The raw segment is therefore rejected before any component can be published.
  const result = computeOffsetCurveComponentsOnCPU(
    {
      original: {positions: Float32Array.of(0, 0), lineOffsets: Uint32Array.of(0, 1)},
      raw: {positions: Float32Array.of(0, 0, 1, 0), lineOffsets: Uint32Array.of(0, 2)}
    },
    {...getProps(), distance: 3.2e38, vertexCapacity: 2, componentCapacity: 1}
  );
  expect(Array.from(result.status)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
  expect(Array.from(result.positions)).toEqual([0, 0, 0, 0]);
  expect(Array.from(result.components)).toEqual([0, 0, 0, 0]);
  expect(Array.from(result.sourceEdges)).toEqual([0, 0]);
});

it('accepts truly empty coordinate buffers and returns zero-filled outputs', () => {
  // GPU noding requires a raw coordinate row, so this documented extension has no GPU comparison.
  const empty = {positions: new Float32Array(0), lineOffsets: Uint32Array.of(0, 0)};
  const result = computeOffsetCurveComponentsOnCPU(
    {original: empty, raw: empty},
    {...getProps(), vertexCapacity: 2, componentCapacity: 1}
  );
  expect(Array.from(result.status)).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
  expect(Array.from(result.positions)).toEqual([0, 0, 0, 0]);
  expect(Array.from(result.components)).toEqual([0, 0, 0, 0]);
  expect(Array.from(result.sourceEdges)).toEqual([0, 0]);
});

it('matches a closed-form right-side bevel offset, retaining the outer join inside the unit circle', () => {
  // The right offset of a left 90-degree turn joins (4,-1) to (5,0) by a bevel.
  // Its midpoint is sqrt(1/2) from the source corner: outerJoin must exempt the two owner edges.
  // Three raw slots per input vertex, including padding, are exactly GPUOffsetCurve's layout.
  const result = computeOffsetCurveComponentsOnCPU(
    {
      original: {positions: Float32Array.of(0, 0, 4, 0, 4, 4), lineOffsets: Uint32Array.of(0, 3)},
      raw: {
        positions: Float32Array.of(0, -1, 0, -1, 0, -1, 4, -1, 5, 0, 5, 0, 5, 4, 5, 4, 5, 4),
        lineOffsets: Uint32Array.of(0, 9)
      }
    },
    {...getProps(), distance: -1, rowsPerVertex: 3}
  );
  expect(Array.from(result.status)).toEqual([1, 4, 1, 4, 0, 0, 0, 0]);
  expect(Array.from(result.positions.subarray(0, 8))).toEqual([0, -1, 4, -1, 5, 0, 5, 4]);
  expect(Array.from(result.sourceEdges.subarray(0, 4))).toEqual([0, 0, 1, 1]);
  const length =
    Math.hypot(
      result.positions[2] - result.positions[0],
      result.positions[3] - result.positions[1]
    ) +
    Math.hypot(
      result.positions[4] - result.positions[2],
      result.positions[5] - result.positions[3]
    ) +
    Math.hypot(
      result.positions[6] - result.positions[4],
      result.positions[7] - result.positions[5]
    );
  expect(length).toBe(8 + Math.SQRT2);
});

it('honours distance sign, distance magnitude and absolute geometric slack', () => {
  const input = {
    original: {positions: Float32Array.of(0, 0, 6, 0), lineOffsets: Uint32Array.of(0, 2)},
    raw: {positions: Float32Array.of(0, -0.9, 6, -0.9), lineOffsets: Uint32Array.of(0, 2)}
  };
  const options = {...getProps(), distance: -1};
  expect(computeOffsetCurveComponentsOnCPU(input, options).status[2]).toBe(0);
  expect(computeOffsetCurveComponentsOnCPU(input, {...options, tolerance: 0.2}).status[2]).toBe(1);
  expect(computeOffsetCurveComponentsOnCPU(input, {...options, distance: -0.8}).status[2]).toBe(1);
});

it('clips both roots of an endpoint circle against a closed-form oracle', () => {
  // At x=-3/4 the unit circle about the source endpoint cuts y at +/-sqrt(7)/4.
  const height = Math.sqrt(7) / 4;
  const result = computeOffsetCurveComponentsOnCPU(
    {
      original: {positions: Float32Array.of(0, 0, 4, 0), lineOffsets: Uint32Array.of(0, 2)},
      raw: {positions: Float32Array.of(-0.75, -2, -0.75, 2), lineOffsets: Uint32Array.of(0, 2)}
    },
    getProps()
  );
  expect(Array.from(result.status)).toEqual([2, 4, 2, 4, 0, 0, 0, 0]);
  expect(Array.from(result.positions.subarray(0, 8))).toEqual([
    -0.75,
    -2,
    -0.75,
    Math.fround(-height),
    -0.75,
    Math.fround(height),
    -0.75,
    2
  ]);
  expect(Array.from(result.components.subarray(0, 8))).toEqual([0, 2, 0, 0, 2, 2, 0, 0]);
});

it('splits a crossing in exact traversal order and removes duplicate fragments', () => {
  const input = {
    original: {positions: Float32Array.of(-10, -10, 10, -10), lineOffsets: Uint32Array.of(0, 2)},
    raw: {positions: Float32Array.of(0, 0, 4, 4, 0, 4, 4, 0), lineOffsets: Uint32Array.of(0, 4)}
  };
  // Everything is well outside the source band. A crossing adds two vertices to the traversal.
  const result = computeOffsetCurveComponentsOnCPU(input, {...getProps(), distance: 1});
  expect(Array.from(result.status)).toEqual([1, 6, 1, 6, 0, 0, 0, 0]);
  expect(Array.from(result.positions.subarray(0, 12))).toEqual([
    0, 0, 2, 2, 4, 4, 0, 4, 2, 2, 4, 0
  ]);
  expect(Array.from(result.sourceEdges.subarray(0, 6))).toEqual([0, 0, 0, 1, 2, 2]);
});

it('rejects inconsistent arrays with argument names and handles empty lines', () => {
  const input = getInput();
  expect(() =>
    computeOffsetCurveComponentsOnCPU(
      {...input, raw: {...input.raw, positions: Float32Array.of(1)}},
      getProps()
    )
  ).toThrow(/raw.positions/);
  expect(() =>
    computeOffsetCurveComponentsOnCPU(
      {...input, original: {...input.original, lineOffsets: Uint32Array.of(0, 5)}},
      getProps()
    )
  ).toThrow(/original.lineOffsets/);
  expect(() =>
    computeOffsetCurveComponentsOnCPU(
      {...input, raw: {...input.raw, lineOffsets: Uint32Array.of(0, 8, 7)}},
      getProps()
    )
  ).toThrow(/raw.lineOffsets/);
  expect(() =>
    computeOffsetCurveComponentsOnCPU(
      {...input, raw: {...input.raw, lineOffsets: Uint32Array.of(0, 5)}},
      getProps()
    )
  ).toThrow(/raw.lineOffsets/);
  const empty = {positions: new Float32Array(0), lineOffsets: Uint32Array.of(0, 0, 0)};
  expect(
    Array.from(computeOffsetCurveComponentsOnCPU({original: empty, raw: empty}, getProps()).status)
  ).toEqual(new Array(8).fill(0));
});

it('validates CPU parameters at construction', () => {
  for (const [name, value] of [
    ['distance', Infinity],
    ['tolerance', -1],
    ['rowsPerVertex', 0],
    ['lineCount', 0],
    ['intersectionCapacity', 0],
    ['segmentCapacity', 0]
  ] as const) {
    expect(() => new CPUOffsetCurveComponentsOperation({...getProps(), [name]: value})).toThrow(
      new RegExp(`^components-test .*${name}`)
    );
  }
});

it.each([
  GPUOffsetCurveComponentsOperation,
  CPUOffsetCurveComponentsOperation
])('%s validates output sizes and disjoint graph names at construction', Operation => {
  expect(() => new Operation(getProps())).not.toThrow();
  expect(() => new Operation({...getProps(), vertexCapacity: -1})).toThrow(/^components-test/);
  expect(() => new Operation({...getProps(), componentCapacity: -1})).toThrow(/^components-test/);
  expect(() => new Operation({...getProps(), vertexCapacity: 0.5})).toThrow(/^components-test/);
  expect(() => new Operation({...getProps(), componentCapacity: 0.5})).toThrow(/^components-test/);
  expect(
    () => new Operation({...getProps(), output: {...getProps().output, positions: 'rawPositions'}})
  ).toThrow(/^components-test/);
});

it('executes every CPU graph port and rejects mismatched graph input lengths', async () => {
  const data = getInput();
  const inputs = {
    originalPositions: {
      format: 'float32x2' as const,
      length: 4,
      chunks: [{values: data.original.positions, length: 4}],
      pack: () => data.original.positions
    },
    originalOffsets: {
      format: 'uint32' as const,
      length: 3,
      chunks: [{values: data.original.lineOffsets, length: 3}],
      pack: () => data.original.lineOffsets
    },
    rawPositions: {
      format: 'float32x2' as const,
      length: 11,
      chunks: [{values: data.raw.positions, length: 11}],
      pack: () => data.raw.positions
    },
    rawOffsets: {
      format: 'uint32' as const,
      length: 3,
      chunks: [{values: data.raw.lineOffsets, length: 3}],
      pack: () => data.raw.lineOffsets
    }
  };
  // Both GPU scratch capacities are deliberately too small for this scene; the CPU twin ignores them.
  const output = await new CPUOffsetCurveComponentsOperation({
    ...getProps(),
    intersectionCapacity: 1,
    segmentCapacity: 1
  }).execute({inputs});
  expect(Array.from(output['status'] as Uint32Array)).toEqual([3, 7, 3, 7, 0, 0, 0, 0]);
  expect((output['positions'] as Float32Array).length).toBe(32);
  expect((output['sourceEdges'] as Uint32Array).length).toBe(16);
  expect((output['components'] as Uint32Array).length).toBe(16);
  expect(() =>
    new CPUOffsetCurveComponentsOperation({...getProps(), lineCount: 1}).execute({inputs})
  ).toThrow(/lineOffsets/);
});
