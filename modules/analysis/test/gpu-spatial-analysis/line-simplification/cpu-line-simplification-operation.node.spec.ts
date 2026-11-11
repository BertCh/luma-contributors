// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  computeLineSimplificationOnCPU,
  CPULineSimplificationOperation
} from '../../../src/gpu-spatial-analysis/line-simplification/cpu-line-simplification-operation';
import {
  GPULineSimplificationOperation,
  type LineSimplificationOperationProps
} from '../../../src/gpu-spatial-analysis/line-simplification/gpu-line-simplification-operation';
import {getGPULineSimplificationParameterValues} from '../../../src/gpu-spatial-analysis/line-simplification/line-simplification-parameters';

it('runs zero-capacity selection and keeps empty-output validation on both operations', async () => {
  const positions = new Float32Array([0, 0, 1, 0]);
  const lineOffsets = new Uint32Array([0, 2]);
  const parameters = getGPULineSimplificationParameterValues({tolerance: 0});
  const props: LineSimplificationOperationProps = {
    id: 'empty-selection',
    positions: 'positions',
    lineOffsets: 'offsets',
    parameters: 'parameters',
    importance: 'importance',
    rowCount: 2,
    lineCount: 1,
    capacity: 0,
    output: {
      ids: 'ids',
      outputMask: 'mask',
      lineCounts: 'lineCounts',
      lineStarts: 'lineStarts',
      status: {
        count: 'count',
        requiredCount: 'required',
        overflow: 'overflow',
        converged: 'converged',
        iterationCount: 'iterations'
      }
    }
  };
  // Both endpoints survive tolerance 0, but neither can be written to an empty output.
  const expected = {
    importance: new Float32Array([Infinity, Infinity]),
    ids: new Uint32Array(0),
    mask: new Uint32Array([1, 1]),
    lineCounts: new Uint32Array([2]),
    lineStarts: new Uint32Array([0]),
    count: new Uint32Array([0]),
    required: new Uint32Array([2]),
    overflow: new Uint32Array([1]),
    converged: new Uint32Array([1]),
    iterations: new Uint32Array([1])
  };
  const pure = computeLineSimplificationOnCPU({positions, lineOffsets, parameters}, {capacity: 0});
  expect(pure.ids.length).toBe(0);
  expect(Array.from(pure.count)).toEqual([0]);
  expect(Array.from(pure.requiredCount)).toEqual([2]);
  expect(Array.from(pure.overflow)).toEqual([1]);
  const gpuOperation = new GPULineSimplificationOperation(props);
  expect(gpuOperation.outputs['ids'].length).toBe(0);
  expect(gpuOperation.contract.outputs).toMatchObject({ids: {length: 0}});
  const cpuOperation = new CPULineSimplificationOperation(props);
  expect(cpuOperation.outputs['ids'].length).toBe(0);
  expect(
    await cpuOperation.execute({
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
          chunks: [{values: lineOffsets, length: 2}],
          pack: () => lineOffsets
        },
        parameters: {
          format: 'float32',
          length: parameters.length,
          chunks: [{values: parameters, length: parameters.length}],
          pack: () => parameters
        }
      }
    })
  ).toEqual(expected);
  for (const Operation of [GPULineSimplificationOperation, CPULineSimplificationOperation]) {
    expect(() => new Operation({...props, capacity: -1})).toThrow(/^empty-selection/);
    expect(() => new Operation({...props, capacity: 0.5})).toThrow(/^empty-selection/);
    expect(() => new Operation({...props, output: {...props.output, ids: 'positions'}})).toThrow(
      /^empty-selection/
    );
  }
});

it('matches hand-calculated Douglas-Peucker distances and ancestor minima on multiple lines', () => {
  const result = computeLineSimplificationOnCPU(
    {
      positions: new Float32Array([0, 0, 1, 1, 2, 3, 3, 1, 4, 0, 0, 0, 1, 2, 2, 0, 3, 2, 4, 0]),
      lineOffsets: new Uint32Array([0, 5, 10]),
      parameters: getGPULineSimplificationParameterValues({tolerance: 0.5})
    },
    {capacity: 6}
  );
  expect(Array.from(result.importance)).toEqual([
    Infinity,
    Math.fround(1 / Math.sqrt(13)),
    3,
    Math.fround(1 / Math.sqrt(13)),
    Infinity,
    Infinity,
    2,
    Math.fround(4 / Math.sqrt(13)),
    Math.fround(4 / Math.sqrt(13)),
    Infinity
  ]);
  expect(Array.from(result.ids)).toEqual([0, 2, 4, 5, 6, 7]);
  expect(Array.from(result.lineCounts)).toEqual([3, 5]);
  expect(Array.from(result.lineStarts)).toEqual([0, 3]);
  expect(Array.from(result.outputMask)).toEqual([1, 0, 1, 0, 1, 1, 1, 1, 1, 1]);
  expect(result.count[0]).toBe(6);
  expect(result.requiredCount[0]).toBe(8);
  expect(result.overflow[0]).toBe(1);
  expect(result.converged[0]).toBe(1);
  expect(result.iterationCount[0]).toBe(1);
});

it('matches hand-computed sequential Visvalingam removal, row ties and strict tolerance', () => {
  // Both first triangles have area 1/2. Remove row 1 first; row 2 then has area 3/2.
  const result = computeLineSimplificationOnCPU(
    {
      positions: new Float32Array([0, 0, 1, 1, 2, 1, 3, 0]),
      lineOffsets: new Uint32Array([0, 4]),
      parameters: getGPULineSimplificationParameterValues({tolerance: 0.5, method: 'visvalingam'})
    },
    {method: 'visvalingam'}
  );
  expect(Array.from(result.importance)).toEqual([Infinity, 0.5, 1.5, Infinity]);
  expect(Array.from(result.ids.slice(0, result.count[0]))).toEqual([0, 2, 3]);
  expect(result.iterationCount[0]).toBe(2);
});

it('honors time interpolation, extrapolation and nonpositive-duration fallback', () => {
  const positions = new Float32Array([0, 0, 2, 0, 4, 0]);
  for (const [timestamps, expected] of [
    [new Float32Array([0, 1, 4]), 1],
    [new Float32Array([0, -1, 4]), 3],
    [new Float32Array([4, 3, 0]), 2]
  ] as const) {
    const result = computeLineSimplificationOnCPU(
      {positions, lineOffsets: new Uint32Array([0, 3]), timestamps},
      {metric: 'time-ratio'}
    );
    expect(result.importance[1]).toBe(expected);
  }
});

it('propagates the removed effective-area floor when a surviving triangle collapses', () => {
  // Initial areas are both 1; row 1 wins the tie. Removing it leaves row 2 on the
  // endpoint chord (raw area 0), but its effective area must retain the removed area 1.
  const result = computeLineSimplificationOnCPU(
    {positions: new Float32Array([0, 0, 1, 1, 2, 0, 4, 0]), lineOffsets: new Uint32Array([0, 4])},
    {method: 'visvalingam'}
  );
  expect(Array.from(result.importance)).toEqual([Infinity, 1, 1, Infinity]);
});

it('matches a closed-form equatorial seam chord distance and a collapsed chord', () => {
  const result = computeLineSimplificationOnCPU(
    {positions: new Float32Array([179, 0, -180, 1, -179, 0]), lineOffsets: new Uint32Array([0, 3])},
    {
      spatialContext: {
        coordinateSpace: 'longitude-latitude',
        metric: 'great-circle',
        sphereRadius: 1
      }
    }
  );
  expect(result.importance[1]).toBe(Math.fround(Math.PI / 180));
  const collapsed = computeLineSimplificationOnCPU({
    positions: new Float32Array([0, 0, 3, 4, 0, 0]),
    lineOffsets: new Uint32Array([0, 3])
  });
  expect(collapsed.importance[1]).toBe(5);
});

it('supports selection-only, uncovered rows, empty lines and zero capacity', () => {
  const result = computeLineSimplificationOnCPU(
    {
      positions: new Float32Array(10),
      lineOffsets: new Uint32Array([1, 1, 4]),
      importance: new Float32Array([Infinity, Infinity, 0.5, Infinity, Infinity]),
      parameters: getGPULineSimplificationParameterValues({tolerance: 0.5})
    },
    {computeImportance: false, capacity: 0}
  );
  expect(Array.from(result.outputMask)).toEqual([0, 1, 0, 1, 0]);
  expect(Array.from(result.lineCounts)).toEqual([0, 2]);
  expect(result.requiredCount[0]).toBe(2);
  expect(result.count[0]).toBe(0);
  expect(result.overflow[0]).toBe(1);
});

it('throws for inconsistent inputs and validates both operation contracts', () => {
  const input = {positions: new Float32Array(6), lineOffsets: new Uint32Array([0, 3])};
  expect(() => computeLineSimplificationOnCPU({...input, positions: new Float32Array(5)})).toThrow(
    /positions/
  );
  expect(() =>
    computeLineSimplificationOnCPU({...input, lineOffsets: new Uint32Array([3, 2])})
  ).toThrow(/lineOffsets/);
  expect(() =>
    computeLineSimplificationOnCPU({...input, lineOffsets: new Uint32Array([0, 4])})
  ).toThrow(/lineOffsets/);
  expect(() => computeLineSimplificationOnCPU({...input, timestamps: new Float32Array(1)})).toThrow(
    /timestamps/
  );
  expect(() => computeLineSimplificationOnCPU({...input, parameters: new Float32Array(1)})).toThrow(
    /parameters/
  );
  expect(() => computeLineSimplificationOnCPU(input, {computeImportance: false})).toThrow(
    /importance/
  );
  const props: LineSimplificationOperationProps = {
    id: 'line-check',
    positions: 'positions',
    lineOffsets: 'offsets',
    importance: 'importance',
    rowCount: 3,
    lineCount: 1
  };
  for (const Operation of [CPULineSimplificationOperation, GPULineSimplificationOperation]) {
    expect(() => new Operation(props)).not.toThrow();
    expect(() => new Operation({...props, rowCount: 0})).toThrow(/^line-check/);
    expect(() => new Operation({...props, importance: 'positions'})).toThrow(/^line-check/);
  }
  expect(() => new CPULineSimplificationOperation({...props, metric: 'time-ratio'})).toThrow(
    /^line-check.*timestamps/
  );
  expect(() => new CPULineSimplificationOperation({...props, maximumIterations: 0})).toThrow(
    /^line-check.*maximumIterations/
  );
});
