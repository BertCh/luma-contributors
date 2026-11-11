// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  computePettittTestOnCPU,
  CPUPettittTestOperation,
  CPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT
} from '../../../src/gpu-spatial-analysis/change-points/cpu-pettitt-test-operation';
import {
  GPUPettittTestOperation,
  type PettittTestOperationProps
} from '../../../src/gpu-spatial-analysis/change-points/gpu-pettitt-test-operation';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {createOperationTestViewGetter} from '../../utils/statistics-dynamics-operation-test-utils';

const PROPS: PettittTestOperationProps = {
  id: 'pettitt-check',
  values: 'values',
  layout: 'cell-major',
  cellCount: 1,
  sliceCount: 5,
  output: {statistic: 'statistic', changeSlice: 'changeSlice', pValues: 'pValues'}
};

// Independent definition: enumerate each split and compare only pairs that cross it (no ranks).
function computeSplitOracle(values: number[]) {
  const finite = values
    .map((value, slice) => ({value, slice}))
    .filter(({value}) => Number.isFinite(value));
  let statistic = 0;
  let changeSlice = finite.length >= 2 ? finite[1].slice : 0xffffffff;
  for (let split = 1; split < finite.length; split++) {
    let total = 0;
    for (let first = 0; first < split; first++) {
      for (let second = split; second < finite.length; second++)
        total += Math.sign(finite[first].value - finite[second].value);
    }
    if (Math.abs(total) > statistic) {
      statistic = Math.abs(total);
      changeSlice = finite[split].slice;
    }
  }
  return {statistic, changeSlice};
}

it('rank prefixes agree with exhaustive cross-split comparisons, ties, gaps and both layouts', () => {
  const series = [
    [1, 1, 4, 4, 4],
    [NaN, 8, NaN, 8, Infinity],
    [NaN, NaN, 2, NaN, NaN],
    [5, 1, 3, 2, 4]
  ];
  for (const layout of ['cell-major', 'period-major'] as const) {
    const values =
      layout === 'cell-major'
        ? Float32Array.from(series.flat())
        : Float32Array.from(
            Array.from({length: 5}, (_, slice) => series.map(row => row[slice])).flat()
          );
    const result = computePettittTestOnCPU(values, {
      cellCount: series.length,
      sliceCount: 5,
      layout
    });
    series.forEach((row, cell) => {
      const expected = computeSplitOracle(row);
      expect(result.statistic[cell]).toBe(expected.statistic);
      expect(result.changeSlice[cell]).toBe(expected.changeSlice);
    });
    // Hand computation: K=6, n=5, p=2 exp(-216/150).
    expect(result.pValues[0]).toBeCloseTo(0.4738555174, 6);
    expect(result.pValues[1]).toBe(1);
    expect(result.pValues[2]).toBe(1);
  }
});

it('supports series above the GPU private-array limit and rejects inconsistent values', () => {
  const values = Float32Array.from({length: 600}, (_, slice) => slice);
  const result = computePettittTestOnCPU(values, {
    cellCount: 1,
    sliceCount: 600,
    layout: 'cell-major'
  });
  expect(result.statistic[0]).toBe(300 * 300);
  expect(result.changeSlice[0]).toBe(300);
  expect(() =>
    computePettittTestOnCPU(values.subarray(1), {
      cellCount: 1,
      sliceCount: 600,
      layout: 'cell-major'
    })
  ).toThrow(/values length/);
});

it('validates CPU construction and delegates GPU shape and limit checks to contributor construction', () => {
  for (const Operation of [CPUPettittTestOperation, GPUPettittTestOperation]) {
    expect(() => new Operation({...PROPS, output: {...PROPS.output, statistic: 'values'}})).toThrow(
      /^pettitt-check outputs.statistic/
    );
  }
  expect(() => new CPUPettittTestOperation({...PROPS, cellCount: 0})).toThrow(
    /^pettitt-check cellCount/
  );
  expect(() => new GPUPettittTestOperation({...PROPS, cellCount: 0})).toThrow(
    /^pettitt-check outputs.statistic.length/
  );
  expect(
    () =>
      new CPUPettittTestOperation({...PROPS, sliceCount: CPU_PETTITT_TEST_MAXIMUM_SLICE_COUNT + 1})
  ).toThrow(/^pettitt-check sliceCount/);
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = createOperationTestViewGetter(graph, {
      values: {format: 'float32', length: 513},
      statistic: {format: 'uint32', length: 1},
      changeSlice: {format: 'uint32', length: 1},
      pValues: {format: 'float32', length: 1}
    });
    expect(() =>
      new GPUPettittTestOperation({...PROPS, sliceCount: 513}).createContributor(getView)
    ).toThrow(/^pettitt-check sliceCount/);
    expect(() => new GPUPettittTestOperation(PROPS).createContributor(getView)).toThrow(
      /^pettitt-check values/
    );
    expect(() => new CPUPettittTestOperation({...PROPS, sliceCount: 513})).not.toThrow();
  } finally {
    device.destroy();
  }
});
