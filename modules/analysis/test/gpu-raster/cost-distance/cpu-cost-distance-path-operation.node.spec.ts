// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {expect, it} from 'vitest';
import {
  computeCostDistancePathOnCPU,
  CPUCostDistancePathOperation
} from '../../../src/raster-analysis/cost-distance/cpu-cost-distance-path-operation';
import {
  GPUCostDistancePathOperation,
  type CostDistancePathOperationProps
} from '../../../src/raster-analysis/cost-distance/gpu-cost-distance-path-operation';
import {constructRasterOperationContributor} from '../../utils/raster-operation-test-utils';

const GRID = makeGPURasterMetadata({width: 3, height: 2});
const PROPS: CostDistancePathOperationProps = {
  id: 'path',
  grid: GRID,
  backLinks: 'backLinks',
  target: 'target',
  capacity: 3,
  output: {
    path: {
      ids: 'ids',
      status: {count: 'count', overflow: 'overflow', requiredCount: 'requiredCount'}
    },
    status: {iterationLimitReached: 'iterationLimitReached'}
  }
};

it('walks a hand-drawn diagonal then west path and publishes exact overflow', () => {
  // Grid: source at 0; cell 1 points west; cell 5 points northwest to 1.
  const links = new Uint32Array([0, 16, 0xffffffff, 0xffffffff, 0xffffffff, 32]);
  const result = computeCostDistancePathOnCPU(links, new Uint32Array([5]), GRID, 2);
  expect(Array.from(result.ids)).toEqual([5, 1]);
  expect(Array.from(result.count)).toEqual([2]);
  expect(Array.from(result.requiredCount)).toEqual([3]);
  expect(Array.from(result.overflow)).toEqual([1]);
  expect(Array.from(result.iterationLimitReached)).toEqual([0]);
});

it('rejects cycles, invalid directions, off-grid links, unreached cells and unresolved walks', () => {
  for (const [links, target, unresolved] of [
    [[1, 16, 0, 0, 0, 0], 0, 0],
    [[3, 0, 0, 0, 0, 0], 0, 0],
    [[16, 0, 0, 0, 0, 0], 0, 0],
    [[1, 0xffffffff, 0, 0, 0, 0], 0, 0],
    [[1, 0xfffffffe, 0, 0, 0, 0], 0, 1],
    [[0, 0, 0, 0, 0, 0], 6, 0]
  ] as const) {
    const result = computeCostDistancePathOnCPU(
      new Uint32Array(links),
      new Uint32Array([target]),
      GRID,
      3
    );
    expect(result.count[0]).toBe(0);
    expect(result.requiredCount[0]).toBe(0);
    expect(result.overflow[0]).toBe(0);
    expect(result.iterationLimitReached[0]).toBe(unresolved);
  }
});

it('validates CPU construction, GPU port construction and deferred contributor construction', () => {
  for (const Operation of [CPUCostDistancePathOperation, GPUCostDistancePathOperation]) {
    expect(() => new Operation({...PROPS, capacity: 0})).toThrow(/^path .*integer/);
  }
  expect(() => new CPUCostDistancePathOperation({...PROPS, grid: {...GRID, width: 0}})).toThrow(
    /^path grid.width/
  );
  const construct = (props: CostDistancePathOperationProps, length = 6) =>
    constructRasterOperationContributor(new GPUCostDistancePathOperation(props), {
      backLinks: {format: 'uint32', length},
      target: {format: 'uint32', length: 1}
    });
  expect(() => construct(PROPS)).not.toThrow();
  expect(() => construct({...PROPS, grid: {...GRID, width: 0}})).toThrow(/^path grid/);
  expect(() => construct(PROPS, 5)).toThrow(/^path backLinks/);
  expect(() =>
    computeCostDistancePathOnCPU(new Uint32Array(5), new Uint32Array([0]), GRID, 3)
  ).toThrow(/backLinks/);
  expect(() =>
    computeCostDistancePathOnCPU(new Uint32Array(6), new Uint32Array(0), GRID, 3)
  ).toThrow(/target/);
});

for (const defect of [
  'empty-result',
  'empty-count',
  'empty-limit',
  'duplicate-result',
  'duplicate-status',
  'input-result',
  'input-status',
  'empty-input'
] as const) {
  it(`FRM-01 cost path rejects ${defect} before keying outputs`, () => {
    const props = {
      ...PROPS,
      output: {
        path: {...PROPS.output.path, status: {...PROPS.output.path.status}},
        status: {...PROPS.output.status}
      }
    };
    if (defect === 'empty-result') props.output.path.ids = '';
    if (defect === 'empty-count') props.output.path.status.count = '';
    if (defect === 'empty-limit') props.output.status.iterationLimitReached = '';
    if (defect === 'duplicate-result') props.output.path.ids = props.output.path.status.count;
    if (defect === 'duplicate-status')
      props.output.path.status.requiredCount = props.output.path.status.count;
    if (defect === 'input-result') props.output.path.ids = props.backLinks;
    if (defect === 'input-status') props.output.status.iterationLimitReached = props.target;
    if (defect === 'empty-input') props.backLinks = '';
    for (const Operation of [CPUCostDistancePathOperation, GPUCostDistancePathOperation]) {
      expect(() => new Operation(props)).toThrow(/^path .*graph value/);
    }
  });
}

it('B3 captures inherited requiredCount before copying the compact status', async () => {
  let destination = 'required';
  const status = {...PROPS.output.path.status};
  Reflect.deleteProperty(status, 'requiredCount');
  Object.setPrototypeOf(
    status,
    new (class {
      get requiredCount() {
        return destination;
      }
    })()
  );
  const operation = new CPUCostDistancePathOperation({
    ...PROPS,
    output: {...PROPS.output, path: {...PROPS.output.path, status}}
  });
  expect(Object.keys(operation.outputs)).toContain('required');
  destination = 'replaced';
  const wrap = (values: Uint32Array) => ({
    format: 'uint32' as const,
    length: values.length,
    chunks: [{values, length: values.length}],
    pack: () => values
  });
  const result = await operation.execute({
    inputs: {
      backLinks: wrap(Uint32Array.of(0, 16, 0xffffffff, 0xffffffff, 0xffffffff, 32)),
      target: wrap(Uint32Array.of(5))
    }
  });
  expect(result['required']).toEqual(Uint32Array.of(3));
  expect(Object.keys(result).sort()).toEqual(Object.keys(operation.outputs).sort());
});
