// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {expect, it} from 'vitest';
import {
  computeRasterRegionGraphOnCPU,
  CPURasterRegionGraphOperation
} from '../../../src/raster-analysis/remote-sensing/cpu-raster-region-graph-operation';
import {
  GPURasterRegionGraphOperation,
  type RasterRegionGraphOperationProps
} from '../../../src/raster-analysis/remote-sensing/gpu-raster-region-graph-operation';
import {constructRasterOperationContributor} from '../../utils/raster-operation-test-utils';
import {makeCPUInputRows} from '../../analysis-core/mixed-execution/output-oracles-test-utils';
import {snapshotRasterRegionGraphOperationProps} from '../../../src/raster-analysis/remote-sensing/gpu-raster-region-graph-operation';

const GRID = makeGPURasterMetadata({width: 3, height: 2});
const PROPS: RasterRegionGraphOperationProps = {
  id: 'regions',
  grid: GRID,
  labels: 'labels',
  values: 'values',
  regionCount: 3,
  mergeThreshold: 2,
  maximumIterations: 4,
  output: {
    adjacency: 'adjacency',
    weights: 'weights',
    mapping: 'mapping',
    status: {iterationLimitReached: 'iterationLimitReached'}
  }
};

it('CPL-05 carries provisional watershed evidence separately from merge exhaustion', () => {
  const options = {...PROPS, grid: makeGPURasterMetadata({width: 3, height: 1}), mergeThreshold: 0};
  // Flat three-cell FIFO watershed after one step: the middle label is unresolved.
  // With all three cells resolved, the sole 1--2 contact has length one.
  const provisional = computeRasterRegionGraphOnCPU(
    new Uint32Array([1, 0xfffffffe, 2]),
    new Float32Array(3),
    {...options, labelsComplete: new Uint32Array([0])}
  );
  expect(provisional.adjacency[5]).toBe(0);
  expect(Array.from(provisional.complete)).toEqual([0]);
  expect(Array.from(provisional.iterationLimitReached)).toEqual([0]);
  const resolved = computeRasterRegionGraphOnCPU(new Uint32Array([1, 1, 2]), new Float32Array(3), {
    ...options,
    labelsComplete: new Uint32Array([1])
  });
  expect(resolved.adjacency[5]).toBe(1);
  expect(Array.from(resolved.mapping)).toEqual([0, 1, 1]);
  expect(Array.from(resolved.complete)).toEqual([1]);
  expect(Array.from(resolved.iterationLimitReached)).toEqual([0]);
  expect(() =>
    computeRasterRegionGraphOnCPU(new Uint32Array(3), new Float32Array(3), {
      ...options,
      labelsComplete: new Uint32Array(0)
    })
  ).toThrow(/labelsComplete/);
});

it('review B5 rejects wrong certificate element types before computing', () => {
  for (const labelsComplete of [Float32Array.of(NaN), Uint8Array.of(1)]) {
    expect(() =>
      computeRasterRegionGraphOnCPU(new Uint32Array(6), new Float32Array(6), {
        ...PROPS,
        labelsComplete: labelsComplete as never
      })
    ).toThrow(/labelsComplete.*Uint32Array/);
  }
});

it('review B3 captures inherited operation inputs, routes and nested status getters', () => {
  const output = Object.create(PROPS.output, {
    status: {
      value: Object.create(null, {
        iterationLimitReached: {get: () => 'pending'},
        complete: {get: () => 'complete'}
      })
    }
  });
  const props: RasterRegionGraphOperationProps = Object.create(
    {
      ...PROPS,
      grid: makeGPURasterMetadata({width: 4, height: 1}),
      regionCount: 4,
      maximumIterations: 1,
      mergeThreshold: 10,
      output
    },
    {
      labelsComplete: {get: () => 'sourceState'},
      searchStructure: {get: () => 'boruvka'}
    }
  );
  expect(snapshotRasterRegionGraphOperationProps(props).searchStructure).toBe('boruvka');
  for (const Operation of [CPURasterRegionGraphOperation, GPURasterRegionGraphOperation]) {
    const operation = new Operation(props);
    expect(operation.inputs).toHaveProperty('labelsComplete', 'sourceState');
    if (operation instanceof CPURasterRegionGraphOperation) {
      const rows = operation.execute({
        inputs: {
          labels: makeCPUInputRows('uint32', Uint32Array.of(0, 1, 2, 3)),
          values: makeCPUInputRows('float32', Float32Array.of(0, 1, 4, 5)),
          labelsComplete: makeCPUInputRows('uint32', Uint32Array.of(0))
        }
      }) as Record<string, Uint32Array>;
      expect(Array.from(rows['complete'])).toEqual([0]);
      // The two weight-1 pairs merge in the first Boruvka phase; the weight-3 bridge waits.
      expect(Array.from(rows['mapping'])).toEqual([0, 0, 2, 2]);
    }
  }
});

it('review S1 and S5 retain annotated no-certificate options and independent status words', () => {
  const options: RasterRegionGraphOperationProps = {...PROPS, maximumIterations: 1};
  // This assignment is also a compile-time check for the previously accepted annotated options.
  const compatibleOptions: Parameters<typeof computeRasterRegionGraphOnCPU>[2] = options;
  const result = computeRasterRegionGraphOnCPU(
    Uint32Array.of(0, 1, 2, 0, 1, 2),
    new Float32Array(6),
    compatibleOptions
  );
  expect(Array.from(result.complete)).toEqual([1]);
  expect(Array.from(result.iterationLimitReached)).toEqual([1]);
});

it('W3-16 validates original CPU destination names, including completeness', () => {
  for (const output of [
    {...PROPS.output, mapping: ''},
    {...PROPS.output, mapping: 'adjacency'},
    {...PROPS.output, mapping: 'labels'},
    {...PROPS.output, status: {...PROPS.output.status, complete: 'mapping'}},
    {...PROPS.output, status: {...PROPS.output.status, complete: ''}}
  ])
    expect(() => new CPURasterRegionGraphOperation({...PROPS, output})).toThrow(/^regions /);
});

it('CPL-05 CPU operation captures and publishes the completeness destination', () => {
  const output = {...PROPS.output, status: {...PROPS.output.status, complete: 'complete'}};
  const operation = new CPURasterRegionGraphOperation({
    ...PROPS,
    labelsComplete: 'labelsComplete',
    output
  });
  output.status.complete = 'mutated';
  const rows = operation.execute({
    inputs: {
      labels: makeCPUInputRows('uint32', new Uint32Array([0, 1, 2, 0, 1, 2])),
      values: makeCPUInputRows('float32', new Float32Array(6)),
      labelsComplete: makeCPUInputRows('uint32', new Uint32Array([0]))
    }
  }) as Record<string, Uint32Array>;
  expect(Array.from(rows['complete'])).toEqual([0]);
  expect(rows['mutated']).toBeUndefined();
  for (const Operation of [CPURasterRegionGraphOperation, GPURasterRegionGraphOperation])
    expect(() => new Operation({...PROPS, labelsComplete: 'labelsComplete'})).toThrow(
      /output.status.complete/
    );
});

it('matches hand-counted shared edges and distinguishes fixed linkage from adaptive means', () => {
  // Each column is a region. Means 0, 2, 4; two edges between successive columns.
  const labels = new Uint32Array([0, 1, 2, 0, 1, 2]);
  const values = new Float32Array([0, 2, 4, 0, 2, 4]);
  const fixed = computeRasterRegionGraphOnCPU(labels, values, {...PROPS, maximumIterations: 1});
  expect(Array.from(fixed.adjacency)).toEqual([0, 2, 0, 2, 0, 2, 0, 2, 0]);
  expect(Array.from(fixed.weights)).toEqual([NaN, 2, NaN, 2, NaN, 2, NaN, 2, NaN]);
  expect(Array.from(fixed.mapping)).toEqual([0, 0, 1]);
  expect(Array.from(fixed.iterationLimitReached)).toEqual([1]);
  // Lexicographic tie picks 0+1, mean 1; distance to region 2 becomes 3, stopping the merge.
  const adaptive = computeRasterRegionGraphOnCPU(labels, values, {...PROPS, mergeMode: 'adaptive'});
  expect(Array.from(adaptive.mapping)).toEqual([0, 0, 2]);
  expect(Array.from(adaptive.weights)).toEqual([NaN, NaN, NaN, NaN, NaN, 3, NaN, 3, NaN]);
});

it('excludes background and missing values from means, retaining boundary counts', () => {
  const result = computeRasterRegionGraphOnCPU(
    new Uint32Array([0, 1, 2, 0, 1, 0xffffffff]),
    new Float32Array([NaN, 1, 2, NaN, 3, 100]),
    PROPS
  );
  expect(Array.from(result.mapping)).toEqual([0, 1, 1]);
  expect(result.adjacency[1]).toBe(2);
  expect(result.weights[1]).toBeNaN();
  expect(result.weights[5]).toBe(0);
});

it('validates both operation forms and drops GPU-only size and round limits on CPU', () => {
  for (const Operation of [CPURasterRegionGraphOperation, GPURasterRegionGraphOperation]) {
    expect(() => new Operation({...PROPS, regionCount: 0})).toThrow(/^regions /);
  }
  const inputs = {
    labels: {format: 'uint32' as const, length: 6},
    values: {format: 'float32' as const, length: 6}
  };
  for (const override of [
    {mergeThreshold: -1},
    {maximumIterations: 0},
    {mergeMode: 'invalid' as never}
  ]) {
    expect(() => new CPURasterRegionGraphOperation({...PROPS, ...override})).toThrow(/^regions /);
    expect(() =>
      constructRasterOperationContributor(
        new GPURasterRegionGraphOperation({...PROPS, ...override}),
        inputs
      )
    ).toThrow(/^regions /);
  }
  expect(
    () => new CPURasterRegionGraphOperation({...PROPS, regionCount: 257, maximumIterations: 257})
  ).not.toThrow();
  // Auto now accepts fixed-linkage Boruvka above the dense reference maximum.
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterRegionGraphOperation({...PROPS, regionCount: 257}),
      inputs
    )
  ).not.toThrow();
  for (const override of [{searchStructure: 'dense' as const}, {mergeMode: 'adaptive' as const}]) {
    expect(() =>
      constructRasterOperationContributor(
        new GPURasterRegionGraphOperation({...PROPS, ...override, regionCount: 257}),
        inputs
      )
    ).toThrow(/^regions regionCount must be an integer in \[1, 256\]$/);
  }
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterRegionGraphOperation({...PROPS, regionCount: 65536}),
      inputs
    )
  ).toThrow(/^regions outputs\.adjacency\.length must be an integer in \[1, 4294967295\]$/);
  expect(() =>
    constructRasterOperationContributor(
      new GPURasterRegionGraphOperation({...PROPS, maximumIterations: 257}),
      inputs
    )
  ).toThrow(/^regions maximumIterations must be an integer in \[1, 256\]$/);
  expect(() =>
    computeRasterRegionGraphOnCPU(new Uint32Array(5), new Float32Array(6), PROPS)
  ).toThrow(/labels/);
  expect(() =>
    computeRasterRegionGraphOnCPU(new Uint32Array(6), new Float32Array(5), PROPS)
  ).toThrow(/values/);
});
