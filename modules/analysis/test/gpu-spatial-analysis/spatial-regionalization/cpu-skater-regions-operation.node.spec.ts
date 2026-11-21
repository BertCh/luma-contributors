// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {
  computeSkaterRegionsOnCPU,
  CPUSkaterRegionsOperation,
  type CPUSkaterRegionsInput
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/cpu-skater-regions-operation';
import {
  GPUSkaterRegionsOperation,
  type SkaterRegionsOperationProps
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions-operation';
import {
  getGPUSkaterRegionsParameterValues,
  GPU_SKATER_NO_CUT
} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-skater-regions';
import {GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT} from '../../../src/gpu-spatial-analysis/spatial-regionalization/gpu-spatial-weights-minimum-spanning-tree';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

const INPUT: CPUSkaterRegionsInput = {
  // Two four-row paths: 0--1--2--3 and 4--5--6--7.
  offsets: Uint32Array.of(0, 1, 3, 5, 6, 7, 9, 11, 12),
  neighbors: Uint32Array.of(1, 0, 2, 1, 3, 2, 5, 4, 6, 5, 7, 6),
  weights: new Float32Array(12).fill(1),
  treeEdgeFlags: Uint32Array.of(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
  componentLabels: Uint32Array.of(0, 0, 0, 0, 4, 4, 4, 4),
  values: Float32Array.of(0, 0, 0, 0, 10, 2, 10, 2, 1, -1, 1, -1, 5, 3, 5, 3),
  parameters: getGPUSkaterRegionsParameterValues({regionCount: 4})
};

it('rejects inconsistent weights metadata at CPU construction and GPU host lowering', () => {
  const information = {rowCount: 2, transform: 'raw', symmetric: true} as const;
  const props: SkaterRegionsOperationProps = {
    id: 'review-skater',
    rowCount: 2,
    edgeCount: 2,
    maximumRegionCount: 2,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights', information},
    treeEdgeFlags: 'treeEdgeFlags',
    componentLabels: 'componentLabels',
    values: 'values',
    parameters: 'parameters',
    output: {
      labels: 'labels',
      regionCount: 'regionCount',
      cutEdges: 'cutEdges',
      cutGains: 'cutGains'
    }
  };
  // The only tree edge separates values 0 and 2. SSD drops from 2 to 0; labels become 0,1.
  const result = computeSkaterRegionsOnCPU(
    {
      offsets: Uint32Array.of(0, 1, 2),
      neighbors: Uint32Array.of(1, 0),
      weights: Float32Array.of(1, 1),
      treeEdgeFlags: Uint32Array.of(1, 0),
      componentLabels: Uint32Array.of(0, 0),
      values: Float32Array.of(0, 2),
      parameters: getGPUSkaterRegionsParameterValues({regionCount: 2, minimumRegionSize: 1})
    },
    {maximumRegionCount: 2}
  );
  expect(result).toEqual({
    labels: Uint32Array.of(0, 1),
    regionCount: Uint32Array.of(2),
    cutEdges: Uint32Array.of(0),
    cutGains: Float32Array.of(2)
  });
  expect(() => new CPUSkaterRegionsOperation(props)).not.toThrow();
  expect(
    () =>
      new CPUSkaterRegionsOperation({...props, weights: {...props.weights, information: undefined}})
  ).not.toThrow();
  const device = createNullWebGPUDevice();
  try {
    for (const [field, invalidInformation] of [
      ['rowCount', {...information, rowCount: 3}],
      ['transform', {...information, transform: 'invalid' as 'raw'}],
      ['symmetric', {...information, symmetric: 'invalid' as 'unknown'}]
    ] as const) {
      const invalidProps = {...props, weights: {...props.weights, information: invalidInformation}};
      const expectedError = new RegExp(`^review-skater weights\\.information\\.${field}`);
      expect(() => new CPUSkaterRegionsOperation(invalidProps)).toThrow(expectedError);
      const graph = new GPUCommandGraph(device);
      const getView = <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format> =>
        createTransientView(
          graph,
          port,
          port === 'weights' || port === 'values' || port === 'cutGains' ? 'float32' : 'uint32',
          port === 'offsets' ? 3 : ['regionCount', 'cutEdges', 'cutGains'].includes(port) ? 1 : 2
        ) as GraphDataView<Format>;
      expect(() => new GPUSkaterRegionsOperation(invalidProps).createContributor(getView)).toThrow(
        expectedError
      );
    }
  } finally {
    device.destroy();
  }
});

it('leaves isolated rows as separate regions with unused cut log entries', () => {
  const result = computeSkaterRegionsOnCPU(
    {
      offsets: Uint32Array.of(0, 0, 0),
      neighbors: new Uint32Array(0),
      weights: new Float32Array(0),
      treeEdgeFlags: new Uint32Array(0),
      componentLabels: Uint32Array.of(0, 1),
      values: Float32Array.of(2, 3),
      parameters: getGPUSkaterRegionsParameterValues({regionCount: 3})
    },
    {maximumRegionCount: 3}
  );
  expect(Array.from(result.labels)).toEqual([0, 1]);
  expect(Array.from(result.regionCount)).toEqual([2]);
  expect(Array.from(result.cutEdges)).toEqual([GPU_SKATER_NO_CUT, GPU_SKATER_NO_CUT]);
  expect(Array.from(result.cutGains)).toEqual([0, 0]);
});

it('two-column, two-tree cuts equal the hand-computed partitions and SSD reductions', () => {
  const result = computeSkaterRegionsOnCPU(INPUT, {columnCount: 2, maximumRegionCount: 5});
  expect(Array.from(result.labels)).toEqual([0, 0, 2, 2, 4, 4, 6, 6]);
  expect(Array.from(result.regionCount)).toEqual([4]);
  expect(Array.from(result.cutEdges)).toEqual([2, 8, GPU_SKATER_NO_CUT, GPU_SKATER_NO_CUT]);
  // Direct SSD oracle: subtract each split's within-side SSD from the pre-cut SSD. This is
  // independent of the implementation's between-means formula.
  const getSquaredDeviation = (rows: number[]) => {
    let total = 0;
    for (let column = 0; column < 2; column++) {
      const mean = rows.reduce((sum, row) => sum + INPUT.values[row * 2 + column], 0) / rows.length;
      for (const row of rows) total += (INPUT.values[row * 2 + column] - mean) ** 2;
    }
    return total;
  };
  expect(result.cutGains[0]).toBe(
    getSquaredDeviation([0, 1, 2, 3]) - getSquaredDeviation([0, 1]) - getSquaredDeviation([2, 3])
  );
  expect(result.cutGains[1]).toBe(
    getSquaredDeviation([4, 5, 6, 7]) - getSquaredDeviation([4, 5]) - getSquaredDeviation([6, 7])
  );
  expect(Array.from(result.cutGains)).toEqual([104, 32, 0, 0]);
});

it('uses the vertex closest to the original root as label, even when it is not the smallest row', () => {
  const result = computeSkaterRegionsOnCPU(
    {
      offsets: Uint32Array.of(0, 1, 2, 4, 6),
      neighbors: Uint32Array.of(3, 2, 1, 3, 0, 2),
      weights: new Float32Array(6).fill(1),
      treeEdgeFlags: Uint32Array.of(1, 1, 0, 1, 0, 0),
      componentLabels: new Uint32Array(4),
      values: Float32Array.of(0, 10, 10, 0),
      parameters: getGPUSkaterRegionsParameterValues({regionCount: 2})
    },
    {maximumRegionCount: 2}
  );
  expect(Array.from(result.labels)).toEqual([0, 2, 2, 0]);
  expect(Array.from(result.cutEdges)).toEqual([3]);
  expect(Array.from(result.cutGains)).toEqual([100]);
});

it('allows zero-gain ties in slot order, honours size floors and clamps the target', () => {
  const tied = {
    ...INPUT,
    values: new Float32Array(16),
    parameters: getGPUSkaterRegionsParameterValues({regionCount: 20})
  };
  const result = computeSkaterRegionsOnCPU(tied, {columnCount: 2, maximumRegionCount: 4});
  expect(Array.from(result.cutEdges)).toEqual([0, 2, GPU_SKATER_NO_CUT]);
  expect(Array.from(result.labels)).toEqual([0, 1, 2, 2, 4, 4, 4, 4]);
  expect(result.regionCount[0]).toBe(4);
  const floored = computeSkaterRegionsOnCPU(
    {
      ...INPUT,
      parameters: getGPUSkaterRegionsParameterValues({regionCount: 8, minimumRegionSize: 2})
    },
    {columnCount: 2, maximumRegionCount: 8}
  );
  expect(floored.regionCount[0]).toBe(4);
  expect(Array.from(floored.cutEdges.slice(2))).toEqual(new Array(5).fill(GPU_SKATER_NO_CUT));
  const stopped = computeSkaterRegionsOnCPU(
    {...INPUT, parameters: getGPUSkaterRegionsParameterValues({regionCount: 1})},
    {columnCount: 2, maximumRegionCount: 4}
  );
  expect(Array.from(stopped.labels)).toEqual(Array.from(INPUT.componentLabels));
  expect(stopped.regionCount[0]).toBe(2);
});

it('tree status stops all cuts, while nonfinite values alone stop only their initial tree', () => {
  const treeStatus = new Uint32Array(GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.length);
  treeStatus[GPU_MINIMUM_SPANNING_TREE_STATUS_LAYOUT.invalidCount] = 1;
  const stopped = computeSkaterRegionsOnCPU(
    {...INPUT, treeStatus},
    {columnCount: 2, maximumRegionCount: 4}
  );
  expect(Array.from(stopped.labels)).toEqual(Array.from(INPUT.componentLabels));
  expect(Array.from(stopped.cutEdges)).toEqual(new Array(3).fill(GPU_SKATER_NO_CUT));
  const values = INPUT.values.slice();
  values[0] = NaN;
  const partial = computeSkaterRegionsOnCPU(
    {...INPUT, values, parameters: getGPUSkaterRegionsParameterValues({regionCount: 3})},
    {columnCount: 2, maximumRegionCount: 4}
  );
  expect(Array.from(partial.cutEdges)).toEqual([8, GPU_SKATER_NO_CUT, GPU_SKATER_NO_CUT]);
  expect(Array.from(partial.labels)).toEqual([0, 0, 0, 0, 4, 4, 6, 6]);
});

it('validates dimensions, array consistency and contributor construction during lowering', () => {
  const props: SkaterRegionsOperationProps = {
    id: 'skater-test',
    rowCount: 8,
    edgeCount: 12,
    columnCount: 2,
    maximumRegionCount: 4,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights'},
    treeEdgeFlags: 'treeEdgeFlags',
    componentLabels: 'componentLabels',
    values: 'values',
    parameters: 'parameters',
    output: {
      labels: 'labels',
      regionCount: 'regionCount',
      cutEdges: 'cutEdges',
      cutGains: 'cutGains'
    }
  };
  expect(new CPUSkaterRegionsOperation(props).contract.outputs).toEqual(
    new GPUSkaterRegionsOperation(props).contract.outputs
  );
  expect(() => new CPUSkaterRegionsOperation({...props, columnCount: 0})).toThrow(
    /^skater-test columnCount/
  );
  expect(() => new CPUSkaterRegionsOperation({...props, columnCount: 17})).not.toThrow();
  expect(() => new GPUSkaterRegionsOperation({...props, rowCount: 0})).toThrow(
    /^skater-test outputs.labels.length/
  );
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const getView = <Format extends GPUGraphVectorFormat>(port: string): GraphDataView<Format> =>
      createTransientView(
        graph,
        port,
        ['values', 'weights', 'cutGains'].includes(port) ? 'float32' : 'uint32',
        port === 'offsets'
          ? 9
          : ['neighbors', 'weights', 'treeEdgeFlags'].includes(port)
            ? 12
            : port === 'values'
              ? 16
              : port === 'parameters'
                ? 2
                : port === 'regionCount'
                  ? 1
                  : ['cutEdges', 'cutGains'].includes(port)
                    ? 3
                    : 8
      ) as GraphDataView<Format>;
    expect(() =>
      new GPUSkaterRegionsOperation({...props, columnCount: 0}).createContributor(getView)
    ).toThrow(/^skater-test columnCount/);
  } finally {
    device.destroy();
  }
  for (const [argument, changed] of [
    ['offsets', {...INPUT, offsets: Uint32Array.of(0, 99)}],
    ['weights', {...INPUT, weights: new Float32Array(1)}],
    ['treeEdgeFlags', {...INPUT, treeEdgeFlags: new Uint32Array(1)}],
    ['componentLabels', {...INPUT, componentLabels: new Uint32Array(1)}],
    ['values', {...INPUT, values: new Float32Array(1)}],
    ['parameters', {...INPUT, parameters: new Uint32Array(1)}]
  ] as const)
    expect(() =>
      computeSkaterRegionsOnCPU(changed, {columnCount: 2, maximumRegionCount: 4})
    ).toThrow(argument);
});
