// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {createSparseGatherComposition} from './emerging-hot-spots-composition-fixture';
import {GPUKeyLookup} from '../../../src/dataframe-analysis/key-join/gpu-key-join';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {computeEmergingHotSpotMoments} from './emerging-hot-spots-oracle';

it('CMP-03 zero-filled sparse gathers retain explicit population moments over two cells and slices', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const view = (name: string, length: number) => createTransientView(graph, name, 'uint32', length);
  try {
    const column = view('counts', 3);
    const gathered = view('gathered', 4);
    const rightRows = view('right-rows', 4);
    const join = new GPUKeyLookup({
      leftKeys: view('dictionary-periods', 4),
      rightKeys: view('sparse-keys', 3),
      gather: [{column, output: gathered, unmatchedValue: 0}],
      output: {rightRows}
    });
    const memory = createCPUGraphMemory();
    // This is the producer/adapter boundary: composite dictionary/period keys align the cube.
    // Dictionary [10,20], periods [0,1]; source keys [200,101,201] have counts [3,1,2].
    memory.write(column, Uint32Array.of(3, 1, 2));
    memory.write(rightRows, Uint32Array.of(0xffffffff, 1, 0, 2));
    executeCommandNodesOnCPU(
      join.getCommandNodes(graph).filter(node => node.id.endsWith('-gather-0')),
      memory
    );
    const values = memory.readUint32(gathered);
    expect(Array.from(values)).toEqual([0, 1, 3, 2]);
    // Hand moments: sum=6, mean=6/4; squared residuals total=5, population variance=5/4.
    expect(
      computeEmergingHotSpotMoments({gridWidth: 2, gridHeight: 1, sliceCount: 2, values})
    ).toEqual({
      count: 4,
      mean: 1.5,
      variance: 1.25,
      standardDeviation: 1.118033988749895
    });
    // The first slice is explicitly [0,3], not [0xffffffff,3].
    expect([values[0], values[2]]).toEqual([0, 3]);
    expect(
      computeEmergingHotSpotMoments({
        gridWidth: 2,
        gridHeight: 1,
        sliceCount: 1,
        values: [values[0], values[2]]
      })
    ).toEqual({count: 2, mean: 1.5, variance: 2.25, standardDeviation: 1.5});
  } finally {
    device.destroy();
  }
});

it('CMP-12 distinguishes observed-empty bins from unknown bins in consumer moments', () => {
  const shape = {gridWidth: 2, gridHeight: 1, sliceCount: 2};
  expect(computeEmergingHotSpotMoments({...shape, values: [0, 1, 3, 2]})).toEqual({
    count: 4,
    mean: 1.5,
    variance: 1.25,
    standardDeviation: 1.118033988749895
  });
  // Unknown removes one population member; remaining [1,3,2] have squared residual total 2.
  expect(computeEmergingHotSpotMoments({...shape, values: [NaN, 1, 3, 2]})).toEqual({
    count: 3,
    mean: 2,
    variance: 2 / 3,
    standardDeviation: 0.816496580927726
  });
});

for (const unknown of [false, true])
  it(`B1 declares the exact sparse-gather device composition on a null graph, unknown=${unknown}`, () => {
    const device = createNullWebGPUDevice();
    const rig = new AnalysisRig(device);
    try {
      const {producers} = createSparseGatherComposition(rig, unknown);
      expect(() => {
        for (const producer of producers) rig.graph.add(producer);
      }).not.toThrow();
    } finally {
      rig.destroy();
      device.destroy();
    }
  });
