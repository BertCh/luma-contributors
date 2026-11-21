// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUGraph, gpuGraph} from '@luma.gl/graph/gpu-core';
import {CPUVector} from '@luma.gl/graph/gpu-data';
import {expect, it} from 'vitest';
import {GPU_CONTRIBUTOR_OPERATION_LOWERING} from '../../../src/analysis-core/operations/gpu-contributor-operation';
import {
  GPUNeighborhoodSummary,
  GPUNeighborhoodSummaryOperation
} from '../../../src/gpu-spatial-analysis/neighborhood-summary/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {AnalysisRig} from '../catchment-accessibility/rig';
import {getCPUVectorRows, makeCPUVector} from '../../utils/mixed-execution-test-utils';

it('pins every summary output word for complete two-row weights without status views', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const rig = new AnalysisRig(device);
  try {
    const table = rig.output('float32', 4);
    const modes = rig.output('uint32', 2);
    const entropy = rig.output('float32', 2);
    const overflow = rig.output('uint32', 1);
    rig.run(
      new GPUNeighborhoodSummary({
        weights: rig.weights({offsets: [0, 1, 2], neighbors: [1, 0], weights: [1, 1]}, 0),
        values: rig.input(Float32Array.of(10, 20), 'float32'),
        categories: rig.input(Uint32Array.of(5, 7), 'uint32'),
        statistics: ['count', 'mean'],
        output: {table, modes, entropy, status: {overflow}}
      })
    );
    // The guide's two-row fixture: each row has the other as its sole neighbor, so the
    // count/mean pairs are [1, 20] and [1, 10], with one category and zero entropy per row.
    const tableWords = new Uint32Array(Float32Array.from(await rig.readFloat(table)).buffer);
    const entropyWords = new Uint32Array(Float32Array.from(await rig.readFloat(entropy)).buffer);
    expect(Array.from(tableWords)).toEqual([0x3f800000, 0x41a00000, 0x3f800000, 0x41200000]);
    expect(await rig.readUint(modes)).toEqual([7, 5]);
    expect(Array.from(entropyWords)).toEqual([0, 0]);
    expect(await rig.readUint(overflow)).toEqual([0]);
  } finally {
    rig.destroy();
  }
});

it('lowers getter-backed candidateOverflow into the named GPU summary loss gate', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  class Status {
    get candidateOverflow() {
      return 'candidate-loss';
    }
  }
  const summary = new GPUNeighborhoodSummaryOperation({
    rowCount: 2,
    weights: {offsets: 'offsets', neighbors: 'neighbors', weights: 'weights', status: new Status()},
    values: 'values',
    statistics: ['mean'],
    output: {table: 'table'}
  });
  const graph = new GPUGraph();
  const addInput = (
    name: string,
    format: 'float32' | 'uint32',
    values: Float32Array | Uint32Array
  ) => {
    graph.addInput(name, gpuGraph.vector(format, {length: values.length}));
    return makeCPUVector(format, values);
  };
  const inputs = {
    offsets: addInput('offsets', 'uint32', Uint32Array.of(0, 1, 2)),
    neighbors: addInput('neighbors', 'uint32', Uint32Array.of(1, 0)),
    weights: addInput('weights', 'float32', Float32Array.of(1, 1)),
    values: addInput('values', 'float32', Float32Array.of(10, 20))
  };
  graph.addInput('candidate-loss', gpuGraph.vector('uint32', {length: 1}));
  graph.add(summary);
  const compiled = graph.compile(device, {
    outputs: ['table'],
    outputLocations: {table: 'cpu'},
    lowerings: [GPU_CONTRIBUTOR_OPERATION_LOWERING]
  });
  try {
    for (const [candidateOverflow, expected] of [
      [0, [20, 10]],
      [1, [NaN, NaN]],
      [0, [20, 10]]
    ] as const) {
      const result = (await compiled.run({
        inputs: {
          ...inputs,
          'candidate-loss': makeCPUVector('uint32', Uint32Array.of(candidateOverflow))
        }
      })) as Record<string, unknown>;
      const table = result['table'];
      if (!(table instanceof CPUVector)) throw new Error('Expected the summary table on the CPU');
      expect(Array.from(getCPUVectorRows(table))).toEqual(expected);
    }
  } finally {
    compiled.destroy();
    summary.destroy();
  }
});
