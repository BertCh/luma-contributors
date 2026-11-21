// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Buffer} from '@luma.gl/core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createLocalPermutationReplay,
  getGPUPermutationParameterValues,
  GPULocalPermutationTest
} from '../../../src/gpu-spatial-analysis/permutation-inference/index';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {createLocalReplayLiteralFixture} from './local-replay-literal-fixture';

it('replay pins literal terms, prefix, count and row bits independently of live draws (Tests §5)', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  const fixture = createLocalReplayLiteralFixture();
  const input = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    values: Uint32Array | Float32Array
  ) => {
    const buffer = createInputBuffer(device, values);
    buffers.push(buffer);
    return importGraphBuffer(graph, name, buffer, format, values.length);
  };
  const output = <Format extends 'uint32' | 'float32'>(
    name: string,
    format: Format,
    length: number
  ) => {
    const buffer = createOutputBuffer(device, length);
    buffers.push(buffer);
    return {buffer, view: importGraphBuffer(graph, name, buffer, format, length)};
  };
  const exceedances = output('exceedances', 'uint32', 4);
  const counts = output('counts', 'uint32', 4);
  const pValues = output('p-values', 'float32', 4);
  const observed = output('observed', 'float32', 4);
  const overflow = output('overflow', 'uint32', 1);
  const test = new GPULocalPermutationTest({
    id: 'literal-replay-test',
    weights: {
      offsets: input('offsets', 'uint32', fixture.weights.offsets),
      neighbors: input('neighbors', 'uint32', fixture.weights.neighbors),
      weights: input('weights', 'float32', fixture.weights.weights)
    },
    values: input('values', 'float32', fixture.values),
    parameters: input(
      'parameters',
      'uint32',
      getGPUPermutationParameterValues({seed: 11, permutationCount: 5, significanceLevel: 0.05})
    ),
    statistic: 'localMoran',
    islands: 'keep',
    alternative: 'greater',
    maximumNeighbors: 1,
    maximumPermutations: 9,
    stopAfterExceedances: 3,
    output: {
      exceedances: exceedances.view,
      permutationCounts: counts.view,
      pseudoPValues: pValues.view,
      observed: observed.view,
      status: {overflow: overflow.view}
    }
  });
  graph.add(test);
  const replays = [1, 3, 4].map(row => {
    const distribution = output(`draws-${row}`, 'float32', 11);
    distribution.buffer.write(new Float32Array(11).fill(1234));
    const distributionView = graph.createDataView(distribution.view.buffer, {
      format: 'float32',
      length: 9,
      byteOffset: 4
    });
    const info = output(`info-${row}`, 'float32', 3);
    // Model reuse after a successful five-draw replay: unavailable pins must reset this count.
    info.buffer.write(Float32Array.of(NaN, 5, NaN));
    graph.add(
      createLocalPermutationReplay({
        id: `literal-replay-${row}`,
        test,
        row: input(`pin-${row}`, 'uint32', Uint32Array.of(row)),
        output: {nullDistribution: distributionView, replayInfo: info.view}
      })
    );
    return {row, distribution, info};
  });
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    const publicObserved = await readFloat32(observed.buffer, 4);
    expect(Number.isFinite(publicObserved[1])).toBe(true);
    expect(publicObserved[1]).toBe(-1.5);
    expect(publicObserved[0]).toBe(0.25);
    expect((await readUint32(exceedances.buffer, 4))[1]).toBe(3);
    expect((await readUint32(counts.buffer, 4))[1]).toBe(3);
    const probabilities = await readFloat32(pValues.buffer, 4);
    expect(probabilities[1]).toBe(1);
    expect(probabilities[3]).toBeNaN();
    expect((await readUint32(overflow.buffer, 1))[0]).toBe(0);
    for (const {row, distribution, info} of replays) {
      const replayInfo = await readFloat32(info.buffer, 3);
      const replayBits = await readUint32(info.buffer, 3);
      const draws = await readFloat32(distribution.buffer, 11);
      // Guard only outside the supplied capacity; the contract defines its reported prefix.
      expect([draws[0], draws[10]]).toEqual([1234, 1234]);
      expect(replayBits[2]).toBe(row);
      if (row === 1) {
        expect(Number.isFinite(replayInfo[0])).toBe(true);
        expect(replayInfo[0]).toBe(-6);
        // Replay uses the full requested budget even when the live test stops at draw 3.
        expect(replayInfo[1]).toBe(5);
        expect(draws.slice(1, 6)).toEqual([-6, -6, -6, -6, -6]);
      } else {
        // Kept island 3 and out-of-range pin 4 both have no replay distribution.
        expect(replayInfo[1]).toBe(0);
        if (row === 4) expect(replayInfo[0]).toBeNaN();
      }
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});
