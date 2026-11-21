// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPUArealInterpolation} from '../../../src/gpu-spatial-analysis/areal-interpolation/gpu-areal-interpolation';
import {GPUSpatialLag} from '../../../src/gpu-spatial-analysis/spatial-weights/gpu-spatial-lag';
import {makeGPURasterMetadata} from '../../../src/raster-analysis/raster-grid/index';
import {requireTestDevice} from '../../../src/testing/require-test-device';
import {createParityGraph} from '../../oracles/parity-wp2-geodesic-harness';
import {readFloat32, readUint32, submitGraph} from '../../utils/gpu-contributor-test-utils';

for (const unweightedFastPath of [true, false]) {
  it(`clears poisoned areal candidate loss before downstream lag on reuse (fast=${unweightedFastPath})`, async () => {
    const device = await getWebGPUTestDevice();
    requireTestDevice(device);
    const fixture = createParityGraph(device);
    const {graph, input, output} = fixture;
    try {
      const overflow = output('overflow', 1, 'uint32');
      const count = output('count', 1, 'uint32');
      const requiredCount = output('required-count', 1, 'uint32');
      const requiredPairCount = output('required-pair-count', 1, 'uint32');
      const candidateOverflow = output('candidate', 1, 'uint32');
      const weights = {
        offsets: output('offsets', 2, 'uint32').view,
        neighbors: output('neighbors', 1, 'uint32').view,
        weights: output('weights', 1, 'float32').view,
        status: {
          overflow: overflow.view,
          count: count.view,
          requiredCount: requiredCount.view,
          candidateOverflow: candidateOverflow.view
        }
      };
      const lag = output('lag', 1, 'float32');
      graph.add(
        new GPUArealInterpolation({
          grid: makeGPURasterMetadata({width: 1, height: 1}),
          sourceZones: input('source', [0], 'uint32').view,
          targetZones: input('target', [0], 'uint32').view,
          sourceZoneCount: 1,
          targetZoneCount: 1,
          unweightedFastPath,
          output: {
            weights,
            status: {overflow: overflow.view, requiredPairCount: requiredPairCount.view}
          }
        })
      );
      graph.add(
        new GPUSpatialLag({weights, values: input('values', [7], 'float32').view, output: lag.view})
      );
      const compiled = graph.compile();
      try {
        for (const poison of [1, 99]) {
          for (const destination of [candidateOverflow, count, requiredCount, requiredPairCount])
            destination.buffer.write(Uint32Array.of(poison));
          submitGraph(device, compiled, undefined);
          expect(await readUint32(candidateOverflow.buffer, 1)).toEqual([0]);
          for (const destination of [count, requiredCount, requiredPairCount])
            expect(await readUint32(destination.buffer, 1)).toEqual([1]);
          expect(await readUint32(overflow.buffer, 1)).toEqual([0]);
          // The sole source occupies the sole target cell: weight=1 and lag=1*7.
          const actual = (await readFloat32(lag.buffer, 1))[0];
          expect(Number.isFinite(actual)).toBe(true);
          expect(actual).toBe(7);
        }
      } finally {
        compiled.destroy();
      }
    } finally {
      fixture.destroy();
    }
  });
}

for (const unweightedFastPath of [true, false]) {
  for (const capacity of [1, 2]) {
    it(`direct areal lag observes outer pair loss without nested status (fast=${unweightedFastPath}, capacity=${capacity})`, async () => {
      const device = await getWebGPUTestDevice();
      requireTestDevice(device);
      const fixture = createParityGraph(device);
      const {graph, input, output} = fixture;
      const overflow = output('overflow', 1, 'uint32');
      const lag = output('lag', 1, 'float32');
      const source = input('source', [0, 1], 'uint32');
      const producer = new GPUArealInterpolation({
        grid: makeGPURasterMetadata({width: 2, height: 1}),
        sourceZones: source.view,
        targetZones: input('target', [0, 0], 'uint32').view,
        sourceZoneCount: 2,
        targetZoneCount: 1,
        unweightedFastPath,
        output: {
          weights: {
            offsets: output('offsets', 2, 'uint32').view,
            neighbors: output('neighbors', capacity, 'uint32').view,
            weights: output('weights', capacity, 'float32').view
          },
          status: {overflow: overflow.view}
        }
      });
      expect(producer.props.output.weights.status?.overflow === overflow.view).toBe(true);
      graph.add(producer);
      graph.add(
        new GPUSpatialLag({
          weights: producer.props.output.weights,
          values: input('values', [7, 11], 'float32').view,
          output: lag.view
        })
      );
      const compiled = graph.compile();
      try {
        for (const split of [true, false, true]) {
          source.buffer.write(Uint32Array.of(0, split ? 1 : 0));
          submitGraph(device, compiled, undefined);
          expect(await readUint32(overflow.buffer, 1)).toEqual([Number(split && capacity === 1)]);
          const actual = (await readFloat32(lag.buffer, 1))[0];
          if (split && capacity === 1) expect(Number.isNaN(actual)).toBe(true);
          else {
            // Each source wholly transfers to target 0: 7+11=18; merged source 0 transfers 7.
            expect(Number.isFinite(actual)).toBe(true);
            expect(actual).toBe(split ? 18 : 7);
          }
        }
      } finally {
        compiled.destroy();
        fixture.destroy();
      }
    });
  }
}
