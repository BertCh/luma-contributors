// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {GPURasterWatershedSegmentation} from '../../src/gpu-raster/gpu-raster-watershed-segmentation';
import {makeGPURasterMetadata} from '../../src/gpu-raster/raster-grid';
import {InterpolationBenchmarkScope} from '../gpu-spatial-analysis/spatial-interpolation/interpolation-performance-harness';
import {requireTestDevice} from '../utils/require-test-device';
import {submitGraph} from '../utils/gpu-contributor-test-utils';
import {
  calculateWatershedForestReference,
  makeWatershedForestFixture,
  type WatershedForestFixture
} from './watershed-forest-reference';

for (const connectivity of [4, 8] as const) {
  it(`rooted raster forest matches independent Kruskal with ${connectivity} neighbors and changing inputs`, {
    timeout: 120000
  }, async () => {
    const device = await getWebGPUTestDevice('core');
    requireTestDevice(device);
    const scope = new InterpolationBenchmarkScope(device, `forest-${connectivity}`);
    const fixture = makeWatershedForestFixture(9, 7);
    fixture.connectivity = connectivity;
    const count = fixture.values.length;
    const values = scope.input('values', fixture.values, 'float32');
    const markers = scope.input('markers', fixture.markers, 'uint32');
    const validity = scope.input('validity', fixture.validity!, 'uint32');
    scope.graph.add(
      new GPURasterWatershedSegmentation({
        method: 'minimum-spanning-forest',
        connectivity,
        grid: makeGPURasterMetadata(fixture),
        input: {id: 'elevation', format: 'float32', validity, storage: {kind: 'buffer', values}},
        markers,
        output: {
          labels: scope.output('labels', count, 'uint32'),
          status: {converged: scope.output('converged', 1, 'uint32')}
        }
      })
    );
    const compiled = scope.graph.compile();
    try {
      const variants: WatershedForestFixture[] = Array.from({length: 12}, (_, seed) => ({
        ...makeWatershedForestFixture(9, 7, seed + 1),
        connectivity
      }));
      const plateau = {...makeWatershedForestFixture(9, 7), connectivity};
      plateau.values.fill(-0);
      plateau.markers.fill(0);
      plateau.markers[0] = 5;
      plateau.markers[count - 1] = 1;
      plateau.validity!.fill(1);
      variants.push(plateau);
      const barriers = {...makeWatershedForestFixture(9, 7), connectivity};
      barriers.markers[2] = 0xffffffff;
      barriers.markers[3] = 0xfffffffe;
      barriers.values[4] = NaN;
      barriers.values[5] = Infinity;
      barriers.values[6] = -Infinity;
      barriers.validity!.fill(1);
      variants.push(barriers);
      const noMarkers = {...makeWatershedForestFixture(9, 7), connectivity};
      noMarkers.markers.fill(0);
      variants.push(noMarkers);
      const allMarkers = {...makeWatershedForestFixture(9, 7), connectivity};
      allMarkers.markers.fill(7);
      variants.push(allMarkers);
      const invalid = {...makeWatershedForestFixture(9, 7), connectivity};
      invalid.validity!.fill(0);
      variants.push(invalid);
      for (const variant of variants) {
        scope.initialWrites[0].buffer.write(variant.values);
        scope.initialWrites[1].buffer.write(variant.markers);
        scope.initialWrites[2].buffer.write(variant.validity!);
        submitGraph(device, compiled, undefined);
        expect(await scope.read()).toEqual({
          labels: calculateWatershedForestReference(variant),
          converged: [1]
        });
      }
    } finally {
      compiled.destroy();
      scope.destroy();
    }
  });
}

it('rooted raster forest fails closed when contraction rounds are exhausted', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const scope = new InterpolationBenchmarkScope(device, 'forest-budget');
  const values = Float32Array.of(0, 0, 9, 0, 0, 9, 0, 0);
  const markers = Uint32Array.of(1, 0, 0, 0, 0, 0, 0, 2);
  scope.graph.add(
    new GPURasterWatershedSegmentation({
      method: 'minimum-spanning-forest',
      maximumIterations: 1,
      grid: makeGPURasterMetadata({width: 8, height: 1}),
      input: {
        id: 'values',
        format: 'float32',
        storage: {kind: 'buffer', values: scope.input('values', values, 'float32')}
      },
      markers: scope.input('markers', markers, 'uint32'),
      output: {
        labels: scope.output('labels', 8, 'uint32'),
        status: {converged: scope.output('converged', 1, 'uint32')}
      }
    })
  );
  const compiled = scope.graph.compile();
  try {
    for (const write of scope.initialWrites) write.buffer.write(write.values);
    submitGraph(device, compiled, undefined);
    expect(await scope.read()).toEqual({
      labels: [1, ...Array(6).fill(0xfffffffe), 2],
      converged: [0]
    });
  } finally {
    compiled.destroy();
    scope.destroy();
  }
});

it('rooted raster forest processes a complete 256-square raster beyond the FIFO limit', {
  timeout: 120000
}, async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const scope = new InterpolationBenchmarkScope(device, 'forest-large');
  const fixture = makeWatershedForestFixture(256, 256, 91);
  const count = fixture.values.length;
  scope.graph.add(
    new GPURasterWatershedSegmentation({
      method: 'minimum-spanning-forest',
      grid: makeGPURasterMetadata(fixture),
      input: {
        id: 'values',
        format: 'float32',
        validity: scope.input('validity', fixture.validity!, 'uint32'),
        storage: {kind: 'buffer', values: scope.input('values', fixture.values, 'float32')}
      },
      markers: scope.input('markers', fixture.markers, 'uint32'),
      output: {
        labels: scope.output('labels', count, 'uint32'),
        status: {converged: scope.output('converged', 1, 'uint32')}
      }
    })
  );
  const compiled = scope.graph.compile();
  try {
    for (const write of scope.initialWrites) write.buffer.write(write.values);
    submitGraph(device, compiled, undefined);
    expect(await scope.read()).toEqual({
      labels: calculateWatershedForestReference(fixture),
      converged: [1]
    });
  } finally {
    compiled.destroy();
    scope.destroy();
  }
});
