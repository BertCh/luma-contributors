// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {GPURasterWatershedSegmentation} from '../../src/raster-analysis/gpu-raster-watershed-segmentation';
import {requireTestDevice} from '../../src/testing/require-test-device';
import {createParityGraph} from '../oracles/parity-wp2-geodesic-harness';
import {readUint32, submitGraph} from '../utils/gpu-contributor-test-utils';
import {CPURasterWatershedSegmentationOperation} from '../../src/raster-analysis/raster-watershed/cpu-raster-watershed-segmentation-operation';
import {GPURasterWatershedSegmentationOperation} from '../../src/raster-analysis/raster-watershed/gpu-raster-watershed-segmentation-operation';
import {runOperationGraph} from '../utils/mixed-execution-test-utils';

it('R7 direct watershed keeps the inherited nodata barrier after nested edits', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const rig = createParityGraph(device);
  const labels = rig.output('labels', 3, 'uint32');
  const done = rig.output('done', 1, 'uint32');
  const limit = rig.output('limit', 1, 'uint32');
  const metadata = {noDataValue: -9};
  const affine: [number, number, number, number, number, number] = [1, 0, 0, 0, 1, 0];
  const grid = {width: 3, height: 1, affine, pixelInterpretation: 'area' as const};
  const input = Object.assign(Object.create(metadata), {
    id: 'surface',
    format: 'float32' as const,
    storage: {kind: 'buffer' as const, values: rig.input('surface', [0, -9, 0], 'float32').view}
  });
  const output = {
    labels: labels.view,
    status: {converged: done.view, iterationLimitReached: limit.view}
  };
  const contributor = new GPURasterWatershedSegmentation({
    grid,
    input,
    markers: rig.input('markers', [7, 0, 0], 'uint32').view,
    maximumSteps: 1,
    output
  });
  metadata.noDataValue = -8;
  grid.width = 2;
  affine[0] = 9;
  input.storage.values = rig.input('replacement-surface', [0, 0, 0], 'float32').view;
  output.labels = rig.output('replacement-labels', 3, 'uint32').view;
  output.status.converged = rig.output('replacement-done', 1, 'uint32').view;
  rig.graph.add(contributor);
  const compiled = rig.graph.compile();
  try {
    done.buffer.write(Uint32Array.of(12345));
    limit.buffer.write(Uint32Array.of(12345));
    submitGraph(device, compiled, undefined);
    // Barrier cell 1 isolates seed 0: its only pop empties the queue; cell 2 stays unseeded.
    expect(await readUint32(labels.buffer, 3)).toEqual([7, 0, 0]);
    expect(await readUint32(done.buffer, 1)).toEqual([1]);
    expect(await readUint32(limit.buffer, 1)).toEqual([0]);
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});

it('R1/R2 watershed forms keep inherited nodata and original output names', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  for (const mutateMetadata of [false, true]) {
    for (const Operation of [
      GPURasterWatershedSegmentationOperation,
      CPURasterWatershedSegmentationOperation
    ]) {
      const metadata = {noDataValue: -9};
      const input = Object.assign(Object.create(metadata), {
        id: 'surface',
        format: 'float32' as const,
        storage: {kind: 'buffer' as const, values: 'surface'}
      });
      const status = Object.create({converged: 'done', iterationLimitReached: 'limit'});
      const output = {labels: 'labels', status};
      const affine: [number, number, number, number, number, number] = [1, 0, 0, 0, 1, 0];
      const grid = {width: 3, height: 1, affine, pixelInterpretation: 'area' as const};
      const operation = new Operation({
        grid,
        input,
        markers: 'markers',
        maximumSteps: 1,
        output
      });
      // The middle raw-nodata cell disconnects seed 0; a single pop empties its queue.
      // Renaming the nested caller objects must not alter the already declared graph outputs.
      status.converged = 'renamed';
      status.iterationLimitReached = 'renamed-limit';
      output.labels = 'renamed-labels';
      if (mutateMetadata) {
        metadata.noDataValue = -8;
        grid.width = 2;
        affine[0] = 9;
      }
      const result = await runOperationGraph(device, {
        inputs: {
          surface: {format: 'float32', values: Float32Array.of(0, -9, 0)},
          markers: {format: 'uint32', values: Uint32Array.of(7, 0, 0)}
        },
        operations: [operation],
        outputs: ['labels', 'done', 'limit']
      });
      expect(Array.from(result['labels'])).toEqual([7, 0, 0]);
      expect(Array.from(result['done'])).toEqual([1]);
      expect(Array.from(result['limit'])).toEqual([0]);
    }
  }
});

for (const method of ['fifo', 'minimum-spanning-forest'] as const) {
  for (const complete of [false, true]) {
    it(`watershed forms pin independent ${method} stopping words (complete=${complete})`, async () => {
      const device = await getWebGPUTestDevice('core');
      requireTestDevice(device);
      const grid = {
        width: 3,
        height: 1,
        affine: [1, 0, 0, 0, 1, 0] as const,
        pixelInterpretation: 'area' as const
      };
      const budget =
        method === 'fifo'
          ? {maximumSteps: complete ? 3 : 1}
          : {maximumIterations: complete ? 2 : 1};
      const expectedLabels = complete ? [7, 7, 7] : [7, 0xfffffffe, 0xfffffffe];
      // The low (1,2) edge joins first in the forest; round two connects it to seed 0.
      // FIFO budget one leaves cell 1 queued and cell 2 undiscovered; three pops exhaust it.
      const rig = createParityGraph(device);
      const labels = rig.output('labels', 3, 'uint32');
      const converged = rig.output('done', 1, 'uint32');
      const limit = rig.output('limit', 1, 'uint32');
      rig.graph.add(
        new GPURasterWatershedSegmentation({
          grid,
          method,
          ...budget,
          input: {
            id: 'surface',
            format: 'float32',
            storage: {kind: 'buffer', values: rig.input('surface', [5, 0, 0], 'float32').view}
          },
          markers: rig.input('markers', [7, 0, 0], 'uint32').view,
          output: {
            labels: labels.view,
            status: {converged: converged.view, iterationLimitReached: limit.view}
          }
        })
      );
      const compiled = rig.graph.compile();
      try {
        converged.buffer.write(Uint32Array.of(12345));
        limit.buffer.write(Uint32Array.of(12345));
        submitGraph(device, compiled, undefined);
        expect(await readUint32(labels.buffer, 3)).toEqual(expectedLabels);
        expect(await readUint32(converged.buffer, 1)).toEqual(complete ? [1] : [0]);
        expect(await readUint32(limit.buffer, 1)).toEqual(complete ? [0] : [1]);
      } finally {
        compiled.destroy();
        rig.destroy();
      }
      for (const Operation of [
        GPURasterWatershedSegmentationOperation,
        CPURasterWatershedSegmentationOperation
      ]) {
        const result = await runOperationGraph(device, {
          inputs: {
            surface: {format: 'float32', values: Float32Array.of(5, 0, 0)},
            markers: {format: 'uint32', values: Uint32Array.of(7, 0, 0)}
          },
          operations: [
            new Operation({
              grid,
              method,
              ...budget,
              input: {
                id: 'surface',
                format: 'float32',
                storage: {kind: 'buffer', values: 'surface'}
              },
              markers: 'markers',
              output: {
                labels: 'labels',
                status: {converged: 'done', iterationLimitReached: 'limit'}
              }
            })
          ],
          outputs: ['labels', 'done', 'limit']
        });
        expect(Array.from(result['labels'])).toEqual(expectedLabels);
        expect(Array.from(result['done'])).toEqual(complete ? [1] : [0]);
        expect(Array.from(result['limit'])).toEqual(complete ? [0] : [1]);
      }
    });
  }
}

it('watershed stopping words distinguish a pending FIFO cell from an exhausted queue on replay', async () => {
  const device = await getWebGPUTestDevice('core');
  requireTestDevice(device);
  const rig = createParityGraph(device);
  const mask = rig.input('mask', [1, 1, 1, 1, 1], 'uint32');
  const labels = rig.output('labels', 5, 'uint32');
  const converged = rig.output('converged', 1, 'uint32');
  const iterationLimitReached = rig.output('limit', 1, 'uint32');
  rig.graph.add(
    new GPURasterWatershedSegmentation({
      grid: {
        width: 5,
        height: 1,
        affine: [1, 0, 0, 0, 1, 0],
        pixelInterpretation: 'area'
      },
      input: {
        id: 'surface',
        format: 'float32',
        storage: {kind: 'buffer', values: rig.input('surface', [0, 0, 0, 0, 0], 'float32').view},
        validity: mask.view
      },
      markers: rig.input('markers', [7, 0, 0, 0, 0], 'uint32').view,
      maximumSteps: 1,
      output: {
        labels: labels.view,
        status: {converged: converged.view, iterationLimitReached: iterationLimitReached.view}
      }
    })
  );
  const compiled = rig.graph.compile();
  try {
    for (const connected of [true, false, true]) {
      // One FIFO pop processes seed 0 and enqueues cell 1. Cell 2 has not been discovered,
      // so the queue cannot be complete. The published contract marks every usable non-seed
      // unresolved (0xfffffffe). Masking cells 1..4 instead leaves no pending work after pop 0.
      mask.buffer.write(Uint32Array.from(connected ? [1, 1, 1, 1, 1] : [1, 0, 0, 0, 0]));
      converged.buffer.write(Uint32Array.of(0xdeadbeef));
      iterationLimitReached.buffer.write(Uint32Array.of(0xdeadbeef));
      submitGraph(device, compiled, undefined);
      expect(await readUint32(labels.buffer, 5)).toEqual(
        connected ? [7, 0xfffffffe, 0xfffffffe, 0xfffffffe, 0xfffffffe] : [7, 0, 0, 0, 0]
      );
      expect(await readUint32(converged.buffer, 1)).toEqual(connected ? [0] : [1]);
      expect(await readUint32(iterationLimitReached.buffer, 1)).toEqual(connected ? [1] : [0]);
    }
  } finally {
    compiled.destroy();
    rig.destroy();
  }
});
