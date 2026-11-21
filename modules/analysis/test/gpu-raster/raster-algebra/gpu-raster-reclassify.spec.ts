// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {
  GPUParameterBuffer,
  importGraphBuffer
} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  getGPURasterReclassifyParameterValues,
  GPURasterReclassify,
  type GPURasterReclassifySettings
} from '../../../src/raster-analysis/raster-algebra/index';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {reclassifyOnCPU, type ReclassifyScene} from './raster-algebra-oracle';
import {createRandom, expectFloatArraysClose} from './raster-algebra-test-utils';
import {requireTestDevice} from '../../../src/testing/require-test-device';

it('B2 exact count supports all terminals and rereads count and closure on resubmission', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const graph = new GPUCommandGraph(device);
  const buffers = [
    createInputBuffer(device, Float32Array.of(1, 7, 100, NaN)),
    createInputBuffer(
      device,
      Float32Array.from({length: 31}, (_, index) => index)
    ),
    createInputBuffer(
      device,
      Float32Array.from({length: 32}, (_, index) => 10 * index)
    ),
    createInputBuffer(device, Uint32Array.of(1, 0, 1, 1)),
    createInputBuffer(device, getGPURasterReclassifyParameterValues({breakCount: 0})),
    createInputBuffer(device, Uint32Array.of(31)),
    createOutputBuffer(device, 4),
    createOutputBuffer(device, 4),
    createOutputBuffer(device, 32)
  ];
  graph.add(
    new GPURasterReclassify({
      values: importGraphBuffer(graph, 'values', buffers[0], 'float32', 4),
      breaks: importGraphBuffer(graph, 'breaks', buffers[1], 'float32', 31),
      classValues: importGraphBuffer(graph, 'class-values', buffers[2], 'float32', 32),
      mask: importGraphBuffer(graph, 'mask', buffers[3], 'uint32', 4),
      parameters: importGraphBuffer(graph, 'parameters', buffers[4], 'float32', 4),
      activeBreakCount: importGraphBuffer(graph, 'active', buffers[5], 'uint32', 1),
      output: {
        classes: importGraphBuffer(graph, 'classes', buffers[6], 'uint32', 4),
        reclassified: importGraphBuffer(graph, 'reclassified', buffers[7], 'float32', 4),
        classCounts: importGraphBuffer(graph, 'counts', buffers[8], 'uint32', 32)
      }
    })
  );
  const compiled = graph.compile();
  try {
    for (const right of [false, true]) {
      if (right) {
        buffers[4].write(getGPURasterReclassifyParameterValues({breakCount: 0, closed: 'right'}));
        buffers[5].write(Uint32Array.of(30));
      }
      submitGraph(device, compiled, undefined);
      // Left counts breaks <= x; right counts breaks < x. Masked/NaN rows never count.
      const first = right ? 1 : 2;
      const last = right ? 30 : 31;
      expect(await readUint32(buffers[6], 4)).toEqual([first, 0xffffffff, last, 0xffffffff]);
      const mapped = await readFloat32(buffers[7], 4);
      expect(mapped[0]).toBe(first * 10);
      expect(mapped[1]).toBeNaN();
      expect(mapped[2]).toBe(last * 10);
      expect(mapped[3]).toBeNaN();
      expect(await readUint32(buffers[8], 32)).toEqual(
        Array.from({length: 32}, (_, index) => (index === first || index === last ? 1 : 0))
      );
    }
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('PCK-08 exact active count retains the final break beyond float32 integer precision', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const count = 16777217;
  // Exact ascending breaks 0..2^24 are all representable. Input is the final break;
  // left closure counts all 2^24+1 breaks, so its literal class is 16777217.
  const breaks = Float32Array.from({length: count}, (_, index) => index);
  const buffers = [
    createInputBuffer(device, breaks),
    createInputBuffer(device, Float32Array.of(16777216)),
    createInputBuffer(device, getGPURasterReclassifyParameterValues({breakCount: 0})),
    createInputBuffer(device, Uint32Array.of(count)),
    createOutputBuffer(device, 1)
  ];
  const graph = new GPUCommandGraph(device);
  graph.add(
    new GPURasterReclassify({
      values: importGraphBuffer(graph, 'values', buffers[1], 'float32', 1),
      breaks: importGraphBuffer(graph, 'breaks', buffers[0], 'float32', count),
      parameters: importGraphBuffer(graph, 'params', buffers[2], 'float32', 4),
      activeBreakCount: importGraphBuffer(graph, 'count', buffers[3], 'uint32', 1),
      output: {classes: importGraphBuffer(graph, 'classes', buffers[4], 'uint32', 1)}
    })
  );
  const compiled = graph.compile();
  try {
    submitGraph(device, compiled, undefined);
    expect(await readUint32(buffers[4], 1)).toEqual([16777217]);
  } finally {
    compiled.destroy();
    for (const buffer of buffers) buffer.destroy();
  }
});

it('GPURasterReclassify matches the CPU oracle as breaks and closure change without rebuilding', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const random = createRandom(7);
  const rowCount = 1531;
  const maximumBreakCount = 6;
  const values = new Float32Array(rowCount);
  const validity = new Uint32Array(rowCount);
  for (let row = 0; row < rowCount; row++) {
    const roll = random();
    // Integer-valued data lands exactly on breaks, which exercises the interval closure.
    values[row] =
      roll < 0.03
        ? NaN
        : roll < 0.06
          ? -9999
          : Math.floor(random() * 12) - 1 + (roll < 0.5 ? 0 : 0.5);
    validity[row] = random() < 0.04 ? 0 : 1;
  }
  const scene: ReclassifyScene = {
    values,
    validity,
    noDataValue: -9999,
    breaks: new Float32Array(maximumBreakCount),
    classValues: new Float32Array(maximumBreakCount + 1)
  };
  const breaksBuffer = createInputBuffer(device, scene.breaks);
  const classValuesBuffer = createInputBuffer(device, scene.classValues!);
  const parameterBuffer = new GPUParameterBuffer(device, {
    id: 'reclassify-parameters',
    format: 'float32',
    length: 4
  });
  const valuesBuffer = createInputBuffer(device, values);
  const validityBuffer = createInputBuffer(device, validity);
  const classesBuffer = createOutputBuffer(device, rowCount);
  const reclassifiedBuffer = createOutputBuffer(device, rowCount);
  const countsBuffer = createOutputBuffer(device, maximumBreakCount + 1);
  const graph = new GPUCommandGraph(device, {id: 'reclassify-graph'});
  graph.add(
    new GPURasterReclassify({
      id: 'reclassify',
      values: importGraphBuffer(graph, 'values', valuesBuffer, 'float32', rowCount),
      mask: importGraphBuffer(graph, 'validity', validityBuffer, 'uint32', rowCount),
      noDataValue: -9999,
      breaks: importGraphBuffer(graph, 'breaks', breaksBuffer, 'float32', maximumBreakCount),
      classValues: importGraphBuffer(
        graph,
        'class-values',
        classValuesBuffer,
        'float32',
        maximumBreakCount + 1
      ),
      parameters: parameterBuffer.importToGraph(graph),
      output: {
        classes: importGraphBuffer(graph, 'classes', classesBuffer, 'uint32', rowCount),
        reclassified: importGraphBuffer(
          graph,
          'reclassified',
          reclassifiedBuffer,
          'float32',
          rowCount
        ),
        classCounts: importGraphBuffer(
          graph,
          'counts',
          countsBuffer,
          'uint32',
          maximumBreakCount + 1
        )
      }
    })
  );
  const compiled = graph.compile();
  const frames: {breaks: number[]; classValues: number[]; settings: GPURasterReclassifySettings}[] =
    [
      {
        breaks: [0, 3, 5.5, 8, 10, 11],
        classValues: [10, 20, 30, 40, 50, 60, 70],
        settings: {breakCount: 6}
      },
      {
        breaks: [0, 3, 5.5, 8, 10, 11],
        classValues: [10, 20, 30, 40, 50, 60, 70],
        settings: {breakCount: 6, closed: 'right'}
      },
      {breaks: [2, 4, 6, 0, 0, 0], classValues: [NaN, 1, 2, 3, 0, 0, 0], settings: {breakCount: 3}},
      {breaks: [0, 0, 0, 0, 0, 0], classValues: [5, 0, 0, 0, 0, 0, 0], settings: {breakCount: 0}},
      // Counts above the compile-time maximum clamp to it.
      {breaks: [-1, 1, 2, 3, 4, 5], classValues: [0, 1, 2, 3, 4, 5, 6], settings: {breakCount: 99}}
    ];
  for (const frame of frames) {
    scene.breaks.set(frame.breaks);
    scene.classValues!.set(frame.classValues);
    breaksBuffer.write(scene.breaks);
    classValuesBuffer.write(scene.classValues!);
    parameterBuffer.write(getGPURasterReclassifyParameterValues(frame.settings));
    submitGraph(device, compiled, undefined);
    const expected = reclassifyOnCPU(
      scene,
      frame.settings.breakCount,
      frame.settings.closed === 'right'
    );
    expect(await readUint32(classesBuffer, rowCount)).toEqual(Array.from(expected.classes));
    expect(await readUint32(countsBuffer, maximumBreakCount + 1)).toEqual(
      Array.from(expected.classCounts)
    );
    expectFloatArraysClose(
      await readFloat32(reclassifiedBuffer, rowCount),
      expected.reclassified,
      0,
      'reclassified'
    );
  }
  // Closure matters on exact break values: 3 is class 2 left-closed, class 1 right-closed.
  const closureScene = {...scene, values: Float32Array.from([3])};
  closureScene.breaks.set([0, 3, 5.5, 8, 10, 11]);
  expect(reclassifyOnCPU(closureScene, 6, false).classes[0]).toBe(2);
  expect(reclassifyOnCPU(closureScene, 6, true).classes[0]).toBe(1);
  compiled.destroy();
  parameterBuffer.destroy();
  for (const buffer of [
    breaksBuffer,
    classValuesBuffer,
    valuesBuffer,
    validityBuffer,
    classesBuffer,
    reclassifiedBuffer,
    countsBuffer
  ]) {
    buffer.destroy();
  }
});
