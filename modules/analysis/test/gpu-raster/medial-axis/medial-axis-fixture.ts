// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPURasterMedialAxis} from '../../../src/raster-analysis/gpu-raster-skeleton';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  createInputBuffer,
  createOutputBuffer,
  readFloat32,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';

export function createMedialAxisFixture(
  device: Device,
  width: number,
  height: number,
  values: Uint32Array
) {
  const count = width * height;
  const graph = new GPUCommandGraph(device, {id: 'medial-axis-test'});
  const input = createInputBuffer(device, values);
  const medialAxis = createOutputBuffer(device, count + 3);
  const radius = createOutputBuffer(device, count + 3);
  const sentinel = new Uint32Array([77, 77, 77]);
  medialAxis.write(sentinel, count * 4);
  radius.write(sentinel, count * 4);
  graph.add(
    new GPURasterMedialAxis({
      // A sheared geographic grid must still produce distances in pixel units.
      grid: {width, height, affine: [2, 3, 400, 5, -7, 600], pixelInterpretation: 'area'},
      input: importGraphBuffer(graph, 'input', input, 'uint32', count),
      output: {
        medialAxis: importGraphBuffer(graph, 'axis', medialAxis, 'uint32', count + 3),
        radius: importGraphBuffer(graph, 'radius', radius, 'float32', count + 3)
      }
    })
  );
  const compiled = graph.compile();
  return {
    submit: () => submitGraph(device, compiled, undefined),
    update: (nextValues: Uint32Array) => input.write(nextValues),
    read: async () => ({
      medialAxis: (await readUint32(medialAxis, count + 3)).slice(0, count),
      radius: (await readFloat32(radius, count + 3)).slice(0, count),
      axisTail: (await readUint32(medialAxis, count + 3)).slice(count),
      radiusTail: (await readUint32(radius, count + 3)).slice(count)
    }),
    destroy: () => {
      compiled.destroy();
      input.destroy();
      medialAxis.destroy();
      radius.destroy();
    }
  };
}
