// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import type {Buffer, Device} from '@luma.gl/core';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {createInputBuffer, createOutputBuffer} from '../../utils/gpu-contributor-test-utils';

type Format = 'float32x3' | 'float32x4' | 'uint32' | 'uint32x2';
const components = (format: Format) =>
  format.endsWith('x3') ? 3 : format.endsWith('x4') ? 4 : format.endsWith('x2') ? 2 : 1;

export function createGlobalCoordinateFixture(device: Device) {
  const graph = new GPUCommandGraph(device);
  const buffers: Buffer[] = [];
  function input<DataFormat extends Format>(format: DataFormat, values: number[]) {
    const buffer = createInputBuffer(
      device,
      format.startsWith('float') ? new Float32Array(values) : new Uint32Array(values)
    );
    buffers.push(buffer);
    return importGraphBuffer(
      graph,
      `input-${buffers.length}`,
      buffer,
      format,
      values.length / components(format)
    );
  }
  function output<DataFormat extends Format>(format: DataFormat, rows: number) {
    const buffer = createOutputBuffer(device, rows * components(format));
    buffers.push(buffer);
    return {
      buffer,
      view: importGraphBuffer(graph, `output-${buffers.length}`, buffer, format, rows)
    };
  }
  return {graph, input, output, destroy: () => buffers.forEach(buffer => buffer.destroy())};
}
