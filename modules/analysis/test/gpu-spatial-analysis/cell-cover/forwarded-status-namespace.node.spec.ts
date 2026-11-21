// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {GPUCellCover} from '../../../src/gpu-spatial-analysis/cell-cover/gpu-cell-cover';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('cell cover private counters do not reserve caller status names', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let serial = 0;
    const view = <Format extends 'uint32' | 'uint32x2' | 'float32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `namespace-${serial++}`, format, length);
    const cover = new GPUCellCover({
      id: 'cover',
      family: 'quadbin',
      resolution: 1,
      candidateCapacity: 16,
      positions: view('float32x2', 4),
      featureOffsets: view('uint32', 2),
      polygonOffsets: view('uint32', 2),
      ringOffsets: view('uint32', 2),
      output: {
        cells: view('uint32x2', 4),
        featureIds: view('uint32', 4),
        status: {
          count: view('uint32', 1),
          overflow: view('uint32', 1),
          candidateOverflow: view('uint32', 1),
          invalidCount: createTransientView(graph, 'cover-invalid-count', 'uint32', 1),
          uncertainCount: createTransientView(graph, 'cover-uncertain-count', 'uint32', 1)
        }
      }
    });
    expect(() => cover.getCommandNodes(graph)).not.toThrow();
  } finally {
    device.destroy();
  }
});
