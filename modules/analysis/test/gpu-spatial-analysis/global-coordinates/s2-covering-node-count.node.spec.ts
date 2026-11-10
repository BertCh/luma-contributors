// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors
import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {GPUS2CapCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-cap-covering';
import {GPUS2PolygonCovering} from '../../../src/gpu-spatial-analysis/cell-topology/gpu-s2-polygon-covering';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it('the level-synchronous polygon covering reports the exact number of command nodes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let sequence = 0;
    const view = <Format extends 'float32x3' | 'uint32' | 'uint32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `s2-node-count-${sequence++}`, format, length);
    // Frontier sizes on both sides of the 256 and 65,536 row scan block boundaries.
    for (const maximumLevel of [3, 15])
      for (const maximumVisitedCells of [200, 1000, 70000, 1048576]) {
        const covering = new GPUS2PolygonCovering({
          id: `polygon-${sequence++}`,
          positions: view('float32x3', 3),
          ringOffsets: view('uint32', 2),
          maximumLevel,
          maximumVisitedCells,
          refinement: 'level-synchronous',
          capacity: 16,
          output: {
            cells: view('uint32x2', 16),
            counts: view('uint32x2', 1),
            status: view('uint32', 1)
          }
        });
        expect(
          covering.getCommandNodes(graph).length,
          `${maximumLevel} ${maximumVisitedCells}`
        ).toBe(covering.estimatedCommandNodeCount);
      }
    const serial = new GPUS2PolygonCovering({
      positions: view('float32x3', 3),
      ringOffsets: view('uint32', 2),
      maximumLevel: 3,
      capacity: 16,
      output: {cells: view('uint32x2', 16), counts: view('uint32x2', 1), status: view('uint32', 1)}
    });
    expect(serial.getCommandNodes(graph).length).toBe(serial.estimatedCommandNodeCount);
  } finally {
    device.destroy();
  }
});

it('the level-synchronous cap covering reports the exact number of command nodes', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    let sequence = 0;
    const view = <Format extends 'float32x4' | 'uint32' | 'uint32x2'>(
      format: Format,
      length: number
    ) => createTransientView(graph, `s2-cap-node-count-${sequence++}`, format, length);
    for (const level of [2, 15])
      for (const [capCount, capacity] of [
        [1, 4],
        [3, 40],
        [200, 90]
      ] as const) {
        const covering = new GPUS2CapCovering({
          id: `cap-${sequence++}`,
          caps: view('float32x4', capCount),
          level,
          capacity,
          refinement: 'level-synchronous',
          output: {
            cells: view('uint32x2', capCount * capacity),
            counts: view('uint32x2', capCount),
            status: view('uint32', capCount)
          }
        });
        expect(covering.getCommandNodes(graph).length, `${level} ${capCount}`).toBe(
          covering.estimatedCommandNodeCount
        );
      }
  } finally {
    device.destroy();
  }
});
