// luma.gl
// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import type {Device} from '@luma.gl/core';
import {GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {getWebGPUTestDevice} from '@luma.gl/core/test-utils';
import {expect, it} from 'vitest';
import {makeGPURasterMetadata} from '@luma-contributors/analysis/raster-analysis';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {
  GPU_TERRAIN_SPIKE_REPAIR_STATISTICS as S,
  GPUTerrainSpikeRepair
} from '../../../src/gpu-terrain/terrain-decode/gpu-terrain-spike-repair';
import {
  createInputBuffer,
  createOutputBuffer,
  readUint32,
  submitGraph
} from '../../utils/gpu-contributor-test-utils';
import {requireTestDevice} from '../../utils/require-test-device';

// The component labels of GPUTerrainSpikeRepair are defined without reference to any labelling
// algorithm: the label of a valid pixel is the lowest pixel index of its 4-connected component
// under the edge rule |dh| <= jump. The expected labels here come from a union-find over that
// definition, so they pin whichever GPU labelling the contributor composes.

async function runLabels(
  device: Device,
  values: Float32Array,
  width: number,
  height: number,
  maximumComponentIterations?: number
): Promise<{labels: number[]; converged: number}> {
  const pixelCount = width * height;
  const inputValues = createInputBuffer(device, values);
  const outputs = {
    values: createOutputBuffer(device, pixelCount),
    validity: createOutputBuffer(device, pixelCount),
    labels: createOutputBuffer(device, pixelCount),
    statistics: createOutputBuffer(device, 5)
  };
  const graph = new GPUCommandGraph(device, {id: 'spike-repair-labels-test'});
  const contributor = new GPUTerrainSpikeRepair({
    grid: makeGPURasterMetadata({width, height}),
    maximumComponentIterations,
    elevation: {
      id: 'elevation',
      format: 'float32',
      storage: {
        kind: 'buffer',
        values: importGraphBuffer(graph, 'elevation', inputValues, 'float32', pixelCount)
      }
    },
    output: {
      values: importGraphBuffer(graph, 'values', outputs.values, 'float32', pixelCount),
      outputValidity: importGraphBuffer(graph, 'validity', outputs.validity, 'uint32', pixelCount),
      labels: importGraphBuffer(graph, 'labels', outputs.labels, 'uint32', pixelCount),
      statistics: importGraphBuffer(graph, 'statistics', outputs.statistics, 'uint32', 5)
    }
  });
  graph.add(contributor);
  const compiled = graph.compile();
  submitGraph(device, compiled, undefined);
  const result = {
    labels: await readUint32(outputs.labels, pixelCount),
    converged: (await readUint32(outputs.statistics, 5))[S.converged]
  };
  compiled.destroy();
  contributor.destroy();
  for (const buffer of [inputValues, ...Object.values(outputs)]) {
    buffer.destroy();
  }
  return result;
}

/** Lowest member index per 4-connected component under `|dh| <= jump`, by union-find. */
function getExpectedLabels(
  values: Float32Array,
  width: number,
  height: number,
  jump: number
): number[] {
  const parent = Int32Array.from({length: width * height}, (_, index) => index);
  const find = (index: number): number => {
    let root = index;
    while (parent[root] !== root) {
      root = parent[root];
    }
    while (parent[index] !== root) {
      const next = parent[index];
      parent[index] = root;
      index = next;
    }
    return root;
  };
  const union = (first: number, second: number): void => {
    const firstRoot = find(first);
    const secondRoot = find(second);
    // The lower index becomes the root, so a root is the lowest member of its component.
    if (firstRoot < secondRoot) {
      parent[secondRoot] = firstRoot;
    } else if (secondRoot < firstRoot) {
      parent[firstRoot] = secondRoot;
    }
  };
  for (let row = 0; row < height; row++) {
    for (let column = 0; column < width; column++) {
      const pixel = row * width + column;
      if (column + 1 < width && Math.abs(values[pixel] - values[pixel + 1]) <= jump) {
        union(pixel, pixel + 1);
      }
      if (row + 1 < height && Math.abs(values[pixel] - values[pixel + width]) <= jump) {
        union(pixel, pixel + width);
      }
    }
  }
  return Array.from(parent, (_, index) => find(index));
}

/**
 * A serpentine corridor: rows alternate between two levels 1000 m apart except for one linking
 * column per row pair, so each level is one long winding component. Label propagation without
 * pointer jumping would need about `width * height / 2` rounds here.
 */
function createSerpentine(width: number, height: number): Float32Array {
  return Float32Array.from({length: width * height}, (_, index) => {
    const column = index % width;
    const row = Math.floor(index / width);
    const band = Math.floor(row / 2);
    // Odd rows are walls 1000 m up, except one gap that joins the two corridor rows around it.
    const isWall = row % 2 === 1 && column !== (band % 2 === 0 ? width - 1 : 0);
    return 500 + (isWall ? 1000 : 0) + 0.25 * ((column + row) % 3);
  });
}

it('GPUTerrainSpikeRepair labels are the lowest pixel index of each component', async () => {
  const device = await getWebGPUTestDevice();
  requireTestDevice(device);
  const cases: {name: string; width: number; height: number; values: Float32Array}[] = [
    {
      name: 'one smooth component',
      width: 256,
      height: 256,
      values: Float32Array.from(
        {length: 256 * 256},
        (_, index) => 800 + 2 * (index % 256) + 3 * Math.floor(index / 256)
      )
    },
    {name: 'serpentine corridors', width: 96, height: 64, values: createSerpentine(96, 64)}
  ];
  for (const {name, width, height, values} of cases) {
    // The default jump threshold of the contributor is 200 m.
    const expected = getExpectedLabels(values, width, height, 200);
    const result = await runLabels(device, values, width, height);
    expect(result.converged, name).toBe(1);
    expect(result.labels, name).toEqual(expected);
  }
});
