// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  createWGSLKernelNode,
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {getStorageTopKNodeRecord} from '../../../src/analysis-core/kernels/storage-backed-top-k';
import {GPUInverseDistanceWeighting} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-inverse-distance-weighting';
import {GPUKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-kriging';
import {getGPUInverseDistanceWeightingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/spatial-interpolation-parameters';
import {getGPUKrigingParameterValues} from '../../../src/gpu-spatial-analysis/spatial-interpolation/kriging-parameters';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {makeExtentGrid} from './extent-grid';

function runIndexedSolve(
  kind: 'idw' | 'kriging',
  query: number,
  longitudes: number[],
  observations: number[],
  radius: number
) {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  recordWGSLKernelNodes(graph);
  const positions = createTransientView(graph, 'positions', 'float32x2', longitudes.length);
  const values = createTransientView(graph, 'observations', 'float32', longitudes.length);
  const parameters = createTransientView(graph, 'parameters', 'float32', 12);
  const result = createTransientView(graph, 'result', 'float32', 1);
  const variance = createTransientView(graph, 'variance', 'float32', 1);
  const counts = createTransientView(graph, 'counts', 'uint32', 1);
  // Symmetric representable endpoints preserve the query exactly after production f32 placement.
  const halfWidth = Math.abs(query) < 180 ? 2 ** -16 : 0.5;
  const grid = makeExtentGrid(1, 1, [query - halfWidth, -0.5, query + halfWidth, 0.5]);
  const props = {
    positions,
    values,
    parameters,
    grid,
    indexGridSize: [12, 1] as const,
    indexBounds: [-180, -1, 180, 1] as const,
    maximumNeighborCount: 4,
    spatialContext: {coordinateSpace: 'longitude-latitude', metric: 'great-circle'} as const,
    longitudeSearch: 'wrapped' as const,
    output: {values: result, variance, counts}
  };
  const contributor =
    kind === 'idw' ? new GPUInverseDistanceWeighting(props) : new GPUKriging(props);
  const node = contributor
    .getCommandNodes(graph)
    .find(node =>
      node.id.startsWith(kind === 'idw' ? 'inverse-distance-weighting-gather' : 'kriging-solve')
    )!;
  const recorded = getWGSLKernelNodeRecord(graph, node.id);
  const topK = getStorageTopKNodeRecord(node);
  const record = recorded ?? topK!;
  // Execute the production declarations/body with a direct one-row dispatch. The host executor
  // does not bind uniform structs; only the runtime dispatch uniform is replaced by its literals.
  let declarations: string;
  let body: string;
  if (recorded) {
    declarations = recorded.declarations!;
    body = recorded.body.replace('getChunkParameter(0u)', '0u');
  } else {
    const source = topK!.source;
    declarations = source.slice(
      source.indexOf('\n', source.indexOf('var<uniform>')) + 1,
      source.indexOf('@compute')
    );
    body = source.slice(
      source.indexOf('\n', source.indexOf('let batchIndex =')) + 1,
      source.lastIndexOf('}')
    );
    const replaceSettings = (text: string) =>
      text.replace(
        /selectionSettings\[(\d+)\]\.([xyzw])/g,
        (_, vector: string, component: string) =>
          `${topK!.settings[Number(vector) * 4 + 'xyzw'.indexOf(component)]}u`
      );
    declarations = replaceSettings(declarations);
    body = replaceSettings(body);
  }
  const probe = createWGSLKernelNode(graph, {
    id: 'direct-solve',
    operation: 'WrappedSolveProbe',
    invocationCount: 1,
    bindings: [...record.bindings],
    declarations,
    body
  });
  const binding = (name: string) => record.bindings.find(binding => binding.name === name)!.view;
  const memory = createCPUGraphMemory();
  memory.write(positions, Float32Array.from(longitudes.flatMap(longitude => [longitude, 0])));
  memory.write(values, Float32Array.from(observations));
  memory.write(
    parameters,
    kind === 'idw'
      ? getGPUInverseDistanceWeightingParameterValues(
          {searchRadius: radius, power: 2, neighborCount: 0},
          grid
        )
      : getGPUKrigingParameterValues(
          {
            searchRadius: radius,
            neighborCount: 4,
            variogram: {model: 'exponential', nugget: 0, sill: 1, range: 10000}
          },
          grid
        )
  );
  // Hand-built world index: 12 half-open 30-degree columns; +180 belongs to the last column.
  const cells = Array.from({length: 12}, () => [] as number[]);
  longitudes.forEach((longitude, row) =>
    cells[Math.min(11, Math.floor((longitude + 180) / 30))].push(row)
  );
  const offsets = [0];
  cells.forEach(cell => offsets.push(offsets[offsets.length - 1] + cell.length));
  memory.write(binding('cellOffsets'), Uint32Array.from(offsets));
  memory.write(binding('sortedIds'), Uint32Array.from(cells.flat()));
  executeCommandNodesOnCPU([probe], memory);
  const output = {
    values: Array.from(memory.readFloat32(result)),
    variance: Array.from(memory.readFloat32(variance)),
    counts: Array.from(memory.readUint32(counts))
  };
  device.destroy();
  return output;
}

it('R1 B1 includes either seam endpoint with production outer column bounds', () => {
  for (const sign of [-1, 1]) {
    const result = runIndexedSolve('idw', sign * 179.99998474121094, [-sign * 180], [7], 2);
    // Equatorial gap is 2^-16 degrees: R*pi/180*2^-16 = 1.696702274071242 m < 2 m.
    expect(result.counts).toEqual([1]);
    expect(result.values).toHaveLength(1);
    expect(Number.isFinite(result.values[0])).toBe(true);
    expect(result.values[0]).toBe(7);
  }
});

it('R1 B2 finds the exact-degree site among zero-distance ties in either row order', () => {
  for (const sign of [-1, 1])
    for (const reverse of [false, true]) {
      const longitudes = [-sign * 180, sign * 180];
      const observations = [7, 11];
      if (reverse) {
        longitudes.reverse();
        observations.reverse();
      }
      const result = runIndexedSolve('kriging', sign * 180, longitudes, observations, 2000);
      // The exact-degree site has value 11; an exact observation has zero kriging variance.
      expect(result.values).toHaveLength(1);
      expect(result.variance).toHaveLength(1);
      expect(Number.isFinite(result.values[0])).toBe(true);
      expect(Number.isFinite(result.variance[0])).toBe(true);
      expect(result.values[0]).toBe(11);
      expect(result.variance[0]).toBe(0);
    }
});

it('R1 S1 wraps queries translated by multiple full turns with production outer bounds', () => {
  for (const longitude of [-720, 720, 1080]) {
    const result = runIndexedSolve('idw', longitude, [0], [7], 2000);
    // Whole turns identify the same spherical location.
    expect(result.counts).toEqual([1]);
    expect(result.values).toEqual([7]);
  }
});
