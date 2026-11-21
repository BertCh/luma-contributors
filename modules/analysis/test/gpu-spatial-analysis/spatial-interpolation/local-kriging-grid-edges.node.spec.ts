// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {expect, it} from 'vitest';
import {
  getWGSLKernelNodeRecord,
  recordWGSLKernelNodes
} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GPULocalKriging} from '../../../src/gpu-spatial-analysis/spatial-interpolation/gpu-local-kriging';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';

it.each([
  {
    count: 18,
    columns: 3,
    span: 0.625,
    height: 10,
    yOffsets: [0, 3, 3.5, 6.5, 7, 10],
    expectedRows: [0, 0, 1, 1, 2, 2],
    offsets: [0, 0.09375, 0.125, 0.1875, 0.21875, 0.3125, 0.40625, 0.4375, 0.5, 0.53125, 0.625],
    expectedCells: [0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
  },
  // cellWidth = f32(.0625/14) = .004464285913854837. The exact quotient of the
  // representable midpoint .03125 by that width is 6.999999687..., so its cell is 6.
  // The previous writer computed f32(f32(.03125/.0625)*14) = 7 instead.
  {
    count: 392,
    columns: 14,
    span: 0.0625,
    height: 1,
    yOffsets: [0, 0.5, 1],
    expectedRows: [0, 6, 13],
    offsets: [0, 0.03125, 0.0625],
    expectedCells: [0, 6, 13]
  },
  // Diagonal tied samples across column 0/1 and row 1/2 boundaries: query=(2,1.5).
  {
    count: 18,
    columns: 3,
    span: 6,
    height: 3,
    offsets: [1, 3, 0, 6],
    expectedCells: [0, 1, 0, 2],
    yOffsets: [1, 2, 0, 3],
    expectedRows: [1, 2, 0, 2]
  }
])('local kriging encodes the reader cells at translated representable edges ($columns columns)', ({
  count,
  columns,
  span,
  offsets,
  expectedCells,
  height,
  yOffsets,
  expectedRows
}) => {
  const graph = new GPUCommandGraph(createNullWebGPUDevice());
  recordWGSLKernelNodes(graph);
  // Derived lattice: ceil(sqrt(count / 2)) cells per axis.
  const positions = createTransientView(graph, 'positions', 'float32x2', count);
  const observations = createTransientView(graph, 'observations', 'float32x4', count);
  const contributor = new GPULocalKriging({
    positions,
    observations,
    supports: createTransientView(graph, 'supports', 'float32x2', 1),
    sampleIndex: 'grid',
    neighborCount: 2,
    mode: 'ordinary',
    covariance: {model: 'exponential', sill: 1, range: 2},
    output: {
      values: createTransientView(graph, 'values', 'float32x4', 1),
      status: createTransientView(graph, 'status', 'uint32', 1)
    }
  });
  const node = contributor
    .getCommandNodes(graph)
    .find(node => node.id === 'local-kriging-index-prepare')!;
  const record = getWGSLKernelNodeRecord(graph, node.id)!;
  const parameters = record.bindings.find(binding => binding.name === 'parameters')!.view;
  const units = record.bindings.find(binding => binding.name === 'gridPositions')!.view;
  const memory = createCPUGraphMemory();
  // Exact binary32 inputs at and adjacent to cell edges, at a projected origin.
  const points = Array.from({length: count}, (_, row) => offsets[row % offsets.length]);
  memory.write(
    positions,
    Float32Array.from(
      points.flatMap((x, row) => [500000 + x, 5000000 + yOffsets[row % yOffsets.length]])
    )
  );
  memory.write(observations, Float32Array.from(points.flatMap(() => [1, 0, 0, 1])));
  memory.write(
    parameters,
    Float32Array.from([500000, 5000000, 500000 + span, 5000000 + height, 0, 0, 0, 0, 0, 0, 0, 0])
  );
  executeCommandNodesOnCPU([node], memory);
  const result = Array.from(memory.readFloat32(units, count * 2));
  expect(result.every(Number.isFinite)).toBe(true);
  // Encoding the actual cell centre leaves a half-cell margin for the published GPUGridIndex.
  expect(
    points.map((_, row) =>
      Math.min(columns - 1, Math.floor(Math.fround(result[row * 2] * columns)))
    )
  ).toEqual(points.map((_, row) => expectedCells[row % offsets.length]));
  expect(
    points.map((_, row) =>
      Math.min(columns - 1, Math.floor(Math.fround(result[row * 2 + 1] * columns)))
    )
  ).toEqual(points.map((_, row) => expectedRows[row % yOffsets.length]));
});
