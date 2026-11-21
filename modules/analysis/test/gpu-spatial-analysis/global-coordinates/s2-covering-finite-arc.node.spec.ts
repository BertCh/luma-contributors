// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {expect, it} from 'vitest';
import {GPUCommandGraph, createTransientView} from '@luma.gl/graph/gpu-core';
import {
  computeS2PolygonCoveringOnCPU,
  CPUS2PolygonCoveringOperation
} from '../../../src/gpu-spatial-analysis/cell-topology/cpu-s2-polygon-covering-operation';
import {getS2CoveringParallelWGSL} from '../../../src/gpu-spatial-analysis/cell-topology/s2-covering-parallel-wgsl';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

function createBoundaryTriangle() {
  const longitude = Math.PI / 4;
  const halfWidth = 0.75e-6;
  return [
    [Math.cos(longitude - halfWidth), Math.sin(longitude - halfWidth), 0],
    [Math.cos(longitude + halfWidth), Math.sin(longitude + halfWidth), 0],
    [
      Math.cos(Math.PI / 18) * Math.cos(longitude),
      Math.cos(Math.PI / 18) * Math.sin(longitude),
      Math.sin(Math.PI / 18)
    ]
  ];
}

it.each([
  'serial',
  'level-synchronous'
] as const)('CPU %s narrow face-boundary cover excludes antipodal faces', refinement => {
  for (const reversed of [false, true]) {
    const vertices = createBoundaryTriangle();
    if (reversed) vertices.reverse();
    const result = computeS2PolygonCoveringOnCPU(
      {
        positions: Float32Array.from(vertices.flat()),
        ringOffsets: Uint32Array.of(0, 3)
      },
      {maximumLevel: 0, maximumVisitedCells: 6, capacity: 6, refinement, mode: 'exterior'}
    );
    // x,y>z>=0 throughout this triangle, which straddles x=y: precisely faces 0 and 1.
    expect(Array.from(result.status)).toEqual([0]);
    expect.soft(Array.from(result.counts), `reversed=${reversed}`).toEqual([2, 2]);
    expect
      .soft(Array.from(result.cells), `reversed=${reversed}`)
      .toEqual([0, 0x10000000, 0, 0x30000000, 0, 0, 0, 0, 0, 0, 0, 0]);
  }
});

it('production S2 WGSL finite-arc helper excludes the antipodal midpoint of a short edge', () => {
  const device = createNullWebGPUDevice();
  try {
    const graph = new GPUCommandGraph(device);
    const positions = createTransientView(graph, 'positions', 'float32x3', 3);
    const output = createTransientView(graph, 'output', 'uint32', 2);
    const memory = createCPUGraphMemory();
    memory.write(positions, Float32Array.from(createBoundaryTriangle().flat()));
    const node = createWGSLKernelNode(graph, {
      id: 'finite-arc',
      operation: 'S2FiniteArcOracle',
      invocationCount: 2,
      bindings: [
        {name: 'positions', view: positions, type: 'f32', access: 'read'},
        {name: 'output', view: output, type: 'u32', access: 'read_write'}
      ],
      declarations: getS2CoveringParallelWGSL(1e-6),
      body: `let first=coverPosition(0u);let second=coverPosition(1u);
let middle=normalize(first+second);let point=select(middle,-middle,index==1u);
output[outputOffset+index]=select(0u,1u,coverOnArc(first,second,point));`
    });
    executeCommandNodesOnCPU([node], memory);
    // The positive midpoint lies on the finite arc; its antipode is pi radians away.
    expect(Array.from(memory.readUint32(output))).toEqual([1, 0]);
  } finally {
    device.destroy();
  }
});

it.each([
  'serial',
  'level-synchronous'
] as const)('named CPU %s excludes the short-edge antipode', async refinement => {
  const positions = Float32Array.from(createBoundaryTriangle().flat());
  const ringOffsets = Uint32Array.of(0, 3);
  const operation = new CPUS2PolygonCoveringOperation({
    positions: 'positions',
    ringOffsets: 'ringOffsets',
    positionCount: 3,
    ringOffsetCount: 2,
    maximumLevel: 0,
    maximumVisitedCells: 6,
    capacity: 6,
    refinement,
    mode: 'exterior',
    output: {cells: 'cells', counts: 'counts', status: 'status'}
  });
  const result = await operation.execute({
    inputs: {
      positions: {
        format: 'float32x3',
        length: 3,
        chunks: [{values: positions, length: 3}],
        pack: () => positions
      },
      ringOffsets: {
        format: 'uint32',
        length: 2,
        chunks: [{values: ringOffsets, length: 2}],
        pack: () => ringOffsets
      }
    }
  });
  // x,y>z>=0 throughout the triangle: only face roots 0 and 1 can intersect it.
  expect(result['counts']).toEqual(Uint32Array.of(2, 2));
  expect(result['status']).toEqual(Uint32Array.of(0));
  expect(result['cells']).toEqual(
    Uint32Array.of(0, 0x10000000, 0, 0x30000000, 0, 0, 0, 0, 0, 0, 0, 0)
  );
});
