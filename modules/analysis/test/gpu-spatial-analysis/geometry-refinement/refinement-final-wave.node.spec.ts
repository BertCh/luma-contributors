// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GPUGraphVectorFormat
} from '@luma.gl/graph/gpu-core';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {GRID_HALF_UP_ROUNDING_WGSL} from '../../../src/gpu-spatial-analysis/geometry-edit/half-up-rounding';
import {usePrecisionExecutorHooks} from './precision-executor-hooks';
import {expect, it} from 'vitest';
import {computeMaximumInscribedCircleOnCPU} from '../../../src/gpu-spatial-analysis/geometry-refinement/cpu-maximum-inscribed-circle-operation';
import {GPUMaximumInscribedCircle} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-maximum-inscribed-circle';
import {GPUNormalizePolygons} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-normalize-polygons';
import {GPUSegmentSnap} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-segment-snap';
import {
  LINE_PRECISION_COORDINATE_WGSL,
  POLYGON_PRECISION_COORDINATE_WGSL
} from '../../../src/gpu-spatial-analysis/geometry-refinement/gpu-precision-reduction';
import {importGraphBuffer} from '../../../src/analysis-core/kernels/gpu-contributor-utils';
import {Buffer} from '@luma.gl/core';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';

useCPUExecutorTestHooks();
usePrecisionExecutorHooks();

it('UGE-05 publishes the radius and relative witness at the stored circle center', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let serial = 0;
  const view = <Format extends GPUGraphVectorFormat>(format: Format, length: number) =>
    createTransientView(graph, `final-${serial++}`, format, length);
  const positions = view('float32x2', 8);
  const ringOffsets = view('uint32', 3);
  const polygonOffsets = view('uint32', 3);
  const featureOffsets = view('uint32', 3);
  const output = {
    circles: view('float32x4', 2),
    witnesses: view('float32x2', 2),
    status: view('uint32', 2),
    publishedCenterClearance: view('float32', 2),
    precisionLimited: view('uint32', 2),
    witnessOffsets: view('float32x2', 2)
  };
  const nodes = new GPUMaximumInscribedCircle({
    positions,
    ringOffsets,
    polygonOffsets,
    featureOffsets,
    tolerance: 0.001,
    output
  }).getCommandNodes(graph);
  memory.write(
    positions,
    Float32Array.of(
      1048576,
      1048576,
      1048576.125,
      1048576,
      1048576.125,
      1048576.125,
      1048576,
      1048576.125,
      0,
      0,
      0.125,
      0,
      0.125,
      0.125,
      0,
      0.125
    )
  );
  for (const offsets of [ringOffsets, polygonOffsets, featureOffsets])
    memory.write(
      offsets,
      offsets === ringOffsets ? Uint32Array.of(0, 4, 8) : Uint32Array.of(0, 1, 2)
    );
  // Side .125 squares: the absolute midpoint rounds to a corner, whereas the local
  // midpoint is representable. Search error is independent of this publication loss.
  memory.write(
    output.circles,
    Float32Array.of(1048576, 1048576, 0.0625, 0.02, 0.0625, 0.0625, 0.0625, 0.02)
  );
  memory.write(output.status, Uint32Array.of(0, 0));
  try {
    const publication = nodes.filter(node => node.id.includes('-published-'));
    expect(publication.length).toBeGreaterThan(0);
    executeCommandNodesOnCPU(publication, memory);
    expect(Array.from(memory.readFloat32(output.circles))).toEqual(
      Array.from(Float32Array.of(1048576, 1048576, 0, 0.02, 0.0625, 0.0625, 0.0625, 0.02))
    );
    expect(Array.from(memory.readFloat32(output.publishedCenterClearance))).toEqual([0, 0.0625]);
    expect(Array.from(memory.readUint32(output.precisionLimited))).toEqual([1, 0]);
    expect(Array.from(memory.readFloat32(output.witnessOffsets))).toEqual([0, 0, -0.0625, 0]);
    expect(Array.from(memory.readFloat32(output.witnesses))).toEqual([1048576, 1048576, 0, 0.0625]);
    expect(Array.from(memory.readUint32(output.status))).toEqual([0, 0]);
  } finally {
    device.destroy();
  }
});

it('UGE-03 certifies original precision operands at decimal and large integral cells', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const input = createTransientView(graph, 'round-input', 'float32x2', 5);
  const output = createTransientView(graph, 'round-output', 'float32x4', 5);
  memory.write(
    input,
    Float32Array.of(20.5, 41, -20.5, 41, 58720228, 7, 1, 2, 0.35, Math.fround(0.1))
  );
  const node = createWGSLKernelNode(graph, {
    id: 'round',
    operation: 'GPULinePrecisionReduction',
    invocationCount: 5,
    bindings: [
      {name: 'input', view: input, type: 'f32', access: 'read'},
      {name: 'output', view: output, type: 'f32', access: 'read_write'}
    ],
    declarations: GRID_HALF_UP_ROUNDING_WGSL,
    body: `let value=input[inputOffset+index*2u]; let spacing=input[inputOffset+index*2u+1u]; output[outputOffset+index*4u]=roundGridHalfUp(value,0.0,spacing);`
  });
  try {
    executeCommandNodesOnCPU([node], memory);
    const values = memory.readFloat32(output);

    expect([values[0], values[4], values[8], values[12], values[16]]).toEqual([
      1, 0, 8388604, 1, 3
    ]);
  } finally {
    device.destroy();
  }
});

it('W5-06 independent CPU bounds pin square optimum and budget-limited triangle', () => {
  const base = {polygonOffsets: Uint32Array.of(0, 1), featureOffsets: Uint32Array.of(0, 1)};
  // A 4-by-4 square attains the half-side bound at (12,22), so no error remains.
  const square = computeMaximumInscribedCircleOnCPU(
    {
      ...base,
      positions: Float32Array.of(10, 20, 14, 20, 14, 24, 10, 24),
      ringOffsets: Uint32Array.of(0, 4)
    },
    3,
    {maximumIterations: 8, cellCapacity: 4}
  );
  expect(Array.from(square.circles)).toEqual([12, 22, 2, 0]);
  expect(Array.from(square.witnesses)).toEqual([10, 22]);
  expect(Array.from(square.status)).toEqual([0]);
  // Root lies on the 8-by-6 right triangle's hypotenuse; the envelope caps radius at 3.
  const triangle = computeMaximumInscribedCircleOnCPU(
    {...base, positions: Float32Array.of(0, 0, 8, 0, 0, 6), ringOffsets: Uint32Array.of(0, 3)},
    0.1,
    {maximumIterations: 0, cellCapacity: 1}
  );
  expect(Array.from(triangle.circles)).toEqual([0, 6, 0, 3]);
  expect(Array.from(triangle.witnesses)).toEqual([0, 6]);
  expect(Array.from(triangle.status)).toEqual([1]);
});

/**
 * Emulate a GPU compiler that lowers `a / b` to `a * (1 / b)` (Apple Metal does), so the
 * interpreter reproduces reciprocal quotients. Every division in the helper text is rewritten.
 */
function useReciprocalDivision(wgsl: string): string {
  return wgsl
    .replaceAll('(value - origin) / spacing', '(value - origin) * (1.0 / spacing)')
    .replaceAll('value / step', 'value * (1.0 / step)');
}

function executeCoordinateKernel(
  declarations: string,
  call: string,
  rows: readonly (readonly [number, number])[]
): number[] {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const input = createTransientView(graph, 'coordinate-input', 'float32x2', rows.length);
  const output = createTransientView(graph, 'coordinate-output', 'float32', rows.length);
  memory.write(input, Float32Array.from(rows.flat()));
  const node = createWGSLKernelNode(graph, {
    id: 'coordinate',
    operation: 'GPUPolygonPrecisionReduction',
    invocationCount: rows.length,
    bindings: [
      {name: 'input', view: input, type: 'f32', access: 'read'},
      {name: 'output', view: output, type: 'f32', access: 'read_write'}
    ],
    declarations,
    body: `let value = input[inputOffset + index * 2u]; let step = input[inputOffset + index * 2u + 1u]; output[outputOffset + index] = ${call};`
  });
  try {
    executeCommandNodesOnCPU([node], memory);
    return Array.from(memory.readFloat32(output));
  } finally {
    device.destroy();
  }
}

it('round 1 B4 polygon quotient never steps past an unrepresentable half boundary', () => {
  // Hand-derived: 78293712 = 7 * 11184816 and 78293600 = 7 * 11184800 are exact grid points;
  // float32(78293712 * float32(1/7)) = 11184817 (NumPy probe). The half boundary 11184816.5 is
  // not a float32 (ulp 1 at 2^23), so the certified integer itself is the snap input.
  // Negative mirror and a decimal half control (20.5 / 41 -> cell 1 just below 1.5) as well.
  const values = executeCoordinateKernel(
    useReciprocalDivision(POLYGON_PRECISION_COORDINATE_WGSL),
    'normalizePrecisionCoordinate(value, step)',
    [
      [78293712, 7],
      [-78293712, 7],
      [78293600, 7],
      [20.5, 41]
    ]
  );
  expect(values.every(Number.isFinite)).toBe(true);
  expect(values.slice(0, 3)).toEqual([11184816, -11184816, 11184800]);
  // 20.5 / 41 = 0.5 exactly: cell 1. The reciprocal quotient 0.49999997 moves to the float32
  // successor of 0.5, 0.5 + 2^-24, just inside cell 1.
  expect(values[3]).toBe(0.5000000596046448);
});

it('round 1 B5 line rounding preserves exact grid points beyond 2^24 cells', () => {
  // 234880912 = 7 * 33554416 (2^25 - 16); float32(234880912 * float32(1/7)) = 33554418.
  // Controls: 58720228 = 7 * 8388604 below 2^24 cells, and 2^25 on grid 2 (power-of-two step).
  const values = executeCoordinateKernel(
    useReciprocalDivision(LINE_PRECISION_COORDINATE_WGSL),
    'roundLineGridCoordinate(value, step)',
    [
      [234880912, 7],
      [-234880912, 7],
      [58720228, 7],
      [33554432, 2]
    ]
  );
  expect(values).toEqual([234880912, -234880912, 58720228, 33554432]);
});

it('round 1 B7 raw circle writes a getter-backed publication destination', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  let serial = 0;
  const view = <Format extends GPUGraphVectorFormat>(format: Format, length: number) =>
    createTransientView(graph, `getter-${serial++}`, format, length);
  const precisionLimited = view('uint32', 1);
  class Output {
    circles = view('float32x4', 1);
    witnesses = view('float32x2', 1);
    status = view('uint32', 1);
    get precisionLimited() {
      return precisionLimited;
    }
  }
  const positions = view('float32x2', 0);
  const ringOffsets = view('uint32', 1);
  const polygonOffsets = view('uint32', 2);
  const featureOffsets = view('uint32', 2);
  const output = new Output();
  const nodes = new GPUMaximumInscribedCircle({
    positions,
    ringOffsets,
    polygonOffsets,
    featureOffsets,
    tolerance: 0.1,
    output
  }).getCommandNodes(graph);
  memory.write(precisionLimited, Uint32Array.of(123));
  memory.write(output.status, Uint32Array.of(123));
  try {
    executeCommandNodesOnCPU(nodes, memory);
    // One declared empty feature: status EMPTY (2) and a cleared precision word.
    expect(Array.from(memory.readUint32(output.status))).toEqual([2]);
    expect(Array.from(memory.readUint32(precisionLimited))).toEqual([0]);
  } finally {
    device.destroy();
  }
});

it('round 1 B11 empty routes bind and validate only the views they write', () => {
  const device = createNullWebGPUDevice();
  const graph = new GPUCommandGraph(device);
  let serial = 0;
  // A zero-row view over four bytes: smaller than one float32x2/float32x4/uint32x4 row.
  const minimal = <Format extends GPUGraphVectorFormat>(format: Format) =>
    importGraphBuffer(
      graph,
      `minimal-${serial++}`,
      device.createBuffer({byteLength: 4, usage: Buffer.STORAGE | Buffer.COPY_DST}),
      format,
      0
    );
  const offsets = (length: number) =>
    createTransientView(graph, `offsets-${serial++}`, 'uint32', length);
  try {
    // Zero features: every circle output is a minimally backed zero-row view.
    expect(
      new GPUMaximumInscribedCircle({
        positions: minimal('float32x2'),
        ringOffsets: offsets(1),
        polygonOffsets: offsets(1),
        featureOffsets: offsets(1),
        tolerance: 0.1,
        output: {
          circles: minimal('float32x4'),
          witnesses: minimal('float32x2'),
          status: minimal('uint32'),
          witnessOffsets: minimal('float32x2')
        }
      }).getCommandNodes(graph)
    ).toEqual([]);
    // One declared empty feature: only the positions view is minimal.
    expect(
      new GPUMaximumInscribedCircle({
        positions: minimal('float32x2'),
        ringOffsets: offsets(1),
        polygonOffsets: offsets(2),
        featureOffsets: offsets(2),
        tolerance: 0.1,
        output: {
          circles: createTransientView(graph, 'circle-row', 'float32x4', 1),
          witnesses: createTransientView(graph, 'witness-row', 'float32x2', 1),
          status: createTransientView(graph, 'status-row', 'uint32', 1)
        }
      })
        .getCommandNodes(graph)
        .map(node => node.id)
    ).toEqual([
      'maximum-inscribed-circle-empty-circles',
      'maximum-inscribed-circle-empty-witnesses',
      'maximum-inscribed-circle-empty-status'
    ]);
    expect(
      new GPUNormalizePolygons({
        positions: minimal('float32x2'),
        ringOffsets: offsets(2),
        polygonOffsets: offsets(2),
        featureOffsets: offsets(2),
        output: {
          positions: minimal('float32x2'),
          ringOffsets: offsets(2),
          polygonOffsets: offsets(2)
        }
      })
        .getCommandNodes(graph)
        .map(node => node.id)
    ).toEqual(['normalize-polygons-empty-ringOffsets', 'normalize-polygons-empty-parts']);
    expect(
      new GPUSegmentSnap({
        positions: minimal('float32x2'),
        lineOffsets: offsets(2),
        referencePositions: minimal('float32x2'),
        tolerance: 0,
        maximumVerticesPerLine: 1,
        output: {
          positions: createTransientView(graph, 'snap-row', 'float32x2', 1),
          sourceEdges: createTransientView(graph, 'snap-edge', 'uint32', 1),
          counts: createTransientView(graph, 'snap-count', 'uint32x4', 1)
        }
      })
        .getCommandNodes(graph)
        .map(node => node.id)
    ).toEqual(['segment-snap-empty-counts']);
  } finally {
    device.destroy();
  }
});
