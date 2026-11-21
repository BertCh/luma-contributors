// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {createTransientView, GPUCommandGraph} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it, vi} from 'vitest';
import {
  createPieceEmitNode,
  createPieceRangeNode,
  LINE_PIECE_STRIDE
} from '../../../src/gpu-spatial-analysis/line-segmentize/line-chunk-kernels';
import {createWGSLKernelNode} from '../../../src/analysis-core/kernels/wgsl-kernel-nodes';
import {
  createNullWebGPUDevice,
  getGPUCommandNodeKernelSources
} from '../../utils/gpu-contributor-test-utils';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';

import {WgslExec, VectorData, ScalarData} from 'wgsl_reflect';

useCPUExecutorTestHooks();
const device = createNullWebGPUDevice();
afterAll(() => device.destroy());
it('t1 production cut helpers preserve shared cuts and terminal zero-width substrings', () => {
  const graph = new GPUCommandGraph(device),
    memory = createCPUGraphMemory();
  const positions = createTransientView(graph, 'positions', 'float32x2', 193),
    rowMeasures = createTransientView(graph, 'prefix', 'float32', 193),
    outputPositions = createTransientView(graph, 'output', 'float32x2', 4);
  const coordinates = new Float32Array(386),
    measures = new Float32Array(193);
  for (let row = 0; row < 193; row++) {
    coordinates[2 * row] = Math.min(row, 64) * 16;
    coordinates[2 * row + 1] = Math.max(row - 64, 0) * 2 ** -20;
    measures[row] = Math.min(row, 64) * 16 + Math.max(row - 64, 0) * 2 ** -20;
  }
  memory.write(positions, coordinates);
  memory.write(rowMeasures, measures);
  const node = createPieceEmitNode(graph, {
    id: 'production-piece',
    operation: 'GPULineChunk',
    coordinateSystem: 'planar',
    positions,
    rowMeasures,
    pieces: createTransientView(graph, 'pieces', 'uint32', 20),
    vertexStarts: createTransientView(graph, 'starts', 'uint32', 2),
    outputPositions
  });
  const source = getGPUCommandNodeKernelSources([node])[0].source;
  // Execute the actual production helpers without the balanced-search workgroup,
  // which the serial interpreter cannot synchronize. No interpolation is reimplemented.
  const first = source.indexOf('fn getMeasure('),
    last = source.indexOf('fn writeVertex(');
  expect(first).toBeGreaterThanOrEqual(0);
  expect(last).toBeGreaterThan(first);
  // wgsl_reflect treats vector mix with scalar weight as ScalarData. Supply only
  // that missing primitive; the production cut search/interpolation stays unchanged.
  const original = WgslExec.prototype._callBuiltinFunction;
  const primitive = vi
    .spyOn(WgslExec.prototype, '_callBuiltinFunction')
    .mockImplementation(function (this: WgslExec, node, context) {
      if (node.name === 'mix') {
        const start = this.evalExpression(node.args![0], context) as VectorData;
        const end = this.evalExpression(node.args![1], context) as VectorData;
        const fraction = this.evalExpression(node.args![2], context) as ScalarData;
        return new VectorData(
          Array.from(start.data, (value, row) =>
            Math.fround(value * (1 - fraction.value) + end.data[row] * fraction.value)
          ),
          start.typeInfo
        );
      }
      return original.call(this, node, context);
    });
  try {
    executeCommandNodesOnCPU(
      [
        createWGSLKernelNode(graph, {
          id: 'cut-helper-probe',
          operation: 'GPULineChunk',
          invocationCount: 4,
          bindings: [
            {name: 'positions', view: positions, type: 'f32', access: 'read'},
            {name: 'rowMeasures', view: rowMeasures, type: 'f32', access: 'read'},
            {name: 'outputPositions', view: outputPositions, type: 'f32', access: 'read_write'}
          ],
          declarations: source.slice(first, last),
          body: `let measure = select(1024.0,1024.0001220703125,index >= 2u);
// Both source segments adjacent to the plateau resolve to the same stored coordinate.
let segment = select(63u,127u,(index % 2u) == 1u);
let point = cutPosition(segment,measure,0u,193u);
outputPositions[outputPositionsOffset+2u*index] = point.x;
outputPositions[outputPositionsOffset+2u*index+1u] = point.y;`
        })
      ],
      memory
    );
  } finally {
    primitive.mockRestore();
  }
  // Binary closed form: the shared 1024 cut is row 64, the known terminal is row 192.
  expect(Array.from(memory.readFloat32(outputPositions))).toEqual([
    1024, 0, 1024, 0, 1024, 0.0001220703125, 1024, 0.0001220703125
  ]);
});

it('forward F2 retains moving interior rows between a shared cut and the terminal vertex', () => {
  const graph = new GPUCommandGraph(device);
  const memory = createCPUGraphMemory();
  const lineOffsets = createTransientView(graph, 'range-offsets', 'uint32', 2);
  const rowMeasures = createTransientView(graph, 'range-measures', 'float32', 193);
  const parameters = createTransientView(graph, 'range-parameters', 'float32', 4);
  const pieceCounts = createTransientView(graph, 'range-counts', 'uint32', 1);
  const pieceStarts = createTransientView(graph, 'range-starts', 'uint32', 1);
  const pieces = createTransientView(graph, 'range-pieces', 'uint32', 2 * LINE_PIECE_STRIDE);
  const vertexCounts = createTransientView(graph, 'range-vertices', 'uint32', 2);
  const pieceTotal = createTransientView(graph, 'range-total', 'uint32', 1);
  memory.write(lineOffsets, Uint32Array.of(0, 193));
  memory.write(
    rowMeasures,
    Float32Array.from(
      {length: 193},
      (_, row) => Math.min(row, 64) * 16 + Math.max(row - 64, 0) * 2 ** -20
    )
  );
  memory.write(parameters, Float32Array.of(1024, 0, 3.4e38, 0));
  memory.write(pieceCounts, Uint32Array.of(2));
  memory.write(pieceStarts, Uint32Array.of(0));
  executeCommandNodesOnCPU(
    [
      createPieceRangeNode(graph, {
        id: 'forward-ranges',
        operation: 'GPULineChunk',
        mode: 'chunk',
        lineOffsets,
        rowMeasures,
        parameters,
        pieceCounts,
        pieceStarts,
        rowCount: 193,
        pieces,
        vertexCounts,
        pieceTotal
      })
    ],
    memory
  );
  const rows = memory.readUint32(pieces);
  // Binary closed form: shared row 64, next interior row 65, final row 192.
  // The second chunk retains all 128 moving increments, giving 129 vertices.
  expect([rows[5], rows[LINE_PIECE_STRIDE + 4], rows[LINE_PIECE_STRIDE + 5]]).toEqual([
    64, 65, 192
  ]);
  expect(Array.from(memory.readUint32(vertexCounts))).toEqual([65, 129]);
});
