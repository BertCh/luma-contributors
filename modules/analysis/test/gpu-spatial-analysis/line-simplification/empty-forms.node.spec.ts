// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright (c) vis.gl contributors

import {
  createTransientView,
  GPUCommandGraph,
  type GraphDataView,
  type GPUGraphPrimitive,
  type GPUCommandNode
} from '@luma.gl/graph/gpu-core';
import {afterAll, expect, it} from 'vitest';
import {
  GPULineSimplificationOperation,
  type LineSimplificationOperationProps
} from '../../../src/gpu-spatial-analysis/line-simplification/gpu-line-simplification-operation';
import {CPULineSimplificationOperation} from '../../../src/gpu-spatial-analysis/line-simplification/cpu-line-simplification-operation';
import {createEmptyLineSimplification} from '../../../src/gpu-spatial-analysis/line-simplification/empty-line-simplification';
import {createNullWebGPUDevice} from '../../utils/gpu-contributor-test-utils';
import {
  createCPUGraphMemory,
  executeCommandNodesOnCPU
} from '../segment-intersection/wgsl-graph-executor';
import {useCPUExecutorTestHooks} from '../segment-intersection/cpu-executor-test-hooks';

useCPUExecutorTestHooks();
const device = createNullWebGPUDevice();
afterAll(() => device.destroy());

for (const {rowCount, lineCount, terminal} of [
  {rowCount: 0, lineCount: 0, terminal: 0},
  {rowCount: 0, lineCount: 1, terminal: 0},
  {rowCount: 4, lineCount: 0, terminal: 2}
]) {
  for (const selection of [false, true]) {
    it(`lowers ${lineCount} empty lines over ${rowCount} rows with selection ${selection} on both named forms`, async () => {
      const props: LineSimplificationOperationProps = {
        positions: 'positions',
        lineOffsets: 'lineOffsets',
        importance: 'importance',
        rowCount,
        lineCount,
        capacity: 0,
        parameters: selection ? 'parameters' : undefined,
        output: {
          ...(selection
            ? {ids: 'ids', outputMask: 'mask', lineCounts: 'counts', lineStarts: 'starts'}
            : {}),
          status: {
            converged: 'converged',
            iterationCount: 'iterations',
            ...(selection
              ? {
                  count: 'count',
                  overflow: 'overflow',
                  requiredCount: 'required',
                  candidateOverflow: 'candidate'
                }
              : {})
          }
        }
      };
      const operation = new GPULineSimplificationOperation(props);
      const graph = new GPUCommandGraph(device);
      const memory = createCPUGraphMemory();
      const ports: Record<string, GraphDataView> = {
        positions: createTransientView(graph, 'positions', 'float32x2', rowCount),
        lineOffsets: createTransientView(graph, 'lineOffsets', 'uint32', lineCount + 1),
        parameters: createTransientView(graph, 'parameters', 'float32', 4)
      };
      const offsets = new Uint32Array(lineCount + 1).fill(terminal);
      memory.write(ports['lineOffsets'], offsets);
      memory.write(ports['parameters'], Float32Array.of(0, 0, 0, 0));
      for (const [port, output] of Object.entries(operation.outputs)) {
        ports[port] = createTransientView(graph, port, output.format, output.length);
        memory.write(
          ports[port],
          output.format === 'float32'
            ? new Float32Array(output.length).fill(99)
            : new Uint32Array(output.length).fill(99)
        );
        expect(output.length).toBe(
          ['converged', 'iterations', 'count', 'overflow', 'required', 'candidate'].includes(
            output.value
          )
            ? 1
            : ['counts', 'starts'].includes(output.value)
              ? lineCount
              : ['importance', 'mask'].includes(output.value)
                ? rowCount
                : 0
        );
      }
      const contributor = operation.createContributor(port => ports[port] as never);
      executeCommandNodesOnCPU(contributor.getCommandNodes(graph), memory);
      expect(Array.from(memory.readUint32(ports['importance']))).toEqual(
        new Array(rowCount).fill(0)
      );
      expect(Array.from(memory.readUint32(ports['converged']))).toEqual([1]);
      expect(Array.from(memory.readUint32(ports['iterationCount']))).toEqual([1]);
      if (selection) {
        expect(Array.from(memory.readUint32(ports['outputMask']))).toEqual(
          new Array(rowCount).fill(0)
        );
        for (const port of ['count', 'overflow', 'requiredCount', 'candidateOverflow'])
          expect(Array.from(memory.readUint32(ports[port]))).toEqual([0]);
        for (const port of ['lineCounts', 'lineStarts'])
          expect(Array.from(memory.readUint32(ports[port]))).toEqual(new Array(lineCount).fill(0));
      }
      expect(Array.from(memory.readUint32(ports['lineOffsets']))).toEqual(Array.from(offsets));
      const positions = new Float32Array(rowCount * 2),
        parameters = new Float32Array(4);
      const cpu = new CPULineSimplificationOperation(props);
      const result = await cpu.execute({
        inputs: {
          positions: {
            format: 'float32x2',
            length: rowCount,
            chunks: [{values: positions, length: rowCount}],
            pack: () => positions
          },
          lineOffsets: {
            format: 'uint32',
            length: offsets.length,
            chunks: [{values: offsets, length: offsets.length}],
            pack: () => offsets
          },
          ...(selection
            ? {
                parameters: {
                  format: 'float32' as const,
                  length: 4,
                  chunks: [{values: parameters, length: 4}],
                  pack: () => parameters
                }
              }
            : {})
        }
      });
      expect(result).toMatchObject({
        importance: new Float32Array(rowCount),
        converged: Uint32Array.of(1),
        iterations: Uint32Array.of(1)
      });
      if (selection)
        expect(result).toMatchObject({
          ids: new Uint32Array(0),
          mask: new Uint32Array(rowCount),
          counts: new Uint32Array(lineCount),
          starts: new Uint32Array(lineCount),
          count: Uint32Array.of(0),
          overflow: Uint32Array.of(0)
        });
    });
  }
}

it('S3 ignores an unused GPU selection capacity in importance-only mode', () => {
  expect(
    () =>
      new GPULineSimplificationOperation({
        positions: 'positions',
        lineOffsets: 'offsets',
        importance: 'importance',
        rowCount: 1,
        lineCount: 1,
        capacity: -1
      })
  ).not.toThrow();
});

it('B3 empty lowering preserves a parameterized graph primitive contract', () => {
  const graph = new GPUCommandGraph<{frame: number}>(device);
  const ports = {
    positions: createTransientView(graph, 'generic-positions', 'float32x2', 0),
    lineOffsets: createTransientView(graph, 'generic-offsets', 'uint32', 1),
    importance: createTransientView(graph, 'generic-importance', 'float32', 0),
    converged: createTransientView(graph, 'generic-converged', 'uint32', 1)
  };
  const primitive: GPUGraphPrimitive = createEmptyLineSimplification(
    {
      positions: 'positions',
      lineOffsets: 'lineOffsets',
      importance: 'importance',
      rowCount: 0,
      lineCount: 0,
      output: {status: {converged: 'converged'}}
    },
    port => ports[port as keyof typeof ports]
  );
  const nodes: readonly GPUCommandNode<{frame: number}>[] = primitive.getCommandNodes(graph);
  expect(nodes.map(node => node.id)).toEqual(['line-simplification-empty-converged']);
});
